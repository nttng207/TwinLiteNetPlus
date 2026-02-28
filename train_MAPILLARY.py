import os
import time
import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn
import yaml
import json
import math
from copy import deepcopy
from argparse import ArgumentParser

from model.model import TwinLiteNetPlus
from loss import TotalLoss
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
from MAPILLARY import MapillaryDataset

class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # Exponential decay ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA model parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def train_net(args, hyp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    cudnn.benchmark = True

    model = TwinLiteNetPlus(args).to(device)
    print("Total params:", netParams(model))

    train_set = MapillaryDataset(
        hyp=hyp,
        root=args.data_root,
        split="training",
        valid=False
    )

    val_set = MapillaryDataset(
        hyp=hyp,
        root=args.data_root,
        split="validation",
        valid=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    criterion = TotalLoss(hyp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyp["lr"],
        betas=(hyp["momentum"], 0.999),
        eps=hyp["eps"],
        weight_decay=hyp["weight_decay"]
    )

    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model) if args.ema else None

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")

        # CASE 1: full training checkpoint
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            print("=> Resuming full checkpoint")
            model.load_state_dict(ckpt["state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            if args.ema and ckpt.get("ema_state_dict") is not None:
                ema.ema.load_state_dict(ckpt["ema_state_dict"])
                ema.updates = ckpt.get("updates", 0)

        # CASE 2: pretrained weights only
        else:
            print("=> Loading pretrained weights only (finetune)")
            model.load_state_dict(ckpt, strict=False)
            start_epoch = 0

    os.makedirs(args.savedir, exist_ok=True)

    for epoch in range(start_epoch, args.max_epochs):
        poly_lr_scheduler(args, hyp, optimizer, epoch)

        model.train()
        start_train = time.time()
        train(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scaler,
            False,
            ema
        )
        end_train = time.time()
        print(f"Epoch {epoch} training time: {end_train - start_train:.2f} seconds")

        model.eval()
        start_val = time.time()
        da_res, ll_res = val(
            val_loader,
            ema.ema if args.ema else model,
            args=args
        )
        end_val = time.time()
        print(f"Epoch {epoch} validation time: {end_val - start_val:.2f} seconds")
        print(
            f"[{epoch}] "
            f"DA mIoU: {da_res[2]:.4f} | "
            f"LL Acc: {ll_res[0]:.4f} IOU: {ll_res[1]:.4f}"
        )

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict() if args.ema else None,
            "updates": ema.updates if args.ema else None,
            "optimizer": optimizer.state_dict(),
        }, os.path.join(args.savedir, "checkpoint.pth.tar"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savedir', default='./testv3', help='Directory to save the results')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML')
    parser.add_argument('--resume', type=str, default='', help='Resume training from a checkpoint')
    parser.add_argument('--config', default='nano', help='Model configuration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--ema', action='store_true', help='Use Exponential Moving Average (EMA)')
    parser.add_argument('--data_root', type=str, help='Path to the Mapillary Vistas dataset root')

    args = parser.parse_args()
    
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters
    
    train_net(args, hyp.copy())
