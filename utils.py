
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, hyp, optimizer, epoch, power=1.5):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


import time
import torch
from tqdm import tqdm

# def train(args, train_loader, model, criterion, optimizer, epoch, scaler=None, verbose=False, ema=None):
#     model.train()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"\nEpoch {epoch}")
#     total_batches = len(train_loader)

#     pbar = tqdm(total=total_batches, dynamic_ncols=True)

#     # ===== timing buffers =====
#     prev_iter_end = time.perf_counter()

#     for i, (_, input, target) in enumerate(train_loader):

#         # ================== DATA LOADING ==================
#         data_time = time.perf_counter() - prev_iter_end

#         # ================== CPU → GPU ==================
#         t0 = time.perf_counter()
#         input = input.to(device, non_blocking=True).float() / 255.0
#         target = [t.to(device, non_blocking=True) for t in target]
#         torch.cuda.synchronize()
#         h2d_time = time.perf_counter() - t0

#         optimizer.zero_grad(set_to_none=True)

#         # ================== FORWARD + LOSS ==================
#         t1 = time.perf_counter()
#         with torch.amp.autocast("cuda", enabled=scaler is not None):
#             output = model(input)
#             focal_loss, tversky_loss, loss = criterion(output, target)
#         torch.cuda.synchronize()
#         fw_time = time.perf_counter() - t1

#         # ================== BACKWARD + OPT ==================
#         t2 = time.perf_counter()
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
#         torch.cuda.synchronize()
#         bw_time = time.perf_counter() - t2

#         # ================== EMA ==================
#         t3 = time.perf_counter()
#         if ema is not None:
#             ema.update(model)
#         torch.cuda.synchronize()
#         ema_time = time.perf_counter() - t3

#         # ================== LOG ==================
#         iter_time = data_time + h2d_time + fw_time + bw_time + ema_time

#         pbar.set_postfix({
#             "data": f"{data_time:.3f}s",
#             "h2d": f"{h2d_time:.3f}s",
#             "fw": f"{fw_time:.3f}s",
#             "bw": f"{bw_time:.3f}s",
#             "it": f"{iter_time:.3f}s",
#             "loss": f"{loss.detach().item():.4f}",
#         })

#         pbar.update(1)

#         prev_iter_end = time.perf_counter()

#     pbar.close()

def train(args, train_loader, model, criterion, optimizer, epoch, scaler=None, verbose=False, ema=None):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEpoch {epoch}")
    total_batches = len(train_loader)

    pbar = tqdm(
        total=total_batches,
        dynamic_ncols=True,
        smoothing=0.05,   # giúp ETA ổn định hơn
    )

    prev_iter_end = time.perf_counter()

    for i, (_, input, target) in enumerate(train_loader):

        # ================== DATA LOADING ==================
        data_time = time.perf_counter() - prev_iter_end

        # ================== CPU → GPU ==================
        t0 = time.perf_counter()
        input = input.to(device, non_blocking=True).float() / 255.0
        target = [t.to(device, non_blocking=True) for t in target]
        torch.cuda.synchronize()
        h2d_time = time.perf_counter() - t0

        optimizer.zero_grad(set_to_none=True)

        # ================== FORWARD + LOSS ==================
        t1 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(input)
            focal_loss, tversky_loss, loss = criterion(output, target)
        torch.cuda.synchronize()
        fw_time = time.perf_counter() - t1

        # ================== BACKWARD + OPT ==================
        t2 = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        bw_time = time.perf_counter() - t2

        # ================== EMA ==================
        t3 = time.perf_counter()
        if ema is not None:
            ema.update(model)
        torch.cuda.synchronize()
        ema_time = time.perf_counter() - t3

        iter_time = data_time + h2d_time + fw_time + bw_time + ema_time

        # ================== LOG (CHỈ LOG MỖI N BATCH) ==================
        if i % 20 == 0:
            pbar.set_postfix({
                "data": f"{data_time:.3f}s",
                "h2d":  f"{h2d_time:.3f}s",
                "fw":   f"{fw_time:.3f}s",
                "bw":   f"{bw_time:.3f}s",
                "it":   f"{iter_time:.3f}s",
                "focal": f"{focal_loss.detach().item():.4f}",
                "tver":  f"{tversky_loss.detach().item():.4f}",
                "loss":  f"{loss.detach().item():.4f}",
            })

        pbar.update(1)
        prev_iter_end = time.perf_counter()

    pbar.close()


@torch.no_grad()
def val(val_loader = None, model = None, half = False, args=None):

    model.eval()

    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().half() / 255.0 if half else input.cuda().float() / 255.0
        
        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)

        ###-------------Drivable Segmetation--------------
        out_da = output[0]
        target_da = target[0]

        _,da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:,12:-12]
        _,da_gt=torch.max(target_da, 1)

        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))
        ###-------------Drivable Segmetation--------------

        ###-------------Lane Segmetation-----------------
        out_ll = output[1]
        target_ll = target[1]


        _,ll_predict=torch.max(out_ll, 1)
        ll_predict = ll_predict[:,12:-12]
        _,ll_gt=torch.max(target_ll, 1)
        
        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))
        ###-------------Lane Segmetation-----------------

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    
    return da_segment_result,ll_segment_result


@torch.no_grad()
def val_one(val_loader = None, model = None, half = False, args=None):

    model.eval()

    RE=SegmentationMetric(2)

    acc_seg = AverageMeter()
    IoU_seg = AverageMeter()
    mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().half() / 255.0 if half else input.cuda().float() / 255.0
        
        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)




        _,predict=torch.max(output, 1)
        predict = predict[:,12:-12]
        _,gt=torch.max(target, 1)
        
        RE.reset()
        RE.addBatch(predict.cpu(), gt.cpu())

        acc = RE.lineAccuracy()
        IoU = RE.IntersectionOverUnion()
        mIoU = RE.meanIntersectionOverUnion()

        acc_seg.update(acc,input.size(0))
        IoU_seg.update(IoU,input.size(0))
        mIoU_seg.update(mIoU,input.size(0))


    segment_result = (acc_seg.avg,IoU_seg.avg,mIoU_seg.avg)
    
    return segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])