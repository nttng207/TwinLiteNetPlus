import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.utils.data
import torchvision.transforms as transforms


# Cityscapes gtFine_labelIds.png raw label IDs (0-33)
DRIVABLE_IDS = [7]        # road
LANE_IDS     = [6, 0]     # ground / road markings (label 0 = unlabeled is NOT lane,
                           # but we keep road marking = any non-road flat marking)
# More precise: Cityscapes lane-relevant IDs
# 6  = ground (includes road markings in some versions)
# We will use only markings visible in gtFine_labelIds:
# Actually the standard approach: road=7 for DA, lane lines don't have a dedicated
# single ID in labelIds — they're part of road. We use label 6 (ground) as proxy.
# This matches the spirit of the BDD100K approach used in TwinLiteNet.
DRIVABLE_IDS = [7]
LANE_IDS     = [6]


def letterbox(im, new_shape=(384, 640), color=(114, 114, 114)):
    """Resize + pad to new_shape (H, W), same as MAPILLARY.py."""
    shape = im.shape[:2]  # (H, W)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return im


class CityscapesDataset(torch.utils.data.Dataset):
    """
    Cityscapes dataset for TwinLiteNetPlus — mirrors MapillaryDataset interface exactly.

    Returns: (img_path, image_tensor, (seg_da, seg_ll))
        img_path   : str
        image      : uint8 Tensor (3, 384, 640), BGR, values 0-255
        seg_da     : float Tensor (2, 360, 640)  one-hot drivable area
        seg_ll     : float Tensor (2, 360, 640)  one-hot lane line

    Folder layout under data_root:
        leftImg8bit/val/<city>/*_leftImg8bit.png
        gtFine_trainvaltest/gtFine/val/<city>/*_gtFine_labelIds.png
    """

    W_, H_      = 640, 384   # final image size (matches letterbox)
    MASK_W      = 640        # label width  (matches MAPILLARY.py)
    MASK_H      = 360        # label height (matches MAPILLARY.py — NOT 384)

    def __init__(self, hyp: dict, root: str, split: str = "val", valid: bool = True):
        super().__init__()
        self.root  = root
        self.split = split
        self.valid = valid
        self.Tensor = transforms.ToTensor()

        img_dir = os.path.join(root, "leftImg8bit", split)
        gt_dir  = os.path.join(root, "gtFine_trainvaltest", "gtFine", split)

        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*", "*_leftImg8bit.png"))
        )
        self.gt_paths = []
        for img_path in self.img_paths:
            city      = os.path.basename(os.path.dirname(img_path))
            base      = os.path.basename(img_path).replace("_leftImg8bit.png", "")
            mask_path = os.path.join(gt_dir, city, f"{base}_gtFine_labelIds.png")
            self.gt_paths.append(mask_path)

        missing = [p for p in self.gt_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} label files not found. First: {missing[0]}\n"
                "Check --data_root contains 'leftImg8bit/' and 'gtFine_trainvaltest/'."
            )
        if not self.img_paths:
            raise RuntimeError(f"No images found under {img_dir}")

        print(f"[CityscapesDataset] {len(self.img_paths)} images | split='{split}' | "
              f"image=({self.H_},{self.W_}) mask=({self.MASK_H},{self.MASK_W})")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # ---------- image ----------
        # Read as BGR uint8 (same as cv2.imread in MAPILLARY.py)
        image = cv2.imread(img_path)                          # (H, W, 3) BGR uint8
        image = letterbox(image, (self.H_, self.W_))          # (384, 640, 3)

        # ---------- label ----------
        label = np.array(Image.open(self.gt_paths[idx]), dtype=np.int32)  # (H, W)

        # Binary masks: 255 = positive, 0 = negative  (matches MAPILLARY.py)
        label1 = np.isin(label, DRIVABLE_IDS).astype(np.uint8) * 255   # drivable
        label2 = np.isin(label, LANE_IDS    ).astype(np.uint8) * 255   # lane

        label1 = cv2.resize(label1, (self.MASK_W, self.MASK_H), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2, (self.MASK_W, self.MASK_H), interpolation=cv2.INTER_NEAREST)

        # One-hot encode: (2, MASK_H, MASK_W) — same cv2.threshold logic as MAPILLARY.py
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1   = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2   = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

        seg1   = self.Tensor(seg1)    # (1, MASK_H, MASK_W)
        seg2   = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)

        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)   # (2, MASK_H, MASK_W)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)

        # image: HWC BGR → CHW, keep uint8 (val() divides by 255 itself)
        image = image[:, :, ::-1].transpose(2, 0, 1)    # RGB, (3, 384, 640)
        image = np.ascontiguousarray(image)

        return img_path, torch.from_numpy(image), (seg_da, seg_ll)
