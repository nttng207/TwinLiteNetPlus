import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
from PIL import Image
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import albumentations as A
import json

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def RandomBilateralBlur(img, sigma_bila_low = 0.05, sigma_bila_high=1.0):
    """
    Apply Bilateral Filtering

    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = random.uniform(sigma_bila_low, sigma_bila_high)
    blurred_img = denoise_bilateral(np.array(img_rgb), sigma_spatial=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


    
def RandomGaussianBlur(img, sigma_gaus_a = 1.15, sigma_gaus_b=0.15):
    """
    Apply Gaussian Blur
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = sigma_gaus_b + random.random() * sigma_gaus_a
    blurred_img = gaussian(np.array(img_rgb), sigma=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, drivable, line = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.5 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            drivable = cv2.warpPerspective(drivable, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            drivable = cv2.warpAffine(drivable, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)



    combination = (img, drivable, line)
    return combination

class MapillaryDataset(torch.utils.data.Dataset):
    def __init__(self, hyp, root, split="training", valid=False):
        self.valid = valid
        self.root = root
        self.split = split

        # ===== augmentation params (GIỮ NGUYÊN) =====
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_hsv = hyp["prob_hsv"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]

        self.Random_Crop = A.RandomCrop(
            width=hyp["width_crop"],
            height=hyp["height_crop"]
        )

        self.Tensor = transforms.ToTensor()

        # ===== paths =====
        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "v2.0", "labels")
        self.names = sorted(os.listdir(self.img_dir))

        # ===== load config =====
        with open(os.path.join(root, "config_v2.0.json"), "r") as f:
            cfg = json.load(f)

        self.id2name = {
            idx: label["name"]
            for idx, label in enumerate(cfg["labels"])
        }

        self.da_ids = self._build_drivable_ids()
        self.ll_ids = self._build_lane_ids()

    def _build_drivable_ids(self):
        keywords = [
            "construction--flat--road",
            "construction--flat--service-lane",
            "construction--flat--driveway",
            "construction--flat--parking",
            "construction--flat--parking-aisle",
            "construction--flat--road-shoulder",
            "construction--flat--bike-lane",
        ]
        return {
            idx for idx, name in self.id2name.items()
            if name in keywords
        }

    def _build_lane_ids(self):
        return {
            idx for idx, name in self.id2name.items()
            if (
                name.startswith("marking--")
                or name.startswith("marking-only--")
                or name == "construction--flat--crosswalk-plain"
            )
        }

    def __len__(self):
        return len(self.names)

    # def __getitem__(self, idx):
    #     W_, H_ = 640, 384

    #     image_name = self.names[idx]
    #     img_path = os.path.join(self.img_dir, image_name)
    #     lbl_path = os.path.join(
    #         self.lbl_dir, image_name.replace(".jpg", ".png")
    #     )

    #     image = cv2.imread(img_path)
    #     label = np.array(Image.open(lbl_path), dtype=np.int32)

    #     # ===== build BDD-style masks =====
    #     label1 = np.isin(label, list(self.da_ids)).astype(np.uint8) * 255
    #     label2 = np.isin(label, list(self.ll_ids)).astype(np.uint8) * 255

    #     # ===== augmentation (GIỮ NGUYÊN LOGIC) =====
    #     if not self.valid:
    #         if random.random() < self.prob_perspective:
    #             image, label1, label2 = random_perspective(
    #                 (image, label1, label2),
    #                 degrees=self.degrees,
    #                 translate=self.translate,
    #                 scale=self.scale,
    #                 shear=self.shear
    #             )

    #         if random.random() < self.prob_hsv:
    #             augment_hsv(image, self.hgain, self.sgain, self.vgain)

    #         if random.random() < self.prob_flip:
    #             image = np.fliplr(image)
    #             label1 = np.fliplr(label1)
    #             label2 = np.fliplr(label2)

    #         if random.random() < self.prob_bilateral:
    #             image = RandomBilateralBlur(image)

    #         if random.random() < self.prob_gaussian:
    #             image = RandomGaussianBlur(image)

    #         if random.random() < self.prob_crop:
    #             masks = np.stack([label1, label2], axis=2)
    #             transformed = self.Random_Crop(image=image, mask=masks)
    #             image = transformed["image"]
    #             label1 = transformed["mask"][:, :, 0]
    #             label2 = transformed["mask"][:, :, 1]

    #     image = letterbox(image, (H_, W_))

    #     label1 = cv2.resize(label1, (W_, 360))
    #     label2 = cv2.resize(label2, (W_, 360))

    #     # ===== same post-process as BDD =====
    #     _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
    #     _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
    #     _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
    #     _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

    #     seg1 = self.Tensor(seg1)
    #     seg2 = self.Tensor(seg2)
    #     seg_b1 = self.Tensor(seg_b1)
    #     seg_b2 = self.Tensor(seg_b2)

    #     seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
    #     seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)

    #     image = image[:, :, ::-1].transpose(2, 0, 1)
    #     image = np.ascontiguousarray(image)

    #     return img_path, torch.from_numpy(image), (seg_da, seg_ll)

    def __getitem__(self, idx):
        W_, H_ = 640, 384
    
        image_name = self.names[idx]
        img_path = os.path.join(self.img_dir, image_name)
        lbl_path = os.path.join(self.lbl_dir, image_name.replace(".jpg", ".png"))
    
        image = cv2.imread(img_path)
        label = np.array(Image.open(lbl_path), dtype=np.int32)
    
        # ===== build BDD-style masks =====
        label1 = np.isin(label, list(self.da_ids)).astype(np.uint8) * 255
        label2 = np.isin(label, list(self.ll_ids)).astype(np.uint8) * 255
    
        # =====================================================
        # =====================================================
        if not self.valid:
            image  = cv2.resize(image,  (1280, 720), interpolation=cv2.INTER_LINEAR)
            label1 = cv2.resize(label1, (1280, 720), interpolation=cv2.INTER_NEAREST)
            label2 = cv2.resize(label2, (1280, 720), interpolation=cv2.INTER_NEAREST)
    
        # ===== augmentation =====
        if not self.valid:
            if random.random() < self.prob_perspective:
                image, label1, label2 = random_perspective(
                    (image, label1, label2),
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
    
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
    
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
    
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
    
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
    
            if random.random() < self.prob_crop:
                masks = np.stack([label1, label2], axis=2)
                transformed = self.Random_Crop(image=image, mask=masks)
                image = transformed["image"]
                label1 = transformed["mask"][:, :, 0]
                label2 = transformed["mask"][:, :, 1]
    
        # ===== final resize về input size =====
        image = letterbox(image, (H_, W_))
        label1 = cv2.resize(label1, (W_, 360), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2, (W_, 360), interpolation=cv2.INTER_NEAREST)
    
        # ===== same post-process as BDD =====
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
    
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
    
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
    
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
    
        return img_path, torch.from_numpy(image), (seg_da, seg_ll)


    
    # def __getitem__(self, idx):
    #     t_start = time.perf_counter()
    
    #     W_, H_ = 640, 384
    #     image_name = self.names[idx]
    
    #     # ================== 1. PATH ==================
    #     t0 = time.perf_counter()
    #     img_path = os.path.join(self.img_dir, image_name)
    #     lbl_path = os.path.join(self.lbl_dir, image_name.replace(".jpg", ".png"))
    #     t1 = time.perf_counter()
    
    #     # ================== 2. DISK IO ==================
    #     image = cv2.imread(img_path)
    #     label = np.array(Image.open(lbl_path), dtype=np.int32)
    #     t2 = time.perf_counter()
    
    #     # ================== 3. BUILD MASK ==================
    #     label1 = np.isin(label, list(self.da_ids)).astype(np.uint8) * 255
    #     label2 = np.isin(label, list(self.ll_ids)).astype(np.uint8) * 255
    #     t3 = time.perf_counter()
    
    #     # ================== 4. AUGMENTATION ==================
    #     if not self.valid:
    #         if random.random() < self.prob_perspective:
    #             image, label1, label2 = random_perspective(
    #                 (image, label1, label2),
    #                 degrees=self.degrees,
    #                 translate=self.translate,
    #                 scale=self.scale,
    #                 shear=self.shear
    #             )
    
    #         if random.random() < self.prob_hsv:
    #             augment_hsv(image, self.hgain, self.sgain, self.vgain)
    
    #         if random.random() < self.prob_flip:
    #             image = np.fliplr(image)
    #             label1 = np.fliplr(label1)
    #             label2 = np.fliplr(label2)
    
    #         if random.random() < self.prob_bilateral:
    #             image = RandomBilateralBlur(image)
    
    #         if random.random() < self.prob_gaussian:
    #             image = RandomGaussianBlur(image)
    
    #         if random.random() < self.prob_crop:
    #             masks = np.stack([label1, label2], axis=2)
    #             transformed = self.Random_Crop(image=image, mask=masks)
    #             image = transformed["image"]
    #             label1 = transformed["mask"][:, :, 0]
    #             label2 = transformed["mask"][:, :, 1]
    #     t4 = time.perf_counter()
    
    #     # ================== 5. RESIZE / LETTERBOX ==================
    #     image = letterbox(image, (H_, W_))
    #     label1 = cv2.resize(label1, (W_, 360))
    #     label2 = cv2.resize(label2, (W_, 360))
    #     t5 = time.perf_counter()
    
    #     # ================== 6. POST PROCESS ==================
    #     _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
    #     _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
    #     _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
    #     _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
    #     t6 = time.perf_counter()
    
    #     # ================== 7. TO TENSOR ==================
    #     seg1 = self.Tensor(seg1)
    #     seg2 = self.Tensor(seg2)
    #     seg_b1 = self.Tensor(seg_b1)
    #     seg_b2 = self.Tensor(seg_b2)
    
    #     seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
    #     seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
    
    #     image = image[:, :, ::-1].transpose(2, 0, 1)
    #     image = np.ascontiguousarray(image)
    #     image = torch.from_numpy(image)
    #     t7 = time.perf_counter()
    
    #     # ================== 8. LOG (mỗi 50 sample) ==================
    #     if idx % 50 == 0:
    #         print(
    #             f"[Dataset idx {idx}] "
    #             f"path={t1-t0:.3f}s | "
    #             f"io={t2-t1:.3f}s | "
    #             f"mask={t3-t2:.3f}s | "
    #             f"aug={t4-t3:.3f}s | "
    #             f"resize={t5-t4:.3f}s | "
    #             f"post={t6-t5:.3f}s | "
    #             f"tensor={t7-t6:.3f}s | "
    #             f"total={t7-t_start:.3f}s"
    #         )

    #     return img_path, image, (seg_da, seg_ll)
