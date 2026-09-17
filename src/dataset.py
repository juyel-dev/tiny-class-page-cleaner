"""Patch dataset for page -> foreground-mask learning."""

from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import make_foreground_mask


class PagePatchDataset(Dataset):
    def __init__(self, files, patch_size=256, stride=192, augment=False):
        self.files = [Path(p) for p in files]
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.samples = []
        for path in self.files:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            ys = list(range(0, max(1, h - patch_size + 1), stride))
            xs = list(range(0, max(1, w - patch_size + 1), stride))
            if not ys or ys[-1] != max(0, h - patch_size):
                ys.append(max(0, h - patch_size))
            if not xs or xs[-1] != max(0, w - patch_size):
                xs.append(max(0, w - patch_size))
            self.samples.extend((path, x, y) for y in ys for x in xs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, x, y = self.samples[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p = self.patch_size
        image = rgb[y:y+p, x:x+p]
        mask = make_foreground_mask(image)

        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.2:
                image = image[::-1].copy()
                mask = mask[::-1].copy()

        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return image, mask
