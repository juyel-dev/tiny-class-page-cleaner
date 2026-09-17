"""Evaluate a checkpoint against pseudo-label validation pages."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .infer import load_model, predict_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    args = p.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    ious, f1s = [], []
    with args.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            image_path = args.manifest.parent / row["image"]
            mask_path = args.manifest.parent / row["mask"]
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            pred = predict_mask(model, image, args.patch_size, args.stride, device) >= args.threshold
            truth = np.asarray(Image.open(mask_path).convert("L")) > 0
            inter = np.logical_and(pred, truth).sum()
            union = np.logical_or(pred, truth).sum()
            ious.append((inter + 1) / (union + 1))
            f1s.append((2 * inter + 1) / (pred.sum() + truth.sum() + 1))

    if not ious:
        raise SystemExit("No validation pages found.")
    print(f"validation pages: {len(ious)}")
    print(f"mean IoU: {np.mean(ious):.4f}")
    print(f"mean F1:  {np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()
