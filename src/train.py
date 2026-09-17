"""Train the tiny foreground segmentation model."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import PagePatchDataset
from .model import TinyUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> tuple[list[Path], list[Path]]:
    train, val = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = val if row["split"] == "val" else train
            target.append(path.parent / row["image"])
    return train, val


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1 - dice.mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # BCE stabilizes pixel-wise learning; Dice handles thin handwriting/strokes.
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return 0.6 * bce + 0.4 * DiceLoss()(logits, target)


def score(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    inter = (pred & truth).sum().item()
    union = (pred | truth).sum().item()
    f1_den = pred.sum().item() + truth.sum().item()
    iou = (inter + 1) / (union + 1)
    f1 = (2 * inter + 1) / (f1_den + 1)
    return iou, f1


def run_epoch(model, loader, optimizer, device, training: bool) -> tuple[float, float, float]:
    model.train(training)
    total_loss = total_iou = total_f1 = 0.0
    count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        iou, f1 = score(logits.detach(), masks)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou += iou * bs
        total_f1 += f1 * bs
        count += bs
    return total_loss / count, total_iou / count, total_f1 / count


def train(args) -> None:
    seed_everything(args.seed)
    manifest = Path(args.manifest)
    train_files, val_files = read_manifest(manifest)
    if not train_files:
        raise RuntimeError("Manifest has no training pages.")
    if not val_files:
        raise RuntimeError("Manifest has no validation pages; increase dataset size or val fraction.")

    train_ds = PagePatchDataset(train_files, patch_size=args.patch_size, stride=args.stride, augment=True)
    val_ds = PagePatchDataset(val_files, patch_size=args.patch_size, stride=args.stride, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyUNet(base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    out = Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)

    best = -1.0
    print(f"device={device} train_patches={len(train_ds)} val_patches={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, True)
        va = run_epoch(model, val_loader, optimizer, device, False)
        scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs}: train loss={tr[0]:.4f} iou={tr[1]:.4f} f1={tr[2]:.4f} | val loss={va[0]:.4f} iou={va[1]:.4f} f1={va[2]:.4f}")
        if va[2] > best:
            best = va[2]
            torch.save({"model": model.state_dict(), "base_channels": args.base_channels, "depth": args.depth, "val_f1": best}, out)
            print(f"  saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    train(p.parse_args())


if __name__ == "__main__":
    main()
