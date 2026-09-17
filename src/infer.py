"""Full-resolution inference with overlapping patches and exact output dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .model import TinyUNet
from .preprocessing import render_black_on_white


def load_model(checkpoint: Path, device: torch.device) -> TinyUNet:
    ckpt = torch.load(checkpoint, map_location=device)
    model = TinyUNet(base_channels=int(ckpt.get("base_channels", 16)), depth=int(ckpt.get("depth", 3))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_mask(model, image: Image.Image, patch_size: int, stride: int, device: torch.device) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    score_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    if not ys or ys[-1] != max(0, h - patch_size): ys.append(max(0, h - patch_size))
    if not xs or xs[-1] != max(0, w - patch_size): xs.append(max(0, w - patch_size))

    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop = arr[y:min(y + patch_size, h), x:min(x + patch_size, w)]
                ph, pw = crop.shape[:2]
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                padded[:ph, :pw] = crop
                tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()[:ph, :pw]
                # Center-weighting reduces visible seams between overlapping patches.
                wy = np.hanning(max(ph, 3))[:ph].astype(np.float32)
                wx = np.hanning(max(pw, 3))[:pw].astype(np.float32)
                weight = np.outer(wy, wx)
                weight = np.maximum(weight, 0.05)
                score_sum[y:y+ph, x:x+pw] += prob * weight
                weight_sum[y:y+ph, x:x+pw] += weight
    return score_sum / np.maximum(weight_sum, 1e-6)


def process(src: Path, dst: Path, checkpoint: Path, threshold: float, patch_size: int, stride: int, device_name: str) -> None:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, device)
    with Image.open(src) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
        prob = predict_mask(model, image, patch_size, stride, device)
        mask = (prob >= threshold).astype(np.uint8) * 255
        result = render_black_on_white(image, mask)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst, "PNG")
        assert result.size == image.size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--device", default="")
    args = p.parse_args()
    process(args.input, args.output, args.checkpoint, args.threshold, args.patch_size, args.stride, args.device)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
