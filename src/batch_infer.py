"""Batch-clean a directory while preserving every page's original dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .infer import process

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description="Clean all supported page images in a folder.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--device", default="")
    args = p.parse_args()

    files = sorted(x for x in args.input.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported images found in {args.input}")

    for i, src in enumerate(files, 1):
        relative = src.relative_to(args.input).with_suffix(".png")
        dst = args.output / relative
        process(src, dst, args.checkpoint, args.threshold, args.patch_size, args.stride, args.device)
        print(f"[{i}/{len(files)}] {src} -> {dst}")


if __name__ == "__main__":
    main()
