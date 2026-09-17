"""Prepare page images and deterministic pseudo-labels for training.

Dataset layout after preparation:
  data/processed/images/*.png
  data/processed/masks/*.png
  data/processed/manifest.csv

The split is done by source page, not by patch, to avoid train/validation leakage.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageOps

from .preprocessing import make_foreground_mask

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def prepare(input_dir: Path, output_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> None:
    files = collect_images(input_dir)
    if not files:
        raise RuntimeError(f"No supported images found in {input_dir}")

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:val_count])

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index, src in enumerate(files):
        with Image.open(src) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            mask = make_foreground_mask(image)
            stem = f"page_{index:06d}"
            image_path = image_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            image.save(image_path, "PNG")
            Image.fromarray(mask).save(mask_path, "PNG")

            rows.append({
                "id": stem,
                "source": str(src),
                "image": str(image_path.relative_to(output_dir)),
                "mask": str(mask_path.relative_to(output_dir)),
                "split": "val" if src in val_set else "train",
                "width": str(image.width),
                "height": str(image.height),
            })

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} pages: {len(rows) - val_count} train / {val_count} val")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create page images and pseudo-label masks.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        raise SystemExit("--val-fraction must be in [0, 1).")
    prepare(args.input, args.output, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
