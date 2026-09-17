"""Deterministic pseudo-label generation for class-page cleaning.

The target is a foreground mask. Rendering the final page is deliberately
kept outside the neural network so the model cannot hallucinate page content.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def make_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Create a conservative foreground mask from an RGB/BGR page.

    Dark/colored ink is retained while large, smooth background regions are
    suppressed. The result is intentionally binary and deterministic.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 image")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Local contrast catches ink on uneven classroom-page backgrounds.
    local = cv2.GaussianBlur(gray, (0, 0), 9)
    detail = cv2.absdiff(gray, local)
    dark = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    contrast = cv2.threshold(detail, 10, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(dark, contrast)

    # Remove isolated compression noise without erasing thin strokes.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def render_black_on_white(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render foreground as black and background as pure white."""
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions differ")
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask > 0] = 0
    return out


def process_file(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        rgb = np.asarray(im.convert("RGB"))
    mask = make_foreground_mask(rgb)
    out = render_black_on_white(rgb, mask)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)
