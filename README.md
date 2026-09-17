# Tiny Class Page Cleaner

A tiny vision model for converting class-note pages into clean, printable black-on-white pages while preserving the original page geometry.

## Goal

- Train on 5,000+ class pages
- Learn foreground/background separation
- Preserve handwriting, diagrams, layout, and dimensions
- Produce deterministic black-on-white output
- Quantize the model to INT8
- Keep the final model under 10 MB

## Planned pipeline

```text
Raw pages
   -> preprocessing / pseudo-label generation
   -> training patches
   -> tiny segmentation CNN
   -> validation
   -> INT8 quantization
   -> printable B/W rendering
```

## Status

Project initialized. Dataset and model pipeline are being built incrementally.
