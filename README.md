# Tiny Class Page Cleaner

A small, deterministic-first pipeline for converting photographed/scanned class pages into clean black-on-white pages **without changing page geometry**.

## Design

`page -> foreground mask -> black/white render`

The neural network predicts only a foreground mask. It never generates replacement pixels, so it cannot hallucinate handwriting, diagrams, or text. Final rendering is deterministic.

## Pipeline

1. `src/prepare.py` — build page-level train/validation splits and deterministic pseudo-label masks.
2. `src/train.py` — train the tiny U-Net with BCE + Dice loss.
3. `src/infer.py` — overlapping full-resolution patch inference; output width/height are preserved exactly.
4. `src/batch_infer.py` — process an entire folder recursively.
5. `src/evaluate.py` — evaluate validation masks using IoU/F1.
6. `src/export_onnx.py` — export ONNX and an INT8 version and report model size.

## Local commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare data:

```bash
python -m src.prepare --input data/raw --output data/processed
```

Train:

```bash
python -m src.train
```

Clean one page:

```bash
python -m src.infer --input page.jpg --output clean.png
```

Clean a folder:

```bash
python -m src.batch_infer --input pages --output cleaned
```

Evaluate:

```bash
python -m src.evaluate
```

Export:

```bash
python -m src.export_onnx
```

## Important

The current pseudo-labeler is an MVP baseline. The intended production model should be trained on representative pages and checked visually on held-out pages before trusting the result for a large archive. The 10 MB target is enforced at export time; if the INT8 model exceeds it, reduce `base_channels`/`depth` and retrain.
