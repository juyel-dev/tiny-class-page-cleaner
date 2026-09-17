"""Export the trained model to ONNX and optionally create an INT8 version."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TinyUNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--output", type=Path, default=Path("models/tiny_cleaner.onnx"))
    p.add_argument("--quantized-output", type=Path, default=Path("models/tiny_cleaner.int8.onnx"))
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TinyUNet(base_channels=int(ckpt.get("base_channels", 16)), depth=int(ckpt.get("depth", 3)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["foreground_logit"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "foreground_logit": {0: "batch", 2: "height", 3: "width"}},
        opset_version=args.opset,
        dynamo=True,
    )

    # Dynamic INT8 quantization reduces weight storage; convolution-heavy graphs
    # are kept portable by quantizing supported weights through ONNX Runtime.
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(args.output), str(args.quantized_output), weight_type=QuantType.QInt8)
    except Exception as exc:
        print(f"INT8 quantization skipped: {exc}")
        return

    for path in (args.output, args.quantized_output):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path}: {size_mb:.2f} MB")
    q_mb = args.quantized_output.stat().st_size / (1024 * 1024)
    if q_mb > 10:
        print("WARNING: INT8 model is above the 10 MB target; reduce base_channels/depth and retrain.")


if __name__ == "__main__":
    main()
