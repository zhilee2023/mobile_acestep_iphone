"""Export the captured Qwen3 Text Encoder call and validate ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]


class TextEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # A 4D additive mask takes the exportable early-exit path in
        # Transformers' masking code instead of its vmap-based mask builder.
        length = input_ids.shape[1]
        upper_triangle = torch.ones((length, length), device=input_ids.device, dtype=torch.bool).triu(1)
        mask = torch.zeros((1, 1, length, length), device=input_ids.device, dtype=torch.float32)
        mask = mask.masked_fill(upper_triangle, torch.finfo(torch.float32).min)
        return self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((candidate - reference).astype(np.float64).ravel()) / max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream" / "ACE-Step-1.5" / "checkpoints" / "Qwen3-Embedding-0.6B")
    parser.add_argument("--capture", type=Path, default=ROOT / "models" / "reference" / "text_encoder")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "onnx" / "text_encoder.onnx")
    args = parser.parse_args()

    torch.set_num_threads(8)
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    golden = torch.load(args.capture / "outputs.pt", map_location="cpu", weights_only=True)["outputs"].numpy()
    ids = inputs["inputs.kwargs.input_ids"]

    model = AutoModel.from_pretrained(str(args.checkpoint), attn_implementation="eager").to("cpu", dtype=torch.float32).eval()
    wrapper = TextEncoderWrapper(model).eval()
    with torch.inference_mode():
        direct = wrapper(ids).numpy()
    direct_error = relative_l2(direct, golden)
    if direct_error > 1e-5:
        raise RuntimeError(f"Cache-free Text Encoder differs from captured output: relative L2 {direct_error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (ids,),
            str(args.output),
            export_params=True,
            external_data=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes=None,
        )
    del wrapper, model
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    candidate = session.run(None, {"input_ids": ids.numpy()})[0]
    report = {
        "component": "text_encoder",
        "input_shape": list(ids.shape),
        "output_shape": list(candidate.shape),
        "pytorch_vs_capture_relative_l2": direct_error,
        "onnx_vs_capture_relative_l2": relative_l2(candidate, golden),
        "onnx_file_bytes": args.output.stat().st_size,
        "external_data_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in args.output.parent.glob(args.output.name + "*")
            if path != args.output
        ],
    }
    (args.output.parent / "text_encoder_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
