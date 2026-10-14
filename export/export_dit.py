"""Export a static Turbo DiT ONNX model for a captured duration/conditioning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = (
    "hidden_states", "timestep", "timestep_r", "attention_mask",
    "encoder_hidden_states", "encoder_attention_mask", "context_latents",
)


class DecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden_states, timestep, timestep_r, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents):
        return self.decoder(
            hidden_states=hidden_states, timestep=timestep, timestep_r=timestep_r,
            attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, context_latents=context_latents,
            use_cache=False, past_key_values=None,
        )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=ROOT / "models" / "reference" / "dit")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "onnx" / "dit.onnx")
    args = parser.parse_args()
    torch.set_num_threads(8)
    sys.path.insert(0, str(ROOT / "upstream" / "ACE-Step-1.5"))
    capture = args.capture
    recorded = torch.load(capture / "inputs.pt", map_location="cpu", weights_only=True)
    inputs = tuple(recorded[f"inputs.kwargs.{name}"] for name in INPUT_NAMES)
    golden = torch.load(capture / "outputs.pt", map_location="cpu", weights_only=True)["outputs"]
    checkpoint = ROOT / "upstream" / "ACE-Step-1.5" / "checkpoints" / "acestep-v15-turbo"
    model = AutoModel.from_pretrained(str(checkpoint), trust_remote_code=True, attn_implementation="sdpa", dtype=torch.float32).eval()
    wrapper = DecoderWrapper(model.decoder).eval()
    with torch.inference_mode():
        direct = wrapper(*inputs)
    rel = float(torch.linalg.vector_norm(direct - golden) / torch.linalg.vector_norm(golden))
    print(f"Cache-free direct vs captured relative L2: {rel}", flush=True)
    if rel > 1e-5:
        raise RuntimeError("Cache-free DiT differs from captured output")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, inputs, str(output), export_params=True, external_data=True,
            opset_version=17, do_constant_folding=True, input_names=list(INPUT_NAMES),
            output_names=["denoised"], dynamic_axes=None,
        )
    onnx.checker.check_model(str(output))
    model_proto = onnx.load(str(output), load_external_data=False)
    locations = {
        entry.value
        for tensor in model_proto.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    report = {
        "component": "dit", "cache_free_vs_capture_relative_l2": rel,
        "input_shapes": {name: list(value.shape) for name, value in zip(INPUT_NAMES, inputs)},
        "onnx_file_bytes": output.stat().st_size,
        "external_data_files": len(locations),
        "external_data_bytes": sum((output.parent / path).stat().st_size for path in locations),
    }
    (output.parent / f"{output.stem}_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
