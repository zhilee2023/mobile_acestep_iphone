"""Probe Turbo DiT's first real denoising call with simulated block-64 W4.

This is a sensitivity screen, not an MNN fidelity or listening test. Only
Linear and Conv weights are quantized; norms, biases and embeddings remain FP.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a = candidate.float().reshape(-1)
    b = reference.float().reshape(-1)
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


@torch.no_grad()
def quantize_weights(model: nn.Module, block: int = 64) -> dict[str, int]:
    counts = {"tensors": 0, "elements": 0, "total_parameters": sum(p.numel() for p in model.parameters())}
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            continue
        weight = module.weight
        if weight is None or weight.numel() < block:
            continue
        original_shape = weight.shape
        flat = weight.detach().float().reshape(-1)
        padding = (-flat.numel()) % block
        if padding:
            flat = torch.nn.functional.pad(flat, (0, padding))
        groups = flat.reshape(-1, block)
        lower = groups.amin(dim=1, keepdim=True)
        upper = groups.amax(dim=1, keepdim=True)
        scale = (upper - lower).clamp_min(1e-12) / 15.0
        restored = ((groups - lower) / scale).round().clamp_(0, 15) * scale + lower
        weight.copy_(restored.reshape(-1)[: weight.numel()].reshape(original_shape).to(weight.dtype))
        counts["tensors"] += 1
        counts["elements"] += weight.numel()
    return counts


def main() -> None:
    torch.set_num_threads(8)
    sys.path.insert(0, str(ROOT / "upstream" / "ACE-Step-1.5"))
    captured = ROOT / "models" / "reference" / "dit"
    inputs = torch.load(captured / "inputs.pt", map_location="cpu", weights_only=True)
    golden = torch.load(captured / "outputs.pt", map_location="cpu", weights_only=True)["outputs"]
    checkpoint = ROOT / "upstream" / "ACE-Step-1.5" / "checkpoints" / "acestep-v15-turbo"
    model = AutoModel.from_pretrained(str(checkpoint), trust_remote_code=True, attn_implementation="sdpa", dtype=torch.float32).eval()
    decoder = model.decoder
    kwargs = {name.split(".")[-1]: value for name, value in inputs.items()}
    kwargs["use_cache"] = True

    started = time.perf_counter()
    with torch.inference_mode():
        baseline = decoder(**kwargs)[0].float()
    baseline_seconds = time.perf_counter() - started
    print(f"DiT FP32 baseline: {baseline_seconds:.2f}s, vs capture {relative_l2(baseline, golden):.6g}", flush=True)

    counts = quantize_weights(decoder)
    started = time.perf_counter()
    with torch.inference_mode():
        candidate = decoder(**kwargs)[0].float()
    quant_seconds = time.perf_counter() - started
    report = {
        "scope": "PyTorch FP32 DiT first denoising step; simulated affine W4 block64 on Linear/Conv weights only; not MNN HQQ",
        "reference_shape": list(golden.shape),
        "baseline_vs_captured_relative_l2": relative_l2(baseline, golden),
        "simulated_w4_vs_baseline_relative_l2": relative_l2(candidate, baseline),
        "simulated_w4_vs_captured_relative_l2": relative_l2(candidate, golden),
        "baseline_seconds": baseline_seconds,
        "w4_seconds": quant_seconds,
        "quantized": counts,
    }
    output = ROOT / "outputs" / "w4-sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    (output / "dit_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
