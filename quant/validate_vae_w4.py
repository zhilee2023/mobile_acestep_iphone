"""Compare real ACE-Step VAE decoder outputs across PyTorch and MNN FP/W4."""

from __future__ import annotations

import gc
import argparse
import json
import time
from pathlib import Path

import MNN
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "models" / "reference" / "vae_decoder"
MODEL_DIR = ROOT / "models" / "mnn"
OUTPUT = ROOT / "outputs" / "vae-w4-validation"


def run_mnn(model_path: Path, latents: np.ndarray) -> tuple[np.ndarray, float]:
    interpreter = MNN.Interpreter(str(model_path))
    session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    input_tensor = interpreter.getSessionInput(session)
    host_input = MNN.Tensor(latents.shape, MNN.Halide_Type_Float, latents, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(host_input)
    started = time.perf_counter()
    interpreter.runSession(session)
    elapsed = time.perf_counter() - started
    output_tensor = interpreter.getSessionOutput(session)
    result = np.asarray(output_tensor.getNumpyData(), dtype=np.float32).reshape(output_tensor.getShape()).copy()
    del output_tensor, host_input, input_tensor, session, interpreter
    gc.collect()
    return result, elapsed


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    return {
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(ref64.ravel()), 1e-12)),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--fp-model", type=Path, default=MODEL_DIR / "vae_decoder_fp.mnn")
    parser.add_argument("--w4-model", type=Path, default=MODEL_DIR / "vae_decoder_w4.mnn")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    outputs = torch.load(args.capture / "outputs.pt", map_location="cpu", weights_only=True)
    latents = inputs["inputs.args.0"].numpy().astype(np.float32)
    reference = outputs["outputs"].numpy().astype(np.float32)
    fp_model = args.fp_model
    w4_model = args.w4_model
    fp, fp_seconds = run_mnn(fp_model, latents) if fp_model.is_file() else (None, None)
    w4, w4_seconds = run_mnn(w4_model, latents)
    if fp is not None and fp.size == reference.size:
        fp = fp.reshape(reference.shape)
    if w4.size == reference.size:
        w4 = w4.reshape(reference.shape)
    if (fp is not None and fp.shape != reference.shape) or w4.shape != reference.shape:
        raise RuntimeError(f"Output shape mismatch: PyTorch {reference.shape}, FP {None if fp is None else fp.shape}, W4 {w4.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gain = 10 ** (-1.0 / 20.0) / max(float(np.max(np.abs(w4))), 1e-12)
    sf.write(args.output_dir / "fp-dit-w4-vae-preview.wav", (w4[0].T * gain), 48_000, subtype="FLOAT")
    report = {
        "scope": "FP Text Encoder and DiT latents; W4+HQQ VAE only, not all-W4",
        "fp_model_bytes": fp_model.stat().st_size if fp is not None else None,
        "w4_model_bytes": w4_model.stat().st_size,
        "w4_to_fp_size_ratio": w4_model.stat().st_size / fp_model.stat().st_size if fp is not None else None,
        "fp_inference_seconds": fp_seconds,
        "w4_inference_seconds": w4_seconds,
        "fp_vs_pytorch": metrics(fp, reference) if fp is not None else None,
        "w4_vs_pytorch": metrics(w4, reference),
        "w4_vs_fp": metrics(w4, fp) if fp is not None else None,
        "w4_raw_peak": float(np.max(np.abs(w4))),
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
