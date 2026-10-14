"""Validate real MNN DiT W4 on the captured first denoising step."""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import MNN
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "models" / "reference" / "dit"
NAMES = ("hidden_states", "timestep", "timestep_r", "encoder_hidden_states", "context_latents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--w4-model", type=Path, default=ROOT / "models" / "mnn" / "dit_w4.mnn")
    parser.add_argument("--fp-model", type=Path, default=ROOT / "models" / "mnn" / "dit_fp.mnn")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "w4-sensitivity" / "dit_mnn_validation.json")
    args = parser.parse_args()
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    golden = torch.load(args.capture / "outputs.pt", map_location="cpu", weights_only=True)["outputs"].numpy()
    report = {"scope": "MNN CPU DiT W4+HQQ block64 on first real denoising call", "models": {}}
    for variant, path in (("w4", args.w4_model), ("fp", args.fp_model)):
        if not path.is_file():
            continue
        interpreter = MNN.Interpreter(str(path))
        session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
        for name in NAMES:
            array = inputs[f"inputs.kwargs.{name}"].numpy().astype(np.float32)
            source = interpreter.getSessionInput(session, name)
            if tuple(source.getShape()) != tuple(array.shape):
                raise RuntimeError(f"Input shape mismatch {name}: MNN {source.getShape()} vs {array.shape}")
            source.copyFrom(MNN.Tensor(array.shape, MNN.Halide_Type_Float, array, MNN.Tensor_DimensionType_Caffe))
        started = time.perf_counter()
        interpreter.runSession(session)
        elapsed = time.perf_counter() - started
        target = interpreter.getSessionOutput(session)
        candidate = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()
        if candidate.size != golden.size:
            raise RuntimeError(f"Output size mismatch: {candidate.shape} vs {golden.shape}")
        candidate = candidate.reshape(golden.shape)
        difference = candidate.astype(np.float64) - golden.astype(np.float64)
        report["models"][variant] = {
            "relative_l2": float(np.linalg.norm(difference.ravel()) / np.linalg.norm(golden.astype(np.float64).ravel())),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "inference_seconds": elapsed,
            "mnn_total_bytes": path.stat().st_size + (path.with_suffix(path.suffix + ".weight").stat().st_size if path.with_suffix(path.suffix + ".weight").exists() else 0),
        }
        del target, session, interpreter
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
