"""Compare real Text Encoder MNN W4 output with the captured ACE-Step call."""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import MNN
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=ROOT / "models" / "reference" / "text_encoder")
    parser.add_argument("--w4-model", type=Path, default=ROOT / "models" / "mnn" / "text_encoder_w4.mnn")
    parser.add_argument("--fp-model", type=Path, default=ROOT / "models" / "mnn" / "text_encoder_fp.mnn")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "w4-sensitivity" / "text_encoder_w4.json")
    args = parser.parse_args()
    capture = args.capture
    ids = torch.load(capture / "inputs.pt", map_location="cpu", weights_only=True)["inputs.kwargs.input_ids"].numpy().astype(np.int32)
    golden = torch.load(capture / "outputs.pt", map_location="cpu", weights_only=True)["outputs"].numpy()
    report = {"scope": "Qwen3 Text Encoder MNN FP vs W4+HQQ block64, actual captured prompt", "models": {}}
    for variant, model_path in (("fp", args.fp_model), ("w4", args.w4_model)):
        if not model_path.is_file():
            continue
        interpreter = MNN.Interpreter(str(model_path))
        session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
        source = interpreter.getSessionInput(session)
        source.copyFrom(MNN.Tensor(ids.shape, MNN.Halide_Type_Int, ids, MNN.Tensor_DimensionType_Caffe))
        started = time.perf_counter()
        interpreter.runSession(session)
        elapsed = time.perf_counter() - started
        target = interpreter.getSessionOutput(session)
        candidate = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()
        diff = candidate.astype(np.float64) - golden.astype(np.float64)
        report["models"][variant] = {
            "output_shape": list(candidate.shape),
            "relative_l2": float(np.linalg.norm(diff.ravel()) / np.linalg.norm(golden.astype(np.float64).ravel())),
            "max_absolute_error": float(np.max(np.abs(diff))),
            "inference_seconds": elapsed,
            "mnn_header_bytes": model_path.stat().st_size,
            "mnn_weight_bytes": model_path.with_suffix(model_path.suffix + ".weight").stat().st_size,
        }
        del target, source, session, interpreter
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
