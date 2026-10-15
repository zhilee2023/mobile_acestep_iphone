"""Numerically check the 73-token lyric W4 graph against the fixed song."""

from __future__ import annotations

import json
from pathlib import Path

import MNN
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "models/onnx/vocal-30s-v2/lyric_encoder_73_validation.json"
MODEL = ROOT / "models/mnn/vocal-30s-v2/lyric_encoder_73_w4.mnn"
TABLE = ROOT / "models/mnn/vocal-30s-v2/lyric_embedding.f16"
GOLDEN = ROOT / "ios/Probe/dit-encoder.f32"


def main() -> None:
    ids = np.asarray(json.loads(REPORT.read_text(encoding="utf-8"))["token_ids"], dtype=np.int64)
    assert ids.shape == (73,) and ids.min() >= 0 and ids.max() < 151669
    table = np.memmap(TABLE, dtype="<f2", mode="r", shape=(151669, 1024))
    embeddings = np.asarray(table[ids], dtype=np.float32).reshape(1, 73, 1024)
    golden = np.fromfile(GOLDEN, dtype="<f4").reshape(1, 181, 2048)[:, :73, :]

    interpreter = MNN.Interpreter(str(MODEL))
    session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    input_tensor = interpreter.getSessionInput(session, "lyric_embeddings")
    assert tuple(input_tensor.getShape()) == embeddings.shape
    host = MNN.Tensor(embeddings.shape, MNN.Halide_Type_Float, embeddings, MNN.Tensor_DimensionType_Caffe)
    assert input_tensor.copyFrom(host)
    interpreter.runSession(session)
    actual = np.asarray(interpreter.getSessionOutput(session, "lyric_hidden_states").getNumpyData(),
                        dtype=np.float32).reshape(1, 73, 2048).copy()
    assert np.isfinite(actual).all()
    relative_l2 = np.linalg.norm((actual - golden).astype(np.float64)) / np.linalg.norm(golden.astype(np.float64))
    print(f"lyric W4 CPU vs captured DiT condition: relative L2={relative_l2:.8g}, peak={np.max(np.abs(actual)):.6g}")
    if relative_l2 >= 0.3:
        raise RuntimeError("Lyric W4 error is too high to proceed to an iPhone pilot")


if __name__ == "__main__":
    main()
