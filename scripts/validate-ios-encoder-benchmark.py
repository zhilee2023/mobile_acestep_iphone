"""Sanity-check the packaged encoder input and W4 graph on Windows CPU."""

from __future__ import annotations

from pathlib import Path
import time

import MNN
import numpy as np


root = Path(__file__).resolve().parents[1]
ids = np.fromfile(root / "ios/Probe/encoder-input-ids.i64", dtype="<i8").reshape(1, 107)
assert np.all((ids >= np.iinfo(np.int32).min) & (ids <= np.iinfo(np.int32).max))
ids = ids.astype(np.int32)
model = root / "models/mnn/vocal-30s-v2/text_encoder_w4.mnn"
net = MNN.Interpreter(str(model))
session = net.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
output = None
for index in range(2):
    tensor = net.getSessionInput(session, "input_ids")
    host = MNN.Tensor(ids.shape, MNN.Halide_Type_Int, ids, MNN.Tensor_DimensionType_Caffe)
    assert tensor.copyFrom(host)
    started = time.perf_counter()
    net.runSession(session)
    current = np.asarray(net.getSessionOutput(session, "last_hidden_state").getNumpyData(),
                         dtype=np.float32).copy().reshape(1, 107, 1024)
    seconds = time.perf_counter() - started
    assert np.isfinite(current).all()
    print(f"CPU encoder run {index + 1}: {seconds:.3f}s, peak={np.max(np.abs(current)):.6f}", flush=True)
    if output is not None:
        print(f"repeat relative L2: {np.linalg.norm(current - output) / np.linalg.norm(output):.6g}", flush=True)
    output = current
