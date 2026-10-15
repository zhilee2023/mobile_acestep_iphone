"""Capture two CPU/High W4 DiT outputs for the iPhone Metal diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import MNN
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "ios" / "Probe"
MODEL = ROOT / "models" / "mnn" / "vocal-30s-v2" / "dit_w4.mnn"


def main() -> None:
    shapes = json.loads((PROBE / "manifest.json").read_text())["shapes"]

    def fixture(name: str, filename: str) -> np.ndarray:
        return np.fromfile(PROBE / filename, dtype="<f4").reshape(shapes[name])

    net = MNN.Interpreter(str(MODEL))
    session = net.createSession({"backend": "CPU", "precision": "high", "memory": "low"})

    def put(name: str, values: np.ndarray) -> None:
        tensor = net.getSessionInput(session, name)
        assert tuple(tensor.getShape()) == tuple(values.shape)
        host = MNN.Tensor(values.shape, MNN.Halide_Type_Float, values, MNN.Tensor_DimensionType_Caffe)
        tensor.copyFrom(host)

    put("encoder_hidden_states", fixture("encoder_hidden_states", "dit-encoder.f32"))
    put("context_latents", fixture("context_latents", "dit-context.f32"))
    hidden = fixture("hidden_states", "dit-hidden.f32").copy()
    for index, time in enumerate((1.0, 0.875), start=1):
        put("hidden_states", hidden)
        put("timestep", np.array([time], dtype=np.float32))
        put("timestep_r", np.array([time], dtype=np.float32))
        net.runSession(session)
        output = net.getSessionOutput(session)
        velocity = np.asarray(output.getNumpyData(), dtype=np.float32).reshape(shapes["denoised"]).copy()
        assert np.isfinite(velocity).all()
        velocity.astype("<f4", copy=False).tofile(PROBE / f"dit-w4-cpu-step{index}.f32")
        hidden -= velocity * np.float32(0.125)
        print(f"step {index}: output peak={np.max(np.abs(velocity)):.8g}; latent peak={np.max(np.abs(hidden)):.8g}", flush=True)


if __name__ == "__main__":
    main()
