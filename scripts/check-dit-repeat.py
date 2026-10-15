"""Check whether one W4 DiT MNN session repeats identical inputs consistently."""

from __future__ import annotations

import json
from pathlib import Path

import MNN
import numpy as np


root = Path(__file__).resolve().parents[1]
probe = root / "ios" / "Probe"
shapes = json.loads((probe / "manifest.json").read_text())["shapes"]


def fixture(name: str, filename: str) -> np.ndarray:
    return np.fromfile(probe / filename, dtype="<f4").reshape(shapes[name])


net = MNN.Interpreter(str(root / "models/mnn/vocal-30s-v2/dit_w4.mnn"))
session = net.createSession({"backend": "CPU", "precision": "high", "memory": "low"})


def put(name: str, values: np.ndarray) -> None:
    tensor = net.getSessionInput(session, name)
    host = MNN.Tensor(values.shape, MNN.Halide_Type_Float, values, MNN.Tensor_DimensionType_Caffe)
    tensor.copyFrom(host)


def run() -> np.ndarray:
    net.runSession(session)
    return np.asarray(net.getSessionOutput(session).getNumpyData(), dtype=np.float32).copy()


put("encoder_hidden_states", fixture("encoder_hidden_states", "dit-encoder.f32"))
put("context_latents", fixture("context_latents", "dit-context.f32"))
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
first = run()
second = run()
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
third = run()
put("encoder_hidden_states", fixture("encoder_hidden_states", "dit-encoder.f32"))
put("context_latents", fixture("context_latents", "dit-context.f32"))
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
fourth = run()
put("encoder_hidden_states", fixture("encoder_hidden_states", "dit-encoder.f32"))
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
fifth = run()
put("encoder_hidden_states", fixture("encoder_hidden_states", "dit-encoder.f32"))
put("context_latents", fixture("context_latents", "dit-context.f32"))
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
sixth = run()
put("context_latents", fixture("context_latents", "dit-context.f32"))
put("hidden_states", fixture("hidden_states", "dit-hidden.f32"))
put("timestep", np.array([1.0], dtype=np.float32))
put("timestep_r", np.array([1.0], dtype=np.float32))
seventh = run()
for name, current in (("first", first), ("second-same-input", second),
                      ("third-recopy-dynamic", third), ("fourth-recopy-all", fourth),
                      ("fifth-recopy-encoder", fifth), ("sixth-recopy-all", sixth),
                      ("seventh-recopy-context", seventh)):
    print(name, "peak", np.max(np.abs(current)),
          "L2-to-first", np.linalg.norm(current - first) / np.linalg.norm(first), flush=True)
