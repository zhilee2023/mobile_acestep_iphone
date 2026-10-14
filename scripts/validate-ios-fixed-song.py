"""Validate the fixed-condition 30-second MNN path before the iOS port.

Only the captured first-step conditioning and noise are reused. DiT and tiled
VAE inference run for real here on Windows CPU; the iOS implementation must
repeat those graph executions on device before claiming on-device generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import MNN
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "ios" / "Probe"
MODELS = ROOT / "models" / "mnn" / "vocal-30s-v2"
OUT = ROOT / "outputs" / "ios-fixed-song-validation"
TIMESTEPS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]


def session(path: Path):
    interpreter = MNN.Interpreter(str(path))
    return interpreter, interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})


def write_input(interpreter, handle, name: str, values: np.ndarray) -> None:
    tensor = interpreter.getSessionInput(handle, name)
    assert tuple(tensor.getShape()) == tuple(values.shape), (name, tensor.getShape(), values.shape)
    source = MNN.Tensor(values.shape, MNN.Halide_Type_Float, values, MNN.Tensor_DimensionType_Caffe)
    tensor.copyFrom(source)


def read_output(interpreter, handle) -> np.ndarray:
    tensor = interpreter.getSessionOutput(handle)
    return np.asarray(tensor.getNumpyData(), dtype=np.float32).reshape(tensor.getShape()).copy()


def main() -> None:
    data = json.loads((PROBE / "manifest.json").read_text())
    shapes = data["shapes"]
    load = lambda name, filename: np.fromfile(PROBE / filename, dtype="<f4").reshape(shapes[name])
    hidden = load("hidden_states", "dit-hidden.f32")
    encoder = load("encoder_hidden_states", "dit-encoder.f32")
    context = load("context_latents", "dit-context.f32")

    dit, dit_handle = session(MODELS / "dit_w4.mnn")
    write_input(dit, dit_handle, "encoder_hidden_states", encoder)
    write_input(dit, dit_handle, "context_latents", context)
    xt = hidden.copy()
    first_step_error = None
    for index, current in enumerate(TIMESTEPS):
        write_input(dit, dit_handle, "hidden_states", xt)
        write_input(dit, dit_handle, "timestep", np.array([current], dtype=np.float32))
        write_input(dit, dit_handle, "timestep_r", np.array([current], dtype=np.float32))
        dit.runSession(dit_handle)
        velocity = read_output(dit, dit_handle)
        if index == 0:
            golden = load("denoised", "dit-golden.f32")
            first_step_error = float(np.linalg.norm(velocity - golden) / np.linalg.norm(golden))
        next_time = TIMESTEPS[index + 1] if index + 1 < len(TIMESTEPS) else 0.0
        xt -= velocity * np.float32(current - next_time)
        print(f"DiT {index + 1}/{len(TIMESTEPS)}: t={current:.6f}", flush=True)

    latents = np.transpose(xt, (0, 2, 1)).copy()
    del dit, dit_handle
    output = np.zeros((2, 750 * 1920), dtype=np.float32)
    stride, overlap = 128, 64
    sessions = {}
    written = 0
    for index in range(math.ceil(750 / stride)):
        core_start = index * stride
        core_end = min(core_start + stride, 750)
        win_start = max(0, core_start - overlap)
        win_end = min(750, core_end + overlap)
        frames = win_end - win_start
        if frames not in sessions:
            sessions[frames] = session(MODELS / f"vae_decoder_{frames}_w4.mnn")
        vae, handle = sessions[frames]
        write_input(vae, handle, "latents", latents[:, :, win_start:win_end].copy())
        vae.runSession(handle)
        audio = read_output(vae, handle)[0, :, :, 0]
        start = (core_start - win_start) * 1920
        end = audio.shape[-1] - (win_end - core_end) * 1920
        core = audio[:, start:end]
        output[:, written:written + core.shape[-1]] = core
        written += core.shape[-1]
        print(f"VAE {index + 1}/6: frames={frames}, samples={core.shape[-1]}", flush=True)
    assert written == 1_440_000, written
    peak = float(np.max(np.abs(output)))
    if peak > 1e-6:
        output *= np.float32(10 ** (-1.0 / 20.0) / peak)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "fixed-vocal-mnn-cpu.wav"
    sf.write(target, output.T, 48_000, subtype="FLOAT")
    report = {"first_step_relative_l2": first_step_error, "samples": written,
              "sample_rate": 48000, "duration_seconds": 30, "peak": float(np.max(np.abs(output))),
              "wav": str(target)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
