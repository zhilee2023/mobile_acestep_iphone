"""Exercise the v28 two-window continuation on Windows MNN CPU.

This validates the mobile window geometry and decoder schedule, not iPhone
Metal performance or whether the repeated fixed lyrics are sung completely.
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
OUT = ROOT / "outputs" / "ios-repaint-validation-v28"
TIMESTEPS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125)
FIRST_FRAMES = 750
CONTEXT_FRAMES = 125
TOTAL_FRAMES = 1375
UPSAMPLE = 1920


def create_session(path: Path, precision: str):
    assert path.is_file(), path
    net = MNN.Interpreter(str(path))
    handle = net.createSession({"backend": "CPU", "precision": precision, "memory": "low"})
    return net, handle


def put(net, handle, name: str, values: np.ndarray) -> None:
    target = net.getSessionInput(handle, name)
    assert tuple(target.getShape()) == values.shape, (name, target.getShape(), values.shape)
    source = MNN.Tensor(values.shape, MNN.Halide_Type_Float, values,
                        MNN.Tensor_DimensionType_Caffe)
    target.copyFrom(source)


def get(net, handle) -> np.ndarray:
    target = net.getSessionOutput(handle)
    return np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()


def load(name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(PROBE / name, dtype="<f4")
    assert values.size == math.prod(shape), (name, values.size, shape)
    return values.reshape(shape)


def main() -> None:
    first = load("vae-final-latent.f32", (1, FIRST_FRAMES, 64))
    initial = load("dit-hidden.f32", (1, FIRST_FRAMES, 64))
    encoder = load("dit-encoder.f32", (1, 181, 2048))
    context = load("dit-context.f32", (1, FIRST_FRAMES, 128)).copy()
    assert np.all(context[:, :, 64:] == 1.0)
    noise = np.roll(initial, -257, axis=1).copy()
    second = noise.copy()
    context[:, :CONTEXT_FRAMES, :64] = first[:, -CONTEXT_FRAMES:, :]
    context[:, :CONTEXT_FRAMES, 64:] = 0.0

    dit, dit_handle = create_session(MODELS / "dit_w4.mnn", "high")
    for index, current in enumerate(TIMESTEPS):
        put(dit, dit_handle, "encoder_hidden_states", encoder)
        put(dit, dit_handle, "context_latents", context)
        put(dit, dit_handle, "hidden_states", second)
        t = np.array([current], dtype=np.float32)
        put(dit, dit_handle, "timestep", t)
        put(dit, dit_handle, "timestep_r", t)
        dit.runSession(dit_handle)
        velocity = get(dit, dit_handle)
        assert velocity.shape == second.shape and np.isfinite(velocity).all()
        next_t = TIMESTEPS[index + 1] if index + 1 < len(TIMESTEPS) else 0.0
        second -= velocity * np.float32(current - next_t)
        if index < len(TIMESTEPS) // 2:
            second[:, :CONTEXT_FRAMES, :] = (
                np.float32(next_t) * noise[:, :CONTEXT_FRAMES, :]
                + np.float32(1.0 - next_t) * first[:, -CONTEXT_FRAMES:, :]
            )
        assert np.isfinite(second).all(), index
        print(f"second-window DiT {index + 1}/8: latent peak {np.max(np.abs(second)):.4f}",
              flush=True)
    del dit, dit_handle

    combined = np.concatenate((first, second[:, CONTEXT_FRAMES:, :]), axis=1)
    assert combined.shape == (1, TOTAL_FRAMES, 64)
    vae, vae_handle = create_session(MODELS / "vae_decoder_192_w4.mnn", "low")
    waveform = np.zeros((2, TOTAL_FRAMES * UPSAMPLE), dtype=np.float32)
    for index, core_start in enumerate(range(0, TOTAL_FRAMES, 128)):
        core_end = min(core_start + 128, TOTAL_FRAMES)
        win_start = min(max(core_start - 32, 0), TOTAL_FRAMES - 192)
        latents = combined[:, win_start:win_start + 192, :].transpose(0, 2, 1).copy()
        put(vae, vae_handle, "latents", latents)
        vae.runSession(vae_handle)
        decoded = get(vae, vae_handle)[0, :, :, 0]
        offset = (core_start - win_start) * UPSAMPLE
        count = (core_end - core_start) * UPSAMPLE
        core = decoded[:, offset:offset + count]
        assert np.isfinite(core).all()
        waveform[:, core_start * UPSAMPLE:core_end * UPSAMPLE] = core
        print(f"VAE {index + 1}/11: {core_end / 25:.2f} s decoded", flush=True)
    del vae, vae_handle

    seam = FIRST_FRAMES * UPSAMPLE
    jump = np.abs(waveform[:, seam] - waveform[:, seam - 1])
    edge_left = waveform[:, seam - 48000:seam]
    edge_right = waveform[:, seam:seam + 48000]
    peak = float(np.max(np.abs(waveform)))
    assert peak > 1e-6 and np.isfinite(peak)
    normalized = waveform * np.float32(10 ** (-1.0 / 20.0) / peak)
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "repaint-55s-local-cpu.wav"
    sf.write(wav, normalized.T, 48000, subtype="PCM_16")
    report = {
        "audio_seconds": TOTAL_FRAMES / 25,
        "samples_per_channel": waveform.shape[1],
        "second_window_peak": float(np.max(np.abs(second))),
        "waveform_peak_before_normalization": peak,
        "seam_single_sample_jump": jump.tolist(),
        "seam_left_1s_rms": float(np.sqrt(np.mean(edge_left ** 2))),
        "seam_right_1s_rms": float(np.sqrt(np.mean(edge_right ** 2))),
        "wav": str(wav),
        "limitations": "CPU numerics only; fixed lyrics repeated; no iPhone Metal or lyric-completion validation",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
