"""Run an unquantized, matched 55-second two-window ACE-Step control on CPU.

The window geometry, fixed prompt/lyrics, noise rotation, Repaint injection,
and 192-frame VAE schedule match the v28 W4 experiment. This remains a
continuity test: the second window receives the same lyrics, not the
automatically determined unsung remainder.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ACE-Step-1.5"
OUT = ROOT / "outputs" / "ios-repaint-fp55"
USER_OUT = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")
FIRST_FRAMES = 750
CONTEXT_FRAMES = 125
TOTAL_FRAMES = 1375
UPSAMPLE = 1920
TIMESTEPS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def main() -> None:
    torch.set_num_threads(8)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(UPSTREAM / "checkpoints")
    sys.path.insert(0, str(UPSTREAM))
    from acestep import model_downloader
    from acestep.audio_utils import normalize_audio
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    reference = json.loads((ROOT / "outputs/vocal-30s-v2/reference_result.json").read_text(encoding="utf-8"))
    assert reference["duration_seconds"] == 30 and reference["seed"] == 17
    required = {
        "acestep-v15-turbo": "model.safetensors",
        "Qwen3-Embedding-0.6B": "model.safetensors",
        "vae": "diffusion_pytorch_model.safetensors",
    }
    handler = AceStepHandler()
    print("Loading official FP model on Windows CPU...", flush=True)
    with patch.object(model_downloader, "MAIN_MODEL_COMPONENTS", list(required)):
        status, success = handler.initialize_service(
            project_root=str(UPSTREAM), config_path="acestep-v15-turbo",
            device="cpu", offload_to_cpu=False,
        )
    if not success:
        raise RuntimeError(status)

    captured: dict[str, torch.Tensor] = {}
    first_latent: list[torch.Tensor] = []
    original_decode = handler.tiled_decode

    def capture_decoder(_module, _args, kwargs):
        if captured:
            return
        for name in ("hidden_states", "encoder_hidden_states", "context_latents"):
            value = kwargs.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Missing FP decoder input {name}")
            captured[name] = value.detach().cpu().float().contiguous().clone()

    def capture_decode(latents, *args, **kwargs):
        first_latent.append(latents.detach().cpu().float().contiguous().clone())
        return original_decode(latents, *args, **kwargs)

    params = GenerationParams(
        caption=reference["caption"], lyrics=reference["lyrics"], instrumental=False,
        vocal_language="zh", duration=30, seed=17, inference_steps=8,
        thinking=False, use_cot_metas=False, use_cot_caption=False,
        use_cot_language=False, dcw_enabled=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[17], audio_format="wav")
    OUT.mkdir(parents=True, exist_ok=True)
    hook = handler.model.decoder.register_forward_pre_hook(capture_decoder, with_kwargs=True)
    handler.tiled_decode = capture_decode
    started = time.monotonic()
    try:
        print("Running first 30-second official FP window...", flush=True)
        result = generate_music(handler, None, params, config, save_dir=str(OUT / "first-window"))
    finally:
        hook.remove()
        handler.tiled_decode = original_decode
    if not result.success or not result.audios or len(first_latent) != 1:
        raise RuntimeError(result.error or result.status_message or "First FP window did not complete")
    first = first_latent[0].transpose(1, 2).contiguous()
    if first.shape != (1, FIRST_FRAMES, 64):
        raise RuntimeError(f"Unexpected first latent shape {tuple(first.shape)}")
    assert captured["hidden_states"].shape == first.shape
    assert captured["encoder_hidden_states"].shape == (1, 181, 2048)
    assert captured["context_latents"].shape == (1, FIRST_FRAMES, 128)
    print(f"First FP window complete in {time.monotonic() - started:.1f}s", flush=True)

    original_noise = captured["hidden_states"]
    encoder = captured["encoder_hidden_states"]
    context = captured["context_latents"].clone()
    if not torch.allclose(context[:, :, 64:], torch.ones_like(context[:, :, 64:])):
        raise RuntimeError("Original context is not an all-generate mask")
    context[:, :CONTEXT_FRAMES, :64] = first[:, -CONTEXT_FRAMES:, :]
    context[:, :CONTEXT_FRAMES, 64:] = 0.0
    noise = torch.roll(original_noise, shifts=-257, dims=1).clone()
    state = noise.clone()
    for index, current in enumerate(TIMESTEPS):
        now = time.monotonic()
        t = torch.tensor([current], dtype=state.dtype)
        with torch.inference_mode():
            velocity = handler.model.decoder(
                hidden_states=state, timestep=t, timestep_r=t,
                attention_mask=None, encoder_hidden_states=encoder,
                encoder_attention_mask=None, context_latents=context,
                use_cache=False, past_key_values=None,
            )[0]
            next_t = TIMESTEPS[index + 1] if index + 1 < len(TIMESTEPS) else 0.0
            state = state - velocity * (current - next_t)
            if index < len(TIMESTEPS) // 2:
                state[:, :CONTEXT_FRAMES, :] = (
                    next_t * noise[:, :CONTEXT_FRAMES, :]
                    + (1.0 - next_t) * first[:, -CONTEXT_FRAMES:, :]
                )
        if not torch.isfinite(state).all():
            raise RuntimeError(f"FP second window step {index + 1} produced non-finite latent")
        print(f"Second FP DiT {index + 1}/8: {time.monotonic() - now:.1f}s, "
              f"peak {state.abs().max().item():.4f}", flush=True)

    combined = torch.cat((first, state[:, CONTEXT_FRAMES:, :]), dim=1)
    assert combined.shape == (1, TOTAL_FRAMES, 64)
    waveform = np.zeros((2, TOTAL_FRAMES * UPSAMPLE), dtype=np.float32)
    for index, core_start in enumerate(range(0, TOTAL_FRAMES, 128)):
        now = time.monotonic()
        core_end = min(core_start + 128, TOTAL_FRAMES)
        win_start = min(max(core_start - 32, 0), TOTAL_FRAMES - 192)
        latents = combined[:, win_start:win_start + 192, :].transpose(1, 2).contiguous()
        with torch.inference_mode():
            decoded = handler.vae.decode(latents.to(handler.vae.dtype)).sample
        audio = decoded[0].detach().cpu().float().numpy()
        offset = (core_start - win_start) * UPSAMPLE
        count = (core_end - core_start) * UPSAMPLE
        core = audio[:, offset:offset + count]
        if core.shape != (2, count) or not np.isfinite(core).all():
            raise RuntimeError(f"FP VAE chunk {index + 1} invalid")
        waveform[:, core_start * UPSAMPLE:core_end * UPSAMPLE] = core
        print(f"FP VAE {index + 1}/11: {time.monotonic() - now:.1f}s, "
              f"decoded {core_end / 25:.2f}s", flush=True)

    normalized = normalize_audio(waveform, -1.0)
    target = USER_OUT / "ACE-Step-v28-55s-local-FP.wav"
    USER_OUT.mkdir(parents=True, exist_ok=True)
    sf.write(target, normalized.T, 48000, subtype="PCM_16")
    seam = FIRST_FRAMES * UPSAMPLE
    w4_target = USER_OUT / "ACE-Step-v28-55s-local-CPU.wav"
    w4, w4_sr = sf.read(w4_target, dtype="float32", always_2d=True)
    if w4_sr != 48000 or w4.shape != (TOTAL_FRAMES * UPSAMPLE, 2):
        raise RuntimeError("Matched W4 55-second WAV has unexpected shape")
    w4 = w4.T
    correlation = float(np.corrcoef(normalized.reshape(-1), w4.reshape(-1))[0, 1])
    report = {
        "scope": "same seed/prompt/lyrics/window geometry; FP PyTorch CPU versus W4 MNN CPU",
        "seed": 17,
        "audio_seconds": 55.0,
        "first_window_frames": FIRST_FRAMES,
        "preserved_context_frames": CONTEXT_FRAMES,
        "second_window_new_frames": FIRST_FRAMES - CONTEXT_FRAMES,
        "fp_wav": str(target),
        "w4_wav": str(w4_target),
        "fp_unscaled_peak": float(np.max(np.abs(waveform))),
        "fp_seam_single_sample_jump": np.abs(waveform[:, seam] - waveform[:, seam - 1]).tolist(),
        "fp_seam_left_1s_rms": rms(waveform[:, seam - 48000:seam]),
        "fp_seam_right_1s_rms": rms(waveform[:, seam:seam + 48000]),
        "normalized_fp_w4_sample_correlation": correlation,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": "Repeated lyrics; first window is full ACE PyTorch reference. This is not automatic lyric completion or iPhone inference.",
    }
    report_path = USER_OUT / "ACE-Step-v28-55s-FP-vs-W4-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
