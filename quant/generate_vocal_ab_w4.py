"""Generate all eight 30-second vocal FP/W4 component combinations on MNN CPU."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import MNN
import numpy as np
import soundfile as sf
import torch

from generate_ab_w4 import audio_stats, dit_mnn_forward, mnn_session, text_mnn_forward


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")
FRAMES = (192, 256, 174)


def vae_mnn_forward(interpreter, session, latent: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    data = latent.detach().cpu().numpy().astype(np.float32)
    source = interpreter.getSessionInput(session)
    if tuple(source.getShape()) != tuple(data.shape):
        raise RuntimeError(f"VAE W4 MNN shape {source.getShape()} vs {data.shape}")
    source.copyFrom(MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe))
    interpreter.runSession(session)
    target = interpreter.getSessionOutput(session)
    audio = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()
    if audio.ndim == 4 and audio.shape[-1] == 1:
        audio = audio[..., 0]
    if audio.shape != tuple(original.shape):
        raise RuntimeError(f"VAE W4 output shape {audio.shape} vs {tuple(original.shape)}")
    return torch.from_numpy(audio).to(original.device, original.dtype)


def main() -> None:
    torch.set_num_threads(8)
    upstream = ROOT / "upstream" / "ACE-Step-1.5"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(upstream / "checkpoints")
    sys.path.insert(0, str(upstream))
    from acestep import model_downloader
    from acestep.audio_utils import normalize_audio
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    reference = json.loads((ROOT / "outputs" / "vocal-30s-v2" / "reference_result.json").read_text(encoding="utf-8"))
    required = {"acestep-v15-turbo": "model.safetensors", "Qwen3-Embedding-0.6B": "model.safetensors", "vae": "diffusion_pytorch_model.safetensors"}
    handler = AceStepHandler()
    with patch.object(model_downloader, "MAIN_MODEL_COMPONENTS", list(required)):
        status, success = handler.initialize_service(project_root=str(upstream), config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False)
    if not success:
        raise RuntimeError(status)

    mnn_dir = ROOT / "models" / "mnn" / "vocal-30s-v2"
    text_mnn = mnn_session(mnn_dir / "text_encoder_w4.mnn")
    dit_mnn = mnn_session(mnn_dir / "dit_w4.mnn")
    vae_mnn = {frames: mnn_session(mnn_dir / f"vae_decoder_{frames}_w4.mnn") for frames in FRAMES}

    params = GenerationParams(
        caption=reference["caption"], lyrics=reference["lyrics"], instrumental=False,
        vocal_language="zh", duration=30, seed=reference["seed"], inference_steps=8,
        thinking=False, use_cot_metas=False, use_cot_caption=False,
        use_cot_language=False, dcw_enabled=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[reference["seed"]], audio_format="wav")
    DEST.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "vocal-30s-v2-ab"
    work.mkdir(parents=True, exist_ok=True)
    report_path = DEST / "ace-step-vocal30-ab-mnn-report.json"
    report = {
        "scope": "Same-seed 30-second Chinese pop vocal A/B, Windows CPU",
        "seed": reference["seed"], "caption": reference["caption"], "lyrics": reference["lyrics"],
        "duration_seconds": 30, "steps": 8,
        "quantization": "MNN weight-only W4+HQQ block64; all component variants use 30-second static shapes",
        "warning": "Voice and audio quality are not guaranteed by successful generation. Not iPhone Metal validated.",
        "combinations": [],
    }

    captured_latents: list[torch.Tensor] = []
    original_tiled_decode = handler.tiled_decode

    def capture_tiled_decode(latents, *args, **kwargs):
        captured_latents.append(latents.detach().cpu().contiguous().clone())
        return original_tiled_decode(latents, *args, **kwargs)

    handler.tiled_decode = capture_tiled_decode

    def text_post_hook(_module, args, kwargs, output):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or not hasattr(output, "last_hidden_state"):
            raise RuntimeError("Unexpected Text Encoder contract")
        replacement = text_mnn_forward(*text_mnn, ids)
        output.last_hidden_state = replacement.to(output.last_hidden_state.dtype)
        return output

    def dit_post_hook(_module, _args, kwargs, output):
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise RuntimeError("Unexpected DiT contract")
        replacement = dit_mnn_forward(*dit_mnn, kwargs, output[0])
        return (replacement, *output[1:])

    def vae_post_hook(_module, args, _kwargs, output):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Unexpected VAE contract")
        frames = args[0].shape[-1]
        if frames not in vae_mnn:
            raise RuntimeError(f"No VAE W4 model for {frames} latent frames")
        return vae_mnn_forward(*vae_mnn[frames], args[0], output)

    try:
        for dit_w4 in (False, True):
            dit_handle = handler.model.decoder.register_forward_hook(dit_post_hook, with_kwargs=True) if dit_w4 else None
            try:
                for text_w4 in (False, True):
                    text_handle = handler.text_encoder.register_forward_hook(text_post_hook, with_kwargs=True) if text_w4 else None
                    label = f"text-{'w4' if text_w4 else 'fp'}_dit-{'w4' if dit_w4 else 'fp'}"
                    print(f"Generating {label}", flush=True)
                    captured_latents.clear()
                    case_dir = work / label
                    case_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        result = generate_music(handler, None, params, config, save_dir=str(case_dir))
                    finally:
                        if text_handle:
                            text_handle.remove()
                    if not result.success or not result.audios:
                        raise RuntimeError(f"{label}: {result.error or result.status_message}")
                    if len(captured_latents) != 1:
                        raise RuntimeError(f"{label}: expected one tiled VAE decode, got {len(captured_latents)}")
                    sources = list(case_dir.glob("*.wav"))
                    if len(sources) != 1:
                        raise RuntimeError(f"{label}: expected one generated WAV, got {len(sources)}")
                    fp_path = DEST / f"ace-step-vocal30_{label}_vae-fp.wav"
                    shutil.copy2(sources[0], fp_path)
                    report["combinations"].append({"text": "w4" if text_w4 else "fp", "dit": "w4" if dit_w4 else "fp", "vae": "fp", "file": str(fp_path), **audio_stats(fp_path)})
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                    vae_handle = handler.vae.decoder.register_forward_hook(vae_post_hook, with_kwargs=True)
                    try:
                        with torch.inference_mode():
                            wav = original_tiled_decode(captured_latents[0], chunk_size=256, overlap=64, offload_wav_to_cpu=True)
                    finally:
                        vae_handle.remove()
                    if wav.ndim != 3 or wav.shape[:2] != (1, 2):
                        raise RuntimeError(f"Unexpected tiled W4 VAE output shape {tuple(wav.shape)}")
                    raw = wav[0].detach().cpu().numpy()
                    normalized = normalize_audio(raw, params.normalization_db)
                    w4_path = DEST / f"ace-step-vocal30_{label}_vae-w4.wav"
                    sf.write(w4_path, normalized.T, 48_000, subtype="FLOAT")
                    report["combinations"].append({"text": "w4" if text_w4 else "fp", "dit": "w4" if dit_w4 else "fp", "vae": "w4", "file": str(w4_path), **audio_stats(w4_path)})
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"Completed {label}: FP and W4 VAE", flush=True)
                    captured_latents.clear()
                    gc.collect()
            finally:
                if dit_handle:
                    dit_handle.remove()
    finally:
        handler.tiled_decode = original_tiled_decode
    print(json.dumps({"completed": len(report["combinations"]), "report": str(report_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
