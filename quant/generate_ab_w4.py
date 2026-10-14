"""Create eight same-seed 10-second A/B WAVs for FP/W4 combinations.

Text and VAE W4 use MNN. DiT W4 may use either a PyTorch block-64
simulation or the real static MNN model, selected by --dit-backend.
These are audio sensitivity experiments, not an iPhone deployment.
"""

from __future__ import annotations

import gc
import argparse
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

from probe_dit_w4 import quantize_weights


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")


def mnn_session(path: Path):
    interpreter = MNN.Interpreter(str(path))
    session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    return interpreter, session


def text_mnn_forward(interpreter, session, ids: torch.Tensor) -> torch.Tensor:
    ids_np = ids.detach().cpu().numpy().astype(np.int32)
    source = interpreter.getSessionInput(session)
    if tuple(source.getShape()) != tuple(ids_np.shape):
        raise RuntimeError(f"Text W4 MNN shape {source.getShape()} cannot serve token shape {ids_np.shape}")
    source.copyFrom(MNN.Tensor(ids_np.shape, MNN.Halide_Type_Int, ids_np, MNN.Tensor_DimensionType_Caffe))
    interpreter.runSession(session)
    target = interpreter.getSessionOutput(session)
    data = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()
    return torch.from_numpy(data).to(ids.device)


def vae_mnn_decode(interpreter, session, latent: torch.Tensor) -> np.ndarray:
    data = latent.detach().cpu().numpy().astype(np.float32)
    source = interpreter.getSessionInput(session)
    if tuple(source.getShape()) != tuple(data.shape):
        raise RuntimeError(f"VAE W4 MNN shape {source.getShape()} cannot serve latent shape {data.shape}")
    source.copyFrom(MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe))
    interpreter.runSession(session)
    target = interpreter.getSessionOutput(session)
    audio = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(target.getShape()).copy()
    if audio.ndim == 4 and audio.shape[-1] == 1:
        audio = audio[..., 0]
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2:
        raise RuntimeError(f"Unexpected VAE waveform shape: {audio.shape}")
    return audio[0]


def dit_mnn_forward(interpreter, session, kwargs: dict, original: torch.Tensor) -> torch.Tensor:
    for name in ("hidden_states", "timestep", "timestep_r", "encoder_hidden_states", "context_latents"):
        array = kwargs[name].detach().cpu().numpy().astype(np.float32)
        source = interpreter.getSessionInput(session, name)
        if tuple(source.getShape()) != tuple(array.shape):
            raise RuntimeError(f"DiT W4 MNN shape for {name}: {source.getShape()} vs {array.shape}")
        source.copyFrom(MNN.Tensor(array.shape, MNN.Halide_Type_Float, array, MNN.Tensor_DimensionType_Caffe))
    interpreter.runSession(session)
    target = interpreter.getSessionOutput(session)
    result = np.asarray(target.getNumpyData(), dtype=np.float32).reshape(-1).copy()
    if result.size != original.numel():
        raise RuntimeError(f"DiT W4 MNN output has {result.size} values, expected {original.numel()}")
    return torch.from_numpy(result.reshape(original.shape)).to(original.device, original.dtype)


def audio_stats(path: Path) -> dict[str, float | int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return {
        "sample_rate": sr,
        "duration_seconds": float(len(audio) / sr),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-backend", choices=("simulated", "mnn"), default="simulated")
    args = parser.parse_args()
    torch.set_num_threads(8)
    upstream = ROOT / "upstream" / "ACE-Step-1.5"
    checkpoints = upstream / "checkpoints"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
    sys.path.insert(0, str(upstream))
    from acestep import model_downloader
    from acestep.audio_utils import normalize_audio
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    required = {"acestep-v15-turbo": "model.safetensors", "Qwen3-Embedding-0.6B": "model.safetensors", "vae": "diffusion_pytorch_model.safetensors"}
    for folder, name in required.items():
        if not (checkpoints / folder / name).is_file():
            raise FileNotFoundError(checkpoints / folder / name)
    handler = AceStepHandler()
    with patch.object(model_downloader, "MAIN_MODEL_COMPONENTS", list(required)):
        status, success = handler.initialize_service(project_root=str(upstream), config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False)
    if not success:
        raise RuntimeError(status)

    generation_config = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))["generation"]
    params = GenerationParams(
        caption=generation_config["caption"], lyrics=generation_config["lyrics"], instrumental=True,
        duration=generation_config["duration_seconds"], seed=generation_config["seed"],
        inference_steps=generation_config["steps"], thinking=False, use_cot_metas=False,
        use_cot_caption=False, use_cot_language=False, dcw_enabled=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[generation_config["seed"]], audio_format="wav")
    text_interpreter, text_session = mnn_session(ROOT / "models" / "mnn" / "text_encoder_w4.mnn")
    vae_interpreter, vae_session = mnn_session(ROOT / "models" / "mnn" / "vae_decoder_w4.mnn")
    dit_interpreter, dit_session = mnn_session(ROOT / "models" / "mnn" / "dit_w4.mnn") if args.dit_backend == "mnn" else (None, None)
    DEST.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "w4-ab"
    work.mkdir(parents=True, exist_ok=True)
    report_path = DEST / ("ace-step-ab-mnn-report.json" if args.dit_backend == "mnn" else "ace-step-ab-report.json")
    report = {
        "warning": ("All W4 components are MNN W4+HQQ, static 10-second shapes. No iPhone Metal validation." if args.dit_backend == "mnn" else "Text/VAE W4 are MNN W4+HQQ; DiT W4 is PyTorch affine block64 simulation, NOT MNN HQQ. Do not treat this as iPhone deployment validation."),
        "caption": params.caption, "seed": params.seed, "duration_seconds": params.duration,
        "steps": params.inference_steps, "combinations": [],
    }

    captured_latents: list[torch.Tensor] = []

    def vae_pre_hook(_module, args, _kwargs):
        if args and isinstance(args[0], torch.Tensor):
            captured_latents.append(args[0].detach().cpu().contiguous().clone())

    vae_handle = handler.vae.decoder.register_forward_pre_hook(vae_pre_hook, with_kwargs=True)

    def text_post_hook(_module, args, kwargs, output):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or not hasattr(output, "last_hidden_state"):
            raise RuntimeError("Unexpected Text Encoder call contract")
        replacement = text_mnn_forward(text_interpreter, text_session, ids)
        output.last_hidden_state = replacement.to(output.last_hidden_state.dtype)
        return output

    def dit_post_hook(_module, _args, kwargs, output):
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise RuntimeError("Unexpected DiT output contract")
        replacement = dit_mnn_forward(dit_interpreter, dit_session, kwargs, output[0])
        return (replacement, *output[1:])

    try:
        for dit_w4 in (False, True):
            dit_handle = None
            if dit_w4:
                if args.dit_backend == "mnn":
                    dit_handle = handler.model.decoder.register_forward_hook(dit_post_hook, with_kwargs=True)
                else:
                    print("Quantizing complete DiT Linear/Conv weights for PyTorch W4 simulation...", flush=True)
                    report["dit_quantized"] = quantize_weights(handler.model.decoder)
            for text_w4 in (False, True):
                text_handle = handler.text_encoder.register_forward_hook(text_post_hook, with_kwargs=True) if text_w4 else None
                dit_label = ("mnn-w4" if args.dit_backend == "mnn" else "w4") if dit_w4 else "fp"
                label = f"text-{'w4' if text_w4 else 'fp'}_dit-{dit_label}"
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
                    raise RuntimeError(f"{label}: expected one VAE decoder call, found {len(captured_latents)}")
                fp_source = Path(result.audios[0]["path"] if isinstance(result.audios[0], dict) else result.audios[0])
                if not fp_source.is_file():
                    candidates = list(case_dir.glob("*.wav"))
                    if len(candidates) != 1:
                        raise RuntimeError(f"{label}: unable to locate FP WAV: {result.audios}")
                    fp_source = candidates[0]
                fp_path = DEST / f"ace-step-ab_{label}_vae-fp.wav"
                shutil.copy2(fp_source, fp_path)
                report["combinations"].append({"text": "w4" if text_w4 else "fp", "dit": ("mnn_w4" if args.dit_backend == "mnn" else "w4_simulated") if dit_w4 else "fp", "vae": "fp", "file": str(fp_path), **audio_stats(fp_path)})

                raw = vae_mnn_decode(vae_interpreter, vae_session, captured_latents[0])
                normalized = normalize_audio(raw, params.normalization_db)
                w4_path = DEST / f"ace-step-ab_{label}_vae-w4.wav"
                sf.write(w4_path, normalized.T, 48_000, subtype="PCM_16")
                report["combinations"].append({"text": "w4" if text_w4 else "fp", "dit": ("mnn_w4" if args.dit_backend == "mnn" else "w4_simulated") if dit_w4 else "fp", "vae": "w4", "file": str(w4_path), **audio_stats(w4_path)})
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"Completed {label}: FP VAE and W4 VAE", flush=True)
                gc.collect()
            if dit_handle:
                dit_handle.remove()
    finally:
        vae_handle.remove()
    print(json.dumps({"completed": len(report["combinations"]), "report": str(report_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
