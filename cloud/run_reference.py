"""Generate fixed, unquantized ACE-Step reference audio on CUDA or CPU.

This is the comparison target for the all-W4 experiment. It does not generate
audio from MNN models. The pinned upstream downloader requires the unused LM
to be present during its precheck; the scoped override below excludes it when
LM generation is explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream" / "ACE-Step-1.5")
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reference")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--capture-contracts", action="store_true")
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "models" / "reference")
    parser.add_argument("--caption")
    parser.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--vocal", action="store_true")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this cloud reference run")

    upstream = args.upstream.resolve()
    checkpoints = (args.checkpoints or upstream / "checkpoints").resolve()
    required = {
        "acestep-v15-turbo": "model.safetensors",
        "Qwen3-Embedding-0.6B": "model.safetensors",
        "vae": "diffusion_pytorch_model.safetensors",
    }
    missing = [str(checkpoints / folder / name) for folder, name in required.items() if not (checkpoints / folder / name).is_file()]
    if missing:
        raise FileNotFoundError("Reference model weights are incomplete: " + ", ".join(missing))

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
    sys.path.insert(0, str(upstream))
    from acestep import model_downloader
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    config = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))["generation"]
    caption = args.caption if args.caption is not None else config["caption"]
    lyrics = args.lyrics_file.read_text(encoding="utf-8").strip() if args.lyrics_file else config["lyrics"]
    duration = args.duration_seconds if args.duration_seconds is not None else config["duration_seconds"]
    seed = args.seed if args.seed is not None else config["seed"]
    steps = args.steps if args.steps is not None else config["steps"]
    handler = AceStepHandler()
    with patch.object(model_downloader, "MAIN_MODEL_COMPONENTS", list(required)):
        status, success = handler.initialize_service(
            project_root=str(upstream),
            config_path="acestep-v15-turbo",
            device=args.device,
            offload_to_cpu=args.device == "cuda",
        )
    if not success:
        raise RuntimeError(status)

    capture_handles = []
    if args.capture_contracts:
        sys.path.insert(0, str(ROOT))
        from export.capture_contracts import attach_capture_hooks

        revision = json.loads((ROOT / "upstream" / "versions.json").read_text(encoding="utf-8"))["ace_step_source"]["commit"]
        capture_handles = attach_capture_hooks(handler, args.capture_dir, revision)

    params = GenerationParams(
        caption=caption,
        lyrics=lyrics,
        instrumental=not args.vocal,
        vocal_language="zh" if args.vocal else "unknown",
        duration=duration,
        seed=seed,
        inference_steps=steps,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        dcw_enabled=False,
    )
    generation = GenerationConfig(
        batch_size=config["batch_size"],
        use_random_seed=False,
        seeds=[seed],
        audio_format="wav",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = generate_music(handler, None, params, generation, save_dir=str(args.output_dir))
    finally:
        for handle in capture_handles:
            handle.remove()
    if not result.success:
        raise RuntimeError(result.error or result.status_message)
    report = {
        "scope": "unquantized ACE-Step reference; not MNN W4",
        "model": "acestep-v15-turbo",
        "device": args.device,
        "seed": seed,
        "caption": caption,
        "lyrics": lyrics,
        "duration_seconds": duration,
        "instrumental": not args.vocal,
        "audios": result.audios,
        "extra_outputs": result.extra_outputs,
    }
    (args.output_dir / "reference_result.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
