"""Generate the 30-second vocal A/B with the real static MNN W4 DiT.

Text encoder and VAE stay in PyTorch FP; only the DiT forward is replaced.
This is a Windows CPU listening experiment, not an iPhone runtime test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from generate_ab_w4 import dit_mnn_forward, mnn_session


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-model", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    upstream = ROOT / "upstream" / "ACE-Step-1.5"
    checkpoints = upstream / "checkpoints"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
    sys.path.insert(0, str(upstream))
    from acestep import model_downloader
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    reference = json.loads((ROOT / "outputs" / "vocal-30s-v2" / "reference_result.json").read_text(encoding="utf-8"))
    if reference["instrumental"] or reference["duration_seconds"] != 30:
        raise RuntimeError("Expected the 30-second vocal FP reference")
    required = {"acestep-v15-turbo": "model.safetensors", "Qwen3-Embedding-0.6B": "model.safetensors", "vae": "diffusion_pytorch_model.safetensors"}
    handler = AceStepHandler()
    with patch.object(model_downloader, "MAIN_MODEL_COMPONENTS", list(required)):
        status, success = handler.initialize_service(project_root=str(upstream), config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False)
    if not success:
        raise RuntimeError(status)

    interpreter, session = mnn_session(args.dit_model)

    def dit_post_hook(_module, _args, kwargs, output):
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise RuntimeError("Unexpected DiT output contract")
        replacement = dit_mnn_forward(interpreter, session, kwargs, output[0])
        return (replacement, *output[1:])

    params = GenerationParams(
        caption=reference["caption"], lyrics=reference["lyrics"], instrumental=False,
        vocal_language="zh", duration=30, seed=reference["seed"], inference_steps=8,
        thinking=False, use_cot_metas=False, use_cot_caption=False,
        use_cot_language=False, dcw_enabled=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[reference["seed"]], audio_format="wav")
    work = ROOT / "outputs" / "vocal-30s-v2-dit-w4"
    work.mkdir(parents=True, exist_ok=True)
    handle = handler.model.decoder.register_forward_hook(dit_post_hook, with_kwargs=True)
    try:
        result = generate_music(handler, None, params, config, save_dir=str(work))
    finally:
        handle.remove()
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message)
    candidates = list(work.glob("*.wav"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one generated WAV, got {len(candidates)}")
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "ace-step-chinese-pop-vocal-30s-dit-mnn-w4.wav"
    shutil.copy2(candidates[0], target)
    report = {
        "scope": "30-second ACE-Step vocal A/B, Windows MNN CPU",
        "source_fp_audio": str(DEST / "ace-step-chinese-pop-vocal-30s-fp-v2.wav"),
        "quantized_audio": str(target),
        "dit": "MNN W4+HQQ block64 weight-only; 30-second static export",
        "text_encoder": "PyTorch FP",
        "vae": "PyTorch FP",
        "seed": reference["seed"],
        "caption": reference["caption"],
        "lyrics": reference["lyrics"],
        "warning": "Not all-component W4 and not iPhone Metal validated; lyric audibility requires listening.",
    }
    report_path = DEST / "ace-step-chinese-pop-vocal-30s-dit-mnn-w4-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audio": str(target), "report": str(report_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
