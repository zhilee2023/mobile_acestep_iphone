"""Reconstruct the captured 30-second DiT condition from upstream weights.

This is a read-only feasibility check for mobile custom lyrics.  It must pass
before exporting a lyric-encoder graph or replacing the phone's fixed tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/ACE-Step-1.5"))
CHECKPOINT = ROOT / "upstream/ACE-Step-1.5/checkpoints/acestep-v15-turbo"
TEXT_CHECKPOINT = ROOT / "upstream/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B"
REFERENCE = ROOT / "models/reference-vocal-30s-v2"
RESULT = ROOT / "outputs/vocal-30s-v2-capture/reference_result.json"


def main() -> None:
    torch.set_num_threads(8)
    recorded = json.loads(RESULT.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT)
    lyrics = f"# Languages\nzh\n\n# Lyric\n{recorded['lyrics']}<|endoftext|>"
    lyric_ids = tokenizer(lyrics, return_tensors="pt").input_ids
    text_inputs = torch.load(REFERENCE / "text_encoder/inputs.pt", map_location="cpu", weights_only=True)
    text_outputs = torch.load(REFERENCE / "text_encoder/outputs.pt", map_location="cpu", weights_only=True)
    dit_inputs = torch.load(REFERENCE / "dit/inputs.pt", map_location="cpu", weights_only=True)
    text_ids = text_inputs["inputs.kwargs.input_ids"]
    text_hidden = text_outputs["outputs"]
    expected = dit_inputs["inputs.kwargs.encoder_hidden_states"]
    context = dit_inputs["inputs.kwargs.context_latents"]
    src_latents = context[:, :, :64]
    print(f"lyric tokens={lyric_ids.shape[1]}, text tokens={text_ids.shape[1]}, target condition={tuple(expected.shape)}", flush=True)

    text_model = AutoModel.from_pretrained(TEXT_CHECKPOINT, attn_implementation="eager", dtype=torch.float32).eval()
    model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.float32).eval()
    with torch.inference_mode():
        lyric_hidden = text_model.embed_tokens(lyric_ids)
        actual, mask = model.encoder(
            text_hidden_states=text_hidden,
            text_attention_mask=torch.ones_like(text_ids, dtype=torch.bool),
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=torch.ones_like(lyric_ids, dtype=torch.bool),
            refer_audio_acoustic_hidden_states_packed=src_latents,
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
        )
    print(f"actual condition={tuple(actual.shape)}, mask valid={int(mask.sum())}", flush=True)
    if actual.shape != expected.shape:
        raise RuntimeError(f"Condition length does not match: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    for name, start, end in (("lyric", 0, lyric_ids.shape[1]), ("timbre+text", lyric_ids.shape[1], expected.shape[1]), ("all", 0, expected.shape[1])):
        error = torch.linalg.vector_norm(actual[:, start:end] - expected[:, start:end])
        norm = torch.linalg.vector_norm(expected[:, start:end]).clamp_min(1e-12)
        print(f"{name} relative L2={float(error / norm):.8g}", flush=True)


if __name__ == "__main__":
    main()
