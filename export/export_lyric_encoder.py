"""Export the real ACE-Step lyric condition path for a 73-token mobile pilot.

The Qwen token embedding table is kept separate so it can be memory-mapped on
iPhone.  This graph receives raw 1024-wide token embeddings and produces the
first 73 x 2048 slice of the captured 181-token DiT condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/ACE-Step-1.5"))
MODEL = ROOT / "upstream/ACE-Step-1.5/checkpoints/acestep-v15-turbo"
TEXT = ROOT / "upstream/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B"
REFERENCE = ROOT / "models/reference-vocal-30s-v2"
SONG = ROOT / "outputs/vocal-30s-v2-capture/reference_result.json"


class LyricWrapper(nn.Module):
    def __init__(self, lyric_encoder: nn.Module) -> None:
        super().__init__()
        self.lyric_encoder = lyric_encoder

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        return self.lyric_encoder(inputs_embeds=embeddings, attention_mask=mask).last_hidden_state


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "models/onnx/vocal-30s-v2/lyric_encoder_73.onnx")
    parser.add_argument("--embedding-output", type=Path, default=ROOT / "models/mnn/vocal-30s-v2/lyric_embedding.f16")
    args = parser.parse_args()
    torch.set_num_threads(8)

    song = json.loads(SONG.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(TEXT)
    text = f"# Languages\nzh\n\n# Lyric\n{song['lyrics']}<|endoftext|>"
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape != (1, 73):
        raise ValueError(f"The captured lyric must tokenize to [1,73], got {tuple(ids.shape)}")
    with safe_open(TEXT / "model.safetensors", framework="pt", device="cpu") as tensors:
        embedding_table = tensors.get_tensor("embed_tokens.weight")
    if embedding_table.shape != (151669, 1024):
        raise ValueError(f"Unexpected Qwen embedding table {tuple(embedding_table.shape)}")
    args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
    # A deterministic little-endian fp16 table supports direct random-access
    # lookup without loading the entire Qwen graph into phone memory.
    embedding_table.float().numpy().astype("<f2").tofile(args.embedding_output)
    embeddings = torch.nn.functional.embedding(ids, embedding_table).float()
    del embedding_table

    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation="sdpa", dtype=torch.float32).eval()
    wrapper = LyricWrapper(model.encoder.lyric_encoder).eval()
    with torch.inference_mode():
        direct = wrapper(embeddings).numpy()
    dit_inputs = torch.load(REFERENCE / "dit/inputs.pt", map_location="cpu", weights_only=True)
    golden = dit_inputs["inputs.kwargs.encoder_hidden_states"][:, :73, :].numpy()
    direct_error = relative_l2(direct, golden)
    print(f"PyTorch lyric graph vs captured condition L2={direct_error:.8g}", flush=True)
    if direct_error > 1e-5:
        raise RuntimeError("Lyric path no longer matches the captured song")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, (embeddings,), str(args.output), opset_version=17,
                          export_params=True, external_data=True, do_constant_folding=True,
                          input_names=["lyric_embeddings"], output_names=["lyric_hidden_states"],
                          dynamic_axes=None)
    del wrapper, model
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    candidate = session.run(None, {"lyric_embeddings": embeddings.numpy()})[0]
    onnx_error = relative_l2(candidate, golden)
    report = {
        "token_count": 73,
        "token_ids": ids.tolist()[0],
        "embedding_shape": [151669, 1024],
        "embedding_bytes": args.embedding_output.stat().st_size,
        "lyric_output_shape": list(candidate.shape),
        "pytorch_vs_capture_l2": direct_error,
        "onnx_vs_capture_l2": onnx_error,
    }
    (args.output.parent / "lyric_encoder_73_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "ios/Probe/lyric-sample-ids.i32").write_bytes(
        np.asarray(ids.tolist()[0], dtype="<i4").tobytes())
    print(json.dumps({k: v for k, v in report.items() if k != "token_ids"}, indent=2), flush=True)
    if onnx_error > 1e-4:
        raise RuntimeError("Exported lyric graph differs from captured condition")


if __name__ == "__main__":
    main()
