"""Confirm a same-length lyric edit actually changes the W4 DiT output."""

from __future__ import annotations

import json
from pathlib import Path

import MNN
import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/mnn/vocal-30s-v2"
PROBE = ROOT / "ios/Probe"


def lyric_ids(tokenizer, lyrics: str) -> np.ndarray:
    text = f"# Languages\nzh\n\n# Lyric\n{lyrics}<|endoftext|>"
    ids = np.asarray(tokenizer(text).input_ids, dtype=np.int64)
    if ids.shape != (73,):
        raise ValueError(f"Test edit must remain 73 tokens; got {ids.shape}")
    return ids


def feed(interpreter, session, name: str, values: np.ndarray) -> None:
    tensor = interpreter.getSessionInput(session, name)
    assert tuple(tensor.getShape()) == values.shape
    host = MNN.Tensor(values.shape, MNN.Halide_Type_Float, values, MNN.Tensor_DimensionType_Caffe)
    assert tensor.copyFrom(host)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B")
    lyrics = json.loads((ROOT / "outputs/vocal-30s-v2-capture/reference_result.json").read_text(encoding="utf-8"))["lyrics"]
    edited = lyrics.replace("听见", "看见", 1)
    assert edited != lyrics
    original_ids = lyric_ids(tokenizer, lyrics)
    edited_ids = lyric_ids(tokenizer, edited)
    assert not np.array_equal(original_ids, edited_ids)
    table = np.memmap(MODELS / "lyric_embedding.f16", dtype="<f2", mode="r", shape=(151669, 1024))
    encoder = MNN.Interpreter(str(MODELS / "lyric_encoder_73_w4.mnn"))
    encoder_session = encoder.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    fixed_condition = np.fromfile(PROBE / "dit-encoder.f32", dtype="<f4").reshape(1, 181, 2048)
    conditions = []
    for ids in (original_ids, edited_ids):
        embeddings = np.asarray(table[ids], dtype=np.float32).reshape(1, 73, 1024)
        feed(encoder, encoder_session, "lyric_embeddings", embeddings)
        encoder.runSession(encoder_session)
        prefix = np.asarray(encoder.getSessionOutput(encoder_session, "lyric_hidden_states").getNumpyData(),
                            dtype=np.float32).reshape(1, 73, 2048).copy()
        condition = fixed_condition.copy()
        condition[:, :73, :] = prefix
        conditions.append(condition)
    del encoder, encoder_session, table

    dit = MNN.Interpreter(str(MODELS / "dit_w4.mnn"))
    session = dit.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    hidden = np.fromfile(PROBE / "dit-hidden.f32", dtype="<f4").reshape(1, 750, 64)
    context = np.fromfile(PROBE / "dit-context.f32", dtype="<f4").reshape(1, 750, 128)
    outputs = []
    for condition in conditions:
        feed(dit, session, "hidden_states", hidden)
        feed(dit, session, "timestep", np.array([1], dtype=np.float32))
        feed(dit, session, "timestep_r", np.array([1], dtype=np.float32))
        feed(dit, session, "context_latents", context)
        feed(dit, session, "encoder_hidden_states", condition)
        dit.runSession(session)
        output = np.asarray(dit.getSessionOutput(session, "denoised").getNumpyData(), dtype=np.float32).copy()
        assert np.isfinite(output).all()
        outputs.append(output)
    difference = float(np.linalg.norm(outputs[1] - outputs[0]) / np.linalg.norm(outputs[0]))
    print(f"Changed lyric produces DiT first-step relative L2={difference:.8g}")
    if difference < 1e-5:
        raise RuntimeError("Lyric edit did not materially change the DiT output")


if __name__ == "__main__":
    main()
