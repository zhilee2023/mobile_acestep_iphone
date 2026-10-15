"""Check the shipped 107-token W4 style graph and ACE-Step projector locally."""

from pathlib import Path

import MNN
import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/mnn/vocal-30s-v2/text_encoder_w4.mnn"
WEIGHTS = ROOT / "models/mnn/vocal-30s-v2/text_projector.f16"
TOKENIZER = ROOT / "upstream/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B"
CAPTION = (ROOT / "ios/Tokenizer/sample-caption.txt").read_text(encoding="utf-8").strip()


def prompt(caption: str) -> str:
    return ("# Instruction\nFill the audio semantic mask based on the given conditions:"
            "\n\n# Caption\n" + caption + "\n\n# Metas\n- bpm: N/A\n"
            "- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n"
            "<|endoftext|>\n")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    captions = [CAPTION, CAPTION.replace("piano", "guitar")]
    tokens = [np.asarray(tokenizer(prompt(caption)).input_ids, dtype=np.int32) for caption in captions]
    if any(ids.shape != (107,) for ids in tokens):
        raise RuntimeError(f"Static graph requires 107 tokens; got {[ids.size for ids in tokens]}")
    captured = np.fromfile(ROOT / "ios/Probe/encoder-input-ids.i64", dtype="<i8")
    if not np.array_equal(tokens[0], captured):
        raise RuntimeError("The style prompt does not match the ACE-Step capture")
    matrix = np.fromfile(WEIGHTS, dtype="<f2").astype(np.float32).reshape(2048, 1024)
    interpreter = MNN.Interpreter(str(MODEL))
    session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    projected = []
    for ids in tokens:
        source = interpreter.getSessionInput(session)
        source.copyFrom(MNN.Tensor((1, 107), MNN.Halide_Type_Int, ids[None], MNN.Tensor_DimensionType_Caffe))
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session)
        qwen = np.asarray(output.getNumpyData(), dtype=np.float32).reshape(107, 1024).copy()
        projected.append(qwen @ matrix.T)
    golden = torch.load(ROOT / "models/reference-vocal-30s-v2/dit/inputs.pt", map_location="cpu", weights_only=True)["inputs.kwargs.encoder_hidden_states"].numpy()[0, 74:181]
    reference_l2 = np.linalg.norm((projected[0] - golden).astype(np.float64)) / np.linalg.norm(golden.astype(np.float64))
    variation_l2 = np.linalg.norm((projected[1] - projected[0]).astype(np.float64)) / np.linalg.norm(projected[0].astype(np.float64))
    if not np.isfinite(projected[0]).all() or not np.isfinite(projected[1]).all():
        raise RuntimeError("W4 style projection contains non-finite values")
    if variation_l2 <= 0.001:
        raise RuntimeError("Changing the caption did not change the projected style")
    print(f"W4 projected style versus FP ACE-Step capture L2={reference_l2:.6g}")
    print(f"piano→guitar projected style L2={variation_l2:.6g}")


if __name__ == "__main__":
    main()
