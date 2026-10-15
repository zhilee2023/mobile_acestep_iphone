"""Export ACE-Step's Qwen-to-DiT style projection for the iPhone runtime.

The 107-token Qwen graph returns 1024-wide states. The DiT expects the
condition encoder's 2048-wide projection in its final 107 slots.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "upstream/ACE-Step-1.5/checkpoints/acestep-v15-turbo/model.safetensors"
REFERENCE = ROOT / "models/reference-vocal-30s-v2"
OUTPUT = ROOT / "models/mnn/vocal-30s-v2/text_projector.f16"


def main() -> None:
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("encoder.text_projector.weight").float().numpy()
    if weight.shape != (2048, 1024):
        raise ValueError(f"Unexpected projector shape: {weight.shape}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    weight.astype("<f2").tofile(OUTPUT)

    qwen = torch.load(REFERENCE / "text_encoder/outputs.pt", map_location="cpu", weights_only=True)["outputs"].numpy()[0]
    condition = torch.load(REFERENCE / "dit/inputs.pt", map_location="cpu", weights_only=True)["inputs.kwargs.encoder_hidden_states"].numpy()[0]
    projected = qwen @ weight.T
    reference = condition[74:181]
    error = np.linalg.norm((projected - reference).astype(np.float64)) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12)
    print(f"projector bytes={OUTPUT.stat().st_size}; float32 reference L2={error:.8g}")
    if error > 1e-4:
        raise RuntimeError("Projector weights do not reproduce the captured DiT style condition")
    quantized = qwen @ weight.astype("<f2").astype(np.float32).T
    quantized_error = np.linalg.norm((quantized - reference).astype(np.float64)) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12)
    print(f"fp16 weight reference L2={quantized_error:.8g}")
    if quantized_error > 0.01:
        raise RuntimeError("fp16 projector is too different from the captured reference")


if __name__ == "__main__":
    main()
