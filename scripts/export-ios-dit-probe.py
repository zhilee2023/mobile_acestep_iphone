"""Bundle the captured 30-second DiT tensors for a real iPhone MNN step test.

These tensors are a diagnostic fixture, not a substitute for on-device text
conditioning or full music generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "reference-vocal-30s-v2" / "dit"
DEST = ROOT / "ios" / "Probe"
INPUTS = {
    "hidden_states": "dit-hidden.f32",
    "timestep": "dit-time.f32",
    "timestep_r": "dit-time-r.f32",
    "encoder_hidden_states": "dit-encoder.f32",
    "context_latents": "dit-context.f32",
}


def main() -> None:
    recorded = torch.load(SOURCE / "inputs.pt", map_location="cpu", weights_only=True)
    golden = torch.load(SOURCE / "outputs.pt", map_location="cpu", weights_only=True)["outputs"]
    DEST.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, filename in INPUTS.items():
        tensor = recorded[f"inputs.kwargs.{name}"]
        values = np.asarray(tensor, dtype="<f4", order="C")
        values.tofile(DEST / filename)
        shapes[name] = list(values.shape)
    np.asarray(golden, dtype="<f4", order="C").tofile(DEST / "dit-golden.f32")
    shapes["denoised"] = list(golden.shape)
    (DEST / "manifest.json").write_text(
        json.dumps({"source": "captured 30-second vocal forward", "shapes": shapes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(INPUTS)} inputs and golden output to {DEST}")


if __name__ == "__main__":
    main()
