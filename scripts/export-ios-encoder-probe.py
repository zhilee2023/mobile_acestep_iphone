"""Package the captured 107-token embedding input for an iPhone benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
capture = root / "models" / "reference-vocal-30s-v2" / "text_encoder" / "inputs.pt"
output = root / "ios" / "Probe" / "encoder-input-ids.i64"

inputs = torch.load(capture, map_location="cpu", weights_only=True)
ids = inputs["inputs.kwargs.input_ids"].numpy()
assert ids.shape == (1, 107), ids.shape
assert ids.dtype == np.int64, ids.dtype
ids.astype("<i8", copy=False).tofile(output)
print(f"{output}: {output.stat().st_size} bytes, {ids.shape[1]} tokens")
