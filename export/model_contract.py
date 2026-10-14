"""Model I/O contracts captured from the real ACE-Step forward calls.

Do not guess tensor names or shapes here. Populate the JSON only after registering
forward pre/post hooks against the pinned upstream revision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: list[int | str]


@dataclass(frozen=True)
class ModelContract:
    component: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    upstream_revision: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

