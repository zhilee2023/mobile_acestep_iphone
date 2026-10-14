"""Compare a converted model output tensor with its golden NumPy tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | list[int]]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float64, copy=False).ravel()
    got = candidate.astype(np.float64, copy=False).ravel()
    delta = got - ref
    denominator = float(np.linalg.norm(ref)) + 1e-12
    cosine_denominator = float(np.linalg.norm(ref) * np.linalg.norm(got)) + 1e-12
    return {
        "shape": list(reference.shape),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "max_absolute_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "cosine_similarity": float(np.dot(ref, got) / cosine_denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = metrics(np.load(args.reference), np.load(args.candidate))
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

