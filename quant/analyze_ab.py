"""Measure saved A/B audio differences against the same-seed FP reference."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


REPORT = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs\ace-step-ab-report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = report["combinations"]
    if len(cases) != 8:
        raise RuntimeError(f"Expected all eight combinations, got {len(cases)}")
    reference, reference_sr = sf.read(cases[0]["file"], dtype="float32", always_2d=True)
    reference = reference.astype(np.float64).reshape(-1)
    for row in cases:
        path = Path(row["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        candidate = waveform.astype(np.float64).reshape(-1)
        if sr != reference_sr or candidate.shape != reference.shape:
            raise RuntimeError(f"Mismatched WAV: {path}")
        row["waveform_relative_l2_vs_fp"] = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
        row["waveform_correlation_vs_fp"] = float(np.corrcoef(candidate, reference)[0, 1])
        row["bytes"] = path.stat().st_size
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in cases:
        print(f"{row['text']:>2} {row['dit']:>12} {row['vae']:>2}  rms={row['rms']:.3f}  corr={row['waveform_correlation_vs_fp']:.4f}  rel={row['waveform_relative_l2_vs_fp']:.3f}")


if __name__ == "__main__":
    main()
