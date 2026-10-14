"""Emit a short local OGG excerpt as base64 for model-side listening QA."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    waveform, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    start = int(args.start * sr)
    stop = min(len(waveform), start + int(args.seconds * sr))
    mono = waveform[start:stop].mean(axis=1)
    if sr != 16_000:
        from math import gcd

        divisor = gcd(sr, 16_000)
        mono = resample_poly(mono, 16_000 // divisor, sr // divisor)
    data = np.asarray(mono, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, data, 16_000, format="OGG", subtype="OPUS")
    print(base64.b64encode(buffer.getvalue()).decode("ascii"))


if __name__ == "__main__":
    main()
