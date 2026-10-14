"""Generate a deterministic PCM WAV to validate the output path (not model inference)."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def write_smoke_wav(path: Path, seconds: float = 1.0, sample_rate: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * sample_rate)
    amplitude = 0.18 * 32767
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frame_count):
            fade = min(1.0, index / 800, (frame_count - index) / 800)
            sample = int(amplitude * fade * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav.writeframesraw(struct.pack("<hh", sample, sample))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    write_smoke_wav(args.output, args.seconds)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

