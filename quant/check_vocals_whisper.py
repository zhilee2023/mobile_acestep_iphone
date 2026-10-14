"""Auxiliary ASR check for whether a generated vocal sample contains words.

Whisper can hallucinate on music, so a nonempty transcript is not proof of
audible singing. Human listening remains the acceptance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    torch.set_num_threads(8)
    recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    waveform, sample_rate = sf.read(args.audio, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        from math import gcd

        divisor = gcd(sample_rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor)
    result = recognizer(
        {"raw": waveform, "sampling_rate": 16000},
        generate_kwargs={"language": "chinese", "task": "transcribe"},
    )
    print(json.dumps({"audio": str(args.audio), "transcript": result.get("text", "")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
