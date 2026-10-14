"""Auxiliary Whisper transcription for each 30-second vocal A/B candidate.

Music ASR can hallucinate. Keep raw transcripts and do not infer that a
nonempty transcript proves intelligible lyrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline


DEST = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")


def main() -> None:
    torch.set_num_threads(8)
    report_path = DEST / "ace-step-vocal30-ab-mnn-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if len(report["combinations"]) != 8:
        raise RuntimeError("Eight audio combinations are required before ASR")
    recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    results = []
    for row in report["combinations"]:
        path = Path(row["file"])
        wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = wave.mean(axis=1)
        if sample_rate != 48_000:
            raise RuntimeError(f"Unexpected sample rate {sample_rate} in {path}")
        mono16 = resample_poly(mono, 1, 3)
        halves = []
        for start in (0, 15):
            segment = mono16[start * 16_000 : (start + 15) * 16_000]
            result = recognizer(
                {"raw": segment, "sampling_rate": 16_000},
                generate_kwargs={"language": "chinese", "task": "transcribe"},
            )
            halves.append(result.get("text", ""))
        result_row = {"text": row["text"], "dit": row["dit"], "vae": row["vae"], "transcript_0_15s": halves[0], "transcript_15_30s": halves[1]}
        results.append(result_row)
        print(json.dumps(result_row, ensure_ascii=False), flush=True)
    output = DEST / "ace-step-vocal30-ab-asr-check.json"
    output.write_text(json.dumps({"warning": "ASR on music may hallucinate; human listening remains required.", "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
