"""Plot eight 30-second waveform envelopes and quantify differences vs FP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont


DEST = Path(r"C:\Users\lizhi\Documents\Codex\2026-10-13\z\outputs")
PREFIX = "ace-step-vocal30"
ORDER = [
    ("fp", "fp", "fp"),
    ("w4", "fp", "fp"),
    ("fp", "w4", "fp"),
    ("fp", "fp", "w4"),
    ("w4", "w4", "fp"),
    ("w4", "fp", "w4"),
    ("fp", "w4", "w4"),
    ("w4", "w4", "w4"),
]


def wav_path(combo: tuple[str, str, str]) -> Path:
    text, dit, vae = combo
    return DEST / f"{PREFIX}_text-{text}_dit-{dit}_vae-{vae}.wav"


def mono_and_stats(path: Path) -> tuple[np.ndarray, int, dict]:
    stereo, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if stereo.shape[1] != 2 or sample_rate != 48_000:
        raise RuntimeError(f"Unexpected audio format in {path}: {stereo.shape}, {sample_rate}")
    if not np.isfinite(stereo).all():
        raise RuntimeError(f"Non-finite samples in {path}")
    stats = {
        "duration_seconds": float(len(stereo) / sample_rate),
        "peak": float(np.max(np.abs(stereo))),
        "rms": float(np.sqrt(np.mean(np.square(stereo, dtype=np.float64)))),
    }
    return stereo.mean(axis=1), sample_rate, stats


def envelope(wave: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(wave) // bins
    trimmed = wave[: count * bins].reshape(bins, count)
    return trimmed.min(axis=1), trimmed.max(axis=1)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    bins = 1200
    width, height = 1600, 1150
    left, right = 295, 40
    top, row_height = 142, 120
    plot_width = width - left - right
    colors = {
        "background": "#ffffff", "ink": "#17212b", "muted": "#5a6570",
        "grid": "#d9e0e6", "fp": "#205888", "w4": "#a14a35",
    }
    image = Image.new("RGB", (width, height), colors["background"])
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = font(29), font(19), font(15)
    draw.text((left, 30), "30-second Chinese pop vocal: waveform comparison", fill=colors["ink"], font=title_font)
    draw.text((left, 74), "Same lyric, prompt and seed 17  |  T = Text Encoder, D = DiT, V = VAE  |  FP vs MNN W4", fill=colors["muted"], font=small_font)

    baseline, sample_rate, _ = mono_and_stats(wav_path(ORDER[0]))
    report = {"scope": "Same-seed 30-second vocal waveform comparison; sample-wise correlation is not a listening score", "sample_rate": sample_rate, "cases": []}
    for row, combo in enumerate(ORDER):
        wave, rate, stats = mono_and_stats(wav_path(combo))
        if rate != sample_rate or wave.shape != baseline.shape:
            raise RuntimeError(f"Incompatible waveform shape for {combo}")
        difference = wave.astype(np.float64) - baseline.astype(np.float64)
        corr = float(np.corrcoef(wave, baseline)[0, 1])
        relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(baseline.astype(np.float64)), 1e-12))
        label = f"T {combo[0].upper()}  D {combo[1].upper()}  V {combo[2].upper()}"
        report["cases"].append({
            "text": combo[0], "dit": combo[1], "vae": combo[2],
            "file": str(wav_path(combo)), "correlation_vs_fp": corr,
            "relative_l2_vs_fp": relative_l2, **stats,
        })
        y_top = top + row * row_height
        y_center = y_top + 49
        draw.text((22, y_top + 20), label, fill=colors["ink"], font=body_font)
        draw.text((22, y_top + 51), f"corr {corr:+.3f}  |  RMS {stats['rms']:.3f}", fill=colors["muted"], font=small_font)
        draw.rectangle((left, y_top, left + plot_width, y_top + 96), outline=colors["grid"], width=1)
        draw.line((left, y_center, left + plot_width, y_center), fill=colors["grid"], width=1)
        for sec in range(0, 31, 5):
            x = left + round(sec / 30 * plot_width)
            draw.line((x, y_top, x, y_top + 96), fill=colors["grid"], width=1)
        low, high = envelope(wave, bins)
        color = colors["fp"] if combo == ORDER[0] else colors["w4"]
        for index in range(bins):
            x = left + round(index / (bins - 1) * plot_width)
            y1 = y_center - round(float(high[index]) * 45)
            y2 = y_center - round(float(low[index]) * 45)
            draw.line((x, max(y_top + 1, y1), x, min(y_top + 95, y2)), fill=color, width=1)
    last_bottom = top + (len(ORDER) - 1) * row_height + 96
    for sec in range(0, 31, 5):
        x = left + round(sec / 30 * plot_width)
        draw.text((x - 7, last_bottom + 11), str(sec), fill=colors["muted"], font=small_font)
    draw.text((left + plot_width // 2 - 43, last_bottom + 42), "Time (s)", fill=colors["ink"], font=small_font)
    output = DEST / "ace-step-vocal30-ab-waveforms.png"
    image.save(output)
    report_path = DEST / "ace-step-vocal30-ab-waveform-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plot": str(output), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
