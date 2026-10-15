from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.validate_quant import metrics
from runtime.smoke_wav import write_smoke_wav


def test_pipeline_contract() -> None:
    config = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))
    assert config["generation"] == {
        "caption": "Soft piano ambient music",
        "lyrics": "[Instrumental]",
        "duration_seconds": 10,
        "seed": 42,
        "steps": 8,
        "batch_size": 1,
        "sample_rate": 48000,
        "channels": 2,
    }
    assert config["runtime"]["load_order"] == ["text_encoder", "dit", "vae"]


def test_all_w4_components_match_pipeline() -> None:
    pipeline = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))
    experiment = json.loads((ROOT / "config" / "all-w4.json").read_text(encoding="utf-8"))
    assert experiment["weight_quant_bits"] == 4
    assert [row["component"] for row in experiment["models"]] == [
        "text_encoder", "dit", "vae_decoder"
    ]
    assert all(row["quant_mnn"].endswith("_w4.mnn") for row in experiment["models"])
    assert {
        key: value for key, value in pipeline["conversion"].items() if key.endswith("_weight_bits")
    } == {
        "text_encoder_weight_bits": 4,
        "dit_weight_bits": 4,
        "vae_weight_bits": 4,
    }


def test_quant_metrics_identity() -> None:
    value = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    result = metrics(value, value.copy())
    assert result["relative_l2"] == 0.0
    assert result["max_absolute_error"] == 0.0
    assert result["cosine_similarity"] > 0.999999


def test_smoke_wav(tmp_path: Path) -> None:
    output = tmp_path / "smoke.wav"
    write_smoke_wav(output, seconds=0.02)
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 48_000
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 960


def test_required_layout() -> None:
    required = [
        ".github/workflows/build-ios.yml",
        "project.yml",
        "runtime/CMakeLists.txt",
        "ios/App/AceStepApp.swift",
        "ios/Bridge/AceBridge.mm",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"missing project files: {missing}"


def test_two_step_cpu_references() -> None:
    probe = ROOT / "ios" / "Probe"
    outputs = [np.fromfile(probe / f"dit-w4-cpu-step{step}.f32", dtype="<f4")
               for step in (1, 2)]
    assert all(output.shape == (750 * 64,) for output in outputs)
    assert all(np.isfinite(output).all() for output in outputs)
    assert not np.array_equal(outputs[0], outputs[1])


def test_encoder_benchmark_fixture() -> None:
    ids = np.fromfile(ROOT / "ios" / "Probe" / "encoder-input-ids.i64", dtype="<i8")
    assert ids.shape == (107,)
    assert np.all(ids >= 0)
    assert np.all(ids <= np.iinfo(np.int32).max)


def test_vae_diagnostic_fixture() -> None:
    latent = np.fromfile(ROOT / "ios" / "Probe" / "vae-final-latent.f32", dtype="<f4")
    assert latent.shape == (750 * 64,)
    assert np.isfinite(latent).all()
    assert 1 < np.max(np.abs(latent)) < 100


def test_lyric_tokenizer_bundle_and_probe() -> None:
    ids = np.fromfile(ROOT / "ios" / "Probe" / "lyric-sample-ids.i32", dtype="<i4")
    assert ids.shape == (73,)
    assert ids[-2:].tolist() == [151643, 151643]
    assert (ROOT / "ios" / "Tokenizer" / "tokenizer.json").stat().st_size > 10_000_000
    assert (ROOT / "ios" / "Tokenizer" / "sample-lyrics.txt").is_file()


def test_v24_connected_condition_and_vae_modes() -> None:
    import plistlib

    info = plistlib.loads((ROOT / "ios" / "Info.plist").read_bytes())
    assert info["CFBundleVersion"] == "24"
    engine = (ROOT / "ios" / "Bridge" / "FixedSongEngine.mm").read_text(encoding="utf-8")
    benchmark = (ROOT / "ios" / "Bridge" / "EncoderBenchmark.mm").read_text(encoding="utf-8")
    ui = (ROOT / "ios" / "App" / "ContentView.swift").read_text(encoding="utf-8")
    assert 'vae-notune-diagnostic.log' in engine
    assert 'MNN_GPU_TUNING_NONE' in engine
    assert 'last-inference-cpu-vae.log' in engine
    assert 'cpuVAE ? MNN_FORWARD_CPU : MNN_FORWARD_METAL, noTuneVAE' in engine
    assert 'encoder-cpu-vae-serial.log' in benchmark
    assert 'generateCustomSong()' in ui
    assert 'generateNoTuneSong()' in ui
    assert 'vaeDiagnosticCard' not in ui.split('var body: some View', 1)[1].split('private var header', 1)[0]
    bridge = (ROOT / "ios" / "Bridge" / "AceBridge.mm").read_text(encoding="utf-8")
    assert '[EncoderBenchmark encodeStyleIDs:' in bridge
    assert '[LyricConditionEngine encodeLyrics:' in bridge
    assert 'generateWithConditionCpuVAE:condition' in bridge
    assert '74ULL * 2048 * sizeof(float)' in bridge
    assert (ROOT / "ios/Tokenizer/sample-caption.txt").is_file()
    assert (ROOT / "export/export_text_projector.py").is_file()
    lyric = (ROOT / "ios/Bridge/LyricConditionEngine.mm").read_text(encoding="utf-8")
    assert 'if (!lyric) [ids addObject:@(kNewline)]' in lyric
    assert 'constexpr int kNewline = 198;' in lyric
    assert '内置样例 %lu token，当前输入 %lu token' in lyric
    assert '使用内置示例风格（107 token）' in ui
    assert 'usesExactSampleCaption ? SongStudioStore.sampleCaption : studio.caption' in ui

