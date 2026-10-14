"""Convert captured ACE-Step ONNX components to float and weight-only W4 MNN.

This tool reports storage size. It does not claim runtime int4 execution or
numerical fidelity; those require separate MNN inference comparisons.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def convert(
    input_dir: Path,
    output_dir: Path,
    experiment: Path,
    converter: str,
) -> dict:
    config = json.loads(experiment.read_text(encoding="utf-8"))
    missing = [str(input_dir / row["onnx"]) for row in config["models"] if not (input_dir / row["onnx"]).is_file()]
    if missing:
        raise FileNotFoundError("Required ONNX exports are missing: " + ", ".join(missing))

    resolved = shutil.which(converter) or (converter if Path(converter).is_file() else None)
    if resolved is None:
        raise FileNotFoundError(f"MNN converter not found: {converter}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in config["models"]:
        source = input_dir / row["onnx"]
        float_path = output_dir / row["float_mnn"]
        quant_path = output_dir / row["quant_mnn"]
        common = [resolved, "-f", "ONNX", "--modelFile", str(source)]
        subprocess.run(common + ["--MNNModel", str(float_path)], check=True)
        quant_args = [
            "--MNNModel",
            str(quant_path),
            "--weightQuantBits",
            str(config["weight_quant_bits"]),
            "--weightQuantBlock",
            str(config["weight_quant_block"]),
        ]
        if config.get("hqq", False):
            quant_args.append("--hqq")
        subprocess.run(
            common + quant_args,
            check=True,
        )
        float_bytes = float_path.stat().st_size
        quant_bytes = quant_path.stat().st_size
        results.append(
            {
                "component": row["component"],
                "float_mnn": str(float_path),
                "float_bytes": float_bytes,
                "w4_mnn": str(quant_path),
                "w4_bytes": quant_bytes,
                "size_ratio": quant_bytes / float_bytes,
            }
        )
    report = {
        "experiment": config["name"],
        "weight_quant_bits_requested": config["weight_quant_bits"],
        "weight_quant_block": config["weight_quant_block"],
        "hqq": config.get("hqq", False),
        "warning": "Only supported float Conv/MatMul/LSTM weights are quantized; verify actual runtime memory, outputs, and audio separately.",
        "results": results,
    }
    (output_dir / "all_w4_sizes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "models" / "onnx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "mnn")
    parser.add_argument("--experiment", type=Path, default=ROOT / "config" / "all-w4.json")
    parser.add_argument("--converter", default="mnnconvert")
    args = parser.parse_args()
    print(json.dumps(convert(args.input_dir, args.output_dir, args.experiment, args.converter), indent=2))


if __name__ == "__main__":
    main()
