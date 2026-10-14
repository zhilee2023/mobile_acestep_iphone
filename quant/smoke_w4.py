"""Small integration check for the installed ONNX→MNN W4 converter and CPU runtime.

This tests the toolchain on a synthetic MatMul, not ACE-Step music quality.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import MNN
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "smoke_w4"
CONVERTER = ROOT / ".venv" / "Scripts" / "mnnconvert.exe"


def run_mnn(path: Path, value: np.ndarray) -> np.ndarray:
    interpreter = MNN.Interpreter(str(path))
    session = interpreter.createSession({"backend": "CPU", "precision": "high", "memory": "low"})
    input_tensor = interpreter.getSessionInput(session)
    host_input = MNN.Tensor(value.shape, MNN.Halide_Type_Float, value, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(host_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    result = np.array(output_tensor.getNumpyData(), dtype=np.float32).reshape(value.shape)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.05, size=(512, 512)).astype(np.float32)
    value = rng.normal(size=(1, 512)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "w4-smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 512])],
        [numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx_path = OUTPUT / "matmul.onnx"
    onnx.save(model, onnx_path)

    sizes = {}
    for name, extra in (
        ("fp", []),
        ("w4", ["--weightQuantBits", "4", "--weightQuantBlock", "64"]),
        ("w4_hqq", ["--weightQuantBits", "4", "--weightQuantBlock", "64", "--hqq"]),
    ):
        dest = OUTPUT / f"matmul_{name}.mnn"
        subprocess.run(
            [str(CONVERTER), "-f", "ONNX", "--modelFile", str(onnx_path), "--MNNModel", str(dest), *extra],
            check=True,
        )
        sizes[name] = dest.stat().st_size

    reference = value @ weights
    fp_output = run_mnn(OUTPUT / "matmul_fp.mnn", value)
    w4_output = run_mnn(OUTPUT / "matmul_w4.mnn", value)
    hqq_output = run_mnn(OUTPUT / "matmul_w4_hqq.mnn", value)
    report = {
        "scope": "synthetic MatMul only; not ACE-Step",
        "fp_mnn_bytes": sizes["fp"],
        "w4_mnn_bytes": sizes["w4"],
        "w4_to_fp_size_ratio": sizes["w4"] / sizes["fp"],
        "w4_hqq_to_fp_size_ratio": sizes["w4_hqq"] / sizes["fp"],
        "fp_relative_l2": float(np.linalg.norm(fp_output - reference) / np.linalg.norm(reference)),
        "w4_relative_l2": float(np.linalg.norm(w4_output - reference) / np.linalg.norm(reference)),
        "w4_hqq_relative_l2": float(np.linalg.norm(hqq_output - reference) / np.linalg.norm(reference)),
    }
    (OUTPUT / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
