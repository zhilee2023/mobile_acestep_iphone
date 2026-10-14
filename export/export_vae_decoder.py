"""Export the captured 10-second ACE-Step VAE decoder and validate ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers.models import AutoencoderOobleck


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream" / "ACE-Step-1.5" / "checkpoints" / "vae")
    parser.add_argument("--capture", type=Path, default=ROOT / "models" / "reference" / "vae_decoder")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "onnx" / "vae_decoder.onnx")
    parser.add_argument("--latent-frames", type=int, help="Export this chunk length using a deterministic synthetic input")
    args = parser.parse_args()

    torch.set_num_threads(8)
    if args.latent_frames is None:
        inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
        golden = torch.load(args.capture / "outputs.pt", map_location="cpu", weights_only=True)["outputs"]
        latents = inputs["inputs.args.0"]
    else:
        if args.latent_frames <= 0:
            raise ValueError("latent frames must be positive")
        generator = torch.Generator(device="cpu").manual_seed(17)
        latents = torch.randn((1, 64, args.latent_frames), generator=generator)
        golden = None

    vae = AutoencoderOobleck.from_pretrained(str(args.checkpoint)).to("cpu", dtype=torch.float32).eval()
    if golden is None:
        with torch.inference_mode():
            golden = vae.decoder(latents).detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            vae.decoder,
            (latents,),
            str(args.output),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["latents"],
            output_names=["waveform"],
            dynamic_axes=None,
        )
    del vae
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    candidate = session.run(None, {"latents": latents.numpy()})[0]
    reference = golden.numpy()
    relative_l2 = float(np.linalg.norm((candidate - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-12))
    report = {
        "component": "vae_decoder",
        "input_shape": list(latents.shape),
        "output_shape": list(candidate.shape),
        "onnx_bytes": args.output.stat().st_size,
        "onnx_vs_pytorch_relative_l2": relative_l2,
        "onnx_vs_pytorch_max_abs": float(np.max(np.abs(candidate - reference))),
    }
    (args.output.parent / f"{args.output.stem}_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
