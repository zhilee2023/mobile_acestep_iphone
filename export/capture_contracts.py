"""Capture first real forward call for the three ACE-Step inference components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _tensor_specs(value: Any, prefix: str, tensors: dict[str, torch.Tensor], misc: dict[str, str]) -> None:
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value.detach().cpu().contiguous().clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            _tensor_specs(item, f"{prefix}.{key}", tensors, misc)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _tensor_specs(item, f"{prefix}.{index}", tensors, misc)
    elif value is None or isinstance(value, (str, int, float, bool)):
        misc[prefix] = repr(value)
    else:
        misc[prefix] = f"<{type(value).__module__}.{type(value).__name__}>"


def _save(output_dir: Path, side: str, value: Any) -> list[dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    misc: dict[str, str] = {}
    _tensor_specs(value, side, tensors, misc)
    torch.save(tensors, output_dir / f"{side}.pt")
    (output_dir / f"{side}_misc.json").write_text(json.dumps(misc, indent=2) + "\n", encoding="utf-8")
    return [
        {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        for name, tensor in tensors.items()
    ]


def attach_capture_hooks(handler: Any, output_dir: Path, upstream_revision: str) -> list[Any]:
    """Attach one-shot hooks and return handles to keep until generation ends."""
    components = {
        "text_encoder": handler.text_encoder,
        "dit": handler.model.decoder,
        "vae_decoder": handler.vae.decoder,
    }
    handles: list[Any] = []

    for name, module in components.items():
        component_dir = output_dir / name
        component_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {"captured": False}

        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, _dir: Path = component_dir, _state: dict[str, Any] = state) -> None:
            if _state["captured"]:
                return
            _state["inputs"] = _save(_dir, "inputs", {"args": args, "kwargs": kwargs})

        def post_hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any, *, _dir: Path = component_dir, _state: dict[str, Any] = state, _name: str = name) -> None:
            if _state["captured"]:
                return
            primary = getattr(output, "last_hidden_state", output)
            if isinstance(primary, (tuple, list)):
                primary = primary[0]
            outputs = _save(_dir, "outputs", primary)
            contract = {
                "component": _name,
                "upstream_revision": upstream_revision,
                "inputs": _state.get("inputs", []),
                "outputs": outputs,
            }
            (_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
            _state["captured"] = True

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    return handles
