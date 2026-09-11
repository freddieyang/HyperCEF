"""Checkpoint serialization with explicit architecture metadata."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import MODEL_VERSION, HyperCEF


def save_checkpoint(model: HyperCEF, path: str | Path, model_config: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "HyperCEF",
            "version": MODEL_VERSION,
            "model_config": dict(model_config),
            "state_dict": model.state_dict(),
            "native_reference_sets": model.native_boundary_sets,
            "chassis_absent_reference_sets": model.hard_boundary_sets,
        },
        destination,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[HyperCEF, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model") != "HyperCEF":
        raise ValueError("Checkpoint does not contain a HyperCEF model")
    config = dict(payload["model_config"])
    model = HyperCEF(**config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.native_boundary_sets = payload.get("native_reference_sets", [])
    model.hard_boundary_sets = payload.get("chassis_absent_reference_sets", [])
    return model, config
