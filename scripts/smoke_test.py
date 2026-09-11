#!/usr/bin/env python3
"""Run a small forward, training, and checkpoint round trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from hypercef import HyperCEF, PreparedDataset, fit, load_checkpoint, predict, save_checkpoint


def main() -> None:
    torch.manual_seed(42)
    x = torch.randint(0, 2, (8, 64), dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 5, 6], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]],
        dtype=torch.long,
    )
    dataset = PreparedDataset(
        x=x,
        edge_index=edge_index,
        native=[[0, 1], [1, 2], [2, 3], [4, 5]],
        chassis_absent=[[3, 4], [6, 7]],
        perturbed=[[0, 7], [2, 6]],
    )
    config = {
        "in_channels": 64,
        "hidden_channels": 32,
        "out_channels": 1,
        "dropout": 0.0,
        "num_layers": 2,
        "heads": 2,
    }
    model = HyperCEF(**config)
    history = fit(model, dataset, epochs=1, batch_size=8, seed=42)
    scores = predict(model, x, edge_index, dataset.native)
    assert history[0]["total"] > 0
    assert scores["chassis_based"].shape == (4,)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(model, path, config)
        restored, _ = load_checkpoint(path)
        restored_scores = predict(restored, x, edge_index, dataset.native)
        assert torch.allclose(
            torch.from_numpy(scores["chassis_based"]),
            torch.from_numpy(restored_scores["chassis_based"]),
            atol=1e-6,
        )
    print("HyperCEF smoke test passed")


if __name__ == "__main__":
    main()
