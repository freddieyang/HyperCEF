#!/usr/bin/env python3
"""Train HyperCEF on a prepared chassis dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hypercef import HyperCEF, fit, load_prepared_dataset, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared dataset directory")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="runs/hypercef")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset = load_prepared_dataset(args.data)
    model_config = {"in_channels": int(dataset.x.shape[1]), **config["model"]}
    model = HyperCEF(**model_config).to(args.device)
    history = fit(model, dataset, **config["training"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output / "model.pt", model_config)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint and training history to {output}")


if __name__ == "__main__":
    main()
