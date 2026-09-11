#!/usr/bin/env python3
"""Predict reaction feasibility with a trained HyperCEF checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hypercef import load_checkpoint, load_prepared_dataset, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared dataset directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--group", choices=("native", "chassis_absent", "perturbed"), required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    dataset = load_prepared_dataset(args.data)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    reactions = getattr(dataset, args.group)
    identifiers = (dataset.reaction_ids or {}).get(args.group) or [f"{args.group}_{i}" for i in range(len(reactions))]
    scores = predict(
        model,
        dataset.x,
        dataset.edge_index,
        reactions,
        batch_size=args.batch_size,
    )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("reaction_id", "chassis_independent_probability", "chassis_based_probability"))
        writer.writerows(zip(identifiers, scores["chassis_independent"], scores["chassis_based"]))
    print(f"Wrote {len(reactions)} predictions to {destination}")


if __name__ == "__main__":
    main()
