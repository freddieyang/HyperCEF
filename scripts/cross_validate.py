#!/usr/bin/env python3
"""Run participant-set-grouped cross-validation on a prepared dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from hypercef import (
    HyperCEF,
    PreparedDataset,
    binary_metrics,
    build_incidence,
    fit,
    load_prepared_dataset,
    predict,
    save_checkpoint,
)


def participant_key(reaction: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(index) for index in reaction)))


def grouped_folds(reactions: list[list[int]], folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    keys = sorted({participant_key(reaction) for reaction in reactions})
    if len(keys) < folds:
        raise ValueError(f"Need at least {folds} unique participant sets; found {len(keys)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys)).tolist()
    shuffled_keys = [keys[index] for index in order]
    validation_keys = [set(shuffled_keys[fold::folds]) for fold in range(folds)]
    assignments = []
    all_indices = list(range(len(reactions)))
    for held_out in validation_keys:
        validation = [index for index, reaction in enumerate(reactions) if participant_key(reaction) in held_out]
        training = [index for index in all_indices if index not in set(validation)]
        assignments.append((training, validation))
    return assignments


def select(rows: list[list[int]], indices: list[int]) -> list[list[int]]:
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="runs/cross_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset = load_prepared_dataset(args.data)
    seed = int(config["training"]["seed"])
    split_groups = {
        "native": grouped_folds(dataset.native, args.folds, seed),
        "chassis_absent": grouped_folds(dataset.chassis_absent, args.folds, seed + 1),
        "perturbed": grouped_folds(dataset.perturbed, args.folds, seed + 2),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict[str, float | int]] = []

    for fold in range(args.folds):
        native_train, native_valid = split_groups["native"][fold]
        absent_train, absent_valid = split_groups["chassis_absent"][fold]
        perturbed_train, perturbed_valid = split_groups["perturbed"][fold]
        train_native = select(dataset.native, native_train)
        train_dataset = PreparedDataset(
            x=dataset.x,
            edge_index=build_incidence(train_native),
            native=train_native,
            chassis_absent=select(dataset.chassis_absent, absent_train),
            perturbed=select(dataset.perturbed, perturbed_train),
        )
        model_config = {"in_channels": int(dataset.x.shape[1]), **config["model"]}
        model = HyperCEF(**model_config).to(args.device)
        history = fit(model, train_dataset, **config["training"])

        valid_native = select(dataset.native, native_valid)
        valid_absent = select(dataset.chassis_absent, absent_valid)
        valid_perturbed = select(dataset.perturbed, perturbed_valid)
        molecular_queries = valid_native + valid_perturbed
        chassis_queries = valid_native + valid_absent
        molecular_scores = predict(model, dataset.x, train_dataset.edge_index, molecular_queries)[
            "chassis_independent"
        ]
        chassis_scores = predict(model, dataset.x, train_dataset.edge_index, chassis_queries)["chassis_based"]
        molecular_labels = np.asarray([1] * len(valid_native) + [0] * len(valid_perturbed))
        chassis_labels = np.asarray([1] * len(valid_native) + [0] * len(valid_absent))
        metrics = {"fold": fold + 1}
        metrics.update({f"independent_{key}": value for key, value in binary_metrics(molecular_labels, molecular_scores).items()})
        metrics.update({f"chassis_{key}": value for key, value in binary_metrics(chassis_labels, chassis_scores).items()})
        fold_metrics.append(metrics)

        fold_dir = output / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, fold_dir / "model.pt", model_config)
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (fold_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("output", "label", "probability"))
            writer.writerows(("chassis_independent", int(label), float(score)) for label, score in zip(molecular_labels, molecular_scores))
            writer.writerows(("chassis_based", int(label), float(score)) for label, score in zip(chassis_labels, chassis_scores))
        print(f"Completed fold {fold + 1}/{args.folds}")

    metric_names = [key for key in fold_metrics[0] if key != "fold"]
    summary = {
        key: {
            "mean": float(np.nanmean([float(row[key]) for row in fold_metrics])),
            "standard_deviation": float(np.nanstd([float(row[key]) for row in fold_metrics])),
        }
        for key in metric_names
    }
    (output / "metrics.json").write_text(
        json.dumps({"folds": fold_metrics, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"Saved cross-validation results to {output}")


if __name__ == "__main__":
    main()
