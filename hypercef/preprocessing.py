"""Helpers for converting tabular metabolite and reaction data to HyperCEF."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .data import PreparedDataset, build_incidence, save_prepared_dataset


VALID_GROUPS = {"native", "chassis_absent", "perturbed"}


def prepare_from_csv(
    metabolites_csv: str | Path,
    reactions_csv: str | Path,
    output_directory: str | Path,
    *,
    fingerprint_dim: int = 2048,
    radius: int = 2,
) -> PreparedDataset:
    """Build a prepared dataset from documented CSV schemas.

    ``metabolites_csv`` must contain ``metabolite_id`` and ``smiles``.
    ``reactions_csv`` must contain ``reaction_id``, ``group``, and a
    semicolon-separated ``metabolite_ids`` field.
    """

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fingerprint_dim)
    metabolite_ids: list[str] = []
    vectors: list[np.ndarray] = []
    with Path(metabolites_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = row["metabolite_id"].strip()
            molecule = Chem.MolFromSmiles(row["smiles"].strip())
            if not identifier or molecule is None:
                raise ValueError(f"Invalid metabolite row: {row}")
            metabolite_ids.append(identifier)
            vectors.append(np.asarray(generator.GetFingerprint(molecule), dtype=np.float32))
    index = {identifier: position for position, identifier in enumerate(metabolite_ids)}
    if len(index) != len(metabolite_ids):
        raise ValueError("metabolite_id values must be unique")

    groups: dict[str, list[list[int]]] = {name: [] for name in VALID_GROUPS}
    reaction_ids: dict[str, list[str]] = {name: [] for name in VALID_GROUPS}
    with Path(reactions_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = row["group"].strip()
            if group not in VALID_GROUPS:
                raise ValueError(f"Unknown reaction group: {group}")
            members = sorted({index[item.strip()] for item in row["metabolite_ids"].split(";") if item.strip()})
            if not members:
                raise ValueError(f"Reaction has no recognized participants: {row}")
            groups[group].append(members)
            reaction_ids[group].append(row["reaction_id"].strip())

    dataset = PreparedDataset(
        x=torch.from_numpy(np.stack(vectors)).float(),
        edge_index=build_incidence(groups["native"]),
        native=groups["native"],
        chassis_absent=groups["chassis_absent"],
        perturbed=groups["perturbed"],
        metabolite_ids=metabolite_ids,
        reaction_ids=reaction_ids,
    )
    save_prepared_dataset(dataset, output_directory)
    return dataset
