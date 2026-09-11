"""Portable on-disk representation for preprocessed HyperCEF inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .reaction import reaction_metabolites


@dataclass
class PreparedDataset:
    """A chassis hypergraph and the three reaction groups used for training."""

    x: torch.Tensor
    edge_index: torch.Tensor
    native: list[list[int]]
    chassis_absent: list[list[int]]
    perturbed: list[list[int]]
    metabolite_ids: list[str] | None = None
    reaction_ids: dict[str, list[str]] | None = None

    def validate(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_metabolites, fingerprint_dim]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_incidences]")
        if self.edge_index.numel() and int(self.edge_index[0].max()) >= self.x.shape[0]:
            raise ValueError("edge_index contains a metabolite index outside x")
        for name in ("native", "chassis_absent", "perturbed"):
            reactions = getattr(self, name)
            for reaction in reactions:
                indices = reaction_metabolites(reaction)
                if any(index >= self.x.shape[0] for index in indices):
                    raise ValueError(f"{name} contains a metabolite index outside x")

    def to(self, device: torch.device | str) -> "PreparedDataset":
        return PreparedDataset(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            native=self.native,
            chassis_absent=self.chassis_absent,
            perturbed=self.perturbed,
            metabolite_ids=self.metabolite_ids,
            reaction_ids=self.reaction_ids,
        )


def build_incidence(native_reactions: list[list[int]], device: str | torch.device = "cpu") -> torch.Tensor:
    """Construct a node-to-hyperedge incidence index from native reactions."""

    nodes: list[int] = []
    hyperedges: list[int] = []
    for hyperedge, reaction in enumerate(native_reactions):
        for metabolite in reaction_metabolites(reaction):
            nodes.append(metabolite)
            hyperedges.append(hyperedge)
    return torch.tensor([nodes, hyperedges], dtype=torch.long, device=device)


def save_prepared_dataset(dataset: PreparedDataset, directory: str | Path) -> None:
    """Write features, incidence indices, and reaction groups without pickling."""

    dataset.validate()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "features.npy", dataset.x.detach().cpu().numpy())
    np.save(destination / "edge_index.npy", dataset.edge_index.detach().cpu().numpy())
    metadata = {
        "native": dataset.native,
        "chassis_absent": dataset.chassis_absent,
        "perturbed": dataset.perturbed,
        "metabolite_ids": dataset.metabolite_ids,
        "reaction_ids": dataset.reaction_ids,
    }
    (destination / "reactions.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    """Load a dataset created by :func:`save_prepared_dataset`."""

    source = Path(directory)
    x = torch.from_numpy(np.load(source / "features.npy", allow_pickle=False)).float()
    edge_index = torch.from_numpy(np.load(source / "edge_index.npy", allow_pickle=False)).long()
    metadata = json.loads((source / "reactions.json").read_text(encoding="utf-8"))
    dataset = PreparedDataset(
        x=x,
        edge_index=edge_index,
        native=[reaction_metabolites(row) for row in metadata.get("native", [])],
        chassis_absent=[reaction_metabolites(row) for row in metadata.get("chassis_absent", [])],
        perturbed=[reaction_metabolites(row) for row in metadata.get("perturbed", [])],
        metabolite_ids=metadata.get("metabolite_ids"),
        reaction_ids=metadata.get("reaction_ids"),
    )
    dataset.validate()
    return dataset
