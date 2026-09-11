"""Minimal training and inference loops for prepared HyperCEF datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch

from .data import PreparedDataset
from .losses import ReactionGroup, hypercef_loss
from .model import HyperCEF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_rows(dataset: PreparedDataset) -> tuple[list[list[int]], torch.Tensor]:
    queries = dataset.native + dataset.perturbed + dataset.chassis_absent
    groups = torch.tensor(
        [int(ReactionGroup.NATIVE)] * len(dataset.native)
        + [int(ReactionGroup.PERTURBED)] * len(dataset.perturbed)
        + [int(ReactionGroup.CHASSIS_ABSENT)] * len(dataset.chassis_absent),
        dtype=torch.long,
    )
    if not queries:
        raise ValueError("The prepared dataset contains no training queries")
    return queries, groups


def fit(
    model: HyperCEF,
    dataset: PreparedDataset,
    *,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    independent_weight: float = 0.5,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit HyperCEF and return one mean-loss record per epoch."""

    dataset.validate()
    set_seed(seed)
    device = next(model.parameters()).device
    dataset = dataset.to(device)
    queries, groups = _training_rows(dataset)
    groups = groups.to(device)
    model.fit_topology_boundary_prototypes(
        dataset.x,
        dataset.edge_index,
        dataset.native,
        dataset.chassis_absent,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(queries), generator=generator).tolist()
        totals = {"total": 0.0, "group": 0.0, "independent": 0.0}
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_queries = [queries[index] for index in indices]
            batch_groups = groups[torch.tensor(indices, device=device)]
            outputs = model.forward_outputs(dataset.x, dataset.edge_index, batch_queries)
            loss, parts = hypercef_loss(
                outputs["final_logit"],
                outputs["chemistry_logit"],
                batch_groups,
                independent_weight=independent_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(indices)
            seen += count
            for name in totals:
                totals[name] += float(parts[name]) * count
        history.append({"epoch": epoch, **{name: value / seen for name, value in totals.items()}})
    return history


@torch.no_grad()
def predict(
    model: HyperCEF,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    reactions: Sequence[Sequence[int]],
    *,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Return chassis-independent and chassis-based probabilities."""

    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    independent: list[np.ndarray] = []
    chassis: list[np.ndarray] = []
    for start in range(0, len(reactions), batch_size):
        batch = list(reactions[start : start + batch_size])
        outputs = model.forward_outputs(x, edge_index, batch)
        independent.append(torch.sigmoid(outputs["chemistry_logit"]).cpu().numpy().reshape(-1))
        chassis.append(torch.sigmoid(outputs["final_logit"]).cpu().numpy().reshape(-1))
    return {
        "chassis_independent": np.concatenate(independent) if independent else np.empty(0),
        "chassis_based": np.concatenate(chassis) if chassis else np.empty(0),
    }
