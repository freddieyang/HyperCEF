"""Label routing and objective used by the primary HyperCEF model."""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn.functional as F


class ReactionGroup(IntEnum):
    """Training groups and their available labels."""

    NATIVE = 0
    PERTURBED = 1
    CHASSIS_ABSENT = 2


def _mean_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        raise ValueError("A loss term received an empty batch")
    return F.binary_cross_entropy_with_logits(logits, targets)


def hypercef_loss(
    chassis_logits: torch.Tensor,
    independent_logits: torch.Tensor,
    groups: torch.Tensor,
    independent_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute group-routed and chassis-independent binary cross-entropy.

    Native reactions are positive for both outputs. Chassis-absent biochemical
    reactions are molecular positives and chassis negatives. Perturbed
    participant sets are molecular negatives and have no chassis label.
    """

    groups = groups.reshape(-1).to(device=chassis_logits.device)
    chassis_logits = chassis_logits.reshape(-1)
    independent_logits = independent_logits.reshape(-1)
    if not (len(groups) == len(chassis_logits) == len(independent_logits)):
        raise ValueError("groups and logits must contain the same number of queries")

    is_native = groups == int(ReactionGroup.NATIVE)
    is_perturbed = groups == int(ReactionGroup.PERTURBED)
    is_absent = groups == int(ReactionGroup.CHASSIS_ABSENT)
    if not torch.all(is_native | is_perturbed | is_absent):
        raise ValueError("groups may contain only native, perturbed, or chassis-absent labels")

    routed_logits = torch.where(is_perturbed, independent_logits, chassis_logits)
    routed_targets = is_native.to(dtype=routed_logits.dtype)
    independent_targets = (is_native | is_absent).to(dtype=independent_logits.dtype)

    group_loss = _mean_bce(routed_logits, routed_targets)
    independent_loss = _mean_bce(independent_logits, independent_targets)
    total = group_loss + float(independent_weight) * independent_loss
    return total, {
        "total": total.detach(),
        "group": group_loss.detach(),
        "independent": independent_loss.detach(),
    }
