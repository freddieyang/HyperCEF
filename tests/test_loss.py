import torch

from hypercef.losses import ReactionGroup, hypercef_loss


def test_group_routing_matches_manual_objective():
    chassis = torch.tensor([[2.0], [8.0], [-1.0]])
    independent = torch.tensor([[1.0], [-2.0], [3.0]])
    groups = torch.tensor(
        [ReactionGroup.NATIVE, ReactionGroup.PERTURBED, ReactionGroup.CHASSIS_ABSENT]
    )
    total, parts = hypercef_loss(chassis, independent, groups)
    routed = torch.tensor([2.0, -2.0, -1.0])
    group_targets = torch.tensor([1.0, 0.0, 0.0])
    molecular_targets = torch.tensor([1.0, 0.0, 1.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(routed, group_targets)
    expected = expected + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        independent.reshape(-1), molecular_targets
    )
    assert torch.allclose(total, expected)
    assert set(parts) == {"total", "group", "independent"}
