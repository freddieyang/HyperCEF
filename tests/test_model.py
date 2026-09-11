import torch

from hypercef import HyperCEF


def test_forward_shapes_and_probabilities():
    torch.manual_seed(1)
    model = HyperCEF(64, 32, 1, dropout=0.0, num_layers=2, heads=2)
    x = torch.randint(0, 2, (8, 64), dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long
    )
    native = [[0, 1], [1, 2], [3, 4], [5, 6]]
    absent = [[0, 7], [2, 6]]
    model.fit_topology_boundary_prototypes(x, edge_index, native, absent)
    outputs = model.forward_outputs(x, edge_index, [[0, 1], [2, 6]])
    assert outputs["chemistry_logit"].shape == (2, 1)
    assert outputs["final_logit"].shape == (2, 1)
    assert torch.isfinite(outputs["chemistry_logit"]).all()
    assert torch.isfinite(outputs["final_logit"]).all()


def test_molecular_residual_is_bounded_and_detached():
    model = HyperCEF(64, 32, 1, dropout=0.0, num_layers=2, heads=2)
    molecular_logit = torch.tensor([[-10.0], [10.0]], requires_grad=True)
    residual = model._molecular_to_chassis_residual(molecular_logit)
    assert residual.abs().max() <= 0.5
    assert not residual.requires_grad
