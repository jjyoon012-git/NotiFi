import torch

from notifi_pose.cal1_kp10 import Cal1KP10Adapter


def _example():
    torch.manual_seed(7)
    support = torch.randn(8, 23, 16)
    support_mask = torch.ones(8, 23, dtype=torch.bool)
    support_class = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8])
    query = torch.randn(3, 23, 16)
    query_mask = torch.ones(3, 23, dtype=torch.bool)
    return support, support_mask, support_class, query, query_mask


def test_cal1_initializes_as_exact_identity():
    model = Cal1KP10Adapter(feature_dim=16, token_dim=24, rank=12)
    model.eval()
    support, support_mask, support_class, query, query_mask = _example()
    output = model(
        query, query_mask, support, support_mask, support_class
    )["features"]
    assert torch.equal(output, query)


def test_cal1_support_encoder_is_permutation_invariant():
    model = Cal1KP10Adapter(feature_dim=16, token_dim=24, rank=12)
    model.eval()
    support, support_mask, support_class, _, _ = _example()
    order = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4])
    first = model.encode_support(support, support_mask, support_class)
    second = model.encode_support(
        support[order], support_mask[order], support_class[order]
    )
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_cal1_strength_zero_preserves_query_after_training_changes():
    model = Cal1KP10Adapter(feature_dim=16, token_dim=24, rank=12)
    support, support_mask, support_class, query, query_mask = _example()
    with torch.no_grad():
        model.film[-1].bias.fill_(0.3)
        model.dynamic[-1].bias.fill_(0.2)
    output = model(
        query, query_mask, support, support_mask, support_class,
        strength=0.0,
    )["features"]
    assert torch.equal(output, query)


def test_cal1_query_path_exposes_no_label_argument():
    import inspect

    parameters = inspect.signature(Cal1KP10Adapter.forward).parameters
    assert "query_class" not in parameters
    assert "query_risk" not in parameters
    assert "query_pose" not in parameters
