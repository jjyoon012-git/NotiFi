import inspect

import torch

from notifi_pose import contract as C
from notifi_pose.cal2_kp10 import RawLinkCanonicalizer, support_statistics


def _example():
    torch.manual_seed(19)
    support = torch.randn(8, 21, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    support_mask = torch.ones(8, 21, C.N_LINKS, dtype=torch.bool)
    support_class = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8])
    query = torch.randn(2, 21, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    query_mask = torch.ones(2, 21, C.N_LINKS, dtype=torch.bool)
    return support, support_mask, support_class, query, query_mask


def test_cal2_strength_zero_is_exact_raw_identity():
    model = RawLinkCanonicalizer(token_dim=24, basis_rank=4, lowpass_window=7)
    support, support_mask, support_class, query, query_mask = _example()
    context = model.encode_support(support, support_mask, support_class)
    output = model(query, query_mask, context, strength=0.0)["csi"]
    assert torch.equal(output, query)


def test_cal2_support_statistics_are_permutation_invariant():
    support, support_mask, _, _, _ = _example()
    order = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    first = support_statistics(support, support_mask)
    second = support_statistics(support[order], support_mask[order])
    for key in first:
        assert torch.allclose(first[key], second[key], atol=1e-6, rtol=1e-6)


def test_cal2_support_token_is_permutation_invariant_in_eval():
    model = RawLinkCanonicalizer(token_dim=24, basis_rank=4, lowpass_window=7)
    model.eval()
    support, support_mask, support_class, _, _ = _example()
    order = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    first = model.encode_support(support, support_mask, support_class)["token"]
    second = model.encode_support(
        support[order], support_mask[order], support_class[order]
    )["token"]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_cal2_fitted_alignment_is_finite_and_shape_preserving():
    model = RawLinkCanonicalizer(token_dim=24, basis_rank=4, lowpass_window=7)
    support, support_mask, support_class, query, query_mask = _example()
    context = model.encode_support(support, support_mask, support_class)
    model.set_reference(context["mean"], context["std"], context["dynamic"])
    output = model(query, query_mask, context, strength=1.0)["csi"]
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_cal2_query_path_has_no_target_arguments():
    parameters = inspect.signature(RawLinkCanonicalizer.forward).parameters
    assert "query_class" not in parameters
    assert "query_risk" not in parameters
    assert "query_pose" not in parameters
