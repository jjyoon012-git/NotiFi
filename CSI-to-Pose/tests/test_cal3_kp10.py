import inspect

import torch

from notifi_pose.cal3_kp10 import SafeSupportFeatureAdapter


def test_cal3_initial_state_is_exact_identity():
    torch.manual_seed(3)
    model = SafeSupportFeatureAdapter(feature_dim=16, rank=4)
    features = torch.randn(2, 31, 16)
    mask = torch.ones(2, 31, dtype=torch.bool)
    assert torch.equal(model(features, mask), features)


def test_cal3_strength_zero_remains_identity_after_fitting():
    model = SafeSupportFeatureAdapter(feature_dim=16, rank=4)
    features = torch.randn(2, 31, 16)
    mask = torch.ones(2, 31, dtype=torch.bool)
    with torch.no_grad():
        model.scale.fill_(0.7)
        model.bias.fill_(0.4)
        model.up.weight.fill_(0.1)
    assert torch.equal(model(features, mask, strength=0.0), features)


def test_cal3_masks_residual_on_invalid_frames():
    model = SafeSupportFeatureAdapter(feature_dim=16, rank=4)
    features = torch.randn(2, 31, 16)
    mask = torch.ones(2, 31, dtype=torch.bool)
    mask[:, -4:] = False
    with torch.no_grad():
        model.bias.fill_(0.5)
    output = model(features, mask)
    assert torch.equal(output[:, -4:], features[:, -4:])


def test_cal3_query_path_has_no_target_arguments():
    parameters = inspect.signature(SafeSupportFeatureAdapter.forward).parameters
    assert "query_class" not in parameters
    assert "query_risk" not in parameters
    assert "query_pose" not in parameters
