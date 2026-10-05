"""Tests for modules.fusion."""

from modules.fusion import fuse_risk


def test_weights_sum_to_one():
    f = fuse_risk(80, 80, 80)
    total = sum(m.weight for m in f.modules)
    assert abs(total - 1.0) < 1e-6


def test_uniform_input_equals_output():
    f = fuse_risk(80, 80, 80)
    assert f.overall_score == 80


def test_all_zero_is_low():
    f = fuse_risk(0, 0, 0)
    assert f.overall_score == 0
    assert f.risk_level == "LOW"


def test_all_hundred_is_high():
    f = fuse_risk(100, 100, 100)
    assert f.overall_score == 100
    assert f.risk_level == "HIGH"


def test_floor_rule_forces_high_when_module_at_90():
    f = fuse_risk(90, 0, 0)
    assert f.risk_level == "HIGH"


def test_floor_rule_forces_medium_when_module_at_70():
    f = fuse_risk(70, 0, 0)
    assert f.risk_level in ("MEDIUM", "HIGH")


def test_boundary_39_is_low():
    f = fuse_risk(39, 39, 39)
    assert f.risk_level == "LOW"


def test_boundary_70_is_high():
    f = fuse_risk(70, 70, 70)
    assert f.risk_level == "HIGH"


def test_uses_default_weights_when_missing_yaml():
    f = fuse_risk(50, 50, 50)
    assert set(f.weights_used.keys()) == {"scam", "transaction", "behavior"}