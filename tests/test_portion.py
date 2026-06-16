"""Tests for the shared portion quantity logic."""
import pytest

from foodvol import nutrition
from foodvol.portion import area_mass_prior, estimate_quantity, geometric_height_prior
from foodvol.volume import VolumeEstimator


def test_without_height_uses_area_mass_prior():
    info = nutrition.lookup("apple")
    est = VolumeEstimator(shape_factor=0.5)
    qty = estimate_quantity(info, area_cm2=45.0, volume_model=est)
    assert qty.source == "area_mass_prior"
    assert qty.mass_g == pytest.approx(area_mass_prior(info, 45.0))
    assert qty.volume_ml == pytest.approx(qty.mass_g / info.density_g_per_ml)


def test_with_height_uses_volume_model_and_density():
    info = nutrition.lookup("apple")
    est = VolumeEstimator(shape_factor=0.5)
    qty = estimate_quantity(
        info,
        area_cm2=40.0,
        volume_model=est,
        height_cm=5.0,
        height_source="side_chessboard",
        clamp=False,
    )
    assert qty.source == "volume_model:side_chessboard"
    assert qty.raw_volume_ml == pytest.approx(100.0)
    assert qty.mass_g == pytest.approx(100.0 * info.density_g_per_ml)


def test_quantity_clamps_implausible_mass():
    info = nutrition.lookup("apple")
    est = VolumeEstimator(shape_factor=10.0)
    qty = estimate_quantity(info, area_cm2=100.0, volume_model=est, height_cm=10.0)
    assert qty.clamped
    assert qty.clamp_bound == "max"
    assert qty.mass_g == pytest.approx(info.mass_max_g)
    assert "clamped_max" in qty.source


def test_geometric_height_prior_is_bounded():
    assert 0.5 <= geometric_height_prior(0.01) <= 6.0
    assert 0.5 <= geometric_height_prior(10000.0) <= 6.0
