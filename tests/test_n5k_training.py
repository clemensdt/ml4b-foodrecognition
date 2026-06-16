"""Tests for Nutrition5k-backed mass-prior training/auditing."""
import json

import pytest

from foodvol import training


def test_n5k_feature_cache_has_expected_scope():
    df = training.load_n5k_features()
    assert len(df) >= 400
    assert df["food_class"].nunique() >= 15
    assert {"area_cm2", "long_cm", "total_mass_g", "total_kcal"} <= set(df.columns)
    assert (df["area_cm2"] > 0).all()
    assert (df["total_mass_g"] > 0).all()


def test_n5k_median_priors_match_committed_metadata():
    df = training.load_n5k_features()
    derived = training.derive_n5k_mass_priors(df, min_samples=5)
    committed = json.load(open("data/n5k_meta/n5k_class_priors.json"))

    # The committed priors should be reproducible from the cached feature table.
    for row in derived.itertuples(index=False):
        meta = committed[row.food_class]
        assert meta["n_samples"] == row.n_samples
        assert meta["mass_per_cm2"] == pytest.approx(row.mass_per_cm2, abs=0.001)
        assert meta["typical_long_cm"] == pytest.approx(row.typical_long_cm, abs=0.001)


def test_n5k_mass_prior_audit_is_finite():
    report = training.evaluate_n5k_mass_priors()
    assert {"food_class", "n", "MAPE", "MAE_g", "mean_mass_g"} <= set(report.columns)
    assert report["MAPE"].notna().all()
    assert report["MAE_g"].notna().all()
    assert (report["n"] > 0).all()
