"""Tests for the volume estimator and its metrics."""
import numpy as np
import pytest

from foodvol.volume import VolumeEstimator, evaluate, feature_matrix, features_for_names, volume_features


def test_physics_fallback_before_training():
    est = VolumeEstimator(shape_factor=0.5)
    assert not est.is_trained
    assert est.predict_volume(60.0, 5.0) == pytest.approx(0.5 * 60.0 * 5.0)


def test_features_dominant_term():
    f = volume_features(60.0, 5.0)
    assert f.tolist() == [60.0, 5.0, 300.0]


def test_proportional_recovers_shape_factor():
    rng = np.random.default_rng(0)
    areas = rng.uniform(20, 120, 200)
    heights = rng.uniform(2, 9, 200)
    volumes = 0.62 * areas * heights  # ground-truth shape factor
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="proportional")
    assert est.is_trained
    assert est.shape_factor == pytest.approx(0.62, rel=1e-6)
    assert est.predict_volume(100.0, 6.0) == pytest.approx(0.62 * 600.0, rel=1e-6)


def test_linear_model_fits_linear_data():
    rng = np.random.default_rng(1)
    areas = rng.uniform(20, 120, 300)
    heights = rng.uniform(2, 9, 300)
    volumes = 0.5 * areas * heights
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="huber")
    preds = est.predict_many(areas, heights)
    assert evaluate(volumes, preds)["MAPE_percent"] < 5.0


def test_save_load_roundtrip(tmp_path):
    areas = np.linspace(20, 120, 50)
    heights = np.linspace(2, 9, 50)
    volumes = 0.55 * areas * heights
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="proportional")
    path = est.save(tmp_path / "vol.joblib")
    loaded = VolumeEstimator.load(path)
    assert loaded.is_trained
    assert loaded.shape_factor == pytest.approx(est.shape_factor)
    assert loaded.feature_names == est.feature_names


def test_named_features_and_artifact_roundtrip(tmp_path):
    areas = np.linspace(20, 120, 50)
    heights = np.linspace(2, 9, 50)
    side_area = areas * 0.7
    volumes = 0.55 * areas * heights
    features = ("area_cm2", "height_cm", "area_x_height", "side_area_cm2")
    est = VolumeEstimator().fit(
        areas,
        heights,
        volumes,
        model_kind="ridge",
        feature_names=features,
        extra_features={"side_area_cm2": side_area},
    )
    loaded = VolumeEstimator.load(est.save(tmp_path / "extended.joblib"))
    pred = loaded.predict_volume(
        80.0,
        4.0,
        extra_features={"side_area_cm2": 56.0},
    )
    assert loaded.feature_names == features
    assert pred > 0


def test_missing_research_feature_falls_back_to_shape_factor():
    est = VolumeEstimator(shape_factor=0.4, feature_names=("area_cm2", "side_area_cm2"))
    class NeedsSideArea:
        def predict(self, X):
            raise AssertionError("model should not be called without side_area_cm2")
    est.model = NeedsSideArea()
    est.fitted = True
    assert est.predict_volume(50.0, 2.0) == pytest.approx(40.0)


def test_feature_matrix_named_columns():
    X = feature_matrix(
        [10.0],
        [2.0],
        feature_names=("area_cm2", "area_x_height", "foo"),
        extra_features={"foo": [7.0]},
    )
    assert X.tolist() == [[10.0, 20.0, 7.0]]
    with pytest.raises(KeyError):
        features_for_names(10.0, 2.0, ("missing",))


def test_evaluate_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = evaluate(y, y)
    assert m["MAE"] == 0 and m["MAPE_percent"] == 0
    assert m["R2"] == pytest.approx(1.0)
