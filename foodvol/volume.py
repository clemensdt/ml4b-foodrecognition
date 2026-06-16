"""Stage E — volume estimation: footprint area + height -> physical volume.

This is the part we *train*. The geometry gives two metric measurements:

* **footprint area** (cm^2) from the top view — pixels inside the food mask scaled
  by the calibration;
* **height** (cm) from the side view — the food mask's vertical extent scaled by the
  calibration.

A bounding prism has volume ``area * height``; the true volume is some fraction of
that (a "shape factor") which depends on how the food piles up. Rather than guess the
factor, we **learn** ``volume = f(area, height)`` from ECUSTFD's ground-truth volumes
with a small, robust regressor. Crucially the model is **class-agnostic**: volume is
pure geometry, and per-class density/energy are handled separately in
:mod:`foodvol.nutrition`, which helps it generalise to unseen foods.

Upgrade path (GPU): replace the area/height features + linear model with a deep
multi-view network that regresses volume directly from the two images. The
:class:`VolumeEstimator` API (``fit``/``predict_volume``/``save``/``load``) is the
seam to swap in such a model without touching the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np

from . import config
from .calibration import Calibration

if TYPE_CHECKING:  # avoid importing the perception stack at module load
    from .segmentation import FoodSegmenter, InstanceMask


# --- geometric measurements ----------------------------------------------------
@dataclass
class Measurement:
    """A single geometric measurement plus the mask it came from."""

    value: float                      # cm^2 (area) or cm (height)
    mask: Optional["InstanceMask"]
    ok: bool


def measure_footprint_area_cm2(
    image_bgr: np.ndarray,
    food_box: tuple[int, int, int, int],
    calib: Calibration,
    segmenter: "FoodSegmenter",
) -> Measurement:
    """Top-view footprint area in cm^2 (segment the food, scale its pixel area)."""
    inst = segmenter.segment_box(image_bgr, food_box)
    if inst is None:
        return Measurement(float("nan"), None, ok=False)
    return Measurement(calib.pixel_area_to_cm2(inst.area_px), inst, ok=True)


def measure_height_cm(
    image_bgr: np.ndarray,
    food_box: tuple[int, int, int, int],
    calib: Calibration,
    segmenter: "FoodSegmenter",
) -> Measurement:
    """Side-view height in cm (vertical extent of the food mask, scaled)."""
    inst = segmenter.segment_box(image_bgr, food_box)
    if inst is None:
        return Measurement(float("nan"), None, ok=False)
    _, _, _, h_px = inst.bbox
    return Measurement(calib.pixel_length_to_cm(h_px), inst, ok=True)


# --- feature engineering -------------------------------------------------------
FEATURE_NAMES = ("area_cm2", "height_cm", "area_x_height")


def volume_features(area_cm2: float, height_cm: float) -> np.ndarray:
    """Feature vector for the regressor.

    ``area * height`` (a bounding-prism volume) is the physically dominant term;
    ``area`` and ``height`` let the model correct for shapes that scale differently.
    """
    return np.array([area_cm2, height_cm, area_cm2 * height_cm], dtype=np.float64)


def _feature_values(
    area_cm2: float,
    height_cm: float,
    extra_features: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    values = {
        "area_cm2": float(area_cm2),
        "height_cm": float(height_cm),
        "area_x_height": float(area_cm2) * float(height_cm),
    }
    if extra_features:
        values.update({k: float(v) for k, v in extra_features.items()})
    return values


def features_for_names(
    area_cm2: float,
    height_cm: float,
    feature_names: Sequence[str] = FEATURE_NAMES,
    extra_features: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Build one feature vector by name.

    The app normally has only the base geometry. Training notebooks may include
    extra descriptors, but the feature list must be stored with the artifact so
    inference can reproduce the same column order.
    """
    values = _feature_values(area_cm2, height_cm, extra_features)
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise KeyError(f"missing volume feature(s): {', '.join(missing)}")
    return np.array([values[name] for name in feature_names], dtype=np.float64)


def feature_matrix(
    areas: Sequence[float],
    heights: Sequence[float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    extra_features: Optional[Mapping[str, Sequence[float]]] = None,
) -> np.ndarray:
    rows = []
    for idx, (area, height) in enumerate(zip(areas, heights)):
        extra = None
        if extra_features:
            extra = {name: np.asarray(values)[idx] for name, values in extra_features.items()}
        rows.append(features_for_names(area, height, feature_names, extra))
    return np.vstack(rows)


# --- estimator -----------------------------------------------------------------
class VolumeEstimator:
    """Predicts food volume (mL) from footprint area and height.

    Before training (or if loading fails) it falls back to the physics estimate
    ``volume = shape_factor * area * height``.
    """

    def __init__(
        self,
        shape_factor: float = 0.5,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ):
        self.shape_factor = shape_factor
        self.model = None              # sklearn regressor once fitted (None for 'proportional')
        self.fitted = False            # True once fit() has run (model or learned shape factor)
        self.metrics: dict[str, float] = {}
        self.feature_names = tuple(feature_names)
        self.model_kind = "physics_fallback"

    # --- training ---
    def fit(
        self,
        areas: Sequence[float],
        heights: Sequence[float],
        volumes_ml: Sequence[float],
        model_kind: str = "huber",
        feature_names: Sequence[str] = FEATURE_NAMES,
        extra_features: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> "VolumeEstimator":
        """Fit ``volume = f(area, height)`` on measured ground-truth volumes."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.feature_names = tuple(feature_names)
        X = feature_matrix(areas, heights, self.feature_names, extra_features)
        y = np.asarray(volumes_ml, dtype=np.float64)
        self.model_kind = model_kind

        if model_kind == "proportional":
            # Physics with a *learned* shape factor: volume = k * area * height.
            # Single parameter through the origin -> robust and extrapolates cleanly,
            # which matters for an open-world app with portions larger than training.
            ah = X[:, 2]
            self.shape_factor = float(np.sum(ah * y) / np.sum(ah * ah))
            self.model = None
            self.fitted = True
            return self
        if model_kind == "huber":
            from sklearn.linear_model import HuberRegressor
            self.model = make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))
        elif model_kind == "ridge":
            from sklearn.linear_model import Ridge
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        elif model_kind == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(
                    learning_rate=0.03,
                    max_depth=2,
                    n_estimators=200,
                    random_state=0,
                ),
            )
        elif model_kind == "hist_gbr":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=0,
            )
        elif model_kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=0,
            )
        elif model_kind == "linear":
            from sklearn.linear_model import LinearRegression
            self.model = make_pipeline(StandardScaler(), LinearRegression())
        else:
            raise ValueError(f"unknown model_kind: {model_kind!r}")

        self.model.fit(X, y)
        self.fitted = True
        return self

    # --- inference ---
    def predict_volume(
        self,
        area_cm2: float,
        height_cm: float,
        extra_features: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Predict volume in mL for a single (area, height) pair."""
        if self.model is not None:
            try:
                X = features_for_names(
                    area_cm2, height_cm, self.feature_names, extra_features,
                ).reshape(1, -1)
                return float(max(0.0, self.model.predict(X)[0]))
            except KeyError:
                # Deployment should keep running if a research artifact needs
                # features the live app cannot measure.
                pass
        return float(self.shape_factor * area_cm2 * height_cm)

    def predict_many(
        self,
        areas: Sequence[float],
        heights: Sequence[float],
        extra_features: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> np.ndarray:
        if self.model is not None:
            try:
                X = feature_matrix(areas, heights, self.feature_names, extra_features)
                preds = self.model.predict(X)
                return np.clip(preds, 0.0, None)
            except KeyError:
                pass
        return np.array([self.shape_factor * a * h for a, h in zip(areas, heights)])

    @property
    def is_trained(self) -> bool:
        return self.fitted

    # --- persistence ---
    def save(self, path: Path = config.VOLUME_MODEL_PATH) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "artifact_version": 2,
            "kind": "foodvol.volume.VolumeEstimator",
            "shape_factor": self.shape_factor,
            "model": self.model,
            "model_kind": self.model_kind,
            "features": list(self.feature_names),
            "fitted": self.fitted,
            "metrics": self.metrics,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path = config.VOLUME_MODEL_PATH) -> "VolumeEstimator":
        import joblib
        est = cls()
        try:
            blob = joblib.load(path)
            if isinstance(blob, cls):
                return blob
            est.shape_factor = blob.get("shape_factor", 0.5)
            est.model = blob.get("model")
            est.fitted = blob.get("fitted", est.model is not None)
            est.feature_names = tuple(blob.get("features", blob.get("feature_names", FEATURE_NAMES)))
            est.model_kind = blob.get("model_kind", blob.get("winner", "loaded_model"))
            est.metrics = blob.get("metrics", {})
            for key in ("metrics_val", "metrics_test", "metrics_test_mass"):
                if key in blob:
                    est.metrics[key] = blob[key]
        except Exception as exc:
            print(f"[volume] could not load trained model ({exc}); using physics fallback.")
        return est


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression error metrics (MAE, RMSE, MAPE, R^2)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}
