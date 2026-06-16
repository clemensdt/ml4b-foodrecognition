"""Reusable training workflow for the volume and mass-estimation notebooks.

The notebooks should explain and visualise the process, but the actual training
logic lives here so it can be tested and reused from scripts or future notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config, nutrition
from .benchmark import FEATURES_EXTENDED_CACHE, N5K_FEATURES_CACHE
from .volume import FEATURE_NAMES, VolumeEstimator, evaluate

BASE_VOLUME_FEATURES = FEATURE_NAMES
EXTENDED_VOLUME_FEATURES = (
    "area_cm2",
    "height_cm",
    "area_x_height",
    "side_area_cm2",
    "top_aspect",
    "top_circ",
    "top_solidity",
    "top_extent",
    "top_elong",
    "side_aspect",
    "side_circ",
    "side_solidity",
    "side_extent",
    "side_elong",
)

DEPLOYMENT_MODEL_KIND = "gbr"
DEPLOYMENT_FEATURES = BASE_VOLUME_FEATURES
MODEL_KINDS = ("proportional", "gbr", "hist_gbr", "random_forest", "huber", "ridge")


def load_ecustfd_features(path: Path = FEATURES_EXTENDED_CACHE) -> pd.DataFrame:
    """Load the cached ECUSTFD training table and add derived columns."""
    df = pd.read_csv(path)
    return prepare_volume_frame(df)


def prepare_volume_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the canonical volume features present."""
    out = df.copy()
    out["area_x_height"] = out["area_cm2"] * out["height_cm"]
    return out


def _extra_mapping(df: pd.DataFrame, feature_names: Sequence[str]) -> Mapping[str, Sequence[float]]:
    base = set(FEATURE_NAMES)
    return {name: df[name].to_numpy() for name in feature_names if name not in base}


def cross_validated_predictions(
    df: pd.DataFrame,
    *,
    feature_names: Sequence[str] = DEPLOYMENT_FEATURES,
    model_kind: str = DEPLOYMENT_MODEL_KIND,
    splitter,
    groups: Sequence[str] | None = None,
) -> np.ndarray:
    """Out-of-fold volume predictions for one model/feature configuration."""
    work = prepare_volume_frame(df)
    areas = work["area_cm2"].to_numpy()
    heights = work["height_cm"].to_numpy()
    target = work["volume_ml"].to_numpy()
    extra = _extra_mapping(work, feature_names)

    preds = np.zeros(len(work), dtype=float)
    split_iter = splitter.split(work, target, groups) if groups is not None else splitter.split(work)
    for train_idx, test_idx in split_iter:
        train_extra = {k: np.asarray(v)[train_idx] for k, v in extra.items()} or None
        test_extra = {k: np.asarray(v)[test_idx] for k, v in extra.items()} or None
        fallback = VolumeEstimator().fit(
            areas[train_idx],
            heights[train_idx],
            target[train_idx],
            model_kind="proportional",
        )
        est = VolumeEstimator(shape_factor=fallback.shape_factor).fit(
            areas[train_idx],
            heights[train_idx],
            target[train_idx],
            model_kind=model_kind,
            feature_names=feature_names,
            extra_features=train_extra,
        )
        preds[test_idx] = est.predict_many(areas[test_idx], heights[test_idx], test_extra)
    return preds


def score_volume_models(
    df: pd.DataFrame,
    *,
    random_state: int = 0,
    model_kinds: Sequence[str] = MODEL_KINDS,
) -> pd.DataFrame:
    """Compare candidate volume models with random and class-grouped CV.

    Random K-fold estimates in-distribution performance. GroupKFold by
    ``food_type`` estimates the harder question: does the geometry model still
    work on food classes not seen during training?
    """
    from sklearn.model_selection import GroupKFold, KFold

    work = prepare_volume_frame(df)
    y = work["volume_ml"].to_numpy()
    groups = work["food_type"].to_numpy()
    feature_sets = {
        "base": BASE_VOLUME_FEATURES,
        "extended": EXTENDED_VOLUME_FEATURES,
    }
    rows = []
    for feature_set_name, feature_names in feature_sets.items():
        for model_kind in model_kinds:
            if model_kind == "proportional" and feature_set_name != "base":
                continue
            pred_kfold = cross_validated_predictions(
                work,
                feature_names=feature_names,
                model_kind=model_kind,
                splitter=KFold(n_splits=5, shuffle=True, random_state=random_state),
            )
            pred_group = cross_validated_predictions(
                work,
                feature_names=feature_names,
                model_kind=model_kind,
                splitter=GroupKFold(n_splits=5),
                groups=groups,
            )
            k_metrics = evaluate(y, pred_kfold)
            g_metrics = evaluate(y, pred_group)
            rows.append({
                "feature_set": feature_set_name,
                "model_kind": model_kind,
                "kfold_MAE": k_metrics["MAE"],
                "kfold_MAPE": k_metrics["MAPE_percent"],
                "kfold_R2": k_metrics["R2"],
                "group_MAE": g_metrics["MAE"],
                "group_MAPE": g_metrics["MAPE_percent"],
                "group_R2": g_metrics["R2"],
            })
    return pd.DataFrame(rows).sort_values(["kfold_MAPE", "group_MAPE"]).reset_index(drop=True)


def fit_final_volume_model(
    df: pd.DataFrame,
    *,
    model_kind: str = DEPLOYMENT_MODEL_KIND,
    feature_names: Sequence[str] = DEPLOYMENT_FEATURES,
    save_path: Path | None = config.VOLUME_MODEL_PATH,
    metrics: dict | None = None,
) -> VolumeEstimator:
    """Fit the deployment volume model on all ECUSTFD rows and optionally save it."""
    work = prepare_volume_frame(df)
    areas = work["area_cm2"].to_numpy()
    heights = work["height_cm"].to_numpy()
    volumes = work["volume_ml"].to_numpy()
    extra = _extra_mapping(work, feature_names) or None

    fallback = VolumeEstimator().fit(areas, heights, volumes, model_kind="proportional")
    est = VolumeEstimator(shape_factor=fallback.shape_factor).fit(
        areas,
        heights,
        volumes,
        model_kind=model_kind,
        feature_names=feature_names,
        extra_features=extra,
    )
    est.metrics = {
        "fallback_shape_factor": fallback.shape_factor,
        "training_rows": float(len(work)),
        "training_classes": float(work["food_type"].nunique()),
    }
    if metrics:
        est.metrics.update(metrics)
    if save_path is not None:
        est.save(save_path)
    return est


def evaluate_n5k_mass_priors(path: Path = N5K_FEATURES_CACHE) -> pd.DataFrame:
    """Evaluate class ``mass_per_cm2`` priors on the cached Nutrition5k subset."""
    df = load_n5k_features(path)
    rows = []
    for food_class, group in df.groupby("food_class"):
        info = nutrition.lookup(food_class)
        pred = group["area_cm2"].to_numpy() * float(info.mass_per_cm2 or 0.0)
        metrics = evaluate(group["total_mass_g"].to_numpy(), pred)
        rows.append({
            "food_class": food_class,
            "n": len(group),
            "MAPE": metrics["MAPE_percent"],
            "MAE_g": metrics["MAE"],
            "mean_mass_g": float(group["total_mass_g"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["MAPE", "n"], ascending=[True, False])


def load_n5k_features(path: Path = N5K_FEATURES_CACHE) -> pd.DataFrame:
    """Load cached Nutrition5k top-view mass features.

    Nutrition5k has measured mass and macros, but no side-view height or measured
    volume. It therefore trains/audits the no-side-view mass fallback, not the
    ECUSTFD volume regressor.
    """
    df = pd.read_csv(path)
    required = {
        "dish_id",
        "food_class",
        "area_cm2",
        "long_cm",
        "short_cm",
        "total_mass_g",
        "total_kcal",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"N5K feature table missing columns: {sorted(missing)}")
    return df


def derive_n5k_mass_priors(
    df: pd.DataFrame,
    *,
    min_samples: int = 5,
    statistic: str = "median",
) -> pd.DataFrame:
    """Derive compact class priors for the top-view mass fallback.

    The robust default is the per-class median of ``mass / area``. This is what
    the committed ``data/n5k_meta/n5k_class_priors.json`` stores for sufficiently
    represented classes.
    """
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    work = load_n5k_features_from_frame(df)
    rows = []
    for food_class, group in work.groupby("food_class"):
        if len(group) < min_samples:
            continue
        ratio = group["total_mass_g"] / group["area_cm2"]
        reducer = ratio.median if statistic == "median" else ratio.mean
        rows.append({
            "food_class": food_class,
            "n_samples": int(len(group)),
            "typical_long_cm": float(group["long_cm"].median()),
            "mass_per_cm2": float(reducer()),
        })
    return pd.DataFrame(rows).sort_values(["n_samples", "food_class"], ascending=[False, True])


def load_n5k_features_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an already-loaded N5K feature DataFrame."""
    required = {"food_class", "area_cm2", "long_cm", "total_mass_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"N5K feature frame missing columns: {sorted(missing)}")
    work = df.copy()
    work = work[
        np.isfinite(work["area_cm2"])
        & np.isfinite(work["long_cm"])
        & np.isfinite(work["total_mass_g"])
        & (work["area_cm2"] > 0)
        & (work["total_mass_g"] > 0)
    ]
    return work
