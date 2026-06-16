"""ECUSTFD feature extraction and volume-model evaluation.

Reusable engine shared by the feasibility notebook and the training script, so the
measurement and evaluation logic lives in one tested place.

For each portion it derives the two geometric features from the dataset's own
annotations (coin box -> metric scale, food box -> region to segment):

* ``area_cm2``   from a **top** image
* ``height_cm``  from a **side** image

and pairs them with the ground-truth ``volume_ml`` and ``weight_g``. It then
cross-validates ``VolumeEstimator`` and compares it to the physics baseline.

Classification is intentionally *not* used here: ECUSTFD's classes (raw fruit) are
outside the Food-101 classifier's vocabulary, and the volume model is class-agnostic
anyway. Per-class density for the mass error comes from :mod:`foodvol.nutrition`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config, nutrition
from .calibration import calibrate_manual
from .datasets import COIN_DIAMETER_CM, ECUSTFD, BBox
from .segmentation import FoodSegmenter
from .volume import VolumeEstimator, evaluate, measure_footprint_area_cm2, measure_height_cm

FEATURES_CACHE = config.ARTIFACTS_DIR / "ecustfd_features.csv"
FEATURES_EXTENDED_CACHE = config.ARTIFACTS_DIR / "ecustfd_features_extended.csv"
N5K_FEATURES_CACHE = config.ARTIFACTS_DIR / "n5k_features.csv"
N5K_SCALE_PATH = config.ARTIFACTS_DIR / "n5k_scale.json"


def _shape_descriptors(mask) -> dict[str, float]:
    """Return scale-invariant shape descriptors of a boolean mask.

    These tell a regressor *how the food is shaped*, independently of size:

    * ``aspect_ratio``  bounding-box width / height (>1: wide, <1: tall)
    * ``circularity``   4πA / P² in [0, 1] — 1.0 == perfect circle, lower == jagged
    * ``solidity``      mask area / convex-hull area in (0, 1] — porous/spiky things
                        like broccoli have low solidity, smooth blobs ~1.0
    * ``extent``        mask area / bounding-box area
    * ``elongation``    1 - (minor/major axis) of the fitted ellipse — round → 0
    """
    import cv2
    import numpy as np

    if mask is None or not mask.any():
        return dict(aspect_ratio=1.0, circularity=0.0, solidity=0.0,
                    extent=0.0, elongation=0.0)
    m = mask.astype("uint8")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return dict(aspect_ratio=1.0, circularity=0.0, solidity=0.0,
                    extent=0.0, elongation=0.0)
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    elong = 0.0
    if len(cnt) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
        major, minor = max(ax1, ax2), min(ax1, ax2)
        elong = 1.0 - minor / max(major, 1e-6)
    return dict(
        aspect_ratio=float(w) / max(h, 1),
        circularity=4 * 3.14159265 * area / max(perim ** 2, 1e-6),
        solidity=area / hull_area,
        extent=area / max(w * h, 1),
        elongation=float(elong),
    )


def _first_usable(ds: ECUSTFD, images) -> Optional[tuple[Path, BBox, BBox]]:
    """Return the first (image, coin_box, food_box) with both boxes annotated."""
    for img in images:
        coin = ds.coin_box(img.stem)
        foods = ds.food_boxes(img.stem)
        if coin is not None and foods:
            return img, coin, foods[0]
    return None


def extract_features(
    ds: Optional[ECUSTFD] = None,
    segmenter: Optional[FoodSegmenter] = None,
    cache_path: Path = FEATURES_CACHE,
    use_cache: bool = True,
    limit: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Build (or load) the per-portion feature table for single-food portions."""
    import cv2

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        if limit:
            df = df.head(limit)
        if progress:
            print(f"Loaded cached features for {len(df)} portions from {cache_path}")
        return df

    ds = ds or ECUSTFD()
    segmenter = segmenter or FoodSegmenter()
    portions = ds.portions(single_food_only=True)
    if limit:
        portions = portions[:limit]

    rows, skipped = [], 0
    for k, p in enumerate(portions, 1):
        top = _first_usable(ds, p.top_images)
        side = _first_usable(ds, p.side_images)
        if top is None or side is None:
            skipped += 1
            continue

        top_path, top_coin, top_food = top
        side_path, side_coin, side_food = side
        top_img = cv2.imread(str(top_path))
        side_img = cv2.imread(str(side_path))

        calib_top = calibrate_manual(top_coin.diameter_px, COIN_DIAMETER_CM, top_coin.center)
        calib_side = calibrate_manual(side_coin.diameter_px, COIN_DIAMETER_CM, side_coin.center)

        area = measure_footprint_area_cm2(top_img, top_food.box, calib_top, segmenter)
        height = measure_height_cm(side_img, side_food.box, calib_side, segmenter)
        if not (area.ok and height.ok):
            skipped += 1
            continue

        rows.append({
            "portion_id": p.portion_id,
            "food_type": p.food_type,
            "area_cm2": round(area.value, 3),
            "height_cm": round(height.value, 3),
            "volume_ml": p.volume_ml,
            "weight_g": p.weight_g,
            "density_g_per_ml": round(p.density_g_per_ml, 4),
        })
        if progress and k % 20 == 0:
            print(f"  processed {k}/{len(portions)} portions ({len(rows)} ok, {skipped} skipped)")

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if progress:
        print(f"Extracted features for {len(df)} portions ({skipped} skipped); cached to {cache_path}")
    return df


def extract_features_extended(
    ds: Optional[ECUSTFD] = None,
    segmenter: Optional[FoodSegmenter] = None,
    cache_path: Path = FEATURES_EXTENDED_CACHE,
    use_cache: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Like :func:`extract_features` but adds top + side shape descriptors.

    Produces the richer feature set used by ``notebooks/01_training.ipynb``:

    * ``area_cm2``, ``height_cm`` (the base geometry)
    * ``side_area_cm2`` — silhouette area from the side view
    * ``top_aspect``, ``top_circ``, ``top_solidity``, ``top_extent``, ``top_elong``
    * ``side_aspect``, ``side_circ``, ``side_solidity``, ``side_extent``, ``side_elong``
    * ``volume_ml`` (target), ``weight_g``, ``density_g_per_ml`` (for analysis)
    """
    import cv2

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        if progress:
            print(f"Loaded cached extended features for {len(df)} portions from {cache_path}")
        return df

    ds = ds or ECUSTFD()
    segmenter = segmenter or FoodSegmenter()
    portions = ds.portions(single_food_only=True)

    rows, skipped = [], 0
    for k, p in enumerate(portions, 1):
        top = _first_usable(ds, p.top_images)
        side = _first_usable(ds, p.side_images)
        if top is None or side is None:
            skipped += 1; continue

        top_path, top_coin, top_food = top
        side_path, side_coin, side_food = side
        top_img = cv2.imread(str(top_path))
        side_img = cv2.imread(str(side_path))

        calib_top = calibrate_manual(top_coin.diameter_px, COIN_DIAMETER_CM, top_coin.center)
        calib_side = calibrate_manual(side_coin.diameter_px, COIN_DIAMETER_CM, side_coin.center)

        top_inst = segmenter.segment_box(top_img, top_food.box)
        side_inst = segmenter.segment_box(side_img, side_food.box)
        if top_inst is None or side_inst is None:
            skipped += 1; continue

        top_shape = _shape_descriptors(top_inst.mask)
        side_shape = _shape_descriptors(side_inst.mask)
        side_h_px = side_inst.bbox[3]

        rows.append({
            "portion_id": p.portion_id, "food_type": p.food_type,
            "area_cm2": round(calib_top.pixel_area_to_cm2(top_inst.area_px), 3),
            "height_cm": round(calib_side.pixel_length_to_cm(side_h_px), 3),
            "side_area_cm2": round(calib_side.pixel_area_to_cm2(side_inst.area_px), 3),
            "top_aspect": round(top_shape["aspect_ratio"], 4),
            "top_circ": round(top_shape["circularity"], 4),
            "top_solidity": round(top_shape["solidity"], 4),
            "top_extent": round(top_shape["extent"], 4),
            "top_elong": round(top_shape["elongation"], 4),
            "side_aspect": round(side_shape["aspect_ratio"], 4),
            "side_circ": round(side_shape["circularity"], 4),
            "side_solidity": round(side_shape["solidity"], 4),
            "side_extent": round(side_shape["extent"], 4),
            "side_elong": round(side_shape["elongation"], 4),
            "volume_ml": p.volume_ml, "weight_g": p.weight_g,
            "density_g_per_ml": round(p.density_g_per_ml, 4),
        })
        if progress and k % 20 == 0:
            print(f"  processed {k}/{len(portions)} portions ({len(rows)} ok, {skipped} skipped)")

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if progress:
        print(f"Extracted extended features for {len(df)} portions; cached to {cache_path}")
    return df


def extract_features_n5k(
    segmenter: Optional[FoodSegmenter] = None,
    cache_path: Path = N5K_FEATURES_CACHE,
    use_cache: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Extract per-portion features for the Nutrition5k subset.

    Produces one row per dish::

        dish_id  food_class  area_cm2  long_cm  short_cm  total_mass_g
        total_kcal  total_fat_g  total_carb_g  total_protein_g

    Uses the fixed ``cm_per_px`` value calibrated once from apple photos
    (stored in ``artifacts/n5k_scale.json``). Caches to CSV so we don't
    re-run FastSAM on every training pass.
    """
    import json
    import cv2
    from .n5k import Nutrition5k

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        if progress:
            print(f"Loaded cached N5K features for {len(df)} portions from {cache_path}")
        return df

    if not N5K_SCALE_PATH.exists():
        raise FileNotFoundError(
            f"N5K scale missing at {N5K_SCALE_PATH}. Run the calibration script."
        )
    cm_per_px = float(json.load(open(N5K_SCALE_PATH))["cm_per_px"])

    n5k = Nutrition5k()
    if not n5k.is_available():
        raise FileNotFoundError(
            "Nutrition5k not downloaded. Run python data/download_nutrition5k.py."
        )

    segmenter = segmenter or FoodSegmenter()
    rows, skipped = [], 0
    portions = n5k.portions()
    for k, p in enumerate(portions, 1):
        img = cv2.imread(str(p.image_path))
        if img is None:
            skipped += 1; continue
        insts = segmenter.segment(img, interior_mask=None,
                                  min_area_frac=0.01, max_area_frac=0.70)
        if not insts:
            skipped += 1; continue
        # The food is the largest non-background mask.
        inst = max(insts, key=lambda x: x.area_px)
        x, y, w, h = inst.bbox
        long_px, short_px = max(w, h), min(w, h)
        rows.append({
            "dish_id": p.dish_id,
            "food_class": p.food_class,
            "area_cm2": round(inst.area_px * (cm_per_px ** 2), 3),
            "long_cm": round(long_px * cm_per_px, 3),
            "short_cm": round(short_px * cm_per_px, 3),
            "total_mass_g": p.total_mass_g,
            "total_kcal": p.total_kcal,
            "total_fat_g": p.total_fat_g,
            "total_carb_g": p.total_carb_g,
            "total_protein_g": p.total_protein_g,
        })
        if progress and k % 25 == 0:
            print(f"  processed {k}/{len(portions)} N5K portions ({len(rows)} ok, {skipped} skipped)")

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if progress:
        print(f"Extracted N5K features for {len(df)} portions; cached to {cache_path}")
    return df


def cross_validate(df: pd.DataFrame, model_kind: str = "huber", n_splits: int = 5,
                   random_state: int = 0) -> dict:
    """K-fold CV of the trained estimator vs. the physics baseline (volume MAPE etc.)."""
    from sklearn.model_selection import KFold

    areas = df["area_cm2"].to_numpy()
    heights = df["height_cm"].to_numpy()
    vols = df["volume_ml"].to_numpy()

    oof_model = np.zeros(len(df))
    oof_phys = 0.5 * areas * heights  # physics baseline needs no fitting

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in kf.split(areas):
        est = VolumeEstimator().fit(areas[tr], heights[tr], vols[tr], model_kind=model_kind)
        oof_model[te] = est.predict_many(areas[te], heights[te])

    return {
        "trained": evaluate(vols, oof_model),
        "physics_baseline": evaluate(vols, oof_phys),
        "oof_model": oof_model,
        "oof_physics": oof_phys,
    }


def mass_error(df: pd.DataFrame, volume_pred_ml: np.ndarray, use_db_density: bool = True) -> dict:
    """Convert predicted volume to mass and compare to ground-truth weight.

    ``use_db_density=True`` uses the bundled nutrition table's per-class density (the
    realistic end-to-end path); ``False`` uses each portion's true density (isolates
    the volume error from density error).
    """
    if use_db_density:
        densities = df["food_type"].map(lambda t: nutrition.lookup(t).density_g_per_ml).to_numpy()
    else:
        densities = df["density_g_per_ml"].to_numpy()
    mass_pred = volume_pred_ml * densities
    return evaluate(df["weight_g"].to_numpy(), mass_pred)


def fit_final(
    df: pd.DataFrame,
    model_kind: str = "huber",
    save: bool = True,
    path: Path = config.VOLUME_BASELINE_MODEL_PATH,
) -> VolumeEstimator:
    """Fit the legacy benchmark estimator and optionally persist it as baseline."""
    est = VolumeEstimator().fit(
        df["area_cm2"].to_numpy(), df["height_cm"].to_numpy(),
        df["volume_ml"].to_numpy(), model_kind=model_kind,
    )
    cv = cross_validate(df, model_kind=model_kind)
    est.metrics = {f"cv_{k}": v for k, v in cv["trained"].items()}
    if save:
        path = est.save(path)
        print(f"Saved trained volume model to {path}")
    return est
