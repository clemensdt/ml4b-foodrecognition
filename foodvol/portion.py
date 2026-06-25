"""Portion quantity logic: geometry -> volume -> grams.

This module is deliberately small and free of image/model loading. The live
pipeline, notebooks and tests can all use the same rules:

* with a reconstructed two-view volume, use it directly and convert it to mass;
* with only a metric side-view height, use the trained volume-model fallback;
* without a side view, use the curated per-class ``mass_per_cm2`` prior as a
  clearly labelled fallback;
* keep unusual-but-possible serving sizes, and only cap extreme outliers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nutrition import NutritionInfo
from .volume import VolumeEstimator

HARD_PLAUSIBILITY_FACTOR = 4.0


@dataclass(frozen=True)
class QuantityEstimate:
    """Mass/volume estimate for one recognised food instance."""

    mass_g: float
    volume_ml: float
    height_cm: float
    source: str
    raw_mass_g: float
    raw_volume_ml: float
    clamped: bool = False
    clamp_bound: Optional[str] = None


def geometric_height_prior(area_cm2: float) -> float:
    """Fallback height in cm when no class prior or side view is available."""
    radius_cm = float(np.sqrt(max(area_cm2, 1e-3) / np.pi))
    return float(np.clip(0.8 * radius_cm, 0.5, 6.0))


def area_mass_prior(info: NutritionInfo, area_cm2: float) -> float:
    """Estimate grams directly from footprint area and class priors."""
    if info.mass_per_cm2 is not None:
        return float(info.mass_per_cm2 * area_cm2)
    if info.typical_mass_g is not None:
        typical_area = (info.typical_long_cm or 10.0) ** 2 * 0.59
        return float(info.typical_mass_g * (area_cm2 / typical_area))
    return float(info.density_g_per_ml * 100.0 * area_cm2 / 50.0)


def _bound_mass(
    mass_g: float,
    info: NutritionInfo,
    *,
    hard_factor: float = HARD_PLAUSIBILITY_FACTOR,
) -> tuple[float, bool, Optional[str]]:
    lo = info.mass_min_g if info.mass_min_g is not None else 0.0
    hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
    factor = max(1.0, float(hard_factor))
    hard_lo = lo / factor if lo > 0 else lo
    hard_hi = hi * factor if np.isfinite(hi) else hi
    if mass_g < hard_lo:
        return float(hard_lo), True, "min"
    if mass_g > hard_hi:
        return float(hard_hi), True, "max"
    return float(mass_g), False, None


def estimate_quantity(
    info: NutritionInfo,
    area_cm2: float,
    volume_model: VolumeEstimator,
    height_cm: Optional[float] = None,
    *,
    height_source: str = "none",
    measured_volume_ml: Optional[float] = None,
    volume_source: str = "two_view_silhouette",
    clamp: bool = True,
    hard_factor: float = HARD_PLAUSIBILITY_FACTOR,
) -> QuantityEstimate:
    """Estimate mass and volume for one item.

    A valid ``measured_volume_ml`` takes precedence over the trained model. If no
    reconstructed volume exists, ``height_cm`` selects the model fallback. Without
    either, the result is an area-based mass prior and is labelled as such.
    """
    has_height = (
        height_cm is not None
        and np.isfinite(height_cm)
        and float(height_cm) > 0
    )

    has_measured_volume = (
        measured_volume_ml is not None
        and np.isfinite(measured_volume_ml)
        and float(measured_volume_ml) > 0
    )

    if has_measured_volume:
        used_height = float(height_cm) if has_height else float("nan")
        raw_volume = float(measured_volume_ml)
        raw_mass = info.mass_from_volume(raw_volume)
        source = f"{volume_source}:{height_source}"
    elif has_height:
        used_height = float(height_cm)
        raw_volume = volume_model.predict_volume(area_cm2, used_height)
        raw_mass = info.mass_from_volume(raw_volume)
        source = f"volume_model:{height_source}"
    else:
        used_height = float("nan")
        raw_mass = area_mass_prior(info, area_cm2)
        raw_volume = raw_mass / info.density_g_per_ml if info.density_g_per_ml else float("nan")
        source = "area_mass_prior"

    mass, clamped, bound = (
        _bound_mass(raw_mass, info, hard_factor=hard_factor)
        if clamp else (raw_mass, False, None)
    )
    volume = mass / info.density_g_per_ml if info.density_g_per_ml else raw_volume
    if clamped:
        source = f"{source}:clamped_{bound}"

    return QuantityEstimate(
        mass_g=float(mass),
        volume_ml=float(volume),
        height_cm=used_height,
        source=source,
        raw_mass_g=float(raw_mass),
        raw_volume_ml=float(raw_volume),
        clamped=clamped,
        clamp_bound=bound,
    )
