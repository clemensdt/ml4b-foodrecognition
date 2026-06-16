"""Portion quantity logic: geometry -> volume -> grams.

This module is deliberately small and free of image/model loading. The live
pipeline, notebooks and tests can all use the same rules:

* with a metric side-view height, predict **volume** and convert it to mass via
  the nutrition-table density;
* without a side view, use the curated per-class ``mass_per_cm2`` prior as a
  clearly labelled fallback;
* clamp to broad serving-size ranges when the estimate is physically implausible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nutrition import NutritionInfo
from .volume import VolumeEstimator


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


def _clamp_mass(
    mass_g: float,
    info: NutritionInfo,
) -> tuple[float, bool, Optional[str]]:
    lo = info.mass_min_g if info.mass_min_g is not None else 0.0
    hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
    if mass_g < lo:
        return float(lo), True, "min"
    if mass_g > hi:
        return float(hi), True, "max"
    return float(mass_g), False, None


def estimate_quantity(
    info: NutritionInfo,
    area_cm2: float,
    volume_model: VolumeEstimator,
    height_cm: Optional[float] = None,
    *,
    height_source: str = "none",
    clamp: bool = True,
) -> QuantityEstimate:
    """Estimate mass and volume for one item.

    ``height_cm`` is the switch: when present, the trained volume model is used.
    Without it, the result is an area-based mass prior and is labelled as such.
    """
    has_height = (
        height_cm is not None
        and np.isfinite(height_cm)
        and float(height_cm) > 0
    )

    if has_height:
        used_height = float(height_cm)
        raw_volume = volume_model.predict_volume(area_cm2, used_height)
        raw_mass = info.mass_from_volume(raw_volume)
        source = f"volume_model:{height_source}"
    else:
        used_height = float("nan")
        raw_mass = area_mass_prior(info, area_cm2)
        raw_volume = raw_mass / info.density_g_per_ml if info.density_g_per_ml else float("nan")
        source = "area_mass_prior"

    mass, clamped, bound = _clamp_mass(raw_mass, info) if clamp else (raw_mass, False, None)
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
