"""Stage F — nutrition lookup: class -> density, energy/macros and portion priors.

Reads the bundled :data:`~foodvol.config.NUTRITION_DB_PATH` table. The ``density``
column converts an estimated **volume** (mL) into **mass** (g). The
``mass_per_cm2`` column is the explicit fallback when no side-view height exists.
The per-100 g energy and macro columns then convert mass into **calories** and
**macronutrients**.

Lookups are normalised (lower-case, spaces/hyphens -> underscores) so they tolerate
small label differences, and fall back to a documented generic value for unknown
classes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from . import config

# Generic cooked-food fallback used when a class is not in the table. Portion priors
# are intentionally None for the default; unknown classes get a generic area-mass
# fallback unless a side-view height lets the volume model run.
DEFAULT_NUTRITION = ("__default__", 0.90, 200.0, 8.0, 25.0, 8.0)


def _normalise(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class NutritionInfo:
    """Per-class nutrition + portion reference.

    All fields besides the nutrition basics are *priors* describing a realistic
    whole serving (a whole apple, one slice of pizza, half an avocado). The
    pipeline uses them to convert a measured pixel footprint into mass — with
    ``mass_min_g`` / ``mass_max_g`` acting as plausibility bounds.

    Attributes
    ----------
    typical_long_cm : the long side of the top-view bounding box for a typical
        serving. Used to self-calibrate cm per pixel from the recognised class.
    typical_mass_g : typical mass of one serving (a whole banana ~ 120 g, a slice
        of pizza ~ 130 g, half an avocado ~ 100 g, …).
    mass_min_g / mass_max_g : the plausible whole-serving mass range; the pipeline
        clamps its estimate to this and warns when clamping happens.
    mass_per_cm2 : derived from ``typical_mass_g`` divided by the typical area;
        used as the area→mass predictor (skips the volume detour entirely).
    """

    food_class: str
    density_g_per_ml: float
    kcal_per_100g: float
    protein_g_per_100g: float
    carbs_g_per_100g: float
    fat_g_per_100g: float
    typical_long_cm: Optional[float] = None
    typical_mass_g: Optional[float] = None
    mass_min_g: Optional[float] = None
    mass_max_g: Optional[float] = None
    mass_per_cm2: Optional[float] = None
    is_default: bool = False

    def mass_from_volume(self, volume_ml: float) -> float:
        """Convert a volume in millilitres to mass in grams."""
        return float(volume_ml) * self.density_g_per_ml

    def for_mass(self, mass_g: float) -> "NutritionEstimate":
        """Scale energy and macros to a given mass in grams."""
        f = mass_g / 100.0
        return NutritionEstimate(
            food_class=self.food_class,
            mass_g=mass_g,
            kcal=self.kcal_per_100g * f,
            protein_g=self.protein_g_per_100g * f,
            carbs_g=self.carbs_g_per_100g * f,
            fat_g=self.fat_g_per_100g * f,
            is_default=self.is_default,
        )

    def for_volume(self, volume_ml: float) -> "NutritionEstimate":
        """Convenience: volume -> mass -> energy/macros."""
        return self.for_mass(self.mass_from_volume(volume_ml))


@dataclass(frozen=True)
class NutritionEstimate:
    """Concrete nutrition values for a specific portion."""

    food_class: str
    mass_g: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    is_default: bool = False


def _optional_float(row: dict, key: str) -> Optional[float]:
    """Read an optional numeric column from a CSV row (blank cell -> None)."""
    val = row.get(key, "")
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _table() -> dict[str, NutritionInfo]:
    table: dict[str, NutritionInfo] = {}
    with open(config.NUTRITION_DB_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            name = _normalise(row["class"])
            table[name] = NutritionInfo(
                food_class=name,
                density_g_per_ml=float(row["density_g_per_ml"]),
                kcal_per_100g=float(row["kcal_per_100g"]),
                protein_g_per_100g=float(row["protein_g_per_100g"]),
                carbs_g_per_100g=float(row["carbs_g_per_100g"]),
                fat_g_per_100g=float(row["fat_g_per_100g"]),
                typical_long_cm=_optional_float(row, "typical_long_cm"),
                typical_mass_g=_optional_float(row, "typical_mass_g"),
                mass_min_g=_optional_float(row, "mass_min_g"),
                mass_max_g=_optional_float(row, "mass_max_g"),
                mass_per_cm2=_optional_float(row, "mass_per_cm2"),
            )
    return table


def lookup(food_class: str) -> NutritionInfo:
    """Return the :class:`NutritionInfo` for a class, or a generic default."""
    info = _table().get(_normalise(food_class))
    if info is not None:
        return info
    _, density, kcal, protein, carbs, fat = DEFAULT_NUTRITION
    return NutritionInfo(food_class=food_class, density_g_per_ml=density,
                         kcal_per_100g=kcal, protein_g_per_100g=protein,
                         carbs_g_per_100g=carbs, fat_g_per_100g=fat, is_default=True)


def known_classes() -> list[str]:
    return sorted(_table().keys())
