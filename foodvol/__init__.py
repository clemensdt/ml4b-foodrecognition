"""foodvol — food recognition and portion (volume / mass / calorie) estimation.

The package is organised as a small, explicit pipeline. Each stage lives in its
own module and can be used in isolation:

    calibration   metric scale (cm/pixel) from a circular reference (plate/coin)
    segmentation  per-item food masks
    classification food class + confidence (drives the nutrition lookup)
    depth          monocular relative depth (optional height cue)
    volume         footprint area + height -> physical volume (the trained part)
    portion        volume/mass decision rules used by app and notebooks
    nutrition      density + energy/macros table -> mass and calories
    training       reusable ECUSTFD/Nutrition5k training workflow
    pipeline       wires the stages together end to end

Heavy models (segmentation, classification, depth) are loaded lazily on first
use and degrade gracefully when their weights cannot be downloaded, so the
package stays importable on a machine without network access.
"""

__version__ = "0.1.0"

__all__ = [
    "calibration",
    "segmentation",
    "classification",
    "depth",
    "volume",
    "portion",
    "nutrition",
    "training",
    "pipeline",
    "config",
]
