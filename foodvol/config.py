"""Central configuration: filesystem paths and compute-device selection.

Every other module imports its paths and the torch device from here, so the
project has a single source of truth for where data, weights and trained
artifacts live.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

# Ensure TLS verification works for model/dataset downloads. The python.org macOS
# builds do not install system certificates by default, which breaks urllib-based
# downloads (e.g. ultralytics/torch.hub); point them at certifi's bundle instead.
try:  # pragma: no cover - environment hardening
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

# --- Filesystem layout ---------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
ECUSTFD_DIR = DATA_DIR / "ECUSTFD"          # populated by data/download_ecustfd.py
MODELS_DIR = PROJECT_ROOT / "models"         # downloaded model weights (gitignored)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"   # trained regressors, evaluation plots

# Bundled, version-controlled nutrition table (density + energy/macros per class).
NUTRITION_DB_PATH = PACKAGE_DIR / "data" / "nutrition_db.csv"

for _d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Trained volume regressor written by the training notebook.
VOLUME_MODEL_PATH = ARTIFACTS_DIR / "volume_model_trained.joblib"
VOLUME_BASELINE_MODEL_PATH = ARTIFACTS_DIR / "volume_regressor.joblib"


# --- Compute device ------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """Return the best available torch device: 'cuda', then 'mps', else 'cpu'.

    CUDA is checked first so the identical code path runs unchanged on a GPU box;
    on this project's target (Apple Silicon) it resolves to 'mps'.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Reference objects ---------------------------------------------------------
# Real-world diameter (centimetres) of circular objects used for metric scale.
REFERENCE_DIAMETERS_CM = {
    "plate_small": 20.0,
    "plate_dinner": 26.0,
    "plate_large": 30.0,
    "coin_ecustfd": 2.5,   # 1-Yuan coin used as the scale reference in ECUSTFD
}

# Default model identifiers (overridable via the respective module APIs).
FASTSAM_WEIGHTS = "FastSAM-s.pt"                                  # ultralytics auto-downloads
FOOD_CLASSIFIER_HF = "nateraw/food"                              # ViT fine-tuned on Food-101 (fallback)
CLIP_MODEL_HF = "openai/clip-vit-base-patch32"                   # zero-shot recognition + non-food gate
DEPTH_MODEL_HF = "depth-anything/Depth-Anything-V2-Small-hf"     # relative monocular depth
