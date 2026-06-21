"""Optional chessboard detection for a real metric scale.

If a flat printed chessboard pattern is visible in the photo, OpenCV's
``findChessboardCorners`` recovers the spacing between corners in pixels. With a
known real-world square size we derive an exact cm/px — much more reliable than
deriving the scale from the recognised food's class prior.

This is **opportunistic**: the pipeline tries to find a board, and if it does,
uses the chessboard scale. If no board is found the pipeline silently falls back
to the class-based self-calibration. The user doesn't have to do anything.

A 7×7-internal-corner board (8×8 squares) is the most common; we try a few
candidate sizes in case the user prints a slightly different one. Square size
defaults to the app's standard 2.0 cm calibration board. The normal UI needs no
manual setting; API users may still override it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# (inner-corners-x, inner-corners-y) — try common board configurations.
_CANDIDATE_PATTERNS = [
    (7, 7), (6, 7), (7, 6), (5, 7), (7, 5),
    (6, 6), (5, 6), (6, 5), (5, 5), (4, 5), (5, 4), (4, 4),
]


@dataclass
class ChessboardScale:
    """Result of a successful chessboard detection."""

    cm_per_px: float
    pattern: tuple[int, int]                # inner-corners (cols, rows)
    square_cm: float
    mean_spacing_px: float
    confidence: float                       # 0..1, based on geometry consistency


def detect_scale(image_bgr: np.ndarray, square_cm: float = 2.0) -> Optional[ChessboardScale]:
    """Try to detect a chessboard and return the resulting scale.

    Parameters
    ----------
    image_bgr : input photo (BGR, as from ``cv2.imread`` or a Streamlit upload).
    square_cm : real-world edge length of one chessboard square. Default 2 cm.

    Returns
    -------
    ChessboardScale or None. ``None`` means no board was found and the pipeline
    should fall back to its other scale source.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Downscale very large images for speed; corner positions get rescaled back.
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 1400:
        scale = 1400 / max(h, w)
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        gray_small = gray

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)

    for cols, rows in _CANDIDATE_PATTERNS:
        ok, corners = cv2.findChessboardCorners(gray_small, (cols, rows), flags=flags)
        if not ok:
            continue
        # Refine corner positions to sub-pixel.
        corners = cv2.cornerSubPix(
            gray_small, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        if scale != 1.0:
            corners = corners / scale       # back to original-image pixel coordinates

        spacings = _corner_spacings(corners, cols, rows)
        if spacings is None:
            continue
        mean_px = float(np.median(spacings))
        # Geometric consistency: low std/mean ratio means the perspective is mild
        # enough for a single scalar cm/px to be a decent approximation.
        std_ratio = float(np.std(spacings) / max(mean_px, 1e-6))
        confidence = float(np.clip(1.0 - 2.0 * std_ratio, 0.0, 1.0))
        return ChessboardScale(
            cm_per_px=square_cm / mean_px,
            pattern=(cols, rows),
            square_cm=square_cm,
            mean_spacing_px=mean_px,
            confidence=confidence,
        )

    return None


def _corner_spacings(corners: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    """Pixel distance between neighbouring corners along rows and columns."""
    pts = corners.reshape(rows, cols, 2)
    horiz = np.linalg.norm(np.diff(pts, axis=1), axis=-1).flatten()
    vert = np.linalg.norm(np.diff(pts, axis=0), axis=-1).flatten()
    spacings = np.concatenate([horiz, vert])
    if spacings.size == 0 or not np.all(np.isfinite(spacings)):
        return None
    return spacings
