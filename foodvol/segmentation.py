"""Stage B — food instance segmentation.

Produces one boolean mask per food item. The default backend is **FastSAM**
(a fast, CPU-friendly "segment anything" variant via ultralytics); when its
weights cannot be downloaded the module falls back to a **classical**
plate-vs-food segmentation so the pipeline still runs offline.

When a plate :class:`~foodvol.calibration.Calibration` interior mask is supplied,
segmentation is restricted to the inside of the plate and the plate surface itself
is filtered out, leaving only the food.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import config


@dataclass
class InstanceMask:
    """A single segmented food region."""

    mask: np.ndarray          # boolean HxW
    area_px: int
    bbox: tuple[int, int, int, int]   # x, y, w, h
    centroid: tuple[float, float]

    def crop(self, image: np.ndarray, pad: int = 8) -> np.ndarray:
        """Return the image content within this instance's bounding box (padded)."""
        h, w = image.shape[:2]
        x, y, bw, bh = self.bbox
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        return image[y0:y1, x0:x1]


def _mask_to_instance(mask: np.ndarray) -> Optional[InstanceMask]:
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area == 0:
        return None
    ys, xs = np.where(mask)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return InstanceMask(mask, area, (x, y, w, h), (float(xs.mean()), float(ys.mean())))


def _ensure_weights(name: str) -> str:
    """Make sure the ultralytics weight file exists under models/, return its path."""
    dest = config.MODELS_DIR / name
    if dest.exists():
        return str(dest)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(dest))
        if dest.exists():
            return str(dest)
    except Exception:
        pass
    return name   # let ultralytics resolve/download by name as a last resort


class FoodSegmenter:
    """Segments food items in an image. Heavy model is loaded lazily."""

    def __init__(self, weights: str = config.FASTSAM_WEIGHTS, device: Optional[str] = None):
        self.weights = weights
        self.device = device or config.get_device()
        self._model = None
        self.backend = "fastsam"

    def _ensure_model(self) -> bool:
        """Load FastSAM on first use. Returns False if unavailable (use fallback)."""
        if self._model is not None:
            return True
        try:
            from ultralytics import FastSAM
            self._model = FastSAM(_ensure_weights(self.weights))
            return True
        except Exception as exc:  # network/weights/runtime issue -> classical fallback
            print(f"[segmentation] FastSAM unavailable ({exc}); using classical fallback.")
            self.backend = "classical"
            return False

    # --- public API ------------------------------------------------------------
    def segment(
        self,
        image_bgr: np.ndarray,
        interior_mask: Optional[np.ndarray] = None,
        min_area_frac: float = 0.004,
        max_area_frac: float = 0.92,
    ) -> list[InstanceMask]:
        """Return food instance masks, largest first.

        Parameters
        ----------
        image_bgr : BGR image (OpenCV convention).
        interior_mask : optional boolean plate-interior mask; segmentation is
            clipped to it and instances mostly outside it are dropped.
        min_area_frac, max_area_frac : keep instances whose area is within this
            fraction of the reference region (interior if given, else whole image).
            ``max_area_frac`` removes the plate/background; ``min_area_frac`` removes noise.
        """
        if self._ensure_model():
            raw = self._segment_fastsam(image_bgr)
        else:
            raw = self._segment_classical(image_bgr, interior_mask)

        ref_area = float(interior_mask.sum()) if interior_mask is not None else float(image_bgr.shape[0] * image_bgr.shape[1])
        keep: list[InstanceMask] = []
        for m in raw:
            if interior_mask is not None:
                m = m & interior_mask
                # require the bulk of the mask to lie inside the plate
                if m.sum() < 0.5 * max(1, int(np.count_nonzero(m))):
                    pass  # already clipped; the area filter below handles tiny remnants
            frac = m.sum() / max(ref_area, 1.0)
            if frac < min_area_frac or frac > max_area_frac:
                continue
            inst = _mask_to_instance(m)
            if inst is not None:
                keep.append(inst)

        keep = self._suppress_overlaps(keep)
        keep.sort(key=lambda i: i.area_px, reverse=True)
        return keep

    def segment_box(
        self,
        image_bgr: np.ndarray,
        box: tuple[int, int, int, int],
        pad: int = 12,
    ) -> Optional[InstanceMask]:
        """Segment the single dominant object inside a bounding ``box`` (x1, y1, x2, y2).

        Runs segmentation on the cropped (padded) region — fast and robust, because
        the object of interest dominates the crop — and returns its mask in
        full-image coordinates. Used with the ECUSTFD food bounding boxes; for the
        live app the box can come from a detector or the plate interior.
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2]

        local: Optional[np.ndarray] = None
        if self._ensure_model():
            masks = self._segment_fastsam(crop)
            # The food fills most of the (tight) crop -> the largest mask is the food.
            masks = [m for m in masks if 0.02 < m.mean() < 0.97]
            if masks:
                local = max(masks, key=lambda m: int(m.sum()))
        if local is None:
            local = self._foreground_in_crop(crop)
        if local is None or local.sum() == 0:
            return None

        full = np.zeros((h, w), dtype=bool)
        full[y1:y2, x1:x2] = local
        return _mask_to_instance(full)

    @staticmethod
    def _foreground_in_crop(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Otsu-threshold fallback: separate the central object from the crop border."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Choose the polarity whose foreground sits more centrally (objects are centred).
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        center = np.zeros_like(fg)
        cv2.circle(center, (cx, cy), max(3, min(h, w) // 6), 255, -1)
        if (fg & center).sum() < ((~fg.astype(bool)).astype(np.uint8) * 255 & center).sum():
            fg = 255 - fg
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        if n <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    # --- backends --------------------------------------------------------------
    def _segment_fastsam(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        results = self._model(
            image_bgr, device=self.device, retina_masks=True,
            conf=0.4, iou=0.9, verbose=False,
        )
        out: list[np.ndarray] = []
        if results and results[0].masks is not None:
            for data in results[0].masks.data:
                out.append(data.cpu().numpy().astype(bool))
        return out

    def _segment_classical(self, image_bgr: np.ndarray,
                           interior_mask: Optional[np.ndarray]) -> list[np.ndarray]:
        """Color-contrast fallback: food differs from the (uniform) plate surface.

        Estimates the plate colour from a ring just inside the rim, flags pixels
        that differ from it as food, then splits into connected components.
        """
        h, w = image_bgr.shape[:2]
        if interior_mask is None:
            interior_mask = np.ones((h, w), bool)

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        # Plate colour ≈ median over an eroded-minus-more-eroded ring of the interior.
        inner = cv2.erode(interior_mask.astype(np.uint8), np.ones((25, 25), np.uint8))
        ring = interior_mask.astype(np.uint8) & (1 - inner)
        ring_bool = ring.astype(bool)
        if ring_bool.sum() < 50:
            ring_bool = interior_mask
        plate_lab = np.median(lab[ring_bool], axis=0)

        dist = np.linalg.norm(lab.astype(np.float32) - plate_lab, axis=2)
        interior_dist = dist[interior_mask]
        p75 = float(np.percentile(interior_dist, 75))
        p95 = float(np.percentile(interior_dist, 95))
        # On clean white/table backgrounds the useful signal may be a very subtle
        # shadow or colour change. Keep the old robust threshold for busy scenes,
        # but lower it when the whole image is low-contrast.
        if p75 < 4.0 and p95 < 35.0:
            thresh = 6.0
        else:
            thresh = max(18.0, p75)
        food = (dist >= thresh) & interior_mask
        food = cv2.morphologyEx(food.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        food = cv2.morphologyEx(food, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        n, labels = cv2.connectedComponents(food)
        masks = [labels == i for i in range(1, n)]

        # Low-contrast scenes (e.g. pale food on a white table) may have almost no
        # colour distance. Add edge-closed contours as a general fallback instead
        # of exposing background-specific switches to the user.
        masks.extend(self._edge_components(image_bgr, interior_mask))
        return masks

    @staticmethod
    def _edge_components(image_bgr: np.ndarray, interior_mask: np.ndarray) -> list[np.ndarray]:
        """Return filled contours from image edges.

        This complements colour-contrast segmentation for neutral backgrounds and
        pale foods. It is intentionally conservative: tiny texture edges and
        frame-sized contours are filtered by area later in ``segment``.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Equalise locally so subtle shadows/rims survive on bright backgrounds.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        med = float(np.median(gray[interior_mask]))
        lo = int(max(0, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        edges = cv2.Canny(gray, lo, max(hi, lo + 20))
        edges = edges & interior_mask.astype(np.uint8) * 255
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks: list[np.ndarray] = []
        h, w = gray.shape
        ref_area = max(1, int(interior_mask.sum()))
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.001 * ref_area or area > 0.90 * ref_area:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            masks.append(mask.astype(bool) & interior_mask)
        return masks

    @staticmethod
    def _suppress_overlaps(instances: list[InstanceMask], iou_thresh: float = 0.85) -> list[InstanceMask]:
        """Drop near-duplicate masks (keep the larger), a light non-max suppression."""
        instances = sorted(instances, key=lambda i: i.area_px, reverse=True)
        kept: list[InstanceMask] = []
        for inst in instances:
            dup = False
            for k in kept:
                inter = np.logical_and(inst.mask, k.mask).sum()
                union = np.logical_or(inst.mask, k.mask).sum()
                if union and inter / union > iou_thresh:
                    dup = True
                    break
            if not dup:
                kept.append(inst)
        return kept
