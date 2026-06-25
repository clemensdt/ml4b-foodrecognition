"""End-to-end orchestration: images -> per-item volume, mass and calories.

Wires the stages together:

    top view  --calibrate--> scale --segment--> items --classify--> class
                                          |                    |
                                          +--> footprint area  +--> density/energy
    side view --segment--> side silhouette --align width with top silhouette
                                          |
        matched top/side profiles --slice integration--> volume --x density--> mass --> calories

With a side view, the dominant food's two masks are integrated as elliptical
cross-sections. The trained :class:`foodvol.volume.VolumeEstimator` remains a
fallback if profile alignment fails. Without a side view, the pipeline uses
explicitly labelled area-to-mass priors from the nutrition table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import config, nutrition
from .nutrition import NutritionEstimate
from .portion import HARD_PLAUSIBILITY_FACTOR, area_mass_prior, estimate_quantity
from .recognition import FoodRecognizer, Recognition, nonfood_kind
from .segmentation import FoodSegmenter, InstanceMask, _mask_to_instance
from .volume import TwoViewVolume, VolumeEstimator, estimate_two_view_volume

ImageInput = Union[str, Path, np.ndarray]

MAX_SEGMENTS = 24        # hard cap: keep interactive inference within a bounded cost
MAX_HEIGHT_CM = 12.0     # clamp measured side-view height to a plausible range
CONTAINMENT_AREA_RATIO = 2.2
CONTAINMENT_SCORE_MARGIN = 0.18

SEGMENTATION_PRESETS = {
    "conservative": {"min_area_frac": 0.008, "max_area_frac": 0.80, "max_segments": 8},
    "balanced": {"min_area_frac": 0.004, "max_area_frac": 0.92, "max_segments": 12},
    "sensitive": {"min_area_frac": 0.0015, "max_area_frac": 0.97, "max_segments": MAX_SEGMENTS},
}


@dataclass
class _SideHeightProfile:
    """Dominant side-view height before item-specific scaling."""

    height_px: float
    cm_per_px: Optional[float] = None
    scale_source: str = "item_scale"
    mask: Optional[InstanceMask] = None
    width_px: float = 0.0


@dataclass
class RegionTrace:
    """One segmentation candidate and the recognition decision made for it."""

    index: int
    mask: InstanceMask
    label: str
    score: float
    is_food: bool
    passed_filter: bool
    kept: bool = False
    reason: str = ""
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    final_label: str = ""
    measurement_mask: Optional[InstanceMask] = None
    removed_px: int = 0


@dataclass(frozen=True)
class ScaleEvidence:
    """One independent physical-scale cue considered for an item."""

    source: str
    cm_per_px: float
    confidence: float
    used: bool = False


@dataclass
class ItemEstimate:
    """Per-food-item result."""

    food_class: str
    confidence: float
    area_cm2: float
    height_cm: float
    volume_ml: float
    nutrition: NutritionEstimate
    mask: InstanceMask
    scale_source: str = "class_prior"        # 'class_prior' | 'chessboard' | 'reranked'
    height_source: str = "none"              # 'side_chessboard' | 'side_item_scale' | 'none'
    mass_source: str = "area_mass_prior"     # 'two_view_silhouette:*' | 'volume_model:*' | fallback
    cm_per_px: float = 0.0
    typical_mass_g: float = 0.0
    mass_range_g: tuple[float, float] = (0.0, 0.0)
    raw_mass_g: float = 0.0
    raw_volume_ml: float = 0.0
    quantity_confidence: float = 0.0         # 0..1, how trustworthy *the mass* is
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # CLIP top-k
    two_view: Optional[TwoViewVolume] = None
    scale_evidence: list[ScaleEvidence] = field(default_factory=list)

    @property
    def mass_g(self) -> float:
        return self.nutrition.mass_g


@dataclass
class PlateEstimate:
    """Full result for one frame (one or more food items)."""

    items: list[ItemEstimate] = field(default_factory=list)
    height_source: str = "none"           # kept for backward compat with old UI code
    chessboard_scale_cm_per_px: float = 0.0  # >0 if a chessboard was used as scale
    notes: list[str] = field(default_factory=list)
    trace_regions: list[RegionTrace] = field(default_factory=list)
    side_trace_regions: list[RegionTrace] = field(default_factory=list)
    side_mask: Optional[InstanceMask] = None
    side_height_px: float = 0.0

    @property
    def total_mass_g(self) -> float:
        return sum(i.mass_g for i in self.items)

    @property
    def total_kcal(self) -> float:
        return sum(i.nutrition.kcal for i in self.items)

    @property
    def total_protein_g(self) -> float:
        return sum(i.nutrition.protein_g for i in self.items)

    @property
    def total_carbs_g(self) -> float:
        return sum(i.nutrition.carbs_g for i in self.items)

    @property
    def total_fat_g(self) -> float:
        return sum(i.nutrition.fat_g for i in self.items)

    def summary(self) -> str:
        lines = [f"{'Food':<22}{'mass(g)':>10}{'kcal':>9}{'vol(mL)':>10}"]
        lines.append("-" * 51)
        for i in sorted(self.items, key=lambda x: x.nutrition.kcal, reverse=True):
            lines.append(f"{i.food_class:<22}{i.mass_g:>10.0f}{i.nutrition.kcal:>9.0f}{i.volume_ml:>10.0f}")
        lines.append("-" * 51)
        lines.append(f"{'TOTAL':<22}{self.total_mass_g:>10.0f}{self.total_kcal:>9.0f}")
        return "\n".join(lines)


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


class FoodVolumePipeline:
    """High-level API. Heavy models are shared and loaded lazily by their stages."""

    def __init__(
        self,
        volume_model_path: Path = config.VOLUME_MODEL_PATH,
        device: Optional[str] = None,
        chessboard_square_cm: float = config.CHESSBOARD_SQUARE_CM,
    ):
        self.segmenter = FoodSegmenter(device=device)
        self.recognizer = FoodRecognizer(device=device)
        self.volume = VolumeEstimator.load(volume_model_path)
        self.device = device
        self.chessboard_square_cm = float(chessboard_square_cm)

    # --- main entry point ------------------------------------------------------
    def estimate(
        self,
        top_image: ImageInput,
        side_image: Optional[ImageInput] = None,
        min_confidence: float = 0.0,
        segmentation_preset: str = "balanced",
        scale_mode: str = "auto",
    ) -> PlateEstimate:
        """Estimate per-item mass and calories.

        The pipeline runs in this order:
          1. Segment the image into region candidates (FastSAM).
          2. Recognise each candidate (CLIP) and drop non-food regions.
          3. **Self-calibrate per item**: convert pixels to centimetres using the
             recognised class's ``typical_long_cm`` from the nutrition table.
          4. If a side view is provided, align both silhouettes and integrate
             elliptical cross-sections. Otherwise use the area-to-mass fallback.

        ``segmentation_preset`` is one of ``conservative``, ``balanced`` or
        ``sensitive``. ``scale_mode`` is ``auto``, ``class_prior`` or
        ``metric_reference``.
        """
        top = _load_bgr(top_image)
        side = _load_bgr(side_image) if side_image is not None else None
        result = PlateEstimate()
        result.height_source = "area_mass_prior"
        seg = self._segmentation_config(segmentation_preset)
        scale_mode = self._normalise_scale_mode(scale_mode)

        # 0. Opportunistic: detect a chessboard. If found, it gives a real cm/px
        #    independent of the recognised class — far more reliable than
        #    deriving the scale from class priors.
        from .chessboard import detect_scale as _detect_chessboard_scale
        cb = (_detect_chessboard_scale(top, square_cm=self.chessboard_square_cm)
              if scale_mode != "class_prior" else None)
        plate_candidate = (
            self._plate_scale_candidate(top) if scale_mode != "class_prior" else None
        )
        if cb is not None:
            result.chessboard_scale_cm_per_px = cb.cm_per_px
            result.notes.append(
                f"Chessboard detected ({cb.pattern[0]}×{cb.pattern[1]} corners, "
                f"square = {cb.square_cm:.1f} cm) — using it as the metric scale."
            )
        elif scale_mode == "metric_reference":
            result.notes.append(
                "No metric reference was detected; falling back to food-size priors."
            )

        side_cb = None
        side_plate_candidate = None
        if side is not None:
            side_cb = (_detect_chessboard_scale(side, square_cm=self.chessboard_square_cm)
                       if scale_mode != "class_prior" else None)
            side_plate_candidate = (
                self._plate_scale_candidate(side) if scale_mode != "class_prior" else None
            )
            if side_cb is not None:
                result.notes.append(
                    f"Chessboard detected in side view — side geometry uses "
                    f"{side_cb.cm_per_px:.4f} cm/px."
                )

        # 1. Segment the whole frame; no plate-interior restriction.
        segments = self.segmenter.segment(
            top,
            interior_mask=None,
            min_area_frac=seg["min_area_frac"],
            max_area_frac=seg["max_area_frac"],
        )[:seg["max_segments"]]
        if not segments:
            result.notes.append("No distinct regions detected.")
            return result

        # 2. Recognise each candidate; keep food only.
        candidates: list[tuple[InstanceMask, "Recognition"]] = []
        n_rejected = 0
        # Recognition consumes the actual FastSAM shape. A plain bounding-box
        # crop can include a neighbouring food (e.g. apple pixels around a leaf)
        # and incorrectly transfer that food label to the small non-food mask.
        crops = [inst.masked_crop(top) for inst in segments]
        recognitions = self.recognizer.recognize_many(crops, batch_size=8)
        for index, (inst, rec) in enumerate(zip(segments, recognitions), start=1):
            rec = self._promote_large_food_part_alternative(inst, rec, top.shape[:2])
            passed_filter = rec.is_food and rec.score >= min_confidence
            visible_label = rec.label if rec.is_food else (rec.top[0][0] if rec.top else "non-food")
            trace = RegionTrace(
                index=index,
                mask=inst,
                label=visible_label,
                score=rec.score,
                is_food=rec.is_food,
                passed_filter=passed_filter,
                alternatives=list(rec.top[:5]),
            )
            if passed_filter:
                candidates.append((inst, rec))
            else:
                n_rejected += 1
                trace.reason = "non-food gate" if not rec.is_food else "below confidence threshold"
            result.trace_regions.append(trace)

        plate_calibration = (
            plate_candidate
            if plate_candidate is not None and self._plate_reference_visible(result.trace_regions)
            else None
        )
        if plate_calibration is not None:
            result.notes.append(
                f"Plate detected and used as a scale cue (assumed diameter "
                f"{config.DEFAULT_PLATE_DIAMETER_CM:.0f} cm, confidence "
                f"{plate_calibration.score:.0%})."
            )

        # FastSAM emits the same object at several scales; collapse overlapping ones.
        kept = self._suppress_nested(candidates)
        kept_mask_ids = {id(inst) for inst, _ in kept}
        for trace in result.trace_regions:
            if trace.passed_filter:
                trace.kept = id(trace.mask) in kept_mask_ids
                if not trace.kept:
                    trace.reason = "overlapping duplicate"

        # Remove small non-food parts (for example an apple leaf) that FastSAM also
        # emitted as a separate region and CLIP explicitly rejected. The recognition
        # decision now affects the physical measurement instead of being display-only.
        measurement_kept: list[tuple[InstanceMask, InstanceMask, Recognition]] = []
        for original_inst, rec in kept:
            measurement_inst, removed_px = self._refine_measurement_mask(
                original_inst,
                rec.label,
                result.trace_regions,
            )
            measurement_kept.append((measurement_inst, original_inst, rec))
            for trace in result.trace_regions:
                if trace.mask is original_inst:
                    trace.measurement_mask = measurement_inst
                    trace.removed_px = removed_px
                    break
            if removed_px:
                result.notes.append(
                    f"Removed {removed_px} non-food/appendage pixel(s) from the "
                    f"{rec.label} measurement mask before area and volume calculation."
                )

        dominant = max(measurement_kept, key=lambda entry: entry[0].area_px) \
            if measurement_kept else None
        side_profile = None
        if side is not None:
            target_label = dominant[2].label if dominant is not None else None
            side_profile, result.side_trace_regions = self._side_height_profile(
                side,
                side_cb,
                seg,
                target_label=target_label,
                min_confidence=min_confidence,
            )
            if side_profile is not None:
                if (
                    side_cb is None
                    and side_plate_candidate is not None
                    and self._plate_reference_visible(result.side_trace_regions)
                ):
                    side_profile.cm_per_px = side_plate_candidate.cm_per_px
                    side_profile.scale_source = "side_plate"
                    result.notes.append(
                        "A detected plate supplied the side-view physical scale."
                    )
                result.side_mask = side_profile.mask
                result.side_height_px = side_profile.height_px
                if side_profile.cm_per_px is None:
                    result.notes.append(
                        "Side-view scale is inferred by matching its selected food "
                        "silhouette width to the top view."
                    )
            else:
                result.notes.append(
                    "The side image contained no region that passed the same food "
                    "recognition gate. Falling back to top-view quantity priors."
                )

        # One unlabelled side photo can describe only one top-view object reliably.
        side_target_id = id(dominant[1]) if side_profile is not None and dominant else None
        if side_target_id is not None and len(measurement_kept) > 1:
            result.notes.append(
                "The selected side food silhouette was paired with the largest "
                "top-view food region only; other items use area-to-mass priors."
            )

        # 3 + 4. For each item: self-calibrate, choose class, estimate quantity.
        for inst, original_inst, rec in measurement_kept:
            # Try the top-1 food class; if its plausible range can't contain the
            # measurement, re-evaluate the same area against CLIP's other food
            # candidates. Non-food sentinels are allowed to reject/carve masks,
            # but must never become final nutrition items.
            chosen_label, chosen_info, cm_per_px, scale_src, area_cm2, raw_mass, \
                was_reranked, scale_evidence = self._pick_class_and_scale(
                    inst,
                    rec,
                    cb,
                    plate_calibration,
                    scale_mode,
                )

            info = chosen_info
            lo = info.mass_min_g if info.mass_min_g is not None else 0.0
            hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
            item_side_profile = side_profile if id(original_inst) == side_target_id else None
            two_view = None
            if item_side_profile is not None and item_side_profile.mask is not None:
                two_view = estimate_two_view_volume(
                    inst.mask,
                    item_side_profile.mask.mask,
                    cm_per_px,
                    side_cm_per_px=item_side_profile.cm_per_px,
                    side_scale_source=item_side_profile.scale_source,
                    max_height_cm=MAX_HEIGHT_CM,
                )
            if two_view is not None:
                height_cm = two_view.height_cm
                height_src = two_view.scale_source
            else:
                height_cm, height_src = self._height_for_item(
                    item_side_profile,
                    cm_per_px,
                    item_long_px=max(inst.bbox[2], inst.bbox[3]),
                )
            qty = estimate_quantity(
                info,
                area_cm2,
                self.volume,
                height_cm=height_cm,
                height_source=height_src,
                measured_volume_ml=two_view.volume_ml if two_view is not None else None,
                hard_factor=self._hard_plausibility_factor(scale_src, height_src),
            )
            if height_src != "none":
                result.height_source = height_src

            # Quantity confidence: how trustworthy is *the mass*?
            q_conf = self._quantity_confidence(
                clip_score=rec.score, raw_mass=qty.raw_mass_g, lo=lo, hi=hi,
                typical=info.typical_mass_g or (lo + hi) / 2 if hi != float("inf") else lo,
                scale_src=scale_src,
                was_reranked=was_reranked,
            )

            item = ItemEstimate(
                food_class=chosen_label, confidence=rec.score, area_cm2=area_cm2,
                height_cm=qty.height_cm, volume_ml=qty.volume_ml,
                nutrition=info.for_mass(qty.mass_g), mask=inst,
                alternatives=[
                    (l, s)
                    for l, s in self._food_rerank_candidates(rec)
                    if l != chosen_label
                ][:3],
            )
            item.scale_source = f"reranked:{scale_src}" if was_reranked else scale_src
            item.height_source = height_src
            item.mass_source = qty.source
            item.cm_per_px = cm_per_px
            item.typical_mass_g = info.typical_mass_g or 0.0
            item.mass_range_g = (lo, hi if hi != float("inf") else 0.0)
            item.raw_mass_g = qty.raw_mass_g
            item.raw_volume_ml = qty.raw_volume_ml
            item.quantity_confidence = q_conf
            item.two_view = two_view
            item.scale_evidence = scale_evidence
            result.items.append(item)

            for trace in result.trace_regions:
                if trace.mask is original_inst:
                    trace.final_label = chosen_label
                    break

            if was_reranked:
                result.notes.append(
                    f"Top-1 class '{rec.label}' didn't fit the measured size; "
                    f"re-ranked to '{chosen_label}' from CLIP's alternatives."
                )
            if qty.clamped:
                result.notes.append(
                    f"{chosen_label}: raw estimate {qty.raw_mass_g:.0f} g was far outside "
                    f"the typical range ({lo:.0f}-{hi:.0f} g); capped to "
                    f"{qty.mass_g:.0f} g."
                )
            elif not self._mass_in_plausible_range(qty.raw_mass_g, lo, hi):
                result.notes.append(
                    f"{chosen_label}: raw estimate {qty.raw_mass_g:.0f} g is outside "
                    f"the typical range ({lo:.0f}-{hi:.0f} g); keeping it as a "
                    "large/small portion rather than snapping to the range."
                )

        if not result.items:
            result.notes.append(
                f"None of the {len(segments)} detected regions were recognised as food."
            )
        elif n_rejected:
            result.notes.append(f"{n_rejected} non-food region(s) were filtered out.")
        if any(item.two_view is not None for item in result.items):
            result.notes.append(
                "Volume was reconstructed from the actual top- and side-view silhouette "
                "profiles (elliptical cross-section integration)."
            )
        if any(it.scale_source == "fallback" for it in result.items):
            result.notes.append(
                "Some items had no typical_long_cm in the nutrition table — their masses "
                "use a generic fallback scale and may be off."
            )
        return result

    @staticmethod
    def _segmentation_config(preset: str) -> dict[str, float]:
        """Return robust segmentation parameters for a user-facing preset."""
        return SEGMENTATION_PRESETS.get(preset, SEGMENTATION_PRESETS["balanced"])

    @staticmethod
    def _normalise_scale_mode(scale_mode: str) -> str:
        if scale_mode in {"auto", "class_prior", "metric_reference"}:
            return scale_mode
        return "auto"

    @staticmethod
    def _plate_scale_candidate(image_bgr: np.ndarray):
        """Return a plausible dinner-plate calibration candidate, if present."""
        from .calibration import CalibrationError, calibrate

        try:
            calibration = calibrate(
                image_bgr,
                real_diameter_cm=config.DEFAULT_PLATE_DIAMETER_CM,
                expect="largest",
            )
        except (CalibrationError, ValueError):
            return None
        height, width = image_bgr.shape[:2]
        cx, cy = calibration.center
        large_enough = calibration.diameter_px >= 0.35 * min(height, width)
        centred = abs(cx - width / 2) <= 0.38 * width and abs(cy - height / 2) <= 0.38 * height
        if not large_enough or not centred or calibration.score < 0.35:
            return None
        return calibration

    @staticmethod
    def _plate_reference_visible(traces: list[RegionTrace]) -> bool:
        """Require CLIP's non-food gate to independently confirm a plate."""
        for trace in traces:
            if not trace.is_food and "plate" in trace.label.lower():
                return True
        return False

    @staticmethod
    def _mass_in_plausible_range(mass_g: float, lo: float, hi: float) -> bool:
        if mass_g < lo:
            return False
        if np.isfinite(hi) and mass_g > hi:
            return False
        return True

    @staticmethod
    def _hard_plausibility_factor(
        scale_src: str,
        height_src: str,
    ) -> float:
        weak_top_only_scale = (
            height_src == "none"
            and ("class_prior" in scale_src or scale_src == "fallback")
        )
        if weak_top_only_scale:
            return 1.0
        return HARD_PLAUSIBILITY_FACTOR

    @staticmethod
    def _promote_large_food_part_alternative(
        inst: InstanceMask,
        rec: Recognition,
        image_shape: tuple[int, int],
    ) -> Recognition:
        """Promote strong food alternatives only for large regions gated as food parts."""
        if rec.is_food or not rec.food_top:
            return rec

        nonfood_label = rec.nonfood_label or (rec.top[0][0] if rec.top else "")
        if nonfood_kind(nonfood_label) != "food_part":
            return rec

        h, w = image_shape
        image_area = max(float(h * w), 1.0)
        area_frac = inst.area_px / image_area
        _, _, bw, bh = inst.bbox
        large_region = area_frac >= 0.025 and max(bw, bh) >= 0.16 * min(h, w)
        if not large_region:
            return rec

        food_top = rec.food_top or [
            (label, score)
            for label, score in rec.top
            if label in nutrition.known_classes()
        ]
        for label, score in food_top:
            close_to_sentinel = score >= 0.20 and score >= 0.60 * rec.score
            if close_to_sentinel:
                return Recognition(
                    label=label,
                    score=score,
                    is_food=True,
                    top=rec.top,
                    food_top=rec.food_top,
                    nonfood_top=rec.nonfood_top,
                    nonfood_label=nonfood_label,
                )
        return rec

    @classmethod
    def _fused_item_scale(
        cls,
        inst: InstanceMask,
        info,
        chessboard,
        plate_calibration,
        scale_mode: str,
    ) -> tuple[float, str, list[ScaleEvidence]]:
        """Fuse all consistent scale cues, preferring direct metric evidence."""
        class_cm, class_source = cls._per_item_scale(inst, info)
        evidence: list[ScaleEvidence] = []
        if scale_mode != "class_prior":
            if chessboard is not None:
                evidence.append(ScaleEvidence(
                    "chessboard",
                    float(chessboard.cm_per_px),
                    float(np.clip(0.75 + 0.25 * chessboard.confidence, 0.0, 1.0)),
                ))
            if plate_calibration is not None:
                evidence.append(ScaleEvidence(
                    "plate",
                    float(plate_calibration.cm_per_px),
                    float(np.clip(0.55 + 0.25 * plate_calibration.score, 0.0, 0.80)),
                ))
        # The recognised item size is always useful as a weak consistency cue and
        # is the final fallback when no physical reference is visible.
        evidence.append(ScaleEvidence(
            class_source,
            class_cm,
            0.35 if class_source == "class_prior" else 0.15,
        ))

        anchor = max(evidence, key=lambda item: item.confidence)
        compatible = [
            item for item in evidence
            if max(item.cm_per_px, anchor.cm_per_px) / max(min(item.cm_per_px, anchor.cm_per_px), 1e-9) <= 1.5
        ]
        weights = np.asarray([item.confidence ** 2 for item in compatible], dtype=np.float64)
        log_scales = np.log([item.cm_per_px for item in compatible])
        fused = float(np.exp(np.average(log_scales, weights=weights)))
        used_sources = [item.source for item in sorted(compatible, key=lambda item: -item.confidence)]
        labelled = [
            ScaleEvidence(item.source, item.cm_per_px, item.confidence, item in compatible)
            for item in evidence
        ]
        source = used_sources[0] if len(used_sources) == 1 else "fused:" + "+".join(used_sources)
        return fused, source, labelled

    @staticmethod
    def _subtract_rejected_parts(
        food_inst: InstanceMask,
        traces: list[RegionTrace],
    ) -> tuple[InstanceMask, int]:
        """Subtract overlapping, local non-food masks from a kept food mask.

        A leaf mask is often not fully contained in the fruit silhouette: it can
        cross the fruit boundary or include a little surrounding background.  The
        old 75%-containment rule therefore left exactly those leaf pixels in the
        volume mask.  We accept partial overlap when the rejected candidate is
        still small relative to the food and spatially attached to it.
        """
        refined = food_inst.mask.copy()
        removed = 0
        for trace in traces:
            # A food candidate that merely missed the user's score threshold is
            # not evidence of non-food and must never carve holes into another food.
            if trace.is_food or trace.mask is food_inst:
                continue
            rejected = trace.mask.mask
            overlap = int(np.count_nonzero(refined & rejected))
            if overlap == 0:
                continue
            contained_ratio = overlap / max(trace.mask.area_px, 1)
            food_ratio = overlap / max(food_inst.area_px, 1)
            candidate_ratio = trace.mask.area_px / max(food_inst.area_px, 1)
            fx, fy, fw, fh = food_inst.bbox
            cx, cy = trace.mask.centroid
            near_food = (
                fx - 0.20 * fw <= cx <= fx + 1.20 * fw
                and fy - 0.20 * fh <= cy <= fy + 1.20 * fh
            )
            # Never subtract plate/background-sized regions. A compact rejected
            # part may be either mostly contained or only partly overlap at the
            # outer contour (the common fruit-leaf case).
            local_part = candidate_ratio <= 0.35 and near_food
            enough_overlap = contained_ratio >= 0.25 or food_ratio >= 0.01
            if food_ratio <= 0.30 and local_part and enough_overlap:
                refined &= ~rejected
                removed += overlap
        if removed == 0:
            return food_inst, 0
        refined_inst = _mask_to_instance(refined)
        return (refined_inst or food_inst), removed

    @classmethod
    def _refine_measurement_mask(
        cls,
        food_inst: InstanceMask,
        food_label: str,
        traces: list[RegionTrace],
    ) -> tuple[InstanceMask, int]:
        """Build the exact silhouette used by area/volume measurement.

        Recognition-gated non-food regions are removed first.  Whole apples get
        one additional conservative shape pass: a small morphological opening
        disconnects thin stems/leaves and the largest compact component is kept.
        This is intentionally class-scoped so irregular foods such as pizza or
        salad are not rounded off.
        """
        refined, removed = cls._subtract_rejected_parts(food_inst, traces)
        if food_label != "apple":
            return refined, removed

        _, _, width, height = refined.bbox
        short_side = min(width, height)
        if short_side < 24:
            return refined, removed
        radius = max(2, int(round(short_side * 0.04)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        opened = cv2.morphologyEx(
            refined.mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        if count <= 1:
            return refined, removed
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        compact = labels == largest
        compact_inst = _mask_to_instance(compact)
        if compact_inst is None:
            return refined, removed

        # Guard against mutilating an unusual but valid apple silhouette. The
        # pass is accepted only when at least 78% of the already gated mask stays.
        retained = compact_inst.area_px / max(refined.area_px, 1)
        shape_removed = refined.area_px - compact_inst.area_px
        # Pure raster-corner smoothing can change a handful of pixels even when
        # no appendage exists. Do not replace the instance in that case.
        meaningful_change = shape_removed >= max(12, int(round(0.005 * refined.area_px)))
        if retained < 0.78 or not meaningful_change:
            return refined, removed
        return compact_inst, removed + shape_removed

    def _side_height_profile(
        self,
        side_bgr: np.ndarray,
        chessboard,
        seg: dict[str, float],
        *,
        target_label: Optional[str],
        min_confidence: float,
    ) -> tuple[Optional[_SideHeightProfile], list[RegionTrace]]:
        """Select a side silhouette using the same FastSAM + CLIP gate as the top."""
        segments = self.segmenter.segment(
            side_bgr,
            interior_mask=None,
            min_area_frac=max(0.001, seg["min_area_frac"] * 0.75),
            max_area_frac=min(0.85, seg["max_area_frac"]),
        )[:int(seg["max_segments"])]
        if not segments:
            return None, []

        recognitions = self.recognizer.recognize_many(
            [inst.masked_crop(side_bgr) for inst in segments],
            batch_size=8,
        )
        traces: list[RegionTrace] = []
        candidates: list[tuple[InstanceMask, Recognition]] = []
        for index, (inst, rec) in enumerate(zip(segments, recognitions), start=1):
            rec = self._promote_large_food_part_alternative(inst, rec, side_bgr.shape[:2])
            passed = rec.is_food and rec.score >= min_confidence
            visible_label = rec.label if rec.is_food else (rec.top[0][0] if rec.top else "non-food")
            trace = RegionTrace(
                index=index,
                mask=inst,
                label=visible_label,
                score=rec.score,
                is_food=rec.is_food,
                passed_filter=passed,
                alternatives=list(rec.top[:5]),
            )
            if passed:
                candidates.append((inst, rec))
            else:
                trace.reason = "non-food gate" if not rec.is_food else "below confidence threshold"
            traces.append(trace)

        kept = self._suppress_nested(candidates)
        if not kept:
            return None, traces

        kept_ids = {id(inst) for inst, _ in kept}
        for trace in traces:
            if trace.passed_filter and id(trace.mask) not in kept_ids:
                trace.reason = "overlapping duplicate"

        def target_score(rec: Recognition) -> float:
            if target_label is None:
                return 0.0
            return max((score for label, score in rec.top if label == target_label), default=0.0)

        # Prefer the top-view label when it appears in the side-view alternatives;
        # otherwise choose the strongest region that still passed the food gate.
        inst, selected_rec = max(
            kept,
            key=lambda pair: (target_score(pair[1]), pair[1].score, pair[0].area_px),
        )
        for trace in traces:
            if trace.mask is inst:
                trace.kept = True
                trace.final_label = target_label or selected_rec.label
            elif trace.passed_filter and id(trace.mask) in kept_ids:
                trace.reason = "food region not selected for height"

        measurement_inst, removed_px = self._refine_measurement_mask(
            inst,
            target_label or selected_rec.label,
            traces,
        )
        for trace in traces:
            if trace.mask is inst:
                trace.measurement_mask = measurement_inst
                trace.removed_px = removed_px
                break

        _, _, width_px, height_px = measurement_inst.bbox
        if height_px <= 0 or width_px <= 0:
            return None, traces
        cm_per_px = chessboard.cm_per_px if chessboard is not None else None
        scale_source = "side_chessboard" if chessboard is not None else "side_item_scale"
        return (
            _SideHeightProfile(
                height_px=float(height_px),
                cm_per_px=cm_per_px,
                scale_source=scale_source,
                mask=measurement_inst,
                width_px=float(width_px),
            ),
            traces,
        )

    @staticmethod
    def _height_for_item(
        side_profile: Optional[_SideHeightProfile],
        item_cm_per_px: float,
        item_long_px: Optional[float] = None,
    ) -> tuple[Optional[float], str]:
        if side_profile is None:
            return None, "none"
        scale = side_profile.cm_per_px
        source = side_profile.scale_source
        if scale is None:
            if item_cm_per_px <= 0:
                return None, "none"
            if item_long_px and item_long_px > 0 and side_profile.width_px > 0:
                item_length_cm = item_long_px * item_cm_per_px
                scale = item_length_cm / side_profile.width_px
                source = "side_width_matched"
            else:
                scale = item_cm_per_px
        height_cm = float(np.clip(side_profile.height_px * scale, 0.3, MAX_HEIGHT_CM))
        return height_cm, source

    @staticmethod
    def _per_item_scale(inst: "InstanceMask", info) -> tuple[float, str]:
        """Return (cm/px, source) for a single item, *class-only* fallback.

        Compares the item's bounding-box long side in pixels to the class's
        ``typical_long_cm``. Used when no chessboard is detected.
        """
        long_side_px = max(inst.bbox[2], inst.bbox[3])
        if info.typical_long_cm is not None and long_side_px > 0:
            return float(info.typical_long_cm / long_side_px), "class_prior"
        return float(10.0 / max(long_side_px, 1)), "fallback"

    def _pick_class_and_scale(
        self,
        inst,
        rec,
        chessboard,
        plate_calibration=None,
        scale_mode: str = "auto",
    ):
        """Decide on (class, cm/px) using all evidence.

        Strategy:
        1. Compute cm/px. If a chessboard is present, use that scale (real metric);
           else fall back to the recognised class's typical_long_cm.
        2. Compute raw_mass = mass_per_cm2[top-1] × area_cm2.
        3. If raw_mass fits the top-1 plausible [min, max] range → keep top-1.
        4. Otherwise look through CLIP's other food candidates. If one of them
           has a plausible range that *contains* the raw_mass (re-scaled to its
           own areal density), pick it. Explicit non-food sentinels are ignored
           here; they can reject/carve masks, but never become nutrition items.
        5. If no candidate fits, keep top-1 and let the soft range warning/cap
           fire later.
        """
        candidates = self._food_rerank_candidates(rec)
        if not candidates:
            candidates = [(rec.label, rec.score)]

        def evaluate(label, forced_scale_mode: Optional[str] = None):
            info = nutrition.lookup(label)
            cm_per_px, scale_src, scale_evidence = self._fused_item_scale(
                inst,
                info,
                chessboard,
                plate_calibration,
                forced_scale_mode or scale_mode,
            )
            area_cm2 = (cm_per_px ** 2) * inst.area_px
            raw_mass = self._raw_mass(info, area_cm2)
            return info, cm_per_px, scale_src, area_cm2, raw_mass, scale_evidence

        # Evaluate top-1 first.
        top_label = candidates[0][0]
        info, cm_per_px, scale_src, area_cm2, raw_mass, scale_evidence = evaluate(top_label)
        lo = info.mass_min_g if info.mass_min_g is not None else 0.0
        hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
        if "plate" in scale_src and "chessboard" not in scale_src and not (lo <= raw_mass <= hi):
            c_info, c_cm, c_src, c_area, c_mass, c_evidence = evaluate(top_label, "class_prior")
            c_lo = c_info.mass_min_g if c_info.mass_min_g is not None else 0.0
            c_hi = c_info.mass_max_g if c_info.mass_max_g is not None else float("inf")
            if self._mass_in_plausible_range(c_mass, c_lo, c_hi):
                return top_label, c_info, c_cm, c_src, c_area, c_mass, False, c_evidence
        if lo <= raw_mass <= hi:
            return top_label, info, cm_per_px, scale_src, area_cm2, raw_mass, False, scale_evidence

        # Top-1 fails plausibility. Try alternatives.
        for alt_label, _ in candidates[1:]:
            a_info, a_cm, a_src, a_area, a_mass, a_evidence = evaluate(alt_label)
            a_lo = a_info.mass_min_g if a_info.mass_min_g is not None else 0.0
            a_hi = a_info.mass_max_g if a_info.mass_max_g is not None else float("inf")
            if a_lo <= a_mass <= a_hi:
                return alt_label, a_info, a_cm, a_src, a_area, a_mass, True, a_evidence

        # No candidate fits; stay with top-1 and let the soft range logic fire.
        return top_label, info, cm_per_px, scale_src, area_cm2, raw_mass, False, scale_evidence

    @staticmethod
    def _food_rerank_candidates(rec: Recognition) -> list[tuple[str, float]]:
        """Return CLIP alternatives that can legally become nutrition items."""
        known_food = set(nutrition.known_classes())
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()
        food_top = rec.food_top or rec.top
        for label, score in [(rec.label, rec.score), *food_top]:
            if label in known_food and label not in seen:
                candidates.append((label, score))
                seen.add(label)
            if len(candidates) >= 5:
                break
        if not candidates and rec.is_food:
            # The fallback Food-101 classifier may emit a label missing from the
            # local nutrition table. Keep that legacy path, but never admit CLIP's
            # explicit non-food sentinels into re-ranking.
            candidates.append((rec.label, rec.score))
        return candidates

    @staticmethod
    def _raw_mass(info, area_cm2: float) -> float:
        """The before-bound mass estimate for one (class, area)."""
        return area_mass_prior(info, area_cm2)

    @staticmethod
    def _quantity_confidence(*, clip_score, raw_mass, lo, hi, typical,
                             scale_src, was_reranked):
        """0..1 confidence in the *mass*, not just the class label.

        Combines:
        - CLIP score (how sure is the recogniser about *some* class)
        - Plausibility: did the raw estimate fall inside the typical range,
          and how close was it to the class's typical value
        - Scale source: a chessboard scale is much more trustworthy than a
          class-prior scale (which is just a guess about typical size)
        - Re-ranking penalty: a re-ranked class is by construction less certain
          than CLIP's top pick
        """
        # Plausibility.
        if hi == float("inf"):
            plaus = 0.5
        elif lo <= raw_mass <= hi:
            band = max(hi - lo, 1e-6)
            dist = abs(raw_mass - typical) / band
            plaus = float(np.clip(1.0 - 0.6 * dist, 0.3, 1.0))
        else:
            plaus = 0.15   # outside the typical range; keep/cap it with low confidence

        if "chessboard" in scale_src:
            scale_factor = 1.0
        elif "plate" in scale_src:
            scale_factor = 0.88
        elif "class_prior" in scale_src:
            scale_factor = 0.7
        elif scale_src == "fallback":
            scale_factor = 0.4
        else:
            scale_factor = 0.6

        clip_factor = float(np.clip(clip_score, 0.0, 1.0))
        rerank_factor = 0.75 if was_reranked else 1.0
        return float(np.clip(clip_factor * plaus * scale_factor * rerank_factor, 0.0, 1.0))

    @staticmethod
    def _suppress_nested(candidates, containment_thresh: float = 0.6):
        """Greedy NMS by containment: keep the most credible mask per object.

        Two masks of the same physical item (e.g. the apple and the apple+plate region)
        overlap heavily even if their IoU is low, so we compare against the smaller mask.
        """
        candidates = list(candidates)

        # A tiny leaf/stem mask can still be misclassified as the same food as
        # its parent fruit, especially when CLIP falls back to Food-101. If that
        # small mask is almost wholly inside a same-label candidate at least 5.5x
        # larger, it is an object part, not the apple instance. This pre-pass is
        # deliberately narrow: the normal tighter-mask/high-score rule below still
        # wins for apple-vs-apple+plate candidates of comparable scale.
        drops: set[int] = set()
        for index, (inst, rec) in enumerate(candidates):
            for other_index, (other_inst, other_rec) in enumerate(candidates):
                if index == other_index:
                    continue
                overlap = int(np.count_nonzero(inst.mask & other_inst.mask))
                if overlap / max(inst.area_px, 1) <= containment_thresh:
                    continue
                area_ratio = other_inst.area_px / max(inst.area_px, 1)
                if rec.label == other_rec.label and area_ratio >= 5.5 and inst.area_px < 0.18 * other_inst.area_px:
                    drops.add(index)
                    break
                if area_ratio >= CONTAINMENT_AREA_RATIO:
                    broad_context = FoodVolumePipeline._nonfood_context_score(
                        other_rec,
                        {"container", "surface", "background"},
                    )
                    broad_is_ambiguous_context = broad_context >= 0.45 * max(other_rec.score, 1e-9)
                    broad_not_decisive = other_rec.score <= rec.score + CONTAINMENT_SCORE_MARGIN
                    if broad_is_ambiguous_context or broad_not_decisive:
                        drops.add(other_index)
        candidates = [
            candidate for index, candidate in enumerate(candidates)
            if index not in drops
        ]

        candidates = sorted(candidates, key=lambda c: c[1].score, reverse=True)
        kept: list[tuple[InstanceMask, "Recognition"]] = []
        for inst, rec in candidates:
            duplicate = False
            for kinst, _ in kept:
                inter = int(np.logical_and(inst.mask, kinst.mask).sum())
                if inter / max(1, min(inst.area_px, kinst.area_px)) > containment_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((inst, rec))
        return kept

    @staticmethod
    def _nonfood_context_score(rec: Recognition, kinds: set[str]) -> float:
        labels = rec.nonfood_top or rec.top
        return max(
            (score for label, score in labels if nonfood_kind(label) in kinds),
            default=0.0,
        )
