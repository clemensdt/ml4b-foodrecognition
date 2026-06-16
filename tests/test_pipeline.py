"""Tests for pipeline helpers that don't require downloading models."""
import numpy as np

from foodvol.pipeline import FoodVolumePipeline, _SideHeightProfile
from foodvol.recognition import Recognition
from foodvol.segmentation import _mask_to_instance


def _inst(y0, y1, x0, x1):
    m = np.zeros((100, 100), bool)
    m[y0:y1, x0:x1] = True
    return _mask_to_instance(m)


def _rec(label, score):
    return Recognition(label=label, score=score, is_food=True, top=[(label, score)])


def test_suppress_nested_keeps_highest_score_and_drops_contained():
    tight = _inst(10, 50, 10, 50)      # the food, area 1600
    big = _inst(10, 60, 10, 60)        # food + plate, contains `tight`
    separate = _inst(70, 90, 70, 90)   # a different item, area 400

    candidates = [(big, _rec("apple", 0.5)),
                  (tight, _rec("apple", 0.8)),
                  (separate, _rec("pizza", 0.7))]
    kept = FoodVolumePipeline._suppress_nested(candidates)

    labels = sorted((inst.area_px, rec.label) for inst, rec in kept)
    assert len(kept) == 2                       # the contained `big` is dropped
    assert (400, "pizza") in labels
    assert (1600, "apple") in labels            # the tighter, higher-scoring mask survives


def test_suppress_nested_empty():
    assert FoodVolumePipeline._suppress_nested([]) == []


def test_height_for_item_uses_side_chessboard_scale():
    profile = _SideHeightProfile(height_px=50, cm_per_px=0.1, scale_source="side_chessboard")
    height, source = FoodVolumePipeline._height_for_item(profile, item_cm_per_px=0.2)
    assert height == 5.0
    assert source == "side_chessboard"


def test_height_for_item_falls_back_to_item_scale():
    profile = _SideHeightProfile(height_px=50, cm_per_px=None, scale_source="side_item_scale")
    height, source = FoodVolumePipeline._height_for_item(profile, item_cm_per_px=0.2)
    assert height == 10.0
    assert source == "side_item_scale"


def test_segmentation_presets_are_ordered():
    conservative = FoodVolumePipeline._segmentation_config("conservative")
    balanced = FoodVolumePipeline._segmentation_config("balanced")
    sensitive = FoodVolumePipeline._segmentation_config("sensitive")

    assert conservative["min_area_frac"] > balanced["min_area_frac"] > sensitive["min_area_frac"]
    assert conservative["max_segments"] < balanced["max_segments"] < sensitive["max_segments"]


def test_unknown_segmentation_preset_uses_balanced():
    assert FoodVolumePipeline._segmentation_config("wat") == FoodVolumePipeline._segmentation_config("balanced")
