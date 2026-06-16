"""Tests for lightweight segmentation fallbacks."""
import cv2
import numpy as np

from foodvol.segmentation import FoodSegmenter, _mask_to_instance


def test_classical_segmentation_handles_low_contrast_white_background():
    img = np.full((220, 260, 3), 255, np.uint8)
    cv2.ellipse(img, (130, 110), (55, 35), 0, 0, 360, (248, 248, 248), -1)
    cv2.ellipse(img, (130, 110), (55, 35), 0, 0, 360, (220, 220, 220), 2)

    seg = FoodSegmenter(device="cpu")
    instances = [_mask_to_instance(mask) for mask in seg._segment_classical(img, None)]
    areas = [inst.area_px for inst in instances if inst is not None]

    assert max(areas) > 4000
