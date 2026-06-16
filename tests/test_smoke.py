"""Lightweight smoke tests: everything imports and wires up without downloading models."""
import importlib

import numpy as np
import pytest


def test_all_modules_import():
    for mod in ["calibration", "segmentation", "classification", "depth", "recognition",
                "volume", "portion", "nutrition", "training", "pipeline", "datasets",
                "benchmark", "video", "config"]:
        importlib.import_module(f"foodvol.{mod}")


def test_pipeline_constructs_without_network():
    # Heavy models are lazy, so construction must not require any download.
    from foodvol.pipeline import FoodVolumePipeline
    pipe = FoodVolumePipeline()
    assert pipe.segmenter is not None
    assert pipe.recognizer is not None
    assert pipe.volume is not None  # loaded trained model or physics fallback


def test_volume_features_shape():
    from foodvol.volume import feature_matrix
    X = feature_matrix([10.0, 20.0], [1.0, 2.0])
    assert X.shape == (2, 3)
    assert np.allclose(X[1], [20.0, 2.0, 40.0])
