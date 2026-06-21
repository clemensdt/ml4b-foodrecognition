"""Open-vocabulary food recognition with a non-food gate (CLIP zero-shot).

The earlier classifier had two problems that made the app fail on real photos:

1. It could only output one of the 101 Food-101 *dishes*, so a plain apple was
   mislabelled, and it had **no way to say "this isn't food"** — every checkerboard
   square or placemat patch became some dish.
2. That meant background patterns were segmented and labelled as food.

This module fixes both with **CLIP zero-shot classification**. CLIP scores an image
against arbitrary text labels, so we score each candidate region against:

* a broad **food vocabulary** (every class in the nutrition table, fruits included), and
* a set of **non-food sentinels** ("a checkerboard pattern", "a plate", "a table"…).

If a non-food sentinel wins, the region is rejected. This both **recognises real
dishes and fruit** and **gates out non-food**, which is exactly what the segmentation
stage needs. CLIP is a strong pretrained internet model; we only train the *portion*
(volume) model ourselves.

Text features for the fixed label set are computed once and cached, so recognising a
region is a single image encode plus a matrix multiply — fast enough for many regions.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from . import config, nutrition

# Non-food labels used purely as a gate. If one of these wins, the region is not food.
NONFOOD_LABELS = [
    "a checkerboard pattern", "a polka dot mat", "a calibration board",
    "an empty plate", "a plate", "a bowl", "a table surface", "a placemat",
    "a fork", "a knife", "a spoon", "cutlery", "a napkin", "a hand",
    "a leaf attached to fruit", "a fruit stem", "fabric", "a wall",
    "the floor", "plain background",
]

HYPOTHESIS = "a photo of {}."

# Some classes benefit from a more descriptive natural-language prompt than the
# bare class name. CLIP is sensitive to wording — "a whole avocado" scores much
# better on a whole-fruit photo than "avocado" alone.
PROMPT_OVERRIDES = {
    "apple":            "a whole apple",
    "banana":           "a whole banana",
    "avocado":          "a whole avocado with skin",
    "half_avocado":     "half an avocado, cut open showing the pit",
    "watermelon_wedge": "a wedge of watermelon",
    "pineapple_slice":  "a slice of pineapple",
    "pizza_slice":      "a slice of pizza",
    "pizza_whole":      "a whole pizza",
    "boiled_egg":       "a boiled egg",
    "fried_egg":        "a fried egg",
    "egg":              "an egg",
    "muffin":           "a muffin",
    "blueberry_muffin": "a blueberry muffin with berries on top",
    "cupcake":          "a frosted cupcake",
    "donut":            "a donut with a hole",
    "cookie":           "a single cookie",
}


@dataclass
class Recognition:
    """Result of recognising one region."""

    label: str          # nutrition-table key (e.g. "apple", "apple_pie"); "unknown" if gated
    score: float        # probability of the winning label
    is_food: bool
    top: list[tuple[str, float]]  # top-k (label, score) for transparency


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class FoodRecognizer:
    """CLIP zero-shot recogniser with a non-food gate. Model loaded lazily."""

    def __init__(self, model_id: str = config.CLIP_MODEL_HF, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or config.get_device()
        self._model = None
        self._processor = None
        self._text_features = None      # (N, D) normalised
        self._labels: list[str] = []     # nutrition keys aligned with text features
        self._is_food: np.ndarray = np.array([])
        self.available = True
        self._fallback = None            # FoodClassifier if CLIP unavailable
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()

    # --- label set -------------------------------------------------------------
    def _build_labels(self) -> tuple[list[str], list[str], np.ndarray]:
        """Return (prompt_texts, nutrition_keys, is_food_mask).

        Food prompts use ``PROMPT_OVERRIDES`` when available — natural-language
        descriptions like "half an avocado, cut open" help CLIP distinguish
        whole vs. cut produce, which then drives the correct portion prior.
        """
        food_keys = nutrition.known_classes()
        food_texts = [PROMPT_OVERRIDES.get(k, k.replace("_", " ")) for k in food_keys]
        keys = food_keys + NONFOOD_LABELS
        texts = food_texts + NONFOOD_LABELS
        is_food = np.array([True] * len(food_keys) + [False] * len(NONFOOD_LABELS))
        return texts, keys, is_food

    def _ensure_model(self) -> bool:
        ready = (
            self._model is not None
            and self._processor is not None
            and self._text_features is not None
            and bool(self._labels)
            and self._is_food.size > 0
        )
        if ready:
            return True

        # Loading used to assign ``_model`` before ``_processor`` was ready. A
        # concurrent Streamlit rerun could then observe the half-loaded object and
        # call ``None`` as a processor. Build everything locally and publish the
        # complete state only once all pieces are usable.
        load_lock = getattr(self, "_load_lock", None)
        if load_lock is None:  # tolerate a cached instance created before this field existed
            self._load_lock = threading.RLock()
            load_lock = self._load_lock
        with load_lock:
            ready = (
                self._model is not None
                and self._processor is not None
                and self._text_features is not None
                and bool(self._labels)
                and self._is_food.size > 0
            )
            if ready:
                return True
            # Drop a half-loaded model before retrying. On MPS this matters: keeping
            # the broken model alive while loading another copy can temporarily
            # double unified-memory pressure.
            self._model = self._processor = self._text_features = None
            self._labels = []
            self._is_food = np.array([])
            if not self.available:
                return False
            if self.device == "mps":
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                processor = CLIPProcessor.from_pretrained(self.model_id)
                model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
                texts, keys, is_food = self._build_labels()
                with torch.inference_mode():
                    prompts = [HYPOTHESIS.format(t) for t in texts]
                    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    feats = self._embeds(model.get_text_features(**inputs))
                    feats = feats / feats.norm(dim=-1, keepdim=True)

                self._model = model
                self._processor = processor
                self._labels = keys
                self._is_food = is_food
                self._text_features = feats
                return True
            except Exception as exc:
                print(f"[recognition] CLIP unavailable ({exc}); falling back to the Food-101 classifier "
                      "(no non-food gate).")
                self._model = self._processor = self._text_features = None
                self.available = False
                return False

    @staticmethod
    def _embeds(out):
        """Return joint-space embeddings, tolerating transformers API drift.

        Older transformers return the projected embedding tensor directly; newer ones
        return a base output whose ``pooler_output`` already holds the projected
        joint-space embedding.
        """
        import torch
        if torch.is_tensor(out):
            return out
        return out.pooler_output

    # --- inference -------------------------------------------------------------
    def recognize(self, image: Union[np.ndarray, Image.Image], top_k: int = 5) -> Recognition:
        return self.recognize_many([image], top_k=top_k)[0]

    def recognize_many(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        top_k: int = 5,
        batch_size: int = 8,
    ) -> list[Recognition]:
        """Recognise regions in small batches instead of one model pass per mask.

        Batching removes most of the overhead that made a single photo trigger
        dozens of separate CLIP calls. The batch is deliberately small on Apple
        Silicon so unified memory usage stays bounded.
        """
        if not images:
            return []
        if not self._ensure_model():
            return [self._fallback_recognize(image) for image in images]
        import torch

        if self.device == "mps":
            batch_size = min(batch_size, 4)
        batch_size = max(1, int(batch_size))
        inference_lock = getattr(self, "_inference_lock", None)
        if inference_lock is None:
            self._inference_lock = threading.Lock()
            inference_lock = self._inference_lock

        results: list[Recognition] = []
        with inference_lock, torch.inference_mode():
            for start in range(0, len(images), batch_size):
                pil_batch = [_to_pil(image) for image in images[start:start + batch_size]]
                inputs = self._processor(images=pil_batch, return_tensors="pt").to(self.device)
                feats = self._embeds(self._model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = self._model.logit_scale.exp() * feats @ self._text_features.t()
                probabilities = logits.softmax(dim=-1).detach().cpu().numpy()

                for probs in probabilities:
                    order = np.argsort(probs)[::-1]
                    top = [(self._labels[i], float(probs[i])) for i in order[:top_k]]
                    best = int(order[0])
                    results.append(Recognition(
                        label=self._labels[best] if self._is_food[best] else "unknown",
                        score=float(probs[best]),
                        is_food=bool(self._is_food[best]),
                        top=top,
                    ))
        if self.device == "mps":
            torch.mps.empty_cache()
        return results

    def _fallback_recognize(self, image) -> Recognition:
        """Without CLIP, use the supervised Food-101 classifier (cannot gate non-food)."""
        if self._fallback is None:
            from .classification import FoodClassifier
            self._fallback = FoodClassifier(device=self.device)
        pred = self._fallback.classify_top1(image)
        return Recognition(label=pred.label, score=pred.score, is_food=True,
                           top=[(pred.label, pred.score)])
