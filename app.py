"""Streamlit demo: estimate food volume, mass, calories and macros from photos.

Run with::

    streamlit run app.py

The user provides a top-down photo and optionally a side photo. The app runs the
:class:`foodvol.pipeline.FoodVolumePipeline` and shows per-item mass / calories /
macros with an annotated overlay.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from foodvol import config
from foodvol.pipeline import FoodVolumePipeline, PlateEstimate

st.set_page_config(page_title="Food Volume & Calorie Estimator", page_icon="🍽️", layout="wide")

# Distinct overlay colours (BGR) cycled per detected item.
_PALETTE = [(60, 60, 255), (60, 200, 60), (255, 160, 0), (200, 60, 200),
            (0, 200, 200), (160, 120, 60), (60, 160, 255)]

FOOD_FILTERS = {
    "Tolerant": 0.0,
    "Balanced": 0.02,
    "Strict": 0.05,
}

SCALE_MODES = {
    "Auto": "auto",
    "Food-size prior": "class_prior",
    "Metric reference": "metric_reference",
}


@st.cache_resource(show_spinner="Loading models (first run downloads weights)…")
def _get_pipeline_for(chessboard_square_cm: float) -> FoodVolumePipeline:
    return FoodVolumePipeline(chessboard_square_cm=chessboard_square_cm)

def get_pipeline() -> FoodVolumePipeline:
    return _get_pipeline_for(chessboard_cm)


def _read_upload(uploaded) -> np.ndarray:
    """Decode an uploaded image to a BGR array."""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _overlay(top_bgr: np.ndarray, result: PlateEstimate) -> np.ndarray:
    """Draw each item's mask outline + label."""
    vis = top_bgr.copy()
    for idx, item in enumerate(result.items):
        color = _PALETTE[idx % len(_PALETTE)]
        mask = item.mask.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)
        cxf, cyf = map(int, item.mask.centroid)
        label = f"{item.food_class} {item.mass_g:.0f}g"
        cv2.putText(vis, label, (cxf - 40, cyf), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, label, (cxf - 40, cyf), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# --- Sidebar controls ----------------------------------------------------------
st.sidebar.header("Settings")
segmentation_preset = st.sidebar.segmented_control(
    "Detection detail",
    options=["Conservative", "Balanced", "Sensitive"],
    default="Balanced",
    help="Sensitive keeps smaller/subtler regions; Conservative suppresses clutter.",
)
food_filter_label = st.sidebar.segmented_control(
    "Food filter",
    options=list(FOOD_FILTERS),
    default="Balanced",
    help="Tolerant keeps more candidates; Strict removes more uncertain regions.",
)
scale_label = st.sidebar.selectbox(
    "Scale",
    options=list(SCALE_MODES),
    index=0,
    help="Auto uses a metric reference if visible, otherwise food-size priors.",
)
chessboard_cm = st.sidebar.number_input(
    "Reference square (cm)", min_value=0.5, max_value=10.0,
    value=2.0, step=0.1,
    help="Used only when a chessboard/reference grid is visible.",
)
st.sidebar.caption(f"Compute device: **{config.get_device()}**")
st.sidebar.caption(
    "Use **Sensitive** for small or low-contrast food, **Strict** when the table or "
    "background creates false positives. **Auto** scale works with or without a visible reference."
)

# --- Header --------------------------------------------------------------------
st.title("🍽️ Food Volume & Calorie Estimator")
st.write(
    "Upload a top-down photo of your food, optionally with a side photo for "
    "height. The app estimates **volume, mass, calories and macros**."
)

# --- Input ---------------------------------------------------------------------
top_bgr: np.ndarray | None = None
side_bgr: np.ndarray | None = None

c1, c2 = st.columns(2)
top_file = c1.file_uploader("Top-down photo (required)", type=["jpg", "jpeg", "png"])
side_file = c2.file_uploader("Side photo (optional)", type=["jpg", "jpeg", "png"])
if top_file is not None:
    top_bgr = _read_upload(top_file)
if side_file is not None:
    side_bgr = _read_upload(side_file)

# --- Run -----------------------------------------------------------------------
if top_bgr is not None and st.button("Estimate", type="primary"):
    pipe = get_pipeline()
    with st.spinner("Analysing image…"):
        result = pipe.estimate(
            top_bgr,
            side_image=side_bgr,
            min_confidence=FOOD_FILTERS[food_filter_label],
            segmentation_preset=segmentation_preset.lower(),
            scale_mode=SCALE_MODES[scale_label],
        )

    if not result.items:
        st.warning("No food recognised in the image. Try a clearer photo.")
        for note in result.notes:
            st.caption(note)
        st.stop()

    # Headline totals: energy + mass + macros.
    a, b, c = st.columns(3)
    a.metric("Total calories", f"{result.total_kcal:.0f} kcal")
    b.metric("Total mass", f"{result.total_mass_g:.0f} g")
    c.metric("Items detected", str(len(result.items)))
    p, k, f = st.columns(3)
    p.metric("Protein", f"{result.total_protein_g:.0f} g")
    k.metric("Carbs", f"{result.total_carbs_g:.0f} g")
    f.metric("Fat", f"{result.total_fat_g:.0f} g")

    st.image(_overlay(top_bgr, result), caption="Detected items", use_container_width=True)

    # Per-item breakdown.
    def _alts(it):
        return ", ".join(f"{lbl} {sc:.0%}" for lbl, sc in it.alternatives) or "—"
    def _range(it):
        lo, hi = it.mass_range_g
        return f"{lo:.0f}–{hi:.0f} g" if lo and hi else "—"
    table = pd.DataFrame([{
        "Food": it.food_class,
        "Class conf": f"{it.confidence:.0%}",
        "Mass conf": f"{it.quantity_confidence:.0%}",
        "Also considered": _alts(it),
        "Area (cm²)": round(it.area_cm2, 1),
        "Height (cm)": None if np.isnan(it.height_cm) else round(it.height_cm, 1),
        "Volume (mL)": round(it.volume_ml, 0),
        "Mass (g)": round(it.mass_g, 0),
        "Plausible range": _range(it),
        "Scale": it.scale_source,
        "Quantity source": it.mass_source,
        "Calories (kcal)": round(it.nutrition.kcal, 0),
        "Protein (g)": round(it.nutrition.protein_g, 1),
        "Carbs (g)": round(it.nutrition.carbs_g, 1),
        "Fat (g)": round(it.nutrition.fat_g, 1),
    } for it in result.items])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Transparency: where the numbers came from and their caveats.
    if result.items:
        scales = ", ".join(f"{it.food_class}: {it.cm_per_px:.4f} cm/px ({it.scale_source})"
                           for it in result.items)
        st.caption(f"Per-item scale (self-calibrated): {scales}")
    for note in result.notes:
        st.caption("ℹ️ " + note)
    if any(it.nutrition.is_default for it in result.items):
        st.caption("⚠️ Some items used a generic nutrition fallback (class not in the table).")
    st.info(
        "Estimates are approximate. With a usable side photo, quantity comes from the "
        "trained ECUSTFD volume model; without one, it falls back to per-class area-to-mass "
        "priors. Target-domain weighed photos are still the main path to better accuracy."
    )
elif top_bgr is None:
    st.info("⬆️ Provide a top-down photo to begin.")
