"""Streamlit demo: estimate food volume, mass, calories and macros from photos.

Run with::

    streamlit run app.py

The user provides a top-down photo and optionally a side photo. The app runs the
:class:`foodvol.pipeline.FoodVolumePipeline` and shows per-item mass / calories /
macros with an annotated overlay.
"""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - dependency is listed, but keep startup robust.
    register_heif_opener = None

if register_heif_opener is not None:
    register_heif_opener()

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

MAX_ANALYSIS_SIDE_PX = 1024
ANALYSIS_VERSION = "leaf-aware-mask-v3"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "heic", "heif"]


@st.cache_resource(show_spinner="Loading models (first run downloads weights)…")
def _get_pipeline(analysis_version: str) -> FoodVolumePipeline:
    # The version is a cache key. Measurement-logic changes must not reuse a
    # pipeline instance created from an older imported class definition.
    del analysis_version  # its value is intentionally used only by Streamlit's cache key
    return FoodVolumePipeline()

def get_pipeline() -> FoodVolumePipeline:
    return _get_pipeline(ANALYSIS_VERSION)


def _estimate_safely(
    pipe: FoodVolumePipeline,
    top_bgr: np.ndarray,
    side_bgr: np.ndarray | None,
    *,
    min_confidence: float,
    segmentation_preset: str,
    scale_mode: str,
) -> PlateEstimate:
    """Keep a failed model load from taking down the whole Streamlit session."""
    try:
        return pipe.estimate(
            top_bgr,
            side_image=side_bgr,
            min_confidence=min_confidence,
            segmentation_preset=segmentation_preset,
            scale_mode=scale_mode,
        )
    except Exception as exc:
        _get_pipeline.clear()
        st.error(
            "Analysis stopped safely and the model cache was reset. "
            f"Please retry once. ({type(exc).__name__}: {exc})"
        )
        return PlateEstimate(notes=["The pipeline stopped before producing an estimate."])


def _decode_with_pillow(raw: bytes) -> np.ndarray | None:
    """Fallback decoder for formats OpenCV cannot read, including HEIC/HEIF."""
    try:
        with Image.open(BytesIO(raw)) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb = np.array(pil_image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_upload(uploaded) -> np.ndarray:
    """Decode and bound phone-sized uploads before they reach the vision models."""
    raw = uploaded.getvalue()
    data = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        image = _decode_with_pillow(raw)
    if image is None:
        raise ValueError(
            "The uploaded image could not be decoded. Please upload a valid "
            "JPEG, PNG, HEIC or HEIF image."
        )
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > MAX_ANALYSIS_SIDE_PX:
        scale = MAX_ANALYSIS_SIDE_PX / longest_side
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


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


def _render_how_it_works() -> None:
    """Render a visual, implementation-accurate tour of the inference pipeline."""
    st.markdown(
        """
        <div class="pipeline-hero">
          <div class="pipeline-kicker">TRANSPARENT STATT MAGISCH</div>
          <h2>Vom Foto bis zur Nährwertschätzung</h2>
          <p>Die App trifft nicht eine einzige große Vermutung. Sie zerlegt das Bild,
          prüft jede Region, übersetzt Pixel in Zentimeter und berechnet daraus die Portion.</p>
          <div class="pipeline-pills">
            <span>📷 Draufsicht erforderlich</span>
            <span>↔️ Seitenfoto optional</span>
            <span>🧠 2 Bildmodelle</span>
            <span>📐 2 Mengenpfade</span>
          </div>
        </div>

        <div class="pipeline-flow">
          <div class="flow-card"><b>01</b><span class="flow-icon">📷</span><strong>Bild</strong><small>Pixel einlesen</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card"><b>02</b><span class="flow-icon">✂️</span><strong>Segmente</strong><small>FastSAM-Masken</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card"><b>03</b><span class="flow-icon">🔎</span><strong>Erkennen</strong><small>CLIP + Filter</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card"><b>04</b><span class="flow-icon">📏</span><strong>Skalieren</strong><small>Pixel → cm</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card"><b>05</b><span class="flow-icon">⚖️</span><strong>Portion</strong><small>mL und Gramm</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card"><b>06</b><span class="flow-icon">🥗</span><strong>Nährwerte</strong><small>kcal + Makros</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Der Weg durch die Bilderkennung")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
            <div class="detail-card accent-blue">
              <div class="detail-number">1</div>
              <div><h4>FastSAM findet zusammenhängende Objekte</h4>
              <p>Aus der Draufsicht entstehen zunächst mehrere binäre Masken – also Flächen,
              die jeweils ein mögliches Objekt markieren. Sehr kleine, sehr große und stark
              überlappende Masken werden entfernt.</p>
              <code>Bild → Maske → Pixelanzahl + Begrenzungsrahmen</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="detail-card accent-violet">
              <div class="detail-number">2</div>
              <div><h4>CLIP benennt jede Region</h4>
              <p>Jeder Bildausschnitt wird mit Textbeschreibungen aller bekannten Lebensmittel
              verglichen. Gleichzeitig treten Nicht-Essen-Begriffe wie Teller, Besteck,
              Tisch oder Schachbrett gegen sie an. Gewinnt ein Nicht-Essen-Begriff, fliegt
              die Region aus der Analyse.</p>
              <code>Ausschnitt ↔ „a photo of apple“ → Ähnlichkeit</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="detail-card accent-amber">
              <div class="detail-number">3</div>
              <div><h4>Die App prüft Alternativen</h4>
              <p>CLIP liefert neben Platz 1 weitere Kandidaten. Passt die errechnete Größe
              nicht zum plausiblen Gewichtsbereich der ersten Klasse, werden die Alternativen
              erneut geprüft. So kann etwa ein großer „Blueberry“-Treffer noch als Muffin
              erkannt werden.</p>
              <code>Top-1 unplausibel → Top-k prüfen → ggf. neu einstufen</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="detail-card accent-green">
              <div class="detail-number">4</div>
              <div><h4>Pixel bekommen eine reale Größe</h4>
              <p>Eine erkannte metrische Referenz liefert direkt Zentimeter pro Pixel. Fehlt
              sie, nutzt die App die typische lange Seite der erkannten Lebensmittelklasse
              als Größen-Prior. Dieser Schritt wird für jedes Objekt einzeln ausgeführt.</p>
              <code>Fläche cm² = Maskenpixel × (cm/Pixel)²</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Zwei Wege zur Portionsgröße")
    quantity_path = st.segmented_control(
        "Erklärpfad",
        options=["Mit Seitenfoto", "Nur Draufsicht"],
        default="Mit Seitenfoto",
        label_visibility="collapsed",
        key="pipeline_explanation_path",
    )
    if quantity_path == "Mit Seitenfoto":
        st.markdown(
            """
            <div class="route-card route-primary">
              <div class="route-badge">GENAUERER PFAD</div>
              <div class="route-line">
                <span><b>Top-Maske</b><small>Grundfläche in cm²</small></span><i>+</i>
                <span><b>Seiten-Maske</b><small>Höhe in cm</small></span><i>→</i>
                <span><b>Volumenmodell</b><small>area, height, area × height</small></span><i>→</i>
                <span><b>Volumen</b><small>Milliliter</small></span><i>→</i>
                <span><b>Dichte</b><small>g pro mL</small></span><i>→</i>
                <span><b>Masse</b><small>Gramm</small></span>
              </div>
              <p>Die Höhe ist die vertikale Ausdehnung der größten brauchbaren Seitenansicht-Maske.
              Ein trainierter Gradient-Boosting-Regressor schätzt aus den drei geometrischen
              Merkmalen das Volumen. Trainiert wurde dieser Teil mit ECUSTFD-Beispielen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="route-card route-fallback">
              <div class="route-badge">TRANSPARENTER FALLBACK</div>
              <div class="route-line compact">
                <span><b>Top-Maske</b><small>Grundfläche in cm²</small></span><i>×</i>
                <span><b>Klassen-Prior</b><small>typische g pro cm²</small></span><i>→</i>
                <span><b>Masse</b><small>Gramm</small></span>
              </div>
              <p>Ohne Höhe wäre ein echtes Volumen nicht belastbar. Deshalb behauptet die App
              hier kein gemessenes 3D-Volumen, sondern nutzt einen hinterlegten Flächen-zu-Masse-
              Wert der erkannten Klasse. Das Ergebnis bleibt eine Schätzung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    formula_a, formula_b, formula_c = st.columns(3)
    formula_a.markdown(
        """<div class="formula-card"><span>01 · Geometrie</span>
        <b>A = Pixel × Maßstab²</b><small>ergibt die Grundfläche in cm²</small></div>""",
        unsafe_allow_html=True,
    )
    formula_b.markdown(
        """<div class="formula-card"><span>02 · Masse</span>
        <b>m = Volumen × Dichte</b><small>oder Fläche × Klassen-Prior</small></div>""",
        unsafe_allow_html=True,
    )
    formula_c.markdown(
        """<div class="formula-card"><span>03 · Energie</span>
        <b>kcal = m / 100 × kcal₁₀₀</b><small>analog für Protein, Kohlenhydrate und Fett</small></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Was am Ende geprüft und angezeigt wird")
    checks = st.columns(3)
    checks[0].info(
        "**Plausibilitätsgrenzen**\n\nLiegt eine Portion außerhalb des hinterlegten "
        "Minimums oder Maximums, wird sie begrenzt und als solche vermerkt."
    )
    checks[1].info(
        "**Zwei Konfidenzen**\n\n*Class conf* bewertet die Erkennung. *Mass conf* bewertet "
        "zusätzlich Maßstab, Plausibilität und mögliche Neueinstufung."
    )
    checks[2].info(
        "**Nachvollziehbare Quellen**\n\nDie Ergebnistabelle zeigt Maßstab, Mengenquelle, "
        "Alternativklassen und den plausiblen Gewichtsbereich."
    )

    with st.expander("Modelle, Daten und ihre genaue Rolle"):
        st.markdown(
            """
            | Baustein | Aufgabe | Herkunft |
            |---|---|---|
            | **FastSAM** | erzeugt Objektmasken aus den Bildern | vortrainiertes Segmentierungsmodell |
            | **CLIP** | vergleicht Ausschnitte mit Food- und Non-Food-Texten | vortrainiertes Zero-Shot-Modell |
            | **Gradient Boosting** | lernt `Fläche + Höhe → Volumen` | auf ECUSTFD-Geometrie trainiert |
            | **Nährwerttabelle** | Dichte, kcal, Makros und Portions-Priors | lokale, kuratierte CSV-Tabelle |
            """
        )

    st.warning(
        "**Wichtig:** Das Ergebnis ist eine Schätzung, keine Messung. Unklare Perspektive, "
        "verdeckte Speisen, falsche Klassifikation und fehlende metrische Referenz können sich "
        "gegenseitig verstärken. Ein sauberes Top-Foto, eine sichtbare Referenz und ein "
        "Seitenfoto liefern den nachvollziehbarsten Pfad."
    )


def _trace_overlay(image_bgr: np.ndarray, traces, *, recognition: bool) -> np.ndarray:
    """Draw the actual candidate masks and decisions from one pipeline run."""
    vis = image_bgr.copy().astype(np.float32)
    for trace in traces:
        if recognition:
            if trace.kept:
                color = (70, 210, 80)       # green: final food region
            elif trace.passed_filter:
                color = (30, 170, 240)      # amber: overlapping duplicate
            else:
                color = (135, 135, 135)     # grey: rejected by recognition gate
        else:
            color = _PALETTE[(trace.index - 1) % len(_PALETTE)]

        shown_inst = (
            trace.measurement_mask
            if recognition and trace.kept and trace.measurement_mask is not None
            else trace.mask
        )
        mask = shown_inst.mask.astype(bool)
        color_arr = np.asarray(color, dtype=np.float32)
        vis[mask] = vis[mask] * 0.58 + color_arr * 0.42
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)

        x, y, _, _ = shown_inst.bbox
        if recognition:
            status = "KEEP" if trace.kept else "DROP"
            label = trace.final_label or trace.label
            text = f"{status}  {label}  {trace.score:.0%}"
        else:
            text = f"candidate {trace.index}"
        origin = (max(4, x), max(22, y - 7))
        cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 4)
        cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2)
    return cv2.cvtColor(np.clip(vis, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _scale_overlay(top_bgr: np.ndarray, result: PlateEstimate) -> np.ndarray:
    """Show the exact footprint mask; the line is calibration length, not area."""
    vis = top_bgr.copy().astype(np.float32)
    for idx, item in enumerate(result.items):
        color = _PALETTE[idx % len(_PALETTE)]
        x, y, w, h = item.mask.bbox
        mask = item.mask.mask.astype(bool)
        color_arr = np.asarray(color, dtype=np.float32)
        vis[mask] = vis[mask] * 0.55 + color_arr * 0.45
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)
        cx, cy = map(int, item.mask.centroid)
        if w >= h:
            p1, p2 = (x, cy), (x + w, cy)
        else:
            p1, p2 = (cx, y), (cx, y + h)
        cv2.line(vis, p1, p2, color, 3)
        text = f"{item.food_class}: {item.cm_per_px:.4f} cm/px | {item.area_cm2:.1f} cm2"
        origin = (max(4, x), max(22, y - 8))
        cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.cvtColor(np.clip(vis, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _side_height_overlay(side_bgr: np.ndarray, result: PlateEstimate) -> np.ndarray:
    """Show the exact side silhouette plus the width/height axes used."""
    vis = side_bgr.copy().astype(np.float32)
    if result.side_mask is None:
        return cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
    mask = result.side_mask.mask.astype(bool)
    color = np.asarray((70, 210, 80), dtype=np.float32)
    vis[mask] = vis[mask] * 0.58 + color * 0.42
    x, y, w, h = result.side_mask.bbox
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (70, 210, 80), 3)
    cx, cy = map(int, result.side_mask.centroid)
    cv2.line(vis, (cx, y), (cx, y + h), (70, 210, 80), 4)
    cv2.line(vis, (x, cy), (x + w, cy), (70, 210, 80), 3)
    text = f"silhouette: {w}px wide x {result.side_height_px:.0f}px high"
    origin = (max(4, x), max(22, y - 8))
    cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 210, 80), 2)
    return cv2.cvtColor(np.clip(vis, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _trace_decision(trace) -> str:
    if trace.kept:
        return "Kept as food"
    if trace.passed_filter:
        return f"Not used: {trace.reason or 'alternate food region'}"
    return f"Rejected: {trace.reason}"


def _render_pipeline_trace() -> None:
    """Expose every observable stage from the latest Estimate run."""
    st.markdown(
        """
        <div class="trace-hero">
          <div><span>LIVE PIPELINE TRACE</span><h2>See what the model does to your image</h2>
          <p>Run an estimate once. Every view below comes from that exact same inference —
          actual masks, actual CLIP decisions and the exact measurements used for the result.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    payload = st.session_state.get("latest_estimate")
    if payload is not None and payload.get("analysis_version") != ANALYSIS_VERSION:
        st.session_state.pop("latest_estimate", None)
        payload = None
    if payload is None:
        st.info("Go to **Estimate**, upload a top-down image and click **Estimate**. Its complete trace will appear here.")
        return

    top_bgr = payload["top"]
    side_bgr = payload["side"]
    result = payload["result"]
    st.caption(f"Showing the exact saved inference for **{payload['filename']}**. Run Estimate again to replace it.")

    stage_tabs = st.tabs([
        "1 · Input",
        "2 · Segmentation",
        "3 · Recognition",
        "4 · Volume reconstruction",
        "5 · Result",
    ])

    with stage_tabs[0]:
        st.markdown("#### The pixels the pipeline received")
        input_cols = st.columns(2) if side_bgr is not None else [st.container()]
        input_cols[0].image(
            cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Top view · {top_bgr.shape[1]} × {top_bgr.shape[0]} px",
            use_container_width=True,
        )
        if side_bgr is not None:
            input_cols[1].image(
                cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Side view · {side_bgr.shape[1]} × {side_bgr.shape[0]} px",
                use_container_width=True,
            )
        st.caption("This is the performance-sized input used by the pipeline (maximum side: 1024 px).")

    with stage_tabs[1]:
        st.markdown("#### FastSAM candidate masks in both views")
        side_traces = getattr(result, "side_trace_regions", [])
        if side_bgr is not None:
            segmentation_cols = st.columns(2)
            segmentation_cols[0].image(
                _trace_overlay(top_bgr, result.trace_regions, recognition=False),
                caption=f"Top view · {len(result.trace_regions)} candidate region(s)",
                use_container_width=True,
            )
            segmentation_cols[1].image(
                _trace_overlay(side_bgr, side_traces, recognition=False),
                caption=f"Side view · {len(side_traces)} candidate region(s)",
                use_container_width=True,
            )
        else:
            st.image(
                _trace_overlay(top_bgr, result.trace_regions, recognition=False),
                caption=f"Top view · {len(result.trace_regions)} candidate region(s)",
                use_container_width=True,
            )
        all_segment_rows = [
            ("Top", trace, top_bgr) for trace in result.trace_regions
        ] + [
            ("Side", trace, side_bgr) for trace in side_traces if side_bgr is not None
        ]
        if all_segment_rows:
            segment_table = pd.DataFrame([{
                "View": view,
                "Candidate": trace.index,
                "Mask pixels": trace.mask.area_px,
                "Image share": f"{trace.mask.area_px / (image.shape[0] * image.shape[1]):.1%}",
                "Bounding box (x, y, w, h)": str(trace.mask.bbox),
            } for view, trace, image in all_segment_rows])
            st.dataframe(segment_table, use_container_width=True, hide_index=True)
        else:
            st.warning("FastSAM did not produce a usable candidate with the current detection-detail setting.")

    with stage_tabs[2]:
        st.markdown("#### The same CLIP food / non-food gate in both views")
        st.markdown(
            '<div class="trace-legend"><span class="keep">● kept as food</span>'
            '<span class="duplicate">● overlapping duplicate</span>'
            '<span class="drop">● rejected by the food gate</span></div>',
            unsafe_allow_html=True,
        )
        if side_bgr is not None:
            recognition_cols = st.columns(2)
            recognition_cols[0].image(
                _trace_overlay(top_bgr, result.trace_regions, recognition=True),
                caption="Top: green mask continues to footprint measurement",
                use_container_width=True,
            )
            recognition_cols[1].image(
                _trace_overlay(side_bgr, side_traces, recognition=True),
                caption="Side: green mask continues to height measurement",
                use_container_width=True,
            )
        else:
            st.image(
                _trace_overlay(top_bgr, result.trace_regions, recognition=True),
                caption="Every candidate is classified; only green regions continue to measurement.",
                use_container_width=True,
            )
        all_recognition_rows = [("Top", trace) for trace in result.trace_regions] + [
            ("Side", trace) for trace in side_traces
        ]
        if all_recognition_rows:
            recognition_table = pd.DataFrame([{
                "View": view,
                "Candidate": trace.index,
                "Decision": _trace_decision(trace),
                "Best match": trace.final_label or trace.label,
                "CLIP score": f"{trace.score:.1%}",
                "Top alternatives": ", ".join(f"{label} {score:.1%}" for label, score in trace.alternatives[:4]),
                "Pixels removed before volume": trace.removed_px or 0,
            } for view, trace in all_recognition_rows])
            st.dataframe(recognition_table, use_container_width=True, hide_index=True)
        st.caption("Green is the exact cleaned mask used downstream. Rejected leaf/stem pixels are removed before area and volume are calculated; the table shows how many pixels changed.")

    with stage_tabs[3]:
        st.markdown("#### Exact footprint mask and pixel-to-centimetre conversion")
        if result.items:
            st.image(
                _scale_overlay(top_bgr, result),
                caption="The coloured shape is the exact mask used for area. The line only calibrates its physical length — the rectangle is not used as area.",
                use_container_width=True,
            )
            scale_table = pd.DataFrame([{
                "Food": item.food_class,
                "Mask pixels": item.mask.area_px,
                "Scale": f"{item.cm_per_px:.5f} cm/px",
                "Scale source": item.scale_source,
                "Area calculation": f"{item.mask.area_px} × {item.cm_per_px:.5f}²",
                "Area": f"{item.area_cm2:.2f} cm²",
            } for item in result.items])
            st.dataframe(scale_table, use_container_width=True, hide_index=True)
            evidence_rows = [{
                "Food": item.food_class,
                "Scale cue": evidence.source,
                "cm per pixel": round(evidence.cm_per_px, 6),
                "Confidence": f"{evidence.confidence:.0%}",
                "Used in fusion": "yes" if evidence.used else "no — inconsistent",
            } for item in result.items for evidence in getattr(item, "scale_evidence", [])]
            if evidence_rows:
                st.markdown("##### Automatic scale evidence")
                st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
            if result.chessboard_scale_cm_per_px > 0:
                st.success(f"Square reference detected: {result.chessboard_scale_cm_per_px:.5f} cm/px.")
            elif any("plate" in item.scale_source for item in result.items):
                st.success("A detected plate contributes physical scale evidence.")
            else:
                st.info("No square or plate reference was reliable. Scale falls back to recognised size and two-view geometry.")
            st.caption("Footprint area = number of coloured mask pixels × (cm per pixel)². Pixels outside the contour contribute nothing.")
        else:
            st.warning("No region survived recognition, so no physical area was calculated.")

    with stage_tabs[3]:
        st.markdown("#### Shape-aware volume from both silhouettes")
        if side_bgr is not None:
            if result.side_mask is not None:
                geometry_cols = st.columns(2)
                geometry_cols[0].image(
                    _scale_overlay(top_bgr, result),
                    caption="Top silhouette → depth profile d(x)",
                    use_container_width=True,
                )
                geometry_cols[1].image(
                    _side_height_overlay(side_bgr, result),
                    caption="Side silhouette → height profile h(x)",
                    use_container_width=True,
                )
            else:
                st.warning("A side image was supplied, but no usable side-view region was found. The area-based fallback was used.")
        elif result.items:
            st.info("No side image: the pipeline uses the per-class area-to-mass prior instead of claiming a measured 3D volume.")

        if result.items:
            reconstructed = [item for item in result.items if getattr(item, "two_view", None) is not None]
            if reconstructed:
                profile_item = reconstructed[0]
                profile = profile_item.two_view
                profile_data = pd.DataFrame({
                    "Position (cm)": profile.axis_cm,
                    "Top-view depth d(x)": profile.top_depth_cm,
                    "Side-view height h(x)": profile.side_height_cm,
                }).set_index("Position (cm)")
                st.line_chart(profile_data, height=260)
                geometry_metrics = st.columns(4)
                geometry_metrics[0].metric("Object length", f"{profile.length_cm:.2f} cm")
                geometry_metrics[1].metric("Measured height", f"{profile.height_cm:.2f} cm")
                geometry_metrics[2].metric("Side scale", f"{profile.side_cm_per_px:.4f} cm/px")
                geometry_metrics[3].metric("Integrated volume", f"{profile.volume_ml:.0f} mL")
                st.code(
                    "slice area(x) = π/4 × top depth d(x) × side height h(x)\n"
                    "volume = sum of all silhouette slice areas × slice width",
                    language=None,
                )
                if profile.scale_source == "side_width_matched":
                    st.info(
                        "The two photos may have different camera distances. The side view is therefore "
                        "scaled by matching its silhouette width to the physical top-view length."
                    )
            elif side_bgr is not None and result.side_mask is not None:
                st.warning("The two silhouettes could not be aligned; the trained area/height model was used as fallback.")

            quantity_table = pd.DataFrame([{
                "Food": item.food_class,
                "Area": f"{item.area_cm2:.2f} cm²",
                "Height": "not measured" if np.isnan(item.height_cm) else f"{item.height_cm:.2f} cm",
                "Quantity path": item.mass_source,
                "Geometry": "two silhouettes" if getattr(item, "two_view", None) is not None else "fallback",
                "Raw volume": f"{item.raw_volume_ml:.1f} mL",
                "Raw mass": f"{item.raw_mass_g:.1f} g",
                "Final mass": f"{item.mass_g:.1f} g",
                "Typical range": f"{item.mass_range_g[0]:.0f}–{item.mass_range_g[1]:.0f} g",
            } for item in result.items])
            st.dataframe(quantity_table, use_container_width=True, hide_index=True)
            st.caption("Only the dominant top-view food item is paired with an unlabelled side photo. Other items use the top-view fallback.")

    with stage_tabs[4]:
        st.markdown("#### Final overlay and nutrition lookup")
        if result.items:
            totals = st.columns(4)
            totals[0].metric("Items", len(result.items))
            totals[1].metric("Mass", f"{result.total_mass_g:.0f} g")
            totals[2].metric("Calories", f"{result.total_kcal:.0f} kcal")
            totals[3].metric("Protein", f"{result.total_protein_g:.1f} g")
            st.image(_overlay(top_bgr, result), caption="Final kept regions, labels and masses", use_container_width=True)
            final_table = pd.DataFrame([{
                "Food": item.food_class,
                "Mass": f"{item.mass_g:.1f} g",
                "Calories": f"{item.nutrition.kcal:.0f} kcal",
                "Protein": f"{item.nutrition.protein_g:.1f} g",
                "Carbs": f"{item.nutrition.carbs_g:.1f} g",
                "Fat": f"{item.nutrition.fat_g:.1f} g",
                "Mass confidence": f"{item.quantity_confidence:.0%}",
            } for item in result.items])
            st.dataframe(final_table, use_container_width=True, hide_index=True)
            st.caption("Nutrition formula: final mass / 100 × the class value per 100 g.")
        else:
            st.warning("This run produced no final food result. Inspect Segmentation and Recognition to see where it stopped.")
        for note in result.notes:
            st.caption("ℹ️ " + note)


st.markdown(
    """
    <style>
      .stTabs [data-baseweb="tab-list"] { gap: .55rem; }
      .stTabs [data-baseweb="tab"] {
        height: 3rem; padding: 0 1.15rem; border-radius: .85rem .85rem 0 0;
        font-weight: 700;
      }
      .pipeline-hero {
        margin: 1.1rem 0 1.5rem; padding: 2.2rem 2.4rem; border-radius: 1.5rem;
        color: #f8fafc;
        background: radial-gradient(circle at 85% 15%, rgba(74,222,128,.26), transparent 28%),
                    linear-gradient(135deg, #10243f 0%, #173f42 55%, #275437 100%);
        box-shadow: 0 18px 50px rgba(8, 29, 41, .20);
      }
      .pipeline-hero h2 { color: #fff; margin: .25rem 0 .55rem; font-size: clamp(1.75rem, 3vw, 2.65rem); }
      .pipeline-hero p { max-width: 52rem; color: #d9f4e3; font-size: 1.04rem; line-height: 1.65; margin: 0; }
      .pipeline-kicker { color: #86efac; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
      .pipeline-pills { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.35rem; }
      .pipeline-pills span {
        padding: .45rem .75rem; border-radius: 999px; background: rgba(255,255,255,.1);
        border: 1px solid rgba(255,255,255,.16); color: #f1f5f9; font-size: .82rem;
      }
      .pipeline-flow {
        display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
        align-items: center; gap: .38rem; margin: .35rem 0 2.35rem;
      }
      .flow-card {
        min-height: 8.1rem; display: flex; flex-direction: column; justify-content: center;
        padding: .8rem; border: 1px solid rgba(125,125,125,.22); border-radius: 1rem;
        background: color-mix(in srgb, var(--secondary-background-color) 78%, transparent);
      }
      .flow-card b { color: #16a34a; font-size: .7rem; letter-spacing: .12em; }
      .flow-card strong { margin-top: .2rem; font-size: .94rem; }
      .flow-card small { opacity: .68; margin-top: .18rem; line-height: 1.3; }
      .flow-icon { font-size: 1.45rem; margin-top: .35rem; }
      .flow-arrow { opacity: .35; font-size: 1.2rem; }
      .detail-card {
        min-height: 15.3rem; display: flex; gap: 1rem; margin-bottom: 1rem; padding: 1.25rem;
        border-radius: 1.1rem; border: 1px solid rgba(125,125,125,.2);
        background: color-mix(in srgb, var(--secondary-background-color) 70%, transparent);
      }
      .detail-card h4 { margin: .1rem 0 .55rem; font-size: 1.05rem; }
      .detail-card p { margin: 0 0 .8rem; opacity: .78; line-height: 1.55; font-size: .91rem; }
      .detail-card code { display: inline-block; white-space: normal; line-height: 1.45; font-size: .76rem; }
      .detail-number {
        flex: 0 0 2.2rem; width: 2.2rem; height: 2.2rem; display: grid; place-items: center;
        border-radius: .72rem; color: #fff; font-weight: 800;
      }
      .accent-blue .detail-number { background: #2563eb; }
      .accent-violet .detail-number { background: #7c3aed; }
      .accent-amber .detail-number { background: #d97706; }
      .accent-green .detail-number { background: #16a34a; }
      .route-card { margin: .9rem 0 1rem; padding: 1.45rem; border-radius: 1.2rem; border: 1px solid; }
      .route-primary { background: rgba(22,163,74,.07); border-color: rgba(22,163,74,.28); }
      .route-fallback { background: rgba(217,119,6,.07); border-color: rgba(217,119,6,.28); }
      .route-badge { font-size: .68rem; font-weight: 800; letter-spacing: .13em; color: #16a34a; margin-bottom: 1rem; }
      .route-fallback .route-badge { color: #d97706; }
      .route-line { display: flex; align-items: stretch; gap: .55rem; }
      .route-line span {
        flex: 1 1 0; min-width: 0; padding: .8rem; border-radius: .8rem;
        background: color-mix(in srgb, var(--background-color) 82%, transparent);
        border: 1px solid rgba(125,125,125,.18);
      }
      .route-line b, .route-line small { display: block; }
      .route-line b { font-size: .87rem; }
      .route-line small { margin-top: .25rem; opacity: .65; font-size: .72rem; line-height: 1.35; }
      .route-line i { align-self: center; opacity: .5; font-style: normal; }
      .route-card > p { margin: 1rem 0 0; opacity: .78; line-height: 1.55; font-size: .9rem; }
      .formula-card {
        min-height: 7.5rem; margin: .25rem 0 1.5rem; padding: 1rem; border-radius: 1rem;
        border: 1px solid rgba(125,125,125,.2);
        background: color-mix(in srgb, var(--secondary-background-color) 70%, transparent);
      }
      .formula-card span, .formula-card b, .formula-card small { display: block; }
      .formula-card span { color: #16a34a; font-size: .7rem; font-weight: 800; letter-spacing: .08em; }
      .formula-card b { margin: .45rem 0 .3rem; }
      .formula-card small { opacity: .65; line-height: 1.4; }
      @media (max-width: 900px) {
        .pipeline-flow { grid-template-columns: repeat(3, 1fr); }
        .flow-arrow { display: none; }
        .route-line { flex-direction: column; }
        .route-line i { transform: rotate(90deg); }
      }
      @media (max-width: 560px) {
        .pipeline-hero { padding: 1.5rem; }
        .pipeline-flow { grid-template-columns: repeat(2, 1fr); }
        .detail-card { min-height: auto; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
      .trace-hero {
        margin: 1.1rem 0 1.4rem; padding: 1.65rem 1.9rem; border-radius: 1.25rem;
        background: linear-gradient(125deg, rgba(37,99,235,.15), rgba(16,185,129,.12));
        border: 1px solid rgba(96,165,250,.28);
      }
      .trace-hero > div > span { color: #60a5fa; font-size: .7rem; font-weight: 800; letter-spacing: .14em; }
      .trace-hero h2 { margin: .25rem 0 .45rem; font-size: clamp(1.55rem, 3vw, 2.25rem); }
      .trace-hero p { max-width: 55rem; margin: 0; opacity: .75; line-height: 1.55; }
      .trace-legend { display: flex; flex-wrap: wrap; gap: .55rem; margin: .3rem 0 .8rem; }
      .trace-legend span {
        padding: .34rem .65rem; border-radius: 999px; border: 1px solid rgba(125,125,125,.2);
        background: color-mix(in srgb, var(--secondary-background-color) 75%, transparent);
        font-size: .78rem; font-weight: 700;
      }
      .trace-legend .keep { color: #4ade80; }
      .trace-legend .duplicate { color: #f59e0b; }
      .trace-legend .drop { color: #a3a3a3; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
st.sidebar.caption(f"Compute device: **{config.get_device()}**")
st.sidebar.caption("Performance guard: uploads are analysed at a maximum side length of **1024 px**.")
st.sidebar.caption(
    "Use **Sensitive** for small or low-contrast food, **Strict** when the table or "
    "background creates false positives. Scale is selected automatically from every available cue."
)

# --- Header --------------------------------------------------------------------
st.title("🍽️ Food Volume & Calorie Estimator")
st.write(
    "Upload a top-down photo of your food, optionally with a side photo for "
    "height. The app estimates **volume, mass, calories and macros**."
)

analyse_tab, trace_tab = st.tabs(["📷  Estimate", "🔬  Pipeline trace"])

with analyse_tab:
    # --- Input -----------------------------------------------------------------
    top_bgr: np.ndarray | None = None
    side_bgr: np.ndarray | None = None

    c1, c2 = st.columns(2)
    top_file = c1.file_uploader("Top-down photo (required)", type=SUPPORTED_UPLOAD_TYPES)
    side_file = c2.file_uploader("Side photo (optional)", type=SUPPORTED_UPLOAD_TYPES)
    if top_file is not None:
        try:
            top_bgr = _read_upload(top_file)
        except ValueError as exc:
            c1.error(str(exc))
    if side_file is not None:
        try:
            side_bgr = _read_upload(side_file)
        except ValueError as exc:
            c2.error(str(exc))

    # --- Run -------------------------------------------------------------------
    if top_bgr is not None and st.button("Estimate", type="primary"):
        pipe = get_pipeline()
        with st.spinner("Analysing image…"):
            result = _estimate_safely(
                pipe,
                top_bgr,
                side_bgr,
                min_confidence=FOOD_FILTERS[food_filter_label],
                segmentation_preset=segmentation_preset.lower(),
                scale_mode="auto",
            )
        st.session_state["latest_estimate"] = {
            "top": top_bgr,
            "side": side_bgr,
            "result": result,
            "filename": top_file.name,
            "analysis_version": ANALYSIS_VERSION,
        }

        if not result.items:
            st.warning("No food recognised in the image. Try a clearer photo.")
            for note in result.notes:
                st.caption(note)
        else:
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
                "Typical range": _range(it),
                "Scale": it.scale_source,
                "Quantity source": it.mass_source,
                "Calories (kcal)": round(it.nutrition.kcal, 0),
                "Protein (g)": round(it.nutrition.protein_g, 1),
                "Carbs (g)": round(it.nutrition.carbs_g, 1),
                "Fat (g)": round(it.nutrition.fat_g, 1),
            } for it in result.items])
            st.dataframe(table, use_container_width=True, hide_index=True)

            # Transparency: where the numbers came from and their caveats.
            scales = ", ".join(
                f"{it.food_class}: {it.cm_per_px:.4f} cm/px ({it.scale_source})"
                for it in result.items
            )
            st.caption(f"Per-item scale (self-calibrated): {scales}")
            for note in result.notes:
                st.caption("ℹ️ " + note)
            if any(it.nutrition.is_default for it in result.items):
                st.caption("⚠️ Some items used a generic nutrition fallback (class not in the table).")
            st.info(
                "Estimates are approximate. With a usable side photo, volume is reconstructed from "
                "the aligned top and side silhouettes; the trained ECUSTFD model remains a geometry "
                "fallback. Without a side view, the app uses per-class area-to-mass priors."
            )
    elif top_bgr is None:
        st.info("⬆️ Provide a top-down photo to begin.")

with trace_tab:
    _render_pipeline_trace()
