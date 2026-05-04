# modules/dashboard/pages/detection.py
"""
Malware Detection & Prediction page.
Implements SRS Mockup M5 — Malware Detection and Prediction Screen.
SRS refs: FR5.1, FR5.2, FR5.3, FR5.4, FR5.5, FR5.6, UC-02
"""
import json
import streamlit as st
import plotly.graph_objects as go

import config
from modules.detection.inference import predict_single
from modules.dashboard.db import log_detection_event
from modules.dashboard import state


def render():
    st.title("🔍 Malware Detection & Classification")
    st.markdown("---")

    # ── Guard: no file uploaded ───────────────────────────────────────────────
    if not state.has_uploaded_file():
        st.warning(
            "⚠️ No binary file loaded. "
            "Please upload a file on the **Binary Upload** page first."
        )
        return

    # ── Guard: no model loaded ────────────────────────────────────────────────
    if not state.is_model_loaded():
        st.warning(
            "⚠️ No trained model available. "
            "Run `python scripts/train.py` to train the model, then restart the dashboard."
        )
        return

    # ── File summary ──────────────────────────────────────────────────────────
    _render_file_summary()
    st.markdown("---")

    # ── Run Detection button ──────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "▶ Run Detection",
            type="primary",
            use_container_width=True,
            help=(
                "Run the CNN malware classifier on the uploaded binary image. "
                "The model classifies the binary into one of 25 known malware families."
            ),
        )

    if run_clicked:
        _run_detection()

    # ── Results ───────────────────────────────────────────────────────────────
    if state.has_detection_result():
        st.markdown("---")
        _render_results()


def _render_file_summary() -> None:
    """Compact summary: thumbnail | metadata | SHA-256."""
    meta      = st.session_state[state.KEY_FILE_META]
    img_array = st.session_state[state.KEY_IMG_ARRAY]

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        from modules.binary_to_image.converter import BinaryConverter
        png_bytes = BinaryConverter(img_size=config.IMG_SIZE).to_png_bytes(img_array)
        st.image(png_bytes, caption="Binary image", width=128)
    with col2:
        st.markdown(f"**File:** `{meta['name']}`")
        st.markdown(f"**Size:** {meta['size_human']}")
        st.markdown(f"**Format:** {meta['format']}")
    with col3:
        st.markdown("**SHA-256:**")
        st.code(meta['sha256'], language=None)


def _run_detection() -> None:
    """
    Execute inference, store result in session_state, log to SQLite.
    Catches all exceptions and displays them as st.error (never crashes the page).
    """
    try:
        with st.spinner("Running malware classification…"):
            model       = st.session_state[state.KEY_MODEL]
            class_names = st.session_state[state.KEY_CLASS_NAMES]
            img_array   = st.session_state[state.KEY_IMG_ARRAY]
            meta        = st.session_state[state.KEY_FILE_META]

            result = predict_single(model, img_array, class_names, config.DEVICE)
            st.session_state[state.KEY_DETECTION] = result

            # Log to SQLite — non-blocking (log_detection_event never raises)
            log_detection_event(
                db_path=config.DB_PATH,
                file_name=meta['name'],
                sha256=meta['sha256'],
                file_format=meta['format'],
                file_size=meta['size_bytes'],
                predicted_family=result['predicted_family'],
                confidence=result['confidence'],
                device_used=str(config.DEVICE),
            )
    except Exception as e:
        st.error(
            "Error: Detection failed. "
            f"Cause: {e}. "
            "Action: Ensure the model is correctly loaded and try again."
        )


def _run_gradcam() -> None:
    """
    Generate Grad-CAM heatmap for the current detection result.
    Stores result in session_state[KEY_HEATMAP].
    Never raises — generate_gradcam() handles all exceptions internally.
    """
    if state.has_heatmap():
        return  # Already generated for this detection — don't regenerate

    result = st.session_state.get(state.KEY_DETECTION)
    if result is None:
        st.warning("Run detection first before generating the heatmap.")
        return

    model = st.session_state[state.KEY_MODEL]
    class_names = st.session_state[state.KEY_CLASS_NAMES]
    img_array = st.session_state[state.KEY_IMG_ARRAY]
    target_class = class_names.index(result['predicted_family'])

    with st.spinner("Generating Grad-CAM heatmap… (this may take a few seconds)"):
        from modules.detection.gradcam import generate_gradcam

        heatmap_data = generate_gradcam(model, img_array, target_class, config.DEVICE)

    if heatmap_data is None:
        st.error(
            "Error: Heatmap generation failed. "
            "Cause: Captum backward pass error or incompatible model state. "
            "Action: Ensure the model was loaded correctly and try again."
        )
        return

    st.session_state[state.KEY_HEATMAP] = heatmap_data
    st.rerun()  # Rerun to render heatmap section below


def _render_heatmap() -> None:
    """
    Display the Grad-CAM heatmap results: overlay + standalone heatmap side by side.
    Also provides a download button for the overlay PNG (used in M8 reports).
    """
    heatmap_data = st.session_state[state.KEY_HEATMAP]
    result = st.session_state[state.KEY_DETECTION]

    col_overlay, col_heatmap = st.columns(2)

    with col_overlay:
        st.markdown("**Heatmap Overlay** — warm regions drove the prediction")
        st.image(
            heatmap_data['overlay_png'],
            caption=(
                f"Grad-CAM overlay for predicted class: "
                f"{result['predicted_family']} "
                f"({result['confidence']*100:.1f}% confidence)"
            ),
            use_container_width=True,
        )

    with col_heatmap:
        st.markdown("**Raw Attribution Map** — with colorbar")
        st.image(
            heatmap_data['heatmap_only_png'],
            caption="Jet colormap: red = high attribution, blue = low attribution",
            use_container_width=True,
        )

    # Interpretation text (SRS M7 FE-3)
    st.markdown("**Interpretation:**")
    st.info(
        "🔴 **Red/warm regions** — byte offsets in these areas of the binary "
        "were the strongest signals for the predicted classification. "
        "These regions often correspond to code sections with distinctive "
        "structural patterns (entry points, encrypted payloads, PE headers). "
        "\n\n"
        "🔵 **Blue/cool regions** — these byte regions had low influence "
        "on the classification decision."
    )

    # Export button for heatmap overlay (feeds into M8 PDF report)
    st.download_button(
        label="📥 Download Heatmap Overlay PNG",
        data=heatmap_data['overlay_png'],
        file_name=f"gradcam_{result['predicted_family'].replace('.', '_')}.png",
        mime="image/png",
        help="Download the Grad-CAM heatmap overlay for inclusion in forensic reports.",
    )


def _render_results() -> None:
    """
    Render the full detection results panel.
    Sections: A) prediction + confidence  B) top-3  C) probability chart
              D) MITRE mapping  E) XAI stub  F) report export
    """
    result     = st.session_state[state.KEY_DETECTION]
    confidence = result['confidence']
    family     = result['predicted_family']

    # ── A: Prediction + Confidence ────────────────────────────────────────────
    st.subheader("Detection Result")

    if confidence >= config.CONFIDENCE_GREEN:
        st.success(f"🎯 **{family}** detected with **{confidence * 100:.1f}%** confidence")
    elif confidence >= config.CONFIDENCE_AMBER:
        st.warning(
            f"⚠️ **{family}** detected with **{confidence * 100:.1f}%** confidence\n\n"
            "Low confidence — results may be unreliable. Manual verification recommended."
        )
    else:
        st.error(
            f"Error: Very low confidence for **{family}** detection "
            f"({confidence * 100:.1f}%). "
            f"Cause: Model confidence fell below the {config.CONFIDENCE_AMBER * 100:.0f}% threshold. "
            "Action: Perform a manual expert review before trusting this result."
        )

    confidence_pct = int(confidence * 100)
    color_label = (
        "🟢 High Confidence"   if confidence >= config.CONFIDENCE_GREEN
        else "🟡 Medium Confidence" if confidence >= config.CONFIDENCE_AMBER
        else "🔴 Low Confidence"
    )
    col_bar, col_label = st.columns([3, 1])
    col_bar.progress(confidence_pct)
    col_label.markdown(f"**{confidence_pct}%** {color_label}")

    # Benign class notice (architectural constraint — SRS_COMPLIANCE.md)
    st.caption(
        "ℹ️ MalTwin classifies binaries into known malware families only. "
        "There is no benign class — every uploaded binary receives a malware family prediction. "
        "Low confidence scores (<50%) indicate the binary may not match any trained family. "
        "This is a known architectural constraint of the Malimg-trained model."
    )

    # ── B: Top-3 Predictions ──────────────────────────────────────────────────
    st.markdown("**Top 3 Predictions:**")
    for i, pred in enumerate(result['top3'], 1):
        st.markdown(f"{i}. `{pred['family']}` — {pred['confidence'] * 100:.2f}%")

    st.markdown("---")

    # ── C: Per-Class Probability Chart ────────────────────────────────────────
    st.subheader("Class Probability Distribution")
    st.caption(
        "All malware families shown. Zero-probability classes display as empty bars.",
        help=(
            "The model outputs a probability for every known malware family. "
            "Higher bars indicate the model believes the binary is more likely "
            "to belong to that family."
        ),
    )
    _render_probability_chart(result['probabilities'])

    st.markdown("---")

    # ── D: MITRE ATT&CK for ICS Mapping ──────────────────────────────────────
    st.subheader("MITRE ATT&CK for ICS Mapping")
    st.caption(
        "Adversary tactics and techniques associated with the detected malware family.",
        help=(
            "MITRE ATT&CK for ICS is a knowledge base of adversary tactics and "
            "techniques specific to operational technology environments."
        ),
    )
    _render_mitre_mapping(family)

    st.markdown("---")

    # ── E: XAI Heatmap ────────────────────────────────────────────────────────
    st.subheader("Explainable AI — Grad-CAM Heatmap")
    st.caption(
        "Highlights which byte regions of the binary image drove the classification decision.",
        help=(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients "
            "flowing into the final convolutional layer to produce a heatmap. "
            "Red/warm regions contributed most strongly to the predicted class."
        ),
    )

    xai_requested = st.checkbox(
        "Generate Grad-CAM Heatmap",
        key="xai_checkbox",
        help="Generates a heatmap showing which byte regions drove the detection. "
             "Adds ~2–8 seconds depending on hardware.",
    )

    if xai_requested:
        _run_gradcam()

    if state.has_heatmap():
        _render_heatmap()

    # ── F: Forensic Report Export ─────────────────────────────────────────────
    st.subheader("Forensic Report")

    report_data = _build_report_data()

    col_pdf, col_json = st.columns(2)

    with col_pdf:
        if st.button(
            "📄 Generate PDF Report",
            type="secondary",
            use_container_width=True,
            help="Generate a PDF forensic report with MITRE ATT&CK mapping "
                 "and Grad-CAM heatmap (if generated).",
        ):
            with st.spinner("Generating PDF report…"):
                from modules.reporting.pdf_report import generate_pdf_report
                pdf_bytes = generate_pdf_report(report_data)

            if pdf_bytes is None:
                st.error(
                    "Error: PDF generation failed. "
                    "Cause: fpdf2 error or missing dependency. "
                    "Action: Use the JSON download instead."
                )
            else:
                # Log report generation (UC-03 postcondition)
                from modules.dashboard.db import log_report_event
                log_report_event(
                    config.DB_PATH,
                    detection_event_id=None,   # not tracked in session state
                    sha256=report_data['sha256'],
                    report_format='PDF',
                    gradcam_included=report_data['gradcam'].get('generated', False),
                )
                meta = st.session_state[state.KEY_FILE_META]
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"maltwin_report_{meta['sha256'][:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with col_json:
        from modules.reporting.json_report import generate_json_report
        from modules.dashboard.db import log_report_event
        json_bytes = generate_json_report(report_data)
        # Log JSON report generation (UC-03 postcondition)
        log_report_event(
            config.DB_PATH,
            detection_event_id=None,
            sha256=report_data['sha256'],
            report_format='JSON',
            gradcam_included=report_data['gradcam'].get('generated', False),
        )
        meta = st.session_state[state.KEY_FILE_META]
        st.download_button(
            label="📥 Download JSON Report",
            data=json_bytes,
            file_name=f"maltwin_report_{meta['sha256'][:8]}.json",
            mime="application/json",
            use_container_width=True,
            help="Full structured JSON report with MITRE ATT&CK mapping and detection metadata.",
        )

    if state.has_heatmap():
        st.caption(
            "✅ Grad-CAM heatmap will be embedded in the PDF report. "
            "Generate the heatmap above first if you haven't already."
        )
    else:
        st.caption(
            "💡 Tip: Generate the Grad-CAM heatmap above to embed it in the PDF report."
        )


def _build_report_data() -> dict:
    """
    Assemble the complete report_data dict from session state.
    Used by both PDF and JSON report generators.
    Called only when a detection result exists.
    """
    from modules.reporting.mitre_mapper import get_mitre_mapping

    result    = st.session_state[state.KEY_DETECTION]
    meta      = st.session_state[state.KEY_FILE_META]
    heatmap   = st.session_state.get(state.KEY_HEATMAP)

    mitre_data = get_mitre_mapping(result['predicted_family'])

    gradcam_info = {'generated': False}
    if heatmap is not None:
        gradcam_info = {
            'generated':         True,
            'target_class':      heatmap['target_class'],
            'layer':             heatmap['captum_layer'],
            'overlay_png_bytes': heatmap['overlay_png'],   # bytes for PDF embedding
        }

    return {
        'file_name':        meta['name'],
        'sha256':           meta['sha256'],
        'file_format':      meta['format'],
        'file_size_bytes':  meta['size_bytes'],
        'upload_time':      meta['upload_time'],
        'predicted_family': result['predicted_family'],
        'confidence':       float(result['confidence']),
        'top3':             result['top3'],
        'all_probabilities': result['probabilities'],
        'gradcam':          gradcam_info,
        'mitre':            mitre_data,
    }


def _render_probability_chart(probabilities: dict) -> None:
    """
    Horizontal bar chart of all class probabilities, sorted descending.
    Top-1 bar is red (#FF4B4B), all others are blue (#4A90D9).
    ALL classes shown including zero-probability ones (SRS FR5.3).
    """
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    families = [item[0] for item in sorted_probs]
    probs    = [item[1] for item in sorted_probs]
    colors   = ['#FF4B4B'] + ['#4A90D9'] * (len(families) - 1)

    fig = go.Figure(go.Bar(
        x=probs,
        y=families,
        orientation='h',
        marker_color=colors,
        text=[f"{p * 100:.2f}%" for p in probs],
        textposition='outside',
    ))
    fig.update_layout(
        title="Detection Probability per Malware Family",
        xaxis_title="Probability",
        yaxis_title="Malware Family",
        template="plotly_dark",
        height=max(400, len(families) * 22),
        xaxis=dict(range=[0, 1.05]),
        margin=dict(l=150, r=80, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_mitre_mapping(predicted_family: str) -> None:
    """
    Load MITRE ATT&CK for ICS mapping from JSON and display for the predicted family.
    Uses st.info (not st.error) when mapping is unavailable — never crashes the page.
    """
    try:
        with open(config.MITRE_JSON_PATH, 'r') as f:
            mitre_db = json.load(f)
    except FileNotFoundError:
        st.info(
            "MITRE ATT&CK mapping database not found. "
            "Ensure data/mitre_ics_mapping.json exists in the repo root."
        )
        return

    mapping = mitre_db.get(predicted_family)
    if not mapping:
        st.info(f"MITRE ATT&CK mapping not available for family: **{predicted_family}**")
        return

    tactics    = mapping.get('tactics', [])
    techniques = mapping.get('techniques', [])

    if tactics:
        st.markdown(f"**Tactics:** {', '.join(tactics)}")
    if techniques:
        st.markdown("**Techniques:**")
        for t in techniques:
            st.markdown(f"  - `{t['id']}` — {t['name']}")
