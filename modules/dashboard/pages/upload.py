# modules/dashboard/pages/upload.py
"""
Binary Upload & Visualization page.
Implements SRS Mockup M3 — Binary Upload and Visualization Screen.
SRS refs: FR3.1, FR3.2, FR3.3, FR3.4, UC-01
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

import config
from modules.binary_to_image.converter import BinaryConverter
from modules.binary_to_image.utils import (
    validate_binary_format,
    compute_pixel_histogram,
    get_file_metadata,
)
from modules.dashboard import state


def render():
    st.title("📂 Binary Upload & Visualization")
    st.markdown(
        "Upload a PE (.exe, .dll) or ELF binary file to convert it into a grayscale "
        "image for analysis. The image captures the structural byte patterns of the "
        "binary and is used as input to the malware detection model."
    )
    st.markdown("---")

    # ── File uploader ─────────────────────────────────────────────────────────
    # SRS ref: FR3.1
    uploaded_file = st.file_uploader(
        label="Upload Binary File",
        type=["exe", "dll", "elf"],
        help=(
            "Accepted formats: PE (.exe, .dll) or ELF (.elf) binaries. "
            "ELF binaries have no extension — rename to .elf if needed. "
            f"Maximum file size: {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        ),
        key="binary_uploader",
    )

    if uploaded_file is not None:
        _process_upload(uploaded_file)

    if state.has_uploaded_file():
        _render_results()
    elif uploaded_file is None:
        st.info("👆 Upload a binary file above to begin.")


def _process_upload(uploaded_file) -> None:
    """
    Reads, validates, and converts the uploaded file.
    Stores img_array and file_meta in session_state.
    Clears previous state before processing a new file.

    Error messages follow SRS USE-3 format:
        "Error: [what]. Cause: [why]. Action: [what to do]."
    """
    file_bytes = uploaded_file.read()

    # ── Size check ────────────────────────────────────────────────────────────
    if len(file_bytes) > config.MAX_UPLOAD_BYTES:
        size_mb = len(file_bytes) // (1024 * 1024)
        st.error(
            f"Error: File too large. "
            f"Cause: File exceeds the 50 MB limit (uploaded: {size_mb} MB). "
            "Action: Upload a smaller binary file."
        )
        return

    # Clear stale results from any previous upload
    state.clear_file_state()

    # ── Format validation ─────────────────────────────────────────────────────
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        st.error(
            "Error: Unsupported file format. "
            f"Cause: {e} "
            "Action: Upload a valid PE (.exe, .dll) or ELF binary file."
        )
        return

    # ── Conversion ────────────────────────────────────────────────────────────
    try:
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        img_array = converter.convert(file_bytes)
    except ValueError as e:
        st.error(
            "Error: Binary too small to convert. "
            f"Cause: {e} "
            "Action: Upload a valid binary file of at least 64 bytes."
        )
        return
    except Exception as e:
        st.error(
            "Error: Conversion failed. "
            f"Cause: {e} "
            "Action: Ensure the file is a valid PE or ELF binary and try again."
        )
        return

    # ── Store in session state ────────────────────────────────────────────────
    st.session_state[state.KEY_IMG_ARRAY] = img_array
    st.session_state[state.KEY_FILE_META] = get_file_metadata(
        file_bytes, uploaded_file.name, file_format
    )
    st.success(
        "✅ File processed successfully. "
        "Navigate to **Malware Detection** in the sidebar to analyze."
    )


def _render_results() -> None:
    """
    Display the grayscale image, metadata table, and pixel intensity histogram.
    Called when session_state has a processed image.
    """
    img_array = st.session_state[state.KEY_IMG_ARRAY]
    meta      = st.session_state[state.KEY_FILE_META]

    col_left, col_right = st.columns(2)

    # ── Left: Grayscale image ─────────────────────────────────────────────────
    with col_left:
        st.subheader("Grayscale Visualization")
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        png_bytes = converter.to_png_bytes(img_array)
        st.image(
            png_bytes,
            caption=f"Grayscale visualization ({config.IMG_SIZE}×{config.IMG_SIZE} pixels, 8-bit)",
            use_column_width=True,
        )

    # ── Right: Metadata + Histogram ───────────────────────────────────────────
    with col_right:
        st.subheader("File Metadata")
        meta_table = {
            "Property": ["File Name", "File Size", "Format", "SHA-256", "Upload Time"],
            "Value":    [
                meta['name'],
                meta['size_human'],
                meta['format'],
                meta['sha256'],
                meta['upload_time'],
            ],
        }
        st.table(meta_table)

        # SHA-256 in monospace for easy copying
        st.markdown("**SHA-256 (copy):**")
        st.code(meta['sha256'], language=None)

        # Pixel intensity histogram
        st.subheader("Pixel Intensity Distribution")
        hist = compute_pixel_histogram(img_array)
        fig = go.Figure(go.Bar(
            x=hist['bins'],
            y=hist['counts'],
            marker_color='#4A90D9',
            name='Byte frequency',
        ))
        fig.update_layout(
            title="Pixel Intensity Distribution (256 bins)",
            xaxis_title="Byte Value (0–255)",
            yaxis_title="Pixel Count",
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("➡️ Navigate to **Malware Detection** in the sidebar to run analysis.")
