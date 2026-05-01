"""
Dataset Gallery page — per-family sample image grid.

SRS refs: M6 FE-5
Displays sample grayscale images from each of the 25 Malimg malware families
directly from config.DATA_DIR. No PyTorch or DataLoader involved.

Layout:
    - Sidebar family selector (radio or selectbox)
    - Main area: family info header + NxM image grid
    - Grid columns configurable via a slider (default 4)
    - Max images per family configurable (default 12)
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

import config


@st.cache_data(ttl=300, show_spinner=False)
def _load_family_names(data_dir: str) -> list[str]:
    """
    Scan data_dir for subdirectory names (one per family).
    Cached for 5 minutes — avoids repeated disk scans.
    Returns sorted list of family name strings.
    Returns [] if data_dir does not exist.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


@st.cache_data(ttl=300, show_spinner=False)
def _load_sample_images(
    data_dir: str,
    family_name: str,
    max_images: int = 12,
) -> list[np.ndarray]:
    """
    Load up to max_images PNG images from data_dir/family_name/.
    Returns list of uint8 numpy arrays shape (H, W).
    Returns [] if directory is missing or has no PNGs.
    Cached per (data_dir, family_name, max_images).
    """
    family_dir = Path(data_dir) / family_name
    if not family_dir.exists():
        return []

    png_files = sorted(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))
    png_files = png_files[:max_images]

    images = []
    for path in png_files:
        try:
            img = Image.open(path).convert('L')
            images.append(np.array(img))
        except Exception:
            continue
    return images


@st.cache_data(ttl=300, show_spinner=False)
def _count_family_images(data_dir: str, family_name: str) -> int:
    """Return total PNG count for a family (not capped at max_images)."""
    family_dir = Path(data_dir) / family_name
    if not family_dir.exists():
        return 0
    return len(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))


def render():
    st.title("🖼️ Dataset Gallery")
    st.markdown(
        "Browse sample grayscale images from each malware family in the Malimg dataset. "
        "Each image is the raw byte structure of a malware binary visualised as a "
        "128×128 grayscale image. Distinctive textures and patterns are visible per family."
    )
    st.markdown("---")

    # ── Dataset availability check ────────────────────────────────────────────
    family_names = _load_family_names(str(config.DATA_DIR))

    if not family_names:
        st.info(
            "📂 Dataset not found. "
            f"The Malimg dataset was not detected at `{config.DATA_DIR}`. "
            "Download it from Kaggle and extract it so that each malware family "
            "has its own subfolder under `data/malimg/`."
        )
        st.markdown("**Expected structure:**")
        st.code(
            "data/malimg/\n"
            "├── Allaple.A/\n"
            "│   ├── 00a5c6a6.png\n"
            "│   └── ...\n"
            "├── Rbot!gen/\n"
            "└── ...",
            language=None,
        )
        return

    # ── Controls sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🖼️ Gallery Controls")
        st.divider()

        selected_family = st.selectbox(
            "Malware Family",
            options=family_names,
            index=0,
            help="Select a malware family to view sample images.",
        )

        max_images = st.slider(
            "Max images to show",
            min_value=4,
            max_value=24,
            value=12,
            step=4,
            help="Maximum number of sample images displayed per family.",
        )

        n_cols = st.slider(
            "Grid columns",
            min_value=2,
            max_value=6,
            value=4,
            help="Number of columns in the image grid.",
        )

        st.divider()
        st.markdown(f"**{len(family_names)} families** in dataset")

    # ── Family overview strip (all families mini-preview) ─────────────────────
    st.subheader("All Families — Quick Overview")
    st.caption(
        "One representative image per family. Click the family selector in the sidebar "
        "for a full grid."
    )

    _render_overview_strip(family_names, str(config.DATA_DIR))

    st.markdown("---")

    # ── Selected family detail grid ───────────────────────────────────────────
    total_count = _count_family_images(str(config.DATA_DIR), selected_family)
    st.subheader(f"📁 {selected_family}")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Samples", total_count)
    col_info2.metric("Showing", min(max_images, total_count))
    col_info3.metric("Image Size", "128 × 128 px")

    try:
        from modules.reporting.mitre_mapper import get_mitre_mapping
        mitre = get_mitre_mapping(selected_family)
        if mitre['found']:
            with st.expander("MITRE ATT&CK for ICS Context", expanded=False):
                st.markdown(f"**Description:** {mitre['description']}")
                st.markdown(f"**Tactics:** {', '.join(mitre['tactics'])}")
                for tech in mitre['techniques']:
                    st.markdown(f"- `{tech['id']}` — {tech['name']}")
    except Exception:
        pass

    st.markdown("---")

    with st.spinner(f"Loading {min(max_images, total_count)} images…"):
        images = _load_sample_images(str(config.DATA_DIR), selected_family, max_images)

    if not images:
        st.warning(f"No PNG images found in `data/malimg/{selected_family}/`.")
        return

    _render_image_grid(images, selected_family, n_cols)


def _render_overview_strip(family_names: list[str], data_dir: str) -> None:
    """
    Render one representative image per family in a horizontal scrolling strip.
    Uses 8 columns per row, wraps to next row automatically.
    """
    STRIP_COLS = 8
    chunks = [
        family_names[i:i + STRIP_COLS]
        for i in range(0, len(family_names), STRIP_COLS)
    ]

    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, family in zip(cols, chunk):
            images = _load_sample_images(data_dir, family, max_images=1)
            with col:
                if images:
                    st.image(
                        images[0],
                        caption=family,
                        use_column_width=True,
                        clamp=True,
                    )
                else:
                    st.caption(family)
                    st.markdown("_(no image)_")


def _render_image_grid(
    images: list[np.ndarray],
    family_name: str,
    n_cols: int,
) -> None:
    """
    Render a grid of grayscale images with filenames as captions.
    Fills rows left-to-right, wraps at n_cols.
    """
    for row_start in range(0, len(images), n_cols):
        row_images = images[row_start:row_start + n_cols]
        cols = st.columns(n_cols)
        for idx, (col, img_array) in enumerate(zip(cols, row_images)):
            with col:
                st.image(
                    img_array,
                    caption=f"{family_name} #{row_start + idx + 1}",
                    use_column_width=True,
                    clamp=True,
                )
