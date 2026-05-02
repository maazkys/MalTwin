# modules/dashboard/app.py
"""
Streamlit application entry point.

Run:
    streamlit run modules/dashboard/app.py --server.port 8501

Responsibilities:
    1. Page configuration  (st.set_page_config — MUST be first Streamlit call)
    2. Session state init
    3. Database init
    4. Model + class names loading (once per session)
    5. Sidebar navigation
    6. Page routing
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from pathlib import Path

import config
from modules.dashboard import state
from modules.dashboard.db import init_db
from modules.dataset.preprocessor import load_class_names
from modules.detection.inference import load_model


def configure_page() -> None:
    """
    MUST be called before any other Streamlit command.
    st.set_page_config() is the very first Streamlit call in the app.
    """
    st.set_page_config(
        page_title=config.DASHBOARD_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': (
                "MalTwin — AI-based IIoT Malware Detection Framework\n"
                "COMSATS University, Islamabad | BS Cyber Security 2023-2027"
            ),
        },
    )


def load_global_resources() -> None:
    """
    Load class names and model into session_state on first run.
    Checks state BEFORE attempting to load — runs once per session only.
    Does NOT use @st.cache_resource; session_state guard is used instead.
    """
    # ── Class names ───────────────────────────────────────────────────────────
    if st.session_state[state.KEY_CLASS_NAMES] is None:
        try:
            class_names = load_class_names(config.CLASS_NAMES_PATH)
            st.session_state[state.KEY_CLASS_NAMES] = class_names
        except FileNotFoundError:
            st.session_state[state.KEY_CLASS_NAMES] = None

    # ── Model ─────────────────────────────────────────────────────────────────
    if (
        st.session_state[state.KEY_MODEL] is None
        and st.session_state[state.KEY_CLASS_NAMES] is not None
    ):
        try:
            with st.spinner("Loading detection model…"):
                num_classes = config.MALIMG_EXPECTED_FAMILIES
                model = load_model(config.BEST_MODEL_PATH, num_classes, config.DEVICE)
                st.session_state[state.KEY_MODEL]        = model
                st.session_state[state.KEY_MODEL_LOADED] = True
                st.session_state[state.KEY_DEVICE_INFO]  = str(config.DEVICE)
        except FileNotFoundError:
            st.session_state[state.KEY_MODEL_LOADED] = False


def render_sidebar() -> str:
    """
    Render sidebar navigation with availability indicators.
    Returns the selected page label string.
    Greyed-out pages are shown with ⚠️ prefix in the selectbox.
    """
    st.sidebar.markdown("# 🛡️ MalTwin")
    st.sidebar.markdown("*IIoT Malware Detection*")
    st.sidebar.divider()

    # Determine which pages are available
    model_ready   = state.is_model_loaded()
    file_ready    = state.has_uploaded_file()
    dataset_ready = config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir())

    # Build options with availability markers
    # Format: (display_label, internal_key, is_available)
    nav_options = [
        ("🏠 Dashboard",        "🏠 Dashboard",        True),
        ("📂 Binary Upload",    "📂 Binary Upload",    True),
        (
            "🔍 Malware Detection" if (model_ready and file_ready)
            else "🔍 Malware Detection ⚠️",
            "🔍 Malware Detection",
            True,     # always selectable — shows its own guard message
        ),
        (
            "🖼️ Dataset Gallery" if dataset_ready
            else "🖼️ Dataset Gallery ⚠️",
            "🖼️ Dataset Gallery",
            True,     # always selectable — gallery shows its own info message
        ),
        ("🖥️ Digital Twin",    "🖥️ Digital Twin",     True),
    ]

    display_labels  = [opt[0] for opt in nav_options]
    internal_keys   = [opt[1] for opt in nav_options]

    selected_display = st.sidebar.radio(
        "Navigation",
        options=display_labels,
        label_visibility="hidden",
    )

    # Map display label back to internal key (strips ⚠️ suffix)
    selected_index = display_labels.index(selected_display)
    page = internal_keys[selected_index]

    # ── System status panel ───────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**System Status**")

    if state.is_model_loaded():
        device_info = st.session_state.get(state.KEY_DEVICE_INFO, 'unknown')
        st.sidebar.success(f"✅ Model Ready ({device_info})")
    else:
        st.sidebar.warning("⚠️ No model loaded")
        st.sidebar.caption("Run `scripts/train.py` first")

    if state.has_uploaded_file():
        meta = st.session_state[state.KEY_FILE_META]
        st.sidebar.info(f"📄 {meta['name']}")
    else:
        st.sidebar.caption("No file uploaded")

    if state.has_detection_result():
        result = st.session_state[state.KEY_DETECTION]
        st.sidebar.success(f"🎯 {result['predicted_family']}")

    # ── Module health summary (compact) ──────────────────────────────────────
    st.sidebar.divider()
    try:
        from modules.dashboard.health import get_all_module_statuses
        statuses  = get_all_module_statuses()
        n_active  = sum(1 for s in statuses if s['status'] == 'active')
        n_total   = len(statuses)
        st.sidebar.caption(f"Modules: {n_active}/{n_total} active")
    except Exception:
        pass

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.caption("COMSATS University, Islamabad")
    st.sidebar.caption("BS Cyber Security 2023-2027")

    return page


def main() -> None:
    """
    Application entry point.
    configure_page() MUST be the first call — st.set_page_config() inside it
    must precede every other Streamlit call.
    """
    configure_page()                        # ← st.set_page_config() happens here
    state.init_session_state()
    init_db(config.DB_PATH)
    load_global_resources()
    page = render_sidebar()

    if page == "🏠 Dashboard":
        from modules.dashboard.pages.home import render
        render()
    elif page == "📂 Binary Upload":
        from modules.dashboard.pages.upload import render
        render()
    elif page == "🔍 Malware Detection":
        from modules.dashboard.pages.detection import render
        render()
    elif page == "🖼️ Dataset Gallery":
        from modules.dashboard.pages.gallery import render
        render()
    elif page == "🖥️ Digital Twin":
        from modules.dashboard.pages.digital_twin import render
        render()


if __name__ == "__main__":
    main()
