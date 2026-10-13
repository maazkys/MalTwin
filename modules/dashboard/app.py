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
        page_title="MalTwin — IIoT Malware Detection",
        page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='12' fill='%23161B22'/><rect x='20' y='20' width='60' height='8' rx='2' fill='%23E8A020'/><rect x='20' y='36' width='42' height='8' rx='2' fill='%23388BFD'/><rect x='20' y='52' width='52' height='8' rx='2' fill='%233FB950'/><rect x='20' y='68' width='30' height='8' rx='2' fill='%238B949E'/></svg>",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "MalTwin v1.0 — COMSATS University Islamabad",
        },
    )


def _check_network_binding() -> None:
    """
    SRS SEC-5: warn if dashboard is accessible on a non-localhost interface.
    Streamlit exposes server address via its config at runtime.
    """
    try:
        from streamlit import runtime
        ctx = runtime.get_instance()
        if ctx is None:
            return
        server_addr = os.environ.get('STREAMLIT_SERVER_ADDRESS', '127.0.0.1')
        if server_addr not in ('localhost', '127.0.0.1', '::1'):
            st.warning(
                "Security Notice (SRS SEC-5): "
                "This dashboard is accessible on a non-localhost network interface "
                f"(`{server_addr}`). "
                "MalTwin is a research prototype and is not hardened for external exposure. "
                "Ensure your network environment is trusted before proceeding.",
            )
    except Exception:
        pass   # non-critical — never block startup


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

    # ── Staleness check ─────────────────────────────────────────────────────
    # If model is loaded but the file on disk has a newer mtime,
    # reset and reload. Handles CLI training completing mid-session.
    if st.session_state[state.KEY_MODEL_LOADED]:
        try:
            stored_mtime = st.session_state.get(state.KEY_MODEL_MTIME, 0)
            current_mtime = config.BEST_MODEL_PATH.stat().st_mtime
            if current_mtime > stored_mtime + 1:   # +1s tolerance
                # Model file changed — reset and reload
                st.session_state[state.KEY_MODEL]        = None
                st.session_state[state.KEY_MODEL_LOADED] = False
                st.session_state[state.KEY_MODEL_MTIME]  = 0
        except Exception:
            pass

    # ── Model load ──────────────────────────────────────────────────────────
    if (
        st.session_state[state.KEY_MODEL] is None
        and st.session_state[state.KEY_CLASS_NAMES] is not None
    ):
        try:
            with st.spinner("Loading detection model…"):
                num_classes = len(st.session_state[state.KEY_CLASS_NAMES])
                model = load_model(config.BEST_MODEL_PATH, num_classes, config.DEVICE)
                st.session_state[state.KEY_MODEL]        = model
                st.session_state[state.KEY_MODEL_LOADED] = True
                st.session_state[state.KEY_DEVICE_INFO]  = str(config.DEVICE)
                st.session_state[state.KEY_MODEL_MTIME]  = config.BEST_MODEL_PATH.stat().st_mtime
        except FileNotFoundError:
            st.session_state[state.KEY_MODEL_LOADED] = False


def render_sidebar() -> str:
    """
    Render sidebar navigation with availability indicators.
    Returns the selected page label string.
    """
    from modules.dashboard.theme import apply_theme, COLORS, status_badge

    apply_theme()

    # Sidebar wordmark
    st.sidebar.markdown(
        f'<div style="padding:20px 0 16px;">'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:18px;font-weight:600;'
        f'letter-spacing:0.04em;color:{COLORS["text_primary"]};">MalTwin</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;font-weight:400;'
        f'letter-spacing:0.08em;text-transform:uppercase;color:{COLORS["text_secondary"]};">'
        f'IIoT Malware Detection</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # Determine availability
    model_ready   = state.is_model_loaded()
    try:
        dataset_ready = config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir())
    except Exception:
        dataset_ready = False

    # Navigation labels are FIXED. Earlier versions appended "(no file)" /
    # "(no dataset)" to labels; Streamlit treats changed option labels as a new
    # widget, so the user's click was lost and the page jumped back after an
    # upload. Availability is shown in the System panel below and each page
    # guards itself with an explanatory message instead.
    nav_pages = [
        "Dashboard",
        "Binary Upload",
        "Malware Detection",
        "Dataset Gallery",
        "Dynamic Analysis",
        "Hybrid Verdict",
        "Model Training",
        "Digital Twin",
    ]
    page = st.sidebar.radio(
        "nav",
        options=nav_pages,
        key="nav_page",
        label_visibility="hidden",
    )

    st.sidebar.divider()

    # System status panel
    st.sidebar.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;font-weight:600;'
        f'letter-spacing:0.08em;text-transform:uppercase;'
        f'color:{COLORS["text_secondary"]};margin-bottom:10px;">System</div>',
        unsafe_allow_html=True,
    )

    model_status = status_badge('active', f'Model loaded ({st.session_state.get(state.KEY_DEVICE_INFO, "cpu")})') \
        if model_ready else status_badge('inactive', 'No model — run training')
    st.sidebar.markdown(model_status, unsafe_allow_html=True)
    st.sidebar.markdown(
        status_badge('active', 'Malimg dataset found') if dataset_ready
        else status_badge('inactive', 'No Malimg dataset (gallery/training)'),
        unsafe_allow_html=True,
    )
    if not state.has_uploaded_file():
        st.sidebar.markdown(status_badge('inactive', 'No binary uploaded'), unsafe_allow_html=True)

    if state.has_uploaded_file():
        meta = st.session_state[state.KEY_FILE_META]
        file_status = status_badge('active', f'{meta["name"]}')
        st.sidebar.markdown(file_status, unsafe_allow_html=True)

    if state.has_detection_result():
        result = st.session_state[state.KEY_DETECTION]
        det_status = status_badge('active', result['predicted_family'])
        st.sidebar.markdown(det_status, unsafe_allow_html=True)

    if state.is_training_running():
        st.sidebar.markdown(
            status_badge('inactive', 'Training in progress...'),
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    # Module health compact
    try:
        from modules.dashboard.health import get_all_module_statuses
        statuses  = get_all_module_statuses()
        n_active  = sum(1 for s in statuses if s['status'] == 'active')
        n_total   = len(statuses)
        st.sidebar.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_secondary"]};">'
            f'Modules: <span style="color:{COLORS["green"]};">{n_active}</span>/{n_total} active'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.sidebar.markdown(
        f'<div style="position:absolute;bottom:20px;left:16px;right:16px;">'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
        f'color:{COLORS["text_secondary"]};">COMSATS University Islamabad</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
        f'color:{COLORS["text_secondary"]};">BS Cyber Security 2023-2027</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    return page


def main() -> None:
    """
    Application entry point.
    configure_page() MUST be the first call — st.set_page_config() inside it
    must precede every other Streamlit call.
    """
    configure_page()                        # ← st.set_page_config() happens here
    _check_network_binding()
    state.init_session_state()
    init_db(config.DB_PATH)
    load_global_resources()
    page = render_sidebar()

    if page == "Dashboard":
        from modules.dashboard.pages.home import render
        render()
    elif page == "Binary Upload":
        from modules.dashboard.pages.upload import render
        render()
    elif page == "Malware Detection":
        from modules.dashboard.pages.detection import render
        render()
    elif page == "Dataset Gallery":
        from modules.dashboard.pages.gallery import render
        render()
    elif page == "Dynamic Analysis":
        from modules.dashboard.pages.dynamic import render
        render()
    elif page == "Hybrid Verdict":
        from modules.dashboard.pages.hybrid import render
        render()
    elif page == "Model Training":
        from modules.dashboard.pages.training import render
        render()
    elif page == "Digital Twin":
        from modules.dashboard.pages.digital_twin import render
        render()


if __name__ == "__main__":
    main()
