# modules/dashboard/state.py
"""
Centralised session_state key definitions and helper functions.

All pages import from here and use these constants exclusively.
Never use raw string literals for session_state keys in page files.
"""
import streamlit as st

# ── Session state key constants ───────────────────────────────────────────────
KEY_MODEL        = 'model'             # MalTwinCNN or None
KEY_CLASS_NAMES  = 'class_names'       # list[str] or None
KEY_IMG_ARRAY    = 'img_array'         # np.ndarray (128,128) uint8 or None
KEY_FILE_META    = 'file_meta'         # dict from get_file_metadata() or None
KEY_DETECTION    = 'detection_result'  # dict from predict_single() or None
KEY_MODEL_LOADED = 'model_loaded'      # bool
KEY_DEVICE_INFO  = 'device_info'       # str e.g. "cuda:0" or "cpu"
KEY_APP_START_TIME = 'app_start_time'   # datetime — set once at first run
KEY_HEATMAP      = 'gradcam_heatmap'   # dict from generate_gradcam() or None


def init_session_state() -> None:
    """
    Initialise all session state keys with default values if not already set.
    Call once at the top of app.py before any page renders.
    """
    defaults = {
        KEY_MODEL:        None,
        KEY_CLASS_NAMES:  None,
        KEY_IMG_ARRAY:    None,
        KEY_FILE_META:    None,
        KEY_DETECTION:    None,
        KEY_MODEL_LOADED: False,
        KEY_DEVICE_INFO:  'unknown',
        KEY_APP_START_TIME: None,
        KEY_HEATMAP:      None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Set start time once — only on very first run of the session
    if st.session_state[KEY_APP_START_TIME] is None:
        from datetime import datetime
        st.session_state[KEY_APP_START_TIME] = datetime.utcnow()


def clear_file_state() -> None:
    """
    Clear file-related keys. Call when a new file is uploaded
    to prevent stale detection results appearing for a different file.
    """
    st.session_state[KEY_IMG_ARRAY] = None
    st.session_state[KEY_FILE_META] = None
    st.session_state[KEY_DETECTION] = None
    st.session_state[KEY_HEATMAP] = None


def has_uploaded_file() -> bool:
    return st.session_state.get(KEY_IMG_ARRAY) is not None


def has_detection_result() -> bool:
    return st.session_state.get(KEY_DETECTION) is not None


def is_model_loaded() -> bool:
    return st.session_state.get(KEY_MODEL_LOADED, False)


def has_heatmap() -> bool:
    return st.session_state.get(KEY_HEATMAP) is not None
