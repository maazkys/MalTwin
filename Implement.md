# MalTwin — Dashboard UI Redesign
### Improving visual quality within Streamlit's constraints

> This is a standalone design improvement pass on top of all previous steps.
> It does not change any logic, routing, or data — only visual presentation.
> Apply after the compliance fixes are complete.

---

## Design Direction

MalTwin is a **security analysis tool for researchers and SOC analysts**. The aesthetic should feel like professional security tooling — precise, data-dense, credible. Not a consumer app, not a generic Streamlit demo.

Chosen direction: **Dark industrial / terminal-adjacent**. Think Grafana meets a high-end threat intel platform. Dark navy background, monospaced accents for hashes and technical values, sharp geometric indicators for status, no gradients, no rounded-everything softness. Every color must carry meaning.

**Key decisions:**
- No emojis anywhere — replace with geometric SVG indicators or text labels
- Monospace font for all hashes, code values, confidence percentages
- A single accent color (amber `#E8A020`) used sparingly for high-priority information only
- Status uses shape + color, not emoji: filled circle for active, hollow for inactive, X for error
- Typography: `JetBrains Mono` for technical values, `DM Sans` for UI text (both available via Google Fonts injected through HTML)
- Sidebar gets a dark panel treatment — visually separated from the main content area

---

## File 1: `modules/dashboard/theme.py`

Create this new file. It is the single source of truth for all visual customisation. Every page imports and calls `apply_theme()` at the top of its `render()` function.

```python
# modules/dashboard/theme.py
"""
MalTwin dashboard visual theme.

Injects custom CSS into the Streamlit page via st.markdown(unsafe_allow_html=True).
Call apply_theme() at the top of every page render() function.

Design direction: dark industrial — precise, data-dense, credible.
No emojis. Monospace for technical values. Amber accent for priority info.
"""
import streamlit as st


# ── Color tokens ──────────────────────────────────────────────────────────────

COLORS = {
    'bg_primary':    '#0D1117',   # near-black background
    'bg_secondary':  '#161B22',   # slightly lighter — card surfaces
    'bg_tertiary':   '#21262D',   # hover states, borders
    'border':        '#30363D',   # subtle borders
    'border_strong': '#484F58',   # visible separators
    'text_primary':  '#E6EDF3',   # main text
    'text_secondary':'#8B949E',   # muted / labels
    'text_mono':     '#A5D6FF',   # monospace values (hashes, IDs)
    'accent':        '#E8A020',   # amber — used for active/important only
    'accent_dim':    '#7D5A00',   # dimmed amber for secondary accent use
    'green':         '#3FB950',   # success / active
    'amber':         '#D29922',   # warning / medium confidence
    'red':           '#F85149',   # error / low confidence
    'blue':          '#388BFD',   # informational
}


STATUS_INDICATORS = {
    'active':   f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{COLORS["green"]};margin-right:6px;"></span>',
    'inactive': f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;border:1.5px solid {COLORS["amber"]};margin-right:6px;"></span>',
    'error':    f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{COLORS["red"]};margin-right:6px;"></span>',
}


def status_badge(status: str, label: str) -> str:
    """Return an HTML status badge with geometric indicator (no emoji)."""
    indicator = STATUS_INDICATORS.get(status, STATUS_INDICATORS['inactive'])
    color = {
        'active':   COLORS['green'],
        'inactive': COLORS['amber'],
        'error':    COLORS['red'],
    }.get(status, COLORS['text_secondary'])
    return (
        f'<span style="display:inline-flex;align-items:center;'
        f'font-size:12px;color:{color};font-family:\'DM Sans\',sans-serif;">'
        f'{indicator}{label}</span>'
    )


def mono(value: str, color: str | None = None) -> str:
    """Wrap a value in monospace styling for technical display (hashes, IDs, etc)."""
    c = color or COLORS['text_mono']
    return (
        f'<code style="font-family:\'JetBrains Mono\',monospace;'
        f'font-size:12px;color:{c};background:{COLORS["bg_tertiary"]};'
        f'padding:2px 6px;border-radius:3px;border:1px solid {COLORS["border"]};">'
        f'{value}</code>'
    )


def section_header(title: str, subtitle: str = '') -> None:
    """Render a section header with optional subtitle. Replaces st.subheader()."""
    subtitle_html = (
        f'<p style="margin:2px 0 16px;font-size:13px;'
        f'color:{COLORS["text_secondary"]};font-family:\'DM Sans\',sans-serif;">'
        f'{subtitle}</p>'
        if subtitle else '<div style="margin-bottom:16px;"></div>'
    )
    st.markdown(
        f'<div style="margin-top:24px;">'
        f'<h3 style="margin:0;font-size:15px;font-weight:600;letter-spacing:0.04em;'
        f'text-transform:uppercase;color:{COLORS["text_secondary"]};'
        f'font-family:\'DM Sans\',sans-serif;">{title}</h3>'
        f'<div style="height:1px;background:{COLORS["border"]};margin:8px 0;"></div>'
        f'{subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def confidence_bar_html(confidence: float, family: str) -> str:
    """
    Render a confidence bar as HTML (replaces the default st.progress widget).
    Uses color-coded fill: green >= 0.80, amber >= 0.50, red < 0.50.
    """
    if confidence >= 0.80:
        fill_color = COLORS['green']
        label_color = COLORS['green']
        level = 'HIGH'
    elif confidence >= 0.50:
        fill_color = COLORS['amber']
        label_color = COLORS['amber']
        level = 'MEDIUM'
    else:
        fill_color = COLORS['red']
        label_color = COLORS['red']
        level = 'LOW'

    pct = confidence * 100
    return f"""
    <div style="margin:16px 0;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
        <span style="font-family:'DM Sans',sans-serif;font-size:13px;
                     color:{COLORS['text_secondary']};">Confidence</span>
        <div style="display:flex;align-items:baseline;gap:8px;">
          <span style="font-family:'JetBrains Mono',monospace;font-size:22px;
                       font-weight:700;color:{label_color};">{pct:.1f}%</span>
          <span style="font-family:'DM Sans',sans-serif;font-size:11px;
                       letter-spacing:0.08em;color:{label_color};opacity:0.7;">{level}</span>
        </div>
      </div>
      <div style="height:6px;background:{COLORS['bg_tertiary']};border-radius:3px;
                  border:1px solid {COLORS['border']};">
        <div style="height:100%;width:{pct:.1f}%;background:{fill_color};
                    border-radius:3px;transition:width 0.4s ease;"></div>
      </div>
    </div>
    """


def kpi_card(label: str, value: str, sub: str = '', accent: bool = False) -> str:
    """Return HTML for a KPI metric card."""
    value_color = COLORS['accent'] if accent else COLORS['text_primary']
    return f"""
    <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                border-radius:6px;padding:16px 20px;height:100%;">
      <div style="font-family:'DM Sans',sans-serif;font-size:11px;font-weight:600;
                  letter-spacing:0.08em;text-transform:uppercase;
                  color:{COLORS['text_secondary']};margin-bottom:8px;">{label}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;
                  font-weight:700;color:{value_color};line-height:1;">{value}</div>
      {f'<div style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:{COLORS["text_secondary"]};margin-top:4px;">{sub}</div>' if sub else ''}
    </div>
    """


def apply_theme() -> None:
    """
    Inject global CSS into the Streamlit page.
    Call once at the top of every page render() function.
    The CSS targets Streamlit's internal class names — these are stable across
    Streamlit 1.30+ but may need updating on major Streamlit version bumps.
    """
    # Load fonts
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&'
        'family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <style>
    /* ── Global resets ──────────────────────────────────────────────────── */
    .stApp {{
        background-color: {COLORS['bg_primary']};
    }}

    /* ── Typography ─────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stMarkdown, .stText, p, li {{
        font-family: 'DM Sans', sans-serif !important;
        color: {COLORS['text_primary']};
    }}

    /* Page title (st.title) */
    h1 {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        color: {COLORS['text_primary']} !important;
        padding-bottom: 4px;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 24px !important;
    }}

    /* st.subheader */
    h2, h3 {{
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* st.caption */
    .stMarkdown small, small {{
        color: {COLORS['text_secondary']} !important;
        font-size: 12px !important;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_secondary']} !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}

    [data-testid="stSidebar"] * {{
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        font-size: 13px !important;
        font-weight: 400 !important;
        color: {COLORS['text_secondary']} !important;
        padding: 6px 8px !important;
        border-radius: 4px !important;
        transition: color 0.15s, background 0.15s;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        color: {COLORS['text_primary']} !important;
        background: {COLORS['bg_tertiary']} !important;
    }}

    /* Selected radio item */
    [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label {{
        color: {COLORS['accent']} !important;
        font-weight: 500 !important;
    }}

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {{
        border-color: {COLORS['border']} !important;
        margin: 12px 0 !important;
    }}

    /* ── Main content area ───────────────────────────────────────────────── */
    .main .block-container {{
        padding: 32px 48px !important;
        max-width: 1200px !important;
    }}

    /* ── Metric cards (st.metric) ────────────────────────────────────────── */
    [data-testid="metric-container"] {{
        background: {COLORS['bg_secondary']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        padding: 16px !important;
    }}

    [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: {COLORS['text_secondary']} !important;
    }}

    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        border-radius: 4px !important;
        border: 1px solid {COLORS['border_strong']} !important;
        background: {COLORS['bg_secondary']} !important;
        color: {COLORS['text_primary']} !important;
        transition: border-color 0.15s, background 0.15s !important;
        padding: 6px 16px !important;
    }}

    .stButton > button:hover {{
        border-color: {COLORS['accent']} !important;
        background: {COLORS['bg_tertiary']} !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: {COLORS['accent']} !important;
        border-color: {COLORS['accent']} !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {COLORS['accent_dim']} !important;
        border-color: {COLORS['accent_dim']} !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* ── Download buttons ────────────────────────────────────────────────── */
    .stDownloadButton > button {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        border-radius: 4px !important;
        border: 1px solid {COLORS['border_strong']} !important;
        background: {COLORS['bg_secondary']} !important;
        color: {COLORS['text_primary']} !important;
    }}

    .stDownloadButton > button:hover {{
        border-color: {COLORS['blue']} !important;
        background: {COLORS['bg_tertiary']} !important;
    }}

    /* ── Inputs ──────────────────────────────────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stSelectbox select,
    [data-baseweb="select"] {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        background: {COLORS['bg_secondary']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 4px !important;
        color: {COLORS['text_primary']} !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {COLORS['accent']} !important;
        box-shadow: 0 0 0 2px {COLORS['accent_dim']} !important;
    }}

    /* ── Sliders ─────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background: {COLORS['accent']} !important;
        border-color: {COLORS['accent']} !important;
    }}

    /* ── Expanders ───────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        background: {COLORS['bg_secondary']} !important;
    }}

    [data-testid="stExpander"] summary {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: {COLORS['text_secondary']} !important;
        padding: 10px 16px !important;
    }}

    [data-testid="stExpander"] summary:hover {{
        color: {COLORS['text_primary']} !important;
    }}

    /* ── Dataframes / tables ─────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] table {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }}

    [data-testid="stDataFrame"] th {{
        background: {COLORS['bg_tertiary']} !important;
        color: {COLORS['text_secondary']} !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid {COLORS['border']} !important;
        padding: 10px 12px !important;
    }}

    [data-testid="stDataFrame"] td {{
        border-bottom: 1px solid {COLORS['border']} !important;
        padding: 8px 12px !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* ── Alerts ──────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 4px !important;
        border-left-width: 3px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }}

    /* Success */
    [data-testid="stAlert"][kind="success"] {{
        background: rgba(63, 185, 80, 0.08) !important;
        border-left-color: {COLORS['green']} !important;
        color: {COLORS['green']} !important;
    }}

    /* Warning */
    [data-testid="stAlert"][kind="warning"] {{
        background: rgba(210, 153, 34, 0.08) !important;
        border-left-color: {COLORS['amber']} !important;
        color: {COLORS['amber']} !important;
    }}

    /* Error */
    [data-testid="stAlert"][kind="error"] {{
        background: rgba(248, 81, 73, 0.08) !important;
        border-left-color: {COLORS['red']} !important;
        color: {COLORS['red']} !important;
    }}

    /* Info */
    [data-testid="stAlert"][kind="info"] {{
        background: rgba(56, 139, 253, 0.08) !important;
        border-left-color: {COLORS['blue']} !important;
        color: {COLORS['blue']} !important;
    }}

    /* ── File uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {COLORS['border_strong']} !important;
        border-radius: 6px !important;
        background: {COLORS['bg_secondary']} !important;
        padding: 24px !important;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS['accent']} !important;
    }}

    /* ── Code blocks ─────────────────────────────────────────────────────── */
    .stCodeBlock, code {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        background: {COLORS['bg_tertiary']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 4px !important;
        color: {COLORS['text_mono']} !important;
    }}

    /* ── Progress bar ────────────────────────────────────────────────────── */
    [data-testid="stProgress"] > div > div > div > div {{
        background: {COLORS['accent']} !important;
    }}

    [data-testid="stProgress"] > div > div > div {{
        background: {COLORS['bg_tertiary']} !important;
        border-radius: 3px !important;
    }}

    /* ── Checkbox ────────────────────────────────────────────────────────── */
    [data-testid="stCheckbox"] label {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] p {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        color: {COLORS['text_secondary']} !important;
    }}

    /* ── Form container ──────────────────────────────────────────────────── */
    [data-testid="stForm"] {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        padding: 20px !important;
        background: {COLORS['bg_secondary']} !important;
    }}

    /* ── Dividers ────────────────────────────────────────────────────────── */
    hr {{
        border: none !important;
        border-top: 1px solid {COLORS['border']} !important;
        margin: 24px 0 !important;
    }}

    /* ── Image display ───────────────────────────────────────────────────── */
    [data-testid="stImage"] img {{
        border-radius: 4px !important;
        border: 1px solid {COLORS['border']} !important;
    }}

    /* ── Plotly charts ───────────────────────────────────────────────────── */
    .js-plotly-plot .plotly {{
        border-radius: 6px !important;
    }}

    /* ── Hide Streamlit branding ─────────────────────────────────────────── */
    #MainMenu, footer, header {{
        visibility: hidden !important;
    }}

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_primary']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border_strong']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['text_secondary']};
    }}
    </style>
    """, unsafe_allow_html=True)
```

---

## File 2: Update `modules/dashboard/app.py`

### 2a — Update `configure_page()`

```python
def configure_page() -> None:
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
```

### 2b — Update `render_sidebar()`

Replace the entire sidebar render function:

```python
def render_sidebar() -> str:
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
    file_ready    = state.has_uploaded_file()
    dataset_ready = config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir())

    nav_options = [
        ("Dashboard",        "Dashboard"),
        ("Binary Upload",    "Binary Upload"),
        ("Malware Detection" + (" (no file)" if not (model_ready and file_ready) else ""),
         "Malware Detection"),
        ("Dataset Gallery"   + (" (no dataset)" if not dataset_ready else ""),
         "Dataset Gallery"),
        ("Model Training",   "Model Training"),
        ("Digital Twin",     "Digital Twin"),
    ]

    display_labels = [opt[0] for opt in nav_options]
    internal_keys  = [opt[1] for opt in nav_options]

    selected_display = st.sidebar.radio(
        "nav",
        options=display_labels,
        label_visibility="hidden",
    )
    selected_index = display_labels.index(selected_display)
    page = internal_keys[selected_index]

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
```

---

## File 3: Update every page — add `apply_theme()` and strip emojis

### 3a — Changes to make in ALL pages

At the top of every `render()` function, add as the first line:
```python
from modules.dashboard.theme import apply_theme, COLORS, section_header, mono, status_badge, confidence_bar_html, kpi_card
apply_theme()
```

### 3b — `pages/home.py` — replace KPI cards and status table

Replace the `st.metric` KPI cards with styled HTML cards:

```python
    # ── KPI Cards ──────────────────────────────────────────────────────────
    stats = get_detection_stats(config.DB_PATH)
    acc   = stats.get('model_accuracy')
    acc_str = f"{acc * 100:.1f}%" if acc is not None else "—"

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(kpi_card("Total Analyzed",  str(stats.get('total_analyzed', 0))), unsafe_allow_html=True)
    col2.markdown(kpi_card("Malware Detected", str(stats.get('total_malware', 0))), unsafe_allow_html=True)
    col3.markdown(kpi_card("Benign Files",    str(stats.get('total_benign', 0))),   unsafe_allow_html=True)
    col4.markdown(kpi_card("Model Accuracy",  acc_str, accent=acc is not None),      unsafe_allow_html=True)
```

Replace `st.title("🏠 System Overview Dashboard")` with:
```python
    st.title("System Overview")
```

Replace all other emoji-prefixed `st.subheader()` calls with `section_header()`:
```python
    section_header("Detection Activity", "Last 7 days")
    section_header("Module Status")
    section_header("Recent Detections")
```

### 3c — `pages/upload.py` — clean up titles and hash display

```python
# Replace:
st.title("📂 Binary Upload & Visualization")
# With:
st.title("Binary Upload")

# Replace the SHA-256 monospace code block:
st.code(meta['sha256'], language=None)
# With:
st.markdown(mono(meta['sha256']), unsafe_allow_html=True)
```

### 3d — `pages/detection.py` — replace confidence bar

Replace the `st.progress()` call and color logic with the styled HTML bar:

```python
# Replace the entire confidence section (banner + progress bar) with:
    st.markdown(
        f'<div style="background:{COLORS["bg_secondary"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;padding:20px 24px;margin-bottom:16px;">'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:13px;'
        f'color:{COLORS["text_secondary"]};margin-bottom:8px;letter-spacing:0.04em;">'
        f'DETECTED FAMILY</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:26px;'
        f'font-weight:700;color:{COLORS["text_primary"]};margin-bottom:16px;">'
        f'{family}</div>'
        f'{confidence_bar_html(confidence, family)}'
        f'</div>',
        unsafe_allow_html=True,
    )
```

Remove all emoji from `st.success()`, `st.warning()`, `st.error()` calls — replace with text only:
```python
# Before:
st.success(f"🎯 **{family}** detected...")
# After:
st.success(f"Detected: {family} — {confidence * 100:.1f}% confidence")
```

Replace `st.title("🔍 Malware Detection & Classification")`:
```python
st.title("Malware Detection")
```

### 3e — `pages/gallery.py`

```python
# Replace:
st.title("🖼️ Dataset Gallery")
# With:
st.title("Dataset Gallery")
```

### 3f — `pages/training.py`

```python
# Replace:
st.title("🏋️ Model Training")
# With:
st.title("Model Training")
```

### 3g — `pages/digital_twin.py`

```python
# Replace:
st.title("🖥️ Digital Twin Simulation")
# With:
st.title("Digital Twin Simulation")
```

---

## File 4: Update Plotly chart themes

All Plotly charts should use the dark theme and the same color tokens. Find every `st.plotly_chart()` call and ensure the figure uses consistent theming:

```python
# Standard chart layout to apply to every Plotly figure
def apply_chart_theme(fig, title: str = '') -> None:
    """Apply consistent dark theme to a Plotly figure. Import from theme.py."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['bg_secondary'],
        font=dict(family='DM Sans, sans-serif', size=12, color=COLORS['text_primary']),
        title=dict(text=title, font=dict(size=13, color=COLORS['text_secondary'])) if title else None,
        margin=dict(l=48, r=24, t=36 if title else 16, b=40),
        xaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border'], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border'], tickfont=dict(size=11)),
        legend=dict(font=dict(size=11)),
    )
```

Add this function to `theme.py` and call it in:

- `home.py` → `_render_activity_chart()` before `st.plotly_chart(fig, ...)`
- `upload.py` → pixel intensity histogram
- `detection.py` → `_render_probability_chart()`, update bar colors to use `COLORS['accent']` and `COLORS['bg_tertiary']`

For the probability chart specifically, update colors:
```python
# In _render_probability_chart():
colors = [COLORS['accent']] + [COLORS['bg_tertiary']] * (len(families) - 1)
# And update border/text colors:
fig.update_traces(marker_line_color=COLORS['border'], marker_line_width=0.5,
                  textfont=dict(color=COLORS['text_secondary'], size=11))
```

---

## File 5: `modules/dashboard/app.py` — update page title in `main()`

The page title displayed in `st.title()` should not repeat the app name. Update routing so each page title is clean:

```python
# In main() routing, the titles are now handled inside each page's render()
# Just ensure the _check_network_binding warning uses no emoji:
# Replace: st.warning("⚠️ **Security Notice...")
# With:    st.warning("Security Notice (SRS SEC-5): ...")
```

---

## Definition of Done

```bash
# No new tests needed — this is purely visual.
# Verify the theme loads without errors:
python -c "from modules.dashboard.theme import apply_theme, COLORS, kpi_card, mono, section_header, status_badge, confidence_bar_html, apply_chart_theme"

# Launch and verify visually:
streamlit run modules/dashboard/app.py --server.port 8501

# Visual checklist:
#   [ ] Background is dark navy (#0D1117), not white
#   [ ] Sidebar is slightly lighter (#161B22) with a right border
#   [ ] Page titles are clean — no emoji, DM Sans font
#   [ ] KPI cards have dark surface with monospace numbers
#   [ ] SHA-256 hashes display in monospace blue (#A5D6FF)
#   [ ] Confidence bar is custom HTML (not st.progress) with correct colors
#   [ ] Status indicators are geometric (filled/hollow circles) — no emoji
#   [ ] All Plotly charts use dark theme matching the page background
#   [ ] Buttons have dark surface with amber hover on primary
#   [ ] No emojis anywhere in titles, labels, or status text
#   [ ] Fonts load: DM Sans for UI, JetBrains Mono for technical values
#   [ ] Sidebar navigation labels are lowercase/sentence case — no emojis
#   [ ] Scrollbar is styled (thin, dark)
#   [ ] Streamlit footer/header is hidden
```

---

## Known Streamlit Styling Limitations

These are constraints you cannot work around without a full custom frontend:

| Limitation | Impact | Workaround |
|---|---|---|
| Streamlit injects its own CSS with high specificity | Some component styles may not apply consistently across Streamlit versions | Use `!important` on critical overrides; test after any Streamlit upgrade |
| `data-testid` attributes may change between minor Streamlit versions | Selectors targeting internal testids could break | Check after upgrading; `data-testid="stSidebar"` is stable as of 1.30+ |
| Google Fonts load from external CDN | Fonts fail if network access is blocked | Fall back to `system-ui` in font stack: `'DM Sans', system-ui, sans-serif` |
| `st.radio` does not support per-item disabled state | Cannot grey out individual nav items natively | Text annotation `(no file)` suffix is the practical substitute |
| Plotly chart background transparency | `paper_bgcolor='rgba(0,0,0,0)'` works but may flash white on load | Set `plot_bgcolor` to match page background as fallback |
| `st.metric` delta colors are hardcoded | Cannot theme the green/red delta arrows | Use custom HTML KPI cards via `kpi_card()` instead |

---

## Color Reference

| Token | Hex | Used for |
|---|---|---|
| `bg_primary` | `#0D1117` | Page background |
| `bg_secondary` | `#161B22` | Cards, sidebar, form surfaces |
| `bg_tertiary` | `#21262D` | Hover states, table headers, code bg |
| `border` | `#30363D` | Default borders |
| `border_strong` | `#484F58` | Visible separators |
| `text_primary` | `#E6EDF3` | Main readable text |
| `text_secondary` | `#8B949E` | Labels, captions, muted text |
| `text_mono` | `#A5D6FF` | Hashes, IDs, technical values |
| `accent` | `#E8A020` | Primary button, active state, important values |
| `green` | `#3FB950` | Active status, high confidence, success |
| `amber` | `#D29922` | Warning, medium confidence, inactive status |
| `red` | `#F85149` | Error, low confidence, failed status |
| `blue` | `#388BFD` | Informational alerts, download buttons |
