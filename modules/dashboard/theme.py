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


def image_full_width(image, **kwargs) -> None:
    """
    st.image stretched to the column width on any Streamlit version.
    Streamlit ≤1.39 uses `use_column_width`; ≥1.40 uses `use_container_width`
    and newer releases removed `use_column_width` entirely.
    """
    import inspect
    params = inspect.signature(st.image).parameters
    if 'use_container_width' in params:
        st.image(image, use_container_width=True, **kwargs)
    elif 'use_column_width' in params:
        st.image(image, use_column_width=True, **kwargs)
    else:
        st.image(image, **kwargs)
