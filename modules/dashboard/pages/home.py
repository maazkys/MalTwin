# modules/dashboard/pages/home.py
"""
Home / System Overview Dashboard page.
Implements SRS Mockup M1 — Main Dashboard Screen.
SRS refs: FR1.1, FR1.2, FR1.4
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

import config
from modules.dashboard.db import (
    get_detection_stats,
    get_events_by_date_range,
)
from modules.dashboard import state


def render():
    st.title("🏠 System Overview Dashboard")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    stats = get_detection_stats(config.DB_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Files Analyzed", value=stats.get('total_analyzed', 0))
    with col2:
        st.metric(label="Malware Detected",     value=stats.get('total_malware', 0))
    with col3:
        st.metric(label="Benign Files",         value=stats.get('total_benign', 0))
    with col4:
        acc = stats.get('model_accuracy')
        st.metric(
            label="Model Accuracy",
            value=f"{acc * 100:.1f}%" if acc is not None else "N/A",
        )

    st.markdown("---")

    # ── Activity Chart + Digital Twin Status ──────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Detection Activity (Last 7 Days)")
        _render_activity_chart(config.DB_PATH)

    with col_right:
        st.subheader("Digital Twin Status")
        st.info("🖥️ Digital Twin simulation is in a future implementation phase.")
        st.markdown("**Status:** Not Configured")
        st.markdown("**Active Nodes:** —")
        st.markdown("**Traffic Flow:** —")

    st.markdown("---")

    # ── Recent Detection Feed (with filters) ──────────────────────────────────
    st.subheader("Detection History")
    _render_history_section()

    st.markdown("---")

    # ── Module Status Table ───────────────────────────────────────────────────
    st.subheader("Module Status")
    _render_module_status()


def _render_activity_chart(db_path):
    """Fetch event counts per day for the last 7 days and render a Plotly line chart."""
    events = get_events_by_date_range(db_path, days_back=7)

    # Build date → count mapping for last 7 days
    today = datetime.utcnow().date()
    date_counts: dict = defaultdict(int)
    for i in range(7):
        day = today - timedelta(days=6 - i)
        date_counts[str(day)] = 0   # pre-fill with zero

    for ev in events:
        try:
            day_str = ev['timestamp'][:10]   # "YYYY-MM-DD"
            if day_str in date_counts:
                date_counts[day_str] += 1
        except (KeyError, TypeError):
            continue

    dates  = sorted(date_counts.keys())
    counts = [date_counts[d] for d in dates]

    fig = go.Figure(go.Scatter(
        x=dates,
        y=counts,
        mode='lines+markers',
        marker=dict(size=8, color='#FF4B4B'),
        line=dict(color='#FF4B4B', width=2),
        name='Detections',
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Detections",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        yaxis=dict(rangemode='nonnegative'),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_module_status():
    """Render a static table showing the implementation status of all 8 modules."""
    model_status = "✅ Active" if config.BEST_MODEL_PATH.exists() else "⚠️ No model trained"
    modules = [
        ("M1", "Digital Twin Simulation",      "⚠️ Deferred — future sprint"),
        ("M2", "Binary-to-Image Conversion",   "✅ Active"),
        ("M3", "Dataset & Preprocessing",      "✅ Active"),
        ("M4", "Data Enhancement & Balancing", "✅ Active"),
        ("M5", "Malware Detection (CNN)",       model_status),
        ("M6", "Dashboard & Visualization",    "✅ Active"),
        ("M7", "Explainable AI (Grad-CAM)",    "⚠️ Deferred — future sprint"),
        ("M8", "Automated Threat Reporting",   "⚠️ Deferred — future sprint"),
    ]
    df = pd.DataFrame(modules, columns=["ID", "Module", "Status"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_history_section() -> None:
    """
    Filterable, sortable detection history table.
    SRS refs: FR1.4, M8 FE-5
    """
    import pandas as pd
    from modules.dashboard.db import get_filtered_events, get_family_list

    # ── Filter controls ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter & Sort", expanded=False):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            family_options = get_family_list(config.DB_PATH)
            family_filter = st.selectbox(
                "Malware Family",
                options=family_options,
                index=0,
                key="history_family_filter",
            )

        with col_f2:
            min_conf_pct = st.slider(
                "Min Confidence",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                format="%d%%",
                key="history_min_conf",
            )

        with col_f3:
            days_options = {
                "All time":    None,
                "Last 7 days":  7,
                "Last 30 days": 30,
                "Last 90 days": 90,
            }
            days_label = st.selectbox(
                "Time Range",
                options=list(days_options.keys()),
                index=0,
                key="history_days",
            )
            days_back = days_options[days_label]

        with col_f4:
            sort_desc = st.radio(
                "Sort Order",
                options=["Newest first", "Oldest first"],
                index=0,
                key="history_sort",
            ) == "Newest first"

        limit = st.slider(
            "Max rows",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="history_limit",
        )

    # ── Fetch events ──────────────────────────────────────────────────────────
    events = get_filtered_events(
        db_path=config.DB_PATH,
        family_filter=family_filter if family_filter != 'All Families' else None,
        min_confidence=min_conf_pct / 100,
        days_back=days_back,
        limit=limit,
        sort_desc=sort_desc,
    )

    if not events:
        st.caption(
            "No detections match the current filters. "
            "Upload a binary file on the Binary Upload page to get started."
        )
        return

    # ── Build display dataframe ───────────────────────────────────────────────
    df = pd.DataFrame(events)

    df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
    df['timestamp']      = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    display_df = df[[
        'timestamp', 'file_name', 'predicted_family',
        'confidence_pct', 'file_format', 'device_used',
    ]].copy()
    display_df.columns = [
        'Timestamp', 'File', 'Predicted Family',
        'Confidence', 'Format', 'Device',
    ]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rows shown", len(events))
    col_m2.metric(
        "Avg confidence",
        f"{df['confidence'].mean() * 100:.1f}%",
    )
    col_m3.metric(
        "Unique families",
        df['predicted_family'].nunique(),
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export to CSV",
        data=csv_bytes,
        file_name="maltwin_detection_history.csv",
        mime="text/csv",
        help="Download the filtered detection history as a CSV file.",
    )
