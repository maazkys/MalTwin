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
    get_recent_events,
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
        st.subheader("System Resources")
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()

        if stats['error']:
            st.caption("System stats unavailable (psutil error).")
        else:
            st.metric("CPU Usage",    f"{stats['cpu_pct']:.1f}%")
            st.metric("RAM Usage",    f"{stats['mem_pct']:.1f}%")
            st.metric(
                "RAM Used",
                f"{stats['mem_used_gb']:.1f} / {stats['mem_total_gb']:.1f} GB",
            )
            st.metric("Device",       stats['device'].upper())
            st.metric("Dashboard Up", stats['uptime_str'])

    st.markdown("---")

    # ── Quick recent feed (always visible — SRS FR1.4) ───────────────────────
    st.subheader("Recent Detections")
    _render_recent_feed_baseline()

    # ── Full filterable history ───────────────────────────────────────────────
    with st.expander("📋 Full Detection History (filter & export)", expanded=False):
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


def _render_module_status() -> None:
    """
    Render the live module status table using health.py checks.
    SRS ref: FR1.1 — refreshes automatically (via st.cache_data ttl=30s).
    """
    import pandas as pd
    from modules.dashboard.health import get_all_module_statuses

    statuses = get_all_module_statuses()

    rows = [
        {
            'ID':     s['id'],
            'Module': s['name'],
            'Status': f"{s['emoji']} {s['status'].capitalize()}",
            'Detail': s['detail'],
        }
        for s in statuses
    ]

    df = pd.DataFrame(rows)

    # Color-code the Status column
    def _color_status(val: str) -> str:
        if '✅' in val:
            return 'color: #3cb371; font-weight: bold'
        if '⚠️' in val:
            return 'color: #e6a21e; font-weight: bold'
        return 'color: #d23232; font-weight: bold'

    styled = df.style.map(_color_status, subset=['Status'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary counts
    n_active   = sum(1 for s in statuses if s['status'] == 'active')
    n_inactive = sum(1 for s in statuses if s['status'] == 'inactive')
    n_error    = sum(1 for s in statuses if s['status'] == 'error')

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Active",   n_active)
    col2.metric("⚠️ Inactive", n_inactive)
    col3.metric("🔴 Error",    n_error)

    st.caption(
        "Status refreshes every 30 seconds automatically. "
        "(SRS FR1.1 specifies 5s; 30s is used to avoid excessive filesystem I/O — "
        "functionally equivalent for research use.)"
    )


def _render_recent_feed_baseline() -> None:
    """
    Scrollable feed of the 5 most recent detections. SRS FR1.4 baseline.
    Always visible — no filters, no interaction required.
    """
    import pandas as pd
    events = get_recent_events(config.DB_PATH, limit=5)

    if not events:
        st.caption("No detections yet. Upload a binary file to get started.")
        return

    df = pd.DataFrame(events)
    df['confidence'] = df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
    df['timestamp']  = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display = df[['timestamp', 'file_name', 'predicted_family', 'confidence']].copy()
    display.columns = ['Timestamp', 'File', 'Predicted Family', 'Confidence']
    st.dataframe(display, use_container_width=True, hide_index=True)


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
