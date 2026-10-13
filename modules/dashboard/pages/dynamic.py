# modules/dashboard/pages/dynamic.py
"""
Dynamic (behavioural) Analysis page — Module 9.

Input:  a Zeek conn.log capture (IoT-23 file, or later a Digital Twin capture),
        or a clearly labelled synthetic demo capture.
Output: capture verdict, behaviour profile, most suspicious flows,
        feature-importance explanation and MITRE ATT&CK for ICS mapping.
The report is stored in session state for the Hybrid Verdict page.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from modules.dashboard import state

DEMO_CAPTURES = {
    'Benign IIoT traffic (DNS, HTTPS, MQTT, Modbus)': {'Benign': 400},
    'Mirai/Okiru-style propagation (scan + Telnet)':  {'Benign': 300, 'PortScan': 90, 'Okiru': 50, 'C&C': 12},
    'DDoS against a Modbus/HTTP target':              {'Benign': 250, 'DDoS': 70, 'C&C': 8},
    'C&C beaconing with payload download':            {'Benign': 300, 'C&C': 30, 'FileDownload': 10},
}


def _load_detector():
    if st.session_state.get(state.KEY_DYNAMIC_MODEL) is not None:
        return st.session_state[state.KEY_DYNAMIC_MODEL]
    if not config.DYNAMIC_MODEL_PATH.exists():
        return None
    from modules.dynamic import DynamicDetector
    try:
        det = DynamicDetector.load(config.DYNAMIC_MODEL_PATH)
    except Exception as e:
        st.error(f"Error: dynamic model could not be loaded. Cause: {e}. "
                 f"Action: retrain with `python scripts/train_dynamic.py`.")
        return None
    st.session_state[state.KEY_DYNAMIC_MODEL] = det
    return det


def _load_metrics() -> dict:
    try:
        with open(config.DYNAMIC_METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def render():
    from modules.dashboard.theme import apply_theme, COLORS, section_header, kpi_card, apply_chart_theme
    apply_theme()
    st.title("Dynamic Analysis")
    st.caption("Behavioural detection from network traffic captured while a sample runs "
               "(Zeek conn.log — IoT-23 today, Digital Twin captures in the next phase).")
    st.markdown("---")

    det = _load_detector()
    if det is None:
        st.warning(
            "No dynamic model available.\n\n"
            "- With IoT-23: extract the labelled `conn.log.labeled` files under `data/iot23/` "
            "and run `python scripts/train_dynamic.py`\n"
            "- Demo only: `python scripts/train_dynamic.py --synthetic 6000`"
        )
        return

    metrics = _load_metrics()
    meta = metrics.get('metadata', getattr(det, 'metadata', {}))
    source = meta.get('data_source', 'unknown')
    _render_model_strip(metrics, source, det, COLORS, kpi_card)

    # ── Input ────────────────────────────────────────────────────────────────
    section_header("Capture input", "Upload a Zeek conn.log, or generate a synthetic demo capture.")
    tab_upload, tab_demo = st.tabs(["Upload conn.log", "Synthetic demo capture"])
    flows, src = None, None
    with tab_upload:
        up = st.file_uploader("Zeek conn.log / conn.log.labeled", type=None,
                              help="Tab-separated Zeek connection log. Labelled IoT-23 files "
                                   "also show ground-truth accuracy for the capture.")
        if up is not None and st.button("Analyse uploaded capture", type="primary"):
            if up.size > config.MAX_UPLOAD_BYTES:
                st.error("Error: capture too large. Cause: file exceeds 50 MB. "
                         "Action: upload a smaller capture or a sample of it.")
                return
            from modules.dynamic import parse_conn_log
            try:
                flows = parse_conn_log(up.getvalue(), max_rows=200_000)
                src = {'name': up.name, 'synthetic': False}
            except Exception as e:
                st.error(f"Error: capture could not be parsed. Cause: {e}. "
                         f"Action: upload a tab-separated Zeek conn.log file.")
                return
    with tab_demo:
        choice = st.selectbox("Scenario", list(DEMO_CAPTURES))
        if st.button("Generate and analyse demo capture"):
            from modules.dynamic.synthetic import generate_flows
            from modules.dynamic.zeek_parser import map_behaviour
            flows = generate_flows(DEMO_CAPTURES[choice], seed=100 + list(DEMO_CAPTURES).index(choice))
            flows['behaviour'] = [map_behaviour(a, b) for a, b in
                                  zip(flows['label'], flows['detailed-label'])]
            src = {'name': f'SYNTHETIC: {choice}', 'synthetic': True}

    if flows is not None:
        from modules.dynamic.detector import load_behaviour_mitre
        with st.spinner("Analysing flows…"):
            report = det.analyse_capture(flows, mitre_db=load_behaviour_mitre(config.MITRE_BEHAVIOUR_JSON_PATH))
            pred = det.predict_proba_df(flows)
        st.session_state[state.KEY_DYNAMIC_REPORT] = report.to_dict()
        st.session_state[state.KEY_DYNAMIC_SOURCE] = src
        st.session_state['_dynamic_flows'] = flows
        st.session_state['_dynamic_pred'] = pred
        st.session_state[state.KEY_FUSION_RESULT] = None

    if st.session_state.get(state.KEY_DYNAMIC_REPORT):
        _render_report(det, COLORS, section_header, kpi_card, apply_chart_theme)


def _render_model_strip(metrics, source, det, COLORS, kpi_card):
    b = metrics.get('binary', {})
    fmt = lambda v: '—' if v is None else f"{v * 100:.1f}%"  # noqa: E731
    cols = st.columns(4)
    badge = ('SYNTHETIC DEMO' if source == 'synthetic' else source)
    cols[0].markdown(kpi_card('Training data', badge, f"{len(det.classes_)} behaviour classes",
                              accent=(source == 'synthetic')), unsafe_allow_html=True)
    cols[1].markdown(kpi_card('Detection rate', fmt(b.get('detection_rate')), 'malicious flows caught'),
                     unsafe_allow_html=True)
    cols[2].markdown(kpi_card('False positive rate', fmt(b.get('false_positive_rate')),
                              'benign flows flagged'), unsafe_allow_html=True)
    cols[3].markdown(kpi_card('Macro F1', fmt(metrics.get('f1_macro')), 'per-behaviour average'),
                     unsafe_allow_html=True)
    if source == 'synthetic':
        st.info("This dynamic model was trained on synthetic traffic for demonstration. "
                "Its scores do not represent real-world performance — retrain on IoT-23.")


def _render_report(det, COLORS, section_header, kpi_card, apply_chart_theme):
    rep = st.session_state[state.KEY_DYNAMIC_REPORT]
    src = st.session_state.get(state.KEY_DYNAMIC_SOURCE) or {}
    flows = st.session_state.get('_dynamic_flows')
    pred = st.session_state.get('_dynamic_pred')

    section_header("Capture verdict", src.get('name', ''))
    color = {'MALICIOUS': COLORS['red'], 'SUSPICIOUS': COLORS['amber'],
             'BENIGN': COLORS['green']}[rep['verdict']]
    c = st.columns(4)
    c[0].markdown(kpi_card('Verdict', f"<span style='color:{color}'>{rep['verdict']}</span>",
                           'capture-level'), unsafe_allow_html=True)
    c[1].markdown(kpi_card('Capture score', f"{rep['capture_score']:.2f}",
                           'mean P(malicious) of top 5% flows'), unsafe_allow_html=True)
    c[2].markdown(kpi_card('Flagged flows', f"{rep['n_malicious_flows']:,}",
                           f"of {rep['n_flows']:,} ({rep['malicious_fraction']:.1%})"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi_card('Dominant behaviour', rep['dominant_behaviour'] or '—', ''),
                  unsafe_allow_html=True)
    if src.get('synthetic'):
        st.caption("Synthetic capture — generated for demonstration, not real traffic.")

    if flows is not None and 'behaviour' in flows.columns and pred is not None:
        acc = (flows['behaviour'].values == pred['behaviour_pred'].values).mean()
        st.caption(f"Capture is labelled — per-flow behaviour accuracy against ground truth: {acc:.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        section_header("Behaviour profile", "Flagged flows by behaviour class")
        prof = rep['behaviour_profile']
        if prof:
            fig = go.Figure(go.Bar(x=list(prof.values()), y=list(prof.keys()), orientation='h',
                                   marker_color=COLORS['red']))
            apply_chart_theme(fig, '')
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("No malicious behaviour detected.")
    with col_b:
        section_header("Why — explanation",
                       "Features where flagged flows differ most from benign traffic")
        if rep['explanation']:
            st.dataframe(pd.DataFrame([{
                'Feature': e['feature'], 'Meaning': e['description'] or '—',
                'Flagged avg': round(e['flagged_mean'], 3),
                'Benign avg': round(e['benign_mean'], 3),
                'Deviation (σ)': round(e['z_score'], 1),
            } for e in rep['explanation']]), hide_index=True, use_container_width=True)
        else:
            st.markdown("Nothing flagged, so nothing to explain.")

    section_header("MITRE ATT&CK for ICS", "Techniques indicated by the observed behaviours")
    if rep['mitre'].get('techniques'):
        st.markdown("**Tactics:** " + ", ".join(rep['mitre']['tactics']))
        st.dataframe(pd.DataFrame(rep['mitre']['techniques']).rename(
            columns={'id': 'Technique ID', 'name': 'Technique'}), hide_index=True,
            use_container_width=True)
    else:
        st.markdown("No techniques — no malicious behaviour observed.")

    if pred is not None and flows is not None:
        section_header("Most suspicious flows", "Top 25 by P(malicious)")
        view = flows[['ts', 'id.resp_p', 'proto', 'service', 'conn_state', 'orig_bytes',
                      'resp_bytes']].copy()
        view['predicted'] = pred['behaviour_pred']
        view['P(malicious)'] = pred['malicious_prob'].round(3)
        st.dataframe(view.sort_values('P(malicious)', ascending=False).head(25),
                     hide_index=True, use_container_width=True)

    with st.expander("Global feature importance (model-wide)"):
        imp = det.feature_importance(top=12)
        fig = go.Figure(go.Bar(x=[r['importance'] for r in imp][::-1],
                               y=[r['feature'] for r in imp][::-1], orientation='h',
                               marker_color=COLORS['blue']))
        apply_chart_theme(fig, '')
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download dynamic report (JSON)",
                       data=json.dumps({'source': src, **rep}, indent=2),
                       file_name="maltwin_dynamic_report.json", mime="application/json")
    st.info("Next: open **Hybrid Verdict** to fuse this with a static CNN detection.")
