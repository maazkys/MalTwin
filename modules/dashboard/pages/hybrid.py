# modules/dashboard/pages/hybrid.py
"""
Hybrid Verdict page — Module 10 (static + dynamic decision-level fusion).

Two modes:
  1. Evidence fusion: combines the latest static CNN detection (Malware
     Detection page) with the latest dynamic capture report (Dynamic Analysis).
  2. What-if simulator: sliders for static confidence and dynamic score, to
     demonstrate the fusion decision table without trained models.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from modules.dashboard import state
from modules.fusion import fuse, CATEGORY_TEXT


def render():
    from modules.dashboard.theme import apply_theme, COLORS, section_header, kpi_card
    apply_theme()
    st.title("Hybrid Verdict")
    st.caption("Decision-level fusion of static (binary image → CNN) and dynamic "
               "(network behaviour → random forest) evidence.")
    st.markdown("---")

    with st.expander("Fusion settings", expanded=False):
        c1, c2 = st.columns(2)
        w_s = c1.slider("Static weight", 0.0, 1.0, float(config.FUSION_STATIC_WEIGHT), 0.05,
                        help="Relative trust in the CNN family match.")
        w_d = c2.slider("Dynamic weight", 0.0, 1.0, float(config.FUSION_DYNAMIC_WEIGHT), 0.05,
                        help="Relative trust in the behavioural score.")
        st.caption(f"Verdict thresholds: MALICIOUS ≥ {config.FUSION_MALICIOUS_THRESHOLD:.2f}, "
                   f"SUSPICIOUS ≥ {config.FUSION_SUSPICIOUS_THRESHOLD:.2f}. "
                   f"Static 'high' = confidence ≥ {config.CONFIDENCE_GREEN:.2f}; "
                   f"dynamic 'high' = score ≥ 0.50.")
    if w_s + w_d == 0:
        st.error("Error: both weights are zero. Cause: no source would contribute. "
                 "Action: give at least one source a positive weight.")
        return

    tab_evidence, tab_whatif = st.tabs(["Fuse current evidence", "What-if simulator"])

    with tab_evidence:
        _render_evidence_mode(w_s, w_d, COLORS, section_header, kpi_card)
    with tab_whatif:
        _render_whatif(w_s, w_d, COLORS, section_header)


def _render_evidence_mode(w_s, w_d, COLORS, section_header, kpi_card):
    det = st.session_state.get(state.KEY_DETECTION)
    dyn = st.session_state.get(state.KEY_DYNAMIC_REPORT)
    dyn_src = st.session_state.get(state.KEY_DYNAMIC_SOURCE) or {}
    meta = st.session_state.get(state.KEY_FILE_META) or {}

    c1, c2 = st.columns(2)
    with c1:
        section_header("Static evidence", "From the Malware Detection page")
        if det:
            st.markdown(kpi_card('Family match', det['predicted_family'],
                                 f"confidence {det['confidence']:.1%} · {meta.get('name', '')}"),
                        unsafe_allow_html=True)
        else:
            st.markdown("No static detection yet — upload a binary and run **Malware Detection**.")
    with c2:
        section_header("Dynamic evidence", "From the Dynamic Analysis page")
        if dyn:
            st.markdown(kpi_card('Behaviour', dyn['dominant_behaviour'] or 'none detected',
                                 f"capture score {dyn['capture_score']:.2f} · {dyn_src.get('name', '')}"),
                        unsafe_allow_html=True)
        else:
            st.markdown("No capture analysed yet — use **Dynamic Analysis**.")

    if not det and not dyn:
        st.info("Provide at least one source of evidence, or try the **What-if simulator** tab.")
        return

    if det and dyn:
        st.warning(
            "Pairing note: in this phase the binary and the traffic capture are supplied "
            "separately, so the analyst asserts they belong to the same sample. Automatic "
            "pairing — running the binary inside the Digital Twin and capturing its traffic — "
            "is part of the Digital Twin module."
        )

    if st.button("Compute hybrid verdict", type="primary"):
        from modules.reporting.mitre_mapper import get_mitre_mapping
        static_mitre = get_mitre_mapping(det['predicted_family']) if det else None
        result = fuse(
            static_conf=float(det['confidence']) if det else None,
            dynamic_score=float(dyn['capture_score']) if dyn else None,
            family=det['predicted_family'] if det else None,
            dominant_behaviour=dyn['dominant_behaviour'] if dyn else None,
            static_mitre=static_mitre,
            dynamic_mitre=dyn['mitre'] if dyn else None,
            w_static=w_s, w_dynamic=w_d,
        )
        st.session_state[state.KEY_FUSION_RESULT] = result.to_dict()

    res = st.session_state.get(state.KEY_FUSION_RESULT)
    if res:
        _render_result(res, COLORS, section_header, kpi_card)
        payload = {
            'binary': {'name': meta.get('name'), 'sha256': meta.get('sha256')},
            'dynamic_source': dyn_src,
            'fusion': res,
        }
        st.download_button("Download hybrid report (JSON)", json.dumps(payload, indent=2),
                           file_name="maltwin_hybrid_report.json", mime="application/json")


def _render_result(res, COLORS, section_header, kpi_card):
    section_header("Hybrid verdict")
    color = {'MALICIOUS': COLORS['red'], 'SUSPICIOUS': COLORS['amber'],
             'BENIGN': COLORS['green']}[res['verdict']]
    c = st.columns(3)
    c[0].markdown(kpi_card('Verdict', f"<span style='color:{color}'>{res['verdict']}</span>",
                           'escalated (novel variant rule)' if res['escalated'] else ''),
                  unsafe_allow_html=True)
    c[1].markdown(kpi_card('Fused score', f"{res['fused_score']:.2f}",
                           f"weights S {res['weights']['static']:.2f} / D {res['weights']['dynamic']:.2f}"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi_card('Agreement', res['category'].replace('_', ' '), ''),
                  unsafe_allow_html=True)
    st.markdown(f"**Interpretation:** {res['explanation']}")

    if res['mitre_techniques']:
        section_header("MITRE ATT&CK for ICS — combined",
                       "Techniques supported by both sources are listed first")
        st.dataframe(pd.DataFrame([{
            'Technique ID': t['id'], 'Technique': t['name'],
            'Evidence': ' + '.join(t['sources']),
        } for t in res['mitre_techniques']]), hide_index=True, use_container_width=True)


def _render_whatif(w_s, w_d, COLORS, section_header):
    section_header("What-if simulator",
                   "Move the sliders to see how the fusion rules respond. No models needed.")
    c1, c2 = st.columns(2)
    s = c1.slider("Static confidence (CNN top-1)", 0.0, 1.0, 0.30, 0.01)
    d = c2.slider("Dynamic capture score", 0.0, 1.0, 0.85, 0.01)
    r = fuse(static_conf=s, dynamic_score=d, w_static=w_s, w_dynamic=w_d)

    color = {'MALICIOUS': COLORS['red'], 'SUSPICIOUS': COLORS['amber'],
             'BENIGN': COLORS['green']}[r.verdict]
    st.markdown(f"<div style='font-size:22px;font-weight:700;color:{color}'>{r.verdict}</div>"
                f"<div style='color:{COLORS['text_secondary']}'>fused score {r.fused_score:.2f} · "
                f"{r.category.replace('_', ' ')}{' · escalated' if r.escalated else ''}</div>",
                unsafe_allow_html=True)
    st.markdown(r.explanation)

    # 2×2 agreement matrix with the current cell highlighted
    cells = {('high', 'high'): 'CONFIRMED', ('high', 'low'): 'STATIC_ONLY',
             ('low', 'high'): 'NOVEL_VARIANT', ('low', 'low'): 'NO_EVIDENCE'}
    z, text = [], []
    for dyn_lvl in ('high', 'low'):
        row_z, row_t = [], []
        for st_lvl in ('low', 'high'):
            cat = cells[(st_lvl, dyn_lvl)]
            row_z.append(1 if cat == r.category else 0)
            row_t.append(cat.replace('_', ' '))
        z.append(row_z); text.append(row_t)
    fig = go.Figure(go.Heatmap(
        z=z, x=['static low', 'static high'], y=['dynamic high', 'dynamic low'],
        text=text, texttemplate='%{text}', showscale=False,
        colorscale=[[0, COLORS['bg_tertiary']], [1, COLORS['accent']]],
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color=COLORS['text_primary']))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("All agreement categories"):
        st.dataframe(pd.DataFrame([{'Category': k.replace('_', ' '), 'Meaning': v}
                                   for k, v in CATEGORY_TEXT.items()]),
                     hide_index=True, use_container_width=True)
