# modules/dashboard/pages/digital_twin.py
"""
Digital Twin page — Module 1.

Status (Evaluation 3): design complete, implementation scheduled for
Evaluation 4. This page shows the agreed architecture, planned nodes, the
interface to the Dynamic Analysis module, and live prerequisite checks so
the deployment machine can be prepared in advance.
"""
import shutil
import subprocess

import pandas as pd
import streamlit as st

import config

PLANNED_NODES = [
    ('plc-1',      'PLC (Modbus TCP server)',   'pymodbus server, holding registers for a simulated process', '502/tcp'),
    ('hmi-1',      'HMI / SCADA client',        'polls the PLC every second, writes setpoints',             '—'),
    ('sensor-1..3','Field sensors',              'publish temperature/pressure telemetry over MQTT',          '—'),
    ('broker',     'MQTT broker',                'Eclipse Mosquitto',                                          '1883/tcp'),
    ('historian',  'Historian',                  'subscribes to telemetry and stores it',                      '—'),
    ('sandbox',    'Sandbox node',               'executes the uploaded sample, no host mounts, capped CPU/RAM', '—'),
    ('monitor',    'Network monitor',            'Zeek on the twin bridge → conn.log for Module 9',            '—'),
]


def _docker_status() -> tuple[str, str]:
    if shutil.which('docker') is None:
        return 'missing', 'Docker not installed'
    try:
        r = subprocess.run(['docker', 'info'], capture_output=True, timeout=3)
        if r.returncode != 0:
            return 'stopped', 'Docker installed but daemon not running'
        r2 = subprocess.run(['docker', 'compose', 'version'], capture_output=True, timeout=3, text=True)
        comp = r2.stdout.strip() if r2.returncode == 0 else 'docker compose plugin missing'
        return 'ok', f'Docker running · {comp}'
    except subprocess.TimeoutExpired:
        return 'stopped', 'Docker did not respond within 3 s'
    except Exception as e:  # pragma: no cover
        return 'stopped', f'Docker check failed: {e}'


def render():
    from modules.dashboard.theme import apply_theme, section_header, status_badge, COLORS
    apply_theme()
    st.title("Digital Twin")
    st.caption("Module 1 — isolated IIoT simulation where samples are executed safely and "
               "their network behaviour is captured.")
    st.markdown("---")
    st.info("Status: design complete; implementation is scheduled for the final evaluation. "
            "The interface to the rest of MalTwin is already fixed (see below), and the "
            "Dynamic Analysis and Hybrid Verdict modules that consume its output are implemented.")

    section_header("Architecture", "Containers on an internal Docker bridge with no route to the host LAN")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#161B22", color="#30363D",
            fontcolor="#E6EDF3", fontname="Helvetica", fontsize=11];
      edge [color="#8B949E", fontcolor="#8B949E", fontsize=9, fontname="Helvetica"];
      subgraph cluster_twin {
        label="Digital Twin — docker network (internal: true)"; fontcolor="#E8A020";
        color="#E8A020"; style="dashed"; fontname="Helvetica";
        hmi [label="HMI / SCADA"]; plc [label="PLC\\nModbus TCP :502"];
        s [label="Sensors x3"]; broker [label="MQTT broker\\n:1883"]; hist [label="Historian"];
        sandbox [label="Sandbox node\\n(sample executes here)", fillcolor="#3a1d1d"];
        zeek [label="Zeek monitor"];
        hmi -> plc [label="Modbus"]; s -> broker [label="MQTT"]; broker -> hist;
        sandbox -> plc [style=dashed, label="?"]; sandbox -> broker [style=dashed, label="?"];
        zeek -> hmi [dir=none, style=dotted]; zeek -> sandbox [dir=none, style=dotted];
      }
      up [label="Uploaded binary\\n(Binary Upload page)"];
      dyn [label="M9 Dynamic Analysis", fillcolor="#1d2a3a"];
      fus [label="M10 Hybrid Verdict", fillcolor="#1d2a3a"];
      cnn [label="M5 CNN (static)", fillcolor="#1d2a3a"];
      up -> sandbox [label="copy in"]; up -> cnn [label="image"];
      zeek -> dyn [label="conn.log"]; dyn -> fus; cnn -> fus;
    }
    """)

    section_header("Planned nodes")
    st.dataframe(pd.DataFrame(PLANNED_NODES, columns=['Container', 'Role', 'Implementation', 'Port']),
                 hide_index=True, use_container_width=True)

    section_header("Interface contract",
                   "What the twin must produce so the rest of MalTwin works unchanged")
    st.markdown(
        "- **Output:** one Zeek `conn.log` per run, covering the sample's execution window "
        "(same format as IoT-23, so the trained dynamic model applies directly).\n"
        "- **Metadata:** SHA-256 of the executed sample, start/stop time, node inventory — "
        "used to pair the capture with the static result automatically.\n"
        "- **Isolation (SRS CON-2, SEC-2):** internal-only Docker network, no host volume "
        "mounts for the sandbox, CPU/memory limits, automatic teardown after each run.\n"
        "- **Control (SRS FR2.1–FR2.4):** start/stop from this page, live traffic log, "
        "node status panel, protocol mix chart."
    )
    st.caption("Mininet emulation from the original proposal needs a native Linux host with root; "
               "the plan uses Docker networking alone so the twin also runs on Windows "
               "(WSL2 + Docker Desktop). Mininet remains an optional extension.")

    section_header("Deployment prerequisites (checked live)")
    status, detail = _docker_status()
    badge = {'ok': 'active', 'stopped': 'inactive', 'missing': 'error'}[status]
    st.markdown(status_badge(badge, detail), unsafe_allow_html=True)
    dyn_ok = config.DYNAMIC_MODEL_PATH.exists()
    st.markdown(status_badge('active' if dyn_ok else 'inactive',
                             'Dynamic model trained' if dyn_ok else
                             'Dynamic model not trained yet (scripts/train_dynamic.py)'),
                unsafe_allow_html=True)
    zeek = shutil.which('zeek')
    st.markdown(status_badge('active' if zeek else 'inactive',
                             f'Zeek found at {zeek}' if zeek else
                             'Zeek not on host (it will run inside its own container)'),
                unsafe_allow_html=True)
