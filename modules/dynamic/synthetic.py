# modules/dynamic/synthetic.py
"""
Synthetic Zeek conn.log generator.

Purpose: unit tests and offline demos when IoT-23 is not available.
Each behaviour class is generated from simple, documented traffic templates
that mirror how the real behaviour looks in a conn.log:

    Benign       DNS lookups, HTTPS/HTTP sessions, MQTT telemetry, Modbus polls
    PortScan     SYNs to many hosts on one port, no reply (state S0)
    C&C          small periodic TCP sessions to one server (IRC/HTTP)
    DDoS         high-rate UDP/TCP floods towards one victim
    Attack       Telnet brute-force-style short sessions, rejected/reset
    FileDownload single TCP session with a large response payload
    Okiru        Okiru/Mirai-style Telnet propagation scans

Records generated here are SYNTHETIC. They are never presented as real
detections — the dashboard labels any synthetic capture clearly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .zeek_parser import ZEEK_CONN_FIELDS


def _ip(rng, prefix='192.168.1.'):
    return prefix + str(int(rng.integers(2, 254)))


def _row(rng, ts, proto, port, service, duration, ob, rb, state, history,
         op, rp, label, detail):
    return {
        'ts': ts, 'uid': 'C' + ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 17)),
        'id.orig_h': '192.168.100.' + str(int(rng.integers(2, 50))),
        'id.orig_p': int(rng.integers(1024, 65535)),
        'id.resp_h': _ip(rng, '10.0.' + str(int(rng.integers(0, 255))) + '.'),
        'id.resp_p': port, 'proto': proto, 'service': service,
        'duration': duration, 'orig_bytes': ob, 'resp_bytes': rb,
        'conn_state': state, 'local_orig': '-', 'local_resp': '-',
        'missed_bytes': 0, 'history': history,
        'orig_pkts': op, 'orig_ip_bytes': ob + 40 * op,
        'resp_pkts': rp, 'resp_ip_bytes': rb + 40 * rp,
        'tunnel_parents': '-', 'label': label, 'detailed-label': detail,
    }


def _benign(rng, ts):
    kind = rng.choice(['dns', 'https', 'http', 'mqtt', 'modbus'], p=[.35, .25, .1, .15, .15])
    if kind == 'dns':
        return _row(rng, ts, 'udp', 53, 'dns', rng.uniform(.001, .1), int(rng.integers(30, 80)),
                    int(rng.integers(60, 300)), 'SF', 'Dd', 1, 1, 'benign', '-')
    if kind == 'https':
        ob, rb = int(rng.integers(500, 4000)), int(rng.integers(2000, 60000))
        return _row(rng, ts, 'tcp', 443, 'ssl', rng.uniform(.2, 20), ob, rb, 'SF',
                    'ShADadFf', int(rng.integers(8, 40)), int(rng.integers(8, 60)), 'benign', '-')
    if kind == 'http':
        ob, rb = int(rng.integers(200, 1500)), int(rng.integers(500, 20000))
        return _row(rng, ts, 'tcp', 80, 'http', rng.uniform(.1, 5), ob, rb, 'SF',
                    'ShADadFf', int(rng.integers(5, 20)), int(rng.integers(5, 30)), 'benign', '-')
    if kind == 'mqtt':
        return _row(rng, ts, 'tcp', 1883, '-', rng.uniform(30, 600), int(rng.integers(200, 3000)),
                    int(rng.integers(50, 500)), 'SF', 'ShADadFf',
                    int(rng.integers(10, 80)), int(rng.integers(10, 80)), 'benign', '-')
    return _row(rng, ts, 'tcp', 502, '-', rng.uniform(.05, 2), int(rng.integers(12, 60)),
                int(rng.integers(12, 260)), 'SF', 'ShADadFf', int(rng.integers(4, 10)),
                int(rng.integers(4, 10)), 'benign', '-')


def _malicious(rng, ts, behaviour):
    if behaviour == 'PortScan':
        return _row(rng, ts, 'tcp', int(rng.choice([23, 80, 8080, 22, 2323])), '-', np.nan,
                    0, 0, 'S0', 'S', 1, 0, 'Malicious', 'PartOfAHorizontalPortScan')
    if behaviour == 'C&C':
        return _row(rng, ts, 'tcp', int(rng.choice([6667, 80, 443, 1883])),
                    str(rng.choice(['irc', 'http', '-'])), rng.uniform(.5, 3),
                    int(rng.integers(20, 120)), int(rng.integers(20, 200)), 'SF', 'ShADadFf',
                    int(rng.integers(3, 8)), int(rng.integers(3, 8)), 'Malicious', 'C&C-HeartBeat')
    if behaviour == 'DDoS':
        proto = str(rng.choice(['udp', 'tcp']))
        op = int(rng.integers(200, 5000))
        return _row(rng, ts, proto, int(rng.choice([80, 53, 502, 123])), '-', rng.uniform(1, 30),
                    op * int(rng.integers(30, 512)), 0, 'S0' if proto == 'tcp' else 'OTH',
                    'S' if proto == 'tcp' else 'D', op, 0, 'Malicious', 'DDoS')
    if behaviour == 'Attack':
        return _row(rng, ts, 'tcp', int(rng.choice([23, 2323, 22])), '-', rng.uniform(.01, .5),
                    int(rng.integers(0, 60)), int(rng.integers(0, 40)),
                    str(rng.choice(['REJ', 'RSTO', 'RSTR'])), 'ShR',
                    int(rng.integers(1, 4)), int(rng.integers(1, 3)), 'Malicious', 'Attack')
    if behaviour == 'FileDownload':
        rb = int(rng.integers(50_000, 2_000_000))
        return _row(rng, ts, 'tcp', int(rng.choice([80, 8080, 443])), 'http', rng.uniform(1, 60),
                    int(rng.integers(100, 400)), rb, 'SF', 'ShADadFf',
                    int(rng.integers(20, 200)), rb // 1400, 'Malicious', 'C&C-FileDownload')
    if behaviour == 'Okiru':
        return _row(rng, ts, 'tcp', int(rng.choice([23, 2323, 37215, 52869])), '-', np.nan,
                    0, 0, str(rng.choice(['S0', 'REJ'])), str(rng.choice(['S', 'Sr'])),
                    int(rng.integers(1, 3)), 0, 'Malicious', 'Okiru')
    raise ValueError(f"Unknown behaviour: {behaviour}")


def generate_flows(counts: dict[str, int], seed: int = 42, start_ts: float = 1.6e9) -> pd.DataFrame:
    """
    Generate a labelled synthetic conn.log DataFrame.

    Args:
        counts: {behaviour_class: number_of_flows}, e.g. {'Benign': 500, 'PortScan': 100}
    """
    rng = np.random.default_rng(seed)
    rows, ts = [], start_ts
    for beh, n in counts.items():
        for _ in range(int(n)):
            ts += float(rng.exponential(0.5))
            rows.append(_benign(rng, ts) if beh == 'Benign' else _malicious(rng, ts, beh))
    df = pd.DataFrame(rows, columns=ZEEK_CONN_FIELDS + ['label', 'detailed-label'])
    return df.sample(frac=1.0, random_state=seed).sort_values('ts').reset_index(drop=True)
