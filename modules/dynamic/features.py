# modules/dynamic/features.py
"""
Flow-level behavioural feature extraction from parsed Zeek conn.log records.

Design choices
  - IP addresses are deliberately NOT used as features. In IoT-23 every
    capture has its own infected host IP, so an IP-based model would learn
    "this IP is bad" rather than "this behaviour is bad" and would not
    transfer to the Digital Twin's address space.
  - Destination ports are reduced to IIoT-relevant indicator flags
    (Modbus 502, MQTT 1883/8883, Telnet 23/2323 used by Mirai-family
    botnets, IRC 6667 C&C, etc.) plus a well-known/registered/ephemeral
    bucket, instead of a raw port number that trees would over-split.
  - Zeek's `history` string (connection state transitions) is summarised
    into counts of handshake / reset / data flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROTOS = ['tcp', 'udp', 'icmp']
SERVICES = ['dns', 'http', 'ssl', 'dhcp', 'irc', 'ssh', 'telnet', 'modbus', 'mqtt']
CONN_STATES = ['S0', 'S1', 'SF', 'REJ', 'S2', 'S3', 'RSTO', 'RSTR',
               'RSTOS0', 'RSTRH', 'SH', 'SHR', 'OTH']

# Ports with known IIoT / IoT-malware relevance
PORT_FLAGS = {
    'port_modbus':  {502},
    'port_mqtt':    {1883, 8883},
    'port_telnet':  {23, 2323},
    'port_http':    {80, 8080, 8000, 8081},
    'port_https':   {443, 8443},
    'port_dns':     {53},
    'port_ssh':     {22},
    'port_irc':     {6667, 6697},
}

BASE_NUMERIC = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']


def _history_counts(h: pd.Series) -> pd.DataFrame:
    h = h.fillna('').astype(str).replace('-', '')
    return pd.DataFrame({
        'hist_len':       h.str.len(),
        'hist_syn':       h.str.count('S') + h.str.count('s'),
        'hist_synack':    h.str.count('h') + h.str.count('H'),
        'hist_data':      h.str.count('D') + h.str.count('d'),
        'hist_reset':     h.str.count('R') + h.str.count('r'),
        'hist_fin':       h.str.count('F') + h.str.count('f'),
        'hist_retrans':   h.str.count('T') + h.str.count('t'),
    })


def extract_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn parsed conn.log rows into a purely numeric feature matrix.

    Returns a DataFrame with the same row order as `df` and the columns
    listed by `feature_names()`. All NaNs are filled with 0.
    """
    out = pd.DataFrame(index=df.index)
    for c in BASE_NUMERIC:
        out[c] = pd.to_numeric(df.get(c, 0), errors='coerce')
    out = out.fillna(0.0)

    # Log-scaled versions dampen heavy-tailed byte counts
    for c in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']:
        out[f'log_{c}'] = np.log1p(out[c].clip(lower=0))

    total_bytes = out['orig_bytes'] + out['resp_bytes']
    total_pkts = out['orig_pkts'] + out['resp_pkts']
    out['bytes_ratio'] = out['orig_bytes'] / (total_bytes + 1.0)
    out['pkts_ratio'] = out['orig_pkts'] / (total_pkts + 1.0)
    out['bytes_per_pkt'] = (out['orig_ip_bytes'] + out['resp_ip_bytes']) / (total_pkts + 1.0)
    out['no_response'] = ((out['resp_pkts'] == 0) & (out['resp_bytes'] == 0)).astype(float)

    port = pd.to_numeric(df.get('id.resp_p', np.nan), errors='coerce')
    for name, ports in PORT_FLAGS.items():
        out[name] = port.isin(ports).astype(float)
    out['port_wellknown'] = (port < 1024).astype(float)
    out['port_registered'] = ((port >= 1024) & (port < 49152)).astype(float)
    out['port_ephemeral'] = (port >= 49152).astype(float)

    proto = df.get('proto', pd.Series('', index=df.index)).astype(str).str.lower()
    for p in PROTOS:
        out[f'proto_{p}'] = (proto == p).astype(float)

    service = df.get('service', pd.Series('', index=df.index)).astype(str).str.lower()
    for s in SERVICES:
        out[f'svc_{s}'] = service.str.contains(s, regex=False).astype(float)
    out['svc_none'] = service.isin(['-', '', 'nan']).astype(float)

    state = df.get('conn_state', pd.Series('', index=df.index)).astype(str)
    for s in CONN_STATES:
        out[f'state_{s}'] = (state == s).astype(float)

    hist = _history_counts(df.get('history', pd.Series('', index=df.index)))
    hist.index = out.index
    out = pd.concat([out, hist.astype(float)], axis=1)

    return out[feature_names()].astype(float).fillna(0.0)


def feature_names() -> list[str]:
    names = list(BASE_NUMERIC)
    names += [f'log_{c}' for c in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']]
    names += ['bytes_ratio', 'pkts_ratio', 'bytes_per_pkt', 'no_response']
    names += list(PORT_FLAGS) + ['port_wellknown', 'port_registered', 'port_ephemeral']
    names += [f'proto_{p}' for p in PROTOS]
    names += [f'svc_{s}' for s in SERVICES] + ['svc_none']
    names += [f'state_{s}' for s in CONN_STATES]
    names += ['hist_len', 'hist_syn', 'hist_synack', 'hist_data',
              'hist_reset', 'hist_fin', 'hist_retrans']
    return names


# Human-readable descriptions for the dashboard / reports
FEATURE_DESCRIPTIONS = {
    'duration': 'Connection duration (s)',
    'orig_bytes': 'Bytes sent by the device',
    'resp_bytes': 'Bytes received by the device',
    'no_response': 'Connection got no reply (typical of scanning)',
    'port_telnet': 'Destination is Telnet 23/2323 (Mirai-style propagation)',
    'port_modbus': 'Destination is Modbus TCP 502',
    'port_mqtt': 'Destination is MQTT 1883/8883',
    'port_irc': 'Destination is IRC 6667 (classic botnet C&C)',
    'state_S0': 'SYN sent, never answered',
    'state_REJ': 'Connection rejected',
    'state_SF': 'Normal established-and-closed connection',
    'proto_udp': 'UDP flow',
    'proto_tcp': 'TCP flow',
    'hist_reset': 'Number of TCP resets',
    'bytes_per_pkt': 'Average bytes per packet',
    'log_orig_pkts': 'Packets sent (log-scaled)',
    'log_resp_pkts': 'Packets received (log-scaled)',
    'log_orig_bytes': 'Bytes sent (log-scaled)',
    'log_resp_bytes': 'Bytes received (log-scaled)',
    'log_duration': 'Connection duration (log-scaled)',
    'missed_bytes': 'Bytes missed by the sensor (gaps)',
    'orig_pkts': 'Packets sent by the device',
    'resp_pkts': 'Packets received by the device',
    'orig_ip_bytes': 'IP-layer bytes sent',
    'resp_ip_bytes': 'IP-layer bytes received',
    'bytes_ratio': 'Share of bytes sent by the device (1 = one-way)',
    'pkts_ratio': 'Share of packets sent by the device (1 = one-way)',
    'port_http': 'Destination is HTTP 80/8080',
    'port_https': 'Destination is HTTPS 443/8443',
    'port_dns': 'Destination is DNS 53',
    'port_ssh': 'Destination is SSH 22',
    'port_wellknown': 'Destination port < 1024',
    'port_registered': 'Destination port 1024–49151',
    'port_ephemeral': 'Destination port ≥ 49152 (unusual for servers)',
    'proto_icmp': 'ICMP flow',
    'svc_none': 'No application protocol recognised',
    'state_S1': 'Established, not terminated',
    'state_RSTO': 'Device reset the connection',
    'state_RSTR': 'Remote side reset the connection',
    'state_OTH': 'No SYN seen (mid-stream / non-TCP)',
    'state_SH': 'SYN then FIN, no SYN-ACK (half-open probe)',
    'hist_len': 'Length of TCP state history',
    'hist_syn': 'Number of SYN packets',
    'hist_synack': 'Number of SYN-ACK packets',
    'hist_data': 'Number of data-carrying packets',
    'hist_fin': 'Number of FIN packets',
    'hist_retrans': 'Number of retransmissions',
}
for _f in feature_names():
    FEATURE_DESCRIPTIONS.setdefault(_f, _f.replace('_', ' '))
