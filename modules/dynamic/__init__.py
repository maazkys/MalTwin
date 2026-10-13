"""
Module 9 (extension) — Dynamic behavioural analysis of IIoT network traffic.

Input:  Zeek conn.log (IoT-23 today; Digital Twin captures in Evaluation 4)
Output: per-flow behaviour classes and a capture-level malicious score,
        explained via feature importance, mapped to MITRE ATT&CK for ICS.
"""
from .zeek_parser import parse_conn_log, load_conn_logs, map_behaviour, BEHAVIOUR_CLASSES
from .features import extract_flow_features, feature_names
from .detector import DynamicDetector, DynamicReport

__all__ = [
    'parse_conn_log', 'load_conn_logs', 'map_behaviour', 'BEHAVIOUR_CLASSES',
    'extract_flow_features', 'feature_names', 'DynamicDetector', 'DynamicReport',
]
