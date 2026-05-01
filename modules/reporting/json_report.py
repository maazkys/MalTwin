# modules/reporting/json_report.py
"""
Structured JSON forensic report builder.

SRS refs: M8 FE-1, FE-2, FE-3, UC-03
"""
import json
from datetime import datetime
from pathlib import Path

import config
from .mitre_mapper import get_mitre_mapping, load_mitre_db


def generate_json_report(report_data: dict) -> bytes:
    """
    Build a structured JSON forensic report from a report_data dict.

    Args:
        report_data: assembled by build_report_data() in detection.py.
            Required keys:
                file_name, sha256, file_format, file_size_bytes, upload_time,
                predicted_family, confidence, top3, all_probabilities,
                gradcam (dict with 'generated': bool),
                mitre (dict from get_mitre_mapping())

    Returns:
        UTF-8 encoded JSON bytes — always returns bytes even on error
        (returns a minimal error JSON on failure).
    """
    try:
        mitre = report_data.get('mitre', {})
        gradcam = report_data.get('gradcam', {'generated': False})

        report = {
            'report_metadata': {
                'generator':       'MalTwin Forensic Reporting Module',
                'report_version':  '1.0',
                'generated_at':    datetime.utcnow().isoformat(),
                'framework':       'MalTwin v1.0 — COMSATS University Islamabad',
            },
            'file_information': {
                'file_name':       report_data['file_name'],
                'sha256':          report_data['sha256'],
                'file_format':     report_data['file_format'],
                'file_size_bytes': report_data['file_size_bytes'],
                'upload_time':     report_data['upload_time'],
            },
            'detection_result': {
                'predicted_family': report_data['predicted_family'],
                'confidence':       round(float(report_data['confidence']), 6),
                'confidence_pct':   round(float(report_data['confidence']) * 100, 2),
                'top3_predictions': [
                    {
                        'rank':       i + 1,
                        'family':     pred['family'],
                        'confidence': round(float(pred['confidence']), 6),
                    }
                    for i, pred in enumerate(report_data.get('top3', []))
                ],
                'all_class_probabilities': {
                    k: round(float(v), 6)
                    for k, v in report_data.get('all_probabilities', {}).items()
                },
            },
            'mitre_attack_ics': {
                'family':        report_data['predicted_family'],
                'mapping_found': mitre.get('found', False),
                'description':   mitre.get('description', ''),
                'tactics':       mitre.get('tactics', []),
                'techniques':    mitre.get('techniques', []),
                'framework':     'MITRE ATT&CK for ICS',
                'reference':     'https://attack.mitre.org/matrices/ics/',
            },
            'explainability': {
                'gradcam_generated': gradcam.get('generated', False),
                'target_class':      gradcam.get('target_class', None),
                'layer':             gradcam.get('layer', None),
                'method':            'Captum LayerGradCam' if gradcam.get('generated') else None,
            },
        }

        return json.dumps(report, indent=2, ensure_ascii=False).encode('utf-8')

    except Exception as e:
        error_report = {
            'error':   'Report generation failed',
            'reason':  str(e),
            'partial': {
                'file_name':        report_data.get('file_name', 'unknown'),
                'sha256':           report_data.get('sha256', 'unknown'),
                'predicted_family': report_data.get('predicted_family', 'unknown'),
            },
        }
        return json.dumps(error_report, indent=2).encode('utf-8')


def save_json_report(report_bytes: bytes, sha256: str) -> Path:
    """
    Save JSON report bytes to config.REPORTS_DIR.

    Args:
        report_bytes: output of generate_json_report()
        sha256:       full SHA-256 hex string of the analyzed file

    Returns:
        Path to the saved file.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"maltwin_report_{sha256[:8]}.json"
    out_path = config.REPORTS_DIR / filename
    out_path.write_bytes(report_bytes)
    return out_path
