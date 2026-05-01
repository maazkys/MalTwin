# modules/reporting/mitre_mapper.py
"""
MITRE ATT&CK for ICS mapping loader and query helper.

Reads from config.MITRE_JSON_PATH (data/mitre_ics_mapping.json).
Never raises — returns empty dict on any failure.

JSON schema expected:
{
  "FamilyName": {
    "tactics":    ["Initial Access", "Execution", ...],
    "techniques": [
      {"id": "T0817", "name": "Drive-by Compromise"},
      ...
    ],
    "description": "Brief family description for the PDF report."
  },
  ...
}
"""
import json
import sys
from pathlib import Path
import config


def load_mitre_db(path: Path = config.MITRE_JSON_PATH) -> dict:
    """
    Load the full MITRE mapping database from JSON.
    Returns empty dict on FileNotFoundError or parse error.
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[MalTwin] MITRE mapping file not found: {path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"[MalTwin] MITRE mapping JSON parse error: {e}", file=sys.stderr)
        return {}


def get_mitre_mapping(family_name: str, db: dict | None = None) -> dict:
    """
    Look up MITRE ATT&CK for ICS data for a single malware family.

    Args:
        family_name: predicted family string e.g. 'Allaple.A'
        db:          pre-loaded DB dict (optional — loads fresh if None)

    Returns:
        {
            'tactics':     list[str],
            'techniques':  list[dict],   # [{'id': str, 'name': str}, ...]
            'description': str,
            'found':       bool,
        }
    """
    if db is None:
        db = load_mitre_db()

    entry = db.get(family_name, {})
    return {
        'tactics':     entry.get('tactics', []),
        'techniques':  entry.get('techniques', []),
        'description': entry.get('description', ''),
        'found':       bool(entry),
    }
