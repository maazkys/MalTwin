# MalTwin — Implementation Step 2: M8 Automated Forensic Reporting
### SRS refs: FR5.6, FR-B4, UC-03, M8 FE-1 through FE-5

> Complete Step 1 (Grad-CAM) and verify `pytest tests/test_gradcam.py -v` passes
> before starting this step. The PDF pipeline embeds the heatmap overlay PNG
> produced by Step 1 — it depends on `generate_gradcam` being correct.

---

## What This Step Delivers

| Item | Status before | Status after |
|---|---|---|
| `modules/reporting/__init__.py` | Does not exist | Package exports |
| `modules/reporting/pdf_report.py` | Does not exist | Full FPDF2 PDF generator |
| `modules/reporting/json_report.py` | Does not exist | Structured JSON report builder |
| `modules/reporting/mitre_mapper.py` | Does not exist | MITRE lookup + full 25-family seed |
| `data/mitre_ics_mapping.json` | Partial / missing families | All 25 Malimg families mapped |
| `modules/dashboard/pages/detection.py` | PDF button disabled, JSON minimal | Both buttons active, full report pipeline |
| `tests/test_reporting.py` | Does not exist | Full test suite |

---

## Mandatory Rules

- **FPDF2** (`fpdf2` package, imported as `from fpdf import FPDF`) — not the older `fpdf` package. The API differs.
- PDF generation **never raises** — all exceptions caught, fallback to JSON communicated to caller.
- `mitre_mapper.py` reads from `config.MITRE_JSON_PATH` — never hardcodes the path.
- The MITRE JSON file uses family name as key, exactly matching the `class_names` strings from training.
- PDF embeds the Grad-CAM overlay PNG **only if** `heatmap_data` is not None — gracefully skips if XAI was not generated.
- All `generate_*` functions accept a `report_data` dict assembled by the dashboard — they do not reach into `st.session_state` directly.
- JSON report values must all be JSON-serialisable — no numpy types, no PIL objects.
- `save_report()` creates `config.REPORTS_DIR` if it does not exist.
- Report filenames are based on the first 8 chars of SHA-256 to avoid collisions.

---

## New config entries needed

Add these to `config.py` if not already present:

```python
# config.py additions
REPORTS_DIR   = BASE_DIR / 'data' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
```

`MITRE_JSON_PATH` and `PROCESSED_DIR` should already exist from Phase 6.

---

## File 1: `modules/reporting/__init__.py`

```python
# modules/reporting/__init__.py
from .pdf_report import generate_pdf_report
from .json_report import generate_json_report
from .mitre_mapper import get_mitre_mapping, load_mitre_db
```

---

## File 2: `modules/reporting/mitre_mapper.py`

```python
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
```

---

## File 3: `data/mitre_ics_mapping.json`

Create this file at `data/mitre_ics_mapping.json`. This is the complete mapping for all 25 Malimg families to their closest MITRE ATT&CK for ICS tactics and techniques.

```json
{
  "Adialer.C": {
    "description": "Dialer trojan that establishes unauthorized premium-rate phone connections. Targets Windows systems.",
    "tactics": ["Initial Access", "Execution", "Command and Control"],
    "techniques": [
      {"id": "T0866", "name": "Exploitation of Remote Services"},
      {"id": "T0858", "name": "Change Operating Mode"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "Agent.FYI": {
    "description": "Remote access trojan providing backdoor access and data exfiltration capabilities.",
    "tactics": ["Persistence", "Command and Control", "Collection"],
    "techniques": [
      {"id": "T0859", "name": "Valid Accounts"},
      {"id": "T0884", "name": "Connection Proxy"},
      {"id": "T0811", "name": "Data from Local System"}
    ]
  },
  "Allaple.A": {
    "description": "Highly prevalent network worm with aggressive self-propagation via network shares and exploits.",
    "tactics": ["Lateral Movement", "Discovery", "Command and Control"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0840", "name": "Network Connection Enumeration"},
      {"id": "T0885", "name": "Commonly Used Port"}
    ]
  },
  "Allaple.L": {
    "description": "Variant of the Allaple worm family with modified propagation and obfuscation techniques.",
    "tactics": ["Lateral Movement", "Defense Evasion", "Command and Control"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0858", "name": "Change Operating Mode"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "Alueron.gen!J": {
    "description": "Generic detection for the Alueron rootkit family. Hides processes and network connections.",
    "tactics": ["Defense Evasion", "Persistence", "Discovery"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0842", "name": "Network Sniffing"}
    ]
  },
  "Autorun.K": {
    "description": "Worm spreading via removable media autorun functionality.",
    "tactics": ["Initial Access", "Lateral Movement", "Execution"],
    "techniques": [
      {"id": "T0847", "name": "Replication Through Removable Media"},
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0859", "name": "Valid Accounts"}
    ]
  },
  "BrowseFox.B": {
    "description": "Adware/browser hijacker that modifies browser settings and injects ads.",
    "tactics": ["Execution", "Collection", "Command and Control"],
    "techniques": [
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0811", "name": "Data from Local System"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "Dialplatform.B": {
    "description": "Dialer platform malware that establishes covert outbound connections to premium services.",
    "tactics": ["Command and Control", "Exfiltration", "Impact"],
    "techniques": [
      {"id": "T0884", "name": "Connection Proxy"},
      {"id": "T0830", "name": "Adversary-in-the-Middle"},
      {"id": "T0831", "name": "Manipulation of Control"}
    ]
  },
  "Dontovo.A": {
    "description": "Trojan dropper that installs additional malware components on infected systems.",
    "tactics": ["Execution", "Persistence", "Defense Evasion"],
    "techniques": [
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0820", "name": "Exploitation for Evasion"}
    ]
  },
  "Fakerean": {
    "description": "Rogue security software displaying false alerts to extort payment from victims.",
    "tactics": ["Impact", "Execution", "Command and Control"],
    "techniques": [
      {"id": "T0831", "name": "Manipulation of Control"},
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0885", "name": "Commonly Used Port"}
    ]
  },
  "Instantaccess": {
    "description": "Adware that provides instant access to premium online content through unauthorized billing.",
    "tactics": ["Execution", "Command and Control", "Impact"],
    "techniques": [
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0884", "name": "Connection Proxy"},
      {"id": "T0831", "name": "Manipulation of Control"}
    ]
  },
  "Lolyda.AA1": {
    "description": "Password stealer targeting browser credentials, email clients, and FTP applications.",
    "tactics": ["Credential Access", "Collection", "Exfiltration"],
    "techniques": [
      {"id": "T0819", "name": "Exploit Public-Facing Application"},
      {"id": "T0811", "name": "Data from Local System"},
      {"id": "T0830", "name": "Adversary-in-the-Middle"}
    ]
  },
  "Lolyda.AA2": {
    "description": "Variant of Lolyda password stealer with enhanced credential harvesting.",
    "tactics": ["Credential Access", "Collection", "Exfiltration"],
    "techniques": [
      {"id": "T0819", "name": "Exploit Public-Facing Application"},
      {"id": "T0811", "name": "Data from Local System"},
      {"id": "T0842", "name": "Network Sniffing"}
    ]
  },
  "Lolyda.AA3": {
    "description": "Third variant of Lolyda with added persistence and anti-analysis techniques.",
    "tactics": ["Credential Access", "Persistence", "Defense Evasion"],
    "techniques": [
      {"id": "T0819", "name": "Exploit Public-Facing Application"},
      {"id": "T0859", "name": "Valid Accounts"},
      {"id": "T0820", "name": "Exploitation for Evasion"}
    ]
  },
  "Lolyda.AT": {
    "description": "Lolyda variant targeting AT-class systems with enhanced lateral movement.",
    "tactics": ["Lateral Movement", "Credential Access", "Collection"],
    "techniques": [
      {"id": "T0866", "name": "Exploitation of Remote Services"},
      {"id": "T0819", "name": "Exploit Public-Facing Application"},
      {"id": "T0811", "name": "Data from Local System"}
    ]
  },
  "Malex.gen!J": {
    "description": "Generic Malex trojan with polymorphic code generation and rootkit capabilities.",
    "tactics": ["Defense Evasion", "Persistence", "Command and Control"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "Obfuscator.AD": {
    "description": "Obfuscated malware loader that decrypts and executes embedded payloads at runtime.",
    "tactics": ["Defense Evasion", "Execution", "Persistence"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0839", "name": "Module Firmware"}
    ]
  },
  "Rbot!gen": {
    "description": "IRC-controlled botnet client with DDoS, spam relay, and credential theft capabilities.",
    "tactics": ["Command and Control", "Impact", "Lateral Movement"],
    "techniques": [
      {"id": "T0885", "name": "Commonly Used Port"},
      {"id": "T0814", "name": "Denial of Service"},
      {"id": "T0812", "name": "Default Credentials"}
    ]
  },
  "Skintrim.N": {
    "description": "Trojan that modifies system settings and installs additional malware components silently.",
    "tactics": ["Persistence", "Defense Evasion", "Execution"],
    "techniques": [
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0863", "name": "User Execution"}
    ]
  },
  "Swizzor.gen!E": {
    "description": "Polymorphic downloader that fetches and installs additional malware from remote servers.",
    "tactics": ["Command and Control", "Execution", "Defense Evasion"],
    "techniques": [
      {"id": "T0884", "name": "Connection Proxy"},
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0820", "name": "Exploitation for Evasion"}
    ]
  },
  "Swizzor.gen!I": {
    "description": "Variant of Swizzor downloader with improved anti-analysis and traffic encryption.",
    "tactics": ["Command and Control", "Defense Evasion", "Execution"],
    "techniques": [
      {"id": "T0884", "name": "Connection Proxy"},
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0858", "name": "Change Operating Mode"}
    ]
  },
  "VB.AT": {
    "description": "Visual Basic-compiled trojan with keylogging and remote shell capabilities.",
    "tactics": ["Collection", "Command and Control", "Exfiltration"],
    "techniques": [
      {"id": "T0811", "name": "Data from Local System"},
      {"id": "T0885", "name": "Commonly Used Port"},
      {"id": "T0830", "name": "Adversary-in-the-Middle"}
    ]
  },
  "Wintrim.BX": {
    "description": "Trimmer malware that strips system protections and installs persistent backdoors.",
    "tactics": ["Defense Evasion", "Persistence", "Impact"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0831", "name": "Manipulation of Control"}
    ]
  },
  "Yuner.A": {
    "description": "Worm with fast self-replication via network drives and email attachment propagation.",
    "tactics": ["Lateral Movement", "Initial Access", "Execution"],
    "techniques": [
      {"id": "T0847", "name": "Replication Through Removable Media"},
      {"id": "T0866", "name": "Exploitation of Remote Services"},
      {"id": "T0863", "name": "User Execution"}
    ]
  },
  "Zlob.gen!D": {
    "description": "Generic Zlob downloader variant. Poses as a media codec, downloads additional payloads.",
    "tactics": ["Initial Access", "Execution", "Command and Control"],
    "techniques": [
      {"id": "T0863", "name": "User Execution"},
      {"id": "T0866", "name": "Exploitation of Remote Services"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  }
}
```

---

## File 4: `modules/reporting/json_report.py`

```python
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
                'family':      report_data['predicted_family'],
                'mapping_found': mitre.get('found', False),
                'description': mitre.get('description', ''),
                'tactics':     mitre.get('tactics', []),
                'techniques':  mitre.get('techniques', []),
                'framework':   'MITRE ATT&CK for ICS',
                'reference':   'https://attack.mitre.org/matrices/ics/',
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
```

---

## File 5: `modules/reporting/pdf_report.py`

```python
# modules/reporting/pdf_report.py
"""
PDF forensic report generator using FPDF2.

SRS refs: M8 FE-1, FE-2, FE-3, UC-03

Uses fpdf2 (imported as `from fpdf import FPDF`).
Never raises — returns None on failure so caller can fall back to JSON.

Layout (A4, portrait):
  Page 1: Header, File Information, Detection Result, Confidence bar
  Page 2: MITRE ATT&CK for ICS mapping
  Page 3: Top-3 Predictions table + Grad-CAM heatmap (if available)
"""
import io
import sys
import tempfile
import os
from datetime import datetime
from pathlib import Path

import config

try:
    from fpdf import FPDF
    FPDF2_AVAILABLE = True
except ImportError:
    FPDF2_AVAILABLE = False


class _MalTwinPDF(FPDF):
    """Custom FPDF subclass with header and footer."""

    def header(self):
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(30, 30, 50)      # dark navy
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, '  MalTwin — IIoT Malware Detection Framework', fill=True, ln=True)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f'  Forensic Analysis Report | Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC', ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | COMSATS University Islamabad | BS Cyber Security 2023-2027', align='C')

    def section_title(self, title: str):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(240, 240, 248)
        self.set_text_color(30, 30, 80)
        self.cell(0, 8, f'  {title}', fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, label: str, value: str, label_width: int = 55):
        self.set_font('Helvetica', 'B', 9)
        self.cell(label_width, 6, label, border='B')
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 6, value, border='B')

    def confidence_bar(self, confidence: float):
        """Draw a colored confidence bar (green/amber/red)."""
        bar_w = 140
        bar_h = 8
        x, y  = self.get_x(), self.get_y()

        # Background
        self.set_fill_color(220, 220, 220)
        self.rect(x, y, bar_w, bar_h, 'F')

        # Fill color
        if confidence >= config.CONFIDENCE_GREEN:
            self.set_fill_color(50, 180, 80)   # green
        elif confidence >= config.CONFIDENCE_AMBER:
            self.set_fill_color(230, 160, 30)  # amber
        else:
            self.set_fill_color(210, 50, 50)   # red

        fill_w = bar_w * confidence
        self.rect(x, y, fill_w, bar_h, 'F')

        # Label
        self.set_xy(x + bar_w + 3, y)
        self.set_font('Helvetica', 'B', 9)
        self.cell(30, bar_h, f'{confidence * 100:.1f}%')
        self.ln(bar_h + 2)


def generate_pdf_report(report_data: dict) -> bytes | None:
    """
    Generate a complete PDF forensic report.

    Args:
        report_data: same dict used by generate_json_report(). Keys:
            file_name, sha256, file_format, file_size_bytes, upload_time,
            predicted_family, confidence, top3, all_probabilities,
            gradcam (dict), mitre (dict from get_mitre_mapping())

    Returns:
        PDF bytes on success.
        None on failure (caller falls back to JSON and notifies user).
    """
    if not FPDF2_AVAILABLE:
        print("[MalTwin] fpdf2 not installed — PDF generation unavailable.", file=sys.stderr)
        return None

    try:
        pdf = _MalTwinPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ── Page 1: File Info + Detection Result ──────────────────────────────
        pdf.section_title('File Information')
        pdf.kv_row('File Name:',    report_data['file_name'])
        pdf.kv_row('File Format:',  report_data['file_format'])
        pdf.kv_row('File Size:',    f"{report_data['file_size_bytes']:,} bytes")
        pdf.kv_row('Upload Time:',  report_data['upload_time'])
        sha = report_data['sha256']
        pdf.kv_row('SHA-256 (1/2):', sha[:32])
        pdf.kv_row('SHA-256 (2/2):', sha[32:])
        pdf.ln(5)

        pdf.section_title('Detection Result')
        confidence = float(report_data['confidence'])
        family     = report_data['predicted_family']

        pdf.set_font('Helvetica', 'B', 16)
        if confidence >= config.CONFIDENCE_GREEN:
            pdf.set_text_color(50, 180, 80)
        elif confidence >= config.CONFIDENCE_AMBER:
            pdf.set_text_color(230, 160, 30)
        else:
            pdf.set_text_color(210, 50, 50)
        pdf.cell(0, 10, f'Predicted Family: {family}', ln=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, 'Confidence:', ln=True)
        pdf.confidence_bar(confidence)

        # Confidence advisory
        pdf.set_font('Helvetica', 'I', 8)
        if confidence >= config.CONFIDENCE_GREEN:
            advisory = 'High confidence detection. Result is reliable.'
        elif confidence >= config.CONFIDENCE_AMBER:
            advisory = 'Medium confidence. Manual verification recommended.'
        else:
            advisory = 'Low confidence. Expert review required before action.'
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, advisory, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Top-3 predictions table
        pdf.section_title('Top 3 Predictions')
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(200, 200, 220)
        pdf.cell(10, 7, 'Rank', fill=True, border=1)
        pdf.cell(100, 7, 'Malware Family', fill=True, border=1)
        pdf.cell(40, 7, 'Confidence', fill=True, border=1)
        pdf.ln()
        pdf.set_font('Helvetica', '', 9)
        for i, pred in enumerate(report_data.get('top3', []), 1):
            pdf.cell(10, 6, str(i), border=1)
            pdf.cell(100, 6, pred['family'], border=1)
            pdf.cell(40, 6, f"{float(pred['confidence'])*100:.2f}%", border=1)
            pdf.ln()
        pdf.ln(5)

        # ── Page 2: MITRE ATT&CK for ICS ─────────────────────────────────────
        pdf.add_page()
        pdf.section_title('MITRE ATT\u0026CK for ICS Mapping')
        mitre = report_data.get('mitre', {})

        if not mitre.get('found', False):
            pdf.set_font('Helvetica', 'I', 9)
            pdf.cell(0, 6, f'No MITRE mapping available for family: {family}', ln=True)
        else:
            pdf.set_font('Helvetica', '', 9)
            if mitre.get('description'):
                pdf.set_font('Helvetica', 'I', 9)
                pdf.multi_cell(0, 5, mitre['description'])
                pdf.ln(3)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, 'Tactics:', ln=True)
            pdf.set_font('Helvetica', '', 9)
            tactics_str = ' | '.join(mitre.get('tactics', []))
            pdf.multi_cell(0, 5, tactics_str or 'None identified')
            pdf.ln(3)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, 'Techniques:', ln=True)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(200, 200, 220)
            pdf.cell(30, 7, 'ID', fill=True, border=1)
            pdf.cell(130, 7, 'Technique Name', fill=True, border=1)
            pdf.ln()
            pdf.set_font('Helvetica', '', 9)
            for tech in mitre.get('techniques', []):
                pdf.cell(30, 6, tech.get('id', ''), border=1)
                pdf.cell(130, 6, tech.get('name', ''), border=1)
                pdf.ln()

        pdf.ln(8)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, 'Reference: MITRE ATT&CK for ICS — https://attack.mitre.org/matrices/ics/', ln=True)
        pdf.set_text_color(0, 0, 0)

        # ── Page 3: Grad-CAM Heatmap (if available) ───────────────────────────
        gradcam = report_data.get('gradcam', {})
        if gradcam.get('generated') and gradcam.get('overlay_png_bytes'):
            pdf.add_page()
            pdf.section_title('Explainable AI — Grad-CAM Heatmap')

            pdf.set_font('Helvetica', '', 9)
            pdf.multi_cell(0, 5,
                'The Grad-CAM heatmap highlights which byte regions of the binary image '
                'most strongly influenced the classification decision. '
                'Warm (red/yellow) regions correspond to high attribution areas. '
                f'Target layer: {gradcam.get("layer", "block3.conv2")}.'
            )
            pdf.ln(3)

            # Save PNG to a temp file for FPDF to embed
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(gradcam['overlay_png_bytes'])
                tmp_path = tmp.name

            try:
                # Centre the image on the page (A4 width = 210mm, margins = 10mm each side)
                img_w = 130
                x_pos = (210 - img_w) / 2
                pdf.image(tmp_path, x=x_pos, w=img_w)
                pdf.ln(5)
                pdf.set_font('Helvetica', 'I', 8)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5,
                    f'Figure: Grad-CAM overlay for {family} | '
                    f'Method: Captum LayerGradCam | Layer: {gradcam.get("layer", "")}',
                    align='C', ln=True,
                )
                pdf.set_text_color(0, 0, 0)
            finally:
                os.unlink(tmp_path)   # always clean up temp file

        # ── Final page: Disclaimer ────────────────────────────────────────────
        pdf.ln(10)
        pdf.set_font('Helvetica', 'I', 7)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 4,
            'DISCLAIMER: This report is generated by an AI-based malware classification system '
            'for research and academic purposes only. Results should be validated by a qualified '
            'cybersecurity professional before any operational decision is made. '
            'MalTwin is a research prototype — COMSATS University Islamabad, BS Cyber Security 2023-2027.'
        )

        return bytes(pdf.output())

    except Exception as e:
        print(f"[MalTwin] PDF generation failed: {e}", file=sys.stderr)
        return None


def save_pdf_report(pdf_bytes: bytes, sha256: str) -> Path:
    """
    Save PDF report bytes to config.REPORTS_DIR.

    Returns:
        Path to the saved file.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"maltwin_report_{sha256[:8]}.pdf"
    out_path = config.REPORTS_DIR / filename
    out_path.write_bytes(pdf_bytes)
    return out_path
```

---

## File 6: Update `modules/dashboard/pages/detection.py`

### 6a — Add helper `build_report_data()`

Add this function alongside the other private helpers:

```python
def _build_report_data() -> dict:
    """
    Assemble the complete report_data dict from session state.
    Used by both PDF and JSON report generators.
    Called only when a detection result exists.
    """
    from modules.reporting.mitre_mapper import get_mitre_mapping

    result    = st.session_state[state.KEY_DETECTION]
    meta      = st.session_state[state.KEY_FILE_META]
    heatmap   = st.session_state.get(state.KEY_HEATMAP)

    mitre_data = get_mitre_mapping(result['predicted_family'])

    gradcam_info = {'generated': False}
    if heatmap is not None:
        gradcam_info = {
            'generated':        True,
            'target_class':     heatmap['target_class'],
            'layer':            heatmap['captum_layer'],
            'overlay_png_bytes': heatmap['overlay_png'],   # bytes for PDF embedding
        }

    return {
        'file_name':        meta['name'],
        'sha256':           meta['sha256'],
        'file_format':      meta['format'],
        'file_size_bytes':  meta['size_bytes'],
        'upload_time':      meta['upload_time'],
        'predicted_family': result['predicted_family'],
        'confidence':       float(result['confidence']),
        'top3':             result['top3'],
        'all_probabilities': result['probabilities'],
        'gradcam':          gradcam_info,
        'mitre':            mitre_data,
    }
```

### 6b — Replace the entire `# ── F: Report Export` section in `_render_results()`

```python
    # ── F: Forensic Report Export ─────────────────────────────────────────────
    st.subheader("Forensic Report")

    report_data = _build_report_data()

    col_pdf, col_json = st.columns(2)

    with col_pdf:
        if st.button(
            "📄 Generate PDF Report",
            type="secondary",
            use_container_width=True,
            help="Generate a PDF forensic report with MITRE ATT&CK mapping "
                 "and Grad-CAM heatmap (if generated).",
        ):
            with st.spinner("Generating PDF report…"):
                from modules.reporting.pdf_report import generate_pdf_report
                pdf_bytes = generate_pdf_report(report_data)

            if pdf_bytes is None:
                st.error(
                    "Error: PDF generation failed. "
                    "Cause: fpdf2 error or missing dependency. "
                    "Action: Use the JSON download instead."
                )
            else:
                meta = st.session_state[state.KEY_FILE_META]
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"maltwin_report_{meta['sha256'][:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with col_json:
        from modules.reporting.json_report import generate_json_report
        json_bytes = generate_json_report(report_data)
        meta = st.session_state[state.KEY_FILE_META]
        st.download_button(
            label="📥 Download JSON Report",
            data=json_bytes,
            file_name=f"maltwin_report_{meta['sha256'][:8]}.json",
            mime="application/json",
            use_container_width=True,
            help="Full structured JSON report with MITRE ATT&CK mapping and detection metadata.",
        )

    if state.has_heatmap():
        st.caption(
            "✅ Grad-CAM heatmap will be embedded in the PDF report. "
            "Generate the heatmap above first if you haven't already."
        )
    else:
        st.caption(
            "💡 Tip: Generate the Grad-CAM heatmap above to embed it in the PDF report."
        )
```

---

## File 7: `tests/test_reporting.py`

```python
"""
Test suite for modules/reporting/

Tests run without the Malimg dataset or a trained model.
All report_data dicts are constructed inline.

Run:
    pytest tests/test_reporting.py -v
"""
import json
import pytest
from pathlib import Path

from modules.reporting.mitre_mapper import load_mitre_db, get_mitre_mapping
from modules.reporting.json_report import generate_json_report
from modules.reporting.pdf_report import generate_pdf_report, FPDF2_AVAILABLE


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_report_data():
    return {
        'file_name':        'suspicious.exe',
        'sha256':           'a' * 64,
        'file_format':      'PE',
        'file_size_bytes':  2048,
        'upload_time':      '2025-05-01T14:30:00',
        'predicted_family': 'Allaple.A',
        'confidence':       0.9312,
        'top3': [
            {'family': 'Allaple.A',  'confidence': 0.9312},
            {'family': 'Allaple.L',  'confidence': 0.0421},
            {'family': 'Rbot!gen',   'confidence': 0.0102},
        ],
        'all_probabilities': {f'Family_{i}': 0.01 for i in range(25)},
        'gradcam': {'generated': False},
        'mitre': {
            'found':       True,
            'description': 'Test family description.',
            'tactics':     ['Lateral Movement', 'Discovery'],
            'techniques':  [
                {'id': 'T0840', 'name': 'Network Connection Enumeration'},
            ],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MITRE mapper
# ─────────────────────────────────────────────────────────────────────────────

class TestMitreMapper:
    def test_load_mitre_db_returns_dict(self):
        db = load_mitre_db()
        assert isinstance(db, dict)

    def test_load_mitre_db_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'MITRE_JSON_PATH', tmp_path / 'missing.json')
        db = load_mitre_db(tmp_path / 'missing.json')
        assert db == {}

    def test_all_25_families_present(self):
        """The seeded JSON must have all 25 Malimg families."""
        db = load_mitre_db()
        expected = {
            'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L',
            'Alueron.gen!J', 'Autorun.K', 'BrowseFox.B', 'Dialplatform.B',
            'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1',
            'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J',
            'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E',
            'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A', 'Zlob.gen!D',
        }
        assert expected.issubset(db.keys()), \
            f"Missing families: {expected - db.keys()}"

    def test_each_family_has_required_keys(self):
        db = load_mitre_db()
        for family, entry in db.items():
            assert 'tactics' in entry,     f"{family} missing 'tactics'"
            assert 'techniques' in entry,  f"{family} missing 'techniques'"
            assert 'description' in entry, f"{family} missing 'description'"

    def test_tactics_are_lists_of_strings(self):
        db = load_mitre_db()
        for family, entry in db.items():
            assert isinstance(entry['tactics'], list), f"{family}: tactics must be list"
            assert all(isinstance(t, str) for t in entry['tactics'])

    def test_techniques_have_id_and_name(self):
        db = load_mitre_db()
        for family, entry in db.items():
            for tech in entry['techniques']:
                assert 'id' in tech,   f"{family}: technique missing 'id'"
                assert 'name' in tech, f"{family}: technique missing 'name'"

    def test_get_mitre_mapping_known_family(self):
        result = get_mitre_mapping('Allaple.A')
        assert result['found'] is True
        assert len(result['tactics']) > 0
        assert len(result['techniques']) > 0

    def test_get_mitre_mapping_unknown_family(self):
        result = get_mitre_mapping('UnknownFamily.XYZ')
        assert result['found'] is False
        assert result['tactics'] == []
        assert result['techniques'] == []

    def test_get_mitre_mapping_returns_required_keys(self):
        result = get_mitre_mapping('Allaple.A')
        assert 'found' in result
        assert 'tactics' in result
        assert 'techniques' in result
        assert 'description' in result

    def test_get_mitre_mapping_with_preloaded_db(self):
        db = load_mitre_db()
        result = get_mitre_mapping('Rbot!gen', db=db)
        assert result['found'] is True


# ─────────────────────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonReport:
    def test_returns_bytes(self, sample_report_data):
        result = generate_json_report(sample_report_data)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_json(self, sample_report_data):
        result = generate_json_report(sample_report_data)
        parsed = json.loads(result.decode('utf-8'))
        assert isinstance(parsed, dict)

    def test_contains_report_metadata(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        assert 'report_metadata' in parsed
        assert 'generated_at' in parsed['report_metadata']

    def test_contains_file_information(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        fi = parsed['file_information']
        assert fi['file_name'] == 'suspicious.exe'
        assert fi['sha256'] == 'a' * 64
        assert fi['file_format'] == 'PE'
        assert fi['file_size_bytes'] == 2048

    def test_contains_detection_result(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        dr = parsed['detection_result']
        assert dr['predicted_family'] == 'Allaple.A'
        assert abs(dr['confidence'] - 0.9312) < 1e-4

    def test_top3_has_three_entries(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        assert len(parsed['detection_result']['top3_predictions']) == 3

    def test_top3_entries_have_rank(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        ranks = [p['rank'] for p in parsed['detection_result']['top3_predictions']]
        assert ranks == [1, 2, 3]

    def test_contains_mitre_section(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        mitre = parsed['mitre_attack_ics']
        assert mitre['mapping_found'] is True
        assert 'tactics' in mitre
        assert 'techniques' in mitre

    def test_contains_explainability_section(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        xai = parsed['explainability']
        assert 'gradcam_generated' in xai
        assert xai['gradcam_generated'] is False

    def test_gradcam_true_when_heatmap_present(self, sample_report_data):
        sample_report_data['gradcam'] = {
            'generated': True, 'target_class': 0, 'layer': 'Conv2d (block3.conv2)',
        }
        parsed = json.loads(generate_json_report(sample_report_data))
        assert parsed['explainability']['gradcam_generated'] is True

    def test_all_confidence_values_are_floats(self, sample_report_data):
        parsed = json.loads(generate_json_report(sample_report_data))
        conf = parsed['detection_result']['confidence']
        assert isinstance(conf, float)

    def test_output_is_utf8_decodable(self, sample_report_data):
        result = generate_json_report(sample_report_data)
        decoded = result.decode('utf-8')
        assert len(decoded) > 0

    def test_handles_missing_top3_gracefully(self, sample_report_data):
        sample_report_data.pop('top3')
        result = generate_json_report(sample_report_data)
        parsed = json.loads(result)
        assert parsed['detection_result']['top3_predictions'] == []

    def test_returns_error_json_on_bad_input(self):
        """Completely broken input should return error JSON, not raise."""
        result = generate_json_report({})
        parsed = json.loads(result)
        # Either a valid report or an error dict — never raises
        assert isinstance(parsed, dict)


# ─────────────────────────────────────────────────────────────────────────────
# PDF report
# ─────────────────────────────────────────────────────────────────────────────

class TestPdfReport:
    def test_returns_bytes_when_fpdf2_available(self, sample_report_data):
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")
        result = generate_pdf_report(sample_report_data)
        assert result is not None
        assert isinstance(result, bytes)

    def test_output_starts_with_pdf_magic(self, sample_report_data):
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")
        result = generate_pdf_report(sample_report_data)
        assert result is not None
        assert result[:4] == b'%PDF'

    def test_output_is_nonempty(self, sample_report_data):
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")
        result = generate_pdf_report(sample_report_data)
        assert result is not None
        assert len(result) > 1000   # any real PDF is at least 1KB

    def test_returns_none_when_fpdf2_missing(self, sample_report_data, monkeypatch):
        """If FPDF2 is not importable, generate_pdf_report must return None."""
        import modules.reporting.pdf_report as pr
        monkeypatch.setattr(pr, 'FPDF2_AVAILABLE', False)
        result = pr.generate_pdf_report(sample_report_data)
        assert result is None

    def test_does_not_raise_on_bad_input(self):
        """Broken input must return None, never raise."""
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")
        result = generate_pdf_report({})
        assert result is None

    def test_with_heatmap_bytes_embedded(self, sample_report_data):
        """PDF with a heatmap PNG embedded must still succeed."""
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")
        import io
        import numpy as np
        from PIL import Image
        # Create a tiny valid PNG as fake heatmap
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, 'RGB').save(buf, format='PNG')
        fake_png = buf.getvalue()

        sample_report_data['gradcam'] = {
            'generated': True,
            'target_class': 0,
            'layer': 'Conv2d (block3.conv2)',
            'overlay_png_bytes': fake_png,
        }
        result = generate_pdf_report(sample_report_data)
        assert result is not None
        assert result[:4] == b'%PDF'
```

---

## Dependency Check

```bash
# Verify fpdf2 is installed (not the old fpdf package)
python -c "from fpdf import FPDF; print('fpdf2 OK')"

# If not installed:
pip install fpdf2 --break-system-packages

# Verify in requirements.txt:
grep fpdf requirements.txt
# Should show: fpdf2>=2.7.0
```

---

## Definition of Done

```bash
# Step 2 tests
pytest tests/test_reporting.py -v
# Expected: all tests pass (PDF tests skip gracefully if fpdf2 missing)

# Full regression suite
pytest tests/ -v -m "not integration"
# Expected: 0 failures across all phases

# Import smoke tests
python -c "from modules.reporting import generate_pdf_report, generate_json_report"
python -c "from modules.reporting.mitre_mapper import get_mitre_mapping"

# Verify all 25 families are in the MITRE JSON
python -c "
from modules.reporting.mitre_mapper import load_mitre_db
db = load_mitre_db()
print(f'Families in MITRE DB: {len(db)}')
assert len(db) == 25, f'Expected 25, got {len(db)}'
print('OK')
"
```

### Checklist

- [ ] `pytest tests/test_reporting.py -v` — 0 failures
- [ ] All earlier tests still pass
- [ ] `data/mitre_ics_mapping.json` exists with all 25 Malimg families
- [ ] `modules/reporting/` package exists with 4 files
- [ ] `config.py` has `REPORTS_DIR` defined
- [ ] PDF button in dashboard is no longer `disabled=True`
- [ ] Clicking "Generate PDF Report" → spinner → "Download PDF Report" button appears
- [ ] PDF opens correctly in a browser PDF viewer
- [ ] JSON report includes `mitre_attack_ics` section with tactics + techniques
- [ ] JSON report includes `explainability.gradcam_generated` field
- [ ] If Grad-CAM was run before PDF generation, heatmap is embedded in the PDF
- [ ] `generate_pdf_report` returns `None` on failure — never raises
- [ ] `generate_json_report` returns error JSON bytes on failure — never raises
- [ ] `os.unlink(tmp_path)` in `generate_pdf_report` is inside a `finally` block

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|---|---|---|
| `import fpdf` instead of `from fpdf import FPDF` | `AttributeError` — old `fpdf` package has different API | Use `from fpdf import FPDF` (fpdf2) |
| `pdf.output()` returns bytearray not bytes | Type error when writing to `st.download_button` | Wrap: `bytes(pdf.output())` |
| Temp PNG file not deleted after PDF embed | Disk fills up over many reports | `os.unlink(tmp_path)` inside `finally` block |
| `generate_pdf_report` raising on bad input | Dashboard page crashes | Wrap entire body in `try/except Exception: return None` |
| MITRE family name case mismatch | `get_mitre_mapping` returns `found: False` for valid families | Keys in JSON must exactly match `class_names` from training (e.g. `Allaple.A` not `allaple.a`) |
| Numpy float32 in `all_probabilities` | `json.dumps` raises `TypeError` | Cast all values: `round(float(v), 6)` |
| `overlay_png_bytes` key missing from gradcam dict | KeyError when PDF tries to embed heatmap | Always check `gradcam.get('overlay_png_bytes')` before embedding |
| `st.download_button` inside `st.button` callback | Streamlit renders button but click does nothing | Generate PDF bytes then immediately call `st.download_button` in same rerun |

---

*Step 2 complete → move to Step 3: Dataset Gallery page + Detection History filtering.*
