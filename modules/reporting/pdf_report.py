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
        self.cell(0, 10, '  MalTwin - IIoT Malware Detection Framework', fill=True, new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f'  Forensic Analysis Report | Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC', new_x='LMARGIN', new_y='NEXT')
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
        self.cell(0, 8, f'  {title}', fill=True, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, label: str, value: str, label_width: int = 55):
        self.set_font('Helvetica', 'B', 9)
        self.cell(label_width, 6, label, border='B')
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 6, value, border='B', new_x='LMARGIN', new_y='NEXT')

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
        pdf.cell(0, 10, f'Predicted Family: {family}', new_x='LMARGIN', new_y='NEXT')
        pdf.set_text_color(0, 0, 0)

        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, 'Confidence:', new_x='LMARGIN', new_y='NEXT')
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
        pdf.cell(0, 5, advisory, new_x='LMARGIN', new_y='NEXT')
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
            pdf.cell(0, 6, f'No MITRE mapping available for family: {family}', new_x='LMARGIN', new_y='NEXT')
        else:
            pdf.set_font('Helvetica', '', 9)
            if mitre.get('description'):
                pdf.set_font('Helvetica', 'I', 9)
                pdf.multi_cell(0, 5, mitre['description'])
                pdf.ln(3)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, 'Tactics:', new_x='LMARGIN', new_y='NEXT')
            pdf.set_font('Helvetica', '', 9)
            tactics_str = ' | '.join(mitre.get('tactics', []))
            pdf.multi_cell(0, 5, tactics_str or 'None identified')
            pdf.ln(3)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, 'Techniques:', new_x='LMARGIN', new_y='NEXT')
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
        pdf.cell(0, 5, 'Reference: MITRE ATT&CK for ICS - https://attack.mitre.org/matrices/ics/', new_x='LMARGIN', new_y='NEXT')
        pdf.set_text_color(0, 0, 0)

        # ── Page 3: Grad-CAM Heatmap (if available) ───────────────────────────
        gradcam = report_data.get('gradcam', {})
        if gradcam.get('generated') and gradcam.get('overlay_png_bytes'):
            pdf.add_page()
            pdf.section_title('Explainable AI - Grad-CAM Heatmap')

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
                    align='C', new_x='LMARGIN', new_y='NEXT',
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
            'MalTwin is a research prototype - COMSATS University Islamabad, BS Cyber Security 2023-2027.'
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
