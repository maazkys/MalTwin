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
