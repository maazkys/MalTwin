"""
End-to-end integration smoke tests for the full MalTwin pipeline.

These tests require:
    - Malimg dataset at config.DATA_DIR
    - Trained model at config.BEST_MODEL_PATH
    - data/mitre_ics_mapping.json with 25 families

Run with dataset + model:
    pytest tests/test_integration.py -v

Skip in CI (no dataset):
    pytest tests/ -v -m "not integration"
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import config


# ── Skip guard ────────────────────────────────────────────────────────────────

def _dataset_available() -> bool:
    return (
        config.DATA_DIR.exists()
        and any(config.DATA_DIR.iterdir())
    )


def _model_available() -> bool:
    return (
        config.BEST_MODEL_PATH.exists()
        and config.CLASS_NAMES_PATH.exists()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: Binary → Image → Detection → GradCAM → Report
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    Tests the complete ML pipeline from raw binary bytes to forensic report.
    Uses the Phase 1 test fixture (sample_pe.exe) as input binary.
    """

    @pytest.fixture
    def sample_pe_bytes(self):
        """Load the Phase 1 test fixture PE binary."""
        fixture = Path('tests/fixtures/sample_pe.exe')
        if not fixture.exists():
            pytest.skip("tests/fixtures/sample_pe.exe not found")
        return fixture.read_bytes()

    @pytest.mark.integration
    def test_binary_to_image_pipeline(self, sample_pe_bytes):
        """Phase 1+2: binary bytes → 128×128 uint8 numpy array."""
        from modules.binary_to_image.converter import BinaryConverter
        from modules.binary_to_image.utils import validate_binary_format, compute_sha256

        fmt = validate_binary_format(sample_pe_bytes)
        assert fmt == 'PE'

        sha = compute_sha256(sample_pe_bytes)
        assert len(sha) == 64

        converter = BinaryConverter(img_size=config.IMG_SIZE)
        img = converter.convert(sample_pe_bytes)
        assert img.shape == (config.IMG_SIZE, config.IMG_SIZE)
        assert img.dtype == np.uint8

    @pytest.mark.integration
    def test_dataset_loads_correctly(self):
        """Phase 3+4: MalimgDataset loads and returns correct tensor shapes."""
        if not _dataset_available():
            pytest.skip("Malimg dataset not available")

        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(config.DATA_DIR, 'train', img_size=config.IMG_SIZE)

        assert len(ds) > 0
        assert len(ds.class_names) == 25

        tensor, label = ds[0]
        assert tensor.shape == (1, config.IMG_SIZE, config.IMG_SIZE)
        assert tensor.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 25

    @pytest.mark.integration
    def test_all_25_classes_in_train_split(self):
        """Stratified split must preserve all 25 classes in train split."""
        if not _dataset_available():
            pytest.skip("Malimg dataset not available")

        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(config.DATA_DIR, 'train')
        assert len(set(ds.get_labels())) == 25

    @pytest.mark.integration
    def test_model_inference_pipeline(self, sample_pe_bytes):
        """Phase 4+5: binary → image → model inference → prediction dict."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.binary_to_image.converter import BinaryConverter
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model, predict_single

        # Binary → image
        img = BinaryConverter(config.IMG_SIZE).convert(sample_pe_bytes)
        assert img.shape == (config.IMG_SIZE, config.IMG_SIZE)

        # Load model
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        assert model is not None

        # Inference
        result = predict_single(model, img, class_names, config.DEVICE)
        assert result['predicted_family'] in class_names
        assert 0.0 <= result['confidence'] <= 1.0
        assert len(result['top3']) == 3
        assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-4

    @pytest.mark.integration
    def test_gradcam_pipeline(self, sample_pe_bytes):
        """Step 1: binary → image → GradCAM heatmap."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.binary_to_image.converter import BinaryConverter
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model, predict_single
        from modules.detection.gradcam import generate_gradcam

        img         = BinaryConverter(config.IMG_SIZE).convert(sample_pe_bytes)
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model       = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        result      = predict_single(model, img, class_names, config.DEVICE)
        target_cls  = class_names.index(result['predicted_family'])

        heatmap = generate_gradcam(model, img, target_cls, config.DEVICE)
        assert heatmap is not None
        assert heatmap['heatmap_array'].shape == (config.IMG_SIZE, config.IMG_SIZE)
        assert heatmap['heatmap_array'].min() >= 0.0
        assert heatmap['heatmap_array'].max() <= 1.0
        assert isinstance(heatmap['overlay_png'], bytes)
        assert len(heatmap['overlay_png']) > 0

    @pytest.mark.integration
    def test_json_report_pipeline(self, sample_pe_bytes):
        """Step 2: full detection → JSON report with MITRE mapping."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.binary_to_image.converter import BinaryConverter
        from modules.binary_to_image.utils import validate_binary_format, compute_sha256
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model, predict_single
        from modules.reporting.json_report import generate_json_report
        from modules.reporting.mitre_mapper import get_mitre_mapping

        img         = BinaryConverter(config.IMG_SIZE).convert(sample_pe_bytes)
        file_format = validate_binary_format(sample_pe_bytes)
        sha256      = compute_sha256(sample_pe_bytes)
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model       = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        result      = predict_single(model, img, class_names, config.DEVICE)
        mitre       = get_mitre_mapping(result['predicted_family'])

        report_data = {
            'file_name':        'sample_pe.exe',
            'sha256':           sha256,
            'file_format':      file_format,
            'file_size_bytes':  len(sample_pe_bytes),
            'upload_time':      '2025-05-01T12:00:00',
            'predicted_family': result['predicted_family'],
            'confidence':       result['confidence'],
            'top3':             result['top3'],
            'all_probabilities':result['probabilities'],
            'gradcam':          {'generated': False},
            'mitre':            mitre,
        }

        json_bytes = generate_json_report(report_data)
        assert isinstance(json_bytes, bytes)
        parsed = json.loads(json_bytes)

        assert parsed['file_information']['sha256'] == sha256
        assert parsed['detection_result']['predicted_family'] == result['predicted_family']
        assert 'mitre_attack_ics' in parsed
        assert 'explainability' in parsed

    @pytest.mark.integration
    def test_pdf_report_pipeline(self, sample_pe_bytes):
        """Step 2: full detection → PDF report with valid PDF header."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.reporting.pdf_report import generate_pdf_report, FPDF2_AVAILABLE
        if not FPDF2_AVAILABLE:
            pytest.skip("fpdf2 not installed")

        from modules.binary_to_image.converter import BinaryConverter
        from modules.binary_to_image.utils import validate_binary_format, compute_sha256
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model, predict_single
        from modules.reporting.mitre_mapper import get_mitre_mapping

        img         = BinaryConverter(config.IMG_SIZE).convert(sample_pe_bytes)
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model       = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        result      = predict_single(model, img, class_names, config.DEVICE)

        report_data = {
            'file_name':        'sample_pe.exe',
            'sha256':           compute_sha256(sample_pe_bytes),
            'file_format':      validate_binary_format(sample_pe_bytes),
            'file_size_bytes':  len(sample_pe_bytes),
            'upload_time':      '2025-05-01T12:00:00',
            'predicted_family': result['predicted_family'],
            'confidence':       result['confidence'],
            'top3':             result['top3'],
            'all_probabilities':result['probabilities'],
            'gradcam':          {'generated': False},
            'mitre':            get_mitre_mapping(result['predicted_family']),
        }

        pdf_bytes = generate_pdf_report(report_data)
        assert pdf_bytes is not None
        assert pdf_bytes[:4] == b'%PDF'
        assert len(pdf_bytes) > 5000

    @pytest.mark.integration
    def test_detection_event_logged_to_db(self, tmp_path, sample_pe_bytes):
        """Step 2+Phase6: detection event persists to SQLite."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.binary_to_image.converter import BinaryConverter
        from modules.binary_to_image.utils import compute_sha256
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model, predict_single
        from modules.dashboard.db import init_db, log_detection_event, get_recent_events

        db_path = tmp_path / "integration_test.db"
        init_db(db_path)

        img         = BinaryConverter(config.IMG_SIZE).convert(sample_pe_bytes)
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model       = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        result      = predict_single(model, img, class_names, config.DEVICE)

        log_detection_event(
            db_path,
            file_name='sample_pe.exe',
            sha256=compute_sha256(sample_pe_bytes),
            file_format='PE',
            file_size=len(sample_pe_bytes),
            predicted_family=result['predicted_family'],
            confidence=result['confidence'],
            device_used=str(config.DEVICE),
        )

        events = get_recent_events(db_path, limit=5)
        assert len(events) == 1
        assert events[0]['predicted_family'] == result['predicted_family']
        assert events[0]['file_name'] == 'sample_pe.exe'

    @pytest.mark.integration
    def test_mitre_coverage_for_trained_classes(self):
        """All class names from training must have MITRE mappings."""
        if not _model_available():
            pytest.skip("Trained model not available")

        from modules.dataset.preprocessor import load_class_names
        from modules.reporting.mitre_mapper import load_mitre_db

        class_names = load_class_names(config.CLASS_NAMES_PATH)
        mitre_db    = load_mitre_db()

        missing = [n for n in class_names if n not in mitre_db]
        assert not missing, (
            f"The following trained classes have no MITRE mapping: {missing}\n"
            "Update data/mitre_ics_mapping.json to add them."
        )
