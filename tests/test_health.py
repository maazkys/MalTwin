"""
Test suite for modules/dashboard/health.py

Health checks interact with the filesystem (config paths) and optionally
Docker. All tests mock filesystem state via tmp_path + monkeypatch.
No real dataset or trained model required.

Run:
    pytest tests/test_health.py -v
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Individual module checks
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckModule2BinaryToImage:
    def test_returns_active_when_module_importable(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        # M2 is always implemented — should be active
        assert result['status'] == 'active'
        assert result['emoji'] == '✅'

    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        assert 'status' in result
        assert 'detail' in result
        assert 'emoji' in result

    def test_status_is_valid_value(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        assert result['status'] in ('active', 'inactive', 'error')


class TestCheckModule3Dataset:
    def test_inactive_when_data_dir_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'nonexistent')
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'inactive'

    def test_inactive_when_data_dir_empty(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'inactive'

    def test_active_when_families_present(self, tmp_path, monkeypatch):
        import config
        (tmp_path / 'Allaple.A').mkdir()
        (tmp_path / 'Rbot_gen').mkdir()
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'active'
        assert '2' in result['detail']   # "2 families found"


class TestCheckModule4Enhancement:
    def test_returns_active(self):
        from modules.dashboard.health import _check_module4_enhancement
        result = _check_module4_enhancement()
        assert result['status'] == 'active'


class TestCheckModule5Detection:
    def test_inactive_when_model_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', tmp_path / 'missing.pt')
        from modules.dashboard.health import _check_module5_detection
        result = _check_module5_detection()
        assert result['status'] == 'inactive'

    def test_inactive_when_class_names_missing(self, tmp_path, monkeypatch):
        import config
        import torch
        from modules.detection.model import MalTwinCNN
        # Create a real model file
        pt_path = tmp_path / 'best_model.pt'
        model = MalTwinCNN(num_classes=25)
        torch.save(model.state_dict(), pt_path)
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', pt_path)
        monkeypatch.setattr(config, 'CLASS_NAMES_PATH', tmp_path / 'missing.json')
        from modules.dashboard.health import _check_module5_detection
        result = _check_module5_detection()
        assert result['status'] == 'inactive'

    def test_active_when_model_and_class_names_present(self, tmp_path, monkeypatch):
        import config, json
        import torch
        from modules.detection.model import MalTwinCNN
        pt_path    = tmp_path / 'best_model.pt'
        names_path = tmp_path / 'class_names.json'
        model      = MalTwinCNN(num_classes=25)
        torch.save(model.state_dict(), pt_path)
        names_path.write_text(json.dumps({'class_names': [f'F{i}' for i in range(25)]}))
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', pt_path)
        monkeypatch.setattr(config, 'CLASS_NAMES_PATH', names_path)
        from modules.dashboard import health
        # Reload module to pick up monkeypatched config
        import importlib; importlib.reload(health)
        result = health._check_module5_detection()
        assert result['status'] == 'active'


class TestCheckModule6Dashboard:
    def test_always_active(self):
        from modules.dashboard.health import _check_module6_dashboard
        result = _check_module6_dashboard()
        assert result['status'] == 'active'


class TestCheckModule7Gradcam:
    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import _check_module7_gradcam
        result = _check_module7_gradcam()
        assert 'status' in result
        assert 'detail' in result
        assert 'emoji' in result

    def test_status_is_valid_value(self):
        from modules.dashboard.health import _check_module7_gradcam
        result = _check_module7_gradcam()
        assert result['status'] in ('active', 'inactive', 'error')

    def test_active_or_inactive_when_captum_installed(self):
        """Captum is installed (Step 1 requirement) — should not be 'error'."""
        try:
            import captum   # noqa
            from modules.dashboard.health import _check_module7_gradcam
            result = _check_module7_gradcam()
            assert result['status'] in ('active', 'inactive')
        except ImportError:
            pytest.skip("Captum not installed")


class TestCheckModule8Reporting:
    def test_inactive_when_mitre_json_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'MITRE_JSON_PATH', tmp_path / 'missing.json')
        from modules.dashboard.health import _check_module8_reporting
        result = _check_module8_reporting()
        assert result['status'] == 'inactive'

    def test_active_when_mitre_json_present(self, tmp_path, monkeypatch):
        import config, json
        mitre_path = tmp_path / 'mitre.json'
        # Write all 25 families
        db = {f'Family_{i}': {'tactics': [], 'techniques': [], 'description': ''} for i in range(25)}
        mitre_path.write_text(json.dumps(db))
        monkeypatch.setattr(config, 'MITRE_JSON_PATH', mitre_path)
        from modules.dashboard import health
        import importlib; importlib.reload(health)
        result = health._check_module8_reporting()
        assert result['status'] == 'active'


# ─────────────────────────────────────────────────────────────────────────────
# get_all_module_statuses
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllModuleStatuses:
    def test_returns_list_of_eight(self):
        from modules.dashboard.health import get_all_module_statuses
        # Clear cache before calling to ensure fresh result
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        assert len(results) == 8

    def test_each_entry_has_required_keys(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        for r in results:
            assert 'id' in r
            assert 'name' in r
            assert 'status' in r
            assert 'detail' in r
            assert 'emoji' in r

    def test_all_statuses_are_valid_values(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        for r in results:
            assert r['status'] in ('active', 'inactive', 'error'), \
                f"Module {r['id']} has invalid status: {r['status']}"

    def test_module_ids_are_m1_through_m8(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        ids = [r['id'] for r in results]
        assert ids == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']

    def test_does_not_raise(self):
        """get_all_module_statuses must never raise regardless of environment."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        try:
            results = get_all_module_statuses()
        except Exception as e:
            pytest.fail(f"get_all_module_statuses raised: {e}")

    def test_m6_is_always_active(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m6 = next(r for r in results if r['id'] == 'M6')
        assert m6['status'] == 'active'

    def test_m2_is_always_active(self):
        """M2 is fully implemented — should always be active."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m2 = next(r for r in results if r['id'] == 'M2')
        assert m2['status'] == 'active'

    def test_m4_is_always_active(self):
        """M4 is fully implemented — should always be active."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m4 = next(r for r in results if r['id'] == 'M4')
        assert m4['status'] == 'active'


# ─────────────────────────────────────────────────────────────────────────────
# get_system_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemStats:
    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        required = {
            'cpu_pct', 'mem_pct', 'mem_used_gb',
            'mem_total_gb', 'uptime_str', 'device', 'error',
        }
        assert required.issubset(stats.keys())

    def test_cpu_pct_in_valid_range(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert 0.0 <= stats['cpu_pct'] <= 100.0

    def test_mem_pct_in_valid_range(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert 0.0 <= stats['mem_pct'] <= 100.0

    def test_mem_used_le_mem_total(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert stats['mem_used_gb'] <= stats['mem_total_gb']

    def test_device_is_nonempty_string(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        assert isinstance(stats['device'], str)
        assert len(stats['device']) > 0

    def test_error_flag_is_bool(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        assert isinstance(stats['error'], bool)

    def test_does_not_raise_when_psutil_missing(self, monkeypatch):
        """If psutil is not importable, stats must return error=True, not raise."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("mocked missing psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        from modules.dashboard import health
        import importlib; importlib.reload(health)
        stats = health.get_system_stats()
        assert stats['error'] is True
