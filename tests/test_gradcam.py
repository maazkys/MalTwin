"""
Test suite for modules/detection/gradcam.py

All tests run without the Malimg dataset or a saved .pt model.
Model is instantiated with random weights. img_array is a random uint8 array.

Run:
    pytest tests/test_gradcam.py -v
"""
import pytest
import numpy as np
import torch
from PIL import Image

from modules.detection.model import MalTwinCNN
from modules.detection.gradcam import generate_gradcam, overlay_heatmap


NUM_CLASSES = 25


@pytest.fixture
def model():
    m = MalTwinCNN(num_classes=NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture
def img_array():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (128, 128), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# generate_gradcam — return structure
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateGradcam:
    def test_returns_dict_on_success(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert isinstance(result, dict)

    def test_returns_required_keys(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        required = {'heatmap_array', 'overlay_png', 'heatmap_only_png',
                    'target_class', 'captum_layer'}
        assert required.issubset(result.keys())

    def test_heatmap_array_shape(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert result['heatmap_array'].shape == (128, 128)

    def test_heatmap_array_dtype_float32(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert result['heatmap_array'].dtype == np.float32

    def test_heatmap_array_range(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        arr = result['heatmap_array']
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_overlay_png_is_bytes(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert isinstance(result['overlay_png'], bytes)
        assert len(result['overlay_png']) > 0

    def test_heatmap_only_png_is_bytes(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert isinstance(result['heatmap_only_png'], bytes)

    def test_overlay_png_is_valid_image(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        pil_img = Image.open(__import__('io').BytesIO(result['overlay_png']))
        assert pil_img.mode == 'RGB'
        assert pil_img.size == (128, 128)

    def test_heatmap_only_png_is_valid_image(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        pil_img = Image.open(__import__('io').BytesIO(result['heatmap_only_png']))
        assert pil_img.mode in ('RGB', 'RGBA')

    def test_target_class_stored_correctly(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=7,
                                  device=torch.device('cpu'))
        assert result is not None
        assert result['target_class'] == 7

    def test_captum_layer_is_nonempty_string(self, model, img_array):
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None
        assert isinstance(result['captum_layer'], str)
        assert len(result['captum_layer']) > 0

    def test_uses_model_gradcam_layer_attribute(self, model, img_array):
        """
        Verify that generate_gradcam uses model.gradcam_layer.
        If the attribute were missing this test would raise AttributeError.
        """
        assert hasattr(model, 'gradcam_layer'), \
            "MalTwinCNN must have self.gradcam_layer set"
        result = generate_gradcam(model, img_array, target_class=0,
                                  device=torch.device('cpu'))
        assert result is not None

    def test_returns_none_on_bad_target_class(self, model, img_array):
        """
        An out-of-range target class should cause Captum to fail.
        generate_gradcam must return None, not raise.
        """
        result = generate_gradcam(model, img_array,
                                  target_class=9999,
                                  device=torch.device('cpu'))
        assert result is None

    def test_deterministic_on_eval_model(self, model, img_array):
        """
        Two calls on the same eval-mode model + same input should return
        identical heatmap arrays (no Dropout influence at eval time).
        """
        r1 = generate_gradcam(model, img_array, 0, torch.device('cpu'))
        r2 = generate_gradcam(model, img_array, 0, torch.device('cpu'))
        assert r1 is not None and r2 is not None
        np.testing.assert_array_almost_equal(
            r1['heatmap_array'], r2['heatmap_array'], decimal=5
        )

    def test_different_target_classes_produce_different_heatmaps(self, model, img_array):
        """
        Heatmaps for class 0 and class 1 should differ (different gradient flows).
        """
        r0 = generate_gradcam(model, img_array, 0, torch.device('cpu'))
        r1 = generate_gradcam(model, img_array, 1, torch.device('cpu'))
        assert r0 is not None and r1 is not None
        # Not strictly guaranteed with random weights, but overwhelmingly likely
        assert not np.array_equal(r0['heatmap_array'], r1['heatmap_array'])


# ─────────────────────────────────────────────────────────────────────────────
# overlay_heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayHeatmap:
    def test_returns_bytes(self, img_array):
        heatmap = np.random.rand(128, 128).astype(np.float32)
        result = overlay_heatmap(img_array, heatmap)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_rgb_png(self, img_array):
        heatmap = np.random.rand(128, 128).astype(np.float32)
        result = overlay_heatmap(img_array, heatmap)
        import io
        pil_img = Image.open(io.BytesIO(result))
        assert pil_img.mode == 'RGB'
        assert pil_img.size == (128, 128)

    def test_zero_heatmap_does_not_crash(self, img_array):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        result = overlay_heatmap(img_array, heatmap)
        assert isinstance(result, bytes)

    def test_ones_heatmap_does_not_crash(self, img_array):
        heatmap = np.ones((128, 128), dtype=np.float32)
        result = overlay_heatmap(img_array, heatmap)
        assert isinstance(result, bytes)

    def test_alpha_zero_returns_grayscale_as_rgb(self, img_array):
        """alpha=0 means no heatmap — output should be the grayscale image as RGB."""
        heatmap = np.ones((128, 128), dtype=np.float32)
        result = overlay_heatmap(img_array, heatmap, alpha=0.0)
        import io
        pil_img = Image.open(io.BytesIO(result))
        arr = np.array(pil_img)
        # All three channels should be equal (grayscale → RGB)
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])
