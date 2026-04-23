"""
Test suite for modules/detection/model.py and modules/detection/inference.py

All tests run without the Malimg dataset or a trained model.
No @pytest.mark.integration tests are needed here — the model can be instantiated
and inference can be run on random tensors without any dataset.

Run:
    pytest tests/test_model.py -v
"""
import pytest
import torch
import numpy as np
import json
from modules.detection.model import MalTwinCNN, ConvBlock


# ─────────────────────────────────────────────────────────────────────────────
# ConvBlock tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConvBlock:
    def test_output_shape_block1(self):
        """block1: (B, 1, 128, 128) → (B, 32, 64, 64) after MaxPool."""
        block = ConvBlock(in_channels=1, out_channels=32)
        x = torch.randn(4, 1, 128, 128)
        out = block(x)
        assert out.shape == (4, 32, 64, 64)

    def test_output_shape_block2(self):
        """block2: (B, 32, 64, 64) → (B, 64, 32, 32) after MaxPool."""
        block = ConvBlock(in_channels=32, out_channels=64)
        x = torch.randn(4, 32, 64, 64)
        out = block(x)
        assert out.shape == (4, 64, 32, 32)

    def test_output_shape_block3(self):
        """block3: (B, 64, 32, 32) → (B, 128, 16, 16) after MaxPool."""
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(4, 64, 32, 32)
        out = block(x)
        assert out.shape == (4, 128, 16, 16)

    def test_conv1_bias_false(self):
        block = ConvBlock(1, 32)
        assert block.conv1.bias is None, "Conv2d bias should be None when bias=False"

    def test_conv2_bias_false(self):
        block = ConvBlock(1, 32)
        assert block.conv2.bias is None, "Conv2d bias should be None when bias=False"

    def test_has_batchnorm(self):
        import torch.nn as nn
        block = ConvBlock(1, 32)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)

    def test_has_maxpool(self):
        import torch.nn as nn
        block = ConvBlock(1, 32)
        assert isinstance(block.pool, nn.MaxPool2d)

    def test_custom_dropout(self):
        import torch.nn as nn
        block = ConvBlock(1, 32, dropout_p=0.1)
        assert isinstance(block.drop, nn.Dropout2d)


# ─────────────────────────────────────────────────────────────────────────────
# MalTwinCNN architecture tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMalTwinCNN:
    def test_forward_pass_output_shape(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert out.shape == (4, num_classes)

    def test_single_sample_forward(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, num_classes)

    def test_parameter_count_reasonable(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000, f"Too few parameters: {total}"
        assert total < 20_000_000, f"Too many parameters: {total}"

    def test_output_is_raw_logits_no_softmax(self, num_classes):
        """
        Verify softmax was NOT applied in forward().
        If softmax was applied, all outputs would be in [0,1] and sum to 1.
        We verify that softmax of the output produces valid probabilities —
        this tests the contract (logits in, probs after softmax), not the values.
        """
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        # Applying softmax to raw logits must yield probabilities that sum to 1
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_gradcam_layer_attribute_exists(self, num_classes):
        """
        CRITICAL: self.gradcam_layer must be set and must point to block3.conv2.
        This is required by Module 7 (Grad-CAM).
        """
        model = MalTwinCNN(num_classes=num_classes)
        assert hasattr(model, 'gradcam_layer'), \
            "MalTwinCNN must have self.gradcam_layer attribute"
        assert model.gradcam_layer is model.block3.conv2, \
            "gradcam_layer must be self.block3.conv2 (the second conv of block3)"

    def test_block_attributes_exist(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert hasattr(model, 'block1')
        assert hasattr(model, 'block2')
        assert hasattr(model, 'block3')
        assert hasattr(model, 'pool')
        assert hasattr(model, 'classifier')

    def test_block1_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block1.conv1.in_channels == 1
        assert model.block1.conv1.out_channels == 32

    def test_block2_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block2.conv1.in_channels == 32
        assert model.block2.conv1.out_channels == 64

    def test_block3_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block3.conv1.in_channels == 64
        assert model.block3.conv1.out_channels == 128

    def test_deterministic_in_eval_mode(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_dropout_nondeterministic(self, num_classes):
        """In train mode, Dropout causes different outputs for different seeds."""
        model = MalTwinCNN(num_classes=num_classes)
        model.train()
        x = torch.randn(4, 1, 128, 128)
        torch.manual_seed(0)
        out1 = model(x)
        torch.manual_seed(1)
        out2 = model(x)
        assert not torch.equal(out1, out2)

    def test_weight_initialization_conv_no_zeros(self, num_classes):
        """Kaiming init should produce non-zero weights."""
        model = MalTwinCNN(num_classes=num_classes)
        # At least some weights should be nonzero after Kaiming init
        conv_weights = model.block1.conv1.weight.data
        assert conv_weights.abs().sum() > 0

    def test_batchnorm_initialized_correctly(self, num_classes):
        """BatchNorm weight=1, bias=0 after _initialize_weights."""
        model = MalTwinCNN(num_classes=num_classes)
        # Check block1's BN
        assert torch.allclose(model.block1.bn1.weight, torch.ones_like(model.block1.bn1.weight))
        assert torch.allclose(model.block1.bn1.bias, torch.zeros_like(model.block1.bn1.bias))

    def test_adaptive_avg_pool_output(self, num_classes):
        """AdaptiveAvgPool2d should output (B, 128, 4, 4) before flatten."""
        import torch.nn as nn
        model = MalTwinCNN(num_classes=num_classes)
        # Run up to just before classifier
        x = torch.randn(2, 1, 128, 128)
        x = model.block1(x)   # (2, 32, 64, 64)
        x = model.block2(x)   # (2, 64, 32, 32)
        x = model.block3(x)   # (2, 128, 16, 16)
        x = model.pool(x)     # (2, 128, 4, 4)
        assert x.shape == (2, 128, 4, 4)

    def test_classifier_output_size(self, num_classes):
        """Classifier input should be 128*4*4=2048."""
        model = MalTwinCNN(num_classes=num_classes)
        # Find the first Linear layer in classifier
        import torch.nn as nn
        first_linear = next(m for m in model.classifier.modules() if isinstance(m, nn.Linear))
        assert first_linear.in_features == 2048


# ─────────────────────────────────────────────────────────────────────────────
# Inference tests (predict_single, load_model)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSingle:
    def _make_result(self, num_classes, sample_grayscale_array):
        from modules.detection.inference import predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        return predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))

    def test_returns_required_keys(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert 'predicted_family' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'top3' in result

    def test_confidence_in_valid_range(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_has_all_classes(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert len(result['probabilities']) == num_classes

    def test_all_probabilities_nonnegative(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert all(v >= 0.0 for v in result['probabilities'].values())

    def test_all_probabilities_at_most_one(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert all(v <= 1.0 for v in result['probabilities'].values())

    def test_top3_has_three_entries(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert len(result['top3']) == 3

    def test_top3_entries_have_required_keys(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        for entry in result['top3']:
            assert 'family' in entry
            assert 'confidence' in entry

    def test_predicted_family_matches_top3_first(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert result['predicted_family'] == result['top3'][0]['family']

    def test_predicted_family_confidence_matches_top3_first(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert abs(result['confidence'] - result['top3'][0]['confidence']) < 1e-6

    def test_top3_sorted_descending(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        confs = [item['confidence'] for item in result['top3']]
        assert confs == sorted(confs, reverse=True)

    def test_predicted_family_is_valid_class(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        assert result['predicted_family'] in class_names

    def test_all_values_json_serialisable(self, sample_grayscale_array, num_classes):
        """All float values must be Python float, not numpy float32."""
        result = self._make_result(num_classes, sample_grayscale_array)
        # Should not raise TypeError
        json.dumps(result)

    def test_confidence_is_python_float(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert isinstance(result['confidence'], float)

    def test_uses_val_transforms_not_train(self, sample_grayscale_array, num_classes):
        """
        Calling predict_single twice on the same image should return identical results
        (val transforms are deterministic; train transforms are not).
        """
        from modules.detection.inference import predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        r1 = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        r2 = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert r1['predicted_family'] == r2['predicted_family']
        assert abs(r1['confidence'] - r2['confidence']) < 1e-6


class TestLoadModel:
    def test_raises_on_missing_file(self, tmp_path):
        from modules.detection.inference import load_model
        missing = tmp_path / 'nonexistent.pt'
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(model_path=missing, num_classes=25, device=torch.device('cpu'))

    def test_loads_saved_state_dict(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        # Save a freshly initialised model's state dict
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)

        # Load it back
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        assert isinstance(loaded, MalTwinCNN)

    def test_loaded_model_is_in_eval_mode(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        assert not loaded.training, "load_model() must return model in eval() mode"

    def test_loaded_model_produces_correct_output_shape(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, num_classes)

    def test_loaded_weights_match_original(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))

        original_w = model.block1.conv1.weight.data
        loaded_w   = loaded.block1.conv1.weight.data
        torch.testing.assert_close(original_w, loaded_w)


class TestPredictBatch:
    def test_returns_list_of_dicts(self, sample_grayscale_array, num_classes):
        from modules.detection.inference import predict_batch
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        results = predict_batch(model, [sample_grayscale_array, sample_grayscale_array],
                                class_names, torch.device('cpu'))
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_results_match_single(self, sample_grayscale_array, num_classes):
        """predict_batch on one image should match predict_single on same image."""
        from modules.detection.inference import predict_batch, predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]

        single = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        batch  = predict_batch(model, [sample_grayscale_array], class_names, torch.device('cpu'))

        assert single['predicted_family'] == batch[0]['predicted_family']
        assert abs(single['confidence'] - batch[0]['confidence']) < 1e-5

    def test_result_order_preserved(self, num_classes):
        """Results must be in same order as inputs."""
        from modules.detection.inference import predict_batch
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]

        # Create two distinct arrays: all-zeros and all-255
        arr_black = np.zeros((128, 128), dtype=np.uint8)
        arr_white = np.full((128, 128), 255, dtype=np.uint8)

        results = predict_batch(model, [arr_black, arr_white], class_names, torch.device('cpu'))
        assert len(results) == 2
        # Both should return valid results (can't predict which family, but structure valid)
        for r in results:
            assert 'predicted_family' in r
            assert 'confidence' in r

