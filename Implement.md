# MalTwin — Implementation Step 1: M7 Grad-CAM XAI
### SRS refs: FR5.4, FR-B6, M7 FE-1 through FE-4

> Implement first, test, verify passing, then move to Step 2 (M8 Reporting).
> Do not start Step 2 until `pytest tests/test_gradcam.py -v` passes clean.

---

## What This Step Delivers

| Item | Status before | Status after |
|---|---|---|
| `modules/detection/gradcam.py` | Does not exist | Full Captum-based GradCAM implementation |
| `modules/detection/__init__.py` | No gradcam export | Exports `generate_gradcam` |
| `modules/dashboard/pages/detection.py` | Stub checkbox + info message | Real heatmap rendered in dashboard |
| `modules/dashboard/state.py` | No heatmap key | `KEY_HEATMAP` constant added |
| `tests/test_gradcam.py` | Does not exist | Full test suite |

---

## Mandatory Rules

- Use **Captum `LayerGradCam`** — not a manual hook implementation.
- The target layer is **always** `model.gradcam_layer` (which is `model.block3.conv2`). Never hardcode the layer name as a string.
- Input to the model during GradCAM must be `(1, 1, 128, 128)` — single-channel, batch size 1.
- `model.eval()` is called before GradCAM, **but** the backward pass requires gradients — do **not** wrap in `torch.no_grad()`.
- The heatmap is resized to `(128, 128)` using `cv2.resize` with `cv2.INTER_LINEAR`.
- The jet colormap overlay is a **PIL Image in RGB mode** — never BGR.
- `generate_gradcam` **never raises** — all exceptions are caught and `None` is returned. The dashboard handles `None` gracefully.
- `plt.close()` is mandatory after any matplotlib figure creation.
- All returned PNG bytes are `bytes` (not `io.BytesIO` objects).

---

## File 1: `modules/detection/gradcam.py`

```python
# modules/detection/gradcam.py
"""
Grad-CAM XAI for MalTwinCNN using Captum LayerGradCam.

SRS refs: FR5.4, FR-B6, M7 FE-1 through FE-4

Public API
----------
generate_gradcam(model, img_array, target_class, device) -> dict | None
    Returns heatmap data dict or None on failure.

overlay_heatmap(img_array, heatmap_array) -> bytes
    Returns PNG bytes of the jet-colormap heatmap overlaid on the grayscale image.
"""
import sys
import io
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import LayerGradCam

import config
from modules.enhancement.augmentor import get_val_transforms


def generate_gradcam(
    model,
    img_array: np.ndarray,
    target_class: int,
    device: torch.device = config.DEVICE,
) -> dict | None:
    """
    Generate a Grad-CAM heatmap for a single 128×128 grayscale image.

    Args:
        model:        MalTwinCNN in eval() mode.
        img_array:    numpy uint8 array shape (128, 128).
        target_class: integer class index to explain (top-1 predicted label).
        device:       inference device.

    Returns:
        On success:
            {
                'heatmap_array':   np.ndarray float32 shape (128,128) values [0,1],
                'overlay_png':     bytes  — PNG of jet heatmap overlaid on grayscale,
                'heatmap_only_png':bytes  — PNG of raw jet colormap heatmap only,
                'target_class':    int,
                'captum_layer':    str    — layer name for audit trail,
            }
        On any failure: None  (caller must handle gracefully)

    Implementation notes:
        1. model.eval() is called here — even if caller already called it.
        2. Gradients must flow — no torch.no_grad() wrapper.
        3. Captum LayerGradCam is instantiated with model.gradcam_layer.
        4. attribute() is called with target=target_class.
        5. Raw attribution shape from Captum: (1, C, H', W') where H'=W'=16 for block3.
        6. Squeeze and average over channels if C > 1, then ReLU, then normalise to [0,1].
        7. Resize to (128, 128) with cv2.resize(heatmap, (128, 128), cv2.INTER_LINEAR).
        8. Both PNG outputs are generated via overlay_heatmap() and _heatmap_to_png().
    """
    try:
        model.eval()

        # ── 1. Prepare input tensor ────────────────────────────────────────────
        transform = get_val_transforms(config.IMG_SIZE)
        pil_img   = Image.fromarray(img_array, mode='L')
        tensor    = transform(pil_img).unsqueeze(0).to(device)  # (1,1,128,128)
        tensor.requires_grad_(True)

        # ── 2. Instantiate Captum LayerGradCam ────────────────────────────────
        grad_cam = LayerGradCam(model, model.gradcam_layer)

        # ── 3. Compute attributions ────────────────────────────────────────────
        # NOTE: do NOT use torch.no_grad() here — backward pass needs gradients
        attributions = grad_cam.attribute(
            tensor,
            target=target_class,
        )
        # attributions shape: (1, out_channels, H', W') e.g. (1, 128, 16, 16)

        # ── 4. Post-process to (128, 128) float32 [0, 1] ──────────────────────
        attr_np = attributions.squeeze().detach().cpu().numpy()  # (128, 16, 16) or (16,16)

        # Average over channel dim if 3D (happens when conv has >1 output channel)
        if attr_np.ndim == 3:
            attr_np = attr_np.mean(axis=0)   # (16, 16)

        # ReLU — keep only positive contributions
        attr_np = np.maximum(attr_np, 0)

        # Normalise to [0, 1]
        attr_max = attr_np.max()
        if attr_max > 0:
            attr_np = attr_np / attr_max
        else:
            attr_np = np.zeros_like(attr_np, dtype=np.float32)

        # Resize to match input image (128, 128)
        # cv2.resize takes (width, height)
        heatmap_128 = cv2.resize(
            attr_np.astype(np.float32),
            (config.IMG_SIZE, config.IMG_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )  # shape (128, 128), values [0.0, 1.0]

        # ── 5. Generate PNG outputs ────────────────────────────────────────────
        overlay_png      = overlay_heatmap(img_array, heatmap_128)
        heatmap_only_png = _heatmap_to_png(heatmap_128)

        # Layer name for audit trail (used in forensic reports)
        layer_name = (
            model.gradcam_layer.__class__.__name__
            + ' (block3.conv2)'
        )

        return {
            'heatmap_array':    heatmap_128,
            'overlay_png':      overlay_png,
            'heatmap_only_png': heatmap_only_png,
            'target_class':     target_class,
            'captum_layer':     layer_name,
        }

    except Exception as e:
        print(f"[MalTwin] GradCAM failed: {e}", file=sys.stderr)
        return None


def overlay_heatmap(
    img_array: np.ndarray,
    heatmap_array: np.ndarray,
    alpha: float = 0.5,
) -> bytes:
    """
    Overlay a jet-colormap heatmap onto the original grayscale image.

    Args:
        img_array:    uint8 numpy array (128, 128) — original grayscale image.
        heatmap_array:float32 numpy array (128, 128) values [0, 1].
        alpha:        blend weight for heatmap overlay (0 = image only, 1 = heatmap only).

    Returns:
        PNG bytes of the blended RGB image.

    Steps:
        1. Convert heatmap float32 [0,1] → uint8 [0,255].
        2. Apply cv2.COLORMAP_JET → BGR uint8 (H, W, 3).
        3. Convert BGR → RGB.
        4. Convert grayscale to RGB by stacking 3 channels.
        5. Blend: result = (1-alpha)*grayscale_rgb + alpha*jet_rgb, clamp to uint8.
        6. Return as PNG bytes via PIL.
    """
    # Jet colormap (uint8 [0,255] heatmap → BGR)
    heatmap_uint8 = (heatmap_array * 255).astype(np.uint8)
    jet_bgr  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb  = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)           # RGB

    # Grayscale → RGB (stack 3 identical channels)
    gray_rgb = np.stack([img_array, img_array, img_array], axis=-1).astype(np.float32)

    # Blend
    blended = (1.0 - alpha) * gray_rgb + alpha * jet_rgb.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(blended, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return buf.getvalue()


def _heatmap_to_png(heatmap_array: np.ndarray) -> bytes:
    """
    Convert a float32 [0,1] heatmap to a standalone PNG with the jet colormap
    and a colorbar for inclusion in forensic reports.

    Args:
        heatmap_array: float32 numpy array (128, 128) values [0, 1].

    Returns:
        PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap_array, cmap='jet', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Grad-CAM Attribution', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight')
    plt.close(fig)   # mandatory — prevent memory leak
    buf.seek(0)
    return buf.getvalue()
```

---

## File 2: Update `modules/detection/__init__.py`

Replace the existing file content:

```python
# modules/detection/__init__.py
from .model import MalTwinCNN
from .trainer import train
from .evaluator import evaluate
from .inference import load_model, predict_single
from .gradcam import generate_gradcam, overlay_heatmap
```

---

## File 3: Update `modules/dashboard/state.py`

Add `KEY_HEATMAP` to the constants and helpers. Add the new key in two places:

**Add to the constants block** (after `KEY_DEVICE_INFO`):
```python
KEY_HEATMAP = 'gradcam_heatmap'  # dict from generate_gradcam() or None
```

**Add to `init_session_state()` defaults dict**:
```python
KEY_HEATMAP: None,
```

**Add to `clear_file_state()`**:
```python
st.session_state[KEY_HEATMAP] = None
```

**Add helper function** (after `is_model_loaded()`):
```python
def has_heatmap() -> bool:
    return st.session_state.get(KEY_HEATMAP) is not None
```

---

## File 4: Update `modules/dashboard/pages/detection.py`

### 4a — Replace the XAI heatmap section (`_render_results`)

Find and replace the entire `# ── E: XAI Heatmap (STUB)` block with:

```python
    # ── E: XAI Heatmap ────────────────────────────────────────────────────────
    st.subheader("Explainable AI — Grad-CAM Heatmap")
    st.caption(
        "Highlights which byte regions of the binary image drove the classification decision.",
        help=(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients "
            "flowing into the final convolutional layer to produce a heatmap. "
            "Red/warm regions contributed most strongly to the predicted class."
        ),
    )

    xai_requested = st.checkbox(
        "Generate Grad-CAM Heatmap",
        key="xai_checkbox",
        help="Generates a heatmap showing which byte regions drove the detection. "
             "Adds ~2–8 seconds depending on hardware.",
    )

    if xai_requested:
        _run_gradcam()

    if state.has_heatmap():
        _render_heatmap()
```

### 4b — Add `_run_gradcam()` function

Add this new function to `detection.py` (alongside the other private functions like `_run_detection`):

```python
def _run_gradcam() -> None:
    """
    Generate Grad-CAM heatmap for the current detection result.
    Stores result in session_state[KEY_HEATMAP].
    Never raises — generate_gradcam() handles all exceptions internally.
    """
    if state.has_heatmap():
        return   # Already generated for this detection — don't regenerate

    result = st.session_state.get(state.KEY_DETECTION)
    if result is None:
        st.warning("Run detection first before generating the heatmap.")
        return

    model     = st.session_state[state.KEY_MODEL]
    class_names = st.session_state[state.KEY_CLASS_NAMES]
    img_array = st.session_state[state.KEY_IMG_ARRAY]
    target_class = class_names.index(result['predicted_family'])

    with st.spinner("Generating Grad-CAM heatmap… (this may take a few seconds)"):
        from modules.detection.gradcam import generate_gradcam
        heatmap_data = generate_gradcam(model, img_array, target_class, config.DEVICE)

    if heatmap_data is None:
        st.error(
            "Error: Heatmap generation failed. "
            "Cause: Captum backward pass error or incompatible model state. "
            "Action: Ensure the model was loaded correctly and try again."
        )
        return

    st.session_state[state.KEY_HEATMAP] = heatmap_data
    st.rerun()   # Rerun to render heatmap section below
```

### 4c — Add `_render_heatmap()` function

```python
def _render_heatmap() -> None:
    """
    Display the Grad-CAM heatmap results: overlay + standalone heatmap side by side.
    Also provides a download button for the overlay PNG (used in M8 reports).
    """
    heatmap_data = st.session_state[state.KEY_HEATMAP]
    result       = st.session_state[state.KEY_DETECTION]

    col_overlay, col_heatmap = st.columns(2)

    with col_overlay:
        st.markdown("**Heatmap Overlay** — warm regions drove the prediction")
        st.image(
            heatmap_data['overlay_png'],
            caption=(
                f"Grad-CAM overlay for predicted class: "
                f"{result['predicted_family']} "
                f"({result['confidence']*100:.1f}% confidence)"
            ),
            use_column_width=True,
        )

    with col_heatmap:
        st.markdown("**Raw Attribution Map** — with colorbar")
        st.image(
            heatmap_data['heatmap_only_png'],
            caption="Jet colormap: red = high attribution, blue = low attribution",
            use_column_width=True,
        )

    # Interpretation text (SRS M7 FE-3)
    st.markdown("**Interpretation:**")
    st.info(
        "🔴 **Red/warm regions** — byte offsets in these areas of the binary "
        "were the strongest signals for the predicted classification. "
        "These regions often correspond to code sections with distinctive "
        "structural patterns (entry points, encrypted payloads, PE headers). "
        "\n\n"
        "🔵 **Blue/cool regions** — these byte regions had low influence "
        "on the classification decision."
    )

    # Export button for heatmap overlay (feeds into M8 PDF report)
    st.download_button(
        label="📥 Download Heatmap Overlay PNG",
        data=heatmap_data['overlay_png'],
        file_name=f"gradcam_{result['predicted_family'].replace('.', '_')}.png",
        mime="image/png",
        help="Download the Grad-CAM heatmap overlay for inclusion in forensic reports.",
    )
```

### 4d — Update the JSON export in `_render_results`

In the existing JSON export block, add the heatmap layer info to `export_data`:

```python
    # Add GradCAM audit info if heatmap was generated
    if state.has_heatmap():
        heatmap_data = st.session_state[state.KEY_HEATMAP]
        export_data['gradcam'] = {
            'generated':   True,
            'target_class': heatmap_data['target_class'],
            'layer':        heatmap_data['captum_layer'],
        }
    else:
        export_data['gradcam'] = {'generated': False}
```

---

## File 5: `tests/test_gradcam.py`

```python
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
```

---

## Dependency Check

Before running tests, verify Captum is installed:

```bash
python -c "import captum; print(captum.__version__)"
```

If not installed:
```bash
pip install captum --break-system-packages
```

Also add to `requirements.txt` if not already present:
```
captum>=0.7.0
```

---

## Definition of Done

```bash
# Run Grad-CAM tests
pytest tests/test_gradcam.py -v

# Expected: all 20 tests pass, 0 failures
# The test_returns_none_on_bad_target_class test verifies the never-raises contract.
# The test_deterministic_on_eval_model test verifies reproducibility.

# Regression — all earlier tests still pass
pytest tests/test_converter.py tests/test_dataset.py \
       tests/test_enhancement.py tests/test_model.py tests/test_db.py \
       -v -m "not integration"

# Dashboard smoke test — import must succeed
python -c "from modules.detection.gradcam import generate_gradcam, overlay_heatmap"
python -c "from modules.detection import generate_gradcam"
```

### Checklist

- [ ] `pytest tests/test_gradcam.py -v` — 0 failures
- [ ] All earlier test files still pass (no regressions)
- [ ] `modules/detection/gradcam.py` exists
- [ ] `modules/detection/__init__.py` exports `generate_gradcam`
- [ ] `state.py` has `KEY_HEATMAP` constant
- [ ] `state.py` `init_session_state()` initialises `KEY_HEATMAP = None`
- [ ] `state.py` `clear_file_state()` clears `KEY_HEATMAP`
- [ ] `state.py` has `has_heatmap()` helper
- [ ] `detection.py` XAI section is no longer a stub — real heatmap renders
- [ ] `_run_gradcam()` uses `model.gradcam_layer` (not a hardcoded string)
- [ ] `_run_gradcam()` does NOT wrap Captum call in `torch.no_grad()`
- [ ] `overlay_heatmap()` returns RGB (not BGR)
- [ ] `plt.close(fig)` called in `_heatmap_to_png()`
- [ ] `generate_gradcam()` returns `None` on failure — does not raise
- [ ] JSON export in `_render_results()` includes `gradcam` key

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|---|---|---|
| `torch.no_grad()` wrapping Captum call | All attributions are zero — no gradients flow | Remove `no_grad()` from around `grad_cam.attribute()` |
| Hardcoding `model.block3.conv2` as string | Breaks if model changes; not using the hook point we set up | Always use `model.gradcam_layer` |
| BGR colormap returned from `cv2.applyColorMap` | Overlay looks blue where it should look red | `cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)` before blending |
| Forgetting `plt.close(fig)` | Memory leak on repeated heatmap generations | Always close after `plt.savefig()` |
| `generate_gradcam` raising instead of returning `None` | Dashboard page crashes on heatmap failure | Wrap entire body in `try/except Exception` |
| Heatmap not cleared on new upload | Previous file's heatmap shows for new file | `clear_file_state()` already clears `KEY_HEATMAP` — ensure it's called |
| `st.rerun()` called inside `_render_heatmap()` | Infinite rerun loop | Only call `st.rerun()` inside `_run_gradcam()` after storing result, never in render |

---

*Step 1 complete → move to Step 2: M8 Automated Forensic Reporting (PDF + JSON pipeline).*
