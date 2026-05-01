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
        pil_img = Image.fromarray(img_array, mode='L')
        tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,1,128,128)
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
            attr_np = attr_np.mean(axis=0)  # (16, 16)

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
        overlay_png = overlay_heatmap(img_array, heatmap_128)
        heatmap_only_png = _heatmap_to_png(heatmap_128)

        # Layer name for audit trail (used in forensic reports)
        layer_name = (
            model.gradcam_layer.__class__.__name__
            + ' (block3.conv2)'
        )

        return {
            'heatmap_array': heatmap_128,
            'overlay_png': overlay_png,
            'heatmap_only_png': heatmap_only_png,
            'target_class': target_class,
            'captum_layer': layer_name,
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
    jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)  # RGB

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
    plt.close(fig)  # mandatory — prevent memory leak
    buf.seek(0)
    return buf.getvalue()
