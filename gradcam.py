"""
Grad-CAM (Gradient-weighted Class Activation Mapping) — pure PyTorch
=====================================================================
Supports ResNet-18 and EfficientNet-B0 out of the box.
No external dependencies beyond torch, torchvision, numpy, and matplotlib.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (must match TransformPipeline)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# GRAD-CAM ENGINE
# ============================================================================

class GradCAM:
    """
    Context-manager-safe Grad-CAM implementation using PyTorch hooks.

    Usage
    -----
    gc = GradCAM(model, target_layer)
    try:
        heatmap, pred_class = gc(image_tensor)
    finally:
        gc.remove_hooks()

    Parameters
    ----------
    model        : trained nn.Module in eval mode
    target_layer : nn.Module whose output activations are used for the CAM
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        def _fwd_hook(module: nn.Module, inp, out: torch.Tensor) -> None:
            self._activations = out  # retain for grad computation

        def _bwd_hook(module: nn.Module, grad_in, grad_out) -> None:
            # grad_out[0]: gradient w.r.t. this layer's OUTPUT (shape B,C,H,W)
            self._gradients = grad_out[0].detach()

        self._fwd_handle = target_layer.register_forward_hook(_fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Always call this after use."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Compute the Grad-CAM heatmap for one image.

        Parameters
        ----------
        image_tensor : Tensor of shape (1, C, H, W) on the same device as model
        target_class : class index to explain; uses argmax prediction if None

        Returns
        -------
        heatmap     : np.ndarray of shape (H_feat, W_feat) in [0, 1]
        pred_class  : int index of the explained class
        logits      : raw model output Tensor (for downstream use)
        """
        self.model.zero_grad()
        logits = self.model(image_tensor)           # (1, num_classes)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Back-prop only the score of the target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot)

        # self._activations: (1, C, Hf, Wf)
        # self._gradients:   (1, C, Hf, Wf)
        weights = self._gradients[0].mean(dim=[1, 2])          # (C,)
        cam = (weights[:, None, None] * self._activations[0]).sum(dim=0)  # (Hf, Wf)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, logits.detach()


# ============================================================================
# LAYER SELECTION HELPERS
# ============================================================================

def get_target_layer(model: nn.Module, model_type: str = "resnet18") -> nn.Module:
    """
    Return the standard Grad-CAM target layer for common architectures.

    Handles both plain torchvision models (main.py) and the wrapped
    BinaryClassificationModel from training_pipeline.py (with a .backbone attr).

    Parameters
    ----------
    model      : the nn.Module to inspect
    model_type : 'resnet18' or 'efficientnet_b0'

    Returns
    -------
    nn.Module that will be hooked

    Raises
    ------
    ValueError  : if model_type is not recognised
    """
    # Unwrap BinaryClassificationModel if needed
    backbone = getattr(model, "backbone", model)

    if model_type == "resnet18":
        return backbone.layer4[-1]
    elif model_type == "efficientnet_b0":
        return backbone.features[-1]
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Pass the target layer directly to generate_gradcam()."
        )


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and return an (H, W, 3) float32 array in [0, 1].

    Parameters
    ----------
    image_tensor : Tensor of shape (1, 3, H, W) or (3, H, W)
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)          # (3, H, W)
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _overlay_heatmap(
    img_arr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a jet-coloured heatmap over an RGB image.

    Parameters
    ----------
    img_arr : float32 (H, W, 3) in [0, 1] — the original image
    heatmap : float32 (H, W) in [0, 1]
    alpha   : opacity of the heatmap overlay

    Returns
    -------
    float32 (H, W, 3) blended image in [0, 1]
    """
    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3].astype(np.float32)   # (H, W, 3)
    overlay = (1.0 - alpha) * img_arr + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0)


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap and the corresponding heatmap-over-image overlay.

    The model is temporarily set to eval() mode for BatchNorm/Dropout stability
    while gradients are enabled for the CAM computation.

    Parameters
    ----------
    model        : trained nn.Module
    image_tensor : Tensor of shape (1, C, H, W); must be on the correct device
    target_layer : nn.Module to hook (use get_target_layer for convenience)
    target_class : class to explain (None → argmax of model output)

    Returns
    -------
    heatmap : np.ndarray of shape (H, W) in [0, 1] — resized to image resolution
    overlay : np.ndarray of shape (H, W, 3) in [0, 1] — heatmap blended over image
    """
    was_training = model.training
    model.eval()

    gc = GradCAM(model, target_layer)
    try:
        with torch.enable_grad():
            raw_cam, pred_class, _ = gc(image_tensor, target_class)
    finally:
        gc.remove_hooks()
        model.train(was_training)

    # Resize CAM to the spatial resolution of the input image
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam_tensor = torch.tensor(raw_cam[None, None])          # (1, 1, Hf, Wf)
    heatmap = (
        F.interpolate(cam_tensor, size=(h, w), mode="bilinear", align_corners=False)
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    img_arr = _denormalize(image_tensor)
    overlay = _overlay_heatmap(img_arr, heatmap)
    return heatmap, overlay


# ============================================================================
# BATCH VISUALISATION
# ============================================================================

def save_gradcam_examples(
    model: nn.Module,
    dataloader,
    output_dir: str,
    target_layer: nn.Module,
    n_per_category: int = 5,
    threshold: float = 0.5,
    positive_label: int = 1,
    device: str = "cpu",
) -> None:
    """
    Generate and save Grad-CAM visualisations for TP, TN, FP, and FN cases.

    Saves individual side-by-side panels (original | overlay) for each example
    and a combined summary grid ``gradcam_grid.png``.

    Parameters
    ----------
    model          : trained nn.Module
    dataloader     : DataLoader returning (images, labels [, metadata])
    output_dir     : directory where Grad-CAM images are written
    target_layer   : nn.Module to use as the Grad-CAM target
    n_per_category : maximum examples to save per category (TP/TN/FP/FN)
    threshold      : decision threshold for positive class
    positive_label : index of the positive class (default: 1)
    device         : 'cpu' or 'cuda'
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # ── Collect representative examples ────────────────────────────────────
    categories: Dict[str, List[torch.Tensor]] = {
        "TP": [], "TN": [], "FP": [], "FN": []
    }
    total_needed = n_per_category * 4

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            labels = batch[1]

            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1)[:, positive_label].cpu()
            preds  = (probs >= threshold).int()

            for img, lbl, pred in zip(images, labels, preds):
                l, p = int(lbl), int(pred)
                key = ("TP" if l == 1 and p == 1 else
                       "TN" if l == 0 and p == 0 else
                       "FP" if l == 0 and p == 1 else "FN")
                if len(categories[key]) < n_per_category:
                    categories[key].append(img.cpu())

            if all(len(v) >= n_per_category for v in categories.values()):
                break

    # ── Generate Grad-CAMs and save individual panels ───────────────────────
    overlays_by_cat: Dict[str, List[np.ndarray]] = {k: [] for k in categories}

    for category, images in categories.items():
        for i, img in enumerate(images):
            img_tensor = img.unsqueeze(0).to(device)
            try:
                heatmap, overlay = generate_gradcam(
                    model, img_tensor, target_layer, target_class=positive_label
                )
            except Exception as exc:
                logger.warning(f"Grad-CAM failed for {category}[{i}]: {exc}")
                continue

            overlays_by_cat[category].append(overlay)

            # Side-by-side panel: original | overlay
            img_arr = _denormalize(img_tensor)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_arr)
            axes[0].set_title("Original", fontsize=11)
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Grad-CAM ({category})", fontsize=11)
            axes[1].axis("off")
            fig.suptitle(f"{category} — example {i+1}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            panel_path = out / f"{category}_{i+1:02d}.png"
            plt.savefig(panel_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    # ── Summary grid (4 columns × n_per_category rows) ──────────────────────
    cat_order = ["TP", "TN", "FP", "FN"]
    cat_labels = {
        "TP": "True Positive\n(correctly detected malignant)",
        "TN": "True Negative\n(correctly dismissed benign)",
        "FP": "False Positive\n(falsely flagged benign)",
        "FN": "False Negative\n(missed malignant)",
    }
    n_rows = max(len(overlays_by_cat[k]) for k in cat_order)
    if n_rows == 0:
        logger.warning("No Grad-CAM examples collected — grid not created.")
        return

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(5 * 4, 4 * n_rows),
        squeeze=False,
    )

    for col, cat in enumerate(cat_order):
        for row in range(n_rows):
            ax = axes[row, col]
            if row < len(overlays_by_cat[cat]):
                ax.imshow(overlays_by_cat[cat][row])
            else:
                ax.set_facecolor("#F3F4F6")
            if row == 0:
                ax.set_title(cat_labels[cat], fontsize=10, fontweight="bold", pad=6)
            ax.axis("off")

    fig.suptitle("Grad-CAM visualisations by prediction category",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    grid_path = out / "gradcam_grid.png"
    plt.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grad-CAM grid saved: {grid_path}")
    print(f"  Grad-CAM grid saved: {grid_path}")
