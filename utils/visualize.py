"""
utils/visualize.py
------------------
Visualization utilities for FreqMerge.

  1. visualize_freq_scores  – JET-colormap heatmap of LFGM output (φ̃)
                              overlaid on the original image.
  2. plot_training_curves   – Loss and accuracy curves over epochs.
  3. plot_confusion_matrix  – Per-class confusion matrix (top-N classes).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless — works on remote / no-display servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

from models.lfgm import LFGM


# ---------------------------------------------------------------------------
# 1.  LFGM frequency heatmap
# ---------------------------------------------------------------------------

def visualize_freq_scores(
    images:     torch.Tensor,
    model:      torch.nn.Module,
    save_path:  str  = None,
    n_samples:  int  = 8,
    grid_h:     int  = 14,
    grid_w:     int  = 14,
    hpf_radius: int  = cfg.HPF_RADIUS,
    device:     str  = "cpu",
    denorm_mean: list = cfg.NORM_MEAN,
    denorm_std:  list = cfg.NORM_STD,
) -> None:
    """
    For each of the first ``n_samples`` images in the batch:
      - Runs a single forward pass through the patch embedding to obtain tokens.
      - Passes the spatial tokens through LFGM to get φ̃ scores.
      - Upsamples the 14×14 score map to the full image size.
      - Overlays it as a JET colormap heatmap on top of the original image.

    Parameters
    ----------
    images     : (B, 3, H, W) normalised image tensor.
    model      : FreqMerge ViT model (used only to extract patch embeddings).
    save_path  : If given, the figure is saved there; otherwise plt.show().
    n_samples  : Number of images to visualise (≤ B).
    """
    model.eval().to(device)
    images = images[:n_samples].to(device)
    B      = images.shape[0]

    lfgm = LFGM(grid_h=grid_h, grid_w=grid_w, hpf_radius=hpf_radius).to(device)

    with torch.no_grad():
        # Extract patch embeddings (before transformer blocks)
        # timm ViT: patch_embed then add pos_embed; cls token prepended
        x = model.patch_embed(images)          # (B, N_s, D)
        # If the model uses absolute pos embed, add it (skip CLS slot)
        if hasattr(model, "pos_embed"):
            x = x + model.pos_embed[:, 1:, :]

        # Compute φ̃ via LFGM
        phi = lfgm(x)                          # (B, N_s)  ∈ [0, 1]

        # Reshape to spatial map (B, 1, H, W) and upsample to image size
        phi_map = phi.reshape(B, 1, grid_h, grid_w)
        H, W    = images.shape[-2:]
        phi_up  = F.interpolate(
            phi_map, size=(H, W), mode="bilinear", align_corners=False
        )                                      # (B, 1, H, W)
        phi_up  = phi_up.squeeze(1).cpu().numpy()  # (B, H, W)

    # De-normalise images for display
    mean = torch.tensor(denorm_mean).view(1, 3, 1, 1)
    std  = torch.tensor(denorm_std).view(1, 3, 1, 1)
    imgs_disp = (images.cpu() * std + mean).clamp(0, 1)
    imgs_np   = imgs_disp.permute(0, 2, 3, 1).numpy()   # (B, H, W, 3)

    # Build figure
    ncols = min(n_samples, 4)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 3, nrows * 6))
    axes = np.array(axes).reshape(nrows * 2, ncols)

    colormap = cm.jet

    for i in range(n_samples):
        row_img  = (i // ncols) * 2
        row_heat = row_img + 1
        col      = i % ncols

        # Original image
        axes[row_img][col].imshow(imgs_np[i])
        axes[row_img][col].set_title(f"Sample {i+1}", fontsize=9)
        axes[row_img][col].axis("off")

        # LFGM heatmap overlay
        heatmap = colormap(phi_up[i])[:, :, :3]   # RGB, drop alpha
        blend   = 0.5 * imgs_np[i] + 0.5 * heatmap
        axes[row_heat][col].imshow(blend.clip(0, 1))
        axes[row_heat][col].set_title("LFGM φ̃ (JET)", fontsize=9)
        axes[row_heat][col].axis("off")

    plt.suptitle("LFGM Frequency Score Heatmaps", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Heatmap saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 2.  Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history:   dict,
    save_path: str = None,
) -> None:
    """
    Plot loss and top-1 accuracy curves for train and validation splits.

    Parameters
    ----------
    history : dict with keys:
                'train_loss', 'val_loss',
                'train_acc1', 'val_acc1'
              each mapping to a list of per-epoch floats.
    save_path : Optional file path to save the figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Loss ----------------------------------------------------------
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Accuracy ------------------------------------------------------
    ax2.plot(epochs, history["train_acc1"], "b-o", markersize=4, label="Train Top-1")
    ax2.plot(epochs, history["val_acc1"],   "r-o", markersize=4, label="Val Top-1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("Training & Validation Top-1 Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Training curves saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3.  Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    all_preds:  list,
    all_labels: list,
    class_names: list = None,
    top_n:      int  = 20,
    save_path:  str  = None,
) -> None:
    """
    Plot a confusion matrix for the top-N most confused classes.

    Parameters
    ----------
    all_preds   : List / array of predicted class indices.
    all_labels  : List / array of ground-truth class indices.
    class_names : Optional list of string class names (length = num_classes).
    top_n       : Show only the top_n classes with the highest error counts.
    save_path   : Optional file path.
    """
    import sklearn.metrics as skm

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = skm.confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]

    # Select top_n classes by total error (off-diagonal sum per row)
    errors = cm.sum(axis=1) - np.diag(cm)
    top_n  = min(top_n, num_classes)
    top_idx = np.argsort(errors)[-top_n:][::-1]

    cm_sub = cm[np.ix_(top_idx, top_idx)]
    # Row-normalise to show recall per class
    row_sums = cm_sub.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm_sub.astype(float) / row_sums

    labels = [class_names[i] if class_names else str(i) for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.6), max(7, top_n * 0.55)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"Confusion Matrix — Top-{top_n} Most Confused Classes", fontsize=11)

    # Annotate cells
    for row in range(top_n):
        for col in range(top_n):
            val = cm_norm[row, col]
            color = "white" if val > 0.5 else "black"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Confusion matrix saved → {save_path}")
    else:
        plt.show()
    plt.close()
