"""
train.py
--------
Main training script for FreqMerge on CIFAR-100.

CUDA features
-------------
  • Automatic Mixed Precision (AMP) via torch.cuda.amp.GradScaler
    — float16 forward/backward, float32 parameter updates
    — ~2× throughput on Ampere GPUs (RTX 30xx) with no accuracy loss
  • cuDNN auto-tune (benchmark=True) for fixed 224×224 inputs
  • TF32 matmul on Ampere GPUs (enabled in setup_cuda)
  • nn.DataParallel across all available GPUs (if USE_MULTI_GPU=True)
  • Per-epoch CUDA memory stats and configurable cache clearing
  • CUDA Event timing for accurate epoch-time measurement
  • Gradient clipping compatible with GradScaler (unscale before clip)

Run
---
    python train.py                          # default config
    python train.py --alpha 0.7              # custom frequency penalty
    python train.py --no_amp                 # disable mixed precision
    python train.py --no_freqmerge           # plain ViT-Small baseline
"""

import os
import sys
import argparse
import random
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from data.cifar100 import get_cifar100_loaders
from models.vit_freqmerge import build_freqmerge_vit
from utils.metrics import AverageMeter, accuracy
from utils.visualize import plot_training_curves
from utils.cuda_utils import (
    setup_cuda,
    print_gpu_info,
    print_memory_stats,
    reset_peak_memory_stats,
    clear_cuda_cache,
    CUDATimer,
    maybe_wrap_data_parallel,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    # Default values are plain literals — never cfg.* — to avoid partial-import
    # issues where config.py attributes are not yet defined.
    p = argparse.ArgumentParser(description="FreqMerge Training")
    p.add_argument("--backbone",      type=str,   default="vit_small_patch16_224")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--keep_rate",     type=float, default=0.7)
    p.add_argument("--alpha",         type=float, default=0.7)
    p.add_argument("--hpf_radius",    type=int,   default=2)
    p.add_argument(
        "--merge_layers", type=int, nargs="+", default=[4, 6, 8, 10],
        help="Encoder block indices where DTM is applied (0-indexed).",
    )
    p.add_argument("--no_freqmerge",  action="store_true",
                   help="Train plain ViT-Small baseline without FreqMerge.")
    p.add_argument("--pretrained",    action="store_true", default=True)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--ckpt_dir",      type=str,   default=None)
    p.add_argument("--log_dir",       type=str,   default=None)
    p.add_argument("--viz_dir",       type=str,   default=None)
    # CUDA-specific flags
    p.add_argument("--no_amp",        action="store_true",
                   help="Disable Automatic Mixed Precision (AMP / float16).")
    p.add_argument("--no_multi_gpu",  action="store_true",
                   help="Disable DataParallel even if multiple GPUs exist.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler:      GradScaler,
    device:      torch.device,
    epoch:       int,
    total_epochs:int,
    use_amp:     bool = True,
) -> tuple[float, float]:
    """
    Train for one epoch with optional AMP (float16) acceleration.

    The GradScaler handles loss scaling to prevent float16 underflow:
      1. scaler.scale(loss).backward()   — backward with scaled gradients
      2. scaler.unscale_(optimizer)      — restore true gradient magnitudes
      3. clip_grad_norm_(...)            — clip AFTER unscaling
      4. scaler.step(optimizer)          — skips update if grads are inf/nan
      5. scaler.update()                 — adjust scale factor for next step

    Returns
    -------
    (avg_loss, avg_top1_acc)
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    acc1_meter = AverageMeter("Top-1")

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch [{epoch}/{total_epochs}] Train",
        ncols=115,
        leave=False,
    )

    for batch_idx, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)   # non_blocking=True
        labels = labels.to(device, non_blocking=True)   # overlaps H→D with compute

        # ── Forward (AMP context) ────────────────────────────────────
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Backward (scaled) ────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)        # faster than zero_grad()
        scaler.scale(loss).backward()

        # ── Gradient clipping (must unscale first) ───────────────────
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        # ── Optimizer step (skips if inf/nan grads) ──────────────────
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics ──────────────────────────────────────────────────
        with torch.no_grad():
            acc1, _ = accuracy(logits.float(), labels, topk=(1, 5))
        B = images.size(0)
        loss_meter.update(loss.item(), B)
        acc1_meter.update(acc1, B)

        if (batch_idx + 1) % cfg.PRINT_FREQ == 0:
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{acc1_meter.avg:.2f}%",
                "scale": f"{scaler.get_scale():.0f}",   # AMP scale factor
            })

    return loss_meter.avg, acc1_meter.avg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device:  torch.device,
    use_amp: bool = True,
) -> tuple[float, float]:
    """Evaluate on the validation set. AMP is used for inference too."""
    model.eval()
    loss_meter = AverageMeter("Val-Loss")
    acc1_meter = AverageMeter("Val-Top-1")

    for images, labels in tqdm(loader, desc="Validation", ncols=115, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        acc1, _ = accuracy(logits.float(), labels, topk=(1, 5))
        B = images.size(0)
        loss_meter.update(loss.item(), B)
        acc1_meter.update(acc1, B)

    return loss_meter.avg, acc1_meter.avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    # Resolve path defaults that couldn't use cfg.* in argparse
    if args.ckpt_dir is None: args.ckpt_dir = cfg.CKPT_DIR
    if args.log_dir  is None: args.log_dir  = cfg.LOG_DIR
    if args.viz_dir  is None: args.viz_dir  = cfg.VIZ_DIR

    use_amp = (not args.no_amp) and cfg.USE_AMP

    # ── CUDA setup ──────────────────────────────────────────────────────
    device = setup_cuda(seed=args.seed)
    print_gpu_info()

    # ── Directories ─────────────────────────────────────────────────────
    for d in (args.ckpt_dir, args.log_dir, args.viz_dir):
        os.makedirs(d, exist_ok=True)

    # ── Banner ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FreqMerge Training — CIFAR-100")
    print(f"  Device       : {device}  "
          f"({'AMP float16 enabled' if use_amp else 'full float32'})")
    print(f"  FreqMerge    : {'DISABLED (baseline)' if args.no_freqmerge else 'ENABLED'}")
    if not args.no_freqmerge:
        print(f"  Merge layers : {args.merge_layers}")
        print(f"  Keep rate    : {args.keep_rate}   Alpha (α): {args.alpha}")
    print(f"  Epochs       : {args.epochs}   Batch: {args.batch_size}   LR: {args.lr}")
    print(f"{'='*62}\n")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = get_cifar100_loaders(batch_size=args.batch_size)

    # ── Model ───────────────────────────────────────────────────────────
    merge_layers = [] if args.no_freqmerge else args.merge_layers
    model = build_freqmerge_vit(
        num_classes  = cfg.NUM_CLASSES,
        merge_layers = merge_layers,
        keep_rate    = args.keep_rate,
        alpha        = args.alpha,
        hpf_radius   = args.hpf_radius,
        pretrained   = args.pretrained,
        backbone     = args.backbone,
    ).to(device)

    # Multi-GPU wrapping (DataParallel)
    if not args.no_multi_gpu:
        model = maybe_wrap_data_parallel(model, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print_memory_stats("after model load", device)

    # ── Loss / Optimizer / Scheduler / GradScaler ───────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Unwrap DataParallel to access named parameters
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    backbone_params  = []
    freqmerge_params = []
    head_params      = []
    for name, param in raw_model.named_parameters():
        if "head" in name:
            head_params.append(param)
        elif "merge_block" in name or "lfgm" in name:
            freqmerge_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params,  "lr": args.lr * 0.1},
        {"params": head_params,      "lr": args.lr},
        {"params": freqmerge_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.LR_MIN)

    # GradScaler — only active when use_amp=True and device is cuda
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ── Training loop ───────────────────────────────────────────────────
    history  = {"train_loss": [], "val_loss": [], "train_acc1": [], "val_acc1": [], "lr": []}
    best_val = 0.0
    best_ckpt = os.path.join(args.ckpt_dir, "best_model.pth")

    print(f"Loading CIFAR-100 and resizing to {cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE}...")
    if not args.no_freqmerge:
        print(f"Starting training with FreqMerge "
              f"(keep_rate={args.keep_rate}, alpha={args.alpha})...")
    else:
        print("Starting training — plain ViT-Small baseline...")

    for epoch in range(1, args.epochs + 1):
        reset_peak_memory_stats(device)

        # ── Timed training pass (CUDA Events for accurate GPU timing) ──
        with CUDATimer(f"epoch {epoch} train", verbose=False) as t_train:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler, device, epoch, args.epochs, use_amp,
            )

        # ── Timed validation pass ──────────────────────────────────────
        with CUDATimer(f"epoch {epoch} val", verbose=False) as t_val:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, use_amp,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep_time_s = (t_train.elapsed_ms + t_val.elapsed_ms) / 1000.0

        # ── Record ─────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc1"].append(train_acc)
        history["val_acc1"].append(val_acc)
        history["lr"].append(current_lr)

        # ── Console output ─────────────────────────────────────────────
        print(f"--- Epoch {epoch}/{args.epochs} Completed in {ep_time_s:.1f}s ---")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Acc: {val_acc:.2f}%")
        print_memory_stats(f"epoch {epoch}", device)

        # ── Save best checkpoint ───────────────────────────────────────
        if val_acc > best_val:
            best_val = val_acc
            save_dict = {
                "epoch":      epoch,
                "state_dict": raw_model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scaler":     scaler.state_dict(),     # save AMP scale state
                "val_acc":    val_acc,
                "args":       vars(args),
            }
            torch.save(save_dict, best_ckpt)
            print(f"  ✓ New best: {val_acc:.2f}%  →  {best_ckpt}")

        # ── Periodic CUDA cache clear ──────────────────────────────────
        n = cfg.EMPTY_CACHE_EVERY_N_EPOCHS
        if n > 0 and epoch % n == 0:
            clear_cuda_cache()
            print(f"  [CUDA] Cache cleared at epoch {epoch}.")

    # ── Post-training ────────────────────────────────────────────────────
    history_path = os.path.join(args.log_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {history_path}")

    curve_path = os.path.join(args.viz_dir, "training_curves.png")
    plot_training_curves(history, save_path=curve_path)

    print(f"\n{'='*62}")
    print(f"  Training complete.")
    print(f"  Best Val Top-1 : {best_val:.2f}%")
    print(f"  Checkpoint     : {best_ckpt}")
    print_memory_stats("final", device)
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
