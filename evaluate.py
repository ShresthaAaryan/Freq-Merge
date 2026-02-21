"""
evaluate.py
-----------
Evaluation script for a trained FreqMerge model.

Run
---
    python evaluate.py --ckpt checkpoints/best_model.pth
    python evaluate.py --ckpt checkpoints/best_model.pth --benchmark
    python evaluate.py --ckpt checkpoints/best_model.pth --visualize

Outputs
-------
    - Top-1 / Top-5 validation accuracy
    - GFLOPs estimate (via a dummy forward pass hook)
    - Throughput (images/sec) if --benchmark is set
    - Heatmap visualisation if --visualize is set
    - Confusion matrix if --confusion is set
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.amp import autocast

import config as cfg
from data.cifar100 import get_cifar100_loaders
from models.vit_freqmerge import build_freqmerge_vit
from utils.metrics import AverageMeter, accuracy, compute_throughput
from utils.visualize import (
    visualize_freq_scores,
    plot_confusion_matrix,
)
from utils.cuda_utils import (
    setup_cuda,
    print_gpu_info,
    print_memory_stats,
    CUDATimer,
)


# ---------------------------------------------------------------------------
# FLOP counting (via forward hook)
# ---------------------------------------------------------------------------

def count_flops(model: nn.Module, input_size=(1, 3, 224, 224), device="cpu") -> float:
    """
    Estimate total GFLOPs for a single forward pass via a forward hook on
    nn.Linear and nn.Conv2d layers.

    Note: This is an approximation — attention FLOPs are counted separately
    because timm's MHSA is implemented as nn.Linear projections.

    Returns
    -------
    float  GFLOPs (10^9 FLOPs)
    """
    flops = [0.0]

    def linear_hook(module, inp, out):
        # FLOPs for a linear layer: 2 * in_features * out_features * batch_elements
        in_feat  = module.in_features
        out_feat = module.out_features
        # inp[0] shape: (B, *, in_feat)
        batch_el = inp[0].numel() // in_feat
        flops[0] += 2.0 * in_feat * out_feat * batch_el

    def conv_hook(module, inp, out):
        # FLOPs for conv: 2 * Cin * Cout * k^2 * H_out * W_out
        Cout, Cin_g, kH, kW = module.weight.shape
        H_out, W_out = out.shape[-2], out.shape[-1]
        groups  = module.groups
        Cin     = Cin_g * groups
        batch   = out.shape[0]
        flops[0] += 2.0 * (Cin / groups) * Cout * kH * kW * H_out * W_out * batch

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    dummy = torch.zeros(*input_size).to(device)
    model.eval().to(device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flops[0] / 1e9


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FreqMerge Evaluation")
    parser.add_argument("--ckpt",        type=str, default=None,
                        help="Path to checkpoint .pth file.")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--benchmark",   action="store_true",
                        help="Run throughput benchmark.")
    parser.add_argument("--visualize",   action="store_true",
                        help="Generate LFGM heatmap visualizations.")
    parser.add_argument("--confusion",   action="store_true",
                        help="Plot confusion matrix.")
    parser.add_argument("--no_freqmerge", action="store_true",
                        help="Evaluate plain ViT-Small baseline.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (run in full float32).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = setup_cuda()
    print_gpu_info()
    use_amp = (not args.no_amp) and cfg.USE_AMP and device.type == "cuda"

    print(f"\n{'='*60}")
    print("  FreqMerge Evaluation")
    print(f"  Device  : {device}")
    print(f"  Ckpt    : {args.ckpt or '(no checkpoint — random weights)'}")
    print(f"{'='*60}\n")

    # ---- Data ----------------------------------------------------------
    _, val_loader = get_cifar100_loaders(batch_size=args.batch_size)

    # ---- Model ---------------------------------------------------------
    merge_layers = [] if args.no_freqmerge else cfg.MERGE_LAYERS
    model = build_freqmerge_vit(
        num_classes  = cfg.NUM_CLASSES,
        merge_layers = merge_layers,
        keep_rate    = cfg.KEEP_RATE,
        alpha        = cfg.ALPHA,
        hpf_radius   = cfg.HPF_RADIUS,
        pretrained   = False,              # load weights from ckpt
    ).to(device)

    # Load checkpoint
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
              f"(recorded val acc: {ckpt.get('val_acc', '?'):.2f}%)\n")
    else:
        print("WARNING: No checkpoint loaded — using current weights.\n")

    # ---- GFLOPs --------------------------------------------------------
    gflops = count_flops(model, input_size=(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                         device=device)
    print(f"GFLOPs (single image): {gflops:.2f} G")

    # ---- Validation accuracy -------------------------------------------
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("Loss")
    acc1_meter = AverageMeter("Top-1")
    acc5_meter = AverageMeter("Top-5")

    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        with CUDATimer("full validation", verbose=False) as t_eval:
            for images, labels in tqdm(val_loader, desc="Evaluating", ncols=100):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss   = criterion(logits, labels)

                acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
                B = images.size(0)
                loss_meter.update(loss.item(), B)
                acc1_meter.update(acc1, B)
                acc5_meter.update(acc5, B)

                if args.confusion:
                    preds = logits.argmax(dim=1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().tolist())

    print(f"\n{'─'*40}")
    print(f"  Eval time  : {t_eval.elapsed_ms/1000:.2f}s")
    print(f"  Val Loss   : {loss_meter.avg:.4f}")
    print(f"  Val Top-1  : {acc1_meter.avg:.2f}%")
    print(f"  Val Top-5  : {acc5_meter.avg:.2f}%")
    print(f"{'─'*40}\n")
    print_memory_stats("after eval", device)

    # ---- Throughput benchmark ------------------------------------------
    if args.benchmark:
        print("Running throughput benchmark …")
        tput = compute_throughput(
            model,
            input_size=(args.batch_size, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            device=str(device),
            n_warmup=30,
            n_measure=100,
        )
        print(f"  Throughput : {tput:.1f} images/sec "
              f"(batch={args.batch_size})\n")

    # ---- LFGM heatmap visualisation ------------------------------------
    if args.visualize:
        os.makedirs(cfg.VIZ_DIR, exist_ok=True)
        # Grab a batch from the val loader
        sample_images, _ = next(iter(val_loader))
        save_path = os.path.join(cfg.VIZ_DIR, "lfgm_heatmaps.png")
        visualize_freq_scores(
            images    = sample_images,
            model     = model,
            save_path = save_path,
            n_samples = min(8, args.batch_size),
            device    = str(device),
        )

    # ---- Confusion matrix ----------------------------------------------
    if args.confusion and all_preds:
        save_path = os.path.join(cfg.VIZ_DIR, "confusion_matrix.png")
        plot_confusion_matrix(
            all_preds  = all_preds,
            all_labels = all_labels,
            top_n      = 20,
            save_path  = save_path,
        )

    # ---- Summary -------------------------------------------------------
    print("Evaluation Results Summary")
    print(f"  Method       : {'ViT-Small (baseline)' if args.no_freqmerge else 'FreqMerge'}")
    print(f"  Top-1 Acc    : {acc1_meter.avg:.2f}%")
    print(f"  Top-5 Acc    : {acc5_meter.avg:.2f}%")
    print(f"  GFLOPs       : {gflops:.2f} G")
    if args.benchmark:
        print(f"  Throughput   : {tput:.1f} img/s")
    print()


if __name__ == "__main__":
    main()
