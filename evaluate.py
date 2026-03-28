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
    - sklearn: balanced accuracy, kappa, macro/micro/weighted P/R/F1, per-class report
    - Params (total / trainable / millions)
    - GFLOPs: hook estimate + thop MACs (2× as FLOPs) when available
    - Latency (ms) mean±std, per-image; throughput (img/s); peak CUDA memory (MB)
    - JSON summary → logs/eval_metrics.json (disable with --no_save_json)
    - Heatmap / confusion matrix with optional flags
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

import config as cfg
from data.skin_datasets import get_skin_loaders
from models.vit_freqmerge import build_freqmerge_vit
from utils.cuda_utils import (
    CUDATimer,
    print_gpu_info,
    print_memory_stats,
    setup_cuda,
)
from utils.metrics import AverageMeter, accuracy, compute_throughput
from utils.paper_metrics import (
    build_full_eval_report,
    estimate_gflops_hook,
    estimate_thop_macs_gflops,
    measure_latency_ms,
    measure_peak_memory_forward_mb,
)
from utils.visualize import plot_confusion_matrix, visualize_freq_scores


def _to_jsonable(obj):
    """Convert numpy types and non-finite floats for JSON."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    default_metrics = os.path.join(cfg.LOG_DIR, "eval_metrics.json")
    parser = argparse.ArgumentParser(description="FreqMerge Evaluation")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint .pth file.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--benchmark", action="store_true",
                        help="Alias: ensures throughput is measured (on by default).")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate LFGM heatmap visualizations.")
    parser.add_argument("--confusion", action="store_true",
                        help="Plot confusion matrix.")
    parser.add_argument("--no_freqmerge", action="store_true",
                        help="Evaluate plain ViT-Small baseline.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (run in full float32).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=cfg.DEFAULT_SKIN_DATASET,
        choices=["ham10000", "ham1000", "isic2019"],
        help="Must match the dataset the checkpoint was trained on.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Folder containing train/ and val/ (or test/). Overrides config paths.",
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default=default_metrics,
        help="Write full metric dict as JSON to this path.",
    )
    parser.add_argument(
        "--no_save_json",
        action="store_true",
        help="Do not write eval_metrics JSON.",
    )
    parser.add_argument(
        "--no_efficiency_suite",
        action="store_true",
        help="Skip thop, latency, throughput, and peak-memory benchmarks.",
    )
    parser.add_argument(
        "--no_sklearn_metrics",
        action="store_true",
        help="Do not collect predictions; skips balanced acc / F1 / per-class report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = setup_cuda()
    print_gpu_info()
    use_amp = (not args.no_amp) and cfg.USE_AMP and device.type == "cuda"

    print(f"\n{'='*60}")
    print("  FreqMerge Evaluation")
    print(f"  Device  : {device}")
    print(f"  Ckpt    : {args.ckpt or '(no checkpoint — random weights)'}")
    print(f"{'='*60}\n")

    _, val_loader, num_classes, class_names = get_skin_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    merge_layers = [] if args.no_freqmerge else cfg.MERGE_LAYERS
    model = build_freqmerge_vit(
        num_classes=num_classes,
        merge_layers=merge_layers,
        keep_rate=cfg.KEEP_RATE,
        alpha=cfg.ALPHA,
        hpf_radius=cfg.HPF_RADIUS,
        pretrained=False,
    ).to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        acc_msg = ckpt.get("val_acc")
        acc_str = f"{acc_msg:.2f}%" if isinstance(acc_msg, (int, float)) else str(acc_msg)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
              f"(recorded val acc: {acc_str})\n")
    else:
        print("WARNING: No checkpoint loaded — using current weights.\n")

    gflops = estimate_gflops_hook(
        model,
        input_size=(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        device=device,
    )
    print(f"GFLOPs hook estimate (1×{cfg.IMAGE_SIZE}, 1 image): {gflops:.2f} G")

    macs_g, flops_thop_g = None, None
    if not args.no_efficiency_suite:
        macs_g, flops_thop_g = estimate_thop_macs_gflops(
            model,
            input_size=(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            device=device,
        )
        if macs_g is not None:
            print(f"thop MACs (1 image): {macs_g:.2f} G  |  2×MACs FLOPs: {flops_thop_g:.2f} G")
        else:
            print("thop: skipped (unavailable or failed on this model graph).")

    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("Loss")
    acc1_meter = AverageMeter("Top-1")
    acc5_meter = AverageMeter("Top-5")

    all_preds: list[int] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        with CUDATimer("full validation", verbose=False) as t_eval:
            for images, labels in tqdm(val_loader, desc="Evaluating", ncols=100):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
                B = images.size(0)
                loss_meter.update(loss.item(), B)
                acc1_meter.update(acc1, B)
                acc5_meter.update(acc5, B)

                if not args.no_sklearn_metrics:
                    preds = logits.argmax(dim=1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().tolist())

    eval_time_s = t_eval.elapsed_ms / 1000.0

    print(f"\n{'─'*40}")
    print(f"  Eval time  : {eval_time_s:.2f}s")
    print(f"  Val Loss   : {loss_meter.avg:.4f}")
    print(f"  Val Top-1  : {acc1_meter.avg:.2f}%")
    print(f"  Val Top-5  : {acc5_meter.avg:.2f}%")
    print(f"{'─'*40}\n")
    print_memory_stats("after eval", device)

    tput: float | None = None
    lat_m: float | None = None
    lat_s: float | None = None
    lat_1_m: float | None = None
    lat_1_s: float | None = None
    peak_mb: float | None = None

    run_efficiency = (not args.no_efficiency_suite) or args.benchmark
    if run_efficiency:
        print("Efficiency benchmarks (eval batch size) …")
        tput = compute_throughput(
            model,
            input_size=(args.batch_size, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            device=str(device),
            n_warmup=30,
            n_measure=100,
        )
        print(f"  Throughput : {tput:.1f} images/sec  (batch={args.batch_size})")

        lat_m, lat_s = measure_latency_ms(
            model,
            device,
            batch_size=args.batch_size,
            image_size=cfg.IMAGE_SIZE,
            n_warmup=30,
            n_measure=100,
        )
        lat_1_m = lat_m / args.batch_size
        lat_1_s = lat_s / args.batch_size
        print(
            f"  Latency    : {lat_m:.2f} ± {lat_s:.2f} ms/batch  "
            f"({lat_1_m:.2f} ± {lat_1_s:.2f} ms/image)"
        )

        peak_mb = measure_peak_memory_forward_mb(
            model,
            device,
            batch_size=args.batch_size,
            image_size=cfg.IMAGE_SIZE,
        )
        if peak_mb is not None:
            print(f"  Peak VRAM (forward, 1 batch): {peak_mb:.1f} MB")
        print()

    if args.visualize:
        os.makedirs(cfg.VIZ_DIR, exist_ok=True)
        sample_images, _ = next(iter(val_loader))
        save_path = os.path.join(cfg.VIZ_DIR, "lfgm_heatmaps.png")
        visualize_freq_scores(
            images=sample_images,
            model=model,
            save_path=save_path,
            n_samples=min(8, args.batch_size),
            device=str(device),
        )

    if args.confusion and all_preds:
        os.makedirs(cfg.VIZ_DIR, exist_ok=True)
        save_path = os.path.join(cfg.VIZ_DIR, "confusion_matrix.png")
        plot_confusion_matrix(
            all_preds=all_preds,
            all_labels=all_labels,
            top_n=20,
            save_path=save_path,
        )

    print("Evaluation summary (paper-style)")
    print(f"  Method       : {'ViT-Small (baseline)' if args.no_freqmerge else 'FreqMerge'}")
    print(f"  Top-1 / Top-5: {acc1_meter.avg:.2f}% / {acc5_meter.avg:.2f}%")

    if not args.no_sklearn_metrics and all_preds:
        from utils.paper_metrics import classification_metrics_block

        blk = classification_metrics_block(all_labels, all_preds, class_names)
        print(f"  Balanced acc : {blk['balanced_accuracy']:.4f}")
        print(f"  Macro F1     : {blk['f1_macro']:.4f}  |  Weighted F1: {blk['f1_weighted']:.4f}")
        print(f"  Cohen κ      : {blk['cohen_kappa']:.4f}")

    if not args.no_save_json:
        _metric_dir = os.path.dirname(os.path.abspath(args.save_metrics))
        if _metric_dir:
            os.makedirs(_metric_dir, exist_ok=True)
        if args.no_sklearn_metrics or not all_preds:
            report = {
                "classification": {
                    "top1_accuracy_pct": acc1_meter.avg,
                    "top5_accuracy_pct": acc5_meter.avg,
                    "val_loss": loss_meter.avg,
                    "note": "sklearn metrics omitted (--no_sklearn_metrics or empty val).",
                },
                "per_class_report": None,
                "efficiency": {
                    "gflops_hook_estimate": gflops,
                    "macs_thop_giga": macs_g,
                    "gflops_thop_2x_macs": flops_thop_g,
                    "throughput_images_per_sec": tput,
                    "latency_batch_ms_mean": lat_m,
                    "latency_batch_ms_std": lat_s,
                    "latency_per_image_ms_mean": lat_1_m,
                    "latency_per_image_ms_std": lat_1_s,
                    "peak_cuda_memory_mb": peak_mb,
                },
                "protocol": {
                    "image_size": cfg.IMAGE_SIZE,
                    "benchmark_batch_size": args.batch_size,
                    "eval_time_s": eval_time_s,
                },
            }
            from utils.paper_metrics import count_parameters

            tot, trn = count_parameters(model)
            report["efficiency"]["params_total"] = tot
            report["efficiency"]["params_trainable"] = trn
            report["efficiency"]["params_millions"] = round(tot / 1e6, 3)
        else:
            report = build_full_eval_report(
                top1=acc1_meter.avg,
                top5=acc5_meter.avg,
                val_loss=loss_meter.avg,
                y_true=all_labels,
                y_pred=all_preds,
                class_names=class_names,
                model=model,
                image_size=cfg.IMAGE_SIZE,
                batch_size_benchmark=args.batch_size,
                gflops_hook=gflops,
                macs_thop_g=macs_g,
                flops_thop_g=flops_thop_g,
                throughput_img_s=tput,
                latency_batch_ms_mean=lat_m,
                latency_batch_ms_std=lat_s,
                latency_per_image_ms_mean=lat_1_m,
                latency_per_image_ms_std=lat_1_s,
                peak_mem_mb=peak_mb,
                eval_time_s=eval_time_s,
            )

        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, indent=2)
        print(f"\nFull metrics JSON → {args.save_metrics}\n")
    elif args.no_save_json:
        print("\n(JSON metrics file skipped: --no_save_json)\n")


if __name__ == "__main__":
    main()
