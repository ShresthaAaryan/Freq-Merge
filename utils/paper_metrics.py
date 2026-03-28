"""
Paper-oriented metrics for CV/ML experiments (accuracy, sklearn aggregates,
parameter count, FLOPs/MACs via thop with hook fallback, latency, VRAM).

thop may fail on some dynamic graphs; in that case only hook-based GFLOPs are reported.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# FLOPs — hook-based (same spirit as prior evaluate.py)
# ---------------------------------------------------------------------------


def estimate_gflops_hook(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: torch.device | str = "cpu",
) -> float:
    """
    Approximate GFLOPs for one forward pass via Linear + Conv2d hooks.
    Attention as implemented in timm is mostly Linear and is included indirectly.
    Treat as an estimate; use thop when available for paper comparisons.
    """
    flops = [0.0]

    def linear_hook(module, inp, out):
        in_feat = module.in_features
        out_feat = module.out_features
        batch_el = inp[0].numel() // in_feat
        flops[0] += 2.0 * in_feat * out_feat * batch_el

    def conv_hook(module, inp, out):
        Cout, Cin_g, kH, kW = module.weight.shape
        H_out, W_out = out.shape[-2], out.shape[-1]
        groups = module.groups
        Cin = Cin_g * groups
        batch = out.shape[0]
        flops[0] += 2.0 * (Cin / groups) * Cout * kH * kW * H_out * W_out * batch

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    dummy = torch.zeros(*input_size, device=device)
    model.eval().to(device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flops[0] / 1e9


def estimate_thop_macs_gflops(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: torch.device | str = "cpu",
) -> tuple[float | None, float | None]:
    """
    MACs and GFLOPs from thop.profile (MACs ≈ 0.5 × FLOPs for mul-add pairs).

    Returns
    -------
    macs_g  : GFLOPS-scale MAC count (thop first number / 1e9), or None if thop fails
    flops_g : 2 * macs_g when MACs are defined (common FLOP convention)
    """
    try:
        from thop import profile
    except ImportError:
        return None, None

    dummy = torch.zeros(*input_size, device=device)
    m = model.eval().to(device)
    try:
        with torch.no_grad():
            total_ops, _params = profile(m, inputs=(dummy,), verbose=False)
        macs_g = float(total_ops) / 1e9
        flops_g = 2.0 * macs_g
        return macs_g, flops_g
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Latency & memory
# ---------------------------------------------------------------------------


def measure_latency_ms(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    image_size: int = 224,
    n_warmup: int = 30,
    n_measure: int = 100,
) -> tuple[float, float]:
    """Mean and std of forward latency in milliseconds (whole batch)."""
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(n_measure):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def measure_peak_memory_forward_mb(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    image_size: int = 224,
) -> float | None:
    """Peak CUDA memory (MB) for one eval forward. CPU returns None."""
    if device.type != "cuda":
        return None
    model.eval().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        model(x)
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Classification (sklearn)
# ---------------------------------------------------------------------------


def classification_metrics_block(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Balanced accuracy, Cohen's kappa, MCC, macro/micro/weighted P/R/F1,
    and sklearn classification_report (per-class).
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    prec_u, rec_u, f1_u, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", zero_division=0
    )

    labels = list(range(len(class_names))) if class_names else None
    report = classification_report(
        yt,
        yp,
        target_names=class_names,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    try:
        mcc = float(matthews_corrcoef(yt, yp))
        if np.isnan(mcc):
            mcc = None
    except ValueError:
        mcc = None

    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "matthews_corrcoef": mcc,
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "precision_micro": float(prec_u),
        "recall_micro": float(rec_u),
        "f1_micro": float(f1_u),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "per_class_report": report,
    }
    return out


def build_full_eval_report(
    *,
    top1: float,
    top5: float,
    val_loss: float,
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None,
    model: nn.Module,
    image_size: int,
    batch_size_benchmark: int,
    gflops_hook: float,
    macs_thop_g: float | None,
    flops_thop_g: float | None,
    throughput_img_s: float | None,
    latency_batch_ms_mean: float | None,
    latency_batch_ms_std: float | None,
    latency_per_image_ms_mean: float | None,
    latency_per_image_ms_std: float | None,
    peak_mem_mb: float | None,
    eval_time_s: float,
) -> dict[str, Any]:
    """Single nested dict suitable for json.dump."""
    total_p, train_p = count_parameters(model)
    cls = classification_metrics_block(y_true, y_pred, class_names)

    return {
        "classification": {
            "top1_accuracy_pct": top1,
            "top5_accuracy_pct": top5,
            "val_loss": val_loss,
            **{k: v for k, v in cls.items() if k != "per_class_report"},
        },
        "per_class_report": cls["per_class_report"],
        "efficiency": {
            "params_total": total_p,
            "params_trainable": train_p,
            "params_millions": round(total_p / 1e6, 3),
            "gflops_hook_estimate": gflops_hook,
            "macs_thop_giga": macs_thop_g,
            "gflops_thop_2x_macs": flops_thop_g,
            "throughput_images_per_sec": throughput_img_s,
            "latency_batch_ms_mean": latency_batch_ms_mean,
            "latency_batch_ms_std": latency_batch_ms_std,
            "latency_per_image_ms_mean": latency_per_image_ms_mean,
            "latency_per_image_ms_std": latency_per_image_ms_std,
            "peak_cuda_memory_mb": peak_mem_mb,
            "note": "gflops_thop uses 2×MACs when thop succeeds; hook FLOPs are a separate estimate.",
        },
        "protocol": {
            "image_size": image_size,
            "benchmark_batch_size": batch_size_benchmark,
            "eval_time_s": eval_time_s,
        },
    }
