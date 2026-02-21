"""
utils/cuda_utils.py
-------------------
CUDA / GPU utilities for FreqMerge.

Provides
--------
  setup_cuda()          – configure cuDNN flags, seed all RNGs, return device
  print_gpu_info()      – print full GPU inventory with VRAM
  get_memory_stats()    – return dict of current CUDA memory usage
  print_memory_stats()  – pretty-print current CUDA memory usage
  clear_cuda_cache()    – torch.cuda.empty_cache() + gc.collect()
  CUDATimer             – context-manager for precise CUDA kernel timing
"""

import gc
import os
import time
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ---------------------------------------------------------------------------
# Primary setup function
# ---------------------------------------------------------------------------

def setup_cuda(seed: int = None) -> torch.device:
    """
    Configure the full CUDA environment and return the active device.

    Actions
    -------
    1. Seeds Python / NumPy / PyTorch (CPU + all CUDA devices) for
       reproducibility.
    2. Sets cuDNN benchmark / deterministic flags from config.py.
    3. Enables TF32 on Ampere GPUs (RTX 30xx) for a free ~3× matmul
       speedup with negligible precision loss.
    4. Prints a one-line summary of the active device.

    Returns
    -------
    torch.device  "cuda" if a GPU is available, else "cpu".
    """
    # ---- Resolve defaults -----------------------------------------------
    if seed is None:
        seed = cfg.SEED

    # ---- Seeding -----------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # covers all GPUs

    # ---- cuDNN flags ---------------------------------------------------
    cudnn.benchmark     = cfg.CUDNN_BENCHMARK      # auto-tune kernels
    cudnn.deterministic = cfg.CUDNN_DETERMINISTIC  # reproducibility

    # ---- TF32 (Ampere+) ------------------------------------------------
    # Allows matmuls to use TensorFloat-32 precision (full float32 range,
    # reduced mantissa).  Enabled by default in PyTorch ≥ 1.12 but we
    # set explicitly for clarity.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ---- Device --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patch cfg.PIN_MEMORY now that we know whether a GPU exists
    cfg.PIN_MEMORY = (device.type == "cuda")

    if device.type == "cuda":
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"[CUDA] Active device : GPU {idx} — {name}  ({vram:.1f} GB VRAM)")
        print(f"[CUDA] cuDNN version  : {torch.backends.cudnn.version()}")
        print(f"[CUDA] cuDNN benchmark: {cudnn.benchmark}  |  "
              f"deterministic: {cudnn.deterministic}")
        print(f"[CUDA] TF32 enabled   : True  (matmul + cuDNN)")
    else:
        print("[CUDA] No GPU detected — running on CPU.")

    return device


# ---------------------------------------------------------------------------
# GPU inventory
# ---------------------------------------------------------------------------

def print_gpu_info() -> None:
    """Print a table of every available GPU with its VRAM and compute cap."""
    n = torch.cuda.device_count()
    if n == 0:
        print("[GPU Info] No CUDA GPUs found.")
        return

    print(f"\n[GPU Info] {n} GPU(s) available:")
    print(f"  {'ID':>3}  {'Name':<35}  {'VRAM (GB)':>10}  {'Compute':>8}")
    print("  " + "─" * 62)
    for i in range(n):
        props  = torch.cuda.get_device_properties(i)
        vram   = props.total_memory / 1024**3
        major, minor = props.major, props.minor
        print(f"  {i:>3}  {props.name:<35}  {vram:>10.1f}  "
              f"{major}.{minor:>4}")
    print()


# ---------------------------------------------------------------------------
# Memory statistics
# ---------------------------------------------------------------------------

def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """
    Return current CUDA memory usage as a dict (all values in MB).

    Keys
    ----
    allocated_mb    : Memory currently occupied by tensors.
    reserved_mb     : Memory reserved by the caching allocator.
    free_mb         : Estimated free VRAM (total − reserved).
    total_mb        : Total VRAM on the device.
    peak_allocated_mb : Maximum allocated since last reset_peak_stats call.
    """
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.device("cuda")

    idx        = device.index if device.index is not None else 0
    props      = torch.cuda.get_device_properties(idx)
    total_b    = props.total_memory

    alloc_b    = torch.cuda.memory_allocated(idx)
    reserved_b = torch.cuda.memory_reserved(idx)
    peak_b     = torch.cuda.max_memory_allocated(idx)

    to_mb = lambda b: b / 1024**2

    return {
        "allocated_mb":     to_mb(alloc_b),
        "reserved_mb":      to_mb(reserved_b),
        "free_mb":          to_mb(total_b - reserved_b),
        "total_mb":         to_mb(total_b),
        "peak_allocated_mb": to_mb(peak_b),
    }


def print_memory_stats(label: str = "", device: Optional[torch.device] = None) -> None:
    """Pretty-print current GPU memory statistics."""
    stats = get_memory_stats(device)
    if not stats:
        print("[Mem] CUDA not available.")
        return
    tag = f" [{label}]" if label else ""
    print(f"[CUDA Mem{tag}]  "
          f"Allocated: {stats['allocated_mb']:.0f} MB  |  "
          f"Reserved: {stats['reserved_mb']:.0f} MB  |  "
          f"Free: {stats['free_mb']:.0f} MB  |  "
          f"Peak: {stats['peak_allocated_mb']:.0f} MB  /  "
          f"{stats['total_mb']:.0f} MB total")


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset the peak-allocated counter (call at epoch start for per-epoch peak)."""
    if torch.cuda.is_available():
        idx = (device.index if device and device.index is not None else 0)
        torch.cuda.reset_peak_memory_stats(idx)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_cuda_cache() -> None:
    """
    Release all unused cached memory back to the OS and run the Python GC.

    Safe to call between epochs on small-VRAM GPUs (≤ 8 GB).
    Has no effect on CPU.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Precise kernel timing
# ---------------------------------------------------------------------------

class CUDATimer:
    """
    Context manager for measuring the wall-clock time of CUDA operations.

    Uses CUDA events for accurate GPU-side timing (accounts for
    asynchronous kernel scheduling).  Falls back to time.perf_counter
    on CPU.

    Example
    -------
    >>> with CUDATimer("forward pass") as t:
    ...     output = model(input)
    >>> print(t.elapsed_ms, "ms")
    """

    def __init__(self, label: str = "", verbose: bool = True):
        self.label    = label
        self.verbose  = verbose
        self.elapsed_ms = 0.0
        self._use_cuda = torch.cuda.is_available()

    def __enter__(self):
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.verbose:
            tag = f" [{self.label}]" if self.label else ""
            print(f"[Timer{tag}]  {self.elapsed_ms:.2f} ms")


# ---------------------------------------------------------------------------
# Multi-GPU wrapper
# ---------------------------------------------------------------------------

def maybe_wrap_data_parallel(
    model:     nn.Module,
    device:    torch.device,
    use_multi: bool = None,
) -> nn.Module:
    """
    Wrap the model in nn.DataParallel if multiple GPUs are available
    and ``use_multi`` is True.

    DataParallel replicates the model on each GPU and splits the batch
    across devices at each forward call — a simple but effective strategy
    for ViT-scale models.

    Parameters
    ----------
    model     : The model to (optionally) wrap.
    device    : Primary device (must be "cuda" for multi-GPU to engage).
    use_multi : Master switch from config.py.

    Returns
    -------
    nn.Module   Original or DataParallel-wrapped model.
    """
    if use_multi is None:
        use_multi = cfg.USE_MULTI_GPU

    if (
        use_multi
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        n = torch.cuda.device_count()
        print(f"[CUDA] Enabling DataParallel across {n} GPUs.")
        model = nn.DataParallel(model)
    return model
