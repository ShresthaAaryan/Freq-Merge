"""
utils/metrics.py
----------------
Training metrics utilities.
"""

import time
import torch


class AverageMeter:
    """Tracks the running average of a scalar metric (loss, accuracy, etc.)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk:   tuple = (1, 5),
) -> list[float]:
    """
    Compute top-k accuracy for the given predictions and labels.

    Parameters
    ----------
    output : (B, C)  raw logits
    target : (B,)    ground-truth class indices
    topk   : tuple of k values to compute accuracy for

    Returns
    -------
    list of float  [top1_acc (%), top5_acc (%)]  (scaled to 0–100)
    """
    with torch.no_grad():
        maxk   = max(topk)
        B      = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()                          # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / B).item())
        return results


def compute_throughput(
    model:      torch.nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device:     str   = "cuda",
    n_warmup:   int   = 50,
    n_measure:  int   = 200,
) -> float:
    """
    Measure inference throughput in images/second.

    Runs ``n_warmup`` forward passes for GPU warm-up, then times
    ``n_measure`` passes and computes the average throughput.

    Parameters
    ----------
    model      : The model to benchmark.
    input_size : Batch input shape, e.g. (32, 3, 224, 224).
    device     : "cuda" or "cpu".
    n_warmup   : Number of warm-up iterations (not timed).
    n_measure  : Number of timed iterations.

    Returns
    -------
    float  Images per second.
    """
    model.eval().to(device)
    dummy = torch.randn(*input_size).to(device)
    batch = input_size[0]

    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed run
        t_start = time.perf_counter()
        for _ in range(n_measure):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    elapsed     = t_end - t_start
    total_imgs  = n_measure * batch
    throughput  = total_imgs / elapsed
    return throughput
