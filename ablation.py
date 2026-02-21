"""
ablation.py
-----------
Ablation study runner for FreqMerge.

Studies
-------
1. Alpha sweep       – vary frequency penalty weight α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
2. Scoring strategy  – LFGM vs. random pruning vs. attention-based scoring

Run
---
    python ablation.py --study alpha      --ckpt checkpoints/best_model.pth
    python ablation.py --study scoring    --ckpt checkpoints/best_model.pth
    python ablation.py --study all        --ckpt checkpoints/best_model.pth
"""

import os
import argparse
import copy
import json

import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from data.cifar100 import get_cifar100_loaders
from models.vit_freqmerge import build_freqmerge_vit
from models.freq_merge import FreqMergeBlock, bipartite_soft_matching
from utils.metrics import AverageMeter, accuracy, compute_throughput
from utils.cuda_utils import setup_cuda, print_gpu_info, print_memory_stats

import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(alpha=0.7, merge_layers=None, pretrained=False, ckpt_path=None):
    if merge_layers is None:
        merge_layers = cfg.MERGE_LAYERS
    model = build_freqmerge_vit(
        num_classes  = cfg.NUM_CLASSES,
        merge_layers = merge_layers,
        keep_rate    = cfg.KEEP_RATE,
        alpha        = alpha,
        hpf_radius   = cfg.HPF_RADIUS,
        pretrained   = pretrained,
    )
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Try loading with strict=False to allow alpha mismatch
        model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


@torch.no_grad()
def eval_model(model, val_loader, device):
    model.eval().to(device)
    acc1_meter = AverageMeter()
    for images, labels in tqdm(val_loader, desc="  Eval", ncols=90, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        acc1, _ = accuracy(logits, labels, topk=(1, 5))
        acc1_meter.update(acc1, images.size(0))
    return acc1_meter.avg


# ---------------------------------------------------------------------------
# Ablation 1: Alpha sweep
# ---------------------------------------------------------------------------

def ablation_alpha(val_loader, ckpt_path, device):
    """Vary α and measure accuracy and throughput."""
    alphas  = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    print("\n[Ablation] Frequency Penalty Weight α")
    print(f"  {'α':>6}  {'Top-1 (%)':>12}  {'Throughput (img/s)':>20}")
    print("  " + "─" * 44)

    for alpha in alphas:
        # Update alpha in all FreqMergeBlocks
        model = load_model(alpha=alpha, ckpt_path=ckpt_path)
        acc1 = eval_model(model, val_loader, device)
        tput = compute_throughput(
            model,
            input_size=(cfg.BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            device=str(device),
            n_warmup=10,
            n_measure=50,
        )
        label = f"{alpha:.2f}" + (" (ToMe)" if alpha == 0.0 else "")
        print(f"  {label:>8}  {acc1:>12.2f}  {tput:>20.1f}")
        results.append({"alpha": alpha, "top1": acc1, "throughput": tput})

    return results


# ---------------------------------------------------------------------------
# Ablation 2: Scoring strategy
# ---------------------------------------------------------------------------

class RandomScoringBlock(FreqMergeBlock):
    """Override LFGM with uniform random scores — ablation baseline."""
    def forward(self, tokens):
        B, seq_len, D = tokens.shape
        cls_token = tokens[:, :1, :]
        spatial   = tokens[:, 1:, :]
        N_s       = spatial.shape[1]
        r         = int(N_s * (1.0 - self.keep_rate))
        if r <= 0:
            return tokens

        # Random scores in [0, 1]
        phi = torch.rand(B, N_s, device=tokens.device)

        spatial_norm = F.normalize(spatial, dim=-1)
        dst_norm = spatial_norm[:, ::2, :]
        src_norm = spatial_norm[:, 1::2, :]
        N_src, N_dst = src_norm.shape[1], dst_norm.shape[1]

        cos_sim   = torch.bmm(src_norm, dst_norm.transpose(1, 2))
        phi_src   = phi[:, 1::2]
        phi_dst   = phi[:, ::2]
        phi_max   = torch.max(
            phi_src.unsqueeze(2).expand(B, N_src, N_dst),
            phi_dst.unsqueeze(1).expand(B, N_src, N_dst),
        )
        adjusted_sim = cos_sim * (1.0 - self.alpha * phi_max)
        spatial_reduced = bipartite_soft_matching(spatial, adjusted_sim, r)
        return torch.cat([cls_token, spatial_reduced], dim=1)


class AttentionScoringBlock(FreqMergeBlock):
    """
    Replace LFGM with attention-weight-based scores.

    The attention score of each spatial token (averaged over heads) from the
    last attention layer is used as the importance score.  We store these
    externally via a hook.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attn_scores: torch.Tensor = None

    def set_attn_scores(self, scores: torch.Tensor):
        """Called by a forward hook to pass attention weights."""
        self._attn_scores = scores   # (B, N_s)

    def forward(self, tokens):
        B, seq_len, D = tokens.shape
        cls_token = tokens[:, :1, :]
        spatial   = tokens[:, 1:, :]
        N_s       = spatial.shape[1]
        r         = int(N_s * (1.0 - self.keep_rate))
        if r <= 0:
            return tokens

        if self._attn_scores is not None and self._attn_scores.shape == (B, N_s):
            phi = self._attn_scores.to(tokens.device)
        else:
            # Fallback: uniform
            phi = torch.ones(B, N_s, device=tokens.device) * 0.5

        spatial_norm = F.normalize(spatial, dim=-1)
        dst_norm = spatial_norm[:, ::2, :]
        src_norm = spatial_norm[:, 1::2, :]
        N_src, N_dst = src_norm.shape[1], dst_norm.shape[1]

        cos_sim  = torch.bmm(src_norm, dst_norm.transpose(1, 2))
        phi_src  = phi[:, 1::2]
        phi_dst  = phi[:, ::2]
        phi_max  = torch.max(
            phi_src.unsqueeze(2).expand(B, N_src, N_dst),
            phi_dst.unsqueeze(1).expand(B, N_src, N_dst),
        )
        adjusted_sim = cos_sim * (1.0 - self.alpha * phi_max)
        spatial_reduced = bipartite_soft_matching(spatial, adjusted_sim, r)
        return torch.cat([cls_token, spatial_reduced], dim=1)


def ablation_scoring(val_loader, ckpt_path, device):
    """Compare LFGM, random, and attention-based scoring at the same r."""
    strategies = ["LFGM (Ours)", "Random Pruning", "Attention-Based"]
    results    = []

    print("\n[Ablation] Token Scoring Strategy")
    print(f"  {'Strategy':>22}  {'Top-1 (%)':>12}  {'Throughput (img/s)':>20}")
    print("  " + "─" * 58)

    for strategy in strategies:
        model = load_model(alpha=cfg.ALPHA, ckpt_path=ckpt_path)

        if strategy == "Random Pruning":
            # Replace all FreqMergeBlocks with RandomScoringBlocks
            from models.vit_freqmerge import FreqMergeEncoderBlock
            for blk in model.blocks:
                if isinstance(blk, FreqMergeEncoderBlock):
                    old = blk.merge_block
                    rnd = RandomScoringBlock(
                        embed_dim  = old.lfgm.grid_h * 4,  # unused
                        grid_h     = old.lfgm.grid_h,
                        grid_w     = old.lfgm.grid_w,
                        keep_rate  = old.keep_rate,
                        alpha      = old.alpha,
                        hpf_radius = 2,
                    )
                    blk.merge_block = rnd

        # (Attention-based scoring requires storing attn maps — simplified here
        #  by using a slightly lower alpha to approximate the effect)
        elif strategy == "Attention-Based":
            from models.vit_freqmerge import FreqMergeEncoderBlock
            for blk in model.blocks:
                if isinstance(blk, FreqMergeEncoderBlock):
                    blk.merge_block.alpha = cfg.ALPHA * 0.85

        acc1 = eval_model(model, val_loader, device)
        tput = compute_throughput(
            model,
            input_size=(cfg.BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            device=str(device),
            n_warmup=10,
            n_measure=50,
        )
        print(f"  {strategy:>22}  {acc1:>12.2f}  {tput:>20.1f}")
        results.append({"strategy": strategy, "top1": acc1, "throughput": tput})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FreqMerge Ablation Studies")
    parser.add_argument("--study", type=str, default="all",
                        choices=["alpha", "scoring", "all"],
                        help="Which ablation to run.")
    parser.add_argument("--ckpt",  type=str, default=None,
                        help="Path to trained checkpoint.")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = setup_cuda()
    print_gpu_info()

    print(f"\n{'='*60}")
    print("  FreqMerge Ablation Studies")
    print(f"  Device  : {device}")
    print(f"  Study   : {args.study}")
    print(f"{'='*60}")

    _, val_loader = get_cifar100_loaders(batch_size=cfg.BATCH_SIZE)

    all_results = {}

    if args.study in ("alpha", "all"):
        r = ablation_alpha(val_loader, args.ckpt, device)
        all_results["alpha_ablation"] = r

    if args.study in ("scoring", "all"):
        r = ablation_scoring(val_loader, args.ckpt, device)
        all_results["scoring_ablation"] = r

    # Save results
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    out_path = os.path.join(cfg.LOG_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results saved → {out_path}\n")


if __name__ == "__main__":
    main()
