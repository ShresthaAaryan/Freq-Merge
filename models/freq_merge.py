"""
models/freq_merge.py
--------------------
Frequency-Guided Dynamic Token Merging (DTM) block.

The modified Bipartite Soft Matching algorithm is implemented here.
Standard cosine similarity between source and destination tokens is
penalised by the frequency score, as defined in the paper:

    S_freq(t_i, t_j) = cos(t_i, t_j) · (1 − α · max(φ̃_i, φ̃_j))

Pairs with a high-frequency token (large φ̃) have their effective
similarity pushed toward zero, so the greedy matcher leaves them alone
and merges only low-frequency, redundant tokens.

Usage
-----
    dtm = FreqMergeBlock(
        embed_dim=384,
        grid_h=14,
        grid_w=14,
        keep_rate=0.7,
        alpha=0.7,
        hpf_radius=2,
    )
    # tokens: (B, 1 + N_s, D)  — includes [CLS] at position 0
    tokens_reduced = dtm(tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfgm import LFGM


# ---------------------------------------------------------------------------
# Bipartite Soft Matching (core algorithm)
# ---------------------------------------------------------------------------

def bipartite_soft_matching(
    tokens:  torch.Tensor,   # (B, N, D)  spatial tokens only — no [CLS]
    scores:  torch.Tensor,   # (B, N_src, N_dst)  adjusted similarity
    r:       int,            # number of token pairs to merge
) -> torch.Tensor:
    """
    Merge the r source–destination pairs with the highest adjusted similarity.

    Partition: even-indexed tokens → dst, odd-indexed → src  (ToMe convention).
    Merge: src token is averaged into its best-matching dst slot.
    Unmerged src tokens are appended to the dst sequence.

    Returns
    -------
    Tensor of shape (B, N − r, D).
    """
    B, N, D = tokens.shape

    dst = tokens[:, ::2,  :]    # (B, N_dst, D)
    src = tokens[:, 1::2, :]    # (B, N_src, D)
    N_src = src.shape[1]
    N_dst = dst.shape[1]

    r = min(r, N_src)
    if r <= 0:
        return tokens

    # For each src, find the best dst
    node_max, node_idx = scores.max(dim=-1)     # (B, N_src)

    # Pick top-r src tokens by their best-match score
    _, edge_idx = node_max.topk(r, dim=-1)      # (B, r)

    # Which dst slot does each selected src map to?
    dst_idx = node_idx.gather(1, edge_idx)       # (B, r)

    # --- Merge: accumulate src into matched dst via scatter_add ----------
    src_to_merge = src.gather(
        1, edge_idx.unsqueeze(-1).expand(B, r, D)
    )                                            # (B, r, D)

    dst_out = dst.clone()
    dst_out.scatter_add_(
        1,
        dst_idx.unsqueeze(-1).expand(B, r, D),
        src_to_merge,
    )
    # Count how many src tokens landed on each dst slot
    ones = torch.ones(B, r, 1, device=tokens.device)
    count = torch.ones(B, N_dst, 1, device=tokens.device)
    count.scatter_add_(1, dst_idx.unsqueeze(-1), ones)
    dst_out = dst_out / count                   # weighted average

    # --- Collect unmerged src tokens ------------------------------------
    # Build a keep mask: True = NOT in the top-r merged set
    keep_mask = torch.ones(B, N_src, dtype=torch.bool, device=tokens.device)
    keep_mask.scatter_(1, edge_idx, False)

    # Gather unmerged src tokens — same count (N_src - r) for every sample
    unmerged = src[keep_mask].reshape(B, N_src - r, D)

    return torch.cat([dst_out, unmerged], dim=1)   # (B, N - r, D)


# ---------------------------------------------------------------------------
# FreqMerge Block
# ---------------------------------------------------------------------------

class FreqMergeBlock(nn.Module):
    """
    Drop-in token reduction block.

    Insert this module after a ViT encoder block's MHSA layer (before FFN)
    or after the complete encoder block. The [CLS] token is always excluded
    from scoring and merging.

    Parameters
    ----------
    embed_dim  : int   Embedding dimension D of the ViT (384 for ViT-Small).
    grid_h     : int   Token grid height  (14 for patch16 / 224 px input).
    grid_w     : int   Token grid width   (14 for patch16 / 224 px input).
    keep_rate  : float Fraction of tokens to KEEP after merging (0 < keep ≤ 1).
    alpha      : float Frequency penalty weight  α ∈ [0, 1].
    hpf_radius : int   High-pass filter radius for LFGM.
    """

    def __init__(
        self,
        embed_dim:  int   = 384,
        grid_h:     int   = 14,
        grid_w:     int   = 14,
        keep_rate:  float = 0.7,
        alpha:      float = 0.7,
        hpf_radius: int   = 2,
    ):
        super().__init__()
        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.keep_rate = keep_rate
        self.alpha     = alpha

        self.N_spatial = grid_h * grid_w          # 196
        self.lfgm = LFGM(grid_h, grid_w, hpf_radius)

    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, 1 + N_s, D)
            Full token sequence including [CLS] at index 0.

        Returns
        -------
        torch.Tensor  (B, 1 + N_reduced, D)
            Sequence with [CLS] prepended and spatial tokens reduced.
        """
        B, seq_len, D = tokens.shape

        # Separate [CLS] from spatial tokens
        cls_token     = tokens[:, :1, :]            # (B, 1, D)
        spatial       = tokens[:, 1:, :]            # (B, N_s, D)
        N_s           = spatial.shape[1]

        # Number of merges r = number of tokens to remove
        r = int(N_s * (1.0 - self.keep_rate))
        if r <= 0:
            return tokens                           # nothing to merge

        # ---- 1. Compute frequency scores via LFGM --------------------
        phi = self.lfgm(spatial)                    # (B, N_s)  ∈ [0, 1]

        # ---- 2. Compute base cosine similarity -----------------------
        # Normalise token features to unit sphere
        spatial_norm = F.normalize(spatial, dim=-1) # (B, N_s, D)

        # Partition into src (odd) and dst (even) for bipartite matching
        dst_norm = spatial_norm[:, ::2,  :]         # (B, N_dst, D)
        src_norm = spatial_norm[:, 1::2, :]         # (B, N_src, D)

        N_src = src_norm.shape[1]
        N_dst = dst_norm.shape[1]

        # Cosine similarity matrix: (B, N_src, N_dst)
        cos_sim = torch.bmm(src_norm, dst_norm.transpose(1, 2))

        # ---- 3. Apply frequency penalty  S_freq = cos · (1 − α·max(φ)) --
        phi_src = phi[:, 1::2]                      # (B, N_src)
        phi_dst = phi[:, ::2]                       # (B, N_dst)

        # max(φ̃_i, φ̃_j) for each (src, dst) pair → (B, N_src, N_dst)
        phi_max = torch.max(
            phi_src.unsqueeze(2).expand(B, N_src, N_dst),
            phi_dst.unsqueeze(1).expand(B, N_src, N_dst),
        )
        freq_penalty = 1.0 - self.alpha * phi_max   # (B, N_src, N_dst)
        adjusted_sim = cos_sim * freq_penalty        # (B, N_src, N_dst)

        # ---- 4. Bipartite soft matching + merge ----------------------
        spatial_reduced = bipartite_soft_matching(spatial, adjusted_sim, r)

        # ---- 5. Re-attach [CLS] token --------------------------------
        out = torch.cat([cls_token, spatial_reduced], dim=1)
        return out
