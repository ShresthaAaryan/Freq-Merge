"""
models/vit_freqmerge.py
-----------------------
ViT-Small backbone (from timm) with FreqMerge blocks injected after
the specified encoder layers.

Architecture overview
---------------------
Standard ViT-Small has 12 encoder blocks.  FreqMerge wraps each target
block so that after the block's forward pass the token sequence is
passed through a FreqMergeBlock, reducing it before the next block sees it.

The [CLS] token is always excluded from scoring / merging and is simply
prepended to the reduced sequence at each step.

Usage
-----
    model = build_freqmerge_vit(
        num_classes  = 100,
        merge_layers = [4, 6, 8, 10],
        keep_rate    = 0.7,
        alpha        = 0.7,
        hpf_radius   = 2,
        pretrained   = True,
    )
"""

import torch
import torch.nn as nn
import timm

from .freq_merge import FreqMergeBlock


# ---------------------------------------------------------------------------
# Wrapper that injects a FreqMergeBlock after a standard timm ViT block
# ---------------------------------------------------------------------------

class FreqMergeEncoderBlock(nn.Module):
    """
    Wraps a single timm ViT encoder block with a FreqMergeBlock appended.

    The wrapped block behaves identically to the original block (including
    residual connections, layer norms, MHSA, FFN) but the output token
    sequence is reduced before being returned.
    """

    def __init__(self, vit_block: nn.Module, merge_block: FreqMergeBlock):
        super().__init__()
        self.vit_block   = vit_block
        self.merge_block = merge_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Strict paper-compliant insertion: apply attention -> merge -> MLP
        # This requires the timm ViT block to expose `norm1`, `attn`, `norm2`,
        # and `mlp`. If those internals are not present, raise an error so the
        # user can install a compatible timm version (paper-strict behaviour).
        block = self.vit_block
        required = ("norm1", "attn", "norm2", "mlp")
        if not all(hasattr(block, a) for a in required):
            missing = [a for a in required if not hasattr(block, a)]
            raise RuntimeError(
                f"Timm ViT block is missing required internals for paper-strict "
                f"FreqMerge insertion: {missing}. Install a compatible timm "
                f"version or modify the code to allow fallback behavior."
            )

        # Pre-norm attention
        x_attn = block.attn(block.norm1(x))
        if hasattr(block, "drop_path") and block.drop_path is not None:
            x_attn = block.drop_path(x_attn)
        x = x + x_attn

        # Frequency-guided token reduction (exclude CLS internally)
        x = self.merge_block(x)

        # MLP (feed-forward) on reduced token set
        x_mlp = block.mlp(block.norm2(x))
        if hasattr(block, "drop_path") and block.drop_path is not None:
            x_mlp = block.drop_path(x_mlp)
        x = x + x_mlp
        return x


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_freqmerge_vit(
    num_classes:  int   = 100,
    merge_layers: list  = None,
    keep_rate:    float = 0.7,
    alpha:        float = 0.7,
    hpf_radius:   int   = 2,
    pretrained:   bool  = True,
    backbone:     str   = "vit_small_patch16_224",
) -> nn.Module:
    """
    Build a ViT-Small model with FreqMerge applied at the specified layers.

    Parameters
    ----------
    num_classes  : Output classes (100 for CIFAR-100).
    merge_layers : List of 0-indexed encoder block indices where DTM is
                   applied.  Defaults to [4, 6, 8, 10].
    keep_rate    : Fraction of spatial tokens kept per merge (0 < r ≤ 1).
    alpha        : Frequency penalty weight.
    hpf_radius   : HPF radius for LFGM.
    pretrained   : Load ImageNet-1k weights from timm.
    backbone     : timm model identifier.

    Returns
    -------
    nn.Module  Ready-to-train / ready-to-eval model.
    """
    if merge_layers is None:
        merge_layers = [4, 6, 8, 10]

    # ---- Load pretrained ViT-Small from timm --------------------------
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # ---- Determine spatial grid dimensions ----------------------------
    # For patch16 / 224 px input: 14 × 14 = 196 spatial tokens
    img_size   = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    grid_h = img_size[0] // patch_size[0]
    grid_w = img_size[1] // patch_size[1]

    embed_dim = model.embed_dim   # 384 for ViT-Small

    # ---- Inject FreqMergeBlock after each target encoder block --------
    # timm ViT-Small stores its blocks in model.blocks (nn.Sequential / list)
    # timm calls self.blocks(x) directly, so it must be nn.Sequential
    # (nn.ModuleList has no forward() and raises NotImplementedError)
    new_blocks = []
    for i, block in enumerate(model.blocks):
        if i in merge_layers:
            merge_blk = FreqMergeBlock(
                embed_dim  = embed_dim,
                grid_h     = grid_h,
                grid_w     = grid_w,
                keep_rate  = keep_rate,
                alpha      = alpha,
                hpf_radius = hpf_radius,
            )
            new_blocks.append(FreqMergeEncoderBlock(block, merge_blk))
        else:
            new_blocks.append(block)

    model.blocks = nn.Sequential(*new_blocks)

    # ---- Adjust the classification head for CIFAR-100 ----------------
    # timm already sets num_classes in create_model, but double-check:
    if model.head.out_features != num_classes:
        model.head = nn.Linear(embed_dim, num_classes)

    return model


# ---------------------------------------------------------------------------
# Convenience: count effective parameters and token budget
# ---------------------------------------------------------------------------

def model_summary(model: nn.Module, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Print total / trainable parameters and a simulated token count trace.
    """
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Trace token count with a dummy forward pass
    hooks, token_counts = [], []

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            token_counts.append((type(module).__name__, out.shape[1]))

    for block in model.blocks:
        hooks.append(block.register_forward_hook(_hook))

    dummy = torch.zeros(*input_size).to(device)
    with torch.no_grad():
        model.to(device)(dummy)

    for h in hooks:
        h.remove()

    print("\nToken count after each block:")
    for name, n in token_counts:
        print(f"  {name:<30s}  tokens = {n}")
