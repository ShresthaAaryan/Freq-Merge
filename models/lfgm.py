"""
models/lfgm.py
--------------
Lightweight Frequency Guidance Module (LFGM)

        Core idea
        ---------
        Given a 1-D sequence of patch tokens (shape: B × N × D), the module:
            1. Averages across the embedding dimension D  →  (B, N)
            2. Reshapes the 1-D sequence into a 2-D spatial grid (grid_h × grid_w)
            3. Applies a 2-D FFT across the spatial axes
            4. Applies a circular high-pass filter (HPF) mask of radius `hpf_radius`
                 to zero low-frequency components
            5. Computes the high-frequency residual via inverse FFT and energy
            6. Normalises per-sample to [0, 1]

        This implementation follows the paper's LFGM description (2-D FFT + HPF)
        so that spatial edges and textures are preserved when scoring tokens.

        Zero learned parameters.
"""

import torch
import torch.nn as nn


class LFGM(nn.Module):
    """
    Lightweight Frequency Guidance Module (1-D FFT, resolution-agnostic).

    Parameters
    ----------
    grid_h, grid_w : kept for API compatibility but no longer used internally.
    hpf_radius     : number of low-frequency bins to zero out from each end
                     of the 1-D FFT output.  Larger values → more aggressive
                     high-pass filtering.
    eps            : small constant for numerically stable normalisation.
    """

    def __init__(
        self,
        grid_h:     int   = 14,   # kept for API compatibility
        grid_w:     int   = 14,   # kept for API compatibility
        hpf_radius: int   = 2,
        eps:        float = 1e-6,
    ):
        super().__init__()
        self.grid_h     = grid_h
        self.grid_w     = grid_w
        self.hpf_radius = hpf_radius
        self.eps        = eps

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, N, D)  — any N, [CLS] must be stripped before calling.

        Returns
        -------
        phi : (B, N)  normalised frequency energy scores in [0, 1].
        """
        B, N, D = tokens.shape

        # Step 1: average across embedding dim → (B, N)
        signal_1d = tokens.mean(dim=2)    # (B, N)

        # Step 2: reshape to 2D spatial grid (B, H, W)
        H0, W0 = self.grid_h, self.grid_w

        # Paper-compliant 2-D FFT on the current token layout.
        # Reshape the current N tokens to the nearest 2-D grid (H' x W'),
        # padding if necessary, apply 2-D FFT + circular HPF, then map
        # the resulting high-frequency energy back to the original N tokens.
        import math

        device = tokens.device

        # Choose grid dimensions close to square for current N
        Hp = int(round(math.sqrt(N)))
        Hp = max(1, Hp)
        Wp = int(math.ceil(N / Hp))
        pad = Hp * Wp - N

        # Build a (B, Hp, Wp) grid filled from signal_1d, padding zeros at end
        sig = signal_1d
        if pad > 0:
            sig = torch.nn.functional.pad(sig, (0, pad), mode="constant", value=0.0)
        signal_2d = sig.view(B, Hp, Wp)

        # 2-D FFT
        F = torch.fft.fft2(signal_2d, norm="ortho")   # (B, Hp, Wp) complex

        # Circular HPF mask on (Hp, Wp)
        u = torch.arange(Hp, device=device)
        v = torch.arange(Wp, device=device)
        u = torch.where(u <= Hp//2, u, u - Hp).to(torch.float32)
        v = torch.where(v <= Wp//2, v, v - Wp).to(torch.float32)
        U, V = torch.meshgrid(u, v, indexing="ij")
        radius = torch.sqrt(U**2 + V**2)
        mask = (radius > float(self.hpf_radius)).to(F.dtype)

        # Apply mask
        F_hp = F * mask[None, :, :]

        # IFFT and energy
        hf2d = torch.fft.ifft2(F_hp, norm="ortho").real   # (B, Hp, Wp)
        energy2d = hf2d ** 2
        energy_flat_padded = energy2d.reshape(B, Hp * Wp)

        # Trim padding to original N
        energy_flat = energy_flat_padded[:, :N]

        # Normalise per sample
        phi_min = energy_flat.min(dim=1, keepdim=True).values
        phi_max = energy_flat.max(dim=1, keepdim=True).values
        phi = (energy_flat - phi_min) / (phi_max - phi_min + self.eps)
        return phi
