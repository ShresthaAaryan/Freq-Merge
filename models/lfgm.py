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

        # If the token count matches the original grid, use the 2D FFT as
        # described in the paper. Otherwise fall back to the 1-D FFT
        # behaviour (resolution-agnostic) to avoid reshape errors after
        # earlier merges.
        if N == H0 * W0:
            H, W = H0, W0
            signal_2d = signal_1d.view(B, H, W)

            # Step 3: 2-D FFT (complex output)
            F = torch.fft.fft2(signal_2d, norm="ortho")   # (B, H, W) complex

            # Step 4: Build circular high-pass filter mask in frequency-domain
            device = tokens.device
            # Create frequency index grids centered at zero
            u = torch.arange(H, device=device)
            v = torch.arange(W, device=device)
            u = torch.where(u <= H//2, u, u - H).to(torch.float32)
            v = torch.where(v <= W//2, v, v - W).to(torch.float32)
            U, V = torch.meshgrid(u, v, indexing="ij")
            radius = torch.sqrt(U**2 + V**2)

            # Zero out frequencies with radius <= hpf_radius (low-freq)
            mask = (radius > float(self.hpf_radius)).to(F.dtype)   # 1 = keep, 0 = zero

            # Apply mask (broadcast over batch)
            F_hp = F * mask[None, :, :]

            # Step 5: inverse FFT → high-frequency residual (real part)
            hf2d = torch.fft.ifft2(F_hp, norm="ortho").real   # (B, H, W)

            # Step 6: energy per token and min-max normalisation per sample
            energy = hf2d ** 2                                # (B, H, W)
            energy_flat = energy.view(B, N)                   # (B, N)
            phi_min = energy_flat.min(dim=1, keepdim=True).values
            phi_max = energy_flat.max(dim=1, keepdim=True).values
            phi = (energy_flat - phi_min) / (phi_max - phi_min + self.eps)
            return phi   # (B, N) in [0, 1]

        # Fallback: resolution-agnostic 1-D FFT approach (previous behaviour)
        # This keeps the module robust when token counts have been reduced
        # by earlier merges and cannot be reshaped into the original 2D grid.
        signal = signal_1d                                  # (B, N)
        F1 = torch.fft.rfft(signal, norm="ortho")         # (B, N//2 + 1)
        r = min(self.hpf_radius, F1.shape[-1])
        F1_hp = F1.clone()
        F1_hp[:, :r] = 0.0
        hf1 = torch.fft.irfft(F1_hp, n=N, norm="ortho")   # (B, N)
        energy1 = hf1 ** 2
        phi_min1 = energy1.min(dim=1, keepdim=True).values
        phi_max1 = energy1.max(dim=1, keepdim=True).values
        phi1 = (energy1 - phi_min1) / (phi_max1 - phi_min1 + self.eps)
        return phi1
