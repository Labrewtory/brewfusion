"""Diffusion Transformer (DiT) for BrewFusion.

AdaLN-Zero conditioned Transformer that denoises recipe sequence embeddings
in continuous latent space, with cross-attention to GNN ingredient embeddings.

Architecture:
  - Sinusoidal timestep embedding
  - Scalar condition embedding (ABV, IBU, Color) + style embedding → AdaLN
  - Self-Attention (inter-token relationships)
  - Cross-Attention (GNN chemical knowledge injection)
  - Pointwise FFN
  - AdaLN-Zero modulation on every sublayer

Optimized for 8GB GPU: d_model=192, nhead=6, num_layers=6, seq_len=64
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] integer timesteps → [B, dim] embeddings."""
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLNZeroModulation(nn.Module):
    """Adaptive LayerNorm Zero modulation.

    Given a condition embedding c, produces (shift, scale, gate)
    for each sublayer.
    """

    def __init__(self, d_model: int, n_modulations: int = 6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, n_modulations * d_model)
        self.n_modulations = n_modulations
        self.d_model = d_model
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """c: [B, d_model] → tuple of n_modulations × [B, 1, d_model]."""
        out = self.linear(self.silu(c))  # [B, n*d]
        chunks = out.chunk(self.n_modulations, dim=-1)  # n × [B, d]
        return tuple(ch.unsqueeze(1) for ch in chunks)  # n × [B, 1, d]


class DiTBlock(nn.Module):
    """Single DiT block with self-attention, cross-attention, and FFN."""

    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)

        # Cross-attention (to GNN embeddings)
        self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1
        )

        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(0.1),
        )

        # AdaLN-Zero: 9 modulations (shift/scale/gate × 3 sublayers)
        self.adaln = AdaLNZeroModulation(d_model, n_modulations=9)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence embeddings [B, L, D]
            c: Condition embedding [B, D]
            memory: GNN embeddings for cross-attention [B, N, D] or None

        Returns:
            [B, L, D]
        """
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = self.adaln(c)

        # Self-Attention with AdaLN-Zero
        h = self.norm1(x)
        h = h * (1 + scale_sa) + shift_sa
        h, _ = self.attn(h, h, h)
        x = x + gate_sa * h

        # Cross-Attention with AdaLN-Zero (if GNN memory available)
        if memory is not None:
            h = self.norm_cross(x)
            h = h * (1 + scale_ca) + shift_ca
            h, _ = self.cross_attn(h, memory, memory)
            x = x + gate_ca * h

        # FFN with AdaLN-Zero
        h = self.norm2(x)
        h = h * (1 + scale_ff) + shift_ff
        h = self.ffn(h)
        x = x + gate_ff * h

        return x


class BrewFusionDiT(nn.Module):
    """Diffusion Transformer for brewing recipe generation.

    Denoises recipe sequence embeddings conditioned on:
    - Scalar targets (ABV, IBU, Color) via AdaLN
    - Style embedding via AdaLN
    - GNN chemical embeddings via Cross-Attention
    """

    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 6,
        seq_len: int = 64,
        num_scalars: int = 3,  # ABV, IBU, Color
        num_styles: int = 180,
        style_emb_dim: int = 64, # Replaced NLP embedding dimension to match GNN dimension
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Positional embedding for sequence positions
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Condition embedding: scalars + style → d_model
        self.style_emb = nn.Embedding(num_styles, style_emb_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(num_scalars + style_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead) for _ in range(num_layers)]
        )

        # Final projection: predict noise
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaln = AdaLNZeroModulation(d_model, n_modulations=2)
        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # GNN embedding projection (if GNN dim != d_model)
        self.gnn_proj = nn.Linear(64, d_model)  # GNN_OUT_DIM → d_model

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        gnn_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t, c).

        Args:
            x_t: Noisy sequence embeddings [B, L, D]
            t: Timesteps [B]
            condition: Condition vector [B, num_scalars + 1] where last dim is style_idx
            gnn_memory: GNN embeddings [N_total, 64] (shared across batch)

        Returns:
            Predicted noise [B, L, D]
        """
        B = x_t.shape[0]

        # Parse condition: scalars + style index
        scalars = condition[:, :3]  # [B, 3] — ABV, IBU, Color
        style_idx = condition[:, 3].long()  # [B]

        # Build condition embedding
        style_vec = self.style_emb(style_idx)  # [B, style_emb_dim]
        cond_input = torch.cat([scalars, style_vec], dim=-1)  # [B, 3+style_emb_dim]
        c = self.cond_proj(cond_input)  # [B, d_model]

        # Add timestep embedding
        t_emb = self.time_emb(t)  # [B, d_model]
        c = c + t_emb

        # Add positional embedding to input
        x = x_t + self.pos_emb[:, : x_t.shape[1], :]

        # Project GNN memory if provided
        memory = None
        if gnn_memory is not None:
            # gnn_memory: [N, 64] → [1, N, d_model] → expand to [B, N, d_model]
            memory = self.gnn_proj(gnn_memory)
            if memory.dim() == 2:
                memory = memory.unsqueeze(0).expand(B, -1, -1)

        # DiT blocks (with optional gradient checkpointing)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, memory, use_reentrant=False
                )
            else:
                x = block(x, c, memory)

        # Final projection
        shift, scale = self.final_adaln(c)
        x = self.final_norm(x)
        x = x * (1 + scale) + shift
        return self.output_proj(x)
