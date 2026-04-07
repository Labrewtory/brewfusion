"""DDPM Noise Scheduler for BrewFusion DiT.

Implements forward diffusion q(x_t | x_0) and reverse process utilities.
Supports both linear and cosine noise schedules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """DDPM noise scheduler with configurable beta schedule."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register as buffers (moved to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _cosine_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021)."""
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        f = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, min=1e-6, max=0.999)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from q(x_t | x_0).

        Args:
            x_0: Clean data [B, L, D]
            t: Timesteps [B]
            noise: Optional pre-sampled noise

        Returns:
            (x_t, noise) tuple
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]  # [B]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]  # [B]

        # Expand dims for broadcasting: [B] → [B, 1, 1]
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Reverse diffusion: sample x_{t-1} from p_theta(x_{t-1} | x_t).

        Args:
            model_output: Predicted noise ε_θ(x_t, t, c)
            x_t: Noisy data at timestep t [B, L, D]
            t: Current timestep (scalar)

        Returns:
            x_{t-1}
        """
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Predict x_0
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        x_0_pred = (
            x_t - sqrt_one_minus_alpha_cumprod * model_output
        ) / self.sqrt_alphas_cumprod[t]

        # Compute mean of posterior q(x_{t-1} | x_t, x_0)
        coeff_1 = beta * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - alpha_cumprod)
        coeff_2 = (
            (1.0 - self.alphas_cumprod_prev[t])
            * torch.sqrt(alpha)
            / (1.0 - alpha_cumprod)
        )
        mean = coeff_1 * x_0_pred + coeff_2 * x_t

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise

        return mean

    @torch.no_grad()
    def sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        gnn_memory: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Full reverse sampling loop.

        Args:
            model: DiT model with forward(x_t, t, c, gnn_memory)
            shape: (B, L, D) output shape
            condition: Condition vector [B, C]
            gnn_memory: GNN embeddings for cross-attention [N_ingredients, D]
            cfg_scale: Classifier-free guidance scale
            device: Target device

        Returns:
            Denoised sample x_0
        """
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)

            # CFG: compute both conditional and unconditional predictions
            if cfg_scale > 1.0:
                # Conditional
                noise_cond = model(x, t_batch, condition, gnn_memory)
                # Unconditional (zero condition)
                null_cond = torch.zeros_like(condition)
                noise_uncond = model(x, t_batch, null_cond, gnn_memory)
                # Guided prediction
                model_output = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                model_output = model(x, t_batch, condition, gnn_memory)

            x = self.p_sample(model_output, x, t)

        return x
