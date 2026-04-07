import torch
from brewfusion.models.dit_brewfusion import BrewFusionDiT


def test_dit_forward_shapes():
    """Verifies that the DiT outputs proper noise prediction and handles AdaLN routing."""

    d_model = 64

    model = BrewFusionDiT(
        d_model=d_model,
        nhead=2,
        num_layers=2,
        seq_len=12,
        num_scalars=3,
        num_styles=10,
        style_emb_dim=16,
    )

    # Mock inputs
    B, S = 4, 12
    x_t = torch.randn(
        B, S, d_model, requires_grad=True
    )  # Continuous embedded noisy sequences
    t = torch.randint(0, 1000, (B,))  # Timesteps

    # Condition: 3 scalars + 1 style_idx
    scalars = torch.rand(B, 3)
    style_idx = torch.randint(0, 10, (B, 1)).float()
    c = torch.cat([scalars, style_idx], dim=-1)  # [B, 4]

    # Mock GNN memory (shared across batch for cross-attn)
    # usually shape [N, 64]
    gnn_memory = torch.randn(20, 64)

    model.eval()  # Disable checkpointing for tests to avoid silent AnyIO crashes

    # Forward pass
    out = model(x_t, t, c, gnn_memory=gnn_memory)

    # Output must be Noise Prediction of same shape as x_t
    assert out.shape == x_t.shape, (
        "DiT must output noise predictions matching input embedding shapes"
    )

    # Test gradients backward flow
    loss = out.sum()
    loss.backward()

    assert model.time_emb[1].weight.grad is not None, "TimeMLP must receive gradients"
    assert model.blocks[0].adaln.linear.weight.grad is not None, (
        "AdaLN Zero layer MUST receive gradients"
    )
    assert model.gnn_proj.weight.grad is not None, (
        "GNN Memory Projection must receive gradients"
    )
