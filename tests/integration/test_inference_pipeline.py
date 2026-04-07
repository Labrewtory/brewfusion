import torch
from brewfusion.models.dit_brewfusion import BrewFusionDiT


def test_inference_sampling_step():
    """Validates the mathematical stability of the inference sampling step (e.g. DDIM/DDPM)."""
    d_model = 64
    # Tiny model for test
    model = BrewFusionDiT(
        d_model=d_model,
        nhead=2,
        num_layers=2,
        seq_len=8,
        num_scalars=3,
        num_styles=10,
        style_emb_dim=16,
    )
    model.eval()

    # Setup inference mock condition [batch_size, 4]
    # Representing: ABV=5.0, IBU=40.0, Color=10.0, StyleIdx=2
    c = torch.tensor([[5.0, 40.0, 10.0, 2.0]])

    # Start from pure noise
    x_t = torch.randn(1, 8, d_model)
    t = torch.tensor([999])

    with torch.no_grad():
        noise_pred = model(x_t, t, c)

    # In standard diffusion, predict noise matches latent shape
    assert noise_pred.shape == x_t.shape, "Inference prediction shape mismatch"
    assert not torch.isnan(noise_pred).any(), (
        "NaN values detected during mock inference"
    )
