"""Recipe generation via DiT reverse diffusion.

Generates recipe sequences by:
  1. Loading trained DiT model + token embeddings
  2. Running DDPM reverse sampling with CFG
  3. Decoding continuous embeddings to tokens via nearest-neighbor lookup
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer

from brewfusion.config import (
    DIT_D_MODEL,
    DIT_NHEAD,
    DIT_NUM_LAYERS,
    DIT_SEQ_LEN,
    GRAPH_DIR,
    PROJECT_ROOT,
)
from brewfusion.models.dit_brewfusion import BrewFusionDiT
from brewfusion.models.hybrid_embedding import HybridTokenEmbedding
from brewfusion.models.scheduler import DDPMScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
) -> tuple[BrewFusionDiT, torch.nn.Embedding, DDPMScheduler, Tokenizer]:
    """Load trained DiT model, token embeddings, scheduler, and tokenizer."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    model = BrewFusionDiT(
        d_model=DIT_D_MODEL,
        nhead=DIT_NHEAD,
        num_layers=DIT_NUM_LAYERS,
        seq_len=DIT_SEQ_LEN,
        num_scalars=3,
        num_styles=180,
        style_emb_dim=32,
    ).to(DEVICE)

    gnn_emb_path = GRAPH_DIR / "gnn_embeddings.pt"
    registry_path = GRAPH_DIR / "ingredient_registry.pt"

    token_emb = HybridTokenEmbedding(
        vocab_size=vocab_size,
        d_model=DIT_D_MODEL,
        gnn_emb_path=str(gnn_emb_path),
        registry_path=str(registry_path),
        gnn_dim=64,
    ).to(DEVICE)
    
    scheduler = DDPMScheduler(num_timesteps=1000, schedule="cosine").to(DEVICE)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    token_emb.load_state_dict(checkpoint["token_emb"])

    model.eval()
    token_emb.eval()

    logger.info("Loaded model from %s", checkpoint_path)
    return model, token_emb, scheduler, tokenizer


def decode_embeddings(
    embeddings: torch.Tensor,
    token_emb: HybridTokenEmbedding,
    tokenizer: Tokenizer,
    pad_id: int | None = None,
) -> list[str]:
    """Decode continuous embeddings to token strings via nearest neighbor.

    Args:
        embeddings: [B, L, D] denoised embeddings
        token_emb: HybridTokenEmbedding (contains learned_emb + gnn_proj)
        tokenizer: Tokenizer for ID → string conversion
        pad_id: PAD token ID to skip

    Returns:
        List of decoded strings
    """
    if pad_id is None:
        pad_id = tokenizer.token_to_id("[PAD]")

    # Build full vocabulary embedding matrix
    # HybridTokenEmbedding maps tokens [0, vocab_size-1]
    # We can compute it by running a forward pass on all token IDs!
    device = embeddings.device
    all_token_ids = torch.arange(tokenizer.get_vocab_size(), device=device).unsqueeze(0) # [1, V]
    with torch.no_grad():
        weight = token_emb(all_token_ids).squeeze(0)  # [V, D]

    decoded_strings = []
    for b in range(embeddings.shape[0]):
        seq_emb = embeddings[b]  # [L, D]

        # Cosine similarity: [L, V]
        seq_norm = torch.nn.functional.normalize(seq_emb, dim=-1)
        weight_norm = torch.nn.functional.normalize(weight, dim=-1)
        similarity = seq_norm @ weight_norm.T  # [L, V]

        # Argmax to get token IDs
        token_ids = similarity.argmax(dim=-1).tolist()

        # Filter PAD tokens and decode
        filtered_ids = [tid for tid in token_ids if tid != pad_id]
        decoded = tokenizer.decode(filtered_ids, skip_special_tokens=False)
        decoded_strings.append(decoded)

    return decoded_strings


@torch.no_grad()
def generate(
    abv: float = 6.0,
    ibu: float = 40.0,
    color: float = 10.0,
    style_idx: int = 0,
    cfg_scale: float = 3.0,
    num_samples: int = 1,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    """Generate recipe sequences with specified scalar constraints.

    Args:
        abv: Target ABV (0-15)
        ibu: Target IBU (0-120)
        color: Target Color SRM (0-50)
        style_idx: Beer style index (0-179)
        cfg_scale: CFG guidance strength (higher = more constrained)
        num_samples: Number of recipes to generate
        checkpoint_path: Path to model checkpoint

    Returns:
        List of generated recipe strings
    """
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "data" / "models" / "dit_best.pt"

    tokenizer_path = (
        PROJECT_ROOT / "src" / "brewfusion" / "data" / "brew_tokenizer.json"
    )

    model, token_emb, scheduler, tokenizer = load_model(checkpoint_path, tokenizer_path)

    # Load GNN memory for cross-attention
    gnn_emb_path = GRAPH_DIR / "gnn_embeddings.pt"
    gnn_memory = None
    if gnn_emb_path.exists():
        gnn_emb = torch.load(gnn_emb_path, weights_only=False)
        memory_parts = []
        for key in ["ingredient", "hop", "yeast"]:
            if key in gnn_emb:
                memory_parts.append(gnn_emb[key])
        if memory_parts:
            full_memory = torch.cat(memory_parts, dim=0)
            if full_memory.shape[0] > 256:
                # Randomly sample 256 memory tokens to avoid OpenMP deadlock with sklearn
                idx = torch.randperm(full_memory.shape[0])[:256]
                gnn_memory = full_memory[idx].to(DEVICE)
            else:
                gnn_memory = full_memory.to(DEVICE)
        logger.info(
            "GNN memory: %s", gnn_memory.shape if gnn_memory is not None else None
        )

    # Build condition vector
    abv_norm = min(abv / 15.0, 1.0)
    ibu_norm = min(ibu / 120.0, 1.0)
    color_norm = min(color / 50.0, 1.0)

    condition = torch.tensor(
        [[abv_norm, ibu_norm, color_norm, float(style_idx)]] * num_samples,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Run reverse diffusion
    logger.info(
        "Generating %d recipes (ABV=%.1f, IBU=%.1f, Color=%.1f, CFG=%.1f)...",
        num_samples,
        abv,
        ibu,
        color,
        cfg_scale,
    )

    x_0 = scheduler.sample_loop(
        model=model,
        shape=(num_samples, DIT_SEQ_LEN, DIT_D_MODEL),
        condition=condition,
        gnn_memory=gnn_memory,
        cfg_scale=cfg_scale,
        device=DEVICE,
    )

    # Decode to text
    recipes = decode_embeddings(x_0, token_emb, tokenizer)

    for i, recipe in enumerate(recipes):
        logger.info("Recipe %d: %s", i + 1, recipe[:200])

    return recipes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate brewing recipes with BrewFusion DiT"
    )
    parser.add_argument("--abv", type=float, default=6.0, help="Target ABV")
    parser.add_argument("--ibu", type=float, default=40.0, help="Target IBU")
    parser.add_argument("--color", type=float, default=10.0, help="Target Color (SRM)")
    parser.add_argument("--style", type=int, default=0, help="Style index (0-179)")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--num", type=int, default=3, help="Number of samples")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Model checkpoint path"
    )
    args = parser.parse_args()

    results = generate(
        abv=args.abv,
        ibu=args.ibu,
        color=args.color,
        style_idx=args.style,
        cfg_scale=args.cfg,
        num_samples=args.num,
        checkpoint_path=args.checkpoint,
    )

    print("\n" + "=" * 60)
    for i, recipe in enumerate(results):
        print(f"\n--- Recipe {i + 1} ---")
        print(recipe)
