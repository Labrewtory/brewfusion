"""Hybrid Token Embedding for BrewFusion DiT.

Routes tokens to different embedding sources:
  - Structural tokens ([MALT], <KG>, etc.) → Learned embedding
  - Ingredient tokens (PALE_MALT_2_ROW, CASCADE, etc.) → GNN embedding + projection
  - Fallback (numbers, unknown) → Learned embedding

This eliminates the disconnect between GNN chemical knowledge and
DiT token representations, and enables zero-shot embedding of new
ingredients via their chemical structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from brewfusion.config import GNN_OUT_DIM, GRAPH_DIR

logger = logging.getLogger(__name__)


class HybridTokenEmbedding(nn.Module):
    """Hybrid embedding that uses GNN vectors for ingredient tokens.

    For each token ID:
      - If it maps to a GNN node → use frozen GNN embedding + learnable projection
      - Otherwise → use standard learned embedding

    This ensures chemical knowledge flows directly into the DiT input,
    and new ingredients can be added without retraining.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        gnn_emb_path: str | Path | None = None,
        registry_path: str | Path | None = None,
        gnn_dim: int = GNN_OUT_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Standard learned embedding for non-ingredient tokens
        self.learned_emb = nn.Embedding(vocab_size, d_model)

        # GNN projection: frozen GNN embedding (gnn_dim) → d_model
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Token ID → GNN embedding index mapping
        # -1 means "use learned embedding" (no GNN match)
        self.register_buffer(
            "token_to_gnn_idx",
            torch.full((vocab_size,), -1, dtype=torch.long),
        )

        # Will be set by register_buffer in _load_gnn_mappings
        self._has_gnn = False

        if gnn_emb_path and registry_path:
            self._load_gnn_mappings(gnn_emb_path, registry_path)

    def _load_gnn_mappings(
        self,
        gnn_emb_path: str | Path,
        registry_path: str | Path,
    ) -> None:
        """Load GNN embeddings and build token→GNN index mapping."""
        from tokenizers import Tokenizer
        from brewfusion.config import PROJECT_ROOT

        # Load GNN embeddings
        gnn_emb = torch.load(gnn_emb_path, weights_only=False)

        # Build concatenated GNN bank: [ingredient | hop | yeast] → flat index
        parts = []
        offsets: dict[str, int] = {}
        current_offset = 0
        for ntype in ["ingredient", "hop", "yeast"]:
            if ntype in gnn_emb:
                offsets[ntype] = current_offset
                parts.append(gnn_emb[ntype])
                current_offset += gnn_emb[ntype].shape[0]

        if not parts:
            logger.warning("No GNN embeddings found!")
            return

        gnn_bank = torch.cat(parts, dim=0)  # [N_total, gnn_dim]
        self.register_buffer("gnn_bank", gnn_bank)
        self._has_gnn = True

        # Load registry: token_name → (node_type, node_index)
        registry = torch.load(registry_path, weights_only=False)

        # Load tokenizer to get token IDs
        tok_path = PROJECT_ROOT / "src" / "brewfusion" / "data" / "brew_tokenizer.json"
        tok = Tokenizer.from_file(str(tok_path))

        # Build mapping
        mapped_count = 0
        for token_name, (ntype, node_idx) in registry.items():
            if ntype not in offsets:
                continue
            token_id = tok.token_to_id(token_name)
            if token_id is not None and token_id > 0:
                flat_idx = offsets[ntype] + node_idx
                self.token_to_gnn_idx[token_id] = flat_idx
                mapped_count += 1

        # Also check BPE multi-token → we only map single-token ingredients
        total_registry = len(registry)
        logger.info(
            "HybridEmb: %d/%d registry entries mapped to BPE tokens (%.1f%%)",
            mapped_count,
            total_registry,
            mapped_count / max(total_registry, 1) * 100,
        )

        gnn_token_count = (self.token_to_gnn_idx >= 0).sum().item()
        logger.info(
            "HybridEmb: %d/%d vocab tokens use GNN embedding (%.1f%%)",
            gnn_token_count,
            self.vocab_size,
            gnn_token_count / self.vocab_size * 100,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            token_ids: [B, L] integer token IDs

        Returns:
            [B, L, d_model] embeddings
        """
        # Start with learned embeddings for everything
        embeddings = self.learned_emb(token_ids)  # [B, L, d_model]

        if self._has_gnn:
            # Find which tokens have GNN mappings
            gnn_idx = self.token_to_gnn_idx[token_ids]  # [B, L]
            gnn_mask = gnn_idx >= 0  # [B, L]

            if gnn_mask.any():
                # Get GNN embeddings for mapped tokens
                valid_indices = gnn_idx[gnn_mask]  # [N_valid]
                gnn_vectors = self.gnn_bank[valid_indices]  # [N_valid, gnn_dim]

                # Project to d_model
                projected = self.gnn_proj(gnn_vectors)  # [N_valid, d_model]

                # Replace learned embeddings with GNN embeddings
                embeddings = embeddings.clone()
                embeddings[gnn_mask] = projected

        return embeddings


def create_hybrid_embedding(
    vocab_size: int,
    d_model: int,
) -> HybridTokenEmbedding:
    """Factory function to create HybridTokenEmbedding with default paths."""
    gnn_emb_path = GRAPH_DIR / "gnn_embeddings.pt"
    registry_path = GRAPH_DIR / "ingredient_registry.pt"

    return HybridTokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        gnn_emb_path=str(gnn_emb_path) if gnn_emb_path.exists() else None,
        registry_path=str(registry_path) if registry_path.exists() else None,
    )
