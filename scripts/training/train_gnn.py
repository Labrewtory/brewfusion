"""GNN training script for BrewFusion.

Trains a HeteroGNNEncoder on the full heterogeneous graph using:
  1. Link Prediction Loss (positive/negative edge sampling)
  2. CSP Loss (Chemical Structure Prediction — Morgan FP reconstruction)

Saves the trained model and per-node-type embeddings to disk.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from brewfusion.config import GNN_HIDDEN_DIM, GNN_OUT_DIM, GRAPH_DIR
from brewfusion.graph.builder import build_graph, load_graph, save_graph
from brewfusion.models.csp_layer import CSPLayer
from brewfusion.models.gnn_encoder import HeteroGNNEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EPOCHS = 100
LR = 1e-3
NEG_RATIO = 3  # negative samples per positive edge
CSP_WEIGHT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_negative_edges_fast(
    num_src: int,
    num_dst: int,
    num_neg: int,
) -> torch.Tensor:
    """Fast approximate negative sampling via random uniform.

    For sparse graphs (density << 1%), random collision with
    positive edges is negligible, so we skip the expensive set lookup.
    """
    neg_src = torch.randint(0, num_src, (num_neg,))
    neg_dst = torch.randint(0, num_dst, (num_neg,))
    return torch.stack([neg_src, neg_dst])


def link_prediction_loss(
    emb_src: torch.Tensor,
    emb_dst: torch.Tensor,
    pos_edge: torch.Tensor,
    neg_edge: torch.Tensor,
) -> torch.Tensor:
    """Compute BCE link prediction loss."""
    # Positive edges
    pos_src_emb = emb_src[pos_edge[0]]
    pos_dst_emb = emb_dst[pos_edge[1]]
    pos_score = (pos_src_emb * pos_dst_emb).sum(dim=-1)

    # Negative edges
    neg_src_emb = emb_src[neg_edge[0]]
    neg_dst_emb = emb_dst[neg_edge[1]]
    neg_score = (neg_src_emb * neg_dst_emb).sum(dim=-1)

    pos_label = torch.ones_like(pos_score)
    neg_label = torch.zeros_like(neg_score)

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([pos_label, neg_label])

    return F.binary_cross_entropy_with_logits(scores, labels)


def train() -> None:
    """Main training loop."""
    # ── Load or build graph ──
    graph_path = GRAPH_DIR / "brewfusion_hetero_graph.pt"
    if graph_path.exists():
        logger.info("Loading pre-built graph...")
        data = load_graph(graph_path)
    else:
        logger.info("No pre-built graph found. Building from JSON...")
        data = build_graph()
        save_graph(data)

    data = data.to(DEVICE)

    # ── Determine edge types for link prediction ──
    # Use only the main structural edges for LP loss.
    # NPMI co-occurrence edges are used for GNN message passing only.
    _LP_RELATIONS = {
        "uses_grain",
        "uses_hop",
        "uses_yeast",
        "uses_adjunct",
        "contains",
        "produces",
        "similar_to",
    }
    lp_edge_types = []
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        _, rel, _ = edge_type
        if ei.shape[1] > 0 and rel in _LP_RELATIONS:
            lp_edge_types.append(edge_type)
            logger.info("  LP edge: %s (%d edges)", edge_type, ei.shape[1])
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        _, rel, _ = edge_type
        if ei.shape[1] > 0 and rel not in _LP_RELATIONS:
            logger.info("  MSG-only edge: %s (%d edges)", edge_type, ei.shape[1])

    # ── Build model ──
    # Determine feature dims from actual data
    feature_dims = {}
    for ntype in data.node_types:
        if hasattr(data[ntype], "x") and data[ntype].x is not None:
            feature_dims[ntype] = data[ntype].x.shape[1]
        else:
            feature_dims[ntype] = 1  # fallback

    edge_types_for_gnn = [
        et for et in data.edge_types if data[et].edge_index.shape[1] > 0
    ]

    encoder = HeteroGNNEncoder(
        feature_dims=feature_dims,
        edge_types=edge_types_for_gnn,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=GNN_OUT_DIM,
        num_layers=2,
    ).to(DEVICE)

    # CSP layer for compound fingerprint reconstruction
    csp = CSPLayer(hidden_dim=GNN_OUT_DIM, fingerprint_size=1024).to(DEVICE)

    # ── Prepare data dicts ──
    x_dict = {
        ntype: data[ntype].x for ntype in data.node_types if hasattr(data[ntype], "x")
    }
    edge_index_dict = {
        et: data[et].edge_index
        for et in data.edge_types
        if data[et].edge_index.shape[1] > 0
    }

    # Lazy initialization: SAGEConv(-1, -1) requires a forward pass to determine input dims
    with torch.no_grad():
        _ = encoder(x_dict, edge_index_dict)

    logger.info("Model params: %d", sum(p.numel() for p in encoder.parameters()))
    logger.info("Training on: %s", DEVICE)

    optimizer = AdamW(
        list(encoder.parameters()) + list(csp.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        csp.train()
        optimizer.zero_grad()

        # Forward pass
        emb_dict = encoder(x_dict, edge_index_dict)

        # ── Link prediction loss ──
        total_lp_loss = torch.tensor(0.0, device=DEVICE)
        num_lp = 0

        for edge_type in lp_edge_types:
            src_type, _, dst_type = edge_type
            if src_type not in emb_dict or dst_type not in emb_dict:
                continue

            pos_edge = data[edge_type].edge_index
            if pos_edge.shape[1] == 0:
                continue

            num_src = emb_dict[src_type].shape[0]
            num_dst = emb_dict[dst_type].shape[0]
            # Cap negatives to keep training fast
            num_neg = min(pos_edge.shape[1] * NEG_RATIO, 50000)

            neg_edge = sample_negative_edges_fast(num_src, num_dst, num_neg).to(DEVICE)

            lp_loss = link_prediction_loss(
                emb_dict[src_type], emb_dict[dst_type], pos_edge, neg_edge
            )
            total_lp_loss = total_lp_loss + lp_loss
            num_lp += 1

        if num_lp > 0:
            total_lp_loss = total_lp_loss / num_lp

        # ── CSP loss ──
        csp_loss = torch.tensor(0.0, device=DEVICE)
        if "compound" in emb_dict and data["compound"].x.shape[0] > 0:
            compound_emb = emb_dict["compound"]
            compound_features = data["compound"].x
            csp_loss = csp.compute_loss(compound_emb, compound_features)

        # ── Total loss ──
        loss = total_lp_loss + CSP_WEIGHT * csp_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | LP: %.4f | CSP: %.4f | Total: %.4f",
                epoch,
                total_lp_loss.item(),
                csp_loss.item(),
                loss.item(),
            )

        if loss.item() < best_loss:
            best_loss = loss.item()

    logger.info("Best loss: %.4f", best_loss)

    # ── Save embeddings ──
    encoder.eval()
    with torch.no_grad():
        emb_dict = encoder(x_dict, edge_index_dict)

    # Move to CPU for saving
    emb_cpu = {k: v.cpu() for k, v in emb_dict.items()}

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = GRAPH_DIR / "gnn_embeddings.pt"
    torch.save(emb_cpu, emb_path)
    logger.info("Saved embeddings to %s", emb_path)

    # Save encoder weights
    model_path = GRAPH_DIR / "gnn_encoder.pt"
    torch.save(encoder.state_dict(), model_path)
    logger.info("Saved GNN encoder to %s", model_path)

    # ── Print embedding stats ──
    for ntype, emb in emb_cpu.items():
        logger.info(
            "  %s: shape=%s, mean=%.4f, std=%.4f",
            ntype,
            emb.shape,
            emb.mean(),
            emb.std(),
        )


if __name__ == "__main__":
    train()
