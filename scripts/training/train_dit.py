"""DiT training script for BrewFusion.

Trains the Diffusion Transformer on recipe sequences using DDPM,
conditioning on scalar targets (ABV, IBU, Color) with CFG support.

Pipeline:
  1. Load tokenized sequences (from train_sequences.txt)
  2. Encode tokens into learned embeddings (d_model=192)
  3. Add DDPM noise to create x_t
  4. DiT predicts noise, train with MSE loss
  5. CFG: randomly drop condition with 15% probability
  6. GNN embeddings provided as cross-attention memory
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
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
from brewfusion.models.hybrid_embedding import create_hybrid_embedding
from brewfusion.models.scheduler import DDPMScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EPOCHS = 50
LR = 3e-4
CFG_DROP_PROB = 0.15
NUM_TIMESTEPS = 1000
GNN_MEMORY_SIZE = 256  # K-means centroids for cross-attention efficiency
TRAIN_BATCH_SIZE = 64  # GPU utilization was only 3.7/8GB at bs=16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regex to extract scalar targets from sequence text
_TARGET_RE = re.compile(
    r"\[TARGET_ABV\]\s+([\d.]+)\s+\[TARGET_IBU\]\s+([\d.]+)\s+\[TARGET_COLOR\]\s+([\d.]+)"
)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class DiffusionSequenceDataset(Dataset):
    """Dataset for DiT training with binary cache support.

    First load tokenizes all sequences and saves a .pt cache.
    Subsequent loads are instant.

    Each sample returns:
      - token_ids: [seq_len] padded/truncated token IDs
      - condition: [4] tensor of [ABV_norm, IBU_norm, Color_norm, style_idx]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_length: int = DIT_SEQ_LEN,
    ):
        cache_path = data_path + f".dit_cache_{max_length}.pt"

        if Path(cache_path).exists():
            logger.info("Loading cached dataset from %s", cache_path)
            cache = torch.load(cache_path, weights_only=False)
            self.token_ids_list = cache["token_ids"]
            self.conditions = cache["conditions"]
            logger.info("Loaded %d cached sequences", len(self.token_ids_list))
            return

        logger.info("Building dataset cache (first time)...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = tokenizer.token_to_id("[PAD]")

        self.token_ids_list: list[torch.Tensor] = []
        self.conditions: list[torch.Tensor] = []

        with open(data_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info("Tokenizing %d sequences...", len(lines))
        for i, line in enumerate(lines):
            match = _TARGET_RE.search(line)
            if not match:
                continue

            abv = float(match.group(1))
            ibu = float(match.group(2))
            color = float(match.group(3))

            abv_norm = min(abv / 15.0, 1.0)
            ibu_norm = min(ibu / 120.0, 1.0)
            color_norm = min(color / 50.0, 1.0)
            style_idx = 0.0

            encoded = tokenizer.encode(line)
            ids = encoded.ids[:max_length]
            if len(ids) < max_length:
                ids = ids + [pad_id] * (max_length - len(ids))

            self.token_ids_list.append(torch.tensor(ids, dtype=torch.long))
            self.conditions.append(
                torch.tensor(
                    [abv_norm, ibu_norm, color_norm, style_idx], dtype=torch.float32
                )
            )

            if (i + 1) % 50000 == 0:
                logger.info("  ...processed %d / %d", i + 1, len(lines))

        # Save cache
        torch.save(
            {
                "token_ids": self.token_ids_list,
                "conditions": self.conditions,
            },
            cache_path,
        )
        logger.info("Cached %d sequences to %s", len(self.token_ids_list), cache_path)

    def __len__(self) -> int:
        return len(self.token_ids_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids_list[idx], self.conditions[idx]


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train() -> None:
    """Main DiT training loop."""
    # ── Paths ──
    train_path = PROJECT_ROOT / "data" / "processed" / "train_sequences.txt"
    val_path = PROJECT_ROOT / "data" / "processed" / "val_sequences.txt"
    tokenizer_path = (
        PROJECT_ROOT / "src" / "brewfusion" / "data" / "brew_tokenizer.json"
    )
    gnn_emb_path = GRAPH_DIR / "gnn_embeddings.pt"
    checkpoint_dir = PROJECT_ROOT / "data" / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    train_ds = DiffusionSequenceDataset(
        str(train_path), str(tokenizer_path), DIT_SEQ_LEN
    )
    val_ds = DiffusionSequenceDataset(str(val_path), str(tokenizer_path), DIT_SEQ_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Load GNN embeddings for cross-attention ──
    # Reduce to GNN_MEMORY_SIZE centroids via K-means to speed up cross-attention
    gnn_memory = None
    if gnn_emb_path.exists():
        gnn_emb = torch.load(gnn_emb_path, weights_only=False)
        memory_parts = []
        for key in ["ingredient", "hop", "yeast"]:
            if key in gnn_emb:
                memory_parts.append(gnn_emb[key])
        if memory_parts:
            full_memory = torch.cat(memory_parts, dim=0)  # [N_total, 64]
            logger.info("GNN memory (full): %s", full_memory.shape)

            # K-means reduction for cross-attention efficiency
            if full_memory.shape[0] > GNN_MEMORY_SIZE:
                from sklearn.cluster import MiniBatchKMeans

                kmeans = MiniBatchKMeans(
                    n_clusters=GNN_MEMORY_SIZE, batch_size=1024, random_state=42
                )
                kmeans.fit(full_memory.numpy())
                gnn_memory = torch.tensor(
                    kmeans.cluster_centers_, dtype=torch.float32
                ).to(DEVICE)
                logger.info("GNN memory (reduced): %s", gnn_memory.shape)
            else:
                gnn_memory = full_memory.to(DEVICE)

    # ── Build model ──
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

    # Hybrid embedding: GNN vectors for ingredients, learned for structure/numbers
    token_emb = create_hybrid_embedding(vocab_size, DIT_D_MODEL).to(DEVICE)

    scheduler = DDPMScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule="cosine",
    ).to(DEVICE)

    # ── Optimizer ──
    all_params = list(model.parameters()) + list(token_emb.parameters())
    optimizer = AdamW(all_params, lr=LR, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda")

    total_params = sum(p.numel() for p in all_params)
    logger.info(
        "DiT params: %d  |  Token Emb params: %d  |  Total: %d",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in token_emb.parameters()),
        total_params,
    )
    logger.info("Training on %s for %d epochs", DEVICE, EPOCHS)

    # ── Training loop ──
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        token_emb.train()
        total_loss = 0.0
        num_batches = 0

        for batch_ids, batch_cond in train_loader:
            batch_ids = batch_ids.to(DEVICE)  # [B, L]
            batch_cond = batch_cond.to(DEVICE)  # [B, 4]

            # Token IDs → continuous embeddings
            x_0 = token_emb(batch_ids)  # [B, L, D]

            # Sample random timesteps
            t = torch.randint(0, NUM_TIMESTEPS, (x_0.shape[0],), device=DEVICE)

            # CFG: randomly drop condition
            if CFG_DROP_PROB > 0:
                drop_mask = torch.rand(x_0.shape[0], device=DEVICE) < CFG_DROP_PROB
                batch_cond = batch_cond.clone()
                batch_cond[drop_mask] = 0.0

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.float16):
                # Forward diffusion
                x_t, noise = scheduler.q_sample(x_0, t)

                # Predict noise
                noise_pred = model(x_t, t, batch_cond, gnn_memory)

                # MSE loss
                loss = nn.functional.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        lr_scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)

        # ── Validation ──
        model.eval()
        token_emb.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_ids, batch_cond in val_loader:
                batch_ids = batch_ids.to(DEVICE)
                batch_cond = batch_cond.to(DEVICE)

                x_0 = token_emb(batch_ids)
                t = torch.randint(0, NUM_TIMESTEPS, (x_0.shape[0],), device=DEVICE)

                with autocast("cuda", dtype=torch.float16):
                    x_t, noise = scheduler.q_sample(x_0, t)
                    noise_pred = model(x_t, t, batch_cond, gnn_memory)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # ── Logging ──
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | Train: %.6f | Val: %.6f | LR: %.2e",
                epoch,
                avg_train_loss,
                avg_val_loss,
                lr_scheduler.get_last_lr()[0],
            )

        # ── Checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "token_emb": token_emb.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                checkpoint_dir / "dit_best.pt",
            )

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)

    # ── Save final model ──
    torch.save(
        {
            "epoch": EPOCHS,
            "model": model.state_dict(),
            "token_emb": token_emb.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "dit_final.pt",
    )
    logger.info("Saved final model to %s", checkpoint_dir / "dit_final.pt")


if __name__ == "__main__":
    train()
