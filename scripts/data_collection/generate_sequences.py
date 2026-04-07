"""Unified sequence generator with GNN-aligned ingredient names.

Uses the SAME normalizers as the GNN graph builder, so ingredient names
in sequences match GNN node names exactly. Also builds an ingredient
registry mapping (token_name → GNN node index) for HybridTokenEmbedding.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re

import torch

from brewfusion.config import GRAPH_DIR, PROJECT_ROOT, RECIPES_JSON
from brewfusion.data.normalizers import (
    normalize_adjunct_name,
    normalize_hop_name,
    normalize_ingredient_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _to_token(name: str) -> str:
    """Convert a normalized lowercase name to a sequence token.

    This is a simple, REVERSIBLE transformation:
      'pale malt 2-row' → 'PALE_MALT_2_ROW'

    The key difference from the old clean_name(): we normalize FIRST
    via GNN normalizers, THEN convert to token form.
    """
    return re.sub(r"[^A-Z0-9]", "_", name.upper()).strip("_")


def generate_sequences(json_path: str | None = None) -> list[str]:
    """Generate recipe sequences with GNN-aligned ingredient names."""
    path = json_path or str(RECIPES_JSON)

    logger.info("Loading recipes from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d recipes", len(data))

    registry_path = PROJECT_ROOT / "src" / "brewfusion" / "data" / "style_registry.json"
    logger.info("Loading style registry from %s...", registry_path)
    with open(registry_path, "r", encoding="utf-8") as f:
        style_registry = json.load(f)

    sequences: list[str] = []

    for _key, recipe in data.items():
        style = recipe.get("style", "")
        if not isinstance(style, str) or style.strip() in ("", "--"):
            continue
        style = style.strip()
        style_idx = style_registry.get(style, 0) # Fallback to 0 if unknown

        # Scalar targets
        abv = _safe_float(recipe.get("abv"))
        ibu = _safe_float(recipe.get("ibu"))
        color = _safe_float(recipe.get("color"))

        if abv == 0 and ibu == 0:
            continue

        seq: list[str] = []
        seq.append("[START]")
        seq.append(f"[TARGET_ABV] {abv:.1f}")
        seq.append(f"[TARGET_IBU] {ibu:.1f}")
        seq.append(f"[TARGET_COLOR] {color:.1f}")
        seq.append(f"[TARGET_STYLE] {style_idx}")

        # Mash step
        seq.append("[MASH_STEP]")

        # Fermentables (using GNN normalizer)
        fermentables = recipe.get("fermentables", [])
        if isinstance(fermentables, list):
            for item in fermentables:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                raw_name = str(item[1])
                norm_name = normalize_ingredient_name(raw_name)
                token = _to_token(norm_name)
                weight = _safe_float(item[0])
                if token and token != "UNKNOWN":
                    seq.append(f"[MALT] {token} {weight:.2f} <KG>")

        # Hops
        hops_list = recipe.get("hops", [])
        boil_hops: list[tuple[str, float, float]] = []
        dryhop_entries: list[tuple[str, float]] = []

        if isinstance(hops_list, list):
            for item in hops_list:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                raw_name = str(item[1])
                norm_name = normalize_hop_name(raw_name)
                token = _to_token(norm_name)
                weight = _safe_float(item[0])
                # Use field at position 3-4 for timing if available
                boil_min = _safe_float(item[3]) if len(item) > 3 else 60.0
                use_type = str(item[2]).lower() if len(item) > 2 else "boil"

                if token and token != "UNKNOWN":
                    if "dry" in use_type:
                        dryhop_entries.append((token, weight))
                    else:
                        boil_hops.append((token, weight, boil_min))

        if boil_hops:
            seq.append("[BOIL_START]")
            for token, weight, mins in boil_hops:
                seq.append(f"[BOIL] {mins:.0f} <MIN>")
                seq.append(f"[HOP] {token} {weight:.1f} <G>")

        if dryhop_entries:
            for token, weight in dryhop_entries:
                seq.append("[DRY_HOP] 7 <DAYS>")
                seq.append(f"[HOP] {token} {weight:.1f} <G>")

        # Other/adjuncts
        others = recipe.get("other", [])
        if isinstance(others, list):
            for item in others:
                if not isinstance(item, list) or len(item) < 5:
                    continue
                category = str(item[2]).strip().lower()
                if category in ("flavor", "spice", "herb"):
                    norm_name = normalize_adjunct_name(str(item[1]))
                    token = _to_token(norm_name)
                    if token and token != "UNKNOWN":
                        seq.append(f"[SPICE] {token}")

        seq.append("[END]")
        sequences.append(" ".join(seq))

    logger.info("Generated %d sequences", len(sequences))
    return sequences


def build_ingredient_registry() -> dict[str, tuple[str, int]]:
    """Build token_name → (node_type, node_index) mapping.

    Loads the GNN graph and creates a lookup from the normalized
    token form of each ingredient to its GNN node index.
    """
    graph_path = GRAPH_DIR / "brewfusion_hetero_graph.pt"
    if not graph_path.exists():
        logger.warning("Graph not found at %s", graph_path)
        return {}

    graph = torch.load(graph_path, weights_only=False)
    registry: dict[str, tuple[str, int]] = {}

    for ntype in ["ingredient", "hop", "yeast"]:
        if ntype not in graph.node_types:
            continue
        names = graph[ntype].names
        for idx, raw_name in enumerate(names):
            token = _to_token(raw_name)
            if token and token != "UNKNOWN" and token not in registry:
                registry[token] = (ntype, idx)

    logger.info("Registry: %d token→GNN mappings", len(registry))
    return registry


def main() -> None:
    """Generate sequences and save splits + registry."""
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate sequences
    sequences = generate_sequences()

    # Shuffle and split
    random.seed(42)
    random.shuffle(sequences)

    n = len(sequences)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": sequences[:n_train],
        "val": sequences[n_train : n_train + n_val],
        "test": sequences[n_train + n_val :],
    }

    for name, data in splits.items():
        path = out_dir / f"{name}_sequences.txt"
        with open(path, "w") as f:
            for s in data:
                f.write(s + "\n")
        logger.info("Saved %d %s sequences to %s", len(data), name, path)

    # Combined for tokenizer training
    all_path = out_dir / "all_sequences.txt"
    with open(all_path, "w") as f:
        for s in sequences:
            f.write(s + "\n")
    logger.info("Saved %d total sequences to %s", len(sequences), all_path)

    # Build ingredient registry
    registry = build_ingredient_registry()

    # Verify coverage
    all_ingredients = set()
    for seq in sequences:
        parts = seq.split()
        for i, p in enumerate(parts):
            if p in ("[MALT]", "[HOP]", "[SPICE]") and i + 1 < len(parts):
                all_ingredients.add(parts[i + 1])

    mapped = sum(1 for name in all_ingredients if name in registry)
    logger.info(
        "Ingredient coverage: %d/%d (%.1f%%)",
        mapped,
        len(all_ingredients),
        mapped / max(len(all_ingredients), 1) * 100,
    )

    # Save registry
    registry_path = GRAPH_DIR / "ingredient_registry.pt"
    torch.save(registry, registry_path)
    logger.info("Saved registry to %s", registry_path)

    # Show examples
    print("\n--- Example Sequence ---")
    print(sequences[0][:300])
    print("\n--- Unmapped ingredients (top 5) ---")
    unmapped = [n for n in sorted(all_ingredients) if n not in registry][:5]
    for u in unmapped:
        print(f"  {u}")


if __name__ == "__main__":
    main()
