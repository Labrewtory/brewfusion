"""Heterogeneous graph builder for BrewFusion.

Orchestrates the full pipeline:
  1. Parse recipes_full.txt (JSON) → nodes & edges
  2. Build compound nodes from curated DB
  3. Compute molecular features via RDKit
  4. Create ingredient/hop/yeast → compound edges
  5. Create compound ↔ compound similarity edges
  6. Compute ingredient-ingredient NPMI co-occurrence edges (FlavorDiffusion insight)
  7. Assemble everything into a PyG HeteroData object
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from brewfusion.chem.fingerprints import compute_compound_features
from brewfusion.chem.similarity import compute_similarity_edges
from brewfusion.config import (
    BEER_NUMERIC_FEATURES,
    EDGE_COMPOUND_SIMILAR,
    EDGE_HOP_COOCCURS,
    EDGE_HOP_CONTAINS,
    EDGE_INGREDIENT_COOCCURS,
    EDGE_INGREDIENT_CONTAINS,
    EDGE_USES_ADJUNCT,
    EDGE_USES_GRAIN,
    EDGE_USES_HOP,
    EDGE_USES_YEAST,
    EDGE_YEAST_PRODUCES,
    GRAPH_DIR,
    NODE_BEER_STYLE,
    PROJECT_ROOT,
    NODE_COMPOUND,
    NODE_HOP,
    NODE_INGREDIENT,
    NODE_YEAST,
)
from brewfusion.data.compound_db import (
    ADJUNCT_COMPOUND_MAP,
    COMPOUNDS,
    GRAIN_COMPOUND_MAP,
    HOP_COMPOUND_MAP,
    YEAST_COMPOUND_MAP,
    classify_yeast_family,
)
from brewfusion.data.json_parser import ParsedGraph, parse_json

logger = logging.getLogger(__name__)


def _build_index_map(names: list[str]) -> dict[str, int]:
    """Create name → integer index mapping."""
    return {name: idx for idx, name in enumerate(names)}


def _build_edge_index(
    edges: list[tuple[str, str, dict]],
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> tuple[torch.Tensor, list[dict]]:
    """Convert named edges to (2, E) tensor + list of edge attrs."""
    src_list, dst_list, attr_list = [], [], []
    for src_name, dst_name, attrs in edges:
        if src_name not in src_map or dst_name not in dst_map:
            continue
        src_list.append(src_map[src_name])
        dst_list.append(dst_map[dst_name])
        attr_list.append(attrs)

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long), []

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index, attr_list


def _compute_npmi_edges(
    parsed: ParsedGraph,
    min_npmi: float = 0.1,
    min_cooccurrence: int = 5,
) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, dict]]]:
    """Compute NPMI co-occurrence edges between ingredients and between hops.

    This is a key insight from FlavorDiffusion (Seo et al., 2025):
    Ingredient-ingredient co-occurrence edges encode which ingredients
    naturally complement each other across recipes.

    Returns:
        (ingredient_cooccur_edges, hop_cooccur_edges)
    """
    # Build co-occurrence from style-level edges
    # For ingredients: count how many styles use both ingredient A and B
    style_ingredients: dict[str, set[str]] = defaultdict(set)
    style_hops: dict[str, set[str]] = defaultdict(set)

    for style, ingredient, _ in parsed.uses_grain:
        style_ingredients[style].add(ingredient)
    for style, hop, _ in parsed.uses_hop:
        style_hops[style].add(hop)

    total_styles = len(parsed.beer_styles)

    def compute_npmi_pairs(
        style_items: dict[str, set[str]],
    ) -> list[tuple[str, str, dict]]:
        # Count individual and pair frequencies
        item_count: dict[str, int] = defaultdict(int)
        pair_count: dict[tuple[str, str], int] = defaultdict(int)

        for items in style_items.values():
            items_list = sorted(items)
            for item in items_list:
                item_count[item] += 1
            for i, a in enumerate(items_list):
                for b in items_list[i + 1 :]:
                    pair_count[(a, b)] += 1

        edges: list[tuple[str, str, dict]] = []
        for (a, b), count in pair_count.items():
            if count < min_cooccurrence:
                continue
            p_ab = count / total_styles
            p_a = item_count[a] / total_styles
            p_b = item_count[b] / total_styles

            if p_a == 0 or p_b == 0 or p_ab == 0:
                continue

            pmi = math.log(p_ab / (p_a * p_b))
            npmi = pmi / (-math.log(p_ab))

            if npmi >= min_npmi:
                edges.append((a, b, {"npmi": round(npmi, 4)}))

        return edges

    ingredient_edges = compute_npmi_pairs(style_ingredients)
    hop_edges = compute_npmi_pairs(style_hops)

    logger.info(
        "NPMI co-occurrence: %d ingredient pairs, %d hop pairs",
        len(ingredient_edges),
        len(hop_edges),
    )
    return ingredient_edges, hop_edges


def build_graph(parsed: ParsedGraph | None = None) -> HeteroData:
    """Build the full Heterogeneous Graph as PyG HeteroData.

    Returns a HeteroData object with all node types, features,
    and edge indices populated.
    """
    if parsed is None:
        parsed = parse_json()

    data = HeteroData()

    # ── 1. Beer Style nodes (Synced with global registry) ──
    import json
    registry_path = PROJECT_ROOT / "src" / "brewfusion" / "data" / "style_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        style_registry = json.load(f)
    
    # Pre-sort style names based on their pre-computed integer indices in the registry
    style_names = [k for k, v in sorted(style_registry.items(), key=lambda item: item[1])]
    style_map = _build_index_map(style_names)

    style_features = []
    for name in style_names:
        feat = parsed.beer_styles.get(name, {})
        style_features.append([feat.get(col, 0.0) for col in BEER_NUMERIC_FEATURES])

    data[NODE_BEER_STYLE].x = torch.tensor(style_features, dtype=torch.float32)
    data[NODE_BEER_STYLE].names = style_names
    logger.info("Beer style nodes: %d", len(style_names))

    # ── 2. Ingredient nodes ──
    ingredient_names = sorted(parsed.ingredients.keys())
    ingredient_map = _build_index_map(ingredient_names)

    ingredient_features = []
    for name in ingredient_names:
        feat = parsed.ingredients[name]
        ingredient_features.append([feat.get("ppg", 0.0)])

    data[NODE_INGREDIENT].x = torch.tensor(ingredient_features, dtype=torch.float32)
    data[NODE_INGREDIENT].names = ingredient_names
    logger.info("Ingredient nodes: %d", len(ingredient_names))

    # ── 3. Hop nodes ──
    hop_names = sorted(parsed.hops.keys())
    hop_map = _build_index_map(hop_names)

    hop_features = []
    for name in hop_names:
        feat = parsed.hops[name]
        hop_features.append([feat.get("alpha_acid", 0.0)])

    data[NODE_HOP].x = torch.tensor(hop_features, dtype=torch.float32)
    data[NODE_HOP].names = hop_names
    logger.info("Hop nodes: %d", len(hop_names))

    # ── 4. Yeast nodes ──
    yeast_names = sorted(parsed.yeasts.keys())
    yeast_map = _build_index_map(yeast_names)

    yeast_features = []
    for name in yeast_names:
        feat = parsed.yeasts[name]
        yeast_features.append(
            [
                feat.get("attenuation", 75.0),
                feat.get("temp_min", 60.0),
                feat.get("temp_max", 72.0),
            ]
        )

    data[NODE_YEAST].x = torch.tensor(yeast_features, dtype=torch.float32)
    data[NODE_YEAST].names = yeast_names
    logger.info("Yeast nodes: %d", len(yeast_names))

    # ── 5. Compound nodes (from curated DB) ──
    seen_cids: set[int] = set()
    unique_compounds: dict[str, Any] = {}
    for key, info in COMPOUNDS.items():
        if info.cid not in seen_cids:
            seen_cids.add(info.cid)
            unique_compounds[key] = info

    compound_names = sorted(unique_compounds.keys())
    compound_map = _build_index_map(compound_names)

    compound_features = []
    valid_compounds: list[str] = []
    compound_fingerprints: dict[str, np.ndarray] = {}

    for name in compound_names:
        info = unique_compounds[name]
        feat_vec = compute_compound_features(info.smiles)
        if feat_vec is not None:
            compound_features.append(feat_vec)
            valid_compounds.append(name)
            compound_fingerprints[name] = feat_vec
        else:
            logger.warning("Skipping compound %s (invalid SMILES)", name)

    compound_names = valid_compounds
    compound_map = _build_index_map(compound_names)

    if compound_features:
        data[NODE_COMPOUND].x = torch.tensor(
            np.stack(compound_features), dtype=torch.float32
        )
    else:
        data[NODE_COMPOUND].x = torch.zeros(0, 1030, dtype=torch.float32)

    data[NODE_COMPOUND].names = compound_names
    logger.info("Compound nodes: %d", len(compound_names))

    # ── 6. Beer ↔ Ingredient/Hop/Yeast edges ──
    ei, _ = _build_edge_index(parsed.uses_grain, style_map, ingredient_map)
    data[EDGE_USES_GRAIN].edge_index = ei
    logger.info("Style→Grain edges: %d", ei.shape[1])

    ei, _ = _build_edge_index(parsed.uses_hop, style_map, hop_map)
    data[EDGE_USES_HOP].edge_index = ei
    logger.info("Style→Hop edges: %d", ei.shape[1])

    ei, _ = _build_edge_index(parsed.uses_yeast, style_map, yeast_map)
    data[EDGE_USES_YEAST].edge_index = ei

    ei, _ = _build_edge_index(parsed.uses_adjunct, style_map, ingredient_map)
    data[EDGE_USES_ADJUNCT].edge_index = ei

    # ── 7. Ingredient/Hop/Yeast → Compound edges ──
    ingredient_compound_edges: list[tuple[str, str, dict]] = []
    for grain_name, compounds_list in GRAIN_COMPOUND_MAP.items():
        if grain_name in ingredient_map:
            for comp_key in compounds_list:
                if comp_key in compound_map:
                    ingredient_compound_edges.append((grain_name, comp_key, {}))
    for adj_name, compounds_list in ADJUNCT_COMPOUND_MAP.items():
        if adj_name in ingredient_map:
            for comp_key in compounds_list:
                if comp_key in compound_map:
                    ingredient_compound_edges.append((adj_name, comp_key, {}))

    ei, _ = _build_edge_index(ingredient_compound_edges, ingredient_map, compound_map)
    data[EDGE_INGREDIENT_CONTAINS].edge_index = ei
    logger.info("Ingredient→Compound edges: %d", ei.shape[1])

    hop_compound_edges: list[tuple[str, str, dict]] = []
    for hop_name, compounds_list in HOP_COMPOUND_MAP.items():
        if hop_name in hop_map:
            for comp_key in compounds_list:
                if comp_key in compound_map:
                    hop_compound_edges.append((hop_name, comp_key, {}))

    ei, _ = _build_edge_index(hop_compound_edges, hop_map, compound_map)
    data[EDGE_HOP_CONTAINS].edge_index = ei
    logger.info("Hop→Compound edges: %d", ei.shape[1])

    yeast_compound_edges: list[tuple[str, str, dict]] = []
    for yeast_name in yeast_names:
        family = classify_yeast_family(yeast_name)
        compounds_list = YEAST_COMPOUND_MAP.get(family, [])
        for comp_key in compounds_list:
            if comp_key in compound_map:
                yeast_compound_edges.append((yeast_name, comp_key, {}))

    ei, _ = _build_edge_index(yeast_compound_edges, yeast_map, compound_map)
    data[EDGE_YEAST_PRODUCES].edge_index = ei
    logger.info("Yeast→Compound edges: %d", ei.shape[1])

    # ── 8. Compound ↔ Compound similarity edges ──
    sim_edges_raw = compute_similarity_edges(compound_fingerprints)
    sim_edges: list[tuple[str, str, dict]] = [
        (a, b, {"similarity": s}) for a, b, s in sim_edges_raw
    ]
    sim_edges_bidir = sim_edges + [(b, a, attrs) for a, b, attrs in sim_edges]
    ei, _ = _build_edge_index(sim_edges_bidir, compound_map, compound_map)
    data[EDGE_COMPOUND_SIMILAR].edge_index = ei
    logger.info("Compound↔Compound similarity edges: %d", ei.shape[1])

    # ── 9. NEW: NPMI co-occurrence edges (FlavorDiffusion insight) ──
    ingr_cooccur, hop_cooccur = _compute_npmi_edges(parsed)

    # Make bidirectional
    ingr_cooccur_bidir = ingr_cooccur + [(b, a, attrs) for a, b, attrs in ingr_cooccur]
    ei, _ = _build_edge_index(ingr_cooccur_bidir, ingredient_map, ingredient_map)
    data[EDGE_INGREDIENT_COOCCURS].edge_index = ei
    logger.info("Ingredient↔Ingredient NPMI edges: %d", ei.shape[1])

    hop_cooccur_bidir = hop_cooccur + [(b, a, attrs) for a, b, attrs in hop_cooccur]
    ei, _ = _build_edge_index(hop_cooccur_bidir, hop_map, hop_map)
    data[EDGE_HOP_COOCCURS].edge_index = ei
    logger.info("Hop↔Hop NPMI edges: %d", ei.shape[1])

    return data


def save_graph(data: HeteroData, path: str | Path | None = None) -> Path:
    """Save HeteroData to disk as .pt file."""
    if path is None:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        path = GRAPH_DIR / "brewfusion_hetero_graph.pt"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, path)
    logger.info("Saved graph to %s", path)
    return path


def load_graph(path: str | Path | None = None) -> HeteroData:
    """Load HeteroData from disk."""
    if path is None:
        path = GRAPH_DIR / "brewfusion_hetero_graph.pt"
    data = torch.load(path, weights_only=False)
    logger.info("Loaded graph from %s", path)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph = build_graph()
    print("\n=== BrewFusion Heterogeneous Graph ===")
    print(graph)
    save_graph(graph)
    print("\n✅ Graph built and saved!")
