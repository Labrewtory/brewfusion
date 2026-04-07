"""Global configuration constants for BrewFusion."""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"
MODEL_DIR = DATA_DIR / "models"

RECIPES_JSON = PROJECT_ROOT / "recipes_full.txt"

# ──────────────────────────────────────────────
# Node type names
# ──────────────────────────────────────────────
NODE_BEER_STYLE = "beer_style"
NODE_INGREDIENT = "ingredient"
NODE_HOP = "hop"
NODE_YEAST = "yeast"
NODE_COMPOUND = "compound"

# ──────────────────────────────────────────────
# Edge type names (src_type, relation, dst_type)
# ──────────────────────────────────────────────
EDGE_USES_GRAIN = (NODE_BEER_STYLE, "uses_grain", NODE_INGREDIENT)
EDGE_USES_HOP = (NODE_BEER_STYLE, "uses_hop", NODE_HOP)
EDGE_USES_YEAST = (NODE_BEER_STYLE, "uses_yeast", NODE_YEAST)
EDGE_USES_ADJUNCT = (NODE_BEER_STYLE, "uses_adjunct", NODE_INGREDIENT)
EDGE_HOP_CONTAINS = (NODE_HOP, "contains", NODE_COMPOUND)
EDGE_INGREDIENT_CONTAINS = (NODE_INGREDIENT, "contains", NODE_COMPOUND)
EDGE_YEAST_PRODUCES = (NODE_YEAST, "produces", NODE_COMPOUND)
EDGE_COMPOUND_SIMILAR = (NODE_COMPOUND, "similar_to", NODE_COMPOUND)
# NEW: Ingredient co-occurrence (FlavorDiffusion insight)
EDGE_INGREDIENT_COOCCURS = (NODE_INGREDIENT, "cooccurs_with", NODE_INGREDIENT)
EDGE_HOP_COOCCURS = (NODE_HOP, "cooccurs_with", NODE_HOP)

ALL_EDGE_TYPES = [
    EDGE_USES_GRAIN,
    EDGE_USES_HOP,
    EDGE_USES_YEAST,
    EDGE_USES_ADJUNCT,
    EDGE_HOP_CONTAINS,
    EDGE_INGREDIENT_CONTAINS,
    EDGE_YEAST_PRODUCES,
    EDGE_COMPOUND_SIMILAR,
    EDGE_INGREDIENT_COOCCURS,
    EDGE_HOP_COOCCURS,
]

# ──────────────────────────────────────────────
# Beer style clustering – numeric feature columns
# ──────────────────────────────────────────────
BEER_NUMERIC_FEATURES = ["og", "fg", "abv", "ibu", "color"]

# ──────────────────────────────────────────────
# Chemistry
# ──────────────────────────────────────────────
MORGAN_FP_RADIUS = 2
MORGAN_FP_NBITS = 1024
TANIMOTO_THRESHOLD = 0.4

# ──────────────────────────────────────────────
# DiT Model Hyperparameters (8GB GPU optimized)
# ──────────────────────────────────────────────
DIT_D_MODEL = 192
DIT_NHEAD = 6
DIT_NUM_LAYERS = 6
DIT_SEQ_LEN = 64
DIT_BATCH_SIZE = 16
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
