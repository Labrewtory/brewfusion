"""Node and edge type schema definitions for BrewFusion."""

from __future__ import annotations

from brewfusion.config import (
    ALL_EDGE_TYPES,
    NODE_BEER_STYLE,
    NODE_COMPOUND,
    NODE_HOP,
    NODE_INGREDIENT,
    NODE_YEAST,
)

ALL_NODE_TYPES = [
    NODE_BEER_STYLE,
    NODE_INGREDIENT,
    NODE_HOP,
    NODE_YEAST,
    NODE_COMPOUND,
]

# Feature dimensions (after encoding)
NODE_FEATURE_DIMS = {
    NODE_BEER_STYLE: 5,  # og, fg, abv, ibu, color
    NODE_INGREDIENT: 1,  # ppg
    NODE_HOP: 1,  # alpha_acid
    NODE_YEAST: 3,  # attenuation, temp_min, temp_max
    NODE_COMPOUND: 1030,  # 6 descriptors + 1024 fingerprint bits
}

NODE_TYPE_DESCRIPTIONS = {
    NODE_BEER_STYLE: "Beer Style (cluster of recipes)",
    NODE_INGREDIENT: "Ingredient (grain, adjunct)",
    NODE_HOP: "Hop variety",
    NODE_YEAST: "Yeast strain",
    NODE_COMPOUND: "Chemical compound (flavor molecule)",
}

EDGE_TYPE_DESCRIPTIONS = {
    edge: f"{edge[0]} --[{edge[1]}]--> {edge[2]}" for edge in ALL_EDGE_TYPES
}
