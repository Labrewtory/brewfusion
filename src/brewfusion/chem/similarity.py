"""Tanimoto similarity computation between compound fingerprints."""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from brewfusion.config import MORGAN_FP_NBITS, TANIMOTO_THRESHOLD

logger = logging.getLogger(__name__)


def tanimoto_similarity(fp1: NDArray[np.float32], fp2: NDArray[np.float32]) -> float:
    """Compute Tanimoto similarity between two binary fingerprint arrays."""
    if len(fp1) > MORGAN_FP_NBITS:
        fp1 = fp1[-MORGAN_FP_NBITS:]
    if len(fp2) > MORGAN_FP_NBITS:
        fp2 = fp2[-MORGAN_FP_NBITS:]

    intersection = float(np.sum(np.minimum(fp1, fp2)))
    union = float(np.sum(np.maximum(fp1, fp2)))
    if union == 0.0:
        return 0.0
    return intersection / union


def compute_similarity_edges(
    compound_fingerprints: dict[str, NDArray[np.float32]],
    threshold: float = TANIMOTO_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Compute all pairwise Tanimoto similarities above threshold."""
    names = list(compound_fingerprints.keys())
    fps = [compound_fingerprints[n] for n in names]
    edges: list[tuple[str, str, float]] = []

    for (i, name_a), (j, name_b) in combinations(enumerate(names), 2):
        sim = tanimoto_similarity(fps[i], fps[j])
        if sim >= threshold:
            edges.append((name_a, name_b, sim))

    logger.info(
        "Found %d similar compound pairs (threshold=%.2f) from %d compounds",
        len(edges),
        threshold,
        len(names),
    )
    return edges
