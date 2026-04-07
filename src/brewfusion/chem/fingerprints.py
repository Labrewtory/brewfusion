"""Molecular fingerprint and descriptor computation via RDKit."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from brewfusion.config import MORGAN_FP_NBITS, MORGAN_FP_RADIUS

logger = logging.getLogger(__name__)

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_FP_RADIUS, fpSize=MORGAN_FP_NBITS
)


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES: %s", smiles)
    return mol


def compute_morgan_fingerprint(smiles: str) -> NDArray[np.float32] | None:
    """Compute a Morgan fingerprint as a float32 numpy array.

    Returns None if SMILES is invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros(MORGAN_FP_NBITS, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def compute_descriptors(smiles: str) -> dict[str, float] | None:
    """Compute common molecular descriptors.

    Returns dict with keys: MolWt, LogP, TPSA, HBD, HBA, NumRotBonds.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return {
        "MolWt": float(getattr(Descriptors, "MolWt")(mol)),
        "LogP": float(getattr(Descriptors, "MolLogP")(mol)),
        "TPSA": float(getattr(Descriptors, "TPSA")(mol)),
        "HBD": float(getattr(Descriptors, "NumHDonors")(mol)),
        "HBA": float(getattr(Descriptors, "NumHAcceptors")(mol)),
        "NumRotBonds": float(getattr(Descriptors, "NumRotatableBonds")(mol)),
    }


def compute_compound_features(
    smiles: str,
) -> NDArray[np.float32] | None:
    """Compute a combined feature vector: [descriptors (6) | fingerprint (1024)].

    Total length = 1030.
    """
    desc = compute_descriptors(smiles)
    fp = compute_morgan_fingerprint(smiles)
    if desc is None or fp is None:
        return None
    desc_arr = np.array(
        [
            desc["MolWt"],
            desc["LogP"],
            desc["TPSA"],
            desc["HBD"],
            desc["HBA"],
            desc["NumRotBonds"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([desc_arr, fp])
