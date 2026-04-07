import numpy as np
import pytest
from brewfusion.chem.fingerprints import compute_compound_features, smiles_to_mol
from brewfusion.chem.similarity import compute_similarity_edges
from brewfusion.config import MORGAN_FP_NBITS

def test_smiles_to_mol():
    mol = smiles_to_mol("CCO") # Ethanol
    assert mol is not None
    
    mol_invalid = smiles_to_mol("INVALID_SMILES")
    assert mol_invalid is None

def test_compute_compound_features():
    feat = compute_compound_features("CCO")
    assert feat is not None
    assert isinstance(feat, np.ndarray)
    # Features = 6 descriptors + 1024 morgan fp
    assert len(feat) == 6 + MORGAN_FP_NBITS
    
    feat_invalid = compute_compound_features("INVALID")
    assert feat_invalid is None

def test_compute_similarity_edges():
    # Make dummy feature dictionary
    features = {
        "A": np.zeros(10, dtype=np.float32),
        "B": np.zeros(10, dtype=np.float32),
        "C": np.zeros(10, dtype=np.float32),
    }
    # A and B are identical
    features["A"][0] = 1.0
    features["B"][0] = 1.0
    # C is completely different
    features["C"][5] = 1.0
    
    # Threshold 0.9. A and B have sim 1.0. A-C and B-C have sim 0.0.
    edges = compute_similarity_edges(features, threshold=0.9)
    
    # combinations returns A-B only
    assert len(edges) == 1
    assert ("A", "B") in [(e[0], e[1]) for e in edges] or ("B", "A") in [(e[0], e[1]) for e in edges]
    
    # Weight should be roughly 1.0
    for u, v, w in edges:
        assert np.isclose(w, 1.0)
