import pytest
import torch
import json
import os


@pytest.fixture
def mock_data_dir(tmp_path):
    """Provides a temporary directory structure mimicking the real data/ repo."""
    data_dir = tmp_path / "data"

    # Create subdirs
    os.makedirs(data_dir / "models", exist_ok=True)
    os.makedirs(data_dir / "graph", exist_ok=True)
    os.makedirs(data_dir / "processed", exist_ok=True)

    return data_dir


@pytest.fixture
def mock_gnn_embeddings(mock_data_dir):
    """Creates a tiny 10-node 64D mock GNN embedding file."""
    emb = torch.randn(10, 64)
    # GNN creates dictionary of node types
    gnn_dict = {"ingredient": emb}
    file_path = mock_data_dir / "graph" / "gnn_embeddings.pt"
    torch.save(gnn_dict, file_path)
    return file_path, gnn_dict


@pytest.fixture
def mock_bpe_tokenizer_files(mock_data_dir):
    """Creates a dummy vocab and merges file."""
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "cascade": 2,
        "citra": 3,
        "[BOIL]": 4,
        "60": 5,
        "<MIN>": 6,
    }
    vocab_path = mock_data_dir / "processed" / "bpe_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    merges = ["c a", "s c"]
    merges_path = mock_data_dir / "processed" / "bpe_merges.txt"
    with open(merges_path, "w") as f:
        f.write("\n".join(merges))

    return vocab_path, merges_path


@pytest.fixture
def mock_ingredient_registry(mock_data_dir):
    """Creates a mock registry mapping token string -> GNN node ID."""
    registry = {"cascade": ("ingredient", 0), "citra": ("ingredient", 1)}
    file_path = mock_data_dir / "processed" / "ingredient_registry.pt"
    torch.save(registry, file_path)
    return file_path, registry


@pytest.fixture
def mock_compound_json(mock_data_dir):
    """Creates a tiny PubChem response mocked JSON."""
    compounds = {
        "100": {"name": "DummyCompound", "smiles": "CC", "molecular_weight": 10.0}
    }
    file_path = mock_data_dir / "chemistry" / "expanded_compounds.json"
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(compounds, f)
    return file_path
