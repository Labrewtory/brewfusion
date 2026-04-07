import pytest
from torch_geometric.data import HeteroData
from brewfusion.data.json_parser import ParsedGraph
from brewfusion.graph.builder import build_graph

def dummy_parse_json(*args, **kwargs):
    graph = ParsedGraph()
    graph.beer_styles["American Stout"] = {"abv": 5.0, "recipe_count": 1}
    graph.ingredients["test malt"] = {"ppg_mean": 35.0, "usage_mean": 1.0}
    graph.hops["test hop"] = {"alpha_mean": 5.0, "usage_mean": 1.0}
    graph.yeasts["test yeast"] = {}
    
    graph.uses_grain.append(("American Stout", "test malt", {"weight": 1.0}))
    graph.uses_hop.append(("American Stout", "test hop", {"weight": 1.0}))
    graph.uses_yeast.append(("American Stout", "test yeast", {"weight": 1.0}))
    return graph

def test_build_graph(monkeypatch):
    # Patch the parse_json call inside builder module so we don't need real JSON
    monkeypatch.setattr("brewfusion.graph.builder.parse_json", dummy_parse_json)
    
    # We also don't want to compute expensive Morgan Fingerprints for all PubChem compounds.
    # So we patch the compound list builder or just let it run if it's fast (it computes 120 features, takes <1 sec).
    # Since we want it to be fast, we can mock the compute_compound_features using monkeypatch.
    import numpy as np
    dummy_features = np.zeros(1030, dtype=np.float32)
    monkeypatch.setattr("brewfusion.graph.builder.compute_compound_features", lambda sm: dummy_features)
    
    data = build_graph()
    
    assert isinstance(data, HeteroData)
    
    # Check node types exist
    assert "beer_style" in data.node_types
    assert "ingredient" in data.node_types
    assert "hop" in data.node_types
    assert "yeast" in data.node_types
    assert "compound" in data.node_types
    
    # Check edge types exist
    edge_types = [et[1] for et in data.edge_types]
    assert "uses_grain" in edge_types
    assert "uses_hop" in edge_types
    assert "uses_yeast" in edge_types
    
    # Verify the test style is in there
    assert "American Stout" in getattr(data["beer_style"], "names", [])
