import pytest
from torch_geometric.data import HeteroData
import torch
import sys

# Because it's a script, we import it directly after adding root to path
import sys
sys.path.insert(0, "/home/yjy20/brewery_v2")

from scripts.visualize_style_graph import build_style_subgraph

def dummy_load_heterodata():
    data = HeteroData()
    data["beer_style"].names = ["Test Style"]
    data["ingredient"].names = ["test malt"]
    data["hop"].names = ["test hop"]
    data["yeast"].names = ["test yeast"]
    data["compound"].names = ["test compound"]
    
    # We need to mock edges so top_k works without crashing
    data["beer_style", "uses_grain", "ingredient"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["beer_style", "uses_grain", "ingredient"].edge_weight = torch.tensor([1.0])
    
    data["beer_style", "uses_hop", "hop"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["beer_style", "uses_hop", "hop"].edge_weight = torch.tensor([1.0])
    
    data["beer_style", "uses_yeast", "yeast"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["beer_style", "uses_yeast", "yeast"].edge_weight = torch.tensor([1.0])
    
    data["ingredient", "contains", "compound"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    return data

def test_build_style_subgraph(monkeypatch):
    monkeypatch.setattr("scripts.visualize_style_graph._load_heterodata", dummy_load_heterodata)
    
    # Run the html builder on our dummy data
    html_output = build_style_subgraph("Test Style")
    
    # Basic sanity checks
    assert isinstance(html_output, str)
    assert "<html" in html_output.lower()
    assert "Test Style" in html_output
