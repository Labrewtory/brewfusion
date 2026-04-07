import torch
from brewfusion.models.gnn_encoder import HeteroGNNEncoder

def test_hetero_gnn_encoder_skip_connection():
    feature_dims = {
        "ingredient": 10,
        "style": 10,  # Isolated node type in this specific mock graph
    }
    
    # We create an edge ONLY for ingredient-ingredient.
    # The 'style' node receives NO messages.
    edge_types = [("ingredient", "rel", "ingredient")]
    
    encoder = HeteroGNNEncoder(
        feature_dims=feature_dims,
        edge_types=edge_types,
        hidden_dim=16,
        out_dim=16,
        num_layers=1
    )
    
    x_dict = {
        "ingredient": torch.ones((2, 10)),
        "style": torch.ones((1, 10))
    }
    
    edge_index_dict = {
        ("ingredient", "rel", "ingredient"): torch.tensor([[0], [1]], dtype=torch.long)
    }
    
    out_dict = encoder(x_dict, edge_index_dict)
    
    assert "ingredient" in out_dict
    assert "style" in out_dict
    
    # Verify shape
    assert out_dict["ingredient"].shape == (2, 16)
    assert out_dict["style"].shape == (1, 16)
    
    # Verify it is not uniformly zero (thanks to the skip connection)
    assert not torch.allclose(out_dict["style"], torch.zeros_like(out_dict["style"]))
