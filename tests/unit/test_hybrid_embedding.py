import torch
from unittest.mock import patch, MagicMock
from brewfusion.models.hybrid_embedding import HybridTokenEmbedding


@patch("tokenizers.Tokenizer.from_file")
def test_hybrid_token_embedding(
    mock_from_file, mock_gnn_embeddings, mock_ingredient_registry
):
    gnn_path, gnn_tensor = mock_gnn_embeddings
    registry_path, registry = mock_ingredient_registry

    # Mock Tokenizer map
    mock_tok = MagicMock()
    mock_tok.token_to_id.side_effect = lambda x: {"cascade": 2, "citra": 3}.get(x, None)
    mock_from_file.return_value = mock_tok

    vocab_size = 10
    d_model = 192
    gnn_dim = 64

    # Initialize embedding
    emb = HybridTokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        gnn_emb_path=str(gnn_path),
        registry_path=str(registry_path),
        gnn_dim=gnn_dim,
    )

    assert emb.gnn_bank is not None
    assert emb.gnn_bank.shape[1] == gnn_dim, "Frozen GNN buffer should match GNN dim"

    # Check forward pass shapes
    batch_size = 4
    seq_len = 8

    # Random tokens (mix of everything)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = emb(x)

    assert out.shape == (batch_size, seq_len, d_model), (
        "Expected (B, S, D) output shape"
    )

    # Verify gradients flow only to allowed parameters
    loss = out.sum()
    loss.backward()

    assert emb.learned_emb.weight.grad is not None, (
        "Structural embedding needs gradients"
    )
    assert emb.gnn_proj[0].weight.grad is not None, "Projection layer needs gradients"

    # GNN embeddings must be frozen, no grad
    assert emb.gnn_bank.grad is None, "GNN buffer MUST be strictly frozen!"
