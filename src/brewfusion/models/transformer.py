import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x


class BrewFusion(nn.Module):
    """
    Auto-regressive Transformer (Decoder-only) for Brewing Sequence Generation.
    Takes tokenized sequence arrays representing physical/thermodynamic brewing steps.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Decoder layer (causal auto-regressive)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Final projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len] with token IDs.
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        # Create causal mask ensuring we can only look at previous tokens
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Embed and scale
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model] for PosEncoding
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]

        # Transform (auto-regressive causality enforced by mask)
        out = self.transformer(x, mask=mask, is_causal=True)

        # Project back to vocabulary logits
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Auto-regressively generates new tokens given a prompt.
        Args:
            prompt: Tensor of shape [1, seq_len] with initial token IDs.
        Returns:
            Tensor of shape [1, seq_len + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Limit the sequence to the maximum positional encoding length (though our data is short, safety first)
            # Take only the last sequence if it gets too long, though we don't have block_size enforced here
            logits = self(prompt)
            # Take logits for the last step
            next_token_logits = logits[:, -1, :] / temperature
            # Greedily sample for simplicity, or use multinomial
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat((prompt, next_token), dim=1)
        self.train()
        return prompt


def test_model():
    # Simple sanity check
    vocab_size = 985
    model = BrewFusion(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2)
    dummy_input = torch.randint(0, vocab_size, (2, 50))  # Batch=2, Seq=50
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape} -> Expected: (2, 50, 985)")


if __name__ == "__main__":
    test_model()
