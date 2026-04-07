import torch
import torch.nn as nn


class CSPLayer(nn.Module):
    """Chemical Structure Prediction Layer.

    Acts as an auxiliary task to constrain node embeddings so that
    they encode the underlying molecular structure (Morgan Fingerprint)
    of the node.
    """

    def __init__(self, hidden_dim: int, fingerprint_size: int = 1024):
        super().__init__()
        self.predictor = nn.Linear(hidden_dim, fingerprint_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Outputs unnormalized logits for the fingerprint dimensions."""
        return self.predictor(embedding)

    def compute_loss(
        self, embedding: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Computes Binary Cross Entropy loss against target fingerprints.

        Args:
            embedding: Graph node embedding [N, hidden_dim]
            target_features: Native compound features where the LAST
                `fingerprint_size` elements are the binary Morgan FP.
        """
        # Ensure we only take the fingerprint portion (assumes 1024-bit Morgan is at the end)
        fp_target = target_features[:, -1024:]

        # Predict logits
        fp_logits = self.forward(embedding)

        # Filter down strictly to binary values (0 or 1) checking
        # just in case since descriptors might be floats
        # BCEWithLogitsLoss expects probabilities or {0, 1}
        return self.loss_fn(fp_logits, fp_target)
