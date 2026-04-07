import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class FeatureProjector(nn.Module):
    """Projects heterogeneous node features into a uniform hidden dimension."""

    def __init__(self, feature_dims: dict[str, int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict(
            {ntype: nn.Linear(dim, hidden_dim) for ntype, dim in feature_dims.items()}
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.projections:
                out_dict[ntype] = F.relu(self.projections[ntype](x.float()))
        return out_dict


class HeteroGNNEncoder(nn.Module):
    """Graph neural network for heterogeneous graphs.

    Uses HeteroConv with standard SAGEConv layers.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.out_dim = out_dim

        # 1. Input feature projection
        self.projector = FeatureProjector(feature_dims, hidden_dim)

        # 2. Heterogeneous Message Passing Layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {etype: SAGEConv((-1, -1), out_dim) for etype in edge_types}
            from typing import Any, cast

            # HeteroConv takes a dictionary of edge types to convolution layers
            # We use cast here because dict is invariant in Python typing
            self.convs.append(HeteroConv(cast(Any, conv_dict), aggr="sum"))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        # Project inputs to uniform size
        h_dict = self.projector(x_dict)

        # Message passing
        for idx, conv in enumerate(self.convs):
            out_dict = conv(h_dict, edge_index_dict)

            # HeteroConv might omit node types that didn't receive any messages.
            # We must restore them and add a residual connection to prevent representation collapse.
            for ntype in h_dict.keys():
                if ntype not in out_dict:
                    out_dict[ntype] = h_dict[ntype]  # Carry over representation
                else:
                    out_dict[ntype] = out_dict[ntype] + h_dict[ntype]  # Skip connection

            h_dict = out_dict

            if idx < len(self.convs) - 1:
                # Add non-linearity and dropout between layers
                h_dict = {
                    key: F.dropout(F.relu(h), p=0.1, training=self.training)
                    for key, h in h_dict.items()
                }

        return h_dict
