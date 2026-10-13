"""
structure_encoder.py
--------------------
GVP-GNN structural encoder.
Stacks 3 GVPConv layers to produce H_struct ∈ R^{L × 512}.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.features.gvp_features import GVPConv


class StructureEncoder(nn.Module):
    """
    GVP-GNN encoder over residue-contact graphs.

    Input:
        node_scalars: (L, node_s_in)   e.g. 10
        node_vectors: (L, node_v_in, 3) e.g. 3 vectors
        edge_index:   (2, E)
        edge_scalars: (E, edge_s_in)   e.g. 2

    Output:
        h_struct: (L, out_dim)          default 512
    """

    def __init__(
        self,
        node_s_in:  int = 10,
        node_v_in:  int = 3,
        edge_s_in:  int = 2,
        hidden_s:   int = 128,
        hidden_v:   int = 16,
        out_dim:    int = 512,
        num_layers: int = 3,
        drop_rate:  float = 0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        in_s, in_v = node_s_in, node_v_in
        for i in range(num_layers):
            out_s = hidden_s if i < num_layers - 1 else out_dim
            out_v = hidden_v if i < num_layers - 1 else 1
            self.layers.append(
                GVPConv(
                    node_s_dim=in_s,
                    node_v_dim=in_v,
                    edge_s_dim=edge_s_in,
                    out_s_dim=out_s,
                    out_v_dim=out_v,
                    drop_rate=drop_rate,
                )
            )
            in_s, in_v = out_s, out_v

        # Final projection to out_dim (scalar only; vectors collapsed)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.out_dim  = out_dim

    def forward(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_index:   Tensor,
        edge_scalars: Tensor,
    ) -> Tensor:
        s, v = node_scalars, node_vectors
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_scalars)
        return self.out_proj(s)   # (L, out_dim)
