"""
gvp_features.py
---------------
GVP (Geometric Vector Perceptron) feature preparation utilities.

Converts raw contact graph data (from structure_builder.py) into
the batched format expected by the GVP-GNN encoder.

Also provides the GVPConv layer implementation based on:
  Jing et al., "Learning from Protein Structure with Geometric Vector Perceptrons"
  ICLR 2021. https://arxiv.org/abs/2009.01411

Usage:
    from src.features.gvp_features import load_graph, GVPConv
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)

GRAPH_DIR = Path("data/processed/contact_graphs")


# ---------------------------------------------------------------------------
# Load a pre-built graph from disk
# ---------------------------------------------------------------------------
def load_graph(pdb_id: str, graph_dir: Path = GRAPH_DIR) -> dict:
    """
    Loads a pre-built contact graph .pt file.
    Returns dict with keys: node_scalars, node_vectors, edge_index, edge_scalars, seq_length.
    """
    path = graph_dir / f"{pdb_id}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found: {path}\n"
            "Run structure_builder.py first."
        )
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# GVP layer — equivariant message passing
# ---------------------------------------------------------------------------
class GVPLayer(nn.Module):
    """
    Single Geometric Vector Perceptron layer.

    Processes both scalar (invariant) and vector (equivariant) features.

    Args:
        in_scalars:   input scalar feature dim
        in_vectors:   input vector feature count
        out_scalars:  output scalar feature dim
        out_vectors:  output vector feature count
        activations:  (scalar_act, vector_act) — defaults (SiLU, sigmoid)
    """
    def __init__(
        self,
        in_scalars:  int,
        in_vectors:  int,
        out_scalars: int,
        out_vectors: int,
        activations: tuple = (nn.SiLU(), torch.sigmoid),
    ) -> None:
        super().__init__()
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.scalar_act, self.vector_act = activations

        # Vector pathway — linear on vector features
        self.W_h = nn.Linear(in_vectors, out_vectors + out_scalars, bias=False)
        # Scalar pathway — linear on scalars + norm of vector output
        self.W_s = nn.Linear(in_scalars + out_vectors + out_scalars, out_scalars)

    def forward(self, scalars: Tensor, vectors: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            scalars: (N, in_scalars)
            vectors: (N, in_vectors, 3)
        Returns:
            out_scalars: (N, out_scalars)
            out_vectors: (N, out_vectors, 3)
        """
        # Vector pathway
        # W_h applied to each vector independently: (N, in_vectors, 3) → (N, out_h, 3)
        vh = torch.einsum("nid,oi->nod", vectors, self.W_h.weight)  # (N, out_h, 3)
        # Split into vector outputs and scalar-contributing vectors
        vv = vh[:, :self.out_vectors, :]                             # (N, out_vectors, 3)
        vs = vh[:, self.out_vectors:, :]                             # (N, out_scalars, 3)

        # Norm of vs to feed into scalar pathway (rotationally invariant)
        vs_norm = torch.norm(vs, dim=-1)                             # (N, out_scalars)

        # Scalar pathway
        cat = torch.cat([scalars, vs_norm, vs_norm], dim=-1)         # (N, in_s + 2*out_s)
        # Trim to expected input size
        cat = cat[:, :self.W_s.in_features]
        s_out = self.scalar_act(self.W_s(cat))                       # (N, out_scalars)

        # Vector activation (gate equivariant outputs)
        if self.vector_act is not None:
            gate = self.vector_act(s_out[:, :self.out_vectors]).unsqueeze(-1)
            vv   = vv * gate

        return s_out, vv


# ---------------------------------------------------------------------------
# GVP-GNN Message Passing Layer
# ---------------------------------------------------------------------------
class GVPConv(nn.Module):
    """
    GVP message passing: aggregate neighbour scalar + vector features
    and update node representations via GVP layers.

    Args:
        node_s_dim:   node scalar feature dim
        node_v_dim:   node vector feature count
        edge_s_dim:   edge scalar feature dim
        out_s_dim:    output node scalar dim
        out_v_dim:    output node vector count
        drop_rate:    dropout on scalar features
    """
    def __init__(
        self,
        node_s_dim: int,
        node_v_dim: int,
        edge_s_dim: int,
        out_s_dim:  int,
        out_v_dim:  int,
        drop_rate:  float = 0.1,
    ) -> None:
        super().__init__()

        # Message GVP: combines src node + edge features
        msg_in_s = node_s_dim + edge_s_dim
        msg_in_v = node_v_dim
        self.message_gvp = GVPLayer(msg_in_s, msg_in_v, out_s_dim, out_v_dim)

        # Update GVP: combines node features + aggregated messages
        upd_in_s = node_s_dim + out_s_dim
        upd_in_v = node_v_dim + out_v_dim
        self.update_gvp = GVPLayer(upd_in_s, upd_in_v, out_s_dim, out_v_dim)

        self.drop = nn.Dropout(drop_rate)
        self.norm_s = nn.LayerNorm(out_s_dim)

    def forward(
        self,
        node_scalars: Tensor,    # (N, node_s_dim)
        node_vectors: Tensor,    # (N, node_v_dim, 3)
        edge_index:   Tensor,    # (2, E) — [src, dst]
        edge_scalars: Tensor,    # (E, edge_s_dim)
    ) -> tuple[Tensor, Tensor]:
        """
        Returns updated (node_scalars, node_vectors) of same shape as input
        but with out_s_dim, out_v_dim dims.
        """
        N = node_scalars.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Gather source node features for each edge
        src_s = node_scalars[src]    # (E, node_s_dim)
        src_v = node_vectors[src]    # (E, node_v_dim, 3)

        # Build message inputs
        msg_s_in = torch.cat([src_s, edge_scalars], dim=-1)   # (E, node_s_dim + edge_s_dim)
        msg_s, msg_v = self.message_gvp(msg_s_in, src_v)      # (E, out_s), (E, out_v, 3)

        # Aggregate messages at destination nodes (mean aggregation)
        agg_s = torch.zeros(N, msg_s.shape[-1], device=msg_s.device)
        agg_v = torch.zeros(N, msg_v.shape[1], 3, device=msg_v.device)
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_s), msg_s)
        agg_v.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg_v), msg_v)

        # Count neighbours for mean
        count = torch.zeros(N, 1, device=node_scalars.device)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=dst.device))
        count = count.clamp(min=1)
        agg_s = agg_s / count
        agg_v = agg_v / count.unsqueeze(-1)

        # Update: concat aggregated with original node features
        upd_s_in = torch.cat([node_scalars, agg_s], dim=-1)
        upd_v_in = torch.cat([node_vectors, agg_v], dim=1)
        out_s, out_v = self.update_gvp(upd_s_in, upd_v_in)

        out_s = self.norm_s(self.drop(out_s))
        return out_s, out_v
