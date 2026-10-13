"""
joint_model.py
--------------
Combines structural GVP-GNN encoder, cross-attention fusion,
and dual prediction heads (HotspotHead + EscapeHead).

Sequence embeddings are pre-extracted by ESM-2 (cached on disk) and passed in directly.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.structure_encoder import StructureEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.heads import HotspotHead, EscapeHead


class JointMutationEscapeModel(nn.Module):
    """
    Multimodal DL framework coupling sequence language model features (ESM-2)
    and 3D structural graph features (GVP-GNN) via cross-attention.

    Inputs:
        h_seq:         (B, L, 1280) pre-cached ESM-2 sequence embeddings
        node_scalars:  (L, 10) or (B, L, 10) structural node scalars
        node_vectors:  (L, 3, 3) or (B, L, 3, 3) structural node vectors
        edge_index:    (2, E) contact graph edge indices
        edge_scalars:  (E, 2) contact graph edge scalar features
        key_padding_mask: (B, L) bool padding mask (True for padding positions)

    Outputs:
        p_mutate:      (B, L) hotspot mutation probability
        s_escape:      (B, L) quantitative DMS escape score
    """

    def __init__(
        self,
        d_seq: int = 1280,
        d_struct: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        num_gvp_layers: int = 3,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.structure_encoder = StructureEncoder(
            node_s_in=10,
            node_v_in=3,
            edge_s_in=2,
            out_dim=d_struct,
            num_layers=num_gvp_layers,
            drop_rate=drop_rate,
        )

        self.fusion = CrossAttentionFusion(
            d_seq=d_seq,
            d_struct=d_struct,
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
        )

        self.hotspot_head = HotspotHead(d_model=d_model, hidden=128)
        self.escape_head  = EscapeHead(d_model=d_model, hidden=128)

    def forward(
        self,
        h_seq: Tensor,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_index: Tensor,
        edge_scalars: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Runs structure encoder, fuses with cached sequence embeddings via cross-attention,
        and outputs dual predictions.
        """
        # If node_scalars is 2D (L, 10), add batch dimension to match h_seq (B, L, 1280)
        if node_scalars.dim() == 2:
            h_struct_single = self.structure_encoder(
                node_scalars, node_vectors, edge_index, edge_scalars
            )  # (L, d_struct)
            B = h_seq.shape[0]
            h_struct = h_struct_single.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_struct)
        else:
            # Batched structures handling if each item in batch has structural features
            B, L, _ = node_scalars.shape
            h_struct_list = []
            for b in range(B):
                h_s = self.structure_encoder(
                    node_scalars[b], node_vectors[b], edge_index[b], edge_scalars[b]
                )
                h_struct_list.append(h_s)
            h_struct = torch.stack(h_struct_list, dim=0)

        # Cross-attention fusion
        h_fused = self.fusion(h_seq, h_struct, key_padding_mask=key_padding_mask)

        # Dual heads
        p_mutate = self.hotspot_head(h_fused)
        s_escape = self.escape_head(h_fused)

        return p_mutate, s_escape
