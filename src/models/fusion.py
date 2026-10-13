"""
fusion.py  —  Cross-attention fusion of sequence + structure representations.

H_seq   (B, L, 1280)  ESM-2 embeddings (queries)
H_struct (B, L, 512)  GVP-GNN outputs  (keys + values)
→  H_fused (B, L, 512)
"""

import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention: sequence representation attends to structural context.
    Each residue's ESM-2 embedding is enriched by the 3D neighborhood
    captured by GVP-GNN.

    Q = proj(H_seq)
    K = proj(H_struct)
    V = proj(H_struct)
    H_fused = LayerNorm(Attention(Q, K, V) + Q)   ← residual
    """

    def __init__(
        self,
        d_seq:    int = 1280,
        d_struct: int = 512,
        d_model:  int = 512,
        n_heads:  int = 8,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_seq,    d_model)
        self.k_proj = nn.Linear(d_struct, d_model)
        self.v_proj = nn.Linear(d_struct, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop_rate)

    def forward(
        self,
        h_seq:    Tensor,              # (B, L, d_seq)
        h_struct: Tensor,              # (B, L, d_struct)
        key_padding_mask: Tensor | None = None,   # (B, L) True = IGNORE
    ) -> Tensor:
        """
        Returns H_fused: (B, L, d_model)
        """
        Q = self.q_proj(h_seq)         # (B, L_seq, d_model)
        K = self.k_proj(h_struct)      # (B, L_struct, d_model)
        V = self.v_proj(h_struct)      # (B, L_struct, d_model)

        if key_padding_mask is not None:
            L_k = K.shape[1]
            if key_padding_mask.shape[1] > L_k:
                key_padding_mask = key_padding_mask[:, :L_k]
            elif key_padding_mask.shape[1] < L_k:
                pad = torch.zeros(key_padding_mask.shape[0], L_k - key_padding_mask.shape[1], dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat([key_padding_mask, pad], dim=1)

        attn_out, _ = self.attn(
            Q, K, V,
            key_padding_mask=key_padding_mask,
        )                              # (B, L_seq, d_model)

        # Residual + norm
        fused = self.norm(Q + self.drop(attn_out))
        return fused                   # (B, L, d_model)
