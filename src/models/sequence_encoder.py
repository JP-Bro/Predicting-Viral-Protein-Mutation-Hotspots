"""
sequence_encoder.py
-------------------
ESM-2 wrapper that loads cached embeddings from disk.
ESM-2 is FULLY FROZEN — never runs during training.
Embeddings were pre-extracted by esm2_embed.py.
"""

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

EMB_DIR = Path("data/processed/esm2_embeddings")


class SequenceEncoder(nn.Module):
    """
    Loads pre-computed ESM-2 embeddings from disk.
    Returns (L, D) tensors per sequence — no forward pass through ESM-2 during training.
    """

    def __init__(self, emb_dir: Path = EMB_DIR) -> None:
        super().__init__()
        self.emb_dir = Path(emb_dir)
        self._cache: dict[str, torch.Tensor] = {}

    def load(self, strain_id: str) -> torch.Tensor:
        """Load embedding for a single strain. Caches in memory after first load."""
        if strain_id in self._cache:
            return self._cache[strain_id]

        strain_id_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", strain_id)
        path = self.emb_dir / f"{strain_id_clean}.pt"
        if not path.exists():
            return torch.zeros((1273, 480))

        try:
            data = torch.load(path, weights_only=True)
            emb = data["embedding"] if isinstance(data, dict) else data   # (L, D)
            self._cache[strain_id] = emb
            return emb
        except Exception as e:
            log.warning(f"Could not load embedding {path} ({e}) — using zero fallback.")
            return torch.zeros((1273, 480))

    def forward(self, strain_ids: list[str], device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and batch embeddings for a list of strain IDs.
        Pads to the longest sequence in the batch.

        Returns:
            embeddings: (B, L_max, D) float32
            mask:       (B, L_max) bool — True where valid (not padding)
        """
        embs = [self.load(sid).to(device) for sid in strain_ids]
        L_max = max(e.shape[0] for e in embs)
        D     = embs[0].shape[1]
        B     = len(embs)

        padded = torch.zeros(B, L_max, D, device=device)
        mask   = torch.zeros(B, L_max, dtype=torch.bool, device=device)

        for i, emb in enumerate(embs):
            L = emb.shape[0]
            padded[i, :L] = emb
            mask[i, :L]   = True

        return padded, mask
