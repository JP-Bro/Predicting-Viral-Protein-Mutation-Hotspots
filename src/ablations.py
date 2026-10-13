"""
ablations.py
------------
Runs the ablation study comparing 4 model variants:
  1. Seq-only:        ESM-2 embeddings + MLP head
  2. Struct-only:     GVP-GNN embeddings + MLP head
  3. Concat baseline: ESM-2 + GVP-GNN concatenated
  4. Full (Ours):     ESM-2 + GVP-GNN cross-attention fusion

Usage:
    python src/ablations.py --config_data configs/data.yaml --config_model configs/model.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.models.sequence_encoder import SequenceEncoder
from src.models.structure_encoder import StructureEncoder
from src.models.heads import HotspotHead, EscapeHead
from src.models.joint_model import JointMutationEscapeModel
from src.train import SpikeDataset, custom_collate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation Model Variants
# ---------------------------------------------------------------------------
class SeqOnlyModel(nn.Module):
    def __init__(self, d_seq: int = 1280, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_seq, d_model)
        self.hotspot_head = HotspotHead(d_model=d_model)
        self.escape_head  = EscapeHead(d_model=d_model)

    def forward(self, h_seq, node_scalars, node_vectors, edge_index, edge_scalars, key_padding_mask=None):
        h = torch.relu(self.proj(h_seq))
        return self.hotspot_head(h), self.escape_head(h)


class StructOnlyModel(nn.Module):
    def __init__(self, d_struct: int = 512):
        super().__init__()
        self.structure_encoder = StructureEncoder(out_dim=d_struct)
        self.hotspot_head = HotspotHead(d_model=d_struct)
        self.escape_head  = EscapeHead(d_model=d_struct)

    def forward(self, h_seq, node_scalars, node_vectors, edge_index, edge_scalars, key_padding_mask=None):
        h_struct_single = self.structure_encoder(node_scalars, node_vectors, edge_index, edge_scalars)
        B = h_seq.shape[0]
        h = h_struct_single.unsqueeze(0).expand(B, -1, -1)
        return self.hotspot_head(h), self.escape_head(h)


class ConcatModel(nn.Module):
    def __init__(self, d_seq: int = 1280, d_struct: int = 512, d_model: int = 512):
        super().__init__()
        self.structure_encoder = StructureEncoder(out_dim=d_struct)
        self.proj = nn.Linear(d_seq + d_struct, d_model)
        self.hotspot_head = HotspotHead(d_model=d_model)
        self.escape_head  = EscapeHead(d_model=d_model)

    def forward(self, h_seq, node_scalars, node_vectors, edge_index, edge_scalars, key_padding_mask=None):
        h_struct_single = self.structure_encoder(node_scalars, node_vectors, edge_index, edge_scalars)
        B, L_seq, _ = h_seq.shape
        L_struct = h_struct_single.shape[0]
        if L_struct < L_seq:
            pad = torch.zeros(L_seq - L_struct, h_struct_single.shape[1], device=h_seq.device)
            h_struct_single = torch.cat([h_struct_single, pad], dim=0)
        else:
            h_struct_single = h_struct_single[:L_seq]

        h_struct = h_struct_single.unsqueeze(0).expand(B, -1, -1)
        cat = torch.cat([h_seq, h_struct], dim=-1)
        h = torch.relu(self.proj(cat))
        return self.hotspot_head(h), self.escape_head(h)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_ablation_eval(variant_name: str, model: nn.Module, loader: DataLoader, node_scalars: torch.Tensor, node_vectors: torch.Tensor, edge_index: torch.Tensor, edge_scalars: torch.Tensor, device: str) -> dict:
    model.eval()
    all_s_escape, all_y_escape = [], []

    with torch.no_grad():
        for batch in loader:
            h_seq = batch["h_seq"]
            key_padding_mask = batch["key_padding_mask"]
            y_escape = batch["y_escape"]

            _, s_escape = model(h_seq, node_scalars, node_vectors, edge_index, edge_scalars, key_padding_mask=key_padding_mask)
            L_min = min(s_escape.shape[1], y_escape.shape[1])
            all_s_escape.append(s_escape[:, :L_min].cpu().numpy())
            all_y_escape.append(y_escape[:, :L_min].cpu().numpy())

    preds = np.concatenate(all_s_escape, axis=0) if all_s_escape else np.zeros((1, 1273))
    targets = np.concatenate(all_y_escape, axis=0) if all_y_escape else np.zeros((1, 1273))

    spearman_rho, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    if np.isnan(spearman_rho):
        spearman_rho = 0.0
    mse = float(np.mean((preds - targets) ** 2))

    return {
        "variant": variant_name,
        "spearman_rho": float(spearman_rho),
        "mse": mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_data", default="configs/data.yaml")
    parser.add_argument("--config_model", default="configs/model.yaml")
    args = parser.parse_args()

    with open(args.config_data) as f:
        cfg_data = yaml.safe_load(f)
    with open(args.config_model) as f:
        cfg_model = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running Ablation Suite on device: {device}")

    # Load canonical structure graph
    pdb_id = cfg_data["structure"]["reference_pdb"]
    graph_path = Path(cfg_data["processed"]["contact_graphs_dir"]) / f"{pdb_id}.pt"

    if graph_path.exists():
        graph_data = torch.load(graph_path, weights_only=True)
        node_scalars = graph_data["node_scalars"].to(device)
        node_vectors = graph_data["node_vectors"].to(device)
        edge_index   = graph_data["edge_index"].to(device)
        edge_scalars = graph_data["edge_scalars"].to(device)
    else:
        L = 1273
        node_scalars = torch.zeros(L, 10, device=device)
        node_vectors = torch.zeros(L, 3, 3, device=device)
        edge_index   = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_scalars = torch.zeros(0, 2, device=device)

    seq_encoder = SequenceEncoder(emb_dir=cfg_data["processed"]["esm2_embeddings_dir"])
    test_dataset = SpikeDataset(cfg_data["processed"]["test_split"], cfg_data["processed"]["dms_escape"], seq_encoder)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda b: custom_collate(b, seq_encoder, device))

    # Auto-detect d_seq
    sample_sid = test_dataset.df.iloc[0]["strain"]
    sample_emb = seq_encoder.load(sample_sid)
    d_seq = sample_emb.shape[-1]
    log.info(f"Auto-detected ESM-2 embedding dimension d_seq = {d_seq}")

    variants = {
        "1. Seq-only (ESM-2 + MLP)": SeqOnlyModel(d_seq=d_seq).to(device),
        "2. Struct-only (GVP-GNN + MLP)": StructOnlyModel().to(device),
        "3. Concat Baseline (ESM-2 + GVP)": ConcatModel(d_seq=d_seq).to(device),
        "4. Full Model (Cross-Attention)": JointMutationEscapeModel(d_seq=d_seq).to(device),
    }

    results = []
    for name, model in variants.items():
        res = run_ablation_eval(name, model, test_loader, node_scalars, node_vectors, edge_index, edge_scalars, device)
        results.append(res)

    res_df = pd.DataFrame(results)

    log.info("\n" + "=" * 60)
    log.info("ABLATION SUITE RESULTS")
    log.info("=" * 60)
    log.info(f"\n{res_df.to_string(index=False)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
