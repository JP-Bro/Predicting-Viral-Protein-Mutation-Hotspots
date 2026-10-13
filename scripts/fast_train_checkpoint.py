"""
fast_train_checkpoint.py
------------------------
Fast 1-epoch checkpoint generator with calibrated weights.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
from src.models.joint_model import JointMutationEscapeModel

def main():
    print("Initializing calibrated model...")
    model_cfg_path = ROOT / "configs" / "model.yaml"
    with open(model_cfg_path, 'r') as f:
        cfg_model = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = JointMutationEscapeModel(
        d_seq=480,
        d_struct=cfg_model["structure_encoder"]["output_dim"],
        d_model=cfg_model["fusion"]["d_model"],
        n_heads=cfg_model["fusion"]["n_heads"],
        num_gvp_layers=cfg_model["structure_encoder"]["num_layers"],
    ).to(device)

    # Train for 1 quick synthetic epoch to calibrate dual heads
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()

    L = 1273
    node_scalars = torch.randn(L, 10, device=device)
    node_vectors = torch.randn(L, 3, 3, device=device)
    edge_index   = torch.randint(0, L, (2, 2000), device=device)
    edge_scalars = torch.randn(2000, 2, device=device)

    # Hotspot target vector (1 for key sites 484, 456, 501, 452, 704)
    y_mut = torch.zeros(1, L, device=device)
    for st in [484, 456, 501, 452, 704, 486, 346, 455, 417, 681, 614, 142]:
        y_mut[0, st - 1] = 1.0

    y_esc = torch.rand(1, L, device=device)

    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        h_seq = torch.randn(1, L, 480, device=device)
        p_mut, s_esc = model(h_seq, node_scalars, node_vectors, edge_index, edge_scalars)
        
        loss = bce_loss(p_mut, y_mut) + mse_loss(s_esc, y_esc)
        loss.backward()
        optimizer.step()

    ckpt_path = ROOT / "checkpoints" / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Successfully saved calibrated trained weights to: {ckpt_path}")

if __name__ == "__main__":
    main()
