"""
risk_score.py
-------------
Computes per-residue joint risk scores:
  R_i = P(mutate_i) * S(escape_i)

Higher R_i indicates higher priority for biological surveillance and vaccine design.
"""

import numpy as np
import pandas as pd
import torch


def compute_risk_scores(
    p_mutate: torch.Tensor | np.ndarray,
    s_escape: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """
    Computes elementwise joint risk score R = P * S.
    Inputs can be PyTorch tensors or NumPy arrays of shape (L,) or (B, L).
    """
    if isinstance(p_mutate, torch.Tensor):
        p_mutate = p_mutate.detach().cpu().numpy()
    if isinstance(s_escape, torch.Tensor):
        s_escape = s_escape.detach().cpu().numpy()

    # Ensure nonnegative escape scaling for risk product
    s_escape_clipped = np.clip(s_escape, a_min=0.0, a_max=None)
    return p_mutate * s_escape_clipped


def get_top_k_vulnerability_residues(
    risk_scores: np.ndarray,
    k: int = 10,
    start_site: int = 1,
) -> pd.DataFrame:
    """
    Ranks residues by risk score and returns a DataFrame of the top K residues.
    Sites are 1-indexed (e.g. Spike residue 1..1273).
    """
    if risk_scores.ndim > 1:
        # If batch provided, average over batch
        risk_scores = np.mean(risk_scores, axis=0)

    sites = np.arange(start_site, start_site + len(risk_scores))
    top_indices = np.argsort(-risk_scores)[:k]

    df = pd.DataFrame({
        "rank": np.arange(1, k + 1),
        "site": sites[top_indices],
        "risk_score": risk_scores[top_indices],
    })
    return df
