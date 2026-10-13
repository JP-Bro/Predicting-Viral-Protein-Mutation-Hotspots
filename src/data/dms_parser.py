"""
dms_parser.py
-------------
Downloads and processes Bloom Lab Deep Mutational Scanning (DMS) escape data.

Source: https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/processed_data/escape_calculator_data.csv

Outputs:
  - data/processed/dms_escape_per_antibody.csv   per-antibody, per-residue escape
  - data/processed/dms_escape_aggregated.csv      per-residue mean escape summary
  - data/processed/dms_escape.pt                  torch tensor (L=1273, n_classes=5)

Run locally — pure Pandas, no GPU, finishes in < 1 min.

Usage:
    python src/data/dms_parser.py
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DMS_DIR = Path("data/raw/dms")
PROCESSED   = Path("data/processed")

DMS_URL = "https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/processed_data/escape_calculator_data.csv"
RAW_DMS_CSV = RAW_DMS_DIR / "escape_calculator_data.csv"

SPIKE_LENGTH = 1273
TOP_DECILE = 0.10

EPITOPE_CLASSES = ["RBD_class1", "RBD_class2", "RBD_class3", "RBD_class4", "NTD_supersite"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_dms() -> Path:
    RAW_DMS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DMS_CSV.exists():
        log.info(f"Already downloaded: {RAW_DMS_CSV}")
        return RAW_DMS_CSV

    log.info(f"Downloading Bloom Lab DMS data from {DMS_URL} ...")
    req = urllib.request.Request(DMS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(RAW_DMS_CSV, "wb") as out:
        out.write(r.read())
    log.info(f"Saved → {RAW_DMS_CSV} ({RAW_DMS_CSV.stat().st_size / 1e6:.1f} MB)")
    return RAW_DMS_CSV


# ---------------------------------------------------------------------------
# Parse & Normalize
# ---------------------------------------------------------------------------
def parse_dms(csv_path: Path) -> pd.DataFrame:
    log.info("Parsing DMS escape data ...")
    df = pd.read_csv(csv_path)
    log.info(f"Raw DMS rows: {len(df):,}, columns: {list(df.columns)}")

    df = df[["condition", "site", "escape"]].dropna().copy()
    df["site"] = df["site"].astype(int)
    df = df[(df["site"] >= 1) & (df["site"] <= SPIKE_LENGTH)]

    # Normalize escape values to [0, 1] per antibody/condition
    log.info("Normalizing escape scores per condition ...")
    def norm(g):
        hi = g.max()
        lo = g.min()
        return (g - lo) / (hi - lo) if hi > lo else 0.0

    df["escape_norm"] = df.groupby("condition")["escape"].transform(norm)
    return df


# ---------------------------------------------------------------------------
# Aggregate per residue & epitope class
# ---------------------------------------------------------------------------
def aggregate_escape(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating top-decile escape scores per site ...")
    records = []
    for site, group in df.groupby("site"):
        vals = group["escape_norm"].values
        threshold = np.quantile(vals, 1.0 - TOP_DECILE)
        top_vals = vals[vals >= threshold]
        records.append({
            "site": int(site),
            "escape_summary": float(top_vals.mean()) if len(top_vals) > 0 else 0.0,
        })
    return pd.DataFrame(records)


def build_escape_tensor(agg_df: pd.DataFrame) -> torch.Tensor:
    tensor = torch.zeros(SPIKE_LENGTH, len(EPITOPE_CLASSES), dtype=torch.float32)
    for _, row in agg_df.iterrows():
        site = int(row["site"]) - 1  # 0-indexed
        if 0 <= site < SPIKE_LENGTH:
            # Map escape summary into class channels
            tensor[site, :] = float(row["escape_summary"])
    return tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = download_dms()
    df = parse_dms(csv_path)

    # Save per-antibody
    per_ab_out = PROCESSED / "dms_escape_per_antibody.csv"
    df.to_csv(per_ab_out, index=False)
    log.info(f"Saved per-antibody escape → {per_ab_out}")

    # Aggregate
    agg = aggregate_escape(df)
    agg_out = PROCESSED / "dms_escape_aggregated.csv"
    agg.to_csv(agg_out, index=False)
    log.info(f"Saved aggregated escape → {agg_out}")

    # Tensor
    tensor = build_escape_tensor(agg)
    tensor_out = PROCESSED / "dms_escape.pt"
    torch.save(tensor, tensor_out)
    log.info(f"Saved escape tensor {tuple(tensor.shape)} → {tensor_out}")

    log.info("dms_parser.py complete.")


if __name__ == "__main__":
    main()
