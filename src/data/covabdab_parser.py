"""
covabdab_parser.py
------------------
Downloads and parses CoV-AbDab (Coronavirus Antibody Database) to map
each antibody → epitope class (RBD Class 1-4, NTD supersite, etc.).

This mapping is used to weight DMS supervision labels by epitope region
during training.

Output:
  - data/processed/covabdab_epitope_map.csv
  - data/processed/covabdab_epitope_map.pt   (dict saved with torch)

Run locally — light Pandas work.

Usage:
    python src/data/covabdab_parser.py
"""

import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR   = Path("data/raw")
PROCESSED = Path("data/processed")

# CoV-AbDab public download (no login needed)
COVABDAB_URL = "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_download.csv"
COVABDAB_RAW = RAW_DIR / "covabdab.csv"
EPITOPE_MAP_CSV = PROCESSED / "covabdab_epitope_map.csv"
EPITOPE_MAP_PT  = PROCESSED / "covabdab_epitope_map.pt"

# Epitope class keywords to match from the 'Protein + Epitope' column
EPITOPE_KEYWORDS = {
    "RBD_class1":   ["class 1", "class1", "rbd-1", "rb1"],
    "RBD_class2":   ["class 2", "class2", "rbd-2", "rb2"],
    "RBD_class3":   ["class 3", "class3", "rbd-3", "rb3"],
    "RBD_class4":   ["class 4", "class4", "rbd-4", "rb4"],
    "NTD_supersite": ["ntd", "n-terminal", "supersite"],
    "RBD_generic":  ["rbd", "receptor binding"],   # fallback for unclassified RBD
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_covabdab() -> None:
    if COVABDAB_RAW.exists():
        log.info(f"Already downloaded: {COVABDAB_RAW} — skipping.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading CoV-AbDab from {COVABDAB_URL} ...")
    try:
        with urllib.request.urlopen(COVABDAB_URL, timeout=30) as r:
            COVABDAB_RAW.write_bytes(r.read())
        log.info(f"Saved → {COVABDAB_RAW} ({COVABDAB_RAW.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log.error(
            f"Download failed: {e}\n"
            f"Visit https://opig.stats.ox.ac.uk/webapps/covabdab/ → Downloads → "
            f"save as {COVABDAB_RAW}"
        )
        raise


# ---------------------------------------------------------------------------
# Parse + classify
# ---------------------------------------------------------------------------
def classify_epitope(epitope_text: str) -> str:
    """
    Map a free-text epitope description to one of the standard epitope classes.
    Returns 'unknown' if no match found.
    """
    if not isinstance(epitope_text, str):
        return "unknown"
    text_lower = epitope_text.lower()

    for cls, keywords in EPITOPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cls
    return "unknown"


def parse_covabdab(path: Path) -> pd.DataFrame:
    log.info(f"Loading CoV-AbDab from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Raw rows: {len(df):,}, columns: {list(df.columns[:10])} ...")

    # Identify relevant columns (column names vary slightly across versions)
    name_col    = next((c for c in df.columns if "name" in c.lower() and "ab" in c.lower()), df.columns[0])
    epitope_col = next((c for c in df.columns if "epitope" in c.lower() or "protein" in c.lower()), None)
    target_col  = next((c for c in df.columns if "target" in c.lower() or "antigen" in c.lower()), None)

    log.info(f"Using: name='{name_col}', epitope='{epitope_col}', target='{target_col}'")

    # Filter to SARS-CoV-2 entries only
    if target_col:
        sars2_mask = df[target_col].str.contains("SARS-CoV-2|SARS2|COVID", case=False, na=False)
        df = df[sars2_mask].copy()
        log.info(f"SARS-CoV-2 entries: {len(df):,}")

    # Build output table
    result = pd.DataFrame()
    result["antibody_name"] = df[name_col].values

    if epitope_col:
        result["epitope_raw"]   = df[epitope_col].values
        result["epitope_class"] = df[epitope_col].apply(classify_epitope)
    else:
        result["epitope_raw"]   = "unknown"
        result["epitope_class"] = "unknown"

    # Class distribution
    dist = result["epitope_class"].value_counts()
    log.info(f"\nEpitope class distribution:\n{dist.to_string()}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    download_covabdab()
    df = parse_covabdab(COVABDAB_RAW)

    # Save CSV
    df.to_csv(EPITOPE_MAP_CSV, index=False)
    log.info(f"Saved epitope map → {EPITOPE_MAP_CSV}")

    # Save as dict {antibody_name: epitope_class} in .pt
    epitope_dict = dict(zip(df["antibody_name"].astype(str), df["epitope_class"].astype(str)))
    torch.save(epitope_dict, EPITOPE_MAP_PT)
    log.info(f"Saved epitope dict ({len(epitope_dict):,} antibodies) → {EPITOPE_MAP_PT}")

    log.info("\ncovabdab_parser.py complete.")
    log.info("Next step: run structure_builder.py to build residue contact graphs.")


if __name__ == "__main__":
    main()
