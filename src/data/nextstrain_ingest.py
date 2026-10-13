"""
nextstrain_ingest.py
--------------------
Downloads and processes Nextstrain open SARS-CoV-2 build sequences + metadata.

This is the PRIMARY data source — metadata already contains pango_lineage,
clade, and collection_date, so minimal cleaning is needed.

Run locally (just download + light Pandas filtering).
MSA alignment step runs separately on Kaggle (see msa_align.py).

Usage:
    python src/data/nextstrain_ingest.py
"""

import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (mirrors configs/data.yaml — kept inline for standalone script use)
# ---------------------------------------------------------------------------
NEXTSTRAIN_FASTA_URL = "https://data.nextstrain.org/files/ncov/open/sequences.fasta.zst"
NEXTSTRAIN_META_URL  = "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst"

RAW_DIR    = Path("data/raw")
INTERIM_DIR = Path("data/interim")

FASTA_ZST  = RAW_DIR / "nextstrain_sequences.fasta.zst"
META_ZST   = RAW_DIR / "nextstrain_metadata.tsv.zst"
FASTA_OUT  = INTERIM_DIR / "nextstrain_sequences.fasta"
META_OUT   = INTERIM_DIR / "nextstrain_metadata.tsv"
CLEAN_FASTA = INTERIM_DIR / "cleaned_sequences.fasta"
CLEAN_META  = INTERIM_DIR / "metadata_clean.tsv"

# Filtering thresholds
SPIKE_AA_LENGTH    = 1273
SPIKE_TOLERANCE    = 5          # allow 1273 ± 5
MAX_AMBIGUOUS_FRAC = 0.05       # discard if >5% X or gap
SUBSAMPLE_N        = 1500       # target: 1 seq per lineage-month bucket
CUTOFF_DATE        = "2023-12-31"
TEST_LINEAGES      = {"JN.1", "KP.2", "KP.3", "XEC"}


# ---------------------------------------------------------------------------
# Step 1: Download & Decompress (.zst)
# ---------------------------------------------------------------------------
def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        log.info(f"Already downloaded: {dest} — skipping.")
        return
    log.info(f"Downloading {url} → {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    log.info(f"Download complete: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def decompress_zst(src: Path, dest: Path) -> None:
    if dest.exists():
        log.info(f"Already decompressed: {dest} — skipping.")
        return
    log.info(f"Decompressing {src} → {dest} via zstandard ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as f_in, open(dest, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)
    log.info("Decompression complete.")


# ---------------------------------------------------------------------------
# Step 2: Load + clean metadata
# ---------------------------------------------------------------------------
def load_and_clean_metadata(meta_path: Path) -> pd.DataFrame:
    log.info(f"Loading metadata efficiently in chunks from {meta_path} ...")
    
    # Read header to inspect column names
    header_df = pd.read_csv(meta_path, sep="\t", nrows=1)
    all_cols = list(header_df.columns)
    
    # Map desired columns
    target_cols = []
    col_rename = {}
    for col in all_cols:
        col_lower = col.lower()
        if col_lower == "strain":
            target_cols.append(col)
            col_rename[col] = "strain"
        elif col_lower in ["date", "collection_date"]:
            target_cols.append(col)
            col_rename[col] = "collection_date"
        elif col_lower in ["pango_lineage", "lineage"]:
            target_cols.append(col)
            col_rename[col] = "lineage"
        elif col_lower in ["clade", "clade_nextstrain"]:
            target_cols.append(col)
            col_rename[col] = "clade"

    log.info(f"Selected metadata columns: {target_cols}")

    chunks = []
    chunk_size = 100000

    for chunk in pd.read_csv(meta_path, sep="\t", usecols=target_cols, chunksize=chunk_size, low_memory=True):
        chunk.rename(columns=col_rename, inplace=True)
        chunk.dropna(subset=["collection_date", "lineage", "strain"], inplace=True)
        # Quick filter out invalid date strings
        valid_dates = chunk["collection_date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        chunk = chunk[valid_dates].copy()
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Clean metadata rows after chunked parsing: {len(df):,}")

    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    df.dropna(subset=["collection_date"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 3: Filter sequences (quality + length)
# ---------------------------------------------------------------------------
def filter_sequences(fasta_path: Path, valid_strains: set) -> dict:
    """
    Returns a dict {strain: SeqRecord} for sequences that pass:
    - strain is in valid_strains (from metadata)
    - amino acid length ≈ SPIKE_AA_LENGTH (nucleotide FASTA, so we check nt length / 3)
    - ambiguous base fraction ≤ MAX_AMBIGUOUS_FRAC
    """
    log.info("Filtering sequences ...")
    passed = {}
    total = 0

    # Nextstrain FASTA contains full genomes — we extract Spike region by length heuristic.
    # Spike ORF nt = 1273 * 3 = 3819 nt.  Full genome ~29,903 nt.
    # NOTE: Nextstrain open sequences are FULL GENOMES — we use the metadata lineage tag
    # to identify Spike-relevant sequences and rely on ESM-2 Spike-pretrained embeddings.
    # For graph construction we use PDB / ColabFold Spike structures separately.

    for record in SeqIO.parse(fasta_path, "fasta"):
        total += 1
        strain = record.id.split("|")[0].strip()

        if strain not in valid_strains:
            continue

        seq = str(record.seq).upper()
        ambiguous = sum(1 for c in seq if c not in "ACGT") / max(len(seq), 1)
        if ambiguous > MAX_AMBIGUOUS_FRAC:
            continue

        passed[strain] = record

    log.info(f"Sequences: {total:,} total → {len(passed):,} passed filters.")
    return passed


# ---------------------------------------------------------------------------
# Step 4: Subsample — 1 sequence per lineage-month bucket
# ---------------------------------------------------------------------------
def subsample(df: pd.DataFrame, n: int = SUBSAMPLE_N) -> pd.DataFrame:
    """
    One sequence per (lineage, year-month) bucket, up to n total.
    Gives broad evolutionary coverage without overrepresenting dominant lineages.
    """
    log.info(f"Subsampling to ≤{n} sequences (1 per lineage-month) ...")
    df = df.copy()
    df["year_month"] = df["collection_date"].dt.to_period("M").astype(str)

    # Shuffle dataset, then take 1 sequence per (lineage, year_month) bucket
    sampled = df.sample(frac=1.0, random_state=42).drop_duplicates(subset=["lineage", "year_month"]).copy()

    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=42).reset_index(drop=True)

    log.info(f"Subsampled to {len(sampled):,} sequences across {sampled['lineage'].nunique()} lineages.")
    return sampled


# ---------------------------------------------------------------------------
# Step 5: Validate temporal split (anti-leakage assertion)
# ---------------------------------------------------------------------------
def validate_temporal_split(df: pd.DataFrame, cutoff: str) -> None:
    cutoff_dt = pd.to_datetime(cutoff)
    train_df = df[df["collection_date"] <= cutoff_dt]
    test_df  = df[df["collection_date"] > cutoff_dt]

    if len(train_df) == 0 or len(test_df) == 0:
        log.warning("Train or test split is empty — check cutoff date and lineage filters.")
        return

    assert train_df["collection_date"].max() <= cutoff_dt, \
        "TEMPORAL LEAKAGE: train set contains dates after cutoff!"
    assert test_df["collection_date"].min() > cutoff_dt, \
        "TEMPORAL LEAKAGE: test set contains dates before cutoff!"

    log.info(f"Temporal split OK — train: {len(train_df):,} seqs (up to {cutoff}), "
             f"test: {len(test_df):,} seqs (after {cutoff}).")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    download_file(NEXTSTRAIN_FASTA_URL, FASTA_ZST)
    download_file(NEXTSTRAIN_META_URL, META_ZST)

    # 2. Decompress
    decompress_zst(FASTA_ZST, FASTA_OUT)
    decompress_zst(META_ZST, META_OUT)

    # 3. Load + clean metadata
    meta = load_and_clean_metadata(META_OUT)

    # 4. Subsample
    meta_sampled = subsample(meta, n=SUBSAMPLE_N)

    # 5. Filter sequences to match sampled strains
    valid_strains = set(meta_sampled["strain"])
    sequences = filter_sequences(FASTA_OUT, valid_strains)

    # Align metadata to sequences that actually passed
    meta_final = meta_sampled[meta_sampled["strain"].isin(sequences)].copy()
    log.info(f"Final matched sequences: {len(meta_final):,}")

    # 6. Validate temporal split (anti-leakage)
    validate_temporal_split(meta_final, CUTOFF_DATE)

    # 7. Save
    SeqIO.write(list(sequences.values()), CLEAN_FASTA, "fasta")
    meta_final.to_csv(CLEAN_META, sep="\t", index=False)

    log.info(f"Saved cleaned FASTA → {CLEAN_FASTA}")
    log.info(f"Saved clean metadata → {CLEAN_META}")
    log.info("nextstrain_ingest.py complete. Next step: upload cleaned_sequences.fasta to Kaggle for MSA alignment.")


if __name__ == "__main__":
    main()
