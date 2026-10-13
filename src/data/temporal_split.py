"""
temporal_split.py
-----------------
Splits the cleaned sequence metadata into train / val / test sets
based on collection date.

Cutoff: Dec 31, 2023
- Train + Val: sequences collected on or before Dec 31, 2023
- Test: sequences from JN.1, KP.2, KP.3, XEC (collected after cutoff)

Includes a hard assertion to guarantee zero temporal leakage.

Run locally — pure Pandas, instant.

Usage:
    python src/data/temporal_split.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLEAN_META  = Path("data/interim/metadata_clean.tsv")
PROCESSED   = Path("data/processed")
TRAIN_OUT   = PROCESSED / "train.csv"
VAL_OUT     = PROCESSED / "val.csv"
TEST_OUT    = PROCESSED / "test.csv"

CUTOFF_DATE   = pd.Timestamp("2023-12-31")
TEST_LINEAGES = {"JN.1", "KP.2", "KP.3", "XEC"}
VAL_FRACTION  = 0.15    # 15% of pre-cutoff data held out for validation
RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------
def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).

    Test set = rows where:
      - collection_date > CUTOFF_DATE, AND
      - lineage matches one of TEST_LINEAGES
    
    Train/val = everything else collected before/on cutoff.
    Val = random 15% of train pool.
    """
    df = df.copy()
    df["collection_date"] = pd.to_datetime(df["collection_date"])

    # Test set: post-cutoff target lineages only
    is_test = (df["collection_date"] > CUTOFF_DATE) & (df["lineage"].isin(TEST_LINEAGES))
    test_df = df[is_test].copy()

    # Train+val pool: pre-cutoff
    trainval_df = df[df["collection_date"] <= CUTOFF_DATE].copy()

    # Random val split from train pool
    val_df   = trainval_df.sample(frac=VAL_FRACTION, random_state=RANDOM_SEED)
    train_df = trainval_df.drop(val_df.index)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anti-leakage assertion — MUST pass before saving
# ---------------------------------------------------------------------------
def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_dates = pd.to_datetime(train["collection_date"])
    val_dates   = pd.to_datetime(val["collection_date"])
    test_dates  = pd.to_datetime(test["collection_date"])

    assert train_dates.max() <= CUTOFF_DATE, \
        f"TEMPORAL LEAKAGE in train set! Max date: {train_dates.max()}"
    assert val_dates.max() <= CUTOFF_DATE, \
        f"TEMPORAL LEAKAGE in val set! Max date: {val_dates.max()}"
    assert test_dates.min() > CUTOFF_DATE, \
        f"TEMPORAL LEAKAGE in test set! Min date: {test_dates.min()}"

    # No strain overlap between train/val and test
    train_val_strains = set(train["strain"]) | set(val["strain"])
    test_strains      = set(test["strain"])
    overlap = train_val_strains & test_strains
    assert len(overlap) == 0, \
        f"LEAKAGE: {len(overlap)} strains appear in both train/val AND test!"

    log.info("✅ Temporal split validated — zero leakage confirmed.")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    log.info("\n" + "=" * 60)
    log.info("SPLIT SUMMARY")
    log.info("=" * 60)
    log.info(f"Train : {len(train):>5,} sequences | "
             f"date range: {train['collection_date'].min().date()} → "
             f"{train['collection_date'].max().date()}")
    log.info(f"Val   : {len(val):>5,} sequences | "
             f"date range: {val['collection_date'].min().date()} → "
             f"{val['collection_date'].max().date()}")
    log.info(f"Test  : {len(test):>5,} sequences | "
             f"date range: {test['collection_date'].min().date()} → "
             f"{test['collection_date'].max().date()}")
    log.info(f"\nTest lineages present: {sorted(test['lineage'].unique())}")
    log.info(f"Train lineages: {train['lineage'].nunique()} unique")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not CLEAN_META.exists():
        raise FileNotFoundError(
            f"{CLEAN_META} not found.\n"
            "Run nextstrain_ingest.py first to generate cleaned metadata."
        )

    log.info(f"Loading {CLEAN_META} ...")
    df = pd.read_csv(CLEAN_META, sep="\t")
    log.info(f"Total sequences: {len(df):,}")

    train, val, test = make_splits(df)
    assert_no_leakage(train, val, test)
    print_summary(train, val, test)

    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    log.info(f"\nSaved → {TRAIN_OUT}, {VAL_OUT}, {TEST_OUT}")
    log.info("Next step: run dms_parser.py to build the escape label matrix.")


if __name__ == "__main__":
    main()
