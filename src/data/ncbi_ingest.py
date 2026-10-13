"""
ncbi_ingest.py
--------------
Downloads SARS-CoV-2 Spike protein sequences from NCBI Virus using the
ncbi-datasets-pylib client. Acts as a SUPPLEMENTARY source to fill gaps
not covered by Nextstrain open data.

Run locally — download is just internet + light filtering, no GPU needed.

Usage:
    python src/data/ncbi_ingest.py

Requirements:
    pip install ncbi-datasets-pylib biopython pandas
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR        = Path("data/raw")
INTERIM_DIR    = Path("data/interim")
NCBI_ZIP       = RAW_DIR / "ncbi_sars2.zip"
NCBI_FASTA_OUT = INTERIM_DIR / "ncbi_spike_protein.fasta"

# SARS-CoV-2 Spike protein (NCBI RefSeq accession for Wuhan-Hu-1 Spike)
SPIKE_PROTEIN_ACC = "YP_009724390.1"   # Wuhan-Hu-1 Spike

# Filtering
SPIKE_AA_LENGTH    = 1273
SPIKE_TOLERANCE    = 5
MAX_AMBIGUOUS_FRAC = 0.05   # X fraction in amino acid sequence


# ---------------------------------------------------------------------------
# Download via NCBI Datasets CLI (subprocess fallback if pylib unavailable)
# ---------------------------------------------------------------------------
def download_via_cli() -> None:
    """
    Fallback: use NCBI Datasets CLI tool if the Python client is unavailable.
    Install: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
    """
    import subprocess
    log.info("Using NCBI Datasets CLI to download Spike proteins ...")
    cmd = [
        "datasets", "download", "virus", "protein",
        "--taxon", "2697049",          # SARS-CoV-2 taxon ID
        "--protein-accessions", SPIKE_PROTEIN_ACC,
        "--filename", str(NCBI_ZIP),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"NCBI datasets CLI failed:\n{result.stderr}")
    log.info(f"Downloaded to {NCBI_ZIP}")


def download_spike_sequences() -> None:
    if NCBI_ZIP.exists():
        log.info(f"Already downloaded: {NCBI_ZIP} — skipping.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ncbi.datasets import DatasetsApiClient
        from ncbi.datasets.openapi import VirusApi
        from ncbi.datasets.openapi.models import V2VirusDataRequestContentType

        log.info("Downloading via ncbi-datasets-pylib ...")
        with DatasetsApiClient() as client:
            virus_api = VirusApi(client)
            # Download protein sequences for SARS-CoV-2 Spike
            response = virus_api.download_virus_protein_package(
                accessions=[SPIKE_PROTEIN_ACC],
                include_annotation_type=["PROT_FASTA"],
                _preload_content=False,
            )
            with open(NCBI_ZIP, "wb") as f:
                f.write(response.data)
        log.info(f"Downloaded to {NCBI_ZIP}")

    except ImportError:
        log.warning("ncbi-datasets-pylib not available — falling back to CLI.")
        download_via_cli()


# ---------------------------------------------------------------------------
# Extract + filter
# ---------------------------------------------------------------------------
def extract_and_filter() -> list:
    """
    Extract FASTA from zip, filter by length and ambiguity.
    Returns list of SeqRecord (amino acid sequences).
    """
    log.info(f"Extracting {NCBI_ZIP} ...")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    fasta_records = []

    with zipfile.ZipFile(NCBI_ZIP) as zf:
        fasta_files = [f for f in zf.namelist() if f.endswith(".faa") or f.endswith(".fasta")]
        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in {NCBI_ZIP}. "
                                    "Check the download completed correctly.")

        for fname in fasta_files:
            log.info(f"  Reading {fname} ...")
            with zf.open(fname) as fh:
                import io
                content = io.TextIOWrapper(fh, encoding="utf-8")
                for record in SeqIO.parse(content, "fasta"):
                    seq = str(record.seq).upper()
                    aa_len = len(seq)

                    # Length filter
                    if abs(aa_len - SPIKE_AA_LENGTH) > SPIKE_TOLERANCE:
                        continue

                    # Ambiguity filter (X = unknown amino acid)
                    x_frac = seq.count("X") / max(aa_len, 1)
                    if x_frac > MAX_AMBIGUOUS_FRAC:
                        continue

                    fasta_records.append(record)

    log.info(f"Filtered to {len(fasta_records):,} Spike sequences.")
    return fasta_records


# ---------------------------------------------------------------------------
# Parse NCBI FASTA headers to extract collection date + strain
# ---------------------------------------------------------------------------
def parse_ncbi_metadata(records: list) -> pd.DataFrame:
    """
    NCBI Spike protein FASTA headers look like:
        >YP_009724390.1 surface glycoprotein [Severe acute respiratory syndrome coronavirus 2]
    or for variant sequences:
        >QXN89565.1 surface glycoprotein, partial [SARS-CoV-2/human/USA/CA-xxxxx/2021]
    
    We extract accession and any strain info available.
    Full date/lineage metadata comes from Nextstrain; NCBI supplements the sequence pool.
    """
    rows = []
    for rec in records:
        parts = rec.description.split("[")
        strain_hint = parts[-1].rstrip("]") if len(parts) > 1 else rec.id
        rows.append({
            "ncbi_accession": rec.id,
            "strain_hint": strain_hint,
            "seq_length": len(rec.seq),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    download_spike_sequences()

    # 2. Extract + filter
    records = extract_and_filter()

    if not records:
        log.warning("No sequences passed filters. Check the downloaded zip contents.")
        return

    # 3. Save filtered FASTA
    SeqIO.write(records, NCBI_FASTA_OUT, "fasta")
    log.info(f"Saved {len(records):,} sequences → {NCBI_FASTA_OUT}")

    # 4. Parse basic metadata
    meta = parse_ncbi_metadata(records)
    meta_out = INTERIM_DIR / "ncbi_metadata_basic.tsv"
    meta.to_csv(meta_out, sep="\t", index=False)
    log.info(f"Saved basic NCBI metadata → {meta_out}")

    log.info(
        "\nncbi_ingest.py complete.\n"
        "NOTE: NCBI sequences supplement Nextstrain data.\n"
        "Primary pipeline: run nextstrain_ingest.py first, then merge if needed.\n"
        "Next step: run temporal_split.py to produce train/val/test CSVs."
    )


if __name__ == "__main__":
    main()
