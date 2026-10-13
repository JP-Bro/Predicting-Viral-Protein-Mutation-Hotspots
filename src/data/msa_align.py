"""
msa_align.py
------------
MAFFT MSA alignment script — designed to run as a Kaggle Notebook (free CPU).

What this does:
  1. Reads cleaned_sequences.fasta (uploaded to Kaggle as a dataset)
  2. Runs MAFFT --auto alignment against Wuhan-Hu-1 Spike reference
  3. Saves aligned.fasta output to /kaggle/working/

How to use on Kaggle:
  1. Upload data/interim/cleaned_sequences.fasta as a Kaggle Dataset
  2. Create a new Kaggle Notebook (CPU accelerator)
  3. Copy-paste this script into a code cell and run
  4. Download /kaggle/working/aligned.fasta → save to data/interim/aligned.fasta locally

Runtime on Kaggle free CPU:
  ~30–60 min for 1,000–1,500 sequences (well within 12-hr session limit)

Local usage (optional, will be slow):
  python src/data/msa_align.py --local
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wuhan-Hu-1 Spike reference sequence (pasted inline so no external file needed)
# GenBank: MN908947.3, positions 21563–25384 (protein, 1273 aa)
# ---------------------------------------------------------------------------
WUHAN_SPIKE_AA = """>Wuhan-Hu-1_Spike_reference
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV
SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFL
GVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINL
VRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENG
TITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWN
RKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKL
PDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYG
FPTTGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRN
VFQAPAGTTIVNPGDYLDYYPQLKSFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVN
NTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGK
YEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_paths(local: bool) -> tuple[Path, Path, Path, Path]:
    if local:
        base = Path("data")
        seq_in  = base / "interim" / "cleaned_sequences.fasta"
        ref_fa  = base / "interim" / "wuhan_spike_reference.fasta"
        seq_ref = base / "interim" / "sequences_with_ref.fasta"
        aln_out = base / "interim" / "aligned.fasta"
    else:
        # Kaggle paths
        input_base = Path("/kaggle/input")
        # Adjust dataset name to match what you uploaded on Kaggle
        dataset_dir = next(input_base.iterdir(), input_base)
        seq_in  = dataset_dir / "cleaned_sequences.fasta"
        ref_fa  = Path("/kaggle/working/wuhan_spike_reference.fasta")
        seq_ref = Path("/kaggle/working/sequences_with_ref.fasta")
        aln_out = Path("/kaggle/working/aligned.fasta")

    return seq_in, ref_fa, seq_ref, aln_out


# ---------------------------------------------------------------------------
# Check MAFFT installed
# ---------------------------------------------------------------------------
def check_mafft() -> None:
    result = subprocess.run(["mafft", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        log.info("MAFFT not found — installing ...")
        # On Kaggle / Ubuntu
        subprocess.run(["apt-get", "install", "-y", "-q", "mafft"], check=True)
        log.info("MAFFT installed.")
    else:
        log.info(f"MAFFT available: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Run MAFFT
# ---------------------------------------------------------------------------
def run_mafft(input_fasta: Path, output_fasta: Path) -> None:
    cmd = [
        "mafft",
        "--auto",
        "--thread", "4",        # 4 threads — safe for Kaggle free CPU
        "--quiet",
        "--reorder",
        str(input_fasta),
    ]
    log.info(f"Running: {' '.join(cmd)}")
    log.info(f"Output → {output_fasta}")

    with open(output_fasta, "w") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        log.error(f"MAFFT failed:\n{result.stderr}")
        sys.exit(1)

    log.info("MAFFT alignment complete.")

    # Quick stats
    from Bio import SeqIO
    records = list(SeqIO.parse(output_fasta, "fasta"))
    log.info(f"Aligned sequences: {len(records)}")
    if records:
        lengths = [len(r.seq) for r in records]
        log.info(f"Alignment length: {max(lengths)} columns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MAFFT MSA alignment (run on Kaggle)")
    parser.add_argument("--local", action="store_true",
                        help="Run locally instead of Kaggle paths")
    args = parser.parse_args()

    seq_in, ref_fa, seq_ref, aln_out = get_paths(args.local)

    if not seq_in.exists():
        log.error(
            f"Input FASTA not found: {seq_in}\n"
            f"{'Run nextstrain_ingest.py first.' if args.local else 'Upload cleaned_sequences.fasta to Kaggle as a Dataset.'}"
        )
        sys.exit(1)

    check_mafft()

    # Write reference sequence to file
    ref_fa.parent.mkdir(parents=True, exist_ok=True)
    ref_fa.write_text(WUHAN_SPIKE_AA)
    log.info(f"Reference sequence written → {ref_fa}")

    # Concatenate reference + sequences for alignment
    log.info("Prepending reference to input sequences ...")
    ref_content = ref_fa.read_text()
    seq_content = seq_in.read_text()
    seq_ref.write_text(ref_content + "\n" + seq_content)
    log.info(f"Combined FASTA → {seq_ref}")

    # Run MAFFT
    aln_out.parent.mkdir(parents=True, exist_ok=True)
    run_mafft(seq_ref, aln_out)

    log.info(f"\nmsa_align.py complete.")
    log.info(f"Download {aln_out} from Kaggle → save to data/interim/aligned.fasta locally.")
    log.info("Next step: run esm2_embed.py on Kaggle T4 to extract ESM-2 embeddings.")


if __name__ == "__main__":
    main()
