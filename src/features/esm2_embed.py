"""
esm2_embed.py
-------------
ESM-2 embedding extraction script.

Reads cleaned_sequences.fasta, extracts per-residue representations,
saves one .pt file per sequence to data/processed/esm2_embeddings/.

Usage:
    python src/features/esm2_embed.py          # Kaggle (default)
    python src/features/esm2_embed.py --local  # local test with small model
"""

import argparse
import logging
import re
import time
from pathlib import Path

import torch
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ESM2_MODEL_LARGE = "esm2_t33_650M_UR50D"   # use on Kaggle T4
ESM2_MODEL_SMALL = "esm2_t12_35M_UR50D"    # use locally for testing pipeline

REPR_LAYER_LARGE = 33
REPR_LAYER_SMALL = 12

MAX_SEQ_LEN = 1280   # ESM-2 max token length (including BOS/EOS)
CHECKPOINT_EVERY = 100  # save progress log every N sequences


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_paths(local: bool) -> tuple[Path, Path]:
    if local:
        seq_in  = Path("data/interim/cleaned_sequences.fasta")
        emb_dir = Path("data/processed/esm2_embeddings")
    else:
        input_base = Path("/kaggle/input")
        dataset_dir = next(input_base.iterdir(), input_base)
        seq_in  = dataset_dir / "cleaned_sequences.fasta"
        emb_dir = Path("/kaggle/working/esm2_embeddings")
    return seq_in, emb_dir


# ---------------------------------------------------------------------------
# Load ESM-2
# ---------------------------------------------------------------------------
def load_esm2(local: bool) -> tuple:
    import esm

    model_name = ESM2_MODEL_SMALL if local else ESM2_MODEL_LARGE
    repr_layer = REPR_LAYER_SMALL if local else REPR_LAYER_LARGE

    log.info(f"Loading {model_name} ...")
    t0 = time.time()

    if model_name == ESM2_MODEL_LARGE:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if device == "cuda":
        model = model.half()

    log.info(f"Model loaded in {time.time() - t0:.1f}s on {device}")
    return model, alphabet, repr_layer, device


# ---------------------------------------------------------------------------
# Embed one sequence
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_sequence(
    seq: str,
    model,
    batch_converter,
    repr_layer: int,
    device: str,
) -> torch.Tensor:
    seq = seq[:MAX_SEQ_LEN - 2]
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    emb = out["representations"][repr_layer][0, 1:-1].float().cpu()
    return emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Use small ESM-2 model for local pipeline testing")
    args = parser.parse_args()

    seq_in, emb_dir = get_paths(args.local)
    emb_dir.mkdir(parents=True, exist_ok=True)

    if not seq_in.exists():
        log.error(f"Input FASTA not found: {seq_in}")
        raise FileNotFoundError(seq_in)

    # Load model
    model, alphabet, repr_layer, device = load_esm2(args.local)
    batch_converter = alphabet.get_batch_converter()

    # Load sequences
    records = list(SeqIO.parse(seq_in, "fasta"))
    log.info(f"Sequences to embed: {len(records):,}")

    # Find already-done (matching clean filename stem)
    done = {p.stem for p in emb_dir.glob("*.pt")}
    remaining = [r for r in records if re.sub(r"[^a-zA-Z0-9_\-]", "_", r.id) not in done]
    log.info(f"Already done: {len(done)} | Remaining: {len(remaining)}")

    if not remaining:
        log.info("All embeddings already extracted. Nothing to do.")
        return

    # Extract
    t_start = time.time()
    for i, record in enumerate(remaining, 1):
        seq_id = record.id
        seq    = str(record.seq).upper().replace("-", "")

        seq_id_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", seq_id)
        out_path = emb_dir / f"{seq_id_clean}.pt"

        try:
            emb = embed_sequence(seq, model, batch_converter, repr_layer, device)
            torch.save({"strain": seq_id, "embedding": emb}, out_path)
        except Exception as e:
            log.warning(f"  [{i}/{len(remaining)}] FAILED {seq_id}: {e} — skipping.")
            continue

        if i % CHECKPOINT_EVERY == 0 or i == len(remaining):
            elapsed = time.time() - t_start
            rate    = i / elapsed
            eta     = (len(remaining) - i) / max(rate, 1e-6)
            log.info(
                f"Progress: {i}/{len(remaining)} | "
                f"{elapsed / 60:.1f} min elapsed | "
                f"ETA: {eta / 60:.1f} min"
            )

    log.info(f"\nesm2_embed.py complete — {len(records)} embeddings saved to {emb_dir}")


if __name__ == "__main__":
    main()
