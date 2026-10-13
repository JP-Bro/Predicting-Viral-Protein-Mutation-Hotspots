# Viral Mutation & Escape Predictor

Multimodal DL framework (ESM-2 + GVP-GNN + cross-attention) for joint mutation-likelihood / immune-escape prediction on SARS-CoV-2 Spike.

**Team**: Aditya (Data) · Vineela (Model) · Jimit (Evaluation)

---

## Setup (Local — runs in seconds, no GPU needed)

```bash
conda create -n viralesc python=3.11 -y
conda activate viralesc
pip install -r requirements.txt
```

---

## Week 1 — Run These Locally

```bash
# Step 1: Download + clean Nextstrain sequences (primary source)
python src/data/nextstrain_ingest.py

# Step 2: (Optional) Supplement with NCBI Spike proteins
python src/data/ncbi_ingest.py

# Step 3: Build train / val / test splits
python src/data/temporal_split.py
```

**Output after these 3 steps:**
- `data/interim/cleaned_sequences.fasta` — 1,000–1,500 sequences
- `data/interim/metadata_clean.tsv` — date + lineage labels
- `data/processed/train.csv`, `val.csv`, `test.csv`

---

## Week 1–2 — Upload to Kaggle (runs on free T4 GPU)

After running the local steps above, upload `cleaned_sequences.fasta` to a Kaggle dataset, then run `notebooks/kaggle_msa_align.ipynb` for MAFFT alignment.

Upload outputs from Phase 1 to Kaggle for ESM-2 extraction in Week 4.

---

## Project Structure

```
viral-escape-predictor/
├── configs/
│   ├── data.yaml       # paths, filtering, temporal split params
│   ├── model.yaml      # ESM-2, GVP-GNN, fusion, head dimensions
│   └── train.yaml      # LR, batch size, loss weights, early stopping
├── data/
│   ├── raw/            # downloaded FASTA, metadata, PDB, DMS files
│   ├── interim/        # cleaned sequences, aligned MSA
│   └── processed/      # ESM-2 cache, contact graphs, splits
├── src/
│   ├── data/
│   │   ├── nextstrain_ingest.py   ← START HERE (Week 1)
│   │   ├── ncbi_ingest.py         ← supplementary
│   │   ├── temporal_split.py      ← Week 1
│   │   ├── dms_parser.py          ← Week 1–2
│   │   ├── covabdab_parser.py     ← Week 2
│   │   ├── structure_builder.py   ← Week 2 (runs locally)
│   │   └── msa_align.py           ← Kaggle notebook
│   ├── features/
│   │   ├── esm2_embed.py          ← Kaggle T4, Week 1–2
│   │   └── gvp_features.py        ← Week 4–5
│   ├── models/
│   │   ├── sequence_encoder.py    ← Week 4
│   │   ├── structure_encoder.py   ← Week 5
│   │   ├── fusion.py              ← Week 6
│   │   ├── heads.py               ← Week 6
│   │   └── joint_model.py         ← Week 6
│   ├── train.py                   ← Week 6–7 (Kaggle)
│   ├── eval.py                    ← Week 8–9 (Kaggle)
│   ├── ablations.py               ← Week 10 (Kaggle)
│   └── risk_score.py              ← Week 7
├── notebooks/
├── reports/
└── requirements.txt
```

---

## Compute Strategy

| Task | Where |
|---|---|
| Code writing, CSV filtering, plotting | **Local** (your laptop) |
| MAFFT MSA alignment | **Kaggle** free CPU |
| ESM-2 embedding extraction | **Kaggle** free T4 (16 GB) |
| All model training + ablations | **Kaggle** free T4 (16 GB) |

**Zero AWS. Zero GCP. Zero paid cloud.**
Kaggle: 30 free GPU-hrs/week. Total project usage: ~12–20 hrs.

---

## Temporal Split

| Set | Date range | Lineages |
|---|---|---|
| Train | ≤ Dec 31, 2023 | All pre-2024 |
| Val | ≤ Dec 31, 2023 (15% holdout) | All pre-2024 |
| **Test** | > Dec 31, 2023 | **JN.1, KP.2, KP.3, XEC** |
