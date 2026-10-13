# Viral Mutation & Immune-Escape Predictor — Project Understanding & Viva Preparation

**Project folder audited:** `viral-escape-predictor/` (plus top-level `implementation_plan_updated.md` and three prior report decks)
**Audit method:** direct inspection of every script, config, dataset file, and model checkpoint on disk — not a summary of the README or the implementation plan. Every claim below cites the file and (where useful) the exact code responsible. This document follows four verification labels throughout:

- ✅ **Implemented & verified** — code exists, runs on real inputs, and does what it says
- 🟡 **Implemented with caveats** — code exists and runs, but with a methodological weakness
- 🟠 **Hardcoded / not learned** — presented as a model output but is a fixed constant or formula
- 🔴 **Discrepancy** — documentation/config claims one thing, the code or artifacts show another

---

## 1. What the Project Is

**Stated goal:** for every one of the 1,273 residues of the SARS-CoV-2 Spike protein, jointly predict

1. `P(mutate)` — the probability that residue becomes a mutation hotspot in future viral lineages, and
2. `S(escape)` — how much a mutation there would help the virus evade neutralizing antibodies,

then rank residues by a joint risk score `R = P × S` for genomic-surveillance / vaccine-design triage.

**Team:** Aditya, Vineela, Jimit (per `implementation_plan_updated.md`), 12-week plan.

**Core architectural idea:** fuse a frozen protein language model (ESM-2) with a structural graph neural network (GVP-GNN) via cross-attention, supervised by real Deep Mutational Scanning (DMS) escape data from the Bloom Lab. ✅ This is a legitimate, literature-grounded idea (Section 5).

---

## 2. Repository Inventory (what matters and why)

| Path | Role | Status |
|---|---|---|
| `configs/data.yaml`, `model.yaml`, `train.yaml` | Dataset paths, architecture hyperparameters, training hyperparameters | ✅ exist, internally consistent |
| `src/data/ncbi_ingest.py`, `nextstrain_ingest.py` | Pull real public Spike sequences + metadata | ✅ real public sources, no auth required |
| `src/data/dms_parser.py` | Downloads & aggregates Bloom Lab DMS escape CSV | ✅ live download from `github.com/jbloomlab/SARS2_RBD_Ab_escape_maps`, top-decile per-site aggregation |
| `src/data/covabdab_parser.py` | Antibody epitope-class cross-reference (CoV-AbDab) | ✅ present, not central to the trained model |
| `src/data/structure_builder.py` | PDB → residue contact graph (BioPython + freesasa) | ✅ builds real graphs for 6VXX/6VSB/7BNN |
| `src/data/msa_align.py` | Multiple sequence alignment utility | present, not deeply inspected in this audit |
| `src/data/temporal_split.py` | Date-based train/val/test split with leakage assertion | ✅ verified, see §4 |
| `src/features/esm2_embed.py` | ESM-2 embedding extraction (Kaggle/Colab, `--local` fallback) | 🔴 see Finding 1 — small model used, not the documented large one |
| `src/features/gvp_features.py` | GVPConv layer + graph loader | ✅ implements Jing et al. 2021 reasonably faithfully |
| `src/models/sequence_encoder.py` | Loads cached ESM-2 embeddings | 🔴 fallback dim is 480, confirming Finding 1 |
| `src/models/structure_encoder.py` | 3-layer GVP-GNN stack | ✅ implemented |
| `src/models/fusion.py` | Cross-attention fusion | ✅ implemented, standard Q/K/V multi-head attention |
| `src/models/heads.py` | HotspotHead / EscapeHead | 🟠 hardcoded domain-prior logit offsets dominate output — Finding 2 |
| `src/models/joint_model.py` | Wires encoder→fusion→heads | ✅ implemented |
| `src/train.py` | Real training loop on cached embeddings | 🟡 implemented, but the **per-example labels are a single fixed vector** — Finding 3 |
| `src/eval.py` | Prospective evaluation on temporal test split | 🟡 implemented but "true mutation positions" are hardcoded/"illustrative"; contains a harmless dead-code bug (`y_escape_all` referenced before assignment, masked by a fallback branch) |
| `src/ablations.py` | Seq-only / Struct-only / Concat / Full-fusion comparison | ✅ implemented, never actually run on the real pipeline (no output artifact found) |
| `src/risk_score.py` | `R = P × S` and top-K ranking | ✅ implemented |
| `scripts/train_well_calibrated_model.py` | **Actually produced `checkpoints/best_model.pt`** | 🔴 trains on `torch.randn(...)` synthetic noise, not real embeddings — Finding 4 |
| `scripts/fast_train_checkpoint.py` | Alternate 1-epoch synthetic calibration script | 🔴 same issue, lighter version |
| `scripts/export_residue_database.py` | Builds `reports/residue_database.json` for the website | 🔴 never calls the model; 14 headline sites are hand-typed constants, the rest are formula + random noise — Finding 6 |
| `tests/test_1000_cases.py` | Produces `reports/test_1000_results.json` | 🔴 "1,000 cases" = residue-position cycling over 2 real sequences; classification ground truth reuses the training hotspot set — Finding 5 |
| `data/processed/*` | Cached embeddings, contact graphs, DMS tensors, splits | ✅ real, verified on disk (counts in §3) |
| `checkpoints/*.pt` | 21 checkpoint files (`best_model.pt`, `ckpt_epoch{5..100}.pt`) | Only `best_model.pt` was inspected in depth; it is the synthetic-noise-calibrated one (Finding 4) |
| `notebooks/Colab_Viral_Escape_Prediction.ipynb` | Colab workflow, not inspected line-by-line in this pass | — |
| `reports/*.pptx`, `reports/index.html` | Prior decks and the interactive 3Dmol.js residue viewer | Viewer reads `residue_database.json`, i.e. the hand-authored/synthetic data in Finding 6 |

---

## 3. Dataset Facts (directly verified, not estimated)

| Quantity | Verified count | Source |
|---|---|---|
| Cleaned Spike sequences | 1,326 | `data/interim/metadata_clean.tsv` |
| Train sequences | 756 | `data/processed/train.csv` |
| Validation sequences | 134 | `data/processed/val.csv` |
| **Test sequences** | **2** | `data/processed/test.csv` — one `USA/LA-CDC-2-8366133/2025` (KP.3), one `USA/IVY-G25Z00X2/2025` (XEC) |
| Sites with real Bloom Lab DMS escape values | 178 of 1,273 | `data/processed/dms_escape_aggregated.csv` |
| Per-antibody DMS rows | 43,500 | `data/processed/dms_escape_per_antibody.csv` |
| Real PDB structures used | 3 (6VXX, 6VSB, 7BNN) | `data/processed/contact_graphs/*.pt` |
| ESM-2 embedding dimensionality (actual, on disk) | **480** (not the configured 1,280) | verified via tensor shape and via zip-member byte size: `2,444,160 bytes ÷ 4 bytes/float32 ÷ 1273 residues = 480.0` exactly |
| Epitope classes used | RBD class 1–4, NTD supersite (5 classes) | `configs/data.yaml` |

**Reading this table honestly:** the engineering to source, clean, and split real sequence, structural, and experimental-escape data is genuine and competently done. The crack is downstream — in how the model was actually trained and evaluated.

---

## 4. What Genuinely Works

1. ✅ **Temporal split with a hard anti-leakage assertion.** `src/data/temporal_split.py::assert_no_leakage` checks that train/val dates never exceed the 2023-12-31 cutoff, test dates never precede it, and that **no strain ID appears in both pools** — and raises an `AssertionError` if any check fails. This is good practice that many student (and professional) projects skip.
2. ✅ **Real external data integration.** Live download of Bloom Lab DMS data; three real cryo-EM Spike structures parsed into proper contact graphs with SASA (via `freesasa`), backbone dihedral angles, and Cα–Cα distance-thresholded edges (10 Å).
3. ✅ **A legitimate architecture.** Frozen ESM-2 → GVP-GNN → cross-attention fusion → dual heads is a coherent, literature-grounded multimodal design (see §6).
4. ✅ **An ablation harness already exists** (`src/ablations.py`) comparing sequence-only, structure-only, naive-concatenation, and full cross-attention-fusion variants — exactly the comparison needed to justify the fusion design, once it is run on valid inputs.
5. ✅ **GVP-GNN implementation** follows Jing et al. (ICLR 2021): separate scalar/vector channels, vector pathway via rotation-equivariant linear maps, scalar pathway folding in vector norms.

---

## 5. The Six Verified Discrepancies (full detail)

### Finding 1 — Embedding dimension mismatch (documented 1,280-dim / 650M model vs. actual 480-dim / 35M model)

- `configs/model.yaml`: `model_name: "esm2_t33_650M_UR50D"`, `output_dim: 1280`.
- `src/features/esm2_embed.py`: defines both `ESM2_MODEL_LARGE = "esm2_t33_650M_UR50D"` (used when **not** `--local`) and `ESM2_MODEL_SMALL = "esm2_t12_35M_UR50D"` (used when `--local`).
- `src/models/sequence_encoder.py:34`: zero-fallback is `torch.zeros((1273, 480))` — the author's own fallback dimension is 480, i.e. the small model's output size.
- Direct verification performed in this audit: unzipped one cached embedding `.pt` file and inspected its raw tensor payload size — `2,444,160 bytes`. At float32 (4 bytes) and 1,273 residues, that is exactly `2444160 / 4 / 1273 = 480.0` dimensions per residue. **Confirmed: the cached embeddings were extracted with the small, local 35M-parameter model, not the documented 650M model.**
- **Consequence:** any report or slide that states "650-million-parameter ESM-2 embeddings" is factually incorrect for the artifacts actually produced. The small model has far less representational capacity, and this should be corrected or the large-model extraction should actually be run (see Roadmap).

### Finding 2 — Hardcoded domain-prior logit offsets dominate the heads

`src/models/heads.py`, both `HotspotHead` and `EscapeHead`, add a **constant** (not learned) offset to the MLP's output logit, keyed purely on residue position, before the sigmoid:

```python
domain_prior = torch.where(
    (indices >= 319) & (indices <= 541), 3.2,   # RBD
    torch.where((indices >= 14) & (indices <= 305), 1.8,   # NTD
    torch.where((indices >= 681) & (indices <= 685), 3.5,  # Furin site
    -3.5)))                                                 # everything else
p_mutate = torch.sigmoid((logits + domain_prior) * 1.5)
```

A freshly-initialized MLP (`std=0.02` or Kaiming-normal, per `train_well_calibrated_model.py`) produces logits on the order of 0.1–1.0 in magnitude; the domain prior is 1.8–3.5 before the ×1.5 scaling. **The output is therefore driven almost entirely by which of four fixed regions a residue sits in, not by anything learned from sequence or structure.** This is the mechanistic explanation for the previously-logged "known weakness" (hardcoded domain-prior offsets dominating the learned representation).

### Finding 3 — Training labels are one fixed vector, repeated with noise

`src/train.py::SpikeDataset.__getitem__`:

```python
self.hotspot_sites = {484, 456, 501, 452, 704, 486, 346, 455, 417, 681, 614, 142, 339, 478}
...
y_mutate = torch.zeros(1273); for st in self.hotspot_sites: y_mutate[st-1] = 1.0
noise = torch.randn(1273) * 0.05
y_mutate = torch.clamp(y_mutate + noise, 0, 1)
y_escape = torch.clamp(self.dms_escape.clone() + noise * 0.1, 0, 1)
```

Every training example — regardless of the real strain it represents — gets the **same** target vector, varied only by small Gaussian noise. There is no per-sequence, per-mutation supervision; the model can reduce its training loss substantially just by reproducing one fixed, slightly-perturbed profile. This compounds with Finding 2: the hardcoded prior already encodes almost exactly this same fixed profile.

### Finding 4 — The deployed checkpoint was calibrated on synthetic noise, not real data

`scripts/train_well_calibrated_model.py` — the script whose own docstring states its purpose is to ensure "outputs have healthy standard deviation (> 0.15)" — is what actually wrote `checkpoints/best_model.pt`:

```python
for epoch in range(100):
    h_seq = torch.randn(4, 1273, 480, device=device) * 0.1   # NOT a real ESM-2 embedding
    p_mut, s_esc = model(h_seq, node_scalars, node_vectors, edge_index, edge_scalars)
    loss = bce_loss(p_mut, y_mut_target.expand(4,-1)) + mse_loss(s_esc, y_esc_target.expand(4,-1))
    ...
torch.save({"model_state_dict": model.state_dict()}, "checkpoints/best_model.pt")
```

`scripts/fast_train_checkpoint.py` is a lighter 50-step variant of the same approach (also random `h_seq`, also overwrites `best_model.pt`). **The real `src/train.py` training loop — which does load genuine cached ESM-2 embeddings via `SequenceEncoder` and the real `train.csv`/`val.csv` splits — exists and is correctly written, but it is not what produced the checkpoint that downstream evaluation and the website actually use.**

### Finding 5 — "1,000 test cases" is residue cycling over 2 real sequences, with circular ground truth

`tests/test_1000_cases.py`:

```python
strains = df_test["strain"].unique()[:10]       # test.csv has exactly 2 rows
for idx in range(1000):
    site   = (idx % 1273) + 1                   # cycles residue positions 1..1273, twice, plus a remainder
    strain = strains[idx % len(strains)]          # cycles between the SAME 2 strains
    is_known_hotspot = site in hotspot_set         # the SAME hardcoded set used as the TRAINING target
```

The reported `AUROC = 0.7357` and `Precision@10 = 0.10` classify residue **positions** of 2 real sequences against the identical hardcoded hotspot set the checkpoint was calibrated on (Finding 4) — this is label leakage by construction, not an independent prospective test. The reported `AUPRC = 0.0368` is the more trustworthy number here: under this much class imbalance, a near-chance AUPRC alongside a moderate AUROC is the classic signature of a classifier whose apparent skill comes from the evaluation construction, not genuine ranking ability. The code additionally defines **hardcoded fallback metric values** (`auroc=0.8950`, `auprc=0.8120`, `spearman_rho=0.8420`) used only if the real computation fails — not triggered in the run that produced the current `test_1000_results.json`, but a latent reproducibility/integrity risk if the script is re-run under different conditions.

### Finding 6 — The interactive website's risk scores are mostly hand-entered, not model inference

`scripts/export_residue_database.py` — which generates `reports/residue_database.json` and `reports/residue_data.js`, consumed by the 3Dmol.js interactive viewer in `reports/index.html` — **never imports or calls `JointMutationEscapeModel`.** Instead:

```python
priority_sites = {
    484: (0.942, 0.915, 0.862), 456: (0.918, 0.890, 0.817), 501: (0.895, 0.880, 0.788),
    ... # 14 sites total, hand-typed plausible-looking (p_mutate, s_escape, risk) triples
}
for i in range(1, 1274):
    if i in priority_sites:
        p_mut, s_esc, risk = priority_sites[i]
    else:
        p_mut = dms_val * 0.7 + np.random.uniform(0.15, 0.45)   # RBD: formula + random noise
        ...
```

The 14 "headline" hotspot residues shown on the public-facing viewer are manually authored constants; the remaining 1,259 residues are a domain-dependent formula blended with `np.random.uniform(...)` noise. **The website should be described as an illustrative/demo visualization, not a live deployment of the trained model.**

---

## 6. Literature Grounding (verified citations)

1. Lin, Z., Akin, H., Rao, R., et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* **Science**, 379(6637), 1123–1130. PMID: 36927031. — introduces the ESM-2 family the project's sequence encoder is specified to use.
2. Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2021). *Learning from Protein Structure with Geometric Vector Perceptrons.* **ICLR 2021**. arXiv:2009.01411. — the architecture `src/features/gvp_features.py` implements.
3. Greaney, A. J., Starr, T. N., Bloom, J. D., et al. (2021). *Complete Mapping of Mutations to the SARS-CoV-2 Spike Receptor-Binding Domain that Escape Antibody Recognition.* **Cell Host & Microbe**, 29(3), 463–476. — the DMS methodology behind the Bloom Lab escape data this project downloads and aggregates.
4. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* **NeurIPS 30.** — the multi-head attention mechanism underlying `src/models/fusion.py`.
5. Bloom Lab data repository: `github.com/jbloomlab/SARS2_RBD_Ab_escape_maps` — the exact source URL hardcoded in `src/data/dms_parser.py`.
6. RCSB PDB entries 6VXX, 6VSB, 7BNN — the three real cryo-EM Spike structures parsed by `src/data/structure_builder.py`.

**Why the design is reasonable despite the execution issues:** ESM-2 captures evolutionary/co-occurrence statistics with no explicit 3D geometry; GVP-GNN captures 3D spatial proximity with no inherent evolutionary context; cross-attention is a standard, well-tested way to let one modality's representation be reweighted by the other's context. Combining them for a structure-aware escape-prediction task is a defensible idea independent of how this particular repository executed it.

---

## 7. Honest Results Table

| Metric | Reported value | Honest interpretation |
|---|---|---|
| AUROC (hotspot classification) | 0.7357 | Inflated by circular, hardcoded ground truth (Finding 5) — not evidence of real generalization |
| AUPRC (hotspot classification) | 0.0368 | The more informative number: close to chance under this label imbalance, and it contradicts the AUROC headline |
| Precision@10 | 0.10 | Only 1 of the top-10 predicted residues matched the (circular) hotspot set |
| Spearman ρ (escape vs. DMS) | 0.3415 | Weak-to-moderate; computed on the synthetic-noise-calibrated checkpoint (Finding 4), not the real pipeline |
| MSE (escape) | 0.0326 | Low in absolute terms, but scores are bounded in [0,1] so this is not surprising |
| Output non-constancy check | Passed (std > thresholds) | Confirms outputs are not collapsed to a single constant — a real but very weak bar |

**Bottom line for the viva:** present these numbers, but lead with the caveat, not the number. An examiner who finds the circularity independently and sees it omitted from the presentation will trust nothing else in the deck; an examiner who sees it flagged first will generally treat it as evidence of rigor.

---

## 8. Limitations

- **Biological scope:** DMS escape data covers only 178 of 1,273 sites and 5 pre-defined epitope classes; the model cannot be meaningfully evaluated on escape from antibodies or epitopes outside that panel.
- **Methodological:** hardcoded priors (Finding 2) and fixed training targets (Finding 3) mean the checkpoint has not been shown to generalize beyond four pre-defined structural regions.
- **Data size:** a 2-sequence held-out test set cannot support any statistically meaningful generalization claim.
- **Structural features are static:** the same 3 PDB-derived contact graphs are reused for every sequence; the model never observes how a specific mutation locally perturbs 3D structure.
- **No uncertainty quantification:** no confidence intervals, cross-validation, or multi-seed variance reported for any metric.
- **No independent prospective validation:** the "true mutation positions" used in `src/eval.py` for Precision@K are explicitly commented as "Illustrative true site set," i.e., not sourced from a verified external database of confirmed JN.1/KP.2/KP.3/XEC mutations.

---

## 9. Prioritized Roadmap

| Priority | Action | Addresses |
|---|---|---|
| 1 | Replace the hardcoded domain-prior offsets with a small, learnable, lower-magnitude biophysical prior gate (the previously-identified top novelty item) | Finding 2 |
| 2 | Re-extract ESM-2 embeddings at the documented 650M/1,280-dim setting and re-run `src/train.py` end-to-end on them; save the resulting checkpoint as the one actually reported | Findings 1 & 4 |
| 3 | Build an independent, non-circular ground-truth set of confirmed JN.1/KP.2/KP.3/XEC Spike mutations (e.g., from Nextstrain clade-defining mutation lists) and expand the temporal test set beyond 2 sequences | Finding 5 |
| 4 | Move from one fixed per-site target to substitution-level (residue × amino-acid) escape labels | Finding 3 |
| 5 | Run the already-implemented `src/ablations.py` suite on the corrected pipeline and report Seq-only vs. Struct-only vs. Concat vs. Full-fusion | validates the fusion design claim |
| 6 (stretch) | Background/lineage-conditioned prediction for epistasis; ensemble structural encoding across all 3 PDB conformations; multi-cutoff prospective benchmark (Split-A/B/C) | longer-term novelty roadmap |

---

## 10. Viva Preparation — 32 Questions with Evidence-Based Answers

### Biology (1–6)

1. **What is immune escape, precisely?** A mutation that reduces how well a given antibody (or antibody class) neutralizes the virus — measured here as the Bloom Lab DMS "escape fraction" for RBD-targeting antibodies.
2. **How can a single mutation affect antibody binding?** By changing a side chain within or near an epitope's contact surface, altering shape, charge, or hydrogen-bonding — e.g. N501Y swaps a small polar Asn for a bulky aromatic Tyr at the ACE2/antibody-contact ridge.
3. **Why doesn't a mutation always imply functional change?** Most point mutations are neutral with respect to protein function; only mutations at structurally/immunologically consequential positions (interfaces, epitopes, catalytic or binding residues) are expected to matter, which is why DMS assays test mutations empirically rather than relying on position alone.
4. **What's the difference between an epitope and a binding site?** An epitope is the surface an *antibody* contacts; a binding site more generally can refer to any functional contact surface (e.g. the ACE2-receptor interface). They overlap at some RBD positions but are conceptually distinct.
5. **Why is escape not the same as transmissibility, virulence, or drug resistance?** These are different selective pressures acting on different molecular mechanisms (antibody affinity vs. receptor affinity vs. replication fitness vs. drug target binding); a mutation can change one without the others.
6. **What experimental evidence would be needed to confirm immune escape in a new variant, beyond this model's prediction?** Direct neutralization assays (e.g., pseudovirus or live-virus neutralization with human sera/monoclonal antibodies) — this project's prediction is a triage/ranking signal for prioritizing which mutations deserve that wet-lab follow-up, not a substitute for it.

### Databases & Data (7–12)

7. **Why NCBI Virus + Nextstrain instead of GISAID?** Both are fully public with no data-use agreement or registration, keeping the pipeline reproducible and redistributable — GISAID requires an access agreement.
8. **What exactly does the Bloom Lab DMS dataset contribute?** Per-residue, per-antibody-condition escape fractions from deep mutational scanning, aggregated here to a top-decile-per-site summary (`dms_parser.py::aggregate_escape`) used as the escape-head's supervision target.
9. **Why use a structure database (PDB) in addition to a sequence database?** Sequence alone cannot tell you which residues are spatially close in the folded protein; GVP-GNN needs real 3D contact information that only structural data provides.
10. **How was data provenance maintained?** Sequences are filtered to full-length Spike (1273±5 aa), deduplicated, discarded if >5% ambiguous residues, and tagged with collection date/lineage from Nextstrain metadata; DMS data is downloaded from a fixed, hardcoded source URL for reproducibility.
11. **What's the difference between the per-antibody and aggregated DMS files?** `dms_escape_per_antibody.csv` (43,500 rows) retains escape values per individual antibody/condition; `dms_escape_aggregated.csv` (178 rows) reduces this to one top-decile summary value per site, which is what the model actually trains against.
12. **Why were only 3 PDB structures used, and is that a limitation?** 6VXX, 6VSB, 7BNN represent closed, one-RBD-up, and RBD-open conformations — a reasonable minimal set, but yes, it is a limitation: no AlphaFold-predicted mutant-specific structures are used, so the model cannot see mutation-induced structural change.

### Machine Learning & Methodology (13–22)

13. **Why fuse sequence and structure with cross-attention rather than concatenation?** Cross-attention lets each residue's sequence representation be adaptively reweighted by the structural context relevant to it, rather than forcing a single fixed linear combination — `ablations.py` exists specifically to test whether this beats naive concatenation, but has not yet been run on valid inputs.
14. **What are protein language model embeddings?** Fixed-length vector representations per residue, produced by a model (ESM-2) trained via masked-language-modeling on millions of protein sequences, capturing evolutionary and contextual patterns.
15. **Was ESM-2 fine-tuned in this project?** No — it is fully frozen; embeddings are extracted once and cached to disk; only the GVP-GNN, fusion layer, and two prediction heads receive gradients.
16. **What does the model actually predict, per residue?** Two numbers: `P(mutate)` (hotspot probability, BCE-trained) and `S(escape)` (continuous escape score, MSE-trained against aggregated DMS values).
17. **How were the labels generated, and what's the weakness there?** From a fixed 14-site hardcoded hotspot set and the aggregated DMS vector, applied identically (plus small noise) to every training example — meaning the model has very little sequence-specific signal to learn from (Finding 3).
18. **What would data leakage look like in this pipeline, and did you find any?** Yes — the "1,000 test case" evaluation (Finding 5) classifies against the same hardcoded hotspot set used in training, which is a textbook case of circular/leaked evaluation.
19. **Why are AUROC, AUPRC, Spearman ρ, and Precision@K the right metrics here?** AUROC/AUPRC suit imbalanced binary hotspot classification (AUPRC is the more trustworthy one under heavy imbalance); Spearman ρ suits a continuous, potentially nonlinearly-scaled escape score; Precision@K reflects the actual use case (ranking a short list of residues for follow-up).
20. **What is a GVP-GNN layer, intuitively?** A graph-convolution layer that processes scalar features (SASA, dihedral angles) with ordinary MLPs and 3D direction vectors (backbone orientation) with rotation-equivariant linear maps, so representations don't change arbitrarily if the structure is rotated in space.
21. **What would an ablation study here need to show to justify the fusion architecture?** That the full cross-attention model (variant 4 in `ablations.py`) outperforms sequence-only, structure-only, and naive-concatenation baselines on the *real*, non-circular test data — currently unverified because the suite has not been run on valid inputs.
22. **If you had another semester, what would you change first?** Follow the roadmap in §9 in order: replace hardcoded priors with a learnable gate, retrain end-to-end on real embeddings, build a non-circular evaluation set, move to substitution-level labels, then run the existing ablation suite.

### Results & Scientific Honesty (23–28)

23. **What is the single strongest, most defensible finding of this project?** That a careful code audit — not assumption — surfaced six specific, verifiable discrepancies between documentation and implementation, each traceable to an exact file and line; that kind of rigor is itself a demonstrable skill, independent of the model's current accuracy.
24. **What do the reported AUROC and AUPRC actually mean together?** AUROC (0.7357) looks respectable in isolation; AUPRC (0.0368) on the *same* labels reveals that the apparent skill is largely an artifact of circular, imbalanced evaluation rather than genuine discriminative power.
25. **How do you know whether the model generalizes to unseen variants?** Currently, there is no evidence that it does — the held-out test set is 2 sequences, and the classification ground truth reuses the training label set. State this directly if asked, rather than defending the number.
26. **What is genuinely novel about this project, separate from its execution issues?** The specific combination of a frozen large protein language model, an equivariant structural GNN, cross-attention fusion, and a leakage-checked *temporal* (not random) split, applied to this exact dual-head mutation/escape task — the architecture pattern itself draws on established 2017–2023 literature rather than inventing new components, but the combination and the temporal-generalization framing for this task is a reasonable contribution once properly executed.
27. **What experiment would provide the strongest validation?** Re-running `eval.py` and `ablations.py` on a checkpoint trained end-to-end on real cached embeddings, evaluated against an independently sourced, non-circular set of confirmed JN.1/KP.2/KP.3/XEC Spike mutations.
28. **Why report negative findings about your own project instead of only the headline metrics?** Because an examiner who independently discovers an unreported issue will discount the entire presentation, while proactively identifying and explaining limitations demonstrates exactly the scientific judgment a viva is meant to assess.

### Research Evaluation (29–32)

29. **How does this project compare to existing published approaches to escape prediction?** It follows the same general multimodal-fusion pattern increasingly used in recent protein-ML literature (language model + structure), but this specific repository has not yet produced a validated, apples-to-apples comparison against a published baseline — that comparison is future work.
30. **What's the biggest single methodological risk going forward?** Re-running training on real embeddings without also fixing the hardcoded domain priors (Finding 2) — the learned signal could still be swamped by the fixed offsets even with correct inputs.
31. **Is the risk score `R = P × S` the right way to combine the two predictions?** It is a reasonable, interpretable first choice (both terms in [0,1], so R is naturally in [0,1] too, and zero in either term zeroes the risk) — but it assumes independence between "will mutate" and "would escape," which is a simplification worth stating explicitly.
32. **What would you say if an examiner asks "so does your model actually work?"** Directly: the architecture is sound and the data-engineering layer is real, but the specific checkpoint and metrics shown today are a calibration/demo artifact rather than a validated end-to-end result — and here is the exact roadmap to get a trustworthy answer to that question.

---

*Document prepared from direct inspection of the repository at* `C:\Users\patel\Desktop\biodb_1\viral-escape-predictor` *(and sibling files), October 2026. No experimental results, datasets, or citations in this document were fabricated; every quantitative claim traces to a specific file verified during this audit.*
