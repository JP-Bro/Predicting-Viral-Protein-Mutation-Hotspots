# AI-Driven Prediction of Viral Mutations and Immune-Escape Potential Using Protein Language Models and Structural Features

**A code-verified project report**
Team: Aditya · Vineela · Jimit Patel — VIT Vellore, B.Tech CSE (Bioinformatics)

---

## Abstract

SARS-CoV-2's Spike glycoprotein accumulates mutations under continuous immune pressure, a small fraction of which materially reduce neutralizing-antibody recognition ("immune escape"). This project builds a multimodal deep learning framework that, for each of the 1,273 residues of Spike, jointly predicts (i) the probability the residue becomes a mutation hotspot in future lineages, `P(mutate)`, and (ii) the degree to which a mutation there would help the virus evade antibodies, `S(escape)`, combined into a joint surveillance-priority score `R = P × S`. The architecture fuses a frozen ESM-2 protein language model with a Geometric Vector Perceptron Graph Neural Network (GVP-GNN) encoding residue-contact structure from three real Spike cryo-EM structures, via cross-attention, and is supervised using real Deep Mutational Scanning (DMS) escape data from the Bloom Laboratory. Data engineering — public sequence acquisition (NCBI Virus, Nextstrain), a temporally leakage-checked train/validation/test split, and real structural-graph construction — is implemented and verified. However, a code-level audit of the delivered checkpoint and its reported metrics (AUROC 0.7357, AUPRC 0.0368, Precision@10 = 0.10, Spearman ρ = 0.3415) found that the checkpoint was calibrated on synthetic Gaussian noise rather than the real cached embeddings, that the "1,000-test-case" evaluation reduces to residue-position cycling over two real sequences scored against a circular, hardcoded ground truth, and that the public visualization's risk scores are substantially hand-authored rather than live model inference. This report presents the architecture, the real data pipeline, an honest reading of the reported metrics in light of these findings, and the concrete steps required to produce a trustworthy evaluation of the underlying idea, which remains methodologically sound.

---

## 1. Introduction

### 1.1 Background and Motivation
SARS-CoV-2 is an RNA virus whose RNA-dependent RNA polymerase lacks strong proofreading, so point mutations accumulate steadily as the virus transmits globally. Most such mutations are neutral or deleterious; a small minority occur in the Spike glycoprotein — the primary target of neutralizing antibodies and the ACE2 receptor-binding machinery — and alter receptor binding or antibody recognition. Identifying which Spike positions are likely to mutate next, and how costly such a mutation would be to antibody recognition, is directly relevant to vaccine-strain selection and genomic-surveillance triage.

### 1.2 Problem Statement
Given the Spike protein sequence and its known 3D structure, predict, per residue: (a) the likelihood that residue will carry a mutation in an emerging lineage, and (b) the degree to which a mutation at that residue would reduce neutralizing-antibody recognition, using supervision from empirical Deep Mutational Scanning escape data.

### 1.3 Research Gap
Sequence-only language-model approaches to mutation/escape prediction capture evolutionary and co-occurrence statistics but no explicit 3D geometry; structure-only approaches capture spatial proximity but lack the broader evolutionary context a language model trained on millions of sequences provides. Few accessible, reproducible pipelines combine both modalities with a genuinely temporal (not randomly shuffled) train/test split for this specific dual-task (hotspot + escape) formulation.

### 1.4 Aim and Objectives
**Aim:** Build and evaluate a multimodal deep learning system that jointly predicts Spike mutation hotspots and antibody-escape potential, trained and tested with realistic temporal generalization constraints.

**Objectives:** (1) assemble a reproducible, public-source data pipeline with a leakage-free temporal split; (2) implement a frozen-language-model + structural-GNN + cross-attention architecture; (3) supervise with real DMS escape data; (4) evaluate prospectively on post-cutoff lineages; (5) rank residues by a joint risk score for surveillance use.

### 1.5 Research Questions
1. Can a cross-attention fusion of sequence and structural embeddings produce a usable per-residue mutation/escape ranking?
2. Does fusing sequence and structure outperform either modality alone (addressed by the project's ablation framework)?
3. Does the resulting model generalize to lineages observed only after the training cutoff?

### 1.6 Scope
In scope: Spike-protein-only analysis, RBD/NTD-class escape from the Bloom Lab DMS panel, three reference cryo-EM conformations. Out of scope: wet-lab validation, non-Spike proteins, antibodies outside the DMS panel, and real-time variant monitoring.

### 1.7 Expected Contribution
A reproducible, public-data pipeline and a concrete architecture for this task, together with — as this report documents — a transparent accounting of which parts of that pipeline are currently validated and which require further work before their output can be trusted.

---

## 2. Literature Review

| Study | Year | Objective | Method | Relevance to this project |
|---|---|---|---|---|
| Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* [1] | 2023 | Large-scale protein language modeling | ESM-2 transformer family trained on evolutionary sequence data | Source of the frozen sequence-encoder embeddings this project's architecture specifies |
| Jing, Eismann, Suriana, Townshend & Dror, *Learning from Protein Structure with Geometric Vector Perceptrons* [2] | 2021 | Rotation-equivariant structural representation learning | GVP layers processing scalar and vector node features jointly | Architecture directly implemented by this project's structure encoder |
| Greaney, Starr, Bloom et al., *Complete Mapping of Mutations to the SARS-CoV-2 Spike RBD that Escape Antibody Recognition* [3] | 2021 | Empirical, high-throughput escape mapping | Deep Mutational Scanning (DMS) against polyclonal/monoclonal sera | Source methodology for this project's escape-score supervision labels |
| Vaswani et al., *Attention Is All You Need* [4] | 2017 | General-purpose sequence modeling | Multi-head scaled dot-product attention | Underlies the cross-attention fusion module combining sequence and structure |

**Synthesis.** The literature establishes two independently mature tools — protein language models for evolutionary/contextual signal and equivariant graph networks for 3D structural signal — plus a general, well-tested mechanism (attention) for combining heterogeneous representations. DMS assays provide the closest available ground truth for antibody escape at residue resolution. This project's research gap is methodological integration: a reproducible, public-data, temporally-split pipeline that combines all three for the joint hotspot/escape prediction task, rather than a new modeling primitive. No claim of being the first such combination is made, as a comprehensive prior-art search was outside this audit's scope.

---

## 3. Methodology

### 3.1 Data Collection
Full-length Spike sequences (1,273 ± 5 aa) were obtained from NCBI Virus and the Nextstrain open build, both public with no registration requirement. Sequences were deduplicated and discarded if more than 5% of residues were ambiguous (`X`/gap).

### 3.2 Temporal Train/Validation/Test Split
Sequences collected on or before 2023-12-31 form the train/validation pool (85%/15% random split within that pool); sequences from lineages JN.1, KP.2, KP.3, and XEC collected after that date form the test set. A hard programmatic assertion verifies zero date leakage and zero strain-ID overlap between pools before the splits are saved.

### 3.3 Feature Extraction
**Sequence features:** per-residue embeddings from a frozen ESM-2 model, extracted once and cached to disk (never updated during training).
**Structural features:** residue-contact graphs built from three real Spike cryo-EM structures (PDB 6VXX, 6VSB, 7BNN) — 10 scalar node features (solvent-accessible surface area, secondary structure, sine/cosine of backbone φ/ψ dihedral angles) and 3 vector node features (backbone orientation), with edges between residues within 10 Å Cα–Cα distance.
**Escape labels:** Bloom Lab DMS escape values, downloaded programmatically, normalized per antibody condition, and aggregated to a top-decile-per-site summary across five epitope classes (RBD classes 1–4, NTD supersite).

### 3.4 Model Architecture
1. **Sequence encoder** — frozen protein language model; embeddings loaded from disk.
2. **Structure encoder** — a 3-layer GVP-GNN stack producing a 512-dimensional structural representation per residue.
3. **Cross-attention fusion** — sequence embeddings act as queries, structural embeddings as keys/values, in an 8-head multi-head attention block with a residual connection and layer normalization.
4. **Dual prediction heads** — a 2-layer MLP (GELU activation, dropout) per task, producing `P(mutate)` (sigmoid, trained with binary cross-entropy) and `S(escape)` (sigmoid, trained with mean-squared error).
5. **Joint risk score** — `R = P(mutate) × S(escape)`, used to rank residues for surveillance priority.

### 3.5 Training and Evaluation Design
The loss function is a weighted sum of the hotspot BCE loss and the escape MSE loss. Early stopping and checkpoint saving are keyed on validation Spearman rank correlation. Prospective evaluation computes Spearman ρ, MSE, AUROC, AUPRC, and Precision@K against the temporally held-out lineages, and an ablation suite (sequence-only, structure-only, naive concatenation, full cross-attention fusion) is implemented to isolate the contribution of structural information and of the fusion mechanism specifically.

---

## 4. Databases and Data Sources

| Database / resource | What it provides | Role in this pipeline | Access basis | Verified status |
|---|---|---|---|---|
| **NCBI Virus** | Full-length SARS-CoV-2 genome/protein sequences with metadata | Primary public sequence source | Public API, no registration | Implemented |
| **Nextstrain (open build)** | Pre-filtered, pre-aligned sequences with clade/lineage/date metadata | Primary metadata source for lineage and collection date, which drive the temporal split | Public direct download | Implemented; 1,326 cleaned sequences verified on disk |
| **Bloom Lab DMS escape data** (`SARS2_RBD_Ab_escape_maps`) | Deep Mutational Scanning escape-fraction measurements across antibody conditions | Supervision labels for the EscapeHead | Public GitHub repository, downloaded programmatically | Implemented; 178 of 1,273 sites carry real aggregated values, 43,500 raw per-antibody rows |
| **RCSB Protein Data Bank** (entries 6VXX, 6VSB, 7BNN) | Experimental cryo-EM Spike trimer structures (closed, one-RBD-up, RBD-open) | Source structures for residue-contact graph construction | Public download | Implemented; three real contact graphs built with BioPython + freesasa |
| **CoV-AbDab** | Catalogue of published antibodies against coronaviruses, with epitope-class annotation | Cross-reference for antibody epitope classes | Public database | Parser implemented in the codebase; not central to the final supervision signal used |

All five sources are public and require no paid licence or data-use agreement, which was a deliberate reproducibility choice over alternatives such as GISAID (sequence) that require an access agreement. Sequence and metadata databases alone cannot establish 3D spatial proximity between residues, which is why a structural database (PDB) is used in addition; likewise, structural databases alone carry no information about which mutations have been empirically shown to affect antibody recognition, which is why DMS experimental data (not a computational proxy) supplies the escape-label ground truth.

---

## 5. Analysis

### 5.1 Dataset Composition
Of 1,326 cleaned Spike sequences, 756 fall in the training split, 134 in validation, and only **2** in the temporally held-out test split (one KP.3-lineage strain, one XEC-lineage strain collected in 2025). Real DMS escape values are available for 178 of 1,273 residues (13.9%); the remaining residues have no direct experimental escape measurement in this panel.

### 5.2 Implementation Verification
Direct inspection of the cached embedding tensors and the trained checkpoint produced three material findings relevant to interpreting any result:

1. **Embedding dimensionality.** The project's configuration specifies a 1,280-dimensional embedding from a 650-million-parameter ESM-2 model. Tensor inspection of the cached embedding files on disk shows a 480-dimensional embedding instead, consistent with a smaller, 35-million-parameter ESM-2 variant. This was confirmed by direct byte-size verification of the stored tensors (2,444,160 bytes ÷ 4 bytes/float32 ÷ 1,273 residues = 480 dimensions exactly).
2. **Supervision signal.** The per-residue mutation-hotspot and escape-score training targets are constructed from one fixed set of 14 hardcoded positions and one fixed aggregated DMS vector, applied identically to every training example with small injected noise, rather than being sequence-specific.
3. **Checkpoint provenance.** The checkpoint used for the project's headline evaluation was produced by a calibration script that trains on randomly generated Gaussian tensors standing in for sequence embeddings, explicitly to ensure non-constant output variance, rather than by the project's main training loop operating on the real cached embeddings.

### 5.3 Evaluation Methodology Check
The "1,000-test-case" prospective evaluation draws its 1,000 cases by cycling residue position (1–1,273) across only the two real test-set sequences, and its binary ground truth for AUROC/AUPRC/Precision@K reuses the identical fixed hotspot-position set used to construct the training labels. This construction makes the classification ground truth circular with respect to the model's training signal, rather than an independent test against confirmed JN.1/KP.2/KP.3/XEC mutations.

---

## 6. Results

| Metric | Reported value | Interpretation given the methodology above |
|---|---|---|
| AUROC (hotspot classification) | 0.7357 | Computed against the circular, hardcoded ground-truth label set; not interpretable as independent discriminative performance |
| AUPRC (hotspot classification) | 0.0368 | Near chance level under the panel's class imbalance; directly contradicts the AUROC figure and is the more diagnostic number |
| Precision@10 | 0.10 | 1 of the top-10 ranked residues matched the (circular) reference set |
| Spearman ρ (predicted vs. DMS escape) | 0.3415 | Weak-to-moderate positive rank correlation, computed from the noise-calibrated checkpoint rather than one trained on real cached embeddings |
| Mean squared error (escape) | 0.0326 | Low in absolute terms, consistent with both predictions and targets being bounded in [0, 1] |
| Output non-constancy diagnostic | Passed | Confirms the model does not collapse to a single constant value across residues — a necessary but weak sanity check, not a measure of accuracy |

The ablation study comparing sequence-only, structure-only, naive-concatenation, and full cross-attention-fusion variants is implemented in the codebase but has not been executed against the real (non-synthetic) embedding pipeline, so no comparative result exists yet to support or refute the specific contribution of the fusion mechanism.

---

## 7. Discussion

The results above should be read as characterizing an unfinished validation, not a finished predictive system. The architectural idea — combining a frozen protein language model with an equivariant structural encoder via cross-attention, trained against real DMS escape measurements — is consistent with current practice in multimodal protein modeling and is not in question. What is not yet established is whether *this specific implementation*, trained end-to-end on its own real cached embeddings with non-circular evaluation, actually achieves the reported accuracy or outperforms the simpler baselines its own ablation code is designed to test.

A predicted escape score, even once properly validated, would constitute a ranking signal for prioritizing residues for experimental follow-up — not direct evidence of real-world immune escape, altered transmissibility, or vaccine failure, all of which require independent experimental or epidemiological confirmation. The correlation observed between predicted and DMS-derived escape values reflects statistical association with an experimentally-derived summary statistic, not a demonstrated causal or mechanistic account of why a given mutation escapes antibody recognition.

**Principal limitations:** DMS coverage spans only 178 of 1,273 residues and five pre-defined epitope classes; the temporally held-out test set contains only two sequences, which is insufficient for a statistically meaningful generalization claim; structural features are drawn from three fixed reference conformations reused identically across all sequences, so the model does not observe mutation-induced structural change; and no confidence intervals, cross-validation, or multi-seed variance are reported for any metric.

---

## 8. Conclusion

This project assembled a genuinely reproducible, public-data pipeline — sequence acquisition, a temporally leakage-checked split, real DMS escape-label aggregation, and real structural-graph construction from experimental cryo-EM structures — and implemented a defensible multimodal architecture (frozen protein language model, equivariant structural GNN, cross-attention fusion, dual prediction heads) for joint mutation-hotspot and immune-escape prediction. A direct, code-level audit found that the checkpoint and metrics currently reported were produced through a calibration procedure using synthetic input and a circular evaluation construction, rather than through end-to-end training and independent prospective testing on the real pipeline. The data-engineering and architectural components required to run that proper evaluation already exist in the codebase; what remains is to execute training on the real cached embeddings, replace the fixed per-site supervision with sequence-specific labels, build a non-circular held-out evaluation set, and run the existing ablation suite — at which point the project's central research question (whether sequence–structure fusion improves mutation/escape prediction over either modality alone) can be answered with evidence rather than assumed.

---

## References

[1] Lin, Z., Akin, H., Rao, R., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.

[2] Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2021). Learning from Protein Structure with Geometric Vector Perceptrons. *International Conference on Learning Representations (ICLR 2021)*. arXiv:2009.01411.

[3] Greaney, A. J., Starr, T. N., Bloom, J. D., et al. (2021). Complete Mapping of Mutations to the SARS-CoV-2 Spike Receptor-Binding Domain that Escape Antibody Recognition. *Cell Host & Microbe*, 29(3), 463–476.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.

[5] Bloom Lab, SARS2_RBD_Ab_escape_maps data repository. github.com/jbloomlab/SARS2_RBD_Ab_escape_maps.

[6] RCSB Protein Data Bank, entries 6VXX, 6VSB, 7BNN (SARS-CoV-2 Spike cryo-EM structures).
