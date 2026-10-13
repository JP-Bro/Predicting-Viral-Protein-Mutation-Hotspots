"""
structure_builder.py
--------------------
Builds residue-contact graphs from PDB structures for GVP-GNN input.

Runs LOCALLY — ~50-100 PDB structures take only 5-15 minutes total.
No Kaggle needed for this step.

For each PDB:
  - Nodes: one per residue
  - Edges: connect residues i,j if Cα–Cα distance < 10 Å
  - Node scalar features: SASA, secondary structure, sin/cos of φ and ψ dihedrals
  - Node vector features: backbone orientation (N→CA, CA→C, CA→CB unit vectors)
  - Edge scalar features: Cα–Cα distance, sequence separation

Output per PDB:
  data/processed/contact_graphs/<pdb_id>.pt
  Each .pt is a dict with keys:
    node_scalars  (L, 10)
    node_vectors  (L, 3, 3)
    edge_index    (2, E)
    edge_scalars  (E, 2)
    seq_length    int
    pdb_id        str

Usage:
    python src/data/structure_builder.py

Install freesasa first:
    pip install freesasa
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

try:
    from Bio import PDB
    from Bio.PDB import DSSP, PDBParser
    BIOPYTHON_OK = True
except ImportError:
    BIOPYTHON_OK = False
    log.warning("BioPython not installed. Run: pip install biopython")

try:
    import freesasa
    FREESASA_OK = True
except ImportError:
    FREESASA_OK = False
    log.warning("freesasa not installed. Run: pip install freesasa")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDB_DIR    = Path("data/raw/pdb")
GRAPH_DIR  = Path("data/processed/contact_graphs")
CONTACT_THRESHOLD = 10.0   # Cα–Cα Angstroms

# Reference Spike PDB structures to build graphs for
# Add more as needed — one graph per unique reference structure
PDB_IDS = [
    "6VXX",   # SARS-CoV-2 Spike trimer (closed)
    "6VSB",   # SARS-CoV-2 Spike trimer (one RBD up)
    "7BNN",   # Spike with RBD in open conformation
]

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

SS_ENCODING = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7}


# ---------------------------------------------------------------------------
# Download PDB
# ---------------------------------------------------------------------------
def download_pdb(pdb_id: str) -> Path:
    dest = PDB_DIR / f"{pdb_id}.pdb"
    if dest.exists():
        log.info(f"  Already have {pdb_id}.pdb")
        return dest

    PDB_DIR.mkdir(parents=True, exist_ok=True)
    url = RCSB_URL.format(pdb_id=pdb_id)
    log.info(f"  Downloading {pdb_id} from RCSB ...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            dest.write_bytes(r.read())
        log.info(f"  Saved → {dest}")
    except Exception as e:
        log.error(f"  Failed to download {pdb_id}: {e}")
        return None
    return dest


# ---------------------------------------------------------------------------
# Extract Spike chain residues
# ---------------------------------------------------------------------------
def get_spike_residues(structure, chain_id: str = "A") -> list:
    """
    Returns list of residues from the specified chain (Spike chain A in most PDBs).
    Filters to standard amino acids only (skips HETATM).
    """
    standard_aa = set([
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
        "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
        "THR","TRP","TYR","VAL",
    ])
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        # Try first available chain
        chain_id = list(model.get_chains())[0].id
        log.info(f"  Chain A not found, using chain {chain_id}")

    residues = [
        r for r in model[chain_id].get_residues()
        if r.resname in standard_aa
    ]
    return residues


# ---------------------------------------------------------------------------
# Compute dihedral angles
# ---------------------------------------------------------------------------
def dihedral(p0, p1, p2, p3) -> float:
    """Praxis–Newell dihedral in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def get_backbone_atoms(res) -> dict:
    """Return N, CA, C, CB atom coordinates (CB = CA if Gly)."""
    atoms = {}
    for atom_name in ["N", "CA", "C", "CB"]:
        if res.has_id(atom_name):
            atoms[atom_name] = res[atom_name].get_vector().get_array()
        elif atom_name == "CB" and res.has_id("CA"):
            atoms["CB"] = res["CA"].get_vector().get_array()  # Gly fallback
    return atoms


# ---------------------------------------------------------------------------
# Build node features
# ---------------------------------------------------------------------------
def build_node_features(
    residues: list,
    dssp_dict: dict | None,
    sasa_vals: list | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      node_scalars: (L, 10) — [SASA, ss_onehot(8), phi_sin, phi_cos, psi_sin, psi_cos]
                    Wait — let's be exact: [SASA, phi_sin, phi_cos, psi_sin, psi_cos,
                                            ss_0..ss_7] truncated to 10 dims
      node_vectors: (L, 3, 3) — [N→CA, CA→C, CA→CB] unit vectors per residue
    """
    L = len(residues)
    node_scalars = np.zeros((L, 10), dtype=np.float32)
    node_vectors = np.zeros((L, 3, 3), dtype=np.float32)

    for i, res in enumerate(residues):
        atoms = get_backbone_atoms(res)
        ca = atoms.get("CA")

        # --- Scalar features ---
        # SASA (normalised 0-1, max typical ~200 Å²)
        if sasa_vals is not None and i < len(sasa_vals):
            node_scalars[i, 0] = min(sasa_vals[i] / 200.0, 1.0)

        # Dihedral angles φ, ψ
        phi, psi = 0.0, 0.0
        if i > 0 and i < L - 1:
            prev_atoms = get_backbone_atoms(residues[i - 1])
            next_atoms = get_backbone_atoms(residues[i + 1])
            if all(k in prev_atoms for k in ["C"]) and all(k in atoms for k in ["N","CA","C"]):
                try:
                    phi = dihedral(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"])
                except Exception:
                    phi = 0.0
            if all(k in atoms for k in ["N","CA","C"]) and "N" in next_atoms:
                try:
                    psi = dihedral(atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"])
                except Exception:
                    psi = 0.0

        node_scalars[i, 1] = np.sin(phi)
        node_scalars[i, 2] = np.cos(phi)
        node_scalars[i, 3] = np.sin(psi)
        node_scalars[i, 4] = np.cos(psi)

        # Secondary structure (one-hot, 5 dims, truncated)
        if dssp_dict is not None:
            res_key = (res.get_parent().id, res.get_id())
            ss_char = dssp_dict.get(res_key, ("-", None, None, None, None, None, None, None, None))[2]
            ss_idx  = SS_ENCODING.get(ss_char, 7)
        else:
            ss_idx = 7

        if 5 + ss_idx < 10:
            node_scalars[i, 5 + ss_idx] = 1.0

        # --- Vector features ---
        n  = atoms.get("N",  ca)
        c  = atoms.get("C",  ca)
        cb = atoms.get("CB", ca)

        if ca is not None and n is not None:
            v = n - ca
            norm = np.linalg.norm(v)
            node_vectors[i, 0] = v / norm if norm > 1e-8 else v

        if ca is not None and c is not None:
            v = c - ca
            norm = np.linalg.norm(v)
            node_vectors[i, 1] = v / norm if norm > 1e-8 else v

        if ca is not None and cb is not None:
            v = cb - ca
            norm = np.linalg.norm(v)
            node_vectors[i, 2] = v / norm if norm > 1e-8 else v

    return node_scalars, node_vectors


# ---------------------------------------------------------------------------
# Build edges (Cα–Cα < 10 Å)
# ---------------------------------------------------------------------------
def build_edges(residues: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      edge_index: (2, E) int64
      edge_scalars: (E, 2) float32  [distance, sequence_separation_normalised]
    """
    ca_coords = []
    for res in residues:
        if res.has_id("CA"):
            ca_coords.append(res["CA"].get_vector().get_array())
        else:
            ca_coords.append(np.zeros(3))

    ca_coords = np.array(ca_coords)   # (L, 3)
    L = len(ca_coords)

    src_list, dst_list, edge_feat_list = [], [], []

    for i in range(L):
        for j in range(i + 1, L):
            dist = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            if dist < CONTACT_THRESHOLD:
                seq_sep = abs(i - j) / max(L, 1)
                feat = [dist / CONTACT_THRESHOLD, seq_sep]   # normalise distance
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_feat_list.extend([feat, feat])

    if not src_list:
        log.warning("  No edges found — structure may be incomplete.")
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    edge_index   = np.array([src_list, dst_list], dtype=np.int64)
    edge_scalars = np.array(edge_feat_list, dtype=np.float32)
    return edge_index, edge_scalars


# ---------------------------------------------------------------------------
# Compute SASA via freesasa
# ---------------------------------------------------------------------------
def compute_sasa(pdb_path: Path) -> list | None:
    if not FREESASA_OK:
        return None
    try:
        structure = freesasa.Structure(str(pdb_path))
        result    = freesasa.calc(structure)
        # Returns per-atom SASA; we sum per residue (chain A only)
        sasa_per_res = {}
        for i in range(structure.nAtoms()):
            res_id = (structure.chainLabel(i), structure.residueNumber(i))
            sasa_per_res[res_id] = sasa_per_res.get(res_id, 0.0) + result.atomArea(i)
        return list(sasa_per_res.values())
    except Exception as e:
        log.warning(f"  freesasa failed: {e} — SASA set to 0.")
        return None


# ---------------------------------------------------------------------------
# Main pipeline per PDB
# ---------------------------------------------------------------------------
def build_graph(pdb_path: Path, pdb_id: str) -> dict | None:
    if not BIOPYTHON_OK:
        log.error("BioPython required. pip install biopython")
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        log.error(f"  Failed to parse {pdb_path}: {e}")
        return None

    residues = get_spike_residues(structure)
    if not residues:
        log.warning(f"  No residues found in {pdb_id}")
        return None

    L = len(residues)
    log.info(f"  {pdb_id}: {L} residues")

    # DSSP (secondary structure)
    dssp_dict = None
    try:
        dssp = DSSP(structure[0], str(pdb_path), dssp="mkdssp")
        dssp_dict = {k: v for k, v in dssp.property_dict.items()}
    except Exception as e:
        log.warning(f"  DSSP failed ({e}) — secondary structure set to unknown.")

    # SASA
    sasa_vals = compute_sasa(pdb_path)

    # Features
    node_scalars, node_vectors = build_node_features(residues, dssp_dict, sasa_vals)
    edge_index, edge_scalars   = build_edges(residues)

    log.info(f"  Edges: {edge_index.shape[1]} (Cα–Cα < {CONTACT_THRESHOLD} Å)")

    return {
        "node_scalars":  torch.tensor(node_scalars),   # (L, 10)
        "node_vectors":  torch.tensor(node_vectors),   # (L, 3, 3)
        "edge_index":    torch.tensor(edge_index),     # (2, E)
        "edge_scalars":  torch.tensor(edge_scalars),   # (E, 2)
        "seq_length":    L,
        "pdb_id":        pdb_id,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Building contact graphs for {len(PDB_IDS)} PDB structures ...")
    log.info(f"Contact threshold: {CONTACT_THRESHOLD} Å | Output: {GRAPH_DIR}\n")

    success = 0
    for pdb_id in PDB_IDS:
        out_path = GRAPH_DIR / f"{pdb_id}.pt"
        if out_path.exists():
            log.info(f"[SKIP] {pdb_id} — already built at {out_path}")
            success += 1
            continue

        log.info(f"[BUILD] {pdb_id}")
        pdb_path = download_pdb(pdb_id)
        if pdb_path is None:
            continue

        graph = build_graph(pdb_path, pdb_id)
        if graph is None:
            continue

        torch.save(graph, out_path)
        log.info(f"  Saved → {out_path}")
        success += 1

    log.info(f"\nDone — {success}/{len(PDB_IDS)} graphs built in {GRAPH_DIR}")
    log.info("Next step: upload data/interim/cleaned_sequences.fasta to Kaggle for MSA alignment.")


if __name__ == "__main__":
    main()
