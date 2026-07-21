import os
import re
import json
import math
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Residue vocabulary
# --------------------------------------------------------------------------- #
# Exactly these — everything else (ligands, ions, waters, modified residues,
# DNA) is dropped. Modified residues such as MSE are deliberately NOT mapped
# back to their parent; they become gaps.
PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
RNA_RES = {"A", "C", "G", "U"}

PROTEIN_ANCHOR = "CA"
RNA_ANCHOR = "C4'"

# One-letter codes. Protein is upper-case, RNA lower-case, so a protein chain
# can never sequence-match an RNA chain.
_PROTEIN_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
_RNA_1 = {"A": "a", "C": "c", "G": "g", "U": "u"}

KIND_PROTEIN = "P"
KIND_RNA = "R"

DEV_OUTLIER_A = 3.0  # deviation above which an anchor counts as an outlier


def residue_kind(resname: str) -> Optional[str]:
    """'P', 'R', or None if the residue is not a standard protein/RNA monomer."""
    if resname in PROTEIN_RES:
        return KIND_PROTEIN
    if resname in RNA_RES:
        return KIND_RNA
    return None


def one_letter(resname: str) -> Optional[str]:
    if resname in _PROTEIN_1:
        return _PROTEIN_1[resname]
    return _RNA_1.get(resname)


def anchor_name(kind: str) -> str:
    return PROTEIN_ANCHOR if kind == KIND_PROTEIN else RNA_ANCHOR


# --------------------------------------------------------------------------- #
# Pairing files across the two directories
# --------------------------------------------------------------------------- #
# Experimental stems look like "11gi"; AF3 stems like "fold_11gi_model_0".
_AF_STRIP = re.compile(r"^fold[_-]|[_-]model[_-]?\d*$|[_-]seed[_-]?\d*$", re.IGNORECASE)


def _af_key(stem: str) -> str:
    """Reduce an AF filename stem to its bare PDB id (best effort)."""
    key = stem.lower()
    prev = None
    while prev != key:  # strip fold_ prefix and _model_N / _seed_N suffixes
        prev = key
        key = _AF_STRIP.sub("", key).strip("_- ")
    return key


def pair_files(pdb_dir: str, af_dir: str) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """Pair experimental and AF cifs by common PDB-id substring in the stem.

    Returns ``({pdb_id: (pdb_path, af_path)}, unpaired_log_lines)``.
    """
    def cifs(d):
        return sorted(f for f in os.listdir(d) if f.lower().endswith((".cif", ".mmcif")))

    pdb_files = {os.path.splitext(f)[0].lower(): os.path.join(pdb_dir, f) for f in cifs(pdb_dir)}
    af_files = {os.path.splitext(f)[0]: os.path.join(af_dir, f) for f in cifs(af_dir)}

    # Exact key match first ("fold_11gi_model_0" -> "11gi"), then containment.
    af_by_key: Dict[str, str] = {}
    for stem, path in af_files.items():
        af_by_key.setdefault(_af_key(stem), path)

    pairs: Dict[str, Tuple[str, str]] = {}
    used_af = set()
    for pdb_id, pdb_path in sorted(pdb_files.items()):
        af_path = af_by_key.get(pdb_id)
        if af_path is None:  # fall back to substring containment
            hits = [p for stem, p in sorted(af_files.items())
                    if pdb_id in stem.lower() and p not in used_af]
            af_path = hits[0] if len(hits) >= 1 else None
        if af_path is None:
            continue
        pairs[pdb_id] = (pdb_path, af_path)
        used_af.add(af_path)

    log = []
    for pdb_id in sorted(set(pdb_files) - set(pairs)):
        log.append(f"[pair] no AlphaFold match for experimental file: {pdb_files[pdb_id]}")
    for stem, path in sorted(af_files.items()):
        if path not in used_af:
            log.append(f"[pair] no experimental match for AlphaFold file: {path}")
    return pairs, log


# --------------------------------------------------------------------------- #
# Structure parsing (model 0, polymer residues only)
# --------------------------------------------------------------------------- #
class Chain:
    """One parsed chain: sequence + anchor coordinates, keyed by author seqid."""

    __slots__ = ("name", "kind", "seq", "anchors", "seqids")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.seq: Dict[int, str] = {}          # seqid -> one-letter code
        self.anchors: Dict[int, np.ndarray] = {}  # seqid -> anchor xyz
        self.seqids: List[int] = []            # sorted, populated on finalise

    def finalise(self) -> "Chain":
        self.seqids = sorted(self.seq)
        return self


def parse_chains(path: str) -> Dict[str, Chain]:
    """Parse model 0 of an mmCIF into ``{chain_name: Chain}`` of polymer residues.

    Coordinates are read via ``atom.pos.x/.y/.z`` (no ordering assumptions).
    A chain's kind is decided by its first standard residue; residues of the
    other kind inside it (rare hybrid chains) are dropped.
    """
    st = gemmi.read_structure(path)
    st.setup_entities()
    if len(st) == 0:
        raise ValueError("structure has no models")
    model = st[0]  # model 0 / first model only

    chains: Dict[str, Chain] = {}
    for ch in model:
        for res in ch:
            kind = residue_kind(res.name)
            if kind is None:
                continue  # ligand, ion, water, modified residue, or DNA
            letter = one_letter(res.name)
            if letter is None:
                continue
            chain = chains.get(ch.name)
            if chain is None:
                chain = chains[ch.name] = Chain(ch.name, kind)
            if chain.kind != kind:
                continue
            seqid = int(res.seqid.num)
            if seqid in chain.seq:
                continue  # microheterogeneity / altloc duplicate: keep the first
            chain.seq[seqid] = letter

            atom = res.find_atom(anchor_name(kind), "*")
            if atom is not None:
                chain.anchors[seqid] = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)

    return {name: c.finalise() for name, c in chains.items() if c.seq}


# --------------------------------------------------------------------------- #
# Unresolved residues
# --------------------------------------------------------------------------- #
def _cat(block, name: str) -> Optional[Dict[str, List[str]]]:
    table = block.get_mmcif_category(name)
    return table if table else None


def _as_int(v) -> Optional[int]:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def read_unobserved(path: str, pdb_chains: Dict[str, Chain]) -> Tuple[Dict[str, List[int]], str]:
    """Depositor-declared unresolved polymer residues: ``{chain: [seqid, ...]}``.

    Authoritative source is ``_pdbx_unobs_or_zero_occ_residues``. Only if that
    category is absent do we derive the set by differencing ``_entity_poly_seq``
    against the observed residues. Returns the map and which path was used
    ('unobs_table' | 'derived').
    """
    block = gemmi.cif.read(path).sole_block()

    unobs = _cat(block, "_pdbx_unobs_or_zero_occ_residues.")
    if unobs:
        missing: Dict[str, List[int]] = {}
        n = len(next(iter(unobs.values())))
        models = unobs.get("PDB_model_num") or [None] * n
        first_model = next((m for m in models if m not in (None, ".", "?")), None)
        for i in range(n):
            if (unobs.get("polymer_flag") or ["Y"] * n)[i] == "N":
                continue
            if first_model is not None and models[i] != first_model:
                continue  # a single model, like the coordinates
            comp = (unobs.get("auth_comp_id") or unobs.get("label_comp_id") or [None] * n)[i]
            chain = (unobs.get("auth_asym_id") or unobs.get("label_asym_id") or [None] * n)[i]
            seqid = _as_int((unobs.get("auth_seq_id") or unobs.get("label_seq_id") or [None] * n)[i])
            if chain is None or seqid is None or residue_kind(str(comp)) is None:
                continue
            missing.setdefault(str(chain), []).append(seqid)
        return {c: sorted(set(v)) for c, v in missing.items()}, "unobs_table"

    return _derive_unobserved(block, pdb_chains), "derived"


def _derive_unobserved(block, pdb_chains: Dict[str, Chain]) -> Dict[str, List[int]]:
    """Fallback: _entity_poly_seq minus the observed residues, per chain.

    Works in label space (entity_poly_seq is label-numbered), then shifts into
    author numbering using the label->auth offset seen in the observed atoms.
    """
    eps = _cat(block, "_entity_poly_seq.")
    atoms = _cat(block, "_atom_site.")
    if not eps or not atoms:
        return {}

    entity_seq: Dict[str, Dict[int, str]] = {}
    for ent, num, mon in zip(eps["entity_id"], eps["num"], eps["mon_id"]):
        n = _as_int(num)
        if n is not None and residue_kind(str(mon)) is not None:
            entity_seq.setdefault(str(ent), {}).setdefault(n, str(mon))

    # Per auth chain: its entity, its observed label seqids, and the label->auth shift.
    obs: Dict[str, Dict[str, object]] = {}
    n_atoms = len(next(iter(atoms.values())))
    models = atoms.get("pdbx_PDB_model_num") or [None] * n_atoms
    first_model = next((m for m in models if m not in (None, ".", "?")), None)
    for i in range(n_atoms):
        if first_model is not None and models[i] != first_model:
            continue
        comp = str((atoms.get("label_comp_id") or atoms.get("auth_comp_id"))[i])
        if residue_kind(comp) is None:
            continue
        auth = str((atoms.get("auth_asym_id") or atoms.get("label_asym_id"))[i])
        ent = str((atoms.get("label_entity_id") or [""] * n_atoms)[i])
        lab_seq = _as_int((atoms.get("label_seq_id") or [None] * n_atoms)[i])
        auth_seq = _as_int((atoms.get("auth_seq_id") or [None] * n_atoms)[i])
        if lab_seq is None or auth_seq is None:
            continue
        rec = obs.setdefault(auth, {"entity": ent, "seen": set(), "shifts": []})
        rec["seen"].add(lab_seq)
        rec["shifts"].append(auth_seq - lab_seq)

    missing: Dict[str, List[int]] = {}
    for auth, rec in obs.items():
        full = entity_seq.get(str(rec["entity"]))
        if not full or not rec["shifts"]:
            continue
        shift = int(np.median(np.asarray(rec["shifts"])))
        gaps = sorted(set(full) - rec["seen"])
        if gaps:
            missing[auth] = [g + shift for g in gaps]
    return missing


# --------------------------------------------------------------------------- #
# Chain matching by sequence substring
# --------------------------------------------------------------------------- #
KMER = 4  # seed length for the offset vote


def _identity(pdb_chain: Chain, af: Chain, offset: int) -> int:
    """How many of this chain's residues agree with the AF sequence at `offset`."""
    return sum(1 for s, letter in pdb_chain.seq.items() if af.seq.get(s + offset) == letter)


def _vote_offset(pdb_chain: Chain, af: Chain) -> Tuple[int, Optional[int]]:
    """Best (identity, offset) for this chain against one AF chain, by k-mer seeding.

    Seeds every contiguous k-mer of the PDB chain against an index of the AF
    chain's k-mers; each hit votes for one seqid offset. The best-voted offsets
    are then scored over the whole chain.
    """
    index: Dict[str, List[int]] = {}
    for s in af.seqids:
        run = [af.seq.get(s + j) for j in range(KMER)]
        if all(run):
            index.setdefault("".join(run), []).append(s)

    votes: Dict[int, int] = {}
    obs = pdb_chain.seqids
    for i in range(len(obs) - KMER + 1):
        run = obs[i:i + KMER]
        if run[-1] - run[0] != KMER - 1:
            continue  # not contiguous in author numbering
        mer = "".join(pdb_chain.seq[s] for s in run)
        for af_start in index.get(mer, ()):
            offset = af_start - run[0]
            votes[offset] = votes.get(offset, 0) + 1

    best = (0, None)
    for offset in sorted(votes, key=lambda o: (-votes[o], o))[:5]:
        score = _identity(pdb_chain, af, offset)
        if score > best[0]:
            best = (score, offset)
    return best


def match_chain(pdb_chain: Chain, missing: List[int], af_chains: Dict[str, Chain],
                taken: set, min_identity: float = 0.9) -> Tuple[Optional[Tuple[str, int, str]], Dict]:
    """Match a PDB chain to its AF counterpart by sequence and recover the seqid offset.

    Primary rule (per spec): the PDB chain's sequence — its observed residues
    plus its declared missing ones, laid out over ``[min_seqid, max_seqid]``,
    with wildcards where a residue was dropped (e.g. MSE) — must appear in the
    AF chain's sequence as an exact substring.

    That rule needs one constant offset across the chain, which a minority of
    entries break by numbering their residues in discontinuous author-numbering
    segments. For those we fall back to a k-mer offset vote and accept the best
    offset only if it reproduces >= `min_identity` of the chain's residues.

    Returns ``((af_chain_name, offset, mode), detail)``; the first element is
    None when nothing matched. ``af_seqid = pdb_seqid + offset``.
    """
    detail: Dict = {"best_af_chain": None, "best_identity": 0.0, "n_af_chains_same_kind": 0}
    same_kind = {n: af for n, af in sorted(af_chains.items())
                 if af.kind == pdb_chain.kind and af.seqids}
    detail["n_af_chains_same_kind"] = len(same_kind)
    if not same_kind or not pdb_chain.seq:
        return None, detail

    span_ids = sorted(set(pdb_chain.seq) | set(missing))
    lo, hi = span_ids[0], span_ids[-1]
    pattern = [pdb_chain.seq.get(s) for s in range(lo, hi + 1)]  # None == wildcard

    # --- exact substring, one offset ---
    exact: List[Tuple[str, int]] = []
    for name, af in same_kind.items():
        af_lo = af.seqids[0]
        af_seq = [af.seq.get(s) for s in range(af_lo, af.seqids[-1] + 1)]
        if len(af_seq) < len(pattern):
            continue
        for start in range(len(af_seq) - len(pattern) + 1):
            if all(p is None or af_seq[start + k] == p for k, p in enumerate(pattern)):
                exact.append((name, (af_lo + start) - lo))
                break
    if exact:
        # Prefer an AF chain not already claimed (homo-oligomers reuse one otherwise).
        name, offset = next(((n, o) for n, o in exact if n not in taken), exact[0])
        detail.update(best_af_chain=name, best_identity=1.0)
        return (name, offset, "substring"), detail

    # --- fallback: dominant offset by k-mer vote, accepted only at high identity ---
    best_score, best_name, best_offset = 0, None, None
    for name, af in same_kind.items():
        score, offset = _vote_offset(pdb_chain, af)
        if score > best_score:
            best_score, best_name, best_offset = score, name, offset
    identity = best_score / len(pdb_chain.seq)
    detail.update(best_af_chain=best_name, best_identity=round(identity, 4))
    if best_offset is not None and identity >= min_identity:
        return (best_name, best_offset, "offset_vote"), detail
    return None, detail


# --------------------------------------------------------------------------- #
# Kabsch
# --------------------------------------------------------------------------- #
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) superposing Q onto P:  P ~= (R @ Q.T).T + t.

    Includes the reflection fix on the determinant.
    """
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (Q - cQ).T @ (P - cP)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cP - R @ cQ
    return R, t


def superpose_stats(P: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Kabsch-superpose Q onto P and describe the residual per-anchor deviations."""
    R, t = kabsch(P, Q)
    dev = np.linalg.norm((Q @ R.T + t) - P, axis=1)
    return {
        "rmsd": float(np.sqrt(np.mean(dev ** 2))),
        "median_dev": float(np.median(dev)),
        "max_dev": float(np.max(dev)),
        "frac_dev_gt_3A": float(np.mean(dev > DEV_OUTLIER_A)),
    }


# --------------------------------------------------------------------------- #
# Missing-residue bucketing
# --------------------------------------------------------------------------- #
def bucket_missing(af_positions: List[int], af_seqids: List[int]) -> Tuple[int, int, int]:
    """Split missing residues into (N-terminal, internal, C-terminal) counts.

    N-terminal = the contiguous run of missing residues starting at the AF
    sequence's first position; C-terminal = the contiguous run ending at its
    last position; everything else is internal.
    """
    if not af_positions or not af_seqids:
        return 0, 0, 0
    miss = set(af_positions)
    first, last = af_seqids[0], af_seqids[-1]

    n_term = 0
    p = first
    while p in miss and p <= last:
        n_term += 1
        p += 1

    c_term = 0
    p = last
    while p in miss and p >= first + n_term:  # don't double-count an all-missing chain
        c_term += 1
        p -= 1

    internal = len(miss & set(af_seqids)) - n_term - c_term
    return n_term, max(internal, 0), c_term


# --------------------------------------------------------------------------- #
# One complex
# --------------------------------------------------------------------------- #
def process_complex(job: Tuple[str, str, str, float]) -> Dict:
    """Align one experimental/AF pair chain by chain. Never raises."""
    pdb_id, pdb_path, af_path, min_identity = job
    out: Dict = {
        "pdb_id": pdb_id,
        "chains": [],
        "missing_source": None,
        "global_complex_rmsd": None,
        "n_chains": 0,
        "n_shared_anchors_total": 0,
        "n_missing_total": 0,
        "n_missing_nterm": 0,
        "n_missing_internal": 0,
        "n_missing_cterm": 0,
        "errors": [],
    }

    try:
        pdb_chains = parse_chains(pdb_path)
        af_chains = parse_chains(af_path)
    except Exception as exc:  # unparseable file: log and move on
        out["errors"].append(f"parse: {type(exc).__name__}: {exc}")
        out["chains"].append(_chain_row(pdb_id, "?", "?", status=f"error:{type(exc).__name__}"))
        return out

    try:
        missing_map, source = read_unobserved(pdb_path, pdb_chains)
    except Exception as exc:
        out["errors"].append(f"unobs: {type(exc).__name__}: {exc}")
        missing_map, source = {}, "derived"
    out["missing_source"] = source

    global_P: List[np.ndarray] = []
    global_Q: List[np.ndarray] = []
    taken: set = set()
    rows: List[Dict] = []
    # (row, P, Q) for each ok chain, so we can re-score it under the global transform
    chain_anchors: List[Tuple[Dict, np.ndarray, np.ndarray]] = []

    for name in sorted(pdb_chains):
        pdb_chain = pdb_chains[name]
        missing = sorted(set(missing_map.get(name, [])))
        row = _chain_row(pdb_id, name, pdb_chain.kind)
        row["n_missing_total"] = len(missing)

        try:
            match, detail = match_chain(pdb_chain, missing, af_chains, taken, min_identity)
            row["match"] = detail
            if match is None:
                row["status"] = "chain_match_failed"
                rows.append(row)
                continue

            af_name, offset, mode = match
            af_chain = af_chains[af_name]
            taken.add(af_name)
            row["chain_len_af"] = len(af_chain.seqids)
            detail.update(af_chain=af_name, offset=offset, mode=mode)

            # Missing residues, expressed as positions along the AF sequence.
            af_missing = [s + offset for s in missing]
            n_t, internal, c_t = bucket_missing(af_missing, af_chain.seqids)
            row["n_missing_nterm"] = n_t
            row["n_missing_internal"] = internal
            row["n_missing_cterm"] = c_t

            # Anchors at every seqid present (with its anchor atom) in both files.
            # The residue names must agree too: under the offset-vote fallback a
            # seqid can land on a non-corresponding AF residue, and pairing those
            # would silently corrupt the superposition.
            P, Q = [], []
            for seqid, p_xyz in sorted(pdb_chain.anchors.items()):
                q_xyz = af_chain.anchors.get(seqid + offset)
                if q_xyz is None:
                    continue  # anchor atom missing on the AF side
                if af_chain.seq.get(seqid + offset) != pdb_chain.seq.get(seqid):
                    continue  # residue mismatch: not a real correspondence
                P.append(p_xyz)
                Q.append(q_xyz)

            row["n_shared_anchors"] = len(P)
            global_P.extend(P)
            global_Q.extend(Q)

            if len(P) < 5:
                row["status"] = "too_few_anchors"
                rows.append(row)
                continue

            row.update(superpose_stats(np.asarray(P), np.asarray(Q)))
            row["status"] = "ok"
            chain_anchors.append((row, np.asarray(P), np.asarray(Q)))
        except Exception as exc:
            row["status"] = f"error:{type(exc).__name__}: {exc}"
            out["errors"].append(f"chain {name}: {traceback.format_exc(limit=1).strip()}")
        rows.append(row)

    # A single global Kabsch over every shared anchor in the complex.
    if len(global_P) >= 5:
        gP, gQ = np.asarray(global_P), np.asarray(global_Q)
        out["global_complex_rmsd"] = superpose_stats(gP, gQ)["rmsd"]
        # Re-score each chain under that ONE global transform: apply (Rg, tg) to the
        # chain's AF anchors and take the RMSD vs its PDB anchors. This is the
        # per-chain residual under GLOBAL alignment (row["rmsd"] is per-chain-aligned).
        Rg, tg = kabsch(gP, gQ)
        for row, Pa, Qa in chain_anchors:
            dev = np.linalg.norm((Qa @ Rg.T + tg) - Pa, axis=1)
            row["rmsd_global"] = float(np.sqrt(np.mean(dev ** 2)))
    for row in rows:
        row["global_complex_rmsd"] = out["global_complex_rmsd"]
        row["missing_source"] = source

    out["chains"] = rows
    out["n_chains"] = len(rows)
    out["n_shared_anchors_total"] = len(global_P)
    for key in ("n_missing_total", "n_missing_nterm", "n_missing_internal", "n_missing_cterm"):
        out[key] = int(sum(r[key] for r in rows))
    return out


def _chain_row(pdb_id: str, chain: str, kind: str, status: str = "ok") -> Dict:
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "kind": kind,
        "chain_len_af": 0,
        "n_shared_anchors": 0,
        "rmsd": None,             # per-chain alignment (chain on its own anchors)
        "rmsd_global": None,      # this chain's residual under the single global transform
        "median_dev": None,
        "max_dev": None,
        "frac_dev_gt_3A": None,
        "n_missing_total": 0,
        "n_missing_nterm": 0,
        "n_missing_internal": 0,
        "n_missing_cterm": 0,
        "global_complex_rmsd": None,
        "missing_source": None,
        "status": status,
        "match": None,   # jsonl only (not a CSV column): how the AF chain was found
    }


CSV_COLUMNS = [
    "pdb_id", "chain", "kind", "chain_len_af", "n_shared_anchors",
    "rmsd", "rmsd_global", "median_dev", "max_dev", "frac_dev_gt_3A",
    "n_missing_total", "n_missing_nterm", "n_missing_internal", "n_missing_cterm",
    "global_complex_rmsd", "missing_source", "status",
]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
RMSD_BINS = [(0, 1, "<1"), (1, 2, "1-2"), (2, 3, "2-3"),
             (3, 5, "3-5"), (5, 10, "5-10"), (10, math.inf, ">10")]


def _histogram(values: List[float]) -> List[Tuple[str, int]]:
    return [(label, int(sum(1 for v in values if lo <= v < hi)))
            for lo, hi, label in RMSD_BINS]


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):5.1f}%" if total else "    -"


def print_report(complexes: List[Dict], df: pd.DataFrame, n_pairs: int) -> None:
    line = "=" * 78
    print(f"\n{line}\nKABSCH DIAGNOSTIC SUMMARY\n{line}")

    n_failed = sum(1 for c in complexes if all(str(r["status"]).startswith("error")
                                               for r in c["chains"]) or c["errors"])
    ok_chain = df["status"] == "ok"
    partial = {c["pdb_id"] for c in complexes
               if any(r["status"] != "ok" for r in c["chains"])
               and any(r["status"] == "ok" for r in c["chains"])}
    print(f"\ncomplexes paired      : {n_pairs}")
    print(f"complexes processed   : {len(complexes)}")
    print(f"  fully processed     : {sum(1 for c in complexes if all(r['status'] == 'ok' for r in c['chains']))}")
    print(f"  partially processed : {len(partial)}   (some chains ok, some not)")
    print(f"  failed / errored    : {n_failed}")
    print(f"chains                : {len(df)}  (aligned ok: {int(ok_chain.sum())})")

    # --- RMSD histogram, protein vs RNA ---
    print("\nper-chain RMSD histogram (A)")
    print(f"  {'bin':>6} | {'protein':>16} | {'RNA':>16}")
    prot = df.loc[ok_chain & (df["kind"] == KIND_PROTEIN), "rmsd"].dropna().tolist()
    rna = df.loc[ok_chain & (df["kind"] == KIND_RNA), "rmsd"].dropna().tolist()
    for (label, np_), (_, nr) in zip(_histogram(prot), _histogram(rna)):
        print(f"  {label:>6} | {np_:6d} {_pct(np_, len(prot))} | {nr:6d} {_pct(nr, len(rna))}")
    print(f"  {'total':>6} | {len(prot):6d}        | {len(rna):6d}")

    print("\nper-chain RMSD (A)")
    for label, vals in (("protein", prot), ("RNA", rna)):
        if vals:
            arr = np.asarray(vals)
            print(f"  {label:>7}: median {np.median(arr):7.3f}   p95 {np.percentile(arr, 95):7.3f}"
                  f"   mean {arr.mean():7.3f}   n={len(arr)}")
        else:
            print(f"  {label:>7}: no chains")

    g = df.drop_duplicates("pdb_id")["global_complex_rmsd"].dropna()
    if len(g):
        print(f"\nglobal-complex RMSD (A): median {g.median():.3f}   p95 "
              f"{np.percentile(g, 95):.3f}   max {g.max():.3f}   n={len(g)}")

    # --- Missing residues ---
    tot = int(df["n_missing_total"].sum())
    print(f"\nunresolved residues: {tot} total over {len(complexes)} complexes")
    for label, col in (("N-terminal", "n_missing_nterm"),
                       ("internal", "n_missing_internal"),
                       ("C-terminal", "n_missing_cterm")):
        n = int(df[col].sum())
        print(f"  {label:>10}: {n:6d} {_pct(n, tot)}")
    unbucketed = tot - int(df[["n_missing_nterm", "n_missing_internal", "n_missing_cterm"]].sum().sum())
    if unbucketed:
        print(f"  {'unbucketed':>10}: {unbucketed:6d} {_pct(unbucketed, tot)}  (chains with no AF match)")

    per_complex = df.groupby("pdb_id")["n_missing_total"].sum()
    print("  per complex: " + "  ".join(
        f"{q}={per_complex.quantile(v):.0f}"
        for q, v in (("min", 0.0), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1.0))))
    print(f"  complexes with zero unresolved residues: {int((per_complex == 0).sum())}")

    # --- Worst chains ---
    print("\ntop 10 chains by RMSD (likely alignment failures, inspect manually)")
    top = df[ok_chain].nlargest(10, "rmsd")
    print(f"  {'pdb_id':>8} {'chain':>5} {'kind':>4} {'anchors':>7} {'rmsd':>8} {'median':>7} "
          f"{'max':>8} {'>3A':>6} {'global':>8}")
    for _, r in top.iterrows():
        gl = "-" if pd.isna(r["global_complex_rmsd"]) else f"{r['global_complex_rmsd']:8.2f}"
        print(f"  {r['pdb_id']:>8} {r['chain']:>5} {r['kind']:>4} {r['n_shared_anchors']:7d} "
              f"{r['rmsd']:8.2f} {r['median_dev']:7.2f} {r['max_dev']:8.2f} "
              f"{r['frac_dev_gt_3A']:6.2f} {gl}")

    # --- Flags ---
    print("\nchain status")
    n_chains = len(df)
    for status, n in df["status"].str.split(":").str[0].value_counts().items():
        print(f"  {status:>18}: {n:5d} {_pct(int(n), n_chains)}")

    # How each matched chain found its AF counterpart, and why the rest did not.
    chains = [r for c in complexes for r in c["chains"]]
    modes = {"substring": 0, "offset_vote": 0}
    reasons = {"no AF chain of same kind": 0, "identity below threshold": 0}
    for r in chains:
        m = r.get("match") or {}
        if r["status"] == "chain_match_failed":
            key = ("no AF chain of same kind" if not m.get("n_af_chains_same_kind")
                   else "identity below threshold")
            reasons[key] += 1
        elif m.get("mode"):
            modes[m["mode"]] += 1
    print("\nchain matching")
    for mode, n in modes.items():
        print(f"  {mode:>18}: {n:5d} {_pct(n, n_chains)}")
    for reason, n in reasons.items():
        if n:
            print(f"  failed, {reason:>10}: {n:5d} {_pct(n, n_chains)}")
    unmatched = [r for r in chains if r["status"] == "chain_match_failed"
                 and (r.get("match") or {}).get("best_identity", 0) >= 0.4]
    if unmatched:
        best = sorted(unmatched, key=lambda r: -r["match"]["best_identity"])[:5]
        print("  near-misses (best single-offset identity, likely segmented author numbering):")
        for r in best:
            print(f"    {r['pdb_id']} chain {r['chain']}: identity "
                  f"{r['match']['best_identity']:.2f} vs AF chain {r['match']['best_af_chain']}")

    src = df.drop_duplicates("pdb_id")["missing_source"].value_counts()
    n_cx = int(src.sum())
    print("\nmissing-residue source (per complex)")
    for source in ("unobs_table", "derived"):
        n = int(src.get(source, 0))
        print(f"  {source:>18}: {n:5d} {_pct(n, n_cx)}")
    print(line)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """numpy scalars -> python; NaN/inf -> None (so the JSON stays strict)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb_dir", default="PDB-CIF", help="experimental mmCIF directory")
    p.add_argument("--af_dir", default="AlphaFold-CIF", help="AlphaFold3 mmCIF directory")
    p.add_argument("--out_dir", default="diagnostics", help="where summary.csv / per_complex.jsonl go")
    p.add_argument("--n_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--limit", type=int, default=None, help="only the first N complexes (debug)")
    p.add_argument("--match_min_identity", type=float, default=0.9,
                   help="identity a k-mer-voted seqid offset must reach before a chain "
                        "counts as matched (only used when exact substring matching fails)")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs, unpaired = pair_files(args.pdb_dir, args.af_dir)
    for msg in unpaired:
        print(msg)
    # sorted -> idempotent
    jobs = [(pdb_id, p, a, args.match_min_identity) for pdb_id, (p, a) in sorted(pairs.items())]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[pair] {len(jobs)} complex(es) paired, {len(unpaired)} file(s) skipped")
    if not jobs:
        raise SystemExit("nothing to do")

    n_workers = max(1, min(args.n_workers, len(jobs)))
    print(f"[run] {n_workers} worker(s)")
    results: List[Dict] = []
    if n_workers == 1:
        for job in tqdm(jobs, desc="[kabsch]"):
            results.append(process_complex(job))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for res in tqdm(pool.map(process_complex, jobs, chunksize=4),
                            total=len(jobs), desc="[kabsch]"):
                results.append(res)

    results.sort(key=lambda r: r["pdb_id"])

    rows = [r for res in results for r in res["chains"]]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS).sort_values(["pdb_id", "chain"], kind="stable")
    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    jsonl_path = os.path.join(args.out_dir, "per_complex.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(_json_safe(res), sort_keys=True, allow_nan=False) + "\n")

    print_report(results, df, n_pairs=len(jobs))
    print(f"\n[out] {csv_path}  ({len(df)} rows)")
    print(f"[out] {jsonl_path}  ({len(results)} complexes)")


if __name__ == "__main__":
    main()
