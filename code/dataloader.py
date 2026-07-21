import io
import os
import re
import gzip
import json
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import model as eg
from utils import CHAIN_PROTEIN, CHAIN_RNA, INTERFACE_CUTOFF


PROTEIN_REP_ATOM = "CA"
RNA_REP_ATOM = "C4'"

_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_RNA = {"A", "C", "G", "U", "I"}

# Stable integer id per residue name, used as a (single) raw node feature.
_RES_VOCAB = sorted(_AA3 | _RNA)
_RES_TO_ID = {r: i for i, r in enumerate(_RES_VOCAB)}


def residue_chain_type(resname: str) -> Optional[int]:
    """CHAIN_PROTEIN, CHAIN_RNA, or None if the residue is neither."""
    if resname in _AA3:
        return CHAIN_PROTEIN
    if resname in _RNA:
        return CHAIN_RNA
    return None


def representative_atom(resname: str) -> Optional[str]:
    t = residue_chain_type(resname)
    if t == CHAIN_PROTEIN:
        return PROTEIN_REP_ATOM
    if t == CHAIN_RNA:
        return RNA_REP_ATOM
    return None


# --------------------------------------------------------------------------- #
# mmCIF parsing (shared by AlphaFold predictions and experimental structures)
# --------------------------------------------------------------------------- #
def _strip_quotes(tok: str) -> str:
    if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0]:
        return tok[1:-1]
    return tok


def _cif_val(rec: Dict[str, str], *keys: str) -> Optional[str]:
    """First present, non-null ('.'/'?') value among `keys`, else None."""
    for k in keys:
        v = rec.get(k)
        if v is not None and v not in (".", "?"):
            return v
    return None


def _read_cif_loop(lines, prefix: str):
    """Yield dict records (item -> token) for the first mmCIF ``loop_`` whose
    header items start with ``prefix`` (e.g. ``"_atom_site."``).

    Simple whitespace tokenisation — values with internal spaces or multi-line
    text blocks are not supported, which is fine for ``_atom_site`` and
    ``_pdbx_unobs_or_zero_occ_residues`` (all single-token values).
    """
    columns: List[str] = []
    state = "search"  # search -> header -> data
    for raw in lines:
        s = raw.strip()
        if state == "search":
            if raw.startswith(prefix):
                columns = [s.split(".", 1)[1]]
                state = "header"
            continue
        if state == "header":
            if raw.startswith(prefix):
                columns.append(s.split(".", 1)[1])
                continue
            # header ended; a control line means the loop had no data rows
            if s == "" or s[0] in "#_" or s.startswith("loop_") or s.startswith("data_"):
                return
            state = "data"  # fall through to parse this first data row
        if state == "data":
            if s == "" or s[0] in "#_" or s.startswith("loop_") or s.startswith("data_"):
                return
            toks = s.split()
            if len(toks) >= len(columns):
                yield dict(zip(columns, toks))


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()


def parse_af_cif(path: str) -> Dict[str, List[Tuple[int, str, np.ndarray]]]:
    """Parse an AlphaFold mmCIF into ``{chain: [(seq_id, resname, xyz), ...]}``.

    One entry per residue (its representative Cα / C4' atom), ordered by seq_id.
    """
    return _parse_atom_site(_read_lines(path))


def _parse_atom_site(lines) -> Dict[str, List[Tuple[int, str, np.ndarray]]]:
    """Parse an ``_atom_site`` loop into residue-level representative atoms.

    Returns ``{chain(auth_asym_id): [(seq_id, resname, xyz), ...]}`` ordered by
    seq_id. First model only. Header-driven, so it works for both AlphaFold and
    experimental mmCIF.
    """
    chains: Dict[str, Dict[int, Tuple[str, np.ndarray]]] = {}
    first_model = None
    for rec in _read_cif_loop(lines, "_atom_site."):
        model = rec.get("pdbx_PDB_model_num")
        if first_model is None:
            first_model = model
        if model is not None and model != first_model:
            continue  # keep a single model (NMR / multi-model files)
        if rec.get("label_alt_id", ".") not in (".", "?", "A"):  # single altloc
            continue
        resname = rec.get("label_comp_id", "")
        rep = representative_atom(resname)
        if rep is None or _strip_quotes(rec.get("label_atom_id", "")) != rep:
            continue
        ch = _cif_val(rec, "auth_asym_id", "label_asym_id")
        try:
            seq_id = int(_cif_val(rec, "auth_seq_id", "label_seq_id"))
            xyz = np.array(
                [float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])],
                dtype=np.float64,
            )
        except (ValueError, TypeError, KeyError):
            continue
        chains.setdefault(ch, {})[seq_id] = (resname, xyz)

    return {ch: [(s, rn, xyz) for s, (rn, xyz) in sorted(m.items())]
            for ch, m in chains.items()}


def parse_experimental_cif(path: str) -> Dict[str, Dict]:
    """Parse an experimental mmCIF into per-chain known/missing residues.

    Returns ``{chain: {"known": {resnum: (resname, xyz)}, "missing": {resnum: resname}}}``
      * ``known``   -> resolved residues (representative atom in ``_atom_site``)
      * ``missing`` -> residues in the ``_pdbx_unobs_or_zero_occ_residues`` loop
    """
    lines = _read_lines(path)
    chains: Dict[str, Dict] = {}

    def _chain(ch: str) -> Dict:
        return chains.setdefault(ch, {"known": {}, "missing": {}})

    for ch, residues in _parse_atom_site(lines).items():
        known = _chain(ch)["known"]
        for seq_id, resname, xyz in residues:
            known[seq_id] = (resname, xyz)

    for rec in _read_cif_loop(lines, "_pdbx_unobs_or_zero_occ_residues."):
        if rec.get("polymer_flag", "Y") == "N":
            continue  # only polymer residues
        resname = _cif_val(rec, "auth_comp_id", "label_comp_id")
        ch = _cif_val(rec, "auth_asym_id", "label_asym_id")
        if resname is None or ch is None or residue_chain_type(resname) is None:
            continue
        try:
            resnum = int(_cif_val(rec, "auth_seq_id", "label_seq_id"))
        except (ValueError, TypeError):
            continue
        _chain(ch)["missing"][resnum] = resname

    # A residue can't be both known and missing; known wins.
    for ch in chains.values():
        for r in set(ch["missing"]) & set(ch["known"]):
            del ch["missing"][r]
    return chains


def chain_full_sequence(chain: Dict) -> List[Tuple[int, str, bool]]:
    """Merge known + missing residues of one chain, ordered by residue number.

    Returns list of (resnum, resname, is_known).
    """
    items = [(resnum, resname, True) for resnum, (resname, _) in chain["known"].items()]
    items += [(resnum, resname, False) for resnum, resname in chain["missing"].items()]
    items.sort(key=lambda t: t[0])
    return items


# --------------------------------------------------------------------------- #
# Chain matching + rigid alignment
# --------------------------------------------------------------------------- #
def _seq_identity(a: List[str], b: List[str]) -> float:
    """Fraction of matching residue names over the overlapping prefix length."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def match_chains(
    exp_chains: Dict[str, Dict],
    af3_chains: Dict[str, List[Tuple[int, str, np.ndarray]]],
    min_identity: float = 0.6,
) -> Dict[str, str]:
    """Match each experimental chain to its best-identity AF3 chain by sequence.

    Handles AF3 renaming/renumbering chains, and allows one AF3 chain to serve
    several *identical* experimental chains (homo-oligomers where AF3 modelled a
    single copy) — each copy is later aligned on its own known anchors.
    """
    exp_seqs = {ch: [r[1] for r in chain_full_sequence(c)] for ch, c in exp_chains.items()}
    af3_seqs = {ch: [r[1] for r in res] for ch, res in af3_chains.items()}

    mapping: Dict[str, str] = {}
    for p, ps in exp_seqs.items():
        best_a, best_ident = None, 0.0
        for a, as_ in af3_seqs.items():
            ident = _seq_identity(ps, as_)
            if ident > best_ident:
                best_a, best_ident = a, ident
        if best_a is not None and best_ident >= min_identity:
            mapping[p] = best_a
    return mapping


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform (R, t) mapping P onto Q:  Q ≈ P @ R.T + t.

    P, Q: [M, 3] corresponding point sets.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_complex_graph(
    exp_cif: str,
    af_cif: str,
    radius: float = 12.0,
    min_chain_identity: float = 0.6,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build a graph from an experimental mmCIF + an AlphaFold ``model_0.cif``.

    Known residues come from the experimental structure; the residues it is
    missing are filled in from the (rigidly aligned) AlphaFold prediction.
    """
    exp_chains = parse_experimental_cif(exp_cif)
    af3_chains = parse_af_cif(af_cif)

    mapping = match_chains(exp_chains, af3_chains, min_identity=min_chain_identity)
    if not mapping:
        raise RuntimeError("Could not match any experimental chain to an AF3 chain by sequence.")

    # ---- 1) Residue-level correspondence; align AF3 -> experimental per chain --
    # A single global rigid transform can't fit a multi-chain complex whose
    # chains sit differently in AF3 vs the crystal. Aligning each chain on its
    # own shared known residues places that chain's missing residues correctly
    # next to its own known part (good local bond geometry).
    node_recs: List[Dict] = []
    # global anchor pools, used as a fallback for chains with too few anchors
    g_af3_anchors: List[np.ndarray] = []
    g_exp_anchors: List[np.ndarray] = []
    chain_anchors: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    chain_missing: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    for exp_ch, af3_ch in mapping.items():
        full = chain_full_sequence(exp_chains[exp_ch])   # [(resnum, resname, known)]
        af3 = af3_chains[af3_ch]                         # [(seq_id, resname, xyz)]
        known_map = exp_chains[exp_ch]["known"]

        n = min(len(full), len(af3))
        if verbose and len(full) != len(af3):
            print(f"[warn] chain {exp_ch}<->{af3_ch} length mismatch "
                  f"(exp {len(full)} vs af3 {len(af3)}); aligning first {n}.")

        chain_anchors.setdefault(exp_ch, ([], []))
        chain_missing.setdefault(exp_ch, [])

        for k in range(n):
            resnum, resname, is_known = full[k]
            af3_xyz = af3[k][2]
            ctype = residue_chain_type(resname)
            if ctype is None:
                continue

            node_index = len(node_recs)
            node_recs.append({
                "coord": None,                    # filled below
                "node_type": eg.EGNN.NODE_KNOWN_UNMASKED if is_known else eg.EGNN.NODE_MISSING,
                "chain_id": ctype,
                "seq_index": k,                   # per-chain ordinal
                "resname": resname,
                "chain_label": exp_ch,            # actual chain, not just its type
                "resnum": resnum,
            })

            if is_known:
                exp_xyz = known_map[resnum][1]
                node_recs[node_index]["coord"] = exp_xyz
                chain_anchors[exp_ch][0].append(af3_xyz)
                chain_anchors[exp_ch][1].append(exp_xyz)
                g_af3_anchors.append(af3_xyz)
                g_exp_anchors.append(exp_xyz)
            else:
                chain_missing[exp_ch].append((node_index, af3_xyz))

    if len(g_exp_anchors) < 3:
        raise RuntimeError(
            f"Only {len(g_exp_anchors)} shared known residues; need >=3 for rigid alignment."
        )
    R_glob, t_glob = kabsch(np.asarray(g_af3_anchors), np.asarray(g_exp_anchors))

    # ---- 2) Per-chain transform (global fallback) applied to missing residues --
    for exp_ch, missing in chain_missing.items():
        af3_a, exp_a = chain_anchors[exp_ch]
        if len(exp_a) >= 3:
            P, Q = np.asarray(af3_a), np.asarray(exp_a)
            R, t = kabsch(P, Q)
            rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(axis=1).mean()))
            tag = "chain"
        else:
            R, t, rmsd, tag = R_glob, t_glob, float("nan"), "global-fallback"
        if verbose:
            print(f"[align] chain {exp_ch}: anchors={len(exp_a)} missing={len(missing)} "
                  f"RMSD={rmsd:.2f} A ({tag})")
        for node_index, af3_xyz in missing:
            node_recs[node_index]["coord"] = af3_xyz @ R.T + t

    # ---- 3) Tensorise node arrays ----
    x = torch.tensor(np.stack([r["coord"] for r in node_recs]), dtype=torch.float32)
    node_type = torch.tensor([r["node_type"] for r in node_recs], dtype=torch.long)
    chain_id = torch.tensor([r["chain_id"] for r in node_recs], dtype=torch.long)
    seq_index = torch.tensor([r["seq_index"] for r in node_recs], dtype=torch.long)
    # single raw node feature: residue-type id, normalised to ~[0,1]
    res_ids = [_RES_TO_ID.get(r["resname"], 0) for r in node_recs]
    h = (torch.tensor(res_ids, dtype=torch.float32) / len(_RES_VOCAB)).unsqueeze(1)

    # Kept for writing predictions back out as CIF; unused by the model/losses.
    res_id = torch.tensor(res_ids, dtype=torch.long)
    chain_order = {lab: i for i, lab in
                   enumerate(dict.fromkeys(r["chain_label"] for r in node_recs))}
    chain_index = torch.tensor([chain_order[r["chain_label"]] for r in node_recs],
                               dtype=torch.long)
    resnum = torch.tensor([int(r["resnum"]) for r in node_recs], dtype=torch.long)

    # ---- 4) Edges: sequential backbone + radius graph ----
    edge_index, edge_type = _build_edges(x, chain_id, seq_index, radius)
    src, dst = edge_index[0], edge_index[1]
    edge_attr = (x[src] - x[dst]).norm(dim=1, keepdim=True)   # raw edge feature: length

    return {
        "h": h,
        "x": x,
        "node_type": node_type,
        "chain_id": chain_id,
        "seq_index": seq_index,
        "res_id": res_id,
        "chain_index": chain_index,
        "resnum": resnum,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_type": edge_type,
        # interface pairs: protein<->RNA within cutoff in the truth coordinates
        "interface_pairs": _interface_pairs(x, chain_id, INTERFACE_CUTOFF),
    }


def _build_edges(
    x: torch.Tensor,
    chain_id: torch.Tensor,
    seq_index: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential (backbone) + radius-graph edges, both directions, de-duplicated.

    edge_type: 0 backbone, 1 intra-chain spatial, 2 inter-chain (interface).
    """
    N = x.size(0)
    edges = {}  # (i,j) -> type

    # sequential backbone edges: consecutive same-chain nodes (ordered by construction)
    for i in range(N - 1):
        if chain_id[i] == chain_id[i + 1] and seq_index[i + 1] == seq_index[i] + 1:
            edges[(i, i + 1)] = 0
            edges[(i + 1, i)] = 0

    # radius graph: every pair within `radius`
    with torch.no_grad():
        d = torch.cdist(x, x)                       # [N, N]
        d.fill_diagonal_(float("inf"))
        within = d <= radius

    for i in range(N):
        for j in torch.nonzero(within[i], as_tuple=False).squeeze(1).tolist():
            if (i, j) in edges:                     # keep backbone typing
                continue
            edges[(i, j)] = 1 if chain_id[i] == chain_id[j] else 2

    if not edges:
        edges = {(0, min(1, N - 1)): 1}  # degenerate fallback

    rows = torch.tensor([e[0] for e in edges], dtype=torch.long)
    cols = torch.tensor([e[1] for e in edges], dtype=torch.long)
    etypes = torch.tensor(list(edges.values()), dtype=torch.long)
    return torch.stack([rows, cols], dim=0), etypes


def _interface_pairs(x: torch.Tensor, chain_id: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Protein(0) <-> RNA(1) node pairs within `cutoff` A. Returns [K, 2] long."""
    prot = (chain_id == CHAIN_PROTEIN).nonzero(as_tuple=True)[0]
    rna = (chain_id == CHAIN_RNA).nonzero(as_tuple=True)[0]
    if prot.numel() == 0 or rna.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    with torch.no_grad():
        d = torch.cdist(x[prot], x[rna])            # [P, R]
        pi, ri = (d <= cutoff).nonzero(as_tuple=True)
    if pi.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.stack([prot[pi], rna[ri]], dim=1)


def collate_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Merge graphs into one big disconnected graph (offsetting all indices).

    NOTE: `losses.compute_loss` keys geometry on chain_id in {0,1}, so it treats
    all protein nodes across the batch as one chain for bond/clash terms. Using
    batch_size=1 (one complex per step) keeps those terms exactly correct; larger
    batches are supported but mix chains of the same type across complexes.
    """
    node_keys = ["h", "x", "node_type", "chain_id", "seq_index", "res_id",
                 "chain_index", "resnum", "edge_attr", "edge_type"]
    # lm_emb is optional (only present when --use_lm_emb / an embedding cache exists).
    if graphs and "lm_emb" in graphs[0]:
        node_keys = node_keys + ["lm_emb"]
    parts: Dict[str, List[torch.Tensor]] = {k: [] for k in node_keys}
    rows, cols, ifpairs = [], [], []
    offset = 0
    for g in graphs:
        for k in node_keys:
            parts[k].append(g[k])
        rows.append(g["edge_index"][0] + offset)
        cols.append(g["edge_index"][1] + offset)
        if g["interface_pairs"].numel() > 0:
            ifpairs.append(g["interface_pairs"] + offset)
        offset += g["x"].size(0)

    out = {k: torch.cat(v, 0) for k, v in parts.items()}
    out["edge_index"] = torch.stack([torch.cat(rows), torch.cat(cols)], 0)
    out["interface_pairs"] = (torch.cat(ifpairs, 0) if ifpairs
                              else torch.zeros((0, 2), dtype=torch.long))
    return out


def _report_unmatched(pdb_names, af_names) -> List[str]:
    """Log names present in only one source; return the sorted intersection."""
    only_pdb = sorted(set(pdb_names) - set(af_names))
    only_af = sorted(set(af_names) - set(pdb_names))
    if only_pdb:
        print(f"[discover] {len(only_pdb)} PDB structure(s) with no AlphaFold match: {only_pdb}")
    if only_af:
        print(f"[discover] {len(only_af)} AlphaFold structure(s) with no PDB match: {only_af}")
    return sorted(set(pdb_names) & set(af_names))


# --------------------------------------------------------------------------- #
# Zip source (PDB.zip.zip  <->  AlphaFold.zip.zip)
# --------------------------------------------------------------------------- #
# PDB archive holds  "<name>.cif.gz"  (gzipped experimental mmCIF).
# AlphaFold archive holds nested zips "fold_<name>[ (n)].zip", each of which
# contains "fold_<name>_model_0.cif" (plus models 1-4 we ignore).
_PDB_MEMBER_RE = re.compile(r"([^/\\]+)\.cif\.gz$", re.IGNORECASE)
_AF_MODEL0_RE = re.compile(r"fold_(.+)_model_0\.cif$", re.IGNORECASE)


def discover_structures(pdb_zip: str, af_zip: str,
                        cache_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Map ``<name> -> {'pdb_member', 'af_nested', 'af_model'}`` across both zips.

    The AlphaFold ``<name>`` is derived from the *inner* ``fold_<name>_model_0.cif``
    filename, so it is robust to nested-zip suffixes like ``fold_11gi (1).zip``.
    Only structures present in **both** archives are returned.

    The mapping is cached to ``<cache_dir>/_index.json`` (if ``cache_dir`` given)
    so subsequent runs skip re-scanning the large AlphaFold archive.
    """
    index_path = os.path.join(cache_dir, "_index.json") if cache_dir else None
    if index_path and os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    pdb_map: Dict[str, str] = {}
    with zipfile.ZipFile(pdb_zip) as zf:
        for member in zf.namelist():
            m = _PDB_MEMBER_RE.search(member)
            if m and not member.endswith("/"):
                pdb_map[m.group(1).lower()] = member

    af_map: Dict[str, Tuple[str, str]] = {}
    with zipfile.ZipFile(af_zip) as zf:
        for nested in zf.namelist():
            if not nested.lower().endswith(".zip"):
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(zf.read(nested))) as inner:
                    for inner_name in inner.namelist():
                        im = _AF_MODEL0_RE.search(inner_name)
                        if im:
                            af_map[im.group(1).lower()] = (nested, inner_name)
                            break
            except zipfile.BadZipFile:
                continue

    structures = {
        name: {
            "pdb_member": pdb_map[name],
            "af_nested": af_map[name][0],
            "af_model": af_map[name][1],
        }
        for name in _report_unmatched(pdb_map, af_map)
    }
    if index_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(structures, fh, indent=2)
    return structures


def extract_structure_cifs(name: str, entry: Dict[str, str], pdb_zip: str,
                           af_zip: str, cache_dir: str) -> Tuple[str, str]:
    """Ensure ``<name>``'s experimental + AF model_0 CIFs exist under ``cache_dir``.

    Returns ``(exp_cif_path, af_cif_path)``. Extraction is skipped if both files
    are already cached — so repeated epochs/runs do no zip work.
    """
    os.makedirs(cache_dir, exist_ok=True)
    exp_path = os.path.join(cache_dir, f"{name}.cif")
    af_path = os.path.join(cache_dir, f"fold_{name}_model_0.cif")

    if not os.path.exists(exp_path):
        with zipfile.ZipFile(pdb_zip) as zf:
            text = gzip.decompress(zf.read(entry["pdb_member"])).decode("utf-8", "replace")
        with open(exp_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    if not os.path.exists(af_path):
        with zipfile.ZipFile(af_zip) as zf:
            nested_bytes = zf.read(entry["af_nested"])
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
            text = inner.read(entry["af_model"]).decode("utf-8", "replace")  # only model_0
        with open(af_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    return exp_path, af_path


# --------------------------------------------------------------------------- #
# Folder source (pre-extracted CIFs)
# --------------------------------------------------------------------------- #
# PDB-CIF/ holds  "<name>.cif"  (experimental mmCIF, already gunzipped).
# AlphaFold-CIF/ holds  "fold_<name>_model_0.cif"  (already unzipped).
_PDB_CIF_RE = re.compile(r"([^/\\]+)\.cif$", re.IGNORECASE)


def discover_structures_folders(pdb_dir: str, af_dir: str) -> Dict[str, Dict[str, str]]:
    """Map ``<name> -> {'exp_cif', 'af_cif'}`` across two folders of CIF files.

    PDB side matches ``<name>.cif``; AlphaFold side matches
    ``fold_<name>_model_0.cif``. Only structures present in **both** are returned.
    """
    pdb_map: Dict[str, str] = {}
    for fn in os.listdir(pdb_dir):
        m = _PDB_CIF_RE.match(fn)
        if m and not fn.lower().startswith("fold_"):
            pdb_map[m.group(1).lower()] = os.path.join(pdb_dir, fn)

    af_map: Dict[str, str] = {}
    for fn in os.listdir(af_dir):
        m = _AF_MODEL0_RE.search(fn)
        if m:
            af_map[m.group(1).lower()] = os.path.join(af_dir, fn)

    return {name: {"exp_cif": pdb_map[name], "af_cif": af_map[name]}
            for name in _report_unmatched(pdb_map, af_map)}


# --------------------------------------------------------------------------- #
# Dataset / DataLoader
# --------------------------------------------------------------------------- #
class RNPDataset(Dataset):
    """One item = one experimental/AlphaFold mmCIF pair, as a prebuilt graph.

    ``pairs`` is ``{name: (exp_cif_path, af_cif_path)}``. Every graph is built
    once up front; structures that fail to build (e.g. no chain match between
    the experimental and AF CIFs) are skipped with a report.
    """

    def __init__(self, pairs: Dict[str, Tuple[str, str]], emb_dir: Optional[str] = None,
                 **build_kwargs):
        self.names: List[str] = []
        self.graphs: List[Dict[str, torch.Tensor]] = []
        skipped: List[Tuple[str, str]] = []
        missing_emb: List[str] = []

        for name in tqdm(sorted(pairs), desc="[dataset] building graphs"):
            exp_cif, af_cif = pairs[name]
            try:
                graph = build_complex_graph(exp_cif, af_cif, **build_kwargs)
                if emb_dir is not None:
                    self._attach_lm_emb(graph, name, emb_dir)  # raises if missing/mismatched
                self.graphs.append(graph)
                self.names.append(name)
            except FileNotFoundError as e:      # embedding cache miss -> skip, but flag it
                missing_emb.append(name)
                skipped.append((name, str(e)))
            except Exception as e:              # skip unbuildable structures, keep going
                skipped.append((name, str(e)))

        print(f"[dataset] built {len(self.names)} structure(s); skipped {len(skipped)}")
        for name, err in skipped[:10]:
            print(f"          skip {name}: {err}")
        if len(skipped) > 10:
            print(f"          ... and {len(skipped) - 10} more")
        if missing_emb:
            print(f"[dataset] {len(missing_emb)} structure(s) skipped for missing embeddings; "
                  f"run compute_embeddings.py first (emb_dir={emb_dir}).")

    @staticmethod
    def _attach_lm_emb(graph: Dict[str, torch.Tensor], name: str, emb_dir: str) -> None:
        """Load the cached [N, lm_emb_dim] per-residue embedding and attach it.

        The cache is one tensor per structure, in the SAME node order as the graph
        (row i == graph node i). Raises FileNotFoundError if absent, ValueError on
        an N mismatch, so silent misalignment can never happen.
        """
        path = os.path.join(emb_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no embedding cache at {path}")
        emb = torch.load(path, map_location="cpu")
        if not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        n_nodes = graph["x"].size(0)
        if emb.dim() != 2 or emb.size(0) != n_nodes:
            raise ValueError(f"embedding for {name} has shape {tuple(emb.shape)}, "
                             f"expected [{n_nodes}, D] to match the graph nodes")
        graph["lm_emb"] = emb.to(torch.float32)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.graphs[idx]


def _select(structures: Dict[str, Dict[str, str]],
            names: Optional[List[str]]) -> List[str]:
    return [n for n in (names or sorted(structures)) if n in structures]


def _make_loader(pairs: Dict[str, Tuple[str, str]], batch_size: int, shuffle: bool,
                 num_workers: int, **build_kwargs) -> DataLoader:
    """Wrap the prebuilt graphs in a DataLoader that collates them into one graph."""
    dataset = RNPDataset(pairs, **build_kwargs)
    if len(dataset) == 0:
        raise RuntimeError("No structures could be built; nothing to train on.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,   # graphs are already in memory: 0 is fastest
        collate_fn=collate_graphs,
    )


def build_folder_dataloader(pdb_dir: str, af_dir: str, batch_size: int = 1,
                            shuffle: bool = True, names: Optional[List[str]] = None,
                            num_workers: int = 0, **build_kwargs) -> DataLoader:
    """DataLoader over two folders of pre-extracted CIFs (PDB-CIF/ + AlphaFold-CIF/)."""
    structures = discover_structures_folders(pdb_dir, af_dir)
    selected = _select(structures, names)
    print(f"[data] {len(selected)} structure(s) from folders")
    pairs = {n: (structures[n]["exp_cif"], structures[n]["af_cif"]) for n in selected}
    return _make_loader(pairs, batch_size, shuffle, num_workers, **build_kwargs)


def build_zip_dataloader(pdb_zip: str, af_zip: str, cache_dir: str = "./.cache_cifs",
                         batch_size: int = 1, shuffle: bool = True,
                         names: Optional[List[str]] = None, num_workers: int = 0,
                         **build_kwargs) -> DataLoader:
    """DataLoader over the raw archives; CIFs are extracted once into ``cache_dir``."""
    structures = discover_structures(pdb_zip, af_zip, cache_dir=cache_dir)
    selected = _select(structures, names)
    print(f"[data] {len(selected)} structure(s) from zips (cache: {cache_dir})")
    pairs = {}
    for name in tqdm(selected, desc="[dataset] extracting CIFs"):
        pairs[name] = extract_structure_cifs(name, structures[name], pdb_zip, af_zip, cache_dir)
    return _make_loader(pairs, batch_size, shuffle, num_workers, **build_kwargs)


# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
def split_names(names: List[str], test_frac: float = 0.1,
                seed: int = 0) -> Tuple[List[str], List[str]]:
    """Deterministic split of complex names into (train, test).

    Seeded independently of the training seed, so the held-out set stays fixed
    while model seeds vary. Sorting first makes the result independent of the
    order the filesystem happened to list the CIFs in.
    """
    import random
    ordered = sorted(names)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_test = int(round(test_frac * len(shuffled)))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return train, test


def _resolve_pairs(args_like) -> Dict[str, Tuple[str, str]]:
    """Discover {name: (exp_cif, af_cif)} from either zip mode or folder mode."""
    if getattr(args_like, "pdb_zip", None) and getattr(args_like, "af_zip", None):
        structures = discover_structures(args_like.pdb_zip, args_like.af_zip,
                                         cache_dir=args_like.cache_dir)
        selected = _select(structures, args_like.names)
        print(f"[data] zip mode: {len(selected)} structure(s)")
        return {n: extract_structure_cifs(n, structures[n], args_like.pdb_zip,
                                          args_like.af_zip, args_like.cache_dir)
                for n in tqdm(selected, desc="[dataset] extracting CIFs")}

    structures = discover_structures_folders(args_like.pdb_dir, args_like.af_dir)
    selected = _select(structures, args_like.names)
    print(f"[data] folder mode: {len(selected)} structure(s)")
    return {n: (structures[n]["exp_cif"], structures[n]["af_cif"]) for n in selected}


def build_split_dataset(args_like, radius: float):
    """Build every graph once, then index it into a train and a test subset.

    Returns ``(dataset, train_idx, test_idx)``. Graph construction dominates the
    cost, so both splits share one ``RNPDataset`` rather than building twice.
    Structures that fail to build are dropped by ``RNPDataset``, so the split is
    computed against ``dataset.names`` (what actually built), not the discovered
    names.
    """
    pairs = _resolve_pairs(args_like)
    emb_dir = getattr(args_like, "emb_dir", None) if getattr(args_like, "use_lm_emb", False) else None
    dataset = RNPDataset(pairs, radius=radius, emb_dir=emb_dir)
    if len(dataset) == 0:
        raise RuntimeError("No structures could be built; nothing to train on.")

    train_names, test_names = split_names(dataset.names, args_like.test_frac,
                                          args_like.split_seed)
    train_set, test_set = set(train_names), set(test_names)
    train_idx = [i for i, n in enumerate(dataset.names) if n in train_set]
    test_idx = [i for i, n in enumerate(dataset.names) if n in test_set]
    print(f"[split] {len(train_idx)} train / {len(test_idx)} test "
          f"(test_frac={args_like.test_frac}, split_seed={args_like.split_seed})")
    return dataset, train_idx, test_idx


def build_train_val_dataset(args_like, radius: float):
    """Build every graph once, then index it into a train and a validation subset.

    Returns ``(dataset, train_idx, val_idx)``. The held-out portion here is the
    VALIDATION set, used for checkpoint / hyperparameter selection. The final
    test set is supplied separately (a different folder), not carved out here.
    """
    pairs = _resolve_pairs(args_like)
    emb_dir = getattr(args_like, "emb_dir", None) if getattr(args_like, "use_lm_emb", False) else None
    dataset = RNPDataset(pairs, radius=radius, emb_dir=emb_dir)
    if len(dataset) == 0:
        raise RuntimeError("No structures could be built; nothing to train on.")

    train_names, val_names = split_names(dataset.names, args_like.val_frac,
                                         args_like.split_seed)
    train_set, val_set = set(train_names), set(val_names)
    train_idx = [i for i, n in enumerate(dataset.names) if n in train_set]
    val_idx = [i for i, n in enumerate(dataset.names) if n in val_set]
    print(f"[split] {len(train_idx)} train / {len(val_idx)} validation "
          f"(val_frac={args_like.val_frac}, split_seed={args_like.split_seed})")
    return dataset, train_idx, val_idx


def build_full_dataset(args_like, radius: float):
    """Build every discovered structure as one set (no split).

    Returns ``(dataset, all_idx)``. Use this for a SEPARATE test set that lives in
    its own folders -- every structure is evaluated, nothing held back.
    """
    pairs = _resolve_pairs(args_like)
    emb_dir = getattr(args_like, "emb_dir", None) if getattr(args_like, "use_lm_emb", False) else None
    dataset = RNPDataset(pairs, radius=radius, emb_dir=emb_dir)
    if len(dataset) == 0:
        raise RuntimeError("No structures could be built; nothing to evaluate.")
    print(f"[test-data] {len(dataset)} structure(s) for evaluation")
    return dataset, list(range(len(dataset)))


def make_train_loader(dataset, train_idx: List[int], batch_size: int = 1,
                      shuffle: bool = True) -> DataLoader:
    """DataLoader over the train subset only."""
    return DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, collate_fn=collate_graphs)
