"""
Extract per-structure metadata from every mmCIF in a folder and write a CSV.

Columns:
    name              PDB id (file stem)
    method            experimental method (_exptl.method): X-RAY, ELECTRON MICROSCOPY, NMR, ...
    resolution_A      resolution in angstroms (X-ray / cryo-EM; blank for NMR)
    mol_weight_Da     macromolecular weight = sum over POLYMER entities of
                      formula_weight * number_of_copies (excludes water/ions/ligands)
    mol_weight_kDa    the same in kilodaltons
    n_polymer_chains  number of polymer chain copies
    q_score           cryo-EM map-model Q-score -- NOT in the coordinate CIF, so
                      left as NA unless a validation field is present (see note below)

Molecular weight uses the deposited `_entity.formula_weight` (the true assembly
weight). If that is missing it falls back to summing atomic weights of the
polymer atoms actually present (an underestimate -- misses hydrogens/unmodelled
residues).

Usage:
    python extract_metadata.py --cif_dir ./PDB-CIF --out pdb_metadata.csv
"""
import argparse
import csv
import glob
import os

import gemmi
from tqdm import tqdm


def _num(v):
    """mmCIF token -> float or None (handles '?', '.', quotes, missing)."""
    if v is None:
        return None
    v = str(v).strip().strip('"\'')
    if v in ("", "?", "."):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def resolution(block, st):
    """Resolution in A, trying the common categories in priority order."""
    # gemmi parses many cases into st.resolution already
    if getattr(st, "resolution", 0):
        return float(st.resolution)
    for item in ("_refine.ls_d_res_high",
                 "_em_3d_reconstruction.resolution",
                 "_reflns.d_resolution_high"):
        r = _num(block.find_value(item))
        if r is not None:
            return r
    return None


def macromolecular_weight(block):
    """(weight_Da, n_polymer_chains) summed over polymer entities.

    weight = sum_entity formula_weight * pdbx_number_of_molecules, polymer only.
    """
    tab = block.find(["_entity.type", "_entity.formula_weight",
                      "_entity.pdbx_number_of_molecules"])
    total, n_chains = 0.0, 0
    found = False
    for row in tab:
        etype = str(row[0]).strip().strip('"\'').lower()
        if etype != "polymer":
            continue
        fw = _num(row[1])
        ncopies = _num(row[2]) or 1
        if fw is not None:
            total += fw * ncopies
            n_chains += int(ncopies)
            found = True
    return (total if found else None), n_chains


def weight_from_atoms(st):
    """Fallback MW: sum atomic weights of polymer atoms in model 0 (underestimate)."""
    if len(st) == 0:
        return None
    total = 0.0
    seen = False
    for chain in st[0]:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not (info.is_amino_acid() or info.is_nucleic_acid()):
                continue
            for atom in res:
                total += atom.element.weight
                seen = True
    return total if seen else None


def q_score(block):
    """Q-score is a cryo-EM map-model metric, not in the coordinate CIF.

    Returned only if some deposition happens to carry it in a validation field;
    otherwise None (written as NA).
    """
    for item in ("_em_validation.Q_score", "_pdbx_vrpt_summary.Q_score"):
        q = _num(block.find_value(item))
        if q is not None:
            return q
    return None


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cif_dir", default="./PDB-CIF")
    p.add_argument("--out", default="pdb_metadata.csv")
    return p.parse_args()


def main():
    args = get_args()
    files = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")) +
                   glob.glob(os.path.join(args.cif_dir, "*.mmcif")))
    print(f"[meta] {len(files)} cif file(s) in {args.cif_dir}")

    rows, errors = [], []
    for path in tqdm(files, desc="[meta] extracting"):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            block = gemmi.cif.read(path).sole_block()
            st = gemmi.read_structure(path)
            method = block.find_value("_exptl.method")
            method = (str(method).strip().strip('"\'') if method
                      and str(method).strip() not in ("?", ".") else "")
            res = resolution(block, st)
            mw, n_chains = macromolecular_weight(block)
            if mw is None:
                mw = weight_from_atoms(st)
            q = q_score(block)
            rows.append({
                "name": name,
                "method": method,
                "resolution_A": f"{res:.2f}" if res is not None else "",
                "mol_weight_Da": f"{mw:.1f}" if mw is not None else "",
                "mol_weight_kDa": f"{mw / 1000.0:.2f}" if mw is not None else "",
                "n_polymer_chains": n_chains,
                "q_score": f"{q:.3f}" if q is not None else "NA",
            })
        except Exception as e:
            errors.append((name, str(e)[:80]))

    fields = ["name", "method", "resolution_A", "mol_weight_Da", "mol_weight_kDa",
              "n_polymer_chains", "q_score"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[meta] wrote {args.out} ({len(rows)} rows, {len(errors)} error(s))")
    for n, e in errors[:10]:
        print(f"  err {n}: {e}")


if __name__ == "__main__":
    main()
