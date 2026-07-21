import os
import csv
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser


def count_observed_residues(structure):
    """
    Count unique observed polymer residues from coordinates.
    """
    residues = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resseq, icode = residue.id

                # Ignore waters and heteroatoms
                if hetflag != " ":
                    continue

                residues.add((chain.id, resseq, icode))

    return len(residues)


def count_missing_residues(cif_dict):
    """
    Count unique missing residues listed in
    _pdbx_unobs_or_zero_occ_residues.
    """
    seq_ids = cif_dict.get("_pdbx_unobs_or_zero_occ_residues.auth_seq_id")

    if seq_ids is None:
        return 0

    chain_ids = cif_dict.get("_pdbx_unobs_or_zero_occ_residues.auth_asym_id")

    # Single missing residue case
    if isinstance(seq_ids, str):
        return 1

    missing = set(zip(chain_ids, seq_ids))
    return len(missing)


def process_folder(folder):
    parser = MMCIFParser(QUIET=True)

    results = []

    for file in sorted(os.listdir(folder)):
        if not file.endswith(".cif"):
            continue

        path = os.path.join(folder, file)

        try:
            cif_dict = MMCIF2Dict(path)
            structure = parser.get_structure(file, path)

            observed = count_observed_residues(structure)
            missing = count_missing_residues(cif_dict)

            total = observed + missing

            if total == 0:
                missing_percent = 0.0
            else:
                missing_percent = 100.0 * missing / total

            results.append({
                "complex": os.path.splitext(file)[0],
                "missing_residue_percent": round(missing_percent, 3)
            })

            print(f"{file}: observed={observed}, missing={missing}, percent={missing_percent:.2f}%")

        except Exception as e:
            print(f"Failed on {file}: {e}")

    return results


def write_csv(results, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "complex",
                "missing_residue_percent"
            ]
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    cif_folder = "PDB-CIF"
    output_csv = "missing_residue_percent.csv"

    results = process_folder(cif_folder)
    write_csv(results, output_csv)

    print(f"\nSaved results to {output_csv}")