"""Perturb known residues and plot per-residue RMSD for selected test structures.

For each structure, only KNOWN residues are noisy-perturbed; missing residues
are left blank in the per-residue plot. The script writes one PNG and one CSV
per structure, plus an optional combined multi-panel figure.
"""
import argparse
import csv
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as eg
from dataloader import RNPDataset, discover_structures_folders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot known-residue noise RMSD for selected test structures."
    )
    parser.add_argument(
        "--pdb_dir", default="./PDB-Test",
        help="experimental CIF folder containing the test structures",
    )
    parser.add_argument(
        "--af_dir", default="./AlphaFold-Test",
        help="AlphaFold CIF folder containing the test structures",
    )
    parser.add_argument(
        "--names", default="9asq,9asm,9asn,9aso,9asp",
        help="comma-separated list of structure names to plot",
    )
    parser.add_argument(
        "--radius", type=float, default=12.0,
        help="radius graph cutoff used when building the graphs",
    )
    parser.add_argument(
        "--noise_std", type=float, nargs="+", default=[1.0],
        help="Gaussian noise std. One value applies uniformly; two values sample per-residue std from [min, max].",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for noise")
    parser.add_argument("--out_dir", default="./plots",
                        help="output folder for PNG/CSV files")
    parser.add_argument("--dpi", type=int, default=150, help="output image DPI")
    return parser.parse_args()


def select_names(structures: dict, names: List[str]) -> List[str]:
    found = []
    for name in names:
        if name in structures:
            found.append(name)
        else:
            print(f"[warn] structure '{name}' not found in {len(structures)} discovered entries")
    return found


def sample_noise_std(noise_std: List[float], n: int, generator: torch.Generator,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(noise_std) == 1:
        return torch.full((n, 1), float(noise_std[0]), device=device, dtype=dtype)
    if len(noise_std) == 2:
        lo, hi = float(noise_std[0]), float(noise_std[1])
        if lo > hi:
            lo, hi = hi, lo
        return torch.rand(n, 1, generator=generator, device=device, dtype=dtype) * (hi - lo) + lo
    raise ValueError("--noise_std must be one or two floats")


def perturb_known_residues(x: torch.Tensor, node_type: torch.Tensor,
                            noise_std: List[float], seed: int) -> torch.Tensor:
    noisy_x = x.clone()
    known_idx = (node_type == eg.EGNN.NODE_KNOWN_UNMASKED).nonzero(as_tuple=True)[0]
    if known_idx.numel() == 0:
        return noisy_x

    generator = torch.Generator(device=x.device).manual_seed(seed)
    std = sample_noise_std(noise_std, known_idx.numel(), generator, x.device, x.dtype)
    noise = torch.randn(known_idx.numel(), x.size(1), generator=generator,
                        device=x.device, dtype=x.dtype) * std
    noisy_x[known_idx] = noisy_x[known_idx] + noise
    return noisy_x


def write_csv(path: str, resnum: np.ndarray, known_mask: np.ndarray,
              rmsd: np.ndarray) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["resnum", "known", "rmsd"])
        for res, known, val in zip(resnum.tolist(), known_mask.tolist(), rmsd.tolist()):
            writer.writerow([res, int(known), "" if not known else f"{val:.6f}"])


def plot_rmsd(name: str, resnum: np.ndarray, known_mask: np.ndarray,
              rmsd: np.ndarray, out_path: str, dpi: int) -> None:
    x = np.arange(len(resnum))
    y = np.full(len(resnum), np.nan, dtype=float)
    y[known_mask] = rmsd[known_mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, color="tab:blue", lw=0.8, marker=".", markersize=4)
    ax.set_title(f"{name}: known-residue noise RMSD")
    ax.set_xlabel("Residue order")
    ax.set_ylabel("RMSD (Å)")
    ax.set_xlim(-0.5, len(resnum) - 0.5)
    if len(resnum) <= 40:
        tick_positions = x
        tick_labels = [str(r) for r in resnum.tolist()]
    else:
        step = max(1, len(resnum) // 20)
        tick_positions = x[::step]
        tick_labels = [str(r) for r in resnum[::step].tolist()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_combined(names: List[str], series: List[dict], out_path: str, dpi: int) -> None:
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name, data in zip(axes, names, series):
        x = np.arange(len(data["resnum"]))
        y = np.full(len(data["resnum"]), np.nan, dtype=float)
        y[data["known_mask"]] = data["rmsd"][data["known_mask"]]
        ax.plot(x, y, color="tab:blue", lw=0.8, marker=".", markersize=3)
        ax.set_title(name)
        ax.set_xlim(-0.5, len(data["resnum"]) - 0.5)
        ax.set_ylabel("RMSD (Å)")
        ax.grid(alpha=0.2)
        if len(data["resnum"]) <= 40:
            tick_positions = x
            tick_labels = [str(r) for r in data["resnum"].tolist()]
        else:
            step = max(1, len(data["resnum"]) // 20)
            tick_positions = x[::step]
            tick_labels = [str(r) for r in data["resnum"][::step].tolist()]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    axes[-1].set_xlabel("Residue order")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    cli = parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    names = [n.strip() for n in cli.names.split(",") if n.strip()]
    if not names:
        raise SystemExit("No structure names provided in --names.")

    structures = discover_structures_folders(cli.pdb_dir, cli.af_dir)
    selected = select_names(structures, names)
    if not selected:
        raise SystemExit("None of the requested structures were found.")

    pairs = {n: (structures[n]["exp_cif"], structures[n]["af_cif"]) for n in selected}
    dataset = RNPDataset(pairs, radius=cli.radius)
    if len(dataset) == 0:
        raise SystemExit("No graphs could be built from the requested structures.")

    index_by_name = {name: idx for idx, name in enumerate(dataset.names)}
    series = []

    for name in names:
        if name not in index_by_name:
            print(f"[skip] {name}: could not build graph or missing from dataset")
            continue
        idx = index_by_name[name]
        graph = dataset[idx]
        resnum = graph["resnum"].cpu().numpy()
        node_type = graph["node_type"]
        true_x = graph["x"]
        noisy_x = perturb_known_residues(true_x, node_type, cli.noise_std, cli.seed)

        rmsd = torch.linalg.norm(noisy_x - true_x, dim=-1).cpu().numpy()
        known_mask = (node_type == eg.EGNN.NODE_KNOWN_UNMASKED).cpu().numpy()

        png_path = os.path.join(cli.out_dir, f"{name}_known_noise_rmsd.png")
        csv_path = os.path.join(cli.out_dir, f"{name}_known_noise_rmsd.csv")
        plot_rmsd(name, resnum, known_mask, rmsd, png_path, cli.dpi)
        write_csv(csv_path, resnum, known_mask, rmsd)

        print(f"[plot] wrote {png_path} and {csv_path}")
        series.append({
            "resnum": resnum,
            "known_mask": known_mask,
            "rmsd": rmsd,
        })

    if len(series) > 1:
        combined_path = os.path.join(cli.out_dir, "combined_known_noise_rmsd.png")
        plot_combined([s for s in names if s in index_by_name], series, combined_path, cli.dpi)
        print(f"[plot] wrote {combined_path}")


if __name__ == "__main__":
    main()