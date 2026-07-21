"""Plot per-chain RMSD distributions from the alignment diagnostics.

Reads the per-chain CSV written by dataset.py (default diagnostics/summary.csv)
and makes two bar charts, split Protein vs RNA:
    1. histogram of per-chain RMSD in fixed bins (<1, 1-2, 2-3, 3-5, 5-10, >10)
    2. summary stats (median, mean, 95th percentile)

Choose which alignment to plot with --metric:
    rmsd         per-chain alignment  (each chain fit on its own anchors)
    rmsd_global  global alignment     (one complex-wide transform, per-chain residual)

    python plot_alignment_rmsd.py --metric rmsd_global --out_dir plots
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the existing figures' look. dataset.py stores kind as "P" / "R".
KIND_PROTEIN, KIND_RNA = "P", "R"
COLORS = {"Protein": "tab:blue", "RNA": "tab:orange"}
BIN_EDGES = [0, 1, 2, 3, 5, 10, np.inf]
BIN_LABELS = ["<1", "1-2", "2-3", "3-5", "5-10", ">10"]


def binned_counts(values):
    counts, _ = np.histogram(values, bins=BIN_EDGES)
    return counts


def plot_histogram(prot, rna, metric, out_path):
    prot_counts = binned_counts(prot)
    rna_counts = binned_counts(rna)
    x = np.arange(len(BIN_LABELS))
    w = 0.4

    plt.figure(figsize=(9, 6))
    plt.bar(x - w / 2, prot_counts, w, label="Protein", color=COLORS["Protein"])
    plt.bar(x + w / 2, rna_counts, w, label="RNA", color=COLORS["RNA"])
    plt.xticks(x, BIN_LABELS)
    plt.xlabel("Per-chain RMSD (Å)")
    plt.ylabel("Number of chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_summary(prot, rna, metric, out_path):
    stats = ["Median", "Mean", "95th percentile"]

    def triplet(v):
        v = np.asarray(v, dtype=float)
        return [np.median(v), np.mean(v), np.percentile(v, 95)]

    prot_s = triplet(prot)
    rna_s = triplet(rna)
    x = np.arange(len(stats))
    w = 0.4

    plt.figure(figsize=(8, 6))
    plt.bar(x - w / 2, prot_s, w, label="Protein", color=COLORS["Protein"])
    plt.bar(x + w / 2, rna_s, w, label="RNA", color=COLORS["RNA"])
    plt.xticks(x, stats)
    plt.ylabel("RMSD (Å)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot per-chain alignment RMSD distributions")
    ap.add_argument("--csv", default="diagnostics/summary.csv",
                    help="per-chain CSV from dataset.py (default: diagnostics/summary.csv)")
    ap.add_argument("--metric", default="rmsd_global", choices=["rmsd", "rmsd_global"],
                    help="which column to plot (default: rmsd_global)")
    ap.add_argument("--out_dir", default="plots", help="folder for the PNGs (default: plots)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(
            f"Column '{args.metric}' not in {args.csv}. Re-run dataset.py to regenerate "
            f"the CSV with the {args.metric} column."
        )

    ok = df["status"] == "ok" if "status" in df.columns else slice(None)
    sub = df[ok]
    prot = sub.loc[sub["kind"] == KIND_PROTEIN, args.metric].dropna().to_numpy()
    rna = sub.loc[sub["kind"] == KIND_RNA, args.metric].dropna().to_numpy()
    if len(prot) == 0 and len(rna) == 0:
        raise SystemExit(f"No non-null '{args.metric}' values found.")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "global" if args.metric == "rmsd_global" else "perchain"
    hist_path = os.path.join(args.out_dir, f"rmsd_{tag}_histogram.png")
    summ_path = os.path.join(args.out_dir, f"rmsd_{tag}_summary.png")

    plot_histogram(prot, rna, args.metric, hist_path)
    plot_summary(prot, rna, args.metric, summ_path)

    print(f"[plot] {args.metric}: {len(prot)} protein / {len(rna)} RNA chain(s)")
    print(f"[plot] wrote {hist_path}")
    print(f"[plot] wrote {summ_path}")


if __name__ == "__main__":
    main()
