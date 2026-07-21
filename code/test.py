import argparse
import csv
import os

import torch

import model as eg
from dataloader import build_full_dataset
from evaluate import evaluate


def get_test_args():
    p = argparse.ArgumentParser(description="Evaluate an EGNN checkpoint on a held-out test set")
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to a .pt written by main.py')
    # The test set lives in its own folders (separate from the training data).
    p.add_argument('--pdb_dir', type=str, default='./pdb-test',
                   help='folder of experimental test <name>.cif files (default: ./pdb-test)')
    p.add_argument('--af_dir', type=str, default='./alphafold-test',
                   help='folder of fold_<name>_model_0.cif test files (default: ./alphafold-test)')
    p.add_argument('--emb_dir', type=str, default='./embeddings-test',
                   help='cached embeddings for the TEST structures (used if the model uses LM embeddings)')
    p.add_argument('--out_csv', type=str, default='./test_results.csv',
                   help='per-complex results (default: ./test_results.csv)')
    p.add_argument('--write_cif', action='store_true',
                   help='also dump predicted coordinates as CIF')
    p.add_argument('--cif_dir', type=str, default='./predictions',
                   help='where to write predicted CIFs (default: ./predictions)')
    p.add_argument('--no_cuda', action='store_true', default=False)
    return p.parse_args()


class _Args:
    """Training args restored from the checkpoint, so the split and the eval mask
    match the run that produced it."""
    def __init__(self, d):
        self.__dict__.update(d)


def main():
    cli = get_test_args()
    ckpt = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    args = _Args(ckpt['args'])
    radius, n_layers = ckpt['radius'], ckpt['n_layers']

    device = torch.device("cuda" if (torch.cuda.is_available() and not cli.no_cuda) else "cpu")
    print(f"[test] {cli.checkpoint}: radius={radius}A n_layers={n_layers} "
          f"(train loss {ckpt['train_loss']:.4f} @ epoch {ckpt['epoch']})")

    # Point the (checkpoint-restored) args at the SEPARATE test folders + test
    # embeddings; the model architecture/eval settings still come from the ckpt.
    args.pdb_dir = cli.pdb_dir
    args.af_dir = cli.af_dir
    args.emb_dir = cli.emb_dir
    args.pdb_zip = None
    args.af_zip = None
    args.names = None

    # Build the whole test folder (no split -- every structure is evaluated), at
    # the checkpoint's radius so the graph edges match how the model was trained.
    dataset, test_idx = build_full_dataset(args, radius)
    if not test_idx:
        raise RuntimeError(f"No test structures built from {cli.pdb_dir} / {cli.af_dir}.")

    model = eg.EGNN(
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        out_node_nf=args.out_node_nf,
        in_edge_nf=args.in_edge_nf,
        node_type_emb_nf=args.node_type_emb_nf,
        edge_type_emb_nf=args.edge_type_emb_nf,
        device=device,
        n_layers=n_layers,
        attention=args.attention,
        normalize=args.normalize,
        tanh=args.tanh,
        use_lm_emb=getattr(args, 'use_lm_emb', False),
        lm_emb_dim=getattr(args, 'lm_emb_dim', 640),
        lm_proj_dim=getattr(args, 'lm_proj_dim', 128),
    )
    model.load_state_dict(ckpt['state_dict'])

    res = evaluate(model, dataset, test_idx, device, args,
                   cif_dir=cli.cif_dir if cli.write_cif else None)
    rows, mean = res['per_complex'], res['mean']

    print(f"\n=== held-out test: {len(rows)} complex(es) ===")
    print(f"{'complex':<10} {'nodes':>6} {'masked':>7} {'GDT_TS':>8} {'base':>8} "
          f"{'delta':>7} {'RMSD':>7}")
    for r in sorted(rows, key=lambda r: r['gdt_ts'], reverse=True):
        print(f"{r['name']:<10} {r['n_nodes']:>6d} {r['n_masked']:>7d} "
              f"{r['gdt_ts']:>8.4f} {r['baseline_gdt_ts']:>8.4f} "
              f"{r['delta_gdt_ts']:>+7.4f} {r['rmsd']:>7.2f}")

    print(f"\nmean GDT-TS   {mean['gdt_ts']:.4f}   "
          f"(noised-input baseline {mean['baseline_gdt_ts']:.4f}, "
          f"delta {mean['delta_gdt_ts']:+.4f})")
    frac_keys = [k for k in mean if k.startswith('frac@')]
    print("  " + "  ".join(f"{k}A={mean[k]:.3f}" for k in frac_keys))
    print(f"mean RMSD     {mean['rmsd']:.2f} A   "
          f"(baseline {mean['baseline_rmsd']:.2f} A)")
    if mean['delta_gdt_ts'] <= 0:
        print("\n[warn] the model did NOT beat its own noised input: it is not "
              "learning anything useful on held-out structures.")

    with open(cli.out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[test] wrote {cli.out_csv}")
    if cli.write_cif:
        print(f"[test] wrote {len(rows)} CIF(s) to {cli.cif_dir}")


if __name__ == "__main__":
    main()
