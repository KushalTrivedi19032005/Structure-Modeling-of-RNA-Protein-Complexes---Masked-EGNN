"""
Inference mode: run a trained checkpoint on complexes with NO masking and NO
noise. Known residues stay fixed at their experimental coordinates (frozen
anchors); only the genuinely-missing residues are refined from their AF3 prior
and written out. This is the deployment task -- fill in the unresolved residues,
leave the experimental structure untouched.

There is no accuracy metric here: the missing residues have no experimental
ground truth to score against (that is exactly why they are missing). Use
test.py for the scored (masked-known) evaluation.

Usage:
    python predict.py --checkpoint checkpoints/r8_l16_m0.5.pt \
        --pdb_dir ./PDB-Test --af_dir ./AlphaFold-Test --emb_dir ./embeddings-test \
        --cif_dir ./predictions
"""
import argparse
import os

import torch
from tqdm import tqdm

import model as eg
from dataloader import build_full_dataset
from evaluate import write_cif


class _Args:
    """Training args restored from the checkpoint (architecture + build settings)."""
    def __init__(self, d):
        self.__dict__.update(d)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='a .pt written by main.py')
    p.add_argument('--pdb_dir', default='./PDB-Test',
                   help='folder of experimental <name>.cif files')
    p.add_argument('--af_dir', default='./AlphaFold-Test',
                   help='folder of fold_<name>_model_0.cif files')
    p.add_argument('--emb_dir', default='./embeddings-test',
                   help='cached embeddings (only used if the model uses LM embeddings)')
    p.add_argument('--cif_dir', default='./predictions',
                   help='where to write <name>_pred.cif (default: ./predictions)')
    p.add_argument('--no_cuda', action='store_true', default=False)
    return p.parse_args()


@torch.no_grad()
def predict(model, dataset, indices, device, args, cif_dir):
    """Forward pass with the graph as-is: no mask_and_perturb, no noise.

    node_type is the ORIGINAL graph label, so the model freezes the known
    anchors (type 1) and refines only the missing residues (type 0) from their
    AF3-prior coordinates. Output keeps every known residue at its experimental
    coordinate and substitutes the model's coordinate only for missing residues.
    """
    model.eval()
    os.makedirs(cif_dir, exist_ok=True)
    rows = []

    for idx in tqdm(indices, desc="[predict]"):
        graph = dataset[idx]
        name = dataset.names[idx]

        h = graph['h'][:, :args.in_node_nf].to(device)
        x_in = graph['x'].to(device)                    # known = exp, missing = AF3 prior
        node_type = graph['node_type'].to(device)       # ORIGINAL labels: no masking
        chain_id = graph['chain_id'].to(device)
        edges = [graph['edge_index'][0].to(device), graph['edge_index'][1].to(device)]
        edge_attr = graph['edge_attr'][:, :args.in_edge_nf].to(device)
        edge_type = graph['edge_type'].to(device)
        lm_emb = graph['lm_emb'].to(device) if 'lm_emb' in graph else None

        # No mask_and_perturb: known anchors frozen, only missing residues move.
        _, pred_x = model(h, x_in, edges, edge_attr, node_type, edge_type,
                          lm_emb=lm_emb, chain_id=chain_id)

        is_missing = (node_type == eg.EGNN.NODE_MISSING).unsqueeze(-1)
        out_x = torch.where(is_missing, pred_x, x_in)   # missing -> predicted, known -> exp
        write_cif(os.path.join(cif_dir, f"{name}_pred.cif"), name, out_x.cpu(), graph)

        rows.append((name, int(node_type.size(0)), int(is_missing.sum())))

    return rows


def main():
    cli = get_args()
    ckpt = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    args = _Args(ckpt['args'])
    radius, n_layers = ckpt['radius'], ckpt['n_layers']

    device = torch.device("cuda" if (torch.cuda.is_available() and not cli.no_cuda) else "cpu")
    print(f"[predict] {cli.checkpoint}: radius={radius}A n_layers={n_layers} "
          f"(train loss {ckpt['train_loss']:.4f} @ epoch {ckpt['epoch']})")
    print("[predict] inference mode: NO masking, NO noise; known residues fixed, "
          "only missing residues refined")

    args.pdb_dir = cli.pdb_dir
    args.af_dir = cli.af_dir
    args.emb_dir = cli.emb_dir
    args.pdb_zip = None
    args.af_zip = None
    args.names = None

    dataset, all_idx = build_full_dataset(args, radius)
    if not all_idx:
        raise RuntimeError(f"No structures built from {cli.pdb_dir} / {cli.af_dir}.")

    model = eg.EGNN(
        in_node_nf=args.in_node_nf, hidden_nf=args.hidden_nf, out_node_nf=args.out_node_nf,
        in_edge_nf=args.in_edge_nf, node_type_emb_nf=args.node_type_emb_nf,
        edge_type_emb_nf=args.edge_type_emb_nf, device=device, n_layers=n_layers,
        attention=args.attention, normalize=args.normalize, tanh=args.tanh,
        use_lm_emb=getattr(args, 'use_lm_emb', False),
        lm_emb_dim=getattr(args, 'lm_emb_dim', 640),
        lm_proj_dim=getattr(args, 'lm_proj_dim', 128),
    )
    model.load_state_dict(ckpt['state_dict'])

    rows = predict(model, dataset, all_idx, device, args, cli.cif_dir)

    total_missing = sum(r[2] for r in rows)
    print(f"\n[predict] wrote {len(rows)} structure(s) to {cli.cif_dir}")
    print(f"[predict] placed {total_missing} missing residue(s) total")
    print(f"  {'name':<10} {'nodes':>7} {'missing_placed':>15}")
    for name, n, miss in rows:
        print(f"  {name:<10} {n:>7d} {miss:>15d}")


if __name__ == "__main__":
    main()
