import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import model as eg
from losses import compute_loss

def get_args():
    parser = argparse.ArgumentParser(description="Masked EGNN coordinate refinement")

    # Optimisation
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup_frac', type=float, default=0.1,
                        help='fraction of total training steps spent linearly warming up '
                             'the lr from 0 before cosine decay (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                        help='optimizer weight decay (default: 1e-16)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='B',
                        help='number of graphs per batch (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--patience', type=int, default=10,
                        help='stop after this many epochs without loss improvement; 0 disables (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='minimum loss decrease that counts as an improvement (default: 0.0)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disable CUDA even if available')

    # Masking / noise (the self-supervised objective)
    parser.add_argument('--mask_fraction', type=float, default=0.15,
                        help='fallback mask fraction if --mask_schedule is not used (default: 0.15)')
    parser.add_argument('--mask_schedule', type=float, nargs='+', default=[0.10],
                        help='per-epoch mask fractions, cycled over epochs (default: 0.10 0.20 0.30)')
    parser.add_argument('--noise_std', type=float, nargs=2, default=[1.0, 3.0],
                    help='uniform range [min max] for Gaussian noise std on masked nodes')
    parser.add_argument('--gdt_thresholds', type=float, nargs='+', default=[1.0, 2.0, 4.0, 8.0],
                        help='GDT-TS distance cutoffs in angstroms (default: 1 2 4 8)')

    # Loss weights (lambdas), one flag per term in compute_loss.
    parser.add_argument('--lambda_coord', type=float, default=0.7,
                        help='weight on coordinate MSE over masked nodes (default: 0.7)')
    parser.add_argument('--lambda_dRMSD', type=float, default=0.7,
                        help='weight on dRMSD over target pairs (default: 0.7)')
    parser.add_argument('--lambda_bond_prot', type=float, default=0.3,
                        help='weight on Ca-Ca bond term (default: 0.3)')
    parser.add_argument('--lambda_angle_prot', type=float, default=0.2,
                        help='weight on Ca virtual-angle hinge (default: 0.2)')
    parser.add_argument('--lambda_clash_prot', type=float, default=0.3,
                        help='weight on protein-protein clash term (default: 0.3)')
    parser.add_argument('--lambda_bond_rna', type=float, default=0.0,
                        help="weight on C4'-C4' bond term (default: 0.3)")
    parser.add_argument('--lambda_clash_rna', type=float, default=0.3,
                        help='weight on RNA-RNA clash term (default: 0.3)')
    parser.add_argument('--lambda_contact', type=float, default=0.2,
                        help='weight on interface contact term (default: 0.2)')

    # Train/validation split, checkpointing, held-out evaluation
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of the provided complexes held out for validation; '
                             'used for checkpoint/hyperparameter selection (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='(legacy) fraction held out for testing when a separate test set '
                             'is not supplied (default: 0.1)')
    parser.add_argument('--split_seed', type=int, default=0,
                        help='seed for the train/validation split; keep fixed across runs (default: 0)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='where to write one checkpoint per sweep trial (default: ./checkpoints)')
    parser.add_argument('--eval_mask_fraction', type=float, default=0.15,
                        help='mask fraction used when scoring held-out complexes (default: 0.15)')
    parser.add_argument('--eval_seed', type=int, default=1234,
                        help='seed for the held-out masks; fixed so every model is scored on the '
                             'identical masked nodes (default: 1234)')

    # Model hyper-parameters (mirror model.EGNN)
    parser.add_argument('--in_node_nf', type=int, default=1,
                        help='number of raw node features (excludes type embedding)')
    parser.add_argument('--in_edge_nf', type=int, default=0,
                        help='number of raw edge features (excludes type embedding)')
    parser.add_argument('--hidden_nf', type=int, default=64,
                        help='hidden feature width (default: 64)')
    parser.add_argument('--out_node_nf', type=int, default=1,
                        help='output node feature width (default: 1)')
    parser.add_argument('--node_type_emb_nf', type=int, default=8,
                        help='width of the node-type embedding (default: 8)')
    # Per-residue language-model embeddings (ESM2 for protein, RNA-FM for RNA).
    parser.add_argument('--use_lm_emb', action='store_true', default=False,
                        help='concatenate projected ESM2/RNA-FM per-residue embeddings onto node features')
    parser.add_argument('--emb_dir', type=str, default='./embeddings',
                        help='folder of cached <name>.pt embeddings (used when --use_lm_emb) (default: ./embeddings)')
    parser.add_argument('--lm_emb_dim', type=int, default=640,
                        help='dimension of the raw LM embeddings (ESM2/RNA-FM = 640) (default: 640)')
    parser.add_argument('--lm_proj_dim', type=int, default=128,
                        help='dimension the LM embeddings are projected to before concat (default: 128)')
    parser.add_argument('--edge_type_emb_nf', type=int, default=8,
                        help='width of the edge-type embedding (default: 8)')
    parser.add_argument('--n_layers', type=int, nargs='+', default=[4, 6, 8],
                        help='EGNN depth(s) to sweep; one training run per value (default: 4 6 8)')
    parser.add_argument('--attention', action='store_true', default=True,
                        help='use attention in the E_GCL layers')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='normalize coordinate messages')
    parser.add_argument('--tanh', action='store_true', default=True,
                        help='bound coordinate updates with a tanh')
    parser.add_argument('--pdb_dir', type=str, default='./PDB-CIF',
                        help='folder of experimental <name>.cif files (default: ./PDB-CIF)')
    parser.add_argument('--af_dir', type=str, default='./AlphaFold-CIF',
                        help='folder of fold_<name>_model_0.cif files (default: ./AlphaFold-CIF)')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='restrict to these structure names (default: all matched)')
    parser.add_argument('--pdb_zip', type=str, default=None,
                        help='PDB.zip.zip (gzipped experimental mmCIFs). Enables zip mode.')
    parser.add_argument('--af_zip', type=str, default=None,
                        help='AlphaFold.zip.zip (nested fold_<name>.zip). Enables zip mode.')
    parser.add_argument('--cache_dir', type=str, default='../.cache_cifs',
                        help='zip mode: folder for extracted CIFs + discovery index')
    parser.add_argument('--radius', type=float, nargs='+', default=[8.0, 10.0, 12.0],
                        help='radius-graph cutoff(s) in angstroms to sweep; one run per value (default: 8 10 12)')

    return parser.parse_args()


def lambdas_from_args(args) -> dict:
    """Collect the --lambda_* flags into the dict compute_loss expects."""
    return {
        "coord":      args.lambda_coord,
        "dRMSD":      args.lambda_dRMSD,
        "bond_prot":  args.lambda_bond_prot,
        "angle_prot": args.lambda_angle_prot,
        "clash_prot": args.lambda_clash_prot,
        "bond_rna":   args.lambda_bond_rna,
        "clash_rna":  args.lambda_clash_rna,
        "contact":    args.lambda_contact,
    }


GDT_THRESHOLDS = (1.0, 2.0, 4.0, 8.0)


def gdt_ts(pred, truth, thresholds=GDT_THRESHOLDS):
    """GDT-TS: mean over thresholds of the fraction of nodes within that cutoff.

    Args:
        pred:  [N, 3] predicted coordinates
        truth: [N, 3] ground-truth coordinates

    Returns:
        score:        scalar in [0, 1], the mean of the per-threshold fractions
        per_threshold: dict {threshold: fraction within it}, for logging
    """
    if pred.size(0) == 0:
        return 0.0, {t: 0.0 for t in thresholds}
    d = (pred - truth).norm(dim=-1)
    fracs = {t: (d <= t).float().mean().item() for t in thresholds}
    return sum(fracs.values()) / len(fracs), fracs


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_epoch(model, optimizer, data_loader, device, args, epoch,
                mask_fraction=None, scheduler=None):
    model.train()
    if mask_fraction is None:
        mask_fraction = args.mask_fraction

    thresholds = tuple(args.gdt_thresholds)
    lambdas = lambdas_from_args(args)

    running_loss = 0.0
    total_nodes = 0
    total_known = 0
    total_masked = 0
    # masked nodes within each GDT cutoff, pooled over every complex in the epoch
    within = {t: 0 for t in thresholds}
    # accumulate each loss component for logging
    comp_sums = {}

    pbar = tqdm(data_loader, desc=f"epoch {epoch}", leave=False)
    for graph in pbar:
        # --in_node_nf selects how many raw node features to feed; 0 means the
        # model runs on the node-type embedding alone.
        h = graph['h'][:, :args.in_node_nf].to(device)
        true_x = graph['x'].to(device)
        node_type = graph['node_type'].to(device)
        chain_id = graph['chain_id'].to(device)
        seq_index = graph['seq_index'].to(device)
        edges = [graph['edge_index'][0].to(device), graph['edge_index'][1].to(device)]
        edge_attr = graph['edge_attr'][:, :args.in_edge_nf].to(device)
        edge_type = graph['edge_type'].to(device)
        interface_pairs = graph['interface_pairs'].to(device)
        lm_emb = graph['lm_emb'].to(device) if 'lm_emb' in graph else None

        # Skip structures whose coordinates are corrupted: a failed per-chain
        # Kabsch alignment can place an AF-filled residue astronomically far
        # away (|x| ~ 1e16), which blows the bond/dRMSD terms up to inf/nan and
        # poisons the whole run. Real complexes span at most a few hundred A.
        if not torch.isfinite(true_x).all() or true_x.abs().max() > 1e4:
            continue

        # Self-supervised masking + perturbation of known (type-1) nodes.
        masked_node_type, x_in, loss_mask = eg.mask_and_perturb(
            node_type, true_x, mask_fraction=mask_fraction, noise_std=args.noise_std)

        optimizer.zero_grad()
        _, pred_x = model(h, x_in, edges, edge_attr, masked_node_type, edge_type,
                          lm_emb=lm_emb, chain_id=chain_id)
        losses = compute_loss(
            x_pred=pred_x,
            x_true=true_x,
            chain_id=chain_id,
            seq_index=seq_index,
            loss_mask=loss_mask,
            interface_pairs=interface_pairs,
            lambdas=lambdas,
        )
        loss = losses['total']

        # A single pathological structure (e.g. a poor alignment placing two
        # masked nodes almost coincident) can produce an exploding / non-finite
        # gradient. Skip that step so it never writes inf/nan into the weights,
        # and clip the gradient norm to keep the rest of training stable.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Per-step lr schedule (linear warmup -> cosine decay), if provided.
        if scheduler is not None:
            scheduler.step()

        # Metrics on the masked nodes.
        n_masked = int(loss_mask.sum())
        if n_masked > 0:
            with torch.no_grad():
                dist = torch.norm(pred_x[loss_mask] - true_x[loss_mask], dim=1)
                for t in thresholds:
                    within[t] += int((dist <= t).sum())

        running_loss += loss.item() * max(n_masked, 1)
        total_nodes += node_type.size(0)
        total_known += int((node_type == eg.EGNN.NODE_KNOWN_UNMASKED).sum())
        total_masked += n_masked
        for k, v in losses.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v)

        pbar.set_postfix(loss=f"{loss.item():.4f}", masked=n_masked)

    n_batches = max(len(data_loader), 1)
    avg_loss = running_loss / max(total_masked, 1)
    # Report masking relative to the KNOWN (type-1) nodes -- those are the only
    # nodes eligible to be masked -- so the number matches the --mask_fraction
    # knob (15% of knowns reads as 15%, not diluted by the AF-filled missing nodes).
    masking_pct = 100.0 * total_masked / max(total_known, 1)
    fracs = {t: within[t] / max(total_masked, 1) for t in thresholds}
    gdt = sum(fracs.values()) / len(fracs)
    return {
        'loss': avg_loss,
        'masking_pct': masking_pct,
        'total_nodes': total_nodes,
        'masked_nodes': total_masked,
        'gdt_ts': gdt,
        'gdt_fracs': fracs,
        'components': {k: v / n_batches for k, v in comp_sums.items()},
    }

