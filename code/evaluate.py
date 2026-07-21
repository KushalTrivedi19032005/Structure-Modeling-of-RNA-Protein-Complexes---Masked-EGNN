"""Held-out evaluation: score a trained model on complexes it never saw.

The training loop scores masked nodes it is actively optimising on that step;
this module scores a frozen model on withheld complexes, under a *fixed* mask so
that every model in the sweep is judged on the identical nodes and noise.
"""
import os
from typing import Dict, List, Optional

import torch

import model as eg
from dataloader import _RES_VOCAB, PROTEIN_REP_ATOM, RNA_REP_ATOM
from train import gdt_ts
from utils import CHAIN_PROTEIN


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.size(0) == 0:
        return float('nan')
    return float(torch.sqrt(((a - b) ** 2).sum(dim=-1).mean()))


@torch.no_grad()
def evaluate(model, dataset, indices: List[int], device, args,
             cif_dir: Optional[str] = None) -> Dict:
    """Score `model` on the complexes at `indices` of `dataset`.

    Each complex is masked with its own fixed-seed generator, so the mask is a
    property of the complex, not of the model or the run order. Returns a dict
    with 'per_complex' (one row per structure) and 'mean' (dataset-level means).
    """
    model.eval()
    thresholds = tuple(args.gdt_thresholds)
    rows = []

    if cif_dir:
        os.makedirs(cif_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        graph = dataset[idx]
        name = dataset.names[idx]

        # match train_epoch: --in_node_nf selects how many raw features to feed
        h = graph['h'][:, :args.in_node_nf].to(device)
        true_x = graph['x'].to(device)
        node_type = graph['node_type'].to(device)
        chain_id = graph['chain_id'].to(device)
        edges = [graph['edge_index'][0].to(device), graph['edge_index'][1].to(device)]
        edge_attr = graph['edge_attr'][:, :args.in_edge_nf].to(device)
        edge_type = graph['edge_type'].to(device)
        lm_emb = graph['lm_emb'].to(device) if 'lm_emb' in graph else None

        # Fixed per-complex mask: same nodes, same noise, for every model scored.
        gen = torch.Generator(device=device).manual_seed(args.eval_seed + i)
        masked_node_type, x_in, loss_mask = eg.mask_and_perturb(
            node_type, true_x, mask_fraction=args.eval_mask_fraction,
            noise_std=args.noise_std, generator=gen)

        _, pred_x = model(h, x_in, edges, edge_attr, masked_node_type, edge_type,
                          lm_emb=lm_emb, chain_id=chain_id)

        pred_m, true_m, in_m = pred_x[loss_mask], true_x[loss_mask], x_in[loss_mask]
        score, fracs = gdt_ts(pred_m, true_m, thresholds)
        # Baseline: the noised input the model was handed. Beating it is the
        # minimum bar -- it is what "do nothing" would have scored.
        base_score, base_fracs = gdt_ts(in_m, true_m, thresholds)

        rows.append({
            'name': name,
            'n_nodes': int(node_type.size(0)),
            'n_masked': int(loss_mask.sum()),
            'n_missing': int((node_type == eg.EGNN.NODE_MISSING).sum()),
            'gdt_ts': score,
            **{f'frac@{t:g}': fracs[t] for t in thresholds},
            'rmsd': _rmsd(pred_m, true_m),
            'baseline_gdt_ts': base_score,
            'baseline_rmsd': _rmsd(in_m, true_m),
            'delta_gdt_ts': score - base_score,
        })

        if cif_dir:
            # Output only moves the genuinely-MISSING (AF-predicted, type-0)
            # residues; every KNOWN residue keeps its exact experimental PDB
            # coordinate. `node_type` here is the original graph label (before the
            # scoring mask), so masked-known nodes are still treated as known and
            # are NOT written with their moved coordinates.
            is_missing = (node_type == eg.EGNN.NODE_MISSING).unsqueeze(-1)
            out_x = torch.where(is_missing, pred_x, true_x)
            write_cif(os.path.join(cif_dir, f"{name}_pred.cif"), name,
                      out_x.cpu(), graph)

    keys = [k for k in rows[0] if k != 'name'] if rows else []
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}
    return {'per_complex': rows, 'mean': mean}


def write_cif(path: str, name: str, x: torch.Tensor, graph: Dict) -> None:
    """Write predicted coords as a minimal CIF: one pseudo-atom per residue.

    One representative atom per node (CA for protein, C4' for RNA) -- that is all
    the model predicts. Enough for PyMOL/ChimeraX to draw a trace. Includes the
    genuinely-missing (type-0) residues the model inpainted; the B-factor column
    carries the node type so they can be selected and coloured.
    """
    chain_index = graph['chain_index'].tolist()
    chain_id = graph['chain_id'].tolist()
    res_id = graph['res_id'].tolist()
    resnum = graph['resnum'].tolist()
    node_type = graph['node_type'].tolist()

    lines = [
        f"data_{name}_pred",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    for i in range(x.size(0)):
        # Quoted, because the RNA representative atom name C4' contains a quote.
        atom = PROTEIN_REP_ATOM if chain_id[i] == CHAIN_PROTEIN else RNA_REP_ATOM
        comp = _RES_VOCAB[res_id[i]]
        asym = chr(ord('A') + chain_index[i] % 26)
        px, py, pz = (float(v) for v in x[i])
        lines.append(
            f'ATOM {i + 1} C "{atom}" . {comp} {asym} {chain_index[i] + 1} {resnum[i]} ? '
            f"{px:.3f} {py:.3f} {pz:.3f} 1.00 {float(node_type[i]):.2f} "
            f"{resnum[i]} {asym} 1"
        )
    lines.append("#")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
