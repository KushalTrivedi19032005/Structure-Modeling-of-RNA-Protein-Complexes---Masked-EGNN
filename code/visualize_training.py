"""
Render a movie of the refinement task for one complex:

  1. the ground-truth structure (known residues at their experimental
     coordinates, missing residues at the AF3 prior),
  2. the training perturbation -- a random fraction of known residues are
     "masked" and their coordinates are kicked by Gaussian noise,
  3. the trained EGNN pulling them back, shown layer by layer (each E_GCL
     updates the coordinates; we capture them with forward hooks).

Output is an animated GIF. This visualises what one training example looks
like, not the whole optimisation over epochs.

Usage:
    python visualize_training.py --checkpoint r8_l8.pt --name 8acb \
        --mask_fraction 0.3 --out training_movie.gif
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import model as eg
from dataloader import (build_complex_graph, discover_structures_folders,
                        PROTEIN_REP_ATOM, RNA_REP_ATOM, _RES_VOCAB)

from utils import CHAIN_PROTEIN

# Node-role colours.
C_ANCHOR = "#b0b0b0"   # known-unmasked: the frozen scaffold
C_MASKED = "#ff0000"   # known-masked: perturbed + refined (has ground truth)
C_MISSING = "#0095ff"  # missing: AF3 prior, refined (no ground truth)
C_TRUTH = "#ff00d4"    # ground-truth ghost for the masked residues


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/r8_l16_m0.5.pt")
    p.add_argument("--pdb_dir", default="./PDB-CIF")
    p.add_argument("--af_dir", default="./AlphaFold-CIF")
    p.add_argument("--name", default="8acb", help="complex to visualise (default: 8acb)")
    p.add_argument("--mask_fraction", type=float, default=0.3,
                   help="fraction of known residues masked+noised (default: 0.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="training_movie.gif")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--bg", choices=["white", "black"], default="black",
                   help="figure background colour (default: white)")
    p.add_argument("--elev", type=float, default=16.0, help="fixed camera elevation")
    p.add_argument("--azim", type=float, default=-60.0, help="fixed camera azimuth")
    p.add_argument("--write_traj", action="store_true",
                   help="also export the per-layer coordinates as a multi-state CIF "
                        "(model 1 = ground truth, 2 = noised input, 3.. = after each layer) "
                        "for ChimeraX")
    p.add_argument("--traj_out", default=None,
                   help="path for the multi-state CIF (default: <name>_traj.cif)")
    p.add_argument("--no_cuda", action="store_true", default=True)
    return p.parse_args()


def build_model(ckpt, device):
    a = ckpt["args"]
    model = eg.EGNN(
        in_node_nf=a["in_node_nf"], hidden_nf=a["hidden_nf"], out_node_nf=a["out_node_nf"],
        in_edge_nf=a["in_edge_nf"], node_type_emb_nf=a["node_type_emb_nf"],
        edge_type_emb_nf=a["edge_type_emb_nf"], device=device, n_layers=ckpt["n_layers"],
        attention=a["attention"], normalize=a["normalize"], tanh=a["tanh"],
        use_lm_emb=a.get("use_lm_emb", False), lm_emb_dim=a.get("lm_emb_dim", 640),
        lm_proj_dim=a.get("lm_proj_dim", 128),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, a


@torch.no_grad()
def run_with_trajectory(model, h, x_in, edges, edge_attr, node_type, edge_type, chain_id):
    """Forward pass capturing the coordinates after every E_GCL layer.

    Returns (pred_x, [x_after_layer_0, ..., x_after_layer_{n-1}]).
    """
    traj = []
    handles = []
    # E_GCL.forward returns (h, coord, edge_attr); grab coord each layer.
    for i in range(model.n_layers):
        layer = model._modules[f"gcl_{i}"]
        handles.append(layer.register_forward_hook(
            lambda mod, inp, out: traj.append(out[1].detach().cpu().numpy().copy())))
    _, pred_x = model(h, x_in, edges, edge_attr, node_type, edge_type, chain_id=chain_id)
    for hd in handles:
        hd.remove()
    return pred_x.detach().cpu().numpy(), traj


def write_traj_cif(path, name, states, labels, graph):
    """Write a multi-state (multi-model) CIF: one pseudo-atom per residue, one
    model per trajectory state, so ChimeraX can scrub through the refinement.

    Open in ChimeraX with:  open <path> coordsets true
    The B-factor column carries the node type (0 missing / 1 anchor / 2 masked)
    for colouring; `labels` is printed as the model->stage mapping.
    """
    chain_index = graph["chain_index"].tolist()
    chain_id = graph["chain_id"].tolist()
    res_id = graph["res_id"].tolist()
    resnum = graph["resnum"].tolist()
    node_type = graph["node_type"].tolist()
    n = len(res_id)

    lines = [
        f"data_{name}_traj", "#", "loop_",
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id", "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num",
    ]
    aid = 0
    for m, coords in enumerate(states, start=1):
        for i in range(n):
            atom = PROTEIN_REP_ATOM if chain_id[i] == CHAIN_PROTEIN else RNA_REP_ATOM
            comp = _RES_VOCAB[res_id[i]]
            asym = chr(ord('A') + chain_index[i] % 26)
            px, py, pz = (float(v) for v in coords[i])
            aid += 1
            lines.append(
                f'ATOM {aid} C "{atom}" . {comp} {asym} {chain_index[i] + 1} {resnum[i]} ? '
                f"{px:.3f} {py:.3f} {pz:.3f} 1.00 {float(node_type[i]):.2f} "
                f"{resnum[i]} {asym} {m}")
    lines.append("#")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[viz] wrote {path} ({len(states)} models):")
    for m, lab in enumerate(labels, start=1):
        print(f"        model {m}: {lab}")


def chain_traces(graph):
    """Node index lists per chain, ordered along the sequence (for backbone lines)."""
    chain_index = graph["chain_index"].tolist()
    seq_index = graph["seq_index"].tolist()
    order = {}
    for i, c in enumerate(chain_index):
        order.setdefault(c, []).append(i)
    for c in order:
        order[c].sort(key=lambda i: seq_index[i])
    return list(order.values())


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def build_frames(true_x, x_in, traj, pred_x, movable, masked, missing):
    """A list of (coords[N,3], subtitle, show_ghost) frames for the whole story."""
    frames = []
    hold_a, perturb, per_layer, hold_d = 12, 14, 3, 16

    # Phase A: ground truth.
    for _ in range(hold_a):
        frames.append((true_x.copy(), "1. ground-truth complex", False))

    # Phase B: kick the masked residues from truth to their noised positions.
    for k in range(1, perturb + 1):
        t = k / perturb
        c = true_x.copy()
        c[masked] = lerp(true_x[masked], x_in[masked], t)
        frames.append((c, "2. perturb: noise the masked residues", False))

    # Phase C: refinement. States are [x_in] then each layer's output.
    states = [x_in] + traj
    for s in range(len(states) - 1):
        a, b = states[s], states[s + 1]
        for k in range(1, per_layer + 1):
            t = k / per_layer
            frames.append((lerp(a, b, t), f"3. EGNN refinement (layer {s + 1}/{len(traj)})", True))

    # Phase D: hold on the prediction, ghost = truth for masked residues.
    for _ in range(hold_d):
        frames.append((pred_x.copy(), "4. refined vs. ground truth", True))
    return frames


def render(frames, traces, roles, true_x, out_path, fps, dpi, masked, rmsd,
           bg="white", elev=16.0, azim=-60.0):
    movable, masked_m, missing_m, anchor_m = roles
    all_xyz = np.concatenate([f[0] for f in frames] + [true_x], axis=0)
    lo, hi = all_xyz.min(0), all_xyz.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05

    bg_color = "black" if bg == "black" else "white"
    fg_color = "white" if bg == "black" else "black"
    trace_color = "#cccccc" if bg == "black" else C_ANCHOR

    fig = plt.figure(figsize=(8, 6.5), dpi=dpi, facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg_color)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    images = []
    for coords, subtitle, ghost in frames:
        ax.clear()
        ax.set_facecolor(bg_color)
        # backbone traces per chain
        for tr in traces:
            xyz = coords[tr]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=trace_color, lw=0.8, alpha=0.6)

        # ground-truth ghost of the masked residues, for comparison
        if ghost and masked_m.any():
            g = true_x[masked_m]
            ax.scatter(g[:, 0], g[:, 1], g[:, 2], color=C_TRUTH, s=14, alpha=0.35,
                       label="masked (truth)")

        # highlight movable node groups
        if missing_m.any():
            p = coords[missing_m]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=C_MISSING, s=10, alpha=0.9,
                       label="missing (AF3)")
        if masked_m.any():
            p = coords[masked_m]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=C_MASKED, s=18, alpha=0.95,
                       label="masked (predicted)")

        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_axis_off()
        try:                                    # zoom in to fill the frame (mpl >= 3.6)
            ax.set_box_aspect((1, 1, 1), zoom=1.5)
        except TypeError:
            pass
        ax.view_init(elev=elev, azim=azim)      # fixed camera (no orbit)

        sub = subtitle
        if subtitle.startswith("4."):
            sub += f"   (masked-residue RMSD = {rmsd:.2f} Å)"
        ax.set_title(sub, fontsize=13, pad=6, color=fg_color)
        leg = ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
        for txt in leg.get_texts():
            txt.set_color(fg_color)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        images.append(img)

    plt.close(fig)
    imageio.mimsave(out_path, images, fps=fps, loop=0)
    return len(images)


def main():
    args = get_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, a = build_model(ckpt, device)
    radius = ckpt["radius"]

    structs = discover_structures_folders(args.pdb_dir, args.af_dir)
    if args.name not in structs:
        raise SystemExit(f"{args.name} not found as a paired structure in "
                         f"{args.pdb_dir} / {args.af_dir}")
    ent = structs[args.name]
    graph = build_complex_graph(ent["exp_cif"], ent["af_cif"], radius=radius)
    n = graph["x"].size(0)
    print(f"[viz] {args.name}: {n} residues, radius {radius} A, {ckpt['n_layers']} layers")

    true_x = graph["x"].to(device)
    node_type = graph["node_type"].to(device)
    chain_id = graph["chain_id"].to(device)
    h = graph["h"][:, :a["in_node_nf"]].to(device)
    edges = [graph["edge_index"][0].to(device), graph["edge_index"][1].to(device)]
    edge_attr = graph["edge_attr"][:, :a["in_edge_nf"]].to(device)
    edge_type = graph["edge_type"].to(device)

    # Training-style perturbation: mask a fraction of known residues + noise them.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    masked_node_type, x_in, loss_mask = eg.mask_and_perturb(
        node_type, true_x, mask_fraction=args.mask_fraction,
        noise_std=a["noise_std"], generator=gen)

    pred_x, traj = run_with_trajectory(
        model, h, x_in, edges, edge_attr, masked_node_type, edge_type, chain_id)

    true_np, xin_np = true_x.cpu().numpy(), x_in.cpu().numpy()
    masked_m = loss_mask.cpu().numpy().astype(bool)
    missing_m = (node_type == eg.EGNN.NODE_MISSING).cpu().numpy().astype(bool)
    anchor_m = (node_type == eg.EGNN.NODE_KNOWN_UNMASKED).cpu().numpy().astype(bool)
    movable = ~anchor_m
    rmsd = float(np.sqrt(((pred_x[masked_m] - true_np[masked_m]) ** 2).sum(1).mean())) \
        if masked_m.any() else float("nan")

    print(f"[viz] masked {int(masked_m.sum())} / {int(anchor_m.sum()) + int(masked_m.sum())} "
          f"known residues; {int(missing_m.sum())} missing (AF3); "
          f"final masked-residue RMSD {rmsd:.2f} A")

    frames = build_frames(true_np, xin_np, traj, pred_x, movable, masked_m, missing_m)
    traces = chain_traces(graph)
    n_img = render(frames, traces, (movable, masked_m, missing_m, anchor_m),
                   true_np, args.out, args.fps, args.dpi, masked_m, rmsd,
                   bg=args.bg, elev=args.elev, azim=args.azim)
    print(f"[viz] wrote {args.out} ({n_img} frames, {args.fps} fps, bg={args.bg})")

    if args.write_traj:
        # model 1 = ground truth, 2 = noised input, 3.. = after each EGNN layer.
        states = [true_np, xin_np] + traj
        labels = ["ground truth", "noised input (masked residues perturbed)"] + \
                 [f"after layer {i + 1}/{len(traj)}" for i in range(len(traj))]
        traj_out = args.traj_out or f"{args.name}_traj.cif"
        write_traj_cif(traj_out, args.name, states, labels, graph)
        print(f"[viz] open in ChimeraX with:  open {traj_out} coordsets true")


if __name__ == "__main__":
    main()
