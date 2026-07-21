"""Precompute per-residue language-model embeddings and cache them per structure.

For every complex we build the SAME graph the training pipeline uses (so node
order matches exactly), reconstruct each chain's sequence, run:
    * ESM2   (esm2_t30_150M_UR50D, 640-d) on PROTEIN chains
    * RNA-FM (rna_fm_t12,          640-d) on RNA     chains
and scatter the per-residue vectors back into an [N, 640] tensor aligned to the
graph nodes. One file per structure is written to  <out_dir>/<name>.pt .

Training then just loads these (see dataloader.RNPDataset._attach_lm_emb) when
run with --use_lm_emb; nothing heavy runs during training.

Run ONCE (needs a GPU ideally, and the two model packages installed):

    pip install fair-esm                     # provides `import esm`
    # RNA-FM: the `fm` package ships in ../RNA-FM (repo you already have)

    python compute_embeddings.py \
        --pdb_dir ./PDB-CIF --af_dir ./AlphaFold-CIF \
        --esm_weights ../ESM-2/esm2_t30_150M_UR50D.pt \
        --rna_fm_dir ../RNA-FM \
        --rna_fm_weights ../RNA-FM/redevelop/pretrained/RNA-FM_pretrained.pth \
        --out_dir ./embeddings

Node order is radius-independent (radius only changes edges), so these caches are
valid for every radius in your sweep -- compute them once.
"""
import os
import sys
import argparse
import esm
import torch
import fm

# PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
# argparse.Namespace stored in the ESM2 / RNA-FM checkpoints. These are trusted
# local weight files, and ESM/RNA-FM call torch.load internally, so default
# weights_only=False for every load in this process (callers can still override).
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

from dataloader import (
    discover_structures_folders, discover_structures, extract_structure_cifs,
    RNPDataset, _RES_VOCAB,
)
from utils import CHAIN_PROTEIN, CHAIN_RNA


# --------------------------------------------------------------------------- #
# Residue name -> single-letter code
# --------------------------------------------------------------------------- #
# _RES_VOCAB stores protein residues as 3-letter codes and RNA as 1-letter.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    "MSE": "M",   # selenomethionine -> Met
    "SEC": "U",   # selenocysteine   -> U (ESM alphabet has it)
    "PYL": "O",   # pyrrolysine      -> O
}
# RNA is already single-letter in the vocab; inosine (I) -> A as a fallback.
_RNA_ONE = {"A": "A", "C": "C", "G": "G", "U": "U", "I": "A"}


def _resname_to_letter(resname: str, is_protein: bool) -> str:
    if is_protein:
        return _THREE_TO_ONE.get(resname, "X")   # unknown protein -> X
    return _RNA_ONE.get(resname, "N")             # unknown RNA -> N


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_core(pretrained_module, weights_path, device):
    """Load weights and build the model via the *core* loader with no regression
    head. `load_model_and_alphabet_local` insists on a co-located
    `<name>-contact-regression.pt` (contact-prediction head) that we neither have
    nor need for embeddings; the core loader takes `regression_data=None`.
    """
    from pathlib import Path
    model_data = torch.load(str(weights_path), map_location="cpu")
    model_name = Path(weights_path).stem
    model, alphabet = pretrained_module.load_model_and_alphabet_core(
        model_name, model_data, None)
    model = model.eval().to(device)
    return model, alphabet.get_batch_converter(), model.num_layers


def load_esm(weights_path, device):
    return _load_core(esm.pretrained, weights_path, device)   # last layer = 30 for t30_150M


def load_rna_fm(rna_fm_dir, weights_path, device):
    # The `fm` package lives inside the RNA-FM repo; put it on the path.
    if rna_fm_dir and rna_fm_dir not in sys.path:
        sys.path.insert(0, rna_fm_dir)
    # Use RNA-FM's own entry point: it selects theme="rna" (the RNA alphabet, not
    # ESM's protein one) and skips the contact-regression head for RNA-FM weights.
    model, alphabet = fm.pretrained.rna_fm_t12(model_location=weights_path)
    model = model.eval().to(device)
    return model, alphabet.get_batch_converter(), model.num_layers    # 12 layers


# --------------------------------------------------------------------------- #
# Per-chain embedding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_sequence(seq, model, batch_converter, repr_layer, device, max_len=1022):
    """Return a [len(seq), D] per-residue embedding for one chain.

    Sequences longer than the model's context are processed in non-overlapping
    windows and concatenated (a pragmatic choice; good enough for local features).
    """
    reps = []
    for start in range(0, len(seq), max_len):
        chunk = seq[start:start + max_len]
        _, _, tokens = batch_converter([("x", chunk)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer])["representations"][repr_layer]
        # drop BOS (index 0) and EOS (index len+1); keep the len(chunk) residues
        reps.append(out[0, 1:len(chunk) + 1].cpu())
    return torch.cat(reps, dim=0)


@torch.no_grad()
def embed_graph(graph, esm_bundle, rna_bundle, emb_dim, device):
    """Build the [N, emb_dim] embedding aligned to this graph's node order."""
    n = graph["x"].size(0)
    res_id = graph["res_id"].tolist()
    chain_id = graph["chain_id"].tolist()
    chain_index = graph["chain_index"].tolist()
    seq_index = graph["seq_index"].tolist()

    emb = torch.zeros(n, emb_dim, dtype=torch.float32)

    # group node indices by chain, ordered along the sequence
    chains = {}
    for i in range(n):
        chains.setdefault(chain_index[i], []).append(i)
    for c, idxs in chains.items():
        idxs.sort(key=lambda i: seq_index[i])
        is_protein = chain_id[idxs[0]] == CHAIN_PROTEIN
        seq = "".join(_resname_to_letter(_RES_VOCAB[res_id[i]], is_protein) for i in idxs)

        model, bc, layer = esm_bundle if is_protein else rna_bundle
        rep = embed_sequence(seq, model, bc, layer, device)   # [len(seq), D]
        if rep.size(0) != len(idxs):
            raise RuntimeError(f"embedding length {rep.size(0)} != chain length {len(idxs)}")
        for row, node_i in enumerate(idxs):
            emb[node_i] = rep[row]
    return emb


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Precompute ESM2/RNA-FM per-residue embeddings")
    p.add_argument("--pdb_dir", default="./PDB-CIF")
    p.add_argument("--af_dir", default="./AlphaFold-CIF")
    p.add_argument("--pdb_zip", default=None, help="zip mode (optional)")
    p.add_argument("--af_zip", default=None, help="zip mode (optional)")
    p.add_argument("--cache_dir", default="../.cache_cifs")
    p.add_argument("--esm_weights", default="../ESM-2/esm2_t30_150M_UR50D.pt")
    p.add_argument("--rna_fm_dir", default="../RNA-FM", help="repo dir containing the `fm` package")
    p.add_argument("--rna_fm_weights", default="../RNA-FM/redevelop/pretrained/RNA-FM_pretrained.pth")
    p.add_argument("--out_dir", default="./embeddings")
    p.add_argument("--emb_dim", type=int, default=640)
    p.add_argument("--radius", type=float, default=12.0, help="any value; node order is radius-independent")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the cache file exists")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def resolve_pairs(args):
    if args.pdb_zip and args.af_zip:
        structures = discover_structures(args.pdb_zip, args.af_zip, cache_dir=args.cache_dir)
        return {n: extract_structure_cifs(n, structures[n], args.pdb_zip, args.af_zip, args.cache_dir)
                for n in sorted(structures)}
    structures = discover_structures_folders(args.pdb_dir, args.af_dir)
    return {n: (structures[n]["exp_cif"], structures[n]["af_cif"]) for n in sorted(structures)}


def main():
    args = get_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[emb] loading ESM2   from {args.esm_weights}")
    esm_bundle = load_esm(args.esm_weights, device)
    print(f"[emb] loading RNA-FM from {args.rna_fm_weights}")
    rna_bundle = load_rna_fm(args.rna_fm_dir, args.rna_fm_weights, device)

    pairs = resolve_pairs(args)
    print(f"[emb] {len(pairs)} structure(s) discovered; building graphs for node order")
    # RNPDataset builds every graph once, skipping unbuildable ones -- the same set
    # (and same node order) training will see.
    dataset = RNPDataset(pairs, radius=args.radius)

    done, skipped = 0, 0
    for name, graph in zip(dataset.names, dataset.graphs):
        out_path = os.path.join(args.out_dir, f"{name}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            emb = embed_graph(graph, esm_bundle, rna_bundle, args.emb_dim, device)
            torch.save(emb, out_path)
            done += 1
            if done % 20 == 0:
                print(f"[emb] wrote {done} embedding file(s)...")
        except Exception as e:
            print(f"[emb] FAILED {name}: {e}")

    print(f"[emb] done: wrote {done}, skipped {skipped} (already cached) -> {args.out_dir}")


if __name__ == "__main__":
    main()
