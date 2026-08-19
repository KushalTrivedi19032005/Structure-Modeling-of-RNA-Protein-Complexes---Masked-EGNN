"""
Count nodes and edges across all paired structures, with total / mean / max.

Nodes  = residues (one representative Cα/C4' node each).
Edges  = sequence + spatial + interface edges built by dataloader at --radius.
         edge_index stores each undirected edge twice (both directions), so we
         report both the directed count (as stored) and the undirected unique count.

Usage:
    python graph_stats.py --pdb_dir ./PDB-CIF --af_dir ./AlphaFold-CIF --radius 8
"""
import argparse
import numpy as np

from dataloader import build_complex_graph, discover_structures_folders


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb_dir", default="./PDB-CIF")
    p.add_argument("--af_dir", default="./AlphaFold-CIF")
    p.add_argument("--radius", type=float, default=8.0,
                   help="spatial-edge cutoff; edge counts depend on it (default: 8)")
    return p.parse_args()


def main():
    args = get_args()
    structs = discover_structures_folders(args.pdb_dir, args.af_dir)
    names = sorted(structs)
    print(f"discovered {len(names)} paired structures; building at radius {args.radius} A")

    nodes, edges_dir, edges_undir, skipped = [], [], [], []
    for i, name in enumerate(names, 1):
        try:
            g = build_complex_graph(structs[name]["exp_cif"], structs[name]["af_cif"],
                                    radius=args.radius)
            ei = g["edge_index"].numpy()
            nodes.append(int(g["x"].size(0)))
            edges_dir.append(int(ei.shape[1]))
            edges_undir.append(len({tuple(p) for p in np.sort(ei, axis=0).T.tolist()}))
        except Exception as e:
            skipped.append((name, str(e)[:70]))
        if i % 50 == 0:
            print(f"  ...{i}/{len(names)}")

    nc, ed, eu = np.array(nodes), np.array(edges_dir), np.array(edges_undir)

    def report(label, a):
        print(f"\n{label}")
        print(f"  total : {a.sum():,}")
        print(f"  mean  : {a.mean():.1f}")
        print(f"  max   : {a.max():,}")
        print(f"  min   : {a.min():,}")

    print(f"\n================ GRAPH STATS (radius {args.radius} A) ================")
    print(f"structures built : {len(nc)}  (skipped {len(skipped)})")
    report("NODES (residues)", nc)
    report("EDGES (directed, as stored in edge_index)", ed)
    report("EDGES (undirected unique pairs)", eu)
    if skipped:
        print("\nskipped:")
        for n, e in skipped:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
