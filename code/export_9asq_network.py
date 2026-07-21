"""
Build the 9asq complex graph exactly as the training/eval pipeline does
(dataloader.build_complex_graph: experimental anchors + AF3-filled missing
residues, sequence/spatial/interface edges) and export it as Cytoscape CSVs.

Usage:
    python export_9asq_network.py
"""
import csv
import os

import networkx as nx

from dataloader import build_complex_graph
from utils import CHAIN_PROTEIN, CHAIN_RNA, INTERFACE_CUTOFF
import model as eg

EXP_CIF = "PDB-Test/9asq.cif"
AF_CIF = "AlphaFold-Test/fold_9asq_model_0.cif"
RADIUS = 10.0          # spatial-edge radius (message-passing graph)
OUT_DIR = "AlphaFold-Test"
OUT_PREFIX = os.path.join(OUT_DIR, "9asq_graph")

# edge_type as produced by dataloader._build_edges: 0 backbone, 1 same molecule
# type within RADIUS, 2 cross-type (protein<->RNA) within RADIUS. The dedicated
# protein-RNA "true contact" set (utils.INTERFACE_CUTOFF, ~4A) is a separate,
# much tighter tensor (graph["interface_pairs"]) used for the interface loss --
# that tighter set is what's reported as "Interface" below.
EDGE_TYPE_NAME = {0: "sequence", 1: "spatial", 2: "interface"}


def main():
    graph = build_complex_graph(EXP_CIF, AF_CIF, radius=RADIUS, verbose=True)

    n = graph["x"].size(0)
    node_type = graph["node_type"].tolist()
    chain_id = graph["chain_id"].tolist()
    chain_index = graph["chain_index"].tolist()
    resnum = graph["resnum"].tolist()
    res_id = graph["res_id"].tolist()
    from dataloader import _RES_VOCAB
    resnames = [_RES_VOCAB[r] for r in res_id]

    # chain_index doesn't carry the original string label; a new chain starts
    # exactly where chain_index changes, so recover A/B/C/... labels from that.
    node_ids = []
    cur_label = None
    for i in range(n):
        if i == 0 or chain_index[i] != chain_index[i - 1]:
            cur_label = chr(ord('A') + chain_index[i]) if chain_index[i] < 26 else f"C{chain_index[i]}"
        node_ids.append(f"{cur_label}_{resnames[i]}{resnum[i]}")

    edge_index = graph["edge_index"]
    edge_type = graph["edge_type"].tolist()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Sequence (0) and same-type spatial (1) edges come straight from the
    # message-passing graph. Cross-type radius edges (2) are dropped in favor
    # of the tighter, purpose-built protein-RNA contact set below (interface_pairs) --
    # that's the real "true contact" definition the pipeline scores against.
    seen_edges = {}
    for s, d, et in zip(src, dst, edge_type):
        if et == 2:
            continue
        key = (s, d) if s < d else (d, s)
        seen_edges[key] = et

    for i, j in graph["interface_pairs"].tolist():
        key = (i, j) if i < j else (j, i)
        seen_edges[key] = 2  # interface, tight cutoff (utils.INTERFACE_CUTOFF)

    edges_path = f"{OUT_PREFIX}_edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "interaction", "edge_type_id",
                    "source_chain_type", "target_chain_type"])
        for (i, j), et in sorted(seen_edges.items()):
            ct_i = "protein" if chain_id[i] == CHAIN_PROTEIN else "RNA"
            ct_j = "protein" if chain_id[j] == CHAIN_PROTEIN else "RNA"
            w.writerow([node_ids[i], node_ids[j], EDGE_TYPE_NAME[et], et, ct_i, ct_j])

    # ---- node CSV ----
    nodes_path = f"{OUT_PREFIX}_nodes.csv"
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "chain_label", "chain_type", "resname", "resnum",
                    "node_role"])
        for i in range(n):
            role = "anchor" if node_type[i] == eg.EGNN.NODE_KNOWN_UNMASKED else "target"
            ct = "protein" if chain_id[i] == CHAIN_PROTEIN else "RNA"
            label = node_ids[i].split("_", 1)[0]
            w.writerow([node_ids[i], label, ct, resnames[i], resnum[i], role])

    # ---- stats, in the requested report shape ----
    n_anchor = sum(1 for t in node_type if t == eg.EGNN.NODE_KNOWN_UNMASKED)
    n_target = n - n_anchor
    n_seq = sum(1 for et in seen_edges.values() if et == 0)
    n_spatial = sum(1 for et in seen_edges.values() if et == 1)
    n_iface = sum(1 for et in seen_edges.values() if et == 2)
    n_edges_total = len(seen_edges)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(seen_edges.keys())
    avg_degree = (2 * n_edges_total / n) if n else 0.0
    n_components = nx.number_connected_components(G)

    print("\n" + "=" * 60)
    print(f"Nodes              : {n:6d}")
    print(f"  Anchor nodes     : {n_anchor:6d}  (experimental coords, frozen)")
    print(f"  Target nodes     : {n_target:6d}  (AF3-initialized coords, will update)")
    print(f"Edges              : {n_edges_total:6d}")
    print(f"  Sequence         : {n_seq:6d}  (backbone connectivity |i-j|<=1)")
    print(f"  Spatial          : {n_spatial:6d}  (Ca/C4' < {RADIUS:.0f}A, same molecule type)")
    print(f"  Interface        : {n_iface:6d}  (protein-RNA < {INTERFACE_CUTOFF:.1f}A)")
    print(f"Avg degree         : {avg_degree:6.1f}")
    print(f"Connected comps    : {n_components:6d}")
    print("=" * 60)
    print(f"\n[out] {nodes_path}")
    print(f"[out] {edges_path}")


if __name__ == "__main__":
    main()
