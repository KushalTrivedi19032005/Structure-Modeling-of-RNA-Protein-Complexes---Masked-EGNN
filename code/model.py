from torch import nn
import torch


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=True, normalize=True, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.radial_max = 1e4

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, node_update_mask=None):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        if node_update_mask is not None:
            # Gate the coordinate update so frozen (anchor) nodes never move.
            # They still influence neighbours through coord_diff above.
            agg = agg * node_update_mask
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        # Cap the squared-distance edge feature so a node that drifts far can't
        # inject an unbounded value into edge_mlp (which drives runaway
        # coordinate updates). Real edges are within the radius cutoff (<=12 A,
        # radial <=144), so this only clamps pathological blow-ups, not normal
        # geometry. clamp() zeroes the gradient above the cap, which is what we
        # want (no signal pushing a node even further out).
        radial = radial.clamp(max=self.radial_max)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_update_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, node_update_mask=node_update_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    # Node type codes used throughout the refinement task.
    NODE_MISSING = 0          # residue absent from the PDB; AF3-predicted prior
    NODE_KNOWN_UNMASKED = 1   # residue with known coords, used as a frozen anchor
    NODE_KNOWN_MASKED = 2     # known residue masked during training (supervised target)

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 node_type_emb_nf=8, edge_type_emb_nf=8, n_node_types=3, n_edge_types=3,
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=True, normalize=True, tanh=False,
                 use_lm_emb=False, lm_emb_dim=640, lm_proj_dim=128):
        '''

        :param in_node_nf: Number of *raw* features for 'h' at the input (excludes the node-type embedding)
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of *raw* edge features (excludes the edge-type embedding)
        :param node_type_emb_nf: Width of the learned node-type embedding (types 0/1/2)
        :param edge_type_emb_nf: Width of the learned edge-type embedding (types 0/1/2)
        :param n_node_types: Number of distinct node types (default 3)
        :param n_edge_types: Number of distinct edge types (default 3)
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Learned embeddings for the discrete node/edge type labels.
        self.node_type_emb = nn.Embedding(n_node_types, node_type_emb_nf)
        self.edge_type_emb = nn.Embedding(n_edge_types, edge_type_emb_nf)

        # Optional per-residue language-model features: ESM2 (protein) and RNA-FM
        # (RNA), both `lm_emb_dim`-wide, each projected to `lm_proj_dim` by its own
        # linear and concatenated onto the node input. Separate projections because
        # the two live in different embedding spaces.
        self.use_lm_emb = use_lm_emb
        self.lm_emb_dim = lm_emb_dim
        self.lm_proj_dim = lm_proj_dim
        lm_extra = 0
        if use_lm_emb:
            self.esm_proj = nn.Linear(lm_emb_dim, lm_proj_dim)      # protein (chain_id==0)
            self.rnafm_proj = nn.Linear(lm_emb_dim, lm_proj_dim)    # RNA     (chain_id==1)
            lm_extra = lm_proj_dim

        # Raw features are concatenated with the type embeddings (and the projected
        # LM embedding, if enabled) before the network.
        self.embedding_in = nn.Linear(in_node_nf + node_type_emb_nf + lm_extra, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        edges_in_d = in_edge_nf + edge_type_emb_nf
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, node_type, edge_type, node_update_mask=None,
                lm_emb=None, chain_id=None):
        '''
        :param h: [N, in_node_nf] raw node features (may be empty, i.e. in_node_nf == 0)
        :param x: [N, 3] coordinates (the AF3 prior; masked/missing entries are the refinement targets)
        :param edges: [rows, cols] edge index
        :param edge_attr: [E, in_edge_nf] raw edge features (may be None when in_edge_nf == 0)
        :param node_type: [N] long tensor of node-type codes (0/1/2)
        :param edge_type: [E] long tensor of edge-type codes (0/1/2)
        :param node_update_mask: optional [N, 1] float mask (1 = movable, 0 = frozen). If None it is
                                 derived from node_type so that only type-1 (known-unmasked) anchors are frozen.
        :return: (h, x) refined node embeddings and refined coordinates
        '''
        # Concatenate raw node features with the node-type embedding.
        node_type_h = self.node_type_emb(node_type)
        parts = []
        if h is not None and h.size(1) > 0:
            parts.append(h)
        parts.append(node_type_h)

        # Project the per-residue LM embedding (640 -> lm_proj_dim) and concat onto
        # h. Protein nodes (chain_id==0) go through esm_proj, RNA nodes (==1) through
        # rnafm_proj; each node gets exactly one, routed by chain_id.
        if self.use_lm_emb:
            if lm_emb is None or chain_id is None:
                raise ValueError("use_lm_emb=True but lm_emb / chain_id were not passed to forward().")
            lm_emb = lm_emb.to(node_type_h.dtype)
            proj = node_type_h.new_zeros(lm_emb.size(0), self.lm_proj_dim)
            is_prot = (chain_id == 0)   # CHAIN_PROTEIN
            is_rna = (chain_id == 1)    # CHAIN_RNA
            if is_prot.any():
                proj[is_prot] = self.esm_proj(lm_emb[is_prot])
            if is_rna.any():
                proj[is_rna] = self.rnafm_proj(lm_emb[is_rna])
            parts.append(proj)

        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        h = self.embedding_in(h)

        # Concatenate raw edge features with the edge-type embedding.
        edge_type_h = self.edge_type_emb(edge_type)
        if edge_attr is not None and edge_attr.size(1) > 0:
            edge_attr = torch.cat([edge_attr, edge_type_h], dim=1)
        else:
            edge_attr = edge_type_h

        # Anchors (type 1) stay fixed; missing (0) and masked (2) nodes are refined.
        if node_update_mask is None:
            node_update_mask = (node_type != self.NODE_KNOWN_UNMASKED).unsqueeze(-1).to(x.dtype)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                  node_update_mask=node_update_mask)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def mask_and_perturb(node_type, coords, mask_fraction, noise_std, generator=None):
    new_node_type = node_type.clone()
    noised_coords = coords.clone()
    loss_mask = torch.zeros(node_type.size(0), dtype=torch.bool, device=node_type.device)

    known_idx = (node_type == EGNN.NODE_KNOWN_UNMASKED).nonzero(as_tuple=True)[0]
    n_known = known_idx.numel()
    if n_known == 0 or mask_fraction <= 0:
        return new_node_type, noised_coords, loss_mask

    n_mask = int(round(mask_fraction * n_known))
    n_mask = max(0, min(n_mask, n_known))
    if n_mask == 0:
        return new_node_type, noised_coords, loss_mask

    perm = torch.randperm(n_known, generator=generator, device=node_type.device)
    chosen = known_idx[perm[:n_mask]]

    new_node_type[chosen] = EGNN.NODE_KNOWN_MASKED
    loss_mask[chosen] = True

    # noise_std is either a scalar or a (min, max) range; for a range each masked
    # node draws its own std, so the model sees a spread of corruption severities.
    if isinstance(noise_std, (list, tuple)):
        lo, hi = float(noise_std[0]), float(noise_std[1])
        std = torch.rand(n_mask, 1, generator=generator, device=coords.device,
                         dtype=coords.dtype) * (hi - lo) + lo
    else:
        std = float(noise_std)

    noise = torch.randn(n_mask, coords.size(1), generator=generator,
                        device=coords.device, dtype=coords.dtype) * std
    noised_coords[chosen] = noised_coords[chosen] + noise

    return new_node_type, noised_coords, loss_mask


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
