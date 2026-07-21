import torch
import torch.nn.functional as F
from typing import Dict, Optional

from utils import (
    CA_CA_BOND_DIST, C4_C4_BOND_MEAN, C4_C4_BOND_STD,
    CA_ANGLE_MIN, CA_ANGLE_MAX,
    INTERFACE_CUTOFF,
    CHAIN_PROTEIN, CHAIN_RNA,
    pairwise_distances,
    consecutive_distances,
    virtual_angles,
    build_clash_mask,
    vdw_radii,
)


def compute_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    chain_id: torch.Tensor,
    seq_index: torch.Tensor,
    loss_mask: torch.Tensor,
    interface_pairs: Optional[torch.Tensor] = None,
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the composite RNP inpainting loss.

    L = L_coord
        + λ1 * L_dRMSD
        + λ2 * L_bond_prot + λ3 * L_angle_prot + λ4 * L_clash_prot
        + λ5 * L_bond_RNA  + λ6 * L_clash_RNA
        + λ7 * L_contact

    Args:
        x_pred:          [N, 3] predicted coordinates (from EGNN).
        x_true:          [N, 3] ground-truth coordinates.
        chain_id:        [N]    0 = protein (Cα), 1 = RNA (C4').
        seq_index:       [N]    integer sequence position within chain.
        loss_mask:       [N]    bool, True for known-masked training targets.
        interface_pairs: [K, 2] optional protein↔RNA contact pairs (indices)
                                known to be in contact in the ground truth.
                                If Nsone, L_contact = 0.
        lambdas:         dict of weights. Missing keys use defaults below.

    Returns:
        dict of scalar tensors: total loss and each component (for logging).
    """
    default_lambdas = {
        "coord":       0.7,
        "dRMSD":       0.7,
        "bond_prot":   0.3,
        "angle_prot":  0.2,
        "clash_prot":  0.3,
        "bond_rna":    0.3,
        "clash_rna":   0.3,
        "contact":     0.2,
    }
    
    if lambdas is not None:
        default_lambdas.update(lambdas)
    w = default_lambdas

    device = x_pred.device
    zero = torch.tensor(0.0, device=device)

    # ---------- Primary: coordinate MSE on masked nodes ----------
    if loss_mask.any():
        L_coord = F.mse_loss(x_pred[loss_mask], x_true[loss_mask])
    else:
        L_coord = zero

    # ---------- Secondary: dRMSD on pairs involving masked nodes ----------
    L_dRMSD = _dRMSD_targets(x_pred, x_true, loss_mask)

    # ---------- Protein geometry (Cα) ----------
    is_prot = chain_id == CHAIN_PROTEIN
    L_bond_prot  = _bond_loss(x_pred, chain_id, is_prot, CA_CA_BOND_DIST)
    L_angle_prot = _angle_hinge_loss(x_pred, chain_id, is_prot,
                                     CA_ANGLE_MIN, CA_ANGLE_MAX)
    L_clash_prot = _clash_loss(x_pred, chain_id, seq_index, restrict_to=is_prot)

    # ---------- RNA geometry (C4') ----------
    is_rna = chain_id == CHAIN_RNA
    L_bond_rna  = _bond_loss(x_pred, chain_id, is_rna,
                             C4_C4_BOND_MEAN, C4_C4_BOND_STD)
    L_clash_rna = _clash_loss(x_pred, chain_id, seq_index, restrict_to=is_rna)

    # ---------- Interface: known contact pairs must stay within cutoff ----------
    L_contact = _contact_loss(x_pred, interface_pairs, INTERFACE_CUTOFF)

    # ---------- Weighted total ----------
    total = (
        w["coord"]      * L_coord
        + w["dRMSD"]      * L_dRMSD
        + w["bond_prot"]  * L_bond_prot
        + w["angle_prot"] * L_angle_prot
        + w["clash_prot"] * L_clash_prot
        + w["bond_rna"]   * L_bond_rna
        + w["clash_rna"]  * L_clash_rna
        + w["contact"]    * L_contact
    )

    return {
        "total":       total,
        "coord":       L_coord.detach(),
        "dRMSD":       L_dRMSD.detach(),
        "bond_prot":   L_bond_prot.detach(),
        "angle_prot":  L_angle_prot.detach(),
        "clash_prot":  L_clash_prot.detach(),
        "bond_rna":    L_bond_rna.detach(),
        "clash_rna":   L_clash_rna.detach(),
        "contact":     L_contact.detach(),
    }


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def _dRMSD_targets(x_pred: torch.Tensor, x_true: torch.Tensor,
                   loss_mask: torch.Tensor) -> torch.Tensor:
    """dRMSD over pairs (i, j) where at least one is a masked target."""
    if not loss_mask.any():
        return x_pred.new_tensor(0.0)

    d_pred = pairwise_distances(x_pred)
    d_true = pairwise_distances(x_true)

    n = x_pred.size(0)
    mask_row = loss_mask.unsqueeze(0).expand(n, n)
    mask_col = loss_mask.unsqueeze(1).expand(n, n)
    involves_target = mask_row | mask_col
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                  device=x_pred.device), diagonal=1)
    pair_mask = involves_target & upper

    if not pair_mask.any():
        return x_pred.new_tensor(0.0)

    diff = (d_pred - d_true)[pair_mask]
    return (diff ** 2).mean()


def _bond_loss(x: torch.Tensor, chain_id: torch.Tensor,
               chain_mask: torch.Tensor, mu: float,
               sigma: Optional[float] = None) -> torch.Tensor:
    """Penalises deviation of consecutive same-chain distances from mu.

    With sigma given, the penalty is the Gaussian NLL 0.5 * ((d - mu) / sigma)^2,
    so deviations are measured in standard deviations rather than angstroms.
    """
    if chain_mask.sum() < 2:
        return x.new_tensor(0.0)

    _, valid = consecutive_distances(x, chain_id)
    # Keep only consecutive pairs that are same-chain AND both endpoints belong
    # to the requested chain type (protein vs RNA).
    both_in = chain_mask[:-1] & chain_mask[1:]
    pair_mask = valid & both_in
    if not pair_mask.any():
        return x.new_tensor(0.0)
    diff = x[1:][pair_mask] - x[:-1][pair_mask]
    d = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    if sigma is None:
        return ((d - mu) ** 2).mean()
    return 0.5 * (((d - mu) / sigma) ** 2).mean()


def _angle_hinge_loss(x: torch.Tensor, chain_id: torch.Tensor,
                      chain_mask: torch.Tensor,
                      angle_min: float, angle_max: float) -> torch.Tensor:
    """Hinge penalty pushing virtual angles into [angle_min, angle_max]."""
    if chain_mask.sum() < 3:
        return x.new_tensor(0.0)

    angles = virtual_angles(x, chain_id)
    if angles.numel() == 0:
        return x.new_tensor(0.0)

    # Restrict to triples fully inside chain_mask.
    triple_in = chain_mask[:-2] & chain_mask[1:-1] & chain_mask[2:]
    same_chain_triple = (chain_id[:-2] == chain_id[1:-1]) & (chain_id[1:-1] == chain_id[2:])
    keep = triple_in[same_chain_triple]
    angles = angles[keep]
    if angles.numel() == 0:
        return x.new_tensor(0.0)

    high = F.relu(angles - angle_max)
    low  = F.relu(angle_min - angles)
    return (high ** 2 + low ** 2).mean()


def _clash_loss(x: torch.Tensor, chain_id: torch.Tensor,
                seq_index: torch.Tensor,
                restrict_to: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hinge on pairwise distances vs vdW-sum, over non-bonded pairs.

    Args:
        restrict_to: [N] bool, if provided only pairs where BOTH nodes satisfy
                     this mask are considered (used to split protein vs RNA).
    """
    n = x.size(0)
    if n < 2:
        return x.new_tensor(0.0)

    d = pairwise_distances(x)
    radii = vdw_radii(chain_id)
    r_sum = radii.unsqueeze(0) + radii.unsqueeze(1)     # [N, N]

    pair_mask = build_clash_mask(chain_id, seq_index)
    if restrict_to is not None:
        both = restrict_to.unsqueeze(0) & restrict_to.unsqueeze(1)
        pair_mask = pair_mask & both

    if not pair_mask.any():
        return x.new_tensor(0.0)

    overlap = F.relu(r_sum - d)
    overlap = overlap * pair_mask.float()
    # Mean over eligible pairs only.
    return (overlap ** 2).sum() / pair_mask.sum().clamp(min=1)


def _contact_loss(x: torch.Tensor,
                  pairs: Optional[torch.Tensor],
                  cutoff: float) -> torch.Tensor:
    """For each known interface pair (i, j), penalise distance beyond cutoff."""
    if pairs is None or pairs.numel() == 0:
        return x.new_tensor(0.0)

    xi = x[pairs[:, 0]]
    xj = x[pairs[:, 1]]
    d = torch.sqrt(((xi - xj) ** 2).sum(dim=-1) + 1e-8)
    return (F.relu(d - cutoff) ** 2).mean()