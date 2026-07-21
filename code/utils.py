import torch
from typing import Tuple

# hyperparameters => angles, interface, c4-c4
CA_CA_BOND_DIST = 3.80
C4_C4_BOND_MEAN = 6.1
C4_C4_BOND_STD = 0.5
CA_ANGLE_MIN = 85
CA_ANGLE_MAX = 135
CA_VDW_RADIUS = 3.40      ## temp
C4_VDW_RADIUS = 3.40      ## temp
INTERFACE_CUTOFF = 4.0
CLASH_SEQ_SEP = 3
CHAIN_PROTEIN = 0
CHAIN_RNA = 1

def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute full pairwise Euclidean distance matrix.

    Args:
        x: [N, 3] coordinates

    Returns:
        [N, N] pairwise distances
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)          # [N, N, 3]
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)


def consecutive_distances(x: torch.Tensor, chain_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distance between residues i and i+1, only within the same chain.

    Args:
        x:        [N, 3] coordinates
        chain_id: [N]    integer chain identifier per node

    Returns:
        distances: [M] distances for valid consecutive pairs
        mask:      [N-1] bool, True where the pair is same-chain
    """
    same_chain = chain_id[:-1] == chain_id[1:]
    diff = x[1:] - x[:-1]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    return dist[same_chain], same_chain


def virtual_angles(x: torch.Tensor, chain_id: torch.Tensor) -> torch.Tensor:
    """Angles at residue i formed by (i-1, i, i+1). Only same-chain triples.

    Args:
        x:        [N, 3]
        chain_id: [N]

    Returns:
        angles in degrees, [K] for K valid triples
    """
    if x.size(0) < 3:
        return x.new_zeros(0)

    same_chain_triple = (chain_id[:-2] == chain_id[1:-1]) & (chain_id[1:-1] == chain_id[2:])
    v1 = x[:-2] - x[1:-1]
    v2 = x[2:] - x[1:-1]
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cos_theta) * (180.0 / torch.pi)
    return angles[same_chain_triple]


def build_clash_mask(chain_id: torch.Tensor, seq_index: torch.Tensor,
                     seq_sep: int = CLASH_SEQ_SEP) -> torch.Tensor:
    """Boolean [N, N] mask marking pairs eligible for clash loss.

    Excludes: self-pairs, and same-chain pairs closer than seq_sep in sequence.
    """
    n = chain_id.size(0)
    same_chain = chain_id.unsqueeze(0) == chain_id.unsqueeze(1)
    seq_diff = (seq_index.unsqueeze(0) - seq_index.unsqueeze(1)).abs()
    too_close_in_seq = same_chain & (seq_diff <= seq_sep)
    eye = torch.eye(n, dtype=torch.bool, device=chain_id.device)
    return ~(too_close_in_seq | eye)


def vdw_radii(chain_id: torch.Tensor) -> torch.Tensor:
    """Per-node vdW radius based on chain type."""
    radii = torch.where(
        chain_id == CHAIN_PROTEIN,
        torch.full_like(chain_id, 0, dtype=torch.float) + CA_VDW_RADIUS,
        torch.full_like(chain_id, 0, dtype=torch.float) + C4_VDW_RADIUS,
    )
    return radii

