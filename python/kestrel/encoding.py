"""PyTorch position and move encoding for training, matching features.py exactly."""

import torch

from .features import (  # noqa: F401
    LEGACY_PLANES,
    NUM_PLANES,
    PIECE_CHARS,
    POLICY_SIZE,
    PROMO_CODES,
    RECORD_BYTES,
    RECORD_DTYPE,
    fen_to_record,
    read_records,
    uci_to_squares,
)

_FLIP = torch.arange(64) ^ 56


def encode_planes(raw: torch.Tensor, num_planes: int = NUM_PLANES) -> torch.Tensor:
    """Returns float planes of shape (B, num_planes, 8, 8) from raw uint8 records of shape (B, 48)."""
    device = raw.device
    b = raw.shape[0]
    packed = raw[:, :32].long()
    squares = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(b, 64)
    black = raw[:, 32].bool()

    # Mover Perspective
    flipped = squares[:, _FLIP.to(device)]
    squares = torch.where(black[:, None], flipped, squares)
    swapped = torch.where(squares == 0, squares, torch.where(squares <= 6, squares + 6, squares - 6))
    squares = torch.where(black[:, None], swapped, squares)

    pieces = torch.nn.functional.one_hot(squares, 13)[..., 1:].permute(0, 2, 1).float()

    castling = raw[:, 33].long()
    bits = torch.stack([(castling >> i) & 1 for i in range(4)], dim=1)
    own_first = torch.stack([bits[:, 0], bits[:, 1], bits[:, 2], bits[:, 3]], dim=1)
    opp_first = torch.stack([bits[:, 2], bits[:, 3], bits[:, 0], bits[:, 1]], dim=1)
    castle_planes = torch.where(black[:, None], opp_first, own_first).float()[:, :, None].expand(b, 4, 64)

    ep = raw[:, 34].long()
    ep_plane = torch.zeros(b, 64, device=device)
    has_ep = ep != 255
    ep_plane[has_ep, 40 + ep[has_ep]] = 1.0

    stm_plane = black.float()[:, None].expand(b, 64)
    ones = torch.ones(b, 64, device=device)

    # History Planes
    repeated = (raw[:, 44] >= 1).float()[:, None].expand(b, 64)
    halfmove = (raw[:, 45].float() / 100.0).clamp(max=1.0)[:, None].expand(b, 64)

    planes = torch.cat(
        [pieces, castle_planes, ep_plane[:, None], stm_plane[:, None], ones[:, None],
         repeated[:, None], halfmove[:, None]],
        dim=1,
    )
    return planes[:, :num_planes].reshape(b, num_planes, 8, 8)


def policy_index(from_sq: torch.Tensor, to_sq: torch.Tensor, promo: torch.Tensor, black: torch.Tensor) -> torch.Tensor:
    """Returns the policy index for moves given in absolute squares, flipped to the mover's view."""
    from_sq = from_sq.long()
    to_sq = to_sq.long()
    promo = promo.long()
    flip = black.long() * 56
    f = from_sq ^ flip
    t = to_sq ^ flip
    base = f * 64 + t
    under = 4096 + (promo - 1).clamp(min=0) * 24 + (f % 8) * 3 + (t % 8 - f % 8 + 1)
    is_under = (promo >= 1) & (promo <= 3)
    return torch.where(is_under, under, base)


def encode_targets(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns policy indices and WDL class targets (0 loss, 1 draw, 2 win) for raw records."""
    idx = policy_index(raw[:, 36], raw[:, 37], raw[:, 38], raw[:, 32].bool())
    return idx, raw[:, 35].long()


def legal_indices(moves: list[str], black: bool) -> torch.Tensor:
    """Returns policy indices for a list of legal UCI moves in one position."""
    sq = torch.tensor([uci_to_squares(m) for m in moves], dtype=torch.long)
    flag = torch.full((len(moves),), bool(black))
    return policy_index(sq[:, 0], sq[:, 1], sq[:, 2], flag)
