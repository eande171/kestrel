"""Position and move encoding shared by training, evaluation and export."""

import numpy as np
import torch

RECORD_BYTES = 44
RECORD_DTYPE = np.dtype(
    [
        ("board", "u1", (32,)),
        ("stm", "u1"),
        ("castling", "u1"),
        ("ep", "u1"),
        ("result", "u1"),
        ("from_sq", "u1"),
        ("to_sq", "u1"),
        ("promo", "u1"),
        ("ply", "u1"),
        ("mover_elo", "<u2"),
        ("opp_elo", "<u2"),
    ]
)
assert RECORD_DTYPE.itemsize == RECORD_BYTES

NUM_PLANES = 19
POLICY_SIZE = 4096 + 72
PIECE_CHARS = "PNBRQKpnbrqk"
PROMO_CODES = {"n": 1, "b": 2, "r": 3, "q": 4}

_FLIP = torch.arange(64) ^ 56


def encode_planes(raw: torch.Tensor) -> torch.Tensor:
    """Returns float planes of shape (B, 19, 8, 8) from raw uint8 records of shape (B, 44)."""
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

    planes = torch.cat(
        [pieces, castle_planes, ep_plane[:, None], stm_plane[:, None], ones[:, None]], dim=1
    )
    return planes.reshape(b, NUM_PLANES, 8, 8)


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


def fen_to_record(fen: str) -> np.ndarray:
    """Returns a 44 byte record for a FEN with an empty move and result."""
    parts = fen.split()
    codes = np.zeros(64, dtype=np.uint8)
    for rank_idx, row in enumerate(parts[0].split("/")):
        rank = 7 - rank_idx
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                codes[rank * 8 + file] = PIECE_CHARS.index(ch) + 1
                file += 1
    rec = np.zeros(RECORD_BYTES, dtype=np.uint8)
    rec[:32] = codes[0::2] | (codes[1::2] << 4)
    rec[32] = 0 if parts[1] == "w" else 1
    castling = parts[2] if len(parts) > 2 else "-"
    rec[33] = sum(bit for ch, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if ch in castling)
    ep = parts[3] if len(parts) > 3 else "-"
    rec[34] = 255 if ep == "-" else ord(ep[0]) - ord("a")
    return rec


def uci_to_squares(move: str) -> tuple[int, int, int]:
    """Returns absolute from square, to square and promotion code for a UCI move string."""
    f = (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)
    t = (ord(move[2]) - ord("a")) + 8 * (int(move[3]) - 1)
    promo = PROMO_CODES.get(move[4], 0) if len(move) > 4 else 0
    return f, t, promo


def legal_indices(moves: list[str], black: bool) -> torch.Tensor:
    """Returns policy indices for a list of legal UCI moves in one position."""
    sq = torch.tensor([uci_to_squares(m) for m in moves], dtype=torch.long)
    flag = torch.full((len(moves),), bool(black))
    return policy_index(sq[:, 0], sq[:, 1], sq[:, 2], flag)
