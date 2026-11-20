"""Position and move encoding in NumPy, shared by training, the search and the ONNX engine."""

import json
from pathlib import Path

import numpy as np

RECORD_BYTES = 48
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
        ("repetitions", "u1"),
        ("halfmove", "u1"),
        ("flags", "u1"),
        ("engine_score", "u1"),
    ]
)
assert RECORD_DTYPE.itemsize == RECORD_BYTES

FLAG_ENGINE_EVAL = 1
NUM_PLANES = 21
LEGACY_PLANES = 19
POLICY_SIZE = 4096 + 72
PIECE_CHARS = "PNBRQKpnbrqk"
PROMO_CODES = {"n": 1, "b": 2, "r": 3, "q": 4}

FLIP = np.arange(64) ^ 56
OPPONENT_FIRST = [2, 3, 0, 1]


def record_width(folder: Path) -> int:
    """Returns the record size of the shards in a folder, 44 bytes for the first format."""
    meta = Path(folder) / "meta.json"
    return json.loads(meta.read_text()).get("record_bytes", 44) if meta.exists() else RECORD_BYTES


def read_records(path: Path) -> np.ndarray:
    """Returns records from one shard as uint8 rows of RECORD_BYTES, padding older 44 byte shards."""
    width = record_width(Path(path).parent)
    rows = np.fromfile(path, dtype=np.uint8).reshape(-1, width)
    if width < RECORD_BYTES:
        rows = np.pad(rows, ((0, 0), (0, RECORD_BYTES - width)))
    return rows


def fen_to_record(fen: str, repetitions: int = 0) -> np.ndarray:
    """Returns a record for a FEN and repetition count with an empty move and result."""
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
    rec[44] = min(repetitions, 2)
    rec[45] = min(int(parts[4]), 255) if len(parts) > 4 else 0
    return rec


def record_to_fen(rec: np.ndarray) -> str:
    """Returns the FEN of a record, with the move number set to 1."""
    codes = np.empty(64, dtype=np.uint8)
    codes[0::2] = rec[:32] & 15
    codes[1::2] = rec[:32] >> 4
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            code = codes[rank * 8 + file]
            if code == 0:
                empty += 1
            else:
                row += (str(empty) if empty else "") + PIECE_CHARS[code - 1]
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    castling = "".join(ch for ch, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if rec[33] & bit) or "-"
    side = "w" if rec[32] == 0 else "b"
    ep = "-" if rec[34] == 255 else f"{chr(ord('a') + rec[34])}{'6' if side == 'w' else '3'}"
    return f"{'/'.join(rows)} {side} {castling} {ep} {rec[45]} 1"


def uci_to_squares(move: str) -> tuple[int, int, int]:
    """Returns absolute from square, to square and promotion code for a UCI move string."""
    f = (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)
    t = (ord(move[2]) - ord("a")) + 8 * (int(move[3]) - 1)
    promo = PROMO_CODES.get(move[4], 0) if len(move) > 4 else 0
    return f, t, promo


def encode_planes_np(raw: np.ndarray, num_planes: int = NUM_PLANES) -> np.ndarray:
    """Returns float32 planes of shape (B, num_planes, 8, 8) from uint8 records of shape (B, 48)."""
    raw = np.asarray(raw, dtype=np.uint8)
    b = raw.shape[0]
    packed = raw[:, :32].astype(np.int64)
    squares = np.stack([packed & 15, packed >> 4], axis=-1).reshape(b, 64)
    black = raw[:, 32] == 1

    # Mover Perspective
    squares = np.where(black[:, None], squares[:, FLIP], squares)
    swapped = np.where(squares == 0, 0, np.where(squares <= 6, squares + 6, squares - 6))
    squares = np.where(black[:, None], swapped, squares)
    pieces = (squares[:, None, :] == np.arange(1, 13)[None, :, None]).astype(np.float32)

    castling = raw[:, 33].astype(np.int64)
    bits = np.stack([(castling >> i) & 1 for i in range(4)], axis=1)
    castle = np.where(black[:, None], bits[:, OPPONENT_FIRST], bits).astype(np.float32)
    castle = np.repeat(castle[:, :, None], 64, axis=2)

    ep = raw[:, 34].astype(np.int64)
    ep_plane = np.zeros((b, 64), dtype=np.float32)
    has_ep = ep != 255
    ep_plane[np.nonzero(has_ep)[0], 40 + ep[has_ep]] = 1.0

    def constant(values: np.ndarray) -> np.ndarray:
        return np.repeat(values.astype(np.float32)[:, None, None], 64, axis=2)

    planes = np.concatenate(
        [
            pieces,
            castle,
            ep_plane[:, None, :],
            constant(black),
            constant(np.ones(b)),
            constant(raw[:, 44] >= 1),
            constant(np.minimum(raw[:, 45] / 100.0, 1.0)),
        ],
        axis=1,
    )
    return np.ascontiguousarray(planes[:, :num_planes].reshape(b, num_planes, 8, 8))


def policy_index_np(from_sq, to_sq, promo, black) -> np.ndarray:
    """Returns policy indices for moves given in absolute squares, flipped to the mover's view."""
    flip = np.asarray(black, dtype=np.int64) * 56
    f = np.asarray(from_sq, dtype=np.int64) ^ flip
    t = np.asarray(to_sq, dtype=np.int64) ^ flip
    promo = np.asarray(promo, dtype=np.int64)
    under = 4096 + np.maximum(promo - 1, 0) * 24 + (f % 8) * 3 + (t % 8 - f % 8 + 1)
    return np.where((promo >= 1) & (promo <= 3), under, f * 64 + t)


def legal_indices_np(moves: list[str], black: bool) -> np.ndarray:
    """Returns policy indices for a list of legal UCI moves in one position."""
    sq = np.array([uci_to_squares(m) for m in moves], dtype=np.int64).reshape(-1, 3)
    return policy_index_np(sq[:, 0], sq[:, 1], sq[:, 2], np.full(len(moves), bool(black)))
