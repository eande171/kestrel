"""Checks records, planes and move indices against the Rust engine's legal move list."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.encoding import (  # noqa: E402
    PIECE_CHARS,
    POLICY_SIZE,
    NUM_PLANES,
    encode_planes,
    encode_targets,
    fen_to_record,
    legal_indices,
    read_records,
)
from kestrel.engine_client import Engine  # noqa: E402


def record_to_fen(rec: np.ndarray) -> str:
    """Returns a FEN for a raw record, with clocks set to zero and one."""
    packed = rec[:32]
    codes = np.empty(64, dtype=np.uint8)
    codes[0::2] = packed & 15
    codes[1::2] = packed >> 4
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            c = codes[rank * 8 + file]
            if c == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += PIECE_CHARS[c - 1]
        rows.append(row + (str(empty) if empty else ""))
    stm = "w" if rec[32] == 0 else "b"
    castling = "".join(ch for ch, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if rec[33] & bit) or "-"
    ep = "-" if rec[34] == 255 else "abcdefgh"[rec[34]] + ("6" if stm == "w" else "3")
    return f"{'/'.join(rows)} {stm} {castling} {ep} {rec[45]} 1"


def main(path: str, samples: int = 3000) -> None:
    data = read_records(Path(path))
    rng = np.random.default_rng(0)
    picks = data[rng.choice(len(data), size=min(samples, len(data)), replace=False)]
    raw = torch.from_numpy(picks)
    planes = encode_planes(raw)
    idx, wdl = encode_targets(raw)
    assert planes.shape == (len(picks), NUM_PLANES, 8, 8)
    assert int(idx.max()) < POLICY_SIZE and int(idx.min()) >= 0
    assert set(wdl.tolist()) <= {0, 1, 2}

    # Own King Plane
    assert torch.all(planes[:, 5].sum(dim=(1, 2)) == 1)
    assert torch.all(planes[:, 11].sum(dim=(1, 2)) == 1)

    checked = 0
    with Engine() as engine:
        for rec, target in zip(picks, idx.tolist()):
            fen = record_to_fen(rec)
            again = fen_to_record(fen, int(rec[44]))
            assert np.array_equal(again[:35], rec[:35]), fen
            assert np.array_equal(again[44:46], rec[44:46]), fen
            engine.position([], fen=fen)
            state = engine.state()
            legal = legal_indices(state.legal, rec[32] == 1).tolist()
            assert len(set(legal)) == len(legal), f"index collision in {fen}"
            assert target in legal, f"played move not legal in {fen}"
            checked += 1
    repeated = float((data[:, 44] >= 1).mean())
    print(f"ok: {checked} positions, planes, round trip and move indices consistent")
    print(f"positions repeated earlier in their game: {repeated:.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
