"""Plays the network's greedy legal move against the baseline engine."""

from dataclasses import dataclass, field

import numpy as np
import torch

from .encoding import encode_planes, fen_to_record, legal_indices
from .engine_client import Engine

# Opening Book
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
]


@dataclass
class MatchResult:
    depth: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    illegal: int = 0
    plies: list[int] = field(default_factory=list)
    endings: dict[str, int] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


@torch.no_grad()
def network_move(model: torch.nn.Module, state, device: str) -> str:
    """Returns the highest scoring legal move in UCI form."""
    rec = torch.from_numpy(fen_to_record(state.fen)[None, :]).to(device)
    logits, _ = model(encode_planes(rec))
    black = state.fen.split()[1] == "b"
    idx = legal_indices(state.legal, black).to(device)
    best = int(torch.argmax(logits[0, idx].float()))
    return state.legal[best]


def play_match(model: torch.nn.Module, depth: int, games: int, device: str, max_plies: int = 300, seed: int = 0) -> MatchResult:
    """Plays games against the baseline at a fixed depth, alternating colours across a small opening book."""
    was_training = model.training
    model.eval()
    result = MatchResult(depth=depth)
    rng = np.random.default_rng(seed)
    with Engine(depth=depth, seed=int(rng.integers(1, 2**31))) as engine:
        for g in range(games):
            engine.set_option("Seed", int(rng.integers(1, 2**31)))
            moves = OPENINGS[(g // 2) % len(OPENINGS)].split()
            net_is_white = g % 2 == 0
            while True:
                engine.position(moves)
                state = engine.state()
                if state.status != "ongoing" or len(moves) >= max_plies:
                    break
                white_to_move = state.fen.split()[1] == "w"
                if white_to_move == net_is_white:
                    mv = network_move(model, state, device)
                else:
                    mv = engine.best_move(depth)
                if mv not in state.legal:
                    result.illegal += 1
                    break
                moves.append(mv)
            status = state.status if state.status != "ongoing" else "move_limit"
            result.endings[status] = result.endings.get(status, 0) + 1
            result.plies.append(len(moves))
            if status.startswith("checkmate"):
                net_won = (status == "checkmate_white") == net_is_white
                if net_won:
                    result.wins += 1
                else:
                    result.losses += 1
            else:
                result.draws += 1
    model.train(was_training)
    return result
