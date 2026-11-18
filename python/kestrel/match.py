"""Plays the network against the baseline engine or another opponent."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .engine_client import Engine
from .inference import greedy_move
from .search import NetSearch

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
    seconds_per_move: float = 0.0
    move_lists: list[list[str]] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


def play_match(
    evaluator,
    depth: int,
    games: int,
    max_plies: int = 300,
    seed: int = 0,
    search_depth: int = 0,
    width: int = 6,
    opponent: Callable[[list[str]], str] | None = None,
    on_game: Callable[[int, MatchResult], None] | None = None,
) -> MatchResult:
    """Plays games against the baseline at a fixed depth, or another opponent, alternating colours over an opening book."""
    result = MatchResult(depth=depth)
    rng = np.random.default_rng(seed)
    with Engine(depth=depth, seed=int(rng.integers(1, 2**31))) as engine:
        searcher = NetSearch(evaluator, engine, search_depth, width) if search_depth > 0 else None
        net_moves, net_seconds = 0, 0.0
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
                    start = time.time()
                    if searcher:
                        mv = searcher.choose(moves, state.fen, state.repetitions)
                    else:
                        mv = greedy_move(evaluator, state.fen, state.repetitions, state.legal)
                    net_seconds += time.time() - start
                    net_moves += 1
                elif opponent:
                    mv = opponent(moves)
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
            result.move_lists.append(moves)
            if on_game:
                on_game(g, result)
        result.seconds_per_move = net_seconds / max(1, net_moves)
    return result
