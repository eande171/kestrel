"""Chooses Kestrel's move for a position and clock, shared by the UCI engine and the Lichess bot."""

import math
import time
from dataclasses import dataclass

from .engine_client import Engine
from .inference import greedy_move
from .search import MATE, NetSearch

# Seconds Per Move Guesses
INITIAL_COST = {0: 0.02, 2: 0.15}
DEPTH_GROWTH = 6.0
MAX_DEPTH = 6


@dataclass
class Clock:
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int = 0
    binc_ms: int = 0
    movestogo: int | None = None
    movetime_ms: int | None = None
    depth: int | None = None


@dataclass
class Decision:
    move: str
    depth: int
    seconds: float
    score: float | None


def score_to_cp(score: float) -> str:
    """Returns a UCI score string from a search score in (-1, 1), or a mate score."""
    if score >= MATE:
        return "mate 1"
    if score <= -MATE:
        return "mate -1"
    p = min(max((score + 1) / 2, 1e-4), 1 - 1e-4)
    return f"cp {round(400 * math.log10(p / (1 - p)))}"


class Player:
    """Owns one rules engine process and picks moves with the shared evaluator."""

    def __init__(self, evaluator, max_depth: int = 4, width: int = 6, randomness: float = 0.01,
                 reserve_seconds: float = 0.0):
        self.evaluator = evaluator
        self.max_depth = min(max_depth, MAX_DEPTH)
        self.reserve_seconds = reserve_seconds
        self.engine = Engine()
        self.search = NetSearch(evaluator, self.engine, max_depth, width, randomness)
        self.cost = dict(INITIAL_COST)

    def budget(self, clock: Clock, white_to_move: bool) -> float:
        """Returns the seconds this move may use."""
        if clock.movetime_ms is not None:
            return clock.movetime_ms / 1000 * 0.9
        remaining = clock.wtime_ms if white_to_move else clock.btime_ms
        if remaining is None:
            return math.inf
        remaining = max(0.0, remaining / 1000 - self.reserve_seconds)
        increment = (clock.winc_ms if white_to_move else clock.binc_ms) / 1000
        moves_left = clock.movestogo or 30
        return min(remaining / moves_left + 0.75 * increment, remaining * 0.2)

    def estimate(self, depth: int) -> float:
        """Returns the expected seconds for a search of this depth, scaling up from the depth below if unmeasured."""
        if depth in self.cost:
            return self.cost[depth]
        return self.estimate(depth - 1) * DEPTH_GROWTH

    def choose(self, moves: list[str], base_fen: str | None, clock: Clock) -> Decision:
        """Returns the move to play after the given moves, within the clock's budget."""
        self.engine.position(moves, fen=base_fen)
        state = self.engine.state()
        if not state.legal:
            return Decision("0000", 0, 0.0, None)
        if clock.depth is not None:
            depth = min(self.max_depth, clock.depth)
        else:
            budget = self.budget(clock, state.fen.split()[1] == "w")
            depth = next((d for d in range(self.max_depth, 1, -1) if self.estimate(d) <= budget), 0)
        if depth == 1:
            depth = 0

        start = time.time()
        if len(state.legal) == 1:
            move, score = state.legal[0], None
        elif depth == 0:
            move, score = greedy_move(self.evaluator, state.fen, state.repetitions, state.legal), None
        else:
            move = self.search.choose(moves, state.fen, state.repetitions, base_fen=base_fen, depth=depth)
            score = self.search.last_score
        elapsed = time.time() - start
        if len(state.legal) > 1:
            self.cost[depth] = 0.7 * self.cost[depth] + 0.3 * elapsed if depth in self.cost else elapsed
        return Decision(move, depth, elapsed, score)

    def close(self) -> None:
        self.engine.close()
