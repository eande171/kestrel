"""Chooses Kestrel's move for a position and clock, shared by the UCI engine and the Lichess bot."""

import math
import threading
import time
from dataclasses import dataclass

from .engine_client import Engine
from .inference import greedy_move
from .search import MATE, NetSearch

MAX_DEPTH = 10


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
    pondered: int = 0


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
        self._ponder_thread: threading.Thread | None = None
        self._ponder_stop = threading.Event()
        self._pondered = 0

    # Pondering
    def start_pondering(self, moves: list[str], base_fen: str | None) -> None:
        """Searches the position after the given moves in the background until stopped, filling the cache."""
        self.stop_pondering()
        self._ponder_stop = threading.Event()
        self._ponder_thread = threading.Thread(target=self._ponder, args=(list(moves), base_fen, self._ponder_stop),
                                               name="ponder", daemon=True)
        self._ponder_thread.start()

    def _ponder(self, moves: list[str], base_fen: str | None, stop: threading.Event) -> None:
        try:
            self.engine.position(moves, fen=base_fen)
            state = self.engine.state()
            if state.status == "ongoing" and not stop.is_set():
                self.search.choose(moves, state.fen, state.repetitions, base_fen=base_fen, depth=self.max_depth,
                                   stop=stop)
                self._pondered = self.search.nodes
        except Exception:
            self._pondered = 0

    def stop_pondering(self) -> int:
        """Stops any background search and returns how many positions it expanded."""
        if self._ponder_thread is None:
            return 0
        self._ponder_stop.set()
        self._ponder_thread.join()
        self._ponder_thread = None
        pondered, self._pondered = self._pondered, 0
        return pondered

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

    def choose(self, moves: list[str], base_fen: str | None, clock: Clock) -> Decision:
        """Returns the move to play after the given moves, at a fixed depth or deepening within the clock's budget."""
        pondered = self.stop_pondering()
        self.engine.position(moves, fen=base_fen)
        state = self.engine.state()
        if not state.legal:
            return Decision("0000", 0, 0.0, None)
        start = time.monotonic()
        if len(state.legal) == 1:
            return Decision(state.legal[0], 0, 0.0, None, pondered)
        if clock.depth is not None:
            depth = min(self.max_depth, clock.depth)
            if depth <= 1:
                move = greedy_move(self.evaluator, state.fen, state.repetitions, state.legal)
                return Decision(move, 0, time.monotonic() - start, None, pondered)
            move = self.search.choose(moves, state.fen, state.repetitions, base_fen=base_fen, depth=depth)
        else:
            budget = self.budget(clock, state.fen.split()[1] == "w")
            move = self.search.choose(moves, state.fen, state.repetitions, base_fen=base_fen,
                                      depth=self.max_depth, deadline=start + budget)
        depth = self.search.completed_depth
        return Decision(move, depth, time.monotonic() - start, self.search.last_score if depth else None, pondered)

    def close(self) -> None:
        self.stop_pondering()
        self.engine.close()
