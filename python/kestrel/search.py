"""Alpha-beta search over the policy's top moves, scored by the value head."""

import math
import random
import time

import numpy as np

from .engine_client import Child, Engine
from .features import fen_to_record, legal_indices_np

MATE = 1.0
CACHE_LIMIT = 200_000
DEFAULT_GROWTH = 3.0


class SearchTimeout(Exception):
    pass


def cache_key(fen: str, repetitions: int) -> tuple[str, int]:
    """Returns the network's inputs for a position: the FEN without its move number, and the capped repetition count."""
    return fen.rsplit(" ", 1)[0], min(repetitions, 2)


class NetSearch:
    """Negamax with alpha-beta pruning over the top policy moves, giving the same scores as a full search of that tree."""

    def __init__(self, evaluator, engine: Engine, depth: int = 2, width: int = 6, randomness: float = 0.0):
        self.evaluator = evaluator
        self.engine = engine
        self.depth = depth
        self.width = width
        self.randomness = randomness
        self.rng = random.Random()
        self.base_fen: str | None = None
        self.deadline: float | None = None
        self.last_score = 0.0
        self.completed_depth = 0
        self.nodes = 0
        self.priors: dict[tuple[str, int], np.ndarray] = {}
        self.values: dict[tuple[str, int], float] = {}
        self.best_child: dict[tuple[str, int], str] = {}

    def choose(self, moves: list[str], fen: str, repetitions: int, base_fen: str | None = None,
               depth: int | None = None, deadline: float | None = None) -> str:
        """Returns the chosen move in UCI form, searching to the given depth or deepening until a time.monotonic deadline."""
        self.base_fen = base_fen
        self.nodes = 0
        if len(self.values) > CACHE_LIMIT:
            self.priors.clear()
            self.values.clear()
            self.best_child.clear()
        depth = depth or self.depth
        if deadline is None:
            self.deadline = None
            self.completed_depth = depth
            return self._pick(self._root(moves, fen, repetitions, depth))
        return self._deepen(moves, fen, repetitions, depth, deadline)

    def _deepen(self, moves: list[str], fen: str, repetitions: int, max_depth: int, deadline: float) -> str:
        self.deadline = None
        policy_best = self._expand(moves, fen, repetitions)[0].move
        self.deadline = deadline
        scored, order, previous = None, None, None
        self.completed_depth = 0
        for depth in range(2, max_depth + 1):
            start = time.monotonic()
            try:
                scored = self._root(moves, fen, repetitions, depth, order)
            except SearchTimeout:
                break
            self.completed_depth = depth
            took = time.monotonic() - start
            growth = min(max(took / previous, 2.0), 8.0) if previous and previous > 0.01 else DEFAULT_GROWTH
            previous = took
            if max(score for score, _ in scored) >= MATE or time.monotonic() + took * growth > deadline:
                break
            order = [move for _, move in sorted(scored, key=lambda item: -item[0])]
        self.deadline = None
        if scored is None:
            self.last_score = 0.0
            return policy_best
        return self._pick(scored)

    def _pick(self, scored: list[tuple[float, str]]) -> str:
        top = max(scored, key=lambda item: item[0])
        pool = [top]
        if self.randomness > 0 and top[0] < MATE:
            pool = [item for item in scored if item[0] >= top[0] - self.randomness]
        self.last_score, choice = self.rng.choice(pool)
        return choice

    def _root(self, moves: list[str], fen: str, repetitions: int, depth: int,
              order: list[str] | None = None) -> list[tuple[float, str]]:
        """Returns exact scores for root moves that can be chosen, and upper bounds for the rest."""
        selected = self._expand(moves, fen, repetitions)
        if order:
            selected.sort(key=lambda c: order.index(c.move) if c.move in order else len(order))
        leaf_values = self._leaf_values(selected) if depth == 1 else {}

        # Root Window
        scored, best = [], -math.inf
        for child in selected:
            score = self._terminal(child, depth)
            if score is None and depth == 1:
                score = -leaf_values[child.move]
            elif score is None:
                alpha = best - self.randomness - 1e-9
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1, -math.inf, -alpha)
            scored.append((score, child.move))
            best = max(best, score)
        return scored

    def _search(self, moves: list[str], fen: str, repetitions: int, depth: int, alpha: float, beta: float) -> float:
        selected = self._expand(moves, fen, repetitions)
        leaf_values = self._leaf_values(selected) if depth == 1 else {}
        best, best_move = -math.inf, None
        for child in selected:
            score = self._terminal(child, depth)
            if score is None and depth == 1:
                score = -leaf_values[child.move]
            elif score is None:
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1,
                                      -beta, -max(alpha, best))
            if score > best:
                best, best_move = score, child.move
            if best >= beta:
                break
        if best_move is not None:
            self.best_child[cache_key(fen, repetitions)] = best_move
        return best

    @staticmethod
    def _terminal(child: Child, depth: int) -> float | None:
        if child.status.startswith("checkmate"):
            return MATE + 0.01 * depth
        if child.status != "ongoing":
            return 0.0
        return None

    def _expand(self, moves: list[str], fen: str, repetitions: int) -> list[Child]:
        """Returns the top policy children of a position, with the best move from an earlier search first."""
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise SearchTimeout
        self.nodes += 1
        self.engine.position(moves, fen=self.base_fen)
        children = self.engine.children()
        key = cache_key(fen, repetitions)
        prior = self.priors.get(key)
        if prior is None:
            policy, value = self.evaluator(fen_to_record(fen, repetitions)[None, :])
            prior = policy[0, legal_indices_np([c.move for c in children], fen.split()[1] == "b")]
            self.values[key] = float(value[0])
            self.priors[key] = prior
        selected = [children[i] for i in np.argsort(-prior, kind="stable")[: self.width]]
        hint = self.best_child.get(key)
        if hint is not None:
            selected.sort(key=lambda c: c.move != hint)
        return selected

    def _leaf_values(self, children: list[Child]) -> dict[str, float]:
        """Returns value head scores for the ongoing children, evaluating the uncached ones in one batch."""
        live = [c for c in children if c.status == "ongoing"]
        missing = [c for c in live if cache_key(c.fen, c.repetitions) not in self.values]
        if missing:
            _, values = self.evaluator(np.stack([fen_to_record(c.fen, c.repetitions) for c in missing]))
            for c, v in zip(missing, values):
                self.values[cache_key(c.fen, c.repetitions)] = float(v)
        return {c.move: self.values[cache_key(c.fen, c.repetitions)] for c in live}
