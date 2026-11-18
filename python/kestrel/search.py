"""Small fixed depth search over the policy's top moves, scored by the value head."""

import random
import time

import numpy as np

from .engine_client import Engine
from .features import fen_to_record, legal_indices_np

MATE = 1.0


class NetSearch:
    """Negamax to a fixed depth over the top policy moves, with value head scores at the leaves."""

    def __init__(self, evaluator, engine: Engine, depth: int = 2, width: int = 6, randomness: float = 0.0,
                 seed: int | None = None):
        self.evaluator = evaluator
        self.engine = engine
        self.depth = depth
        self.width = width
        self.randomness = randomness
        self.rng = random.Random(seed)
        self.base_fen: str | None = None
        self.nodes = 0
        self.seconds = 0.0
        self.last_score = 0.0

    def _evaluate(self, positions: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluator(np.stack([fen_to_record(fen, reps) for fen, reps in positions]))

    def choose(self, moves: list[str], fen: str, repetitions: int, base_fen: str | None = None,
               depth: int | None = None) -> str:
        """Returns the chosen move in UCI form for the position reached by the move list."""
        start = time.time()
        self.base_fen = base_fen
        scored = self._search(moves, fen, repetitions, depth or self.depth, root=True)
        best_score = max(score for score, _ in scored)
        candidates = [move for score, move in scored if score >= best_score - self.randomness]
        if self.randomness <= 0 or best_score >= MATE:
            candidates = [next(move for score, move in scored if score == best_score)]
        choice = self.rng.choice(candidates)
        self.last_score = next(score for score, move in scored if move == choice)
        self.seconds += time.time() - start
        return choice

    def _search(self, moves: list[str], fen: str, repetitions: int, depth: int, root: bool = False):
        self.nodes += 1
        self.engine.position(moves, fen=self.base_fen)
        children = self.engine.children()
        policy, _ = self._evaluate([(fen, repetitions)])
        prior = policy[0, legal_indices_np([c.move for c in children], fen.split()[1] == "b")]
        order = np.argsort(-prior, kind="stable")[: self.width]
        selected = [children[i] for i in order]

        leaf_values = {}
        live = [c for c in selected if c.status == "ongoing"]
        if depth == 1 and live:
            _, values = self._evaluate([(c.fen, c.repetitions) for c in live])
            leaf_values = {c.move: float(v) for c, v in zip(live, values)}

        scored = []
        for child in selected:
            if child.status.startswith("checkmate"):
                score = MATE + 0.01 * depth
            elif child.status != "ongoing":
                score = 0.0
            elif depth == 1:
                score = -leaf_values[child.move]
            else:
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1)[0]
            scored.append((score, child.move))
        if root:
            return scored
        return max(scored, key=lambda item: item[0])
