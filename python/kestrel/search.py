"""Alpha-beta search over the policy's top moves, scored by the value head."""

import math
import random
import threading
import time
from collections.abc import Callable

import numpy as np

from .engine_client import Child, Engine
from .features import fen_to_record, legal_indices_np

MATE = 1.0
CACHE_LIMIT = 200_000
DEFAULT_GROWTH = 3.0

# Endgame Tuning
PIECE_POINTS = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
MOP_UP_WEIGHT = 0.05
MOP_UP_LEAD = 4
MOP_UP_DEFENDER_MAX = 5
AHEAD_VALUE = 0.5
DRAW_PENALTY = 0.1
CONVERSION_SCALE = 0.85
CONVERSION_WEIGHT = 0.01
CONVERSION_CAP = 0.08

# Forcing Move Extensions
EXTRA_FORCING = 3
QUIESCENCE_PLIES = 4
QUIESCENCE_WIDTH = 4
CHECK_EXTENSIONS = 2

# Leaf Batching
FIRST_LEAVES = 2


class SearchTimeout(Exception):
    pass


def is_forcing(child: Child, white_moved: bool) -> bool:
    """Returns True for a capture, a promotion, or a pawn push to the rank before promotion."""
    if "x" in child.san or "=" in child.san:
        return True
    return child.san[0] in "abcdefgh" and child.move[3] == ("7" if white_moved else "2")


def material_and_kings(fen: str) -> tuple[dict[str, int], dict[str, tuple[int, int]], str]:
    """Returns each side's material in pawns, each king's file and rank, and the side to move."""
    board, side = fen.split()[:2]
    material = {"w": 0, "b": 0}
    kings = {}
    rank = 7
    file = 0
    for ch in board:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        else:
            colour = "w" if ch.isupper() else "b"
            if ch.lower() == "k":
                kings[colour] = (file, rank)
            else:
                material[colour] += PIECE_POINTS[ch.lower()]
            file += 1
    return material, kings, side


def endgame_adjusted(fen: str, value: float) -> float:
    """Returns a value head score plus the mop-up bonus and, once the position is decided, a small material tie-break."""
    material, kings, side = material_and_kings(fen)
    other = "b" if side == "w" else "w"
    score = value + mop_up(material, kings, side)
    if abs(value) > AHEAD_VALUE:
        lead = material[side] - material[other]
        score = CONVERSION_SCALE * score + max(-CONVERSION_CAP, min(CONVERSION_CAP, CONVERSION_WEIGHT * lead))
    return min(max(score, -0.999), 0.999)


def mop_up(material: dict[str, int], kings: dict[str, tuple[int, int]], side: str) -> float:
    """Returns a bonus for the side to move when it is well ahead in a sparse endgame, for cornering the losing king."""
    strong = "w" if material["w"] > material["b"] else "b"
    weak = "b" if strong == "w" else "w"
    if material[strong] - material[weak] < MOP_UP_LEAD or material[weak] > MOP_UP_DEFENDER_MAX:
        return 0.0
    (wf, wr), (sf, sr) = kings[weak], kings[strong]
    edge = max(3 - wf, wf - 4) + max(3 - wr, wr - 4)
    apart = abs(wf - sf) + abs(wr - sr)
    bonus = MOP_UP_WEIGHT * (0.6 * edge / 6 + 0.4 * (14 - apart) / 14)
    return bonus if side == strong else -bonus


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
        self.halt: Callable[[], bool] | None = None
        self.last_score = 0.0
        self.completed_depth = 0
        self.root_depth = 0
        self.draw_score = 0.0
        self.endgame_tuning = True
        self.extensions = True
        self.check_extensions = CHECK_EXTENSIONS
        self.nodes = 0
        self.priors: dict[tuple[str, int], np.ndarray] = {}
        self.values: dict[tuple[str, int], float] = {}
        self.best_child: dict[tuple[str, int], str] = {}

    def choose(self, moves: list[str], fen: str, repetitions: int, base_fen: str | None = None,
               depth: int | None = None, deadline: float | None = None, stop: threading.Event | None = None) -> str:
        """Returns the chosen move in UCI form, searching to the given depth or deepening until a deadline or stop."""
        self.base_fen = base_fen
        self.nodes = 0
        if len(self.values) > CACHE_LIMIT:
            self.priors.clear()
            self.values.clear()
            self.best_child.clear()
        depth = depth or self.depth
        if deadline is None and stop is None:
            self.halt = None
            self.completed_depth = depth
            return self._pick(self._root(moves, fen, repetitions, depth))
        return self._deepen(moves, fen, repetitions, depth, math.inf if deadline is None else deadline, stop)

    def _deepen(self, moves: list[str], fen: str, repetitions: int, max_depth: int, deadline: float,
                stop: threading.Event | None) -> str:
        self.halt = None
        policy_best = self._expand(moves, fen, repetitions)[0].move
        self.halt = lambda: time.monotonic() > deadline or (stop is not None and stop.is_set())
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
        self.halt = None
        if scored is None:
            self.last_score = 0.0
            return policy_best
        return self._pick(scored)

    def _pick(self, scored: list[tuple[float, str]]) -> str:
        top = max(scored, key=lambda item: item[0])
        pool = [top]
        if self.randomness > 0 and top[0] < AHEAD_VALUE:
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
        self.root_depth = depth
        ahead = self.endgame_tuning and self.values[cache_key(fen, repetitions)] > AHEAD_VALUE
        self.draw_score = -DRAW_PENALTY if ahead else 0.0

        # Root Window
        scored, best = [], -math.inf
        for child in selected:
            score = self._terminal(child, 1)
            alpha = best - self.randomness - 1e-9
            extend = self._extends(child, 0)
            if score is None and depth == 1 and not extend:
                score = self._leaf_score(moves, fen, child, leaf_values, alpha, math.inf, 1)
            elif score is None:
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1 + extend,
                                      -math.inf, -alpha, 1, extend)
            scored.append((score, child.move))
            best = max(best, score)
        return scored

    def _search(self, moves: list[str], fen: str, repetitions: int, depth: int, alpha: float, beta: float,
                ply: int = 0, extended: int = 0) -> float:
        selected = self._expand(moves, fen, repetitions)
        leaf_values: dict[str, float] = {}
        best, best_move = -math.inf, None
        for index, child in enumerate(selected):
            score = self._terminal(child, ply + 1)
            extend = self._extends(child, extended)
            if score is None and depth == 1 and not extend:
                if child.move not in leaf_values:
                    leaf_values.update(self._leaf_values(selected[index:] if index else selected[:FIRST_LEAVES]))
                score = self._leaf_score(moves, fen, child, leaf_values, max(alpha, best), beta, ply + 1)
            elif score is None:
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1 + extend,
                                      -beta, -max(alpha, best), ply + 1, extended + extend)
            if score > best:
                best, best_move = score, child.move
            if best >= beta:
                break
        if best_move is not None:
            self.best_child[cache_key(fen, repetitions)] = best_move
        return best

    def _extends(self, child: Child, extended: int) -> int:
        """Returns 1 when a checking move should be searched a ply deeper, within the line's extension allowance."""
        return int("+" in child.san and extended < self.check_extensions)

    def _terminal(self, child: Child, ply: int) -> float | None:
        """Returns the score of a finished child, ply moves from the root, for the side that moved into it."""
        if child.status.startswith("checkmate"):
            return MATE + 0.01 * max(self.root_depth - ply + 1, 0)
        if child.status != "ongoing":
            return self.draw_score if ply % 2 == 1 else -self.draw_score
        return None

    def _ordered_children(self, moves: list[str], fen: str, repetitions: int) -> list[Child]:
        """Returns every child of a position, highest policy first."""
        if self.halt is not None and self.halt():
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
        return [children[i] for i in np.argsort(-prior, kind="stable")]

    def _expand(self, moves: list[str], fen: str, repetitions: int) -> list[Child]:
        """Returns the top policy children plus a few forcing ones, with the best move from an earlier search first."""
        ordered = self._ordered_children(moves, fen, repetitions)
        selected = ordered[: self.width]
        if self.extensions:
            white = fen.split()[1] == "w"
            selected += [c for c in ordered[self.width:] if is_forcing(c, white)][:EXTRA_FORCING]
        hint = self.best_child.get(cache_key(fen, repetitions))
        if hint is not None:
            selected.sort(key=lambda c: c.move != hint)
        return selected

    def _leaf_score(self, moves: list[str], fen: str, child: Child, leaf_values: dict[str, float],
                    alpha: float, beta: float, ply: int) -> float:
        """Returns a depth 1 child's score for the side that moved, following forcing replies after a forcing move."""
        if self.extensions and is_forcing(child, fen.split()[1] == "w"):
            return -self._quiesce(moves + [child.move], child, -beta, -alpha, 0, ply)
        return -leaf_values[child.move]

    def _quiesce(self, moves: list[str], node: Child, alpha: float, beta: float, depth: int, ply: int) -> float:
        """Returns the score for the side to move, standing pat on the value head or playing forcing moves until quiet."""
        stand = self._adjusted_value(node)
        if stand >= beta or depth <= -QUIESCENCE_PLIES:
            return stand
        white = node.fen.split()[1] == "w"
        forcing = [c for c in self._ordered_children(moves, node.fen, node.repetitions) if is_forcing(c, white)]
        if forcing:
            self._leaf_values(forcing[:QUIESCENCE_WIDTH])
        best = stand
        for child in forcing[:QUIESCENCE_WIDTH]:
            score = self._terminal(child, ply + 1)
            if score is None:
                score = -self._quiesce(moves + [child.move], child, -beta, -max(alpha, best), depth - 1, ply + 1)
            best = max(best, score)
            if best >= beta:
                break
        return best

    def _adjusted_value(self, child: Child) -> float:
        value = self.values[cache_key(child.fen, child.repetitions)]
        return endgame_adjusted(child.fen, value) if self.endgame_tuning else value

    def _leaf_values(self, children: list[Child]) -> dict[str, float]:
        """Returns value head scores for the ongoing children, evaluating the uncached ones in one batch."""
        live = [c for c in children if c.status == "ongoing"]
        missing = [c for c in live if cache_key(c.fen, c.repetitions) not in self.values]
        if missing:
            _, values = self.evaluator(np.stack([fen_to_record(c.fen, c.repetitions) for c in missing]))
            for c, v in zip(missing, values):
                self.values[cache_key(c.fen, c.repetitions)] = float(v)
        return {c.move: self._adjusted_value(c) for c in live}
