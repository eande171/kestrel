"""Small fixed depth search over the policy's top moves, scored by the value head."""

import time

import numpy as np
import torch

from .encoding import encode_planes, fen_to_record, legal_indices
from .engine_client import Engine

MATE = 1.0


class NetSearch:
    """Negamax to a fixed depth over the top policy moves, with value head scores at the leaves."""

    def __init__(self, model: torch.nn.Module, engine: Engine, device: str, depth: int = 2, width: int = 6):
        self.model = model
        self.engine = engine
        self.device = device
        self.depth = depth
        self.width = width
        self.nodes = 0
        self.seconds = 0.0

    @torch.no_grad()
    def _evaluate(self, positions: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        records = np.stack([fen_to_record(fen, reps) for fen, reps in positions])
        raw = torch.from_numpy(records).to(self.device)
        policy, wdl = self.model(encode_planes(raw, self.model.in_planes))
        probs = wdl.float().softmax(-1)
        return policy.float(), probs[:, 2] - probs[:, 0]

    def choose(self, moves: list[str], fen: str, repetitions: int) -> str:
        """Returns the best move in UCI form for the position reached by the move list."""
        start = time.time()
        _, best = self._search(moves, fen, repetitions, self.depth)
        self.seconds += time.time() - start
        return best

    def _search(self, moves: list[str], fen: str, repetitions: int, depth: int) -> tuple[float, str]:
        self.nodes += 1
        self.engine.position(moves)
        children = self.engine.children()
        policy, _ = self._evaluate([(fen, repetitions)])
        black = fen.split()[1] == "b"
        prior = policy[0, legal_indices([c.move for c in children], black).to(self.device)]
        order = torch.argsort(prior, descending=True)[: self.width].tolist()
        selected = [children[i] for i in order]

        leaf_values = {}
        live = [c for c in selected if c.status == "ongoing"]
        if depth == 1 and live:
            _, values = self._evaluate([(c.fen, c.repetitions) for c in live])
            leaf_values = {c.move: float(v) for c, v in zip(live, values)}

        best_score, best_move = -float("inf"), selected[0].move
        for child in selected:
            if child.status.startswith("checkmate"):
                score = MATE + 0.01 * depth
            elif child.status != "ongoing":
                score = 0.0
            elif depth == 1:
                score = -leaf_values[child.move]
            else:
                score = -self._search(moves + [child.move], child.fen, child.repetitions, depth - 1)[0]
            if score > best_score:
                best_score, best_move = score, child.move
        return best_score, best_move
