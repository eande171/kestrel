"""Remembers how Kestrel's early moves fared against each opponent, to replay wins and avoid losses."""

import json
import os
from pathlib import Path

MEMORY_PLIES = 30


class OpeningMemory:
    """Per opponent counts of wins, draws and losses for each Kestrel move after each opening move sequence."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.data: dict[str, dict[str, dict[str, list[int]]]] = {}
        if path and path.exists():
            self.data = json.loads(path.read_text())

    def record(self, opponent: str, moves: list[str], kestrel_white: bool, points: float) -> None:
        """Adds a finished game's early Kestrel moves with its result, 1 for a win down to 0 for a loss."""
        book = self.data.setdefault(opponent.lower(), {})
        outcome = 0 if points == 1 else 1 if points == 0.5 else 2
        for ply in range(min(len(moves), MEMORY_PLIES)):
            if (ply % 2 == 0) == kestrel_white:
                stats = book.setdefault(" ".join(moves[:ply]), {}).setdefault(moves[ply], [0, 0, 0])
                stats[outcome] += 1
        self.save()

    def advice(self, opponent: str, moves: list[str]) -> tuple[str | None, set[str]]:
        """Returns a move that has only won here against this opponent, and the moves that have only lost."""
        if len(moves) >= MEMORY_PLIES:
            return None, set()
        options = self.data.get(opponent.lower(), {}).get(" ".join(moves), {})
        winners = [(wins, move) for move, (wins, _, losses) in options.items() if wins and not losses]
        losers = {move for move, (wins, _, losses) in options.items() if losses and not wins}
        return (max(winners)[1] if winners else None), losers

    def save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data))
        os.replace(temporary, self.path)
