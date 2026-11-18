"""UCI engine for Kestrel: the network plus a short search, run on the CPU with ONNX Runtime."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

from .engine_client import Engine
from .inference import OnnxEvaluator, greedy_move
from .search import MATE, NetSearch

DEFAULT_MODEL = Path(os.environ.get("KESTREL_MODEL", Path(__file__).resolve().parents[2] / "runs" / "rep" / "model.onnx"))
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Seconds Per Move Guesses
INITIAL_COST = {0: 0.02, 2: 0.15, 3: 0.6, 4: 3.0}
DEPTHS_DESCENDING = [4, 3, 2, 0]


def score_to_cp(score: float) -> str:
    """Returns a UCI score string from a search score in (-1, 1), or a mate score."""
    if score >= MATE:
        return "mate 1"
    if score <= -MATE:
        return "mate -1"
    p = min(max((score + 1) / 2, 1e-4), 1 - 1e-4)
    return f"cp {round(400 * math.log10(p / (1 - p)))}"


class KestrelUci:
    def __init__(self, model: Path, threads: int, depth: int, width: int, randomness: float, out=sys.stdout):
        self.model_path = model
        self.threads = threads
        self.max_depth = depth
        self.width = width
        self.randomness = randomness
        self.out = out
        self.evaluator = None
        self.engine = None
        self.search = None
        self.base_fen = None
        self.moves: list[str] = []
        self.cost = dict(INITIAL_COST)

    def send(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()

    def ensure_ready(self) -> None:
        if self.evaluator is None:
            self.evaluator = OnnxEvaluator(self.model_path, self.threads)
            self.engine = Engine()
            self.search = NetSearch(self.evaluator, self.engine, self.max_depth, self.width, self.randomness)

    def set_option(self, tokens: list[str]) -> None:
        if "name" not in tokens or "value" not in tokens:
            return
        name = " ".join(tokens[tokens.index("name") + 1 : tokens.index("value")]).lower()
        value = " ".join(tokens[tokens.index("value") + 1 :])
        if name == "depth":
            self.max_depth = max(0, min(4, int(value)))
        elif name == "width":
            self.width = max(1, min(30, int(value)))
        elif name == "randomness":
            self.randomness = max(0, min(100, int(value))) / 1000
        if self.search:
            self.search.width = self.width
            self.search.randomness = self.randomness

    def set_position(self, tokens: list[str]) -> None:
        moves_at = tokens.index("moves") if "moves" in tokens else len(tokens)
        if len(tokens) > 1 and tokens[1] == "fen":
            fen = " ".join(tokens[2:moves_at])
            self.base_fen = None if fen == START_FEN else fen
        else:
            self.base_fen = None
        self.moves = tokens[moves_at + 1 :]

    def budget(self, tokens: list[str], white_to_move: bool) -> float:
        """Returns the seconds this move may use, from the go command's clock fields."""
        values = {}
        for key in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth"):
            if key in tokens:
                values[key] = int(tokens[tokens.index(key) + 1])
        if "movetime" in values:
            return values["movetime"] / 1000 * 0.9
        remaining = values.get("wtime" if white_to_move else "btime")
        if remaining is None:
            return math.inf
        increment = values.get("winc" if white_to_move else "binc", 0) / 1000
        moves_left = values.get("movestogo", 30)
        remaining /= 1000
        return min(remaining / moves_left + 0.75 * increment, remaining * 0.2)

    def go(self, tokens: list[str]) -> None:
        self.ensure_ready()
        self.engine.position(self.moves, fen=self.base_fen)
        state = self.engine.state()
        if not state.legal:
            self.send("bestmove 0000")
            return
        if "depth" in tokens:
            depth = min(self.max_depth, int(tokens[tokens.index("depth") + 1]))
        else:
            budget = self.budget(tokens, state.fen.split()[1] == "w")
            depth = next((d for d in DEPTHS_DESCENDING if d <= self.max_depth and self.cost[d] <= budget), 0)
        if depth == 1:
            depth = 0

        start = time.time()
        if len(state.legal) == 1:
            move, score = state.legal[0], None
        elif depth == 0:
            move, score = greedy_move(self.evaluator, state.fen, state.repetitions, state.legal), None
        else:
            move = self.search.choose(self.moves, state.fen, state.repetitions, base_fen=self.base_fen, depth=depth)
            score = self.search.last_score
        elapsed = time.time() - start
        if len(state.legal) > 1:
            self.cost[depth] = 0.7 * self.cost[depth] + 0.3 * elapsed
        info = f"info depth {depth} time {round(elapsed * 1000)}"
        if score is not None:
            info += f" score {score_to_cp(score)}"
        self.send(f"{info} pv {move}")
        self.send(f"bestmove {move}")

    def run(self, lines) -> None:
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            command = tokens[0]
            if command == "uci":
                self.send("id name Kestrel")
                self.send("id author Kestrel project")
                self.send(f"option name Depth type spin default {self.max_depth} min 0 max 4")
                self.send(f"option name Width type spin default {self.width} min 1 max 30")
                self.send(f"option name Randomness type spin default {round(self.randomness * 1000)} min 0 max 100")
                self.send("uciok")
            elif command == "isready":
                self.ensure_ready()
                self.send("readyok")
            elif command == "setoption":
                self.set_option(tokens)
            elif command == "ucinewgame":
                self.base_fen, self.moves = None, []
            elif command == "position":
                self.set_position(tokens)
            elif command == "go":
                self.go(tokens)
            elif command == "quit":
                break
        if self.engine:
            self.engine.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--threads", type=int, default=int(os.environ.get("KESTREL_THREADS", 2)))
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--randomness", type=int, default=10, help="thousandths of a point; root moves this close to the best are picked at random")
    args = p.parse_args(argv)
    KestrelUci(args.model, args.threads, args.depth, args.width, args.randomness / 1000).run(sys.stdin)


if __name__ == "__main__":
    main()
