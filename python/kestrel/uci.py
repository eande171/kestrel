"""UCI engine for Kestrel: the network plus a short search, run on the CPU with ONNX Runtime."""

import argparse
import os
import sys
from pathlib import Path

from .inference import OnnxEvaluator
from .player import MAX_DEPTH, Clock, Player, score_to_cp

DEFAULT_MODEL = Path(os.environ.get("KESTREL_MODEL", Path(__file__).resolve().parents[2] / "runs" / "rep" / "model.onnx"))
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class KestrelUci:
    def __init__(self, model: Path, threads: int, depth: int, width: int, randomness: float, out=sys.stdout):
        self.model_path = model
        self.threads = threads
        self.max_depth = depth
        self.width = width
        self.randomness = randomness
        self.out = out
        self.player = None
        self.base_fen = None
        self.moves: list[str] = []

    def send(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()

    def ensure_ready(self) -> None:
        if self.player is None:
            evaluator = OnnxEvaluator(self.model_path, self.threads)
            self.player = Player(evaluator, self.max_depth, self.width, self.randomness)

    def set_option(self, tokens: list[str]) -> None:
        if "name" not in tokens or "value" not in tokens:
            return
        name = " ".join(tokens[tokens.index("name") + 1 : tokens.index("value")]).lower()
        value = " ".join(tokens[tokens.index("value") + 1 :])
        if name == "depth":
            self.max_depth = max(0, min(MAX_DEPTH, int(value)))
        elif name == "width":
            self.width = max(1, min(30, int(value)))
        elif name == "randomness":
            self.randomness = max(0, min(100, int(value))) / 1000
        if self.player:
            self.player.max_depth = self.max_depth
            self.player.search.width = self.width
            self.player.search.randomness = self.randomness

    def set_position(self, tokens: list[str]) -> None:
        moves_at = tokens.index("moves") if "moves" in tokens else len(tokens)
        if len(tokens) > 1 and tokens[1] == "fen":
            fen = " ".join(tokens[2:moves_at])
            self.base_fen = None if fen == START_FEN else fen
        else:
            self.base_fen = None
        self.moves = tokens[moves_at + 1 :]

    def go(self, tokens: list[str]) -> None:
        self.ensure_ready()
        values = {}
        for key in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth"):
            if key in tokens:
                values[key] = int(tokens[tokens.index(key) + 1])
        clock = Clock(values.get("wtime"), values.get("btime"), values.get("winc", 0), values.get("binc", 0),
                      values.get("movestogo"), values.get("movetime"), values.get("depth"))
        decision = self.player.choose(self.moves, self.base_fen, clock)
        info = f"info depth {decision.depth} time {round(decision.seconds * 1000)}"
        if decision.score is not None:
            info += f" score {score_to_cp(decision.score)}"
        self.send(f"{info} pv {decision.move}")
        self.send(f"bestmove {decision.move}")

    def run(self, lines) -> None:
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            command = tokens[0]
            if command == "uci":
                self.send("id name Kestrel")
                self.send("id author Kestrel project")
                self.send(f"option name Depth type spin default {self.max_depth} min 0 max {MAX_DEPTH}")
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
        if self.player:
            self.player.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--threads", type=int, default=int(os.environ.get("KESTREL_THREADS", 2)))
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--randomness", type=int, default=10, help="thousandths of a point; root moves this close to the best are picked at random")
    args = p.parse_args(argv)
    KestrelUci(args.model, args.threads, args.depth, args.width, args.randomness / 1000).run(sys.stdin)


if __name__ == "__main__":
    main()
