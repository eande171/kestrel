"""Plays the network against Stockfish at fixed UCI_Elo levels and estimates a rating."""

import argparse
import json
import math
import time
from pathlib import Path

from .engine_client import Engine, UciProcess, result_for
from .inference import OnnxEvaluator, TorchEvaluator
from .match import MatchResult, pgn, play_match

DEFAULT_STOCKFISH = Path(__file__).resolve().parents[2] / "tools" / "stockfish-19" / "stockfish" / "stockfish-windows-x86-64-universal.exe"


class Stockfish(UciProcess):
    def __init__(self, path: Path, elo: int, movetime_ms: int):
        super().__init__([str(path)])
        self.movetime_ms = movetime_ms
        self._send("uci")
        self._read_until("uciok")
        for name, value in (("Threads", 1), ("Hash", 16), ("UCI_LimitStrength", "true"), ("UCI_Elo", elo)):
            self.set_option(name, value)
        self.new_game()

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def move(self, moves: list[str]) -> str:
        self.position(moves)
        return self.go(f"movetime {self.movetime_ms}")


def expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400))


def fit_rating(levels: list[tuple[int, float, int]]) -> tuple[int, int, int]:
    """Returns the maximum likelihood rating and a 95% range from (opponent rating, points, games) rows."""
    def log_likelihood(r: float) -> float:
        total = 0.0
        for opp, points, games in levels:
            e = min(max(expected(r, opp), 1e-9), 1 - 1e-9)
            total += points * math.log(e) + (games - points) * math.log(1 - e)
        return total

    grid = range(600, 3601)
    scores = {r: log_likelihood(r) for r in grid}
    best = max(scores, key=scores.get)
    inside = [r for r in grid if scores[r] >= scores[best] - 1.92]
    return best, min(inside), max(inside)


def sans_for(engine: Engine, moves: list[str]) -> list[str]:
    """Returns the SAN form of each move in a game played from the start."""
    sans = []
    for i, mv in enumerate(moves):
        engine.position(moves[:i])
        parsed = engine.parse(mv)
        sans.append(parsed[1] if parsed else mv)
    return sans


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/rep")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--levels", type=int, nargs="+", default=[1600, 1900, 2200])
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--search-depth", type=int, default=3)
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--movetime", type=int, default=200, help="Stockfish milliseconds per move")
    p.add_argument("--stockfish", type=Path, default=DEFAULT_STOCKFISH)
    p.add_argument("--label", default=None)
    p.add_argument("--onnx", type=Path, default=None, help="use this ONNX model on the CPU instead of the PyTorch checkpoint")
    p.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads")
    args = p.parse_args(argv)

    run = Path(args.run)
    if args.onnx:
        evaluator, step = OnnxEvaluator(args.onnx, args.threads), None
    else:
        from .model import load_model

        model, ckpt = load_model(run / "checkpoints" / args.checkpoint, "cuda")
        evaluator, step = TorchEvaluator(model, "cuda"), ckpt.get("step")
    mode = f"search depth {args.search_depth}" if args.search_depth else "top move only"
    mode += f", {evaluator.description}"
    label = args.label or (f"s{args.search_depth}" if args.search_depth else "greedy")
    out_dir = run / "stockfish" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kestrel ({mode}) vs Stockfish 19, {args.games} games per level, Stockfish {args.movetime} ms per move")
    print("| Stockfish UCI_Elo | games | wins | draws | losses | score | avg plies | net s/move |")
    print("|---|---|---|---|---|---|---|---|")
    rows, fit_rows = [], []
    with Engine() as referee:
        for elo in args.levels:
            start = time.time()
            with Stockfish(args.stockfish, elo, args.movetime) as sf:
                r: MatchResult = play_match(
                    evaluator, 0, args.games, seed=elo, search_depth=args.search_depth, width=args.width,
                    opponent=sf.move, on_game=lambda g, res: sf.new_game(),
                )
            avg = sum(r.plies) / max(1, len(r.plies))
            print(f"| {elo} | {r.games} | {r.wins} | {r.draws} | {r.losses} | {r.score:.2f} | {avg:.0f} | "
                  f"{r.seconds_per_move:.2f} |", flush=True)
            rows.append({"elo": elo, "games": r.games, "wins": r.wins, "draws": r.draws, "losses": r.losses,
                         "score": r.score, "endings": r.endings, "illegal": r.illegal,
                         "seconds": round(time.time() - start, 1)})
            fit_rows.append((elo, r.wins + 0.5 * r.draws, r.games))
            pgns = []
            for g, moves in enumerate(r.move_lists):
                net_white = g % 2 == 0
                white, black = ("Kestrel", f"Stockfish {elo}") if net_white else (f"Stockfish {elo}", "Kestrel")
                pgns.append(pgn(sans_for(referee, moves), white, black, result_for(r.statuses[g])))
            (out_dir / f"games_{elo}.pgn").write_text("\n".join(pgns))

    best, low, high = fit_rating(fit_rows)
    print(f"\nEstimated rating on Stockfish's UCI_Elo scale: {best} (95% range {low} to {high})")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"mode": mode, "movetime_ms": args.movetime, "levels": rows,
                   "estimate": best, "range": [low, high], "step": step}, f, indent=2)


if __name__ == "__main__":
    main()
