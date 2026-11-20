"""Plays Kestrel from lost positions against a faster, weaker version of itself and counts the games it saves."""

import argparse
import time
from pathlib import Path

import numpy as np

from .engine_client import Engine, result_for
from .features import FLAG_ENGINE_EVAL, read_records, record_to_fen
from .inference import OnnxEvaluator
from .search import NetSearch

DEFAULT_SHARD = Path("D:/chess-data/evals/val_000.bin")


def lost_positions(shard: Path, count: int, low: float, high: float, min_pieces: int, seed: int) -> list[str]:
    """Returns FENs where Stockfish gives the side to move an expected score between low and high."""
    rows = read_records(shard)
    rows = rows[(rows[:, 46] & FLAG_ENGINE_EVAL) > 0]
    q = rows[:, 47] / 255
    board = rows[:, :32]
    pieces = ((board & 15) != 0).sum(1) + ((board >> 4) != 0).sum(1)
    pool = rows[(q >= low) & (q <= high) & (pieces >= min_pieces)]
    picks = np.random.default_rng(seed).choice(len(pool), size=min(count, len(pool)), replace=False)
    return [record_to_fen(pool[i]) for i in picks]


def defend(defender: NetSearch, attacker: NetSearch, engine: Engine, fen: str, seconds: float,
           attacker_seconds: float, max_plies: int) -> tuple[float, int]:
    """Returns the defender's points from a position it starts, and how many plies the game lasted."""
    moves: list[str] = []
    for ply in range(max_plies):
        engine.position(moves, fen=fen)
        state = engine.state()
        if state.status != "ongoing":
            outcome = result_for(state.status)
            if outcome == "1/2-1/2":
                return 0.5, ply
            defender_white = fen.split()[1] == "w"
            return (1.0 if (outcome == "1-0") == defender_white else 0.0), ply
        search, limit = (defender, seconds) if ply % 2 == 0 else (attacker, attacker_seconds)
        moves.append(search.choose(moves, state.fen, state.repetitions, base_fen=fen, depth=10,
                                   deadline=time.monotonic() + limit))
    return 0.5, max_plies


def parse_option(text: str) -> tuple[str, float]:
    name, value = text.split("=", 1)
    return name, float(value)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("runs/evals6/model.onnx"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--shard", type=Path, default=DEFAULT_SHARD, help="held out records with Stockfish scores")
    p.add_argument("--positions", type=int, default=40)
    p.add_argument("--low", type=float, default=0.05, help="lowest Stockfish expected score for the defender")
    p.add_argument("--high", type=float, default=0.2, help="highest Stockfish expected score for the defender")
    p.add_argument("--min-pieces", type=int, default=12, help="kings included, so the attacker has work to do")
    p.add_argument("--seconds", type=float, default=1.0, help="defender's time per move")
    p.add_argument("--attacker-seconds", type=float, default=0.1, help="attacker's time per move")
    p.add_argument("--max-plies", type=int, default=200, help="plies before the game counts as saved")
    p.add_argument("--option", type=parse_option, action="append", default=[],
                   help="search setting for the defender only, such as check_extensions=0; may repeat")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    evaluator = OnnxEvaluator(args.model, args.threads)
    fens = lost_positions(args.shard, args.positions, args.low, args.high, args.min_pieces, args.seed)
    points, lengths = [], []
    with Engine() as engine:
        for i, fen in enumerate(fens, 1):
            defender = NetSearch(evaluator, engine, 10, 6)
            attacker = NetSearch(evaluator, engine, 10, 6)
            for name, value in args.option:
                setattr(defender, name, type(getattr(defender, name))(value))
            score, plies = defend(defender, attacker, engine, fen, args.seconds, args.attacker_seconds, args.max_plies)
            points.append(score)
            lengths.append(plies)
            print(f"{i}/{len(fens)} {fen}: {score} after {plies} plies", flush=True)
    settings = ", ".join(f"{n}={v:g}" for n, v in args.option) or "default search"
    print(f"{settings}: saved {sum(points):g} points from {len(fens)} lost positions "
          f"({sum(p > 0 for p in points)} games not lost), average {np.mean(lengths):.0f} plies")


if __name__ == "__main__":
    main()
