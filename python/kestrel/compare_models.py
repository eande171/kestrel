"""Compares two exported models: a match at equal time, won endgame conversion, real game mistakes, and agreement with Stockfish."""

import argparse
import random
import time
from pathlib import Path

import numpy as np

from .engine_client import Engine, EngineError, State
from .features import FLAG_ENGINE_EVAL, policy_index_np, read_records
from .inference import OnnxEvaluator
from .match import play_match
from .mistakes import DEFAULT_SUITE, grade, load
from .search import NetSearch

# Endgame Suite
ENDGAMES = {
    "K+Q vs K": "Q",
    "K+R vs K": "R",
    "K+Q vs K+R": "Qr",
    "K+R+P vs K+R": "RPr",
}


def head_to_head(a: OnnxEvaluator, b: OnnxEvaluator, games: int, seconds: float) -> str:
    with Engine() as engine_a, Engine() as engine_b:
        search_a = NetSearch(a, engine_a, 8, 6, randomness=0.01)
        search_b = NetSearch(b, engine_b, 8, 6, randomness=0.01)
        depths = {"a": [], "b": []}

        def move_a(moves: list[str], state: State) -> str:
            move = search_a.choose(moves, state.fen, state.repetitions, depth=8, deadline=time.monotonic() + seconds)
            depths["a"].append(search_a.completed_depth)
            return move

        def move_b(moves: list[str]) -> str:
            engine_b.position(moves)
            state = engine_b.state()
            move = search_b.choose(moves, state.fen, state.repetitions, depth=8, deadline=time.monotonic() + seconds)
            depths["b"].append(search_b.completed_depth)
            return move

        r = play_match(None, 1, games, seed=21, mover=move_a, opponent=move_b)
    return (f"A {r.wins}-{r.draws}-{r.losses} B (A scores {r.score:.2f}) at {seconds}s per move, "
            f"average depth {np.mean(depths['a']):.1f} vs {np.mean(depths['b']):.1f}, endings {r.endings}")


def random_endgame(engine: Engine, pieces: str, rng: random.Random) -> str:
    """Returns a legal ongoing position with white kings and pieces from upper case letters, black from lower case."""
    while True:
        squares = rng.sample(range(64), 2 + len(pieces))
        if max(abs(squares[0] % 8 - squares[1] % 8), abs(squares[0] // 8 - squares[1] // 8)) <= 1:
            continue
        board = ["."] * 64
        board[squares[0]], board[squares[1]] = "K", "k"
        for square, piece in zip(squares[2:], pieces):
            if piece in "Pp" and square // 8 in (0, 7):
                break
            board[square] = piece
        else:
            rows = []
            for rank in range(7, -1, -1):
                row, empty = "", 0
                for file in range(8):
                    ch = board[rank * 8 + file]
                    if ch == ".":
                        empty += 1
                    else:
                        row += (str(empty) if empty else "") + ch
                        empty = 0
                rows.append(row + (str(empty) if empty else ""))
            fen = f"{'/'.join(rows)} w - - 0 1"
            try:
                engine.position([], fen=fen)
                state = engine.state()
            except EngineError:
                continue
            if state.status == "ongoing" and len(state.legal) > 1:
                return fen


def convert(evaluator: OnnxEvaluator, defender: OnnxEvaluator, engine: Engine, fen: str, seconds: float,
            max_plies: int = 100) -> int | None:
    """Returns the plies the attacker needs to mate from the position, or None if it does not within max_plies."""
    attacker = NetSearch(evaluator, engine, 8, 6)
    defence = NetSearch(defender, engine, 8, 6)
    moves: list[str] = []
    for ply in range(max_plies):
        engine.position(moves, fen=fen)
        state = engine.state()
        if state.status.startswith("checkmate"):
            return ply
        if state.status != "ongoing":
            return None
        search = attacker if ply % 2 == 0 else defence
        moves.append(search.choose(moves, state.fen, state.repetitions, base_fen=fen, depth=8,
                                   deadline=time.monotonic() + seconds))
    return None


def endgame_suite(a: OnnxEvaluator, b: OnnxEvaluator, defender: OnnxEvaluator, count: int, seconds: float) -> list[str]:
    lines = []
    with Engine() as engine:
        for kind, (name, pieces) in enumerate(ENDGAMES.items()):
            fens = [random_endgame(engine, pieces, random.Random(1000 * kind + i)) for i in range(count)]
            row = []
            for label, model in (("A", a), ("B", b)):
                mates = [r for r in (convert(model, defender, engine, fen, seconds) for fen in fens) if r is not None]
                average = f"{np.mean(mates) / 2:.1f} moves" if mates else "none"
                row.append(f"{label} mates {len(mates)}/{count} (average {average})")
            lines.append(f"{name}: " + ", ".join(row))
    return lines


def stockfish_agreement(evaluator: OnnxEvaluator, shard: Path, limit: int) -> str:
    """Returns how often the policy's top move is Stockfish's and how well the value tracks Stockfish's score."""
    rows = read_records(shard)
    rows = rows[(rows[:, 46] & FLAG_ENGINE_EVAL) > 0][:limit]
    policy, value = evaluator(rows)
    best = policy_index_np(rows[:, 36], rows[:, 37], rows[:, 38], rows[:, 32] == 1)
    target = 2 * rows[:, 47].astype(np.float32) / 255 - 1
    board = rows[:, :32]
    pieces = ((board & 15) != 0).sum(1) + ((board >> 4) != 0).sum(1)
    parts = []
    for name, mask in (("all", np.ones(len(rows), bool)), ("12 pieces or fewer", pieces <= 12)):
        top1 = float((policy[mask].argmax(1) == best[mask]).mean())
        corr = float(np.corrcoef(value[mask], target[mask])[0, 1])
        mae = float(np.abs(value[mask] - target[mask]).mean())
        parts.append(f"{name} ({mask.sum()}): top move = Stockfish {top1:.1%}, value correlation {corr:.3f}, "
                     f"mean error {mae:.3f}")
    return "; ".join(parts)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="candidate ONNX model")
    p.add_argument("--b", type=Path, required=True, help="reference ONNX model")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--seconds", type=float, default=0.5, help="time per move in the match and the endgames")
    p.add_argument("--endgames", type=int, default=8, help="positions per endgame type, 0 to skip")
    p.add_argument("--eval-shard", type=Path, default=None, help="held out records with Stockfish scores")
    p.add_argument("--eval-positions", type=int, default=20000)
    p.add_argument("--mistakes", type=Path, default=DEFAULT_SUITE, help="positions from real games, see mistakes.py")
    p.add_argument("--mistake-seconds", type=float, default=3.0)
    args = p.parse_args(argv)

    a = OnnxEvaluator(args.a, args.threads)
    b = OnnxEvaluator(args.b, args.threads)
    print(f"A = {args.a}\nB = {args.b}", flush=True)
    if args.eval_shard:
        for label, model in (("A", a), ("B", b)):
            print(f"Stockfish agreement {label}: {stockfish_agreement(model, args.eval_shard, args.eval_positions)}",
                  flush=True)
    if args.mistakes.is_file():
        suite = load(args.mistakes)
        for label, model in (("A", a), ("B", b)):
            print(f"game mistakes {label}: {grade(model, suite, args.mistake_seconds)}", flush=True)
    if args.endgames:
        for line in endgame_suite(a, b, b, args.endgames, args.seconds):
            print(f"endgames, defended by B: {line}", flush=True)
    if args.games:
        print(f"match: {head_to_head(a, b, args.games, args.seconds)}", flush=True)


if __name__ == "__main__":
    main()
