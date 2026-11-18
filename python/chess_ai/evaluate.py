"""Plays a checkpoint against the baseline at several depths and prints a results table."""

import argparse
import json
import time
from pathlib import Path

from .match import play_match
from .model import load_model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--search-depth", type=int, default=0, help="0 plays the greedy policy move")
    p.add_argument("--width", type=int, default=6, help="policy moves searched at each node")
    args = p.parse_args(argv)

    run = Path(args.run)
    model, ckpt = load_model(run / "checkpoints" / args.checkpoint, "cuda")
    mode = f"search depth {args.search_depth} width {args.width}" if args.search_depth else "greedy"

    rows = []
    print(f"checkpoint {args.checkpoint} step {ckpt.get('step', ckpt.get('state', {}).get('step'))}, {mode}")
    print("| baseline depth | games | wins | draws | losses | score | illegal | avg plies | s per move | endings |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for depth in args.depths:
        start = time.time()
        r = play_match(model, depth, args.games, "cuda", seed=args.seed + depth,
                       search_depth=args.search_depth, width=args.width)
        avg = sum(r.plies) / max(1, len(r.plies))
        endings = ", ".join(f"{k} {v}" for k, v in sorted(r.endings.items()))
        print(f"| {depth} | {r.games} | {r.wins} | {r.draws} | {r.losses} | {r.score:.2f} | {r.illegal} | "
              f"{avg:.0f} | {r.seconds_per_move:.3f} | {endings} |", flush=True)
        rows.append({"depth": depth, "games": r.games, "wins": r.wins, "draws": r.draws, "losses": r.losses,
                     "score": r.score, "illegal": r.illegal, "avg_plies": avg, "endings": r.endings,
                     "seconds_per_move": r.seconds_per_move, "seconds": round(time.time() - start, 1)})
    suffix = f"s{args.search_depth}w{args.width}" if args.search_depth else "greedy"
    with open(run / f"match_{Path(args.checkpoint).stem}_{suffix}.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
