"""Plays a checkpoint against the baseline at several depths and prints a results table."""

import argparse
import json
import time
from pathlib import Path

import torch

from .match import play_match
from .model import ChessNet


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)

    run = Path(args.run)
    ckpt = torch.load(run / "checkpoints" / args.checkpoint, map_location="cuda", weights_only=False)
    model = ChessNet(**ckpt["model_config"]).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    print(f"checkpoint {args.checkpoint} step {ckpt.get('step', ckpt.get('state', {}).get('step'))}")
    print("| depth | games | wins | draws | losses | score | illegal | avg plies | endings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for depth in args.depths:
        start = time.time()
        r = play_match(model, depth, args.games, "cuda", seed=args.seed + depth)
        avg = sum(r.plies) / max(1, len(r.plies))
        endings = ", ".join(f"{k} {v}" for k, v in sorted(r.endings.items()))
        print(f"| {depth} | {r.games} | {r.wins} | {r.draws} | {r.losses} | {r.score:.2f} | {r.illegal} | {avg:.0f} | {endings} |", flush=True)
        rows.append({"depth": depth, "games": r.games, "wins": r.wins, "draws": r.draws, "losses": r.losses,
                     "score": r.score, "illegal": r.illegal, "avg_plies": avg, "endings": r.endings,
                     "seconds": round(time.time() - start, 1)})
    with open(run / f"match_{Path(args.checkpoint).stem}.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
