"""Plays the baseline against itself at neighbouring depths to calibrate the yardstick."""

import argparse
import time

from .engine_client import Engine
from .match import OPENINGS


def play(deep: int, shallow: int, games: int, max_plies: int = 300) -> tuple[int, int, int, float]:
    wins = draws = losses = 0
    deep_seconds, deep_moves = 0.0, 0
    with Engine(depth=deep, seed=7) as a, Engine(depth=shallow, seed=11) as b:
        for g in range(games):
            a.set_option("Seed", 1000 + g)
            b.set_option("Seed", 2000 + g)
            moves = OPENINGS[(g // 2) % len(OPENINGS)].split()
            deep_white = g % 2 == 0
            while True:
                a.position(moves)
                state = a.state()
                if state.status != "ongoing" or len(moves) >= max_plies:
                    break
                white = state.fen.split()[1] == "w"
                if white == deep_white:
                    start = time.time()
                    moves.append(a.best_move(deep))
                    deep_seconds += time.time() - start
                    deep_moves += 1
                else:
                    b.position(moves)
                    moves.append(b.best_move(shallow))
            if state.status.startswith("checkmate"):
                if (state.status == "checkmate_white") == deep_white:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
    return wins, draws, losses, deep_seconds / max(1, deep_moves)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--games", type=int, default=20)
    args = p.parse_args()
    print("| deeper | shallower | wins | draws | losses | score | deeper s/move |")
    print("|---|---|---|---|---|---|---|")
    for deep in range(2, args.max_depth + 1):
        w, d, l, spm = play(deep, deep - 1, args.games)
        print(f"| {deep} | {deep - 1} | {w} | {d} | {l} | {(w + d / 2) / args.games:.2f} | {spm:.3f} |", flush=True)


if __name__ == "__main__":
    main()
