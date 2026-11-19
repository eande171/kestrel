"""Plays the baseline against itself at neighbouring depths to calibrate the yardstick."""

import argparse

from .engine_client import Engine, State
from .match import MatchResult, play_match


def play(deep: int, shallow: int, games: int, max_plies: int = 300) -> MatchResult:
    with Engine(depth=deep, seed=7) as deeper:
        def deep_move(moves: list[str], state: State) -> str:
            deeper.position(moves)
            return deeper.best_move(deep)

        return play_match(None, shallow, games, max_plies, seed=deep, mover=deep_move)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--games", type=int, default=20)
    args = p.parse_args()
    print("| deeper | shallower | wins | draws | losses | score | deeper s/move |")
    print("|---|---|---|---|---|---|---|")
    for deep in range(2, args.max_depth + 1):
        r = play(deep, deep - 1, args.games)
        print(f"| {deep} | {deep - 1} | {r.wins} | {r.draws} | {r.losses} | {r.score:.2f} | "
              f"{r.seconds_per_move:.3f} |", flush=True)


if __name__ == "__main__":
    main()
