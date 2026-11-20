"""Builds a suite of positions where Kestrel went wrong in real games, scored by Stockfish, and grades models on it."""

import argparse
import json
import math
import re
import time
from pathlib import Path

from .engine_client import Engine, UciProcess
from .search import NetSearch
from .stockfish_match import DEFAULT_STOCKFISH

DEFAULT_SUITE = Path(__file__).resolve().parent / "suites" / "mistakes.jsonl"
LICHESS_SCALE = 0.00368208
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def expected_score(kind: str, value: int) -> float:
    """Returns the mover's expected score for a UCI score, using the Lichess win model for centipawns."""
    if kind == "mate":
        return 1.0 if value > 0 else 0.0
    return 1 / (1 + math.exp(-LICHESS_SCALE * value))


class Analyst(UciProcess):
    """Runs Stockfish at full strength on one thread."""

    def __init__(self, path: Path = DEFAULT_STOCKFISH, depth: int = 14):
        super().__init__([str(path)])
        self.depth = depth
        self._send("uci")
        self._read_until("uciok")
        self.set_option("Threads", 1)
        self.set_option("Hash", 64)
        self._send("isready")
        self._read_until("readyok")

    def scores(self, fen: str, lines: int = 1) -> dict[str, float]:
        """Returns the mover's expected score for each of the top lines' first moves."""
        self.set_option("MultiPV", lines)
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        found = {}
        for line in self._read_until("bestmove"):
            m = re.search(r" depth (\d+) .*?multipv \d+ score (cp|mate) (-?\d+).* pv (\S+)", line)
            if m and int(m.group(1)) == self.depth:
                found[m.group(4)] = expected_score(m.group(2), int(m.group(3)))
        return found


def read_pgn(text: str):
    """Yields the headers and SAN moves of each game in a PGN file."""
    for block in re.split(r"\n\s*\n(?=\[)", text.strip()):
        headers = dict(re.findall(r'\[(\w+) "([^"]*)"\]', block))
        body = re.sub(r"\{[^}]*\}|\([^)]*\)", " ", re.sub(r"^\[.*$", "", block, flags=re.M))
        sans = [t for t in body.split() if not re.match(r"^\d+\.+$", t) and t not in RESULTS]
        if headers and sans:
            yield headers, sans


def after_score(engine: Engine, analyst: Analyst, moves: list[str]) -> float:
    """Returns the expected score of the side that just moved."""
    engine.position(moves)
    state = engine.state()
    if state.status.startswith("checkmate"):
        return 1.0
    if state.status != "ongoing":
        return 0.5
    reply = analyst.scores(state.fen)
    return 1.0 - max(reply.values())


def build(pgns: list[Path], player: str, min_drop: float, depth: int) -> list[dict]:
    """Returns every position where the player's move lost at least min_drop of expected score."""
    analyst = Analyst(depth=depth)
    suite = []
    with Engine() as engine:
        for path in pgns:
            for headers, sans in read_pgn(path.read_text(encoding="utf-8")):
                if "FEN" in headers or player.lower() not in (headers.get("White", "").lower(), headers.get("Black", "").lower()):
                    continue
                white = headers["White"].lower() == player.lower()
                moves = []
                for ply, san in enumerate(sans):
                    engine.position(moves)
                    state = engine.state()
                    parsed = engine.parse(san)
                    if parsed is None:
                        break
                    if (ply % 2 == 0) == white:
                        best = max(analyst.scores(state.fen).values())
                        played = after_score(engine, analyst, moves + [parsed[0]])
                        if best - played >= min_drop:
                            scores = analyst.scores(state.fen, len(state.legal))
                            scores[parsed[0]] = min(scores.get(parsed[0], played), played)
                            suite.append({"game": headers.get("Site", ""), "move": f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'}{san}",
                                          "fen": state.fen, "played": parsed[0],
                                          "scores": {m: round(s, 4) for m, s in scores.items()}})
                            print(f"{suite[-1]['game']} {suite[-1]['move']}: loses {best - played:.2f}", flush=True)
                    moves.append(parsed[0])
    analyst.close()
    return suite


def load(path: Path = DEFAULT_SUITE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grade(evaluator, suite: list[dict], seconds: float) -> str:
    """Returns the model's average expected score lost against Stockfish's best and how often it repeats the game move."""
    lost, repeats, best_found = 0.0, 0, 0
    with Engine() as engine:
        search = NetSearch(evaluator, engine, 8, 6)
        for case in suite:
            engine.position([], fen=case["fen"])
            state = engine.state()
            move = search.choose([], case["fen"], state.repetitions, base_fen=case["fen"], depth=10,
                                 deadline=time.monotonic() + seconds)
            scores = case["scores"]
            top = max(scores.values())
            lost += top - scores.get(move, min(scores.values()))
            repeats += move == case["played"]
            best_found += scores.get(move, -1) >= top - 0.02
    n = len(suite)
    return (f"{n} positions: average expected score lost {lost / n:.3f}, near best {best_found}/{n}, "
            f"repeated the game move {repeats}/{n}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pgn", type=Path, nargs="+", required=True)
    p.add_argument("--player", default="KestrelChess")
    p.add_argument("--min-drop", type=float, default=0.15, help="expected score lost by the game move")
    p.add_argument("--depth", type=int, default=14, help="Stockfish depth")
    p.add_argument("--out", type=Path, default=DEFAULT_SUITE)
    args = p.parse_args(argv)
    suite = build(args.pgn, args.player, args.min_drop, args.depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    known = {(c["fen"], c["played"]) for c in (load(args.out) if args.out.exists() else [])}
    added = [c for c in suite if (c["fen"], c["played"]) not in known]
    with args.out.open("a", encoding="utf-8") as f:
        for case in added:
            f.write(json.dumps(case) + "\n")
    print(f"added {len(added)} positions to {args.out}")


if __name__ == "__main__":
    main()
