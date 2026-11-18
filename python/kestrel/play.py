"""Plays a game against the network in the terminal."""

import argparse
import random
import time
from pathlib import Path

import torch

from .engine_client import Engine
from .match import network_move
from .model import load_model
from .search import NetSearch

RESULTS = {
    "checkmate_white": ("1-0", "White wins by checkmate"),
    "checkmate_black": ("0-1", "Black wins by checkmate"),
    "stalemate": ("1/2-1/2", "Draw by stalemate"),
    "repetition": ("1/2-1/2", "Draw by threefold repetition"),
    "fifty_moves": ("1/2-1/2", "Draw by the fifty move rule"),
    "insufficient_material": ("1/2-1/2", "Draw, neither side can mate"),
}
HELP = "Type moves like e4, Nf3, exd5, O-O, e8=Q or e2e4. Commands: undo, flip, moves, resign, help."


def render(fen: str, white_bottom: bool, last: str | None) -> str:
    rows = fen.split()[0].split("/")
    grid = []
    for row in rows:
        line = []
        for ch in row:
            line.extend(["."] * int(ch) if ch.isdigit() else [ch])
        grid.append(line)
    marked = set()
    if last:
        for sq in (last[0:2], last[2:4]):
            marked.add((8 - int(sq[1]), ord(sq[0]) - ord("a")))
    ranks = range(8) if white_bottom else range(7, -1, -1)
    files = range(8) if white_bottom else range(7, -1, -1)
    out = []
    for r in ranks:
        cells = []
        for f in files:
            piece = grid[r][f]
            cells.append(f"[{piece}]" if (r, f) in marked else f" {piece} ")
        out.append(f" {8 - r} " + "".join(cells))
    out.append("    " + "".join(f" {'abcdefgh'[f]} " for f in files))
    return "\n".join(out)


def pgn(sans: list[str], result: str, human_white: bool, label: str) -> str:
    white, black = ("You", label) if human_white else (label, "You")
    body = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            body.append(f"{i // 2 + 1}.")
        body.append(san)
    body.append(result)
    headers = [
        '[Event "Kestrel test game"]',
        f'[Date "{time.strftime("%Y.%m.%d")}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        f'[Result "{result}"]',
    ]
    return "\n".join(headers) + "\n\n" + " ".join(body) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/rep")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--color", choices=["white", "black", "random"], default="random")
    p.add_argument("--search-depth", type=int, default=3, help="0 plays the top move with no look ahead")
    p.add_argument("--width", type=int, default=6)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(Path(args.run) / "checkpoints" / args.checkpoint, device)
    human_white = args.color == "white" or (args.color == "random" and random.random() < 0.5)
    white_bottom = human_white
    label = f"Kestrel ({'top move' if args.search_depth == 0 else f'search depth {args.search_depth}'})"

    moves: list[str] = []
    sans: list[str] = []
    result = "*"
    print(f"\nYou are {'White' if human_white else 'Black'} against {label}. {HELP}\n")
    with Engine() as engine:
        searcher = NetSearch(model, engine, device, args.search_depth, args.width) if args.search_depth else None
        while True:
            engine.position(moves)
            state = engine.state()
            print(render(state.fen, white_bottom, moves[-1] if moves else None))
            if state.status != "ongoing":
                result, text = RESULTS[state.status]
                print(f"\n{text}.")
                break
            white_to_move = state.fen.split()[1] == "w"
            if white_to_move == human_white:
                text = input(f"\nYour move ({len(sans) // 2 + 1}{'.' if human_white else '...'}): ").strip()
                if not text:
                    continue
                command = text.lower()
                if command in ("quit", "exit", "resign"):
                    result = "0-1" if human_white else "1-0"
                    print("\nYou resigned.")
                    break
                if command == "help":
                    print(HELP)
                    continue
                if command == "flip":
                    white_bottom = not white_bottom
                    continue
                if command == "moves":
                    print(" ".join(sans) or "(no moves yet)")
                    continue
                if command == "undo":
                    if len(moves) >= 2:
                        del moves[-2:], sans[-2:]
                    else:
                        print("Nothing to undo.")
                    continue
                parsed = engine.parse(text)
                if parsed is None:
                    print(f"'{text}' is not a legal move here. Pieces use capital letters (Nf3, not nf3).")
                    continue
                uci, san = parsed
                moves.append(uci)
                sans.append(san)
            else:
                start = time.time()
                if searcher:
                    uci = searcher.choose(moves, state.fen, state.repetitions)
                    engine.position(moves)
                else:
                    uci = network_move(model, state, device)
                _, san = engine.parse(uci)
                moves.append(uci)
                sans.append(san)
                number = len(sans) // 2 + (len(sans) % 2)
                print(f"\nKestrel plays{number}{'.' if not human_white else '...'} {san}  ({time.time() - start:.2f}s)\n")

    game_dir = Path(args.run).parent / "games"
    game_dir.mkdir(parents=True, exist_ok=True)
    path = game_dir / f"game_{time.strftime('%Y%m%d_%H%M%S')}.pgn"
    path.write_text(pgn(sans, result, human_white, label))
    print(f"\nGame saved to {path}. You can paste it into lichess.org/analysis to review it.")


if __name__ == "__main__":
    main()
