"""Serves the web UI and a small JSON API backed by the network and the Rust engine."""

import argparse
import json
import mimetypes
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from .engine_client import Engine
from .match import network_move
from .model import count_parameters, load_model
from .search import NetSearch

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
MAX_DEPTH = 4


class ChessApi:
    def __init__(self, run: Path, checkpoint: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, ckpt = load_model(run / "checkpoints" / checkpoint, self.device)
        self.engine = Engine()
        self.info = {
            "run": run.name,
            "step": ckpt.get("step") or ckpt.get("state", {}).get("step"),
            "parameters": count_parameters(self.model),
            "device": self.device,
        }

    def position(self, moves: list[str]) -> dict:
        self.engine.position(moves)
        state = self.engine.state()
        if state.plies != len(moves):
            raise ValueError("illegal move list")
        children = self.engine.children() if state.status == "ongoing" else []
        return {
            "fen": state.fen,
            "status": state.status,
            "repetitions": state.repetitions,
            "legal": [{"uci": c.move, "san": c.san} for c in children],
        }

    def reply(self, moves: list[str], depth: int, width: int) -> dict:
        depth = max(0, min(MAX_DEPTH, depth))
        self.engine.position(moves)
        state = self.engine.state()
        if state.status != "ongoing":
            raise ValueError("game is over")
        start = time.time()
        if depth == 0:
            uci = network_move(self.model, state, self.device)
        else:
            uci = NetSearch(self.model, self.engine, self.device, depth, width).choose(
                moves, state.fen, state.repetitions
            )
        self.engine.position(moves)
        _, san = self.engine.parse(uci)
        return {"uci": uci, "san": san, "seconds": round(time.time() - start, 3)}


def make_handler(api: ChessApi):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            if self.path == "/api/info":
                return self._json(200, api.info)
            rel = self.path.split("?")[0].lstrip("/") or "index.html"
            path = (WEB_ROOT / rel).resolve()
            if WEB_ROOT not in path.parents or not path.is_file():
                return self._send(404, b"not found", "text/plain")
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), content_type)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                moves = [str(m) for m in body.get("moves", [])]
                if self.path == "/api/position":
                    return self._json(200, api.position(moves))
                if self.path == "/api/move":
                    api.position(moves)
                    return self._json(200, api.reply(moves, int(body.get("depth", 3)), int(body.get("width", 6))))
                self._json(404, {"error": "unknown endpoint"})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/rep")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = p.parse_args(argv)

    api = ChessApi(Path(args.run), args.checkpoint)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(api))
    url = f"http://localhost:{args.port}/"
    print(f"chess-ai web UI on {url} (model {api.info['run']} step {api.info['step']}, {api.info['device']}). Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        api.engine.close()


if __name__ == "__main__":
    main()
