"""Runs the Lichess bot against a local mock of the Lichess Bot API and checks what it did."""

import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.engine_client import Engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "test-token-not-real"
PLY_LIMIT = 40
DROP_AFTER_PLIES = 8


def challenge(cid: str, user: str, variant: str, speed: str, limit: int, increment: int) -> dict:
    return {
        "type": "challenge",
        "challenge": {
            "id": cid,
            "status": "created",
            "challenger": {"id": user, "name": user.title(), "rating": 1500, "title": None},
            "destUser": {"id": "kestrelbot", "name": "KestrelBot", "title": "BOT"},
            "variant": {"key": variant},
            "rated": False,
            "speed": speed,
            "timeControl": {"type": "clock", "limit": limit, "increment": increment, "show": f"{limit // 60}+{increment}"},
            "color": "random",
        },
    }


class MockLichess:
    def __init__(self):
        self.events: queue.Queue = queue.Queue()
        self.game_updates: queue.Queue = queue.Queue()
        self.declined: dict[str, str] = {}
        self.accepted: list[str] = []
        self.chat: list[str] = []
        self.illegal: list[str] = []
        self.moves: list[str] = []
        self.bot_moves = 0
        self.game_streams = 0
        self.finished = threading.Event()
        self.matchmade = threading.Event()
        self.sent_challenges: list[tuple[str, dict]] = []
        self.cancelled: list[str] = []
        self.engine = Engine(depth=1, seed=3)
        self.lock = threading.Lock()
        for event in (
            challenge("c960", "alice", "chess960", "blitz", 180, 2),
            challenge("cbullet", "bob", "standard", "bullet", 60, 0),
            challenge("cblitz", "carol", "standard", "blitz", 180, 2),
            challenge("cbusy", "dave", "standard", "blitz", 180, 2),
            challenge("stale1", "kestrelbot", "standard", "blitz", 180, 2),
        ):
            self.events.put(event)

    def state_event(self, status: str = "started", winner: str | None = None) -> dict:
        event = {"type": "gameState", "moves": " ".join(self.moves), "wtime": 170000, "btime": 170000,
                 "winc": 2000, "binc": 2000, "status": status}
        if winner:
            event["winner"] = winner
        return event

    def bot_moved(self, move: str) -> bool:
        with self.lock:
            self.engine.position(self.moves)
            state = self.engine.state()
            if state.plies % 2 != 0 or move not in state.legal:
                self.illegal.append(move)
                return False
            self.moves.append(move)
            self.bot_moves += 1
            self.engine.position(self.moves)
            state = self.engine.state()
            if state.status != "ongoing" or len(self.moves) >= PLY_LIMIT:
                status = "mate" if state.status.startswith("checkmate") else "resign"
                self.game_updates.put(self.state_event(status, "white"))
                return True
            self.moves.append(self.engine.best_move(1))
            self.engine.position(self.moves)
            state = self.engine.state()
            if state.status != "ongoing":
                self.game_updates.put(self.state_event("mate" if state.status.startswith("checkmate") else "draw", "black"))
            else:
                self.game_updates.put(self.state_event())
            return True


def make_handler(mock: MockLichess):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def authorized(self) -> bool:
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                self.send_error(401)
                return False
            return True

        def json_reply(self, data: dict, code: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def start_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()

        def write_line(self, data: dict | None) -> None:
            self.wfile.write((json.dumps(data) + "\n").encode() if data else b"\n")
            self.wfile.flush()

        def do_GET(self):
            if not self.authorized():
                return
            path = urlparse(self.path).path
            if path == "/api/account":
                return self.json_reply({"id": "kestrelbot", "username": "KestrelBot", "title": "BOT",
                                        "perfs": {"blitz": {"rating": 2000, "games": 40}}})
            if path == "/api/bot/online":
                self.start_stream()
                for name, rating in (("KestrelBot", 2000), ("FarBot", 2900), ("GoodBot", 2050), ("RapidOnly", None)):
                    perfs = {"blitz": {"rating": rating, "games": 50}} if rating else {"rapid": {"rating": 2000}}
                    self.write_line({"id": name.lower(), "username": name, "title": "BOT", "perfs": perfs})
                return
            if path == "/api/stream/event":
                self.start_stream()
                while not mock.finished.is_set():
                    try:
                        self.write_line(mock.events.get(timeout=1))
                    except queue.Empty:
                        self.write_line(None)
                return
            if path == "/api/bot/game/stream/cblitz":
                mock.game_streams += 1
                first_stream = mock.game_streams == 1
                self.start_stream()
                full = {"type": "gameFull", "id": "cblitz", "variant": {"key": "standard"}, "speed": "blitz",
                        "rated": False, "white": {"id": "kestrelbot", "name": "KestrelBot", "title": "BOT"},
                        "black": {"id": "carol", "name": "Carol", "title": None}, "initialFen": "startpos",
                        "clock": {"initial": 180000, "increment": 2000}, "state": mock.state_event()}
                self.write_line(full)
                while True:
                    try:
                        event = mock.game_updates.get(timeout=1)
                    except queue.Empty:
                        self.write_line(None)
                        continue
                    self.write_line(event)
                    if first_stream and len(event["moves"].split()) >= DROP_AFTER_PLIES:
                        return
                    if event["status"] != "started":
                        time.sleep(0.5)
                        mock.finished.set()
                        return
            self.send_error(404)

        def do_POST(self):
            if not self.authorized():
                return
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
            parts = urlparse(self.path).path.strip("/").split("/")
            if parts[:2] == ["api", "challenge"] and len(parts) == 3:
                mock.sent_challenges.append((parts[2], form))
                mock.matchmade.set()
                return self.json_reply({"id": "out1", "status": "created", "direction": "out"})
            if parts[:2] == ["api", "challenge"] and parts[3] == "cancel":
                mock.cancelled.append(parts[2])
                return self.json_reply({"ok": True})
            if parts[:2] == ["api", "challenge"] and parts[3] == "decline":
                mock.declined[parts[2]] = form.get("reason", "")
                return self.json_reply({"ok": True})
            if parts[:2] == ["api", "challenge"] and parts[3] == "accept":
                mock.accepted.append(parts[2])
                mock.events.put({"type": "gameStart", "game": {"gameId": "cblitz", "id": "cblitz", "color": "white"}})
                return self.json_reply({"ok": True})
            if parts[:4] == ["api", "bot", "game", "cblitz"] and parts[4] == "move":
                ok = mock.bot_moved(parts[5])
                return self.json_reply({"ok": ok} if ok else {"error": "illegal"}, 200 if ok else 400)
            if parts[:4] == ["api", "bot", "game", "cblitz"] and parts[4] == "chat":
                mock.chat.append(form.get("text", ""))
                return self.json_reply({"ok": True})
            self.send_error(404)

    return Handler


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def main() -> None:
    mock = MockLichess()
    server = QuietServer(("127.0.0.1", 0), make_handler(mock))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    folder = Path(tempfile.mkdtemp(prefix="kestrel-bot-test-"))
    (folder / "token.txt").write_text(TOKEN)
    config = (ROOT / "bot.example.toml").read_text()
    config = config.replace('base_url = "https://lichess.org"', f'base_url = "http://127.0.0.1:{port}"')
    config = config.replace("max_depth = 10", "max_depth = 4")
    config = config.replace("enabled = false", "enabled = true").replace("idle_minutes = 2.0", "idle_minutes = 0.02")
    config = config.replace('model = "model.onnx"', f'model = "{(ROOT / "runs/rep/model.onnx").as_posix()}"')
    (folder / "bot.toml").write_text(config)

    bot = subprocess.Popen(
        [sys.executable, "-m", "kestrel.lichess_bot", "--config", str(folder / "bot.toml")],
        cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "python")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(bot.stdout), daemon=True).start()
    finished = mock.finished.wait(timeout=300)
    matchmade = mock.matchmade.wait(timeout=15)
    deadline = time.time() + 40
    while "out1" not in mock.cancelled and time.time() < deadline:
        time.sleep(0.5)
    bot.terminate()
    bot.wait(timeout=10)
    server.shutdown()
    shutil.rmtree(folder, ignore_errors=True)

    print("".join(line for line in output if ": move " not in line))
    assert finished, "game did not finish within 5 minutes"
    assert mock.declined.get("c960") == "variant", mock.declined
    assert mock.declined.get("cbullet") == "tooFast", mock.declined
    assert mock.accepted == ["cblitz"], mock.accepted
    assert mock.declined.get("cbusy") == "later", mock.declined
    assert mock.game_streams == 2, f"{mock.game_streams} game stream connections, expected one reconnect"
    assert not mock.illegal, f"illegal moves sent: {mock.illegal}"
    assert mock.bot_moves >= 5, f"only {mock.bot_moves} bot moves"
    assert len(mock.chat) == 2, mock.chat
    assert not any(TOKEN in line for line in output), "token appeared in the bot log"
    assert matchmade, "no matchmaking challenge after the game"
    assert all(name.lower() == "goodbot" for name, _ in mock.sent_challenges), mock.sent_challenges
    _, form = mock.sent_challenges[0]
    assert form["rated"] == "true" and (form["clock.limit"], form["clock.increment"]) in (("180", "2"), ("300", "3")), form
    assert mock.cancelled == ["stale1", "out1"], f"stale or unanswered challenge not cancelled: {mock.cancelled}"
    print(f"ok: declined chess960 (variant), bullet (tooFast) and a second blitz while busy (later), accepted blitz, "
          f"reconnected after a dropped stream, {mock.bot_moves} legal bot moves over {len(mock.moves)} plies, "
          f"greeting and goodbye sent, token never logged, then challenged the one online bot in rating range and cancelled it when unanswered, after cancelling one left from an earlier run")


if __name__ == "__main__":
    main()
