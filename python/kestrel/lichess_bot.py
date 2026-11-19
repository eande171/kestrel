"""Plays Kestrel on Lichess through the Bot API, configured by a TOML file kept outside the repo."""

import argparse
import http.client
import json
import os
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path

from .inference import OnnxEvaluator
from .player import Clock, Player

USER_AGENT = "kestrel-bot"
SPEEDS = ["ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"]
STREAM_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, json.JSONDecodeError)
MAX_STREAM_FAILURES = 10
MOVE_ATTEMPTS = 4
RESERVE_SECONDS = 30


def default_config_path() -> Path:
    if os.environ.get("KESTREL_BOT_CONFIG"):
        return Path(os.environ["KESTREL_BOT_CONFIG"])
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "kestrel" / "bot.toml"
    return Path.home() / ".config" / "kestrel" / "bot.toml"


@dataclass
class BotConfig:
    token: str
    base_url: str = "https://lichess.org"
    model: Path = Path("model.onnx")
    threads: int = 2
    max_depth: int = 8
    width: int = 6
    randomness: float = 0.01
    reserve_seconds: float = 1.0
    max_games: int = 1
    variants: list[str] = field(default_factory=lambda: ["standard"])
    speeds: list[str] = field(default_factory=lambda: ["blitz", "rapid", "classical"])
    min_initial_seconds: int = 60
    max_initial_seconds: int = 3600
    min_increment_seconds: int = 0
    accept_rated: bool = True
    accept_casual: bool = True
    accept_bots: bool = True
    accept_humans: bool = True
    allow_users: list[str] = field(default_factory=list)
    block_users: list[str] = field(default_factory=list)
    greeting: str = ""
    goodbye: str = ""


CONVERTERS = {int: int, float: float, bool: bool, str: str, list[str]: list}

def load_config(path: Path) -> BotConfig:
    """Returns the bot settings from a TOML file, reading the token from the file, token_file or LICHESS_BOT_TOKEN."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    lichess, engine, challenges, chat = (data.get(k, {}) for k in ("lichess", "engine", "challenges", "chat"))

    token = os.environ.get("LICHESS_BOT_TOKEN") or lichess.get("token", "")
    if not token and lichess.get("token_file"):
        token_path = Path(lichess["token_file"])
        if not token_path.is_absolute():
            token_path = path.parent / token_path
        token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("no Lichess token: set lichess.token_file in the config or LICHESS_BOT_TOKEN")

    model = Path(engine.get("model", BotConfig.model))
    if not model.is_absolute():
        model = path.parent / model

    settings = {**engine, **challenges, **chat}
    types = {f.name: f.type for f in fields(BotConfig)}
    kwargs = {k: CONVERTERS[types[k]](v) for k, v in settings.items() if k in types and k not in ("token", "model")}
    if "base_url" in lichess:
        kwargs["base_url"] = lichess["base_url"].rstrip("/")
    if "randomness" in kwargs:
        kwargs["randomness"] /= 1000
    for key in ("allow_users", "block_users"):
        if key in kwargs:
            kwargs[key] = [u.lower() for u in kwargs[key]]
    unknown = [s for s in kwargs.get("speeds", []) if s not in SPEEDS]
    if unknown:
        raise SystemExit(f"unknown speeds in config: {unknown}, expected some of {SPEEDS}")
    return BotConfig(token=token, model=model, **kwargs)


def log(text: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}", flush=True)


class LichessApi:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self._token = token

    def _request(self, method: str, path: str, form: dict | None = None, timeout: float = 30):
        data = urllib.parse.urlencode(form).encode() if form is not None else (b"" if method == "POST" else None)
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("User-Agent", USER_AGENT)
        if form is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        return urllib.request.urlopen(request, timeout=timeout)

    def get_json(self, path: str) -> dict:
        with self._request("GET", path) as response:
            return json.loads(response.read())

    def post(self, path: str, form: dict | None = None) -> bool:
        for attempt in range(3):
            try:
                with self._request("POST", path, form) as response:
                    response.read()
                return True
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    log("rate limited by Lichess, waiting 60 seconds")
                    time.sleep(60)
                    continue
                log(f"POST {path.split('?')[0]} failed: HTTP {e.code} {e.read()[:200]!r}")
                return False
            except STREAM_ERRORS as e:
                log(f"POST {path.split('?')[0]} failed: {e!r}, retrying")
                time.sleep(1 + attempt)
        return False

    def stream(self, path: str):
        """Yields JSON objects from an ndjson stream, skipping keep alive lines."""
        with self._request("GET", path, timeout=60) as response:
            for raw in response:
                line = raw.strip()
                if line:
                    yield json.loads(line)


class Bot:
    def __init__(self, config: BotConfig, api: LichessApi | None = None):
        self.config = config
        self.api = api or LichessApi(config.base_url, config.token)
        self.evaluator = None
        self.username = ""
        self.games: dict[str, threading.Thread] = {}
        self.reserved: dict[str, float] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def active_games(self) -> int:
        """Returns running games plus accepted challenges whose game has not started yet."""
        with self.lock:
            self.games = {gid: t for gid, t in self.games.items() if t.is_alive()}
            now = time.time()
            self.reserved = {cid: t for cid, t in self.reserved.items()
                             if now - t < RESERVE_SECONDS and cid not in self.games}
            return len(self.games) + len(self.reserved)

    def release(self, challenge_id: str) -> None:
        with self.lock:
            self.reserved.pop(challenge_id, None)

    # Challenge Policy
    def decline_reason(self, challenge: dict) -> str | None:
        c = self.config
        challenger = challenge.get("challenger") or {}
        name = (challenger.get("id") or "").lower()
        is_bot = challenger.get("title") == "BOT"
        control = challenge.get("timeControl", {})
        if name in c.block_users or (c.allow_users and name not in c.allow_users):
            return "generic"
        if challenge.get("variant", {}).get("key") not in c.variants:
            return "variant" if "standard" in c.variants else "standard"
        if is_bot and not c.accept_bots:
            return "noBot"
        if not is_bot and not c.accept_humans:
            return "onlyBot"
        if challenge.get("rated") and not c.accept_rated:
            return "casual"
        if not challenge.get("rated") and not c.accept_casual:
            return "rated"
        speed = challenge.get("speed")
        if control.get("type") != "clock":
            return "timeControl"
        if not c.speeds:
            return "timeControl"
        if speed not in c.speeds:
            slowest_allowed = min(SPEEDS.index(s) for s in c.speeds)
            return "tooFast" if speed in SPEEDS and SPEEDS.index(speed) < slowest_allowed else "tooSlow"
        limit, increment = control.get("limit", 0), control.get("increment", 0)
        if limit < c.min_initial_seconds or increment < c.min_increment_seconds:
            return "tooFast"
        if limit > c.max_initial_seconds:
            return "tooSlow"
        if self.active_games() >= c.max_games:
            return "later"
        return None

    def handle_challenge(self, challenge: dict) -> None:
        if (challenge.get("challenger") or {}).get("id", "").lower() == self.username.lower():
            return
        reason = self.decline_reason(challenge)
        who = (challenge.get("challenger") or {}).get("name", "?")
        summary = f"{challenge.get('speed')} {challenge.get('timeControl', {}).get('show', '')} " \
                  f"{'rated' if challenge.get('rated') else 'casual'} {challenge.get('variant', {}).get('key')}"
        if reason:
            log(f"declining challenge {challenge['id']} from {who} ({summary}): {reason}")
            self.api.post(f"/api/challenge/{challenge['id']}/decline", {"reason": reason})
        else:
            log(f"accepting challenge {challenge['id']} from {who} ({summary})")
            if self.api.post(f"/api/challenge/{challenge['id']}/accept"):
                with self.lock:
                    self.reserved[challenge["id"]] = time.time()

    def start_game(self, game_id: str) -> None:
        with self.lock:
            if game_id in self.games and self.games[game_id].is_alive():
                return
            thread = threading.Thread(target=self.play_game, args=(game_id,), name=f"game-{game_id}", daemon=True)
            self.games[game_id] = thread
        thread.start()

    # Game Loop
    def play_game(self, game_id: str) -> None:
        player = Player(self.evaluator, self.config.max_depth, self.config.width, self.config.randomness,
                        self.config.reserve_seconds)
        session = {"color": None, "base_fen": None, "greeted": False, "last_played": -1}
        failures = 0
        try:
            while failures < MAX_STREAM_FAILURES:
                try:
                    for event in self.api.stream(f"/api/bot/game/stream/{game_id}"):
                        failures = 0
                        if self.on_game_event(game_id, event, player, session):
                            return
                    log(f"game {game_id}: stream ended before the game did, reconnecting")
                except STREAM_ERRORS as e:
                    log(f"game {game_id}: stream error {e!r}, reconnecting")
                failures += 1
                time.sleep(min(30, 2 * failures))
            log(f"game {game_id}: giving up after {MAX_STREAM_FAILURES} failed reconnects in a row")
        except Exception as e:
            log(f"game {game_id}: stopped by an error: {e!r}")
            raise
        finally:
            player.close()

    def on_game_event(self, game_id: str, event: dict, player: Player, session: dict) -> bool:
        """Handles one game stream event, playing a move when it is Kestrel's turn, and returns True once the game is over."""
        kind = event.get("type")
        if kind == "gameFull":
            me = self.username.lower()
            session["color"] = "white" if (event.get("white") or {}).get("id", "").lower() == me else "black"
            initial = event.get("initialFen", "startpos")
            session["base_fen"] = None if initial in ("startpos", "") else initial
            opponent = event.get("black" if session["color"] == "white" else "white") or {}
            log(f"game {game_id}: playing {session['color']} against {opponent.get('name', opponent.get('aiLevel', '?'))}")
            if self.config.greeting and not session["greeted"]:
                self.api.post(f"/api/bot/game/{game_id}/chat", {"room": "player", "text": self.config.greeting})
                session["greeted"] = True
            state = event.get("state", {})
        elif kind == "gameState":
            state = event
        else:
            return False

        if state.get("status") not in ("started", "created"):
            log(f"game {game_id}: finished ({state.get('status')}, winner {state.get('winner', 'none')})")
            if self.config.goodbye:
                self.api.post(f"/api/bot/game/{game_id}/chat", {"room": "player", "text": self.config.goodbye})
            return True
        moves = state.get("moves", "").split()
        base_fen = session["base_fen"]
        start_white = base_fen is None or base_fen.split()[1] == "w"
        white_to_move = (len(moves) % 2 == 0) == start_white
        if session["color"] is None or white_to_move != (session["color"] == "white") \
                or len(moves) == session["last_played"]:
            return False

        clock = Clock(state.get("wtime"), state.get("btime"), state.get("winc", 0), state.get("binc", 0))
        decision = player.choose(moves, base_fen, clock)
        if decision.move == "0000":
            return False
        for attempt in range(MOVE_ATTEMPTS):
            if self.api.post(f"/api/bot/game/{game_id}/move/{decision.move}"):
                session["last_played"] = len(moves)
                log(f"game {game_id}: move {len(moves) // 2 + 1} {decision.move} "
                    f"(depth {decision.depth}, {decision.seconds:.2f}s)")
                return False
            time.sleep(1 + attempt)
        log(f"game {game_id}: could not send {decision.move} after {MOVE_ATTEMPTS} tries")
        return False

    def run(self) -> None:
        account = self.api.get_json("/api/account")
        self.username = account.get("username", account.get("id", ""))
        if account.get("title") != "BOT":
            raise SystemExit(f"account {self.username} is not a BOT account yet, see the README for the upgrade step")
        self.evaluator = OnnxEvaluator(self.config.model, self.config.threads)
        log(f"Kestrel bot running as {self.username} (depth up to {self.config.max_depth}, "
            f"speeds {', '.join(self.config.speeds)}, max {self.config.max_games} game(s))")
        backoff = 1
        while not self.stop.is_set():
            try:
                for event in self.api.stream("/api/stream/event"):
                    backoff = 1
                    kind = event.get("type")
                    if kind == "challenge":
                        self.handle_challenge(event["challenge"])
                    elif kind in ("challengeCanceled", "challengeDeclined"):
                        self.release(event.get("challenge", {}).get("id", ""))
                    elif kind == "gameStart":
                        game = event.get("game", {})
                        game_id = game.get("gameId") or game.get("id")
                        self.start_game(game_id)
                        self.release(game_id)
                    if self.stop.is_set():
                        break
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(f"Lichess rejected the token (HTTP {e.code}). Create a new bot:play token "
                                     "and save it to the token file.")
                wait = 60 if e.code == 429 else min(60, backoff)
                log(f"event stream HTTP {e.code}, reconnecting in {wait}s")
                time.sleep(wait)
            except STREAM_ERRORS as e:
                log(f"event stream lost ({e!r}), reconnecting in {backoff}s")
                time.sleep(backoff)
            backoff = min(60, backoff * 2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="bot.toml location, default KESTREL_BOT_CONFIG or the user config folder")
    args = p.parse_args(argv)
    path = args.config or default_config_path()
    if not path.exists():
        raise SystemExit(f"config not found at {path}. Copy bot.example.toml there and edit it.")
    config = load_config(path)
    log(f"using config {path}")
    try:
        Bot(config).run()
    except KeyboardInterrupt:
        log("stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
