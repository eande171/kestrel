"""Talks to the Rust baseline engine over its UCI style stdin and stdout protocol."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENGINE = Path(
    os.environ.get(
        "KESTREL_ENGINE",
        Path(__file__).resolve().parents[2] / "engine" / "target" / "release" / ("kestrel.exe" if os.name == "nt" else "kestrel"),
    )
)


@dataclass
class State:
    fen: str
    status: str
    legal: list[str]
    repetitions: int = 0
    plies: int = 0


@dataclass
class Child:
    move: str
    san: str
    status: str
    repetitions: int
    fen: str


class EngineError(ValueError):
    """Raised when the engine rejects a position or command."""


def result_for(status: str) -> str:
    """Returns the PGN result for an engine status, counting anything but checkmate as a draw."""
    return {"checkmate_white": "1-0", "checkmate_black": "0-1"}.get(status, "1/2-1/2")


class UciProcess:
    """Runs an engine process that speaks UCI style lines over stdin and stdout."""

    def __init__(self, args: list[str]):
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix: str) -> list[str]:
        """Returns the lines up to one starting with prefix, raising EngineError afterwards if one reported an error."""
        lines, error = [], None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith("info string error "):
                error = line[len("info string error "):]
                continue
            lines.append(line)
            if line.startswith(prefix):
                if error:
                    raise EngineError(f"engine rejected a command: {error}")
                return lines

    def set_option(self, name: str, value) -> None:
        self._send(f"setoption name {name} value {value}")

    def position(self, moves: list[str], fen: str | None = None) -> None:
        start = f"fen {fen}" if fen else "startpos"
        self._send(f"position {start} moves {' '.join(moves)}" if moves else f"position {start}")

    def go(self, limits: str) -> str:
        """Returns the best move for the current position under UCI go limits such as 'depth 3'."""
        self._send(f"go {limits}")
        return self._read_until("bestmove")[-1].split()[1]

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class SearchResult:
    move: str
    depth: int
    score: float
    nodes: int


class NetEngine(UciProcess):
    """Runs the Rust network search, which keeps its caches between searches and pondering."""

    def __init__(self, weights: Path, path: Path = DEFAULT_ENGINE):
        super().__init__([str(path), "net-uci", "--weights", str(weights)])
        self._send("isready")
        self._read_until("readyok")

    def start(self, moves: list[str], base_fen: str | None, limits: str) -> None:
        self.position(moves, fen=base_fen)
        self._send(f"go {limits}")

    def result(self) -> SearchResult:
        """Waits for the running search and returns its move, depth, score and positions expanded."""
        lines = self._read_until("bestmove")
        info = next((line.split() for line in reversed(lines) if line.startswith("info depth")), [])
        fields = dict(zip(info[1::2], info[2::2]))
        return SearchResult(lines[-1].split()[1], int(fields.get("depth", 0)), float(fields.get("score", 0.0)),
                            int(fields.get("nodes", 0)))

    def stop(self) -> SearchResult:
        self._send("stop")
        return self.result()


class Engine(UciProcess):
    """Kestrel's Rust rules engine and material baseline."""

    def __init__(self, path: Path = DEFAULT_ENGINE, depth: int = 2, seed: int = 1):
        super().__init__([str(path), "uci", "--depth", str(depth), "--seed", str(seed)])
        self._send("isready")
        self._read_until("readyok")

    def state(self) -> State:
        self._send("state")
        lines = self._read_until("endstate")
        fields = {}
        for line in lines:
            key, _, rest = line.partition(" ")
            fields[key] = rest
        if "fen" not in fields:
            raise RuntimeError(f"engine error: {lines}")
        return State(
            fields["fen"],
            fields["status"],
            fields.get("legal", "").split(),
            int(fields.get("repetitions", 0)),
            int(fields.get("plies", 0)),
        )

    def children(self) -> list[Child]:
        self._send("children")
        children = []
        for line in self._read_until("endchildren")[:-1]:
            _, move, san, status, reps, fen = line.split(" ", 5)
            children.append(Child(move, san, status, int(reps), fen))
        return children

    def parse(self, text: str) -> tuple[str, str] | None:
        """Returns the UCI and SAN forms of a move typed in either notation, or None if it is not legal."""
        self._send(f"parse {text}")
        parts = self._read_until("parsed")[-1].split()
        return None if parts[1] == "none" else (parts[1], parts[2])

    def best_move(self, depth: int) -> str:
        return self.go(f"depth {depth}")
