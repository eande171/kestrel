"""Talks to the Rust baseline engine over its UCI style stdin and stdout protocol."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENGINE = Path(__file__).resolve().parents[2] / "engine" / "target" / "release" / "chess-ai.exe"


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


class Engine:
    def __init__(self, path: Path = DEFAULT_ENGINE, depth: int = 2, seed: int = 1):
        self.proc = subprocess.Popen(
            [str(path), "uci", "--depth", str(depth), "--seed", str(seed)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix: str) -> list[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def set_option(self, name: str, value) -> None:
        self._send(f"setoption name {name} value {value}")

    def position(self, moves: list[str], fen: str | None = None) -> None:
        start = f"fen {fen}" if fen else "startpos"
        self._send(f"position {start} moves {' '.join(moves)}" if moves else f"position {start}")

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
        self._send(f"go depth {depth}")
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
