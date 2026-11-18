# Kestrel

**K**een-**E**yed **S**earch **T**rained on **R**eal **E**ncounters from **L**ichess.

A neural network chess AI, built mainly as a learning project, with a decent opponent to play against as a bonus.

## Goals

Long term, the plan is a small network plus a short search, compiled to WebAssembly so it runs in the browser on a static website. It should also do "opponent modelling": learn a player's recurring mistakes from their own games and help them improve. Self-play, WebAssembly, the website and opponent modelling are later milestones.

Milestone 1 (this repository so far):

- a data pipeline that turns a month of Lichess games into training positions
- a small supervised policy and value network trained on those positions
- a training setup that can run unattended and resume after a stop or crash
- a deliberately simple baseline engine to measure the network against

## Layout

```
engine/          Rust crate: board wrapper, perft tests, baseline engine, UCI loop, PGN to shards pipeline
python/kestrel/  encoding, model, training, search, evaluation matches, ONNX export, UCI engine
python/tests/    encoding consistency check against the Rust engine
scripts/         train.ps1 (start or resume training), web.ps1 and play.ps1 (play the network)
web/             browser UI served by python/kestrel/web.py
data/            training shards (not committed)
runs/            checkpoints, logs and exported models (not committed)
```

## Setup

Rust (stable, MSVC toolchain) and Python 3.13.

```powershell
cd engine
cargo build --release
cargo test --release
```

The Python environment lives outside the repo, in `D:\chess-venv` (about 4.5 GB, mostly PyTorch):

```powershell
python -m venv D:\chess-venv
D:\chess-venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
D:\chess-venv\Scripts\python.exe -m pip install -r python\requirements.txt
```

## Playing the network

In the browser:

```powershell
.\scripts\web.ps1                     # starts a local server on http://localhost:8000 and opens it
.\scripts\web.ps1 -Port 8080 -NoOpen
```

Click or drag pieces to move. Choose your colour and strength (Instinct plays its first idea, Careful looks 2 moves ahead, Strong looks 3 ahead) and start a new game. Undo, flip, resign and copy PGN are next to the board. Ctrl+C in the terminal stops the server.

The page in `web/` is plain HTML, CSS and JavaScript with no build step. It holds the game itself and asks the server only two things: the legal moves and status for a move list (`POST /api/position`) and Kestrel's reply (`POST /api/move`). The server keeps no game state and only listens on 127.0.0.1. A later in-browser engine can answer the same two questions without the page changing.

In the terminal:

```powershell
.\scripts\play.ps1                    # random colour, search depth 3 (about a quarter of a second per move)
.\scripts\play.ps1 -Color white -Depth 2
.\scripts\play.ps1 -Depth 0           # top move only, no look ahead (weakest)
```

Type moves in standard notation (`e4`, `Nf3`, `exd5`, `O-O`, `e8=Q`) or as squares (`e2e4`). Pieces need capital letters. Other commands are `undo`, `flip`, `moves`, `resign` and `help`. The last move is shown in brackets on the board, uppercase letters are White and lowercase are Black. Each game is saved as a PGN file in `runs/games/`, which can be pasted into lichess.org/analysis.

## Kestrel as a UCI engine

`python -m kestrel.uci` runs Kestrel (network plus search) as a standard UCI engine, so any chess GUI or bot bridge can use it. It runs on the CPU with ONNX Runtime and does not need PyTorch or a GPU.

```powershell
$env:PYTHONPATH = "python"
D:\chess-venv\Scripts\python.exe -m kestrel.uci --model runs/rep/model.onnx --threads 2
```

- Needs `numpy`, `onnxruntime`, an exported model and the Rust `kestrel` binary (found next to the repo, or set `KESTREL_ENGINE`). `KESTREL_MODEL` and `KESTREL_THREADS` set the defaults.
- UCI options: `Depth` (0 to 4, default 3), `Width` (moves searched at each step, default 6) and `Randomness` (in thousandths; root moves this close to the best are picked at random, default 10, so it does not repeat the same game).
- Time management: from `wtime`/`btime`/`winc`/`binc` it allows about a thirtieth of the remaining time plus most of the increment, and drops to a shallower search when a deeper one would not fit. It learns how long each depth takes as it plays.

Speed on an i7-12700KF with 2 threads: depth 3 takes about 0.1 s per move and depth 4 about 0.6 s. ONNX and PyTorch choose the same move on 100% of 5,000 test positions (`python/tests/test_features.py`).

## Baseline engine

```powershell
.\engine\target\release\kestrel.exe play --depth 3              # play it in the terminal (SAN or UCI moves)
.\engine\target\release\kestrel.exe uci --depth 3               # UCI engine on stdin and stdout
.\engine\target\release\kestrel.exe perft 6                     # move generation check
.\engine\target\release\kestrel.exe bench --depth 5             # search speed
```

Material only evaluation, fixed depth alpha-beta, capture-only quiescence search and random tie breaking. It does not know about piece placement, king safety or pawn structure, so it is weak on purpose. Depth 1 to 4 all reply in well under a second.

Besides standard UCI commands it accepts two extra commands, so Python never needs its own move generator:

- `state` prints the FEN, the game status (including repetition, fifty moves and insufficient material), how many times the current position has occurred before, and the legal moves.
- `children` prints one line per legal move with the move, the resulting status, repetition count and FEN. The network search uses this to expand a node in one round trip.

## Data

The prep tool streams a Lichess monthly file over HTTP, decompresses it on the fly, filters games and stops once it has enough positions. The full monthly file is around 30 GB but only the part that is read gets downloaded.

```powershell
.\engine\target\release\kestrel.exe prep `
  --input https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst `
  --out data/shards_2026-10_20m_v2 --positions 20000000 --shards 16 --min-elo 1800
```

Run `kestrel.exe` with no arguments for all options (rating threshold, minimum time control, plies skipped, opening position cap and so on). A local `.pgn` or `.pgn.zst` file works too.

## Training

One command starts a run, and the same command resumes it from the latest checkpoint:

```powershell
.\scripts\train.ps1 -Run runs/rep -Data data/shards_2026-10_20m_v2 --max-hours 2
```

- Any `train.py` option can be added after the script options, for example `--max-steps 300000 --eval-every 10000 --patience 5`.
- `-Quiet` halves the GPU duty cycle and caps the batch at 512 (about 137 W instead of 220 W on the 3070 Ti). `--duty 0.7` gives finer control.
- To stop cleanly, create an empty file called `STOP` in the run folder, or press Ctrl+C. Both save a checkpoint first.
- If Python crashes, the script restarts it after 30 seconds and training resumes from the latest checkpoint.

Training stops on `--max-steps`, `--max-hours` (cumulative across resumes) or when held-out loss has not improved for `--patience` evaluations. A `DONE` file in the run folder records why.

Each run folder contains:

- `log.csv`: step, losses, policy accuracy, value error, learning rate, positions per second, GPU temperature, power and utilisation, VRAM and RAM
- `eval.csv`: held-out loss, top-1 and top-3 accuracy, value error, and a short match against the baseline
- `events.log`: starts, resumes, checkpoints and evaluations in plain text
- `checkpoints/latest.pt`, `checkpoints/best.pt` and the previous copy of each

**Before a long run, stop Windows from sleeping.** The training script asks Windows to stay awake while it runs, but set Settings, System, Power, "Make my device sleep after" to Never (plugged in) to be safe, and do not let the laptop lid or a Windows Update restart end the run.

## Evaluation and export

```powershell
$env:PYTHONPATH = "python"
D:\chess-venv\Scripts\python.exe -m kestrel.evaluate --run runs/rep --depths 1 2 3 4 --games 20
D:\chess-venv\Scripts\python.exe -m kestrel.evaluate --run runs/rep --depths 3 4 5 --games 20 --search-depth 2 --width 6
D:\chess-venv\Scripts\python.exe -m kestrel.export_onnx --run runs/main
D:\chess-venv\Scripts\python.exe python\tests\test_encoding.py data\shards_2026-10_20m_v2\val_000.bin
```

Without `--search-depth` the network plays its single highest scoring legal move. With it, the network looks ahead: at each position it takes its `--width` most likely moves, goes `--search-depth` plies deep, and scores the final positions with its value output. Checkmates and draws (including repetition) found along the way are scored exactly.

Shards from the first data format (44 byte records, without repetition and fifty move fields) still load, and older 19 plane checkpoints still evaluate and export.

Strength against Stockfish (unpack a Stockfish release into `tools/stockfish-19/`, which git ignores):

```powershell
D:\chess-venv\Scripts\python.exe -m kestrel.stockfish_match --run runs/rep --levels 1500 1700 1900 --games 40 --search-depth 3
```

This plays the network against Stockfish with its strength capped at each `UCI_Elo` level, saves every game as PGN under `runs/<run>/stockfish/`, and fits one rating across all levels with a 95% range.

## Licences

The Rust side uses `cozy-chess` (MIT) and `ruzstd` (MIT). The Python side uses PyTorch, NumPy, ONNX and onnxruntime, all permissive. No GPL chess library is used. Lichess game data is CC0. This repository has no licence file yet.
