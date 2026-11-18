# chess-ai

A neural network chess AI, built mainly as a learning project, with a decent opponent to play against as a bonus.

## Goals

Long term, the plan is a small network plus a short search, compiled to WebAssembly so it runs in the browser on a static website. It should also do "opponent modelling": learn a player's recurring mistakes from their own games and help them improve. Self-play, WebAssembly, the website and opponent modelling are later milestones.

Milestone 1 (this repository so far):

- a data pipeline that turns a month of Lichess games into training positions
- a small supervised policy and value network trained on those positions
- a training setup that can run unattended and resume after a stop or crash
- a deliberately simple baseline engine to measure the network against

See `NOTES.md` for the decisions behind this and `REPORT.md` for the results.

## Layout

```
engine/          Rust crate: board wrapper, perft tests, baseline engine, UCI loop, PGN to shards pipeline
python/chess_ai/ encoding, model, training, evaluation matches, ONNX export
python/tests/    encoding consistency check against the Rust engine
scripts/         train.ps1 (start or resume training), gpu_power_limit.ps1
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

## Baseline engine

```powershell
.\engine\target\release\chess-ai.exe play --depth 3             # play it in the terminal (SAN or UCI moves)
.\engine\target\release\chess-ai.exe uci --depth 3              # UCI engine on stdin and stdout
.\engine\target\release\chess-ai.exe perft 6                    # move generation check
.\engine\target\release\chess-ai.exe bench --depth 5            # search speed
```

Material only evaluation, fixed depth alpha-beta, capture-only quiescence search and random tie breaking. It does not know about piece placement, king safety or pawn structure, so it is weak on purpose. Depth 1 to 4 all reply in well under a second.

Besides standard UCI commands it accepts `state`, which prints the FEN, the game status (including repetition, fifty moves and insufficient material) and the legal moves. The Python match code uses this so Python never needs its own move generator.

## Data

The prep tool streams a Lichess monthly file over HTTP, decompresses it on the fly, filters games and stops once it has enough positions. The full monthly file is around 30 GB but only the part that is read gets downloaded.

```powershell
.\engine\target\release\chess-ai.exe prep `
  --input https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst `
  --out data/shards_2026-10_20m --positions 20000000 --shards 16 --min-elo 1800
```

Run `chess-ai.exe` with no arguments for all options (rating threshold, minimum time control, plies skipped, opening position cap and so on). A local `.pgn` or `.pgn.zst` file works too.

## Training

One command starts a run, and the same command resumes it from the latest checkpoint:

```powershell
.\scripts\train.ps1 -Run runs/main -Data data/shards_2026-10_20m --max-hours 2
```

- Any `train.py` option can be added after the script options, for example `--max-steps 300000 --eval-every 10000 --patience 5`.
- `-Quiet` halves the GPU duty cycle and caps the batch at 512 (about 137 W instead of 220 W on the 3070 Ti). `--duty 0.7` gives finer control.
- For a hard power cap, run `.\scripts\gpu_power_limit.ps1 200` in an Administrator PowerShell. The 3070 Ti accepts 100 to 310 W. The limit resets on reboot, or set it back with `310`.
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
D:\chess-venv\Scripts\python.exe -m chess_ai.evaluate --run runs/main --depths 1 2 3 4 --games 20
D:\chess-venv\Scripts\python.exe -m chess_ai.export_onnx --run runs/main
D:\chess-venv\Scripts\python.exe python\tests\test_encoding.py data\shards_2026-10_20m\val_000.bin
```

The network plays its single highest scoring legal move with no search.

## Licences

The Rust side uses `cozy-chess` (MIT) and `ruzstd` (MIT). The Python side uses PyTorch, NumPy, ONNX and onnxruntime, all permissive. No GPL chess library is used. Lichess game data is CC0. This repository has no licence file yet.
