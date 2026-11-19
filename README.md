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
- UCI options: `Depth` (the deepest search allowed, 0 to 8, default 4), `Width` (moves searched at each step, default 6) and `Randomness` (in thousandths; root moves this close to the best are picked at random, default 10, so it does not repeat the same game).
- Time management: from `wtime`/`btime`/`winc`/`binc` it allows about a thirtieth of the remaining time plus most of the increment. With a clock it searches depth 2, then 3 and so on up to `Depth`, keeps the result of the deepest search that finished, and stops at the deadline even in the middle of a search. It does not start another depth that would probably not finish, judging by how much longer each depth took than the one before. `go depth N` searches exactly that depth.
- Search: negamax with alpha-beta pruning over the policy's top moves, with value head scores at the leaves. Moves are tried in policy order, except that the best move found by the previous, shallower search goes first, which prunes more. Pruning returns exactly the same move and score as searching the whole tree (checked on 260 positions at depths 2 to 5). Network outputs are cached by position, so transpositions, the deepening steps and the next move's search reuse them.
- Forcing moves: up to 3 captures or promotions outside the policy's top moves are always searched, and a leaf reached by a capture, promotion or pawn push to the seventh rank keeps following forcing replies (up to 4 at each step, 4 extra plies at most) until the position is quiet, standing pat on the value head. This roughly doubles the positions searched at a given depth, but won 22-4-14 against the search without it at 0.5 s per move, reaching depth 3.8 on average against 4.2.
- Endgame tuning: when one side is at least 4 points of material ahead and the other has at most a rook's worth left, leaf scores get a small mop-up bonus (up to 0.05) for driving the losing king to the edge and bringing the winning king closer, and the losing side is pushed the other way. When the root position is clearly winning (value above 0.5), draws score -0.1 for Kestrel, so it avoids repetitions and the fifty move rule. In tests it mated a lone king about 10% faster (queen 9.0 moves against 9.8, rook 14.2 against 16.2), and scored 20-5-15 against the untuned search over 40 games at depth 4.

Speed on an i7-12700KF with 2 threads, averaged over 12 positions: depth 4 takes about 0.15 s per move, depth 5 about 0.45 s, depth 6 about 1.1 s and depth 7 about 3.3 s (up to 6 s). So depth 5 fits 1+0 bullet, depth 6 fits blitz and depth 7 fits rapid. Without pruning depth 5 took 3.3 s and depth 6 about 21 s. ONNX and PyTorch choose the same move on 100% of 5,000 test positions (`python/tests/test_features.py`).

## Lichess bot

`python -m kestrel.lichess_bot` plays Kestrel on Lichess through the Bot API. It uses only the Python standard library plus the same `numpy` and `onnxruntime` as the UCI engine, and reads all settings from a TOML file that can live anywhere, so the token never sits in the repo.

### One time setup

1. Create a new Lichess account for the bot. It must never play a game before the upgrade.
2. Signed in as the bot, create an API token with the "Play games with the bot API" (`bot:play`) scope at lichess.org/account/oauth/token/create. Save it in a text file on its own, for example `C:\Users\<you>\.kestrel\token.txt`.
3. Upgrade the account to a bot. **This cannot be undone**, and the account can only ever play as a bot afterwards:
   ```powershell
   curl.exe -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <token>"
   ```
4. Copy `bot.example.toml` to a folder of your choice, next to the token file, along with the exported `model.onnx`.

### Config file

The bot looks for the config in this order: `--config <path>`, the `KESTREL_BOT_CONFIG` environment variable, then `%APPDATA%\kestrel\bot.toml` on Windows or `~/.config/kestrel/bot.toml` on Linux. Relative paths inside it are resolved from the config file's own folder.

| Section | Setting | Meaning |
|---|---|---|
| `lichess` | `token_file` | file holding only the token. `LICHESS_BOT_TOKEN` in the environment overrides it |
| `lichess` | `base_url` | `https://lichess.org` |
| `engine` | `model`, `threads` | ONNX model and CPU threads |
| `engine` | `max_depth`, `width` | deepest search, up to 8; with a clock it searches as deep as the time allows up to this (on the Stockfish scale 4 is about 1975, 5 about 2160 and 6 about 2360) and moves searched at each step |
| `engine` | `randomness` | thousandths of a point; root moves this close to the best are picked at random |
| `engine` | `reserve_seconds` | seconds kept back from the clock for network lag |
| `engine` | `ponder` | search on the opponent's time, so whatever they reply, much of the next search is already cached. After 8 seconds of pondering the next search is about twice as fast. Uses the CPU for the whole game |
| `challenges` | `speeds`, `variants` | any of `ultraBullet`, `bullet`, `blitz`, `rapid`, `classical`, `correspondence`; variants by Lichess key |
| `challenges` | `min_initial_seconds`, `max_initial_seconds`, `min_increment_seconds` | clock limits |
| `challenges` | `accept_rated`, `accept_casual`, `accept_bots`, `accept_humans`, `max_games` | who and how many |
| `challenges` | `allow_users`, `block_users` | if `allow_users` is not empty, only those usernames may challenge it. Use this with your own username while testing |
| `chat` | `greeting`, `goodbye` | messages sent to the opponent at the start and end of a game, empty to stay quiet |
| `matchmaking` | `enabled`, `idle_minutes` | when on, after this many minutes without a game it challenges a random online bot |
| `matchmaking` | `time_controls`, `rated` | clocks to offer, such as `"5+3"`, picked at random each time; each must be one of the accepted `speeds` |
| `matchmaking` | `max_rating_diff`, `cooldown_minutes` | only bots within this many points of Kestrel's rating at that speed (any rating while Kestrel's is provisional), and not the same bot again within the cooldown. Blocked users are never challenged |

Declined challenges get the matching Lichess reason (`variant`, `tooFast`, `tooSlow`, `timeControl`, `noBot`, `onlyBot`, `rated`, `casual`, `later`).

### Running

```powershell
$env:PYTHONPATH = "python"
D:\chess-venv\Scripts\python.exe -m kestrel.lichess_bot --config C:\Users\<you>\.kestrel\bot.toml
```

It checks that the account has the BOT title, then listens for challenges. Games in progress are picked up again after a restart, and dropped connections reconnect on their own. Ctrl+C stops it.

`python/tests/test_lichess_bot.py` runs the whole bot against a local mock of the Lichess API (challenges declined and accepted, a full game, chat, and a check that the token is never logged), so it can be tested without an account.

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

The Lichess evaluation database (`lichess_db_eval.jsonl.zst`, 22.5 GB, about 418 million positions with Stockfish analysis, CC0) adds positions with an engine score:

```powershell
.\engine\target\release\kestrel.exe prep-evals --input D:\chess-data\lichess_db_eval.jsonl.zst `
  --out D:\chess-data\evals --positions 30000000 --keep-percent 8
```

It keeps a random `--keep-percent` of lines, takes each position's deepest analysis, and stores Stockfish's best move as the move and its score (converted to an expected score with Lichess's win model, from the side to move's view) in the record. The scores in the file are from White's point of view.

Records are 48 bytes (format 3): the board as 32 bytes of 4-bit piece codes, side to move, castling rights, en passant file, result, move, ply, both ratings, repetition count, halfmove clock, a flags byte (1 means the next byte holds an engine score) and the engine's expected score from 0 to 255. Format 2 shards have zeros in the last two bytes and load unchanged.

## Training

One command starts a run, and the same command resumes it from the latest checkpoint:

```powershell
.\scripts\train.ps1 -Run runs/rep -Data data/shards_2026-10_20m_v2 --max-hours 2
```

- Any `train.py` option can be added after the script options, for example `--max-steps 300000 --eval-every 10000 --patience 5`.
- `-Data` takes several folders, for example `-Data D:\chess-data\games_2026-10, D:\chess-data\evals`. Data larger than `--gpu-data-gb` (2.5) stays in pinned system memory and batches are copied to the GPU, at the same speed.
- Records with an engine score train the value head towards that score, spread into win, draw and loss chances of q², 2q(1-q) and (1-q)² for an expected score q. Game records use the result. Both train the policy on their move (the human's move or Stockfish's).
- `--endgame-weight 3` samples positions with `--endgame-pieces` (12) or fewer pieces, kings included, three times as often.
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

Results so far for the current model, with Stockfish at 200 ms per move:

| Kestrel setting | Seconds per move now | Estimate (95% range) |
|---|---|---|
| Top move only | under 0.01 | about 1410 |
| Depth 3 | 0.05 | 1693 (1639 to 1748) |
| Depth 4 | 0.15 | 1975 (1906 to 2044) |
| Depth 5 | 0.45 | 2158 (2060 to 2254) |
| Depth 6 | 1.1 | 2357 (2260 to 2455) |

These are on Stockfish's `UCI_Elo` scale, which is not the same as a Lichess rating. Depths 3 to 5 were measured before alpha-beta pruning was added, but pruning picks the same moves, so those ratings still apply at the new speeds. Each depth used 60 games over three Stockfish levels (depth 3 used more).

## Licences

The Rust side uses `cozy-chess` (MIT) and `ruzstd` (MIT). The Python side uses PyTorch, NumPy, ONNX and onnxruntime, all permissive. No GPL chess library is used. Lichess game data is CC0. This repository has no licence file yet.
