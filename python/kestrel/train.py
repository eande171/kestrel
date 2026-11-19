"""Supervised training that can be started, stopped and resumed with one command."""

import argparse
import csv
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import encode_planes, encode_targets
from .features import FLAG_ENGINE_EVAL, RECORD_BYTES, read_records, record_width
from .inference import TorchEvaluator
from .match import play_match
from .model import ChessNet, checkpoint_config, count_parameters

LOG_FIELDS = [
    "wall_time", "step", "epoch", "loss", "policy_loss", "value_loss", "policy_acc", "value_mse",
    "lr", "positions_per_sec", "gpu_temp_c", "gpu_power_w", "gpu_util_pct", "vram_alloc_mb",
    "vram_peak_mb", "ram_mb",
]
EVAL_FIELDS = [
    "wall_time", "step", "epoch", "train_hours", "val_loss", "val_policy_loss", "val_value_loss",
    "top1", "top3", "value_mse", "match_depth", "match_games", "wins", "draws", "losses", "score",
    "is_best",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", default=["data/shards_2026-10"], help="one or more shard folders")
    p.add_argument("--endgame-weight", type=float, default=1.0, help="how many times as often to sample endgames")
    p.add_argument("--endgame-pieces", type=int, default=12, help="positions with this many pieces or fewer, kings included")
    p.add_argument("--gpu-data-gb", type=float, default=2.5, help="keep training data on the GPU up to this size")
    p.add_argument("--run", default="runs/main")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--max-hours", type=float, default=0.0, help="cumulative training hours, 0 for no limit")
    p.add_argument("--patience", type=int, default=4, help="evals without held out loss improvement before stopping")
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--match-depth", type=int, default=2)
    p.add_argument("--match-games", type=int, default=10)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-minutes", type=float, default=10.0)
    p.add_argument("--duty", type=float, default=1.0, help="fraction of time the GPU is busy, below 1 for quieter running")
    p.add_argument("--quiet", action="store_true", help="shortcut for --duty 0.5 and --batch-size 512")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-keep-awake", action="store_true")
    args = p.parse_args(argv)
    if args.quiet:
        args.duty = min(args.duty, 0.5)
        args.batch_size = min(args.batch_size, 512)
    return args


# System Telemetry
def gpu_stats() -> tuple[str, str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        temp, power, util = (s.strip() for s in out.split(","))
        return temp, power, util
    except Exception:
        return "", "", ""


def process_ram_mb() -> float:
    if sys.platform != "win32":
        return 0.0

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6


def keep_awake(enable: bool) -> None:
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | es_system_required if enable else es_continuous
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def load_shards(data_dirs: list[Path], prefix: str) -> np.ndarray:
    """Returns every shard with the prefix from all data folders as one array, read into place without a second copy."""
    files = [f for d in data_dirs for f in sorted(d.glob(f"{prefix}_*.bin"))]
    if not files:
        raise SystemExit(f"no {prefix} shards in {', '.join(map(str, data_dirs))}")
    rows = sum(f.stat().st_size // record_width(f.parent) for f in files)
    out = np.empty((rows, RECORD_BYTES), dtype=np.uint8)
    total = 0
    for f in files:
        part = read_records(f)
        out[total : total + len(part)] = part
        total += len(part)
    return out


def piece_counts(rows: np.ndarray, chunk: int = 1 << 22) -> np.ndarray:
    """Returns how many pieces, kings included, each record has on the board."""
    counts = np.empty(len(rows), dtype=np.uint8)
    for start in range(0, len(rows), chunk):
        board = rows[start : start + chunk, :32]
        counts[start : start + chunk] = ((board & 15) != 0).sum(1) + ((board >> 4) != 0).sum(1)
    return counts


def epoch_order(counts: np.ndarray, endgame_pieces: int, endgame_weight: float, seed: int) -> torch.Tensor:
    """Returns a shuffled index order for one epoch, repeating endgame records so they appear endgame_weight times as often."""
    rng = np.random.default_rng(seed)
    endgames = np.flatnonzero(counts <= endgame_pieces)
    extra = [endgames] * (int(endgame_weight) - 1)
    fraction = endgame_weight - int(endgame_weight)
    if fraction > 0:
        extra.append(endgames[rng.random(len(endgames)) < fraction])
    order = np.concatenate([np.arange(len(counts)), *extra]) if extra else np.arange(len(counts))
    rng.shuffle(order)
    return torch.from_numpy(order)


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def value_targets(raw: torch.Tensor) -> torch.Tensor:
    """Returns loss, draw and win probabilities: the game result, or a spread around an engine's expected score."""
    _, result = encode_targets(raw)
    hard = F.one_hot(result, 3).float()
    q = raw[:, 47].float() / 255.0
    soft = torch.stack([(1 - q) ** 2, 2 * q * (1 - q), q ** 2], dim=1)
    has_eval = (raw[:, 46] & FLAG_ENGINE_EVAL).bool()[:, None]
    return torch.where(has_eval, soft, hard)


def losses(model, raw, value_weight):
    planes = encode_planes(raw, model.in_planes)
    target_idx, _ = encode_targets(raw)
    target_wdl = value_targets(raw)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy, value = model(planes)
    policy = policy.float()
    value = value.float()
    policy_loss = F.cross_entropy(policy, target_idx)
    value_loss = F.cross_entropy(value, target_wdl)
    probs = value.softmax(-1)
    expected = probs[:, 2] - probs[:, 0]
    value_mse = ((expected - (target_wdl[:, 2] - target_wdl[:, 0])) ** 2).mean()
    return policy_loss + value_weight * value_loss, policy_loss, value_loss, policy, target_idx, value_mse


@torch.no_grad()
def evaluate_heldout(model, val: torch.Tensor, value_weight: float, batch: int = 4096) -> dict:
    model.eval()
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "top1": 0.0, "top3": 0.0, "mse": 0.0}
    n = 0
    for i in range(0, len(val), batch):
        raw = val[i : i + batch]
        loss, pl, vl, policy, target, mse = losses(model, raw, value_weight)
        k = len(raw)
        top3 = policy.topk(3, dim=1).indices
        totals["loss"] += loss.item() * k
        totals["policy"] += pl.item() * k
        totals["value"] += vl.item() * k
        totals["top1"] += (top3[:, 0] == target).float().sum().item()
        totals["top3"] += (top3 == target[:, None]).any(dim=1).float().sum().item()
        totals["mse"] += mse.item() * k
        n += k
    model.train()
    return {key: value / n for key, value in totals.items()}


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    if path.exists():
        os.replace(path, path.with_suffix(".prev.pt"))
    os.replace(tmp, path)


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def log_event(run: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with open(run / "events.log", "a") as f:
        f.write(line + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run = Path(args.run)
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    stop_file = run / "STOP"
    if stop_file.exists():
        stop_file.unlink()

    torch.manual_seed(args.seed)
    model = ChessNet(args.blocks, args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state = {"step": 0, "epoch": 0, "epoch_pos": 0, "best_val": float("inf"), "bad_evals": 0, "train_seconds": 0.0}

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint_config(ckpt) != model.config:
            raise SystemExit(f"checkpoint model config {ckpt['model_config']} differs from {model.config}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        log_event(run, f"resumed from step {state['step']} epoch {state['epoch']}")
    else:
        log_event(run, "starting new run")
    with open(run / "config.json", "w") as f:
        json.dump({**vars(args), "model_config": model.config, "parameters": count_parameters(model)}, f, indent=2)

    data_dirs = [Path(d) for d in args.data]
    train_np = load_shards(data_dirs, "train")
    counts = piece_counts(train_np)
    on_gpu = train_np.nbytes <= args.gpu_data_gb * 1e9
    train = torch.from_numpy(train_np)
    train = train.to(device) if on_gpu else train.pin_memory()
    del train_np
    val = torch.from_numpy(load_shards(data_dirs, "val")).to(device)
    endgames = int((counts <= args.endgame_pieces).sum())
    log_event(
        run,
        f"data train {len(train):,} val {len(val):,} positions ({endgames:,} with {args.endgame_pieces} pieces or "
        f"fewer, weighted x{args.endgame_weight}), kept on {'GPU' if on_gpu else 'pinned CPU memory'}, "
        f"model {count_parameters(model):,} parameters, batch {args.batch_size}, duty {args.duty}",
    )

    def save(reason: str) -> None:
        atomic_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
             "model_config": model.config, "args": vars(args)},
            latest_path,
        )
        log_event(run, f"checkpoint at step {state['step']} ({reason})")

    def finished() -> str | None:
        if state["step"] >= args.max_steps:
            return f"reached max steps {args.max_steps}"
        if args.max_hours > 0 and state["train_seconds"] >= args.max_hours * 3600:
            return f"reached max hours {args.max_hours}"
        if state["bad_evals"] >= args.patience:
            return f"held out loss did not improve for {args.patience} evals"
        return None

    keep_awake(not args.no_keep_awake)
    model.train()
    perm = None
    window = {"loss": 0.0, "policy": 0.0, "value": 0.0, "acc": 0.0, "mse": 0.0, "n": 0}
    window_start = time.time()
    last_ckpt = time.time()
    reason = finished()
    try:
        while reason is None:
            if perm is not None and state["epoch_pos"] >= len(perm):
                state["epoch"] += 1
                state["epoch_pos"] = 0
                perm = None
            if perm is None:
                perm = epoch_order(counts, args.endgame_pieces, args.endgame_weight,
                                   args.seed * 100_003 + state["epoch"])
                perm = perm.to(device) if on_gpu else perm

            step_start = time.time()
            for group in optimizer.param_groups:
                group["lr"] = lr_at(state["step"], args)
            batch = train[perm[state["epoch_pos"] : state["epoch_pos"] + args.batch_size]]
            if not on_gpu:
                batch = batch.to(device, non_blocking=True)
            loss, pl, vl, policy, target, mse = losses(model, batch, args.value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            state["step"] += 1
            state["epoch_pos"] += len(batch)

            window["loss"] += loss.detach()
            window["policy"] += pl.detach()
            window["value"] += vl.detach()
            window["acc"] += (policy.argmax(1) == target).float().mean().detach()
            window["mse"] += mse.detach()
            window["n"] += 1

            if args.duty < 1.0:
                torch.cuda.synchronize()
                busy = time.time() - step_start
                time.sleep(busy * (1.0 - args.duty) / max(args.duty, 0.05))
            state["train_seconds"] += time.time() - step_start

            if state["step"] % args.log_every == 0:
                elapsed = time.time() - window_start
                n = window["n"]
                temp, power, util = gpu_stats()
                row = {
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": state["step"],
                    "epoch": round(state["epoch"] + state["epoch_pos"] / len(perm), 3),
                    "loss": round(float(window["loss"]) / n, 4),
                    "policy_loss": round(float(window["policy"]) / n, 4),
                    "value_loss": round(float(window["value"]) / n, 4),
                    "policy_acc": round(float(window["acc"]) / n, 4),
                    "value_mse": round(float(window["mse"]) / n, 4),
                    "lr": f"{lr_at(state['step'], args):.6f}",
                    "positions_per_sec": round(n * args.batch_size / elapsed),
                    "gpu_temp_c": temp,
                    "gpu_power_w": power,
                    "gpu_util_pct": util,
                    "vram_alloc_mb": round(torch.cuda.memory_allocated() / 1e6),
                    "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
                    "ram_mb": round(process_ram_mb()),
                }
                append_csv(run / "log.csv", LOG_FIELDS, row)
                print(
                    f"step {row['step']} epoch {row['epoch']} loss {row['loss']} acc {row['policy_acc']} "
                    f"lr {row['lr']} {row['positions_per_sec']} pos/s gpu {temp}C {power}W",
                    flush=True,
                )
                window = {k: 0.0 for k in window}
                window["n"] = 0
                window_start = time.time()

            if state["step"] % args.eval_every == 0:
                metrics = evaluate_heldout(model, val, args.value_weight)
                model.eval()
                match = play_match(TorchEvaluator(model, device), args.match_depth, args.match_games, seed=state["step"])
                model.train()
                is_best = metrics["loss"] < state["best_val"] - 1e-4
                if is_best:
                    state["best_val"] = metrics["loss"]
                    state["bad_evals"] = 0
                    atomic_save({"model": model.state_dict(), "model_config": model.config, "step": state["step"],
                                 "metrics": metrics}, best_path)
                else:
                    state["bad_evals"] += 1
                append_csv(run / "eval.csv", EVAL_FIELDS, {
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": state["step"],
                    "epoch": round(state["epoch"] + state["epoch_pos"] / len(perm), 3),
                    "train_hours": round(state["train_seconds"] / 3600, 3),
                    "val_loss": round(metrics["loss"], 4),
                    "val_policy_loss": round(metrics["policy"], 4),
                    "val_value_loss": round(metrics["value"], 4),
                    "top1": round(metrics["top1"], 4),
                    "top3": round(metrics["top3"], 4),
                    "value_mse": round(metrics["mse"], 4),
                    "match_depth": args.match_depth,
                    "match_games": match.games,
                    "wins": match.wins,
                    "draws": match.draws,
                    "losses": match.losses,
                    "score": round(match.score, 3),
                    "is_best": int(is_best),
                })
                log_event(
                    run,
                    f"eval step {state['step']} val_loss {metrics['loss']:.4f} top1 {metrics['top1']:.3f} "
                    f"top3 {metrics['top3']:.3f} vs depth {args.match_depth}: +{match.wins} ={match.draws} "
                    f"-{match.losses}{' best' if is_best else ''}",
                )
                save("eval")
                last_ckpt = time.time()

            if time.time() - last_ckpt > args.ckpt_minutes * 60:
                save("timer")
                last_ckpt = time.time()
            if stop_file.exists():
                save("stop file")
                stop_file.unlink()
                log_event(run, "stopped by STOP file, run again to resume")
                return 0
            reason = finished()
    except KeyboardInterrupt:
        save("interrupted")
        log_event(run, "interrupted, run again to resume")
        return 0
    finally:
        keep_awake(False)

    save("finished")
    (run / "DONE").write_text(reason + "\n")
    log_event(run, f"finished: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
