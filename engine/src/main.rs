use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use kestrel::game::{ascii_board, Game, Outcome};
use kestrel::prep::{self, PrepConfig};
use kestrel::search::Baseline;
use kestrel::{perft, uci};
use cozy_chess::{Board, Color};

const USAGE: &str = "\
usage: kestrel <command> [options]

commands:
  uci   [--depth N] [--seed N]            run the baseline as a UCI engine on stdin/stdout
  play  [--depth N] [--color white|black] play against the baseline in the terminal
  perft <depth> [--fen FEN]               count leaf nodes for a position
  bench [--depth N]                       time the baseline on a few positions
  prep  --input PATH_OR_URL [options]     turn a Lichess PGN (.pgn or .pgn.zst) into training shards
        --out DIR (data/shards) --positions N (4000000) --min-elo N (1800)
        --min-seconds N (180) --min-plies N (20) --skip-plies N (8)
        --common-plies N (24) --common-cap N (20, 0 disables)
        --shards N (8) --val-every N (50) --seed N (1)";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().cloned() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let opts = parse_opts(&args[1..]);
    let get = |key: &str, default: u64| -> u64 {
        opts.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    };

    match command.as_str() {
        "uci" => uci::run(get("depth", 3) as u32, get("seed", 1)),
        "play" => play(get("depth", 3) as u32, opts.get("color").map(String::as_str) != Some("black")),
        "perft" => {
            let depth: u32 = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
            let board = match opts.get("fen") {
                Some(fen) => Board::from_fen(fen, false).expect("bad fen"),
                None => Board::default(),
            };
            let start = Instant::now();
            let nodes = perft(&board, depth);
            let secs = start.elapsed().as_secs_f64();
            println!("perft {depth}: {nodes} nodes in {secs:.2}s ({:.1} Mnps)", nodes as f64 / secs / 1e6);
        }
        "bench" => bench(get("depth", 4) as u32),
        "prep" => {
            let Some(input) = opts.get("input") else {
                eprintln!("prep needs --input");
                std::process::exit(2);
            };
            let d = PrepConfig::default();
            let cfg = PrepConfig {
                input: input.clone(),
                out_dir: opts.get("out").map(PathBuf::from).unwrap_or(d.out_dir),
                min_elo: get("min-elo", d.min_elo as u64) as u32,
                min_seconds: get("min-seconds", d.min_seconds as u64) as u32,
                min_plies: get("min-plies", d.min_plies as u64) as u32,
                skip_plies: get("skip-plies", d.skip_plies as u64) as u32,
                positions: get("positions", d.positions),
                shards: get("shards", d.shards as u64).max(1) as u32,
                val_every: get("val-every", d.val_every),
                common_plies: get("common-plies", d.common_plies as u64) as u32,
                common_cap: get("common-cap", d.common_cap as u64) as u32,
                seed: get("seed", d.seed),
            };
            eprintln!("[prep] {cfg:?}");
            if let Err(e) = prep::run(&cfg) {
                eprintln!("prep failed: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn parse_opts(args: &[String]) -> HashMap<String, String> {
    let mut opts = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            let value = args.get(i + 1).filter(|v| !v.starts_with("--")).cloned();
            if value.is_some() {
                i += 1;
            }
            opts.insert(key.to_string(), value.unwrap_or_else(|| "true".to_string()));
        }
        i += 1;
    }
    opts
}

fn play(depth: u32, human_white: bool) {
    let mut game = Game::startpos();
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let mut engine = Baseline::new(depth, seed);
    let human = if human_white { Color::White } else { Color::Black };
    let stdin = io::stdin();
    println!("Baseline depth {depth}. Enter moves as SAN (Nf3) or UCI (g1f3), 'quit' to stop.");
    loop {
        let outcome = game.outcome();
        if outcome != Outcome::Ongoing {
            print!("{}", ascii_board(&game.board));
            println!("game over: {}", outcome.name());
            return;
        }
        if game.board.side_to_move() == human {
            print!("{}your move: ", ascii_board(&game.board));
            let _ = io::stdout().flush();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            let text = line.trim();
            if text == "quit" {
                return;
            }
            if game.play_san(text).is_err() && game.play_uci(text).is_err() {
                println!("not a legal move: {text}");
            }
        } else {
            let result = engine.search(&game.board);
            let mv = result.best.expect("ongoing game has a move");
            println!("baseline plays {} (score {} cp, {} nodes)", game.uci(mv), result.score, result.nodes);
            game.play(mv).expect("search returns legal moves");
        }
    }
}

fn bench(depth: u32) {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    for d in 1..=depth {
        let mut engine = Baseline::new(d, 1);
        let start = Instant::now();
        let mut nodes = 0;
        for fen in fens {
            nodes += engine.search(&Board::from_fen(fen, false).unwrap()).nodes;
        }
        let secs = start.elapsed().as_secs_f64();
        println!("depth {d}: {nodes} nodes, {:.3}s for {} positions ({:.2}s per move)", secs, fens.len(), secs / fens.len() as f64);
    }
}
