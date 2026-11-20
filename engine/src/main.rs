use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use kestrel::game::{ascii_board, Game, Outcome};
use kestrel::evals::{self, EvalConfig};
use kestrel::prep::{self, PrepConfig};
use kestrel::search::Baseline;
use kestrel::{game, net, netsearch, perft, uci};
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
        --shards N (8) --val-every N (50) --seed N (1)
  prep-evals --input PATH_OR_URL [options] turn the Lichess evaluation database (.jsonl.zst) into shards
        --out DIR (data/evals) --positions N (30000000) --keep-percent X (8)
        --skip-percent X (0, skips the lines an earlier run with the same seed kept at this percent)
        --shards N (8) --val-every N (50) --seed N (1)
  net-eval --weights FILE --fens FILE --out FILE   write wdl and policy logits for each \"fen<TAB>repetitions\" line
  net-bench --weights FILE                          time the network on a few positions
  net-match --weights FILE [--games N] [--seconds S] [--a no_check_extensions,no_extensions,no_endgame_tuning]
        play a search with the listed settings against the default one at equal time
  net-search --weights FILE --fens FILE [--depth N] [--seconds S]
        search each \"fen<TAB>moves\" line and print the move, score, depth reached and positions expanded";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().cloned() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let opts = parse_opts(&args[1..]);

    match command.as_str() {
        "uci" => uci::run(get_opt(&opts, "depth", 3), get_opt(&opts, "seed", 1)),
        "play" => play(get_opt(&opts, "depth", 3), opts.get("color").map(String::as_str) != Some("black")),
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
        "bench" => bench(get_opt(&opts, "depth", 4)),
        "prep" => {
            let Some(input) = opts.get("input") else {
                eprintln!("prep needs --input");
                std::process::exit(2);
            };
            let d = PrepConfig::default();
            let cfg = PrepConfig {
                input: input.clone(),
                out_dir: opts.get("out").map(PathBuf::from).unwrap_or(d.out_dir),
                min_elo: get_opt(&opts, "min-elo", d.min_elo),
                min_seconds: get_opt(&opts, "min-seconds", d.min_seconds),
                min_plies: get_opt(&opts, "min-plies", d.min_plies),
                skip_plies: get_opt(&opts, "skip-plies", d.skip_plies),
                positions: get_opt(&opts, "positions", d.positions),
                shards: get_opt(&opts, "shards", d.shards).max(1),
                val_every: get_opt(&opts, "val-every", d.val_every),
                common_plies: get_opt(&opts, "common-plies", d.common_plies),
                common_cap: get_opt(&opts, "common-cap", d.common_cap),
                seed: get_opt(&opts, "seed", d.seed),
            };
            eprintln!("[prep] {cfg:?}");
            if let Err(e) = prep::run(&cfg) {
                eprintln!("prep failed: {e}");
                std::process::exit(1);
            }
        }
        "net-eval" => {
            let (Some(weights), Some(fens), Some(out)) = (opts.get("weights"), opts.get("fens"), opts.get("out")) else {
                eprintln!("net-eval needs --weights, --fens and --out");
                std::process::exit(2);
            };
            let network = load_network(weights);
            let all: Vec<usize> = (0..net::POLICY_SIZE).collect();
            let mut bytes = Vec::new();
            for line in std::fs::read_to_string(fens).expect("cannot read fens").lines() {
                let (fen, reps) = line.split_once('\t').unwrap_or((line, "0"));
                let board = Board::from_fen(fen, false).expect("bad fen");
                let (wdl, policy) = network.evaluate(&net::encode(&board, reps.trim().parse().unwrap_or(0)), &all);
                for v in wdl.iter().chain(&policy) {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(out, bytes).expect("cannot write output");
        }
        "net-search" => {
            let (Some(weights), Some(fens)) = (opts.get("weights"), opts.get("fens")) else {
                eprintln!("net-search needs --weights and --fens");
                std::process::exit(2);
            };
            let network = std::rc::Rc::new(load_network(weights));
            let clock = Instant::now();
            let mut search = netsearch::NetSearch::new(network, move || clock.elapsed().as_secs_f64(), 0.0, 1);
            let depth = get_opt(&opts, "depth", 4u32);
            let seconds: Option<f64> = opts.get("seconds").and_then(|s| s.parse().ok());
            for line in std::fs::read_to_string(fens).expect("cannot read fens").lines() {
                let (fen, moves) = line.split_once('\t').unwrap_or((line, ""));
                let game = Game::from_position(fen, moves).expect("bad position");
                let start = Instant::now();
                let best = search.choose(&game, depth, seconds);
                let uci = best.map(|mv| game.uci(mv)).unwrap_or_else(|| "none".into());
                println!("{uci} {:.6} {} {} {:.3} {}", search.last_score, search.completed_depth, search.nodes,
                         start.elapsed().as_secs_f64(), search.evaluations);
            }
        }
        "net-match" => {
            let Some(weights) = opts.get("weights") else {
                eprintln!("net-match needs --weights");
                std::process::exit(2);
            };
            let settings: Vec<String> = opts.get("a").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default();
            net_match(std::rc::Rc::new(load_network(weights)), get_opt(&opts, "games", 40u32), get_opt(&opts, "seconds", 1.0f64), &settings);
        }
        "net-bench" => {
            let Some(weights) = opts.get("weights") else {
                eprintln!("net-bench needs --weights");
                std::process::exit(2);
            };
            net_bench(&load_network(weights));
        }
        "prep-evals" => {
            let Some(input) = opts.get("input") else {
                eprintln!("prep-evals needs --input");
                std::process::exit(2);
            };
            let d = EvalConfig::default();
            let cfg = EvalConfig {
                input: input.clone(),
                out_dir: opts.get("out").map(PathBuf::from).unwrap_or(d.out_dir),
                positions: get_opt(&opts, "positions", d.positions),
                keep_percent: get_opt(&opts, "keep-percent", d.keep_percent),
                skip_percent: get_opt(&opts, "skip-percent", d.skip_percent),
                shards: get_opt(&opts, "shards", d.shards).max(1),
                val_every: get_opt(&opts, "val-every", d.val_every),
                seed: get_opt(&opts, "seed", d.seed),
            };
            eprintln!("[evals] {cfg:?}");
            if let Err(e) = evals::run(&cfg) {
                eprintln!("prep-evals failed: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn get_opt<T: FromStr>(opts: &HashMap<String, String>, key: &str, default: T) -> T {
    opts.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
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
            let result = engine.search(&game.board, depth);
            let mv = result.best.expect("ongoing game has a move");
            println!("baseline plays {} (score {} cp, {} nodes)", game.uci(mv), result.score, result.nodes);
            game.play(mv).expect("search returns legal moves");
        }
    }
}

fn load_network(path: &str) -> net::Network {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    net::Network::load(&bytes).unwrap_or_else(|e| panic!("{e}"))
}

// Search Variant Match
const OPENINGS: [&str; 10] = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
];

fn net_match(network: std::rc::Rc<net::Network>, games: u32, seconds: f64, settings: &[String]) {
    let clock = Instant::now();
    let make = |seed| netsearch::NetSearch::new(network.clone(), move || clock.elapsed().as_secs_f64(), 0.01, seed);
    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    let mut depths = [0u64, 0u64];
    let mut moves_made = [0u64, 0u64];
    for g in 0..games {
        let mut sides = [make(2 * g as u64 + 1), make(2 * g as u64 + 2)];
        for name in settings {
            match name.as_str() {
                "no_check_extensions" => sides[0].check_extensions = 0,
                "no_extensions" => sides[0].extensions = false,
                "no_endgame_tuning" => sides[0].endgame_tuning = false,
                "" => {}
                other => panic!("unknown setting {other}"),
            }
        }
        let mut game = Game::from_position("startpos", OPENINGS[(g as usize / 2) % OPENINGS.len()]).unwrap();
        let a_is_white = g % 2 == 0;
        while game.outcome() == Outcome::Ongoing && game.plies < 300 {
            let white = game.board.side_to_move() == Color::White;
            let side = if white == a_is_white { 0 } else { 1 };
            let mv = sides[side].choose(&game, 10, Some(seconds)).expect("no move");
            depths[side] += sides[side].completed_depth as u64;
            moves_made[side] += 1;
            game.play(mv).unwrap();
        }
        let outcome = game.outcome();
        match outcome {
            Outcome::Checkmate { winner } if (winner == Color::White) == a_is_white => wins += 1,
            Outcome::Checkmate { .. } => losses += 1,
            _ => draws += 1,
        }
        eprintln!("game {} {}: A {wins}-{draws}-{losses}", g + 1, outcome.name());
    }
    let score = (wins as f64 + draws as f64 / 2.0) / games as f64;
    println!("A ({}) {wins}-{draws}-{losses} against the default search at {seconds}s per move, score {score:.3}, \
              average depth {:.2} vs {:.2}", settings.join(","), depths[0] as f64 / moves_made[0].max(1) as f64,
             depths[1] as f64 / moves_made[1].max(1) as f64);
}

fn net_bench(network: &net::Network) {
    let board = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", false).unwrap();
    let planes = net::encode(&board, 0);
    let legal: Vec<usize> = game::legal_moves(&board).into_iter().map(|mv| net::policy_index(&board, mv)).collect();
    for (label, policy) in [("value only", false), ("value and legal move policy", true)] {
        let runs = 2000;
        let start = Instant::now();
        let mut total = 0.0;
        for _ in 0..runs {
            let wdl = if policy { network.evaluate(&planes, &legal).0 } else { network.wdl(&planes) };
            total += net::expected_score(wdl);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        println!("{label}: {ms:.3} ms per position (checksum {total:.3})");
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
            nodes += engine.search(&Board::from_fen(fen, false).unwrap(), d).nodes;
        }
        let secs = start.elapsed().as_secs_f64();
        println!("depth {d}: {nodes} nodes, {:.3}s for {} positions ({:.2}s per move)", secs, fens.len(), secs / fens.len() as f64);
    }
}
