use std::io::{self, BufRead, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::game::Game;
use crate::net::Network;
use crate::netsearch::NetSearch;
use crate::uci::parse_position;

const PONDER_SECONDS: f64 = 1e9;

enum Job {
    Search { game: Game, depth: u32, seconds: Option<f64> },
    Randomness(f32),
    Seed(u64),
}

// Network UCI Loop
pub fn run(network: Network, default_depth: u32) {
    let stop = Arc::new(AtomicBool::new(false));
    let (jobs, inbox) = mpsc::channel::<Job>();
    let flag = stop.clone();
    let worker = thread::spawn(move || {
        let clock = Instant::now();
        let now = move || if flag.load(Ordering::Relaxed) { f64::INFINITY } else { clock.elapsed().as_secs_f64() };
        let mut search = NetSearch::new(Rc::new(network), now, 0.0, 1);
        for job in inbox {
            match job {
                Job::Randomness(r) => search.randomness = r,
                Job::Seed(s) => search.seed(s),
                Job::Search { game, depth, seconds } => {
                    let best = search.choose(&game, depth, seconds);
                    let uci = best.map(|mv| game.uci(mv)).unwrap_or_else(|| "0000".into());
                    let mut out = io::stdout().lock();
                    let _ = writeln!(out, "info depth {} score {:.6} nodes {}", search.completed_depth, search.last_score,
                                     search.nodes);
                    let _ = writeln!(out, "bestmove {uci}");
                    let _ = out.flush();
                }
            }
        }
    });

    let mut game = Game::startpos();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => println!("id name Kestrel net\nuciok"),
            Some("isready") => println!("readyok"),
            Some("position") => match parse_position(&tokens) {
                Ok(g) => game = g,
                Err(e) => println!("info string error {e}"),
            },
            Some("setoption") => {
                let value = tokens.last().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                match tokens.get(2).copied() {
                    Some("Randomness") => jobs.send(Job::Randomness(value as f32 / 1000.0)).unwrap(),
                    Some("Seed") => jobs.send(Job::Seed(value as u64)).unwrap(),
                    _ => println!("info string error unknown option"),
                }
            }
            Some("go") => {
                let number = |name: &str| tokens.iter().position(|&t| t == name).and_then(|i| tokens.get(i + 1)?.parse::<f64>().ok());
                let depth = number("depth").map_or(default_depth, |d| d as u32);
                let seconds = match number("movetime") {
                    Some(ms) => Some(ms / 1000.0),
                    None if number("depth").is_some() && !tokens.contains(&"infinite") => None,
                    None => Some(PONDER_SECONDS),
                };
                stop.store(false, Ordering::Relaxed);
                jobs.send(Job::Search { game: game.clone(), depth, seconds }).unwrap();
            }
            Some("stop") => stop.store(true, Ordering::Relaxed),
            Some("quit") => {
                stop.store(true, Ordering::Relaxed);
                break;
            }
            _ => {}
        }
        let _ = io::stdout().flush();
    }
    drop(jobs);
    let _ = worker.join();
}
