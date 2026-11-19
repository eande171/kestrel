use std::io::{self, BufRead, Write};

use cozy_chess::util::{display_san_move, parse_uci_move};

use crate::game::{ascii_board, parse_san, Game};
use crate::search::Baseline;

pub fn run(default_depth: u32, seed: u64) {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut game = Game::startpos();
    let mut engine = Baseline::new(default_depth, seed);

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = tokens.first() else { continue };
        let reply = match command {
            "uci" => format!(
                "id name kestrel baseline\nid author kestrel\n\
                 option name Depth type spin default {default_depth} min 1 max 12\n\
                 option name Seed type spin default {seed} min 0 max 4294967295\n\
                 option name Quiescence type check default true\nuciok"
            ),
            "isready" => "readyok".to_string(),
            "ucinewgame" => {
                game = Game::startpos();
                String::new()
            }
            "setoption" => {
                set_option(&tokens, &mut engine);
                String::new()
            }
            "position" => match parse_position(&tokens) {
                Ok(g) => {
                    game = g;
                    String::new()
                }
                Err(e) => {
                    game = Game::startpos();
                    format!("info string error {e}")
                }
            },
            "go" => {
                let depth = tokens
                    .iter()
                    .position(|&t| t == "depth")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|d| d.parse().ok())
                    .unwrap_or(engine.depth);
                let result = engine.search(&game.board, depth);
                match result.best {
                    Some(mv) => format!(
                        "info depth {depth} score cp {} nodes {}\nbestmove {}",
                        result.score,
                        result.nodes,
                        game.uci(mv)
                    ),
                    None => "bestmove 0000".to_string(),
                }
            }
            "state" => {
                let legal: Vec<String> = game.legal_moves().into_iter().map(|m| game.uci(m)).collect();
                format!(
                    "fen {}\nstatus {}\nrepetitions {}\nplies {}\nlegal {}\nendstate",
                    game.board,
                    game.outcome().name(),
                    game.repetitions(),
                    game.plies,
                    legal.join(" ")
                )
            }
            "children" => {
                let mut lines = Vec::new();
                for mv in game.legal_moves() {
                    let mut child = game.clone();
                    let uci = game.uci(mv);
                    let san = display_san_move(&game.board, mv).to_string();
                    if child.play(mv).is_ok() {
                        lines.push(format!(
                            "child {uci} {san} {} {} {}",
                            child.outcome().name(),
                            child.repetitions(),
                            child.board
                        ));
                    }
                }
                lines.push("endchildren".to_string());
                lines.join("\n")
            }
            "parse" => {
                let text = tokens.get(1).copied().unwrap_or("");
                let parsed = parse_san(&game.board, text)
                    .ok()
                    .or_else(|| parse_uci_move(&game.board, text).ok().filter(|&m| game.board.is_legal(m)));
                match parsed {
                    Some(mv) => format!("parsed {} {}", game.uci(mv), display_san_move(&game.board, mv)),
                    None => "parsed none".to_string(),
                }
            }
            "d" => format!("{}fen {}", ascii_board(&game.board), game.board),
            "quit" => break,
            _ => format!("info string unknown command {command}"),
        };
        if !reply.is_empty() {
            let _ = writeln!(out, "{reply}");
            let _ = out.flush();
        }
    }
}

fn set_option(tokens: &[&str], engine: &mut Baseline) {
    let name = tokens.iter().position(|&t| t == "name").and_then(|i| tokens.get(i + 1));
    let value = tokens.iter().position(|&t| t == "value").and_then(|i| tokens.get(i + 1));
    match (name.map(|n| n.to_ascii_lowercase()), value) {
        (Some(n), Some(v)) if n == "depth" => {
            if let Ok(d) = v.parse() {
                engine.depth = d;
            }
        }
        (Some(n), Some(v)) if n == "seed" => {
            if let Ok(s) = v.parse() {
                engine.set_seed(s);
            }
        }
        (Some(n), Some(v)) if n == "quiescence" => engine.quiescence = *v == "true",
        _ => {}
    }
}

pub fn parse_position(tokens: &[&str]) -> Result<Game, String> {
    let moves_at = tokens.iter().position(|&t| t == "moves");
    let mut game = match tokens.get(1) {
        Some(&"startpos") => Game::startpos(),
        Some(&"fen") => {
            let end = moves_at.unwrap_or(tokens.len());
            Game::from_fen(&tokens[2..end].join(" "))?
        }
        _ => return Err("expected startpos or fen".to_string()),
    };
    if let Some(i) = moves_at {
        for mv in &tokens[i + 1..] {
            game.play_uci(mv)?;
        }
    }
    Ok(game)
}
