//! Kestrel for the browser: plain WebAssembly exports with strings passed through linear memory.

use std::cell::RefCell;

use cozy_chess::util::display_san_move;
use kestrel::game::{Game, Outcome};
use kestrel::net::Network;
use kestrel::netsearch::NetSearch;

unsafe extern "C" {
    fn now_ms() -> f64;
}

thread_local! {
    static NETWORK: RefCell<Option<&'static Network>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn text(ptr: *const u8, len: usize) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn reply(out: String) -> usize {
    OUTPUT.with(|o| {
        let mut o = o.borrow_mut();
        *o = out.into_bytes();
        o.len()
    })
}

fn game_from(fen: &str, moves: &str) -> Result<Game, String> {
    let mut game = if fen.is_empty() || fen == "startpos" { Game::startpos() } else { Game::from_fen(fen)? };
    for mv in moves.split_whitespace() {
        game.play_uci(mv)?;
    }
    Ok(game)
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    unsafe { drop(Vec::from_raw_parts(ptr, 0, len)) };
}

/// Returns the address of the last reply, whose length the call returned.
#[unsafe(no_mangle)]
pub extern "C" fn output() -> *const u8 {
    OUTPUT.with(|o| o.borrow().as_ptr())
}

/// Loads network weights from a .kstn file, returning 0 on success or the length of an error message.
#[unsafe(no_mangle)]
pub extern "C" fn load(ptr: *const u8, len: usize) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    match Network::load(bytes) {
        Ok(network) => {
            NETWORK.with(|n| *n.borrow_mut() = Some(Box::leak(Box::new(network))));
            0
        }
        Err(e) => reply(e),
    }
}

/// Describes a position as lines of status, repetitions, FEN and legal moves as "uci san" pairs.
#[unsafe(no_mangle)]
pub extern "C" fn state(fen: *const u8, fen_len: usize, moves: *const u8, moves_len: usize) -> usize {
    let out = match game_from(&text(fen, fen_len), &text(moves, moves_len)) {
        Ok(game) => {
            let legal: Vec<String> = game
                .legal_moves()
                .into_iter()
                .map(|mv| format!("{} {}", game.uci(mv), display_san_move(&game.board, mv)))
                .collect();
            let status = match game.outcome() {
                Outcome::Ongoing => "ongoing",
                other => other.name(),
            };
            format!("status {status}\nrepetitions {}\nfen {}\nlegal {}", game.repetitions(), game.board, legal.join(","))
        }
        Err(e) => format!("error {e}"),
    };
    reply(out)
}

/// Searches a position and replies "move score depth positions", deepening up to depth until millis have passed.
#[unsafe(no_mangle)]
pub extern "C" fn think(fen: *const u8, fen_len: usize, moves: *const u8, moves_len: usize, depth: u32,
                        millis: f64, randomness: f32, seed: u32) -> usize {
    let Some(network) = NETWORK.with(|n| *n.borrow()) else {
        return reply("error weights not loaded".into());
    };
    let out = match game_from(&text(fen, fen_len), &text(moves, moves_len)) {
        Ok(game) => {
            let mut search = NetSearch::new(network, || unsafe { now_ms() } / 1000.0, 6, randomness, seed as u64);
            let limit = (millis > 0.0).then_some(millis / 1000.0);
            match search.choose(&game, depth.max(1), limit) {
                Some(mv) => format!("{} {:.4} {} {}", game.uci(mv), search.last_score, search.completed_depth, search.nodes),
                None => "none".into(),
            }
        }
        Err(e) => format!("error {e}"),
    };
    reply(out)
}
