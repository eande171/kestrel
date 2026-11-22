// Browser Exports

use std::cell::RefCell;
use std::rc::Rc;

use cozy_chess::util::display_san_move;
use kestrel::game::Game;
use kestrel::net::Network;
use kestrel::netsearch::NetSearch;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn now_ms() -> f64;
}

thread_local! {
    static SEARCH: RefCell<Option<NetSearch>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn text(ptr: *const u8, len: usize) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn reply(out: String) -> usize {
    OUTPUT.with_borrow_mut(|o| {
        *o = out.into_bytes();
        o.len()
    })
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

// Last Reply Address
#[unsafe(no_mangle)]
pub extern "C" fn output() -> *const u8 {
    OUTPUT.with_borrow(|o| o.as_ptr())
}

// Load Weights
#[unsafe(no_mangle)]
pub extern "C" fn load(ptr: *const u8, len: usize, int8: u32) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    match Network::load(bytes) {
        Ok(mut network) => {
            if int8 != 0 {
                network.use_int8();
            }
            let search = NetSearch::new(Rc::new(network), || unsafe { now_ms() } / 1000.0, 0.0, 1);
            SEARCH.with_borrow_mut(|s| *s = Some(search));
            0
        }
        Err(e) => reply(e),
    }
}

// Position State
#[unsafe(no_mangle)]
pub extern "C" fn state(fen: *const u8, fen_len: usize, moves: *const u8, moves_len: usize) -> usize {
    let out = match Game::from_position(&text(fen, fen_len), &text(moves, moves_len)) {
        Ok(game) => {
            let legal: Vec<String> = game
                .legal_moves()
                .into_iter()
                .map(|mv| format!("{} {}", game.uci(mv), display_san_move(&game.board, mv)))
                .collect();
            format!("status {}\nrepetitions {}\nfen {}\nlegal {}", game.outcome().name(), game.repetitions(), game.board,
                    legal.join(","))
        }
        Err(e) => format!("error {e}"),
    };
    reply(out)
}

// Search A Move
#[unsafe(no_mangle)]
pub extern "C" fn think(fen: *const u8, fen_len: usize, moves: *const u8, moves_len: usize, depth: u32,
                        millis: f64, randomness: f32, seed: u32) -> usize {
    let game = match Game::from_position(&text(fen, fen_len), &text(moves, moves_len)) {
        Ok(game) => game,
        Err(e) => return reply(format!("error {e}")),
    };
    let out = SEARCH.with_borrow_mut(|slot| {
        let Some(search) = slot.as_mut() else {
            return "error weights not loaded".to_string();
        };
        search.randomness = randomness;
        search.seed(seed as u64);
        let limit = (millis > 0.0).then_some(millis / 1000.0);
        match search.choose(&game, depth, limit) {
            Some(mv) => format!("{} {:.4} {} {}", game.uci(mv), search.last_score, search.completed_depth, search.nodes),
            None => "none".into(),
        }
    });
    reply(out)
}
