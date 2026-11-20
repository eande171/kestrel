pub mod evals;
pub mod game;
pub mod net;
pub mod netsearch;
pub mod netuci;
pub mod qconv;
pub mod prep;
pub mod search;
pub mod uci;

use cozy_chess::Board;

pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0;
    board.generate_moves(|moves| {
        if depth == 1 {
            nodes += moves.len() as u64;
        } else {
            for mv in moves {
                let mut child = board.clone();
                child.play_unchecked(mv);
                nodes += perft(&child, depth - 1);
            }
        }
        false
    });
    nodes
}
