use cozy_chess::{Board, Color, Move, Piece};

use crate::game::legal_moves;

pub const MATE: i32 = 30_000;
const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub nodes: u64,
}

// Baseline Searcher
pub struct Baseline {
    pub depth: u32,
    pub quiescence: bool,
    rng: u64,
    nodes: u64,
}

impl Baseline {
    pub fn new(depth: u32, seed: u64) -> Self {
        Baseline { depth, quiescence: true, rng: seed | 1, nodes: 0 }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = seed | 1;
    }

    // Xorshift Tiebreak
    fn next_random(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    pub fn search(&mut self, board: &Board) -> SearchResult {
        self.nodes = 0;
        let mut moves = legal_moves(board);
        if moves.is_empty() {
            let score = if board.checkers().is_empty() { 0 } else { -MATE };
            return SearchResult { best: None, score, nodes: 0 };
        }
        for i in (1..moves.len()).rev() {
            let j = (self.next_random() % (i as u64 + 1)) as usize;
            moves.swap(i, j);
        }
        order_moves(board, &mut moves);

        let depth = self.depth.max(1);
        let mut best = moves[0];
        let mut alpha = -MATE - 1;
        for mv in moves {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -self.negamax(&child, depth - 1, 1, -MATE - 1, -alpha);
            if score > alpha {
                alpha = score;
                best = mv;
            }
        }
        SearchResult { best: Some(best), score: alpha, nodes: self.nodes }
    }

    fn negamax(&mut self, board: &Board, depth: u32, ply: i32, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let mut moves = legal_moves(board);
        if moves.is_empty() {
            return if board.checkers().is_empty() { 0 } else { -MATE + ply };
        }
        if board.halfmove_clock() >= 100 {
            return 0;
        }
        if depth == 0 {
            return if self.quiescence {
                self.quiesce(board, alpha, beta)
            } else {
                evaluate(board)
            };
        }
        order_moves(board, &mut moves);
        for mv in moves {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    fn quiesce(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let stand_pat = evaluate(board);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
        let them = board.colors(!board.side_to_move());
        let mut captures: Vec<Move> = legal_moves(board)
            .into_iter()
            .filter(|mv| them.has(mv.to) || mv.promotion == Some(Piece::Queen))
            .collect();
        order_moves(board, &mut captures);
        for mv in captures {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -self.quiesce(&child, -beta, -alpha);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }
}

// Material Only
pub fn evaluate(board: &Board) -> i32 {
    let us = board.side_to_move();
    let mut score = 0;
    for piece in Piece::ALL {
        let value = PIECE_VALUES[piece as usize];
        let ours = board.colored_pieces(us, piece).len() as i32;
        let theirs = board.colored_pieces(!us, piece).len() as i32;
        score += value * (ours - theirs);
    }
    score
}

// MVV LVA
fn order_moves(board: &Board, moves: &mut [Move]) {
    let them: Color = !board.side_to_move();
    let theirs = board.colors(them);
    moves.sort_by_key(|mv| {
        let mut key = 0;
        if theirs.has(mv.to) {
            let victim = board.piece_on(mv.to).map_or(0, |p| PIECE_VALUES[p as usize]);
            let attacker = board.piece_on(mv.from).map_or(0, |p| PIECE_VALUES[p as usize]);
            key -= 10 * victim - attacker / 10;
        }
        if let Some(p) = mv.promotion {
            key -= PIECE_VALUES[p as usize];
        }
        key
    });
}
