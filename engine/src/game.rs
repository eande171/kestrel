use cozy_chess::util::{display_uci_move, parse_san_move, parse_uci_move};
use cozy_chess::{Board, Color, Move, Piece};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Checkmate { winner: Color },
    Stalemate,
    FiftyMoves,
    Repetition,
    InsufficientMaterial,
}

impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Ongoing => "ongoing",
            Outcome::Checkmate { winner: Color::White } => "checkmate_white",
            Outcome::Checkmate { winner: Color::Black } => "checkmate_black",
            Outcome::Stalemate => "stalemate",
            Outcome::FiftyMoves => "fifty_moves",
            Outcome::Repetition => "repetition",
            Outcome::InsufficientMaterial => "insufficient_material",
        }
    }
}

// Game With History
#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub plies: usize,
    hashes: Vec<u64>,
}

impl Game {
    pub fn new(board: Board) -> Self {
        let hashes = vec![board.hash()];
        Game { board, plies: 0, hashes }
    }

    pub fn startpos() -> Self {
        Game::new(Board::default())
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        Board::from_fen(fen, false)
            .map(Game::new)
            .map_err(|e| format!("bad fen: {e}"))
    }

    pub fn play(&mut self, mv: Move) -> Result<(), String> {
        self.board.try_play(mv).map_err(|_| format!("illegal move {mv}"))?;
        if self.board.halfmove_clock() == 0 {
            self.hashes.clear();
        }
        self.hashes.push(self.board.hash());
        self.plies += 1;
        Ok(())
    }

    pub fn play_uci(&mut self, text: &str) -> Result<Move, String> {
        let mv = parse_uci_move(&self.board, text).map_err(|_| format!("bad move {text}"))?;
        self.play(mv)?;
        Ok(mv)
    }

    pub fn play_san(&mut self, text: &str) -> Result<Move, String> {
        let mv = parse_san(&self.board, text)?;
        self.play(mv)?;
        Ok(mv)
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        legal_moves(&self.board)
    }

    pub fn repetitions(&self) -> usize {
        let current = self.board.hash();
        self.hashes.iter().filter(|&&h| h == current).count().saturating_sub(1)
    }

    pub fn uci(&self, mv: Move) -> String {
        display_uci_move(&self.board, mv).to_string()
    }

    pub fn outcome(&self) -> Outcome {
        let mut any = false;
        self.board.generate_moves(|_| {
            any = true;
            true
        });
        if !any {
            return if self.board.checkers().is_empty() {
                Outcome::Stalemate
            } else {
                Outcome::Checkmate { winner: !self.board.side_to_move() }
            };
        }
        if self.board.halfmove_clock() >= 100 {
            return Outcome::FiftyMoves;
        }
        let current = self.board.hash();
        if self.hashes.iter().filter(|&&h| h == current).count() >= 3 {
            return Outcome::Repetition;
        }
        if insufficient_material(&self.board) {
            return Outcome::InsufficientMaterial;
        }
        Outcome::Ongoing
    }
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    board.generate_moves(|pm| {
        moves.extend(pm);
        false
    });
    moves
}

// SAN With Annotations
pub fn parse_san(board: &Board, text: &str) -> Result<Move, String> {
    let trimmed = text.trim_end_matches(['!', '?']);
    let mv = parse_san_move(board, trimmed).map_err(|_| format!("bad san {text}"))?;
    if board.is_legal(mv) {
        Ok(mv)
    } else {
        Err(format!("illegal san {text}"))
    }
}

fn insufficient_material(board: &Board) -> bool {
    let heavy = board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    if !heavy.is_empty() {
        return false;
    }
    let minors = board.pieces(Piece::Knight) | board.pieces(Piece::Bishop);
    minors.len() <= 1
}

pub fn ascii_board(board: &Board) -> String {
    let mut out = String::new();
    for rank in (0..8).rev() {
        out.push_str(&format!("{} ", rank + 1));
        for file in 0..8 {
            let sq = cozy_chess::Square::index(rank * 8 + file);
            let ch = match (board.piece_on(sq), board.color_on(sq)) {
                (Some(p), Some(c)) => {
                    let ch: char = p.into();
                    if c == Color::White { ch.to_ascii_uppercase() } else { ch }
                }
                _ => '.',
            };
            out.push(ch);
            out.push(' ');
        }
        out.push('\n');
    }
    out.push_str("  a b c d e f g h\n");
    out
}
