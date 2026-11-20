use std::collections::HashMap;

use cozy_chess::{Board, Color, Move, Piece, Rank};

use crate::game::{any_move, legal_moves, Game};
use crate::net::{encode, policy_index, Network};

pub const MATE: f32 = 1.0;
const CACHE_LIMIT: usize = 200_000;
const DEFAULT_GROWTH: f64 = 3.0;

// Endgame Tuning
const MOP_UP_WEIGHT: f32 = 0.05;
const MOP_UP_LEAD: i32 = 4;
const MOP_UP_DEFENDER_MAX: i32 = 5;
const AHEAD_VALUE: f32 = 0.5;
const DRAW_PENALTY: f32 = 0.1;
const CONVERSION_SCALE: f32 = 0.85;
const CONVERSION_WEIGHT: f32 = 0.01;
const CONVERSION_CAP: f32 = 0.08;

// Forcing Move Extensions
const EXTRA_FORCING: usize = 3;
const QUIESCENCE_PLIES: i32 = 4;
const QUIESCENCE_WIDTH: usize = 4;
pub const CHECK_EXTENSIONS: u32 = 2;

// Leaf Batching
const FIRST_LEAVES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ongoing,
    Checkmate,
    Draw,
}

#[derive(Clone)]
struct Child {
    mv: Move,
    board: Board,
    status: Status,
    repetitions: usize,
    forcing: bool,
    check: bool,
}

type Key = (u64, u8, u8);

fn key(board: &Board, repetitions: usize) -> Key {
    (board.hash(), board.halfmove_clock(), repetitions.min(2) as u8)
}

struct Timeout;

/// Returns each side's material in pawns and each king's file and rank.
fn material_and_kings(board: &Board) -> ([i32; 2], [(i32, i32); 2]) {
    let mut material = [0; 2];
    let mut kings = [(0, 0); 2];
    for color in [Color::White, Color::Black] {
        let own = board.colors(color);
        for (piece, points) in [(Piece::Pawn, 1), (Piece::Knight, 3), (Piece::Bishop, 3), (Piece::Rook, 5), (Piece::Queen, 9)] {
            material[color as usize] += points * (own & board.pieces(piece)).len() as i32;
        }
        let king = board.king(color);
        kings[color as usize] = (king.file() as i32, king.rank() as i32);
    }
    (material, kings)
}

fn mop_up(material: [i32; 2], kings: [(i32, i32); 2], side: usize) -> f32 {
    let strong = if material[0] > material[1] { 0 } else { 1 };
    let weak = 1 - strong;
    if material[strong] - material[weak] < MOP_UP_LEAD || material[weak] > MOP_UP_DEFENDER_MAX {
        return 0.0;
    }
    let ((wf, wr), (sf, sr)) = (kings[weak], kings[strong]);
    let edge = (3 - wf).max(wf - 4) + (3 - wr).max(wr - 4);
    let apart = (wf - sf).abs() + (wr - sr).abs();
    let bonus = MOP_UP_WEIGHT * (0.6 * edge as f32 / 6.0 + 0.4 * (14 - apart) as f32 / 14.0);
    if side == strong { bonus } else { -bonus }
}

/// Returns a value head score plus the mop-up bonus and, once the position is decided, a small material tie-break.
fn endgame_adjusted(board: &Board, value: f32) -> f32 {
    let (material, kings) = material_and_kings(board);
    let side = board.side_to_move() as usize;
    let mut score = value + mop_up(material, kings, side);
    if value.abs() > AHEAD_VALUE {
        let lead = (material[side] - material[1 - side]) as f32;
        score = CONVERSION_SCALE * score + (CONVERSION_WEIGHT * lead).clamp(-CONVERSION_CAP, CONVERSION_CAP);
    }
    score.clamp(-0.999, 0.999)
}

fn insufficient_material(board: &Board) -> bool {
    let heavy = board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    heavy.is_empty() && (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)).len() <= 1
}

/// Negamax with alpha-beta pruning over the top policy moves, scored by the value head.
pub struct NetSearch<'a> {
    network: &'a Network,
    now: Box<dyn Fn() -> f64 + 'a>,
    pub width: usize,
    pub randomness: f32,
    pub endgame_tuning: bool,
    pub extensions: bool,
    pub check_extensions: u32,
    pub nodes: u64,
    pub evaluations: u64,
    pub completed_depth: u32,
    pub last_score: f32,
    rng: u64,
    deadline: f64,
    root_depth: u32,
    draw_score: f32,
    history: Vec<u64>,
    priors: HashMap<Key, Vec<f32>>,
    values: HashMap<Key, f32>,
    best_child: HashMap<Key, Move>,
}

impl<'a> NetSearch<'a> {
    /// Creates a search, with now returning seconds from any fixed point.
    pub fn new(network: &'a Network, now: impl Fn() -> f64 + 'a, width: usize, randomness: f32, seed: u64) -> Self {
        NetSearch {
            network,
            now: Box::new(now),
            width,
            randomness,
            endgame_tuning: true,
            extensions: true,
            check_extensions: CHECK_EXTENSIONS,
            nodes: 0,
            evaluations: 0,
            completed_depth: 0,
            last_score: 0.0,
            rng: seed | 1,
            deadline: f64::INFINITY,
            root_depth: 0,
            draw_score: 0.0,
            history: Vec::new(),
            priors: HashMap::new(),
            values: HashMap::new(),
            best_child: HashMap::new(),
        }
    }

    /// Returns the chosen move, searching exactly to depth when no time limit is given, or deepening until it.
    pub fn choose(&mut self, game: &Game, depth: u32, seconds: Option<f64>) -> Option<Move> {
        self.nodes = 0;
        if self.values.len() > CACHE_LIMIT {
            self.priors.clear();
            self.values.clear();
            self.best_child.clear();
        }
        self.history = game.history().to_vec();
        let reps = game.repetitions();
        let board = &game.board;
        match seconds {
            None => {
                self.deadline = f64::INFINITY;
                self.completed_depth = depth;
                let scored = self.root(board, reps, depth, None).ok()?;
                self.pick(&scored)
            }
            Some(limit) => self.deepen(board, reps, depth, (self.now)() + limit),
        }
    }

    fn deepen(&mut self, board: &Board, reps: usize, max_depth: u32, deadline: f64) -> Option<Move> {
        self.deadline = f64::INFINITY;
        let policy_best = self.expand(board, reps).ok()?.first()?.mv;
        self.deadline = deadline;
        let (mut scored, mut order, mut previous): (Option<Vec<(f32, Move)>>, Option<Vec<Move>>, Option<f64>) =
            (None, None, None);
        self.completed_depth = 0;
        for depth in 2..=max_depth {
            let start = (self.now)();
            match self.root(board, reps, depth, order.as_deref()) {
                Ok(result) => scored = Some(result),
                Err(Timeout) => break,
            }
            self.completed_depth = depth;
            let took = (self.now)() - start;
            let growth = match previous {
                Some(p) if p > 0.01 => (took / p).clamp(2.0, 8.0),
                _ => DEFAULT_GROWTH,
            };
            previous = Some(took);
            let current = scored.as_ref().unwrap();
            if current.iter().any(|&(s, _)| s >= MATE) || (self.now)() + took * growth > deadline {
                break;
            }
            let mut ranked = current.clone();
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            order = Some(ranked.into_iter().map(|(_, mv)| mv).collect());
        }
        self.deadline = f64::INFINITY;
        match scored {
            Some(result) => self.pick(&result),
            None => {
                self.last_score = 0.0;
                Some(policy_best)
            }
        }
    }

    fn pick(&mut self, scored: &[(f32, Move)]) -> Option<Move> {
        let top = scored.iter().cloned().fold(None, |best: Option<(f32, Move)>, item| match best {
            Some(b) if b.0 >= item.0 => Some(b),
            _ => Some(item),
        })?;
        let pool: Vec<(f32, Move)> = if self.randomness > 0.0 && top.0 < AHEAD_VALUE {
            scored.iter().cloned().filter(|item| item.0 >= top.0 - self.randomness).collect()
        } else {
            vec![top]
        };
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let (score, mv) = pool[(self.rng % pool.len() as u64) as usize];
        self.last_score = score;
        Some(mv)
    }

    /// Returns exact scores for root moves that can be chosen, and upper bounds for the rest.
    fn root(&mut self, board: &Board, reps: usize, depth: u32, order: Option<&[Move]>) -> Result<Vec<(f32, Move)>, Timeout> {
        let mut selected = self.expand(board, reps)?;
        if let Some(order) = order {
            let rank = |mv: &Move| order.iter().position(|m| m == mv).unwrap_or(order.len());
            selected.sort_by_key(|c| rank(&c.mv));
        }
        let mut leaf_values = HashMap::new();
        if depth == 1 {
            leaf_values = self.leaf_values(&selected);
        }
        self.root_depth = depth;
        let ahead = self.endgame_tuning && self.values[&key(board, reps)] > AHEAD_VALUE;
        self.draw_score = if ahead { -DRAW_PENALTY } else { 0.0 };

        let mut scored = Vec::with_capacity(selected.len());
        let mut best = f32::NEG_INFINITY;
        for child in &selected {
            let alpha = best - self.randomness - 1e-9;
            let extend = self.extends(child, 0);
            let score = match self.terminal(child, 1) {
                Some(s) => s,
                None if depth == 1 && extend == 0 => self.leaf_score(child, &leaf_values, alpha, f32::INFINITY, 1)?,
                None => -self.search(child, depth - 1 + extend, f32::NEG_INFINITY, -alpha, 1, extend)?,
            };
            scored.push((score, child.mv));
            best = best.max(score);
        }
        Ok(scored)
    }

    fn search(&mut self, node: &Child, depth: u32, alpha: f32, beta: f32, ply: u32, extended: u32) -> Result<f32, Timeout> {
        self.history.push(node.board.hash());
        let result = self.search_inner(node, depth, alpha, beta, ply, extended);
        self.history.pop();
        result
    }

    fn search_inner(&mut self, node: &Child, depth: u32, alpha: f32, beta: f32, ply: u32, extended: u32) -> Result<f32, Timeout> {
        let selected = self.expand(&node.board, node.repetitions)?;
        let mut leaf_values: HashMap<Move, f32> = HashMap::new();
        let mut best = f32::NEG_INFINITY;
        let mut best_move = None;
        for (index, child) in selected.iter().enumerate() {
            let extend = self.extends(child, extended);
            let score = match self.terminal(child, ply + 1) {
                Some(s) => s,
                None if depth == 1 && extend == 0 => {
                    if !leaf_values.contains_key(&child.mv) {
                        let batch = if index > 0 { &selected[index..] } else { &selected[..FIRST_LEAVES.min(selected.len())] };
                        leaf_values.extend(self.leaf_values(batch));
                    }
                    self.leaf_score(child, &leaf_values, alpha.max(best), beta, ply + 1)?
                }
                None => -self.search(child, depth - 1 + extend, -beta, -alpha.max(best), ply + 1, extended + extend)?,
            };
            if score > best {
                best = score;
                best_move = Some(child.mv);
            }
            if best >= beta {
                break;
            }
        }
        if let Some(mv) = best_move {
            self.best_child.insert(key(&node.board, node.repetitions), mv);
        }
        Ok(best)
    }

    fn extends(&self, child: &Child, extended: u32) -> u32 {
        (child.check && extended < self.check_extensions) as u32
    }

    /// Returns the score of a finished child, ply moves from the root, for the side that moved into it.
    fn terminal(&self, child: &Child, ply: u32) -> Option<f32> {
        match child.status {
            Status::Checkmate => Some(MATE + 0.01 * (self.root_depth as f32 - ply as f32 + 1.0).max(0.0)),
            Status::Draw => Some(if ply % 2 == 1 { self.draw_score } else { -self.draw_score }),
            Status::Ongoing => None,
        }
    }

    /// Returns every child of a position, highest policy first.
    fn ordered_children(&mut self, board: &Board, reps: usize) -> Result<Vec<Child>, Timeout> {
        if (self.now)() > self.deadline {
            return Err(Timeout);
        }
        self.nodes += 1;
        let children = self.children(board);
        let k = key(board, reps);
        if !self.priors.contains_key(&k) {
            let indices: Vec<usize> = children.iter().map(|c| policy_index(board, c.mv)).collect();
            let output = self.network.evaluate(&encode(board, reps), Some(&indices));
            self.evaluations += 1;
            self.values.insert(k, output.value());
            self.priors.insert(k, output.policy);
        }
        let prior = &self.priors[&k];
        let mut order: Vec<usize> = (0..children.len()).collect();
        order.sort_by(|&a, &b| prior[b].partial_cmp(&prior[a]).unwrap());
        Ok(order.into_iter().map(|i| children[i].clone()).collect())
    }

    fn children(&self, board: &Board) -> Vec<Child> {
        let white = board.side_to_move() == Color::White;
        legal_moves(board)
            .into_iter()
            .map(|mv| {
                let mut next = board.clone();
                next.play_unchecked(mv);
                let hash = next.hash();
                let repetitions = if next.halfmove_clock() == 0 {
                    0
                } else {
                    self.history.iter().filter(|&&h| h == hash).count()
                };
                let status = if !any_move(&next) {
                    if next.checkers().is_empty() { Status::Draw } else { Status::Checkmate }
                } else if next.halfmove_clock() >= 100 || repetitions >= 2 || insufficient_material(&next) {
                    Status::Draw
                } else {
                    Status::Ongoing
                };
                let pawn = board.piece_on(mv.from) == Some(Piece::Pawn);
                let capture = board.color_on(mv.to) == Some(!board.side_to_move())
                    || (pawn && mv.from.file() != mv.to.file());
                let seventh = pawn && mv.to.rank() == if white { Rank::Seventh } else { Rank::Second };
                Child {
                    mv,
                    status,
                    repetitions,
                    forcing: capture || mv.promotion.is_some() || seventh,
                    check: !next.checkers().is_empty() && status != Status::Checkmate,
                    board: next,
                }
            })
            .collect()
    }

    /// Returns the top policy children plus a few forcing ones, with the best move from an earlier search first.
    fn expand(&mut self, board: &Board, reps: usize) -> Result<Vec<Child>, Timeout> {
        let ordered = self.ordered_children(board, reps)?;
        let mut selected: Vec<Child> = ordered.iter().take(self.width).cloned().collect();
        if self.extensions {
            selected.extend(ordered.iter().skip(self.width).filter(|c| c.forcing).take(EXTRA_FORCING).cloned());
        }
        if let Some(hint) = self.best_child.get(&key(board, reps)) {
            selected.sort_by_key(|c| c.mv != *hint);
        }
        Ok(selected)
    }

    /// Returns a depth 1 child's score for the side that moved, following forcing replies after a forcing move.
    fn leaf_score(&mut self, child: &Child, leaf_values: &HashMap<Move, f32>, alpha: f32, beta: f32,
                  ply: u32) -> Result<f32, Timeout> {
        if self.extensions && child.forcing {
            return Ok(-self.quiesce(child, -beta, -alpha, 0, ply)?);
        }
        Ok(-leaf_values[&child.mv])
    }

    fn quiesce(&mut self, node: &Child, alpha: f32, beta: f32, depth: i32, ply: u32) -> Result<f32, Timeout> {
        let stand = self.adjusted_value(node);
        if stand >= beta || depth <= -QUIESCENCE_PLIES {
            return Ok(stand);
        }
        self.history.push(node.board.hash());
        let result = self.quiesce_moves(node, alpha, beta, depth, ply, stand);
        self.history.pop();
        result
    }

    fn quiesce_moves(&mut self, node: &Child, alpha: f32, beta: f32, depth: i32, ply: u32, stand: f32) -> Result<f32, Timeout> {
        let forcing: Vec<Child> = self
            .ordered_children(&node.board, node.repetitions)?
            .into_iter()
            .filter(|c| c.forcing)
            .take(QUIESCENCE_WIDTH)
            .collect();
        if !forcing.is_empty() {
            self.leaf_values(&forcing);
        }
        let mut best = stand;
        for child in &forcing {
            let score = match self.terminal(child, ply + 1) {
                Some(s) => s,
                None => -self.quiesce(child, -beta, -alpha.max(best), depth - 1, ply + 1)?,
            };
            best = best.max(score);
            if best >= beta {
                break;
            }
        }
        Ok(best)
    }

    fn adjusted_value(&self, child: &Child) -> f32 {
        let value = self.values[&key(&child.board, child.repetitions)];
        if self.endgame_tuning { endgame_adjusted(&child.board, value) } else { value }
    }

    /// Returns value head scores for the ongoing children, evaluating the uncached ones.
    fn leaf_values(&mut self, children: &[Child]) -> HashMap<Move, f32> {
        let mut out = HashMap::new();
        for child in children.iter().filter(|c| c.status == Status::Ongoing) {
            let k = key(&child.board, child.repetitions);
            if !self.values.contains_key(&k) {
                let value = self.network.evaluate(&encode(&child.board, child.repetitions), None).value();
                self.evaluations += 1;
                self.values.insert(k, value);
            }
            out.insert(child.mv, self.adjusted_value(child));
        }
        out
    }
}
