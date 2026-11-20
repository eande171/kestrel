use cozy_chess::{Board, Color, Move, Piece, Square};

use crate::game::standard_move;

pub const POLICY_SIZE: usize = 4096 + 72;
pub const PLANES: usize = 21;
const MAGIC: &[u8; 4] = b"KSTN";
const MASKED: f32 = -1e4;

struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
    outputs: usize,
}

// Weight Reader
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, len: Option<usize>) -> Result<&[u8], String> {
        let end = len.and_then(|n| self.pos.checked_add(n)).ok_or("weights file has impossible sizes")?;
        let raw = self.bytes.get(self.pos..end).ok_or("weights file is truncated")?;
        self.pos = end;
        Ok(raw)
    }

    fn u32(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(Some(4))?.try_into().unwrap()) as usize)
    }

    fn floats(&mut self, n: Option<usize>) -> Result<Vec<f32>, String> {
        let raw = self.take(n.and_then(|n| n.checked_mul(4)))?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn conv(&mut self, inputs: usize, outputs: usize, taps: usize) -> Result<Conv, String> {
        let weights = taps.checked_mul(inputs).and_then(|n| n.checked_mul(outputs));
        Ok(Conv { weight: self.floats(weights)?, bias: self.floats(Some(outputs))?, inputs, outputs })
    }
}

// Policy And Value Network
pub struct Network {
    stem: Conv,
    trunk: Vec<(Conv, Conv)>,
    policy: Conv,
    query: Conv,
    key: Conv,
    square_bias: Vec<f32>,
    value_conv: Conv,
    fc1: Conv,
    fc2: Conv,
}

// Expected Score
pub fn expected_score(wdl: [f32; 3]) -> f32 {
    let [loss, draw, win] = wdl;
    let top = loss.max(draw).max(win);
    let (loss, draw, win) = ((loss - top).exp(), (draw - top).exp(), (win - top).exp());
    (win - loss) / (loss + draw + win)
}

impl Network {
    pub fn load(bytes: &[u8]) -> Result<Network, String> {
        if bytes.get(..4) != Some(MAGIC) {
            return Err("not a Kestrel weights file".into());
        }
        let mut r = Reader { bytes, pos: 4 };
        let version = r.u32()?;
        if version != 1 {
            return Err(format!("unsupported weights version {version}"));
        }
        let (blocks, channels, key_dim, in_planes) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if in_planes != PLANES {
            return Err(format!("weights expect {in_planes} input planes, not {PLANES}"));
        }
        let stem = r.conv(in_planes, channels, 9)?;
        let mut trunk = Vec::with_capacity(blocks.min(64));
        for _ in 0..blocks {
            trunk.push((r.conv(channels, channels, 9)?, r.conv(channels, channels, 9)?));
        }
        let policy = r.conv(channels, channels, 9)?;
        let query = r.conv(channels, key_dim, 1)?;
        let key = r.conv(channels, key_dim * 4, 1)?;
        let square_bias = r.floats(Some(64 * 64))?;
        let value_conv = r.conv(channels, 8, 1)?;
        let fc1 = r.conv(8 * 64, 128, 1)?;
        let fc2 = r.conv(128, 3, 1)?;
        if r.pos != bytes.len() {
            return Err("weights file has extra bytes".into());
        }
        Ok(Network { stem, trunk, policy, query, key, square_bias, value_conv, fc1, fc2 })
    }

    // Value Only
    pub fn wdl(&self, planes: &[f32]) -> [f32; 3] {
        self.value_head(&self.body(planes))
    }

    // Value And Policy
    pub fn evaluate(&self, planes: &[f32], policy_indices: &[usize]) -> ([f32; 3], Vec<f32>) {
        let x = self.body(planes);
        (self.value_head(&x), self.policy_head(&x, policy_indices))
    }

    fn body(&self, planes: &[f32]) -> Vec<f32> {
        let size = 64 * self.stem.outputs;
        let (mut x, mut y, mut z) = (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
        conv3x3(&self.stem, planes, &mut x);
        relu(&mut x);
        for (first, second) in &self.trunk {
            conv3x3(first, &x, &mut y);
            relu(&mut y);
            conv3x3(second, &y, &mut z);
            for (a, b) in x.iter_mut().zip(&z) {
                *a = (*a + b).max(0.0);
            }
        }
        x
    }

    fn value_head(&self, x: &[f32]) -> [f32; 3] {
        let mut v = [0.0; 64 * 8];
        conv1x1(&self.value_conv, x, &mut v);
        relu(&mut v);
        let mut flat = [0.0; 64 * 8];
        for sq in 0..64 {
            for ch in 0..8 {
                flat[ch * 64 + sq] = v[sq * 8 + ch];
            }
        }
        let mut hidden = [0.0; 128];
        conv1x1(&self.fc1, &flat, &mut hidden);
        relu(&mut hidden);
        let mut out = [0.0; 3];
        conv1x1(&self.fc2, &hidden, &mut out);
        out
    }

    fn policy_head(&self, x: &[f32], indices: &[usize]) -> Vec<f32> {
        let kd = self.query.outputs;
        let mut p = vec![0.0; x.len()];
        conv3x3(&self.policy, x, &mut p);
        relu(&mut p);
        let mut q = vec![0.0; 64 * kd];
        conv1x1(&self.query, &p, &mut q);
        let mut k = vec![0.0; 64 * kd * 4];
        conv1x1(&self.key, &p, &mut k);
        let scale = 1.0 / (kd as f32).sqrt();
        let dot = |from: usize, to: usize, piece: usize| -> f32 {
            let qs = &q[from * kd..(from + 1) * kd];
            let ks = &k[(to * 4 + piece) * kd..(to * 4 + piece + 1) * kd];
            qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() * scale
        };
        indices
            .iter()
            .map(|&index| {
                if index < 4096 {
                    return dot(index / 64, index % 64, 0) + self.square_bias[index];
                }
                let i = index - 4096;
                let (piece, file, direction) = (i / 24, (i % 24) / 3, i % 3);
                let to_file = file as isize + direction as isize - 1;
                if !(0..8).contains(&to_file) {
                    return MASKED;
                }
                dot(48 + file, 56 + to_file as usize, piece + 1)
            })
            .collect()
    }
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

fn conv3x3(conv: &Conv, input: &[f32], out: &mut [f32]) {
    match conv.outputs {
        64 => conv3x3_fixed::<64>(conv, input, out),
        96 => conv3x3_fixed::<96>(conv, input, out),
        _ => conv3x3_any(conv, input, out),
    }
}

// Board Neighbours
fn neighbours(sq: usize) -> impl Iterator<Item = (usize, usize)> {
    let (rank, file) = ((sq / 8) as isize, (sq % 8) as isize);
    (0..9).filter_map(move |tap| {
        let (r, f) = (rank + tap as isize / 3 - 1, file + tap as isize % 3 - 1);
        ((0..8).contains(&r) && (0..8).contains(&f)).then_some((tap, (r * 8 + f) as usize))
    })
}

// Zero Skipping
const SKIP_ZEROS: bool = cfg!(target_arch = "wasm32");

// Register Accumulators
fn conv3x3_fixed<const N: usize>(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let cin = conv.inputs;
    for (sq, dest) in out.chunks_exact_mut(N).enumerate() {
        let mut acc: [f32; N] = conv.bias[..].try_into().unwrap();
        for (tap, nsq) in neighbours(sq) {
            let inp = &input[nsq * cin..(nsq + 1) * cin];
            let taps = &conv.weight[tap * cin * N..(tap + 1) * cin * N];
            for (&v, w) in inp.iter().zip(taps.chunks_exact(N)) {
                if SKIP_ZEROS && v == 0.0 {
                    continue;
                }
                let w: &[f32; N] = w.try_into().unwrap();
                for co in 0..N {
                    acc[co] += v * w[co];
                }
            }
        }
        dest.copy_from_slice(&acc);
    }
}

fn conv3x3_any(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let (cin, cout) = (conv.inputs, conv.outputs);
    for (sq, acc) in out.chunks_exact_mut(cout).enumerate() {
        acc.copy_from_slice(&conv.bias);
        for (tap, nsq) in neighbours(sq) {
            let taps = &conv.weight[tap * cin * cout..(tap + 1) * cin * cout];
            for (&v, w) in input[nsq * cin..(nsq + 1) * cin].iter().zip(taps.chunks_exact(cout)) {
                for (a, w) in acc.iter_mut().zip(w) {
                    *a += v * w;
                }
            }
        }
    }
}

fn conv1x1(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let (cin, cout) = (conv.inputs, conv.outputs);
    for (inp, acc) in input.chunks_exact(cin).zip(out.chunks_exact_mut(cout)) {
        acc.copy_from_slice(&conv.bias);
        for (&v, w) in inp.iter().zip(conv.weight.chunks_exact(cout)) {
            if v == 0.0 {
                continue;
            }
            for (a, w) in acc.iter_mut().zip(w) {
                *a += v * w;
            }
        }
    }
}

// Input Planes
pub fn encode(board: &Board, repetitions: usize) -> Vec<f32> {
    let mut planes = vec![0.0; 64 * PLANES];
    let us = board.side_to_move();
    let flip = if us == Color::Black { 56 } else { 0 };
    for sq in Square::ALL {
        if let (Some(piece), Some(color)) = (board.piece_on(sq), board.color_on(sq)) {
            let owner = if color == us { 0 } else { 6 };
            planes[(sq as usize ^ flip) * PLANES + piece as usize + owner] = 1.0;
        }
    }
    let rights = [board.castle_rights(us).short, board.castle_rights(us).long,
                  board.castle_rights(!us).short, board.castle_rights(!us).long];
    let ep = board.en_passant().map(|f| 40 + f as usize);
    let constants = [(us == Color::Black) as u8 as f32, 1.0, (repetitions >= 1) as u8 as f32,
                     (board.halfmove_clock() as f32 / 100.0).min(1.0)];
    for (sq, row) in planes.chunks_exact_mut(PLANES).enumerate() {
        for (i, right) in rights.iter().enumerate() {
            row[12 + i] = right.is_some() as u8 as f32;
        }
        row[16] = (ep == Some(sq)) as u8 as f32;
        row[17..21].copy_from_slice(&constants);
    }
    planes
}

// Policy Index
pub fn policy_index(board: &Board, mv: Move) -> usize {
    let flip = if board.side_to_move() == Color::Black { 56 } else { 0 };
    let mv = standard_move(board, mv);
    let (f, t) = (mv.from as usize ^ flip, mv.to as usize ^ flip);
    let code = match mv.promotion {
        Some(Piece::Knight) => 0,
        Some(Piece::Bishop) => 1,
        Some(Piece::Rook) => 2,
        _ => return f * 64 + t,
    };
    4096 + code * 24 + (f % 8) * 3 + (t % 8 + 1 - f % 8)
}
