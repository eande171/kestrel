use cozy_chess::{Board, Color, File, Move, Piece, Square};

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

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<usize, String> {
        let raw = self.bytes.get(self.pos..self.pos + 4).ok_or("weights file is truncated")?;
        self.pos += 4;
        Ok(u32::from_le_bytes(raw.try_into().unwrap()) as usize)
    }

    fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let raw = self.bytes.get(self.pos..self.pos + 4 * n).ok_or("weights file is truncated")?;
        self.pos += 4 * n;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn conv(&mut self, inputs: usize, outputs: usize, taps: usize) -> Result<Conv, String> {
        Ok(Conv { weight: self.floats(taps * inputs * outputs)?, bias: self.floats(outputs)?, inputs, outputs })
    }
}

/// The policy and value network, evaluated one position at a time on the CPU.
pub struct Network {
    pub channels: usize,
    key_dim: usize,
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

/// Network outputs for one position: win, draw and loss logits and, when asked for, policy logits.
pub struct Output {
    pub wdl: [f32; 3],
    pub policy: Vec<f32>,
}

impl Output {
    /// Returns the expected score for the side to move, from -1 for a loss to 1 for a win.
    pub fn value(&self) -> f32 {
        let top = self.wdl.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = self.wdl.iter().map(|x| (x - top).exp()).collect();
        (e[2] - e[0]) / (e[0] + e[1] + e[2])
    }
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
        let mut trunk = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            trunk.push((r.conv(channels, channels, 9)?, r.conv(channels, channels, 9)?));
        }
        let policy = r.conv(channels, channels, 9)?;
        let query = r.conv(channels, key_dim, 1)?;
        let key = r.conv(channels, key_dim * 4, 1)?;
        let square_bias = r.floats(64 * 64)?;
        let value_conv = r.conv(channels, 8, 1)?;
        let fc1 = r.conv(8 * 64, 128, 1)?;
        let fc2 = r.conv(128, 3, 1)?;
        if r.pos != bytes.len() {
            return Err("weights file has extra bytes".into());
        }
        Ok(Network { channels, key_dim, stem, trunk, policy, query, key, square_bias, value_conv, fc1, fc2 })
    }

    /// Returns the network's outputs, with policy logits only for the given policy indices.
    pub fn evaluate(&self, planes: &[f32], policy_indices: Option<&[usize]>) -> Output {
        let c = self.channels;
        let mut x = vec![0.0; 64 * c];
        let mut y = vec![0.0; 64 * c];
        conv3x3(&self.stem, planes, &mut x);
        relu(&mut x);
        for (first, second) in &self.trunk {
            conv3x3(first, &x, &mut y);
            relu(&mut y);
            let mut z = vec![0.0; 64 * c];
            conv3x3(second, &y, &mut z);
            for (a, b) in x.iter_mut().zip(&z) {
                *a = (*a + b).max(0.0);
            }
        }
        let wdl = self.value_head(&x);
        let policy = match policy_indices {
            Some(indices) => self.policy_head(&x, indices),
            None => Vec::new(),
        };
        Output { wdl, policy }
    }

    fn value_head(&self, x: &[f32]) -> [f32; 3] {
        let mut v = vec![0.0; 64 * 8];
        conv1x1(&self.value_conv, x, &mut v);
        relu(&mut v);
        let mut flat = vec![0.0; 64 * 8];
        for sq in 0..64 {
            for ch in 0..8 {
                flat[ch * 64 + sq] = v[sq * 8 + ch];
            }
        }
        let mut hidden = vec![0.0; 128];
        conv1x1(&self.fc1, &flat, &mut hidden);
        relu(&mut hidden);
        let mut out = vec![0.0; 3];
        conv1x1(&self.fc2, &hidden, &mut out);
        [out[0], out[1], out[2]]
    }

    fn policy_head(&self, x: &[f32], indices: &[usize]) -> Vec<f32> {
        let (c, kd) = (self.channels, self.key_dim);
        let mut p = vec![0.0; 64 * c];
        conv3x3(&self.policy, x, &mut p);
        relu(&mut p);
        let mut q = vec![0.0; 64 * kd];
        conv1x1(&self.query, &p, &mut q);
        let mut k = vec![0.0; 64 * kd * 4];
        conv1x1(&self.key, &p, &mut k);
        let scale = 1.0 / (kd as f32).sqrt();
        let dot = |from: usize, to: usize, piece: usize| -> f32 {
            let qs = &q[from * kd..(from + 1) * kd];
            let ks = &k[to * kd * 4 + piece * kd..to * kd * 4 + (piece + 1) * kd];
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

// Register Accumulators
fn conv3x3_fixed<const N: usize>(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let cin = conv.inputs;
    for sq in 0..64 {
        let mut acc = [0.0f32; N];
        acc.copy_from_slice(&conv.bias);
        let (rank, file) = ((sq / 8) as isize, (sq % 8) as isize);
        for ky in 0..3 {
            let r = rank + ky as isize - 1;
            if !(0..8).contains(&r) {
                continue;
            }
            for kx in 0..3 {
                let f = file + kx as isize - 1;
                if !(0..8).contains(&f) {
                    continue;
                }
                let nsq = (r * 8 + f) as usize;
                let inp = &input[nsq * cin..(nsq + 1) * cin];
                let taps = &conv.weight[(ky * 3 + kx) * cin * N..(ky * 3 + kx + 1) * cin * N];
                for (&v, w) in inp.iter().zip(taps.chunks_exact(N)) {
                    let w: &[f32; N] = w.try_into().unwrap();
                    for co in 0..N {
                        acc[co] += v * w[co];
                    }
                }
            }
        }
        out[sq * N..(sq + 1) * N].copy_from_slice(&acc);
    }
}

fn conv3x3_any(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let (cin, cout) = (conv.inputs, conv.outputs);
    for sq in 0..64 {
        let acc = &mut out[sq * cout..(sq + 1) * cout];
        acc.copy_from_slice(&conv.bias);
        let (rank, file) = ((sq / 8) as isize, (sq % 8) as isize);
        for ky in 0..3 {
            let r = rank + ky as isize - 1;
            if !(0..8).contains(&r) {
                continue;
            }
            for kx in 0..3 {
                let f = file + kx as isize - 1;
                if !(0..8).contains(&f) {
                    continue;
                }
                let nsq = (r * 8 + f) as usize;
                let inp = &input[nsq * cin..(nsq + 1) * cin];
                let taps = &conv.weight[(ky * 3 + kx) * cin * cout..(ky * 3 + kx + 1) * cin * cout];
                for (ci, &v) in inp.iter().enumerate() {
                    if v == 0.0 {
                        continue;
                    }
                    for (a, w) in acc.iter_mut().zip(&taps[ci * cout..(ci + 1) * cout]) {
                        *a += v * w;
                    }
                }
            }
        }
    }
}

fn conv1x1(conv: &Conv, input: &[f32], out: &mut [f32]) {
    let (cin, cout) = (conv.inputs, conv.outputs);
    for (inp, acc) in input.chunks_exact(cin).zip(out.chunks_exact_mut(cout)) {
        acc.copy_from_slice(&conv.bias);
        for (ci, &v) in inp.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            for (a, w) in acc.iter_mut().zip(&conv.weight[ci * cout..(ci + 1) * cout]) {
                *a += v * w;
            }
        }
    }
}

/// Returns the input planes for a position from the side to move's point of view, square by square.
pub fn encode(board: &Board, repetitions: usize) -> Vec<f32> {
    let mut planes = vec![0.0; 64 * PLANES];
    let black = board.side_to_move() == Color::Black;
    for sq in Square::ALL {
        if let (Some(piece), Some(color)) = (board.piece_on(sq), board.color_on(sq)) {
            let target = if black { sq as usize ^ 56 } else { sq as usize };
            let own = color == board.side_to_move();
            planes[target * PLANES + piece as usize + if own { 0 } else { 6 }] = 1.0;
        }
    }
    let us = board.side_to_move();
    let rights = [board.castle_rights(us).short, board.castle_rights(us).long,
                  board.castle_rights(!us).short, board.castle_rights(!us).long];
    let ep = board.en_passant().map(|f| 40 + f as usize);
    let fifty = (board.halfmove_clock().min(255) as f32 / 100.0).min(1.0);
    for sq in 0..64 {
        let row = &mut planes[sq * PLANES..(sq + 1) * PLANES];
        for (i, right) in rights.iter().enumerate() {
            row[12 + i] = right.is_some() as u8 as f32;
        }
        row[16] = (ep == Some(sq)) as u8 as f32;
        row[17] = black as u8 as f32;
        row[18] = 1.0;
        row[19] = (repetitions >= 1) as u8 as f32;
        row[20] = fifty;
    }
    planes
}

/// Returns a legal move's policy index, flipped to the side to move's view, with castling as the king's two square move.
pub fn policy_index(board: &Board, mv: Move) -> usize {
    let flip = if board.side_to_move() == Color::Black { 56 } else { 0 };
    let mut to = mv.to;
    if board.piece_on(mv.from) == Some(Piece::King) && board.color_on(mv.to) == Some(board.side_to_move()) {
        let file = if mv.to.file() as usize > mv.from.file() as usize { File::G } else { File::C };
        to = Square::new(file, mv.from.rank());
    }
    let (f, t) = (mv.from as usize ^ flip, to as usize ^ flip);
    match mv.promotion {
        Some(piece @ (Piece::Knight | Piece::Bishop | Piece::Rook)) => {
            let code = match piece {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                _ => 2,
            };
            4096 + code * 24 + (f % 8) * 3 + (t % 8 + 1 - f % 8)
        }
        _ => f * 64 + t,
    }
}
