use std::fs::{self, File};
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use cozy_chess::util::parse_uci_move;
use cozy_chess::{Board, Color, Move};
use serde_json::Value;

use crate::prep::{encode_record, open_input, FLAG_ENGINE_EVAL, FORMAT_VERSION, RECORD_BYTES};

// Lichess Win Model
const WIN_SCALE: f64 = 0.00368208;

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub input: String,
    pub out_dir: PathBuf,
    pub positions: u64,
    pub keep_percent: f64,
    pub shards: u32,
    pub val_every: u64,
    pub seed: u64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig {
            input: String::new(),
            out_dir: PathBuf::from("data/evals"),
            positions: 30_000_000,
            keep_percent: 8.0,
            shards: 8,
            val_every: 50,
            seed: 1,
        }
    }
}

#[derive(Default, Debug)]
struct EvalStats {
    lines: u64,
    sampled: u64,
    rejected: u64,
    positions_train: u64,
    positions_val: u64,
    bytes_read: u64,
}

pub fn expected_score(cp: Option<i64>, mate: Option<i64>) -> Option<f64> {
    match (cp, mate) {
        (Some(cp), _) => Some(1.0 / (1.0 + (-WIN_SCALE * cp as f64).exp())),
        (None, Some(m)) => Some(if m > 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

// Eval Line
pub fn parse_eval_line(line: &[u8]) -> Option<(Board, Move, f64)> {
    let v: Value = serde_json::from_slice(line).ok()?;
    let fen = v.get("fen")?.as_str()?;
    let board = Board::from_fen(&format!("{fen} 0 1"), false).ok()?;
    let deepest = v
        .get("evals")?
        .as_array()?
        .iter()
        .max_by_key(|e| e.get("depth").and_then(Value::as_u64).unwrap_or(0))?;
    let pv = deepest.get("pvs")?.as_array()?.first()?;
    let first = pv.get("line")?.as_str()?.split(' ').next()?;
    let mv = first
        .parse::<Move>()
        .ok()
        .filter(|&m| board.is_legal(m))
        .or_else(|| parse_uci_move(&board, first).ok().filter(|&m| board.is_legal(m)))?;
    let white_score = expected_score(pv.get("cp").and_then(Value::as_i64), pv.get("mate").and_then(Value::as_i64))?;
    Some((board, mv, white_score))
}

pub fn encode_eval_record(board: &Board, mv: Move, white_score: f64) -> [u8; RECORD_BYTES] {
    let mover_score = if board.side_to_move() == Color::White { white_score } else { 1.0 - white_score };
    let white_result = match white_score {
        s if s > 0.6 => 2,
        s if s < 0.4 => 0,
        _ => 1,
    };
    let mut r = encode_record(board, mv, 0, white_result, 0, 0, 0);
    r[46] = FLAG_ENGINE_EVAL;
    r[47] = (mover_score * 255.0).round().clamp(0.0, 255.0) as u8;
    r
}

pub fn run(cfg: &EvalConfig) -> io::Result<()> {
    fs::create_dir_all(&cfg.out_dir)?;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (mut reader, mut child) = open_input(&cfg.input, counter.clone())?;
    let mut train: Vec<BufWriter<File>> = (0..cfg.shards)
        .map(|i| File::create(cfg.out_dir.join(format!("train_{i:03}.bin"))).map(BufWriter::new))
        .collect::<io::Result<_>>()?;
    let mut val = BufWriter::new(File::create(cfg.out_dir.join("val_000.bin"))?);

    let keep_below = (cfg.keep_percent / 100.0 * u64::MAX as f64) as u64;
    let mut rng = cfg.seed | 1;
    let mut stats = EvalStats::default();
    let mut line = Vec::new();
    let start = Instant::now();
    let mut last_report = Instant::now();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) if stats.lines > 0 => {
                eprintln!("[evals] input ended early ({e}), keeping the {} positions so far", stats.positions_train);
                break;
            }
            Err(e) => return Err(e),
        }
        stats.lines += 1;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        if rng > keep_below {
            continue;
        }
        stats.sampled += 1;
        let Some((board, mv, white_score)) = parse_eval_line(&line) else {
            stats.rejected += 1;
            continue;
        };
        let record = encode_eval_record(&board, mv, white_score);
        if cfg.val_every > 0 && stats.sampled % cfg.val_every == 0 {
            val.write_all(&record)?;
            stats.positions_val += 1;
        } else {
            train[(rng % cfg.shards as u64) as usize].write_all(&record)?;
            stats.positions_train += 1;
        }
        if last_report.elapsed().as_secs() >= 5 {
            stats.bytes_read = counter.load(std::sync::atomic::Ordering::Relaxed);
            report(&stats, start);
            last_report = Instant::now();
        }
        if stats.positions_train >= cfg.positions {
            break;
        }
    }
    for w in train.iter_mut() {
        w.flush()?;
    }
    val.flush()?;
    drop(reader);
    if let Some(mut c) = child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    stats.bytes_read = counter.load(std::sync::atomic::Ordering::Relaxed);
    report(&stats, start);
    let json = format!(
        "{{\n  \"input\": \"{}\",\n  \"source\": \"lichess_eval\",\n  \"format_version\": {FORMAT_VERSION},\n  \
         \"record_bytes\": {RECORD_BYTES},\n  \"keep_percent\": {},\n  \"lines_read\": {},\n  \"sampled\": {},\n  \
         \"rejected\": {},\n  \"positions_train\": {},\n  \"positions_val\": {},\n  \"compressed_bytes_read\": {},\n  \
         \"seconds\": {:.1}\n}}\n",
        cfg.input.replace('\\', "/"),
        cfg.keep_percent,
        stats.lines,
        stats.sampled,
        stats.rejected,
        stats.positions_train,
        stats.positions_val,
        stats.bytes_read,
        start.elapsed().as_secs_f64()
    );
    fs::write(cfg.out_dir.join("meta.json"), json)
}

fn report(stats: &EvalStats, start: Instant) {
    let secs = start.elapsed().as_secs_f64().max(1e-6);
    eprintln!(
        "[evals] {:.0}s lines {} sampled {} rejected {} train {} val {} | compressed {:.1} MB ({:.1} MB/s)",
        secs,
        stats.lines,
        stats.sampled,
        stats.rejected,
        stats.positions_train,
        stats.positions_val,
        stats.bytes_read as f64 / 1e6,
        stats.bytes_read as f64 / 1e6 / secs
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_deepest_eval_and_converts_castling() {
        let line = br#"{"fen":"r3k2r/8/8/8/8/8/8/R3K2R w KQkq -","evals":[{"pvs":[{"cp":20,"line":"a1a2 a8a7"}],"knodes":10,"depth":12},{"pvs":[{"cp":311,"line":"e1h1 e8h8"}],"knodes":900,"depth":30}]}"#;
        let (board, mv, white_score) = parse_eval_line(line).unwrap();
        assert!(board.is_legal(mv));
        let r = encode_eval_record(&board, mv, white_score);
        assert_eq!(r[36], cozy_chess::Square::E1 as u8);
        assert_eq!(r[37], cozy_chess::Square::G1 as u8);
        assert_eq!(r[46], FLAG_ENGINE_EVAL);
        assert!(r[47] > 190);
    }

    #[test]
    fn mate_scores_flip_for_black_to_move() {
        let line = br#"{"fen":"6k1/5ppp/8/8/8/8/8/R5K1 b - -","evals":[{"pvs":[{"mate":-1,"line":"g8f8"}],"knodes":5,"depth":20}]}"#;
        let (board, mv, white_score) = parse_eval_line(line).unwrap();
        assert_eq!(white_score, 0.0);
        let r = encode_eval_record(&board, mv, white_score);
        assert_eq!(r[47], 255);
    }
}
