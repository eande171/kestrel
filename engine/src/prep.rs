use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use cozy_chess::{Board, Color, Move, Piece, Square};
use ruzstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};
use ruzstd::decoding::{BlockDecodingStrategy, FrameDecoder};

use crate::game::{parse_san, Game};

pub const RECORD_BYTES: usize = 48;
pub const FORMAT_VERSION: u32 = 3;
pub const FLAG_ENGINE_EVAL: u8 = 1;

#[derive(Debug, Clone)]
pub struct PrepConfig {
    pub input: String,
    pub out_dir: PathBuf,
    pub min_elo: u32,
    pub min_seconds: u32,
    pub min_plies: u32,
    pub skip_plies: u32,
    pub positions: u64,
    pub shards: u32,
    pub val_every: u64,
    pub common_plies: u32,
    pub common_cap: u32,
    pub seed: u64,
}

impl Default for PrepConfig {
    fn default() -> Self {
        PrepConfig {
            input: String::new(),
            out_dir: PathBuf::from("data/shards"),
            min_elo: 1800,
            min_seconds: 180,
            min_plies: 20,
            skip_plies: 8,
            positions: 4_000_000,
            shards: 8,
            val_every: 50,
            common_plies: 24,
            common_cap: 20,
            seed: 1,
        }
    }
}

#[derive(Default, Debug)]
struct Stats {
    games_read: u64,
    games_kept: u64,
    rejected_rating: u64,
    rejected_speed: u64,
    rejected_termination: u64,
    rejected_short: u64,
    rejected_parse: u64,
    positions_train: u64,
    positions_val: u64,
    skipped_common: u64,
    bytes_read: u64,
    truncated: bool,
}

#[derive(Default)]
struct Headers {
    white_elo: Option<u32>,
    black_elo: Option<u32>,
    result: Option<String>,
    time_control: Option<String>,
    termination: Option<String>,
}

// Multi Frame Zstd
struct MultiFrame<R: BufRead> {
    source: R,
    decoder: FrameDecoder,
    in_frame: bool,
}

impl<R: BufRead> MultiFrame<R> {
    fn new(source: R) -> Self {
        MultiFrame { source, decoder: FrameDecoder::new(), in_frame: false }
    }

    fn start_frame(&mut self) -> io::Result<bool> {
        loop {
            if self.source.fill_buf()?.is_empty() {
                return Ok(false);
            }
            self.decoder.set_max_window_size(1 << 31);
            match self.decoder.init(&mut self.source) {
                Ok(()) => return Ok(true),
                Err(FrameDecoderError::ReadFrameHeaderError(ReadFrameHeaderError::SkipFrame { length, .. })) => {
                    io::copy(&mut (&mut self.source).take(length as u64), &mut io::sink())?;
                }
                Err(e) => return Err(io::Error::other(e)),
            }
        }
    }
}

impl<R: BufRead> Read for MultiFrame<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if !self.in_frame {
                if !self.start_frame()? {
                    return Ok(0);
                }
                self.in_frame = true;
            }
            while self.decoder.can_collect() == 0 && !self.decoder.is_finished() {
                self.decoder
                    .decode_blocks(&mut self.source, BlockDecodingStrategy::UptoBlocks(1))
                    .map_err(io::Error::other)?;
            }
            let n = self.decoder.read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            if self.decoder.is_finished() && self.decoder.can_collect() == 0 {
                self.in_frame = false;
            }
        }
    }
}

// Byte Counter
struct Counting<R: Read> {
    inner: R,
    count: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

pub(crate) fn open_input(
    input: &str,
    counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> io::Result<(Box<dyn BufRead>, Option<Child>)> {
    let (raw, child): (Box<dyn Read>, Option<Child>) = if input.starts_with("http://") || input.starts_with("https://") {
        let mut child = Command::new("curl")
            .args(["-sSL", "--fail", input])
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().expect("curl stdout");
        (Box::new(stdout), Some(child))
    } else {
        (Box::new(File::open(input)?), None)
    };
    let counted = BufReader::with_capacity(1 << 20, Counting { inner: raw, count: counter });
    let reader: Box<dyn BufRead> = if input.ends_with(".zst") {
        Box::new(BufReader::with_capacity(1 << 20, MultiFrame::new(counted)))
    } else {
        Box::new(counted)
    };
    Ok((reader, child))
}

pub fn run(cfg: &PrepConfig) -> io::Result<()> {
    fs::create_dir_all(&cfg.out_dir)?;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (mut reader, mut child) = open_input(&cfg.input, counter.clone())?;

    let mut train: Vec<BufWriter<File>> = (0..cfg.shards)
        .map(|i| File::create(cfg.out_dir.join(format!("train_{i:03}.bin"))).map(BufWriter::new))
        .collect::<io::Result<_>>()?;
    let mut val = BufWriter::new(File::create(cfg.out_dir.join("val_000.bin"))?);

    let mut stats = Stats::default();
    let mut common: HashMap<u64, u32> = HashMap::new();
    let mut rng = cfg.seed | 1;
    let mut headers = Headers::default();
    let mut movetext = String::new();
    let mut line = Vec::new();
    let start = Instant::now();
    let mut last_report = Instant::now();

    loop {
        line.clear();
        let n = match reader.read_until(b'\n', &mut line) {
            Ok(n) => n,
            Err(e) if stats.games_read > 0 => {
                eprintln!("[prep] input ended early ({e}), keeping the {} positions so far", stats.positions_train);
                stats.truncated = true;
                break;
            }
            Err(e) => return Err(e),
        };
        let text = String::from_utf8_lossy(&line);
        let trimmed = text.trim();
        let at_end = n == 0;

        if trimmed.starts_with('[') && movetext.is_empty() {
            parse_header(trimmed, &mut headers);
            continue;
        }
        if !trimmed.is_empty() {
            movetext.push_str(trimmed);
            movetext.push(' ');
            continue;
        }
        if !movetext.is_empty() {
            stats.games_read += 1;
            if let Some(game) = filter_and_parse(cfg, &headers, &movetext, &mut stats) {
                stats.games_kept += 1;
                let to_val = cfg.val_every > 0 && stats.games_kept % cfg.val_every == 0;
                for (ply, (board, mv, repetitions)) in game.positions.iter().enumerate() {
                    let ply = ply as u32;
                    if ply < cfg.skip_plies {
                        continue;
                    }
                    if cfg.common_cap > 0 && ply < cfg.common_plies {
                        let count = common.entry(board.hash()).or_insert(0);
                        *count += 1;
                        if *count > cfg.common_cap {
                            stats.skipped_common += 1;
                            continue;
                        }
                    }
                    let record = encode_record(board, *mv, *repetitions, game.white_result, ply, game.white_elo, game.black_elo);
                    if to_val {
                        val.write_all(&record)?;
                        stats.positions_val += 1;
                    } else {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let shard = (rng % cfg.shards as u64) as usize;
                        train[shard].write_all(&record)?;
                        stats.positions_train += 1;
                    }
                }
            }
            headers = Headers::default();
            movetext.clear();
        }
        stats.bytes_read = counter.load(std::sync::atomic::Ordering::Relaxed);
        if last_report.elapsed().as_secs() >= 5 {
            report(&stats, start);
            last_report = Instant::now();
        }
        if at_end || stats.positions_train >= cfg.positions {
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
    write_meta(cfg, &stats, start.elapsed().as_secs_f64())?;
    Ok(())
}

fn report(stats: &Stats, start: Instant) {
    let secs = start.elapsed().as_secs_f64().max(1e-6);
    eprintln!(
        "[prep] {:.0}s games {} kept {} positions train {} val {} | compressed {:.1} MB ({:.1} MB/s)",
        secs,
        stats.games_read,
        stats.games_kept,
        stats.positions_train,
        stats.positions_val,
        stats.bytes_read as f64 / 1e6,
        stats.bytes_read as f64 / 1e6 / secs
    );
}

fn write_meta(cfg: &PrepConfig, s: &Stats, seconds: f64) -> io::Result<()> {
    let json = format!(
        "{{\n  \"input\": \"{}\",\n  \"format_version\": {FORMAT_VERSION},\n  \"record_bytes\": {RECORD_BYTES},\n  \
         \"min_elo\": {},\n  \"min_seconds\": {},\n  \
         \"min_plies\": {},\n  \"skip_plies\": {},\n  \"common_plies\": {},\n  \"common_cap\": {},\n  \
         \"val_every\": {},\n  \"shards\": {},\n  \"games_read\": {},\n  \"games_kept\": {},\n  \
         \"rejected_rating\": {},\n  \"rejected_speed\": {},\n  \"rejected_termination\": {},\n  \
         \"rejected_short\": {},\n  \"rejected_parse\": {},\n  \"skipped_common\": {},\n  \
         \"positions_train\": {},\n  \"positions_val\": {},\n  \"compressed_bytes_read\": {},\n  \"truncated\": {},\n  \
         \"seconds\": {:.1}\n}}\n",
        cfg.input.replace('\\', "/"),
        cfg.min_elo,
        cfg.min_seconds,
        cfg.min_plies,
        cfg.skip_plies,
        cfg.common_plies,
        cfg.common_cap,
        cfg.val_every,
        cfg.shards,
        s.games_read,
        s.games_kept,
        s.rejected_rating,
        s.rejected_speed,
        s.rejected_termination,
        s.rejected_short,
        s.rejected_parse,
        s.skipped_common,
        s.positions_train,
        s.positions_val,
        s.bytes_read,
        s.truncated,
        seconds
    );
    fs::write(cfg.out_dir.join("meta.json"), json)
}

fn parse_header(line: &str, h: &mut Headers) {
    let inner = line.trim_start_matches('[').trim_end_matches(']');
    let Some((key, rest)) = inner.split_once(' ') else { return };
    let value = rest.trim().trim_matches('"').to_string();
    match key {
        "WhiteElo" => h.white_elo = value.parse().ok(),
        "BlackElo" => h.black_elo = value.parse().ok(),
        "Result" => h.result = Some(value),
        "TimeControl" => h.time_control = Some(value),
        "Termination" => h.termination = Some(value),
        _ => {}
    }
}

struct ParsedGame {
    positions: Vec<(Board, Move, u8)>,
    white_result: u8,
    white_elo: u16,
    black_elo: u16,
}

// Estimated Seconds
fn estimated_seconds(tc: &str) -> Option<u32> {
    let (base, inc) = tc.split_once('+')?;
    Some(base.parse::<u32>().ok()? + 40 * inc.parse::<u32>().ok()?)
}

fn filter_and_parse(cfg: &PrepConfig, h: &Headers, movetext: &str, stats: &mut Stats) -> Option<ParsedGame> {
    let white_result = match h.result.as_deref() {
        Some("1-0") => 2,
        Some("1/2-1/2") => 1,
        Some("0-1") => 0,
        _ => {
            stats.rejected_termination += 1;
            return None;
        }
    };
    if h.termination.as_deref() != Some("Normal") {
        stats.rejected_termination += 1;
        return None;
    }
    let (Some(we), Some(be)) = (h.white_elo, h.black_elo) else {
        stats.rejected_rating += 1;
        return None;
    };
    if we < cfg.min_elo || be < cfg.min_elo {
        stats.rejected_rating += 1;
        return None;
    }
    match h.time_control.as_deref().and_then(estimated_seconds) {
        Some(s) if s >= cfg.min_seconds => {}
        _ => {
            stats.rejected_speed += 1;
            return None;
        }
    }

    let mut game = Game::startpos();
    let mut positions = Vec::with_capacity(128);
    for token in san_tokens(movetext) {
        let Ok(mv) = parse_san(&game.board, token) else {
            stats.rejected_parse += 1;
            return None;
        };
        positions.push((game.board.clone(), mv, game.repetitions().min(2) as u8));
        game.play(mv).ok()?;
    }
    if (positions.len() as u32) < cfg.min_plies {
        stats.rejected_short += 1;
        return None;
    }
    Some(ParsedGame {
        positions,
        white_result,
        white_elo: we.min(u16::MAX as u32) as u16,
        black_elo: be.min(u16::MAX as u32) as u16,
    })
}

// Movetext Tokens
fn san_tokens(movetext: &str) -> impl Iterator<Item = &str> {
    let mut depth = 0i32;
    let mut cleaned: Vec<&str> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, ch) in movetext.char_indices() {
        let boundary = ch.is_whitespace() || ch == '{' || ch == '}' || ch == '(' || ch == ')';
        if boundary {
            if let Some(s) = start.take() {
                if depth == 0 {
                    cleaned.push(&movetext[s..i]);
                }
            }
            match ch {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                _ => {}
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        if depth == 0 {
            cleaned.push(&movetext[s..]);
        }
    }
    cleaned.into_iter().filter_map(|t| {
        if matches!(t, "1-0" | "0-1" | "1/2-1/2" | "*") {
            return None;
        }
        let t = t.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        (!t.is_empty() && !t.starts_with('$')).then_some(t)
    })
}

fn piece_code(board: &Board, sq: Square) -> u8 {
    match (board.piece_on(sq), board.color_on(sq)) {
        (Some(p), Some(Color::White)) => 1 + p as u8,
        (Some(p), Some(Color::Black)) => 7 + p as u8,
        _ => 0,
    }
}

// Standard Castling Squares
pub fn standard_move(board: &Board, mv: Move) -> Move {
    if board.piece_on(mv.from) == Some(Piece::King) && board.color_on(mv.to) == Some(board.side_to_move()) {
        let file = if mv.to.file() > mv.from.file() { cozy_chess::File::G } else { cozy_chess::File::C };
        return Move { from: mv.from, to: Square::new(file, mv.from.rank()), promotion: None };
    }
    mv
}

// Record Layout
pub fn encode_record(
    board: &Board,
    mv: Move,
    repetitions: u8,
    white_result: u8,
    ply: u32,
    white_elo: u16,
    black_elo: u16,
) -> [u8; RECORD_BYTES] {
    let mut r = [0u8; RECORD_BYTES];
    for i in 0..32 {
        let lo = piece_code(board, Square::index(2 * i));
        let hi = piece_code(board, Square::index(2 * i + 1));
        r[i] = lo | (hi << 4);
    }
    let stm = board.side_to_move();
    r[32] = if stm == Color::White { 0 } else { 1 };
    let w = board.castle_rights(Color::White);
    let b = board.castle_rights(Color::Black);
    r[33] = (w.short.is_some() as u8) | (w.long.is_some() as u8) << 1 | (b.short.is_some() as u8) << 2 | (b.long.is_some() as u8) << 3;
    r[34] = board.en_passant().map_or(255, |f| f as u8);
    r[35] = if stm == Color::White { white_result } else { 2 - white_result };
    let std_mv = standard_move(board, mv);
    r[36] = std_mv.from as u8;
    r[37] = std_mv.to as u8;
    r[38] = std_mv.promotion.map_or(0, |p| p as u8);
    r[39] = ply.min(255) as u8;
    let (mover, other) = if stm == Color::White { (white_elo, black_elo) } else { (black_elo, white_elo) };
    r[40..42].copy_from_slice(&mover.to_le_bytes());
    r[42..44].copy_from_slice(&other.to_le_bytes());
    r[44] = repetitions;
    r[45] = board.halfmove_clock();
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_skip_comments_and_numbers() {
        let text = "1. e4 { [%clk 0:03:00] } 1... e5 { [%eval 0.2] } 2. Nf3?! $6 Nc6 1-0";
        let tokens: Vec<&str> = san_tokens(text).collect();
        assert_eq!(tokens, vec!["e4", "e5", "Nf3?!", "Nc6"]);
    }

    #[test]
    fn castling_is_stored_as_king_two_squares() {
        let board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false).unwrap();
        let mv = parse_san(&board, "O-O").unwrap();
        let r = encode_record(&board, mv, 0, 1, 0, 2000, 2000);
        assert_eq!(r[36], Square::E1 as u8);
        assert_eq!(r[37], Square::G1 as u8);
        assert_eq!(r[33], 15);
    }

    #[test]
    fn repetitions_are_counted_within_a_game() {
        let cfg = PrepConfig { min_elo: 0, min_plies: 0, ..PrepConfig::default() };
        let headers = Headers {
            white_elo: Some(2000),
            black_elo: Some(2000),
            result: Some("1/2-1/2".to_string()),
            time_control: Some("600+0".to_string()),
            termination: Some("Normal".to_string()),
        };
        let text = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. e4 1/2-1/2";
        let game = filter_and_parse(&cfg, &headers, text, &mut Stats::default()).unwrap();
        let reps: Vec<u8> = game.positions.iter().map(|p| p.2).collect();
        assert_eq!(reps, vec![0, 0, 0, 0, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn estimated_seconds_matches_lichess_categories() {
        assert_eq!(estimated_seconds("60+0"), Some(60));
        assert_eq!(estimated_seconds("180+2"), Some(260));
        assert_eq!(estimated_seconds("-"), None);
    }
}
