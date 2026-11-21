// Int8 Convolution

use std::cell::RefCell;

thread_local! {
    static SCRATCH: RefCell<(Vec<u8>, Vec<i32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

pub struct QConv {
    weight: Vec<i16>,
    quads: Vec<i8>,
    vnni: bool,
    scale: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
    outputs: usize,
}

impl QConv {
    // Quantize Weights
    pub fn new(weight: &[f32], bias: &[f32], inputs: usize, outputs: usize) -> Option<QConv> {
        if inputs % 2 != 0 || outputs % 8 != 0 {
            return None;
        }
        let scale: Vec<f32> = (0..outputs)
            .map(|co| {
                let top = (0..9 * inputs).map(|i| weight[i * outputs + co].abs()).fold(0.0f32, f32::max);
                if top > 0.0 { top / 127.0 } else { 1.0 }
            })
            .collect();
        let mut packed = vec![0i16; 9 * inputs * outputs];
        let mut quads = vec![0i8; 9 * inputs * outputs];
        for tap in 0..9 {
            for ci in 0..inputs {
                for co in 0..outputs {
                    let w = (weight[(tap * inputs + ci) * outputs + co] / scale[co]).round().clamp(-127.0, 127.0);
                    packed[((tap * (inputs / 2) + ci / 2) * outputs + co) * 2 + ci % 2] = w as i16;
                    quads[((tap * (inputs / 4) + ci / 4) * outputs + co) * 4 + ci % 4] = w as i8;
                }
            }
        }
        let vnni = inputs % 4 == 0 && matches!(outputs, 32 | 64 | 96) && vnni_available();
        Some(QConv { weight: packed, quads, vnni, scale, bias: bias.to_vec(), inputs, outputs })
    }

    // Inputs Are Never Negative
    pub fn apply(&self, input: &[f32], out: &mut [f32]) {
        let top = input.iter().fold(0.0f32, |a, &b| a.max(b));
        let act_scale = if top > 0.0 { top / 255.0 } else { 1.0 };
        let inverse = 1.0 / act_scale;
        SCRATCH.with_borrow_mut(|(bytes, acc)| {
            acc.resize(64 * self.outputs, 0);
            if self.vnni {
                if bytes.len() != 100 * self.inputs {
                    *bytes = vec![0; 100 * self.inputs];
                }
                for (sq, square) in input.chunks_exact(self.inputs).enumerate() {
                    let start = padded(sq) * self.inputs;
                    for (dest, &x) in bytes[start..start + self.inputs].iter_mut().zip(square) {
                        *dest = (x * inverse + 0.5) as u8;
                    }
                }
                accumulate_vnni(&self.quads, bytes, self.inputs / 4, self.outputs, acc);
            } else {
                let level = |x: f32| (x * inverse + 0.5) as i32;
                let pairs: Vec<i32> = input.chunks_exact(2).map(|p| level(p[0]) | (level(p[1]) << 16)).collect();
                self.accumulate(&pairs, acc);
            }
            for (row, dest) in acc.chunks_exact(self.outputs).zip(out.chunks_exact_mut(self.outputs)) {
                for (((d, &total), &scale), &bias) in dest.iter_mut().zip(row).zip(&self.scale).zip(&self.bias) {
                    *d = total as f32 * (act_scale * scale) + bias;
                }
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn accumulate(&self, pairs: &[i32], acc: &mut [i32]) {
        use std::arch::x86_64::*;
        let (half, cout) = (self.inputs / 2, self.outputs);
        let lanes = cout / 8;
        for (sq, total) in acc.chunks_exact_mut(cout).enumerate() {
            let mut sums = [unsafe { _mm256_setzero_si256() }; 12];
            for (tap, nsq) in neighbours(sq) {
                let row = &pairs[nsq * half..(nsq + 1) * half];
                let taps = &self.weight[tap * half * cout * 2..(tap + 1) * half * cout * 2];
                for (p, &pair) in row.iter().enumerate() {
                    let w = &taps[p * cout * 2..(p + 1) * cout * 2];
                    unsafe {
                        let a = _mm256_set1_epi32(pair);
                        for (k, sum) in sums.iter_mut().take(lanes).enumerate() {
                            let wv = _mm256_loadu_si256(w.as_ptr().add(k * 16) as *const __m256i);
                            *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(a, wv));
                        }
                    }
                }
            }
            for (k, sum) in sums.iter().take(lanes).enumerate() {
                unsafe { _mm256_storeu_si256(total.as_mut_ptr().add(k * 8) as *mut __m256i, *sum) };
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn accumulate(&self, pairs: &[i32], acc: &mut [i32]) {
        use std::arch::wasm32::*;
        let (half, cout) = (self.inputs / 2, self.outputs);
        let lanes = cout / 4;
        for (sq, total) in acc.chunks_exact_mut(cout).enumerate() {
            let mut sums = [i32x4_splat(0); 24];
            for (tap, nsq) in neighbours(sq) {
                let row = &pairs[nsq * half..(nsq + 1) * half];
                let taps = &self.weight[tap * half * cout * 2..(tap + 1) * half * cout * 2];
                for (p, &pair) in row.iter().enumerate() {
                    if pair == 0 {
                        continue;
                    }
                    let w = &taps[p * cout * 2..(p + 1) * cout * 2];
                    let a = i32x4_splat(pair);
                    for (k, sum) in sums.iter_mut().take(lanes).enumerate() {
                        let wv = unsafe { v128_load(w.as_ptr().add(k * 8) as *const v128) };
                        *sum = i32x4_add(*sum, i32x4_dot_i16x8(a, wv));
                    }
                }
            }
            for (k, sum) in sums.iter().take(lanes).enumerate() {
                unsafe { v128_store(total.as_mut_ptr().add(k * 4) as *mut v128, *sum) };
            }
        }
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"),
                  all(target_arch = "wasm32", target_feature = "simd128"))))]
    fn accumulate(&self, pairs: &[i32], acc: &mut [i32]) {
        let (half, cout) = (self.inputs / 2, self.outputs);
        for (sq, total) in acc.chunks_exact_mut(cout).enumerate() {
            for (tap, nsq) in neighbours(sq) {
                for p in 0..half {
                    let pair = pairs[nsq * half + p];
                    let (lo, hi) = (pair as i16 as i32, pair >> 16);
                    let w = &self.weight[(tap * half + p) * cout * 2..(tap * half + p + 1) * cout * 2];
                    for (co, t) in total.iter_mut().enumerate() {
                        *t += lo * w[co * 2] as i32 + hi * w[co * 2 + 1] as i32;
                    }
                }
            }
        }
    }
}

// Vector Neural Network Instructions
#[cfg(target_arch = "x86_64")]
fn vnni_available() -> bool {
    std::env::var_os("KESTREL_NO_VNNI").is_none() && is_x86_feature_detected!("avxvnni")
}

#[cfg(not(target_arch = "x86_64"))]
fn vnni_available() -> bool {
    false
}

// Padded Board
fn padded(sq: usize) -> usize {
    (sq / 8 + 1) * 10 + sq % 8 + 1
}

#[cfg(target_arch = "x86_64")]
fn accumulate_vnni(weight: &[i8], bytes: &[u8], quarter: usize, cout: usize, acc: &mut [i32]) {
    assert!(weight.len() >= 9 * quarter * cout * 4 && bytes.len() >= 400 * quarter && acc.len() >= 64 * cout);
    unsafe {
        match cout {
            32 => vnni_kernel::<4>(weight, bytes, quarter, acc),
            64 => vnni_kernel::<8>(weight, bytes, quarter, acc),
            96 => vnni_kernel::<12>(weight, bytes, quarter, acc),
            _ => unreachable!(),
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn accumulate_vnni(_: &[i8], _: &[u8], _: usize, _: usize, _: &mut [i32]) {
    unreachable!()
}

// Register Accumulators
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vnni_kernel<const L: usize>(weight: &[i8], bytes: &[u8], quarter: usize, acc: &mut [i32]) {
    use std::arch::x86_64::*;
    let cout = L * 8;
    unsafe {
        for sq in 0..64 {
            let (rank, file) = (sq / 8, sq % 8);
            let mut sums = [_mm256_setzero_si256(); L];
            for tap in 0..9 {
                let row = bytes.as_ptr().add(((rank + tap / 3) * 10 + file + tap % 3) * quarter * 4) as *const i32;
                let taps = weight.as_ptr().add(tap * quarter * cout * 4);
                for p in 0..quarter {
                    let a = _mm256_set1_epi32(row.add(p).read_unaligned());
                    let w = taps.add(p * cout * 4);
                    for (k, sum) in sums.iter_mut().enumerate() {
                        *sum = _mm256_dpbusd_avx_epi32(*sum, a, _mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    }
                }
            }
            for (k, sum) in sums.iter().enumerate() {
                _mm256_storeu_si256(acc.as_mut_ptr().add(sq * cout + k * 8) as *mut __m256i, *sum);
            }
        }
    }
}

pub fn neighbours(sq: usize) -> impl Iterator<Item = (usize, usize)> {
    let (rank, file) = ((sq / 8) as isize, (sq % 8) as isize);
    (0..9).filter_map(move |tap| {
        let (r, f) = (rank + tap as isize / 3 - 1, file + tap as isize % 3 - 1);
        ((0..8).contains(&r) && (0..8).contains(&f)).then_some((tap, (r * 8 + f) as usize))
    })
}
