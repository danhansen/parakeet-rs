//! NeMo's inference-time Nemotron frontend on a global sample/frame grid.
//!
//! Evaluation dither is disabled. Only complete centered windows are emitted
//! during streaming. `finish` emits the valid right-edge frames and NeMo's
//! final, masked (zero) frame. Synthetic encoder padding is not raw audio.
use crate::error::{Error, Result};
use ndarray::{Array1, Array2};
use realfft::{num_complex::Complex32, RealToComplex};
use std::collections::VecDeque;
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const PRE_ENCODE_CACHE: usize = 9;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

pub struct NemotronFrontend {
    mel_basis: Arc<Array2<f32>>,
    plan: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    powers: Array1<f32>,
    audio: Vec<f32>,
    audio_base: usize,
    total_samples: usize,
    next_frame: usize,
    finished: bool,
}

/// Fixed-shape transport of NeMo's cache-aware feature chunks. `input_frames`
/// excludes transport padding and must be kept separate from the tensor shape.
pub struct NemotronMelChunk {
    pub features: ndarray::Array3<f32>,
    pub input_frames: usize,
}

pub struct NemotronMelChunker {
    chunk_frames: usize,
    pending: VecDeque<Vec<f32>>,
    emitted: usize,
    finished: bool,
}

impl NemotronMelChunker {
    pub fn new(chunk_frames: usize) -> Self {
        assert!(chunk_frames > 0);
        Self {
            chunk_frames,
            pending: VecDeque::from(vec![vec![0.; N_MELS]; PRE_ENCODE_CACHE]),
            emitted: 0,
            finished: false,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new(self.chunk_frames);
    }
    pub fn push(&mut self, frames: Vec<Vec<f32>>) -> Result<Vec<NemotronMelChunk>> {
        if self.finished {
            return Err(Error::Audio(
                "features supplied after chunk finalization".into(),
            ));
        }
        if frames.iter().any(|frame| frame.len() != N_MELS) {
            return Err(Error::Audio("incorrect mel feature width".into()));
        }
        self.pending.extend(frames);
        let mut chunks = Vec::new();
        while self.pending.len() >= PRE_ENCODE_CACHE + self.chunk_frames {
            chunks.push(self.take_chunk(self.chunk_frames));
        }
        Ok(chunks)
    }
    pub fn finish(&mut self) -> Vec<NemotronMelChunk> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let remaining = self.pending.len() - PRE_ENCODE_CACHE;
        // CacheAwareStreamingAudioBuffer's causal dw_striding sampling_frames
        // are [1, 8]: the first slice can be short; later slices require 8.
        if remaining > 0 && (self.emitted == 0 || remaining >= 8) {
            vec![self.take_chunk(remaining)]
        } else {
            Vec::new()
        }
    }
    fn take_chunk(&mut self, new_frames: usize) -> NemotronMelChunk {
        let width = PRE_ENCODE_CACHE + self.chunk_frames;
        let mut features = ndarray::Array3::<f32>::zeros((1, N_MELS, width));
        for (f, frame) in self
            .pending
            .iter()
            .take(PRE_ENCODE_CACHE + new_frames)
            .enumerate()
        {
            for m in 0..N_MELS {
                features[[0, m, f]] = frame[m];
            }
        }
        self.pending.drain(..new_frames);
        self.emitted += 1;
        NemotronMelChunk {
            features,
            input_frames: PRE_ENCODE_CACHE + new_frames,
        }
    }
}

impl NemotronFrontend {
    pub const SAMPLE_RATE: usize = SAMPLE_RATE;

    /// Use checkpoint-provided window coefficients when available. This also
    /// permits parity tests to isolate FFT/reduction error from window rounding.
    pub fn with_window(mel_basis: Arc<Array2<f32>>, window: &[f32]) -> Result<Self> {
        if window.len() != WIN_LENGTH || window.iter().any(|v| !v.is_finite()) {
            return Err(Error::Config("invalid Nemotron window coefficients".into()));
        }
        let mut frontend = Self::new(mel_basis);
        frontend.window.copy_from_slice(window);
        Ok(frontend)
    }
    pub fn new(mel_basis: Arc<Array2<f32>>) -> Self {
        assert_eq!(mel_basis.shape(), &[N_MELS, N_FFT / 2 + 1]);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(N_FFT);
        // Torch's hamming_window(alpha=beta=0.5) first converts the F64
        // angular increment to the tensor dtype, then multiplies each index.
        // Computing (2*pi*i)/(N-1) instead adds an extra FP32 rounding step.
        let increment = (2.0 * std::f64::consts::PI / (WIN_LENGTH - 1) as f64) as f32;
        Self {
            mel_basis,
            spectrum: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            plan,
            window: (0..WIN_LENGTH)
                .map(|i| 0.5 - 0.5 * (i as f32 * increment).cos())
                .collect(),
            input: vec![0.; N_FFT],
            powers: Array1::zeros(N_FFT / 2 + 1),
            audio: Vec::new(),
            audio_base: 0,
            total_samples: 0,
            next_frame: 0,
            finished: false,
        }
    }

    pub fn reset(&mut self) {
        self.audio.clear();
        self.audio_base = 0;
        self.total_samples = 0;
        self.next_frame = 0;
        self.finished = false;
    }

    pub fn sample_count(&self) -> usize {
        self.total_samples
    }
    pub fn buffered_samples(&self) -> usize {
        self.audio.len()
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if self.finished {
            return Err(Error::Audio(
                "audio supplied after frontend finalization; reset first".into(),
            ));
        }
        self.audio.extend_from_slice(samples);
        self.total_samples += samples.len();
        let mut frames = Vec::new();
        // Samples outside the 400-sample Hann window have zero weight. We need
        // 200 real samples to the right, not arbitrary chunk-end zero padding.
        while self.next_frame * HOP_LENGTH + WIN_LENGTH / 2 <= self.total_samples {
            frames.push(self.compute_frame()?);
            self.next_frame += 1;
        }
        self.trim();
        Ok(frames)
    }

    pub fn finish(&mut self) -> Result<Vec<Vec<f32>>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut frames = Vec::new();
        let valid_frames = self.total_samples / HOP_LENGTH;
        while self.next_frame < valid_frames {
            frames.push(self.compute_frame()?);
            self.next_frame += 1;
        }
        // torch.stft(center=True) returns floor(samples / hop) + 1 frames;
        // FilterbankFeatures masks everything beyond the valid length to zero.
        frames.push(vec![0.; N_MELS]);
        self.finished = true;
        self.audio.clear();
        self.audio_base = self.total_samples;
        Ok(frames)
    }

    fn compute_frame(&mut self) -> Result<Vec<f32>> {
        self.input.fill(0.);
        let center = self.next_frame * HOP_LENGTH;
        let fft_offset = (N_FFT - WIN_LENGTH) / 2;
        for i in 0..WIN_LENGTH {
            let index = center as i64 + i as i64 - (WIN_LENGTH / 2) as i64;
            let sample = if index < 0 || index as usize >= self.total_samples {
                0.
            } else {
                let index = index as usize;
                let value = self.audio[index - self.audio_base];
                if index == 0 {
                    value
                } else {
                    value - PREEMPH * self.audio[index - 1 - self.audio_base]
                }
            };
            self.input[fft_offset + i] = sample * self.window[i];
        }
        self.plan
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;
        for (power, value) in self.powers.iter_mut().zip(&self.spectrum) {
            *power = value.norm_sqr();
        }
        Ok(self
            .mel_basis
            .dot(&self.powers)
            .mapv(|v| (v + LOG_ZERO_GUARD).ln())
            .to_vec())
    }

    fn trim(&mut self) {
        // Keep the previous RAW sample too, for preemphasis at the next frame's
        // left edge. Storage movement never changes the global frame cursor.
        let keep_from = (self.next_frame * HOP_LENGTH).saturating_sub(WIN_LENGTH / 2 + 1);
        let remove = keep_from
            .saturating_sub(self.audio_base)
            .min(self.audio.len());
        self.audio.drain(..remove);
        self.audio_base += remove;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frontend() -> NemotronFrontend {
        NemotronFrontend::new(Arc::new(crate::audio::create_mel_filterbank(
            N_FFT,
            N_MELS,
            SAMPLE_RATE,
        )))
    }
    #[test]
    fn arbitrary_packetization_is_bit_exact_and_storage_is_bounded() {
        let samples: Vec<f32> = (0..32_001).map(|i| (i as f32 * 0.17).sin() * 0.1).collect();
        let mut whole = frontend();
        let mut expected = whole.push(&samples).unwrap();
        expected.extend(whole.finish().unwrap());
        for packet in [1, 159, 160, 257, 1280, 2560, 8960, 17920] {
            let mut streaming = frontend();
            let mut actual = Vec::new();
            for part in samples.chunks(packet) {
                actual.extend(streaming.push(part).unwrap());
                assert!(streaming.buffered_samples() <= WIN_LENGTH + HOP_LENGTH);
            }
            actual.extend(streaming.finish().unwrap());
            assert_eq!(actual, expected, "packet={packet}");
            assert_eq!(actual.len(), samples.len() / HOP_LENGTH + 1);
        }
    }
    #[test]
    fn finish_is_idempotent_and_reset_clears_stream_state() {
        let mut frontend = frontend();
        assert!(frontend.push(&[0.; 199]).unwrap().is_empty());
        let frames = frontend.finish().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames.last().unwrap(), &vec![0.; N_MELS]);
        assert!(frontend.finish().unwrap().is_empty());
        assert!(frontend.push(&[0.]).is_err());
        frontend.reset();
        assert_eq!(frontend.finish().unwrap(), vec![vec![0.; N_MELS]]);
    }
    #[test]
    fn chunking_preserves_short_history_lengths_and_transport_padding() {
        for width in [8, 16, 56, 112] {
            for length in [1, 7, 8, 9, 15, 16, 17, 113, 227] {
                let frames: Vec<Vec<f32>> =
                    (0..length).map(|i| vec![(i + 1) as f32; N_MELS]).collect();
                let mut chunker = NemotronMelChunker::new(width);
                let mut chunks = Vec::new();
                for packet in frames.chunks(3) {
                    chunks.extend(chunker.push(packet.to_vec()).unwrap());
                }
                chunks.extend(chunker.finish());
                for (index, chunk) in chunks.iter().enumerate() {
                    let start = index * width;
                    let main = (length - start).min(width);
                    assert_eq!(chunk.input_frames, PRE_ENCODE_CACHE + main);
                    for column in 0..PRE_ENCODE_CACHE + width {
                        let frame = start as i64 + column as i64 - PRE_ENCODE_CACHE as i64;
                        let value = if frame < 0 || column >= chunk.input_frames {
                            0.
                        } else {
                            (frame + 1) as f32
                        };
                        assert!(chunk
                            .features
                            .slice(ndarray::s![0, .., column])
                            .iter()
                            .all(|&v| v == value));
                    }
                }
                assert!(chunker.finish().is_empty());
                assert!(chunker.push(vec![vec![0.; N_MELS]]).is_err());
                chunker.reset();
                assert!(chunker.push(vec![vec![0.; 3]]).is_err());
            }
        }
    }
}
