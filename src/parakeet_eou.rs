use crate::android_log;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_eou::{EncoderCache, ParakeetEOUModel};
use ndarray::{linalg::general_mat_mul, s, Array2, Array3};
use realfft::{num_complex::Complex32, RealFftPlanner, RealToComplex};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const SAMPLE_RATE: usize = 16000;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const FMAX: f32 = 8000.0;
const PRE_ENCODE_CACHE: usize = 9;
const FRAMES_PER_CHUNK: usize = 16;
const SLICE_LEN: usize = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK;
const FREQ_BINS: usize = N_FFT / 2 + 1;
const FEATURE_WINDOW_SAMPLES: usize = (FRAMES_PER_CHUNK - 1) * HOP_LENGTH + WIN_LENGTH;
const STFT_FRAMES: usize = 1 + (FEATURE_WINDOW_SAMPLES + N_FFT - WIN_LENGTH) / HOP_LENGTH;

fn required_audio_samples(num_frames: usize) -> usize {
    if num_frames == 0 {
        return 0;
    }
    (num_frames - 1) * HOP_LENGTH + WIN_LENGTH
}

/// Parakeet RealTime EOU model for streaming ASR with end-of-utterance detection.
/// Uses cache-aware streaming with a bounded raw-audio history just large enough
/// to regenerate the next encoder input window.
pub struct ParakeetEOU {
    model: ParakeetEOUModel,
    tokenizer: tokenizers::Tokenizer,
    encoder_cache: EncoderCache,
    state_h: Array3<f32>,
    state_c: Array3<f32>,
    next_state_h: Array3<f32>,
    next_state_c: Array3<f32>,
    last_token: Array2<i32>,
    blank_id: i32,
    eou_id: i32,
    mel_basis: Array2<f32>,
    mel_frame_cache: Array2<f32>,
    feature_window: Vec<f32>,
    preemphasis_buffer: Vec<f32>,
    spec: Array2<f32>,
    mel: Array2<f32>,
    new_mel_frames: Array2<f32>,
    features: Array3<f32>,
    encoder_out: Array3<f32>,
    decoder_frame: Array3<f32>,
    logits: Array3<f32>,
    window: Vec<f32>,
    audio_buffer: VecDeque<f32>,
    new_frame_window_samples: usize,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    chunk_counter: u64,
    first_non_empty_emitted: bool,
    segment_has_text: bool,
}

impl ParakeetEOU {
    /// Load Parakeet EOU model from path
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder.onnx, decoder_joint.onnx, and tokenizer.json
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let started_at = Instant::now();
        let path = path.as_ref();
        let tokenizer_path = path.join("tokenizer.json");
        android_log::info(format!(
            "crate from_pretrained begin modelDir={}",
            path.display()
        ));
        android_log::info(format!(
            "crate tokenizer load begin path={}",
            tokenizer_path.display()
        ));
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            let message = format!("Failed to load tokenizer: {e}");
            android_log::error(format!(
                "crate tokenizer load failed elapsedMs={} error={}",
                started_at.elapsed().as_millis(),
                message
            ));
            Error::Config(message)
        })?;
        android_log::info(format!(
            "crate tokenizer load end elapsedMs={}",
            started_at.elapsed().as_millis()
        ));

        let vocab_size = tokenizer.get_vocab_size(true);
        // NeMo RNNT uses blank_id = len(vocabulary); it is outside the tokenizer
        // vocabulary and is present only in the joint network logits.
        let blank_id = vocab_size as i32;
        let eou_id = tokenizer
            .token_to_id("<EOU>")
            .map(|id| id as i32)
            .unwrap_or(1024);
        android_log::info(format!(
            "crate tokenIds vocabSize={} blankId={} eouId={}",
            vocab_size, blank_id, eou_id
        ));

        let exec_config = config.unwrap_or_default();
        android_log::info(format!(
            "crate model load begin elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let model = ParakeetEOUModel::from_pretrained(path, exec_config, vocab_size + 1)?;
        let encoder_layout = model.encoder_layout();
        if encoder_layout.batch != 1
            || encoder_layout.n_mels != N_MELS
            || encoder_layout.input_frames != SLICE_LEN
        {
            return Err(Error::Model(format!(
                "Encoder frontend shape mismatch: model layout={encoder_layout:?}, \
                 frontend expects batch=1 n_mels={N_MELS} input_frames={SLICE_LEN}"
            )));
        }
        android_log::info(format!(
            "crate model load end elapsedMs={}",
            started_at.elapsed().as_millis()
        ));

        // Keep only enough raw audio to generate the next batch of fresh mel frames.
        // The pre-encode context is carried separately as mel-frame cache.
        let new_frame_window_samples = required_audio_samples(FRAMES_PER_CHUNK);
        debug_assert_eq!(new_frame_window_samples, FEATURE_WINDOW_SAMPLES);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);
        let fft_input = vec![0.0f32; N_FFT];
        let fft_output = fft_plan.make_output_vec();
        let fft_scratch = fft_plan.make_scratch_vec();

        let state = Self {
            model,
            tokenizer,
            encoder_cache: EncoderCache::new(encoder_layout),
            state_h: Array3::zeros((1, 1, 640)),
            state_c: Array3::zeros((1, 1, 640)),
            next_state_h: Array3::zeros((1, 1, 640)),
            next_state_c: Array3::zeros((1, 1, 640)),
            last_token: Array2::from_elem((1, 1), blank_id),
            blank_id,
            eou_id,
            mel_basis: Self::create_mel_filterbank(),
            mel_frame_cache: Array2::zeros((N_MELS, PRE_ENCODE_CACHE)),
            feature_window: vec![0.0f32; FEATURE_WINDOW_SAMPLES],
            preemphasis_buffer: vec![0.0f32; FEATURE_WINDOW_SAMPLES],
            spec: Array2::zeros((FREQ_BINS, STFT_FRAMES)),
            mel: Array2::zeros((N_MELS, STFT_FRAMES)),
            new_mel_frames: Array2::zeros((N_MELS, FRAMES_PER_CHUNK)),
            features: Array3::zeros((
                encoder_layout.batch,
                encoder_layout.n_mels,
                encoder_layout.input_frames,
            )),
            encoder_out: Array3::zeros((
                encoder_layout.batch,
                encoder_layout.hidden_size,
                encoder_layout.output_frames,
            )),
            decoder_frame: Array3::zeros((encoder_layout.batch, encoder_layout.hidden_size, 1)),
            logits: Array3::zeros((1, 1, vocab_size + 1)),
            window: Self::create_window(),
            audio_buffer: VecDeque::with_capacity(new_frame_window_samples),
            new_frame_window_samples,
            fft_plan,
            fft_input,
            fft_output,
            fft_scratch,
            chunk_counter: 0,
            first_non_empty_emitted: false,
            segment_has_text: false,
        };
        android_log::info(format!(
            "crate from_pretrained ready elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        Ok(state)
    }

    /// Transcribe a chunk of audio samples.
    ///
    /// # Arguments
    /// * `chunk` - Audio chunk (typically 160ms / 2560 samples at 16kHz)
    /// * `emit_eou_marker` - If true, append an EOU marker when the model detects an
    ///   utterance boundary after text has been emitted for the current segment.
    ///
    /// # Streaming Behavior
    /// Cache-aware streaming
    /// - Maintains only the raw-audio tail required for the next fresh mel frames
    /// - Reuses the previous mel-frame cache for encoder pre-encode context
    /// - Sends (pre_encode_cache + new_frames) mel frames to the encoder
    pub fn transcribe(&mut self, chunk: &[f32], emit_eou_marker: bool) -> Result<String> {
        let chunk_started_at = Instant::now();
        self.chunk_counter += 1;
        let chunk_index = self.chunk_counter;

        // Add new chunk to rolling buffer
        self.audio_buffer.extend(chunk.iter().copied());

        // Trim buffer to keep only the raw-audio context needed for the next fresh mel frames.
        while self.audio_buffer.len() > self.new_frame_window_samples {
            self.audio_buffer.pop_front();
        }

        if self.audio_buffer.len() < WIN_LENGTH {
            android_log::info(format!(
                "chunk idx={} stage=buffering bufferedSamples={} requiredSamples={} elapsedMs={}",
                chunk_index,
                self.audio_buffer.len(),
                WIN_LENGTH,
                chunk_started_at.elapsed().as_millis()
            ));
            return Ok(String::new());
        }

        self.build_feature_window();
        let feature_started_at = Instant::now();
        self.extract_new_mel_frames()?;
        let feature_ms = feature_started_at.elapsed().as_millis();

        self.features
            .slice_mut(s![0, .., 0..PRE_ENCODE_CACHE])
            .assign(&self.mel_frame_cache);
        self.features
            .slice_mut(s![0, .., PRE_ENCODE_CACHE..SLICE_LEN])
            .assign(&self.new_mel_frames);
        let time_steps = SLICE_LEN;

        // Encode with cache - encoder sees full buffer context
        let encoder_started_at = Instant::now();
        let total_frames = self.model.run_encoder_into(
            &self.features,
            time_steps as i64,
            &mut self.encoder_cache,
            &mut self.encoder_out,
        )?;
        let encoder_ms = encoder_started_at.elapsed().as_millis();
        let cache_len = self.encoder_cache_len();
        self.mel_frame_cache.assign(&self.features.slice(s![
            0,
            ..,
            SLICE_LEN - PRE_ENCODE_CACHE..SLICE_LEN
        ]));

        if total_frames == 0 {
            return Ok(String::new());
        }

        let mut text_output = String::new();
        let decoder_started_at = Instant::now();
        let mut decoder_calls = 0usize;
        let mut blank_breaks = 0usize;
        let mut emitted_tokens = 0usize;
        let mut eou_hits = 0usize;

        for t in 0..total_frames {
            self.decoder_frame
                .assign(&self.encoder_out.slice(s![.., .., t..t + 1]));
            let mut syms_added = 0;

            while syms_added < 5 {
                decoder_calls += 1;
                self.model.run_decoder_into(
                    &self.decoder_frame,
                    &self.last_token,
                    &self.state_h,
                    &self.state_c,
                    &mut self.logits,
                    &mut self.next_state_h,
                    &mut self.next_state_c,
                )?;

                let vocab = self.logits.slice(s![0, 0, ..]);

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &val) in vocab.iter().enumerate() {
                    if val.is_finite() && val > max_val {
                        max_val = val;
                        max_idx = i as i32;
                    }
                }

                if max_idx == self.blank_id {
                    blank_breaks += 1;
                    break;
                }

                if max_idx == 0 {
                    android_log::info(format!(
                        "metaToken kind=unk chunk={} frame={} symbol={} logit={} emittedTokens={} textLen={}",
                        chunk_index,
                        t,
                        syms_added,
                        max_val,
                        emitted_tokens,
                        text_output.len()
                    ));
                    blank_breaks += 1;
                    break;
                }

                if max_idx == self.eou_id {
                    eou_hits += 1;
                    android_log::info(format!(
                        "metaToken kind=eou chunk={} frame={} symbol={} logit={} emittedTokens={} textLen={} segmentHasText={}",
                        chunk_index,
                        t,
                        syms_added,
                        max_val,
                        emitted_tokens,
                        text_output.len(),
                        self.segment_has_text
                    ));
                    if emit_eou_marker && (self.segment_has_text || !text_output.is_empty()) {
                        self.segment_has_text = false;
                        let decoder_ms = decoder_started_at.elapsed().as_millis();
                        android_log::info(format!(
                            "chunk idx={} featureMs={} encoderMs={} decoderMs={} decoderCalls={} blankBreaks={} emittedTokens={} eouHits={} cacheLen={} totalMs={} emittedTextLen={} endpoint=true",
                            chunk_index,
                            feature_ms,
                            encoder_ms,
                            decoder_ms,
                            decoder_calls,
                            blank_breaks,
                            emitted_tokens,
                            eou_hits,
                            cache_len,
                            chunk_started_at.elapsed().as_millis(),
                            text_output.len()
                        ));
                        return Ok(text_output + " [EOU]");
                    }
                    break;
                }

                if max_idx as usize >= self.tokenizer.get_vocab_size(true) {
                    break;
                }

                self.state_h.assign(&self.next_state_h);
                self.state_c.assign(&self.next_state_c);
                self.last_token.fill(max_idx);

                if let Ok(decoded) = self.tokenizer.decode(&[max_idx as u32], true) {
                    text_output.push_str(&decoded);
                }
                emitted_tokens += 1;
                self.segment_has_text = true;
                syms_added += 1;
            }
        }
        let decoder_ms = decoder_started_at.elapsed().as_millis();
        if !text_output.is_empty() && !self.first_non_empty_emitted {
            self.first_non_empty_emitted = true;
            android_log::info(format!(
                "firstNonEmpty chunk={} featureMs={} encoderMs={} decoderMs={} decoderCalls={} blankBreaks={} emittedTokens={} cacheLen={} totalMs={} textLen={}",
                chunk_index,
                feature_ms,
                encoder_ms,
                decoder_ms,
                decoder_calls,
                blank_breaks,
                emitted_tokens,
                cache_len,
                chunk_started_at.elapsed().as_millis(),
                text_output.len()
            ));
        }
        android_log::info(format!(
            "chunk idx={} featureMs={} encoderMs={} decoderMs={} decoderCalls={} blankBreaks={} emittedTokens={} eouHits={} cacheLen={} totalMs={} emittedTextLen={} endpoint=false",
            chunk_index,
            feature_ms,
            encoder_ms,
            decoder_ms,
            decoder_calls,
            blank_breaks,
            emitted_tokens,
            eou_hits,
            cache_len,
            chunk_started_at.elapsed().as_millis(),
            text_output.len()
        ));
        Ok(text_output)
    }

    pub fn end_profiling(&mut self) -> Result<Vec<String>> {
        self.model.end_profiling()
    }

    /// Reset all streaming state for a new recognition session.
    ///
    /// EOU detection finalizes text, but it does not reset model state; this
    /// reset is reserved for a new recognition session.
    pub fn reset(&mut self) {
        self.encoder_cache = EncoderCache::new(self.model.encoder_layout());
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.next_state_h.fill(0.0);
        self.next_state_c.fill(0.0);
        self.last_token.fill(self.blank_id);
        self.mel_frame_cache.fill(0.0);
        self.feature_window.fill(0.0);
        self.preemphasis_buffer.fill(0.0);
        self.spec.fill(0.0);
        self.mel.fill(0.0);
        self.new_mel_frames.fill(0.0);
        self.features.fill(0.0);
        self.encoder_out.fill(0.0);
        self.decoder_frame.fill(0.0);
        self.logits.fill(0.0);
        self.fft_input.fill(0.0);
        self.fft_output.fill(Complex32::new(0.0, 0.0));
        self.fft_scratch.fill(Complex32::new(0.0, 0.0));
        self.audio_buffer.clear();
        self.chunk_counter = 0;
        self.first_non_empty_emitted = false;
        self.segment_has_text = false;
    }

    fn build_feature_window(&mut self) {
        let pad = self
            .new_frame_window_samples
            .saturating_sub(self.audio_buffer.len());
        self.feature_window[..pad].fill(0.0);
        for (dst, sample) in self.feature_window[pad..]
            .iter_mut()
            .zip(self.audio_buffer.iter().copied())
        {
            *dst = sample;
        }
    }

    fn encoder_cache_len(&self) -> i64 {
        self.encoder_cache
            .cache_last_channel_len
            .as_slice()
            .and_then(|values| values.first().copied())
            .unwrap_or(-1)
    }

    fn extract_new_mel_frames(&mut self) -> Result<()> {
        self.apply_preemphasis();
        self.stft()?;
        general_mat_mul(1.0, &self.mel_basis, &self.spec, 0.0, &mut self.mel);
        let total_frames = self.mel.shape()[1];
        let start_frame = total_frames.saturating_sub(FRAMES_PER_CHUNK);
        for mel_bin in 0..N_MELS {
            for frame in 0..FRAMES_PER_CHUNK {
                let value = self.mel[[mel_bin, start_frame + frame]];
                self.new_mel_frames[[mel_bin, frame]] = (value.max(0.0) + LOG_ZERO_GUARD).ln();
            }
        }
        Ok(())
    }

    fn apply_preemphasis(&mut self) {
        let safe_x = |x: f32| if x.is_finite() { x } else { 0.0 };
        if self.feature_window.is_empty() {
            return;
        }

        self.preemphasis_buffer[0] = safe_x(self.feature_window[0]);
        for i in 1..self.feature_window.len() {
            self.preemphasis_buffer[i] =
                safe_x(self.feature_window[i]) - PREEMPH * safe_x(self.feature_window[i - 1]);
        }
    }

    fn stft(&mut self) -> Result<()> {
        let pad_amount = N_FFT / 2;
        let padded_len = self.preemphasis_buffer.len() + 2 * pad_amount;
        let num_frames = 1 + (padded_len.saturating_sub(WIN_LENGTH)) / HOP_LENGTH;
        if num_frames != STFT_FRAMES {
            return Err(Error::Audio(format!(
                "Unexpected STFT frame count: {num_frames}, expected {STFT_FRAMES}"
            )));
        }

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            if start + WIN_LENGTH > padded_len {
                break;
            }

            self.fft_input.fill(0.0);
            for i in 0..WIN_LENGTH {
                let padded_idx = start + i;
                let sample = if padded_idx < pad_amount {
                    0.0
                } else {
                    let audio_idx = padded_idx - pad_amount;
                    self.preemphasis_buffer
                        .get(audio_idx)
                        .copied()
                        .unwrap_or(0.0)
                };
                self.fft_input[i] = sample * self.window[i];
            }

            self.fft_plan
                .process_with_scratch(
                    &mut self.fft_input,
                    &mut self.fft_output,
                    &mut self.fft_scratch,
                )
                .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

            for (i, val) in self.fft_output.iter().take(FREQ_BINS).enumerate() {
                let mag_sq = val.norm_sqr();
                self.spec[[i, frame_idx]] = if mag_sq.is_finite() { mag_sq } else { 0.0 };
            }
        }
        Ok(())
    }

    fn create_window() -> Vec<f32> {
        (0..WIN_LENGTH)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / ((WIN_LENGTH - 1) as f32)).cos())
            .collect()
    }

    fn create_mel_filterbank() -> Array2<f32> {
        let num_freqs = N_FFT / 2 + 1;

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(FMAX);

        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32))
            .collect();

        let fft_freqs: Vec<f32> = (0..num_freqs)
            .map(|i| (SAMPLE_RATE as f32 / N_FFT as f32) * i as f32)
            .collect();

        let mut weights = Array2::zeros((N_MELS, num_freqs));

        for i in 0..N_MELS {
            let left = mel_points[i];
            let center = mel_points[i + 1];
            let right = mel_points[i + 2];
            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= left && freq <= center {
                    weights[[i, j]] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    weights[[i, j]] = (right - freq) / (right - center);
                }
            }
        }

        for i in 0..N_MELS {
            let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
            for j in 0..num_freqs {
                weights[[i, j]] *= enorm;
            }
        }

        weights
    }
}
