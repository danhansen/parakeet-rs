use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_nemotron::{NemotronEncoderCache, NemotronModel};
use crate::nemotron_frontend::{NemotronFrontend, NemotronMelChunk, NemotronMelChunker};
use ndarray::{Array1, Array2, Array3};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};

// Nemotron 0.6B model constants
// note that those numbers are coming from offical impl. and of course my onnx export decisions.
// Buffer logic and cache slicing strategy derived from:
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/utils/streaming_utils.py
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/audio_preprocessing.py
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;

/// Language → prompt embedding index for the multilingual model. Mirrors
/// `cfg.model_defaults.prompt_dictionary` from the .nemo. Embedded here so
/// we don't require a sidecar `config.json` next to the ONNX files.
///
/// NVIDIA's model card documents 40 language-locales across 3 tiers:
///   - **Transcription-ready (19):** en, es, fr, it, pt, nl, de, tr, ru, ar,
///     hi, ja, ko, vi, uk (with locales).
///   - **Broad-coverage (13):** pl, sv, cs, nb, da, bg, fi, hr, sk, zh-CN,
///     hu, ro, et.
///   - **Adaptation-ready (8):** el, lt, lv, mt, sl, he, th, nn — recognized
///     by the tokenizer but need fine-tuning for production quality.
///
/// The full dictionary below contains additional entries because (a) several
/// codes alias the same prompt index (`en` == `en-US`, `hi` == `hi-IN`, ...)
/// and (b) some experimental languages (e.g. `qu-PE`, `mi-NZ`, `haw-US`)
/// have prompt slots but are not in the model card. Using those will work
/// but quality is not guaranteed.
///
/// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
const PROMPT_DICTIONARY: &[(&str, i64)] = &[
    ("af-ZA", 54),
    ("am-ET", 49),
    ("ar", 7),
    ("ar-AR", 7),
    ("auto", 101),
    ("ay-BO", 81),
    ("az-AZ", 66),
    ("bg", 30),
    ("bg-BG", 30),
    ("bn-IN", 36),
    ("cs", 22),
    ("cs-CZ", 22),
    ("da", 25),
    ("da-DK", 25),
    ("de", 9),
    ("de-DE", 9),
    ("el", 21),
    ("el-GR", 21),
    ("en", 0),
    ("en-GB", 1),
    ("en-US", 0),
    ("enGB", 1),
    ("es", 3),
    ("es-ES", 2),
    ("es-US", 3),
    ("esES", 2),
    ("et", 60),
    ("et-EE", 60),
    ("fa-IR", 38),
    ("fi", 26),
    ("fi-FI", 26),
    ("fr", 8),
    ("fr-CA", 100),
    ("fr-FR", 8),
    ("gn-PY", 82),
    ("gu-IN", 42),
    ("ha-NG", 50),
    ("haw-US", 97),
    ("he-IL", 64),
    ("hi", 6),
    ("hi-HI", 6),
    ("hi-IN", 6),
    ("hr", 29),
    ("hr-HR", 29),
    ("hu", 23),
    ("hu-HU", 23),
    ("hy-AM", 68),
    ("id-ID", 34),
    ("ig-NG", 53),
    ("it", 15),
    ("it-IT", 15),
    ("ja-JA", 10),
    ("ja-JP", 10),
    ("ka-GE", 67),
    ("km-KH", 47),
    ("kn-IN", 43),
    ("ko", 14),
    ("ko-KO", 14),
    ("ko-KR", 14),
    ("ku-TR", 65),
    ("ky-KG", 71),
    ("ln-CD", 58),
    ("lt", 31),
    ("lt-LT", 31),
    ("lv", 61),
    ("lv-LV", 61),
    ("mi-NZ", 96),
    ("ml-IN", 44),
    ("mr-IN", 41),
    ("ms-MY", 35),
    ("mt-MT", 102),
    ("nah-MX", 83),
    ("nb", 103),
    ("nb-NO", 103),
    ("ne-NP", 46),
    ("nl", 16),
    ("nl-NL", 16),
    ("nn", 104),
    ("nn-NO", 104),
    ("no", 27),
    ("no-NO", 27),
    ("ny-MW", 57),
    ("or-KE", 59),
    ("pl", 17),
    ("pl-PL", 17),
    ("pt", 13),
    ("pt-BR", 12),
    ("pt-PT", 13),
    ("qu-PE", 80),
    ("ro", 20),
    ("ro-RO", 20),
    ("ru", 11),
    ("ru-RU", 11),
    ("rw-RW", 55),
    ("si-LK", 45),
    ("sk", 28),
    ("sk-SK", 28),
    ("sl", 62),
    ("sl-SI", 62),
    ("sm-WS", 98),
    ("so-SO", 56),
    ("sv", 24),
    ("sv-SE", 24),
    ("sw-KE", 48),
    ("ta-IN", 39),
    ("te-IN", 40),
    ("tg-TJ", 70),
    ("th-TH", 32),
    ("to-TO", 99),
    ("tr", 18),
    ("tr-TR", 18),
    ("uk", 19),
    ("uk-UA", 19),
    ("ur-PK", 37),
    ("uz-UZ", 69),
    ("vi-VN", 33),
    ("yo-NG", 52),
    ("zh-CN", 4),
    ("zh-TW", 5),
    ("zh-ZH", 4),
    ("zu-ZA", 51),
];

/// Detect SentencePiece pieces that encode a language tag like `<en-US>` or
/// `<en>`. The multilingual model emits these inline with text; they're
/// stripped from the user-visible transcript.
fn is_lang_tag(piece: &str) -> bool {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return false;
    }
    let inner = &bytes[1..bytes.len() - 1];
    match inner.len() {
        2 => inner[0].is_ascii_lowercase() && inner[1].is_ascii_lowercase(),
        5 => {
            inner[0].is_ascii_lowercase()
                && inner[1].is_ascii_lowercase()
                && inner[2] == b'-'
                && inner[3].is_ascii_uppercase()
                && inner[4].is_ascii_uppercase()
        }
        _ => false,
    }
}

/// Which Nemotron variant a handle wraps. Detected automatically from
/// the encoder ONNX graph (multilingual exposes a `prompt_index` input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronMode {
    /// English-only Nemotron 0.6B (vocab 1024, no language conditioning).
    EnglishOnly,
    /// Multilingual Nemotron 3.5 0.6B with `prompt_index` input,
    /// vocab ~13k, supports `target_lang` selection.
    Multilingual,
}

#[derive(Debug, Clone)]
pub struct NemotronTopToken {
    pub id: usize,
    pub piece: String,
    pub logit: f32,
}

#[derive(Debug, Clone)]
pub struct NemotronTokenDecision {
    pub chunk_index: usize,
    pub frame_index: usize,
    pub symbol_index: usize,
    pub input_token_id: i32,
    pub token_id: usize,
    pub piece: String,
    pub logit: f32,
    pub blank_logit: f32,
    pub margin: f32,
    pub top: Vec<NemotronTopToken>,
}

#[derive(Debug, Clone)]
pub struct NemotronChunkTrace {
    pub text: String,
    pub decisions: Vec<NemotronTokenDecision>,
}

/// Minimal SentencePiece vocabulary loader.
/// Parses the protobuf .model file to extract token strings.
/// Note that, our vocab.rs cannot parse protobuf format. I haven't test it with digit spacing yet, at least for this initial impl.
pub struct SentencePieceVocab {
    pieces: Vec<String>,
}

impl SentencePieceVocab {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open tokenizer.model: {e}")))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Tokenizer(format!("Failed to read tokenizer.model: {e}")))?;

        let pieces = Self::parse_sentencepiece_model(&data)?;
        Ok(Self { pieces })
    }

    fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize > data.len() {
                        break;
                    }

                    let piece_data = &data[pos..pos + len as usize];
                    pos += len as usize;

                    if let Ok(piece) = Self::parse_piece_message(piece_data) {
                        pieces.push(piece);
                    }
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        if pieces.is_empty() {
            return Err(Error::Tokenizer("No tokens found in model".into()));
        }

        Ok(pieces)
    }

    fn parse_piece_message(data: &[u8]) -> Result<String> {
        let mut pos = 0;
        let mut piece = String::new();

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize <= data.len() {
                        piece = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                    }
                    pos += len as usize;
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        Ok(piece)
    }

    fn read_varint(data: &[u8]) -> Result<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut pos = 0;

        while pos < data.len() && pos < 10 {
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;

            if byte & 0x80 == 0 {
                return Ok((result, pos));
            }
            shift += 7;
        }

        Err(Error::Tokenizer("Invalid varint".into()))
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id < self.pieces.len() {
                let piece = &self.pieces[id];
                let decoded = piece.replace('\u{2581}', " ");
                result.push_str(&decoded);
            }
        }
        result.trim_start().to_string()
    }

    pub fn decode_single(&self, id: usize) -> String {
        if id < self.pieces.len() {
            self.pieces[id].replace('\u{2581}', " ")
        } else {
            String::new()
        }
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    /// Token IDs whose SentencePiece pieces look like language tags
    /// (`<en-US>`, `<fr>`, ...). and ofc empty for the en only vocab.
    pub fn lang_tag_ids(&self) -> Vec<usize> {
        self.pieces
            .iter()
            .enumerate()
            .filter_map(|(i, p)| is_lang_tag(p).then_some(i))
            .collect()
    }
}

/// Shared handle to a loaded Nemotron model.
/// ONNX session is only loaded once and reference counted.
///
/// Use [`NemotronHandle::load`] to load from disk, then [`Nemotron::from_shared`]
/// to spawn each stream with its own independent decoder state.
/// Variant is auto-detected: both the en only 0.6B and the multi lang
/// 3.5 0.6B drop into the same type.
#[derive(Clone)]
pub struct NemotronHandle {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mel_basis: Arc<Array2<f32>>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    decoder_lstm_dim: usize,
    decoder_lstm_layers: usize,
    vocab_size: usize,
    blank_id: usize,
    has_projected_kv_cache: bool,
    chunk_size: usize,
    /// Empty for en. populated for multilingual with all `<xx-XX>` token ids.
    lang_tag_ids: Arc<Vec<usize>>,
}

/// Nemotron streaming ASR model (0.6B parameters).
/// We dont apply mel normalization unlike others...
///
/// For a single stream, use [`Nemotron::from_pretrained`]. For multiple
/// concurrent streams (e.g. mic + system audio) sharing one loaded model,
/// use [`NemotronHandle::load`] followed by [`Nemotron::from_shared`].
///
/// For the multilingual variant call [`Nemotron::set_target_lang`] before
/// transcribing if you know the language; otherwise it defaults to `auto`
/// (prompt index 101) and lets the model pick.
pub struct Nemotron {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    vocab_size: usize,
    blank_id: usize,
    lang_tag_ids: Arc<Vec<usize>>,
    encoder_cache: NemotronEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    /// `None` for English-only mode; `Some(idx)` for multilingual.
    prompt_index: Option<i64>,
    frontend: NemotronFrontend,
    mel_chunker: NemotronMelChunker,
    chunk_size: usize,
    chunk_idx: usize,
    accumulated_tokens: Vec<usize>,
}

impl NemotronHandle {
    /// Load the Nemotron model and vocabulary from a directory.
    ///
    /// Required files in `path`:
    /// - `encoder.onnx` + `encoder.onnx.data`
    /// - `decoder_joint.onnx`
    /// - `tokenizer.model`
    ///
    /// The returned handle is cheap to clone and can be used to spawn any
    /// number of [`Nemotron`] instances via [`Nemotron::from_shared`], each
    /// with its own independent decoder state.
    pub fn load<P: AsRef<Path>>(path: P, exec_config: Option<ExecutionConfig>) -> Result<Self> {
        let path = path.as_ref();

        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let vocab_size = vocab.size();

        let exec = exec_config.unwrap_or_default();
        let model = NemotronModel::from_pretrained(path, exec, vocab_size)?;
        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        let mode = if model.has_prompt {
            NemotronMode::Multilingual
        } else {
            NemotronMode::EnglishOnly
        };
        let cfg = model.config.clone();
        let lang_tag_ids = if mode == NemotronMode::Multilingual {
            vocab.lang_tag_ids()
        } else {
            Vec::new()
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(vocab),
            mel_basis: Arc::new(mel_basis),
            mode,
            num_encoder_layers: cfg.num_encoder_layers,
            hidden_dim: cfg.hidden_dim,
            left_context: cfg.left_context,
            conv_context: cfg.conv_context,
            decoder_lstm_dim: cfg.decoder_lstm_dim,
            decoder_lstm_layers: cfg.decoder_lstm_layers,
            vocab_size: cfg.vocab_size,
            blank_id: cfg.blank_id,
            has_projected_kv_cache: cfg.has_projected_kv_cache,
            chunk_size: cfg.chunk_size_output_frames * 8,
            lang_tag_ids: Arc::new(lang_tag_ids),
        })
    }

    /// Which variant this handle wraps (auto detected at load time).
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Languages this model can transcribe, as accepted by
    /// [`Nemotron::set_target_lang`]. Empty for the English-only variant.
    pub fn available_languages(&self) -> Vec<&'static str> {
        match self.mode {
            NemotronMode::Multilingual => PROMPT_DICTIONARY.iter().map(|(k, _)| *k).collect(),
            NemotronMode::EnglishOnly => Vec::new(),
        }
    }
}

impl Nemotron {
    /// Load Nemotron from a directory and return a ready to use instance.
    /// Convenience wrapper for the single-stream case.
    ///
    /// For multiple concurrent streams sharing one loaded model, use
    /// [`NemotronHandle::load`] + [`Nemotron::from_shared`] instead.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Ok(Self::from_shared(&NemotronHandle::load(path, exec_config)?))
    }

    /// Spawn a new Nemotron instance bound to a shared model.
    ///
    /// Each instance owns independent decoder state (~7.5 MB) while the
    /// expensive ONNX session is shared through the handle.
    /// The model lock is held only during encoder/decoder inference
    /// (~20-50 ms per 560 ms audio chunk).
    ///
    /// For the multilingual variant the new instance defaults to `auto`
    /// (prompt index 101) — the model picks the language itself. Override
    /// via [`Self::set_target_lang`] when you know the language; that's
    /// strictly more accurate.
    pub fn from_shared(handle: &NemotronHandle) -> Self {
        let encoder_cache = if handle.has_projected_kv_cache {
            NemotronEncoderCache::with_projected_dims(
                handle.num_encoder_layers,
                handle.left_context,
                handle.hidden_dim,
                handle.conv_context,
            )
        } else {
            NemotronEncoderCache::with_dims(
                handle.num_encoder_layers,
                handle.left_context,
                handle.hidden_dim,
                handle.conv_context,
            )
        };

        let prompt_index = match handle.mode {
            NemotronMode::Multilingual => Some(101),
            NemotronMode::EnglishOnly => None,
        };

        Self {
            model: Arc::clone(&handle.model),
            vocab: Arc::clone(&handle.vocab),
            mode: handle.mode,
            num_encoder_layers: handle.num_encoder_layers,
            hidden_dim: handle.hidden_dim,
            left_context: handle.left_context,
            conv_context: handle.conv_context,
            vocab_size: handle.vocab_size,
            blank_id: handle.blank_id,
            lang_tag_ids: Arc::clone(&handle.lang_tag_ids),
            encoder_cache,
            state_1: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            state_2: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            last_token: handle.blank_id as i32,
            prompt_index,
            frontend: NemotronFrontend::new(Arc::clone(&handle.mel_basis)),
            mel_chunker: NemotronMelChunker::new(handle.chunk_size),
            chunk_size: handle.chunk_size,
            chunk_idx: 0,
            accumulated_tokens: Vec::new(),
        }
    }

    /// Which variant this instance wraps.
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Set the target language for the multilingual model. Accepts any key
    /// from [`NemotronHandle::available_languages`] (e.g. `"en-US"`, `"es-ES"`,
    /// `"ja-JP"`, `"auto"` for language-agnostic decoding).
    ///
    /// **Quality note:** NVIDIA's model card documents 40 language-locales
    /// across 3 tiers (transcription-ready, broad-coverage, adaptation-ready).
    /// Adaptation-ready locales need fine-tuning for production quality.
    /// The full prompt dictionary accepts additional codes (e.g. `qu-PE`,
    /// `mi-NZ`, `haw-US`) that the model has prompt slots for but are not
    /// in the model card — those will run, but accuracy is not guaranteed.
    /// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
    ///
    /// Returns an error on the English-only variant or for an unknown language.
    /// The new language takes effect on the next encoder call — for clean
    /// switching mid-utterance you usually also want [`Self::reset`].
    pub fn set_target_lang(&mut self, lang: &str) -> Result<()> {
        if self.mode != NemotronMode::Multilingual {
            return Err(Error::Config(
                "set_target_lang is only available on the multilingual variant".into(),
            ));
        }
        let idx = PROMPT_DICTIONARY
            .iter()
            .find_map(|(k, v)| (*k == lang).then_some(*v))
            .ok_or_else(|| {
                Error::Config(format!(
                    "Unknown target language '{lang}'. Try one of: en-US, es-ES, de-DE, fr-FR, ja-JP, zh-CN, auto, ..."
                ))
            })?;
        self.prompt_index = Some(idx);
        Ok(())
    }

    /// Reset all state for new utterance. Preserves the configured target
    /// language (call [`Self::set_target_lang`] again to change it).
    pub fn reset(&mut self) {
        self.encoder_cache = {
            let has_projected_kv_cache = {
                let model = self.model.lock();
                model
                    .as_ref()
                    .map(|model| model.has_projected_kv_cache)
                    .unwrap_or(false)
            };
            if has_projected_kv_cache {
                NemotronEncoderCache::with_projected_dims(
                    self.num_encoder_layers,
                    self.left_context,
                    self.hidden_dim,
                    self.conv_context,
                )
            } else {
                NemotronEncoderCache::with_dims(
                    self.num_encoder_layers,
                    self.left_context,
                    self.hidden_dim,
                    self.conv_context,
                )
            }
        };
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.frontend.reset();
        self.mel_chunker.reset();
        self.chunk_idx = 0;
        self.accumulated_tokens.clear();
    }

    /// Get the full accumulated transcript. Language tag tokens (e.g. `<en-US>`)
    /// emitted by the multilingual model are stripped.
    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|t| *t < self.vocab_size && !self.lang_tag_ids.contains(t))
            .collect();
        self.vocab.decode(&valid)
    }

    /// note that, offline transcription for testing/debugging and for some curious ppl :-). with following function too (transcribe_audio)
    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;

        let audio = if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        };

        self.transcribe_audio(&audio)
    }

    /// Transcribe audio samples (non-streaming)
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        self.reset();
        for packet in audio.chunks(self.chunk_samples()) {
            self.transcribe_chunk(packet)?;
        }
        self.finish()?;
        Ok(self.get_transcript())
    }

    /// Required number of 16 kHz audio samples per streaming chunk.
    pub fn chunk_samples(&self) -> usize {
        self.chunk_size * HOP_LENGTH
    }

    /// Stream transcribe a chunk of audio (call repeatedly for real-time).
    ///
    /// Input may have any packet size. Complete centered windows are produced
    /// on a global grid; a window needing future samples waits for more audio.
    /// Call [`Self::finish`] once at end of stream, not with synthetic silence.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        Ok(self.transcribe_chunk_internal(audio_chunk, false)?.text)
    }

    pub fn transcribe_chunk_with_trace(
        &mut self,
        audio_chunk: &[f32],
    ) -> Result<NemotronChunkTrace> {
        self.transcribe_chunk_internal(audio_chunk, true)
    }

    fn transcribe_chunk_internal(
        &mut self,
        audio_chunk: &[f32],
        trace_tokens: bool,
    ) -> Result<NemotronChunkTrace> {
        let started = std::time::Instant::now();
        let frames = self.frontend.push(audio_chunk)?;
        let chunks = self.mel_chunker.push(frames)?;
        self.process_mel_chunks(chunks, trace_tokens, started.elapsed().as_secs_f64())
    }

    /// Complete the feature tail using the true audio length and NeMo's masked
    /// terminal frame. Idempotent; reset before supplying a new stream.
    pub fn finish(&mut self) -> Result<String> {
        Ok(self.finish_with_trace_internal(false)?.text)
    }

    pub fn finish_with_trace(&mut self) -> Result<NemotronChunkTrace> {
        self.finish_with_trace_internal(true)
    }

    fn finish_with_trace_internal(&mut self, trace_tokens: bool) -> Result<NemotronChunkTrace> {
        let started = std::time::Instant::now();
        let frames = self.frontend.finish()?;
        if frames.is_empty() || self.frontend.sample_count() == 0 {
            return Ok(NemotronChunkTrace {
                text: String::new(),
                decisions: Vec::new(),
            });
        }
        let mut chunks = self.mel_chunker.push(frames)?;
        chunks.extend(self.mel_chunker.finish());
        self.process_mel_chunks(chunks, trace_tokens, started.elapsed().as_secs_f64())
    }

    fn process_mel_chunks(
        &mut self,
        chunks: Vec<NemotronMelChunk>,
        trace_tokens: bool,
        frontend_seconds: f64,
    ) -> Result<NemotronChunkTrace> {
        let mut result = NemotronChunkTrace {
            text: String::new(),
            decisions: Vec::new(),
        };
        let mut first = true;
        for chunk in chunks {
            let decoded = self.process_mel_chunk(
                chunk,
                trace_tokens,
                if first { frontend_seconds } else { 0. },
            )?;
            first = false;
            result.text.push_str(&decoded.text);
            result.decisions.extend(decoded.decisions);
        }
        Ok(result)
    }

    fn process_mel_chunk(
        &mut self,
        chunk: NemotronMelChunk,
        trace_tokens: bool,
        frontend_seconds: f64,
    ) -> Result<NemotronChunkTrace> {
        let stage_trace = std::env::var_os("PARAKEET_STAGE_TRACE").is_some();
        let chunk_started = std::time::Instant::now();
        let encoder_started = std::time::Instant::now();
        let (encoded, enc_len) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder_into(
                &chunk.features,
                chunk.input_frames as i64,
                &mut self.encoder_cache,
                self.prompt_index,
            )?
        };
        let encoder_seconds = encoder_started.elapsed().as_secs_f64();

        let chunk_index = self.chunk_idx;
        let decoder_started = std::time::Instant::now();
        let (tokens, decisions) =
            self.decode_chunk_with_trace(&encoded, enc_len as usize, chunk_index, trace_tokens)?;
        if stage_trace {
            eprintln!(
                "[parakeet-stages] {}",
                serde_json::json!({
                    "chunk_index": chunk_index,
                    "encoder_frames": enc_len,
                    "frontend_seconds": frontend_seconds,
                    "encoder_seconds": encoder_seconds,
                    "decoder_seconds": decoder_started.elapsed().as_secs_f64(),
                    "total_seconds": chunk_started.elapsed().as_secs_f64(),
                })
            );
        }
        self.accumulated_tokens.extend(&tokens);

        self.chunk_idx += 1;

        let mut result = String::new();
        for &t in &tokens {
            if t < self.vocab_size && !self.lang_tag_ids.contains(&t) {
                result.push_str(&self.vocab.decode_single(t));
            }
        }
        Ok(NemotronChunkTrace {
            text: result,
            decisions,
        })
    }

    fn decode_chunk_with_trace(
        &mut self,
        encoder_out: &Array3<f32>,
        enc_frames: usize,
        chunk_index: usize,
        trace_tokens: bool,
    ) -> Result<(Vec<usize>, Vec<NemotronTokenDecision>)> {
        let mut tokens = Vec::new();
        let mut decisions = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let max_symbols_per_step = 10;
        let mut frame = Array3::<f32>::zeros((1, hidden_dim, 1));
        let vocab = Arc::clone(&self.vocab);
        let vocab_size = self.vocab_size;
        let blank_id = self.blank_id;

        // Lock the model once for the entire decode loop to minimise
        // lock acquire/release overhead (many decoder steps per chunk).
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;

        for t in 0..enc_frames {
            for h in 0..hidden_dim {
                frame[[0, h, 0]] = encoder_out[[0, h, t]];
            }

            for symbol_index in 0..max_symbols_per_step {
                let input_token_id = self.last_token;
                let (logits, new_state_1, new_state_2) =
                    model.run_decoder(&frame, input_token_id, &self.state_1, &self.state_2)?;

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }

                if max_idx == self.blank_id {
                    break;
                }

                if trace_tokens {
                    let blank_logit = logits.get(blank_id).copied().unwrap_or(f32::NEG_INFINITY);
                    decisions.push(NemotronTokenDecision {
                        chunk_index,
                        frame_index: t,
                        symbol_index,
                        input_token_id,
                        token_id: max_idx,
                        piece: token_piece(&vocab, vocab_size, blank_id, max_idx),
                        logit: max_val,
                        blank_logit,
                        margin: max_val - blank_logit,
                        top: top_tokens(&logits, &vocab, vocab_size, blank_id, 8),
                    });
                }
                tokens.push(max_idx);
                self.last_token = max_idx as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok((tokens, decisions))
    }
}

fn token_piece(
    vocab: &SentencePieceVocab,
    vocab_size: usize,
    blank_id: usize,
    token_id: usize,
) -> String {
    if token_id == blank_id {
        "<blank>".to_string()
    } else if token_id < vocab_size {
        vocab.decode_single(token_id)
    } else {
        format!("<out:{token_id}>")
    }
}

fn top_tokens(
    logits: &Array1<f32>,
    vocab: &SentencePieceVocab,
    vocab_size: usize,
    blank_id: usize,
    k: usize,
) -> Vec<NemotronTopToken> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    pairs
        .into_iter()
        .take(k)
        .map(|(id, logit)| NemotronTopToken {
            id,
            piece: token_piece(vocab, vocab_size, blank_id, id),
            logit,
        })
        .collect()
}
