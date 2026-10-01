use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array3, Array4};
use ort::session::{InMemorySession, Session, SessionInputValue};
use ort::value::{TensorRef, ValueType};
use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

fn trace_load(message: impl AsRef<str>) {
    if std::env::var_os("PARAKEET_LOAD_TRACE").is_some() {
        eprintln!("[parakeet-load] {}", message.as_ref());
    }
}

struct OptimizedModelCachePaths {
    final_path: PathBuf,
    temp_path: PathBuf,
    external_data_source_path: Option<PathBuf>,
    external_data_cache_path: Option<PathBuf>,
    extra_external_data: Vec<(PathBuf, PathBuf)>,
}

struct ModelBytes {
    data: Box<[u8]>,
    #[cfg(test)]
    probe: Option<std::sync::Arc<LifetimeProbe>>,
}

impl ModelBytes {
    fn new(data: Box<[u8]>) -> Self {
        Self {
            data,
            #[cfg(test)]
            probe: None,
        }
    }
}

#[cfg(test)]
#[derive(Default)]
struct LifetimeProbe {
    freed: std::sync::atomic::AtomicUsize,
    native_session_gone: std::sync::atomic::AtomicBool,
    session: std::sync::Mutex<Option<std::sync::Weak<ort::session::SharedSessionInner>>>,
}

impl Drop for ModelBytes {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(probe) = &self.probe {
            use std::sync::atomic::Ordering;
            let gone = probe
                .session
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|weak| weak.upgrade().is_none());
            probe.native_session_gone.store(gone, Ordering::SeqCst);
            probe.freed.fetch_add(1, Ordering::SeqCst);
        }
    }
}

self_cell::self_cell!(
    struct DirectOrtSession {
        owner: ModelBytes,
        #[covariant]
        dependent: InMemorySession,
    }
);

enum LoadedSession {
    File(Session),
    DirectOrt(DirectOrtSession),
}

impl LoadedSession {
    // Neither sessions nor ORT outputs may escape these owner-bound closures.
    // All public inference results are copied into owned ndarrays before return.
    fn with_session<R>(&self, f: impl FnOnce(&Session) -> R) -> R {
        match self {
            Self::File(session) => f(session),
            Self::DirectOrt(cell) => f(cell.borrow_dependent()),
        }
    }

    fn with_session_mut<R>(&mut self, f: impl FnOnce(&mut Session) -> R) -> R {
        match self {
            Self::File(session) => f(session),
            Self::DirectOrt(cell) => cell.with_dependent_mut(|_, session| f(session)),
        }
    }
}

fn model_component_path(model_dir: &Path, stem: &str) -> Result<PathBuf> {
    let ort_path = model_dir.join(format!("{stem}.ort"));
    if ort_path.exists() {
        return Ok(ort_path);
    }

    let onnx_path = model_dir.join(format!("{stem}.onnx"));
    if onnx_path.exists() {
        return Ok(onnx_path);
    }

    Err(Error::Model(format!(
        "Missing {stem}.ort or {stem}.onnx in {}",
        model_dir.display()
    )))
}

fn external_weight_files(source_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = vec![source_path.with_file_name(format!(
        "{}.data",
        source_path.file_name().unwrap().to_string_lossy()
    ))];
    let config_path = source_path.with_file_name("config.json");
    if config_path.exists() {
        let payload: serde_json::Value = serde_json::from_slice(&fs::read(config_path)?)?;
        if let Some(shared) = payload.get("shared_weight_files") {
            let names = shared
                .as_array()
                .ok_or_else(|| Error::Model("invalid shared weight manifest".into()))?;
            for name in names {
                let name = name
                    .as_str()
                    .ok_or_else(|| Error::Model("invalid shared weight path".into()))?;
                let path = Path::new(name);
                if path.is_absolute()
                    || path
                        .components()
                        .any(|part| matches!(part, std::path::Component::ParentDir))
                {
                    return Err(Error::Model("unsafe shared weight path".into()));
                }
                files.push(source_path.parent().unwrap().join(path));
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files.into_iter().filter(|path| path.is_file()).collect())
}

fn optimized_model_cache_paths(
    exec_config: &ExecutionConfig,
    component: &str,
    source_path: &Path,
) -> Result<Option<OptimizedModelCachePaths>> {
    let Some(cache_dir) = exec_config.ort_optimized_model_cache_dir() else {
        return Ok(None);
    };
    // An opaque session-builder callback cannot be safely fingerprinted.
    if exec_config.configure.is_some() {
        return Ok(None);
    }
    fs::create_dir_all(cache_dir)?;
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_path_component)
        .unwrap_or_else(|| "model".to_string());
    let key = optimized_model_cache_key(exec_config, component, source_path)?;
    let source_file_name = source_path
        .file_name()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_path_component)
        .unwrap_or_else(|| "model.onnx".to_string());
    let artifact_dir = cache_dir.join(format!("{component}.{stem}.{key}"));
    fs::create_dir_all(&artifact_dir)?;

    let external_data_file_name = format!("{source_file_name}.data");
    let external_data_source_path = source_path.with_file_name(&external_data_file_name);
    let (external_data_source_path, external_data_cache_path) =
        if external_data_source_path.exists() {
            (
                Some(external_data_source_path),
                Some(artifact_dir.join(&external_data_file_name)),
            )
        } else {
            (None, None)
        };

    let extra_external_data = external_weight_files(source_path)?
        .into_iter()
        .filter(|path| Some(path) != external_data_source_path.as_ref())
        .map(|source| {
            let relative = source.strip_prefix(source_path.parent().unwrap()).unwrap();
            let target = artifact_dir.join(relative);
            (source, target)
        })
        .collect();
    Ok(Some(OptimizedModelCachePaths {
        final_path: artifact_dir.join(&source_file_name),
        temp_path: artifact_dir.join(format!("{source_file_name}.{}.tmp", std::process::id())),
        external_data_source_path,
        external_data_cache_path,
        extra_external_data,
    }))
}

fn optimized_model_cache_key(
    exec_config: &ExecutionConfig,
    component: &str,
    source_path: &Path,
) -> Result<String> {
    let metadata = fs::metadata(source_path)?;
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    update_fnv1a(&mut hash, b"wordpipe-nemotron-ort-cache-v2");
    update_fnv1a(&mut hash, component.as_bytes());
    update_fnv1a(&mut hash, source_path.to_string_lossy().as_bytes());
    update_fnv1a(&mut hash, metadata.len().to_string().as_bytes());
    update_fnv1a(&mut hash, modified_ns.to_string().as_bytes());
    update_fnv1a(
        &mut hash,
        format!("{:?}", exec_config.execution_provider).as_bytes(),
    );
    update_fnv1a(
        &mut hash,
        format!("{:?}", exec_config.graph_optimization).as_bytes(),
    );
    update_fnv1a(&mut hash, exec_config.intra_threads.to_string().as_bytes());
    update_fnv1a(&mut hash, exec_config.inter_threads.to_string().as_bytes());
    update_fnv1a(
        &mut hash,
        exec_config.force_spinning_stop.to_string().as_bytes(),
    );
    update_fnv1a(
        &mut hash,
        format!("{:?}", exec_config.dimension_overrides).as_bytes(),
    );
    update_fnv1a(
        &mut hash,
        format!(
            "{:?}:{}:{:?}",
            exec_config.memory_pattern, exec_config.parallel_execution, exec_config.cpu_arena
        )
        .as_bytes(),
    );
    update_fnv1a(&mut hash, std::env::consts::ARCH.as_bytes());
    update_fnv1a(&mut hash, std::env::consts::OS.as_bytes());
    #[cfg(target_arch = "x86_64")]
    update_fnv1a(
        &mut hash,
        format!(
            "sse2={},avx={},avx2={},fma={},avx512f={}",
            std::is_x86_feature_detected!("sse2"),
            std::is_x86_feature_detected!("avx"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx512f")
        )
        .as_bytes(),
    );
    let mut fingerprint_files = external_weight_files(source_path)?;
    fingerprint_files.push(source_path.with_file_name("config.json"));
    for path in fingerprint_files {
        if let Ok(metadata) = fs::metadata(&path) {
            update_fnv1a(&mut hash, metadata.len().to_string().as_bytes());
            update_fnv1a(
                &mut hash,
                format!("{:?}", metadata.modified().ok()).as_bytes(),
            );
        }
    }
    update_fnv1a(&mut hash, ort::info().as_bytes());
    Ok(format!("{hash:016x}"))
}

fn update_fnv1a(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn sanitize_cache_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn materialize_external_data(paths: &OptimizedModelCachePaths, component: &str) {
    for (source_path, cache_path) in &paths.extra_external_data {
        if let Some(parent) = cache_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if !cache_path.exists() && fs::hard_link(source_path, cache_path).is_err() {
            let _ = fs::copy(source_path, cache_path);
        }
    }
    let (Some(source_path), Some(cache_path)) = (
        paths.external_data_source_path.as_ref(),
        paths.external_data_cache_path.as_ref(),
    ) else {
        return;
    };

    let _ = fs::remove_file(cache_path);
    if let Err(link_err) = fs::hard_link(source_path, cache_path) {
        if let Err(copy_err) = fs::copy(source_path, cache_path) {
            trace_load(format!(
                "optimized cache external data materialization failed {component} {} -> {}: hard_link={link_err}; copy={copy_err}",
                source_path.display(),
                cache_path.display()
            ));
        }
    }
}

fn load_direct_ort_session(
    exec_config: &ExecutionConfig,
    source_path: &Path,
    log_id: &str,
) -> Result<LoadedSession> {
    trace_load(format!("direct ORT format load {}", source_path.display()));
    let bytes = fs::read(source_path)?.into_boxed_slice();
    let builder = Session::builder()?;
    let mut builder = exec_config.apply_to_session_builder_for_cached_model(builder)?;
    builder = builder.with_log_id(log_id)?;
    Ok(LoadedSession::DirectOrt(DirectOrtSession::try_new(
        ModelBytes::new(bytes),
        |bytes| builder.commit_from_memory_directly(&bytes.data),
    )?))
}

fn load_session_with_optional_cache(
    exec_config: &ExecutionConfig,
    component: &str,
    source_path: &Path,
    log_id: &str,
) -> Result<LoadedSession> {
    if source_path.extension().and_then(|value| value.to_str()) == Some("ort") {
        return load_direct_ort_session(exec_config, source_path, log_id);
    }

    fn build_session(
        exec_config: &ExecutionConfig,
        load_path: &Path,
        log_id: &str,
        use_cached_model: bool,
        optimized_output_path: Option<&Path>,
    ) -> Result<LoadedSession> {
        let builder = Session::builder()?;
        let mut builder = if use_cached_model {
            exec_config.apply_to_session_builder_for_cached_model(builder)?
        } else {
            exec_config.apply_to_session_builder(builder)?
        };
        builder = builder.with_log_id(log_id)?;
        if let Some(optimized_output_path) = optimized_output_path {
            builder = builder.with_optimized_model_path(optimized_output_path)?;
            let filename = format!(
                "{}.data",
                optimized_output_path.file_name().unwrap().to_string_lossy()
            );
            builder = builder
                .with_config_entry(
                    "session.optimized_model_external_initializers_file_name",
                    filename,
                )?
                .with_config_entry(
                    "session.optimized_model_external_initializers_min_size_in_bytes",
                    "65536",
                )?;
        }
        Ok(LoadedSession::File(builder.commit_from_file(load_path)?))
    }

    let Some(paths) = optimized_model_cache_paths(exec_config, component, source_path)? else {
        return build_session(exec_config, source_path, log_id, false, None);
    };

    if paths.final_path.exists() {
        trace_load(format!(
            "optimized cache hit {component} {}",
            paths.final_path.display()
        ));
        match build_session(exec_config, &paths.final_path, log_id, true, None) {
            Ok(session) => return Ok(session),
            Err(err) => {
                trace_load(format!(
                    "optimized cache load failed {component} {}: {err}",
                    paths.final_path.display()
                ));
                let _ = fs::remove_file(&paths.final_path);
                let _ = fs::remove_file(&paths.temp_path);
                if let Some(external_data_path) = paths.external_data_cache_path.as_ref() {
                    let _ = fs::remove_file(external_data_path);
                }
            }
        }
    }

    trace_load(format!(
        "optimized cache miss {component} source={}",
        source_path.display()
    ));
    let _ = fs::remove_file(&paths.temp_path);
    match build_session(
        exec_config,
        source_path,
        log_id,
        false,
        Some(&paths.temp_path),
    ) {
        Ok(session) => {
            if paths.temp_path.exists() {
                materialize_external_data(&paths, component);
                if let Err(err) = fs::rename(&paths.temp_path, &paths.final_path) {
                    trace_load(format!(
                        "optimized cache rename failed {component} {} -> {}: {err}",
                        paths.temp_path.display(),
                        paths.final_path.display()
                    ));
                    let _ = fs::remove_file(&paths.temp_path);
                }
            }
            Ok(session)
        }
        Err(err) => {
            trace_load(format!(
                "optimized cache write load failed {component}; retrying without cache: {err}"
            ));
            let _ = fs::remove_file(&paths.temp_path);
            build_session(exec_config, source_path, log_id, false, None)
        }
    }
}

#[derive(Clone)]
pub struct ProjectedKvCache {
    pub key: Array4<f32>,
    pub value: Array4<f32>,
}

/// Encoder cache state for Nemotron streaming inference.
/// Shapes are model/export-dependent, so always construct via
/// [`NemotronEncoderCache::with_dims`].
#[derive(Clone)]
pub struct NemotronEncoderCache {
    pub cache_last_channel: Array4<f32>,
    pub cache_last_time: Array4<f32>,
    pub cache_last_channel_len: Array1<i64>,
    pub projected_kv: Option<ProjectedKvCache>,
}

impl NemotronEncoderCache {
    pub fn with_dims(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            cache_last_channel: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            cache_last_time: Array4::zeros((num_layers, 1, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
            projected_kv: None,
        }
    }

    pub fn with_projected_dims(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            cache_last_channel: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            cache_last_time: Array4::zeros((num_layers, 1, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
            projected_kv: Some(ProjectedKvCache {
                key: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
                value: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            }),
        }
    }
}

fn streaming_frames_from_config(
    payload: &serde_json::Value,
    requested: Option<usize>,
) -> Result<usize> {
    let default_frames = payload
        .get("right_context")
        .and_then(|value| value.as_u64())
        .unwrap_or(6) as usize
        + 1;
    let frames = requested.unwrap_or(default_frames);
    if frames == 0 || default_frames == 0 {
        return Err(Error::Model(
            "streaming chunk frames must be positive".into(),
        ));
    }
    if let Some(dynamic) = payload
        .get("dynamic_streaming")
        .filter(|value| !value.is_null())
    {
        let valid = dynamic.get("format").and_then(|value| value.as_u64()) == Some(1)
            && dynamic
                .get("shape_derived_attention_context")
                .and_then(|value| value.as_bool())
                == Some(true)
            && dynamic
                .get("subsampling_factor")
                .and_then(|value| value.as_u64())
                == Some(8)
            && dynamic
                .get("mel_frames_overhead")
                .and_then(|value| value.as_u64())
                == Some(9)
            && dynamic
                .get("supported_chunk_frames")
                .and_then(|value| value.as_array())
                .is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| value.as_u64() == Some(frames as u64))
                });
        if !valid {
            return Err(Error::Model(
                "invalid or unsupported dynamic streaming contract".into(),
            ));
        }
    } else if frames != default_frames {
        return Err(Error::Model(
            "fixed encoder cannot change attention context; install a dynamic export".into(),
        ));
    }
    Ok(frames)
}

/// Nemotron ONNX wrapper.
/// Encoder and decoder_joint sessions live side by side; [`Self::has_prompt`]
/// flips on automatically when the encoder graph exposes a `prompt_index` input
/// (the multilingual variant).
pub struct NemotronModel {
    encoder: LoadedSession,
    decoder_joint: LoadedSession,
    pub config: NemotronModelConfig,
    pub has_prompt: bool,
    pub has_projected_kv_cache: bool,
    pub has_signal_length_input: bool,
}

/// cfg for Nemotron model dims.
#[derive(Debug, Clone)]
pub struct NemotronModelConfig {
    pub chunk_size_output_frames: usize,
    pub num_encoder_layers: usize,
    pub hidden_dim: usize,
    pub left_context: usize,
    pub conv_context: usize,
    pub decoder_lstm_dim: usize,
    pub decoder_lstm_layers: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
    pub has_projected_kv_cache: bool,
}

impl NemotronModel {
    /// Load encoder + decoder/joint sessions and read all dimension info
    /// straight from the encoder graph. `vocab_size` is supplied by the
    /// caller (it comes from the tokenizer).
    ///
    /// Note that, multilang graph is identified by the presence of a
    /// `prompt_index` input that flips [`Self::has_prompt`] on.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut exec_config = exec_config;
        let sidecar = model_dir.join("config.json");
        let selected_frames = if sidecar.exists() {
            let payload: serde_json::Value = serde_json::from_slice(&fs::read(&sidecar)?)?;
            let frames =
                streaming_frames_from_config(&payload, exec_config.streaming_chunk_frames)?;
            exec_config = exec_config
                .with_dimension_override("batch", 1)
                .with_dimension_override("mel_frames", (frames * 8 + 9) as i64)
                .with_dimension_override("encoder_frames", frames as i64)
                .with_dimension_override("current_frames", frames as i64);
            Some(frames)
        } else {
            if exec_config.streaming_chunk_frames.is_some() {
                return Err(Error::Model(
                    "explicit streaming context requires config.json".into(),
                ));
            }
            None
        };

        let encoder_path = model_component_path(model_dir, "encoder")?;
        let decoder_path = model_component_path(model_dir, "decoder_joint")?;

        trace_load(format!("loading encoder {}", encoder_path.display()));
        let encoder = load_session_with_optional_cache(
            &exec_config,
            "encoder",
            &encoder_path,
            "wordpipe-nemotron-encoder",
        )?;

        trace_load(format!("loading decoder {}", decoder_path.display()));
        let decoder_joint = load_session_with_optional_cache(
            &exec_config,
            "decoder",
            &decoder_path,
            "wordpipe-nemotron-decoder",
        )?;
        trace_load("sessions loaded");

        let mut config = NemotronModelConfig {
            num_encoder_layers: 24,
            hidden_dim: 1024,
            left_context: 70,
            conv_context: 8,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            vocab_size,
            blank_id: vocab_size,
            has_projected_kv_cache: false,
            chunk_size_output_frames: 7,
        };

        if let Some(frames) = encoder.with_session(|session| {
            session
                .metadata()
                .ok()
                .and_then(|meta| meta.custom("chunk_size_output_frames"))
                .and_then(|value| value.parse::<usize>().ok())
        }) {
            config.chunk_size_output_frames = frames;
        }
        if let Some(frames) = selected_frames {
            config.chunk_size_output_frames = frames;
        }
        if config.chunk_size_output_frames == 0 {
            return Err(Error::Model(
                "streaming chunk frames must be positive".into(),
            ));
        }

        let mut has_prompt = false;
        let mut has_projected_kv_cache = false;
        let mut has_signal_length_input = false;
        encoder.with_session(|session| {
        for outlet in session.inputs() {
            let name = outlet.name();
            if name == "prompt_index" {
                has_prompt = true;
                continue;
            }
            if name == "processed_signal_length" {
                has_signal_length_input = true;
                continue;
            }
            if name == "cache_key_layer_0" {
                has_projected_kv_cache = true;
                config.has_projected_kv_cache = true;
                continue;
            }
            let ValueType::Tensor { shape, .. } = outlet.dtype() else { continue };
            let dims: &[i64] = shape;
            match name {
                "processed_signal" if dims.len() == 3 && dims[2] > 0 => {
                    let expected = (config.chunk_size_output_frames * 8 + 9) as i64;
                    if dims[2] != expected {
                        return Err(Error::Model(format!(
                            "encoder input has {} mel frames, but mode requires {expected}",
                            dims[2]
                        )));
                    }
                }
                "cache_last_channel" if dims.len() == 4 => {
                    config.num_encoder_layers = dims[0] as usize;
                    config.left_context = dims[2] as usize;
                    config.hidden_dim = dims[3] as usize;
                }
                "cache_last_time" if dims.len() == 4 => {
                    config.conv_context = dims[3] as usize;
                }
                _ => {}
            }
        }

        trace_load(format!(
            "config layers={} left_context={} hidden={} conv_context={} prompt={} projected={} length_input={}",
            config.num_encoder_layers,
            config.left_context,
            config.hidden_dim,
            config.conv_context,
            has_prompt,
            has_projected_kv_cache,
            has_signal_length_input
        ));
        Ok(())
        })?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
            has_prompt,
            has_projected_kv_cache,
            has_signal_length_input,
        })
    }

    /// Run encoder with cache-aware streaming.
    /// `prompt_index` must be `Some(_)` for multilingual models and `None`
    /// for eng only mistmaching will produce an ORT InvalidArgument err.
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &NemotronEncoderCache,
        prompt_index: Option<i64>,
    ) -> Result<(Array3<f32>, i64, NemotronEncoderCache)> {
        let mut new_cache = cache.clone();
        let (encoded, encoded_len) =
            self.run_encoder_into(features, length, &mut new_cache, prompt_index)?;
        Ok((encoded, encoded_len, new_cache))
    }

    /// Run encoder with cache-aware streaming and update `cache` in place.
    pub fn run_encoder_into(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &mut NemotronEncoderCache,
        prompt_index: Option<i64>,
    ) -> Result<(Array3<f32>, i64)> {
        self.encoder.with_session_mut(|session| {
            let features_value = TensorRef::<f32>::from_array_view(features.view())?;
            let cache_last_channel_value =
                TensorRef::<f32>::from_array_view(cache.cache_last_channel.view())?;
            let cache_last_time_value =
                TensorRef::<f32>::from_array_view(cache.cache_last_time.view())?;
            let cache_last_channel_len_value =
                TensorRef::<i64>::from_array_view(cache.cache_last_channel_len.view())?;

            let mut inputs = ort::inputs![
                "processed_signal" => features_value,
                "cache_last_channel" => cache_last_channel_value,
                "cache_last_time" => cache_last_time_value,
                "cache_last_channel_len" => cache_last_channel_len_value
            ];
            let length_arr = [length];
            if self.has_signal_length_input {
                let length_value = TensorRef::<i64>::from_array_view(([1usize], &length_arr[..]))?;
                inputs.push((
                    Cow::Borrowed("processed_signal_length"),
                    SessionInputValue::from(length_value),
                ));
            }
            let prompt_arr = prompt_index.map(|idx| [idx]);
            if let Some(prompt_arr) = prompt_arr.as_ref() {
                let prompt_value = TensorRef::<i64>::from_array_view(([1usize], &prompt_arr[..]))?;
                inputs.push((
                    Cow::Borrowed("prompt_index"),
                    SessionInputValue::from(prompt_value),
                ));
            }
            if self.has_projected_kv_cache {
                let projected = cache.projected_kv.as_ref().ok_or_else(|| {
                    Error::Model("projected K/V cache graph requires projected cache state".into())
                })?;
                let layers = projected.key.shape()[0];
                for layer in 0..layers {
                    let key = TensorRef::<f32>::from_array_view((
                        [1usize, projected.key.shape()[2], projected.key.shape()[3]],
                        projected_layer_slice(&projected.key, layer)?,
                    ))?;
                    inputs.push((
                        Cow::Owned(format!("cache_key_layer_{layer}")),
                        SessionInputValue::from(key),
                    ));

                    let value = TensorRef::<f32>::from_array_view((
                        [
                            1usize,
                            projected.value.shape()[2],
                            projected.value.shape()[3],
                        ],
                        projected_layer_slice(&projected.value, layer)?,
                    ))?;
                    inputs.push((
                        Cow::Owned(format!("cache_value_layer_{layer}")),
                        SessionInputValue::from(value),
                    ));
                }
            }

            let outputs = session.run(inputs)?;

            // [1, hidden_dim, time]
            let (shape, data) = outputs["encoded"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

            let shape_dims = shape.as_ref();
            let b = shape_dims[0] as usize;
            let d = shape_dims[1] as usize;
            let t = shape_dims[2] as usize;

            let encoder_out = Array3::from_shape_vec((b, d, t), data.to_vec())
                .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

            // on here we are extracting encoded length and new cache states.. and so on...
            let (_, enc_len_data) = outputs["encoded_len"]
                .try_extract_tensor::<i64>()
                .map_err(|e| Error::Model(format!("Failed to extract encoded_len: {e}")))?;
            let encoded_len = enc_len_data[0];

            let (ch_shape, ch_data) = outputs["cache_last_channel_next"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract cache_last_channel: {e}")))?;

            let (tm_shape, tm_data) = outputs["cache_last_time_next"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract cache_last_time: {e}")))?;

            let (len_shape, len_data) = outputs["cache_last_channel_len_next"]
                .try_extract_tensor::<i64>()
                .map_err(|e| Error::Model(format!("Failed to extract cache_len: {e}")))?;

            if self.has_projected_kv_cache {
                let projected = cache.projected_kv.as_mut().ok_or_else(|| {
                    Error::Model(
                        "projected K/V cache graph returned outputs but cache state is missing"
                            .into(),
                    )
                })?;
                for layer in 0..self.config.num_encoder_layers {
                    let key_name = format!("projected_current_key_layer_{layer}");
                    let value_name = format!("projected_current_value_layer_{layer}");
                    let (key_shape, key_data) = outputs[key_name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| Error::Model(format!("Failed to extract {key_name}: {e}")))?;
                    roll_projected_layer_from_slice(
                        &mut projected.key,
                        layer,
                        key_shape.as_ref(),
                        key_data,
                    )?;

                    let (value_shape, value_data) = outputs[value_name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| {
                            Error::Model(format!("Failed to extract {value_name}: {e}"))
                        })?;
                    roll_projected_layer_from_slice(
                        &mut projected.value,
                        layer,
                        value_shape.as_ref(),
                        value_data,
                    )?;
                }
            }

            copy_output_to_array4(
                "cache_last_channel",
                &mut cache.cache_last_channel,
                ch_shape.as_ref(),
                ch_data,
            )?;
            copy_output_to_array4(
                "cache_last_time",
                &mut cache.cache_last_time,
                tm_shape.as_ref(),
                tm_data,
            )?;
            copy_output_to_array1_i64(
                "cache_last_channel_len",
                &mut cache.cache_last_channel_len,
                len_shape.as_ref(),
                len_data,
            )?;

            Ok((encoder_out, encoded_len))
        })
    }

    /// Run decoder step.
    /// Returns: (logits [vocab_size], new_state_1, new_state_2)
    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, hidden_dim, 1]
        target_token: i32,
        state_1: &Array3<f32>, // [2, 1, 640]
        state_2: &Array3<f32>, // [2, 1, 640]
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let targets = [target_token];
        let target_len = [1i32];
        let encoder_frame_value = TensorRef::<f32>::from_array_view(encoder_frame.view())?;
        let targets_value = TensorRef::<i32>::from_array_view(([1usize, 1usize], &targets[..]))?;
        let target_len_value = TensorRef::<i32>::from_array_view(([1usize], &target_len[..]))?;
        let state_1_value = TensorRef::<f32>::from_array_view(state_1.view())?;
        let state_2_value = TensorRef::<f32>::from_array_view(state_2.view())?;

        self.decoder_joint.with_session_mut(|session| {
            let outputs = session.run(ort::inputs![
                "encoder_outputs" => encoder_frame_value,
                "targets" => targets_value,
                "target_length" => target_len_value,
                "input_states_1" => state_1_value,
                "input_states_2" => state_2_value
            ])?;

            // logits for others I think you can understand by looking at the error msgs right?
            let (_l_shape, l_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

            let logits = Array1::from_vec(l_data.to_vec());

            let (h_shape, h_data) = outputs["output_states_1"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract state_1: {e}")))?;

            let (c_shape, c_data) = outputs["output_states_2"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract state_2: {e}")))?;

            let new_state_1 = Array3::from_shape_vec(
                (
                    h_shape[0] as usize,
                    h_shape[1] as usize,
                    h_shape[2] as usize,
                ),
                h_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape state_1: {e}")))?;

            let new_state_2 = Array3::from_shape_vec(
                (
                    c_shape[0] as usize,
                    c_shape[1] as usize,
                    c_shape[2] as usize,
                ),
                c_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape state_2: {e}")))?;

            Ok((logits, new_state_1, new_state_2))
        })
    }
}

fn projected_layer_slice(cache: &Array4<f32>, layer: usize) -> Result<&[f32]> {
    let shape = cache.shape();
    if layer >= shape[0] {
        return Err(Error::Model(format!(
            "projected cache layer out of range: {layer}"
        )));
    }
    let layer_values = shape[1] * shape[2] * shape[3];
    let start = layer * layer_values;
    let end = start + layer_values;
    cache
        .as_slice()
        .and_then(|values| values.get(start..end))
        .ok_or_else(|| Error::Model("projected cache is not contiguous".to_string()))
}

fn projected_layer_slice_mut(cache: &mut Array4<f32>, layer: usize) -> Result<&mut [f32]> {
    let cache_shape = cache.shape();
    if layer >= cache_shape[0] {
        return Err(Error::Model(format!(
            "projected cache layer out of range: {layer}"
        )));
    }
    let layer_values = cache_shape[1] * cache_shape[2] * cache_shape[3];
    let start = layer * layer_values;
    let end = start + layer_values;
    cache
        .as_slice_mut()
        .and_then(|values| values.get_mut(start..end))
        .ok_or_else(|| Error::Model("projected cache is not contiguous".to_string()))
}

fn roll_projected_layer_from_slice(
    cache: &mut Array4<f32>,
    layer: usize,
    current_shape: &[i64],
    current: &[f32],
) -> Result<()> {
    let cache_shape = cache.shape();
    if current_shape.len() != 3
        || current_shape[0] as usize != cache_shape[1]
        || current_shape[2] as usize != cache_shape[3]
    {
        return Err(Error::Model(format!(
            "projected cache shape mismatch: cache={cache_shape:?}, current={current_shape:?}",
        )));
    }

    let batch = cache_shape[1];
    let cache_len = cache_shape[2];
    let hidden = cache_shape[3];
    let current_frames = (current_shape[1] as usize).min(cache_len);
    if current_frames == 0 {
        return Ok(());
    }
    let input_frames = current_shape[1] as usize;
    let expected_values = batch * input_frames * hidden;
    if current.len() != expected_values {
        return Err(Error::Model(format!(
            "projected cache data length mismatch: got {}, expected {}",
            current.len(),
            expected_values
        )));
    }

    let keep = cache_len - current_frames;
    let layer_cache = projected_layer_slice_mut(cache, layer)?;
    for b in 0..batch {
        let cache_batch_start = b * cache_len * hidden;
        let cache_batch =
            &mut layer_cache[cache_batch_start..cache_batch_start + cache_len * hidden];
        let input_batch_start = b * input_frames * hidden;
        let input_batch = &current[input_batch_start..input_batch_start + input_frames * hidden];
        if current_frames >= cache_len {
            let input_start = (input_frames - cache_len) * hidden;
            cache_batch.copy_from_slice(&input_batch[input_start..]);
        } else {
            cache_batch.copy_within(current_frames * hidden.., 0);
            let input_start = (input_frames - current_frames) * hidden;
            cache_batch[keep * hidden..].copy_from_slice(&input_batch[input_start..]);
        }
    }
    Ok(())
}

fn copy_output_to_array4(
    name: &str,
    destination: &mut Array4<f32>,
    shape: &[i64],
    data: &[f32],
) -> Result<()> {
    let expected_shape = destination.shape();
    if shape.len() != 4
        || shape
            .iter()
            .map(|dim| *dim as usize)
            .ne(expected_shape.iter().copied())
    {
        return Err(Error::Model(format!(
            "Unexpected {name} shape: got {shape:?}, expected {expected_shape:?}"
        )));
    }
    let destination = destination
        .as_slice_mut()
        .ok_or_else(|| Error::Model(format!("{name} is not contiguous")))?;
    if destination.len() != data.len() {
        return Err(Error::Model(format!(
            "Unexpected {name} data length: got {}, expected {}",
            data.len(),
            destination.len()
        )));
    }
    destination.copy_from_slice(data);
    Ok(())
}

#[cfg(test)]
mod runtime_lifetime_tests {
    use super::*;
    use std::sync::{atomic::Ordering, Arc};

    #[test]
    fn generic_encoder_accepts_each_supported_context() {
        let payload = serde_json::json!({
            "right_context": 6,
            "dynamic_streaming": {
                "format": 1, "shape_derived_attention_context": true,
                "subsampling_factor": 8, "mel_frames_overhead": 9,
                "supported_chunk_frames": [1, 2, 7, 14]
            }
        });
        assert_eq!(streaming_frames_from_config(&payload, None).unwrap(), 7);
        assert_eq!(streaming_frames_from_config(&payload, Some(7)).unwrap(), 7);
        assert_eq!(streaming_frames_from_config(&payload, Some(1)).unwrap(), 1);
        assert_eq!(streaming_frames_from_config(&payload, Some(2)).unwrap(), 2);
        assert_eq!(
            streaming_frames_from_config(&payload, Some(14)).unwrap(),
            14
        );
        assert!(streaming_frames_from_config(&payload, Some(8)).is_err());
    }

    #[test]
    fn fixed_encoder_cannot_silently_switch_attention_context() {
        let payload = serde_json::json!({"right_context": 6});
        assert_eq!(streaming_frames_from_config(&payload, Some(7)).unwrap(), 7);
        assert!(streaming_frames_from_config(&payload, Some(14)).is_err());
    }

    #[test]
    fn malformed_dynamic_contract_is_not_treated_as_fixed() {
        let payload = serde_json::json!({"right_context": 6, "dynamic_streaming": true});
        assert!(streaming_frames_from_config(&payload, None).is_err());
    }

    #[test]
    fn backing_bytes_are_freed_when_session_creation_fails() {
        let probe = Arc::new(LifetimeProbe::default());
        let owner = ModelBytes {
            data: vec![0_u8; 8].into_boxed_slice(),
            probe: Some(probe.clone()),
        };
        let result = DirectOrtSession::try_new(owner, |_| Err::<InMemorySession<'_>, _>("failure"));
        assert!(result.is_err());
        assert_eq!(probe.freed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn direct_ort_session_drops_before_its_backing_bytes() {
        let Some(directory) = std::env::var_os("WORDPIPE_ORT_TEST_FIXTURES") else {
            return;
        };
        let path = PathBuf::from(directory).join("session.ort");
        for _ in 0..3 {
            let probe = Arc::new(LifetimeProbe::default());
            let owner = ModelBytes {
                data: fs::read(&path).unwrap().into_boxed_slice(),
                probe: Some(probe.clone()),
            };
            let mut builder = Session::builder().unwrap();
            let mut cell = DirectOrtSession::try_new(owner, |bytes| {
                builder.commit_from_memory_directly(&bytes.data)
            })
            .unwrap();
            cell.with_dependent_mut(|_, session| {
                *probe.session.lock().unwrap() = Some(Arc::downgrade(&session.inner()));
                let data = Array3::<f32>::zeros((1, 128, 65));
                let outputs = session
                    .run(ort::inputs![
                        "audio" => TensorRef::from_array_view(data.view()).unwrap()
                    ])
                    .unwrap();
                let (_, shape) = outputs["shape"].try_extract_tensor::<i64>().unwrap();
                assert_eq!(shape, &[1, 128, 65]);
            });
            assert_eq!(probe.freed.load(Ordering::SeqCst), 0);
            drop(cell);
            assert_eq!(probe.freed.load(Ordering::SeqCst), 1);
            assert!(probe.native_session_gone.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn optimized_cache_separates_dimensions_and_reuses_each_mode() {
        let Some(directory) = std::env::var_os("WORDPIPE_ORT_TEST_FIXTURES") else {
            return;
        };
        let source = PathBuf::from(directory).join("source.onnx");
        let cache = std::env::temp_dir().join(format!(
            "wordpipe-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut previous = None;
        for frames in [65_i64, 121, 65] {
            let config = ExecutionConfig::default()
                .with_ort_optimized_model_cache_dir(&cache)
                .with_dimension_override("frames", frames);
            let paths = optimized_model_cache_paths(&config, "encoder", &source)
                .unwrap()
                .unwrap();
            let mut session =
                load_session_with_optional_cache(&config, "encoder", &source, "shape-test")
                    .unwrap();
            session.with_session_mut(|session| {
                let data = Array3::<f32>::zeros((1, 128, frames as usize));
                let outputs = session
                    .run(ort::inputs![
                        "audio" => TensorRef::from_array_view(data.view()).unwrap()
                    ])
                    .unwrap();
                let (_, shape) = outputs["shape"].try_extract_tensor::<i64>().unwrap();
                assert_eq!(shape, &[1, 128, frames]);
            });
            assert!(paths.final_path.is_file());
            if frames == 65 {
                if let Some(previous) = &previous {
                    assert_eq!(previous, &paths.final_path);
                } else {
                    previous = Some(paths.final_path);
                }
            } else {
                assert_ne!(previous.as_ref().unwrap(), &paths.final_path);
            }
        }
        fs::remove_dir_all(cache).unwrap();
    }

    #[test]
    fn pool_stop_policy_has_a_distinct_cache_key_without_disabling_cache() {
        let Some(directory) = std::env::var_os("WORDPIPE_ORT_TEST_FIXTURES") else {
            return;
        };
        let source = PathBuf::from(directory).join("source.onnx");
        let config = ExecutionConfig::default().with_ort_optimized_model_cache_dir(
            std::env::temp_dir().join("wordpipe-cache-key-only"),
        );
        let normal = optimized_model_cache_paths(&config, "encoder", &source)
            .unwrap()
            .unwrap();
        let stopped =
            optimized_model_cache_paths(&config.with_force_spinning_stop(true), "encoder", &source)
                .unwrap()
                .unwrap();
        assert_ne!(normal.final_path, stopped.final_path);
    }
}

fn copy_output_to_array1_i64(
    name: &str,
    destination: &mut Array1<i64>,
    shape: &[i64],
    data: &[i64],
) -> Result<()> {
    if shape.len() != 1 || shape[0] as usize != destination.len() {
        return Err(Error::Model(format!(
            "Unexpected {name} shape: got {shape:?}, expected [{}]",
            destination.len()
        )));
    }
    let destination = destination
        .as_slice_mut()
        .ok_or_else(|| Error::Model(format!("{name} is not contiguous")))?;
    if destination.len() != data.len() {
        return Err(Error::Model(format!(
            "Unexpected {name} data length: got {}, expected {}",
            data.len(),
            destination.len()
        )));
    }
    destination.copy_from_slice(data);
    Ok(())
}
