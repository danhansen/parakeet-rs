use crate::android_log;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{IoBinding, Session};
use ort::value::{Outlet, Tensor, TensorRef, ValueType};
use std::ffi::{c_char, c_void, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;
use std::time::{Instant, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EncoderCacheAbi {
    RawChannel,
    ProjectedKvStacked,
    ProjectedKvLayered,
}

#[derive(Clone, Copy, Debug)]
pub struct EncoderLayout {
    pub batch: usize,
    pub n_mels: usize,
    pub input_frames: usize,
    pub output_frames: usize,
    pub num_layers: usize,
    pub cache_len: usize,
    pub hidden_size: usize,
    pub time_cache_len: usize,
}

impl EncoderLayout {
    fn projected_cache_layer_values(&self) -> usize {
        self.batch * self.cache_len * self.hidden_size
    }

    fn projected_current_layer_values(&self) -> usize {
        self.batch * self.output_frames * self.hidden_size
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiteRtEncoderLayout {
    pub batch: i64,
    pub n_mels: i64,
    pub input_frames: i64,
    pub output_frames: i64,
    pub num_layers: i64,
    pub cache_len: i64,
    pub hidden_size: i64,
    pub time_cache_len: i64,
    pub cache_abi: i64,
}

impl TryFrom<LiteRtEncoderLayout> for EncoderLayout {
    type Error = Error;

    fn try_from(value: LiteRtEncoderLayout) -> Result<Self> {
        let dims = [
            ("batch", value.batch),
            ("n_mels", value.n_mels),
            ("input_frames", value.input_frames),
            ("output_frames", value.output_frames),
            ("num_layers", value.num_layers),
            ("cache_len", value.cache_len),
            ("hidden_size", value.hidden_size),
            ("time_cache_len", value.time_cache_len),
        ];
        for (name, dim) in dims {
            if dim <= 0 {
                return Err(Error::Model(format!(
                    "LiteRT encoder layout has invalid {name}={dim}"
                )));
            }
        }
        Ok(Self {
            batch: value.batch as usize,
            n_mels: value.n_mels as usize,
            input_frames: value.input_frames as usize,
            output_frames: value.output_frames as usize,
            num_layers: value.num_layers as usize,
            cache_len: value.cache_len as usize,
            hidden_size: value.hidden_size as usize,
            time_cache_len: value.time_cache_len as usize,
        })
    }
}

fn litert_cache_abi_from_i64(value: i64) -> Result<EncoderCacheAbi> {
    match value {
        0 => Ok(EncoderCacheAbi::RawChannel),
        2 => Ok(EncoderCacheAbi::ProjectedKvLayered),
        other => Err(Error::Model(format!(
            "LiteRT encoder reported unsupported cache ABI {other}"
        ))),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiteRtEncoderApi {
    pub version: u32,
    pub create: Option<
        unsafe extern "C" fn(
            model_path: *const c_char,
            num_threads: i32,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> *mut c_void,
    >,
    pub destroy: Option<unsafe extern "C" fn(handle: *mut c_void)>,
    pub run: Option<
        unsafe extern "C" fn(
            handle: *mut c_void,
            audio_signal: *const f32,
            audio_signal_len: usize,
            cache_last_channel: *const f32,
            cache_last_channel_len: usize,
            cache_last_time: *const f32,
            cache_last_time_len: usize,
            cache_last_channel_len_values: *const i64,
            cache_last_channel_len_count: usize,
            outputs: *mut f32,
            outputs_len: usize,
            encoded_lengths: *mut i64,
            encoded_lengths_len: usize,
            new_cache_last_channel: *mut f32,
            new_cache_last_channel_len: usize,
            new_cache_last_time: *mut f32,
            new_cache_last_time_len: usize,
            new_cache_last_channel_len_values: *mut i64,
            new_cache_last_channel_len_count: usize,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> bool,
    >,
    pub run_projected_layered: Option<
        unsafe extern "C" fn(
            handle: *mut c_void,
            audio_signal: *const f32,
            audio_signal_len: usize,
            cache_last_channel: *const f32,
            cache_last_channel_len: usize,
            cache_last_time: *const f32,
            cache_last_time_len: usize,
            cache_last_channel_len_values: *const i64,
            cache_last_channel_len_count: usize,
            cache_last_key: *const f32,
            cache_last_key_len: usize,
            cache_last_value: *const f32,
            cache_last_value_len: usize,
            outputs: *mut f32,
            outputs_len: usize,
            encoded_lengths: *mut i64,
            encoded_lengths_len: usize,
            new_cache_last_channel: *mut f32,
            new_cache_last_channel_len: usize,
            new_cache_last_time: *mut f32,
            new_cache_last_time_len: usize,
            new_cache_last_channel_len_values: *mut i64,
            new_cache_last_channel_len_count: usize,
            projected_current_key: *mut f32,
            projected_current_key_len: usize,
            projected_current_value: *mut f32,
            projected_current_value_len: usize,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> bool,
    >,
    pub get_encoder_layout: Option<
        unsafe extern "C" fn(
            handle: *mut c_void,
            layout: *mut LiteRtEncoderLayout,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> bool,
    >,
    pub create_decoder: Option<
        unsafe extern "C" fn(
            model_path: *const c_char,
            num_threads: i32,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> *mut c_void,
    >,
    pub destroy_decoder: Option<unsafe extern "C" fn(handle: *mut c_void)>,
    pub run_decoder: Option<
        unsafe extern "C" fn(
            handle: *mut c_void,
            encoder_frame: *const f32,
            encoder_frame_len: usize,
            targets: *const i32,
            targets_len: usize,
            state_h: *const f32,
            state_h_len: usize,
            state_c: *const f32,
            state_c_len: usize,
            logits: *mut f32,
            logits_len: usize,
            new_h: *mut f32,
            new_h_len: usize,
            new_c: *mut f32,
            new_c_len: usize,
            error_buffer: *mut c_char,
            error_buffer_len: usize,
        ) -> bool,
    >,
}

const LITERT_ENCODER_API_VERSION: u32 = 3;
static LITERT_ENCODER_API: Mutex<Option<LiteRtEncoderApi>> = Mutex::new(None);

pub fn set_litert_encoder_api(api: *const LiteRtEncoderApi) -> bool {
    if api.is_null() {
        return false;
    }
    let api = unsafe { *api };
    if api.version != LITERT_ENCODER_API_VERSION
        || api.create.is_none()
        || api.destroy.is_none()
        || api.run.is_none()
        || api.run_projected_layered.is_none()
        || api.get_encoder_layout.is_none()
        || api.create_decoder.is_none()
        || api.destroy_decoder.is_none()
        || api.run_decoder.is_none()
    {
        return false;
    }
    match LITERT_ENCODER_API.lock() {
        Ok(mut guard) => {
            *guard = Some(api);
            true
        }
        Err(_) => false,
    }
}

fn get_litert_encoder_api() -> Option<LiteRtEncoderApi> {
    LITERT_ENCODER_API.lock().ok().and_then(|guard| *guard)
}

fn read_c_error(buffer: &[c_char]) -> String {
    let nul_index = buffer
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(buffer.len());
    let bytes = buffer[..nul_index]
        .iter()
        .map(|value| *value as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

struct OptimizedModelCachePaths {
    final_path: PathBuf,
    temp_path: PathBuf,
}

fn optimized_model_cache_paths(
    exec_config: &ExecutionConfig,
    component: &str,
    source_path: &Path,
) -> Result<Option<OptimizedModelCachePaths>> {
    let Some(cache_dir) = exec_config.ort_optimized_model_cache_dir() else {
        return Ok(None);
    };
    fs::create_dir_all(cache_dir)?;
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_path_component)
        .unwrap_or_else(|| "model".to_string());
    let key = optimized_model_cache_key(exec_config, component, source_path)?;
    let filename = format!("{component}.{stem}.{key}.optimized.onnx");
    Ok(Some(OptimizedModelCachePaths {
        final_path: cache_dir.join(&filename),
        temp_path: cache_dir.join(format!("{filename}.tmp")),
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
    update_fnv1a(&mut hash, b"sayboard-parakeet-ort-cache-v1");
    update_fnv1a(&mut hash, component.as_bytes());
    update_fnv1a(&mut hash, source_path.to_string_lossy().as_bytes());
    update_fnv1a(&mut hash, metadata.len().to_string().as_bytes());
    update_fnv1a(&mut hash, modified_ns.to_string().as_bytes());
    update_fnv1a(
        &mut hash,
        format!("{:?}", exec_config.execution_provider).as_bytes(),
    );
    update_fnv1a(&mut hash, exec_config.intra_threads.to_string().as_bytes());
    update_fnv1a(&mut hash, exec_config.inter_threads.to_string().as_bytes());
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

/// Encoder cache state for streaming inference
/// The cache maintains temporal context across chunks
pub struct EncoderCache {
    /// Raw channel cache, shaped from the encoder's `cache_last_channel` input.
    pub cache_last_channel: Array4<f32>,
    /// Projected attention key cache, shaped from the encoder's cache ABI.
    pub cache_last_key: Array4<f32>,
    /// Projected attention value cache, shaped from the encoder's cache ABI.
    pub cache_last_value: Array4<f32>,
    /// Time cache, shaped from the encoder's `cache_last_time` input.
    pub cache_last_time: Array4<f32>,
    /// Cache length vector, one value per batch item.
    pub cache_last_channel_len: Array1<i64>,
}

impl EncoderCache {
    pub fn new(layout: EncoderLayout) -> Self {
        Self {
            cache_last_channel: Array4::zeros((
                layout.num_layers,
                layout.batch,
                layout.cache_len,
                layout.hidden_size,
            )),
            cache_last_key: Array4::zeros((
                layout.num_layers,
                layout.batch,
                layout.cache_len,
                layout.hidden_size,
            )),
            cache_last_value: Array4::zeros((
                layout.num_layers,
                layout.batch,
                layout.cache_len,
                layout.hidden_size,
            )),
            cache_last_time: Array4::zeros((
                layout.num_layers,
                layout.batch,
                layout.hidden_size,
                layout.time_cache_len,
            )),
            cache_last_channel_len: Array1::zeros(layout.batch),
        }
    }
}

fn cache_layer_input_name(kind: &str, layer: usize) -> String {
    format!("cache_{kind}_layer_{layer}")
}

fn projected_current_output_name(kind: &str, layer: usize) -> String {
    format!("projected_current_{kind}_layer_{layer}")
}

fn has_all_layered_projected_cache_inputs(input_names: &[String], num_layers: usize) -> bool {
    (0..num_layers).all(|layer| {
        input_names
            .iter()
            .any(|name| name == &cache_layer_input_name("key", layer))
            && input_names
                .iter()
                .any(|name| name == &cache_layer_input_name("value", layer))
    })
}

fn has_all_layered_projected_cache_outputs(output_names: &[String], num_layers: usize) -> bool {
    (0..num_layers).all(|layer| {
        output_names
            .iter()
            .any(|name| name == &projected_current_output_name("key", layer))
            && output_names
                .iter()
                .any(|name| name == &projected_current_output_name("value", layer))
    })
}

fn contiguous_layer_count(
    names: &[String],
    key_prefix: &str,
    value_prefix: &str,
) -> Result<Option<usize>> {
    let mut key_layers = Vec::new();
    let mut value_layers = Vec::new();
    for name in names {
        if let Some(suffix) = name.strip_prefix(key_prefix) {
            if let Ok(layer) = suffix.parse::<usize>() {
                key_layers.push(layer);
            }
        } else if let Some(suffix) = name.strip_prefix(value_prefix) {
            if let Ok(layer) = suffix.parse::<usize>() {
                value_layers.push(layer);
            }
        }
    }
    if key_layers.is_empty() || value_layers.is_empty() {
        return Ok(None);
    }
    key_layers.sort_unstable();
    key_layers.dedup();
    value_layers.sort_unstable();
    value_layers.dedup();
    let count = key_layers.iter().max().copied().unwrap_or(0) + 1;
    let expected = (0..count).collect::<Vec<_>>();
    if key_layers != expected || value_layers != expected {
        return Err(Error::Model(format!(
            "Layered projected-cache ABI has non-contiguous layers: keys={key_layers:?} values={value_layers:?}"
        )));
    }
    Ok(Some(count))
}

fn layered_projected_cache_layer_count(encoder: &Session) -> Result<Option<usize>> {
    let input_names = encoder
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect::<Vec<_>>();
    let output_names = encoder
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect::<Vec<_>>();
    let input_count =
        contiguous_layer_count(&input_names, "cache_key_layer_", "cache_value_layer_")?;
    let output_count = contiguous_layer_count(
        &output_names,
        "projected_current_key_layer_",
        "projected_current_value_layer_",
    )?;
    match (input_count, output_count) {
        (Some(inputs), Some(outputs)) if inputs != outputs => Err(Error::Model(format!(
            "Layered projected-cache input/output layer mismatch: inputs={inputs} outputs={outputs}"
        ))),
        (Some(count), _) | (_, Some(count)) => Ok(Some(count)),
        (None, None) => Ok(None),
    }
}

fn tensor_shape(outlet: &Outlet, expected_rank: usize) -> Result<Vec<usize>> {
    let shape = match outlet.dtype() {
        ValueType::Tensor { shape, .. } => shape,
        other => {
            return Err(Error::Model(format!(
                "{} is not a tensor: {other:?}",
                outlet.name()
            )));
        }
    };
    if shape.len() != expected_rank {
        return Err(Error::Model(format!(
            "{} has rank {}, expected {}: {:?}",
            outlet.name(),
            shape.len(),
            expected_rank,
            shape
        )));
    }
    shape
        .iter()
        .map(|dim| {
            if *dim <= 0 {
                Err(Error::Model(format!(
                    "{} has non-static shape {:?}; use the fixed-shape encoder variant",
                    outlet.name(),
                    shape
                )))
            } else {
                Ok(*dim as usize)
            }
        })
        .collect()
}

fn input_shape(encoder: &Session, name: &str, expected_rank: usize) -> Result<Vec<usize>> {
    let outlet = encoder
        .inputs()
        .iter()
        .find(|input| input.name() == name)
        .ok_or_else(|| Error::Model(format!("Missing encoder input {name}")))?;
    tensor_shape(outlet, expected_rank)
}

fn output_shape(encoder: &Session, name: &str, expected_rank: usize) -> Result<Vec<usize>> {
    let outlet = encoder
        .outputs()
        .iter()
        .find(|output| output.name() == name)
        .ok_or_else(|| Error::Model(format!("Missing encoder output {name}")))?;
    tensor_shape(outlet, expected_rank)
}

fn infer_encoder_layout(encoder: &Session) -> Result<EncoderLayout> {
    let audio_signal = input_shape(encoder, "audio_signal", 3)?;
    let cache_last_channel = input_shape(encoder, "cache_last_channel", 4)?;
    let cache_last_time = input_shape(encoder, "cache_last_time", 4)?;
    let outputs = output_shape(encoder, "outputs", 3)?;
    let layered_num_layers = layered_projected_cache_layer_count(encoder)?;
    let num_layers = layered_num_layers.unwrap_or(cache_last_channel[0]);

    if audio_signal[0] != outputs[0] {
        return Err(Error::Model(format!(
            "Encoder batch mismatch: audio_signal={audio_signal:?}, outputs={outputs:?}"
        )));
    }
    if layered_num_layers.is_none() && cache_last_channel[0] != cache_last_time[0] {
        return Err(Error::Model(format!(
            "Encoder raw cache layer mismatch: cache_last_channel={cache_last_channel:?}, \
             cache_last_time={cache_last_time:?}"
        )));
    }
    if cache_last_channel[1] != audio_signal[0]
        || cache_last_time[1] != audio_signal[0]
        || cache_last_channel[3] != cache_last_time[2]
        || outputs[1] != cache_last_channel[3]
    {
        return Err(Error::Model(format!(
            "Encoder cache/output shapes are inconsistent: audio_signal={audio_signal:?}, \
             outputs={outputs:?}, cache_last_channel={cache_last_channel:?}, \
             cache_last_time={cache_last_time:?}"
        )));
    }

    Ok(EncoderLayout {
        batch: audio_signal[0],
        n_mels: audio_signal[1],
        input_frames: audio_signal[2],
        output_frames: outputs[2],
        num_layers,
        cache_len: cache_last_channel[2],
        hidden_size: cache_last_channel[3],
        time_cache_len: cache_last_time[3],
    })
}

fn projected_cache_layer_slice(
    cache: &Array4<f32>,
    layer: usize,
    layout: EncoderLayout,
) -> Result<&[f32]> {
    let all = cache
        .as_slice()
        .ok_or_else(|| Error::Model("projected cache is not contiguous".to_string()))?;
    let layer_values = layout.projected_cache_layer_values();
    let start = layer * layer_values;
    let end = start + layer_values;
    all.get(start..end)
        .ok_or_else(|| Error::Model(format!("projected cache layer {layer} is out of bounds")))
}

fn projected_current_layer_slice(
    cache: &Array4<f32>,
    layer: usize,
    layout: EncoderLayout,
) -> Result<&[f32]> {
    let all = cache
        .as_slice()
        .ok_or_else(|| Error::Model("projected current cache is not contiguous".to_string()))?;
    let layer_values = layout.projected_current_layer_values();
    let start = layer * layer_values;
    let end = start + layer_values;
    all.get(start..end).ok_or_else(|| {
        Error::Model(format!(
            "projected current cache layer {layer} is out of bounds"
        ))
    })
}

fn roll_projected_cache_layer(
    cache: &mut Array4<f32>,
    layer: usize,
    layout: EncoderLayout,
    shape: &[i64],
    current_data: &[f32],
) -> Result<()> {
    if shape.len() != 3 || shape[0] != layout.batch as i64 || shape[2] != layout.hidden_size as i64
    {
        return Err(Error::Model(format!(
            "Unexpected projected cache output shape for layer {layer}: {shape:?}"
        )));
    }
    let current_frames = shape[1] as usize;
    let expected_values = layout.batch * current_frames * layout.hidden_size;
    if current_data.len() != expected_values {
        return Err(Error::Model(format!(
            "Unexpected projected cache output size for layer {layer}: got {}, expected {}",
            current_data.len(),
            expected_values
        )));
    }

    let all = cache
        .as_slice_mut()
        .ok_or_else(|| Error::Model("projected cache is not contiguous".to_string()))?;
    let layer_values = layout.projected_cache_layer_values();
    let layer_start = layer * layer_values;
    let layer_end = layer_start + layer_values;
    let layer_slice = all
        .get_mut(layer_start..layer_end)
        .ok_or_else(|| Error::Model(format!("projected cache layer {layer} is out of bounds")))?;

    if current_frames >= layout.cache_len {
        let src_start = (current_frames - layout.cache_len) * layout.hidden_size;
        layer_slice.copy_from_slice(&current_data[src_start..]);
        return Ok(());
    }

    let appended_values = current_frames * layout.hidden_size;
    let kept_values = layer_values - appended_values;
    layer_slice.copy_within(appended_values..layer_values, 0);
    layer_slice[kept_values..layer_values].copy_from_slice(current_data);
    Ok(())
}

struct LiteRtEncoderBackend {
    handle: *mut c_void,
    api: LiteRtEncoderApi,
    layout: EncoderLayout,
    cache_abi: EncoderCacheAbi,
}

impl LiteRtEncoderBackend {
    fn create(model_path: &Path, num_threads: usize) -> Result<Option<Self>> {
        if !model_path.exists() {
            return Ok(None);
        }
        let Some(api) = get_litert_encoder_api() else {
            android_log::info(format!(
                "crate litert encoder unavailable path={} reason=no_api",
                model_path.display()
            ));
            return Ok(None);
        };
        let create = api
            .create
            .ok_or_else(|| Error::Config("LiteRT encoder create callback missing".to_string()))?;
        let path = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|_| Error::Config("LiteRT encoder path contains NUL".to_string()))?;
        let mut error_buffer = vec![0 as c_char; 1024];
        let handle = unsafe {
            create(
                path.as_ptr(),
                num_threads as i32,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if handle.is_null() {
            let message = read_c_error(&error_buffer);
            return Err(Error::Model(format!(
                "Failed to create LiteRT encoder {}: {}",
                model_path.display(),
                if message.is_empty() {
                    "unknown error"
                } else {
                    &message
                }
            )));
        }
        let raw_layout = match Self::read_layout(api, handle) {
            Ok(layout) => layout,
            Err(err) => {
                if let Some(destroy) = api.destroy {
                    unsafe { destroy(handle) };
                }
                return Err(err);
            }
        };
        let cache_abi = match litert_cache_abi_from_i64(raw_layout.cache_abi) {
            Ok(cache_abi) => cache_abi,
            Err(err) => {
                if let Some(destroy) = api.destroy {
                    unsafe { destroy(handle) };
                }
                return Err(err);
            }
        };
        let layout = match EncoderLayout::try_from(raw_layout) {
            Ok(layout) => layout,
            Err(err) => {
                if let Some(destroy) = api.destroy {
                    unsafe { destroy(handle) };
                }
                return Err(err);
            }
        };
        android_log::info(format!(
            "crate litert encoder ready path={} handle={:?} abi={:?} layout={:?}",
            model_path.display(),
            handle,
            cache_abi,
            layout
        ));
        Ok(Some(Self {
            handle,
            api,
            layout,
            cache_abi,
        }))
    }

    fn read_layout(api: LiteRtEncoderApi, handle: *mut c_void) -> Result<LiteRtEncoderLayout> {
        let get_encoder_layout = api
            .get_encoder_layout
            .ok_or_else(|| Error::Config("LiteRT encoder layout callback missing".to_string()))?;
        let mut layout = LiteRtEncoderLayout {
            batch: 0,
            n_mels: 0,
            input_frames: 0,
            output_frames: 0,
            num_layers: 0,
            cache_len: 0,
            hidden_size: 0,
            time_cache_len: 0,
            cache_abi: -1,
        };
        let mut error_buffer = vec![0 as c_char; 1024];
        let ok = unsafe {
            get_encoder_layout(
                handle,
                &mut layout,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if !ok {
            let message = read_c_error(&error_buffer);
            return Err(Error::Model(format!(
                "LiteRT encoder layout failed: {}",
                if message.is_empty() {
                    "unknown error"
                } else {
                    &message
                }
            )));
        }
        Ok(layout)
    }

    fn encoder_layout(&self) -> EncoderLayout {
        self.layout
    }

    fn encoder_cache_abi(&self) -> EncoderCacheAbi {
        self.cache_abi
    }

    fn run_encoder_into(
        &mut self,
        features: &Array3<f32>,
        cache: &mut EncoderCache,
        encoder_out: &mut Array3<f32>,
        projected_current_key: &mut Array4<f32>,
        projected_current_value: &mut Array4<f32>,
    ) -> Result<usize> {
        let (features_ptr, features_len) = features
            .as_slice()
            .map(|values| (values.as_ptr(), values.len()))
            .ok_or_else(|| Error::Model("features is not contiguous".to_string()))?;
        let (cache_last_channel_ptr, cache_last_channel_len) = cache
            .cache_last_channel
            .as_slice()
            .map(|values| (values.as_ptr(), values.len()))
            .ok_or_else(|| Error::Model("cache_last_channel is not contiguous".to_string()))?;
        let (cache_last_time_ptr, cache_last_time_len) = cache
            .cache_last_time
            .as_slice()
            .map(|values| (values.as_ptr(), values.len()))
            .ok_or_else(|| Error::Model("cache_last_time is not contiguous".to_string()))?;
        let (cache_len_ptr, cache_len_count) = cache
            .cache_last_channel_len
            .as_slice()
            .map(|values| (values.as_ptr(), values.len()))
            .ok_or_else(|| Error::Model("cache_last_channel_len is not contiguous".to_string()))?;
        let encoder_out = encoder_out
            .as_slice_mut()
            .ok_or_else(|| Error::Model("encoder_out is not contiguous".to_string()))?;
        let new_cache_last_channel = cache
            .cache_last_channel
            .as_slice_mut()
            .ok_or_else(|| Error::Model("cache_last_channel is not contiguous".to_string()))?;
        let new_cache_last_time = cache
            .cache_last_time
            .as_slice_mut()
            .ok_or_else(|| Error::Model("cache_last_time is not contiguous".to_string()))?;
        let new_cache_last_channel_len = cache
            .cache_last_channel_len
            .as_slice_mut()
            .ok_or_else(|| Error::Model("cache_last_channel_len is not contiguous".to_string()))?;
        let mut encoded_lengths = vec![0i64; new_cache_last_channel_len.len()];
        let mut error_buffer = vec![0 as c_char; 1024];

        let ok = match self.cache_abi {
            EncoderCacheAbi::RawChannel => {
                let run = self.api.run.ok_or_else(|| {
                    Error::Config("LiteRT encoder run callback missing".to_string())
                })?;
                unsafe {
                    run(
                        self.handle,
                        features_ptr,
                        features_len,
                        cache_last_channel_ptr,
                        cache_last_channel_len,
                        cache_last_time_ptr,
                        cache_last_time_len,
                        cache_len_ptr,
                        cache_len_count,
                        encoder_out.as_mut_ptr(),
                        encoder_out.len(),
                        encoded_lengths.as_mut_ptr(),
                        encoded_lengths.len(),
                        new_cache_last_channel.as_mut_ptr(),
                        new_cache_last_channel.len(),
                        new_cache_last_time.as_mut_ptr(),
                        new_cache_last_time.len(),
                        new_cache_last_channel_len.as_mut_ptr(),
                        new_cache_last_channel_len.len(),
                        error_buffer.as_mut_ptr(),
                        error_buffer.len(),
                    )
                }
            }
            EncoderCacheAbi::ProjectedKvLayered => {
                let run_projected = self.api.run_projected_layered.ok_or_else(|| {
                    Error::Config("LiteRT projected encoder run callback missing".to_string())
                })?;
                let (cache_last_key_ptr, cache_last_key_len) = cache
                    .cache_last_key
                    .as_slice()
                    .map(|values| (values.as_ptr(), values.len()))
                    .ok_or_else(|| Error::Model("cache_last_key is not contiguous".to_string()))?;
                let (cache_last_value_ptr, cache_last_value_len) = cache
                    .cache_last_value
                    .as_slice()
                    .map(|values| (values.as_ptr(), values.len()))
                    .ok_or_else(|| {
                        Error::Model("cache_last_value is not contiguous".to_string())
                    })?;
                let (projected_key_ptr, projected_key_len) = projected_current_key
                    .as_slice_mut()
                    .map(|values| (values.as_mut_ptr(), values.len()))
                    .ok_or_else(|| {
                        Error::Model("projected_current_key is not contiguous".to_string())
                    })?;
                let (projected_value_ptr, projected_value_len) = projected_current_value
                    .as_slice_mut()
                    .map(|values| (values.as_mut_ptr(), values.len()))
                    .ok_or_else(|| {
                        Error::Model("projected_current_value is not contiguous".to_string())
                    })?;
                unsafe {
                    run_projected(
                        self.handle,
                        features_ptr,
                        features_len,
                        cache_last_channel_ptr,
                        cache_last_channel_len,
                        cache_last_time_ptr,
                        cache_last_time_len,
                        cache_len_ptr,
                        cache_len_count,
                        cache_last_key_ptr,
                        cache_last_key_len,
                        cache_last_value_ptr,
                        cache_last_value_len,
                        encoder_out.as_mut_ptr(),
                        encoder_out.len(),
                        encoded_lengths.as_mut_ptr(),
                        encoded_lengths.len(),
                        new_cache_last_channel.as_mut_ptr(),
                        new_cache_last_channel.len(),
                        new_cache_last_time.as_mut_ptr(),
                        new_cache_last_time.len(),
                        new_cache_last_channel_len.as_mut_ptr(),
                        new_cache_last_channel_len.len(),
                        projected_key_ptr,
                        projected_key_len,
                        projected_value_ptr,
                        projected_value_len,
                        error_buffer.as_mut_ptr(),
                        error_buffer.len(),
                    )
                }
            }
            EncoderCacheAbi::ProjectedKvStacked => {
                return Err(Error::Config(
                    "LiteRT stacked projected-cache ABI is not supported".to_string(),
                ));
            }
        };
        if !ok {
            let message = read_c_error(&error_buffer);
            return Err(Error::Model(format!(
                "LiteRT encoder run failed: {}",
                if message.is_empty() {
                    "unknown error"
                } else {
                    &message
                }
            )));
        }

        if self.cache_abi == EncoderCacheAbi::ProjectedKvLayered {
            let shape = [
                self.layout.batch as i64,
                self.layout.output_frames as i64,
                self.layout.hidden_size as i64,
            ];
            for layer in 0..self.layout.num_layers {
                let key = projected_current_layer_slice(projected_current_key, layer, self.layout)?;
                roll_projected_cache_layer(
                    &mut cache.cache_last_key,
                    layer,
                    self.layout,
                    &shape,
                    key,
                )?;
                let value =
                    projected_current_layer_slice(projected_current_value, layer, self.layout)?;
                roll_projected_cache_layer(
                    &mut cache.cache_last_value,
                    layer,
                    self.layout,
                    &shape,
                    value,
                )?;
            }
        }

        Ok(encoded_lengths
            .first()
            .copied()
            .unwrap_or(encoder_out.len() as i64) as usize)
    }
}

impl Drop for LiteRtEncoderBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            if let Some(destroy) = self.api.destroy {
                unsafe { destroy(self.handle) };
            }
            self.handle = ptr::null_mut();
        }
    }
}

struct LiteRtDecoderBackend {
    handle: *mut c_void,
    api: LiteRtEncoderApi,
}

impl LiteRtDecoderBackend {
    fn create(model_path: &Path, num_threads: usize) -> Result<Option<Self>> {
        if !model_path.exists() {
            return Ok(None);
        }
        let Some(api) = get_litert_encoder_api() else {
            android_log::info(format!(
                "crate litert decoder unavailable path={} reason=no_api",
                model_path.display()
            ));
            return Ok(None);
        };
        let create = api
            .create_decoder
            .ok_or_else(|| Error::Config("LiteRT decoder create callback missing".to_string()))?;
        let path = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|_| Error::Config("LiteRT decoder path contains NUL".to_string()))?;
        let mut error_buffer = vec![0 as c_char; 1024];
        let handle = unsafe {
            create(
                path.as_ptr(),
                num_threads as i32,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if handle.is_null() {
            let message = read_c_error(&error_buffer);
            return Err(Error::Model(format!(
                "Failed to create LiteRT decoder {}: {}",
                model_path.display(),
                if message.is_empty() {
                    "unknown error"
                } else {
                    &message
                }
            )));
        }
        android_log::info(format!(
            "crate litert decoder ready path={} handle={:?}",
            model_path.display(),
            handle
        ));
        Ok(Some(Self { handle, api }))
    }

    fn run_decoder_into(
        &mut self,
        encoder_frame: &Array3<f32>,
        last_token: &Array2<i32>,
        state_h: &Array3<f32>,
        state_c: &Array3<f32>,
        logits: &mut Array3<f32>,
        new_h: &mut Array3<f32>,
        new_c: &mut Array3<f32>,
    ) -> Result<()> {
        let run = self
            .api
            .run_decoder
            .ok_or_else(|| Error::Config("LiteRT decoder run callback missing".to_string()))?;
        let encoder_frame = encoder_frame
            .as_slice()
            .ok_or_else(|| Error::Model("decoder encoder_frame is not contiguous".to_string()))?;
        let last_token = last_token
            .as_slice()
            .ok_or_else(|| Error::Model("decoder last_token is not contiguous".to_string()))?;
        let state_h = state_h
            .as_slice()
            .ok_or_else(|| Error::Model("decoder state_h is not contiguous".to_string()))?;
        let state_c = state_c
            .as_slice()
            .ok_or_else(|| Error::Model("decoder state_c is not contiguous".to_string()))?;
        let logits = logits
            .as_slice_mut()
            .ok_or_else(|| Error::Model("decoder logits is not contiguous".to_string()))?;
        let new_h = new_h
            .as_slice_mut()
            .ok_or_else(|| Error::Model("decoder new_h is not contiguous".to_string()))?;
        let new_c = new_c
            .as_slice_mut()
            .ok_or_else(|| Error::Model("decoder new_c is not contiguous".to_string()))?;
        let mut error_buffer = vec![0 as c_char; 1024];
        let ok = unsafe {
            run(
                self.handle,
                encoder_frame.as_ptr(),
                encoder_frame.len(),
                last_token.as_ptr(),
                last_token.len(),
                state_h.as_ptr(),
                state_h.len(),
                state_c.as_ptr(),
                state_c.len(),
                logits.as_mut_ptr(),
                logits.len(),
                new_h.as_mut_ptr(),
                new_h.len(),
                new_c.as_mut_ptr(),
                new_c.len(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if !ok {
            let message = read_c_error(&error_buffer);
            return Err(Error::Model(format!(
                "LiteRT decoder run failed: {}",
                if message.is_empty() {
                    "unknown error"
                } else {
                    &message
                }
            )));
        }
        Ok(())
    }
}

impl Drop for LiteRtDecoderBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            if let Some(destroy) = self.api.destroy_decoder {
                unsafe { destroy(self.handle) };
            }
            self.handle = ptr::null_mut();
        }
    }
}

pub struct ParakeetEOUModel {
    encoder: Option<Session>,
    encoder_binding: Option<IoBinding>,
    litert_encoder: Option<LiteRtEncoderBackend>,
    decoder_joint: Option<Session>,
    decoder_binding: Option<IoBinding>,
    litert_decoder: Option<LiteRtDecoderBackend>,
    encoder_cache_abi: EncoderCacheAbi,
    encoder_layout: EncoderLayout,
    encoder_accepts_length: bool,
    encoder_has_raw_channel_cache: bool,
    encoder_outputs_raw_channel_cache: bool,
    encoder_length: Array1<i64>,
    litert_projected_current_key: Array4<f32>,
    litert_projected_current_value: Array4<f32>,
    decoder_target_length: Array1<i32>,
}

impl ParakeetEOUModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        decoder_vocab_size: usize,
    ) -> Result<Self> {
        let started_at = Instant::now();
        let model_dir = model_dir.as_ref();
        let litert_encoder_path = model_dir.join("encoder.tflite");
        let litert_encoder_enabled = model_dir.join("encoder.tflite.enabled");
        let litert_decoder_path = model_dir.join("decoder_joint.tflite");
        let full_litert = litert_encoder_path.exists()
            && litert_encoder_enabled.exists()
            && litert_decoder_path.exists();

        if full_litert {
            android_log::info(format!(
                "crate backend selection full_litert encoder={} marker={} decoder={}",
                litert_encoder_path.display(),
                litert_encoder_enabled.display(),
                litert_decoder_path.display()
            ));
            let litert_encoder =
                LiteRtEncoderBackend::create(&litert_encoder_path, exec_config.intra_threads)?
                    .ok_or_else(|| {
                        Error::Config(
                            "Full LiteRT model selected, but LiteRT encoder API is unavailable"
                                .to_string(),
                        )
                    })?;
            let encoder_layout = litert_encoder.encoder_layout();
            let encoder_cache_abi = litert_encoder.encoder_cache_abi();
            android_log::info(format!(
                "crate litert encoder layout={:?} abi={:?}",
                encoder_layout, encoder_cache_abi
            ));
            let litert_decoder =
                LiteRtDecoderBackend::create(&litert_decoder_path, exec_config.intra_threads)?
                    .ok_or_else(|| {
                        Error::Config(
                            "Full LiteRT model selected, but LiteRT decoder API is unavailable"
                                .to_string(),
                        )
                    })?;
            android_log::info(format!(
                "crate full_litert ready elapsedMs={}",
                started_at.elapsed().as_millis()
            ));
            return Ok(Self {
                encoder: None,
                encoder_binding: None,
                litert_encoder: Some(litert_encoder),
                decoder_joint: None,
                decoder_binding: None,
                litert_decoder: Some(litert_decoder),
                encoder_cache_abi,
                encoder_layout,
                encoder_accepts_length: false,
                encoder_has_raw_channel_cache: true,
                encoder_outputs_raw_channel_cache: true,
                encoder_length: Array1::zeros(encoder_layout.batch),
                litert_projected_current_key: Array4::zeros((
                    encoder_layout.num_layers,
                    encoder_layout.batch,
                    encoder_layout.output_frames,
                    encoder_layout.hidden_size,
                )),
                litert_projected_current_value: Array4::zeros((
                    encoder_layout.num_layers,
                    encoder_layout.batch,
                    encoder_layout.output_frames,
                    encoder_layout.hidden_size,
                )),
                decoder_target_length: Array1::from_vec(vec![1i32]),
            });
        }

        let encoder_path = {
            let projected_kv_layered_fixed =
                model_dir.join("encoder.projected_kv_cache.layered.fixed.onnx");
            let projected_kv_layered = model_dir.join("encoder.projected_kv_cache.layered.onnx");
            let projected_kv_int8 = model_dir.join("encoder.projected_kv_cache.onnx");
            let fixed_raw_cache_int8 =
                model_dir.join("encoder.fixed_raw_cache.dynamic_int8.matmul_gemm.onnx");
            let fixed_raw_cache = model_dir.join("encoder.fixed_raw_cache.onnx");
            let posconst_int8 =
                model_dir.join("encoder.matmul_gemm.dynamic_int8.fullpre_cacheabi.posconst.onnx");
            let conservative_int8 =
                model_dir.join("encoder.matmul_gemm.dynamic_int8.fullpre_cacheabi.onnx");
            if litert_encoder_path.exists() && litert_encoder_enabled.exists() {
                android_log::info(format!(
                    "crate encoder selection prefer raw-cache ONNX because LiteRT encoder is explicitly enabled path={} marker={}",
                    litert_encoder_path.display(),
                    litert_encoder_enabled.display()
                ));
                if fixed_raw_cache_int8.exists() {
                    fixed_raw_cache_int8
                } else if fixed_raw_cache.exists() {
                    fixed_raw_cache
                } else if posconst_int8.exists() {
                    posconst_int8
                } else if conservative_int8.exists() {
                    conservative_int8
                } else {
                    model_dir.join("encoder.onnx")
                }
            } else if litert_encoder_path.exists() {
                android_log::info(format!(
                    "crate encoder selection ignoring LiteRT encoder without opt-in marker path={} marker={}",
                    litert_encoder_path.display(),
                    litert_encoder_enabled.display()
                ));
                if projected_kv_layered_fixed.exists() {
                    projected_kv_layered_fixed
                } else if projected_kv_layered.exists() {
                    projected_kv_layered
                } else if projected_kv_int8.exists() {
                    projected_kv_int8
                } else if posconst_int8.exists() {
                    posconst_int8
                } else if conservative_int8.exists() {
                    conservative_int8
                } else {
                    model_dir.join("encoder.onnx")
                }
            } else if projected_kv_layered_fixed.exists() {
                projected_kv_layered_fixed
            } else if projected_kv_layered.exists() {
                projected_kv_layered
            } else if projected_kv_int8.exists() {
                projected_kv_int8
            } else if posconst_int8.exists() {
                posconst_int8
            } else if conservative_int8.exists() {
                conservative_int8
            } else {
                model_dir.join("encoder.onnx")
            }
        };
        let decoder_path = model_dir.join("decoder_joint.onnx");

        if !encoder_path.exists() || !decoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing ONNX files in {}. Expected encoder.onnx and decoder_joint.onnx",
                model_dir.display()
            )));
        }

        android_log::info(format!(
            "crate session builder begin elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let encoder_profile_path = exec_config.ort_profile_path("encoder");
        let encoder_cache_paths =
            optimized_model_cache_paths(&exec_config, "encoder", &encoder_path)?;
        let build_encoder_session = |load_path: &Path,
                                     use_cached_model: bool,
                                     optimized_output_path: Option<&Path>|
         -> Result<Session> {
            let builder = Session::builder()?;
            android_log::info(format!(
                "crate encoder builder created elapsedMs={}",
                started_at.elapsed().as_millis()
            ));
            let mut builder = if use_cached_model {
                exec_config.apply_to_session_builder_for_cached_model(builder)?
            } else {
                exec_config.apply_to_session_builder(builder)?
            };
            builder = builder.with_log_id("sayboard-parakeet-eou-encoder")?;
            android_log::info(format!(
                "crate encoder builder configured elapsedMs={} intraThreads={} interThreads={} optimizedCacheLoad={}",
                started_at.elapsed().as_millis(),
                exec_config.intra_threads,
                exec_config.inter_threads,
                use_cached_model
            ));
            if let Some(profile_path) = encoder_profile_path.as_ref() {
                android_log::info(format!(
                    "crate encoder profiling enabled path={}",
                    profile_path.display()
                ));
                builder = builder.with_profiling(profile_path)?;
            }
            if let Some(optimized_output_path) = optimized_output_path {
                android_log::info(format!(
                    "crate encoder optimized cache output path={}",
                    optimized_output_path.display()
                ));
                builder = builder.with_optimized_model_path(optimized_output_path)?;
            }
            android_log::info(format!(
                "crate encoder commit begin elapsedMs={} path={}",
                started_at.elapsed().as_millis(),
                load_path.display()
            ));
            let session = builder.commit_from_file(load_path)?;
            android_log::info(format!(
                "crate encoder commit end elapsedMs={}",
                started_at.elapsed().as_millis()
            ));
            Ok(session)
        };

        let encoder = match encoder_cache_paths.as_ref() {
            Some(paths) if paths.final_path.exists() => {
                android_log::info(format!(
                    "crate encoder optimized cache hit path={}",
                    paths.final_path.display()
                ));
                match build_encoder_session(&paths.final_path, true, None) {
                    Ok(session) => session,
                    Err(err) => {
                        android_log::error(format!(
                            "crate encoder optimized cache load failed path={} error={}",
                            paths.final_path.display(),
                            err
                        ));
                        if let Err(remove_err) = fs::remove_file(&paths.final_path) {
                            android_log::error(format!(
                                "crate encoder optimized cache remove failed path={} error={}",
                                paths.final_path.display(),
                                remove_err
                            ));
                        }
                        let _ = fs::remove_file(&paths.temp_path);
                        build_encoder_session(&encoder_path, false, Some(&paths.temp_path))?
                    }
                }
            }
            Some(paths) => {
                android_log::info(format!(
                    "crate encoder optimized cache miss source={} path={}",
                    encoder_path.display(),
                    paths.final_path.display()
                ));
                let _ = fs::remove_file(&paths.temp_path);
                build_encoder_session(&encoder_path, false, Some(&paths.temp_path))?
            }
            None => build_encoder_session(&encoder_path, false, None)?,
        };
        if let Some(paths) = encoder_cache_paths.as_ref() {
            if paths.temp_path.exists() {
                let _ = fs::remove_file(&paths.final_path);
                match fs::rename(&paths.temp_path, &paths.final_path) {
                    Ok(()) => {
                        let size_bytes = fs::metadata(&paths.final_path)
                            .map(|metadata| metadata.len())
                            .unwrap_or(0);
                        android_log::info(format!(
                            "crate encoder optimized cache stored path={} sizeBytes={}",
                            paths.final_path.display(),
                            size_bytes
                        ));
                    }
                    Err(err) => {
                        android_log::error(format!(
                            "crate encoder optimized cache store failed temp={} final={} error={}",
                            paths.temp_path.display(),
                            paths.final_path.display(),
                            err
                        ));
                    }
                }
            }
        }
        let encoder_input_names = encoder
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        let encoder_output_names = encoder
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect::<Vec<_>>();
        let encoder_layout = infer_encoder_layout(&encoder)?;
        let encoder_has_stacked_projected_kv_cache = encoder_input_names
            .iter()
            .any(|name| name == "cache_last_key")
            && encoder_input_names
                .iter()
                .any(|name| name == "cache_last_value");
        let encoder_has_layered_projected_kv_cache =
            has_all_layered_projected_cache_inputs(&encoder_input_names, encoder_layout.num_layers);
        let encoder_has_raw_channel_cache = encoder_input_names
            .iter()
            .any(|name| name == "cache_last_channel");
        let encoder_accepts_length = encoder_input_names.iter().any(|name| name == "length");
        let encoder_outputs_raw_channel_cache = encoder_output_names
            .iter()
            .any(|name| name == "new_cache_last_channel");
        let encoder_outputs_stacked_projected_kv_cache = encoder_output_names
            .iter()
            .any(|name| name == "new_cache_last_key")
            && encoder_output_names
                .iter()
                .any(|name| name == "new_cache_last_value");
        let encoder_outputs_layered_projected_kv_cache = has_all_layered_projected_cache_outputs(
            &encoder_output_names,
            encoder_layout.num_layers,
        );
        let encoder_cache_abi = if encoder_has_layered_projected_kv_cache {
            if !encoder_outputs_layered_projected_kv_cache {
                return Err(Error::Config(
                    "Layered projected K/V encoder inputs are present, but projected current K/V outputs are missing"
                        .to_string(),
                ));
            }
            EncoderCacheAbi::ProjectedKvLayered
        } else if encoder_has_stacked_projected_kv_cache {
            if !encoder_outputs_stacked_projected_kv_cache {
                return Err(Error::Config(
                    "Projected K/V encoder inputs are present, but projected K/V outputs are missing"
                        .to_string(),
                ));
            }
            EncoderCacheAbi::ProjectedKvStacked
        } else {
            EncoderCacheAbi::RawChannel
        };
        android_log::info(format!(
            "crate encoder abi={:?} layout={:?} inputs=[{}] outputs=[{}]",
            encoder_cache_abi,
            encoder_layout,
            encoder_input_names.join(","),
            encoder_output_names.join(",")
        ));
        let mut encoder_binding = encoder.create_binding()?;
        encoder_binding.bind_output(
            "outputs",
            Tensor::<f32>::new(
                encoder.allocator(),
                [
                    encoder_layout.batch,
                    encoder_layout.hidden_size,
                    encoder_layout.output_frames,
                ],
            )?,
        )?;
        if encoder_outputs_raw_channel_cache {
            encoder_binding.bind_output(
                "new_cache_last_channel",
                Tensor::<f32>::new(
                    encoder.allocator(),
                    [
                        encoder_layout.num_layers,
                        encoder_layout.batch,
                        encoder_layout.cache_len,
                        encoder_layout.hidden_size,
                    ],
                )?,
            )?;
        }
        encoder_binding.bind_output(
            "new_cache_last_time",
            Tensor::<f32>::new(
                encoder.allocator(),
                [
                    encoder_layout.num_layers,
                    encoder_layout.batch,
                    encoder_layout.hidden_size,
                    encoder_layout.time_cache_len,
                ],
            )?,
        )?;
        encoder_binding.bind_output(
            "new_cache_last_channel_len",
            Tensor::<i64>::new(encoder.allocator(), [encoder_layout.batch])?,
        )?;
        if encoder_cache_abi == EncoderCacheAbi::ProjectedKvStacked {
            encoder_binding.bind_output(
                "new_cache_last_key",
                Tensor::<f32>::new(
                    encoder.allocator(),
                    [
                        encoder_layout.num_layers,
                        encoder_layout.batch,
                        encoder_layout.cache_len,
                        encoder_layout.hidden_size,
                    ],
                )?,
            )?;
            encoder_binding.bind_output(
                "new_cache_last_value",
                Tensor::<f32>::new(
                    encoder.allocator(),
                    [
                        encoder_layout.num_layers,
                        encoder_layout.batch,
                        encoder_layout.cache_len,
                        encoder_layout.hidden_size,
                    ],
                )?,
            )?;
        } else if encoder_cache_abi == EncoderCacheAbi::ProjectedKvLayered {
            for layer in 0..encoder_layout.num_layers {
                encoder_binding.bind_output(
                    projected_current_output_name("key", layer),
                    Tensor::<f32>::new(
                        encoder.allocator(),
                        [
                            encoder_layout.batch,
                            encoder_layout.output_frames,
                            encoder_layout.hidden_size,
                        ],
                    )?,
                )?;
                encoder_binding.bind_output(
                    projected_current_output_name("value", layer),
                    Tensor::<f32>::new(
                        encoder.allocator(),
                        [
                            encoder_layout.batch,
                            encoder_layout.output_frames,
                            encoder_layout.hidden_size,
                        ],
                    )?,
                )?;
            }
        }
        android_log::info(format!(
            "crate encoder io binding ready elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let litert_encoder = if encoder_cache_abi == EncoderCacheAbi::RawChannel {
            android_log::info(format!(
                "crate litert encoder create using threads={} ortIntraThreads={}",
                exec_config.intra_threads, exec_config.intra_threads
            ));
            LiteRtEncoderBackend::create(&litert_encoder_path, exec_config.intra_threads)?
        } else {
            if litert_encoder_path.exists() {
                android_log::info(format!(
                    "crate litert encoder unavailable path={} reason=incompatible_encoder_abi abi={:?}",
                    litert_encoder_path.display(),
                    encoder_cache_abi
                ));
            }
            None
        };

        // Load decoder
        let builder = Session::builder()?;
        android_log::info(format!(
            "crate decoder builder created elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        builder = builder.with_log_id("sayboard-parakeet-eou-decoder")?;
        android_log::info(format!(
            "crate decoder builder configured elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        if let Some(profile_path) = exec_config.ort_profile_path("decoder") {
            android_log::info(format!(
                "crate decoder profiling enabled path={}",
                profile_path.display()
            ));
            builder = builder.with_profiling(&profile_path)?;
        }
        android_log::info(format!(
            "crate decoder commit begin elapsedMs={} path={}",
            started_at.elapsed().as_millis(),
            decoder_path.display()
        ));
        let decoder_joint = builder.commit_from_file(&decoder_path)?;
        android_log::info(format!(
            "crate decoder commit end elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let mut decoder_binding = decoder_joint.create_binding()?;
        decoder_binding.bind_output(
            "outputs",
            Tensor::<f32>::new(
                decoder_joint.allocator(),
                [1usize, 1, 1, decoder_vocab_size],
            )?,
        )?;
        decoder_binding.bind_output(
            "output_states_1",
            Tensor::<f32>::new(decoder_joint.allocator(), [1usize, 1, 640])?,
        )?;
        decoder_binding.bind_output(
            "output_states_2",
            Tensor::<f32>::new(decoder_joint.allocator(), [1usize, 1, 640])?,
        )?;
        android_log::info(format!(
            "crate decoder io binding ready elapsedMs={} vocabSize={}",
            started_at.elapsed().as_millis(),
            decoder_vocab_size
        ));

        Ok(Self {
            encoder: Some(encoder),
            encoder_binding: Some(encoder_binding),
            litert_encoder,
            decoder_joint: Some(decoder_joint),
            decoder_binding: Some(decoder_binding),
            litert_decoder: None,
            encoder_cache_abi,
            encoder_layout,
            encoder_accepts_length,
            encoder_has_raw_channel_cache,
            encoder_outputs_raw_channel_cache,
            encoder_length: Array1::zeros(encoder_layout.batch),
            litert_projected_current_key: Array4::zeros((
                encoder_layout.num_layers,
                encoder_layout.batch,
                encoder_layout.output_frames,
                encoder_layout.hidden_size,
            )),
            litert_projected_current_value: Array4::zeros((
                encoder_layout.num_layers,
                encoder_layout.batch,
                encoder_layout.output_frames,
                encoder_layout.hidden_size,
            )),
            decoder_target_length: Array1::from_vec(vec![1i32]),
        })
    }

    pub fn encoder_layout(&self) -> EncoderLayout {
        self.encoder_layout
    }

    pub fn end_profiling(&mut self) -> Result<Vec<String>> {
        let mut paths = Vec::new();
        if let Some(encoder) = self.encoder.as_mut() {
            match encoder.end_profiling() {
                Ok(path) if !path.is_empty() => paths.push(path),
                Ok(_) => {}
                Err(err) => {
                    android_log::error(format!("crate encoder endProfiling failed error={err}"));
                }
            }
        }
        if let Some(decoder_joint) = self.decoder_joint.as_mut() {
            match decoder_joint.end_profiling() {
                Ok(path) if !path.is_empty() => paths.push(path),
                Ok(_) => {}
                Err(err) => {
                    android_log::error(format!("crate decoder endProfiling failed error={err}"));
                }
            }
        }
        Ok(paths)
    }

    /// Run the stateful encoder with cache
    /// Input: features [1, 128, T], cache state
    /// Output: (encoded [1, 512, T], new_cache)
    pub fn run_encoder_into(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &mut EncoderCache,
        encoder_out: &mut Array3<f32>,
    ) -> Result<usize> {
        if let Some(litert_encoder) = self.litert_encoder.as_mut() {
            return litert_encoder.run_encoder_into(
                features,
                cache,
                encoder_out,
                &mut self.litert_projected_current_key,
                &mut self.litert_projected_current_value,
            );
        }

        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| Error::Config("ONNX encoder session is not available".to_string()))?;
        let encoder_binding = self
            .encoder_binding
            .as_mut()
            .ok_or_else(|| Error::Config("ONNX encoder binding is not available".to_string()))?;

        self.encoder_length[0] = length;
        let audio_signal_value = ort::value::TensorRef::<f32>::from_array_view(features.view())?;
        let cache_last_channel_value =
            ort::value::TensorRef::<f32>::from_array_view(cache.cache_last_channel.view())?;
        let cache_last_time_value =
            ort::value::TensorRef::<f32>::from_array_view(cache.cache_last_time.view())?;
        let cache_last_channel_len_value =
            ort::value::TensorRef::<i64>::from_array_view(cache.cache_last_channel_len.view())?;

        let outputs = {
            let mut projected_cache_input_values = Vec::new();

            encoder_binding.bind_input("audio_signal", &audio_signal_value)?;
            let length_value = if self.encoder_accepts_length {
                Some(ort::value::TensorRef::<i64>::from_array_view(
                    self.encoder_length.view(),
                )?)
            } else {
                None
            };
            if let Some(length_value) = length_value.as_ref() {
                encoder_binding.bind_input("length", length_value)?;
            }
            if self.encoder_has_raw_channel_cache {
                encoder_binding.bind_input("cache_last_channel", &cache_last_channel_value)?;
            }
            match self.encoder_cache_abi {
                EncoderCacheAbi::ProjectedKvStacked => {
                    let key = TensorRef::<f32>::from_array_view(cache.cache_last_key.view())?;
                    encoder_binding.bind_input("cache_last_key", &key)?;
                    projected_cache_input_values.push(key);

                    let value = TensorRef::<f32>::from_array_view(cache.cache_last_value.view())?;
                    encoder_binding.bind_input("cache_last_value", &value)?;
                    projected_cache_input_values.push(value);
                }
                EncoderCacheAbi::ProjectedKvLayered => {
                    projected_cache_input_values.reserve(self.encoder_layout.num_layers * 2);
                    for layer in 0..self.encoder_layout.num_layers {
                        let key = TensorRef::<f32>::from_array_view((
                            [
                                self.encoder_layout.batch,
                                self.encoder_layout.cache_len,
                                self.encoder_layout.hidden_size,
                            ],
                            projected_cache_layer_slice(
                                &cache.cache_last_key,
                                layer,
                                self.encoder_layout,
                            )?,
                        ))?;
                        encoder_binding.bind_input(cache_layer_input_name("key", layer), &key)?;
                        projected_cache_input_values.push(key);

                        let value = TensorRef::<f32>::from_array_view((
                            [
                                self.encoder_layout.batch,
                                self.encoder_layout.cache_len,
                                self.encoder_layout.hidden_size,
                            ],
                            projected_cache_layer_slice(
                                &cache.cache_last_value,
                                layer,
                                self.encoder_layout,
                            )?,
                        ))?;
                        encoder_binding
                            .bind_input(cache_layer_input_name("value", layer), &value)?;
                        projected_cache_input_values.push(value);
                    }
                }
                EncoderCacheAbi::RawChannel => {}
            }
            encoder_binding.bind_input("cache_last_time", &cache_last_time_value)?;
            encoder_binding.bind_input("cache_last_channel_len", &cache_last_channel_len_value)?;

            encoder.run_binding(encoder_binding)?
        };

        // Extract encoder output [1, 512, T]
        let (shape, data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        let b = shape_dims[0] as usize;
        let d = shape_dims[1] as usize;
        let t = shape_dims[2] as usize;

        if encoder_out.shape() != [b, d, t] {
            return Err(Error::Model(format!(
                "Unexpected encoder output shape: {:?}, reusable buffer shape: {:?}",
                shape_dims,
                encoder_out.shape()
            )));
        }
        encoder_out
            .as_slice_mut()
            .ok_or_else(|| Error::Model("encoder_out is not contiguous".to_string()))?
            .copy_from_slice(data);

        // Extract new cache states
        let (tm_shape, tm_data) = outputs["new_cache_last_time"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_time: {e}")))?;

        let (len_shape, len_data) = outputs["new_cache_last_channel_len"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_len: {e}")))?;

        if cache.cache_last_time.shape()
            != [
                tm_shape[0] as usize,
                tm_shape[1] as usize,
                tm_shape[2] as usize,
                tm_shape[3] as usize,
            ]
        {
            return Err(Error::Model(format!(
                "Unexpected cache_last_time shape: {:?}",
                tm_shape
            )));
        }
        if cache.cache_last_channel_len.len() != len_shape[0] as usize {
            return Err(Error::Model(format!(
                "Unexpected cache_last_channel_len shape: {:?}",
                len_shape
            )));
        }

        if self.encoder_outputs_raw_channel_cache {
            let (ch_shape, ch_data) = outputs["new_cache_last_channel"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract cache_last_channel: {e}")))?;
            if cache.cache_last_channel.shape()
                != [
                    ch_shape[0] as usize,
                    ch_shape[1] as usize,
                    ch_shape[2] as usize,
                    ch_shape[3] as usize,
                ]
            {
                return Err(Error::Model(format!(
                    "Unexpected cache_last_channel shape: {:?}",
                    ch_shape
                )));
            }
            cache
                .cache_last_channel
                .as_slice_mut()
                .ok_or_else(|| Error::Model("cache_last_channel is not contiguous".to_string()))?
                .copy_from_slice(ch_data);
        }
        if self.encoder_cache_abi == EncoderCacheAbi::ProjectedKvStacked {
            let (key_shape, key_data) =
                outputs["new_cache_last_key"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Model(format!("Failed to extract cache_last_key: {e}")))?;
            let (value_shape, value_data) = outputs["new_cache_last_value"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract cache_last_value: {e}")))?;
            if cache.cache_last_key.shape()
                != [
                    key_shape[0] as usize,
                    key_shape[1] as usize,
                    key_shape[2] as usize,
                    key_shape[3] as usize,
                ]
            {
                return Err(Error::Model(format!(
                    "Unexpected cache_last_key shape: {:?}",
                    key_shape
                )));
            }
            if cache.cache_last_value.shape()
                != [
                    value_shape[0] as usize,
                    value_shape[1] as usize,
                    value_shape[2] as usize,
                    value_shape[3] as usize,
                ]
            {
                return Err(Error::Model(format!(
                    "Unexpected cache_last_value shape: {:?}",
                    value_shape
                )));
            }
            cache
                .cache_last_key
                .as_slice_mut()
                .ok_or_else(|| Error::Model("cache_last_key is not contiguous".to_string()))?
                .copy_from_slice(key_data);
            cache
                .cache_last_value
                .as_slice_mut()
                .ok_or_else(|| Error::Model("cache_last_value is not contiguous".to_string()))?
                .copy_from_slice(value_data);
        } else if self.encoder_cache_abi == EncoderCacheAbi::ProjectedKvLayered {
            for layer in 0..self.encoder_layout.num_layers {
                let key_name = projected_current_output_name("key", layer);
                let (key_shape, key_data) = outputs[key_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        Error::Model(format!(
                            "Failed to extract projected current key layer {layer}: {e}"
                        ))
                    })?;
                roll_projected_cache_layer(
                    &mut cache.cache_last_key,
                    layer,
                    self.encoder_layout,
                    key_shape,
                    key_data,
                )?;

                let value_name = projected_current_output_name("value", layer);
                let (value_shape, value_data) = outputs[value_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        Error::Model(format!(
                            "Failed to extract projected current value layer {layer}: {e}"
                        ))
                    })?;
                roll_projected_cache_layer(
                    &mut cache.cache_last_value,
                    layer,
                    self.encoder_layout,
                    value_shape,
                    value_data,
                )?;
            }
        }
        cache
            .cache_last_time
            .as_slice_mut()
            .ok_or_else(|| Error::Model("cache_last_time is not contiguous".to_string()))?
            .copy_from_slice(tm_data);
        cache
            .cache_last_channel_len
            .as_slice_mut()
            .ok_or_else(|| Error::Model("cache_last_channel_len is not contiguous".to_string()))?
            .copy_from_slice(len_data);

        Ok(t)
    }

    /// Run the stateful decoder
    /// Returns: (logits [1, 1, 1, vocab], new_state_h, new_state_c)
    pub fn run_decoder_into(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, 512, 1]
        last_token: &Array2<i32>,    // [1, 1]
        state_h: &Array3<f32>,       // [1, 1, 640]
        state_c: &Array3<f32>,       // [1, 1, 640]
        logits: &mut Array3<f32>,
        new_h: &mut Array3<f32>,
        new_c: &mut Array3<f32>,
    ) -> Result<()> {
        if let Some(litert_decoder) = self.litert_decoder.as_mut() {
            return litert_decoder.run_decoder_into(
                encoder_frame,
                last_token,
                state_h,
                state_c,
                logits,
                new_h,
                new_c,
            );
        }

        let decoder_joint = self
            .decoder_joint
            .as_mut()
            .ok_or_else(|| Error::Config("ONNX decoder session is not available".to_string()))?;
        let decoder_binding = self
            .decoder_binding
            .as_mut()
            .ok_or_else(|| Error::Config("ONNX decoder binding is not available".to_string()))?;

        let encoder_outputs_value =
            ort::value::TensorRef::<f32>::from_array_view(encoder_frame.view())?;
        let targets_value = ort::value::TensorRef::<i32>::from_array_view(last_token.view())?;
        let target_length_value =
            ort::value::TensorRef::<i32>::from_array_view(self.decoder_target_length.view())?;
        let state_h_value = ort::value::TensorRef::<f32>::from_array_view(state_h.view())?;
        let state_c_value = ort::value::TensorRef::<f32>::from_array_view(state_c.view())?;

        decoder_binding.bind_input("encoder_outputs", &encoder_outputs_value)?;
        decoder_binding.bind_input("targets", &targets_value)?;
        decoder_binding.bind_input("target_length", &target_length_value)?;
        decoder_binding.bind_input("input_states_1", &state_h_value)?;
        decoder_binding.bind_input("input_states_2", &state_c_value)?;

        let outputs = decoder_joint.run_binding(decoder_binding)?;

        // 1. Extract Logits
        let (l_shape, l_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        // 2. Extract States (output_states_1, output_states_2)
        let (_h_shape, h_data) = outputs["output_states_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state h: {e}")))?;

        let (_c_shape, c_data) = outputs["output_states_2"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state c: {e}")))?;

        let vocab_size = l_shape[3] as usize;
        if logits.shape() != [1, 1, vocab_size] {
            return Err(Error::Model(format!(
                "Unexpected logits shape: {:?}, reusable buffer shape: {:?}",
                l_shape,
                logits.shape()
            )));
        }
        if new_h.shape() != [1, 1, 640] {
            return Err(Error::Model(format!(
                "Unexpected state h buffer shape: {:?}",
                new_h.shape()
            )));
        }
        if new_c.shape() != [1, 1, 640] {
            return Err(Error::Model(format!(
                "Unexpected state c buffer shape: {:?}",
                new_c.shape()
            )));
        }

        logits
            .as_slice_mut()
            .ok_or_else(|| Error::Model("logits buffer is not contiguous".to_string()))?
            .copy_from_slice(l_data);
        new_h
            .as_slice_mut()
            .ok_or_else(|| Error::Model("state h buffer is not contiguous".to_string()))?
            .copy_from_slice(h_data);
        new_c
            .as_slice_mut()
            .ok_or_else(|| Error::Model("state c buffer is not contiguous".to_string()))?
            .copy_from_slice(c_data);

        Ok(())
    }
}
