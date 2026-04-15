use crate::android_log;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{IoBinding, Session};
use ort::value::{Outlet, Tensor, TensorRef, ValueType};
use std::path::Path;
use std::time::Instant;

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
        self.cache_len * self.hidden_size
    }
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

pub struct ParakeetEOUModel {
    encoder: Session,
    encoder_binding: IoBinding,
    decoder_joint: Session,
    decoder_binding: IoBinding,
    encoder_cache_abi: EncoderCacheAbi,
    encoder_layout: EncoderLayout,
    encoder_accepts_length: bool,
    encoder_has_raw_channel_cache: bool,
    encoder_outputs_raw_channel_cache: bool,
    encoder_length: Array1<i64>,
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

        let encoder_path = {
            let projected_kv_layered_fixed =
                model_dir.join("encoder.projected_kv_cache.layered.fixed.onnx");
            let projected_kv_layered = model_dir.join("encoder.projected_kv_cache.layered.onnx");
            let projected_kv_int8 = model_dir.join("encoder.projected_kv_cache.onnx");
            let posconst_int8 =
                model_dir.join("encoder.matmul_gemm.dynamic_int8.fullpre_cacheabi.posconst.onnx");
            let conservative_int8 =
                model_dir.join("encoder.matmul_gemm.dynamic_int8.fullpre_cacheabi.onnx");
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
        // Load encoder
        let builder = Session::builder()?;
        android_log::info(format!(
            "crate encoder builder created elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        builder = builder.with_log_id("sayboard-parakeet-eou-encoder")?;
        android_log::info(format!(
            "crate encoder builder configured elapsedMs={} intraThreads={} interThreads={}",
            started_at.elapsed().as_millis(),
            exec_config.intra_threads,
            exec_config.inter_threads
        ));
        if let Some(profile_path) = exec_config.ort_profile_path("encoder") {
            android_log::info(format!(
                "crate encoder profiling enabled path={}",
                profile_path.display()
            ));
            builder = builder.with_profiling(&profile_path)?;
        }
        android_log::info(format!(
            "crate encoder commit begin elapsedMs={} path={}",
            started_at.elapsed().as_millis(),
            encoder_path.display()
        ));
        let encoder = builder.commit_from_file(&encoder_path)?;
        android_log::info(format!(
            "crate encoder commit end elapsedMs={}",
            started_at.elapsed().as_millis()
        ));
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
            encoder,
            encoder_binding,
            decoder_joint,
            decoder_binding,
            encoder_cache_abi,
            encoder_layout,
            encoder_accepts_length,
            encoder_has_raw_channel_cache,
            encoder_outputs_raw_channel_cache,
            encoder_length: Array1::zeros(encoder_layout.batch),
            decoder_target_length: Array1::from_vec(vec![1i32]),
        })
    }

    pub fn encoder_layout(&self) -> EncoderLayout {
        self.encoder_layout
    }

    pub fn end_profiling(&mut self) -> Result<Vec<String>> {
        let mut paths = Vec::new();
        match self.encoder.end_profiling() {
            Ok(path) if !path.is_empty() => paths.push(path),
            Ok(_) => {}
            Err(err) => {
                android_log::error(format!("crate encoder endProfiling failed error={err}"));
            }
        }
        match self.decoder_joint.end_profiling() {
            Ok(path) if !path.is_empty() => paths.push(path),
            Ok(_) => {}
            Err(err) => {
                android_log::error(format!("crate decoder endProfiling failed error={err}"));
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

            self.encoder_binding
                .bind_input("audio_signal", &audio_signal_value)?;
            let length_value = if self.encoder_accepts_length {
                Some(ort::value::TensorRef::<i64>::from_array_view(
                    self.encoder_length.view(),
                )?)
            } else {
                None
            };
            if let Some(length_value) = length_value.as_ref() {
                self.encoder_binding.bind_input("length", length_value)?;
            }
            if self.encoder_has_raw_channel_cache {
                self.encoder_binding
                    .bind_input("cache_last_channel", &cache_last_channel_value)?;
            }
            match self.encoder_cache_abi {
                EncoderCacheAbi::ProjectedKvStacked => {
                    let key = TensorRef::<f32>::from_array_view(cache.cache_last_key.view())?;
                    self.encoder_binding.bind_input("cache_last_key", &key)?;
                    projected_cache_input_values.push(key);

                    let value = TensorRef::<f32>::from_array_view(cache.cache_last_value.view())?;
                    self.encoder_binding
                        .bind_input("cache_last_value", &value)?;
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
                        self.encoder_binding
                            .bind_input(cache_layer_input_name("key", layer), &key)?;
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
                        self.encoder_binding
                            .bind_input(cache_layer_input_name("value", layer), &value)?;
                        projected_cache_input_values.push(value);
                    }
                }
                EncoderCacheAbi::RawChannel => {}
            }
            self.encoder_binding
                .bind_input("cache_last_time", &cache_last_time_value)?;
            self.encoder_binding
                .bind_input("cache_last_channel_len", &cache_last_channel_len_value)?;

            self.encoder.run_binding(&self.encoder_binding)?
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
        let encoder_outputs_value =
            ort::value::TensorRef::<f32>::from_array_view(encoder_frame.view())?;
        let targets_value = ort::value::TensorRef::<i32>::from_array_view(last_token.view())?;
        let target_length_value =
            ort::value::TensorRef::<i32>::from_array_view(self.decoder_target_length.view())?;
        let state_h_value = ort::value::TensorRef::<f32>::from_array_view(state_h.view())?;
        let state_c_value = ort::value::TensorRef::<f32>::from_array_view(state_c.view())?;

        self.decoder_binding
            .bind_input("encoder_outputs", &encoder_outputs_value)?;
        self.decoder_binding.bind_input("targets", &targets_value)?;
        self.decoder_binding
            .bind_input("target_length", &target_length_value)?;
        self.decoder_binding
            .bind_input("input_states_1", &state_h_value)?;
        self.decoder_binding
            .bind_input("input_states_2", &state_c_value)?;

        let outputs = self.decoder_joint.run_binding(&self.decoder_binding)?;

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
