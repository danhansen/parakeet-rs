use std::path::{Path, PathBuf};
use std::{fmt, rc::Rc};

use crate::error::Result;
use ort::session::builder::SessionBuilder;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, MIGraphX) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML EP currently runs slower than CPU for Sortformer/Parakeet models because
// the ONNX graphs have dynamic input shapes, preventing CoreML from building optimised
// execution plans for ANE/GPU. CoreML claims nodes but runs them on CPU with overhead.
//
// WebGPU is experimental and may produce incorrect results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    pub ort_profile_dir: Option<PathBuf>,
    pub ort_profile_session: Option<String>,
    /// Optional directory for ONNX Runtime optimized model artifacts.
    /// When set, callers can save ORT's offline-optimized graph and reuse it
    /// on subsequent loads to reduce session startup latency.
    pub ort_optimized_model_cache_dir: Option<PathBuf>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    pub coreml_cache_dir: Option<PathBuf>,
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .field("ort_profile_dir", &self.ort_profile_dir)
            .field("ort_profile_session", &self.ort_profile_session)
            .field(
                "ort_optimized_model_cache_dir",
                &self.ort_optimized_model_cache_dir,
            )
            .finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            configure: None,
            coreml_cache_dir: None,
            ort_profile_dir: None,
            ort_profile_session: None,
            ort_optimized_model_cache_dir: None,
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + 'static,
    ) -> Self {
        self.configure = Some(Rc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    pub fn with_ort_profiling(
        mut self,
        profile_dir: impl Into<PathBuf>,
        session_name: impl Into<String>,
    ) -> Self {
        self.ort_profile_dir = Some(profile_dir.into());
        self.ort_profile_session = Some(session_name.into());
        self
    }

    pub fn with_ort_optimized_model_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.ort_optimized_model_cache_dir = Some(path.into());
        self
    }

    pub(crate) fn ort_profile_path(&self, component: &str) -> Option<PathBuf> {
        let dir = self.ort_profile_dir.as_ref()?;
        let session = self.ort_profile_session.as_ref()?;
        Some(dir.join(format!("{session}-{component}.json")))
    }

    pub(crate) fn ort_optimized_model_cache_dir(&self) -> Option<&Path> {
        self.ort_optimized_model_cache_dir.as_deref()
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        self.apply_to_session_builder_with_optimization(builder, GraphOptimizationPolicy::All)
    }

    pub(crate) fn apply_to_session_builder_for_cached_model(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        self.apply_to_session_builder_with_optimization(builder, GraphOptimizationPolicy::Disable)
    }

    fn apply_to_session_builder_with_optimization(
        &self,
        builder: SessionBuilder,
        optimization: GraphOptimizationPolicy,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::logging::LogLevel;
        use ort::session::builder::GraphOptimizationLevel;

        let optimization_level = match optimization {
            GraphOptimizationPolicy::All => GraphOptimizationLevel::All,
            GraphOptimizationPolicy::Disable => GraphOptimizationLevel::Disable,
        };
        let mut builder = builder
            .with_optimization_level(optimization_level)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;
        if self.ort_profile_dir.is_some() {
            builder = builder
                .with_log_level(LogLevel::Verbose)?
                .with_log_verbosity(1)?;
        }

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML};
                let mut coreml = CoreML::default().with_compute_units(ComputeUnits::CPUAndGPU);

                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum GraphOptimizationPolicy {
    All,
    Disable,
}
