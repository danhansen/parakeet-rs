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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphOptimization {
    Disable,
    Level1,
    Level2,
    #[default]
    Level3,
    All,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub graph_optimization: GraphOptimization,
    pub memory_pattern: Option<bool>,
    pub parallel_execution: bool,
    pub cpu_arena: Option<bool>,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    /// Optional directory for ORT-optimized model artifacts. On first load,
    /// ORT can save its optimized graph here; later loads can open that graph
    /// with runtime graph optimization disabled to reduce session startup work.
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
            .field("graph_optimization", &self.graph_optimization)
            .field("memory_pattern", &self.memory_pattern)
            .field("parallel_execution", &self.parallel_execution)
            .field("cpu_arena", &self.cpu_arena)
            .field(
                "ort_optimized_model_cache_dir",
                &self.ort_optimized_model_cache_dir,
            )
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            graph_optimization: GraphOptimization::default(),
            memory_pattern: None,
            parallel_execution: false,
            cpu_arena: None,
            configure: None,
            ort_optimized_model_cache_dir: None,
            coreml_cache_dir: None,
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

    pub fn with_graph_optimization(mut self, level: GraphOptimization) -> Self {
        self.graph_optimization = level;
        self
    }

    pub fn with_memory_pattern(mut self, enabled: Option<bool>) -> Self {
        self.memory_pattern = enabled;
        self
    }

    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }

    pub fn with_cpu_arena(mut self, enabled: Option<bool>) -> Self {
        self.cpu_arena = enabled;
        self
    }

    pub fn with_ort_optimized_model_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.ort_optimized_model_cache_dir = Some(path.into());
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

    pub(crate) fn ort_optimized_model_cache_dir(&self) -> Option<&Path> {
        self.ort_optimized_model_cache_dir.as_deref()
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        self.apply_to_session_builder_with_optimization(builder, self.graph_optimization)
    }

    pub(crate) fn apply_to_session_builder_for_cached_model(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        self.apply_to_session_builder_with_optimization(builder, GraphOptimization::Disable)
    }

    fn apply_to_session_builder_with_optimization(
        &self,
        builder: SessionBuilder,
        graph_optimization: GraphOptimization,
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
        use ort::session::builder::GraphOptimizationLevel;

        let graph_optimization = match graph_optimization {
            GraphOptimization::Disable => GraphOptimizationLevel::Disable,
            GraphOptimization::Level1 => GraphOptimizationLevel::Level1,
            GraphOptimization::Level2 => GraphOptimizationLevel::Level2,
            GraphOptimization::Level3 => GraphOptimizationLevel::Level3,
            GraphOptimization::All => GraphOptimizationLevel::All,
        };
        let mut builder = builder
            .with_optimization_level(graph_optimization)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        if let Some(memory_pattern) = self.memory_pattern {
            builder = builder.with_memory_pattern(memory_pattern)?;
        }
        if self.parallel_execution {
            builder = builder.with_parallel_execution(true)?;
        }

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => {
                if let Some(cpu_arena) = self.cpu_arena {
                    builder.with_execution_providers([
                        ort::ep::CPU::default()
                            .with_arena_allocator(cpu_arena)
                            .build(),
                    ])?
                } else {
                    builder
                }
            }

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
