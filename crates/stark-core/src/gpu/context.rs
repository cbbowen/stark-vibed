//! GPU context: the wgpu handles the engine renders with (DESIGN.md §7).
//!
//! Per GOALS §Inputs the backend is *given* its wgpu resources by the frontend.
//! [`GpuContext::from_parts`] is that path; [`GpuContext::headless`] is a
//! convenience for tests and tools that need an offscreen device (DESIGN.md §9).

use crate::error::{EngineError, Result};

/// Max surface texture edge. `Limits::downlevel_defaults()` (and WebGL2) cap 2D
/// textures at 2048, so larger sources are box-downsampled by an integer factor
/// (which preserves tileability) — applied on every target so deposition stays
/// deterministic across native and web.
pub(crate) const MAX_TEXTURE_DIM_2D: u32 = 2048;

/// The wgpu device, queue, and adapter the engine draws with.
///
/// `wgpu::Device` and `wgpu::Queue` are cheaply clonable (reference-counted),
/// so this struct is too.
#[derive(Clone)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Wrap wgpu handles supplied by the frontend (GOALS §Inputs).
    pub fn from_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    pub fn minimum_required_limits() -> wgpu::Limits {
        // downlevel defaults keep us within web/WebGL2 limits too.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_texture_dimension_2d = required_limits
            .max_texture_dimension_2d
            .max(MAX_TEXTURE_DIM_2D);
        required_limits
    }

    /// Create an offscreen context with no surface, for headless rendering.
    pub async fn headless() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stark headless device"),
                required_features: wgpu::Features::empty(),
                required_limits: Self::minimum_required_limits(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        Ok(Self::from_parts(instance, adapter, device, queue))
    }

    /// Map a [`NoAdapter`](EngineError::NoAdapter)-style absence into an error.
    /// (Reserved for future surface-bound construction.)
    #[allow(dead_code)]
    pub(crate) fn require<T>(value: Option<T>) -> Result<T> {
        value.ok_or(EngineError::NoAdapter)
    }
}
