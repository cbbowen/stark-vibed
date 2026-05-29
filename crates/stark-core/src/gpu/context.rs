//! GPU context: the wgpu handles the engine renders with (DESIGN.md §7).
//!
//! Per GOALS §Inputs the backend is *given* its wgpu resources by the frontend.
//! [`GpuContext::from_parts`] is that path; [`GpuContext::headless`] is a
//! convenience for tests and tools that need an offscreen device (DESIGN.md §9).

use crate::error::{EngineError, Result};

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

    /// Create an offscreen context with no surface, for headless rendering.
    pub async fn headless() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stark headless device"),
                required_features: wgpu::Features::empty(),
                // downlevel defaults keep us within web/WebGL2 limits too.
                required_limits: wgpu::Limits::downlevel_defaults(),
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
