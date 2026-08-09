//! GPU context: the wgpu handles the engine renders with (§7).
//!
//! Per CLAUDE.md the backend is *given* its wgpu resources by the frontend.
//! [`GpuContext::from_parts`] is that path; [`GpuContext::headless`] is a
//! convenience for tests and tools that need an offscreen device (§9).

use crate::error::Result;

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
    /// Wrap wgpu handles supplied by the frontend (CLAUDE.md).
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
        // **The stamp loop's `exchange` writes six storage textures where WebGPU
        // guarantees four.**
        //
        // The four it always wrote — the footprint snapshot's colour and aux, and the
        // reservoir's colour and aux, since the segment's `snapshot` rides in the tail
        // of that same dispatch (§6.2) — sit exactly on the downlevel limit, so the
        // residual channel's two (§6.7) put it over. This is the one limit Stark asks
        // for above the guaranteed floor for a *feature* rather than for canvas size,
        // and it is worth saying what that buys and what it costs.
        //
        // It is asked of every device, including one that will only ever open Oklab
        // documents, because limits are settled when the device is created and the
        // colour space is a property of a document opened long after. Every adapter
        // Stark targets — D3D12, Vulkan, Metal, and WebGPU in Chrome — reports at
        // least eight; a conformant device reporting exactly four would fail to start
        // rather than fail to open a Mixbox file, which is the honest failure but not
        // a graceful one.
        //
        // The way back to four, if such a device ever turns up, is packing rather than
        // a second code path, and both halves of it are free: `brush_dst_aux_w` and
        // `under_aux_w` each carry height in `.x` and nothing in `.yzw`, so each one's
        // residual fits beside the height it belongs to. That is the whole excess —
        // no other entry point in the module declares more than three.
        required_limits.max_storage_textures_per_shader_stage =
            required_limits.max_storage_textures_per_shader_stage.max(6);
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
}
