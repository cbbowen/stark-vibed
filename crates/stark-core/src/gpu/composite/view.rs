//! The canvas → NDC mapping every pass that rasterizes canvas geometry reads.
//!
//! One buffer and one sampler, shared: pass A binds them vertex-only, the outline
//! pass binds the same buffer to both stages (it converts a canvas-space distance
//! to screen px with the zoom, §6.8), and the brush-dynamics loop writes its own
//! [`ViewUniform`] into a buffer of its own for the region it composites (§6.2).

use bytemuck::{Pod, Zeroable};

use crate::geom::{INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE, ViewTransform};
use crate::gpu::wesl::mirrors_wesl;

/// Mirrors `View` in `composite.wesl` and `overlay.wesl`.
///
/// **The one definition of that struct on this side of the boundary.** The
/// brush-dynamics loop composites its 1:1 canvas region through the very same
/// `composite.wesl`, and used to declare a second, identical `ViewUniform` of its
/// own with a doc comment asking that the two match exactly — which is a job for
/// the compiler, not for a sentence. It builds one of these through
/// [`ViewUniform::new`] instead.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct ViewUniform {
    // The canvas px -> NDC linear map, column-major (`mat2x2` in the shaders). A
    // full 2x2 rather than a scale pair because the view can be turned and mirrored
    // (§18.1.2); upright and unmirrored it is diagonal, and every
    // shader that reads it multiplies the same way either way.
    st: [f32; 4],
    // translate.xy, then padding to the 16-byte stride a uniform member takes
    // anyway.
    xlate: [f32; 4],
    misc: [f32; 4], // tile_size, interior uv scale, interior uv bias, zoom
}
mirrors_wesl!(ViewUniform, 48);

impl ViewUniform {
    /// The canvas px → NDC map `st` (column-major) with translation `xlate`, at
    /// `zoom`.
    ///
    /// The three tile constants in `misc` are filled here rather than at the call
    /// sites, because they are facts about the tile layout (§6.4) that no caller
    /// should be choosing: a consumer that quoted a different `INTERIOR_UV_BIAS`
    /// would sample its neighbours' aprons and the seam would show only on that
    /// one path.
    ///
    /// `zoom` reaches only the overlay pass, which measures its outline width in
    /// screen px from a canvas-space distance (§6.8). Anything drawing into a frame
    /// with no outline over it passes 0.
    pub(crate) fn new(st: [f32; 4], xlate: crate::geom::Vec2, zoom: f32) -> Self {
        Self {
            st,
            xlate: [xlate.x, xlate.y, 0.0, 0.0],
            misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, zoom],
        }
    }
}

/// The view uniform's buffer and the sampler beside it, shared by every pass that
/// draws in canvas space.
pub(super) struct View {
    pub(super) buf: wgpu::Buffer,
    pub(super) sampler: wgpu::Sampler,
}

impl View {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stark composite view"),
                size: std::mem::size_of::<ViewUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("stark composite sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// Write `view`'s canvas px → NDC mapping, for every pass in the frame.
    ///
    /// One write rather than one per pass: they all read the same buffer, and a
    /// render is a single submit, so a second write would only overwrite the first
    /// (`write_buffer` is a queue operation — the reason the *blend* uniform needs
    /// a slot per group rather than a rewrite per pass).
    pub(super) fn write(&self, queue: &wgpu::Queue, view: ViewTransform) {
        let (m, translate) = view.canvas_to_ndc();
        // `zoom` rides in `misc.w` for the outline pass, which measures its width in
        // screen px from a canvas-space distance (§6.8).
        queue.write_buffer(
            &self.buf,
            0,
            bytemuck::bytes_of(&ViewUniform::new(m.to_cols_array(), translate, view.zoom)),
        );
    }
}
