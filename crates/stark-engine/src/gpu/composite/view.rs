//! The canvas → NDC mapping every pass that rasterizes canvas geometry reads.
//!
//! One buffer and one sampler per **consumer**: pass A binds them vertex-only, the
//! outline pass binds the same buffer to both stages (it converts a canvas-space
//! distance to screen px with the zoom, §6.8), and the brush-dynamics loop writes its
//! own [`ViewUniform`] into a buffer of its own for the region it composites (§6.2).
//!
//! [`View`] is the shared, never-changing half (the sampler); [`ViewBindings`] the
//! per-target half (what this render is looking at, and the groups over it).

use crate::gpu::uniforms::UniformSlots;
use crate::gpu::{INTERIOR_UV_BIAS, INTERIOR_UV_SCALE};
use crate::view::ViewTransform;
use stark_model::geom::TILE_SIZE;
use stark_shaders::mirror::composite::binding as cb;

// Generated from `composite.wesl`'s declaration of `View`, which `matte.wesl` and
// `overlay.wesl` declare identically — the generator checks all three agree (§6.7).
//
// The one definition of that struct on the host side: every consumer, the
// brush-dynamics loop included, builds one through `view_uniform` rather than
// declaring a second copy.
pub(crate) use stark_shaders::mirror::composite::View as ViewUniform;

/// The canvas px → NDC map `st` (column-major) with translation `xlate`, at `zoom`.
///
/// The three tile constants in `misc` are filled here rather than by the caller: they
/// are facts about the tile layout (§6.4) that no caller should be choosing, and one
/// quoting a different `INTERIOR_UV_BIAS` would sample its neighbours' aprons, the
/// seam showing on that path alone.
///
/// `zoom` reaches only the overlay pass, which measures its outline width in screen
/// px from a canvas-space distance (§6.8). Pass 0 when nothing outlines the frame.
///
/// A free function rather than an inherent `new` because the type is generated into
/// `stark-shaders`, but still the only way one is built.
pub(crate) fn view_uniform(st: [f32; 4], xlate: stark_model::geom::Vec2, zoom: f32) -> ViewUniform {
    ViewUniform {
        st,
        xlate: [xlate.x, xlate.y, 0.0, 0.0],
        misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, zoom],
    }
}

/// The sampler every pass that draws in canvas space reads its tiles through —
/// the whole of what the view contributes to the **shared** pipeline kit.
///
/// The buffer is deliberately not here; see [`ViewBindings`].
pub(super) struct View {
    pub(super) sampler: wgpu::Sampler,
}

impl View {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("stark composite sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }
}

/// The view uniform's buffer and the two group-0 bind groups over it — **one
/// consumer's**, not the pipeline's.
///
/// The buffer holds what this render is looking at, which is per-target by
/// definition: the canvas and the navigator's miniature draw the same document
/// through different transforms, at different sizes, in the same frame. The shared
/// [`CompositorPasses`](super::CompositorPasses) keeps only the layouts and
/// pipelines, which never change.
///
/// **A slot per view, not one uniform**, for the reason `gpu::uniforms` gives:
/// `write_buffer` is a queue operation, so N rewrites before one submit leave every
/// pass reading the last value written. A frame has one view; the eyedropper has up
/// to [`MAX_SAMPLES`](stark_model::gradient::MAX_SAMPLES), a gradient trace sampling
/// a line through the painting (§22.2), and with one uniform each would need a submit
/// of its own.
pub(super) struct ViewBindings {
    slots: UniformSlots<ViewUniform>,
    /// Pass A's group 0 — the uniform vertex-only, plus the tile sampler.
    pub(super) tiles: wgpu::BindGroup,
    /// Pass C's group 0. Its own because the outline's fragment stage reads the
    /// uniform too (it measures its width in screen px from a canvas-space distance,
    /// §6.8) where pass A declares it vertex-only.
    pub(super) overlay: wgpu::BindGroup,
}

/// The three never-changing things a view group is built from — all owned by the
/// shared [`CompositorPasses`](super::CompositorPasses).
///
/// **Handed in at every build rather than kept**, so a [`ViewBindings`] cannot
/// outlive the layouts it answers to: a color-space rebuild replaces the passes, and
/// a consumer holding the old layouts would go on building groups against them with
/// nothing to say so (§6.7).
pub(super) struct ViewGroups<'a> {
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) tiles: &'a wgpu::BindGroupLayout,
    pub(super) overlay: &'a wgpu::BindGroupLayout,
}

impl ViewBindings {
    pub(super) fn new(device: &wgpu::Device, parts: ViewGroups<'_>) -> Self {
        let slots = UniformSlots::new(device, "stark composite view", 1);
        let (tiles, overlay) = groups(device, &slots, parts);
        Self {
            slots,
            tiles,
            overlay,
        }
    }

    /// The dynamic offset that selects view `i` of the last [`write`](Self::write).
    pub(super) fn offset(i: usize) -> u32 {
        UniformSlots::<ViewUniform>::offset(i as u32)
    }

    /// Write one slot per view, in order — every pass of the submit that follows binds
    /// its own by [`offset`](Self::offset).
    ///
    /// Call this once per submit, not once per pass: passes sharing a view share a
    /// slot, and a second write of the same slot before the submit would only
    /// overwrite the first. Growing the buffer **replaces** it, so the groups are
    /// rebuilt when it moves.
    pub(super) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parts: ViewGroups<'_>,
        views: &[ViewTransform],
    ) {
        let uniforms: Vec<ViewUniform> = views
            .iter()
            .map(|view| {
                let (m, translate) = view.canvas_to_ndc();
                // `zoom` rides in `misc.w` for the outline pass, which measures its
                // width in screen px from a canvas-space distance (§6.8).
                view_uniform(m.to_cols_array(), translate, view.zoom)
            })
            .collect();
        if self.slots.write(device, queue, &uniforms) {
            let (tiles, overlay) = groups(device, &self.slots, parts);
            self.tiles = tiles;
            self.overlay = overlay;
        }
    }
}

/// The two group-0 bind groups over `slots` — pass A's and the outline's.
///
/// The two hold the same pair against two layouts, so one closure answers for both —
/// but each names its *own* shader's slot list (§6.10), rather than asserting by hand
/// that `composite.wesl` and `overlay.wesl` number these alike.
fn groups(
    device: &wgpu::Device,
    slots: &UniformSlots<ViewUniform>,
    parts: ViewGroups<'_>,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let group = |label, layout, list| {
        crate::gpu::desc::bind_group_for(device, label, layout, list, false, |i| match i {
            cb::VIEW => slots.resource(),
            cb::SAMP => wgpu::BindingResource::Sampler(parts.sampler),
            other => unreachable!("a view group lists no binding {other}"),
        })
    };
    (
        group(
            "stark composite view bg",
            parts.tiles,
            super::tiles::VIEW_SLOTS,
        ),
        group(
            "stark overlay view bg",
            parts.overlay,
            super::overlay::VIEW_SLOTS,
        ),
    )
}
