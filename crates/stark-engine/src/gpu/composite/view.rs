//! The canvas → NDC mapping every pass that rasterizes canvas geometry reads.
//!
//! One buffer and one sampler per **consumer**: pass A binds them vertex-only, the
//! outline pass binds the same buffer to both stages (it converts a canvas-space
//! distance to screen px with the zoom, §6.8), and the brush-dynamics loop writes its
//! own [`ViewUniform`] into a buffer of its own for the region it composites (§6.2).
//!
//! The split here is the module's own, one level down: [`View`] is the shared,
//! never-changing half (the sampler) and [`ViewBindings`] the per-target half (the
//! buffer holding *what this render is looking at*, and the groups over it).

use crate::gpu::uniforms::UniformSlots;
use crate::gpu::{INTERIOR_UV_BIAS, INTERIOR_UV_SCALE};
use crate::view::ViewTransform;
use stark_model::geom::TILE_SIZE;
use stark_shaders::mirror::composite::binding as cb;

// Generated from `composite.wesl`'s declaration of `View`, which `matte.wesl` and
// `overlay.wesl` declare identically — the generator checks all three agree (§6.7).
//
// **The one definition of that struct on this side of the boundary.** The
// brush-dynamics loop composites its 1:1 canvas region through the very same
// `composite.wesl`, and builds one of these through [`view_uniform`] rather than
// declaring a second, identical `ViewUniform` of its own — keeping two structs in step
// is a job for the compiler, not for a doc comment asking.
pub(crate) use stark_shaders::mirror::composite::View as ViewUniform;

/// The canvas px → NDC map `st` (column-major) with translation `xlate`, at `zoom`.
///
/// The three tile constants in `misc` are filled here rather than at the call sites,
/// because they are facts about the tile layout (§6.4) that no caller should be
/// choosing: a consumer that quoted a different `INTERIOR_UV_BIAS` would sample its
/// neighbours' aprons and the seam would show only on that one path.
///
/// `zoom` reaches only the overlay pass, which measures its outline width in screen
/// px from a canvas-space distance (§6.8). Anything drawing into a frame with no
/// outline over it passes 0.
///
/// A free function rather than the `ViewUniform::new` it replaced: the type is
/// generated into `stark-shaders` now, and an inherent impl on another crate's type
/// is not allowed. It is still the only way one is built.
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
/// definition: the substrate and the navigator's miniature draw the same document
/// through different transforms, at different sizes, in the same frame. It sat on
/// the shared [`CompositorPasses`](super::CompositorPasses) for as long as the
/// argument "every render writes it immediately before the submit that reads it, and
/// submits on one queue are ordered" held — which is true of every path today and is
/// a property of the *call sequence*, not of the types. Two `Compositor`s are the
/// documented arrangement and a `&CompositorPipeline` is all either needs, so nothing
/// stopped a future caller from straddling a write and its submit with another
/// render. Here the question cannot be asked.
///
/// The passes keep their layouts and their pipelines, which is the expensive half and
/// genuinely never changes; what a consumer owns is a slot buffer and two bind groups
/// over it.
///
/// **A slot per view, not one uniform**, for the reason `gpu::uniforms` gives at
/// length: `write_buffer` is a queue operation, so N rewrites before a single submit
/// leave every pass reading the last value written. A frame has one view and never
/// noticed. The eyedropper has up to
/// [`MAX_SAMPLES`](stark_model::gradient::MAX_SAMPLES) of them — a gradient trace
/// samples a line through the painting (§22.2) — and with one uniform the only thing
/// ordering them was a *submit* between each pair, which is what made a pick cost a
/// hundred round trips to the queue instead of one.
pub(super) struct ViewBindings {
    slots: UniformSlots<ViewUniform>,
    /// The layouts the groups below answer to, kept so growing the buffer can rebuild
    /// them: growth *replaces* the allocation, and a group over the old one names a
    /// buffer too small for the offsets it is about to be given.
    tile_bgl: wgpu::BindGroupLayout,
    overlay_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Pass A's group 0 — the uniform vertex-only, plus the tile sampler.
    pub(super) tiles: wgpu::BindGroup,
    /// Pass C's group 0. Its own because the outline's fragment stage reads the
    /// uniform too (it measures its width in screen px from a canvas-space distance,
    /// §6.8) where pass A declares it vertex-only.
    pub(super) overlay: wgpu::BindGroup,
}

impl ViewBindings {
    pub(super) fn new(
        device: &wgpu::Device,
        view: &View,
        tile_view_bgl: &wgpu::BindGroupLayout,
        overlay_view_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut this = Self {
            slots: UniformSlots::new(device, "stark composite view", 1),
            tile_bgl: tile_view_bgl.clone(),
            overlay_bgl: overlay_view_bgl.clone(),
            sampler: view.sampler.clone(),
            // Replaced by `rebuild` on the next line, which is the one place these are
            // described — a second description here would be the one that drifts.
            tiles: placeholder(device),
            overlay: placeholder(device),
        };
        this.rebuild(device);
        this
    }

    /// The dynamic offset that selects view `i` of the last [`write`](Self::write).
    pub(super) fn offset(i: usize) -> u32 {
        UniformSlots::<ViewUniform>::offset(i as u32)
    }

    /// Build the two group-0 bind groups over the current slot buffer.
    fn rebuild(&mut self, device: &wgpu::Device) {
        // Pass A's view group and the overlay's hold the same two things against two
        // layouts, so one closure answers for both — but each names its *own* shader's
        // declarations (§6.10), because "`composite.wesl` and `overlay.wesl` happen to
        // number these alike" is exactly the kind of agreement this stops asserting by
        // hand.
        let slots = &self.slots;
        let sampler = &self.sampler;
        let group = |label, layout, list| {
            crate::gpu::desc::bind_group_for(device, label, layout, list, false, |i| match i {
                cb::VIEW => slots.resource(),
                cb::SAMP => wgpu::BindingResource::Sampler(sampler),
                other => unreachable!("a view group lists no binding {other}"),
            })
        };
        let tiles = group(
            "stark composite view bg",
            &self.tile_bgl,
            super::tiles::VIEW_SLOTS,
        );
        let overlay = group(
            "stark overlay view bg",
            &self.overlay_bgl,
            super::overlay::VIEW_SLOTS,
        );
        self.tiles = tiles;
        self.overlay = overlay;
    }

    /// Write one slot per view, in order — every pass of the submit that follows binds
    /// its own by [`offset`](Self::offset).
    ///
    /// One call rather than one per pass: passes sharing a view share a slot, and a
    /// second write of the same slot before the submit would only overwrite the first.
    pub(super) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
            self.rebuild(device);
        }
    }
}

/// A bind group standing in until [`ViewBindings::rebuild`] replaces it, which the
/// constructor does before returning.
///
/// An empty group over an empty layout: `BindGroup` has no `Default`, and the
/// alternative — `Option` fields unwrapped at every draw — would put a question at
/// every call site to answer one at construction.
fn placeholder(device: &wgpu::Device) -> wgpu::BindGroup {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark view placeholder bgl"),
        entries: &[],
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark view placeholder bg"),
        layout: &layout,
        entries: &[],
    })
}
