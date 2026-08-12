//! The wgpu descriptors this subsystem writes over and over (§7).
//!
//! Nothing here decides anything. Every function is one shape of descriptor with
//! the fields that never vary already filled in — `depth_stencil: None`,
//! `multisample: Default::default()`, `multiview_mask: None`, `cache: None`,
//! `immediate_size: 0` — so a call site states only what makes it different from
//! its neighbours.
//!
//! That is the whole argument for the module. Before it there were ten separately
//! written closures for "a fragment-visible float texture", three
//! `clear_attachment`s, two byte-identical `zero_texture`s, and seventeen render
//! pipelines each restating the same five defaults. None of that was wrong; it was
//! just impossible to see, at a call site, which of the differences from the one
//! beside it were *meant*. A shared descriptor makes the meant ones the only ones
//! written down.
//!
//! The one judgement encoded here is in the pair [`load_tex`] / [`sample_tex`]: a
//! texture read with `textureLoad` needs no filtering and so no `filterable`
//! requirement on its format, while one read through a sampler does. Choosing the
//! wrong one is a wgpu validation error rather than a wrong pixel, but the names
//! say which is which, where `Float { filterable: false }` at ten call sites did
//! not.

use crate::gpu::context::GpuContext;

// ---- bind group layout entries -------------------------------------------------

/// A texture read with `textureLoad` only.
///
/// No filtering, so the format need not be filterable and no sampler accompanies
/// it. This is also what lets a 1×1 [`constant_texture`] stand in for a missing
/// tile: a clamped load returns the constant for every texel, so the same shader
/// reads a real tile and a stand-in without branching (§6.8).
pub(crate) fn load_tex(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    tex_entry(binding, vis, false, wgpu::TextureViewDimension::D2)
}

/// A texture read through a sampler, and therefore required to be filterable.
pub(crate) fn sample_tex(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    tex_entry(binding, vis, true, wgpu::TextureViewDimension::D2)
}

/// [`load_tex`] over a 2-D **array** view — the prefix-τ volume, whose layers are
/// the brush shape's orientations (§6.6). Read with `textureLoad` so the shader can
/// do its own trilinear lookup across them.
pub(crate) fn load_tex_array(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    tex_entry(binding, vis, false, wgpu::TextureViewDimension::D2Array)
}

fn tex_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    filterable: bool,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// A write-only storage texture — how the stamp loop's compute passes write the
/// region they are evolving (§6.2).
pub(crate) fn storage_tex(
    binding: u32,
    vis: wgpu::ShaderStages,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// A filtering sampler.
pub(crate) fn sampler(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// A whole uniform buffer, bound at offset 0.
pub(crate) fn uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, vis, false, None)
}

/// One **dynamic-offset slot** of a uniform buffer, `slot` bytes wide — how both
/// render paths vary a uniform across the draws or dispatches of a single pass
/// without a buffer per draw (`UNIFORM_STRIDE`).
///
/// `slot` is the struct's own size, and declaring it as `min_binding_size` is free
/// validation against a truncated write: the layouts that pass `None` here get none.
pub(crate) fn uniform_slot(
    binding: u32,
    vis: wgpu::ShaderStages,
    slot: u64,
) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, vis, true, wgpu::BufferSize::new(slot))
}

fn buffer_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    has_dynamic_offset: bool,
    min_binding_size: Option<wgpu::BufferSize>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size,
        },
        count: None,
    }
}

/// A bind group layout of `entries`.
pub(crate) fn bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// A pipeline layout over `bgls`.
pub(crate) fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    bgls: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: bgls,
        immediate_size: 0,
    })
}

// ---- bind group entries --------------------------------------------------------

/// A texture-view binding.
pub(crate) fn tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// A whole-buffer uniform binding — the entry a [`uniform`] layout slot takes.
///
/// The one-liner it replaces was written out at a dozen call sites as a
/// `BindGroupEntry` with an `as_entire_binding()` inside; a dynamic-offset slot needs
/// the explicit [`wgpu::BufferBinding`] form instead and still says so where it is
/// used, which is the difference worth seeing at a call site.
pub(crate) fn uniform_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A sampler binding.
pub(crate) fn samp(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

// ---- render pass attachments ---------------------------------------------------

/// Clear to transparent, then store. What every target fully rewritten by its own
/// pass wants — which here is nearly all of them.
pub(crate) const CLEAR: wgpu::Operations<wgpu::Color> = wgpu::Operations {
    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
    store: wgpu::StoreOp::Store,
};

/// Keep what is there, then store — for a pass drawn *over* a finished image.
pub(crate) const LOAD: wgpu::Operations<wgpu::Color> = wgpu::Operations {
    load: wgpu::LoadOp::Load,
    store: wgpu::StoreOp::Store,
};

/// Clear to `color`, then store. Used where the clear value carries meaning: the
/// coverage that reigns outside a selection's own tiles (§6.8).
pub(crate) fn clear_to(color: wgpu::Color) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: wgpu::LoadOp::Clear(color),
        store: wgpu::StoreOp::Store,
    }
}

/// A colour attachment on `view` with `ops`.
pub(crate) fn attach(
    view: &wgpu::TextureView,
    ops: wgpu::Operations<wgpu::Color>,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops,
    }
}

// ---- pipelines -----------------------------------------------------------------

/// What a render pipeline in this subsystem actually varies. Everything absent from
/// this struct is a default no pass here has ever wanted to change.
pub(crate) struct RenderPipe<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    /// One module for both stages — every shader here declares its vertex and
    /// fragment entry points together.
    pub module: &'a wgpu::ShaderModule,
    pub vs: &'a str,
    pub fs: &'a str,
    pub primitive: wgpu::PrimitiveState,
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

/// A render pipeline, with the five fields no pass here varies filled in.
pub(crate) fn render_pipeline(device: &wgpu::Device, p: RenderPipe<'_>) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(p.label),
        layout: Some(p.layout),
        vertex: wgpu::VertexState {
            module: p.module,
            entry_point: Some(p.vs),
            compilation_options: Default::default(),
            buffers: p.buffers,
        },
        primitive: p.primitive,
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: p.module,
            entry_point: Some(p.fs),
            compilation_options: Default::default(),
            targets: p.targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// [`render_pipeline`] for the **fullscreen triangle** shape: no vertex buffers, no
/// instancing, `draw(0..3, 0..1)`. Ten of the passes here are this — every one that
/// computes a whole target from what it reads rather than rasterizing geometry.
pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entries: (&str, &str),
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    render_pipeline(
        device,
        RenderPipe {
            label,
            layout,
            module,
            vs: entries.0,
            fs: entries.1,
            primitive: wgpu::PrimitiveState::default(),
            buffers: &[],
            targets,
        },
    )
}

/// An instanced quad drawn as a triangle strip — the shape every pass that
/// rasterizes per-tile geometry takes (`draw(0..4, i..i+1)`).
///
/// Spelled out field by field because a `const` cannot call `Default::default()`,
/// which is the one thing here that could change a pixel silently: this replaced
/// six pipelines written as `{ topology: TriangleStrip, ..Default::default() }`
/// and four of the transform's written with an explicit `cull_mode: None`, and a
/// field that disagreed with those would alter what they rasterize without failing
/// anything. `quad_strip_is_the_default_with_a_strip_topology` is what checks it.
///
/// `cull_mode: None` is load-bearing for the transform in particular: a
/// negative-determinant affine (a flip) reverses winding, so both faces must draw
/// (§16).
pub(crate) const QUAD_STRIP: wgpu::PrimitiveState = wgpu::PrimitiveState {
    topology: wgpu::PrimitiveTopology::TriangleStrip,
    strip_index_format: None,
    front_face: wgpu::FrontFace::Ccw,
    cull_mode: None,
    unclipped_depth: false,
    polygon_mode: wgpu::PolygonMode::Fill,
    conservative: false,
};

/// A colour target that replaces what it writes — the pass computes the finished
/// texel, so there is nothing for fixed-function blending to do.
pub(crate) fn target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    blended_target(format, None)
}

/// A colour target with an explicit blend state.
pub(crate) fn blended_target(
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

// ---- constant textures ---------------------------------------------------------

/// A 1×1 texture of `format` holding one texel of `bytes`.
///
/// Bound wherever a consumer has no real texture to give: the tile a layer does not
/// have yet, the coverage outside a selection's own tiles, the lasso edge list an
/// analytic shape never reads (§6.8). Every such consumer clamps its load to the
/// bound texture's own extent, so the constant answers for every texel and the
/// shader needs no branch — which is the reason these exist rather than a pipeline
/// variant per case.
pub(crate) fn constant_texture(
    ctx: &GpuContext,
    format: wgpu::TextureFormat,
    bytes: &[u8],
    label: &str,
) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes.len() as u32),
            rows_per_image: Some(1),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// [`constant_texture`] holding zeros — "no paint here", in whichever format the
/// consumer reads.
pub(crate) fn zero_texture(
    ctx: &GpuContext,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let bytes = format
        .block_copy_size(None)
        .expect("uncompressed tile format") as usize;
    constant_texture(ctx, format, &vec![0u8; bytes], label)
}

/// The colour attachments of a pass that writes a **tile**: colour, aux, and the
/// residual a pigment space adds (§6.7).
///
/// Returned as a fixed array plus how many of it are real, so the caller slices —
/// `&attachments[..n]` — rather than allocating a `Vec` per render pass. These are
/// encoded once per tile per stroke, which is the rate `ScopedResources` exists to
/// keep allocations off (§6.2).
pub(crate) fn tile_attachments<'a>(
    color: &'a wgpu::TextureView,
    aux: &'a wgpu::TextureView,
    resid: Option<&'a wgpu::TextureView>,
    ops: wgpu::Operations<wgpu::Color>,
) -> ([Option<wgpu::RenderPassColorAttachment<'a>>; 3], usize) {
    (
        [
            Some(attach(color, ops)),
            Some(attach(aux, ops)),
            resid.map(|v| attach(v, ops)),
        ],
        2 + usize::from(resid.is_some()),
    )
}

/// The 1×1 stand-ins a pass binds where a tile does not exist — one per persistent
/// tile channel.
///
/// **"There is no tile here" is one question, and this is its one answer.** The fill,
/// the transform's combine and the stroke's integrate all bind these and read them
/// through clamped loads, so the same shader code reads a real tile and a hole
/// (§6.8's pattern). Built once and cloned into each renderer — a `TextureView` is an
/// `Arc` handle, so a clone is a bump.
///
/// They were built twice before this, once inside `FillRenderer` and once inside
/// `TransformRenderer`, for the same two formats at the same moment in `build_gpu`;
/// and the stroke integrate answered the question a third way, by acquiring a whole
/// pooled tile and clearing it on every pointer move.
#[derive(Clone)]
pub(crate) struct Zeroes {
    pub(crate) color: wgpu::TextureView,
    pub(crate) aux: wgpu::TextureView,
    /// The residual channel's stand-in, in a space that has one (§6.7). Bare canvas
    /// has no residual for the same reason it has no colour, and this is what the
    /// clamped loads read there.
    pub(crate) resid: Option<wgpu::TextureView>,
}

impl Zeroes {
    pub(crate) fn new(
        ctx: &GpuContext,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self {
            color: zero_texture(ctx, color_format, "stark zero color"),
            aux: zero_texture(ctx, aux_format, "stark zero aux"),
            resid: resid_format.map(|f| zero_texture(ctx, f, "stark zero resid")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`QUAD_STRIP`] is exactly `PrimitiveState::default()` with a strip topology,
    /// and [`fullscreen_pipeline`]'s primitive is exactly the default.
    ///
    /// Both were written out by hand when the pipelines that had spelled them
    /// inline were folded into [`render_pipeline`], so this is the assertion that
    /// the fold changed nothing. It is the only way that pass could have moved a
    /// pixel: every other field these helpers fill in is one wgpu validates or one
    /// the shader ignores, whereas winding, culling and topology decide what gets
    /// rasterized and would simply come out different, on ten pipelines, with
    /// nothing failing to say so.
    ///
    /// It also pins the pair against wgpu itself. A future release that changed a
    /// `PrimitiveState` default would leave these constants behind — which is the
    /// safe direction, and this says which way round it happened.
    #[test]
    fn quad_strip_is_the_default_with_a_strip_topology() {
        assert_eq!(
            QUAD_STRIP,
            wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            "QUAD_STRIP has drifted from the form the strip pipelines used",
        );
        // The transform's four gated/parcel pipelines spelled this one out, because
        // a flip reverses winding and both faces have to draw (§16).
        assert_eq!(
            QUAD_STRIP,
            wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            "QUAD_STRIP culls a face the transform needs drawn",
        );
    }
}
