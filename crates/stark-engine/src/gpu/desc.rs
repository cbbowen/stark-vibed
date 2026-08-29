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
//! **A layout is no longer written here at all**, it is read off the shader (§6.10).
//! [`layout_for`] and [`bind_group_for`] take a list of [`Slot`]s naming the generated
//! declarations, and everything a `desc::` call used to spell — the index, whether the
//! slot is a uniform and how wide, a sampler, a texture, or a storage texture of a
//! particular format, and whether the residual build has it — comes from the WESL. What
//! a list still says is [`How`] the host binds: through a sampler, as a dynamic-offset
//! slot, in which stages, and (once) that a slot exists only where the space has a
//! residual. Each of those four is a fact about the *host*, and each has a case in this
//! codebase that proves it cannot be read off the declaration.

use crate::gpu::context::GpuContext;

// ---- bind group layout entries -------------------------------------------------

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

/// A filtering sampler.
pub(crate) fn sampler(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// One **dynamic-offset slot** of a uniform buffer, `slot` bytes wide — how both
/// render paths vary a uniform across the draws or dispatches of a single pass
/// without a buffer per draw (`gpu::uniforms`).
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

// ---- shader-declared binding lists (§6.10) --------------------------------------

/// The **two** things a `@binding` declaration does not decide, so that a slot list
/// says only these and reads everything else off the shader.
///
/// Both are properties of how the host *binds*, not of what the shader declares, and
/// each has a case in this codebase that proves it:
///
/// * **Filterability.** `dynamics.wesl`'s `region_color` is `textureLoad`ed by
///   `snapshot` and `textureSample`d by `exchange` — the same slot, non-filterable in
///   one layout and filterable in the next.
/// * **Dynamic offset.** `fill.wesl` declares `f` and `tile` both `var<uniform>`; the
///   first is one buffer for the whole fill and the second is a per-tile slot of one
///   (`UniformSlots`). The WGSL is identical either way.
#[derive(Clone, Copy, PartialEq, Eq)]
enum How {
    /// `textureLoad`ed, or a uniform bound whole — the common case.
    Plain,
    /// Read through a sampler, which makes its texture required to be filterable and
    /// its sampler a filtering one.
    Sampled,
    /// A uniform bound as one **dynamic-offset slot** of a larger buffer, which is how
    /// both render paths vary a uniform across the draws or dispatches of a pass.
    Dynamic,
}

/// One binding an entry point reads, as the **host** has to name it: the shader's own
/// declaration, plus [`How`] the host binds it.
///
/// Everything else about a slot — whether it is a uniform and how wide, a sampler, a
/// texture, or a storage texture of a particular format, and whether it exists at all
/// in a build without the residual — rides in the generated
/// [`Binding`](stark_shaders::Binding).
///
/// **The declaration is carried, not looked up.** A slot list names
/// `decl::REGION_COLOR`, so there is no index to resolve against a table — which is
/// what makes a multi-group module safe to describe at all: `@binding(0)` means a
/// different slot in each of a module's groups, and a lookup keyed on the index alone
/// silently answered with whichever came first.
#[derive(Clone, Copy)]
pub(crate) struct Slot {
    decl: stark_shaders::Binding,
    how: How,
    /// Whether this slot exists only in a residual build **although its declaration is
    /// unconditional** — see [`Slot::only_with_resid`].
    resid_only: bool,
    /// This slot's own visibility, where it differs from the layout's — see
    /// [`Slot::in_stages`].
    vis: Option<wgpu::ShaderStages>,
}

impl Slot {
    /// A slot this entry point reads with `textureLoad`, or a whole-buffer uniform —
    /// anything with no filterability and no dynamic offset to speak of.
    pub(crate) const fn at(decl: stark_shaders::Binding) -> Self {
        Self {
            decl,
            how: How::Plain,
            resid_only: false,
            vis: None,
        }
    }

    /// A slot this entry point reads **through a sampler**, which makes its texture
    /// required to be filterable and its sampler a filtering one.
    pub(crate) const fn sampled(decl: stark_shaders::Binding) -> Self {
        Self {
            decl,
            how: How::Sampled,
            resid_only: false,
            vis: None,
        }
    }

    /// A uniform bound as one **dynamic-offset slot** ([`uniform_slot`]).
    pub(crate) const fn dynamic(decl: stark_shaders::Binding) -> Self {
        Self {
            decl,
            how: How::Dynamic,
            resid_only: false,
            vis: None,
        }
    }

    /// The slot's `@binding` index — what a bind-group entry is keyed on, once the
    /// group is fixed.
    pub(crate) const fn binding(&self) -> u32 {
        self.decl.index
    }

    /// The shader's declaration of this slot.
    pub(crate) const fn decl(&self) -> &stark_shaders::Binding {
        &self.decl
    }

    /// A slot the shader declares **unconditionally** but that only a residual build
    /// has, because the module declaring it is reached only by a space with a residual
    /// (§6.7).
    ///
    /// The one thing about the residual the WESL cannot say. `blend_mixbox.wesl`
    /// declares bindings 7 and 8 with no `@if`, past where `blend_common` (0–4) and
    /// `mixbox_lut` (5–6) stop, and it is right not to gate them: that shader is *only*
    /// compiled for the pigment space, so inside it they always exist. What varies is
    /// which shader the document runs, and that is `colorspace.rs`'s business — so the
    /// host states it, here, once per slot.
    ///
    /// Orthogonal to [`How`] rather than folded into it, because the two genuinely vary
    /// independently: `filter_mixbox.wesl` **samples** its `back_resid` for the
    /// chromatic gather (§21.10) where `blend_mixbox.wesl` loads its two.
    pub(crate) const fn only_with_resid(mut self) -> Self {
        self.resid_only = true;
        self
    }

    /// The stages that read **this** slot, where they are not the whole layout's.
    ///
    /// Most layouts are one stage's, and pass it once. The two that are not are pass A's
    /// view group and the overlay's: a `View` uniform is read by the vertex stage to
    /// place the quad while the sampler beside it is the fragment's, so declaring the
    /// pair `VERTEX_FRAGMENT` would ask for a visibility neither binding uses.
    pub(crate) const fn in_stages(mut self, vis: wgpu::ShaderStages) -> Self {
        self.vis = Some(vis);
        self
    }

    /// Whether this build has the slot at all: a `@if(resid)` declaration, or one the
    /// host has gated with [`Self::only_with_resid`], exists only in a color space that
    /// carries a residual (§6.7).
    const fn present(&self, resid: bool) -> bool {
        resid || !(self.decl.resid || self.resid_only)
    }
}

/// The layout entry for one slot, from the shader's own declaration.
///
/// Returns `None` for a slot the shader declares `@if(resid)` when this build has no
/// residual, so a layout never restates that gate as an element count
/// (`[..12 + 4 * usize::from(resid)]`).
fn slot_entry(
    slot: Slot,
    vis: wgpu::ShaderStages,
    resid: bool,
) -> Option<wgpu::BindGroupLayoutEntry> {
    if !slot.present(resid) {
        return None;
    }
    let decl = slot.decl();
    let vis = match slot.vis {
        Some(v) => v,
        None => vis,
    };
    Some(match decl.kind {
        // `min_binding_size` is the declared struct's own WGSL size either way — free
        // validation against a truncated write, against the size the shader reads.
        stark_shaders::BindKind::Uniform { min_size } => match slot.how {
            How::Dynamic => uniform_slot(decl.index, vis, min_size),
            _ => buffer_entry(decl.index, vis, false, wgpu::BufferSize::new(min_size)),
        },
        stark_shaders::BindKind::Sampler => sampler(decl.index, vis),
        stark_shaders::BindKind::Texture { dim } => {
            tex_entry(decl.index, vis, slot.how == How::Sampled, dim)
        }
        stark_shaders::BindKind::Storage { dim, format } => wgpu::BindGroupLayoutEntry {
            binding: decl.index,
            visibility: vis,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: dim,
            },
            count: None,
        },
    })
}

/// A bind group layout for the `slots` one entry point reads, typed from the shader's
/// own declarations (§6.10).
///
/// The list of slots is the *only* thing written on the host, and it is written once:
/// [`bind_group_for`] builds the matching group from the same list, so a layout and its
/// group cannot disagree about which bindings are present, in what order, or of what
/// type. Two hand-kept arrays per entry point joined by a magic element count — seven
/// pairs of them for `dynamics.wesl` alone — is what that saves.
///
/// A bind group layout describes exactly one `@group`, so a list spanning two is a
/// mistake in the list rather than a layout with a meaning: the assertion below is what
/// says so, and it is only sayable because the declaration carries its group.
pub(crate) fn layout_for(
    device: &wgpu::Device,
    label: &str,
    slots: &[Slot],
    vis: wgpu::ShaderStages,
    resid: bool,
) -> wgpu::BindGroupLayout {
    if let Some(first) = slots.first() {
        for s in slots {
            assert_eq!(
                s.decl().group,
                first.decl().group,
                "`{label}` lists `{}` from @group({}) beside `{}` from @group({}); one \
                 layout describes one group",
                s.decl().name,
                s.decl().group,
                first.decl().name,
                first.decl().group,
            );
        }
    }
    let entries: Vec<_> = slots
        .iter()
        .filter_map(|s| slot_entry(*s, vis, resid))
        .collect();
    bind_group_layout(device, label, &entries)
}

/// The bind group for the same `slots` [`layout_for`] built a layout from, with
/// `resource` asked for each one that is present in this build.
///
/// `resource` is never asked about a slot the residual gate excluded, so a caller has
/// nothing to say about the residual beyond supplying the views when it has them.
pub(crate) fn bind_group_for<'a>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    slots: &[Slot],
    resid: bool,
    mut resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
) -> wgpu::BindGroup {
    let entries: Vec<_> = slots
        .iter()
        .filter(|s| s.present(resid))
        .map(|s| wgpu::BindGroupEntry {
            binding: s.binding(),
            resource: resource(s.binding()),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
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

/// A color attachment on `view` with `ops`.
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
/// which is the one thing here that could change a pixel silently: the ten pipelines
/// that take this would otherwise each write
/// `{ topology: TriangleStrip, ..Default::default() }`, and a field here disagreeing
/// with that default would alter what they rasterize without failing anything.
/// `quad_strip_is_the_default_with_a_strip_topology` is what checks it.
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

/// A color target that replaces what it writes — the pass computes the finished
/// texel, so there is nothing for fixed-function blending to do.
pub(crate) fn target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    blended_target(format, None)
}

/// A color target with an explicit blend state.
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
