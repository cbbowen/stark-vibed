//! The wgpu descriptors this subsystem writes over and over (§7).
//!
//! Nothing here decides anything. Every function is one shape of descriptor with
//! the fields that never vary already filled in — `depth_stencil: None`,
//! `multisample: Default::default()`, `multiview_mask: None`, `cache: None`,
//! `immediate_size: 0` — so a call site states only what makes it different from
//! its neighbours.
//!
//! That is the whole argument for the module: a shared descriptor makes the meant
//! differences the only ones written down.
//!
//! **A layout is not written here at all**, it is read off the shader (§6.10).
//! [`stark_shaders::layout_of`] derives a whole group's entries from the pipeline that
//! binds it — [`stark_shaders::layout_shared_by`] where several do — and [`Bindings`]
//! keeps them beside the layout so every group over it is built from the same ones.
//!
//! [`Slot`] is what is left of the hand-written form, for the layouts not yet derived:
//! a list naming the generated declarations, saying only [`How`] the host binds —
//! through a sampler, as a dynamic-offset slot, in which stages.

use stark_shaders::{EntryPoint, Stages};

use crate::gpu::context::GpuContext;

// ---- bind group layout entries -------------------------------------------------

fn tex_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// A filtering sampler.
fn sampler(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
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
fn uniform_slot(binding: u32, vis: wgpu::ShaderStages, slot: u64) -> wgpu::BindGroupLayoutEntry {
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
/// different slot in each of a module's groups, and an index alone cannot tell them
/// apart.
#[derive(Clone, Copy)]
pub(crate) struct Slot {
    decl: stark_shaders::Binding,
    how: How,
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
            vis: None,
        }
    }

    /// A slot this entry point reads **through a sampler**, which makes its texture
    /// required to be filterable and its sampler a filtering one.
    pub(crate) const fn sampled(decl: stark_shaders::Binding) -> Self {
        Self {
            decl,
            how: How::Sampled,
            vis: None,
        }
    }

    /// A uniform bound as one **dynamic-offset slot** ([`uniform_slot`]).
    pub(crate) const fn dynamic(decl: stark_shaders::Binding) -> Self {
        Self {
            decl,
            how: How::Dynamic,
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

    /// Whether this list says the entry point reads the slot **through a sampler** —
    /// the half of [`How`] the shader also knows, and so the half a generated
    /// [`Use`](stark_shaders::Use) can be checked against.
    ///
    /// The layout path reads [`How`] directly; this exists for
    /// [`slot_agreement`](crate::gpu::slot_agreement), which is the only thing that has
    /// a second opinion to compare it with.
    #[cfg(test)]
    pub(crate) const fn is_sampled(&self) -> bool {
        matches!(self.how, How::Sampled)
    }

    /// The stages that read **this** slot, where they are not the whole layout's.
    ///
    /// Most layouts are one stage's and pass it once. Pass A's view group and the
    /// overlay's are not: a `View` uniform is the vertex stage's while the sampler
    /// beside it is the fragment's, and declaring the pair `VERTEX_FRAGMENT` would ask
    /// for a visibility neither binding uses.
    pub(crate) const fn in_stages(mut self, vis: wgpu::ShaderStages) -> Self {
        self.vis = Some(vis);
        self
    }

    /// Whether this build has the slot at all: an `@if(resid)` declaration exists only
    /// in a color space that carries a residual (§6.7).
    pub(crate) const fn present(&self, resid: bool) -> bool {
        resid || !self.decl.resid
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
        // Filterability is the only half of the sample type the declaration does not
        // decide, and `Sample::of` is where the two meet (§6.10).
        stark_shaders::BindKind::Texture { dim, sample } => {
            tex_entry(decl.index, vis, sample.of(slot.how == How::Sampled), dim)
        }
        stark_shaders::BindKind::Storage {
            dim,
            format,
            access,
        } => wgpu::BindGroupLayoutEntry {
            binding: decl.index,
            visibility: vis,
            ty: wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension: dim,
            },
            count: None,
        },
    })
}

/// The most bindings one group may hold — [`bind_group_over`] fills its entries into
/// an array of this size on the stack, since it runs once per tile per pass. Checked
/// where the **layout** is built, both by [`slot_entries`] and by [`Bindings::of`], so
/// a longer one fails where its renderer is built rather than at its first draw. The
/// longest today is the dynamics' deposit at 17.
const MAX_SLOTS: usize = 24;

/// The layout entries for `slots`, in list order, with the residual gate applied.
///
/// A bind group layout describes exactly one `@group`, so a list spanning two is a
/// mistake in the list rather than a layout with a meaning — which is only sayable
/// because the declaration carries its group. Panics if it does, or if the list is
/// longer than [`MAX_SLOTS`].
fn slot_entries(
    label: &str,
    slots: &[Slot],
    vis: wgpu::ShaderStages,
    resid: bool,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    assert!(
        slots.len() <= MAX_SLOTS,
        "`{label}` lists {} slots; a bind group holds at most {MAX_SLOTS}",
        slots.len()
    );
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
    slots
        .iter()
        .filter_map(|s| slot_entry(*s, vis, resid))
        .collect()
}

/// A bind group layout for the `slots` one entry point reads, typed from the shader's
/// own declarations (§6.10).
///
/// The list of slots is the *only* thing written on the host, and it is written once:
/// [`bind_group_for`] builds the matching group from the same list, so a layout and its
/// group cannot disagree about which bindings are present, in what order, or of what
/// type.
pub(crate) fn layout_for(
    device: &wgpu::Device,
    label: &str,
    slots: &[Slot],
    vis: wgpu::ShaderStages,
    resid: bool,
) -> wgpu::BindGroupLayout {
    bind_group_layout(device, label, &slot_entries(label, slots, vis, resid))
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
    resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
) -> wgpu::BindGroup {
    bind_group_over(
        device,
        label,
        layout,
        slots.iter().filter(|s| s.present(resid)).map(Slot::binding),
        resource,
    )
}

/// The bind group for the `entries` a layout was built from ([`Bindings`]), with
/// `resource` asked for each.
///
/// Private, and reached only through [`Bindings`]: the entry list *is* the description
/// of the group, so a caller that could name a different one is the whole disagreement
/// this path removes.
fn bind_group_of<'a>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupLayoutEntry],
    resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
) -> wgpu::BindGroup {
    bind_group_over(
        device,
        label,
        layout,
        entries.iter().map(|e| e.binding),
        resource,
    )
}

/// One bind group over the `@binding` indices its layout declares, `resource` asked
/// for each.
fn bind_group_over<'a>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    mut bindings: impl Iterator<Item = u32>,
    mut resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
) -> wgpu::BindGroup {
    // On the stack, filled in slot order; the tail past `count` names no resource and
    // is never handed to wgpu — the descriptor takes `[..count]`.
    let mut count = 0;
    let entries: [wgpu::BindGroupEntry<'a>; MAX_SLOTS] =
        std::array::from_fn(|_| match bindings.next() {
            Some(binding) => {
                count += 1;
                wgpu::BindGroupEntry {
                    binding,
                    resource: resource(binding),
                }
            }
            None => wgpu::BindGroupEntry {
                binding: u32::MAX,
                resource: wgpu::BindingResource::BufferArray(&[]),
            },
        });
    assert!(
        bindings.next().is_none(),
        "`{label}` binds more than {MAX_SLOTS} slots, which is all this array holds",
    );
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries[..count],
    })
}

/// A bind group layout **with the entries it was built from**, so every group over it
/// is built from the same ones by construction.
///
/// Two calls that each decide for themselves which bindings a layout holds — one for
/// the layout, one for the group — can disagree, and the disagreement is a group
/// missing the entries its layout declares: a validation error only a pigment document
/// reaches, which the build without Mixbox cannot render (§6.7).
#[derive(Clone)]
pub(crate) struct Bindings {
    layout: wgpu::BindGroupLayout,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// The `@group` this describes, for [`pipeline_layout_of`].
    group: u32,
}

impl Bindings {
    /// The layout **one pipeline** needs for `anchor`'s group
    /// ([`stark_shaders::layout_of`]), keeping its entries for [`Self::group`] and its
    /// group number for [`pipeline_layout_of`].
    ///
    /// The anchor is taken rather than the finished entries, so a [`Bindings`] cannot
    /// be built over a list that came from nowhere — and so the `@group` it stands at
    /// is the shader's answer rather than a position a call site counted.
    pub(crate) fn of(
        device: &wgpu::Device,
        label: &str,
        stages: Stages,
        anchor: stark_shaders::Binding,
        dynamic: &[stark_shaders::Binding],
    ) -> Self {
        Self::over(
            device,
            label,
            stark_shaders::layout_of(stages, anchor, dynamic),
            anchor.group,
        )
    }

    /// [`Self::of`] for a layout several pipelines **share**
    /// ([`stark_shaders::layout_shared_by`], which states what sharing costs).
    pub(crate) fn shared_by(
        device: &wgpu::Device,
        label: &str,
        eps: &[EntryPoint],
        anchor: stark_shaders::Binding,
        dynamic: &[stark_shaders::Binding],
    ) -> Self {
        Self::over(
            device,
            label,
            stark_shaders::layout_shared_by(eps, anchor, dynamic),
            anchor.group,
        )
    }

    /// A layout over `entries`, which describe `@group(group)`.
    fn over(
        device: &wgpu::Device,
        label: &str,
        entries: Vec<wgpu::BindGroupLayoutEntry>,
        group: u32,
    ) -> Self {
        assert!(
            entries.len() <= MAX_SLOTS,
            "`{label}` holds {} bindings; a bind group holds at most {MAX_SLOTS}",
            entries.len(),
        );
        Self {
            layout: bind_group_layout(device, label, &entries),
            entries,
            group,
        }
    }

    /// [`Self::of`] over a hand-written slot list, whose group is its first slot's
    /// declaration ([`slot_entries`] refuses a list spanning two).
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &str,
        slots: &'static [Slot],
        vis: wgpu::ShaderStages,
        resid: bool,
    ) -> Self {
        let group = slots
            .first()
            .expect("a slot list names at least one binding")
            .decl()
            .group;
        Self::over(device, label, slot_entries(label, slots, vis, resid), group)
    }

    /// The layout, for a [`pipeline_layout`].
    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// [`bind_group_of`] against this layout, so `resource` is asked for exactly the
    /// entries the layout declares.
    pub(crate) fn group<'a>(
        &self,
        device: &wgpu::Device,
        label: &str,
        resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        bind_group_of(device, label, &self.layout, &self.entries, resource)
    }
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

/// [`pipeline_layout`] over layouts that know which `@group` they are — so the one
/// thing a pipeline layout states, **position**, is checked rather than trusted.
///
/// A pipeline layout is positional: the i-th layout *is* `@group(i)`. That is the last
/// shader-stated fact the host still writes by hand, and a swapped pair is otherwise a
/// failure only a device reports.
///
/// # Panics
/// If any layout stands at a group other than its position.
pub(crate) fn pipeline_layout_of(
    device: &wgpu::Device,
    label: &str,
    bindings: &[&Bindings],
) -> wgpu::PipelineLayout {
    for (i, b) in bindings.iter().enumerate() {
        assert_eq!(
            b.group, i as u32,
            "`{label}` binds a layout for @group({}) at position {i}",
            b.group,
        );
    }
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> =
        bindings.iter().map(|b| Some(b.layout())).collect();
    pipeline_layout(device, label, &bgls)
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

/// A compiled shader module **and the artifact it was compiled from** (§6.10).
///
/// `wgpu` ties a module to nothing: a pipeline is a module plus two entry-point names,
/// and handing it one shader's module with another's entry points creates a pipeline
/// that runs. The plain stamp module under the *ceiling* record's `fs_main` would pass
/// [`targets_agree`] against `[0, 1, 3]`, attach location 3, and have the plain
/// fragment write nothing to it — different pixels, and no error anywhere.
///
/// One argument, not two: the artifact is the record's own, so a call site cannot mix
/// one shader's WGSL with another's name.
pub(crate) struct Module {
    module: wgpu::ShaderModule,
    artifact: &'static str,
}

impl Module {
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &str,
        of: &dyn stark_shaders::Artifact,
    ) -> Self {
        Self {
            module: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(of.wgsl().into()),
            }),
            artifact: of.name(),
        }
    }

    /// Fail unless `entry` is one this module declares.
    fn declares(&self, label: &str, entry: EntryPoint) {
        assert_eq!(
            self.artifact, entry.artifact,
            "`{label}` builds a stage from `{}`'s `{}`, over `{}`'s module",
            entry.artifact, entry.name, self.artifact,
        );
    }
}

/// What a render pipeline in this subsystem actually varies. Everything absent from
/// this struct is a default no pass here has ever wanted to change.
///
/// The two stages are the shader's own [`EntryPoint`]s, not their names, so
/// [`render_pipeline`] can check what the pass attaches against what the entry point
/// writes — and so no call site spells a function name `wgpu` would only miss on a
/// GPU.
pub(crate) struct RenderPipe<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    /// One module for both stages — every shader here declares its vertex and
    /// fragment entry points together.
    pub module: &'a Module,
    pub vs: EntryPoint,
    pub fs: EntryPoint,
    pub primitive: wgpu::PrimitiveState,
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

/// A render pipeline, with the five fields no pass here varies filled in.
///
/// # Panics
/// If a stage is handed the other stage's entry point, if either entry point is not
/// one `module` declares, or if the color targets are not the ones the fragment entry
/// point writes ([`targets_agree`]).
pub(crate) fn render_pipeline(device: &wgpu::Device, p: RenderPipe<'_>) -> wgpu::RenderPipeline {
    p.module.declares(p.label, p.vs);
    p.module.declares(p.label, p.fs);
    assert_eq!(
        p.vs.stage,
        wgpu::ShaderStages::VERTEX,
        "`{}` builds its vertex stage from `{}`, which is not one",
        p.label,
        p.vs.name,
    );
    assert_eq!(
        p.fs.stage,
        wgpu::ShaderStages::FRAGMENT,
        "`{}` builds its fragment stage from `{}`, which is not one",
        p.label,
        p.fs.name,
    );
    targets_agree(p.label, p.fs, p.targets);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(p.label),
        layout: Some(p.layout),
        vertex: wgpu::VertexState {
            module: &p.module.module,
            entry_point: Some(p.vs.name),
            compilation_options: Default::default(),
            buffers: p.buffers,
        },
        primitive: p.primitive,
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &p.module.module,
            entry_point: Some(p.fs.name),
            compilation_options: Default::default(),
            targets: p.targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Fail unless `targets` carries a state at exactly the `@location`s `fs` writes.
///
/// **A short target list is silent.** WebGPU discards a fragment output with no
/// target, so a pass that stops one entry early loses that lane with nothing to say
/// so — and the lists here are sliced by hand against the residual and the ceiling
/// (`swept`, `erase`), which is where the count comes from. The entry point knows its
/// own `@location`s (§6.10), so the two are compared rather than one trusted.
///
/// A `None` in the middle is a hole the space does not have (a residual at location 2
/// with the ceiling at 3), and the shader writes no location there either — so the
/// comparison is over the positions that carry a state, not over the length.
fn targets_agree(label: &str, fs: EntryPoint, targets: &[Option<wgpu::ColorTargetState>]) {
    let attached: Vec<u32> = targets
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_some())
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(
        attached, fs.targets,
        "`{label}` attaches color targets at {attached:?}, where `{}` writes {:?}",
        fs.name, fs.targets,
    );
}

/// [`render_pipeline`] for the **fullscreen triangle** shape: no vertex buffers, no
/// instancing, `draw(0..3, 0..1)`. Ten of the passes here are this — every one that
/// computes a whole target from what it reads rather than rasterizing geometry.
pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &Module,
    entries: (EntryPoint, EntryPoint),
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

/// A compute pipeline over one kernel, with the fields no dispatch here varies filled
/// in.
///
/// # Panics
/// If `entry` is not a compute entry point, or not one `module` declares.
pub(crate) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &Module,
    entry: EntryPoint,
) -> wgpu::ComputePipeline {
    module.declares(label, entry);
    assert_eq!(
        entry.stage,
        wgpu::ShaderStages::COMPUTE,
        "`{label}` dispatches `{}`, which is not a compute entry point",
        entry.name,
    );
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module.module,
        entry_point: Some(entry.name),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// An instanced quad drawn as a triangle strip — the shape every pass that
/// rasterizes per-tile geometry takes (`draw(0..4, i..i+1)`).
///
/// Spelled out field by field because a `const` cannot call `Default::default()`, so
/// a field here disagreeing with that default would alter what ten pipelines
/// rasterize without failing anything. `quad_strip_is_the_default_with_a_strip_topology`
/// is what checks it.
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
    /// Both are written out by hand rather than derived from the pipelines that use
    /// them, and winding, culling and topology are the fields that decide what gets
    /// rasterized: a drift here comes out as different pixels on ten pipelines with
    /// nothing failing to say so. It also pins the pair against wgpu itself, whose
    /// own `PrimitiveState` defaults could move underneath them.
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

    /// The residual gate a hand-written list applies is [`Slot::present`]: an
    /// `@if(resid)` declaration exists only with the residual, a plain one always.
    /// Device-free, so it can run where the pigment build's bind groups themselves
    /// cannot be built.
    #[test]
    fn a_gated_slot_is_present_only_with_the_residual() {
        use stark_shaders::mirror::fill::decl as fd;
        const _: () = assert!(
            fd::BASE_RESID.resid && !fd::BASE_COLOR.resid,
            "the fixtures no longer stand for a gated and an ungated declaration",
        );
        let declared = Slot::at(fd::BASE_RESID);
        let plain = Slot::at(fd::BASE_COLOR);
        for resid in [false, true] {
            assert_eq!(
                declared.present(resid),
                resid,
                "@if(resid) at resid={resid}"
            );
            assert!(plain.present(resid), "a plain slot at resid={resid}");
        }
    }
}
