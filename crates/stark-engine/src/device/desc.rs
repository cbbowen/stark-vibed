//! The wgpu descriptors this subsystem writes over and over (§7).
//!
//! Nothing here decides anything. Every function is one shape of descriptor with
//! the fields that never vary already filled in — `depth_stencil: None`,
//! `multisample: Default::default()`, `multiview_mask: None`, `cache: None`,
//! `immediate_size: 0` — so a call site states only what makes it different from
//! its neighbours.
//!
//! That is the whole argument for the module: a shared descriptor makes the meant
//! differences the only ones written down. The same goes for the one pass shape
//! written everywhere, the fullscreen triangle: [`OnePass`] builds its layout and
//! pipeline from one pair of entry points, and [`fullscreen_pass`] encodes it.
//!
//! **No layout is written here at all**, every one is read off the shader
//! ([`stark_shaders::layout_of`], §6.10). What [`Bindings`] adds is that the entries
//! stay beside the layout, so every group over it is built from the same ones.

use std::sync::Arc;

use stark_shaders::{EntryPoint, Stages};

use crate::device::context::GpuContext;

// ---- bind group layouts ---------------------------------------------------------

/// A bind group layout of `entries`.
fn bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// The most bindings one group may hold — [`bind_group_of`] fills its entries into an
/// array of this size on the stack, since it runs once per tile per pass. Checked where
/// the **layout** is built ([`Bindings::over`]), so a longer one fails where its
/// renderer is built rather than at its first draw. The wet loop's deposit is the
/// longest today.
const MAX_SLOTS: usize = 24;

/// The bind group for the `declared` entries a layout was built from ([`Bindings`]),
/// with `resource` asked for each.
///
/// Private, and reached only through [`Bindings`]: the entry list *is* the description
/// of the group, so a caller that could name a different one is the whole disagreement
/// this path removes.
fn bind_group_of<'a>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    declared: &[wgpu::BindGroupLayoutEntry],
    mut resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
) -> wgpu::BindGroup {
    // On the stack, filled in layout order; the tail past `declared.len()` names no
    // resource and is never handed to wgpu — the descriptor takes that prefix.
    let mut it = declared.iter();
    let entries: [wgpu::BindGroupEntry<'a>; MAX_SLOTS] = std::array::from_fn(|_| match it.next() {
        Some(e) => wgpu::BindGroupEntry {
            binding: e.binding,
            resource: resource(e.binding),
        },
        None => wgpu::BindGroupEntry {
            binding: u32::MAX,
            resource: wgpu::BindingResource::BufferArray(&[]),
        },
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries[..declared.len()],
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
    /// `Arc`-backed like the layout beside it: a renderer holding one is cloned per
    /// `Action::Context`, which on the live-paint path is once per pointer move.
    entries: Arc<[wgpu::BindGroupLayoutEntry]>,
    /// The `@group` this describes, for [`pipeline_layout_of`].
    group: u32,
}

impl Bindings {
    /// The layout **one pipeline** needs for `anchor`'s group
    /// ([`stark_shaders::layout_of`]), keeping its entries for [`Self::group`] and its
    /// group number for [`pipeline_layout_of`].
    ///
    /// The `@group` it stands at is then the shader's answer rather than a position a
    /// call site counted.
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

    /// [`Self::of`] for a group a pipeline may reach nothing of
    /// ([`stark_shaders::layout_if_reached`]): the colour space's own `@group(1)`,
    /// which a colorimetric space declares nothing in and binds empty (§6.7).
    pub(crate) fn if_reached(
        device: &wgpu::Device,
        label: &str,
        pipelines: &[Stages],
        anchor: stark_shaders::Binding,
    ) -> Self {
        Self::over(
            device,
            label,
            stark_shaders::layout_if_reached(pipelines, anchor),
            anchor.group,
        )
    }

    /// [`Self::of`] for a layout several pipelines **share**
    /// ([`stark_shaders::layout_shared_by`], which states what sharing costs).
    pub(crate) fn shared_by(
        device: &wgpu::Device,
        label: &str,
        pipelines: &[Stages],
        anchor: stark_shaders::Binding,
        dynamic: &[stark_shaders::Binding],
    ) -> Self {
        Self::over(
            device,
            label,
            stark_shaders::layout_shared_by(pipelines, anchor, dynamic),
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
            entries: entries.into(),
            group,
        }
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

    /// Set `group` — one built over this layout — at the `@group` this layout
    /// describes, so no call site counts the index.
    pub(crate) fn set(
        &self,
        pass: &mut impl SetBindGroup,
        group: &wgpu::BindGroup,
        offsets: &[u32],
    ) {
        pass.set_group(self.group, group, offsets);
    }
}

/// A pass a bind group is set on. `wgpu` gives its render and compute passes no
/// common trait, and [`Bindings::set`] serves both.
pub(crate) trait SetBindGroup {
    fn set_group(&mut self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]);
}

impl SetBindGroup for wgpu::RenderPass<'_> {
    fn set_group(&mut self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) {
        self.set_bind_group(index, group, offsets);
    }
}

impl SetBindGroup for wgpu::ComputePass<'_> {
    fn set_group(&mut self, index: u32, group: &wgpu::BindGroup, offsets: &[u32]) {
        self.set_bind_group(index, group, offsets);
    }
}

/// The bind groups one pass sets, in `@group` order, with the dynamic offsets
/// `@group(0)` takes.
///
/// A pass over a colour-space-split shader binds two — the shared module's whole group
/// 0 and the space's own group 1, empty where the space declares nothing (§6.7) — and
/// every other pass here binds one. Both go through [`Self::set`], so a pipeline layout
/// of two and a pass that set one is a shape no call site can write.
#[derive(Clone, Copy)]
pub(crate) struct Bound<'a> {
    groups: &'a [wgpu::BindGroup],
    offsets: &'a [u32],
}

impl<'a> Bound<'a> {
    /// `groups` in `@group` order; `offsets` are `@group(0)`'s.
    pub(crate) fn new(groups: &'a [wgpu::BindGroup], offsets: &'a [u32]) -> Self {
        Self { groups, offsets }
    }

    /// One group, no dynamic offset.
    pub(crate) fn one(group: &'a wgpu::BindGroup) -> Self {
        Self::dynamic(group, &[])
    }

    /// One group at `offsets` — a per-draw slot of a staged uniform.
    pub(crate) fn dynamic(group: &'a wgpu::BindGroup, offsets: &'a [u32]) -> Self {
        Self::new(std::slice::from_ref(group), offsets)
    }

    pub(crate) fn set(self, pass: &mut wgpu::RenderPass<'_>) {
        for (i, group) in self.groups.iter().enumerate() {
            let offsets = if i == 0 { self.offsets } else { &[][..] };
            pass.set_bind_group(i as u32, group, offsets);
        }
    }
}

/// A pipeline layout over `bgls`.
fn pipeline_layout(
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

/// The most color targets a pass here attaches: the channel trio and the ceiling
/// lane (§6.2, §6.7).
const MAX_ATTACHMENTS: usize = 4;

/// The color attachments of one render pass, **every one cleared** unless the pass
/// draws over what they hold ([`Self::loaded`]) — a channel trio (from
/// [`Targets`](crate::device::channels::Targets)), one target (from a `&TextureView`), or
/// one target cleared to a value that means something ([`Self::cleared_to`]).
///
/// A fixed array rather than a `Vec`: one is built per pass, and a pass per tile is
/// the common case (§6.2).
pub(crate) struct Attachments<'a> {
    slots: [Option<wgpu::RenderPassColorAttachment<'a>>; MAX_ATTACHMENTS],
    /// Up to and including the last attached location, so a trailing hole is never
    /// handed to `wgpu`.
    len: usize,
}

impl<'a> Attachments<'a> {
    /// `targets` in `@location` order, each cleared to transparent; a `None` leaves its
    /// location unattached.
    ///
    /// # Panics
    /// If `targets` names more than [`MAX_ATTACHMENTS`] locations.
    pub(crate) fn cleared(
        targets: impl IntoIterator<Item = Option<&'a wgpu::TextureView>>,
    ) -> Self {
        Self::with(CLEAR, targets)
    }

    /// [`Self::cleared`], keeping what each target holds — a pass that adds to a total
    /// an earlier pass began (a resumed stroke parcel, §6.2).
    pub(crate) fn loaded(targets: impl IntoIterator<Item = Option<&'a wgpu::TextureView>>) -> Self {
        Self::with(LOAD, targets)
    }

    /// [`Self::cleared`] with `ops` on each, for a caller that decides between the two
    /// (a pass A run that clears the frame's first tiles and loads onto the rest).
    pub(crate) fn with(
        ops: wgpu::Operations<wgpu::Color>,
        targets: impl IntoIterator<Item = Option<&'a wgpu::TextureView>>,
    ) -> Self {
        Self::at_locations(targets.into_iter().map(|t| t.map(|view| attach(view, ops))))
    }

    /// The attachments at the `@location`s they stand at, each carrying its **own**
    /// load op — a clear-only pass whose targets mean unlike things (a freshly charged
    /// tip, §6.2) is the case `ops` cannot express.
    ///
    /// **This is [`targets_for`] on the attachment side.** The length is read off where
    /// the last `Some` falls rather than counted by the caller, so a list sliced to a
    /// count — `&attachments[..t.count()]`, once per pass — cannot stop one location
    /// early and lose that lane in silence.
    ///
    /// # Panics
    /// If `targets` names more than [`MAX_ATTACHMENTS`] locations.
    pub(crate) fn at_locations(
        targets: impl IntoIterator<Item = Option<wgpu::RenderPassColorAttachment<'a>>>,
    ) -> Self {
        let mut out = Self {
            slots: [const { None }; MAX_ATTACHMENTS],
            len: 0,
        };
        for (location, target) in targets.into_iter().enumerate() {
            assert!(
                location < MAX_ATTACHMENTS,
                "a pass attaches at most {MAX_ATTACHMENTS} targets",
            );
            if target.is_some() {
                out.len = location + 1;
            }
            out.slots[location] = target;
        }
        out
    }

    /// One target cleared to `color`, for where the clear value carries meaning: the
    /// coverage outside a selection's own tiles (§6.8).
    pub(crate) fn cleared_to(view: &'a wgpu::TextureView, color: wgpu::Color) -> Self {
        Self::at_locations([Some(attach(view, clear_to(color)))])
    }

    fn as_slice(&self) -> &[Option<wgpu::RenderPassColorAttachment<'a>>] {
        &self.slots[..self.len]
    }
}

impl<'a> From<&'a wgpu::TextureView> for Attachments<'a> {
    fn from(view: &'a wgpu::TextureView) -> Self {
        Self::cleared([Some(view)])
    }
}

/// Begin a render pass writing `into`.
pub(crate) fn begin_pass<'e, 'a>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    into: impl Into<Attachments<'a>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: into.into().as_slice(),
        ..Default::default()
    })
}

/// Encode one **fullscreen pass**: `pipeline` over `bound`, drawn as the fullscreen
/// triangle into `into` — the whole of what a pass that computes every texel of its
/// targets records ([`fullscreen_pipeline`]).
///
/// Clears rather than loads: the triangle writes every texel, so what was there is
/// never read.
pub(crate) fn fullscreen_pass<'a>(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    into: impl Into<Attachments<'a>>,
    pipeline: &wgpu::RenderPipeline,
    bound: Bound<'_>,
) {
    let mut pass = begin_pass(encoder, label, into);
    pass.set_pipeline(pipeline);
    bound.set(&mut pass);
    pass.draw(FULLSCREEN_VERTICES, 0..1);
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
    ///
    /// **Every pipeline crosses this**, vertex, fragment and compute alike, which is
    /// why the coverage recorder hangs off it rather than off the three constructors.
    fn declares(&self, label: &str, entry: EntryPoint) {
        assert_eq!(
            self.artifact, entry.artifact,
            "`{label}` builds a stage from `{}`'s `{}`, over `{}`'s module",
            entry.artifact, entry.name, self.artifact,
        );
        #[cfg(test)]
        record::built(entry);
    }
}

/// Which entry points the engine actually builds pipelines from
/// ([`pipeline_coverage`](crate::gpu::pipeline_coverage)).
///
/// Per thread and only while a [`Recording`](record::Recording) is open, so what comes
/// back is what *this* test built rather than whatever else the binary has compiled.
#[cfg(test)]
pub(crate) mod record {
    use std::cell::RefCell;
    use std::collections::BTreeSet;

    use stark_shaders::EntryPoint;

    /// An entry point as the recorder identifies it: the artifact it belongs to and its
    /// name, which together are unique — and which stay comparable across the two
    /// builds of a shader, where the names alone would not be.
    pub(crate) type Built = (&'static str, &'static str);

    thread_local! {
        static OPEN: RefCell<Option<BTreeSet<Built>>> = const { RefCell::new(None) };
    }

    /// Records every entry point a pipeline is built from on this thread until
    /// [`take`](Self::take) — or until it is dropped, which discards the recording.
    pub(crate) struct Recording(());

    impl Recording {
        /// # Panics
        /// If one is already open on this thread: two overlapping recordings would
        /// each be missing whatever the other took.
        pub(crate) fn open() -> Self {
            OPEN.with(|open| {
                let mut open = open.borrow_mut();
                assert!(open.is_none(), "a recording is already open on this thread");
                *open = Some(BTreeSet::new());
            });
            Self(())
        }

        pub(crate) fn take(self) -> BTreeSet<Built> {
            OPEN.with(|open| open.borrow_mut().take())
                .expect("the recording is open until it is taken")
        }
    }

    impl Drop for Recording {
        fn drop(&mut self) {
            OPEN.with(|open| *open.borrow_mut() = None);
        }
    }

    pub(super) fn built(entry: EntryPoint) {
        OPEN.with(|open| {
            if let Some(set) = open.borrow_mut().as_mut() {
                set.insert((entry.artifact, entry.name));
            }
        });
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

/// The color targets `fs` writes, out of what the pass has to offer at each
/// `@location`.
///
/// **A short target list is silent.** WebGPU discards a fragment output with no target,
/// so a pass that stops one entry early loses that lane with nothing to say so. The
/// counts were sliced by hand against whichever flag decided them — `[..2 +
/// usize::from(has_resid)]`, a `truncate`, a `match` on the ceiling axis — one per pass
/// and one per build of it. The entry point states its own `@location`s (§6.10), so the
/// list is built from those instead of checked against them afterwards.
///
/// `by_location` is indexed by `@location`, `None` where the pass has nothing for one —
/// which is also the shape `wgpu` wants back, a fragment state's array being indexed by
/// location. So a hole survives: a space with no residual at 2 under a ceiling at 3
/// attaches `[color, aux, None, ceiling]`, and [`targets_agree`] then holds by
/// construction.
///
/// # Panics
/// If `fs` writes a location `by_location` offers nothing for — the silent loss, said
/// out loud at the one place that can still see both halves.
pub(crate) fn targets_for(
    fs: EntryPoint,
    by_location: &[Option<wgpu::ColorTargetState>],
) -> Vec<Option<wgpu::ColorTargetState>> {
    // Ascending, so the last is the highest; a hole below it is a location this entry
    // point does not write.
    let width = fs.targets.last().map_or(0, |&at| at as usize + 1);
    let mut out = vec![None; width];
    for &at in fs.targets {
        let at = at as usize;
        let state = by_location.get(at).cloned().flatten().unwrap_or_else(|| {
            panic!(
                "`{}` writes `@location({at})`, which this pass offers no color target for",
                fs.name,
            )
        });
        out[at] = Some(state);
    }
    out
}

/// Fail unless `targets` carries a state at exactly the `@location`s `fs` writes.
///
/// [`targets_for`] satisfies this by construction; what is left for it to catch is a
/// list assembled any other way.
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

/// The vertices of the fullscreen triangle every [`fullscreen_pipeline`] draws, off
/// `lib/quad.wesl`'s own count (§6.10) — the vertex function decides how many it is
/// written for, and a `0..3` here would be that number a second time.
pub(crate) const FULLSCREEN_VERTICES: std::ops::Range<u32> =
    0..stark_shaders::mirror::quad::FULLSCREEN_VERTS;

/// [`render_pipeline`] for the **fullscreen triangle** shape: no vertex buffers, no
/// instancing, [`FULLSCREEN_VERTICES`] once. Ten of the passes here are this — every
/// one that computes a whole target from what it reads rather than rasterizing
/// geometry.
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

/// A fullscreen pass's pipeline and the one bind group layout it binds, **both built
/// from one vertex/fragment pair** — so the layout is derived from exactly the entry
/// points the pipeline runs, which two separately written pairs could not promise.
#[derive(Clone)]
pub(crate) struct OnePass {
    pub(crate) bindings: Bindings,
    pub(crate) pipeline: wgpu::RenderPipeline,
}

impl OnePass {
    /// [`Bindings::of`] `anchor`'s group and a [`fullscreen_pipeline`] over it, from
    /// one `(vs, fs)` pair of `module`'s.
    pub(crate) fn fullscreen(
        device: &wgpu::Device,
        label: &str,
        module: &Module,
        (vs, fs): (EntryPoint, EntryPoint),
        anchor: stark_shaders::Binding,
        dynamic: &[stark_shaders::Binding],
        targets: &[Option<wgpu::ColorTargetState>],
    ) -> Self {
        let bindings = Bindings::of(
            device,
            &format!("{label} bgl"),
            Stages::Render(vs, fs),
            anchor,
            dynamic,
        );
        let layout = pipeline_layout_of(device, &format!("{label} layout"), &[&bindings]);
        let pipeline = fullscreen_pipeline(device, label, &layout, module, (vs, fs), targets);
        Self { bindings, pipeline }
    }
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
/// rasterizes per-tile geometry takes ([`QUAD_VERTICES`], one instance a draw).
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

/// The vertices of one [`QUAD_STRIP`] quad, off `lib/quad.wesl`'s own count, for
/// [`FULLSCREEN_VERTICES`]' reason.
pub(crate) const QUAD_VERTICES: std::ops::Range<u32> = 0..stark_shaders::mirror::quad::QUAD_VERTS;

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

// ---- textures with data --------------------------------------------------------

/// A 2D texture of `size` that shaders sample and an upload fills: one mip, one
/// sample. A `size` deeper than 1 is an array of that many layers.
pub(crate) fn sampled_texture(
    label: &str,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
) -> wgpu::TextureDescriptor<'_> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

/// A texture `desc` describes, created holding `data`: **the one way a texture is
/// made with its contents** ([`create_texture_with_data`]).
///
/// `data` is tightly packed, layer by layer and each layer's mips in order
/// ([`TextureDataOrder::LayerMajor`]); [`upload_len`] is its length. The bytes go
/// through the queue rather than a buffer mapped at creation, so no mapping limit
/// applies to their size.
///
/// # Panics
/// If `data` is not exactly [`upload_len`] bytes. The util reads a prefix of a longer
/// slice and indexes off the end of a shorter one; neither is an upload anyone meant.
///
/// [`create_texture_with_data`]: wgpu::util::DeviceExt::create_texture_with_data
/// [`TextureDataOrder::LayerMajor`]: wgpu::util::TextureDataOrder::LayerMajor
pub(crate) fn texture_with_data(
    ctx: &GpuContext,
    desc: &wgpu::TextureDescriptor<'_>,
    data: &[u8],
) -> wgpu::Texture {
    use wgpu::util::DeviceExt as _;
    assert_eq!(
        data.len(),
        upload_len(desc),
        "`{}` is uploaded from the wrong number of bytes",
        desc.label.unwrap_or("a texture"),
    );
    ctx.device.create_texture_with_data(
        &ctx.queue,
        desc,
        wgpu::util::TextureDataOrder::LayerMajor,
        data,
    )
}

/// A texture view as the resource a bind-group entry takes.
pub(crate) fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

/// `texture` at `origin`, as a copy addresses it: mip 0, every aspect.
pub(crate) fn copy_at(
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin,
        aspect: wgpu::TextureAspect::All,
    }
}

/// A bilinear sampler, clamped at the edge — what every pass here that filters a
/// texture samples with, short of the few that tile or mip.
pub(crate) fn linear_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// One texel of `format`, in bytes — **the one answer to that question**.
///
/// It was asked four ways, each with a different fallback for a format it is not
/// defined on (`unwrap_or(8)` in one place, `unwrap_or(0)` in three), so a format that
/// stopped answering would have costed a texture at eight bytes here and at nothing
/// there. Every caller is pricing a texture for a budget or an upload, and for every
/// one of them a format with no answer is a bug in this build rather than a size.
///
/// # Panics
/// For a compressed or depth-stencil format, which nothing here allocates.
pub(crate) fn texel_bytes(format: wgpu::TextureFormat) -> u32 {
    assert_eq!(
        format.block_dimensions(),
        (1, 1),
        "{format:?} is compressed, so it has no texel size",
    );
    format
        .block_copy_size(None)
        .expect("an uncompressed format has one aspect")
}

/// The bytes of every layer and mip of `desc`, tightly packed — what
/// [`texture_with_data`] is handed.
///
/// # Panics
/// For a compressed or depth-stencil format, which no upload here has.
pub(crate) fn upload_len(desc: &wgpu::TextureDescriptor<'_>) -> usize {
    let texel = texel_bytes(desc.format) as usize;
    let per_layer: usize = (0..desc.mip_level_count)
        .map(|mip| {
            let size = desc.mip_level_size(mip).expect("a mip below the count");
            // A 2D level's depth counts layers, not texels.
            let depth = match desc.dimension {
                wgpu::TextureDimension::D3 => size.depth_or_array_layers,
                _ => 1,
            };
            size.width as usize * size.height as usize * depth as usize
        })
        .sum();
    texel * per_layer * desc.array_layer_count() as usize
}

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
    let one = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    texture_with_data(ctx, &sampled_texture(label, format, one), bytes)
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// [`constant_texture`] holding zeros — "no paint here", in whichever format the
/// consumer reads.
pub(crate) fn zero_texture(
    ctx: &GpuContext,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    constant_texture(ctx, format, &vec![0u8; texel_bytes(format) as usize], label)
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

    /// A pass attaches up to its last target: a hole before it stays a hole, one after
    /// it is not attached at all — the ceiling lane sits at 3 with or without a
    /// residual at 2 (§6.2).
    #[test]
    fn a_pass_attaches_through_its_last_target_and_keeps_the_holes_before_it() {
        let built = pollster::block_on(GpuContext::headless());
        let Some(ctx) = crate::testing::or_skip(built, "GPU tests") else {
            return;
        };
        let view = zero_texture(&ctx, wgpu::TextureFormat::R16Float, "stark attach test");
        let v = Some(&view);
        let shape = |a: Attachments<'_>| -> Vec<Option<bool>> {
            a.as_slice()
                .iter()
                .map(|s| s.as_ref().map(|s| matches!(s.ops.load, wgpu::LoadOp::Load)))
                .collect()
        };
        assert_eq!(shape(Attachments::cleared([None, None])), []);
        assert_eq!(
            shape(Attachments::cleared([v, v, None, None])),
            [Some(false); 2]
        );
        assert_eq!(
            shape(Attachments::loaded([v, v, None, v])),
            [Some(true), Some(true), None, Some(true)]
        );
    }

    /// [`upload_len`] counts what `create_texture_with_data` reads: every mip of every
    /// layer, a 2D level's depth as layers and a 3D level's as texels.
    #[test]
    fn an_upload_is_every_mip_of_every_layer() {
        use wgpu::TextureFormat as F;
        let size = |width, height, depth_or_array_layers| wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers,
        };
        assert_eq!(
            upload_len(&sampled_texture("t", F::R8Unorm, size(1, 1, 1))),
            1
        );
        // An array: three layers of 4×2 eight-byte texels.
        let array = sampled_texture("t", F::Rg32Float, size(4, 2, 3));
        assert_eq!(upload_len(&array), 4 * 2 * 8 * 3);
        // A full chain on a non-square base: 8×2, 4×1, 2×1, 1×1.
        let chain = wgpu::TextureDescriptor {
            mip_level_count: 4,
            ..sampled_texture("t", F::Rgba16Float, size(8, 2, 1))
        };
        assert_eq!(upload_len(&chain), (16 + 4 + 2 + 1) * 8);
        // A volume's depth halves with its mips, where an array's layers do not.
        let volume = wgpu::TextureDescriptor {
            dimension: wgpu::TextureDimension::D3,
            mip_level_count: 2,
            ..sampled_texture("t", F::R8Unorm, size(4, 4, 4))
        };
        assert_eq!(upload_len(&volume), 64 + 8);
        let mipped_array = wgpu::TextureDescriptor {
            mip_level_count: 2,
            ..array
        };
        assert_eq!(upload_len(&mipped_array), (4 * 2 + 2) * 8 * 3);
    }
}
