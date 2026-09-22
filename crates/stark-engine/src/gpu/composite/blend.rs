//! Per-layer blend modes and the isolation they are defined against (§18.0.4,
//! §14.7).
//!
//! A group with a mode of its own composites *alone* into a scratch pair, and one
//! fullscreen pass then merges that into the accumulator. Because the merge reads
//! the accumulator and writes the result, and a texture cannot be both, the
//! accumulator ping-pongs between the caller's pair and this module's `swap`.

use super::attachment::Trio;
use std::sync::OnceLock;

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Bindings};
use crate::gpu::pigment::PigmentLut;
use crate::gpu::uniforms::UniformSlots;
use crate::view::Extent2;
use stark_model::document::BlendMode;
use stark_shaders::Stages;
use stark_shaders::mirror::blend::binding as bm;
use stark_shaders::mirror::blend::decl as bmd;
use stark_shaders::mirror::blend_common::binding as bc;
use stark_shaders::mirror::blend_common::decl as bcd;

use super::plan::Phase;

// Generated from `blend_common.wesl`'s own declaration (§6.7).
pub(crate) use stark_shaders::mirror::blend_common::Blend as BlendUniform;

/// The shader ABI for [`BlendMode`], kept here rather than on the enum: which `u32` a
/// mode is numbered is a fact about `blend_common.wesl`, not about the document, and
/// the codes are generated from its declaration (§6.10) rather than transcribed.
///
/// `Normal` reaches the pass only when the group is **clipped** or carries an opacity
/// of its own (§14.4); an ordinary normal layer is the absence of a pass.
fn blend_code(mode: BlendMode) -> u32 {
    use stark_shaders::mirror::blend_common as bc;
    match mode {
        BlendMode::Normal => bc::MODE_NORMAL,
        BlendMode::Reinhard => bc::MODE_REINHARD,
        BlendMode::Drago { .. } => bc::MODE_DRAGO,
        BlendMode::Multiply => bc::MODE_MULTIPLY,
    }
}

/// The blend pass's uniform for one merge (§18.0.4).
///
/// **The one place it is assembled**, screen and merge alike: `gpu::merge` runs this
/// very pipeline on tile-sized targets (§14.11).
///
/// `opacity` is the caller's because the two mean different things by it: the screen
/// folds the group's slider in here, where a merge has already folded the source
/// layer's into its expansion and has nothing left to fade.
pub(crate) fn blend_uniform(mode: BlendMode, clip: bool, opacity: f32) -> BlendUniform {
    BlendUniform {
        mode: blend_code(mode),
        k: mode.drago_k(),
        clip: u32::from(clip),
        opacity,
    }
}

/// The blend pass: one fullscreen draw merging an isolated group into the
/// accumulator.
///
/// **Shared, not owned.** A merge-down through a blend mode runs this very pipeline on
/// tile-sized targets (§14.11, `gpu::merge`), so a merged layer cannot drift from the
/// stack it stands in for. Behind an `Arc` also because building one decodes the
/// Mixbox LUT, which is not a thing to do twice per document.
pub(crate) struct BlendPass {
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// `blend_common`'s whole `@group(0)`, then the space's own `@group(1)` — the LUT
    /// and the two residuals, or nothing at all in a colorimetric space (§6.7).
    bgls: [Bindings; 2],
    /// Mixbox's LUT, `None` in a space whose blend shader declares none — which is
    /// every colorimetric one (§6.7).
    pigment: Option<PigmentLut>,
}

impl BlendPass {
    pub(crate) fn new(ctx: &GpuContext, color_space: &dyn ColorSpace) -> Self {
        let device = &ctx.device;
        let blend = color_space.blend_shader();
        let shader = desc::Module::new(device, "stark blend", blend);
        // A group per module (§6.10): `blend_common`'s, then the space's own, which is
        // empty in a colorimetric one. Every texture here is `textureLoad`ed at the
        // fragment's own coordinate except the LUT, which Mixbox interpolates in
        // hardware, so it alone comes out filterable.
        let formats = ChannelFormats::of(color_space);
        let stages = Stages::Render(blend.vs_main, blend.fs_main);
        let bgls = [
            Bindings::of(device, "stark blend bgl", stages, bcd::B, &[bcd::B]),
            Bindings::if_reached(device, "stark blend space bgl", &[stages], bmd::PIGMENT_LUT),
        ];
        let layout = desc::pipeline_layout_of(device, "stark blend layout", &[&bgls[0], &bgls[1]]);
        // No fixed-function blend on either target: the pass computes the whole
        // merge — backdrop included — and *replaces* what it writes. That is the
        // point of the ping-pong.
        let targets = formats.targets();
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark blend pipeline",
            &layout,
            &shader,
            (blend.vs_main, blend.fs_main),
            &targets,
        );
        let pigment = PigmentLut::of(ctx, blend.fs_main, bmd::PIGMENT_LUT);
        Self {
            pipeline,
            bgls,
            pigment,
        }
    }

    /// The LUT this pass decoded, for the filter pass to bind too
    /// ([`FilterPass::new`](super::filter::FilterPass::new)): both ask it the same
    /// question of the same space.
    pub(crate) fn pigment(&self) -> Option<&PigmentLut> {
        self.pigment.as_ref()
    }

    /// The bind group for one merge: the isolated `src` over the accumulator `back`
    /// (§18.0.4). **The one description of `blend_common.wesl`'s group** — the
    /// screen's and `gpu::merge`'s alike (§14.11), which is what keeps a merged layer
    /// from drifting from the stack it stands in for.
    ///
    /// The uniform arrives as a resource because it is the one thing the two callers
    /// differ in: the screen binds a slot of a buffer holding every merge in the
    /// frame, the merge binds the first slot of a buffer holding one.
    pub(crate) fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform: wgpu::BindingResource<'_>,
        back: Targets<'_>,
        src: Targets<'_>,
    ) -> [wgpu::BindGroup; 2] {
        // The two residual `expect`s hold together: `back` and `src` are targets of one
        // document, so the space that gave either a residual gave both one.
        let lut = || {
            self.pigment
                .as_ref()
                .expect("the layout holds the LUT only where the shader reads it")
        };
        [
            self.bgls[0].group(device, "stark blend bg", |i| match i {
                bc::B => uniform.clone(),
                bc::BACK_COLOR => wgpu::BindingResource::TextureView(back.color),
                bc::BACK_AUX => wgpu::BindingResource::TextureView(back.aux),
                bc::SRC_COLOR => wgpu::BindingResource::TextureView(src.color),
                bc::SRC_AUX => wgpu::BindingResource::TextureView(src.aux),
                other => unreachable!("`blend_common`'s group has no binding {other}"),
            }),
            // Empty in a colorimetric space, where the closure is asked for nothing.
            self.bgls[1].group(device, "stark blend space bg", |i| match i {
                bm::PIGMENT_LUT => wgpu::BindingResource::TextureView(&lut().view),
                bm::PIGMENT_SAMP => wgpu::BindingResource::Sampler(&lut().sampler),
                bm::BACK_RESID => wgpu::BindingResource::TextureView(
                    back.resid.expect("a residual build has one"),
                ),
                bm::SRC_RESID => {
                    wgpu::BindingResource::TextureView(src.resid.expect("a residual build has one"))
                }
                other => unreachable!("`blend.wesl`'s own group has no binding {other}"),
            }),
        ]
    }

    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b: Bounce<'_>,
        src: Targets<'_>,
        slots: &UniformSlots<BlendUniform>,
    ) {
        // Two per level rather than one per merge per frame: the views this names are
        // fixed by the phase, and the cached groups are dropped when they change (see
        // [`ScratchLevel::blend_bg`]).
        let bg = b.here.blend_bg(b.phase.back_is_swap, || {
            self.bind_group(&ctx.device, slots.resource(), b.back, src)
        });
        let offsets = [UniformSlots::<BlendUniform>::offset(b.slot)];
        desc::fullscreen_pass(
            encoder,
            "stark blend pass",
            b.out,
            &self.pipeline,
            desc::Bound::new(bg, &offsets),
        );
    }
}

/// One **bouncing** pass: what a merge and a filter both need beyond the pipeline kit
/// (§18.0.4, §21.3).
///
/// The two are the same shape — read the accumulator, write the other half of the
/// ping-pong, off a uniform slot of their own — and differ only in that a merge also
/// binds an isolated source. They bounce through the same scratch, which this module
/// owns.
pub(super) struct Bounce<'a> {
    /// The accumulator this pass reads.
    pub(super) back: Targets<'a>,
    /// The other half of the ping-pong, which it wholly replaces: one fullscreen
    /// triangle writes every texel, reading nothing from it.
    pub(super) out: Targets<'a>,
    /// Which dynamic-offset slot holds this pass's uniform.
    pub(super) slot: u32,
    /// The level it bounces at, which owns the bind groups it reads through.
    pub(super) here: &'a ScratchLevel,
    pub(super) phase: Phase,
}

/// The extra viewport-sized targets **one level** of isolation needs (§18.0.4).
///
/// `swap` is the other half of the ping-pong, and every level has one. `iso`, where a
/// group composites alone, exists only on a level whose stack actually isolates
/// something: a level that only ping-pongs (its sole non-direct members are filters,
/// §21.3) allocates no trio it cannot bind.
pub(super) struct ScratchLevel {
    swap: Trio,
    iso: Option<Trio>,
    /// The blend and filter bind groups this level's passes read through, one per
    /// **phase** of its ping-pong.
    ///
    /// A bind group over an accumulator is fully determined by which way round the
    /// ping-pong is: a pass at level `l` reads either this level's `swap` or the
    /// stack's own target (the caller's at level 0, level `l−1`'s `iso` below that),
    /// and a merge's source is always this level's `iso`. Two phases, so two of each,
    /// however many merges the document has.
    ///
    /// **Three things invalidate one:** a rebuilt accumulator (`ensure_targets` drops
    /// the whole scratch with it); each patch of a pick's trace, a target of its own
    /// ([`Compositor::composite_channels`] drops the groups before it); and a replaced
    /// uniform buffer — a frame with more merges than any before it replaces the pass's
    /// ([`UniformSlots::write`]), which `Streams::upload` reports for its callers to
    /// invalidate.
    ///
    /// [`Compositor::composite_channels`]: super::Compositor::composite_channels
    blend_bg: [OnceLock<[wgpu::BindGroup; 2]>; 2],
    filter_bg: [OnceLock<[wgpu::BindGroup; 2]>; 2],
}

impl ScratchLevel {
    fn new(device: &wgpu::Device, size: Extent2, iso: bool, formats: ChannelFormats) -> Self {
        let trio = |labels| Trio::new(device, size, labels, formats);
        Self {
            swap: trio((
                "stark blend swap color",
                "stark blend swap aux",
                "stark blend swap resid",
            )),
            iso: iso.then(|| {
                trio((
                    "stark blend iso color",
                    "stark blend iso aux",
                    "stark blend iso resid",
                ))
            }),
            blend_bg: Default::default(),
            filter_bg: Default::default(),
        }
    }

    /// The blend bind group for the phase in which the backdrop is (or is not) this
    /// level's `swap`, built on first use.
    pub(super) fn blend_bg(
        &self,
        back_is_swap: bool,
        make: impl FnOnce() -> [wgpu::BindGroup; 2],
    ) -> &[wgpu::BindGroup; 2] {
        self.blend_bg[usize::from(back_is_swap)].get_or_init(make)
    }

    /// [`Self::blend_bg`] for the filter pass, which reads the same accumulator
    /// through a layout of its own (§21.3).
    pub(super) fn filter_bg(
        &self,
        back_is_swap: bool,
        make: impl FnOnce() -> [wgpu::BindGroup; 2],
    ) -> &[wgpu::BindGroup; 2] {
        self.filter_bg[usize::from(back_is_swap)].get_or_init(make)
    }

    /// Drop both caches, for the one thing they name that the scratch's own lifetime
    /// does not cover — see [`Self::blend_bg`].
    fn invalidate_bind_groups(&mut self) {
        self.blend_bg = Default::default();
        self.filter_bg = Default::default();
    }

    pub(super) fn swap(&self) -> Targets<'_> {
        self.swap.targets()
    }

    pub(super) fn iso(&self) -> Targets<'_> {
        self.iso
            .as_ref()
            .expect("a merge at a level allocated without iso scratch (Plan::scratch)")
            .targets()
    }
}

/// What one level of the scratch holds (§18.0.4, §21.3). Ordered by what it holds, so
/// the level that serves two needs is the greater of them.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Level {
    /// The ping-pong's other half alone: a level whose only bounces are filters,
    /// which isolate nothing.
    Swap,
    /// The swap and an isolation trio, for a group composited alone.
    SwapIso,
}

impl Level {
    /// How many channel trios the level allocates.
    fn trios(self) -> u64 {
        match self {
            Level::Swap => 1,
            Level::SwapIso => 2,
        }
    }
}

/// The scratch a frame needs: one [`Level`] per level of nesting it bounces at, and
/// empty for a document whose every group draws straight into the accumulator — the
/// common case, which allocates nothing.
///
/// Decided by [`Plan::build`](super::plan::Plan::build), and what a [`ScratchTargets`]
/// records it was allocated for.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(super) struct ScratchNeeds(Vec<Level>);

impl ScratchNeeds {
    #[cfg(test)]
    pub(super) fn of(levels: &[Level]) -> Self {
        Self(levels.to_vec())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn levels(&self) -> &[Level] {
        &self.0
    }

    /// Make sure level `level` exists, as a [`Level::Swap`] if it is new.
    pub(super) fn reach(&mut self, level: usize) {
        if self.0.len() <= level {
            self.0.resize(level + 1, Level::Swap);
        }
    }

    /// Give level `level` an isolation trio, reaching it first.
    pub(super) fn isolate(&mut self, level: usize) {
        self.reach(level);
        self.0[level] = Level::SwapIso;
    }

    /// Whether scratch allocated for `have` serves these needs: at least as deep, and
    /// isolating wherever these isolate.
    pub(super) fn served_by(&self, have: &Self) -> bool {
        self.0.len() <= have.0.len() && self.0.iter().zip(&have.0).all(|(need, has)| need <= has)
    }

    /// The least scratch that serves both.
    pub(super) fn union(&self, other: &Self) -> Self {
        let (long, short) = if self.0.len() >= other.0.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut levels = long.0.clone();
        for (level, &other) in levels.iter_mut().zip(&short.0) {
            *level = (*level).max(other);
        }
        Self(levels)
    }

    /// What one texel of this scratch costs, given what one channel `trio` does —
    /// [`resolve::attachment_bytes`](super::resolve::attachment_bytes)'s scratch term.
    pub(super) fn bytes_per_px(&self, trio: u64) -> u64 {
        self.0.iter().map(|level| trio * level.trios()).sum()
    }
}

/// One [`ScratchLevel`] per level of group nesting the document actually reaches
/// (§14.7).
///
/// A group's members isolate into *its* level's `iso`, which is what the next level
/// down composites into — so nesting costs one of these per level, not one per group.
/// Allocated only when something has to be isolated at all (an ordinary painting never
/// pays the ~40 MB), and each level allocates only the half its stack uses: a document
/// whose only non-`Normal` thing is a filter pays for the ping-pong pair alone.
pub(super) struct ScratchTargets {
    size: Extent2,
    /// What the levels were allocated for.
    needs: ScratchNeeds,
    pub(super) levels: Vec<ScratchLevel>,
}

impl ScratchTargets {
    /// Drop every level's cached bind groups, because the uniform buffer they name
    /// has been replaced — see [`ScratchLevel::blend_bg`]. Cheap: at most two of each
    /// per level, and only on the frame that grew the buffer.
    pub(super) fn invalidate_bind_groups(&mut self) {
        for level in &mut self.levels {
            level.invalidate_bind_groups();
        }
    }

    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        needs: ScratchNeeds,
        formats: ChannelFormats,
    ) -> Self {
        let levels = needs
            .levels()
            .iter()
            .map(|&level| ScratchLevel::new(device, size, level == Level::SwapIso, formats))
            .collect();
        Self {
            size,
            needs,
            levels,
        }
    }

    /// Make sure `cache` holds scratch at `size` serving `needs`. Nothing when the frame
    /// bounces nowhere — the common case, which never allocates.
    ///
    /// Grown to the most the document has *reached* — depth and per-level isolation
    /// alike — and never shrunk within a size: a group opened and closed again over and
    /// over would otherwise reallocate viewport-sized targets each time. A new size
    /// starts over, since nothing at the old one can be kept.
    pub(super) fn ensure(
        cache: &mut Option<Self>,
        device: &wgpu::Device,
        size: Extent2,
        needs: &ScratchNeeds,
        formats: ChannelFormats,
    ) {
        let cached = cache.as_ref().map(|s| (s.size, &s.needs));
        if let Some(levels) = rebuild_for(cached, size, needs) {
            *cache = Some(Self::new(device, size, levels, formats));
        }
    }
}

/// What to allocate for a frame at `size` needing `needs`, given the cached scratch's
/// size and levels — `None` when the cache already serves, or the frame needs none.
fn rebuild_for(
    cached: Option<(Extent2, &ScratchNeeds)>,
    size: Extent2,
    needs: &ScratchNeeds,
) -> Option<ScratchNeeds> {
    if needs.is_empty() {
        return None;
    }
    match cached {
        Some((at, have)) if at == size => (!needs.served_by(have)).then(|| needs.union(have)),
        _ => Some(needs.clone()),
    }
}

/// A render pass that only clears. Encoded when the bottom of the stack is a blend
/// group: that pass *reads* the accumulator, so unlike a run of tiles it cannot
/// fold the clear into its own load op.
pub(super) fn clear_targets(encoder: &mut wgpu::CommandEncoder, into: Targets<'_>) {
    desc::begin_pass(encoder, "stark composite clear", into);
}

#[cfg(test)]
mod tests {
    use super::*;
    use Level::{Swap, SwapIso};

    fn size() -> Extent2 {
        Extent2::new(64, 64)
    }

    /// [`rebuild_for`] with the cache at `size()`, or with none.
    fn plan(cached: Option<&[Level]>, needs: &[Level]) -> Option<Vec<Level>> {
        let cached = cached.map(ScratchNeeds::of);
        rebuild_for(
            cached.as_ref().map(|have| (size(), have)),
            size(),
            &ScratchNeeds::of(needs),
        )
        .map(|n| n.levels().to_vec())
    }

    #[test]
    fn no_cache_allocates_exactly_what_the_frame_needs() {
        assert_eq!(plan(None, &[SwapIso, Swap]), Some(vec![SwapIso, Swap]));
    }

    #[test]
    fn a_frame_that_bounces_nowhere_allocates_nothing() {
        assert_eq!(plan(None, &[]), None);
    }

    #[test]
    fn an_equal_cache_is_reused() {
        assert_eq!(plan(Some(&[SwapIso, Swap]), &[SwapIso, Swap]), None);
    }

    #[test]
    fn a_deeper_cache_than_the_frame_is_reused() {
        assert_eq!(plan(Some(&[SwapIso, SwapIso, Swap]), &[SwapIso]), None);
    }

    #[test]
    fn a_shallower_cache_is_rebuilt_to_the_union() {
        assert_eq!(
            plan(Some(&[SwapIso]), &[Swap, SwapIso]),
            Some(vec![SwapIso, SwapIso])
        );
    }

    #[test]
    fn iso_needed_where_the_cache_has_none_rebuilds_keeping_the_rest() {
        assert_eq!(
            plan(Some(&[Swap, SwapIso]), &[SwapIso, Swap]),
            Some(vec![SwapIso, SwapIso])
        );
    }

    #[test]
    fn iso_cached_where_the_frame_needs_none_is_reused() {
        assert_eq!(plan(Some(&[SwapIso]), &[Swap]), None);
    }

    #[test]
    fn a_resized_target_starts_the_scratch_over() {
        // Not the union: nothing at the old size can be kept, so the new set is the
        // frame's own needs and no deeper.
        let have = ScratchNeeds::of(&[SwapIso, SwapIso]);
        assert_eq!(
            rebuild_for(
                Some((Extent2::new(65, 64), &have)),
                size(),
                &ScratchNeeds::of(&[Swap]),
            ),
            Some(ScratchNeeds::of(&[Swap]))
        );
    }

    /// A level that only ping-pongs costs one trio, an isolating one two.
    #[test]
    fn a_level_costs_a_trio_per_target_it_holds() {
        assert_eq!(ScratchNeeds::default().bytes_per_px(10), 0);
        assert_eq!(ScratchNeeds::of(&[Swap, SwapIso]).bytes_per_px(10), 30);
    }
}
