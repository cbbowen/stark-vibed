//! GPU execution of the transforms (§16): the whole-plane affine, and the
//! rect-scoped perspective (§16.8) and warp (§16.9).
//!
//! [`TransformRenderer::apply`] takes a layer's tile map, the author's selection
//! and a [`TransformMap`], and produces the transformed tile map plus the carried
//! selection — copy-on-write like every stroke, so old history versions keep
//! their tiles.
//!
//! The plan (which tiles, which quads) is CPU-side and pure ([`plan`]); this module
//! owns only the passes of `transform.wesl`:
//!
//! - **parcel** — source interiors forward-rasterized as transformed quads into a
//!   scratch pair. Source interiors tile the plane, so their images are disjoint
//!   and the pass needs no blending and no order; the gated families draw *pieces*
//!   (`tile ∩ rect`, split further per warp sub-cell), watertight the same way.
//! - **combine** — cut the destination's own base by its mask (the lift law) and
//!   stack the parcel by the shared parcel-deposit law (`paint_common.wesl`), the
//!   cut additionally scoped by the source rect for the gated families.
//! - **mask** — the selection mask resampled under the same map: pure Replace
//!   for the affine, residue-plus-max-blended-union for the gated families
//!   (§16.8).
//!
//! The uniform each draw reads is packed on the CPU, in [`pack`].
//!
//! Like [`StrokeRenderer`](super::stroke::StrokeRenderer) this holds only
//! immutable GPU objects, so it is cheap to `Clone` and rides in the
//! `Action::Context` (§5).

use crate::gpu::tile::TilePairHandle;
use std::collections::BTreeMap;

use rpds::HashTrieMap;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::gpu::channels::Zeroes;
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::scratch::{LeasedSlots, ScratchPool, SubmitScope};
use crate::gpu::selection::{MaskSource, SelectionRenderer, outside_clear};
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TileMap, TilePool};
use stark_model::document::TransformMap;
use stark_model::geom::{Affine2, TileCoord, TileRect, Vec2};
use stark_shaders::mirror::transform::binding as t;
use stark_shaders::mirror::transform::decl as td;
use stark_shaders::mirror::transform::{
    Combine as CombineUniform, Gated as GatedUniform, Quad as QuadUniform,
};
use stark_shaders::{EntryPoint, Stages};

mod pack;
pub(crate) mod plan;

use plan::{
    gated_geometry, plan_float, plan_gated_mask, plan_gated_paint, plan_mask, plan_paint,
    shifted_mask_cover, shifted_mask_sources,
};

use pack::{
    Gated, combine_uniform, slot_count, stage_combines, stage_gated_masks, stage_gated_parcels,
    stage_quads,
};

/// Every group-0 uniform of one kind that an `apply` draws with, staged in full before
/// anything is submitted ([`SubmitScope::run_slots`]), and a cursor handing the draws
/// their slots in the order they were staged.
///
/// So a transform walks its plan twice: [`stage_quads`] and its siblings build the
/// uniforms in draw order, and [`Self::take`] hands them back in that same order. The
/// two walks agree exactly when they visit the same `(destination, source)` pairs — a
/// source the draw skips (no tile at that coord) still takes its slot — and the plan's
/// shape is what both read, so the count is known before a command is encoded
/// ([`slot_count`]).
struct Slots<T> {
    /// `None` when nothing was staged, and then no draw will ask.
    staged: Option<LeasedSlots<T>>,
    next: usize,
}

impl<T: bytemuck::Pod> Slots<T> {
    fn stage(scope: &mut SubmitScope, label: &'static str, uniforms: &[T]) -> Self {
        Self {
            staged: scope.run_slots(label, uniforms),
            next: 0,
        }
    }

    /// The next draw's dynamic offset.
    fn take(&mut self) -> u32 {
        let offset = self.staged().offset(self.next);
        self.next += 1;
        offset
    }

    /// Whether every staged slot was taken — the two walks agreed to the end.
    fn spent(&self) -> bool {
        self.next == self.staged.as_ref().map_or(0, LeasedSlots::len)
    }

    /// One slot's binding, which each draw displaces by its [`Self::take`].
    fn window(&self) -> wgpu::BindingResource<'_> {
        self.staged().window()
    }

    fn staged(&self) -> &LeasedSlots<T> {
        self.staged
            .as_ref()
            .expect("a transform draw past the slots its plan staged")
    }
}

/// [`Slots`] with the one group-0 bind group its draws share, and the layout it was
/// built over: the quad and gated passes' group is the uniform and the sampler,
/// nothing per draw.
struct Group<'r, T> {
    slots: Slots<T>,
    bg: Option<wgpu::BindGroup>,
    bindings: &'r desc::Bindings,
}

impl<T: bytemuck::Pod> Group<'_, T> {
    fn take(&mut self) -> u32 {
        self.slots.take()
    }

    /// Bind the group at one draw's slot ([`Self::take`]).
    fn set(&self, pass: &mut wgpu::RenderPass<'_>, offset: u32) {
        let bg = self.bg.as_ref().expect("a draw was staged for it");
        self.bindings.set(pass, bg, &[offset]);
    }

    fn spent(&self) -> bool {
        self.slots.spent()
    }

    /// Draw one quad per `(slot, group 1)`: this group at the draw's slot, and `src`'s
    /// group over what that draw samples.
    fn draw_quads(&self, pass: &mut wgpu::RenderPass<'_>, src: &desc::Bindings, draws: &[Draw]) {
        for (offset, src_bg) in draws {
            self.set(pass, *offset);
            src.set(pass, src_bg, &[]);
            pass.draw(desc::QUAD_VERTICES, 0..1);
        }
    }
}

/// One quad draw: its group-0 slot, and the group-1 bind over what it samples.
type Draw = (u32, wgpu::BindGroup);

/// The base tiles around `coord` that the identity parcel draws — the tile and its
/// neighbours, not the tile alone: a tile's texture reaches an apron past its
/// interior, and under the identity that band lies in the neighbours' interiors, so
/// drawing them is what keeps the child's aprons bit-identical to its neighbours
/// (§6.4).
fn neighbours_in(base: &TileMap, coord: TileCoord) -> Vec<TileCoord> {
    (-1..=1)
        .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| {
            let c = TileCoord::new(coord.x.checked_add(dx)?, coord.y.checked_add(dy)?);
            base.get(&c).map(|_| c)
        })
        .collect()
}

/// What a transform's passes draw **from**, as against which piece they are drawing.
/// Assembled once per [`TransformRenderer::apply`] and threaded through every pass.
///
/// `src_bgs` rides along because it is scoped to exactly this: a source tile's bind
/// group is shared across every destination its image reaches, and there is no
/// destination outside one `apply` for it to be shared with.
struct Source<'a> {
    pool: &'a TilePool,
    base: &'a TileMap,
    selection: &'a Selection,
    src_bgs: BTreeMap<TileCoord, wgpu::BindGroup>,
}

impl<'a> Source<'a> {
    fn new(pool: &'a TilePool, base: &'a TileMap, selection: &'a Selection) -> Self {
        Self {
            pool,
            base,
            selection,
            src_bgs: BTreeMap::new(),
        }
    }
}

/// A tile's paint channels as this module passes them around: the moved parcel a
/// quad pass writes, and the destination a combine writes.
type Parcel = Channels;

/// The tile rect spanning `base`'s keys — every coord whose gate a cut can read
/// (sources are base tiles; a destination's mask only ever cuts base paint), so
/// the shifted mask is built no wider than the layer's extent (§14.12).
fn base_rect(base: &TileMap) -> TileRect {
    base.keys()
        .map(|c| TileRect {
            min: (c.x, c.y),
            max: (c.x, c.y),
        })
        .fold(TileRect::EMPTY, TileRect::union)
}

#[derive(Clone)]
pub(crate) struct TransformRenderer {
    ctx: GpuContext,
    /// The channel formats this transform's tiles carry — the color space's,
    /// resolved once (§6.7).
    formats: ChannelFormats,
    parcel_pipeline: wgpu::RenderPipeline,
    mask_pipeline: wgpu::RenderPipeline,
    /// The cut and stack of one destination tile, the only pass whose layout no
    /// other pipeline shares.
    combine: desc::OnePass,
    /// The rect-scoped maps' passes (§16.8, §16.9): pieces through `vs_gated`,
    /// and the mask's residue + max-blended moved coverage.
    parcel_gated_pipeline: wgpu::RenderPipeline,
    mask_gated_pipeline: wgpu::RenderPipeline,
    mask_base_pipeline: wgpu::RenderPipeline,
    quad_bindings: desc::Bindings,
    /// The rect-scoped maps' own group. Separate from [`Self::quad_bindings`] because
    /// their uniform is a different, larger struct at the same `@group(0) @binding(0)`
    /// — `transform.wesl` puts `Quad` and `Gated` there because no entry point reaches
    /// both, and `wgpu` wants a bound size at least what the shader reads, so one
    /// layout cannot cover the two.
    gated_bindings: desc::Bindings,
    src_bindings: desc::Bindings,
    mask_src_bindings: desc::Bindings,
    sampler: wgpu::Sampler,
    /// The base of a virgin destination and the parcel of a cut-only tile, so the
    /// combine is one shader whatever exists.
    zeroes: Zeroes,
    scratch: ScratchPool,
    /// For the selection constants (0/1 coverage) bound where a mask has no tile.
    selection: SelectionRenderer,
}

impl TransformRenderer {
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        selection: SelectionRenderer,
        zeroes: Zeroes,
        scratch: ScratchPool,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space);

        let xf = stark_shaders::transform(color_space.resid());
        let shader = desc::Module::new(device, "stark transform", xf);

        // Each pipeline's stages, written once for the layouts that serve it and for
        // the pipeline itself.
        let parcel = (xf.vs_quad, xf.fs_parcel);
        let mask = (xf.vs_quad, xf.fs_mask);
        let parcel_gated = (xf.vs_gated, xf.fs_parcel_gated);
        let mask_gated = (xf.vs_gated, xf.fs_mask_gated);
        let mask_base = (xf.vs_fill, xf.fs_mask_base);

        // Four layouts, each off the pipelines that share it — safely, every slot
        // between them being read-only. Group 0 is shared along one axis (the affine
        // pipelines against the rect-scoped ones, which is why `Quad` and `Gated` need
        // layouts of their own) and group 1 along the other (what carries a parcel
        // against what carries only a mask).
        let shared = |label: &str, over: &[(EntryPoint, EntryPoint)], anchor, dynamic: &[_]| {
            let over: Vec<Stages> = over
                .iter()
                .map(|&(vs, fs)| Stages::Render(vs, fs))
                .collect();
            desc::Bindings::shared_by(device, label, &over, anchor, dynamic)
        };
        let quad_bindings = shared("stark transform quad bgl", &[parcel, mask], td::Q, &[td::Q]);
        let gated_bindings = shared(
            "stark transform gated bgl",
            &[parcel_gated, mask_gated, mask_base],
            td::QG,
            &[td::QG],
        );
        // The source tile being carried, at group 1 — its channels and the mask over
        // it, all sampled, since the parcel resamples under the map (§16).
        let src_bindings = shared(
            "stark transform src bgl",
            &[parcel, parcel_gated],
            td::SRC_COLOR,
            &[],
        );
        let mask_src_bindings = shared(
            "stark transform mask src bgl",
            &[mask, mask_gated, mask_base],
            td::SRC_MASK,
            &[],
        );

        let gated_layout = desc::pipeline_layout_of(
            device,
            "stark transform gated layout",
            &[&gated_bindings, &src_bindings],
        );
        let gated_mask_layout = desc::pipeline_layout_of(
            device,
            "stark transform gated mask layout",
            &[&gated_bindings, &mask_src_bindings],
        );
        let quad_layout = desc::pipeline_layout_of(
            device,
            "stark transform quad layout",
            &[&quad_bindings, &src_bindings],
        );
        let mask_layout = desc::pipeline_layout_of(
            device,
            "stark transform mask layout",
            &[&quad_bindings, &mask_src_bindings],
        );

        // A negative-determinant affine (a flip) reverses winding, so both faces must
        // draw — which `QUAD_STRIP` already leaves unculled.
        let strip = desc::QUAD_STRIP;
        // Parcels are disjoint; the combine computes rather than blends.
        let target = desc::target;

        // The paint channels, and the mask alone. Both paint pipelines — the moved
        // parcel and the combine — write the residual as a third target where the
        // space has one (§6.7).
        let paint = formats.targets();
        let mask_target = [target(MASK_FORMAT)];
        // Moved mask coverage lands with **max** blending over the residue: the soft
        // union of what stayed and what arrived (§16.8), and — unlike the paint
        // parcels — safe under any draw order.
        let max_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let mask_union = [desc::blended_target(
            MASK_FORMAT,
            Some(wgpu::BlendState {
                color: max_blend,
                alpha: max_blend,
            }),
        )];
        // One quad per source unit, forward-rasterized; and the two fullscreen passes
        // that read a whole destination tile back (`vs_fill`).
        let quad = |label,
                    layout,
                    (vs, fs): (EntryPoint, EntryPoint),
                    targets: &[Option<wgpu::ColorTargetState>]| {
            desc::render_pipeline(
                device,
                desc::RenderPipe {
                    label,
                    layout,
                    module: &shader,
                    vs,
                    fs,
                    primitive: strip,
                    buffers: &[],
                    targets,
                },
            )
        };
        let parcel_pipeline = quad("stark transform parcel", &quad_layout, parcel, &paint);
        let mask_pipeline = quad("stark transform mask", &mask_layout, mask, &mask_target);
        let parcel_gated_pipeline = quad(
            "stark transform parcel gated",
            &gated_layout,
            parcel_gated,
            &paint,
        );
        let mask_gated_pipeline = quad(
            "stark transform mask gated",
            &gated_mask_layout,
            mask_gated,
            &mask_union,
        );
        // The combine's inputs claim group 0 from 2 up, the quad passes never seeing
        // them (`transform.wesl`'s header).
        let combine = desc::OnePass::fullscreen(
            device,
            "stark transform combine",
            &shader,
            (xf.vs_fill, xf.fs_combine),
            td::BASE_COLOR,
            &[td::QC],
            &paint,
        );
        // The residue is a *gated* pass — it reads `qg`'s source rect (`fs_mask_base`)
        // — so it takes the gated group, like the two pipelines it is drawn beside.
        let mask_base_pipeline = desc::fullscreen_pipeline(
            device,
            "stark transform mask base",
            &gated_mask_layout,
            &shader,
            mask_base,
            &mask_target,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark transform sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            ctx: ctx.clone(),
            formats,
            parcel_pipeline,
            mask_pipeline,
            combine,
            parcel_gated_pipeline,
            mask_gated_pipeline,
            mask_base_pipeline,
            quad_bindings,
            gated_bindings,
            src_bindings,
            mask_src_bindings,
            sampler,
            zeroes,
            scratch,
            selection,
        }
    }

    /// Transform `base` (one layer's tiles) and `selection` (the author's mask)
    /// under `map`, stated on the canvas, against a layer whose frame sits at
    /// `translation` (§14.12): the paint side runs the map conjugated into the
    /// frame and gates through the mask brought into it, while the mask side
    /// moves the canvas mask under the canvas map — so the moved paint and its
    /// outline land together on the canvas. `None` rejects the whole action —
    /// an unusable map, or more tiles than the caps allow — deterministically,
    /// so peers and replays agree (§16.1).
    pub(crate) fn apply(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        map: &TransformMap,
        translation: stark_model::geom::IVec2,
    ) -> Option<(TileMap, Selection)> {
        let local = map.under_translation(translation);
        match (&local, map) {
            (TransformMap::Affine(l), TransformMap::Affine(c)) => {
                self.apply_affine(pool, base, selection, *l, *c, translation)
            }
            _ => self.apply_gated(pool, base, selection, &local, map, translation),
        }
    }

    /// The author's selection as it stands over a layer's frame placed at
    /// `translation` (§14.12): a mask whose value at a frame position `p` is the
    /// canvas mask's at `p + translation`, built only over the tiles in `within`
    /// — everywhere else it answers with `outside`, so it may gate exactly the
    /// tiles it was scoped to and nothing more.
    ///
    /// **Exact.** The frame is whole pixels, so every resample tap lands on a
    /// texel centre (§16.4's second exactness property, which the mask pass
    /// shares with the parcel). A zero translation and a universal mask are both
    /// the mask itself, by handle: no pass runs.
    pub(crate) fn shifted_selection(
        &self,
        pool: &TilePool,
        selection: &Selection,
        translation: stark_model::geom::IVec2,
        within: &[TileCoord],
    ) -> Selection {
        if translation == stark_model::geom::IVec2::ZERO || selection.is_universal() {
            return selection.clone();
        }
        let f = Vec2::new(translation.x as f32, translation.y as f32);
        // dest(p) = src(A⁻¹·p) is the pass's contract, and src is to be read at
        // p + translation — so the drawn map is the *negated* translation.
        let affine = Affine2::from_translation(-f);
        let rewrites: Vec<(TileCoord, Vec<TileCoord>)> = within
            .iter()
            .filter_map(|coord| {
                let sources = shifted_mask_sources(selection, *coord, translation);
                (!sources.is_empty()).then_some((*coord, sources))
            })
            .collect();
        let mut scope = self.scratch.scope(&self.ctx, "stark selection shift");
        let mut uniforms = Vec::with_capacity(slot_count(&rewrites));
        stage_quads(affine, &rewrites, 1.0, &mut uniforms);
        let mut quads = self.group(
            &mut scope,
            &self.quad_bindings,
            t::Q,
            "stark selection shift quads",
            &uniforms,
        );
        let mut tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> = HashTrieMap::new();
        for (coord, sources) in &rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut scope, selection, &mut quads, sources, &dst);
            tiles = tiles.insert(*coord, dst);
            scope.tile_done();
        }
        debug_assert!(quads.spent(), "the shift's staging and draw walks disagree");
        scope.finish();
        Selection::from_parts(
            tiles,
            selection.outside(),
            selection.level(),
            selection.opacity(),
            selection.hull().map(|(lo, hi)| (lo - f, hi - f)),
        )
    }

    /// [`Self::shifted_selection`], scoped by a tile rect instead of a coord
    /// list: the destinations are enumerated from the mask's own tiles — a
    /// stroke's gate stays the stroke's size however large the layer is, and a
    /// claim of the whole layer (`TileRect::ALL`) stays the mask's.
    pub(crate) fn shifted_selection_in(
        &self,
        pool: &TilePool,
        selection: &Selection,
        translation: stark_model::geom::IVec2,
        rect: stark_model::geom::TileRect,
    ) -> Selection {
        if translation == stark_model::geom::IVec2::ZERO || selection.is_universal() {
            return selection.clone();
        }
        let within = shifted_mask_cover(selection, translation, rect);
        self.shifted_selection(pool, selection, translation, &within)
    }

    /// Cut what `selection` holds of `base` into a second map — the float's GPU
    /// half (§16.12): `(what remains, what lifted)`. The two stack back to the
    /// picture that was there, §14.11.1's law run backwards: the cut is §16.2's
    /// lift, fully-covered tiles cross **by handle**, and only the boundary
    /// tiles pay a parcel and two combines. `None` rejects the action — nothing
    /// selected over the layer, or past the cap — deterministically.
    pub(crate) fn float(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        translation: stark_model::geom::IVec2,
    ) -> Option<(TileMap, TileMap)> {
        let plan = plan_float(base, selection, translation)?;
        // The mask, brought into the layer's frame once — every gate below reads it.
        let shifted = self.shifted_selection(pool, selection, translation, &plan.partial);

        let mut scope = self.scratch.scope(&self.ctx, "stark float");
        let mut from = Source::new(pool, base, &shifted);
        // The child's combines stack their parcel over nothing: an empty base
        // binds the 1×1 zeroes through the same door a virgin destination does.
        let empty = TileMap::new();
        let onto = Source::new(pool, &empty, &shifted);

        // Every group-0 uniform, staged before the first submit (`Slots`): the
        // identity quads of each partial tile's neighbourhood, and two combines per
        // partial tile — the cut and the lift, over the same uniform.
        let partial: Vec<(TileCoord, Vec<TileCoord>)> = plan
            .partial
            .iter()
            .map(|coord| (*coord, neighbours_in(base, *coord)))
            .collect();
        let mut quad_uniforms = Vec::with_capacity(slot_count(&partial));
        stage_quads(
            Affine2::IDENTITY,
            &partial,
            shifted.opacity(),
            &mut quad_uniforms,
        );
        let mut quads = self.group(
            &mut scope,
            &self.quad_bindings,
            t::Q,
            "stark float quads",
            &quad_uniforms,
        );
        let combine_uniforms: Vec<CombineUniform> = partial
            .iter()
            .flat_map(|(coord, _)| {
                let u = combine_uniform(*coord, None, shifted.opacity());
                [u, u]
            })
            .collect();
        let mut combines = Slots::stage(&mut scope, "stark float combines", &combine_uniforms);

        let mut remaining = base.clone();
        let mut lifted = TileMap::new();
        for coord in &plan.full {
            let handle = base.get(coord).expect("planned from this map").clone();
            lifted = lifted.insert(*coord, handle);
            remaining = remaining.remove(coord);
        }
        for (coord, sources) in &partial {
            let parcel = self.render_parcel_from(
                &mut scope,
                &mut from,
                &mut quads,
                sources.iter().copied(),
                &self.parcel_pipeline,
                "stark transform parcel",
            );
            // The cut: the base under its mask, no parcel stacked back (§16.2).
            let cut = Channels::acquire(pool, self.formats, AllocSource::TransformDestination);
            self.combine(&mut scope, &from, &mut combines, *coord, None, &cut);
            remaining = remaining.insert(*coord, cut.into_tile());
            // The lift: the parcel alone, over nothing.
            let up = Channels::acquire(pool, self.formats, AllocSource::TransformDestination);
            self.combine(
                &mut scope,
                &onto,
                &mut combines,
                *coord,
                parcel.as_ref(),
                &up,
            );
            lifted = lifted.insert(*coord, up.into_tile());
            scope.tile_done();
        }
        debug_assert!(
            quads.spent() && combines.spent(),
            "the float's staging and draw walks disagree"
        );
        scope.finish();
        Some((remaining, lifted))
    }

    /// The whole-plane affine (§16): one quad per selected source tile, pure
    /// Replace on the mask. `local` is the map in the layer's frame — the paint
    /// side — and `affine` the canvas map the mask moves under; at a zero
    /// translation the two are the same value.
    fn apply_affine(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        local: Affine2,
        affine: Affine2,
        translation: stark_model::geom::IVec2,
    ) -> Option<(TileMap, Selection)> {
        // The mask as it stands over the layer's frame — what classifies and
        // gates every paint tile. The canvas mask stays what *moves*, below.
        let gate = self.shifted_selection_in(pool, selection, translation, base_rect(base));
        let plan = plan_paint(base, &gate, local)?;
        let mask_plan = plan_mask(selection, affine)?;

        let mut scope = self.scratch.scope(&self.ctx, "stark transform");

        // Source-tile bind groups are shared across every destination they reach.
        let mut from = Source::new(pool, base, &gate);

        // Every group-0 uniform, staged before the first submit (`Slots`): the paint
        // quads, then the mask quads — one buffer, since both passes bind the quad
        // group — and one combine per destination.
        let mut quad_uniforms =
            Vec::with_capacity(slot_count(&plan.rewrites) + slot_count(&mask_plan.rewrites));
        stage_quads(local, &plan.rewrites, gate.opacity(), &mut quad_uniforms);
        stage_quads(affine, &mask_plan.rewrites, 1.0, &mut quad_uniforms);
        let mut quads = self.group(
            &mut scope,
            &self.quad_bindings,
            t::Q,
            "stark transform quads",
            &quad_uniforms,
        );
        let combine_uniforms = stage_combines(&plan.rewrites, None, gate.opacity());
        let mut combines = Slots::stage(&mut scope, "stark transform combines", &combine_uniforms);

        let tiles = self.rewrite_paint(
            &mut scope,
            &mut from,
            &mut combines,
            &plan.rewrites,
            &plan.drops,
            |scope, from, sources| {
                self.render_parcel_from(
                    scope,
                    from,
                    &mut quads,
                    sources.iter().copied(),
                    &self.parcel_pipeline,
                    "stark transform parcel",
                )
            },
        );

        // The mask, carried under the canvas affine (pure Replace — §16.1).
        let mut mask_tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> =
            HashTrieMap::new();
        for (dest, sources) in &mask_plan.rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut scope, selection, &mut quads, sources, &dst);
            mask_tiles = mask_tiles.insert(*dest, dst);
            scope.tile_done();
        }
        debug_assert!(
            quads.spent() && combines.spent(),
            "the transform's staging and draw walks disagree"
        );
        // The hull rides along: the AABB of the affine image of its corners.
        let hull = selection.hull().map(|(lo, hi)| {
            let corners = [
                affine.transform_point2(lo),
                affine.transform_point2(Vec2::new(hi.x, lo.y)),
                affine.transform_point2(Vec2::new(lo.x, hi.y)),
                affine.transform_point2(hi),
            ];
            let min = corners.iter().fold(corners[0], |a, p| a.min(*p));
            let max = corners.iter().fold(corners[0], |a, p| a.max(*p));
            (min, max)
        });
        // A transform *moves* coverage; it does not restrike it, so the opacity
        // the selection is read at rides through unchanged (§6.8).
        let moved_selection = Selection::from_parts(
            mask_tiles,
            selection.outside(),
            selection.level(),
            selection.opacity(),
            hull,
        );

        scope.finish();
        Some((tiles, moved_selection))
    }

    /// The rect-scoped maps (§16.8, §16.9): pieces of `tile ∩ rect` (further
    /// split per warp sub-cell) forward-rasterized through `vs_gated`, the cut
    /// gated by the rect's coverage, and the mask carried as
    /// `max(old · (1 − box), moved)` — the residue unioned with what landed.
    /// `local` is `canvas` restated in the layer's frame
    /// ([`TransformMap::under_translation`]);
    /// the paint runs under it, the mask under `canvas`, and at a zero translation the
    /// two are the same value.
    fn apply_gated(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        local: &TransformMap,
        canvas: &TransformMap,
        translation: stark_model::geom::IVec2,
    ) -> Option<(TileMap, Selection)> {
        let (lrect, lgeo) = gated_geometry(local)?;
        let (rect, geo) = gated_geometry(canvas)?;
        // The mask over the layer's frame, for the cut; the canvas mask moves.
        let gate = self.shifted_selection_in(pool, selection, translation, base_rect(base));
        let plan = plan_gated_paint(base, &gate, lrect, &lgeo)?;
        let mask_plan = plan_gated_mask(selection, rect, &geo)?;

        let mut scope = self.scratch.scope(&self.ctx, "stark transform gated");

        let mut from = Source::new(pool, base, &gate);
        let paint = Gated::new(&plan.units, lrect, &lgeo);
        let mask = Gated::new(&mask_plan.units, rect, &geo);

        // Every group-0 uniform, staged before the first submit (`Slots`): the paint
        // pieces, then the mask's residues and pieces — one buffer, since all three
        // passes bind the gated group — and one combine per destination. A universal
        // selection plans no mask work and draws none (below).
        let mut piece_uniforms = Vec::with_capacity(
            slot_count(&plan.rewrites) + mask_plan.rewrites.len() + slot_count(&mask_plan.rewrites),
        );
        stage_gated_parcels(&paint, &plan.rewrites, gate.opacity(), &mut piece_uniforms);
        if !selection.is_universal() {
            stage_gated_masks(&mask, &mask_plan.rewrites, &mut piece_uniforms);
        }
        let mut pieces = self.group(
            &mut scope,
            &self.gated_bindings,
            t::QG,
            "stark transform pieces",
            &piece_uniforms,
        );
        let combine_uniforms = stage_combines(&plan.rewrites, Some(lrect), gate.opacity());
        let mut combines = Slots::stage(
            &mut scope,
            "stark transform gated combines",
            &combine_uniforms,
        );

        let tiles = self.rewrite_paint(
            &mut scope,
            &mut from,
            &mut combines,
            &plan.rewrites,
            &plan.drops,
            |scope, from, unit_idxs| {
                self.render_parcel_from(
                    scope,
                    from,
                    &mut pieces,
                    unit_idxs.iter().map(|i| paint.units[*i].src),
                    &self.parcel_gated_pipeline,
                    "stark transform parcel gated",
                )
            },
        );

        // The mask: tiles outside the rect keep their handles; touched and
        // receiving tiles are rebuilt as residue + moved coverage. A universal
        // selection plans no mask work at all and stays universal.
        let moved_selection = if selection.is_universal() {
            selection.clone()
        } else {
            let mut mask_tiles = selection.tile_map().clone();
            for coord in &mask_plan.drops {
                mask_tiles = mask_tiles.remove(coord);
            }
            for rewrite in &mask_plan.rewrites {
                let dst = pool.acquire_mask(AllocSource::TransformMask);
                self.render_gated_mask(&mut scope, selection, &mask, &mut pieces, rewrite, &dst);
                mask_tiles = mask_tiles.insert(rewrite.0, dst);
                scope.tile_done();
            }
            // The hull rides along: what stayed plus wherever the map can have
            // carried coverage, conservatively.
            let hull = selection
                .hull()
                .map(|(lo, hi)| (lo.min(geo.image_aabb.0), hi.max(geo.image_aabb.1)));
            Selection::from_parts(
                mask_tiles,
                selection.outside(),
                selection.level(),
                selection.opacity(),
                hull,
            )
        };
        debug_assert!(
            pieces.spent() && combines.spent(),
            "the gated transform's staging and draw walks disagree"
        );

        scope.finish();
        Some((tiles, moved_selection))
    }

    /// The group-0 slots of one pass family for this `apply`, and the one bind group
    /// its draws share: `uniforms` at `at` (`Q` or `QG`), the sampler beside it
    /// ([`Slots`]).
    fn group<'r, T: bytemuck::Pod>(
        &self,
        scope: &mut SubmitScope,
        bindings: &'r desc::Bindings,
        at: u32,
        label: &'static str,
        uniforms: &[T],
    ) -> Group<'r, T> {
        let slots = Slots::stage(scope, label, uniforms);
        let bg = slots.staged.as_ref().map(|staged| {
            bindings.group(&self.ctx.device, label, |i| match i {
                i if i == at => staged.window(),
                t::SAMP => wgpu::BindingResource::Sampler(&self.sampler),
                other => unreachable!("neither quad group holds binding {other}"),
            })
        });
        Group {
            slots,
            bg,
            bindings,
        }
    }

    /// The paint half of an apply: each destination `rewrites` names gets the parcel
    /// `render` draws from its sources, cut and stacked over its base by the combine
    /// into a fresh tile; then `drops` leave the map.
    fn rewrite_paint<S>(
        &self,
        scope: &mut SubmitScope,
        from: &mut Source<'_>,
        combines: &mut Slots<CombineUniform>,
        rewrites: &[(TileCoord, Vec<S>)],
        drops: &[TileCoord],
        mut render: impl FnMut(&mut SubmitScope, &mut Source<'_>, &[S]) -> Option<Parcel>,
    ) -> TileMap {
        let mut tiles = from.base.clone();
        for (dest, sources) in rewrites {
            let parcel = render(scope, from, sources);
            let dst = Channels::acquire(from.pool, self.formats, AllocSource::TransformDestination);
            self.combine(scope, from, combines, *dest, parcel.as_ref(), &dst);
            tiles = tiles.insert(*dest, dst.into_tile());
            scope.tile_done();
        }
        for coord in drops {
            tiles = tiles.remove(coord);
        }
        tiles
    }

    /// Rasterize the transformed sources reaching one destination into a fresh scratch
    /// pair through `pipeline`: `(premult color as-is, height·mask)` — the moved parcel.
    /// `None` when nothing reaches the tile (a cut with no incoming paint).
    ///
    /// `srcs` is one source tile per draw, in the order `group`'s slots were staged:
    /// the affine's whole-tile quads, or a gated map's pieces, whose deposit its
    /// uniform gates by the source rect.
    fn render_parcel_from<T: bytemuck::Pod>(
        &self,
        scope: &mut SubmitScope,
        from: &mut Source<'_>,
        group: &mut Group<'_, T>,
        srcs: impl ExactSizeIterator<Item = TileCoord>,
        pipeline: &wgpu::RenderPipeline,
        label: &str,
    ) -> Option<Parcel> {
        if srcs.len() == 0 {
            return None;
        }
        // The parcel carries the residual it was cut with: the lift scales height
        // alone, so the color — both halves of it — rides through unscaled (§16.2).
        let parcel = Channels::scratch(
            scope,
            from.pool,
            self.formats,
            AllocSource::TransformScratch,
        );

        let mut draws: Vec<Draw> = Vec::with_capacity(srcs.len());
        for src in srcs {
            // Taken before the skip: the staging walk counted this source (`Slots`).
            let offset = group.take();
            let Some(tile) = from.base.get(&src) else {
                continue;
            };
            let src_bg = from
                .src_bgs
                .entry(src)
                .or_insert_with(|| {
                    // A gating read — the opacity reaches the fragment stage through
                    // the draw's own uniform (`pack`).
                    let mask = self.selection.mask_for(from.selection, src);
                    self.src_bg(tile, &mask)
                })
                .clone();
            draws.push((offset, src_bg));
        }

        let mut pass = desc::begin_pass(scope.encoder(), label, parcel.targets());
        pass.set_pipeline(pipeline);
        group.draw_quads(&mut pass, &self.src_bindings, &draws);
        drop(pass);
        Some(parcel)
    }

    /// One destination mask tile under a rect-scoped map: the residue
    /// `old · (1 − box)` laid down fullscreen, then the moved coverage pieces
    /// drawn over with max blending — the soft union (§16.8).
    fn render_gated_mask(
        &self,
        scope: &mut SubmitScope,
        selection: &Selection,
        g: &Gated<'_>,
        pieces: &mut Group<'_, GatedUniform>,
        rewrite: &(TileCoord, Vec<usize>),
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let (dest, unit_idxs) = (rewrite.0, &rewrite.1);
        // The residue reads the destination's *old* coverage — a real tile or
        // the outside constant, through the same clamped-read pattern. Its slot
        // was staged first (`stage_gated_masks`), then the pieces' in order.
        let old = self.selection.mask_for(selection, dest);
        let base_draw = (pieces.take(), self.mask_src_bg(old.view()));
        let mut draws: Vec<Draw> = Vec::with_capacity(unit_idxs.len());
        for idx in unit_idxs {
            let unit = &g.units[*idx];
            let src = self.selection.mask_for(selection, unit.src);
            draws.push((pieces.take(), self.mask_src_bg(src.view())));
        }

        let mut pass = desc::begin_pass(scope.encoder(), "stark transform mask gated", dst.view());
        pass.set_pipeline(&self.mask_base_pipeline);
        pieces.set(&mut pass, base_draw.0);
        self.mask_src_bindings.set(&mut pass, &base_draw.1, &[]);
        pass.draw(desc::FULLSCREEN_VERTICES, 0..1);
        pass.set_pipeline(&self.mask_gated_pipeline);
        pieces.draw_quads(&mut pass, &self.mask_src_bindings, &draws);
    }

    /// The group-1 bind for one source tile: its channels and the mask over it, all
    /// sampled under the map. Shared across every destination the
    /// tile's image reaches (`Source::src_bgs`).
    fn src_bg(&self, tile: &TilePairHandle, mask: &MaskSource) -> wgpu::BindGroup {
        self.src_bindings
            .group(&self.ctx.device, "stark transform src bg", |i| {
                wgpu::BindingResource::TextureView(match i {
                    t::SRC_COLOR => tile.color_view(),
                    t::SRC_AUX => tile.aux_view(),
                    t::SRC_MASK => mask.view(),
                    t::SRC_RESID => tile.resid_view().expect("a residual space's tile has one"),
                    other => unreachable!("the transform's source group holds no binding {other}"),
                })
            })
    }

    /// The group-1 bind for a mask pass: the one source mask it carries
    /// (the same group 1, one slot of it).
    fn mask_src_bg(&self, view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.mask_src_bindings
            .group(&self.ctx.device, "stark transform mask src bg", |_| {
                wgpu::BindingResource::TextureView(view)
            })
    }

    /// Cut `dest`'s base by its own mask and stack the parcel over it, into the
    /// fresh CoW `(color, aux)` pair `dst`. Whether the cut is scoped to a source
    /// rect (§16.8) or is the affine's whole-plane one rides in the staged uniform
    /// ([`combine_uniform`]); `combines` hands this tile its slot.
    fn combine(
        &self,
        scope: &mut SubmitScope,
        from: &Source<'_>,
        combines: &mut Slots<CombineUniform>,
        dest: TileCoord,
        parcel: Option<&Parcel>,
        dst: &Parcel,
    ) {
        let device = &self.ctx.device;
        let offset = combines.take();
        // A virgin destination and a cut-only tile read the 1×1 zeroes — all three at
        // once, through the one answer to that question (`Zeroes::or`), so the combine
        // is one shader whatever exists (§6.8's pattern).
        let under = self
            .zeroes
            .or(from.base.get(&dest).map(TilePairHandle::targets));
        let carried = self.zeroes.or(parcel.map(Channels::targets));
        // A gating read: the base is cut by `coverage · opacity`, the identical
        // factor the parcel side took at the source (`pack::combine_uniform`).
        let base_mask = self.selection.mask_for(from.selection, dest);
        let bg = self
            .combine
            .bindings
            .group(device, "stark transform combine bg", |i| match i {
                t::BASE_COLOR => wgpu::BindingResource::TextureView(under.color),
                t::BASE_AUX => wgpu::BindingResource::TextureView(under.aux),
                t::BASE_MASK => wgpu::BindingResource::TextureView(base_mask.view()),
                t::PARCEL_COLOR => wgpu::BindingResource::TextureView(carried.color),
                t::PARCEL_AUX => wgpu::BindingResource::TextureView(carried.aux),
                t::QC => combines.window(),
                t::BASE_RESID => wgpu::BindingResource::TextureView(
                    under.resid.expect("a residual build has one"),
                ),
                t::PARCEL_RESID => wgpu::BindingResource::TextureView(
                    carried.resid.expect("a residual build has one"),
                ),
                other => unreachable!("the combine's group holds no binding {other}"),
            });
        scope.fullscreen_pass(
            "stark transform combine",
            dst.targets(),
            &self.combine.pipeline,
            desc::Bound::dynamic(&bg, &[offset]),
        );
    }

    /// One destination mask tile: cleared to the coverage that reigns outside the
    /// mask's tiles, with the transformed source mask quads drawn over.
    fn render_mask(
        &self,
        scope: &mut SubmitScope,
        selection: &Selection,
        quads: &mut Group<'_, QuadUniform>,
        sources: &[TileCoord],
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let mut draws: Vec<Draw> = Vec::with_capacity(sources.len());
        for src in sources {
            // Taken before the skip: the staging walk counted this source (`Slots`).
            // Its uniform was staged at 1.0: this pass carries the mask, it does not
            // gate by it — the opacity rides on the moved `Selection` (§6.8).
            let offset = quads.take();
            let Some(handle) = selection.tile(*src) else {
                continue;
            };
            draws.push((offset, self.mask_src_bg(handle.view())));
        }

        let mut pass = desc::begin_pass(
            scope.encoder(),
            "stark transform mask",
            desc::Attachments::cleared_to(dst.view(), outside_clear(selection)),
        );
        pass.set_pipeline(&self.mask_pipeline);
        quads.draw_quads(&mut pass, &self.mask_src_bindings, &draws);
    }
}
