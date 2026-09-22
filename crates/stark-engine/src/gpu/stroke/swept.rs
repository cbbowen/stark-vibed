//! The swept fast path (§6.2): one quad per segment, coverage integrated along the
//! sweep through a precomputed prefix-τ texture, over-blended so optical depth sums
//! exactly. Carries no brush state between segments — a range needs nothing from its
//! predecessor but the arc length.

use std::ops::Range;

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::desc;
use stark_model::document::StrokeRecord;
use stark_model::geom::{TILE_APRON, TILE_TEX, TileCoord, Vec2};
use stark_shaders::mirror::integrate::binding as ib;
use stark_shaders::mirror::integrate::decl as id;
use stark_shaders::mirror::stamp_common::binding as sc;
use stark_shaders::mirror::stamp_common::decl as sd;
use stark_shaders::{EntryPoint, Lane, Stages};

use crate::gpu::tile::{SCRATCH_AUX_FORMAT, TileMap};

use super::accum::{
    LandStep, Landing, LaneKeys, Lanes, ParcelEffect, SweepStep, land_tiles, lane_key,
};
use super::region::tiles_with_segments;
use super::segments::{Segment, SegmentInstance};
use super::{ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene};
use crate::gpu::scratch::{BufKey, Key, LeasedSlots, SubmitScope};
use crate::gpu::substrate::SubstrateMap;

// A segment's centreline is an arc rather than a chord (§6.2), so its swept geometry is
// a triangle strip of two rims across `SWEEP_SLICES` steps of travel. Both numbers are
// `stamp_common.wesl`'s, which is where the strip is built.
use stark_shaders::mirror::stamp_common::{SWEEP_SLICES, SWEEP_VERTS};

/// The draw call and the strip agree on the vertex count.
const _: () = assert!(
    SWEEP_VERTS == 2 * (SWEEP_SLICES + 1),
    "the sweep strip's slice count and its vertex count have diverged",
);

// The per-tile uniform, generated from `stamp_common.wesl`'s own declaration
// (§6.7): the tile *texture's* top-left in canvas px + canvas→NDC scale, plus the
// brush's stroke-constant color channels.
use super::tips::ResolvedTip;
use stark_shaders::mirror::integrate::Integrate;
use stark_shaders::mirror::stamp_common::TileXform;

/// A pipeline over the sweep's three layouts (§6.2, §6.12) — its row of [`SWEEPS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SweepPass {
    /// The deposit: one instanced quad strip per segment, over-blended into the
    /// stroke's scratch parcel.
    Deposit,
    /// The deposit with the **ceiling lane** as a fourth target, for a stroke whose
    /// opacity the pen drives. A pipeline of its own because a target list is fixed at
    /// pipeline creation, and an artifact of its own because the shader's output is
    /// `@if(ceiling)`-gated, so no other stroke pays the lane.
    DepositCeiling,
    /// The ceiling lane **alone** (`fs_levels`): what the stamp loop draws per painting
    /// segment into its region, so its claim advances by the sweep's own sums.
    Levels,
    /// The erase sweep (`fs_erase`): the stroke's transparency mass.
    Erase,
    /// The erase sweep with the ceiling lane and the mass's moment beside it.
    EraseCeiling,
}

impl SweepPass {
    /// The deposit, with the ceiling lane if `lane` is the ceiling build.
    pub(super) fn deposit(lane: Lane) -> Self {
        match lane {
            Lane::Plain => Self::Deposit,
            Lane::Ceiling => Self::DepositCeiling,
        }
    }

    /// The erase sweep, with the ceiling lane if `lane` is the ceiling build.
    pub(super) fn erase(lane: Lane) -> Self {
        match lane {
            Lane::Plain => Self::Erase,
            Lane::Ceiling => Self::EraseCeiling,
        }
    }
}

/// One pipeline over the sweep's layouts: which build of `stamp.wesl` it runs, which of
/// that build's fragment entry points (the vertex stage is always `vs_main`), and the
/// color targets at that entry point's `@location`s.
struct SweepRow {
    pass: SweepPass,
    label: &'static str,
    lane: Lane,
    fs: fn(&stark_shaders::Stamp) -> EntryPoint,
    targets: fn(&ChannelFormats) -> Vec<Option<wgpu::ColorTargetState>>,
}

impl SweepRow {
    /// The row's vertex and fragment entry points, out of `color_space`'s build of
    /// the stamp shader (§6.10).
    fn entries(&self, color_space: &dyn ColorSpace) -> (EntryPoint, EntryPoint) {
        let shader = color_space.stamp_shader(self.lane);
        (shader.vs_main, (self.fs)(shader))
    }
}

/// Every pipeline the sweep's layouts serve, in [`SweepPass`] order: the one list the
/// layouts are derived over and the pipelines are built from.
const SWEEPS: [SweepRow; 5] = [
    SweepRow {
        pass: SweepPass::Deposit,
        label: "stark sweep pipeline",
        lane: Lane::Plain,
        fs: |s| s.fs_main,
        targets: |f| deposit_targets(f, Lane::Plain),
    },
    SweepRow {
        pass: SweepPass::DepositCeiling,
        label: "stark sweep ceiling pipeline",
        lane: Lane::Ceiling,
        fs: |s| s.fs_main,
        targets: |f| deposit_targets(f, Lane::Ceiling),
    },
    // Over the plain build: `fs_levels` is not `@if(ceiling)`-gated, since it writes
    // nothing but the lane.
    SweepRow {
        pass: SweepPass::Levels,
        label: "stark sweep levels pipeline",
        lane: Lane::Plain,
        fs: |s| s.fs_levels,
        targets: |_| vec![ceiling_target()],
    },
    SweepRow {
        pass: SweepPass::Erase,
        label: "stark erase sweep pipeline",
        lane: Lane::Plain,
        fs: |s| s.fs_erase,
        targets: |_| super::erase::sweep_targets(Lane::Plain),
    },
    SweepRow {
        pass: SweepPass::EraseCeiling,
        label: "stark erase sweep ceiling pipeline",
        lane: Lane::Ceiling,
        fs: |s| s.fs_erase,
        targets: |_| super::erase::sweep_targets(Lane::Ceiling),
    },
];

const _: () = {
    let mut i = 0;
    while i < SWEEPS.len() {
        assert!(
            SWEEPS[i].pass as usize == i,
            "`SWEEPS` is out of `SweepPass` order"
        );
        i += 1;
    }
};

/// The deposit's targets (§6.1, §6.7): the channel trio as a deposit lands it, into a
/// *scratch* parcel whose aux is the wide [`SCRATCH_AUX_FORMAT`]. The ceiling lane sits
/// at location 3 whether or not the space has a residual at 2, so the plain list stops
/// short of it and the ceiling list carries the hole (`accum::MAX_LANES`).
fn deposit_targets(formats: &ChannelFormats, lane: Lane) -> Vec<Option<wgpu::ColorTargetState>> {
    let mut targets = formats.deposit_targets(SCRATCH_AUX_FORMAT).to_vec();
    match lane {
        Lane::Plain => targets.truncate(formats.count()),
        Lane::Ceiling => targets.push(ceiling_target()),
    }
    targets
}

/// The pipelines that rasterize a stroke's swept extent ([`SWEEPS`]), built once, and
/// the three layouts they share: the per-tile transform at group 0, the prefix-τ volume
/// at group 1, the noise and substrate fields at group 2. One set of layouts, so one set
/// of bind groups serves every pipeline ([`SweepDraws`]).
///
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
#[derive(Clone)]
pub(super) struct SweepKit {
    pub(super) uniform_bgl: desc::Bindings,
    pub(super) prefix_bgl: desc::Bindings,
    pub(super) noise_bgl: desc::Bindings,
    pipelines: [wgpu::RenderPipeline; SWEEPS.len()],
}

impl SweepKit {
    pub(super) fn new(device: &wgpu::Device, color_space: &dyn ColorSpace) -> Self {
        // Each layout is the union over every row (§6.10). Group 0 is one buffer for the
        // whole piece with a slot per tile, selected by a dynamic offset; group 1 the
        // prefix-τ volume (§6.6); group 2 the noise field with its repeat sampler beside
        // the canvas substrate's map (§6.2, §6.4).
        let over = SWEEPS.map(|row| {
            let (vs, fs) = row.entries(color_space);
            Stages::Render(vs, fs)
        });
        let uniform_bgl =
            desc::Bindings::shared_by(device, "stark sweep uniform bgl", &over, sd::XF, &[sd::XF]);
        let prefix_bgl =
            desc::Bindings::shared_by(device, "stark sweep prefix bgl", &over, sd::PREFIX_TEX, &[]);
        let noise_bgl =
            desc::Bindings::shared_by(device, "stark sweep noise bgl", &over, sd::NOISE_TEX, &[]);
        let layout = desc::pipeline_layout_of(
            device,
            "stark sweep layout",
            &[&uniform_bgl, &prefix_bgl, &noise_bgl],
        );

        // One module per build, shared by the rows over it: a second
        // `create_shader_module` is a second parse and translation of source already in
        // hand — on the web, startup the artist waits through.
        let plain = stamp_module(device, color_space, Lane::Plain);
        let ceiling = stamp_module(device, color_space, Lane::Ceiling);
        let formats = ChannelFormats::of(color_space);
        let pipelines = SWEEPS.map(|row| {
            let (vs, fs) = row.entries(color_space);
            desc::render_pipeline(
                device,
                desc::RenderPipe {
                    label: row.label,
                    layout: &layout,
                    module: match row.lane {
                        Lane::Plain => &plain,
                        Lane::Ceiling => &ceiling,
                    },
                    vs,
                    fs,
                    primitive: desc::QUAD_STRIP,
                    buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
                        wgpu::VertexStepMode::Instance,
                    ))],
                    targets: &(row.targets)(&formats),
                },
            )
        });
        Self {
            uniform_bgl,
            prefix_bgl,
            noise_bgl,
            pipelines,
        }
    }

    pub(super) fn pipeline(&self, pass: SweepPass) -> &wgpu::RenderPipeline {
        &self.pipelines[pass as usize]
    }
}

/// Compile `stamp.wesl` for `color_space` (§6.2, §6.7) — with or without the ceiling
/// lane, which is the axis the caller picks and the residual the space does.
fn stamp_module(device: &wgpu::Device, color_space: &dyn ColorSpace, lane: Lane) -> desc::Module {
    let label = match lane {
        Lane::Ceiling => "stark stamp ceiling",
        Lane::Plain => "stark stamp",
    };
    desc::Module::new(device, label, color_space.stamp_shader(lane))
}

/// The format of the ceiling lane (§6.2, `paint_common::level_sums`): the parcel's
/// mass above each of two ceiling levels and its moment above each — four sums,
/// f16 for the reason the aux's own mass is: the interesting range is a few times
/// `OPAQUE_MASS`, and past f16's mantissa the coverage they decide is 1 anyway.
pub(super) const CEILING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The ceiling lane as a sweep target: **additive**, like the aux, because each of its
/// four channels is a sum over segments. That is why the law is stated in gated sums
/// (`paint_common::claimed_coverage`) rather than as a max — fixed-function blending
/// has no max of a running quantity, but the mass above a level is a sum, and the
/// point-wise max of the ceilings falls out of two of them and their moments. The erase
/// sweep attaches the same lane by the same rule (§6.12), as does the stamp loop's
/// per-segment draw of it (`fs_levels`).
pub(super) fn ceiling_target() -> Option<wgpu::ColorTargetState> {
    desc::blended_target(CEILING_FORMAT, Some(wgpu::BlendState::ADDITIVE))
}

/// A landing pass (§6.2, §6.12): the fullscreen pass that applies a stroke's parcel to
/// the paint under one tile. It computes the finished texel, so no target blends, into
/// the space's own channels (§6.7).
pub(super) fn landing_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    label: &str,
    shader: &dyn stark_shaders::Artifact,
    entries: (EntryPoint, EntryPoint),
    anchor: stark_shaders::Binding,
) -> desc::OnePass {
    let module = desc::Module::new(device, label, shader);
    let targets = ChannelFormats::of(color_space).targets();
    desc::OnePass::fullscreen(device, label, &module, entries, anchor, &[], &targets)
}

/// The integrate (`integrate.wesl`, §6.2/§6.1): the landing pass that stacks a stroke's
/// parcel over the base tile — the layer's paint, the parcel, the selection each is
/// gated by, the effect's opacity and the ceiling lane in; `new = f(base, parcel)` out,
/// through the shared law in `paint_common.wesl` that a fill and the stamp loop's
/// `deposit` also land by.
pub(super) fn integrate_pass(device: &wgpu::Device, color_space: &dyn ColorSpace) -> desc::OnePass {
    let shader = stark_shaders::integrate(color_space.resid());
    landing_pipeline(
        device,
        color_space,
        "stark integrate",
        shader,
        (shader.vs_main, shader.fs_main),
        id::BASE_COLOR,
    )
}

impl StrokeRenderer {
    /// [`Self::render_range`] through the swept deposit. The segments and constants
    /// arrive already resolved and flattened ([`ResolvedRange`]), and every tile lands
    /// through [`land_tiles`].
    pub(super) fn render_swept(
        &self,
        scene: StrokeScene<'_>,
        range: ResolvedRange<'_>,
    ) -> (TileMap, StrokeCarry) {
        // The control every dynamics row is read against: the same geometry, the
        // same tiles, one instanced draw instead of a dispatch chain per segment.
        // A change that moves this row has moved something shared — tile
        // acquisition, the scope — rather than the loop.
        crate::timing::span!("stroke.swept");
        let StrokeScene {
            base, substrate, ..
        } = scene;
        let ResolvedRange {
            rec,
            consts: k,
            end_dist,
            resume,
            ..
        } = range;

        // Uncapped, the parcel lands by one of §6.2's piece-composable forms, so a range
        // needs nothing from its predecessor but the arc length. Under a ceiling
        // (`StrokeConstants::capped`) its finished coverage is scaled, which is neither
        // — so the path takes the erase pass's shape instead: the parcel accumulates
        // across pieces, every piece re-derived from pristine paint under the whole of
        // it (`accum::ParcelCarry`), or a stroke crossing itself would outrun the dial
        // or the mask.
        //
        // A supersampled stroke (`budget::supersample_scale`) needs the same shape one
        // level up: the resolve averages the parcel's *finished* visible alpha, and an
        // average of per-piece parcels differs from the whole parcel averaged by the
        // covariance of the subsample alphas — worth a quarter of the alpha at a rim
        // texel where pieces meet.
        let ss = super::budget::supersample_scale(&rec.brush);
        let carries = k.capped || ss > 1;

        // The per-stroke buffers and the working parcels ride in the scope, and only
        // the `finish` that submits the commands naming them can release them
        // (`scratch::SubmitScope`).
        let mut scope = self.scratch.scope(&self.ctx, "stark swept stroke");
        let Some(draws) = SweepDraws::new(self, &mut scope, &range, substrate, ss) else {
            return reaches_nothing(scope, base, end_dist);
        };
        // Exactly 1 where nothing is carried: the integrate's identity branch, which
        // the layout binds either way.
        let opacity = opacity_uniform(
            &mut scope,
            if carries { k.opacity } else { 1.0 },
            ss,
            k.ceiling_lane,
        );

        // The parcel's lanes at `stamp.wesl`'s own targets ([`parcel_keys`]), from the
        // scratch pool rather than the tile pool, since nothing here ever becomes a
        // document tile. A supersampled stroke holds them at `ss` texels per px.
        let formats = ChannelFormats::of(self.color_space.as_ref());
        let keys = parcel_keys(formats.color, formats.resid, k.ceiling_lane)
            .map(|key| key.map(|key| key.scaled(ss)));
        let lanes = if carries {
            Lanes::Resume(resume)
        } else {
            Lanes::Fresh
        };
        land_tiles(
            self,
            scene,
            scope,
            lanes,
            &SweepStep {
                label: "stark sweep pass",
                pipeline: self.swept.pipeline(SweepPass::deposit(k.lane())),
                draws: &draws,
                keys,
            },
            &LandStep {
                label: "stark integrate",
                pipeline: &self.integrate.pipeline,
                effect: ParcelEffect::Deposit,
            },
            |l: &Landing<'_>| {
                let parcel = Targets {
                    color: l.parcel.lane(COLOR),
                    aux: l.parcel.lane(AUX),
                    resid: l.base.resid.is_some().then(|| l.parcel.lane(RESID)),
                };
                let ceiling = if k.ceiling_lane {
                    l.parcel.lane(CEILING)
                } else {
                    &self.zeroes.aux
                };
                integrate_bind_group(self, l.base, parcel, l.mask, &opacity, ceiling)
            },
        )
        .finished(end_dist)
    }
}

/// What a swept path returns for a range whose segments reach no tile: nothing drawn,
/// exactly as for a range with no segments (`render_range`).
pub(super) fn reaches_nothing(
    scope: SubmitScope,
    base: &TileMap,
    dist: f32,
) -> (TileMap, StrokeCarry) {
    scope.finish();
    (base.clone(), StrokeCarry::unchanged(dist))
}

/// The deposit parcel's lanes: the index is the shader's own `@location` in
/// `stamp.wesl`, so the ceiling is at 3 with or without a residual at 2. [`parcel_keys`]
/// fills them by these names, which keeps the attach order and the bind order one list
/// ([`ParcelLanes`](super::accum::ParcelLanes)).
const COLOR: usize = 0;
const AUX: usize = 1;
const RESID: usize = 2;
const CEILING: usize = 3;

/// The lanes' pool keys at the lanes' own indices — `None` at one this stroke does
/// not carry. A space with no residual under a pen-driven ceiling leaves [`RESID`]
/// a hole between two lanes it does carry (§6.7, §6.2), and the parcel attaches
/// exactly that.
fn parcel_keys(
    color: wgpu::TextureFormat,
    resid: Option<wgpu::TextureFormat>,
    ceiling_lane: bool,
) -> LaneKeys {
    let mut keys = LaneKeys::default();
    keys[COLOR] = Some(parcel_key(color));
    keys[AUX] = Some(parcel_key(SCRATCH_AUX_FORMAT));
    keys[RESID] = resid.map(parcel_key);
    keys[CEILING] = ceiling_lane.then(|| parcel_key(CEILING_FORMAT));
    keys
}

/// One parcel lane's pool key — [`lane_key`]'s usages, at the parcel's own formats.
/// One label for the lanes: they are one purpose, and the formats already give each
/// its own line in the pool's free list (`scratch::Key`).
fn parcel_key(format: wgpu::TextureFormat) -> Key {
    lane_key(format, "stark sweep parcel")
}

/// The brush as the sweep binds it: the prefix-τ volume at group 1, and the noise and
/// canvas substrate fields at group 2. [`SweepDraws`] holds its own; this is for a draw
/// that binds its own group 0 (the stamp loop's ceiling lane, §6.2).
pub(super) fn sweep_brush_groups(
    r: &StrokeRenderer,
    scope: &mut SubmitScope,
    tip: &ResolvedTip,
    rec: &StrokeRecord,
    substrate: &SubstrateMap,
    k: &super::StrokeConstants,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let device = &r.ctx.device;
    // The tip `render_range`'s own gate resolved: an image brush's from the asset
    // store, the round tip's generated from its hardness.
    let prefix = r
        .swept
        .prefix_bgl
        .group(device, "stark sweep prefix bg", |_| {
            wgpu::BindingResource::TextureView(&tip.prefix)
        });
    // Color dynamics (§6.2): the noise tile baked for this stroke, of the brush's
    // kind. An inactive brush binds the zero tile with zero amplitudes, so the deposit
    // is exactly the constant color. The canvas substrate beside it (§6.4): the
    // deposition tooth's height and the rise ahead of it.
    let noise = r.tips.noise(&rec.brush.color_dynamics(), k.noise_seed);
    scope.hold(noise.clone());
    let noise = r
        .swept
        .noise_bgl
        .group(device, "stark sweep noise bg", |b| match b {
            sc::NOISE_TEX => wgpu::BindingResource::TextureView(noise.view()),
            sc::NOISE_SAMP => wgpu::BindingResource::Sampler(&r.tips.noise_sampler),
            sc::SUBSTRATE_TEX => wgpu::BindingResource::TextureView(&substrate.view),
            other => unreachable!("the sweep's noise group holds no binding {other}"),
        });
    (prefix, noise)
}

/// The integrate's opacity uniform (`integrate.wesl::Integrate`) for one piece:
/// the stroke's ceiling — the effect's dial with the mask's opacity folded in
/// (`stroke_constants`) — or exactly 1, the shader's identity branch, where nothing
/// is carried.
///
/// `ss` is how many subsample texels per canvas px the scratch holds (§6.2), which
/// tells the shader whether to box-resolve the parcel — 1 is the plain 1:1 load.
/// `lane` says whether the ceiling lane bound beside the parcel is real or the 1×1
/// zero the shader must not read.
///
/// Run tier: every tile of the piece binds it.
fn opacity_uniform(
    scope: &mut SubmitScope,
    opacity: f32,
    ss: u32,
    lane: bool,
) -> LeasedSlots<Integrate> {
    let u = Integrate {
        params: [opacity, ss as f32, f32::from(u8::from(lane)), 0.0],
    };
    scope.run_uniform("stark integrate opacity", &u)
}

/// Everything one piece's sweep draws with: which tiles, each tile's contiguous
/// instance range, the instance buffer, and the sweep's three bind groups.
///
/// One derivation for every pass that rasterizes the swept extent — the deposit and the
/// erase sweep — for [`StrokeConstants`](super::StrokeConstants)'s reason: they draw
/// the *same* extent, gated by the same drain, tooth and jitter, and differ by pipeline
/// alone.
pub(super) struct SweepDraws {
    /// The tiles the piece's segments reach, texture (interior + apron) included —
    /// [`tiles_with_segments`]' answer, in its order, which `runs` is laid out to match.
    coords: Vec<TileCoord>,
    /// Each tile's slice of the instance buffer.
    runs: Vec<Range<u32>>,
    instances: wgpu::Buffer,
    /// A [`TileXform`] per tile, selected in group 0 by its dynamic offset.
    xforms: LeasedSlots<TileXform>,
    /// In `@group` order: the transform slots, the prefix-τ volume, and the noise and
    /// substrate fields ([`sweep_brush_groups`]).
    groups: [wgpu::BindGroup; 3],
}

impl SweepDraws {
    /// Lay out a piece's draws: which segments reach which tile, and the instance
    /// buffer to match — each tile's segments contiguous, so its draw is one instance
    /// *range* rather than the whole stroke. A segment writes exactly zero outside the
    /// tiles it is listed under, and zero is an exact identity through both blends, so
    /// this is the same picture as drawing everything everywhere, for
    /// `Σ tiles-per-segment` instances instead of `segments × tiles`
    /// ([`tiles_with_segments`]).
    ///
    /// `ss` is the supersampling factor the piece rasterizes at (§6.2): it reaches the
    /// shader as the pixel footprint's half-width, `0.5 / ss` canvas px
    /// (`TileXform::params.w`), so each subsample box-filters over its own footprint
    /// rather than the whole pixel's. 1 on every path that does not supersample, the
    /// erase sweep included.
    ///
    /// `None` where the segments reach no tile, before anything is leased.
    pub(super) fn new(
        r: &StrokeRenderer,
        scope: &mut SubmitScope,
        range: &ResolvedRange<'_>,
        substrate: &SubstrateMap,
        ss: u32,
    ) -> Option<Self> {
        let ResolvedRange {
            rec,
            tip,
            consts: k,
            segments,
            ..
        } = *range;
        let pairs = tiles_with_segments(segments);
        // A tile's pairs are one contiguous run of the sorted list, so its instances
        // are one contiguous run of the buffer. All three reserved: the pair count is
        // the instance count, and the run count is one pass over it.
        let tiles = pairs.chunk_by(|a, b| a.0 == b.0).count();
        let mut coords: Vec<TileCoord> = Vec::with_capacity(tiles);
        let mut instances: Vec<SegmentInstance> = Vec::with_capacity(pairs.len());
        let mut runs: Vec<Range<u32>> = Vec::with_capacity(tiles);
        for run in pairs.chunk_by(|a, b| a.0 == b.0) {
            let from = instances.len() as u32;
            instances.extend(
                run.iter()
                    .map(|&(_, i)| segment_instance(&segments[i as usize])),
            );
            coords.push(run[0].0);
            runs.push(from..instances.len() as u32);
        }

        // Per-tile sweep transforms. The texture top-left is the interior origin
        // shifted out by the apron, so the full TILE_TEX target maps to NDC [-1, 1];
        // everything else is a stroke constant, repeated per slot because the slot is
        // what the shader reads.
        let apron = Vec2::splat(TILE_APRON as f32);
        let xforms: Vec<TileXform> = coords
            .iter()
            .map(|coord| {
                let size = (TILE_TEX as f32, TILE_TEX as f32);
                tile_xform(k, coord.origin() - apron, size, ss)
            })
            .collect();
        let xforms = scope.run_slots("stark sweep xforms", &xforms)?;

        // Leased and written through the scope, never `create_buffer_init`: that maps
        // at creation, and Chrome/Dawn caps map-at-creation buffers well below the
        // normal `maxBufferSize`, so a long stroke would panic in `createBuffer`. It is
        // also the largest buffer the stroke path builds, rebuilt on every pointer move,
        // so the pool hands back the previous move's (`scratch::BufKey`).
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        let instances = scope.take_run_buffer(BufKey {
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            label: "stark sweep instances",
        });
        scope.write_lease(&instances, bytes);

        let xform = r
            .swept
            .uniform_bgl
            .group(&r.ctx.device, "stark sweep bg", |_| xforms.window());
        let (prefix, noise) = sweep_brush_groups(r, scope, tip, rec, substrate, k);
        Some(Self {
            coords,
            runs,
            instances,
            xforms,
            groups: [xform, prefix, noise],
        })
    }

    /// The tiles the piece reaches, in draw order: [`draw`](Self::draw)'s `i` indexes
    /// this.
    pub(super) fn coords(&self) -> &[TileCoord] {
        &self.coords
    }

    /// Draw tile `i`'s segments with `pipeline`, one of the sweep's. Just this tile's:
    /// every other segment differences its prefix-τ taps to zero here anyway, so what
    /// is skipped is the shading, not a contribution.
    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        i: usize,
    ) {
        pass.set_pipeline(pipeline);
        desc::Bound::new(&self.groups, &[self.xforms.offset(i)]).set(pass);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..SWEEP_VERTS, self.runs[i].clone());
    }
}

/// One segment as the sweep is instanced with — the same record for every draw of
/// the swept extent, the stamp loop's lane draw included (§6.2).
pub(super) fn segment_instance(segment: &Segment) -> SegmentInstance {
    let Segment { sweep, paint } = segment;
    SegmentInstance {
        start: sweep.start.to_array(),
        dir: sweep.dir.to_array(),
        // The tip's radius, which is the frame brush-local coordinates are read in
        // (§6.6, [`Sweep::radius`]) — the ramp rides beside it unscaled, that being
        // the point of its being *relative*.
        geom: [sweep.radius, sweep.length, sweep.radius_ramp],
        extra: [sweep.orient, sweep.dist, sweep.curvature, paint.add],
        tooth_give: paint.tooth_give,
        // The ceiling's factor across the segment (§6.2) — its mean and its ramp,
        // `(1, 0)` unless the pen drives it, and read by no pipeline but the
        // ceiling lane's.
        opacity: [paint.opacity, paint.opacity_ramp],
        // The other two per-texel targets' ramps (§6.2); their means ride
        // `extra.w` and `tooth_give` above.
        ramps: [paint.add_ramp, paint.tooth_give_ramp],
        // The solved stretch map (§6.6). Unscaled for the ramp's reason: it acts on
        // brush-local coordinates, which are already the frame's own units.
        stretch: [
            sweep.stretch.travel,
            sweep.stretch.shear,
            sweep.stretch.lateral,
        ],
    }
}

/// The sweep's per-target uniform (§6.2): the target's top-left in canvas px and
/// its size — a tile texture's, apron included, or the stamp loop's region's — with
/// the stroke constants every fragment reads. `ss` is the supersampling factor the
/// target is rasterized at.
pub(super) fn tile_xform(
    k: &super::StrokeConstants,
    origin: Vec2,
    size: (f32, f32),
    ss: u32,
) -> TileXform {
    TileXform {
        params: [
            origin.x,
            origin.y,
            0.0,
            // Half the deposit's own footprint in canvas px (§6.2): the box filter's
            // half-window at 1×, each subsample's own share of the pixel above that.
            // The canvas→NDC scale below does *not* move — a supersampled target
            // covers the same canvas extent, only on a finer grid.
            0.5 / ss as f32,
        ],
        color: [k.channels[0], k.channels[1], k.channels[2], 1.0],
        resid: k.resid,
        paint: [k.drain_px, 0.0, k.tooth_softness, 0.0],
        noise_freq: k.nfreq,
        noise_amp: k.namp,
        jitter_eps: k.jitter_eps,
        jitter_seed: k.jitter_seed,
        ndc: [2.0 / size.0, 2.0 / size.1],
        substrate: [k.substrate.num[0], k.substrate.num[1], k.substrate.den, 0],
    }
}

/// One [`TileXform`] bound as the sweep's group 0 — for a draw over a single
/// target, the stamp loop's lane draw (§6.2), where [`SweepDraws`] stages a slot
/// per tile. Drawn at offset 0.
pub(super) fn xform_group(
    r: &StrokeRenderer,
    scope: &mut SubmitScope,
    xform: &TileXform,
) -> wgpu::BindGroup {
    let slot = scope.piece_uniform("stark sweep region xform", xform);
    r.swept
        .uniform_bgl
        .group(&r.ctx.device, "stark sweep region bg", |_| slot.window())
}

/// The integrate pass's bind group: the pristine paint under a tile, the parcel the
/// stroke laid over it, the selection coverage, the stroke's opacity ceiling, and
/// the ceiling lane — the parcel's own under a pen-driven opacity, the 1×1 zero
/// otherwise (the uniform says which, and the shader reads the lane only when
/// it does).
///
/// A residual build asks for both residual slots, and both are there: the base's
/// residual, the zero standing in for it and the parcel's third lane are all present
/// exactly when the space has one (§6.7).
fn integrate_bind_group(
    r: &StrokeRenderer,
    base: Targets<'_>,
    parcel: Targets<'_>,
    mask: &wgpu::TextureView,
    opacity: &LeasedSlots<Integrate>,
    ceiling: &wgpu::TextureView,
) -> wgpu::BindGroup {
    r.integrate
        .bindings
        .group(&r.ctx.device, "stark integrate bg", |b| match b {
            ib::BASE_COLOR => wgpu::BindingResource::TextureView(base.color),
            ib::BASE_AUX => wgpu::BindingResource::TextureView(base.aux),
            ib::SCRATCH_COLOR => wgpu::BindingResource::TextureView(parcel.color),
            ib::SCRATCH_AUX => wgpu::BindingResource::TextureView(parcel.aux),
            ib::SELECTION => wgpu::BindingResource::TextureView(mask),
            ib::IG => opacity.window(),
            ib::SCRATCH_CEILING => wgpu::BindingResource::TextureView(ceiling),
            ib::BASE_RESID => {
                wgpu::BindingResource::TextureView(base.resid.expect("a residual build has one"))
            }
            ib::SCRATCH_RESID => {
                wgpu::BindingResource::TextureView(parcel.resid.expect("a residual build has one"))
            }
            other => unreachable!("the integrate's group holds no binding {other}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which lanes exist, by index: the shape the sweep's targets and the landing
    /// pass's binds both read.
    #[test]
    fn parcel_keys_fill_exactly_the_lanes_the_stroke_carries() {
        let present = |keys: LaneKeys| keys.map(|key| key.is_some());
        let color = wgpu::TextureFormat::Rgba16Float;
        let resid = Some(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            present(parcel_keys(color, None, false)),
            [true, true, false, false]
        );
        assert_eq!(
            present(parcel_keys(color, resid, false)),
            [true, true, true, false]
        );
        // The hole: a space with no residual under a pen-driven ceiling.
        assert_eq!(
            present(parcel_keys(color, None, true)),
            [true, true, false, true]
        );
        assert_eq!(
            present(parcel_keys(color, resid, true)),
            [true, true, true, true]
        );
    }
}
