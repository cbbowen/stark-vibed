//! The swept fast path (§6.2): one quad per segment, coverage integrated
//! along the sweep through a precomputed prefix-τ texture, over-blended so optical
//! depth sums exactly.
//!
//! Carries no brush state between segments, which is what makes it the fast path —
//! a range needs nothing from its predecessor but the arc length.

use crate::colorspace::ColorSpace;
use crate::gpu::channels::Targets;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use stark_model::document::StrokeRecord;
use stark_model::geom::{TILE_APRON, TILE_TEX, TileCoord, Vec2};
use stark_shaders::mirror::integrate::binding as ib;
use stark_shaders::mirror::integrate::decl as id;
use stark_shaders::mirror::stamp_common::binding as sc;
use stark_shaders::mirror::stamp_common::decl as sd;

/// The stamp pass's three groups (§6.2, §6.10), and the integrate's one.
///
/// `stamp.wesl` declares none of these itself: they are `stamp_common.wesl`'s, which is
/// where the swept-segment scaffolding lives — so naming the declarations is also what
/// says which file to read.
///
/// One buffer for the whole stroke with a slot per tile, rather than a buffer built per
/// tile on every pointer move.
const XFORM_SLOTS: &[Slot] = &[Slot::dynamic(sd::XF)];

/// The prefix-τ volume at group 1 — an Rg32Float 2-D array (x, y, + orientation
/// layers; travel prefix in `r`, its lateral prefix in `g`, §6.2) read with
/// `textureLoad`, since the shader does its own trilinear lookup (§6.6). The
/// array dimension comes from the declaration; the host used to name it.
pub(super) const PREFIX_SLOTS: &[Slot] = &[Slot::at(sd::PREFIX_TEX)];

/// Group 2: the color-dynamics noise field and its repeat sampler (§6.2), and beside
/// them the canvas substrate's map — height and the rise ahead — with its own sampler,
/// for the deposition tooth (§6.4). In this group rather than one of its own because it
/// is the same kind of thing as the noise: a tileable field the deposit samples per
/// fragment, resolved per stroke.
pub(super) const NOISE_SLOTS: &[Slot] = &[
    Slot::sampled(sd::NOISE_TEX),
    Slot::at(sd::NOISE_SAMP),
    // The substrate is read **nearest** (`substrate_texel_at`, §6.4), so it needs no filtering and
    // has no sampler — which is what naming the declarations turned up here. The host
    // had been declaring a fourth entry, a filtering sampler at binding 3, and binding
    // `substrate.sampler` into it; `stamp_common.wesl` declares no such slot and says on
    // its face that it does not ("No sampler: the tap is nearest"). A layout may carry
    // an entry no shader reads, so nothing failed — it was a sampler bound for a
    // lookup that stopped existing.
    Slot::at(sd::SUBSTRATE_TEX),
];

/// The integrate pass (`integrate.wesl`, §6.2/§6.1): the layer's resident paint, the
/// stroke's scratch parcel, the selection each is gated by, the paint
/// effect's opacity — bound on every stroke, exactly 1 (the shader's identity
/// branch) on the unscaled path — and the ceiling lane, which is the parcel's
/// fourth lane under a pen-driven opacity and the 1×1 zero everywhere else.
const INTEGRATE_SLOTS: &[Slot] = &[
    Slot::at(id::BASE_COLOR),
    Slot::at(id::BASE_AUX),
    Slot::at(id::SCRATCH_COLOR),
    Slot::at(id::SCRATCH_AUX),
    Slot::at(id::SELECTION),
    Slot::at(id::IG),
    Slot::at(id::BASE_RESID),
    Slot::at(id::SCRATCH_RESID),
    Slot::at(id::SCRATCH_CEILING),
];
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TileMap};

use super::accum::{
    BareCanvas, IncrementalTileAccumulator, Land, Landed, Landing, Sweep, lane_key,
};
use super::incremental::{Carried, Resume};
use super::region::tiles_with_segments;
use super::segments::{Segment, SegmentInstance, generate_segments_in};
use super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState};
use crate::gpu::scratch::{BufKey, Key};
use crate::gpu::uniforms::UniformSlots;

// Vertices in one segment's swept geometry: a triangle strip of two rims across
// `SWEEP_SLICES` steps along the travel, since a segment's centreline is an arc rather
// than a chord (§6.2). Generated from `stamp_common.wesl`, which is where the strip is
// actually built — asking for fewer would clip the sweep short, more would fold the
// strip back over itself.
use stark_shaders::mirror::stamp_common::{SWEEP_SLICES, SWEEP_VERTS};

/// The draw call and the strip agree on the vertex count.
///
/// Both numbers are the shader's now, so this is the shader's own invariant rather
/// than a boundary check — and it holds at compile time, where the runtime test that
/// scraped `SWEEP_SLICES` out of the linked source used to.
const _: () = assert!(
    SWEEP_VERTS == 2 * (SWEEP_SLICES + 1),
    "the sweep strip's slice count and its vertex count have diverged",
);

// The per-tile uniform, generated from `stamp_common.wesl`'s own declaration
// (§6.7): the tile *texture's* top-left in canvas px + canvas→NDC scale, plus the
// brush's stroke-constant color channels.
use super::tips::ResolvedTip;
use stark_shaders::mirror::stamp_common::TileXform;

/// One tile's window into the stroke's transform buffer — the `min_binding_size` the
/// sweep's layout declares, taken from the struct rather than written down.
const XFORM_SLOT: u64 = std::mem::size_of::<TileXform>() as u64;

/// Stride between the per-tile [`TileXform`] slots, from the type rather than from a
/// number written here (§6.10, and `gpu::uniforms`' own law): the padded size a
/// dynamic offset must be a multiple of.
///
/// **Taken from [`UniformSlots`] without taking its buffer**, which is the one place
/// this path departs from that type. What `UniformSlots` owns is a grow-only
/// allocation per pass, and the stroke path has something better — a leased buffer
/// from the scratch pool (`scratch::BufKey`), recycled across strokes rather than
/// across frames of one. What it does *not* have on its own is the stride law: spelled
/// as a bare `256` beside a `copy_from_slice` of the uniform's real size, a uniform
/// that outgrew the quantum would write over the next slot rather than widening them —
/// silently, and only for whichever brush reached it.
const XFORM_STRIDE: u64 = UniformSlots::<TileXform>::STRIDE;

/// How many scratch pairs the sweep rotates through (§6.2) — see
/// [`render_swept`](StrokeRenderer::render_swept), where the ring is acquired.
///
/// The floor is the dependency being removed: at 1, tile `n+1`'s sweep waits on tile
/// `n`'s integrate, and the path's `2N` render passes cannot overlap at all. The
/// ceiling is memory — one pair is `TILE_TEX²` of the color format plus the wide
/// scratch aux plus a residual, so ~1.5 MB in a pigment space. Three is enough for the
/// driver to keep a sweep, an integrate and a spare in flight; past that the win falls
/// off well before the megabytes do.
const SCRATCH_RING: usize = 3;

/// The swept fast path's GPU objects, built once (§6.2) — the sweep that accumulates
/// a stroke's extent into a scratch tile, and the integrate that stacks that
/// scratch over the base into a fresh CoW tile.
///
/// A kit for the same reason [`DynamicsKit`](super::DynamicsKit) is one, and it is
/// overdue: these five sat loose on [`StrokeRenderer`] among the caches, so a struct
/// documented as holding "only immutable GPU objects" held one path's pipelines by
/// name and the other's behind a type. Both are behind a type now, and the renderer is
/// composition rather than storage.
///
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
#[derive(Clone)]
pub(super) struct SweptKit {
    /// The sweep: one instanced quad strip per segment, over-blended into the scratch
    /// pair, with the per-tile transform at group 0, the prefix-τ volume at group 1
    /// and the noise + substrate fields at group 2.
    pub(super) pipeline: wgpu::RenderPipeline,
    /// The same sweep with the **ceiling lane** as a fourth target (§6.2): the
    /// pipeline a stroke whose opacity the pen drives takes. Its own pipeline
    /// because a target list is fixed at pipeline creation, and its own artifact
    /// because the shader's output is `@if(ceiling)` — every other stroke never
    /// pays the lane's two bytes a fragment.
    pub(super) pipeline_ceiling: wgpu::RenderPipeline,
    /// The ceiling lane **alone** (`stamp.wesl::fs_levels`): what the stamp loop
    /// draws per painting segment into its region, so its claim advances by the
    /// sweep's own sums (§6.2). Over the same three layouts, so the loop binds
    /// the brush exactly as the swept path does.
    pub(super) levels_pipeline: wgpu::RenderPipeline,
    pub(super) uniform_bgl: wgpu::BindGroupLayout,
    pub(super) prefix_bgl: wgpu::BindGroupLayout,
    pub(super) noise_bgl: wgpu::BindGroupLayout,
    /// The integrate (§6.2/§6.1): a fullscreen pass reading the base tile + the
    /// stroke's extent scratch and writing `new = f(base, scratch)` into a fresh CoW
    /// tile's color+aux MRT — the scratch's accumulated parcel stacked on the base
    /// through the shared law in `paint_common.wesl`, the same one a fill lands through
    /// and the stamp loop's `deposit` uses.
    pub(super) integrate_pipeline: wgpu::RenderPipeline,
    pub(super) integrate_bgl: wgpu::BindGroupLayout,
}

/// Compile `stamp.wesl` for `color_space` (§6.2, §6.7) — with or without the
/// ceiling lane, the two artifacts `stark_shaders::stamp` chooses between.
///
/// **Once per renderer per variant, lent to both kits.** The erase pass builds its
/// own pipelines over the very same modules (§6.12) — only the fragment entry point
/// and the target list differ — so a second `create_shader_module` was a second
/// parse and a second translation of source already in hand, which on the web is
/// startup the artist waits through. A module is immutable and entry points are
/// resolved per pipeline, so lending it is all there is to it.
pub(super) fn stamp_module(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    ceiling: bool,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(if ceiling {
            "stark stamp ceiling"
        } else {
            "stark stamp"
        }),
        source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader(ceiling).into()),
    })
}

/// The format of the ceiling lane (§6.2, `paint_common::level_sums`): the parcel's
/// mass above each of two ceiling levels and its moment above each — four sums,
/// f16 for the reason the aux's own mass is: the interesting range is a few times
/// `OPAQUE_MASS`, and past f16's mantissa the coverage they decide is 1 anyway.
pub(super) const CEILING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The ceiling lane as a sweep target: **additive**, like the aux, because every
/// one of its four channels is a sum over segments. That is the whole reason the
/// law is stated in gated sums (`paint_common::claimed_coverage`): a max is not
/// something fixed-function blending can take of a running quantity, but the mass
/// above a level is a sum a blend can accumulate, and the point-wise max of the
/// ceilings falls out of two of them and their moments. For the erase sweep too,
/// which attaches the same lane by the same rule (§6.12), and for the stamp loop's
/// per-segment draw of it over its region (`fs_levels`).
pub(super) fn ceiling_target(color_space: &dyn ColorSpace) -> Option<wgpu::ColorTargetState> {
    desc::blended_target(CEILING_FORMAT, Some(color_space.aux_blend()))
}

/// Build the swept fast path's kit (§6.2): the sweep pipeline over its three bind
/// group layouts — twice, with and without the ceiling lane — and the integrate
/// that lands its scratch on the base.
pub(super) fn build_swept_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    shader: &wgpu::ShaderModule,
    shader_ceiling: &wgpu::ShaderModule,
) -> SweptKit {
    let frag = wgpu::ShaderStages::FRAGMENT;
    // One slot per affected tile, selected by a dynamic offset
    // ([`XFORM_STRIDE`]) — so a stroke crossing many tiles binds one buffer rather
    // than building one per tile on every pointer move.
    let uniform_bgl = desc::layout_for(
        device,
        "stark sweep uniform bgl",
        XFORM_SLOTS,
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        false,
    );

    // The prefix-τ texture is an Rg32Float 2D-array (x, y, + orientation layers),
    // sampled via textureLoad (not filterable), so the shader does its own trilinear
    // lookup.
    let prefix_bgl = desc::layout_for(device, "stark sweep prefix bgl", PREFIX_SLOTS, frag, false);

    // Group 2: the color-dynamics noise field (a tileable 3-D volume) + its
    // repeat sampler (§6.2), and beside it the canvas substrate's map (height +
    // the rise ahead) + its own repeat sampler — the deposition tooth (§6.4). In
    // this group rather than one of its own because it is the same kind of thing
    // as the noise: a tileable field the deposit samples per fragment, resolved
    // per stroke.
    let noise_bgl = desc::layout_for(device, "stark sweep noise bgl", NOISE_SLOTS, frag, false);

    let layout = desc::pipeline_layout(
        device,
        "stark sweep layout",
        &[Some(&uniform_bgl), Some(&prefix_bgl), Some(&noise_bgl)],
    );
    let targets = [
        desc::blended_target(color_space.color_format(), Some(color_space.color_blend())),
        // The stamp renders into a *scratch* tile, whose aux is the wide
        // SCRATCH_AUX_FORMAT — not the compact persistent aux. Additive
        // blend across overlapping segments.
        desc::blended_target(SCRATCH_AUX_FORMAT, Some(color_space.aux_blend())),
        // The parcel's residual (§6.7), over-blended by the color's rule
        // because it is the rest of the same color.
        color_space
            .resid_format()
            .and_then(|f| desc::blended_target(f, Some(color_space.color_blend()))),
        // The ceiling lane (§6.2), additive (`ceiling_target`). At the shader's
        // location 3 whether or not the space has a residual at 2, so the plain
        // list stops short of it and the ceiling list carries the hole
        // (`accum::MAX_LANES`).
        ceiling_target(color_space),
    ];
    let sweep_pipeline = |label, module, fs, targets: &[Option<wgpu::ColorTargetState>]| {
        desc::render_pipeline(
            device,
            desc::RenderPipe {
                label,
                layout: &layout,
                module,
                vs: "vs_main",
                fs,
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
                    wgpu::VertexStepMode::Instance,
                ))],
                targets,
            },
        )
    };
    let pipeline = sweep_pipeline(
        "stark sweep pipeline",
        shader,
        "fs_main",
        &targets[..2 + usize::from(color_space.has_resid())],
    );
    let pipeline_ceiling = sweep_pipeline(
        "stark sweep ceiling pipeline",
        shader_ceiling,
        "fs_main",
        &targets,
    );
    // The lane alone, for the stamp loop's per-segment draw of it (§6.2) — over
    // the plain module, since `fs_levels` is not `@if(ceiling)`-gated: it writes
    // nothing else.
    let levels_pipeline = sweep_pipeline(
        "stark sweep levels pipeline",
        shader,
        "fs_levels",
        &[ceiling_target(color_space)],
    );

    let (integrate_pipeline, integrate_bgl) = build_integrate_pipeline(device, color_space);
    SweptKit {
        pipeline,
        pipeline_ceiling,
        levels_pipeline,
        uniform_bgl,
        prefix_bgl,
        noise_bgl,
        integrate_pipeline,
        integrate_bgl,
    }
}

impl StrokeRenderer {
    /// [`Self::render_range`] through the plain swept fast path: no carried brush
    /// state at all, so a range needs nothing from its predecessor but the arc length.
    /// `tol` comes from [`dynamics_setup`](super::dynamics::dynamics_setup), which has
    /// already decided — from the brush — that this stroke takes the fast path, or
    /// that the loop cannot draw it. Handed over rather than recomputed, so one place
    /// answers what a stroke's segments are.
    pub(super) fn render_swept(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tol: crate::path::FlattenTolerance,
        resume: Resume<'_>,
        tip: &ResolvedTip,
    ) -> (TileMap, StrokeCarry) {
        // The control every dynamics row is read against: the same geometry, the
        // same tiles, one instanced draw instead of a dispatch chain per segment.
        // A change that moves this row has moved something shared — segment
        // generation, tile acquisition, the scope — rather than the loop.
        crate::timing::span!("stroke.swept");
        let StrokeScene {
            pool,
            base,
            selection,
            substrate,
            ..
        } = scene;
        // Everything both paths share, resolved once (see [`StrokeConstants`]).
        let k = self.stroke_constants(rec, substrate, selection);
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        if segments.is_empty() {
            return (base.clone(), StrokeCarry::unchanged(end_dist));
        }

        // Below full opacity the parcel's finished coverage is scaled, which is
        // neither of §6.2's two piece-composable forms — so the path stops being
        // stateless and takes the erase pass's shape instead: the parcel
        // accumulates across pieces, every piece re-derived from pristine paint
        // under the whole of it (`accum::ParcelCarry`). A branch on the brush and on
        // the mask, so a live tail and its commit make the same choice for free.
        //
        // Under a pen-driven ceiling too, whatever the dial: the ceiling lane is
        // a claim the pieces of a stroke build up together, and the coverage it
        // admits is below 1 wherever the pen was light.
        //
        // On the mask as well as the dial, because the mask is the ceiling's other
        // factor *per texel* (§6.8): every selection has a rim at least a pixel
        // soft, and a texel under it scales the parcel exactly as a dial below 1
        // does. A ring of stateless pieces there would cap each piece on its own
        // and let a stroke crossing itself outrun the mask.
        //
        // A **supersampled** stroke (§6.2, `budget::supersample_scale`) takes the
        // same shape for the same theorem, one level up: the resolve averages the
        // parcel's *finished* visible alpha, and an average of parcels over-blended
        // per piece differs from the whole parcel averaged by the covariance of the
        // subsample alphas — a cut-dependent term worth a quarter of the alpha at a
        // rim texel where pieces meet. Landing the whole accumulated parcel from
        // pristine paint each piece is what makes the resolve exact whatever the
        // pointer's cadence, and it is why the ring below never supersamples.
        let ss = super::budget::supersample_scale(&rec.brush);
        if k.opacity < 1.0 || k.ceiling_lane || selection.is_active() || ss > 1 {
            return self.render_swept_scaled(scene, rec, &k, &segments, end_dist, resume, tip, ss);
        }

        // The submit scope: the per-stroke buffers and the shared scratch pair ride
        // in it, and only the `finish` that submits the commands naming them can
        // release them (`scratch::SubmitScope`). The ordering is the scope's shape,
        // not a pair of `drop`s placed after the submit and defended by a comment.
        let mut scope = self.scratch.scope(&self.ctx, "stark stroke commit");

        let (prefix_bg, noise_bg) = sweep_binds(self, &mut scope, tip, rec, substrate, &k);
        // The per-tile draw list, instance buffer and transform slots — shared with
        // the erase pass ([`sweep_draws`]). Always at 1×: a supersampled stroke
        // never reaches this loop (the branch above).
        let draws = sweep_draws(self, &mut scope, rec, &k, &segments, 1);
        // The integrate's opacity uniform, at this path's identity: the layout
        // names it on every stroke, and the shader's exact branch at 1 is what
        // keeps this path bit-for-bit what it was. No ceiling lane either — a
        // stroke that has one never comes this way.
        let opacity_buf = opacity_uniform(&mut scope, 1.0, 1, false);
        let SweepDraws {
            coords,
            runs,
            instances: instance_buf,
            xforms: bind_group,
        } = &draws;

        let carry = StrokeCarry {
            dist: end_dist,
            tool: None,
            dirty: coords.clone(),
            deferred: false,
        };

        // Extent → cleared scratch tile: within-stroke accumulation of the parcel
        // this stroke lays (the color target over-blends the parcel's visible alpha
        // with the latent premultiplied by it, the aux accumulates its height and
        // optical mass additively). The scratch aux is the wide format.
        //
        // **A ring, not one pair and not one per tile.** Sharing *one* pair across
        // every tile is sound — each sweep pass clears both targets, so no tile can see
        // what
        // the tile before it left — but it serializes the path: tile `n+1`'s sweep
        // writes the very texture tile `n`'s integrate reads, a write-after-read the
        // driver has to order, so the `2N` passes ran strictly back to back. A ring of
        // [`SCRATCH_RING`] lets that many tiles' work be in flight before the
        // dependency comes round again, for a few MB against the pool's budget — and a
        // stroke touching fewer tiles than the ring takes only as many as it has.
        //
        // The first reason is gone with the pool it borrowed from. These are
        // `ScratchPool` leases now, taken through the scope, and they are taken on the
        // **run** tier: the ring is the loop's running state, not the current tile's,
        // and `take_run` is what says so. That matters beyond tidiness — the piece tier
        // is released at every `flush`, so a ring held there would be correct only for
        // as long as this loop never flushed, which nothing here states and every other
        // tile-writing loop in the crate already does not honour. They are the very
        // keys the scaled path's parcel lanes take, so the two share one warm set.
        let ring: Vec<RingSlot> = (0..SCRATCH_RING.min(coords.len()))
            .map(|_| RingSlot::take(self, &mut scope))
            .collect();

        let mut new_map = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let xform_off = draws.xform_offset(i);
            // Round-robin, so the reuse distance is the ring's length.
            let scratch = &ring[i % ring.len()];

            // This tile's segments into the shared scratch, cleared as it goes.
            {
                let sweep_targets = scratch.targets();
                let sweep_att = sweep_targets.attachments(desc::CLEAR);
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark sweep pass"),
                        color_attachments: &sweep_att[..sweep_targets.count()],
                        ..Default::default()
                    });
                pass.set_pipeline(&self.swept.pipeline);
                pass.set_bind_group(0, bind_group, &[xform_off]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &noise_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                // Just this tile's segments. Every other one differences its prefix-τ
                // taps to zero here anyway, so what is skipped is the shading, not a
                // contribution.
                pass.draw(0..SWEEP_VERTS, runs[i].clone());
            }

            // Integrate the scratch slab over the base into a fresh CoW tile, gated
            // by this tile's selection coverage — its own mask if it has one, or the
            // 1×1 constant standing in for the rest of the canvas (§6.8).
            let dst = self.acquire_tile(pool, AllocSource::IntegrateDestination);
            // The layer's resident paint here, or the 1×1 zero where it has none —
            // the integrate clamps its loads, so bare canvas costs no tile at all
            // (§6.8's pattern), where acquiring a real pooled pair would mean
            // allocating and clearing one on every pointer move whether or not the
            // stroke reached anything unpainted.
            // The pristine paint under this tile, or the 1×1 zeroes where the layer has
            // none — the accumulator's own derivation (`accum::base_targets`), which
            // this loop open-coded a third time.
            let base_t = super::accum::base_targets(self, base.get(coord));
            // The coverage alone: the mask's opacity is in `k.opacity` already
            // (`stroke_constants`), which the integrate multiplies this by.
            let mask_view = self.selection.mask_for(selection, *coord);
            let integrate_bg = integrate_bind_group(
                self,
                base_t,
                scratch.targets(),
                &mask_view,
                &opacity_buf,
                &self.zeroes.aux,
            );
            {
                let int_targets = dst.targets();
                let int_att = int_targets.attachments(desc::CLEAR);
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark integrate"),
                        color_attachments: &int_att[..int_targets.count()],
                        ..Default::default()
                    });
                pass.set_pipeline(&self.swept.integrate_pipeline);
                pass.set_bind_group(0, &integrate_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            new_map = new_map.insert(*coord, dst);
        }

        // `finish` submits and releases everything behind that submit: the ring's
        // leases back to the scratch pool, the destination tiles to theirs by drop, the
        // per-stroke buffers by lease as well (`scratch::SubmitScope`).
        scope.finish();
        (new_map, carry)
    }

    /// [`Self::render_swept`] below full opacity (§6.2): the same sweep, but
    /// into a **per-tile accumulator carried across the pieces of a live
    /// stroke**, with every piece's integrate re-deriving its tiles from
    /// pristine paint under the whole parcel — the erase pass's shape
    /// (`erase.rs`), forced by the same theorem: the opacity scales the parcel's
    /// finished coverage, which is not composable per piece, so pieces cannot
    /// stack scaled parcels and be the whole.
    ///
    /// The same shape *and the same code*: the bookkeeping the two share is
    /// [`accum`](super::accum), and what stays here is `integrate.wesl`'s slots,
    /// this path's pipelines, and the one decision that is the deposit's own — a
    /// stroke onto nothing mints a tile ([`BareCanvas::Mint`]), where an erase
    /// has nothing to erase.
    ///
    /// What that shape costs relative to the ring above: a persistent scratch
    /// pair per touched tile for the stroke's lifetime, a copy per resumed tile
    /// per piece, and no ring overlap. The full-opacity path — every stroke
    /// whose dial is at 1 — never comes here, which is why the branch is on the
    /// brush and not a uniform alone.
    ///
    /// `ss` is the supersampling factor (§6.2, `budget::supersample_scale`): at 2
    /// the parcel lanes hold 2 subsample texels per canvas px, the sweep
    /// rasterizes into them with its pixel-footprint window halved to match
    /// (`TileXform::params.w`), and the integrate box-resolves each destination
    /// texel from the finished 2×2 block (`integrate.wesl`). 1 is this path
    /// exactly as it was, to the bit.
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a distinct type or a plain scalar resolved once, so the transposition the lint guards cannot be written; the five beyond `scene` are what `render_swept` already flattened and resolved, and re-deriving any of them here is what this path exists to avoid"
    )]
    fn render_swept_scaled(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        k: &super::StrokeConstants,
        segments: &[Segment],
        end_dist: f32,
        resume: Resume<'_>,
        tip: &ResolvedTip,
        ss: u32,
    ) -> (TileMap, StrokeCarry) {
        // The pool and the selection are the accumulator's, as in `erase.rs`; the
        // base it reads pristine paint out of is too.
        let StrokeScene { substrate, .. } = scene;
        let mut scope = self.scratch.scope(&self.ctx, "stark stroke scaled commit");
        let (prefix_bg, noise_bg) = sweep_binds(self, &mut scope, tip, rec, substrate, k);
        let draws = sweep_draws(self, &mut scope, rec, k, segments, ss);
        let opacity_buf = opacity_uniform(&mut scope, k.opacity, ss, k.ceiling_lane);

        // The carried parcel's lanes, in the order `stamp.wesl`'s deposit declares
        // its targets — the same three channels at the same formats the unscaled
        // path's ring pairs carry, differing only in outliving their piece (and in
        // coming from the scratch pool rather than the tile pool, since nothing
        // here ever becomes a document tile). A space with no residual builds two
        // lanes and binds two, the `[..2 + has_resid]` count every list of these
        // takes (§6.7) — unless the stroke carries the ceiling lane, which sits at
        // the shader's location 3 and leaves the residual's slot a hole
        // (`accum::MAX_LANES`). A supersampled stroke's lanes are the same set at
        // `ss` texels per px — the one place the 4× memory is paid, and it is
        // transient: no document tile ever holds a subsample.
        let mut keys: Vec<Option<Key>> = vec![
            Some(parcel_key(self.color_space.color_format())),
            Some(parcel_key(SCRATCH_AUX_FORMAT)),
        ];
        match self.color_space.resid_format() {
            Some(f) => keys.push(Some(parcel_key(f))),
            None if k.ceiling_lane => keys.push(None),
            None => {}
        }
        if k.ceiling_lane {
            keys.push(Some(parcel_key(CEILING_FORMAT)));
        }
        let keys: Vec<Option<Key>> = keys
            .into_iter()
            .map(|key| key.map(|key| key.scaled(ss)))
            .collect();

        let Landed { map, carry, dirty } = IncrementalTileAccumulator::resume(
            self,
            scene,
            scope,
            &keys,
            BareCanvas::Mint,
            resume.prior.map(ToolState::swept),
        )
        .run(
            &Sweep {
                label: "stark sweep pass",
                pipeline: if k.ceiling_lane {
                    &self.swept.pipeline_ceiling
                } else {
                    &self.swept.pipeline
                },
                draws: &draws,
                prefix: &prefix_bg,
                noise: &noise_bg,
            },
            &Land {
                label: "stark integrate",
                pipeline: &self.swept.integrate_pipeline,
            },
            |l: &Landing<'_>| {
                // The parcel's lanes as the trio they are, so this and the unscaled
                // loop ask `integrate_bind_group` the same question.
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
                integrate_bind_group(self, l.base, parcel, l.mask, &opacity_buf, ceiling)
            },
        );

        (
            map,
            StrokeCarry {
                dist: end_dist,
                tool: resume.capture.then(|| ToolState(Carried::Sweep(carry))),
                dirty,
                deferred: false,
            },
        )
    }
}

/// The carried parcel's lanes (`render_swept_scaled`), named beside the keys that
/// build them so the attach order and the bind order stay one list
/// ([`Parcel`](super::accum::Parcel)) — and in the order `stamp.wesl` declares its
/// own targets, which is what the sweep attaches them as. The ceiling is at 3
/// with or without a residual at 2: the index is the shader's `@location`.
const COLOR: usize = 0;
const AUX: usize = 1;
const RESID: usize = 2;
const CEILING: usize = 3;

/// One carried-parcel lane's pool key — [`lane_key`]'s usages, at the parcel's own
/// formats. One label for the three: they are one purpose, and the formats already
/// give each its own line in the pool's free list (`scratch::Key`).
fn parcel_key(format: wgpu::TextureFormat) -> Key {
    lane_key(format, "stark sweep parcel")
}

/// One slot of the sweep's scratch ring: the three working textures a tile's sweep
/// writes and its integrate reads, leased for the piece.
///
/// Views only — the pass attaches and binds them, and nothing here copies — where the
/// leases themselves live in the scope until the submit that releases them.
pub(super) struct RingSlot {
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    resid: Option<wgpu::TextureView>,
}

impl RingSlot {
    /// Check one out. The **wide** scratch aux (§6.2), and the color space's own color
    /// and residual beside it — the same trio a parcel lane carries, at the same keys,
    /// which is what lets the two paths draw from one free list.
    fn take(r: &StrokeRenderer, scope: &mut crate::gpu::scratch::SubmitScope) -> Self {
        let mut view = |format| scope.take_run(parcel_key(format)).1;
        Self {
            color: view(r.color_space.color_format()),
            aux: view(crate::gpu::tile::SCRATCH_AUX_FORMAT),
            resid: r.color_space.resid_format().map(&mut view),
        }
    }

    /// The slot's three views as the trio every consumer wants — the same shape a
    /// parcel's lanes make, which is what lets one `integrate_bind_group` serve the
    /// unscaled loop and the accumulator alike.
    fn targets(&self) -> Targets<'_> {
        Targets {
            color: &self.color,
            aux: &self.aux,
            resid: self.resid.as_ref(),
        }
    }
}

/// The sweep's brush-resolved bind groups — the prefix-τ volume at group 1, the
/// noise + substrate fields at group 2. One derivation for the three passes that
/// rasterize the swept extent (the plain deposit, its scaled sibling and the
/// erase sweep), for [`sweep_draws`]' reason: they draw the *same* extent, and a
/// second copy of its inputs would be a place to disagree about what that is.
pub(super) fn sweep_binds(
    r: &StrokeRenderer,
    scope: &mut crate::gpu::scratch::SubmitScope,
    tip: &ResolvedTip,
    rec: &StrokeRecord,
    substrate: &crate::gpu::substrate::SubstrateMap,
    k: &super::StrokeConstants,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let device = &r.ctx.device;
    // The brush's prefix-τ texture — image brushes from the asset store, the round tip
    // generated and cached from its hardness — resolved by `render_range`'s own gate
    // and handed down rather than re-resolved here behind an `expect` naming that
    // gate, which is a claim about a caller where passing the value is a fact.
    let prefix_view = tip.prefix.clone();
    let prefix_bg = desc::bind_group_for(
        device,
        "stark sweep prefix bg",
        &r.swept.prefix_bgl,
        PREFIX_SLOTS,
        false,
        |_| wgpu::BindingResource::TextureView(&prefix_view),
    );
    // Color dynamics (§6.2): the noise tile baked for this stroke, of the
    // brush's kind. An inactive brush binds the zero tile with zero amplitudes
    // — the deposit is exactly the constant color. The canvas substrate beside
    // it (§6.4): the deposition tooth's height and the rise ahead of it, in the
    // same group because it is the same kind of thing — a field the deposit
    // samples per fragment.
    let noise = r.tips.noise(&rec.brush.color_dynamics(), k.noise_seed);
    scope.hold(noise.clone());
    let noise_bg = desc::bind_group_for(
        device,
        "stark sweep noise bg",
        &r.swept.noise_bgl,
        NOISE_SLOTS,
        false,
        |b| match b {
            sc::NOISE_TEX => wgpu::BindingResource::TextureView(noise.view()),
            sc::NOISE_SAMP => wgpu::BindingResource::Sampler(&r.tips.noise_sampler),
            sc::SUBSTRATE_TEX => wgpu::BindingResource::TextureView(&substrate.view),
            other => unreachable!("`NOISE_SLOTS` lists no binding {other}"),
        },
    );
    (prefix_bg, noise_bg)
}

/// The integrate's opacity uniform (`integrate.wesl::Integrate`) for one piece:
/// the stroke's ceiling — the effect's dial with the mask's opacity folded in
/// (`stroke_constants`) — or exactly 1, the shader's identity branch, on the
/// unscaled path, which binds it because the layout names it either way.
///
/// `ss` rides beside it: how many subsample texels per canvas px the scratch
/// holds (§6.2), which is what tells the shader to box-resolve the parcel — 1,
/// the plain 1:1 load, everywhere the sweep did not supersample. And `lane`,
/// whether the ceiling lane bound beside the parcel is real — the stroke's
/// opacity is pen-driven — or the 1×1 zero the shader must not read.
pub(super) fn opacity_uniform(
    scope: &mut crate::gpu::scratch::SubmitScope,
    opacity: f32,
    ss: u32,
    lane: bool,
) -> wgpu::Buffer {
    let u = stark_shaders::mirror::integrate::Integrate {
        params: [opacity, ss as f32, f32::from(u8::from(lane)), 0.0],
    };
    let buf = scope.take_run_buffer(BufKey {
        size: std::mem::size_of_val(&u) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        label: "stark integrate opacity",
    });
    scope.write_lease(&buf, bytemuck::bytes_of(&u));
    buf
}

/// Everything one piece's sweep draws with, per tile: which tiles, each tile's
/// contiguous instance range, the instance buffer, and the per-tile transform
/// slots behind one dynamic-offset bind group.
///
/// One derivation for both consumers — the plain deposit ([`StrokeRenderer::render_swept`])
/// and the erase pass (`erase.rs`) — for [`StrokeConstants`](super::StrokeConstants)'s reason: the two
/// rasterize the *same* extent, gated by the same drain, tooth and jitter, and a
/// second copy of this construction would be a place for them to disagree about
/// what the extent is.
pub(super) struct SweepDraws {
    /// The tiles the piece's segments reach, texture (interior + apron) included —
    /// [`tiles_with_segments`]' answer, in its order, which the instance runs below
    /// are laid out to match.
    pub(super) coords: Vec<TileCoord>,
    /// Each tile's slice of the instance buffer.
    pub(super) runs: Vec<std::ops::Range<u32>>,
    pub(super) instances: wgpu::Buffer,
    /// The per-tile [`TileXform`] slots, selected by
    /// [`xform_offset`](Self::xform_offset).
    pub(super) xforms: wgpu::BindGroup,
}

impl SweepDraws {
    /// The dynamic offset that selects tile `i` of [`coords`](Self::coords) in
    /// [`xforms`](Self::xforms).
    ///
    /// Asked of the draws rather than computed at each pass, so the stride the slots
    /// were *written* at and the stride they are *bound* at are one expression.
    pub(super) fn xform_offset(&self, i: usize) -> u32 {
        UniformSlots::<TileXform>::offset(i as u32)
    }
}

/// Build a piece's [`SweepDraws`]: which segments reach which tile, and the
/// instance buffer laid out to match — each tile's segments contiguous, so its
/// draw is one instance *range* rather than the whole stroke. A segment writes
/// exactly zero outside the tiles it is listed under, and zero is an exact
/// identity through both blends, so this is the same picture as drawing everything
/// everywhere — for `Σ tiles-per-segment` instances instead of `segments × tiles`
/// ([`tiles_with_segments`]).
///
/// The duplication is real but small: a segment is at most a tip wide, so it
/// appears under a handful of tiles. What it replaces grew with the *stroke*.
///
/// `ss` is the supersampling factor the piece rasterizes at (§6.2): it reaches the
/// shader as the pixel footprint's half-width, `0.5 / ss` canvas px
/// (`TileXform::params.w`), so each subsample box-filters over its own footprint
/// rather than the whole pixel's. 1 — half a px, the value the lane always meant —
/// on every path that does not supersample, the erase sweep included.
pub(super) fn sweep_draws(
    r: &StrokeRenderer,
    scope: &mut crate::gpu::scratch::SubmitScope,
    rec: &StrokeRecord,
    k: &super::StrokeConstants,
    segments: &[Segment],
    ss: u32,
) -> SweepDraws {
    let device = &r.ctx.device;
    let touched = tiles_with_segments(segments);
    let coords: Vec<TileCoord> = touched.keys().copied().collect();
    // Both reserved: the instance count is the sum of the per-tile lists, which
    // `touched` already holds. `runs` was reserved and `instances` — the larger of the
    // two by an order of magnitude — was not.
    let mut instances: Vec<SegmentInstance> =
        Vec::with_capacity(touched.values().map(Vec::len).sum());
    let mut runs: Vec<std::ops::Range<u32>> = Vec::with_capacity(touched.len());
    for idx in touched.values() {
        let from = instances.len() as u32;
        instances.extend(idx.iter().map(|&i| segment_instance(&segments[i as usize])));
        runs.push(from..instances.len() as u32);
    }
    // Leased and written through the scope (`SubmitScope::write_lease`), never
    // `create_buffer_init`: that maps at creation, and Chrome/Dawn caps
    // map-at-creation buffers well below the normal `maxBufferSize`, so a long stroke
    // would panic in `createBuffer`. Leasing is the other half of the same problem —
    // this is the largest buffer the stroke path builds and it was built afresh on
    // every pointer move, where the pool hands back the one the previous move used
    // (`scratch::BufKey`).
    let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
    let instance_buf = scope.take_run_buffer(BufKey {
        size: instance_bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        label: "stark sweep instances",
    });
    scope.write_lease(&instance_buf, instance_bytes);

    // Per-tile sweep transforms, one [`XFORM_STRIDE`] slot each in a single
    // buffer the draws select with a dynamic offset. The texture top-left is
    // the interior origin shifted out by the apron, so the full TILE_TEX target
    // maps to NDC [-1, 1]; everything else is a stroke constant, repeated per slot
    // because the slot is what the shader reads.
    //
    // One buffer and one bind group for the stroke, not one of each per tile: this
    // path redraws on every pointer move, and the allocation *rate* is what OOMs
    // the tab (`gpu::uniforms`).
    let apron = TILE_APRON as f32;
    let mut xform_data = vec![0u8; coords.len() * XFORM_STRIDE as usize];
    for (i, coord) in coords.iter().enumerate() {
        let origin = coord.origin();
        let xform = tile_xform(
            rec,
            k,
            origin - Vec2::splat(apron),
            (TILE_TEX as f32, TILE_TEX as f32),
            ss,
        );
        let at = i * XFORM_STRIDE as usize;
        xform_data[at..at + XFORM_SLOT as usize].copy_from_slice(bytemuck::bytes_of(&xform));
    }
    let xform_buf = scope.take_run_buffer(BufKey {
        size: xform_data.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        label: "stark sweep xforms",
    });
    scope.write_lease(&xform_buf, &xform_data);
    // Through the slot list the layout was built from, like every other group in the
    // crate — a hand-written `binding: 0` is the shader's own number transcribed onto
    // the host, which is the drift §6.10 exists to remove. The window is the
    // uniform's size and the offset is the draw's, so the entry names a slot rather
    // than the whole buffer.
    let xforms = desc::bind_group_for(
        device,
        "stark sweep bg",
        &r.swept.uniform_bgl,
        XFORM_SLOTS,
        false,
        |_| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &xform_buf,
                offset: 0,
                size: wgpu::BufferSize::new(XFORM_SLOT),
            })
        },
    );

    SweepDraws {
        coords,
        runs,
        instances: instance_buf,
        xforms,
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
        geom: [sweep.radius, sweep.length, sweep.ramp],
        extra: [sweep.orient, sweep.dist, sweep.curvature, paint.add],
        tooth_give: paint.tooth_give,
        // The ceiling's factor across the segment (§6.2) — its mean and its ramp,
        // `(1, 0)` unless the pen drives it, and read by no pipeline but the
        // ceiling lane's.
        opacity: [paint.opacity, paint.opacity_ramp],
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
    rec: &StrokeRecord,
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
            // Half the deposit's own footprint in canvas px (§6.2): the box
            // filter's half-window at 1×, and each subsample's own share of
            // the pixel when the sweep supersamples. The canvas→NDC scale
            // below does *not* move — a supersampled target covers the
            // same canvas extent; only the rasterizer's grid is finer.
            0.5 / ss as f32,
        ],
        color: [k.channels[0], k.channels[1], k.channels[2], 1.0],
        resid: k.resid,
        paint: [
            rec.brush.drain_px(),
            k.substrate_uv_scale,
            k.tooth_softness,
            0.0,
        ],
        noise_freq: k.nfreq,
        noise_amp: k.namp,
        jitter_eps: k.jitter_eps,
        jitter_seed: k.jitter_seed,
        ndc: [2.0 / size.0, 2.0 / size.1],
    }
}

/// One [`TileXform`] bound as the sweep's group 0 — for a draw over a single
/// target, the stamp loop's lane draw (§6.2), where [`sweep_draws`] builds a slot
/// per tile.
pub(super) fn xform_group(
    r: &StrokeRenderer,
    scope: &mut crate::gpu::scratch::SubmitScope,
    xform: &TileXform,
) -> wgpu::BindGroup {
    let buf = scope.take_piece_buffer(BufKey {
        size: XFORM_SLOT,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        label: "stark sweep region xform",
    });
    scope.write_lease(&buf, bytemuck::bytes_of(xform));
    desc::bind_group_for(
        &r.ctx.device,
        "stark sweep region bg",
        &r.swept.uniform_bgl,
        XFORM_SLOTS,
        false,
        |_| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buf,
                offset: 0,
                size: wgpu::BufferSize::new(XFORM_SLOT),
            })
        },
    )
}

/// Build the stroke integrate pipeline (`integrate` shader) — §6.2/§6.1. A
/// fullscreen pass with four sampled tiles (base/scratch color/aux), writing the
/// color+aux MRT of a fresh tile.
pub(super) fn build_integrate_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let resid = color_space.has_resid();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark integrate"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::integrate(resid).into()),
    });
    let frag = wgpu::ShaderStages::FRAGMENT;
    let bgl = desc::layout_for(device, "stark integrate bgl", INTEGRATE_SLOTS, frag, resid);
    let layout = desc::pipeline_layout(device, "stark integrate layout", &[Some(&bgl)]);
    // No blend on any target: the shader does the combine and writes straight
    // through.
    let pipeline = desc::fullscreen_pipeline(
        device,
        "stark integrate pipeline",
        &layout,
        &shader,
        ("vs_main", "fs_main"),
        // The space's own three, as `ChannelFormats` counts them — the last of the
        // hand-counted `[..2 + usize::from(resid)]` slices (§6.7).
        &crate::gpu::channels::ChannelFormats::of(color_space).targets(),
    );
    (pipeline, bgl)
}

/// The integrate pass's bind group: the pristine paint under a tile, the parcel the
/// stroke laid over it, the selection coverage, the stroke's opacity ceiling, and
/// the ceiling lane — the parcel's own under a pen-driven opacity, the 1×1 zero
/// otherwise (the uniform says which, and the shader reads the lane only when
/// it does).
///
/// **One derivation for both swept paths.** The unscaled loop and the scaled
/// accumulator each spelled these eight slots out — the same list, the same
/// `unreachable!`, the same two `expect`s on the residual lanes — differing only in
/// where the six views came from: a resident tile and a ring slot on one side, a
/// `Targets` and a parcel's lanes on the other. Both of those *are* trios, so the
/// difference disappears once they are asked for as trios, and what is left is the one
/// thing that was genuinely shared: what the integrate reads. Two spellings of that
/// was a place for the two paths to disagree about it.
///
/// The residual predicate is `base && parcel` because both must be bound for the
/// `_resid` build to be legal. On the scaled path the two are the same question — the
/// base's residual, the zero standing in for it and the parcel's third lane are all
/// present exactly when the space has one (§6.7) — so requiring both costs it nothing.
fn integrate_bind_group(
    r: &StrokeRenderer,
    base: Targets<'_>,
    parcel: Targets<'_>,
    mask: &wgpu::TextureView,
    opacity: &wgpu::Buffer,
    ceiling: &wgpu::TextureView,
) -> wgpu::BindGroup {
    desc::bind_group_for(
        &r.ctx.device,
        "stark integrate bg",
        &r.swept.integrate_bgl,
        INTEGRATE_SLOTS,
        base.resid.is_some() && parcel.resid.is_some(),
        |b| match b {
            ib::BASE_COLOR => wgpu::BindingResource::TextureView(base.color),
            ib::BASE_AUX => wgpu::BindingResource::TextureView(base.aux),
            ib::SCRATCH_COLOR => wgpu::BindingResource::TextureView(parcel.color),
            ib::SCRATCH_AUX => wgpu::BindingResource::TextureView(parcel.aux),
            ib::SELECTION => wgpu::BindingResource::TextureView(mask),
            ib::IG => opacity.as_entire_binding(),
            ib::SCRATCH_CEILING => wgpu::BindingResource::TextureView(ceiling),
            ib::BASE_RESID => {
                wgpu::BindingResource::TextureView(base.resid.expect("a residual build has one"))
            }
            ib::SCRATCH_RESID => {
                wgpu::BindingResource::TextureView(parcel.resid.expect("a residual build has one"))
            }
            other => unreachable!("`INTEGRATE_SLOTS` lists no binding {other}"),
        },
    )
}
