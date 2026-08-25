//! The swept fast path (§6.2): one quad per segment, coverage integrated
//! along the sweep through a precomputed prefix-τ texture, over-blended so optical
//! depth sums exactly.
//!
//! Carries no brush state between segments, which is what makes it the fast path —
//! a range needs nothing from its predecessor but the arc length.

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use stark_model::document::StrokeRecord;
use stark_model::geom::{TILE_APRON, TILE_TEX, TileCoord};
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

/// The prefix-τ volume at group 1 — an R32Float 2-D array (x, y, + orientation layers)
/// read with `textureLoad`, since the shader does its own trilinear lookup (§6.6). The
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
/// stroke's scratch parcel, the selection each is gated by, and the paint
/// effect's opacity — bound on every stroke, exactly 1 (the shader's identity
/// branch) on the unscaled path.
const INTEGRATE_SLOTS: &[Slot] = &[
    Slot::at(id::BASE_COLOR),
    Slot::at(id::BASE_AUX),
    Slot::at(id::SCRATCH_COLOR),
    Slot::at(id::SCRATCH_AUX),
    Slot::at(id::SELECTION),
    Slot::at(id::IG),
    Slot::at(id::BASE_RESID),
    Slot::at(id::SCRATCH_RESID),
];
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TileMap};

use std::collections::BTreeMap;
use std::sync::Arc;

use super::incremental::{Carried, SweepAccum, SweepCarry, SweepTile};
use super::region::tiles_with_segments;
use super::scratch::Key;
use super::segments::{Segment, SegmentInstance, generate_segments_in};
use super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState, UNIFORM_STRIDE};

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
use stark_shaders::mirror::stamp_common::TileXform;

/// One tile's window into the stroke's transform buffer — the `min_binding_size` the
/// sweep's layout declares, taken from the struct rather than written down.
const XFORM_SLOT: u64 = std::mem::size_of::<TileXform>() as u64;

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

/// Build the swept fast path's kit (§6.2): the sweep pipeline over its three bind
/// group layouts, and the integrate that lands its scratch on the base.
pub(super) fn build_swept_kit(device: &wgpu::Device, color_space: &dyn ColorSpace) -> SweptKit {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark sweep"),
        source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader().into()),
    });

    let frag = wgpu::ShaderStages::FRAGMENT;
    // One slot per affected tile, selected by a dynamic offset
    // ([`UNIFORM_STRIDE`](super::UNIFORM_STRIDE)) — so a stroke crossing many tiles
    // binds one buffer rather than building one per tile on every pointer move.
    let uniform_bgl = desc::layout_for(
        device,
        "stark sweep uniform bgl",
        XFORM_SLOTS,
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        false,
    );

    // The prefix-τ texture is a R32Float 2D-array (x, y, + orientation layers), sampled
    // via textureLoad (not filterable), so the shader does its own trilinear lookup.
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
    let pipeline = desc::render_pipeline(
        device,
        desc::RenderPipe {
            label: "stark sweep pipeline",
            layout: &layout,
            module: &shader,
            vs: "vs_main",
            fs: "fs_main",
            primitive: desc::QUAD_STRIP,
            buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
                wgpu::VertexStepMode::Instance,
            ))],
            targets: &[
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
            ][..2 + usize::from(color_space.has_resid())],
        },
    );

    let (integrate_pipeline, integrate_bgl) = build_integrate_pipeline(device, color_space);
    SweptKit {
        pipeline,
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
        tool: Option<&ToolState>,
    ) -> (TileMap, StrokeCarry) {
        // The control every dynamics row is read against: the same geometry, the
        // same tiles, one instanced draw instead of a dispatch chain per segment.
        // A change that moves this row has moved something shared — segment
        // generation, tile acquisition, the scope — rather than the loop.
        crate::timing::span!("stroke.swept");
        let StrokeScene {
            pool,
            assets,
            base,
            selection,
            substrate,
        } = scene;
        // Everything both paths share, resolved once (see [`StrokeConstants`]).
        let k = self.stroke_constants(rec, substrate);
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        if segments.is_empty() {
            return (
                base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                    dirty: Vec::new(),
                },
            );
        }

        // Below full opacity the parcel's finished coverage is scaled, which is
        // neither of §6.2's two piece-composable forms — so the path stops being
        // stateless and takes the erase pass's shape instead: the parcel
        // accumulates across pieces, every piece re-derived from pristine paint
        // under the whole of it ([`SweepCarry`]). A branch on the brush, so a
        // live tail and its commit make the same choice for free.
        if k.opacity < 1.0 {
            return self.render_swept_scaled(scene, rec, &k, &segments, end_dist, tool);
        }

        // The submit scope: the per-stroke buffers and the shared scratch pair ride
        // in it, and only the `finish` that submits the commands naming them can
        // release them (`scratch::SubmitScope`). The ordering is the scope's shape,
        // not a pair of `drop`s placed after the submit and defended by a comment.
        let mut scope = self.scratch.scope(&self.ctx, "stark stroke commit");

        let device = &self.ctx.device;
        let (prefix_bg, noise_bg) = sweep_binds(self, assets, rec, substrate);
        // The per-tile draw list, instance buffer and transform slots — shared with
        // the erase pass ([`sweep_draws`]).
        let draws = sweep_draws(self, &mut scope, rec, &k, &segments);
        // The integrate's opacity uniform, at this path's identity: the layout
        // names it on every stroke, and the shader's exact branch at 1 is what
        // keeps this path bit-for-bit what it was.
        let opacity_buf = opacity_uniform(self, &mut scope, 1.0, selection.strength());
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
        };

        // Extent → cleared scratch tile: within-stroke accumulation of the parcel
        // this stroke lays (the color target over-blends the parcel's visible alpha
        // with the latent premultiplied by it, the aux accumulates its height and
        // optical mass additively). The scratch aux is the wide format.
        //
        // **A ring, held to the submit — not one pair, and not one per tile.** Two
        // separate rules meet here:
        //
        // * Every scratch must outlive the submit of the passes naming it. A pair
        //   acquired per tile and dropped at the end of its iteration goes back on the
        //   pool's free list while those passes are still only recorded — and the free
        //   list is where `TilePool::trim` takes from, tail first, on any `acquire_tex`
        //   that happens to end an epoch. Destroying a texture this command buffer names
        //   fails the submit, so every destination tile in it keeps whatever paint the
        //   pool last had there: one frame of other tiles' work, gone on the next
        //   render. Same rule as `transform::Recording`. Hence `scope.hold` below.
        //
        // * Sharing **one** pair across every tile is sound — each sweep pass clears
        //   both targets, so no tile can see what the tile before it left — but it
        //   serializes the path. Tile n+1's sweep writes the very texture tile n's
        //   integrate reads, which is a write-after-read the driver has to order, so
        //   the `2N` passes ran strictly back to back with no overlap at all.
        //
        // A ring satisfies the first and drops the second: `SCRATCH_RING` tiles' worth
        // of work can be in flight before the dependency comes round again. At
        // `TILE_TEX = 256` a target is 512 KB, so the whole ring is a few MB against
        // `ScratchPool`'s 256 MB budget — and a stroke touching fewer tiles than the
        // ring takes only as many pairs as it has tiles.
        let ring: Vec<_> = (0..SCRATCH_RING.min(coords.len()))
            .map(|_| self.acquire_scratch(pool, AllocSource::StrokeScratch))
            .collect();

        let mut new_map = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let xform_off = (i * UNIFORM_STRIDE) as u32;
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
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
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
            let (base_color, base_aux) = match base.get(coord) {
                Some(tile) => (tile.color_view(), tile.aux_view()),
                None => (&self.zeroes.color, &self.zeroes.aux),
            };
            // The resident residual, or the 1×1 zero on bare canvas — the same pairing
            // the color above makes, since the two are one color (§6.7).
            let base_resid = self
                .zeroes
                .resid
                .as_ref()
                .map(|zero| base.get(coord).and_then(|t| t.resid_view()).unwrap_or(zero));
            // A gating read: the strength travels with the integrate's opacity
            // uniform, which is built from the same selection.
            let mask_view = self.selection.gate_for(selection, *coord);
            let integrate_bg = desc::bind_group_for(
                device,
                "stark integrate bg",
                &self.swept.integrate_bgl,
                INTEGRATE_SLOTS,
                base_resid.is_some() && scratch.resid_view().is_some(),
                |b| match b {
                    ib::BASE_COLOR => wgpu::BindingResource::TextureView(base_color),
                    ib::BASE_AUX => wgpu::BindingResource::TextureView(base_aux),
                    ib::SCRATCH_COLOR => wgpu::BindingResource::TextureView(scratch.color_view()),
                    ib::SCRATCH_AUX => wgpu::BindingResource::TextureView(scratch.aux_view()),
                    ib::SELECTION => wgpu::BindingResource::TextureView(mask_view.view()),
                    ib::IG => opacity_buf.as_entire_binding(),
                    ib::BASE_RESID => wgpu::BindingResource::TextureView(
                        base_resid.expect("a residual build has one"),
                    ),
                    ib::SCRATCH_RESID => wgpu::BindingResource::TextureView(
                        scratch.resid_view().expect("a residual build has one"),
                    ),
                    other => unreachable!("`INTEGRATE_SLOTS` lists no binding {other}"),
                },
            );
            {
                let int_targets = dst.targets();
                let int_att = int_targets.attachments(desc::CLEAR);
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark integrate"),
                        color_attachments: &int_att[..int_targets.count()],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.swept.integrate_pipeline);
                pass.set_bind_group(0, &integrate_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            new_map = new_map.insert(*coord, dst);
        }

        // The scratch ring rides the scope past the submit, for the reason given
        // where it is acquired: released any earlier these are *pooled* textures this
        // command buffer still names, free to be handed out — or destroyed — before
        // the submit. `finish` then submits and releases everything behind that
        // submit: the tile pairs back to their pool by drop, the per-stroke buffers by
        // `destroy()` (left to JS GC they pile up and OOM the tab, §6.2).
        scope.hold(ring);
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
    /// What that shape costs relative to the ring above: a persistent scratch
    /// pair per touched tile for the stroke's lifetime, a copy per resumed tile
    /// per piece, and no ring overlap. The full-opacity path — every stroke
    /// whose dial is at 1 — never comes here, which is why the branch is on the
    /// brush and not a uniform alone.
    fn render_swept_scaled(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        k: &super::StrokeConstants,
        segments: &[Segment],
        end_dist: f32,
        tool: Option<&ToolState>,
    ) -> (TileMap, StrokeCarry) {
        let StrokeScene {
            pool,
            assets,
            base,
            selection,
            substrate,
        } = scene;
        let mut scope = self.scratch.scope(&self.ctx, "stark stroke scaled commit");
        let device = &self.ctx.device;
        let (prefix_bg, noise_bg) = sweep_binds(self, assets, rec, substrate);
        let draws = sweep_draws(self, &mut scope, rec, k, segments);
        let opacity_buf = opacity_uniform(self, &mut scope, k.opacity, selection.strength());

        // The carry this piece hands on: everything the pieces before it
        // accumulated — shared, never rewritten — with this piece's tiles
        // replacing theirs below (`EraseCarry`'s contract).
        let mut tiles: BTreeMap<TileCoord, SweepTile> = match tool.map(ToolState::swept) {
            Some(prior) => prior
                .tiles
                .iter()
                .map(|(c, t)| {
                    (
                        *c,
                        SweepTile {
                            pristine: t.pristine.clone(),
                            accum: Arc::clone(&t.accum),
                        },
                    )
                })
                .collect(),
            None => BTreeMap::new(),
        };

        let mut new_map = base.clone();
        let mut dirty = Vec::new();
        for (i, coord) in draws.coords.iter().enumerate() {
            // The paint the stroke found under this tile: what an earlier piece
            // recorded, or — for a tile this stroke reaches for the first time —
            // the base itself, which no earlier piece can have rewritten. `None`
            // is bare canvas, and unlike an erase the deposit keeps going: a
            // stroke onto nothing mints a tile, over the 1×1 zeroes.
            let pristine = match tiles.get(coord) {
                Some(t) => t.pristine.clone(),
                None => base.get(coord).cloned(),
            };

            // This piece's working parcel: the carried total copied in, or a
            // clear for a first touch — either way every texel is written before
            // the integrate reads it (the pool's no-zero-init contract). The
            // carried textures themselves are only ever read: the live tail
            // resumes the same frozen carry on every pointer move.
            let work = SweepAccum {
                color: self
                    .scratch
                    .keep(device, parcel_key(self.color_space.color_format())),
                aux: self.scratch.keep(device, parcel_key(SCRATCH_AUX_FORMAT)),
                resid: self
                    .color_space
                    .resid_format()
                    .map(|f| self.scratch.keep(device, parcel_key(f))),
            };
            let resumed = tiles.get(coord).map(|t| Arc::clone(&t.accum));
            if let Some(old) = &resumed {
                for (src, dst) in [(&old.color, &work.color), (&old.aux, &work.aux)]
                    .into_iter()
                    .chain(old.resid.iter().zip(work.resid.iter()))
                {
                    scope.encoder().copy_texture_to_texture(
                        src.tex().as_image_copy(),
                        dst.tex().as_image_copy(),
                        PARCEL_EXTENT,
                    );
                }
            }
            {
                let ops = if resumed.is_some() {
                    desc::LOAD
                } else {
                    desc::CLEAR
                };
                let att = [
                    Some(desc::attach(work.color.view(), ops)),
                    Some(desc::attach(work.aux.view(), ops)),
                    work.resid.as_ref().map(|r| desc::attach(r.view(), ops)),
                ];
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark sweep pass"),
                        color_attachments: &att[..2 + usize::from(work.resid.is_some())],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.swept.pipeline);
                pass.set_bind_group(0, &draws.xforms, &[(i * UNIFORM_STRIDE) as u32]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &noise_bg, &[]);
                pass.set_vertex_buffer(0, draws.instances.slice(..));
                pass.draw(0..SWEEP_VERTS, draws.runs[i].clone());
            }

            // The whole stroke's parcel so far, scaled and stacked on the
            // pristine paint — never on the base in hand, which for a resumed
            // tile is an earlier piece's output and would compound the scale
            // per piece.
            let dst = self.acquire_tile(pool, AllocSource::IntegrateDestination);
            let (base_color, base_aux) = match &pristine {
                Some(tile) => (tile.color_view(), tile.aux_view()),
                None => (&self.zeroes.color, &self.zeroes.aux),
            };
            let base_resid = self.zeroes.resid.as_ref().map(|zero| {
                pristine
                    .as_ref()
                    .and_then(|t| t.resid_view())
                    .unwrap_or(zero)
            });
            // A gating read: the strength travels with the integrate's opacity
            // uniform, which is built from the same selection.
            let mask_view = self.selection.gate_for(selection, *coord);
            let integrate_bg = desc::bind_group_for(
                device,
                "stark integrate bg",
                &self.swept.integrate_bgl,
                INTEGRATE_SLOTS,
                base_resid.is_some() && work.resid.is_some(),
                |b| match b {
                    ib::BASE_COLOR => wgpu::BindingResource::TextureView(base_color),
                    ib::BASE_AUX => wgpu::BindingResource::TextureView(base_aux),
                    ib::SCRATCH_COLOR => wgpu::BindingResource::TextureView(work.color.view()),
                    ib::SCRATCH_AUX => wgpu::BindingResource::TextureView(work.aux.view()),
                    ib::SELECTION => wgpu::BindingResource::TextureView(mask_view.view()),
                    ib::IG => opacity_buf.as_entire_binding(),
                    ib::BASE_RESID => wgpu::BindingResource::TextureView(
                        base_resid.expect("a residual build has one"),
                    ),
                    ib::SCRATCH_RESID => wgpu::BindingResource::TextureView(
                        work.resid
                            .as_ref()
                            .expect("a residual build has one")
                            .view(),
                    ),
                    other => unreachable!("`INTEGRATE_SLOTS` lists no binding {other}"),
                },
            );
            {
                let int_targets = dst.targets();
                let int_att = int_targets.attachments(desc::CLEAR);
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark integrate"),
                        color_attachments: &int_att[..int_targets.count()],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.swept.integrate_pipeline);
                pass.set_bind_group(0, &integrate_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            new_map = new_map.insert(*coord, dst);
            dirty.push(*coord);
            tiles.insert(
                *coord,
                SweepTile {
                    pristine,
                    accum: Arc::new(work),
                },
            );
        }

        // Submit before the carry leaves this call: a `Kept` may reach the
        // pool's free list only behind the submit of the commands naming it, and
        // handing the carry out first would let a caller drop it ahead of one.
        scope.finish();
        (
            new_map,
            StrokeCarry {
                dist: end_dist,
                tool: Some(ToolState(Carried::Sweep(SweepCarry { tiles }))),
                dirty,
            },
        )
    }
}

/// One carried-parcel texture's pool key (`render_swept_scaled`): a full tile
/// (interior + apron), renderable (the sweep accumulates into it), bindable (the
/// integrate reads it), and copyable both ways (a resuming piece copies the
/// carried total into its working texture) — the erase accumulator's key, at the
/// parcel's own formats.
fn parcel_key(format: wgpu::TextureFormat) -> Key {
    Key {
        size: (TILE_TEX, TILE_TEX),
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        label: "stark sweep parcel",
    }
}

/// One whole parcel texture, as a copy extent.
const PARCEL_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: TILE_TEX,
    height: TILE_TEX,
    depth_or_array_layers: 1,
};

/// The sweep's brush-resolved bind groups — the prefix-τ volume at group 1, the
/// noise + substrate fields at group 2. One derivation for the three passes that
/// rasterize the swept extent (the plain deposit, its scaled sibling and the
/// erase sweep), for [`sweep_draws`]' reason: they draw the *same* extent, and a
/// second copy of its inputs would be a place to disagree about what that is.
pub(super) fn sweep_binds(
    r: &StrokeRenderer,
    assets: &crate::assets::AssetStore,
    rec: &StrokeRecord,
    substrate: &crate::gpu::substrate::SubstrateMap,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let device = &r.ctx.device;
    // Resolve the brush's prefix-τ texture: image brushes from the asset
    // store; the round tip generated (and cached) from its hardness.
    let prefix_view = r.tips.prefix_view(assets, &rec.brush);
    let prefix_bg = desc::bind_group_for(
        device,
        "stark sweep prefix bg",
        &r.swept.prefix_bgl,
        PREFIX_SLOTS,
        false,
        |_| wgpu::BindingResource::TextureView(&prefix_view),
    );
    // Color dynamics (§6.2): the noise tile for this brush and the stroke's
    // lookup parameters. An inactive brush binds the zero tile with zero
    // amplitudes — the deposit is exactly the constant color. The canvas
    // substrate beside it (§6.4): the deposition tooth's height and the rise
    // ahead of it, in the same group because it is the same kind of thing — a
    // field the deposit samples per fragment.
    let noise_view = r.tips.noise_view(&rec.brush.color_dynamics());
    let noise_bg = desc::bind_group_for(
        device,
        "stark sweep noise bg",
        &r.swept.noise_bgl,
        NOISE_SLOTS,
        false,
        |b| match b {
            sc::NOISE_TEX => wgpu::BindingResource::TextureView(&noise_view),
            sc::NOISE_SAMP => wgpu::BindingResource::Sampler(&r.tips.noise_sampler),
            sc::SUBSTRATE_TEX => wgpu::BindingResource::TextureView(&substrate.view),
            other => unreachable!("`NOISE_SLOTS` lists no binding {other}"),
        },
    );
    (prefix_bg, noise_bg)
}

/// The integrate's uniform (`integrate.wesl::Integrate`) for one piece: the paint
/// effect's ceiling — or exactly 1, the shader's identity branch, on the unscaled
/// path, which binds it because the layout names it either way — and the strength
/// the selection mask gates at ([`Gate::strength`](crate::gpu::selection::Gate)).
///
/// The two travel together because the integrate reads them together: the ceiling
/// says how much of the parcel is laid, the strength how much of that the mask lets
/// through (§6.2, §6.8).
pub(super) fn opacity_uniform(
    r: &StrokeRenderer,
    scope: &mut super::scratch::SubmitScope,
    opacity: f32,
    gate: f32,
) -> wgpu::Buffer {
    let u = stark_shaders::mirror::integrate::Integrate {
        params: [opacity, gate, 0.0, 0.0],
    };
    let buf = scope.buffer(r.ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark integrate opacity"),
        size: std::mem::size_of_val(&u) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    r.ctx.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
    buf
}

/// Everything one piece's sweep draws with, per tile: which tiles, each tile's
/// contiguous instance range, the instance buffer, and the per-tile transform
/// slots behind one dynamic-offset bind group.
///
/// One derivation for both consumers — the plain deposit ([`StrokeRenderer::render_swept`])
/// and the erase pass (`erase.rs`) — for [`StrokeConstants`]'s reason: the two
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
    /// The per-tile [`TileXform`] slots, selected by a dynamic offset of
    /// `i * UNIFORM_STRIDE` for tile `i` of [`coords`](Self::coords).
    pub(super) xforms: wgpu::BindGroup,
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
pub(super) fn sweep_draws(
    r: &StrokeRenderer,
    scope: &mut super::scratch::SubmitScope,
    rec: &StrokeRecord,
    k: &super::StrokeConstants,
    segments: &[Segment],
) -> SweepDraws {
    let device = &r.ctx.device;
    let touched = tiles_with_segments(segments);
    let coords: Vec<TileCoord> = touched.keys().copied().collect();
    let mut instances: Vec<SegmentInstance> = Vec::new();
    let mut runs: Vec<std::ops::Range<u32>> = Vec::with_capacity(touched.len());
    for idx in touched.values() {
        let from = instances.len() as u32;
        instances.extend(idx.iter().map(|&i| {
            let Segment { sweep, paint } = &segments[i as usize];
            SegmentInstance {
                start: sweep.start.to_array(),
                dir: sweep.dir.to_array(),
                // The tip's radius, which is the frame brush-local coordinates
                // are read in (§6.6, [`Sweep::radius`]) — the ramp rides beside
                // it unscaled, that being the point of its being *relative*.
                geom: [sweep.radius, sweep.length, sweep.ramp],
                extra: [sweep.orient, sweep.dist, sweep.curvature, paint.add],
                tooth_give: paint.tooth_give,
                // The solved stretch map (§6.6). Unscaled for the ramp's reason:
                // it acts on brush-local coordinates, which are already the
                // frame's own units.
                stretch: [
                    sweep.stretch.travel,
                    sweep.stretch.shear,
                    sweep.stretch.lateral,
                ],
            }
        }));
        runs.push(from..instances.len() as u32);
    }
    // Written via `write_buffer` (not `create_buffer_init`, which maps-at-creation):
    // a long stroke makes this buffer large, and Chrome/Dawn caps map-at-creation
    // buffers well below the normal `maxBufferSize`, so a long stroke would panic
    // in `createBuffer`.
    let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
    let instance_buf = scope.buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark sweep instances"),
        size: instance_bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    r.ctx.queue.write_buffer(&instance_buf, 0, instance_bytes);

    // Per-tile sweep transforms, one [`UNIFORM_STRIDE`] slot each in a single
    // buffer the draws select with a dynamic offset. The texture top-left is
    // the interior origin shifted out by the apron, so the full TILE_TEX target
    // maps to NDC [-1, 1]; everything else is a stroke constant, repeated per slot
    // because the slot is what the shader reads.
    //
    // One buffer and one bind group for the stroke, not one of each per tile: this
    // path redraws on every pointer move, and the allocation *rate* is what OOMs
    // the tab (see [`ScopedResources`] and [`UNIFORM_STRIDE`]).
    let apron = TILE_APRON as f32;
    let mut xform_data = vec![0u8; coords.len() * UNIFORM_STRIDE];
    for (i, coord) in coords.iter().enumerate() {
        let origin = coord.origin();
        let xform = TileXform {
            params: [
                origin.x - apron,
                origin.y - apron,
                2.0 / TILE_TEX as f32,
                0.0,
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
            noise_off: k.noff,
            jitter_eps: k.jitter_eps,
            jitter_seed: k.jitter_seed,
            // The struct's own trailing padding, generated because the two
            // scalars above end 8 bytes short of the uniform's 16-byte round
            // (§6.10).
            _pad_9: [0; 8],
        };
        let at = i * UNIFORM_STRIDE;
        xform_data[at..at + XFORM_SLOT as usize].copy_from_slice(bytemuck::bytes_of(&xform));
    }
    let xform_buf = scope.buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark sweep xforms"),
        size: xform_data.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    r.ctx.queue.write_buffer(&xform_buf, 0, &xform_data);
    let xforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark sweep bg"),
        layout: &r.swept.uniform_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &xform_buf,
                offset: 0,
                size: wgpu::BufferSize::new(XFORM_SLOT),
            }),
        }],
    });

    SweepDraws {
        coords,
        runs,
        instances: instance_buf,
        xforms,
    }
}

// `the_draw_call_and_the_strip_agree_on_the_vertex_count` stood here. It had to check
// through `SWEEP_SLICES` rather than the shader's own `SWEEP_VERTS`, because the
// shader states that one for the host's benefit and never computes with it — so the
// linker stripped it and the check could not see it. Reading the *unlinked* source
// retires that limitation, and the assertion above holds at compile time.

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
