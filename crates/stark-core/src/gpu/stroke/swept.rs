//! The swept fast path (§6.2): one quad per segment, coverage integrated
//! along the sweep through a precomputed prefix-τ texture, over-blended so optical
//! depth sums exactly.
//!
//! Carries no brush state between segments, which is what makes it the fast path —
//! a range needs nothing from its predecessor but the arc length.

use crate::colorspace::ColorSpace;
use crate::document::StrokeRecord;
use crate::geom::{TILE_APRON, TILE_TEX, TileCoord};
use crate::gpu::desc;
use crate::gpu::desc::Slot;
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
const PREFIX_SLOTS: &[Slot] = &[Slot::at(sd::PREFIX_TEX)];

/// Group 2: the color-dynamics noise field and its repeat sampler (§6.2), and beside
/// them the canvas surface's ground — height and the rise ahead — with its own sampler,
/// for the deposition tooth (§6.4). In this group rather than one of its own because it
/// is the same kind of thing as the noise: a tileable field the deposit samples per
/// fragment, resolved per stroke.
const NOISE_SLOTS: &[Slot] = &[
    Slot::sampled(sd::NOISE_TEX),
    Slot::at(sd::NOISE_SAMP),
    // The ground is read **nearest** (`ground_at`, §6.4), so it needs no filtering and
    // has no sampler — which is what naming the declarations turned up here. The host
    // had been declaring a fourth entry, a filtering sampler at binding 3, and binding
    // `surface.sampler` into it; `stamp_common.wesl` declares no such slot and says on
    // its face that it does not ("No sampler: the tap is nearest"). A layout may carry
    // an entry no shader reads, so nothing failed — it was a sampler bound for a
    // lookup that stopped existing.
    Slot::at(sd::SURFACE_TEX),
];

/// The integrate pass (`integrate.wesl`, §6.2/§6.1): the layer's resident paint, the
/// stroke's scratch parcel, and the selection each is gated by.
const INTEGRATE_SLOTS: &[Slot] = &[
    Slot::at(id::BASE_COLOR),
    Slot::at(id::BASE_AUX),
    Slot::at(id::SCRATCH_COLOR),
    Slot::at(id::SCRATCH_AUX),
    Slot::at(id::SELECTION),
    Slot::at(id::BASE_RESID),
    Slot::at(id::SCRATCH_RESID),
];
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TileMap};

use super::region::tiles_with_segments;
use super::segments::{Segment, SegmentInstance, generate_segments_in};
use super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, UNIFORM_STRIDE};

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
/// a stroke's footprint into a scratch tile, and the integrate that stacks that
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
    /// and the noise + ground fields at group 2.
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) uniform_bgl: wgpu::BindGroupLayout,
    pub(super) prefix_bgl: wgpu::BindGroupLayout,
    pub(super) noise_bgl: wgpu::BindGroupLayout,
    /// The integrate (§6.2/§6.1): a fullscreen pass reading the base tile + the
    /// stroke's footprint scratch and writing `new = f(base, scratch)` into a fresh CoW
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
    // repeat sampler (§6.2), and beside it the canvas surface's ground (height +
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
            surface,
        } = scene;
        // Everything both paths share, resolved once (see [`StrokeConstants`]).
        let k = self.stroke_constants(rec, surface);
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

        // The submit scope: the per-stroke buffers and the shared scratch pair ride
        // in it, and only the `finish` that submits the commands naming them can
        // release them (`scratch::SubmitScope`). The ordering is the scope's shape,
        // not a pair of `drop`s placed after the submit and defended by a comment.
        let mut scope = self.scratch.scope(&self.ctx, "stark stroke commit");

        // Resolve the brush's prefix-τ texture: image brushes from the asset
        // store; the round tip generated (and cached) from its hardness.
        let prefix_view = self.tips.prefix_view(assets, &rec.brush);

        let device = &self.ctx.device;
        let prefix_bg = desc::bind_group_for(
            device,
            "stark sweep prefix bg",
            &self.swept.prefix_bgl,
            PREFIX_SLOTS,
            false,
            |_| wgpu::BindingResource::TextureView(&prefix_view),
        );

        // Color dynamics (§6.2): the noise tile for this brush and
        // the stroke's lookup parameters. An inactive brush binds the zero
        // tile with zero amplitudes — the deposit is exactly the constant
        // color.
        let noise_view = self.tips.noise_view(&rec.brush.color_dynamics);
        // The canvas ground beside it (§6.4): the deposition tooth's height and the
        // rise ahead of it, in the same group because it is the same kind of thing —
        // a field the deposit samples per fragment.
        let noise_bg = desc::bind_group_for(
            device,
            "stark sweep noise bg",
            &self.swept.noise_bgl,
            NOISE_SLOTS,
            false,
            |b| match b {
                sc::NOISE_TEX => wgpu::BindingResource::TextureView(&noise_view),
                sc::NOISE_SAMP => wgpu::BindingResource::Sampler(&self.tips.noise_sampler),
                sc::SURFACE_TEX => wgpu::BindingResource::TextureView(&surface.view),
                other => unreachable!("`NOISE_SLOTS` lists no binding {other}"),
            },
        );
        // Which segments reach which tile, and the instance buffer laid out to match:
        // each tile's segments contiguous, so its draw is one instance *range* rather
        // than the whole stroke. A segment writes exactly zero outside the tiles it is
        // listed under, and zero is an exact identity through both blends, so this is
        // the same picture as drawing everything everywhere — for
        // `Σ tiles-per-segment` instances instead of `segments × tiles`
        // ([`tiles_with_segments`]).
        //
        // The duplication is real but small: a segment is at most a tip wide, so it
        // appears under a handful of tiles. What it replaces grew with the *stroke*.
        let touched = tiles_with_segments(&segments);
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
                    // The **frame**, not the tip: brush-local coordinates are the
                    // volume's, and a padded one is wider than the shape inside it
                    // (§6.6, [`Sweep::frame`]). The two are the same number for
                    // every brush but a pen-oriented stamp.
                    //
                    // The ramp rides here unscaled, and that is the point of its being
                    // *relative*: the frame is the tip times a constant, so the tip's
                    // fractional growth is the frame's ([`Sweep::ramp`]).
                    geom: [sweep.frame, sweep.length, sweep.ramp],
                    extra: [sweep.orient, sweep.dist, sweep.curvature, paint.add],
                    tooth: paint.tooth,
                    // The solved stretch map (§6.6). Unscaled by the frame for the
                    // ramp's reason: it acts on brush-local coordinates, which are
                    // already in the frame's units whatever the frame is.
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
        self.ctx
            .queue
            .write_buffer(&instance_buf, 0, instance_bytes);

        let carry = StrokeCarry {
            dist: end_dist,
            tool: None,
            dirty: coords.clone(),
        };

        // Per-tile sweep transforms, one [`UNIFORM_STRIDE`] slot each in a single
        // buffer the draws below select with a dynamic offset. The texture top-left is
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
                color: k.channels,
                resid: k.resid,
                paint: [rec.brush.drain, k.grain_uv, 0.0, 0.0],
                noise_freq: k.nfreq,
                noise_amp: k.namp,
                noise_off: k.noff,
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
        self.ctx.queue.write_buffer(&xform_buf, 0, &xform_data);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep bg"),
            layout: &self.swept.uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &xform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(XFORM_SLOT),
                }),
            }],
        });

        // Footprint → cleared scratch tile: within-stroke accumulation of the parcel
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
                pass.set_bind_group(0, &bind_group, &[xform_off]);
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
            let mask_view = self.selection.mask_for(selection, *coord);
            let integrate_bg = desc::bind_group_for(
                device,
                "stark integrate bg",
                &self.swept.integrate_bgl,
                INTEGRATE_SLOTS,
                base_resid.is_some() && scratch.resid_view().is_some(),
                |b| {
                    wgpu::BindingResource::TextureView(match b {
                        ib::BASE_COLOR => base_color,
                        ib::BASE_AUX => base_aux,
                        ib::SCRATCH_COLOR => scratch.color_view(),
                        ib::SCRATCH_AUX => scratch.aux_view(),
                        ib::SELECTION => &mask_view,
                        ib::BASE_RESID => base_resid.expect("a residual build has one"),
                        ib::SCRATCH_RESID => {
                            scratch.resid_view().expect("a residual build has one")
                        }
                        other => unreachable!("`INTEGRATE_SLOTS` lists no binding {other}"),
                    })
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
