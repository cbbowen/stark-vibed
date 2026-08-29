//! GPU execution of a layer merge-down (§14.11).
//!
//! [`MergeRenderer::apply`] takes the two layers' tile maps and their opacities and
//! produces the one tile map that composites to the same texels — copy-on-write like
//! every other rewrite, so old history versions keep their tiles and the pool reclaims
//! what falls out of reach.
//!
//! The arithmetic is `merge.wesl`'s and the rule is
//! [`document::merge`](crate::document::merge)'s; what is left here is the tile
//! bookkeeping, and the one part of it worth stating is **what this does not draw**.
//! A merge is normally lopsided — a few tiles of stroke onto a background that spans
//! the canvas — so a tile only one side has, and which that side's opacity leaves
//! unchanged, is passed through **by handle**. No pass is encoded and no texture is
//! acquired for it, which means merging onto a large layer costs the tiles the small
//! one actually covers rather than the tiles the large one has. The `Clip` law goes
//! further: a clipped layer is deleted where its backdrop has no coverage, so a tile
//! the destination lacks is not merged at all.
//!
//! Like the other renderers this holds only immutable GPU objects, so it is cheap to
//! `Clone` and rides in the `Action::Context` (§5).

use wgpu::util::DeviceExt;

use std::sync::Arc;

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Channels, Targets};
use crate::gpu::composite::{BlendPass, BlendUniform, FilterDraw, FilterPass, FilterUniform};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::submit::TileScope;
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle, TilePool};
use crate::gpu::uniforms::UniformSlots;
use crate::view::ViewTransform;
use stark_model::document::BlendMode;

// Generated from the shaders' own declarations (§6.10).
use stark_shaders::mirror::merge::Merge as MergeUniform;
use stark_shaders::mirror::merge::binding as m;
use stark_shaders::mirror::merge::decl as md;
use stark_shaders::mirror::slab::Slab as SlabUniform;
use stark_shaders::mirror::slab::binding as sl;
use stark_shaders::mirror::slab::decl as sd;

/// Which bindings `merge.wesl` reads, in layout order (§6.10).
///
/// One list, read by both sides: [`layout_for`](desc::layout_for) builds the layout
/// from it and [`bind_group_for`](desc::bind_group_for) the group, so neither can
/// disagree with the other about which slots are present or of what type. The two
/// residual entries sit beside the colors they ride with rather than in a countable
/// tail — the `@if(resid)` gate is on the declaration, so the `if resid { push }` this
/// replaces had nothing left to say (§6.7).
const MERGE_SLOTS: &[desc::Slot] = &[
    desc::Slot::at(md::M),
    desc::Slot::at(md::LOWER_COLOR),
    desc::Slot::at(md::LOWER_AUX),
    desc::Slot::at(md::UPPER_COLOR),
    desc::Slot::at(md::UPPER_AUX),
    desc::Slot::at(md::LOWER_RESID),
    desc::Slot::at(md::UPPER_RESID),
];

/// Which bindings `slab.wesl` reads — one list for both directions, since they take
/// the same shapes in and put the same shapes out, which is what makes them one module.
const SLAB_SLOTS: &[desc::Slot] = &[
    desc::Slot::at(sd::S),
    desc::Slot::at(sd::IN_COLOR),
    desc::Slot::at(sd::IN_AUX),
    desc::Slot::at(sd::IN_RESID),
];

/// A texture view as the resource a bind-group entry takes.
fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

/// One side of a merge: a layer's tiles and the opacity slider that scales them.
///
/// The two travel together because neither means anything without the other here —
/// the pass folds the slider into the tiles, which is what lets the merged layer come
/// out at full opacity (§14.11) — and taking them as one is what stops a call site
/// pairing the destination's map with the source's slider.
#[derive(Copy, Clone)]
pub struct MergeSide<'a> {
    pub tiles: &'a TileMap,
    pub opacity: f32,
}

/// One merge, described: the two layers and how the upper's paint meets the lower's.
///
/// The four travel together because they are one description of the operation, decided
/// in one place — [`merge::plan`](crate::document::merge::plan) — and meaningless
/// apart. `blend` and `clip` are the **source layer's own** (§14.11): a merge folds the
/// upper layer into the lower through exactly the merge the compositor would have run
/// between them.
#[derive(Copy, Clone)]
pub struct MergeScene<'a> {
    pub lower: MergeSide<'a>,
    pub upper: MergeSide<'a>,
    pub blend: BlendMode,
    pub clip: bool,
}

impl MergeScene<'_> {
    /// Whether this merge is settled in tile space directly, with no trip out to the
    /// composited representation and back — a `Normal` merge, clipped or not, which is
    /// the ordinary one and the whole of what `merge.wesl` is written for (§14.11.3).
    ///
    /// The **clip is not a reason to go the long way round**: it rides as a flag the
    /// shader branches on, so both of §14.11.3's two laws are settled here. Only the
    /// blend mode decides, because only a mode needs the composited representation
    /// the tile does not carry.
    fn is_direct(&self) -> bool {
        self.blend.is_normal()
    }
}

#[derive(Clone)]
pub struct MergeRenderer {
    ctx: GpuContext,
    /// The channel formats this merge's tiles carry — the color space's, resolved
    /// once (§6.7).
    formats: ChannelFormats,
    /// The direct tile-space law: an unclipped `Normal` merge, settled in one pass
    /// (`merge.wesl`).
    direct: wgpu::RenderPipeline,
    direct_bgl: wgpu::BindGroupLayout,
    /// The slab law's two directions (`slab.wesl`), which is what lets a merge through
    /// a blend mode borrow the compositor's own pass rather than restate its algebra.
    expand: wgpu::RenderPipeline,
    store: wgpu::RenderPipeline,
    slab_bgl: wgpu::BindGroupLayout,
    /// **The compositor's blend pass, shared** — the same pipeline the screen runs,
    /// pointed at tile-sized targets (§14.11). A merged layer therefore cannot drift
    /// from the stack it replaced, which no amount of care in a second implementation
    /// could promise.
    blend: Arc<BlendPass>,
    /// **The compositor's filter pass, shared** — on `blend`'s argument exactly, and
    /// for the entry point beside the one the screen runs: `fs_tile` adjusts a tile's
    /// stored channels in place, which is the whole of merging a filter layer into
    /// the paint beneath it (§14.11.7).
    filter: Arc<FilterPass>,
    /// Bound for whichever side has no tile at a coordinate the other does — the
    /// §6.8 pattern, so a one-sided tile runs the same shader as a two-sided one.
    zeroes: Zeroes,
}

impl MergeRenderer {
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        zeroes: Zeroes,
        blend: Arc<BlendPass>,
        filter: Arc<FilterPass>,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space);
        let resid = formats.has_resid();
        let frag = wgpu::ShaderStages::FRAGMENT;

        // The channel targets both of this module's own passes write.
        let targets = formats.targets();

        let merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark merge"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::merge(resid).into()),
        });
        let direct_bgl = desc::layout_for(device, "stark merge bgl", MERGE_SLOTS, frag, resid);
        let direct = desc::fullscreen_pipeline(
            device,
            "stark merge pipeline",
            &desc::pipeline_layout(device, "stark merge layout", &[Some(&direct_bgl)]),
            &merge_shader,
            ("vs_main", "fs_main"),
            &targets,
        );

        // One layout for both slab directions: they take the same shapes in and put
        // the same shapes out, which is what makes them one module (`slab.wesl`).
        let slab_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark slab"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::slab(resid).into()),
        });
        let slab_bgl = desc::layout_for(device, "stark slab bgl", SLAB_SLOTS, frag, resid);
        let slab_layout = desc::pipeline_layout(device, "stark slab layout", &[Some(&slab_bgl)]);
        let slab = |label, fs| {
            desc::fullscreen_pipeline(
                device,
                label,
                &slab_layout,
                &slab_shader,
                ("vs_main", fs),
                &targets,
            )
        };

        Self {
            ctx: ctx.clone(),
            formats,
            direct,
            direct_bgl,
            expand: slab("stark slab expand", "fs_expand"),
            store: slab("stark slab store", "fs_store"),
            slab_bgl,
            blend,
            filter,
            zeroes,
        }
    }

    /// The tiles of `scene.lower` with `scene.upper` merged into them — what the
    /// merged layer holds.
    ///
    /// Infallible, unlike the transform's and the fill's: there is no map to be
    /// unusable and no cap to exceed, because the result spans the union of two tile
    /// sets that the document already holds.
    pub fn apply(&self, pool: &TilePool, scene: MergeScene<'_>) -> TileMap {
        let MergeScene {
            lower,
            upper,
            blend,
            clip,
        } = scene;

        // Uniforms, written once for the whole merge rather than once per tile: every
        // tile of one merge runs the same law on the same two sliders.
        let direct_uniform = self.uniform(
            "stark merge uniform",
            &MergeUniform {
                p: [lower.opacity, upper.opacity, f32::from(u8::from(clip)), 0.0],
            },
        );
        let expand_lower = self.uniform("stark slab expand", &slab_uniform(lower.opacity));
        let expand_upper = self.uniform("stark slab expand", &slab_uniform(upper.opacity));
        // `store` folds no slider — it writes tiles at opacity 1 and leaves the fade to
        // the surviving layer — so its slot is the identity.
        let store_uniform = self.uniform("stark slab store", &slab_uniform(1.0));
        // The blend pass reads a **dynamic-offset** slot, so its buffer is one slot
        // wide and this merge binds the first. The layer's own opacity is already
        // inside the expansion, so the merge itself runs at 1.
        let blend_uniform = self.blend_uniform(blend, clip);

        // The five uniforms above are per *merge* and outlive every flush below,
        // which is what lets the recording be cut at any tile boundary.
        let mut scope = TileScope::new(&self.ctx, "stark merge");
        let mut tiles = lower.tiles.clone();
        for coord in self.rewritten(&scene) {
            let src = upper.tiles.get(&coord);
            let dst = lower.tiles.get(&coord);
            let out = self.acquire(pool, AllocSource::MergeDestination);
            if scene.is_direct() {
                self.encode_direct(&mut scope, &direct_uniform, dst, src, &out);
            } else {
                self.encode_blended(
                    &mut scope,
                    pool,
                    Uniforms {
                        expand: (&expand_lower, &expand_upper),
                        store: &store_uniform,
                        blend: &blend_uniform,
                    },
                    (dst, src),
                    &out,
                );
            }
            tiles = tiles.insert(coord, out.into_tile());
            // Everything this tile needs is recorded, so this is the one point at
            // which the scratch behind it is safe to hand back.
            scope.tile_done();
        }

        // Tiles the source alone has, which every pass here would only be copying. A
        // layer over an empty backdrop is that layer whatever its mode — the identity
        // `tests/blend.rs` pins to the byte — so at full opacity the source's own
        // handle *is* the answer, and merging onto virgin canvas costs neither a pass
        // nor a texture. **Clipping is the exception**: a clipped layer has no backdrop
        // there, so nothing of it survives and the tile is dropped rather than copied.
        if !clip && upper.opacity >= 1.0 {
            for (coord, handle) in upper.tiles.iter() {
                if lower.tiles.get(coord).is_none() {
                    tiles = tiles.insert(*coord, handle.clone());
                }
            }
        }

        scope.finish();
        tiles
    }

    /// The tiles of `dest` with `draw`'s filter run over each — what a **filter
    /// layer** merged into the paint beneath it leaves behind (§14.11.7).
    ///
    /// A different shape from [`apply`](Self::apply) beside it, and the difference is
    /// the operation's: nothing is stacked, so there is no second tile map, no
    /// coverage arithmetic and no slab conversion. One pass per tile rewrites the
    /// channels and copies everything else, which is what a filter does to what it
    /// sits on — see the tile entry point in `filter_oklab.wesl` for why a tile needs
    /// no trip out to composite space to be filtered.
    ///
    /// **Every tile is rewritten**, with no passthrough-by-handle: a filter has an
    /// opinion about every texel it can reach, so there is no counterpart to the
    /// lopsided-merge shortcut above. The one tile that costs nothing is the one that
    /// does not exist — an empty destination merges to an empty destination.
    pub fn apply_filter(&self, pool: &TilePool, dest: &TileMap, draw: &FilterDraw) -> TileMap {
        debug_assert!(
            draw.kind != stark_shaders::mirror::filter_common::FILTER_CHROMATIC,
            "a resampling filter cannot be merged (§14.11.7) — `merge::plan` declines              it, because no apron makes a gather a function of canvas position (§6.4)",
        );
        let uniform = self.filter_uniform(draw);
        let mut scope = TileScope::new(&self.ctx, "stark merge filter");
        let mut tiles = dest.clone();
        for (coord, handle) in dest.iter() {
            let out = self.acquire(pool, AllocSource::MergeDestination);
            self.encode_filter(&mut scope, &uniform, handle, &out);
            tiles = tiles.insert(*coord, out.into_tile());
            scope.tile_done();
        }
        scope.finish();
        tiles
    }

    /// One tile's channels through the compositor's own filter shader, tile-space
    /// entry point (§14.11.7).
    ///
    /// The bind group is [`Compositor::encode_filter`]'s, slot for slot, because it is
    /// the same layout: the tile's three channel textures stand where the
    /// accumulator's do, and the pigment LUT is the blend pass's, as it is on screen.
    /// The sampler at 3 is bound and never read — `fs_tile` takes no taps — because a
    /// bind group answers to the whole layout.
    ///
    /// [`Compositor::encode_filter`]: crate::gpu::composite
    /// The tile is not an `Option` where the merge's other encoders take one: a
    /// filter layer is *defined* as a function of the paint beneath it (§21), so a
    /// tile the lower layer does not have is a tile this merge never plans — the
    /// caller walks the lower map's own coords. Bare canvas is a real case for the
    /// blend and has no meaning here.
    fn encode_filter(
        &self,
        scope: &mut TileScope,
        uniform: &UniformSlots<FilterUniform>,
        tile: &TilePairHandle,
        out: &Channels,
    ) {
        // **The filter pass's own group**, built by the filter pass: a merged filter
        // runs the very pipeline the screen would (§14.11.7), so it binds the very
        // group rather than a second description of one.
        let bg = self.filter.bind_group(
            &self.ctx.device,
            uniform.resource(),
            Targets {
                color: tile.color_view(),
                aux: tile.aux_view(),
                resid: tile.resid_view().filter(|_| self.formats.has_resid()),
            },
            &self.blend.pigment,
        );
        pass(
            scope,
            "stark merge filter",
            &self.filter.tile,
            &bg,
            &[0],
            out,
        );
    }

    /// The filter pass's uniform, in a buffer wide enough for its dynamic-offset slot.
    ///
    /// [`blend_uniform`](Self::blend_uniform)'s twin, and it differs in one number:
    /// this uniform is wider than a slot (it carries the gradient map's ramp), so the
    /// buffer is sized to the struct rather than to `UNIFORM_SLOT`. The `disp` lane is
    /// zero and stays zero — it is the *view's* number, and the only kind that reads
    /// it is the one this merge refuses.
    fn filter_uniform(&self, draw: &FilterDraw) -> UniformSlots<FilterUniform> {
        // The compositor's own assembly, at the identity view. The one lane that is a
        // fact about the *view* is the chromatic gather's dispersion — measured in
        // screen px (§21.10) — and a merge has no view: `merge::plan` refuses the
        // chromatic filter outright, since a filter baked into a tile cannot depend on
        // how the tile is being looked at (§14.11.7). So a view that disperses by
        // nothing is not a stand-in here, it is the only one that means anything.
        self.slot(
            "stark merge filter uniform",
            &crate::gpu::composite::filter_uniform(
                draw,
                ViewTransform::identity(stark_model::geom::Extent2::new(1, 1)),
            ),
        )
    }

    /// The direct tile-space law: one pass, `merge.wesl` (§14.11).
    fn encode_direct(
        &self,
        scope: &mut TileScope,
        uniform: &wgpu::Buffer,
        dst: Option<&TilePairHandle>,
        src: Option<&TilePairHandle>,
        out: &Channels,
    ) {
        // Both sides through the one "a tile, or the 1×1 zeroes" answer
        // (`Zeroes::or`): a layer that has no tile at this coord reads the stand-in,
        // which is what lets `merge.wesl` be one shader whatever exists (§6.8).
        let (lower, upper) = (self.views_of(dst), self.views_of(src));
        fn resid<'a>(t: Targets<'a>) -> &'a wgpu::TextureView {
            t.resid
                .expect("a residual is asked for only in a space that has one")
        }
        let bg = desc::bind_group_for(
            &self.ctx.device,
            "stark merge bg",
            &self.direct_bgl,
            MERGE_SLOTS,
            self.formats.has_resid(),
            |b| match b {
                m::M => uniform.as_entire_binding(),
                m::LOWER_COLOR => view(lower.color),
                m::LOWER_AUX => view(lower.aux),
                m::UPPER_COLOR => view(upper.color),
                m::UPPER_AUX => view(upper.aux),
                m::LOWER_RESID => view(resid(lower)),
                m::UPPER_RESID => view(resid(upper)),
                other => unreachable!("`MERGE_SLOTS` lists no binding {other}"),
            },
        );
        pass(scope, "stark merge tile", &self.direct, &bg, &[], out);
    }

    /// The general law: expand both sides into what they composite to, run the
    /// **compositor's own blend pass** between them, and store the result back as a
    /// tile (§14.11).
    ///
    /// Four passes and three scratch trios per tile where the direct path takes one and
    /// none — which is the right trade for an action, not a frame: a merge runs once
    /// over the tiles the two layers share, and what it buys is that the merged tile is
    /// produced by the very shader the screen would have run.
    fn encode_blended(
        &self,
        scope: &mut TileScope,
        pool: &TilePool,
        u: Uniforms<'_>,
        (dst, src): (Option<&TilePairHandle>, Option<&TilePairHandle>),
        out: &Channels,
    ) {
        // Acquired *as scratch*, so each is registered with the recording that
        // names it and cannot be released before the submit (`Channels::scratch`).
        let lower = self.scratch(scope, pool, AllocSource::MergeScratch);
        let upper = self.scratch(scope, pool, AllocSource::MergeScratch);
        let blended = self.scratch(scope, pool, AllocSource::MergeScratch);
        self.encode_slab(scope, &self.expand, u.expand.0, self.views_of(dst), &lower);
        self.encode_slab(scope, &self.expand, u.expand.1, self.views_of(src), &upper);
        self.encode_blend(scope, u.blend, &lower, &upper, &blended);
        self.encode_slab(scope, &self.store, u.store, blended.targets(), out);
    }

    /// One direction of the slab law over one tile (`slab.wesl`).
    fn encode_slab(
        &self,
        scope: &mut TileScope,
        pipeline: &wgpu::RenderPipeline,
        uniform: &wgpu::Buffer,
        input: Targets<'_>,
        out: &Channels,
    ) {
        let bg = desc::bind_group_for(
            &self.ctx.device,
            "stark slab bg",
            &self.slab_bgl,
            SLAB_SLOTS,
            input.resid.is_some(),
            |b| match b {
                sl::S => uniform.as_entire_binding(),
                sl::IN_COLOR => view(input.color),
                sl::IN_AUX => view(input.aux),
                sl::IN_RESID => view(input.resid.expect("a residual build has one")),
                other => unreachable!("`SLAB_SLOTS` lists no binding {other}"),
            },
        );
        pass(scope, "stark slab tile", pipeline, &bg, &[], out);
    }

    /// The compositor's blend pass, on tile-sized targets.
    ///
    /// The bind group is this module's; the layout, the pipeline and the pigment LUT
    /// are the compositor's, borrowed whole. Binding 0 is a dynamic-offset slot there
    /// (several merges share one buffer in a frame), and there is exactly one here, so
    /// the offset is always the first.
    fn encode_blend(
        &self,
        scope: &mut TileScope,
        uniform: &UniformSlots<BlendUniform>,
        back: &Channels,
        src: &Channels,
        out: &Channels,
    ) {
        // **The compositor's own group**, built by the compositor (§18.0.4) — which is
        // the whole argument for merging through this pass rather than restating its
        // algebra (§14.11), and was not true while this file spelled the group out
        // arm for arm beside it.
        let bg = self.blend.bind_group(
            &self.ctx.device,
            uniform.resource(),
            back.targets(),
            src.targets(),
        );
        pass(
            scope,
            "stark merge blend",
            &self.blend.pipeline,
            &bg,
            &[0],
            out,
        );
    }

    /// A tile's three channel textures, or the 1×1 zeroes where the layer has none.
    fn views_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> Targets<'a> {
        self.zeroes.or(tile.map(TilePairHandle::targets))
    }

    fn acquire(&self, pool: &TilePool, source: AllocSource) -> Channels {
        Channels::acquire(pool, self.formats, source)
    }

    /// [`acquire`](Self::acquire) for a trio the recording reads back — registered
    /// with the scope, so it outlives the submit by construction.
    fn scratch(&self, scope: &mut TileScope, pool: &TilePool, source: AllocSource) -> Channels {
        Channels::scratch(scope, pool, self.formats, source)
    }

    fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// The blend pass's uniform, in a buffer wide enough for its dynamic-offset slot.
    fn blend_uniform(&self, blend: BlendMode, clip: bool) -> UniformSlots<BlendUniform> {
        // The compositor's own assembly. The source layer's own opacity is folded in
        // by its expansion, which is what pass A would have done to its tiles — so the
        // merge itself has nothing left to fade.
        self.slot(
            "stark merge blend uniform",
            &crate::gpu::composite::blend_uniform(blend, clip, 1.0),
        )
    }

    /// One uniform in a buffer of exactly one slot.
    ///
    /// [`UniformSlots`] rather than a hand-rolled buffer of [`UNIFORM_SLOT`] bytes,
    /// which is what this was: the layouts a merge borrows are the screen's and
    /// declare a dynamic offset, since several merges share one buffer in a frame —
    /// and the type that answers "what is a dynamic-offset slot" already exists, gets
    /// the stride from the uniform rather than from a constant, and is what the screen
    /// side binds. A merge has one merge in flight, so the count is one and the offset
    /// is always the first.
    fn slot<T: bytemuck::Pod>(&self, label: &'static str, uniform: &T) -> UniformSlots<T> {
        let mut slots = UniformSlots::new(&self.ctx.device, label, 1);
        slots.write(
            &self.ctx.device,
            &self.ctx.queue,
            std::slice::from_ref(uniform),
        );
        slots
    }

    /// The coordinates a pass has to be encoded for: everything both sides touch,
    /// less the tiles that pass through by handle.
    ///
    /// Stated as one list rather than as conditions inside the loop, because "which
    /// tiles change" is also what the caller's footprint and the history's tile diff
    /// are about (§12.6) — a tile that keeps its handle is a tile the undo has nothing
    /// to restore.
    fn rewritten(&self, scene: &MergeScene<'_>) -> Vec<stark_model::geom::TileCoord> {
        let MergeScene { lower, upper, .. } = *scene;
        let mut out: Vec<_> = lower
            .tiles
            .keys()
            .filter(|c| {
                // The destination's own tiles are left alone where the source has
                // nothing to add — a transparent source is the identity of every mode,
                // clipped or not — *and* where the destination's own slider changes
                // nothing.
                upper.tiles.get(c).is_some() || lower.opacity < 1.0
            })
            .copied()
            .collect();
        // …and the source's, where the destination has none. An unclipped merge at
        // full opacity hands the handle across instead (see `apply`); a clipped one
        // deletes them.
        if !scene.clip && upper.opacity < 1.0 {
            out.extend(
                upper
                    .tiles
                    .keys()
                    .filter(|c| lower.tiles.get(c).is_none())
                    .copied(),
            );
        }
        // Sorted so the encoding order is a function of the document rather than of a
        // hash seed. Nothing here depends on the order — the tiles are disjoint — but
        // a deterministic one is what keeps a captured command stream comparable
        // between runs when something else goes wrong (§12.1).
        out.sort_unstable_by_key(|c| (c.y, c.x));
        out
    }
}

/// One fullscreen pass over a tile's three channel targets.
///
/// Thin, because [`TileScope::fullscreen_pass`] carries the attachment count — the
/// residual's `Option` (§6.7) — for this pass, the transform and the fill alike,
/// rather than each deciding it again.
fn pass(
    scope: &mut TileScope,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    offsets: &[u32],
    out: &Channels,
) {
    scope.fullscreen_pass(label, pipeline, bg, offsets, out.targets(), desc::CLEAR);
}

/// The four uniform slots one blended merge binds, so `encode_blended` takes one
/// parameter rather than four that could be passed in the wrong order.
#[derive(Copy, Clone)]
struct Uniforms<'a> {
    /// The two expansions', lower then upper — the one place the two sides differ.
    expand: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    store: &'a wgpu::Buffer,
    blend: &'a UniformSlots<BlendUniform>,
}

fn slab_uniform(opacity: f32) -> SlabUniform {
    SlabUniform {
        p: [opacity, 0.0, 0.0, 0.0],
    }
}
