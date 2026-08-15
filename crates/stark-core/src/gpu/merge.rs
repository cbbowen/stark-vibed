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
use crate::document::BlendMode;
use crate::gpu::channels::{ChannelFormats, Channels, Targets};
use crate::gpu::composite::{
    BlendPass, BlendUniform, FilterDraw, FilterPass, FilterUniform, blend_code,
};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::submit::TileScope;
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle, TilePool};
use crate::gpu::uniforms::UNIFORM_SLOT;

// Generated from the shaders' own declarations (§6.10).
use stark_shaders::mirror::merge::Merge as MergeUniform;
use stark_shaders::mirror::slab::Slab as SlabUniform;

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
    /// composited representation and back — an unclipped `Normal` merge, which is the
    /// ordinary one and the only one `merge.wesl` knows how to do.
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
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag), // the destination's color
            desc::load_tex(2, frag), // …and its height
            desc::load_tex(3, frag), // the source's color
            desc::load_tex(4, frag), // …and its height
        ];
        if resid {
            entries.push(desc::load_tex(5, frag)); // the two residuals (§6.7)
            entries.push(desc::load_tex(6, frag));
        }
        let direct_bgl = desc::bind_group_layout(device, "stark merge bgl", &entries);
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
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag),
            desc::load_tex(2, frag),
        ];
        if resid {
            entries.push(desc::load_tex(3, frag));
        }
        let slab_bgl = desc::bind_group_layout(device, "stark slab bgl", &entries);
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
        // The blend pass reads a **dynamic-offset** slot (`UNIFORM_SLOT`), so its
        // buffer is a slot wide and this merge binds the first one. The layer's own
        // opacity is already inside the expansion, so the merge itself runs at 1.
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
            self.encode_filter(&mut scope, &uniform, Some(handle), &out);
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
    fn encode_filter(
        &self,
        scope: &mut TileScope,
        uniform: &wgpu::Buffer,
        tile: Option<&TilePairHandle>,
        out: &Channels,
    ) {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<FilterUniform>() as u64),
                }),
            },
            desc::tex(1, self.color_of(tile)),
            desc::tex(2, self.aux_of(tile)),
            desc::samp(3, &self.filter.sampler),
            desc::tex(5, &self.blend.pigment.view),
            desc::samp(6, &self.blend.pigment.sampler),
        ];
        if self.formats.has_resid() {
            entries.push(desc::tex(7, self.resid_of(tile)));
        }
        let bg = self.bind("stark merge filter bg", &self.filter.bgl, &entries);
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
    fn filter_uniform(&self, draw: &FilterDraw) -> wgpu::Buffer {
        let size = std::mem::size_of::<FilterUniform>() as u64;
        let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark merge filter uniform"),
            size: size.max(UNIFORM_SLOT),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(
            &buf,
            0,
            bytemuck::bytes_of(&FilterUniform {
                kind: draw.kind,
                strength: draw.strength,
                clip: u32::from(draw.clip),
                disp: [0.0; 2],
                params: draw.params,
                params2: draw.params2,
                stops: draw.stops.as_deref().copied().unwrap_or([[0.0; 4]; 16]),
                ..Default::default()
            }),
        );
        buf
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
        let mut entries = vec![
            desc::uniform_entry(0, uniform),
            desc::tex(1, self.color_of(dst)),
            desc::tex(2, self.aux_of(dst)),
            desc::tex(3, self.color_of(src)),
            desc::tex(4, self.aux_of(src)),
        ];
        if self.formats.has_resid() {
            entries.push(desc::tex(5, self.resid_of(dst)));
            entries.push(desc::tex(6, self.resid_of(src)));
        }
        let bg = self.bind("stark merge bg", &self.direct_bgl, &entries);
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
        let lower = self.acquire(pool, AllocSource::MergeScratch);
        let upper = self.acquire(pool, AllocSource::MergeScratch);
        let blended = self.acquire(pool, AllocSource::MergeScratch);
        self.encode_slab(scope, &self.expand, u.expand.0, self.views_of(dst), &lower);
        self.encode_slab(scope, &self.expand, u.expand.1, self.views_of(src), &upper);
        self.encode_blend(scope, u.blend, &lower, &upper, &blended);
        self.encode_slab(scope, &self.store, u.store, blended.targets(), out);
        // Held, not dropped: nothing in a recorded encoder has run, so an early
        // release hands these straight to the next tile's expand (`TileScope`).
        scope.hold(lower);
        scope.hold(upper);
        scope.hold(blended);
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
        let mut entries = vec![
            desc::uniform_entry(0, uniform),
            desc::tex(1, input.color),
            desc::tex(2, input.aux),
        ];
        if let Some(r) = input.resid {
            entries.push(desc::tex(3, r));
        }
        let bg = self.bind("stark slab bg", &self.slab_bgl, &entries);
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
        uniform: &wgpu::Buffer,
        back: &Channels,
        src: &Channels,
        out: &Channels,
    ) {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<BlendUniform>() as u64),
                }),
            },
            desc::tex(1, back.color.view()),
            desc::tex(2, back.aux.view()),
            desc::tex(3, src.color.view()),
            desc::tex(4, src.aux.view()),
            desc::tex(5, &self.blend.pigment.view),
            desc::samp(6, &self.blend.pigment.sampler),
        ];
        if let (Some(b), Some(s)) = (&back.resid, &src.resid) {
            entries.push(desc::tex(7, b.view()));
            entries.push(desc::tex(8, s.view()));
        }
        let bg = self.bind("stark merge blend bg", &self.blend.bgl, &entries);
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
        Targets {
            color: self.color_of(tile),
            aux: self.aux_of(tile),
            resid: self.formats.has_resid().then(|| self.resid_of(tile)),
        }
    }

    fn color_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        tile.map_or(&self.zeroes.color, TilePairHandle::color_view)
    }

    fn aux_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        tile.map_or(&self.zeroes.aux, TilePairHandle::aux_view)
    }

    /// The residual, in a space that has one. Bare canvas reads the 1×1 zero here
    /// exactly as it does for the color (§6.8's pattern).
    fn resid_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        let zero = self
            .zeroes
            .resid
            .as_ref()
            .expect("a residual is asked for only in a space that has one");
        tile.and_then(TilePairHandle::resid_view).unwrap_or(zero)
    }

    fn acquire(&self, pool: &TilePool, source: AllocSource) -> Channels {
        Channels::acquire(pool, self.formats, source)
    }

    fn bind(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
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
    fn blend_uniform(&self, blend: BlendMode, clip: bool) -> wgpu::Buffer {
        let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark merge blend uniform"),
            size: UNIFORM_SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(
            &buf,
            0,
            bytemuck::bytes_of(&BlendUniform {
                mode: blend_code(blend),
                k: blend.drago_k(),
                clip: u32::from(clip),
                // The source layer's own opacity is folded in by its expansion, which
                // is what pass A would have done to its tiles — so the merge itself
                // has nothing left to fade.
                opacity: 1.0,
            }),
        );
        buf
    }

    /// The coordinates a pass has to be encoded for: everything both sides touch,
    /// less the tiles that pass through by handle.
    ///
    /// Stated as one list rather than as conditions inside the loop, because "which
    /// tiles change" is also what the caller's footprint and the history's tile diff
    /// are about (§12.6) — a tile that keeps its handle is a tile the undo has nothing
    /// to restore.
    fn rewritten(&self, scene: &MergeScene<'_>) -> Vec<crate::geom::TileCoord> {
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
/// A thin call now: [`TileScope::fullscreen_pass`] carries the attachment count,
/// which is the residual's `Option` (§6.7) and used to be decided here as well as at
/// the transform and the fill.
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
    blend: &'a wgpu::Buffer,
}

fn slab_uniform(opacity: f32) -> SlabUniform {
    SlabUniform {
        p: [opacity, 0.0, 0.0, 0.0],
    }
}
