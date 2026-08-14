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
use crate::gpu::composite::{BlendPass, BlendUniform, UNIFORM_SLOT, blend_code};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::tile::{AllocSource, TexHandle, TileMap, TilePairHandle, TilePool};

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

/// A tile's three channel textures, as this module passes them around — the shape
/// [`TilePairHandle`] is built from, before it becomes one.
type Trio = (TexHandle, TexHandle, Option<TexHandle>);

/// A merge's command encoder together with the scratch it will read.
///
/// Nothing in a recorded encoder has *run*, so releasing a scratch handle early hands
/// its texture straight back to the pool, which gives it to the next tile in this very
/// encoder — whose expand overwrites it before the earlier tile's blend ever reads it.
/// [`TransformRenderer`](super::transform::TransformRenderer)'s `Recording` learned
/// that the hard way; this is the same guard, and it works the same way: [`submit`]
/// takes `self`, so the scratch cannot be released except by submitting first.
///
/// [`submit`]: Recording::submit
struct Recording {
    encoder: wgpu::CommandEncoder,
    scratch: Vec<TexHandle>,
    /// Whether anything was encoded at all — a merge whose every tile passed through
    /// by handle submits nothing rather than an empty command buffer.
    drawn: bool,
}

impl Recording {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            encoder: device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark merge"),
            }),
            scratch: Vec::new(),
            drawn: false,
        }
    }

    /// Hold a trio until this recording is submitted.
    fn keep(&mut self, trio: &Trio) {
        self.scratch.push(trio.0.clone());
        self.scratch.push(trio.1.clone());
        self.scratch.extend(trio.2.clone());
    }

    /// Submit, then release the scratch — in that order, which is the whole point.
    fn submit(self, queue: &wgpu::Queue) {
        if self.drawn {
            queue.submit([self.encoder.finish()]);
        }
    }
}

#[derive(Clone)]
pub struct MergeRenderer {
    ctx: GpuContext,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    /// The residual channel's format, or `None` in a space that has none (§6.7).
    resid_format: Option<wgpu::TextureFormat>,
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
    ) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();
        let resid_format = color_space.resid_format();
        let resid = resid_format.is_some();
        let frag = wgpu::ShaderStages::FRAGMENT;

        // The channel targets both of this module's own passes write.
        let mut targets = vec![desc::target(color_format), desc::target(aux_format)];
        if let Some(f) = resid_format {
            targets.push(desc::target(f));
        }

        let merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark merge"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::merge(resid).into()),
        });
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag), // the destination's colour
            desc::load_tex(2, frag), // …and its height
            desc::load_tex(3, frag), // the source's colour
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
            color_format,
            aux_format,
            resid_format,
            direct,
            direct_bgl,
            expand: slab("stark slab expand", "fs_expand"),
            store: slab("stark slab store", "fs_store"),
            slab_bgl,
            blend,
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
        let device = &self.ctx.device;
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

        let mut rec = Recording::new(device);
        let mut tiles = lower.tiles.clone();
        for coord in self.rewritten(&scene) {
            let src = upper.tiles.get(&coord);
            let dst = lower.tiles.get(&coord);
            let out = self.acquire(pool, AllocSource::MergeDestination);
            if scene.is_direct() {
                self.encode_direct(&mut rec, &direct_uniform, dst, src, &out);
            } else {
                self.encode_blended(
                    &mut rec,
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
            rec.drawn = true;
            tiles = tiles.insert(coord, TilePairHandle::new(out.0, out.1, out.2));
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

        rec.submit(&self.ctx.queue);
        tiles
    }

    /// The direct tile-space law: one pass, `merge.wesl` (§14.11).
    fn encode_direct(
        &self,
        rec: &mut Recording,
        uniform: &wgpu::Buffer,
        dst: Option<&TilePairHandle>,
        src: Option<&TilePairHandle>,
        out: &Trio,
    ) {
        let mut entries = vec![
            desc::uniform_entry(0, uniform),
            desc::tex(1, self.color_of(dst)),
            desc::tex(2, self.aux_of(dst)),
            desc::tex(3, self.color_of(src)),
            desc::tex(4, self.aux_of(src)),
        ];
        if self.resid_format.is_some() {
            entries.push(desc::tex(5, self.resid_of(dst)));
            entries.push(desc::tex(6, self.resid_of(src)));
        }
        let bg = self.bind("stark merge bg", &self.direct_bgl, &entries);
        self.pass(rec, "stark merge tile", &self.direct, &bg, &[], out);
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
        rec: &mut Recording,
        pool: &TilePool,
        u: Uniforms<'_>,
        (dst, src): (Option<&TilePairHandle>, Option<&TilePairHandle>),
        out: &Trio,
    ) {
        let lower = self.acquire(pool, AllocSource::MergeScratch);
        let upper = self.acquire(pool, AllocSource::MergeScratch);
        let blended = self.acquire(pool, AllocSource::MergeScratch);
        self.encode_slab(rec, &self.expand, u.expand.0, self.views_of(dst), &lower);
        self.encode_slab(rec, &self.expand, u.expand.1, self.views_of(src), &upper);
        self.encode_blend(rec, u.blend, &lower, &upper, &blended);
        self.encode_slab(rec, &self.store, u.store, views(&blended), out);
        rec.keep(&lower);
        rec.keep(&upper);
        rec.keep(&blended);
    }

    /// One direction of the slab law over one tile (`slab.wesl`).
    fn encode_slab(
        &self,
        rec: &mut Recording,
        pipeline: &wgpu::RenderPipeline,
        uniform: &wgpu::Buffer,
        input: (
            &wgpu::TextureView,
            &wgpu::TextureView,
            Option<&wgpu::TextureView>,
        ),
        out: &Trio,
    ) {
        let mut entries = vec![
            desc::uniform_entry(0, uniform),
            desc::tex(1, input.0),
            desc::tex(2, input.1),
        ];
        if let Some(r) = input.2 {
            entries.push(desc::tex(3, r));
        }
        let bg = self.bind("stark slab bg", &self.slab_bgl, &entries);
        self.pass(rec, "stark slab tile", pipeline, &bg, &[], out);
    }

    /// The compositor's blend pass, on tile-sized targets.
    ///
    /// The bind group is this module's; the layout, the pipeline and the pigment LUT
    /// are the compositor's, borrowed whole. Binding 0 is a dynamic-offset slot there
    /// (several merges share one buffer in a frame), and there is exactly one here, so
    /// the offset is always the first.
    fn encode_blend(
        &self,
        rec: &mut Recording,
        uniform: &wgpu::Buffer,
        back: &Trio,
        src: &Trio,
        out: &Trio,
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
            desc::tex(1, back.0.view()),
            desc::tex(2, back.1.view()),
            desc::tex(3, src.0.view()),
            desc::tex(4, src.1.view()),
            desc::tex(5, &self.blend.pigment.view),
            desc::samp(6, &self.blend.pigment.sampler),
        ];
        if let (Some(b), Some(s)) = (&back.2, &src.2) {
            entries.push(desc::tex(7, b.view()));
            entries.push(desc::tex(8, s.view()));
        }
        let bg = self.bind("stark merge blend bg", &self.blend.bgl, &entries);
        self.pass(
            rec,
            "stark merge blend",
            &self.blend.pipeline,
            &bg,
            &[0],
            out,
        );
    }

    /// One fullscreen pass over a tile's three channel targets.
    fn pass(
        &self,
        rec: &mut Recording,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        offsets: &[u32],
        out: &Trio,
    ) {
        let (attachments, n) = desc::tile_attachments(
            out.0.view(),
            out.1.view(),
            out.2.as_ref().map(TexHandle::view),
            desc::CLEAR,
        );
        let mut pass = rec.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &attachments[..n],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, offsets);
        pass.draw(0..3, 0..1);
    }

    /// A tile's three channel textures, or the 1×1 zeroes where the layer has none.
    fn views_of<'a>(
        &'a self,
        tile: Option<&'a TilePairHandle>,
    ) -> (
        &'a wgpu::TextureView,
        &'a wgpu::TextureView,
        Option<&'a wgpu::TextureView>,
    ) {
        (
            self.color_of(tile),
            self.aux_of(tile),
            self.resid_format.is_some().then(|| self.resid_of(tile)),
        )
    }

    fn color_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        tile.map_or(&self.zeroes.color, TilePairHandle::color_view)
    }

    fn aux_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        tile.map_or(&self.zeroes.aux, TilePairHandle::aux_view)
    }

    /// The residual, in a space that has one. Bare canvas reads the 1×1 zero here
    /// exactly as it does for the colour (§6.8's pattern).
    fn resid_of<'a>(&'a self, tile: Option<&'a TilePairHandle>) -> &'a wgpu::TextureView {
        let zero = self
            .zeroes
            .resid
            .as_ref()
            .expect("a residual is asked for only in a space that has one");
        tile.and_then(TilePairHandle::resid_view).unwrap_or(zero)
    }

    fn acquire(&self, pool: &TilePool, source: AllocSource) -> Trio {
        (
            pool.acquire_tex(self.color_format, source),
            pool.acquire_tex(self.aux_format, source),
            self.resid_format.map(|f| pool.acquire_tex(f, source)),
        )
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

/// A scratch trio's own views, for the pass that reads it.
fn views(
    trio: &Trio,
) -> (
    &wgpu::TextureView,
    &wgpu::TextureView,
    Option<&wgpu::TextureView>,
) {
    (
        trio.0.view(),
        trio.1.view(),
        trio.2.as_ref().map(TexHandle::view),
    )
}
