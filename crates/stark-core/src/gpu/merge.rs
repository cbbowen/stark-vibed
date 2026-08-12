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

use crate::colorspace::ColorSpace;
use crate::document::MergeKind;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle, TilePool};

// Generated from `merge.wesl`'s own declaration (§6.10).
use stark_shaders::mirror::merge::Merge as MergeUniform;

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

#[derive(Clone)]
pub struct MergeRenderer {
    ctx: GpuContext,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    /// The residual channel's format, or `None` in a space that has none (§6.7).
    resid_format: Option<wgpu::TextureFormat>,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// Bound for whichever side has no tile at a coordinate the other does — the
    /// §6.8 pattern, so a one-sided tile runs the same shader as a two-sided one.
    zeroes: Zeroes,
}

impl MergeRenderer {
    pub(crate) fn new(ctx: &GpuContext, color_space: &dyn ColorSpace, zeroes: Zeroes) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();
        let resid_format = color_space.resid_format();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark merge"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::merge(resid_format.is_some()).into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag), // the destination's colour
            desc::load_tex(2, frag), // …and its height
            desc::load_tex(3, frag), // the source's colour
            desc::load_tex(4, frag), // …and its height
        ];
        if resid_format.is_some() {
            entries.push(desc::load_tex(5, frag)); // the two residuals (§6.7)
            entries.push(desc::load_tex(6, frag));
        }
        let bgl = desc::bind_group_layout(device, "stark merge bgl", &entries);
        let layout = desc::pipeline_layout(device, "stark merge layout", &[Some(&bgl)]);
        let mut targets = vec![desc::target(color_format), desc::target(aux_format)];
        if let Some(f) = resid_format {
            targets.push(desc::target(f));
        }
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark merge pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &targets,
        );

        Self {
            ctx: ctx.clone(),
            color_format,
            aux_format,
            resid_format,
            pipeline,
            bgl,
            zeroes,
        }
    }

    /// The tiles of `lower` with `upper` merged into them under `kind` — what the
    /// merged layer holds.
    ///
    /// Infallible, unlike the transform's and the fill's: there is no map to be
    /// unusable and no cap to exceed, because the result spans the union of two tile
    /// sets that the document already holds.
    pub fn apply(
        &self,
        pool: &TilePool,
        lower: MergeSide<'_>,
        upper: MergeSide<'_>,
        kind: MergeKind,
    ) -> TileMap {
        let device = &self.ctx.device;
        let uniform = MergeUniform {
            p: [
                lower.opacity,
                upper.opacity,
                f32::from(u8::from(kind == MergeKind::Clip)),
                0.0,
            ],
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark merge uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark merge"),
        });

        let mut tiles = lower.tiles.clone();
        let mut drawn = false;
        for coord in self.rewritten(lower, upper, kind) {
            let src = upper.tiles.get(&coord);
            let dst = lower.tiles.get(&coord);
            // A coordinate only one side has still runs the shader — the other side
            // reads the 1×1 zero, which is exactly "no paint here" in the tile
            // representation. The pass-through cases never reach this loop.
            let color = |tile: Option<&TilePairHandle>| {
                tile.map_or_else(|| self.zeroes.color.clone(), |t| t.color_view().clone())
            };
            let aux = |tile: Option<&TilePairHandle>| {
                tile.map_or_else(|| self.zeroes.aux.clone(), |t| t.aux_view().clone())
            };
            let (dst_color, dst_aux) = (color(dst), aux(dst));
            let (src_color, src_aux) = (color(src), aux(src));
            let out = (
                pool.acquire_tex(self.color_format, AllocSource::MergeDestination),
                pool.acquire_tex(self.aux_format, AllocSource::MergeDestination),
                self.resid_format
                    .map(|f| pool.acquire_tex(f, AllocSource::MergeDestination)),
            );
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                desc::tex(1, &dst_color),
                desc::tex(2, &dst_aux),
                desc::tex(3, &src_color),
                desc::tex(4, &src_aux),
            ];
            // The residual, where the space has one — read through the same
            // absent-side stand-in as the colour it belongs to.
            let resid = self.zeroes.resid.as_ref().map(|zero| {
                let pick = |t: Option<&TilePairHandle>| {
                    t.and_then(TilePairHandle::resid_view)
                        .unwrap_or(zero)
                        .clone()
                };
                (pick(dst), pick(src))
            });
            if let Some((d, s)) = &resid {
                entries.push(desc::tex(5, d));
                entries.push(desc::tex(6, s));
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark merge bg"),
                layout: &self.bgl,
                entries: &entries,
            });
            let (attachments, n) = desc::tile_attachments(
                out.0.view(),
                out.1.view(),
                out.2.as_ref().map(crate::gpu::tile::TexHandle::view),
                desc::CLEAR,
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark merge tile"),
                color_attachments: &attachments[..n],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            drawn = true;
            tiles = tiles.insert(coord, TilePairHandle::new(out.0, out.1, out.2));
        }

        // Tiles the source alone has, which the shader would only be copying. Under
        // `Clip` they are deleted rather than copied — a clipped layer has no backdrop
        // there, so nothing of it survives — and at full opacity under `Over` the
        // source's own handle *is* the answer, which is what keeps a merge onto virgin
        // canvas free of both a pass and a texture.
        if kind == MergeKind::Over && upper.opacity >= 1.0 {
            for (coord, handle) in upper.tiles.iter() {
                if lower.tiles.get(coord).is_none() {
                    tiles = tiles.insert(*coord, handle.clone());
                }
            }
        }

        if drawn {
            self.ctx.queue.submit([encoder.finish()]);
        }
        tiles
    }

    /// The coordinates a pass has to be encoded for: everything both sides touch,
    /// less the tiles that pass through by handle.
    ///
    /// Stated as one list rather than as conditions inside the loop, because "which
    /// tiles change" is also what the caller's footprint and the history's tile diff
    /// are about (§12.6) — a tile that keeps its handle is a tile the undo has nothing
    /// to restore.
    fn rewritten(
        &self,
        lower: MergeSide<'_>,
        upper: MergeSide<'_>,
        kind: MergeKind,
    ) -> Vec<crate::geom::TileCoord> {
        let mut out: Vec<_> = lower
            .tiles
            .keys()
            .filter(|c| {
                // The destination's own tiles are left alone where the source has
                // nothing to add *and* the destination's slider changes nothing —
                // the shader's own untouched branch, hoisted to whole tiles.
                upper.tiles.get(c).is_some() || lower.opacity < 1.0
            })
            .copied()
            .collect();
        // …and the source's, where the destination has none. `Over` at full opacity
        // hands the handle across instead (see `apply`); `Clip` deletes them.
        if kind == MergeKind::Over && upper.opacity < 1.0 {
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
