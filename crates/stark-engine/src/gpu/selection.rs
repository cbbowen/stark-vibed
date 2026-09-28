//! Selection-mask rasterization and gathering (§6.8).
//!
//! Three small pieces, all color-space independent (a mask is one coverage channel,
//! whatever the paint is made of), so this survives a color-space rebuild untouched:
//!
//! - [`SelectionRenderer::apply`] — rasterize one [`SelectionOp`] into fresh mask
//!   tiles, combining with the previous mask. Copy-on-write, exactly like paint: the
//!   old tiles stay valid in older history versions and return to the pool when the
//!   last one drops.
//! - [`SelectionRenderer::region_mask`] — gather the mask into the 1:1 canvas region
//!   the brush-dynamics stamp loop works over (§6.2).
//! - [`SelectionRenderer::constant`] — the 1×1 textures standing in for "there is no
//!   tile here". Every consumer clamps its load to the bound texture's own extent, so
//!   the same shader code reads a real tile and a constant.
//!
//! Like [`StrokeRenderer`](super::stroke::StrokeRenderer) this holds only immutable
//! GPU objects, so it is cheap to `Clone` and can ride in the `Action::Context` (§5).

use rpds::HashTrieMap;
use stark_model::document::SelectionMode;
use std::sync::{Arc, OnceLock};

use crate::device::context::GpuContext;
use crate::device::desc;
use crate::device::tile::mask_tex_origin;
use crate::document::selection::Selection;
use crate::view::Extent2;
use stark_model::document::{SelectionOp, SelectionShape};
use stark_model::geom::{TILE_TEX, TileCoord, Vec2};

/// The lasso's closed edge list, as `selection.wesl` reads it: one texel per edge
/// holding `(a.xy, b.xy)` in canvas px. Empty for a polygon that cannot enclose area.
///
/// Here rather than in the model (§2): the document says what a lasso *is* — its
/// vertex list, and `MAX_LASSO_POINTS` bounding it — while turning one into edge
/// texels is a statement about this pass's buffer layout.
fn lasso_edges(points: &[Vec2]) -> Vec<[f32; 4]> {
    if points.len() < 3 {
        return Vec::new();
    }
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            [a.x, a.y, b.x, b.y]
        })
        .collect()
}

/// The narrowest `maxTextureDimension1D` a WebGPU adapter may report, and so the
/// longest edge list [`SelectionRenderer::edge_texture`] can upload as one row.
///
/// A floor from the spec rather than a number this build chose, which is why it is
/// not read off the adapter: an op that validates on this machine and fails on a
/// peer's is the §6.8 disagreement, so the bound has to be the *guaranteed* one.
const MIN_MAX_TEXTURE_DIM_1D: usize = 8192;

// One edge per vertex, so `MAX_LASSO_POINTS` and the row bound are directly
// comparable — and the bound is worth asserting here, where `edge_texture` sizes a
// texture straight from `edges.len()`.
const _: () = assert!(
    stark_model::document::MAX_LASSO_POINTS <= MIN_MAX_TEXTURE_DIM_1D,
    "a lasso can name more edges than a guaranteed 1-D texture row holds, so the \
     op would fail wgpu validation on some adapters instead of rasterizing"
);
use stark_shaders::Stages;
use stark_shaders::mirror::mask_region::decl as mrd;
use stark_shaders::mirror::selection as sel;
use stark_shaders::mirror::selection::binding as sb;
use stark_shaders::mirror::selection::decl as sd;

use crate::device::scratch::{BufKey, Key, LeasedTex, ScratchPool, SubmitScope};
use crate::device::tile::{AllocSource, MASK_FORMAT, TilePool};

// Generated from the two shaders' own declarations (§6.7). `selection.wesl` and
// `slice.wesl` both call theirs `Params`, which is why the mirrors are namespaced by
// the WESL module they came from.
use stark_shaders::mirror::mask_region::Region as RegionUniform;
use stark_shaders::mirror::selection::Params as MaskUniform;

// Generated from `mask_region.wesl`'s vertex parameters (§6.10).
use stark_shaders::mirror::mask_region::MaskInstance;

#[derive(Clone)]
pub(crate) struct SelectionRenderer {
    ctx: GpuContext,
    rasterize: desc::OnePass,
    region_pipeline: wgpu::RenderPipeline,
    region_view_bindings: desc::Bindings,
    region_tile_bindings: desc::Bindings,
    /// The 1×1 constant masks, one slot per quantized coverage byte, built on first
    /// ask (see [`Self::constant`]).
    ///
    /// Shared across clones, which is what lets it be a cache at all: a
    /// `SelectionRenderer` is cloned into every action's context, so a per-instance
    /// cache would be re-filled by each of them.
    constants: Arc<[OnceLock<wgpu::TextureView>; 256]>,
    /// 1×1 stand-in for the lasso edge list, bound by the analytic shapes.
    dummy_edges: wgpu::TextureView,
    scratch: ScratchPool,
}

/// The shape half of a selection rasterize: what to draw, as opposed to where.
struct RasterShape<'a> {
    /// Coverage outside the rasterized tiles (§6.8).
    outside: f32,
    /// The result's peak coverage ([`Selection::level`]).
    level: f32,
    /// The result's overall opacity ([`Selection::opacity`]) — the previous
    /// selection's, since an op never sets it. Carried rather than read off `prev`
    /// so the plan stays the one place the algebra is stated.
    opacity: f32,
    /// The result's analytic hull, as the plan computed it ([`Selection::hull`]).
    hull: Option<(Vec2, Vec2)>,
    /// The op's own half of `selection.wesl`'s uniform ([`shader_params`]) —
    /// everything but the tile origin, which the rasterize fills per tile.
    params: MaskUniform,
    /// The lasso's edge list ([`lasso_edges`]); empty for every other shape, which
    /// binds the 1×1 stand-in instead.
    edges: &'a [[f32; 4]],
}

impl SelectionRenderer {
    pub(crate) fn new(ctx: &GpuContext, scratch: ScratchPool) -> Self {
        let device = &ctx.device;

        let selection = stark_shaders::selection();
        let shader = desc::Module::new(device, "stark selection", selection);
        // The mask targets take no blend: the shader does the combine and writes
        // straight through.
        let mask_target = [desc::target(MASK_FORMAT)];
        // One op's rasterize into a mask tile (§6.8): the shape — per tile, so a
        // dynamic-offset slot rather than a buffer each — the mask it combines with,
        // and the lasso's edge list.
        let rasterize = desc::OnePass::fullscreen(
            device,
            "stark selection",
            &shader,
            (selection.vs_main, selection.fs_main),
            sd::P,
            &[sd::P],
            &mask_target,
        );

        // ---- Region gather (for the brush-dynamics stamp loop, §6.2).
        let region = stark_shaders::mask_region();
        let region_shader = desc::Module::new(device, "stark selection region", region);
        // The region gather's two groups (§6.8/§6.2) — where the region sits, and one
        // mask tile drawn into it.
        let region_stages = Stages::Render(region.vs_main, region.fs_main);
        let region_view_bindings = desc::Bindings::of(
            device,
            "stark selection region view bgl",
            region_stages,
            mrd::R,
            &[],
        );
        let region_tile_bindings = desc::Bindings::of(
            device,
            "stark selection region tile bgl",
            region_stages,
            mrd::MASK,
            &[],
        );
        let region_layout = desc::pipeline_layout_of(
            device,
            "stark selection region layout",
            &[&region_view_bindings, &region_tile_bindings],
        );
        let region_pipeline = desc::render_pipeline(
            device,
            desc::RenderPipe {
                label: "stark selection region pipeline",
                layout: &region_layout,
                module: &region_shader,
                vs: region.vs_main,
                fs: region.fs_main,
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(
                    stark_shaders::mirror::mask_region::mask_instance_layout(
                        wgpu::VertexStepMode::Instance,
                    ),
                )],
                targets: &mask_target,
            },
        );

        // The 1×1 stand-ins: the coverage that reigns where the selection has no
        // tile, and the edge list every analytic shape binds but never reads.
        let constants: Arc<[OnceLock<wgpu::TextureView>; 256]> =
            Arc::new(std::array::from_fn(|_| OnceLock::new()));
        let dummy_edges = desc::constant_texture(
            ctx,
            wgpu::TextureFormat::Rgba32Float,
            bytemuck::cast_slice(&[0.0f32; 4]),
            "stark selection dummy edges",
        );

        Self {
            ctx: ctx.clone(),
            rasterize,
            region_pipeline,
            region_view_bindings,
            region_tile_bindings,
            constants,
            dummy_edges,
            scratch,
        }
    }

    /// The 1×1 mask holding `coverage` — what consumers bind wherever the selection
    /// has no tile. Their clamped `textureLoad` then reads the constant for every
    /// texel, so nothing branches on whether a mask exists.
    ///
    /// Quantized to a byte, which is not a loss: [`MASK_FORMAT`] is `R8Unorm`, so
    /// this is the same rounding the mask tiles themselves took, and a texel of the
    /// constant has to answer as one of theirs would.
    ///
    /// **Cached per byte, built on first ask** — 256 slots of one texel.
    /// [`mask_for`](Self::mask_for) asks once per tile of every fill, transform and
    /// stroke under a selection, and a partially selected plane (reachable only by
    /// inverting a partial selection, §6.8) asks for a byte between 0 and 255.
    pub(crate) fn constant(&self, coverage: f32) -> wgpu::TextureView {
        let byte = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
        self.constants[byte as usize]
            .get_or_init(|| {
                desc::constant_texture(
                    &self.ctx,
                    MASK_FORMAT,
                    &[byte],
                    "stark selection constant mask",
                )
            })
            .clone()
    }

    /// The mask bound for `coord`: the selection's own tile, or the constant that
    /// reigns outside its tile set.
    ///
    /// **The coverage the mask holds, and only that** — the whole mask's opacity
    /// ([`Selection::opacity`]) is *not* folded in, since it is not in the tiles.
    /// It scales every read of the mask, so a caller applies it once and elsewhere:
    /// the fill into its uniform lane, the stroke paths into the one ceiling both
    /// renderers read (§6.2). This is asked per tile and has nowhere per-tile to put
    /// a scalar. The transform must not apply it at all — it carries coverage rather
    /// than gating by it, and the opacity rides on the moved selection instead (§16).
    pub(crate) fn mask_for(&self, selection: &Selection, coord: TileCoord) -> MaskSource {
        match selection.tile(coord) {
            Some(handle) => MaskSource::Tile(handle.clone()),
            None => MaskSource::Constant(self.constant(selection.outside())),
        }
    }

    /// Apply `op` to `prev`, returning the new selection. `None` when the op's shape
    /// would need more than [`MAX_SELECTION_TILES`](stark_model::document::MAX_SELECTION_TILES)
    /// mask tiles — the caller leaves the selection alone rather than clipping it.
    pub(crate) fn apply(
        &self,
        pool: &TilePool,
        prev: &Selection,
        op: &SelectionOp,
    ) -> Option<Selection> {
        let plan = prev.plan(op)?;
        let edges = match &op.shape() {
            SelectionShape::Lasso(points) => lasso_edges(points),
            _ => Vec::new(),
        };
        // A lasso that encloses nothing has no boundary; treat it as a no-op rather
        // than rasterizing an empty edge list (which would read as "all outside").
        if matches!(op.shape(), SelectionShape::Lasso(_)) && edges.is_empty() {
            return Some(prev.clone());
        }
        let params = shader_params(op, edges.len());

        // Union and Subtract build on the previous mask (it survives where the shape
        // does not reach); Replace and Intersect start from nothing.
        let base = if plan.keep_prev {
            prev.tile_map().clone()
        } else {
            HashTrieMap::new()
        };
        Some(self.rasterize(
            pool,
            prev,
            base,
            &plan.rasterize,
            RasterShape {
                outside: plan.outside,
                level: plan.level,
                opacity: plan.opacity,
                hull: plan.hull,
                params,
                edges: &edges,
            },
        ))
    }

    /// Invert the selection: every mask tile reflects through the mask's own level,
    /// and so does the coverage outside them. Constant cost on an unbounded canvas —
    /// the whole point of carrying `outside` as one number (§6.8).
    ///
    /// The level rides in the lane the shader reads an op's opacity from: an inversion
    /// has no shape, and both numbers are "the strength this mask is drawn at"
    /// (`selection.wesl`).
    pub(crate) fn invert(&self, pool: &TilePool, prev: &Selection) -> Selection {
        let plan = prev.plan_invert();
        self.rasterize(
            pool,
            prev,
            HashTrieMap::new(),
            &plan.rasterize,
            RasterShape {
                outside: plan.outside,
                level: plan.level,
                opacity: plan.opacity,
                hull: plan.hull,
                // Not a [`SelectionMode`] — inverting is not a way of combining a
                // shape, it is its own edit — so the mode is named here rather than in
                // `mode_code`. No shape and no feather: every member but these two is
                // the zero the mirror's `Default` gives.
                params: MaskUniform {
                    mode: sel::MODE_INVERT,
                    strength: plan.level,
                    ..Default::default()
                },
                edges: &[],
            },
        )
    }

    /// The usage a [`region_mask`](Self::region_mask) target needs: drawn into by the
    /// gather below, then sampled by the loop. Stated here rather than at the caller
    /// so the lease and the pass cannot disagree about it.
    pub(crate) const REGION_MASK_USAGE: wgpu::TextureUsages =
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

    /// Gather `selection` into `into` — a region-sized mask for the stamp loop,
    /// matching the region `stroke.rs` composited the paint into. Tiles the selection
    /// has no mask for are left at the clear value, so the pass draws only what
    /// actually exists.
    ///
    /// **The target is the caller's**, at [`REGION_MASK_USAGE`](Self::REGION_MASK_USAGE)
    /// and `MASK_FORMAT` — a region is up to `MAX_REGION_DIM`² of `R8Unorm`, so a live
    /// stroke leases one from the scratch pool rather than creating one per pointer
    /// move. The pass writes every texel it owns, clear included, so the pool's
    /// no-zero-init contract is met (`gpu::scratch`).
    ///
    /// The coverage alone, like [`mask_for`](Self::mask_for): the loop's ceiling
    /// already carries the mask's opacity (`Stamp::opacity`), and a mask that
    /// carried it too would gate the plane at its square.
    pub(crate) fn region_mask(
        &self,
        scope: &mut SubmitScope,
        into: &wgpu::TextureView,
        selection: &Selection,
        tiles: &[TileCoord],
        region: (Vec2, Extent2),
    ) {
        let (region_origin, size) = region;
        let (w, h) = (size.width, size.height);
        let device = &self.ctx.device;
        // Leased for the same reason the target is: a live stroke re-gathers this per
        // piece per pointer move.
        let u = scope.piece_uniform(
            "stark selection region uniform",
            &RegionUniform {
                size: [w as f32, h as f32],
                tile_tex: TILE_TEX as f32,
                ..Default::default()
            },
        );
        let view_bg =
            self.region_view_bindings
                .group(device, "stark selection region view bg", |_| u.window());

        let mut origins: Vec<MaskInstance> = Vec::new();
        let mut tile_bgs: Vec<&wgpu::BindGroup> = Vec::new();
        for coord in tiles {
            let Some(handle) = selection.tile(*coord) else {
                continue;
            };
            let off = mask_tex_origin(*coord) - region_origin;
            origins.push(MaskInstance {
                origin: off.to_array(),
            });
            // Cached on the tile: this runs per piece per pointer move, and a mask
            // tile is never rewritten, so the group built for it stays right.
            tile_bgs.push(handle.region_bg(|| {
                self.region_tile_bindings
                    .group(device, "stark selection region tile bg", |_| {
                        wgpu::BindingResource::TextureView(handle.view())
                    })
            }));
        }
        let instances = (!origins.is_empty()).then(|| {
            let bytes: &[u8] = bytemuck::cast_slice(&origins);
            let buf = scope.take_piece_buffer(BufKey {
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                label: "stark selection region instances",
            });
            scope.write_lease(&buf, bytes);
            buf
        });

        // Everything the selection has no tile for takes the constant coverage that
        // reigns there.
        let mut pass = desc::begin_pass(
            scope.encoder(),
            "stark selection region gather",
            desc::Attachments::cleared_to(into, outside_clear(selection)),
        );
        if let Some(inst) = &instances {
            pass.set_pipeline(&self.region_pipeline);
            self.region_view_bindings.set(&mut pass, &view_bg, &[]);
            pass.set_vertex_buffer(0, inst.slice(..));
            for (i, bg) in tile_bgs.iter().enumerate() {
                let idx = i as u32;
                self.region_tile_bindings.set(&mut pass, bg, &[]);
                pass.draw(desc::QUAD_VERTICES, idx..idx + 1);
            }
        }
    }

    /// Rasterize `coords` into fresh mask tiles on top of `base`, reading `prev` for
    /// the combine. One draw per tile — they are independent, so this is a single
    /// encoder with no barriers between them.
    fn rasterize(
        &self,
        pool: &TilePool,
        prev: &Selection,
        base: HashTrieMap<TileCoord, crate::device::tile::MaskHandle>,
        coords: &[TileCoord],
        shape: RasterShape<'_>,
    ) -> Selection {
        let RasterShape {
            outside,
            level,
            opacity,
            hull,
            params,
            edges,
        } = shape;
        if coords.is_empty() {
            return Selection::from_parts(base, outside, level, opacity, hull);
        }
        let device = &self.ctx.device;
        // The lasso's edge texture and the per-tile parameters both belong to the whole
        // rasterize, so both are leased for the run.
        let mut scope = self.scratch.scope(&self.ctx, "stark selection edit");
        let edge_lease;
        let edges: &wgpu::TextureView = if edges.is_empty() {
            &self.dummy_edges
        } else {
            edge_lease = self.edge_texture(&mut scope, edges);
            &edge_lease
        };
        let params: Vec<MaskUniform> = coords
            .iter()
            .map(|coord| MaskUniform {
                origin: mask_tex_origin(*coord).to_array(),
                ..params
            })
            .collect();
        let slots = scope
            .run_slots("stark selection params", &params)
            .expect("a rasterize that reaches here has a tile to draw");

        let mut tiles = base;

        for (i, coord) in coords.iter().enumerate() {
            let dst = pool.acquire_mask(AllocSource::SelectionMask);
            let prev_view = self.mask_for(prev, *coord);
            let bg = self
                .rasterize
                .bindings
                .group(device, "stark selection bg", |i| match i {
                    sb::P => slots.window(),
                    sb::PREV => wgpu::BindingResource::TextureView(prev_view.view()),
                    sb::EDGES => wgpu::BindingResource::TextureView(edges),
                    other => unreachable!("the rasterize group holds no binding {other}"),
                });
            scope.fullscreen_pass(
                "stark selection rasterize",
                dst.view(),
                &self.rasterize.pipeline,
                desc::Bound::dynamic(&bg, &[slots.offset(i)]),
            );
            tiles = tiles.insert(*coord, dst);
            scope.tile_done();
        }
        scope.finish();
        Selection::from_parts(tiles, outside, level, opacity, hull)
    }

    /// Upload the lasso's edge list as an `N×1` texture (see `selection.wesl`),
    /// **leased for the scope** at a power-of-two width.
    ///
    /// `apply` runs live through a lasso drag, so a drag settles into a handful of
    /// pooled widths rather than a texture per pointer move. The shader reads the live
    /// edge count from [`MaskUniform::edge_count`] (`sd_lasso`), so the slack texels
    /// past `edges.len()` are never loaded.
    fn edge_texture(&self, scope: &mut SubmitScope, edges: &[[f32; 4]]) -> wgpu::TextureView {
        let n = edges.len() as u32;
        let format = wgpu::TextureFormat::Rgba32Float;
        let LeasedTex { tex, view } = scope.take_run(Key {
            size: (n.next_power_of_two(), 1),
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: "stark selection lasso edges",
        });
        scope.write_texture_lease(
            &tex,
            bytemuck::cast_slice(edges),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * desc::texel_bytes(format)),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: n,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        view
    }
}

/// The mask bound for one tile, **holding whatever keeps it alive**.
///
/// A `TextureView` on its own keeps the texture alive and nothing else: the pool slot
/// behind a mask tile is reserved by its handle, so a reader left with only the view
/// after the `Selection` dropped would read a slot the pool has re-handed out — or,
/// past a `tick` that `destroy()`ed it, nothing at all. A tile therefore answers with
/// the handle, and the constant — which no pool owns — with its view.
pub(crate) enum MaskSource {
    /// The selection's own tile at this coordinate.
    Tile(crate::device::tile::MaskHandle),
    /// The constant coverage that reigns outside its tile set ([`Selection::outside`]).
    Constant(wgpu::TextureView),
}

impl MaskSource {
    /// The coverage, to bind.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        match self {
            MaskSource::Tile(handle) => handle.view(),
            MaskSource::Constant(view) => view,
        }
    }
}

/// What a fresh mask target is cleared to: the constant coverage that reigns
/// wherever `selection` has no tile of its own (§6.8).
///
/// In `r`, the only channel [`MASK_FORMAT`] has. Shared with the transform, which
/// carries masks under its maps and has to leave the untouched plane saying the
/// same thing this does — an inverted selection whose two paths disagreed about
/// the outside would gain or lose the whole canvas.
pub(crate) fn outside_clear(selection: &Selection) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(selection.outside()),
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }
}

// —— packing an op for `selection.wesl` ————————————————————————————————————————
//
// The op is a fact about the document and lives in `stark-model`; how it is packed
// into a uniform is a fact about the shader and lives here (§2) — on the side that
// has the generated mirror the codes come from (§6.10).

/// The mode's discriminant as the mask shader sees it.
///
/// The numbers are `selection.wesl`'s own, generated from its declarations (§6.10)
/// rather than transcribed: which `u32` a mode is numbered is a fact about the shader.
///
/// `u32` because `Params::mode` is one: a tag is an enumerand, and the member now
/// says so (§6.10), which is what retired the `as f32` this used to end on.
fn mode_code(mode: SelectionMode) -> u32 {
    match mode {
        SelectionMode::Replace => sel::MODE_REPLACE,
        SelectionMode::Union => sel::MODE_UNION,
        SelectionMode::Subtract => sel::MODE_SUBTRACT,
        SelectionMode::Intersect => sel::MODE_INTERSECT,
    }
}

/// The op's own half of `selection.wesl`'s uniform: its shape, kind, mode, edge count
/// and strength. The tile origin is the rasterize's, once per tile.
pub(crate) fn shader_params(op: &SelectionOp, edges: usize) -> MaskUniform {
    // The kind codes are `selection.wesl`'s, generated from its declarations
    // (§6.10) — see `SelectionMode::code` for why they are not written here.
    let (kind, shape) = match &op.shape() {
        SelectionShape::All => (sel::KIND_ALL, [0.0; 4]),
        SelectionShape::Rect { min, max } => (sel::KIND_RECT, [min.x, min.y, max.x, max.y]),
        // Radii non-negative by `SelectionShape::sanitized`, which every op passes
        // through — the shader takes them as written.
        SelectionShape::Ellipse { center, radii } => {
            (sel::KIND_ELLIPSE, [center.x, center.y, radii.x, radii.y])
        }
        SelectionShape::Lasso(_) => (sel::KIND_LASSO, [0.0; 4]),
    };
    MaskUniform {
        feather: op.feather(),
        shape,
        kind,
        mode: mode_code(op.mode()),
        edge_count: edges as u32,
        strength: op.opacity(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The edge list closes the polygon, which is what makes it a *region* rather
    /// than a polyline — `selection.wesl` counts crossings, so a loop that did not
    /// return to its start would leave the winding open and the coverage undefined.
    #[test]
    fn lasso_edges_close_the_loop() {
        let pts = vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let edges = lasso_edges(&pts);
        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges[2],
            [0.0, 1.0, 0.0, 0.0],
            "last edge returns to the start"
        );
        assert!(
            lasso_edges(&pts[..2]).is_empty(),
            "a segment encloses nothing"
        );
    }
}
