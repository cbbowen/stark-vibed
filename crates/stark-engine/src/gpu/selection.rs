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

use crate::document::selection::Selection;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use crate::gpu::{MASK_TEX, mask_tex_origin};
use crate::view::Extent2;
use stark_model::document::{SelectionOp, SelectionShape};
use stark_model::geom::{TileCoord, Vec2};

/// The lasso's closed edge list, as `selection.wesl` reads it: one texel per edge
/// holding `(a.xy, b.xy)` in canvas px. Empty for a polygon that cannot enclose area.
///
/// Here rather than in `stark_model::geom`, where it grew up: it is a statement about
/// this pass's buffer layout and nothing in the model ever read it. What the *document*
/// says about a lasso is its vertex list and the bound on how long that may be
/// (`SelectionShape`, `MAX_LASSO_POINTS`); turning one into edge texels is the
/// shader's own business.
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

// `MAX_LASSO_POINTS` is what keeps a lasso inside it, and until this assert existed
// that was a sentence in the model's doc comment with nothing behind it: the
// constant was not even re-exported, so this file could name it only in prose while
// `edge_texture` sized a texture straight from `edges.len()`. One edge per vertex,
// so the two numbers are directly comparable.
//
// Same shape and same reason as `gpu::fill`'s assert against the shader's stop
// count: a bound is worth having where the thing it bounds is built.
const _: () = assert!(
    stark_model::document::MAX_LASSO_POINTS <= MIN_MAX_TEXTURE_DIM_1D,
    "a lasso can name more edges than a guaranteed 1-D texture row holds, so the \
     op would fail wgpu validation on some adapters instead of rasterizing"
);
use stark_shaders::mirror::mask_region::decl as mrd;
use stark_shaders::mirror::selection::binding as sb;
use stark_shaders::mirror::selection::decl as sd;

/// One op's rasterize into a mask tile (§6.8): the shape, the mask it combines with,
/// and the lasso's edge list (a 1×1 stand-in for the analytic shapes).
const RASTERIZE_SLOTS: &[Slot] = &[
    // Per tile, so a dynamic-offset slot rather than a buffer each.
    Slot::dynamic(sd::P),
    Slot::at(sd::PREV),
    Slot::at(sd::EDGES),
];

/// The region gather's two groups (`mask_region.wesl`, §6.8/§6.2) — where the region
/// sits, and one mask tile drawn into it.
const REGION_VIEW_SLOTS: &[Slot] = &[Slot::at(mrd::R)];
const REGION_TILE_SLOTS: &[Slot] = &[Slot::at(mrd::MASK)];
use crate::gpu::scratch::{BufKey, ScratchPool, SubmitScope};
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TilePool};
use crate::gpu::uniforms::UniformSlots;

// Generated from the two shaders' own declarations (§6.7). `selection.wesl` and
// `slice.wesl` both call theirs `Params`, which is why the mirrors are namespaced by
// the WESL module they came from.
use stark_shaders::mirror::mask_region::Region as RegionUniform;
use stark_shaders::mirror::selection::Params as MaskUniform;

// Generated from `mask_region.wesl`'s vertex parameters (§6.10).
use stark_shaders::mirror::mask_region::MaskInstance;

/// The mode code `selection.wesl` reads for an inversion. Not a [`SelectionMode`] —
/// inverting is not a way of combining a shape, it is its own edit — which is why it
/// is named here rather than in `SelectionMode::code`, and why it is still the
/// shader's own number (§6.10) rather than a `4.0` written twice.
const MODE_INVERT: f32 = stark_shaders::mirror::selection::MODE_INVERT as f32;

#[derive(Clone)]
pub struct SelectionRenderer {
    ctx: GpuContext,
    rasterize_pipeline: wgpu::RenderPipeline,
    rasterize_bgl: wgpu::BindGroupLayout,
    region_pipeline: wgpu::RenderPipeline,
    region_view_bgl: wgpu::BindGroupLayout,
    region_tile_bgl: wgpu::BindGroupLayout,
    /// The 1×1 constant masks, one slot per quantized coverage byte, built on first
    /// ask (see [`Self::constant`]).
    ///
    /// Shared across clones and lock-free after the first write, which is what lets
    /// it be a cache at all: a `SelectionRenderer` is cloned into every action's
    /// context, so a per-instance cache would be re-filled by each of them.
    constants: Arc<[OnceLock<wgpu::TextureView>; 256]>,
    /// 1×1 stand-in for the lasso edge list, bound by the analytic shapes.
    dummy_edges: wgpu::TextureView,
    /// The scratch this renderer opens its scope on — **the one the stroke path uses
    /// too** (`gpu::scratch`), so that when its working textures do move onto the pool
    /// the two paths draw from one free list. Today it is the scope alone: the working
    /// textures still come from `TilePool` through `Channels::scratch`, so what the
    /// pool buys here is the submit-then-release ordering (§7).
    scratch: ScratchPool,
}

/// The shape half of a selection rasterize: what to draw, as opposed to where.
///
/// `b` and `c` are the shape's two uniform vectors — their meaning depends on the
/// [`SelectionShape`] being rasterized (rect corners, ellipse centre + radii, lasso
/// bounds) — and `edges` is the lasso's edge buffer, unused by the analytic shapes.
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
    b: [f32; 4],
    c: [f32; 4],
    feather: f32,
    edges: &'a wgpu::TextureView,
}

impl SelectionRenderer {
    pub(crate) fn new(ctx: &GpuContext, scratch: ScratchPool) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::selection().into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        // The mask targets take no blend: the shader does the combine and writes
        // straight through.
        let mask_target = [desc::target(MASK_FORMAT)];
        let rasterize_bgl =
            desc::layout_for(device, "stark selection bgl", RASTERIZE_SLOTS, frag, false);
        let layout =
            desc::pipeline_layout(device, "stark selection layout", &[Some(&rasterize_bgl)]);
        let rasterize_pipeline = desc::fullscreen_pipeline(
            device,
            "stark selection pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &mask_target,
        );

        // ---- Region gather (for the brush-dynamics stamp loop, §6.2).
        let region_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection region"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::mask_region().into()),
        });
        let region_view_bgl = desc::layout_for(
            device,
            "stark selection region view bgl",
            REGION_VIEW_SLOTS,
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            false,
        );
        let region_tile_bgl = desc::layout_for(
            device,
            "stark selection region tile bgl",
            REGION_TILE_SLOTS,
            frag,
            false,
        );
        let region_layout = desc::pipeline_layout(
            device,
            "stark selection region layout",
            &[Some(&region_view_bgl), Some(&region_tile_bgl)],
        );
        let region_pipeline = desc::render_pipeline(
            device,
            desc::RenderPipe {
                label: "stark selection region pipeline",
                layout: &region_layout,
                module: &region_shader,
                vs: "vs_main",
                fs: "fs_main",
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
            rasterize_pipeline,
            rasterize_bgl,
            region_pipeline,
            region_view_bgl,
            region_tile_bgl,
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
    /// **Cached per byte, built on first ask.** 0 and 255 are what nearly every
    /// selection asks for, but a *partially* selected plane — reachable by inverting
    /// a partial selection, and only that way (§6.8) — asks for a byte in between,
    /// and [`mask_for`](Self::mask_for) asks once per tile of every fill, transform
    /// and stroke under one. Creating and uploading a texture per tile for a value
    /// that never changes is the shape of cost this whole module is arranged to
    /// avoid; the cache is 256 slots of one texel.
    pub fn constant(&self, coverage: f32) -> wgpu::TextureView {
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
    /// **The coverage the mask holds**, and only that — the whole mask's opacity
    /// ([`Selection::opacity`]) is *not* folded in, since it is not in the tiles.
    /// Three kinds of reader want exactly this: one *carrying* the mask (the
    /// transform's own mask pass), one reading a shape that is not a selection at
    /// all (a fill's region, rasterized through the same shader), and one that has
    /// **a ceiling of its own to fold the opacity into** — the stroke paths, whose
    /// `stroke_constants` already multiplies it into the one ceiling both renderers
    /// read (§6.2). A reader with no such ceiling takes [`gate_for`](Self::gate_for)
    /// instead, so the scalar cannot be left on the floor.
    pub fn mask_for(&self, selection: &Selection, coord: TileCoord) -> wgpu::TextureView {
        match selection.tile(coord) {
            Some(handle) => handle.view().clone(),
            None => self.constant(selection.outside()),
        }
    }

    /// The mask bound for `coord` **as a gate** — the coverage, and the opacity
    /// every read of it must be scaled by (§6.8). For a pass with no ceiling of its
    /// own to carry the opacity in; see [`mask_for`](Self::mask_for).
    pub fn gate_for(&self, selection: &Selection, coord: TileCoord) -> Gate {
        Gate {
            view: self.mask_for(selection, coord),
            opacity: selection.opacity(),
        }
    }

    /// Apply `op` to `prev`, returning the new selection. `None` when the op's shape
    /// would need more than [`MAX_SELECTION_TILES`](stark_model::document::MAX_SELECTION_TILES)
    /// mask tiles — the caller leaves the selection alone rather than clipping it.
    pub fn apply(&self, pool: &TilePool, prev: &Selection, op: &SelectionOp) -> Option<Selection> {
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
        let (_edge_tex, edge_view) = match edges.is_empty() {
            true => (None, self.dummy_edges.clone()),
            false => {
                let (t, v) = self.edge_texture(&edges);
                (Some(t), v)
            }
        };
        let (b, c) = shader_params(op, edges.len());

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
                b,
                c,
                feather: op.feather(),
                edges: &edge_view,
            },
        ))
    }

    /// Invert the selection: every mask tile reflects through the mask's own level,
    /// and so does the coverage outside them. Constant cost on an unbounded canvas —
    /// the whole point of carrying `outside` as one number (§6.8).
    ///
    /// The level rides in the shape lane the shader reads an op's opacity from: an
    /// inversion has no shape, and both numbers are "the strength this mask is drawn
    /// at" (`selection.wesl`).
    pub fn invert(&self, pool: &TilePool, prev: &Selection) -> Selection {
        let plan = prev.plan_invert();
        let edges = self.dummy_edges.clone();
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
                b: [0.0; 4],
                c: [0.0, MODE_INVERT, 0.0, plan.level],
                feather: 0.0,
                edges: &edges,
            },
        )
    }

    /// The usage a [`region_mask`](Self::region_mask) target needs: drawn into by the
    /// gather below, then sampled by the loop. Stated here rather than at the caller
    /// so the lease and the pass cannot disagree about it.
    pub const REGION_MASK_USAGE: wgpu::TextureUsages =
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

    /// Gather `selection` into `into` — a region-sized mask for the stamp loop,
    /// matching the region `stroke.rs` composited the paint into. Tiles the selection
    /// has no mask for are left at the clear value, so the pass draws only what
    /// actually exists.
    ///
    /// **The target is the caller's**, at [`REGION_MASK_USAGE`](Self::REGION_MASK_USAGE)
    /// and `MASK_FORMAT`. It used to be created here and destroyed at the piece's
    /// submit: a region is up to `MAX_REGION_DIM`² of `R8Unorm`, so a live stroke
    /// under any selection was creating and destroying megabytes of texture per piece
    /// per pointer move, and never reusing one. Leased from the stroke's scratch pool
    /// it is the same texture every move, and the pool's own rule — a lease returns to
    /// the free list only through a submit — is what keeps that sound.
    ///
    /// The pass writes every texel it owns, clear included, so the pool's
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
        // Leased for the same reason the target above it is: a live stroke re-gathers
        // this per piece per pointer move, and these two were the last things on the
        // path still created and thrown away every time.
        let u = RegionUniform {
            a: [w as f32, h as f32, MASK_TEX as f32, 0.0],
        };
        let ubuf = scope.take_piece_buffer(BufKey {
            size: std::mem::size_of_val(&u) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark selection region uniform",
        });
        scope.write_lease(&ubuf, bytemuck::bytes_of(&u));
        let view_bg = desc::bind_group_for(
            device,
            "stark selection region view bg",
            &self.region_view_bgl,
            REGION_VIEW_SLOTS,
            false,
            |_| ubuf.as_entire_binding(),
        );

        let mut origins: Vec<MaskInstance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for coord in tiles {
            let Some(handle) = selection.tile(*coord) else {
                continue;
            };
            let off = mask_tex_origin(*coord) - region_origin;
            origins.push(MaskInstance {
                origin: off.to_array(),
            });
            tile_bgs.push(desc::bind_group_for(
                device,
                "stark selection region tile bg",
                &self.region_tile_bgl,
                REGION_TILE_SLOTS,
                false,
                |_| wgpu::BindingResource::TextureView(handle.view()),
            ));
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

        {
            // Everything the selection has no tile for takes the constant coverage
            // that reigns there.
            let outside = outside_clear(selection);
            let mut pass = scope
                .encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark selection region gather"),
                    color_attachments: &[Some(desc::attach(into, outside))],
                    ..Default::default()
                });
            if let Some(inst) = &instances {
                pass.set_pipeline(&self.region_pipeline);
                pass.set_bind_group(0, &view_bg, &[]);
                pass.set_vertex_buffer(0, inst.slice(..));
                for (i, bg) in tile_bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
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
        base: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle>,
        coords: &[TileCoord],
        shape: RasterShape<'_>,
    ) -> Selection {
        let RasterShape {
            outside,
            level,
            opacity,
            hull,
            b,
            c,
            feather,
            edges,
        } = shape;
        if coords.is_empty() {
            return Selection::from_parts(base, outside, level, opacity, hull);
        }
        let device = &self.ctx.device;
        // The lasso's edge texture and the shape's parameters both belong to the
        // whole rasterize and outlive every flush; only the per-tile uniform below
        // is scoped to one submit.
        let mut scope = self.scratch.scope(&self.ctx, "stark selection edit");
        // One slot per tile, all written before the first submit (`UniformSlots`).
        let params: Vec<MaskUniform> = coords
            .iter()
            .map(|coord| {
                let origin = mask_tex_origin(*coord);
                MaskUniform {
                    a: [origin.x, origin.y, 2.0 / MASK_TEX as f32, feather],
                    b,
                    c,
                }
            })
            .collect();
        let mut slots =
            UniformSlots::<MaskUniform>::new(device, "stark selection params", coords.len());
        slots.write(device, &self.ctx.queue, &params);

        let mut tiles = base;

        for (i, coord) in coords.iter().enumerate() {
            let dst = pool.acquire_mask(AllocSource::SelectionMask);
            let prev_view = self.mask_for(prev, *coord);
            let bg = desc::bind_group_for(
                device,
                "stark selection bg",
                &self.rasterize_bgl,
                RASTERIZE_SLOTS,
                false,
                |i| match i {
                    sb::P => slots.resource(),
                    sb::PREV => wgpu::BindingResource::TextureView(&prev_view),
                    sb::EDGES => wgpu::BindingResource::TextureView(edges),
                    other => unreachable!("`RASTERIZE_SLOTS` lists no binding {other}"),
                },
            );
            {
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark selection rasterize"),
                        color_attachments: &[Some(desc::attach(dst.view(), desc::CLEAR))],
                        ..Default::default()
                    });
                pass.set_pipeline(&self.rasterize_pipeline);
                pass.set_bind_group(0, &bg, &[UniformSlots::<MaskUniform>::offset(i as u32)]);
                pass.draw(0..3, 0..1);
            }
            tiles = tiles.insert(*coord, dst);
            scope.tile_done();
        }
        scope.finish();
        Selection::from_parts(tiles, outside, level, opacity, hull)
    }

    /// Upload the lasso's edge list as an `N×1` texture (see `selection.wesl`).
    fn edge_texture(&self, edges: &[[f32; 4]]) -> (wgpu::Texture, wgpu::TextureView) {
        let extent = wgpu::Extent3d {
            width: edges.len() as u32,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark selection lasso edges"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ctx.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(edges),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(edges.len() as u32 * 16),
                rows_per_image: Some(1),
            },
            extent,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

/// The selection as a **gating** pass reads it over one tile (§6.8): the coverage
/// bound as a texture, and the scalar the shader multiplies every read of it by.
///
/// One value rather than two returns, because the two halves are one fact. A pass
/// that took the view alone would compile, run, and quietly ignore the Opacity
/// slider — the failure mode the whole-mask opacity invites, since the mask tiles
/// look exactly the same at every opacity ([`Selection::opacity`]). Handing the
/// binding and its scale back together is what makes the second half hard to leave
/// on the floor. The stroke paths are the readers that legitimately take the view
/// alone, and only because their ceiling took the scalar first
/// (`SelectionRenderer::mask_for`).
pub struct Gate {
    view: wgpu::TextureView,
    opacity: f32,
}

impl Gate {
    /// The coverage, to bind.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The scalar to multiply every read of it by, for the pass's uniform.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }
}

/// What a fresh mask target is cleared to: the constant coverage that reigns
/// wherever `selection` has no tile of its own (§6.8).
///
/// In `r`, the only channel [`MASK_FORMAT`] has. Shared with the transform, which
/// carries masks under its maps and has to leave the untouched plane saying the
/// same thing this does — an inverted selection whose two paths disagreed about
/// the outside would gain or lose the whole canvas.
pub(crate) fn outside_clear(selection: &Selection) -> wgpu::Operations<wgpu::Color> {
    desc::clear_to(wgpu::Color {
        r: f64::from(selection.outside()),
        g: 0.0,
        b: 0.0,
        a: 0.0,
    })
}

// —— packing an op for `selection.wesl` ————————————————————————————————————————
//
// The op is a fact about the document and lives in `stark-model`; how it is packed
// into a uniform is a fact about the shader and lives here (§2). The codes come
// through the generated mirror (§6.10) rather than being transcribed, which is why
// this had to end up on the side that has the shaders at all.

/// The mode's discriminant as the mask shader sees it.
///
/// The numbers are `selection.wesl`'s own, generated from its declarations
/// (§6.10) rather than transcribed: which `u32` a mode is numbered is a fact
/// about the shader, and a `match` writing `0.0, 1.0, 2.0, 3.0` beside it was a
/// second declaration of it with nothing checking the correspondence.
///
/// `f32` because the lane it lands in is one — `Params::c` packs the kind, the
/// mode, the edge count and the opacity into one `vec4<f32>`, and the shader
/// compares them as floats.
fn mode_code(mode: SelectionMode) -> f32 {
    use stark_shaders::mirror::selection as sel;
    let code = match mode {
        SelectionMode::Replace => sel::MODE_REPLACE,
        SelectionMode::Union => sel::MODE_UNION,
        SelectionMode::Subtract => sel::MODE_SUBTRACT,
        SelectionMode::Intersect => sel::MODE_INTERSECT,
    };
    code as f32
}

/// The shape/feather packed for `selection.wesl`'s uniform: `(b, c)` where `b`
/// carries the analytic shape's parameters and `c` the kind/mode/edge
/// count/opacity.
pub(crate) fn shader_params(op: &SelectionOp, edges: usize) -> ([f32; 4], [f32; 4]) {
    // The kind codes are `selection.wesl`'s, generated from its declarations
    // (§6.10) — see `SelectionMode::code` for why they are not written here.
    use stark_shaders::mirror::selection as sel;
    let (kind, b) = match &op.shape() {
        SelectionShape::All => (sel::KIND_ALL, [0.0; 4]),
        SelectionShape::Rect { min, max } => (sel::KIND_RECT, [min.x, min.y, max.x, max.y]),
        // Radii non-negative by `SelectionShape::sanitized`, which every op passes
        // through — the shader takes them as written.
        SelectionShape::Ellipse { center, radii } => {
            (sel::KIND_ELLIPSE, [center.x, center.y, radii.x, radii.y])
        }
        SelectionShape::Lasso(_) => (sel::KIND_LASSO, [0.0; 4]),
    };
    (
        b,
        [
            kind as f32,
            mode_code(op.mode()),
            edges as f32,
            op.opacity(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The edge list closes the polygon, which is what makes it a *region* rather
    /// than a polyline — `selection.wesl` counts crossings, so a loop that did not
    /// return to its start would leave the winding open and the coverage undefined.
    ///
    /// Moved here with [`lasso_edges`] from `stark_model::geom`, where it was the
    /// only test of a function the model never called.
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
