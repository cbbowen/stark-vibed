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
use wgpu::util::DeviceExt;

use crate::document::selection::{
    MASK_TEX, Selection, SelectionOp, SelectionShape, lasso_edges, mask_tex_origin,
};
use crate::geom::{TileCoord, Vec2};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::submit::TileScope;
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
    /// 1×1 masks holding exactly 0 and 1 — the two coverages a selection has
    /// outside its own tiles unless it is a partial one (see [`Self::constant`]).
    constants: [wgpu::TextureView; 2],
    /// 1×1 stand-in for the lasso edge list, bound by the analytic shapes.
    dummy_edges: wgpu::TextureView,
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
    /// The result's analytic hull, as the plan computed it ([`Selection::hull`]).
    hull: Option<(Vec2, Vec2)>,
    b: [f32; 4],
    c: [f32; 4],
    feather: f32,
    edges: &'a wgpu::TextureView,
}

impl SelectionRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::selection().into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        // The mask targets take no blend: the shader does the combine and writes
        // straight through.
        let mask_target = [desc::target(MASK_FORMAT)];
        let rasterize_bgl = desc::bind_group_layout(
            device,
            "stark selection bgl",
            &[
                // Per tile, so a dynamic-offset slot rather than a buffer each.
                UniformSlots::<MaskUniform>::layout(0, frag),
                desc::load_tex(1, frag), // previous mask
                desc::load_tex(2, frag), // lasso edges
            ],
        );
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
        let region_view_bgl = desc::bind_group_layout(
            device,
            "stark selection region view bgl",
            &[desc::uniform(0, wgpu::ShaderStages::VERTEX)],
        );
        let region_tile_bgl = desc::bind_group_layout(
            device,
            "stark selection region tile bgl",
            &[desc::load_tex(0, frag)],
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
        let constants = [
            desc::constant_texture(ctx, MASK_FORMAT, &[0], "stark selection constant mask"),
            desc::constant_texture(ctx, MASK_FORMAT, &[255], "stark selection constant mask"),
        ];
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
        }
    }

    /// The 1×1 mask holding `coverage` — what consumers bind wherever the selection
    /// has no tile. Their clamped `textureLoad` then reads the constant for every
    /// texel, so nothing branches on whether a mask exists.
    ///
    /// Quantized to a byte, which is not a loss: [`MASK_FORMAT`] is `R8Unorm`, so
    /// this is the same rounding the mask tiles themselves took, and a texel of the
    /// constant has to answer as one of theirs would.
    pub fn constant(&self, coverage: f32) -> wgpu::TextureView {
        let byte = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
        match byte {
            0 => self.constants[0].clone(),
            255 => self.constants[1].clone(),
            // A *partially* selected plane — reachable by inverting a partial
            // selection, and only that way (§6.8). Built on the spot rather than
            // cached: it is one texel, and a cache keyed by the byte would have to
            // be interior-mutable inside a type that is cloned into every action's
            // context.
            _ => desc::constant_texture(
                &self.ctx,
                MASK_FORMAT,
                &[byte],
                "stark selection constant mask",
            ),
        }
    }

    /// The mask bound for `coord`: the selection's own tile, or the constant that
    /// reigns outside its tile set.
    pub fn mask_for(&self, selection: &Selection, coord: TileCoord) -> wgpu::TextureView {
        match selection.tile(coord) {
            Some(handle) => handle.view().clone(),
            None => self.constant(selection.outside()),
        }
    }

    /// Apply `op` to `prev`, returning the new selection. `None` when the op's shape
    /// would need more than [`MAX_SELECTION_TILES`](crate::document::selection::MAX_SELECTION_TILES)
    /// mask tiles — the caller leaves the selection alone rather than clipping it.
    pub fn apply(&self, pool: &TilePool, prev: &Selection, op: &SelectionOp) -> Option<Selection> {
        let plan = prev.plan(op)?;
        let edges = match &op.shape {
            SelectionShape::Lasso(points) => lasso_edges(points),
            _ => Vec::new(),
        };
        // A lasso that encloses nothing has no boundary; treat it as a no-op rather
        // than rasterizing an empty edge list (which would read as "all outside").
        if matches!(op.shape, SelectionShape::Lasso(_)) && edges.is_empty() {
            return Some(prev.clone());
        }
        let (_edge_tex, edge_view) = match edges.is_empty() {
            true => (None, self.dummy_edges.clone()),
            false => {
                let (t, v) = self.edge_texture(&edges);
                (Some(t), v)
            }
        };
        let (b, c) = op.shader_params(edges.len());

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
                hull: plan.hull,
                b,
                c,
                feather: op.feather,
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
                hull: plan.hull,
                b: [0.0; 4],
                c: [0.0, MODE_INVERT, 0.0, plan.level],
                feather: 0.0,
                edges: &edges,
            },
        )
    }

    /// Gather `selection` into a region-sized mask for the stamp loop, matching the
    /// region `stroke.rs` composited the paint into. Tiles the selection has no mask
    /// for are left at the clear value, so the pass draws only what actually exists.
    /// Returns the texture too, so the caller can register it for prompt destruction.
    pub fn region_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        selection: &Selection,
        tiles: &[TileCoord],
        region_origin: Vec2,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let device = &self.ctx.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark selection region mask"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark selection region uniform"),
            contents: bytemuck::bytes_of(&RegionUniform {
                a: [w as f32, h as f32, MASK_TEX as f32, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark selection region view bg"),
            layout: &self.region_view_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });

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
            tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark selection region tile bg"),
                layout: &self.region_tile_bgl,
                entries: &[desc::tex(0, handle.view())],
            }));
        }
        let instances = (!origins.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark selection region instances"),
                contents: bytemuck::cast_slice(&origins),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        {
            // Everything the selection has no tile for takes the constant coverage
            // that reigns there.
            let outside = outside_clear(selection);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark selection region gather"),
                color_attachments: &[Some(desc::attach(&view, outside))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
        (texture, view)
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
            hull,
            b,
            c,
            feather,
            edges,
        } = shape;
        if coords.is_empty() {
            return Selection::from_parts(base, outside, level, hull);
        }
        let device = &self.ctx.device;
        // The lasso's edge texture and the shape's parameters both belong to the
        // whole rasterize and outlive every flush; only the per-tile uniform below
        // is scoped to one submit.
        let mut scope = TileScope::new(&self.ctx, "stark selection edit");
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
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark selection bg"),
                layout: &self.rasterize_bgl,
                entries: &[
                    slots.binding(0),
                    desc::tex(1, &prev_view),
                    desc::tex(2, edges),
                ],
            });
            {
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark selection rasterize"),
                        color_attachments: &[Some(desc::attach(dst.view(), desc::CLEAR))],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.rasterize_pipeline);
                pass.set_bind_group(0, &bg, &[UniformSlots::<MaskUniform>::offset(i as u32)]);
                pass.draw(0..3, 0..1);
            }
            tiles = tiles.insert(*coord, dst);
            scope.tile_done();
        }
        scope.finish();
        Selection::from_parts(tiles, outside, level, hull)
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
