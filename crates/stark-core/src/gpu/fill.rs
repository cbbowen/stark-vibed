//! GPU execution of a region fill (§18.0.4).
//!
//! [`FillRenderer::apply`] takes a layer's tile map, the author's selection, and a
//! [`FillOp`], and produces the filled tile map — copy-on-write like every stroke,
//! so old history versions keep their tiles and the pool reclaims what falls out of
//! reach.
//!
//! Almost all of it is borrowed, which is the point. The region's coverage is
//! rasterized by [`SelectionRenderer`] — a `Replace` op over an empty mask *is* the
//! shape's coverage, so the analytic shapes, the feather ramp and the tile caps are
//! shared with the selection tools rather than reimplemented for paint. What is left
//! is one pass of `fill.wesl` per written tile, stacking the parcel by the law
//! `paint_common.wesl` shares with the brush.
//!
//! Like the other renderers this holds only immutable GPU objects, so it is cheap to
//! `Clone` and rides in the `Action::Context` (§5).

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::fill::{FillOp, GradientAxis, Parcel, plan};
use crate::document::selection::{
    Selection, SelectionMode, SelectionOp, SelectionShape, mask_tex_origin,
};
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::submit::TileScope;
use crate::gpu::tile::{AllocSource, TileMap, TilePool};
use crate::gpu::uniforms::UniformSlots;

// Generated from `fill.wesl`'s own declarations (§6.7).
use stark_shaders::mirror::fill::Fill as FillUniform;
use stark_shaders::mirror::fill::Tile as TileUniform;

// The shader's stop capacity is the fitter's (§22.1) — asserted rather than
// commented, since a gradient with more stops than the uniform holds would
// truncate silently (§6.10).
const _: () =
    assert!(stark_shaders::mirror::fill::MAX_GRADIENT_STOPS as usize == crate::gradient::MAX_STOPS);

#[derive(Clone)]
pub struct FillRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    /// The channel formats this fill's tiles carry — the color space's, resolved
    /// once (§6.7).
    formats: ChannelFormats,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// The base of a tile the layer does not have yet, so a fill onto virgin canvas
    /// runs the same shader as a fill onto paint.
    zeroes: Zeroes,
    /// Borrowed for the coverage rasterize and for the 0/1 constants bound where a
    /// mask has no tile.
    selection: SelectionRenderer,
}

impl FillRenderer {
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: Arc<dyn ColorSpace>,
        selection: SelectionRenderer,
        zeroes: Zeroes,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space.as_ref());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark fill"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::fill(formats.has_resid()).into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag), // base color
            desc::load_tex(2, frag), // base aux (height)
            desc::load_tex(3, frag), // the shape's coverage
            desc::load_tex(4, frag), // the author's selection
        ];
        if formats.has_resid() {
            entries.push(desc::load_tex(5, frag)); // base residual (§6.7)
        }
        // The tile's canvas origin — per tile, where binding 0 is per fill, so it
        // is a dynamic-offset slot rather than a buffer each (`UniformSlots`). Bound
        // unconditionally at 6 so its index does not move with the resid feature.
        entries.push(desc::uniform_slot(
            6,
            frag,
            std::mem::size_of::<TileUniform>() as u64,
        ));
        let bgl = desc::bind_group_layout(device, "stark fill bgl", &entries);
        let layout = desc::pipeline_layout(device, "stark fill layout", &[Some(&bgl)]);
        let targets = formats.targets();
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark fill pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &targets,
        );

        Self {
            ctx: ctx.clone(),
            color_space,
            formats,
            pipeline,
            bgl,
            zeroes,
            selection,
        }
    }

    /// Fill `op`'s region on `base` (one layer's tiles), gated by `gate` (the
    /// author's selection). `None` rejects the whole action — an unbounded fill, or
    /// more tiles than the cap allows — deterministically, so peers and replays
    /// agree (see [`crate::document::fill::plan`]).
    pub fn apply(
        &self,
        pool: &TilePool,
        base: &TileMap,
        gate: &Selection,
        op: &FillOp,
    ) -> Option<TileMap> {
        let coords = plan(op, gate)?;
        if coords.is_empty() {
            return Some(base.clone());
        }

        // The shape's own coverage, as mask tiles. `Replace` over the unrestricted
        // selection is exactly "the shape, and nothing else", so this is the
        // selection rasterizer doing its ordinary job — the reason a fill's edge and
        // a marquee's edge cannot drift apart. `All` needs no rasterize: its
        // coverage is 1 everywhere, and the gate is then the whole boundary.
        let region = match &op.shape {
            SelectionShape::All => Selection::everything(),
            shape => self.selection.apply(
                pool,
                &Selection::everything(),
                &SelectionOp::new(SelectionMode::Replace, shape.clone(), op.feather),
            )?,
        };

        let device = &self.ctx.device;
        let mut scope = TileScope::new(&self.ctx, "stark fill");

        // Every stop's color converts to this space's channels **on the CPU, once
        // per fill** — the shader then interpolates in the working space, which is
        // what makes an Oklab ramp the library strip's and a Mixbox ramp a pigment
        // mixture (§22.4).
        let mut uniform = FillUniform::default();
        uniform.p[0] = op.opacity;
        match &op.paint {
            Parcel::Solid(color) => {
                let channels = self.color_space.rgb_to_channels(*color);
                let resid = self.color_space.rgb_to_resid(*color);
                uniform.c = [channels[0], channels[1], channels[2], 0.0];
                uniform.r = [resid[0], resid[1], resid[2], 0.0];
            }
            Parcel::Gradient(g) => {
                let stops = g.gradient.stops();
                uniform.p[1] = stops.len() as f32;
                uniform.axis = match g.axis {
                    GradientAxis::Linear { from, to } => [from.x, from.y, to.x, to.y],
                    GradientAxis::Radial { center, radius } => {
                        uniform.p[2] = 1.0;
                        [center.x, center.y, radius, 0.0]
                    }
                };
                for (i, stop) in stops.iter().enumerate() {
                    let channels = self.color_space.rgb_to_channels(stop.color);
                    let resid = self.color_space.rgb_to_resid(stop.color);
                    uniform.stop_c[i] = [channels[0], channels[1], channels[2], stop.t];
                    uniform.stop_r[i] = [resid[0], resid[1], resid[2], 0.0];
                }
            }
        }
        // Per *fill*, and so deliberately not registered with the scope: it is
        // bound by every tile below and has to outlive each of their submits.
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark fill uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // One slot per tile, all written before the first submit — see
        // [`UniformSlots`] for why they cannot share a single rewritten buffer.
        let origins: Vec<TileUniform> = coords
            .iter()
            .map(|coord| {
                // Mask and paint tiles share their geometry, so the selection's
                // origin is this pass's too.
                let origin = mask_tex_origin(*coord);
                TileUniform {
                    origin: [origin.x, origin.y, 0.0, 0.0],
                }
            })
            .collect();
        let mut tile_slots =
            UniformSlots::<TileUniform>::new(device, "stark fill tile", coords.len());
        tile_slots.write(device, &self.ctx.queue, &origins);

        let mut tiles = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let (base_color, base_aux) = match base.get(coord) {
                Some(tile) => (tile.color_view().clone(), tile.aux_view().clone()),
                None => (self.zeroes.color.clone(), self.zeroes.aux.clone()),
            };
            // Bare canvas reads the 1×1 zero here exactly as it does for the color.
            let base_resid = self.zeroes.resid.as_ref().map(|zero| {
                base.get(coord)
                    .and_then(|t| t.resid_view())
                    .unwrap_or(zero)
                    .clone()
            });
            let region_mask = self.selection.mask_for(&region, *coord);
            let gate_mask = self.selection.mask_for(gate, *coord);
            let dst = Channels::acquire(pool, self.formats, AllocSource::FillDestination);
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                desc::tex(1, &base_color),
                desc::tex(2, &base_aux),
                desc::tex(3, &region_mask),
                desc::tex(4, &gate_mask),
            ];
            if let Some(view) = &base_resid {
                entries.push(desc::tex(5, view));
            }
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: tile_slots.buffer(),
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<TileUniform>() as u64),
                }),
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark fill bg"),
                layout: &self.bgl,
                entries: &entries,
            });
            scope.fullscreen_pass(
                "stark fill tile",
                &self.pipeline,
                &bg,
                &[UniformSlots::<TileUniform>::offset(i as u32)],
                dst.targets(),
                desc::CLEAR,
            );
            tiles = tiles.insert(*coord, dst.into_tile());
            scope.tile_done();
        }

        scope.finish();
        Some(tiles)
    }
}
