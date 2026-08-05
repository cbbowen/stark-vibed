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

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::fill::{FillOp, plan};
use crate::document::selection::{Selection, SelectionMode, SelectionOp, SelectionShape};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle, TilePool};
use crate::gpu::wesl::mirrors_wesl;

/// Mirrors `Fill` in `fill.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FillUniform {
    c: [f32; 4], // latent .xyz, per-unit opacity .w
    p: [f32; 4], // height at full coverage, _, _, _
}
mirrors_wesl!(FillUniform, 32);

#[derive(Clone)]
pub struct FillRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// 1×1 zero color/aux — the base of a tile the layer does not have yet, so a
    /// fill onto virgin canvas runs the same shader as a fill onto paint.
    zero_color: wgpu::TextureView,
    zero_aux: wgpu::TextureView,
    /// Borrowed for the coverage rasterize and for the 0/1 constants bound where a
    /// mask has no tile.
    selection: SelectionRenderer,
}

impl FillRenderer {
    pub fn new(
        ctx: &GpuContext,
        color_space: Arc<dyn ColorSpace>,
        selection: SelectionRenderer,
    ) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark fill"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::fill().into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = desc::bind_group_layout(
            device,
            "stark fill bgl",
            &[
                desc::uniform(0, frag),
                desc::load_tex(1, frag), // base color
                desc::load_tex(2, frag), // base aux (height)
                desc::load_tex(3, frag), // the shape's coverage
                desc::load_tex(4, frag), // the author's selection
            ],
        );
        let layout = desc::pipeline_layout(device, "stark fill layout", &[Some(&bgl)]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark fill pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &[desc::target(color_format), desc::target(aux_format)],
        );

        Self {
            ctx: ctx.clone(),
            color_space,
            color_format,
            aux_format,
            pipeline,
            bgl,
            zero_color: desc::zero_texture(ctx, color_format, "stark fill zero"),
            zero_aux: desc::zero_texture(ctx, aux_format, "stark fill zero"),
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark fill"),
        });

        let channels = self
            .color_space
            .rgb_to_channels([op.color[0], op.color[1], op.color[2]]);
        let uniform = FillUniform {
            c: [channels[0], channels[1], channels[2], op.color[3]],
            p: [op.height, 0.0, 0.0, 0.0],
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark fill uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut tiles = base.clone();
        for coord in &coords {
            let (base_color, base_aux) = match base.get(coord) {
                Some(tile) => (tile.color_view().clone(), tile.aux_view().clone()),
                None => (self.zero_color.clone(), self.zero_aux.clone()),
            };
            let region_mask = self.selection.mask_for(&region, *coord);
            let gate_mask = self.selection.mask_for(gate, *coord);
            let dst = (
                pool.acquire_tex(self.color_format, AllocSource::FillDestination),
                pool.acquire_tex(self.aux_format, AllocSource::FillDestination),
            );
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark fill bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubuf.as_entire_binding(),
                    },
                    desc::tex(1, &base_color),
                    desc::tex(2, &base_aux),
                    desc::tex(3, &region_mask),
                    desc::tex(4, &gate_mask),
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark fill tile"),
                color_attachments: &[
                    Some(desc::attach(dst.0.view(), desc::CLEAR)),
                    Some(desc::attach(dst.1.view(), desc::CLEAR)),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            tiles = tiles.insert(*coord, TilePairHandle::new(dst.0, dst.1));
        }

        self.ctx.queue.submit([encoder.finish()]);
        Some(tiles)
    }
}
