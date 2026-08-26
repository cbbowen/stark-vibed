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
use crate::document::fill::plan;
use crate::document::selection::Selection;
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::mask_tex_origin;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::submit::TileScope;
use crate::gpu::tile::{AllocSource, TileMap, TilePool};
use crate::gpu::uniforms::UniformSlots;
use stark_model::document::{FillOp, GradientAxis, Parcel};
use stark_model::document::{SelectionMode, SelectionOp, SelectionShape};

// Generated from `fill.wesl`'s own declarations (§6.7).
use stark_shaders::mirror::fill::Fill as FillUniform;
use stark_shaders::mirror::fill::Tile as TileUniform;
use stark_shaders::mirror::fill::binding as f;
use stark_shaders::mirror::fill::decl as fd;

/// Which bindings `fill.wesl` reads, in layout order (§6.10).
///
/// One list, read by both sides — the layout and the group are built from it, so
/// neither can disagree with the other. The residual sits beside the base color it
/// rides with, its `@if(resid)` gate carried on the declaration, where the
/// `if resid { push }` this replaces had to restate it.
///
/// `TILE` is the one slot whose *binding* the shader does not decide: `f` and `tile`
/// are both `var<uniform>` in the WESL, and the difference is that the first is one
/// buffer for the whole fill while the second is a per-tile slot of one
/// (`UniformSlots`). That is what [`desc::Slot::dynamic`] says.
const FILL_SLOTS: &[desc::Slot] = &[
    desc::Slot::at(fd::F),
    desc::Slot::at(fd::BASE_COLOR),
    desc::Slot::at(fd::BASE_AUX),
    desc::Slot::at(fd::REGION),
    desc::Slot::at(fd::GATE),
    desc::Slot::at(fd::BASE_RESID),
    desc::Slot::dynamic(fd::TILE),
];

// The shader's stop capacity is the ramp's own bound (§22.1) — asserted rather than
// commented, since the two are declared in different crates and nothing else would
// notice them parting (§6.10).
//
// **What this does not guard is the data**, and it cannot: it is a statement about
// two constants. The loop below indexes `stop_c` by a stop's position, so a ramp
// longer than the array is an index off the end of a uniform — a panic, not the
// silent truncation this comment used to claim. That bound belongs to `Gradient`,
// which holds it in `new`; this assert only keeps the number the same on both sides
// of the seam.
const _: () = assert!(
    stark_shaders::mirror::fill::MAX_GRADIENT_STOPS as usize == stark_model::gradient::MAX_STOPS,
    "the shader's stop count and the model's have drifted apart across the seam"
);

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
        let bgl = desc::layout_for(
            device,
            "stark fill bgl",
            FILL_SLOTS,
            frag,
            formats.has_resid(),
        );
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
        let region = match op.shape() {
            SelectionShape::All => Selection::everything(),
            shape => self.selection.apply(
                pool,
                &Selection::everything(),
                &SelectionOp::new(SelectionMode::Replace, shape.clone(), op.feather()),
            )?,
        };

        let device = &self.ctx.device;
        let mut scope = TileScope::new(&self.ctx, "stark fill");

        // Every stop's color converts to this space's channels **on the CPU, once
        // per fill** — the shader then interpolates in the working space, which is
        // what makes an Oklab ramp the library strip's and a Mixbox ramp a pigment
        // mixture (§22.4).
        let mut uniform = FillUniform::default();
        uniform.p[0] = op.opacity();
        // How strongly the author's mask gates this fill (§6.8) — the whole mask's
        // opacity, which `gate`'s own texels do not carry. The fill's *region* takes
        // no such factor: it is this gesture's shape rasterized as a selection, and
        // has no opacity of its own.
        uniform.p[3] = gate.opacity();
        match op.paint() {
            Parcel::Solid(color) => {
                let l = self.color_space.rgb_to_latent(*color);
                uniform.c = [l.lat[0], l.lat[1], l.lat[2], 0.0];
                uniform.r = [l.res[0], l.res[1], l.res[2], 0.0];
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
                    let l = self.color_space.rgb_to_latent(stop.color);
                    uniform.stop_c[i] = [l.lat[0], l.lat[1], l.lat[2], stop.t];
                    uniform.stop_r[i] = [l.res[0], l.res[1], l.res[2], 0.0];
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
            // As a gate: the coverage and the opacity it is read at, which the
            // uniform above already carries (`SelectionRenderer::gate_for`).
            let gate_mask = self.selection.gate_for(gate, *coord);
            let dst = Channels::acquire(pool, self.formats, AllocSource::FillDestination);
            let bg = desc::bind_group_for(
                device,
                "stark fill bg",
                &self.bgl,
                FILL_SLOTS,
                self.formats.has_resid(),
                |b| match b {
                    f::F => ubuf.as_entire_binding(),
                    f::BASE_COLOR => wgpu::BindingResource::TextureView(&base_color),
                    f::BASE_AUX => wgpu::BindingResource::TextureView(&base_aux),
                    f::REGION => wgpu::BindingResource::TextureView(&region_mask),
                    f::GATE => wgpu::BindingResource::TextureView(gate_mask.view()),
                    f::BASE_RESID => wgpu::BindingResource::TextureView(
                        base_resid
                            .as_ref()
                            .expect("a residual build has a base residual"),
                    ),
                    f::TILE => tile_slots.resource(),
                    other => unreachable!("`FILL_SLOTS` lists no binding {other}"),
                },
            );
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
