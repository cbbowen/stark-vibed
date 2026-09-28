//! GPU execution of a region fill (§18.0.4).
//!
//! [`FillRenderer::apply`] takes a layer's tile map, the author's selection, and a
//! [`FillOp`], and produces the filled tile map — copy-on-write like every stroke,
//! so old history versions keep their tiles and the pool reclaims what falls out of
//! reach.
//!
//! Which tiles it writes is settled first, on the CPU and with no device, by [`plan()`].
//!
//! The region's coverage is rasterized by [`SelectionRenderer`] — a `Replace` op over an
//! empty mask *is* the shape's coverage, so the analytic shapes, the feather ramp and the
//! tile caps are shared with the selection tools rather than reimplemented for paint.
//! What is left is one pass of `fill.wesl` per written tile, stacking the parcel by the
//! law `paint_common.wesl` shares with the brush.
//!
//! Like the other renderers this holds only immutable GPU objects, so it is cheap to
//! `Clone` and rides in the `Action::Context` (§5).

use crate::gpu::tile::TilePairHandle;
use std::sync::Arc;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::gpu::channels::Zeroes;
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::scratch::ScratchPool;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::mask_tex_origin;
use crate::gpu::tile::{AllocSource, TileMap, TilePool};
use stark_model::document::{FillOp, GradientAxis, Parcel};
use stark_model::document::{SelectionMode, SelectionOp, SelectionShape};

// Generated from `fill.wesl`'s own declarations (§6.7).
use stark_shaders::mirror::fill::Fill as FillUniform;
use stark_shaders::mirror::fill::Tile as TileUniform;
use stark_shaders::mirror::fill::binding as f;
use stark_shaders::mirror::fill::decl as fd;

mod plan;

use self::plan::plan;

// The shader's stop capacity is the ramp's own bound (§22.1) — asserted rather than
// commented, since the two are declared in different crates and nothing else would
// notice them parting (§6.10).
//
// **What this does not guard is the data**: the loop below indexes `stop_c` by a stop's
// position, so a ramp longer than the array is an index off the end of a uniform. That
// bound belongs to `Gradient`, which holds it in `new`.
const _: () = assert!(
    stark_shaders::mirror::fill::MAX_GRADIENT_STOPS as usize == stark_model::gradient::MAX_STOPS,
    "the shader's stop count and the model's have drifted apart across the seam"
);

#[derive(Clone)]
pub(crate) struct FillRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    /// The channel formats this fill's tiles carry — the color space's, resolved
    /// once (§6.7).
    formats: ChannelFormats,
    pass: desc::OnePass,
    /// The base of a tile the layer does not have yet, so a fill onto virgin canvas
    /// runs the same shader as a fill onto paint.
    zeroes: Zeroes,
    scratch: ScratchPool,
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
        scratch: ScratchPool,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space.as_ref());

        let fill = stark_shaders::fill(color_space.resid());
        let shader = desc::Module::new(device, "stark fill", fill);
        // `TILE` is the one thing the shader does not decide: `f` and `tile` are both
        // `var<uniform>`, and the difference is that the first is one uniform for the
        // whole fill while the second is a per-tile slot of one (`uniforms::Slot`).
        let pass = desc::OnePass::fullscreen(
            device,
            "stark fill",
            &shader,
            (fill.vs_main, fill.fs_main),
            fd::F,
            &[fd::TILE],
            &formats.targets(),
        );

        Self {
            ctx: ctx.clone(),
            color_space,
            formats,
            pass,
            zeroes,
            scratch,
            selection,
        }
    }

    /// Fill `op`'s region on `base` (one layer's tiles), gated by `gate` (the
    /// author's selection). `None` rejects the whole action — an unbounded fill, or
    /// more tiles than the cap allows — deterministically, so peers and replays
    /// agree (see [`plan()`]).
    pub(crate) fn apply(
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

        // The shape's own coverage, as mask tiles: `Replace` over the unrestricted
        // selection is exactly "the shape, and nothing else", which is why a fill's
        // edge and a marquee's edge cannot drift apart. `All` needs no rasterize — its
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
        let mut scope = self.scratch.scope(&self.ctx, "stark fill");

        let uniform = fill_uniform(self.color_space.as_ref(), op, gate.opacity());
        let fill = scope.run_uniform("stark fill uniform", &uniform);
        let origins: Vec<TileUniform> = coords
            .iter()
            .map(|coord| {
                // Mask and paint tiles share their geometry, so the selection's
                // origin is this pass's too.
                let origin = mask_tex_origin(*coord);
                TileUniform {
                    origin: origin.to_array(),
                    ..Default::default()
                }
            })
            .collect();
        let tile_slots = scope
            .run_slots("stark fill tile", &origins)
            .expect("a fill that reaches here has a tile to rewrite");

        let mut tiles = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            // Bare canvas reads the 1×1 zeroes, all three at once, through the one
            // answer to that question (`Zeroes::or`, §6.8's pattern).
            let under = self.zeroes.or(base.get(coord).map(TilePairHandle::targets));
            let region_mask = self.selection.mask_for(&region, *coord);
            // As a gate: the coverage and the opacity it is read at, which the
            // uniform above already carries (`SelectionRenderer::gate_for`).
            let gate_mask = self.selection.mask_for(gate, *coord);
            let dst = Channels::acquire(pool, self.formats, AllocSource::FillDestination);
            let bg = self
                .pass
                .bindings
                .group(device, "stark fill bg", |b| match b {
                    f::F => fill.window(),
                    f::BASE_COLOR => wgpu::BindingResource::TextureView(under.color),
                    f::BASE_AUX => wgpu::BindingResource::TextureView(under.aux),
                    f::REGION => wgpu::BindingResource::TextureView(region_mask.view()),
                    f::GATE => wgpu::BindingResource::TextureView(gate_mask.view()),
                    f::BASE_RESID => wgpu::BindingResource::TextureView(
                        under.resid.expect("a residual build has a base residual"),
                    ),
                    f::TILE => tile_slots.window(),
                    other => {
                        unreachable!(
                            "`fill.wesl` declares @binding({other}), which this fills with nothing"
                        )
                    }
                });
            scope.fullscreen_pass(
                "stark fill tile",
                dst.targets(),
                &self.pass.pipeline,
                desc::Bound::dynamic(&bg, &[tile_slots.offset(i)]),
            );
            tiles = tiles.insert(*coord, dst.into_tile());
            scope.tile_done();
        }

        scope.finish();
        Some(tiles)
    }
}

/// The whole of `fill.wesl`'s per-fill uniform, from the op and the mask's opacity.
///
/// Every stop's color converts to this space's channels here, on the CPU, once per
/// fill; the shader interpolates in the working space, which is what makes an Oklab
/// ramp the library strip's and a Mixbox ramp a pigment mixture (§22.4).
///
/// A free function over the space, like `place::texel`: it is the whole of what a
/// caller can get wrong about the parcel, and it holds no device.
fn fill_uniform(space: &dyn ColorSpace, op: &FillOp, gate_opacity: f32) -> FillUniform {
    let mut uniform = FillUniform {
        opacity: op.opacity(),
        // How strongly the author's mask gates this fill (§6.8) — the whole mask's
        // opacity, which the gate's own texels do not carry. The fill's *region* takes
        // no such factor: it is this gesture's shape rasterized as a selection, and has
        // no opacity of its own.
        gate_opacity,
        ..Default::default()
    };
    match op.paint() {
        Parcel::Solid(color) => {
            let l = space.rgb_to_latent(*color);
            uniform.c = l.lat;
            uniform.r = l.res;
        }
        Parcel::Gradient(g) => {
            let stops = g.gradient.stops();
            uniform.stops = stops.len() as f32;
            uniform.axis = match g.axis {
                GradientAxis::Linear { from, to } => [from.x, from.y, to.x, to.y],
                GradientAxis::Radial { center, radius } => {
                    uniform.axis_kind = 1.0;
                    [center.x, center.y, radius, 0.0]
                }
            };
            for (i, stop) in stops.iter().enumerate() {
                let l = space.rgb_to_latent(stop.color);
                uniform.stop_c[i] = [l.lat[0], l.lat[1], l.lat[2], stop.t];
                uniform.stop_r[i] = [l.res[0], l.res[1], l.res[2], 0.0];
            }
        }
    }
    uniform
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colorspace::OkLabColorSpace;
    use stark_model::Srgb;
    use stark_model::document::GradientParcel;
    use stark_model::geom::Vec2;
    use stark_model::gradient::{Gradient, GradientStop};

    fn shape() -> SelectionShape {
        SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(10.0))
    }

    /// Black to white, the two-stop ramp every case here reads.
    fn ramp() -> Gradient {
        let stop = |t, v| GradientStop {
            t,
            color: Srgb::new([v; 3]),
        };
        Gradient::new(vec![stop(0.0, 0.0), stop(1.0, 1.0)]).expect("two ordered stops")
    }

    fn gradient_fill(axis: GradientAxis) -> FillOp {
        FillOp::with_paint(
            shape(),
            0.0,
            Parcel::Gradient(GradientParcel {
                gradient: ramp(),
                axis,
            }),
            1.0,
        )
    }

    fn uniform_of(op: &FillOp, gate_opacity: f32) -> FillUniform {
        fill_uniform(&OkLabColorSpace, op, gate_opacity)
    }

    /// **The two opacities ride in different lanes**, and neither is the other: the
    /// op's is the fill's own, `gate_opacity` is how hard the author's mask gates it
    /// (§6.8).
    #[test]
    fn a_solid_fill_carries_its_own_opacity_and_the_masks_separately() {
        let op = FillOp::new(shape(), 0.0, Srgb::new([1.0, 0.0, 0.0]), 0.25);
        let u = uniform_of(&op, 0.5);
        assert_eq!((u.opacity, u.gate_opacity), (0.25, 0.5));
        // A solid parcel names no stops and no axis, which is what the shader reads
        // `stops` for.
        assert_eq!(u.stops, 0.0);
        assert_eq!(u.axis, [0.0; 4]);
        // Red in Oklab is a positive `a` — the latent, not the sRGB triple.
        assert!(u.c[1] > 0.0, "a red fill converted to {:?}", u.c);
    }

    /// A gradient fills `stop_c` in the ramp's order with each stop's own `t` in `.w`,
    /// and says how many are live — the shader walks that prefix (§22.1).
    #[test]
    fn a_linear_gradient_lands_its_stops_in_order_with_its_axis() {
        let op = gradient_fill(GradientAxis::Linear {
            from: Vec2::new(1.0, 2.0),
            to: Vec2::new(3.0, 4.0),
        });
        let u = uniform_of(&op, 1.0);
        assert_eq!(u.stops, 2.0);
        // Linear, so the radial flag stays down and the axis is the two endpoints.
        assert_eq!(u.axis_kind, 0.0);
        assert_eq!(u.axis, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!((u.stop_c[0][3], u.stop_c[1][3]), (0.0, 1.0));
        assert!(
            u.stop_c[0][0] < u.stop_c[1][0],
            "black should be darker than white in the latent's first channel",
        );
    }

    /// The radial axis is the same four lanes read differently, and `axis_kind` is the
    /// one thing that says which — so a centre and radius written into the linear lanes
    /// is a ramp drawn along a line between two unrelated points.
    #[test]
    fn a_radial_gradient_raises_the_flag_that_reinterprets_the_axis() {
        let op = gradient_fill(GradientAxis::Radial {
            center: Vec2::new(5.0, 6.0),
            radius: 7.0,
        });
        let u = uniform_of(&op, 1.0);
        assert_eq!(u.axis_kind, 1.0);
        assert_eq!(u.axis, [5.0, 6.0, 7.0, 0.0]);
    }
}
