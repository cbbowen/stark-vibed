//! The erase pass (§6.12): the swept extent every brush rasterizes, turned on
//! the layer's **visible** opacity instead of laid as paint.
//!
//! Two pieces. The *sweep* is `swept.rs`'s own pipeline geometry — the same
//! segments, the same prefix-τ lookups, the same drain/tooth/jitter gates — with
//! `stamp.wesl::fs_erase` as the fragment, accumulating the stroke's transparency
//! mass into one `R16Float` tile-sized accumulator per touched tile. The
//! *integrate* (`erase.wesl`) rewrites each tile from its **pristine** base — the
//! paint the stroke found, not the previous piece's output — scaling what the eye
//! sees by `1 − opacity·w` and inverting the slab law into a height (§6.1).
//!
//! **The accumulator spans the stroke, not the piece.** `1 − opacity·w` is not
//! exponential in swept depth, so applying it per piece would compound at every cut
//! a live stroke makes (§6.2's two composable forms). The mass sums additively — so
//! re-cutting the path changes nothing — and every piece re-derives its tiles from
//! pristine paint under the total. That bookkeeping is any such effect's, so it
//! lives in [`accum`](super::accum); what stays here is `erase.wesl`'s own layout and
//! pipelines, and the one decision that is this pass's — a tile the layer does not
//! have is nothing to erase ([`ParcelEffect::Erase`]).

use stark_shaders::Lane;
use stark_shaders::mirror::erase::binding as eb;
use stark_shaders::mirror::erase::decl as ed;

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::tile::TileMap;

use super::accum::{
    LandStep, Landing, LaneKeys, Lanes, ParcelEffect, SweepStep, land_tiles, lane_key,
};
use super::swept::{SweepDraws, SweepPass, landing_pipeline, reaches_nothing};
use super::{ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene};
use crate::gpu::scratch::Key;

/// The accumulator's format: one channel — the transparency mass — additive. f16
/// is enough because the interesting range is a few times `OPAQUE_MASS`, and by the
/// time accumulation outruns its mantissa the survival `exp(−m)` has reached zero.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The parcel's lanes: this pass rasterizes one number per texel where the deposit
/// rasterizes the channel trio, with the ceiling lane — the coverage the stroke has
/// claimed, each segment's share at its own ceiling — beside it under a pen-driven
/// opacity (§6.2). [`parcel_keys`] fills them by these names, which keeps the attach
/// order and the bind order one list ([`ParcelLanes`](super::accum::ParcelLanes)).
const MASS: usize = 0;
const CEILING: usize = 1;
/// The moment of the whole mass over the pen's factor — the lane's companion,
/// which the deposit keeps in its aux's spare channel and this pass, whose
/// accumulator has none, keeps in a lane of its own.
const MOMENT: usize = 2;

/// The accumulator's pool key — [`lane_key`]'s usages, at this pass's own format
/// and label.
fn accum_key() -> Key {
    lane_key(ACCUM_FORMAT, "stark erase accum")
}

/// The ceiling lane's pool key: the swept path's format, since the two lanes
/// carry the same sums by the same rule.
fn ceiling_key() -> Key {
    lane_key(super::swept::CEILING_FORMAT, "stark erase ceiling")
}

/// The moment lane's pool key: one channel, the accumulator's own format.
fn moment_key() -> Key {
    lane_key(ACCUM_FORMAT, "stark erase moment")
}

/// The lanes' pool keys at the lanes' own indices: the mass always, and under a
/// pen-driven ceiling the two beside it (§6.2) — the same sweep the deposit takes,
/// at the shader's own locations.
fn parcel_keys(ceiling_lane: bool) -> LaneKeys {
    let mut keys = LaneKeys::default();
    keys[MASS] = Some(accum_key());
    keys[CEILING] = ceiling_lane.then(ceiling_key);
    keys[MOMENT] = ceiling_lane.then(moment_key);
    keys
}

/// The erase sweep's targets, at `stamp.wesl::fs_erase`'s own locations
/// ([`SweepKit`](super::swept::SweepKit)): the transparency mass, additive across
/// overlapping segment quads and, through the load, across the pieces of a live
/// stroke — and under a pen-driven ceiling the lane and the mass's moment beside it,
/// additive like the deposit's (`swept::ceiling_target`, §6.2).
pub(super) fn sweep_targets(lane: Lane) -> Vec<Option<wgpu::ColorTargetState>> {
    let mass = desc::blended_target(ACCUM_FORMAT, Some(wgpu::BlendState::ADDITIVE));
    match lane {
        Lane::Plain => vec![mass],
        Lane::Ceiling => vec![mass.clone(), super::swept::ceiling_target(), mass],
    }
}

/// The erase's landing pass (`erase.wesl`): the pristine tile, the stroke's accumulated
/// mass, the selection, the opacity uniform and the ceiling lane in; the erased tile's
/// channels out.
pub(super) fn erase_pass(device: &wgpu::Device, color_space: &dyn ColorSpace) -> desc::OnePass {
    let shader = stark_shaders::erase(color_space.resid());
    landing_pipeline(
        device,
        color_space,
        "stark erase",
        shader,
        (shader.vs_main, shader.fs_main),
        ed::BASE_COLOR,
    )
}

impl StrokeRenderer {
    /// [`Self::render_range`] through the erase pass. The segments and constants
    /// arrive resolved ([`ResolvedRange`]), as for both of its siblings — one place
    /// answers what a stroke's segments are.
    pub(super) fn render_erase(
        &self,
        scene: StrokeScene<'_>,
        range: ResolvedRange<'_>,
    ) -> (TileMap, StrokeCarry) {
        crate::timing::span!("stroke.erase");
        // The pool, the base and the selection are `land_tiles`'s — it is what
        // acquires the copy-on-write destinations and gates each one by its mask
        // (§6.8).
        let StrokeScene { substrate, .. } = scene;
        let ResolvedRange {
            consts: k,
            end_dist,
            resume,
            ..
        } = range;

        let mut scope = self.scratch.scope(&self.ctx, "stark erase stroke");
        let device = &self.ctx.device;

        // The swept path's own draws, and `fs_erase` reads the same prefix-τ, substrate
        // and stroke uniform (the noise field rides along unread). At 1× always: the
        // supersampled resolve averages the *paint* parcel's finished visible alpha
        // (§6.2), and the erase's transparency mass runs a different law through
        // `erase.wesl` — its resolve would live there, and nothing gates an eraser
        // today (`budget::supersample_scale`).
        let Some(draws) = SweepDraws::new(self, &mut scope, &range, substrate, 1) else {
            return reaches_nothing(scope, scene.base(), end_dist);
        };

        // The stroke's ceiling, once per *call*: `StrokeConstants` resolved it with
        // the color and the mask's opacity folded in, so this path cannot disagree
        // with the others about what the dial said (`BrushEffect::opacity`, §6.8).
        // Run tier: every tile of `accum::land_tiles` binds it.
        let opacity = scope.run_uniform(
            "stark erase opacity",
            &stark_shaders::mirror::erase::Erase {
                params: [k.opacity, f32::from(u8::from(k.ceiling_lane)), 0.0, 0.0],
            },
        );

        // The shared procedure (§6.12, `accum`): resume what the pieces before this
        // one accumulated, extend it over this piece's tiles, and turn the total on
        // the pristine paint. A tile the layer does not have is nothing to erase —
        // no output, no accumulator, no entry in the carry — so a stroke over bare
        // canvas mints no tiles at all.
        land_tiles(
            self,
            scene,
            scope,
            Lanes::Resume(resume),
            &SweepStep {
                label: "stark erase sweep pass",
                pipeline: self.swept.pipeline(SweepPass::erase(k.lane())),
                draws: &draws,
                keys: parcel_keys(k.ceiling_lane),
            },
            &LandStep {
                label: "stark erase integrate",
                pipeline: &self.erase.pipeline,
                effect: ParcelEffect::Erase,
            },
            |l: &Landing<'_>| {
                self.erase
                    .bindings
                    .group(device, "stark erase bg", |b| match b {
                        eb::BASE_COLOR => wgpu::BindingResource::TextureView(l.base.color),
                        eb::BASE_AUX => wgpu::BindingResource::TextureView(l.base.aux),
                        eb::ACCUM => wgpu::BindingResource::TextureView(l.parcel.lane(MASS)),
                        eb::SELECTION => wgpu::BindingResource::TextureView(l.mask),
                        eb::E => opacity.window(),
                        eb::BASE_RESID => wgpu::BindingResource::TextureView(
                            l.base.resid.expect("a residual build has one"),
                        ),
                        eb::CEILING => wgpu::BindingResource::TextureView(if k.ceiling_lane {
                            l.parcel.lane(CEILING)
                        } else {
                            &self.zeroes.aux
                        }),
                        eb::MOMENT => wgpu::BindingResource::TextureView(if k.ceiling_lane {
                            l.parcel.lane(MOMENT)
                        } else {
                            &self.zeroes.aux
                        }),
                        other => unreachable!("the erase group holds no binding {other}"),
                    })
            },
        )
        .finished(end_dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parcel_keys_fill_exactly_the_lanes_the_stroke_carries() {
        let present = |keys: LaneKeys| keys.map(|key| key.is_some());
        assert_eq!(present(parcel_keys(false)), [true, false, false, false]);
        assert_eq!(present(parcel_keys(true)), [true, true, true, false]);
    }
}
