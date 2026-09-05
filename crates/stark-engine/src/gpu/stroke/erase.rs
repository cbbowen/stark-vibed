//! The erase pass (§6.12): the swept extent every brush rasterizes, turned on
//! the layer's **visible** opacity instead of laid as paint.
//!
//! Two pieces, mirroring the fast path's shape. The *sweep* is the very pipeline
//! geometry `swept.rs` draws — the same segments, the same prefix-τ lookups, the
//! same drain/tooth/jitter gates — with `stamp.wesl::fs_erase` as the fragment,
//! accumulating the stroke's transparency mass into one `R16Float` tile-sized
//! accumulator per touched tile. The *integrate* (`erase.wesl`) then rewrites each
//! tile from its **pristine** base — the paint the stroke found, not the previous
//! piece's output — scaling what the eye sees by `1 − opacity·w` and inverting
//! the slab law into a height (§6.1).
//!
//! **The accumulator spans the stroke, not the piece**, and that is the design.
//! `1 − opacity·w` is not exponential in swept depth, so applying it per piece
//! would compound at every cut a live stroke makes (§6.2's two composable forms).
//! Instead the mass keeps summing — additively, so re-cutting the path changes
//! nothing — and every piece re-derives its tiles from pristine paint under the
//! total. The accumulators and the pristine handles ride the stroke's carry, the
//! way the stamp loop's reservoir does; a piece copies the accumulator it resumes
//! rather than writing it, which is what lets the live tail re-render every frame
//! from the same frozen head.
//!
//! That last paragraph is not this pass's alone — it is the law any effect outside
//! §6.2's two composable forms obeys, and the swept deposit obeys it too below full
//! opacity. So the bookkeeping it describes lives in
//! [`accum`](super::accum) and is run from here rather than written here: what
//! stays in this file is `erase.wesl`'s own slots, its pipelines, and the one
//! decision that is genuinely this pass's — a tile the layer does not have is
//! nothing to erase ([`BareCanvas::Skip`]).

use stark_model::document::StrokeRecord;
use stark_shaders::mirror::erase::binding as eb;
use stark_shaders::mirror::erase::decl as ed;

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use crate::gpu::tile::TileMap;

use super::accum::{
    BareCanvas, IncrementalTileAccumulator, Land, Landed, Landing, LaneKeys, Sweep, lane_key,
};
use super::incremental::{Carried, Resume};
use super::segments::generate_segments_in;
use super::swept::{SweptKit, sweep_binds, sweep_draws};
use super::tips::ResolvedTip;
use super::{Progress, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState};
use crate::gpu::scratch::{BufKey, Key};

/// The integrate's one group (`erase.wesl`): the pristine tile, the stroke's
/// accumulated mass, the selection, the opacity uniform, and the ceiling lane —
/// the parcel's second lane under a pen-driven opacity, the 1×1 zero otherwise.
const ERASE_SLOTS: &[Slot] = &[
    Slot::at(ed::BASE_COLOR),
    Slot::at(ed::BASE_AUX),
    Slot::at(ed::ACCUM),
    Slot::at(ed::SELECTION),
    Slot::at(ed::E),
    Slot::at(ed::BASE_RESID),
    Slot::at(ed::CEILING),
    Slot::at(ed::MOMENT),
];

/// The accumulator's format: one channel — the transparency mass — additive, like
/// the persistent aux it sits beside in size and precision. f16 is enough for the
/// same reason it is enough there: the interesting range is a few times
/// `OPAQUE_MASS`, and by the time accumulation outruns f16's mantissa the survival
/// `exp(−m)` has long since reached zero.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The accumulator is the parcel's first lane: this pass rasterizes one number per
/// texel, where the deposit rasterizes the channel trio. Under a pen-driven
/// opacity the ceiling lane rides beside it (§6.2) — the coverage the stroke has
/// claimed, each segment's share at its own ceiling. [`parcel_keys`] fills the
/// lanes by these names, which is what keeps the attach order and the bind order
/// one list ([`Parcel`](super::accum::Parcel)).
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

/// The erase pass's GPU objects, built once beside the two paths' kits.
///
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
#[derive(Clone)]
pub(super) struct EraseKit {
    /// The erase sweep: the swept path's pipeline shape — same module, same
    /// instance layout, same three bind group layouts (shared with [`SweptKit`],
    /// so one set of bind groups serves either pipeline) — with `fs_erase` writing
    /// the single accumulator target.
    pub(super) sweep: wgpu::RenderPipeline,
    /// The same sweep with the ceiling lane as a second target (§6.2, §6.12) —
    /// the swept kit's `pipeline_ceiling`, on the removing side.
    pub(super) sweep_ceiling: wgpu::RenderPipeline,
    /// The integrate (`erase.wesl`): a fullscreen pass reading the pristine tile
    /// and the accumulated mass, writing the erased tile's color+aux(+resid) MRT.
    pub(super) integrate: wgpu::RenderPipeline,
    pub(super) integrate_bgl: wgpu::BindGroupLayout,
}

/// Build the erase kit (§6.12). Takes the [`SweptKit`] because the sweep
/// half *is* that path's pipeline over the same layouts — only the fragment entry
/// point and the target list differ — and both stamp modules, for the same two
/// sweeps that kit builds.
pub(super) fn build_erase_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    swept: &SweptKit,
    shader: &wgpu::ShaderModule,
    shader_ceiling: &wgpu::ShaderModule,
) -> EraseKit {
    let layout = desc::pipeline_layout(
        device,
        "stark erase sweep layout",
        &[
            Some(&swept.uniform_bgl),
            Some(&swept.prefix_bgl),
            Some(&swept.noise_bgl),
        ],
    );
    let targets = [
        // The transparency mass, additive across overlapping segment quads —
        // and, through the load below, across the pieces of a live stroke.
        desc::blended_target(ACCUM_FORMAT, Some(color_space.aux_blend())),
        // The ceiling lane and the mass's moment beside it, additive like the
        // deposit's (`swept::ceiling_target`, §6.2).
        super::swept::ceiling_target(color_space),
        desc::blended_target(ACCUM_FORMAT, Some(color_space.aux_blend())),
    ];
    let sweep_pipeline = |label, module, targets: &[Option<wgpu::ColorTargetState>]| {
        desc::render_pipeline(
            device,
            desc::RenderPipe {
                label,
                layout: &layout,
                module,
                vs: "vs_main",
                fs: "fs_erase",
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
                    wgpu::VertexStepMode::Instance,
                ))],
                targets,
            },
        )
    };
    let sweep = sweep_pipeline("stark erase sweep pipeline", shader, &targets[..1]);
    let sweep_ceiling = sweep_pipeline(
        "stark erase sweep ceiling pipeline",
        shader_ceiling,
        &targets,
    );

    let resid = color_space.has_resid();
    let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark erase"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::erase(resid).into()),
    });
    let frag = wgpu::ShaderStages::FRAGMENT;
    let integrate_bgl = desc::layout_for(device, "stark erase bgl", ERASE_SLOTS, frag, resid);
    let integrate_layout =
        desc::pipeline_layout(device, "stark erase layout", &[Some(&integrate_bgl)]);
    // No blend on any target: the shader computes the finished texel.
    let integrate = desc::fullscreen_pipeline(
        device,
        "stark erase pipeline",
        &integrate_layout,
        &integrate_shader,
        ("vs_main", "fs_main"),
        &crate::gpu::channels::ChannelFormats::of(color_space).targets(),
    );

    EraseKit {
        sweep,
        sweep_ceiling,
        integrate,
        integrate_bgl,
    }
}

impl StrokeRenderer {
    /// [`Self::render_range`] through the erase pass. `tol` comes from
    /// [`dynamics_setup`](super::dynamics::dynamics_setup), like both of its
    /// siblings' — one place answers what a stroke's segments are.
    pub(super) fn render_erase(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        resume: Resume<'_>,
        tol: crate::path::FlattenTolerance,
        tip: &ResolvedTip,
    ) -> (TileMap, StrokeCarry) {
        crate::timing::span!("stroke.erase");
        // The pool and the selection are the accumulator's — it is what acquires
        // the copy-on-write destinations and gates each one by its mask (§6.8).
        let StrokeScene {
            base, substrate, ..
        } = scene;
        let k = self.stroke_constants(rec, substrate, scene.selection);
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        if segments.is_empty() {
            return (base.clone(), StrokeCarry::unchanged(end_dist));
        }

        let mut scope = self.scratch.scope(&self.ctx, "stark erase stroke");
        let device = &self.ctx.device;

        // The brush's textures, bound exactly as the swept path binds them — one
        // derivation (`sweep_binds`), and `fs_erase` reads the same prefix-τ,
        // substrate and stroke uniform (the noise field rides along unread).
        let (prefix_bg, noise_bg) = sweep_binds(self, &mut scope, tip, rec, substrate, &k);
        // At 1× always: the supersampled resolve averages the *paint* parcel's
        // finished visible alpha (§6.2), and the erase's transparency mass runs a
        // different law through `erase.wesl` — its resolve would live there, and
        // nothing gates an eraser today (`budget::supersample_scale`).
        let draws = sweep_draws(self, &mut scope, rec, &k, &segments, 1);

        // The stroke's ceiling, once per *call* — `StrokeConstants` resolved it with
        // the color, the mask's opacity folded in, so this path cannot disagree
        // with the others about what the dial said (`BrushEffect::opacity`, §6.8).
        //
        // Run tier, like the two `sweep_draws` builds above it and like the swept
        // path's identical uniform: every tile of `accum::run` binds this one buffer,
        // so its lifetime is the whole call and not the tile being recorded.
        // `accum::run` does not flush today, which would make the piece tier correct —
        // and that is exactly the argument `swept.rs` writes out as the reason the ring
        // had to leave it. Correct because a loop happens not to flush is not correct.
        let opacity = stark_shaders::mirror::erase::Erase {
            params: [k.opacity, f32::from(u8::from(k.ceiling_lane)), 0.0, 0.0],
        };
        let opacity_buf = scope.take_run_buffer(BufKey {
            size: std::mem::size_of_val(&opacity) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark erase opacity",
        });
        scope.write_lease(&opacity_buf, bytemuck::bytes_of(&opacity));

        let keys = parcel_keys(k.ceiling_lane);

        // The shared procedure (§6.12, `accum`): resume everything the pieces
        // before this one accumulated, extend it over this piece's tiles, and turn
        // the total on the pristine paint. A tile the layer does not have is
        // nothing to erase — no output, no accumulator, no entry in the carry — so
        // a stroke over bare canvas mints no tiles at all, which is the whole of
        // what this pass says about the shape.
        let Landed { map, carry, dirty } = IncrementalTileAccumulator::resume(
            self,
            scene,
            scope,
            keys,
            BareCanvas::Skip,
            resume.prior.map(ToolState::erased),
        )
        .run(
            &Sweep {
                label: "stark erase sweep pass",
                pipeline: if k.ceiling_lane {
                    &self.erase.sweep_ceiling
                } else {
                    &self.erase.sweep
                },
                draws: &draws,
                prefix: &prefix_bg,
                noise: &noise_bg,
            },
            &Land {
                label: "stark erase integrate",
                pipeline: &self.erase.integrate,
            },
            |l: &Landing<'_>| {
                desc::bind_group_for(
                    device,
                    "stark erase bg",
                    &self.erase.integrate_bgl,
                    ERASE_SLOTS,
                    l.base.resid.is_some(),
                    |b| match b {
                        eb::BASE_COLOR => wgpu::BindingResource::TextureView(l.base.color),
                        eb::BASE_AUX => wgpu::BindingResource::TextureView(l.base.aux),
                        eb::ACCUM => wgpu::BindingResource::TextureView(l.parcel.lane(MASS)),
                        eb::SELECTION => wgpu::BindingResource::TextureView(l.mask),
                        eb::E => opacity_buf.as_entire_binding(),
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
                        other => unreachable!("`ERASE_SLOTS` lists no binding {other}"),
                    },
                )
            },
        );

        (
            map,
            StrokeCarry {
                dist: end_dist,
                progress: Progress::Finished {
                    tool: resume.capture.then(|| ToolState(Carried::Erase(carry))),
                    dirty,
                },
            },
        )
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
