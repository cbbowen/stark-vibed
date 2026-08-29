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
    BareCanvas, IncrementalTileAccumulator, Land, Landed, Landing, Sweep, lane_key,
};
use super::incremental::Carried;
use super::scratch::{BufKey, Key};
use super::segments::generate_segments_in;
use super::swept::{SweptKit, sweep_binds, sweep_draws};
use super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState};

/// The integrate's one group (`erase.wesl`): the pristine tile, the stroke's
/// accumulated mass, the selection, and the opacity uniform.
const ERASE_SLOTS: &[Slot] = &[
    Slot::at(ed::BASE_COLOR),
    Slot::at(ed::BASE_AUX),
    Slot::at(ed::ACCUM),
    Slot::at(ed::SELECTION),
    Slot::at(ed::E),
    Slot::at(ed::BASE_RESID),
];

/// The accumulator's format: one channel — the transparency mass — additive, like
/// the persistent aux it sits beside in size and precision. f16 is enough for the
/// same reason it is enough there: the interesting range is a few times
/// `OPAQUE_MASS`, and by the time accumulation outruns f16's mantissa the survival
/// `exp(−m)` has long since reached zero.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The accumulator is the parcel's only lane: this pass rasterizes one number per
/// texel, where the deposit rasterizes the channel trio. Named beside the key it is
/// taken with, so the attach order and the bind order are one list
/// ([`Parcel`](super::accum::Parcel)).
const MASS: usize = 0;

/// The accumulator's pool key — [`lane_key`]'s usages, at this pass's own format
/// and label.
fn accum_key() -> Key {
    lane_key(ACCUM_FORMAT, "stark erase accum")
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
    /// The integrate (`erase.wesl`): a fullscreen pass reading the pristine tile
    /// and the accumulated mass, writing the erased tile's color+aux(+resid) MRT.
    pub(super) integrate: wgpu::RenderPipeline,
    pub(super) integrate_bgl: wgpu::BindGroupLayout,
}

/// Build the erase kit (§6.12). Takes the [`SweptKit`] because the sweep
/// half *is* that path's pipeline over the same layouts — only the fragment entry
/// point and the target list differ.
pub(super) fn build_erase_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    swept: &SweptKit,
) -> EraseKit {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark erase sweep"),
        source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader().into()),
    });
    let layout = desc::pipeline_layout(
        device,
        "stark erase sweep layout",
        &[
            Some(&swept.uniform_bgl),
            Some(&swept.prefix_bgl),
            Some(&swept.noise_bgl),
        ],
    );
    let sweep = desc::render_pipeline(
        device,
        desc::RenderPipe {
            label: "stark erase sweep pipeline",
            layout: &layout,
            module: &shader,
            vs: "vs_main",
            fs: "fs_erase",
            primitive: desc::QUAD_STRIP,
            buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
                wgpu::VertexStepMode::Instance,
            ))],
            // The transparency mass, additive across overlapping segment quads —
            // and, through the load below, across the pieces of a live stroke.
            targets: &[desc::blended_target(
                ACCUM_FORMAT,
                Some(color_space.aux_blend()),
            )],
        },
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
        tool: Option<&ToolState>,
        tol: crate::path::FlattenTolerance,
    ) -> (TileMap, StrokeCarry) {
        crate::timing::span!("stroke.erase");
        // The pool and the selection are the accumulator's — it is what acquires
        // the copy-on-write destinations and gates each one by its mask (§6.8).
        let StrokeScene {
            assets,
            base,
            substrate,
            ..
        } = scene;
        let k = self.stroke_constants(rec, substrate, scene.selection);
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        if segments.is_empty() {
            return (
                base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                    dirty: Vec::new(),
                },
            );
        }

        let mut scope = self.scratch.scope(&self.ctx, "stark erase stroke");
        let device = &self.ctx.device;

        // The brush's textures, bound exactly as the swept path binds them — one
        // derivation (`sweep_binds`), and `fs_erase` reads the same prefix-τ,
        // substrate and stroke uniform (the noise field rides along unread).
        let (prefix_bg, noise_bg) = sweep_binds(self, &mut scope, assets, rec, substrate, &k);
        let draws = sweep_draws(self, &mut scope, rec, &k, &segments);

        // The stroke's ceiling, once per piece — `StrokeConstants` resolved it with
        // the color, the mask's opacity folded in, so this path cannot disagree
        // with the others about what the dial said (`BrushEffect::opacity`, §6.8).
        let opacity = stark_shaders::mirror::erase::Erase {
            params: [k.opacity, 0.0, 0.0, 0.0],
        };
        let opacity_buf = scope.take_piece_buffer(BufKey {
            size: std::mem::size_of_val(&opacity) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark erase opacity",
        });
        self.ctx
            .queue
            .write_buffer(&opacity_buf, 0, bytemuck::bytes_of(&opacity));

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
            &[accum_key()],
            BareCanvas::Skip,
            tool.map(ToolState::erased),
        )
        .run(
            &Sweep {
                label: "stark erase sweep pass",
                pipeline: &self.erase.sweep,
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
                        other => unreachable!("`ERASE_SLOTS` lists no binding {other}"),
                    },
                )
            },
        );

        (
            map,
            StrokeCarry {
                dist: end_dist,
                tool: Some(ToolState(Carried::Erase(carry))),
                dirty,
            },
        )
    }
}
