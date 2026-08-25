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
//! total. The accumulators and the pristine handles ride the stroke's carry
//! ([`EraseCarry`]), the way the stamp loop's reservoir does; a piece copies the
//! accumulator it resumes rather than writing it, which is what lets the live
//! tail re-render every frame from the same frozen head.

use std::collections::BTreeMap;
use std::sync::Arc;

use stark_model::document::StrokeRecord;
use stark_model::geom::{TILE_TEX, TileCoord};
use stark_shaders::mirror::erase::binding as eb;
use stark_shaders::mirror::erase::decl as ed;
use stark_shaders::mirror::stamp_common::SWEEP_VERTS;

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use crate::gpu::tile::{AllocSource, TileMap};

use super::incremental::{Carried, EraseCarry, EraseTile};
use super::scratch::Key;
use super::segments::generate_segments_in;
use super::swept::{SweptKit, sweep_binds, sweep_draws};
use super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState, UNIFORM_STRIDE};

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

/// The accumulator's pool key: a full tile texture (interior + apron), renderable
/// (the sweep accumulates into it), bindable (the integrate reads it), and
/// copyable both ways (a resuming piece copies the carried total into its working
/// texture).
fn accum_key() -> Key {
    Key {
        size: (TILE_TEX, TILE_TEX),
        format: ACCUM_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        label: "stark erase accum",
    }
}

/// One whole accumulator, as a copy extent.
const ACCUM_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: TILE_TEX,
    height: TILE_TEX,
    depth_or_array_layers: 1,
};

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
        let StrokeScene {
            pool,
            assets,
            base,
            selection,
            substrate,
        } = scene;
        let k = self.stroke_constants(rec, substrate);
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
        let (prefix_bg, noise_bg) = sweep_binds(self, assets, rec, substrate);
        let draws = sweep_draws(self, &mut scope, rec, &k, &segments);

        // The erase opacity, once per piece — `StrokeConstants` resolved it with
        // the color, so this path cannot disagree with the others about what the
        // dial said (`BrushEffect::opacity`) — beside the strength the mask gates at
        // (§6.8), which the mask tiles this pass binds do not carry.
        let opacity = stark_shaders::mirror::erase::Erase {
            params: [k.opacity, selection.strength(), 0.0, 0.0],
        };
        let opacity_buf = scope.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark erase opacity"),
            size: std::mem::size_of_val(&opacity) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx
            .queue
            .write_buffer(&opacity_buf, 0, bytemuck::bytes_of(&opacity));

        // The carry this piece hands on: everything the pieces before it
        // accumulated — shared, never rewritten — with this piece's tiles
        // replacing theirs below.
        let mut tiles: BTreeMap<TileCoord, EraseTile> = match tool.map(ToolState::erased) {
            Some(prior) => prior
                .tiles
                .iter()
                .map(|(c, t)| {
                    (
                        *c,
                        EraseTile {
                            pristine: t.pristine.clone(),
                            accum: Arc::clone(&t.accum),
                        },
                    )
                })
                .collect(),
            None => BTreeMap::new(),
        };

        let mut new_map = base.clone();
        let mut dirty = Vec::new();
        for (i, coord) in draws.coords.iter().enumerate() {
            // The paint the stroke found under this tile: what an earlier piece
            // recorded, or — for a tile this stroke reaches for the first time —
            // the base itself, which no earlier piece can have rewritten. A tile
            // the layer does not have is nothing to erase: no output, no
            // accumulator, and no entry in the carry, so a stroke over bare
            // canvas mints no tiles at all.
            let Some(pristine) = tiles
                .get(coord)
                .map(|t| t.pristine.clone())
                .or_else(|| base.get(coord).cloned())
            else {
                continue;
            };

            // This piece's working accumulator: the carried total copied in, or a
            // clear for a first touch — either way every texel is written before
            // the integrate reads it, the pool's no-zero-init contract
            // (`scratch`). The carried texture itself is only ever read: the live
            // tail resumes the same frozen carry on every pointer move.
            let work = self.scratch.keep(device, accum_key());
            let resumed = tiles.get(coord).map(|t| Arc::clone(&t.accum));
            if let Some(old) = &resumed {
                scope.encoder().copy_texture_to_texture(
                    old.tex().as_image_copy(),
                    work.tex().as_image_copy(),
                    ACCUM_EXTENT,
                );
            }
            {
                let ops = if resumed.is_some() {
                    desc::LOAD
                } else {
                    desc::CLEAR
                };
                let att = [Some(desc::attach(work.view(), ops))];
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark erase sweep pass"),
                        color_attachments: &att,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.erase.sweep);
                pass.set_bind_group(0, &draws.xforms, &[(i * UNIFORM_STRIDE) as u32]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &noise_bg, &[]);
                pass.set_vertex_buffer(0, draws.instances.slice(..));
                pass.draw(0..SWEEP_VERTS, draws.runs[i].clone());
            }

            // The whole stroke's extent so far, turned on the pristine paint —
            // never on the base in hand, which for a resumed tile is an earlier
            // piece's output and would compound the erase per piece.
            let dst = self.acquire_tile(pool, AllocSource::IntegrateDestination);
            // A gating read: the strength travels with the erase opacity
            // uniform above, built from this same selection.
            let mask_view = self.selection.gate_for(selection, *coord);
            let has_resid = pristine.resid_view().is_some();
            let bg = desc::bind_group_for(
                device,
                "stark erase bg",
                &self.erase.integrate_bgl,
                ERASE_SLOTS,
                has_resid,
                |b| match b {
                    eb::BASE_COLOR => wgpu::BindingResource::TextureView(pristine.color_view()),
                    eb::BASE_AUX => wgpu::BindingResource::TextureView(pristine.aux_view()),
                    eb::ACCUM => wgpu::BindingResource::TextureView(work.view()),
                    eb::SELECTION => wgpu::BindingResource::TextureView(mask_view.view()),
                    eb::E => opacity_buf.as_entire_binding(),
                    eb::BASE_RESID => wgpu::BindingResource::TextureView(
                        pristine.resid_view().expect("a residual build has one"),
                    ),
                    other => unreachable!("`ERASE_SLOTS` lists no binding {other}"),
                },
            );
            {
                let int_targets = dst.targets();
                let int_att = int_targets.attachments(desc::CLEAR);
                let mut pass = scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark erase integrate"),
                        color_attachments: &int_att[..int_targets.count()],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                pass.set_pipeline(&self.erase.integrate);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }

            new_map = new_map.insert(*coord, dst);
            dirty.push(*coord);
            tiles.insert(
                *coord,
                EraseTile {
                    pristine,
                    accum: Arc::new(work),
                },
            );
        }

        // Submit before the carry leaves this call: a `Kept` may reach the pool's
        // free list only behind the submit of the commands naming it, and handing
        // the carry out first would let a caller drop it ahead of one.
        scope.finish();
        (
            new_map,
            StrokeCarry {
                dist: end_dist,
                tool: Some(ToolState(Carried::Erase(EraseCarry { tiles }))),
                dirty,
            },
        )
    }
}
