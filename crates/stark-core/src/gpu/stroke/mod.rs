//! The brush engine: **swept-segment** stroke rasterization with copy-on-write
//! tiles (§6.2, §5.2, §6.6, §6.7).
//!
//! Rather than stamping discrete dabs, each short segment of the flattened curve
//! is drawn as one oriented quad whose coverage is the brush *swept* along it —
//! the path integral of the footprint. Because alpha-"over" is additive in
//! optical depth `τ = −ln(1−α)`, the swept depth of a segment is a difference of
//! the brush's precomputed prefix-τ texture (`prefix(u) − prefix(u−d)`), and the
//! existing premultiplied-over blend across overlapping segment quads sums those
//! depths *exactly* — reconstructing the continuous stroke with no banding, no
//! scratch buffer, and no second pass.
//!
//! That is the plain **add** fast path: footprint → cleared scratch tile →
//! integrate over the base into a fresh CoW tile. A brush that also moves paint
//! already on the canvas (`lift` / `deposit` / `charge`, §6.2) instead runs
//! the sequential swept-exchange loop in `dynamics.wesl`; `dynamics_setup`
//! decides which path a record takes.
//!
//! The renderer is parameterized by a [`ColorSpace`] (formats, blends, channel
//! mapping, shader). It holds only immutable GPU objects plus `Arc`-backed
//! handles, so it is cheap to `Clone` and can live in the `Action::Context` (§5).

use std::sync::{Arc, Mutex};

use rpds::HashTrieMap;

use crate::assets::{AssetStore, build_coverage_r8, build_prefix_tau};
use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::document::{BrushParams, BrushShape, ColorDynamics, NoiseKind, StrokeRecord};
use crate::geom::TileCoord;
use crate::gpu::context::GpuContext;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TilePairHandle, TilePool};

mod dynamics;
mod segments;
mod swept;

use dynamics::{
    DynamicsKit, StrokePath, build_dynamics_kit, build_integrate_pipeline, dynamics_setup,
};
use segments::{SegmentInstance, round_coverage};

/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

/// Resolution (texels per side) of the stamp loop's tool reservoir
/// (§6.2). Brush-local, so carried colour detail is ~radius/32 canvas px — plenty
/// for smeared paint, and small enough that the per-stamp reservoir update is
/// nearly free.
const BRUSH_RES: u32 = 64;
/// Resolution of the per-segment **swept prefix** of the reservoir
/// (`dynamics.wesl::bake`). Finer than the reservoir along the travel axis, since
/// it also has to resolve the footprint's optical-depth density it integrates
/// against; the bake is a one-workgroup-per-row shared-memory scan, so this costs
/// almost nothing. Must match the shader's own `BAKE_RES` (its workgroup width).
const BAKE_RES: u32 = 128;
/// fp32, for the same reason the prefix-τ volume is: every fragment reads it as a
/// *difference* of two prefix sums (§6.2).
const BAKE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The optical depth one full pass of an opaque tip lays over a point — the τ
/// ceiling `assets::build_prefix_tau` clamps to.
///
/// Every exchange in the stamp loop is a rate *per unit optical depth*, because
/// that is the currency the swept integral is denominated in and the only one both
/// sides can agree on (§6.2). But τ ≈ 7 for a single pass, so read
/// literally a `lift` of 0.5 would strip 99% of the canvas in one pass. Dividing
/// the rates through by this makes an axis mean a fraction **per pass of the tip**
/// — hardness-independent, and what a 0..1 knob is expected to mean.
const TAU_PER_PASS: f32 = 6.9;
/// Largest region edge (canvas px) the stamp loop composites at once. A stroke that
/// wants more is drawn in as many region-sized *pieces* as it takes
/// ([`segments::chunk_segments`]), so this bounds the loop's transient GPU memory —
/// at 2048² the colour and aux regions are ~34 MB together — rather than deciding
/// which strokes the loop can draw at all (§6.2).
const MAX_REGION_DIM: u32 = 2048;
/// Cap on the segments one piece dispatches, which bounds its stamp uniform buffer.
/// Reached only by a stroke fine enough to fill a whole region with segments, and it
/// cuts a new piece rather than coarsening anything.
const MAX_STAMPS: usize = 4096;
/// Gain on the `add` axis in the stamp loop, tuned so `add = 1` lays roughly a
/// full-thickness deposit per pass of the tip. The swept fast path needs no
/// counterpart: it lays height directly as the brush's rate times the swept
/// optical depth (`stamp_oklab.wesl`), with no correction factor.
const ADD_GAIN: f32 = 2.0;
/// How far the tool may travel per exchange, as a fraction of the brush radius
/// (§6.2) — which, since the tool now exchanges once per *segment*, is
/// simply a cap on the flattened segment length for a dynamics brush
/// (see [`flatten_tolerance`]).
///
/// A property of the exchange loop, not of the tip: it sets how finely the reservoir
/// tracks the evolving canvas, and nothing about a shape's coverage mask should change
/// it. It was once a cadence of its own — the tool reloaded every `spacing·radius`
/// while the canvas was stripped every segment — and the lag between the two is what
/// left a stroke's last footprint short of paint (`dynamics.wesl`).
///
/// **The loop is first order in this constant, and nothing here is free.** Measured on
/// `golden_drained_brush_length_independent` — a tip that runs dry and then *carries*
/// paint 400px into view, so every visible pixel arrived through the reservoir and the
/// transport error has nowhere to hide — against a reference at 0.03125:
///
/// ```text
///   step     error vs reference     length-dependence     order
///   1.0        93 max / 49.7 rms      15.8 max / 3.69
///   0.5        51 max / 23.2 rms       8.0 max / 2.27      1.10
///   0.25       24 max / 10.1 rms       3.6 max / 0.82      1.20
///   0.125      13 max /  5.3 rms       1.6 max / 0.34      0.94
///   0.0625      8 max /  2.7 rms       1.6 max / 0.28      0.96
/// ```
///
/// Clean first order, no knee to sit on. The second column is what makes it a bug and
/// not a tolerance: the flattener bisects, so a span's segment length depends on the
/// *whole path's* length, and the same visible stretch of stroke therefore renders
/// differently depending on where the pen went afterwards. The error prints as one
/// tip-shaped arc per segment — the tool lifts at a point and lays back down swept, so
/// the smear translates the canvas by exactly one segment length per segment, which is
/// a delay line ringing at the segment cadence. 0.125 is where that falls into the
/// 8-bit quantization noise.
///
/// It is worth knowing why 0.5 looked fine for a while. The goldens could not see it:
/// nearly every one of them paints with the shared `brush()` helper, which sets
/// `drain = 0.0015`, and `drain` used to impose its own `0.02 / drain` = 13.3px cap on
/// segment length. For any tip wider than 13.3px that cap was the tighter of the two,
/// so the goldens rendered at 13.3px segments *whatever this constant said* — a change
/// here moved nothing, and looked free. Only once the drain cap was retired (it is
/// evaluated per fragment now, see [`flatten_tolerance`]) did this become the binding
/// constraint and start deciding pixels. A benchmark or a golden that does not move is
/// evidence about the test, not about the change. For a radius-80 tip that old cap
/// worked out to 0.166, so this value is very close to what actually shipped; the step
/// was never really at 0.5.
///
/// **Four cheaper things were tried and none of them work**, which is worth recording
/// because each looks obvious:
///
/// * *Averaging the canvas along the reservoir texel's track* instead of the single
///   midpoint tap `dynamics.wesl::exchange` takes. Changes the result by less than the
///   8-bit noise floor on both this test and the pointer-sample-density spread it was
///   meant for. The midpoint tap is not the error.
/// * *Sub-stepping the tool's own kernel* over `e/N`. It looks like refinement and is a
///   different model: the tool lifts its share of a canvas held fixed, N times over,
///   while the deposit gives up a single share of `e`, so the halves stop being
///   complements. At a step four times finer than this one it lands 12 levels rms from
///   where the single step converges.
/// * *Baking the post-exchange reservoir* rather than the entering one. Tempting — it
///   scores 5.0 rms at a step of 0.5, better than the honest scheme manages at 0.125 —
///   and it is a leak: the canvas receives a share of a reservoir the tool never gave
///   up. It converges to a *different* answer, 3.2 rms from the true one, and stalls
///   there however fine the step. The good score at 0.5 is discretization error
///   cancelling the bias.
/// * *Matching `BAKE_RES` to the prefix-τ volume's 256.* No effect; the two grids
///   meeting in `deposit`'s ratio are not the problem.
///
/// The one thing that does help is replacing the closed-pair kernel with a **sliding**
/// one — `keep = exp(−k_lift·e)` rather than the pair's `1 − k_lift·w(e)`, on the
/// grounds that a canvas point does not stay under one reservoir cell for a segment but
/// slides through a stream of them, so the pair's saturation at `k_lift/s` is wrong at
/// any coarse step. That converges to the same answer and is ~2.5× more accurate at
/// every step. It also gives up the exact complementarity the transfer is built on
/// (`dynamics.wesl`, and the 39%-of-height-vanishing story there), so it is a design
/// change and not a constant, and it is not made here.
const RESERVOIR_EXCHANGE_STEP: f32 = 0.125;
/// Cap on `radius · |curvature|`: how fat the tip may be relative to the turn it is
/// swept through before the segment goes back to being straight (§6.2).
///
/// Both shaders sweep a curved segment by **unrolling** the annulus about its centre
/// of curvature into the straight travel frame, which treats a canvas point as sliding
/// through the tip frame along a line of constant lateral offset. It does not: the
/// true track is an arc of radius `ρ`, so a point out at the footprint's shoulder is
/// off that line by `≈ r²/2R`, i.e. **`radius · |curvature| / 2` as a fraction of the
/// tip radius**. That is the constant's real job. The annular sector the swept path
/// rasterizes also folds over itself once `radius ≥ |R|`, but that bound (1.0) is five
/// times looser and never the one that bites.
///
/// 0.1 holds the lateral error to 5% of the tip. It was 0.5 — 25% — which the plain
/// swept deposit absorbed (its segments overlap heavily and the error is smooth) but
/// the dynamics loop did not: there the same offset picks the wrong reservoir texel to
/// serve a canvas texel, and because the loop is sequential the error compounds down
/// the stroke into crescent seams at the reservoir cadence, worst where the tool is
/// dragging paint with nothing left to `add` over them.
const MAX_TIP_TURN: f32 = 0.1;

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    pipeline: wgpu::RenderPipeline,
    uniform_bgl: wgpu::BindGroupLayout,
    prefix_bgl: wgpu::BindGroupLayout,
    /// Cached round-tip prefix-τ, keyed by `hardness.to_bits()`.
    round_prefix: Arc<Mutex<Option<(u32, wgpu::TextureView)>>>,
    /// Colour dynamics (§6.2): the sweep's noise bind group layout
    /// (group 2), the shared wrap/linear sampler, the 1×1×1 zero volume bound
    /// when a brush's jitter is off, and the lazily-baked per-kind fields.
    noise_bgl: wgpu::BindGroupLayout,
    noise_sampler: wgpu::Sampler,
    dummy_noise: wgpu::TextureView,
    noise_cache: Arc<Mutex<Vec<(NoiseKind, wgpu::TextureView)>>>,

    // Stroke integrate (§6.2/§6.1): a fullscreen pass reads the base tile +
    // the stroke's footprint scratch and writes `new = f(base, scratch)` into a fresh
    // CoW tile's color+aux MRT — premultiplied-over + additive height.
    integrate_pipeline: wgpu::RenderPipeline,
    integrate_bgl: wgpu::BindGroupLayout,

    // Brush dynamics: the sequential stamp loop (§6.2), used when the
    // brush manipulates existing paint (`load` / `deposit` / `charge`).
    dynamics: DynamicsKit,

    /// Selection masks (§6.8): the per-tile mask bound into the integrate
    /// pass, and the region gather the stamp loop reads. Colour-space independent, so
    /// it is handed in rather than rebuilt with the rest of this renderer.
    selection: SelectionRenderer,
}

/// The stamp loop's carried state at a cut point in a stroke (§6.2).
///
/// The sequential loop threads exactly two things from one segment to the next that
/// do not already live on the canvas: the **tool reservoir** — what paint the tip is
/// carrying, and where on the tip it sits — and how far the tip has travelled since
/// it last exchanged with the canvas. Remember those at a span boundary and the rest
/// of the stroke can be drawn later, over the already-composited head, for the same
/// result as one pass. That is what lets a `lift`/`deposit`/`charge` brush get the
/// same incremental repaint the swept path gets.
///
/// The reservoir is brush-*local*, which is why this works at all: it says nothing
/// about where the stroke is, so the region rectangle may change completely between
/// the piece that produced this state and the piece that resumes from it.
pub struct ToolState {
    /// Reservoir colour: per texel, the latent paint (rgb) and its per-unit opacity.
    color: wgpu::Texture,
    /// Reservoir aux: per texel, the carried amount (height).
    aux: wgpu::Texture,
}

impl Drop for ToolState {
    fn drop(&mut self) {
        // Destroyed eagerly for the same reason as `ScopedResources`: one of these is
        // allocated per pointer move, and waiting on JS GC to free them OOMs the tab.
        // Safe because WebGPU defers the real free until the submitted work that reads
        // them retires.
        self.color.destroy();
        self.aux.destroy();
    }
}

/// Everything a stroke is drawn *against*, as opposed to the stroke itself.
///
/// [`StrokeRenderer`] holds only immutable GPU objects — pipelines, layouts, the
/// prefix-τ cache — so the mutable scene is handed in per call. These four travel
/// together through every entry point ([`StrokeRenderer::render`],
/// [`render_range`](StrokeRenderer::render_range), and both paths underneath), so
/// they are one parameter rather than four repeated at each hop.
#[derive(Copy, Clone)]
pub struct StrokeScene<'a> {
    pub pool: &'a TilePool,
    pub assets: &'a AssetStore,
    /// The layer's committed tiles: what the stroke composites over.
    pub base: &'a HashTrieMap<TileCoord, TilePairHandle>,
    /// The selection in force, which gates the deposit (§6.8).
    pub selection: &'a Selection,
}

/// What a range render leaves behind for the range that resumes after it.
pub struct StrokeCarry {
    /// Arc length at the end of the range. Not derivable from the span index — it is
    /// measured along the flattened polyline — and both the `drain` falloff and the
    /// colour-dynamics noise read it, so restarting it at zero would make the middle
    /// of a stroke look like the start of one.
    pub dist: f32,
    /// The brush state to resume with, for a stroke that runs the stamp loop. `None`
    /// means *nothing changed*: the swept path carries no state at all, a range that
    /// reaches the end of the stroke has nothing following it to hand off to, and a
    /// range with no geometry leaves the brush as it found it — so a caller holding
    /// earlier state should keep it rather than treat this as a reset.
    pub tool: Option<ToolState>,
    /// The tiles this range rewrote — everything in the returned map that differs
    /// from `scene.base`.
    ///
    /// The renderer already enumerates these to decide what to draw
    /// ([`affected_tiles`](segments::affected_tiles)), and reporting them is what
    /// lets several in-flight strokes be composited over one committed document
    /// without diffing whole tile maps (§17.6).
    pub dirty: Vec<TileCoord>,
}

/// GPU resources scoped to one `render()` call (currently the instance buffer;
/// per-stroke region textures register here too as dynamics return). They're sized
/// per-stroke, so — unlike the fixed-`TILE_TEX` tile pool — they can't be recycled,
/// and a *live* stroke re-renders on every pointer move. Left to drop they'd only
/// release the JS handle and wait on GC, which can't keep up → the tab OOMs. So
/// they're collected here (cheap `Arc` clones) and **`destroy()`d on drop**, which
/// `render` arranges to happen right after the submit — safe, because WebGPU defers
/// the real free until the in-flight work referencing them completes.
#[derive(Default)]
struct ScopedResources {
    textures: Vec<wgpu::Texture>,
    buffers: Vec<wgpu::Buffer>,
}

impl ScopedResources {
    /// Register a per-stroke texture; returns it unchanged (the clone keeps the GPU
    /// resource alive until this `ScopedResources` drops).
    fn texture(&mut self, tex: wgpu::Texture) -> wgpu::Texture {
        self.textures.push(tex.clone());
        tex
    }

    /// Register a per-stroke buffer; returns it unchanged.
    fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.buffers.push(buf.clone());
        buf
    }

    /// Whether anything is registered — how the stamp loop tells a first piece of a
    /// stroke from a later one (see `dynamics::DynamicsRun::flush`).
    fn is_empty(&self) -> bool {
        self.textures.is_empty() && self.buffers.is_empty()
    }
}

impl Drop for ScopedResources {
    fn drop(&mut self) {
        if !self.textures.is_empty() || !self.buffers.is_empty() {
            tracing::trace!(
                textures = self.textures.len(),
                buffers = self.buffers.len(),
                "destroying scoped stroke resources",
            );
        }
        for tex in self.textures.drain(..) {
            tex.destroy();
        }
        for buf in self.buffers.drain(..) {
            buf.destroy();
        }
    }
}

impl StrokeRenderer {
    pub fn new(
        ctx: &GpuContext,
        color_space: Arc<dyn ColorSpace>,
        selection: SelectionRenderer,
    ) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark sweep"),
            source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader().into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep uniform bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // The prefix-τ texture is a R32Float 2D-array (x, y, + orientation layers), sampled
        // via textureLoad (not filterable), so the shader does its own trilinear lookup.
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep prefix bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            }],
        });

        // Group 2: the colour-dynamics noise field (a tileable 3-D volume) + its
        // repeat sampler (§6.2).
        let noise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep noise bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Wrapping on both axes — the noise tile tiles (that's the whole point).
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark noise sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (_dummy_tex, dummy_noise) = crate::noise::dummy_noise_texture(ctx);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark sweep layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&prefix_bgl), Some(&noise_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark sweep pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x4
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_space.color_format(),
                        blend: Some(color_space.color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // The stamp renders into a *scratch* tile, whose aux is the wide
                    // SCRATCH_AUX_FORMAT — not the compact persistent aux. Additive
                    // blend across overlapping segments.
                    Some(wgpu::ColorTargetState {
                        format: SCRATCH_AUX_FORMAT,
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let (integrate_pipeline, integrate_bgl) =
            build_integrate_pipeline(device, color_space.as_ref());
        let dynamics = build_dynamics_kit(device, color_space.as_ref());

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
            noise_bgl,
            noise_sampler,
            dummy_noise,
            noise_cache: Arc::new(Mutex::new(Vec::new())),
            integrate_pipeline,
            integrate_bgl,
            dynamics,
            selection,
        }
    }

    /// Render `rec` over `base`, gated by `selection`, returning a copy-on-write tile
    /// map.
    ///
    /// The selection is applied at the *end* of each path — the integrate pass's
    /// merge on the fast path, the deposit's write-back in the stamp loop — rather
    /// than by clipping the footprint. That keeps one rule for both paths (a texel
    /// receives the mask's fraction of whatever the stroke did there) and is what
    /// makes a feathered selection fade a stroke out instead of scaling its optical
    /// depth, which for an opaque brush would barely fade at all (§6.8).
    pub fn render(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
    ) -> HashTrieMap<TileCoord, TilePairHandle> {
        self.render_range(scene, rec, StrokeSpans::whole(rec), None)
            .0
    }

    /// Render just `spans` of `rec` over `base`, resuming the brush from `tool` — the
    /// state the preceding range left it in — and returning what a range picking up
    /// where this one stops must resume from ([`StrokeCarry`]).
    ///
    /// This is what makes a live stroke cost its *tail* rather than its whole length
    /// (§6.2). On the swept path it is sound because the deposit is a
    /// definite integral over each segment and composes by summing optical depth, so
    /// cutting the path at a span boundary and compositing the two halves in order
    /// gives the same result as one pass — the same property that lets adaptive
    /// flattening choose segment lengths freely. On the stamp loop it is sound because
    /// [`ToolState`] carries the only thing the loop threads between segments that is
    /// not already on the canvas. Adjacent ranges share exactly one flattened point
    /// (`path::flatten_spans`), so their segments tile with no gap and no overlap.
    pub fn render_range(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&ToolState>,
    ) -> (HashTrieMap<TileCoord, TilePairHandle>, StrokeCarry) {
        // Which path the stroke takes — and how finely it flattens — is decided from
        // the record, never from the piece in hand. A live tail and the commit that
        // eventually replaces it have to make the same choice, or releasing the pointer
        // would visibly redraw the stroke. See `dynamics_setup`.
        match dynamics_setup(rec) {
            StrokePath::Loop(tol) => self.render_dynamic(scene, rec, spans, tool, tol),
            StrokePath::Swept => self.render_swept(scene, rec, spans),
            StrokePath::TipTooLarge => {
                // An error, not a warning: what lands is not a rougher version of the
                // stroke that was asked for but a different brush — the swept deposit
                // only ever adds paint, so `lift`, `deposit` and `charge` all silently
                // do nothing. It is the one degradation left (stroke *length* is
                // handled by drawing the stroke in pieces, §6.2), and no
                // brush the UI can build reaches it, so hitting it means a record came
                // from somewhere else and is not being honoured. It repeats per
                // pointer move, because the gate is re-asked per render.
                tracing::error!(
                    radius = rec.brush.radius,
                    max_region_dim = MAX_REGION_DIM,
                    "brush tip too large for one dynamics region: falling back to the \
                     swept deposit, so this stroke's lift/deposit/charge do nothing",
                );
                self.render_swept(scene, rec, spans)
            }
        }
    }

    /// Acquire a persistent tile: the colour space's `color` + `aux` formats, paired.
    ///
    /// The pool hands out textures, not tiles (see [`TexHandle`](crate::gpu::tile::TexHandle)).
    /// Pairing them here is what keeps the two formats coming from the colour space
    /// actually in use rather than from a constant — the pool previously hardcoded
    /// `R16Float` for aux, which happened to match every colour space but would have
    /// panicked on the first one that chose otherwise (§6.7).
    fn acquire_tile(&self, pool: &TilePool, source: AllocSource) -> TilePairHandle {
        TilePairHandle::new(
            pool.acquire_tex(self.color_space.color_format(), source),
            pool.acquire_tex(self.color_space.aux_format(), source),
        )
    }

    /// Acquire a brush-dynamics *scratch* tile: the same colour channel, but a wider
    /// [`SCRATCH_AUX_FORMAT`] aux (an extra channel the deposit/integrate use
    /// internally, §6.2).
    fn acquire_scratch(&self, pool: &TilePool, source: AllocSource) -> TilePairHandle {
        TilePairHandle::new(
            pool.acquire_tex(self.color_space.color_format(), source),
            pool.acquire_tex(SCRATCH_AUX_FORMAT, source),
        )
    }

    /// The brush's swept-footprint prefix-τ texture: an image brush's from the asset
    /// store, the round tip's generated (and cached) from its hardness.
    ///
    /// Both render paths resolve it the same way — they differ in which bind-group
    /// layout they hang it off, not in how the texture is chosen.
    fn prefix_view(&self, assets: &AssetStore, brush: &BrushParams) -> wgpu::TextureView {
        match brush.shape {
            BrushShape::Stamp(id) => assets
                .prefix_view(id)
                .unwrap_or_else(|| self.round_prefix(BrushShape::DEFAULT_HARDNESS)),
            BrushShape::Round { hardness } => self.round_prefix(hardness),
        }
    }

    /// The round tip's prefix-τ texture for a given `hardness`, cached so live
    /// preview (which re-renders per pointer move) doesn't rebuild it each frame.
    fn round_prefix(&self, hardness: f32) -> wgpu::TextureView {
        let key = hardness.to_bits();
        let mut cache = self.round_prefix.lock().expect("round prefix poisoned");
        if let Some((k, view)) = cache.as_ref()
            && *k == key
        {
            return view.clone();
        }
        // The round tip is rotation-invariant, so a single orientation layer suffices —
        // the shader's wrapping lookup reads it for every orientation (§6.6).
        let coverage = round_coverage(hardness, ROUND_RES);
        let (_tex, view) = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, 1, &coverage);
        *cache = Some((key, view.clone()));
        view
    }

    /// The colour-dynamics noise tile for a brush: the baked field for its
    /// kind (built once, cached — the bake is a fixed pure function, so at most
    /// one texture per [`NoiseKind`] ever exists), or the 1×1 zero tile when
    /// the jitter is off (amplitudes all 0 ⇒ the shader adds exactly nothing).
    fn noise_view(&self, cd: &ColorDynamics) -> wgpu::TextureView {
        if !cd.is_active() {
            return self.dummy_noise.clone();
        }
        let mut cache = self.noise_cache.lock().expect("noise cache poisoned");
        if let Some((_, view)) = cache.iter().find(|(k, _)| *k == cd.noise) {
            return view.clone();
        }
        let (_tex, view) = crate::noise::build_noise_texture(&self.ctx, cd.noise);
        cache.push((cd.noise, view.clone()));
        view
    }

    /// The round tip's coverage texture for `hardness`, cached like the prefix.
    fn round_coverage_view(&self, hardness: f32) -> wgpu::TextureView {
        let key = hardness.to_bits();
        let mut cache = self
            .dynamics
            .round_cov
            .lock()
            .expect("round coverage poisoned");
        if let Some((k, view)) = cache.as_ref()
            && *k == key
        {
            return view.clone();
        }
        let cov = round_coverage(hardness, ROUND_RES);
        let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
        let (_tex, view) = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
        *cache = Some((key, view.clone()));
        view
    }
}

/// The flattening budget for a brush (§6.2). The error bounds are
/// brush-independent — sub-pixel position, a small tangent turn, a small attribute
/// step — but a segment is swept with *constant* attributes, so any brush quantity
/// that varies with distance travelled and is applied per segment (rather than
/// recovered per fragment, as the colour-dynamics arc is) needs a length cap too.
pub(crate) fn flatten_tolerance(b: &BrushParams) -> crate::path::FlattenTolerance {
    let mut tol = crate::path::FLATTEN_TOLERANCE;
    // Use a more relaxed tolerance for larger brushes.
    tol.position = tol.position.max(0.01 * b.radius);
    // The tightest arc this tip may be swept along (§6.2). Both the
    // flattener and the segment generator get it from here, so an edge too tight to
    // sweep as an arc is priced as a chord as well as drawn as one.
    tol.max_arc_curvature = MAX_TIP_TURN / b.radius.max(0.5);
    // `drain` used to be bought here, at `0.02 / drain` px per segment — a cap that
    // could dominate everything else (at `drain = 0.02`, one segment per pixel). It is
    // gone because the falloff is no longer a per-segment constant: both paths
    // evaluate it from the fragment's own arc length, so the amount laid is exactly
    // independent of how the path was cut and there is nothing left for a length cap
    // to bound (`generate_segments_in`).
    // The stamp loop exchanges once per segment, so the segment length *is* the step
    // at which the tool reloads and drains — and unlike the canvas side, which the
    // prefix-τ integral makes exact at any length, that step is a plain first-order
    // discretization of a coupled ODE. [`RESERVOIR_EXCHANGE_STEP`] is what keeps it
    // fine enough. The cap also bounds the snapshot scratch, which is sized by the
    // longest segment.
    let d = b.dynamics;
    if d.lift > 0.0 || d.deposit > 0.0 || d.charge > 0.0 {
        tol.max_len = tol
            .max_len
            .min((RESERVOIR_EXCHANGE_STEP * b.radius).max(0.5));
    }
    tol
}

/// How many leading spans of a *live* stroke may be rendered once and kept, given
/// that the fitter has settled `frozen` of them (§6.2).
///
/// Freezing is what makes a long live stroke cost its tail rather than its length
/// ([`StrokeRenderer::render_range`]), and it rests on a frozen span's pixels being
/// final. A taper breaks that on its own terms: it is measured from the ends of the
/// **whole** stroke, and while the pointer is down the far end has not happened yet.
/// Bake the trailing taper into a span too early and the stroke would carry a pinch
/// in its middle that the commit does not — the live == committed invariant (§1.3),
/// failing in the one place it cannot be repainted.
///
/// So a span is held back unless both of the taper's inputs are already settled for
/// it:
///
/// * it is at least the trailing taper's length before the stroke's end, so its
///   trailing factor is 1 — and stays 1, since the stroke only gets longer;
/// * it is at least the leading taper's length past the start, which together with
///   the first condition proves the stroke is already longer than the two zones
///   together, so the "scale both to fit" compression ([`segments::Taper`]) is 1 and
///   likewise stays 1.
///
/// Both are tested on **chords**, which under-estimate arc length — so a span this
/// admits genuinely satisfies them, and a stroke that doubles back near its own
/// start or end merely re-renders a little more than it had to. Only the last span
/// in the candidate prefix is tested: arc length increases monotonically along the
/// stroke, so it is the hardest case, and once a prefix is admitted it stays
/// admissible however the stroke continues (which is what lets a kept head survive
/// this shrinking under it).
pub fn taper_safe_frozen(rec: &StrokeRecord, frozen: usize) -> usize {
    let (start_px, end_px) = rec.brush.taper_px();
    let last = crate::path::span_count(rec.path.len());
    if (start_px <= 0.0 && end_px <= 0.0) || last == 0 {
        return frozen;
    }
    let head = rec.path[0].pos;
    let tip = crate::path::span_end(&rec.path, last - 1);
    let mut spans = frozen.min(last);
    while spans > 0 {
        let cut = crate::path::span_end(&rec.path, spans - 1);
        if (tip - cut).length() >= end_px && (cut - head).length() >= start_px {
            break;
        }
        spans -= 1;
    }
    spans
}

/// Which part of a stroke to build segments for, and the arc length its first
/// sample carries.
///
/// `dist` is not derivable from `range` — it is the arc length accumulated along
/// everything *before* it — so an incremental caller has to carry it forward. It
/// matters because the `drain` falloff and the colour-dynamics noise are both
/// parameterized by distance travelled: restarting it at zero would make the tail
/// of a stroke look like the head of one.
#[derive(Clone, Debug)]
pub struct StrokeSpans {
    pub range: std::ops::Range<usize>,
    pub dist: f32,
}

impl StrokeSpans {
    /// The whole stroke, from the beginning.
    pub fn whole(rec: &StrokeRecord) -> Self {
        StrokeSpans {
            range: 0..crate::path::span_count(rec.path.len()),
            dist: 0.0,
        }
    }
}
