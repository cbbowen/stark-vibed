//! The brush engine: **swept-segment** stroke rasterization with copy-on-write
//! tiles (§6.2, §5.2, §6.6, §6.7).
//!
//! Rather than stamping discrete dabs, each short segment of the flattened curve
//! is drawn as one oriented quad whose coverage is the brush *swept* along it —
//! the path integral of the footprint. Because alpha-"over" is additive in
//! optical depth `τ = −ln(1−α)`, the swept depth of a segment is a difference of
//! the brush's precomputed prefix-τ texture (`prefix(u) − prefix(u−d)`), and the
//! premultiplied-over blend across overlapping segment quads sums those depths
//! *exactly* — reconstructing the continuous stroke with no banding and no
//! double-counting at the joints.
//!
//! That is the plain **add** fast path: footprint → cleared scratch tile →
//! integrate over the base into a fresh CoW tile. A brush that also moves paint
//! already on the canvas (`lift` / `deposit` / `charge` / `bleed`, §6.2) instead runs
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

mod budget;
mod dynamics;
mod incremental;
mod segments;
mod swept;

use budget::MAX_REGION_DIM;
use dynamics::{
    DynamicsKit, StrokePath, build_dynamics_kit, build_integrate_pipeline, dynamics_setup,
};
use segments::{SegmentInstance, round_coverage};

// The module's surface, re-exported so callers name `gpu::stroke::X` rather than the
// file X happens to live in — the split below is about where a maintainer reads, not
// about what the engine depends on.
pub(crate) use budget::flatten_tolerance;
pub use incremental::{StrokeCarry, StrokeSpans, ToolState};
// Not part of the module's public surface: the engine calls it, nothing outside the
// crate does, and keeping it crate-visible is what lets its doc comment point at the
// `segments` internals the rule is actually about.
pub(crate) use incremental::safe_frozen;

/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

/// Stride between the slots of a uniform buffer read through **dynamic offsets**,
/// which is how both render paths vary a uniform across the draws or dispatches of
/// one pass. A dynamic offset must be a multiple of the device's
/// `min_uniform_buffer_offset_alignment`, whose spec maximum is 256, so this clears it
/// on every adapter — at the cost of the padding past each slot's real size.
///
/// **One buffer per stroke or per piece, not one per tile.** Every uniform here is
/// tens of bytes and a live stroke re-renders on every pointer move, so a buffer and
/// a bind group per affected tile is a rate of small WebGPU allocations rather than an
/// amount of memory — and the rate is the thing JS GC cannot keep up with
/// ([`ScopedResources`]). Laid out this way, a stroke's per-tile uniforms cost one
/// registered buffer and one bind group however many tiles it crosses.
const UNIFORM_STRIDE: usize = 256;

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    pipeline: wgpu::RenderPipeline,
    uniform_bgl: wgpu::BindGroupLayout,
    prefix_bgl: wgpu::BindGroupLayout,
    /// The round tip's baked textures, keyed by `hardness.to_bits()` (§6.6).
    ///
    /// **One entry, replaced rather than accumulated** — and that is a fact about the
    /// key, not caution. Hardness is a continuous slider, so a live preview walks it
    /// through a fresh value per frame while the user drags one: keeping every
    /// position would bank ~320 KB of GPU texture apiece and never hand it back, while
    /// keeping the last is exactly the working set of *adjust the knob and look*.
    /// [`noise_cache`](Self::noise_cache) below grows without bound for the opposite
    /// reason — its key is a small enum, so the whole domain fits and nothing evicts.
    round_tip: Arc<Mutex<Option<(u32, RoundTip)>>>,
    /// Colour dynamics (§6.2): the sweep's noise bind group layout
    /// (group 2), the shared wrap/linear sampler, the 1×1×1 zero volume bound
    /// when a brush's jitter is off, and the lazily-baked per-kind fields.
    noise_bgl: wgpu::BindGroupLayout,
    noise_sampler: wgpu::Sampler,
    dummy_noise: wgpu::TextureView,
    noise_cache: Arc<Mutex<Vec<(NoiseKind, wgpu::TextureView)>>>,

    // Stroke integrate (§6.2/§6.1): a fullscreen pass reads the base tile +
    // the stroke's footprint scratch and writes `new = f(base, scratch)` into a fresh
    // CoW tile's color+aux MRT — the scratch's accumulated parcel stacked on the base
    // through the shared law in `paint_common.wesl`, the same one a fill lands through
    // and the stamp loop's `deposit` uses.
    integrate_pipeline: wgpu::RenderPipeline,
    integrate_bgl: wgpu::BindGroupLayout,

    // Brush dynamics: the sequential stamp loop (§6.2), used when the
    // brush manipulates existing paint (`lift` / `deposit` / `charge` / `bleed` —
    // the four axes `dynamics_setup` gates on).
    dynamics: DynamicsKit,

    /// Selection masks (§6.8): the per-tile mask bound into the integrate
    /// pass, and the region gather the stamp loop reads. Colour-space independent, so
    /// it is handed in rather than rebuilt with the rest of this renderer.
    selection: SelectionRenderer,
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
    /// The canvas surface the document was on when this stroke was made (§6.4) —
    /// the ground whose tooth gates how much of the brush's own paint lands
    /// (`BrushParams::tooth`).
    ///
    /// Handed in per call, like everything else here, rather than held on the
    /// renderer: it is *document* state, and a renderer that cached it would answer
    /// a replayed stroke with whatever the compositor happens to be showing. That is
    /// the shape the deleted `StrokeRenderer::set_surface` had (§6.4).
    pub surface: &'a crate::gpu::surface::Surface,
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

        // One slot per affected tile, selected by a dynamic offset ([`UNIFORM_STRIDE`])
        // — so a stroke crossing many tiles binds one buffer rather than building one
        // per tile on every pointer move.
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep uniform bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(swept::XFORM_SLOT),
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
                // The canvas surface's height map + its own repeat sampler — the
                // deposition tooth (§6.4). In this group rather than one of its own
                // because it is the same kind of thing as the noise beside it: a
                // tileable field the deposit samples per fragment, resolved per
                // stroke.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x4,
                        4 => Float32
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
            round_tip: Arc::new(Mutex::new(None)),
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
                .unwrap_or_else(|| self.round_tip(BrushShape::DEFAULT_HARDNESS).prefix),
            BrushShape::Round { hardness } => self.round_tip(hardness).prefix,
        }
    }

    /// The brush's plain coverage mask — the weights a reservoir texel carries
    /// (§6.2). Resolved exactly as [`Self::prefix_view`] is, from the same two
    /// sources; only the stamp loop asks for it.
    fn coverage_view(&self, assets: &AssetStore, brush: &BrushParams) -> wgpu::TextureView {
        match brush.shape {
            BrushShape::Stamp(id) => assets
                .coverage_view(id)
                .unwrap_or_else(|| self.round_tip(BrushShape::DEFAULT_HARDNESS).coverage),
            BrushShape::Round { hardness } => self.round_tip(hardness).coverage,
        }
    }

    /// The round tip's baked textures for a given `hardness`, cached so live preview
    /// — which re-renders per pointer move — doesn't rebuild them each frame.
    ///
    /// The pair is built and cached **together**, off a single [`round_coverage`]
    /// evaluation, because they are two readings of one field: 256² texels of `powf`
    /// that used to be run twice for the same hardness, once per texture. Cached as
    /// one entry for a second reason — held apart, the stamp loop could find its
    /// prefix hot and its coverage cold, and pay the field again anyway.
    fn round_tip(&self, hardness: f32) -> RoundTip {
        let key = hardness.to_bits();
        let mut cache = self.round_tip.lock().expect("round tip cache poisoned");
        if let Some((k, tip)) = cache.as_ref()
            && *k == key
        {
            return tip.clone();
        }
        let cov = round_coverage(hardness, ROUND_RES);
        // The round tip is rotation-invariant, so a single orientation layer suffices —
        // the shader's wrapping lookup reads it for every orientation (§6.6).
        let (_tex, prefix) = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, 1, &cov);
        let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
        let (_tex, coverage) = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
        let tip = RoundTip { prefix, coverage };
        *cache = Some((key, tip.clone()));
        tip
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
}

/// The two textures a round tip bakes to (§6.6): the swept-footprint prefix-τ both
/// render paths integrate against, and the plain coverage mask the stamp loop's
/// reservoir texels weight by.
///
/// One type because they are one thing — the same coverage field, read two ways —
/// and keeping them so is what makes a cache entry able to say it holds *the tip*
/// rather than a texture that happens to be a tip's.
#[derive(Clone)]
struct RoundTip {
    prefix: wgpu::TextureView,
    coverage: wgpu::TextureView,
}
