//! What an engine is made of (§6.7, §11): the GPU half that is expensive to build
//! and shared between siblings, the three ways an engine comes to hold one — built
//! fresh, built over a donor's, rebuilt for a new color space — and the one
//! constructor every path ends in.

use std::sync::Arc;

use super::{
    Authoring, DEFAULT_FAST_COMMIT, DEFAULT_HISTORY_BUDGET, Engine, EngineShared, ROOT_LAYER,
};
use crate::Result;
use crate::assets::AssetStore;
use crate::colorspace::ColorSpace;
use crate::document::{ApplyCtx, DocState, LinearTimeline, Timeline};
use crate::gpu::channels::Zeroes;
use crate::gpu::scratch::ScratchPool;
use crate::gpu::{
    BlendPass, Compositor, CompositorPipeline, Environment, EnvironmentId, FillRenderer,
    FilterPass, GpuContext, MergeRenderer, Registry, SelectionRenderer, StrokeRenderer, Substrate,
    SubstrateMap, TilePool, TransformRenderer,
};
use crate::peer::Peers;
use crate::projection::{Memo, Revision};
use crate::view::{Extent2, ViewTransform};
use stark_model::{ColorSpaceId, DocError, SubstrateId};

impl EngineShared {
    /// The device these engines draw with.
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// The texture format every pipeline here was compiled against. A substrate a
    /// sibling presents to has to be configured for it.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// The color space these pipelines were built for (§6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.color_space.id()
    }
}

impl Engine {
    /// Build an engine that presents to `target_format` (a substrate format, or a
    /// test target), in the default Oklab color space. Takes wgpu handles from
    /// the frontend (CLAUDE.md).
    pub fn new(gpu: GpuContext, target_format: wgpu::TextureFormat, viewport: Extent2) -> Self {
        // Oklab is in every build by construction — it is the space with no optional
        // dependency behind it — so the only fallible case cannot arise here.
        Self::new_with_color_space(gpu, target_format, viewport, ColorSpaceId::Oklab)
            .expect("Oklab is unconditional")
    }

    /// Build an engine in a chosen color space (§6.7).
    ///
    /// Fails with [`DocError::UnsupportedColorSpace`] if this build does not carry
    /// the space — today only `Mixbox` without the `mixbox` feature. A frontend that
    /// builds its picker from
    /// [`colorspace::all_available`](crate::colorspace::all_available)
    /// never sees it.
    pub fn new_with_color_space(
        gpu: GpuContext,
        target_format: wgpu::TextureFormat,
        viewport: Extent2,
        color_space: ColorSpaceId,
    ) -> Result<Self> {
        let color_space = crate::colorspace::make(color_space)
            .ok_or(DocError::UnsupportedColorSpace(color_space))?;
        // The registry starts on the builtin flat substrate — it is all that can be
        // built before any bytes exist, and it is also what a fresh document is on
        // (`DEFAULT_SUBSTRATE`), so there is nothing to reconcile between the two. A
        // substrate is named by the hash of its height map (§6.4), so an engine with no
        // bytes has exactly one substrate it can truthfully name, and a frontend that
        // wants another opens a document on it.
        let substrates = Registry::<Substrate>::new(&gpu, Substrate::default());
        // Lighting starts on the procedural neutral environment; image HDRs are
        // registered later by the frontend (§6.3).
        let environments = Registry::<EnvironmentId>::new(&gpu, EnvironmentId::default());
        let substrate = substrates.current();
        // Read out before the registry moves into the keep — the live object, not the
        // registry, is what the media pass binds.
        let environment = environments.current();
        let scratch = ScratchPool::default();
        let built = build_gpu(GpuBuild {
            keep: GpuKeep {
                assets: AssetStore::new(gpu.clone()),
                selection: SelectionRenderer::new(&gpu, scratch.clone()),
                scratch,
                gpu: gpu.clone(),
                substrates,
                environments,
            },
            target_format,
            cs: &color_space,
            substrate: &substrate,
            environment: &environment,
        });

        let initial = DocState::with_layer(ROOT_LAYER);
        let initial_substrate = initial.substrate;
        let timeline = Timeline::Linear(LinearTimeline::new(initial));

        Ok(Self::assemble(
            built.shared,
            built.compositor,
            built.compositor_pipeline,
            initial_substrate,
            timeline,
            viewport,
        ))
    }

    /// A second engine on `donor`'s device, **sharing** everything expensive and
    /// immutable — the compiled pipelines (stroke, compositing, selection,
    /// transform, fill, merge, the blend pass and its pigment LUT), the tile
    /// allocator, the content-addressed brush assets, and the decoded substrate and
    /// environment caches — around a fresh document of its own.
    ///
    /// This is what a *preview* engine is (§11): the brush editor's test canvas and
    /// a preset thumbnail both paint strokes that must render exactly as the main
    /// canvas would, which is an argument for sharing the machinery, not just an
    /// economy. Sharing keeps the cost to a document, a compositor's attachments and a
    /// fistful of `Arc` bumps, where building one standalone means recompiling ~19
    /// shaders and ~30 pipelines and re-decoding every image the app has already
    /// decoded once.
    ///
    /// What is shared is exactly what cannot disagree: the shared pieces are either
    /// immutable (pipelines), content-addressed (assets, the substrate/environment
    /// byte-and-build caches), or an allocator (the tile pool). Everything an engine
    /// can *set* stays per-engine — the document, the session view, and the three
    /// compositor view settings, which start mirroring the donor's current look
    /// (substrate, lighting, media parameters) and move independently from there.
    ///
    /// The document opens on the donor's current substrate, so a preview needs no
    /// `SetSubstrate` step — and no substrate bytes handed across, which is the point.
    ///
    /// Divergence after construction is safe but not tracked: a
    /// [`new_document`](Self::new_document) that changes *this* engine's color
    /// space rebuilds it an unshared set (`rebuild_gpu_for`), and the donor doing
    /// the same simply stops feeding the shared caches this engine keeps using.
    pub fn new_sharing(donor: &Engine, viewport: Extent2) -> Self {
        Self::on_shared(donor.shared(), viewport)
    }

    /// Build an engine on an already-built [`EngineShared`] — the general form of
    /// [`new_sharing`](Self::new_sharing), for a caller that holds the shared half
    /// without holding an engine.
    ///
    /// That is the difference worth having: a preset thumbnail wants the device and
    /// the pipelines, and requiring a *donor engine* would mean borrowing whichever
    /// live one happens to exist — with its substrate, its document and its in-flight
    /// gesture — for the length of the call.
    ///
    /// The document opens on `shared`'s current substrate, so a preview needs no
    /// `SetSubstrate` step — and no substrate bytes handed across, which is the point.
    pub fn on_shared(shared: EngineShared, viewport: Extent2) -> Self {
        let substrate = shared.apply.substrates.id();
        let initial_substrate = substrate.id;
        let timeline = Timeline::Linear(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER)
                .with_substrate(initial_substrate)
                .with_substrate_scale(substrate.scale),
        ));
        // Its own three view settings over the shared passes — the whole of what a
        // sibling's compositor costs ([`CompositorPipeline::sharing`]), seeded from
        // `shared` so it opens mirroring the canvas it came from.
        let compositor_pipeline = CompositorPipeline::sharing(
            shared.passes.clone(),
            shared.apply.substrates.current(),
            shared.environment.current(),
            shared.media,
            shared.output,
        );
        let compositor = Compositor::new(&compositor_pipeline);
        Self::assemble(
            shared,
            compositor,
            compositor_pipeline,
            initial_substrate,
            timeline,
            viewport,
        )
    }

    /// The fields every engine opens with, wrapped around the handful that differ —
    /// and the [`apply_document_substrate`](Self::apply_document_substrate) both
    /// constructors owe once they are set.
    ///
    /// **[`EngineShared`]'s argument, one level up.** Two struct literals naming
    /// fourteen identical fields are the same shape waiting to go wrong: a field given
    /// a value in one and forgotten in the other is invisible on the main canvas and
    /// shows up only on a preview or a thumbnail, the hardest surface in the app to
    /// notice on. A field added to [`Engine`] has one place to be given a value, and
    /// the compiler asks for it there.
    ///
    /// Every parameter is a distinct type, so a transposed argument list is a compile
    /// error rather than a silently wrong engine — which is what makes six positional
    /// values safe here, where a parameter struct would only move the literal.
    fn assemble(
        shared: EngineShared,
        compositor: Compositor,
        compositor_pipeline: CompositorPipeline,
        initial_substrate: SubstrateId,
        timeline: Timeline,
        viewport: Extent2,
    ) -> Self {
        let mut engine = Self {
            shared,
            compositor,
            compositor_pipeline,
            initial_substrate,
            timeline,
            // Built here rather than handed in, because both constructors built the
            // same one: an engine opens on its viewport, aimed at the root layer its
            // timeline was seeded with.
            session: crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER),
            peers: Peers::new(),
            now: 0.0,
            preview: Default::default(),
            doc_revision: 0,
            doc_origin: 0,
            draw_cache: Memo::default(),
            layer_cache: Memo::default(),
            guide_cache: Memo::default(),
            guide_epoch: Revision::default(),
            history_budget: DEFAULT_HISTORY_BUDGET,
            fast_commit: DEFAULT_FAST_COMMIT,
            strokes_reused: 0,
            #[cfg(feature = "debug-unfrozen")]
            debug_samples: Vec::new(),
            authoring: Authoring::solo(),
        };
        // Park the substrate registry on the document's substrate. A no-op for a fresh
        // document (both are `Flat`) and for a sibling, whose two halves were just
        // seeded from the same place — and not for one `new_document` seeded, where it
        // is what makes the substrate actually render. Here rather than at the two
        // call sites for the reason the fields above are: an invariant every engine
        // holds belongs where every engine is built.
        engine.apply_document_substrate();
        engine
    }

    /// The expensive half of this engine, for building another on the same device
    /// (§11) — see [`EngineShared`].
    ///
    /// The three view settings ride along as the look a sibling **opens** on, read
    /// live rather than from when this engine was built, so a preview of the canvas
    /// mirrors the canvas as it stands.
    pub fn shared(&self) -> EngineShared {
        debug_assert!(
            Arc::ptr_eq(&self.shared.passes, &self.compositor_pipeline.passes()),
            "the shared passes and this engine's pipeline have come apart",
        );
        EngineShared {
            media: self.compositor_pipeline.media(),
            output: self.compositor_pipeline.output(),
            ..self.shared.clone()
        }
    }

    /// Rebuild the GPU subsystems (pool/stroke/compositor) for `id`. Assumes the
    /// document is already empty (no tiles of the old format are referenced).
    /// Takes the *resolved* space rather than an id, which is what keeps this
    /// infallible: every caller has already had to obtain one, so there is no
    /// "unsupported space" case left to handle here or to forget.
    pub(super) fn rebuild_gpu_for(&mut self, cs: Arc<dyn ColorSpace>) {
        // Cloned out before the rebuild: the registry lives on `self.shared.apply`, which is
        // replaced below, and a `SubstrateMap` is two reference-counted wgpu handles.
        let substrate = self.shared.apply.substrates.current();
        let environment = self.shared.environment.current();
        let built = build_gpu(GpuBuild {
            // What a rebuild does not touch, moved through into the new context —
            // stated as a list rather than as four arguments, because "what survives
            // a color-space change" is the interesting half of this function.
            keep: GpuKeep {
                gpu: self.shared.gpu.clone(),
                assets: self.shared.apply.assets.clone(),
                selection: self.shared.apply.selection.clone(),
                scratch: self.shared.scratch.clone(),
                substrates: self.shared.apply.substrates.clone(),
                environments: self.shared.environment.clone(),
            },
            target_format: self.shared.target_format,
            cs: &cs,
            substrate: &substrate,
            environment: &environment,
        });
        // Whole, not field by field: anything added to the shared half is rebuilt
        // here by construction rather than by somebody remembering this line — which
        // now includes the compiled `passes`, and that one matters. Assigned
        // piecemeal, a rebuild left `shared.passes` naming the pipelines it had just
        // replaced, so the next sibling built off `shared()` would have taken the old
        // ones. `Engine::shared`'s `debug_assert` is the guard on exactly that.
        self.shared = built.shared;
        self.compositor = built.compositor;
        self.compositor_pipeline = built.compositor_pipeline;
    }
}

/// What the color-space-dependent GPU subsystems are built from.
///
/// Grouped because they are always supplied together: the pool, stroke renderer and
/// compositor are torn down and rebuilt as a set whenever the color space changes
/// (§6.7).
struct GpuBuild<'a> {
    /// What the rebuild does **not** touch, moved through into the context it comes
    /// back in.
    keep: GpuKeep,
    target_format: wgpu::TextureFormat,
    // No viewport: nothing built here is sized by one. A `Compositor` given one at
    // construction would overwrite it on its first render anyway, since that is the
    // only moment the zoom — and so the supersampled size — is known.
    cs: &'a Arc<dyn ColorSpace>,
    substrate: &'a SubstrateMap,
    environment: &'a Environment,
}

/// The pieces of [`ApplyCtx`] a color-space rebuild **survives** — the device, and
/// the three stores whose contents are either content-addressed or independent of
/// how color is represented (§6.7).
///
/// A struct rather than four parameters so that "what survives a rebuild" is stated
/// once and read as a list. The two callers differ only in where they get it: a
/// fresh engine builds these, a rebuild clones them off the context it is replacing.
struct GpuKeep {
    gpu: GpuContext,
    /// Brush shapes, named by the hash of their bytes — so nothing about them
    /// changes when the color space does (§6.6).
    assets: AssetStore,
    /// A mask is one coverage channel whatever the paint is, so the rasterizer is
    /// color-space independent and is handed back in rather than rebuilt (§6.8).
    selection: SelectionRenderer,
    /// The working textures and buffers every recording leases and gives back
    /// (`gpu::scratch`) — **one pool for the whole stack**, so a stroke's ring, a
    /// transform's parcel and a merge's expansions feed one another's free lists.
    ///
    /// Kept across a color-space rebuild, unlike the renderers it serves: what a
    /// checkout asks for is a size, a format and a usage, so a pool holds no opinion
    /// about the space and would only have to warm up again (§6.7). Nothing in it is
    /// live at the moment a rebuild happens — a rebuild needs an empty document, and
    /// a lease outlives no submit.
    scratch: ScratchPool,
    /// The canvas substrates and their registered bytes: a height map, likewise
    /// nothing to do with how color is represented (§6.4). Keyed by the substrate *and
    /// the scale it is laid at*, since that is what a substrate is baked from
    /// (`gpu::substrate::Substrate`).
    substrates: Registry<Substrate>,
    /// The lighting environments and their registered bytes — a *view* setting, and
    /// color-space independent, so a rebuild carries it rather than re-decoding the
    /// HDR and its whole mip chain (§6.3).
    environments: Registry<EnvironmentId>,
}

/// Everything a build hands back, in the shape the engine stores it.
///
/// The whole [`EngineShared`] rather than its parts loose, which is the point: a
/// rebuild is then `self.shared = built.shared` and anything added to the shared half
/// is rebuilt by construction. Assigned field by field, each renderer has to be
/// remembered in three places — the tuple, the constructor and the rebuild — and the
/// rebuild is the one whose omission shows up only in a document that changed color
/// space.
///
/// The two compositor values come back beside it rather than inside it because they
/// are **per-engine**: the attachments are this target's, and the pipeline carries
/// this engine's three view settings. What `shared` keeps of them is the compiled
/// `passes`, read off the pipeline built here so the two cannot come apart.
struct GpuBuilt {
    shared: EngineShared,
    compositor_pipeline: CompositorPipeline,
    compositor: Compositor,
}

fn build_gpu(b: GpuBuild<'_>) -> GpuBuilt {
    let GpuBuild {
        keep:
            GpuKeep {
                gpu,
                assets,
                selection,
                scratch,
                substrates,
                environments,
            },
        target_format,
        cs,
        substrate,
        environment,
    } = b;
    // The color space's formats — the only ones this call site knows. The pool
    // unions in its own (the selection mask, the wide scratch aux), so none can be
    // forgotten here (`TilePool::new`). The residual's is `Rgba16Float`, which every
    // space's color already is, but it is passed rather than assumed for the same
    // reason the aux is: the first space to choose otherwise would meet
    // `acquire_tex`'s "unsupported format" panic on its first stroke.
    let pool = TilePool::new(
        gpu.clone(),
        [cs.color_format(), cs.aux_format()]
            .into_iter()
            .chain(cs.resid_format()),
    );
    let zeroes = Zeroes::new(&gpu, crate::gpu::channels::ChannelFormats::of(cs.as_ref()));
    // Built here rather than inside either consumer, because both bind the group a
    // *tile* caches over its own channels and a cached group answers to one layout:
    // pass A composites the document, and the stamp loop composites the very same
    // tiles into its working region (§6.2). Same bargain as `blend` and `filter`
    // below — built once at the top, handed to everyone who needs it.
    let tile_bgl = crate::gpu::composite::tile_bind_group_layout(&gpu.device, cs.as_ref());
    let stroke = StrokeRenderer::new(
        &gpu,
        cs.clone(),
        selection.clone(),
        zeroes.clone(),
        tile_bgl.clone(),
        scratch.clone(),
    );
    // Built once and shared: `gpu::merge` runs this very pipeline on tile-sized
    // targets to merge a layer down through its mode (§14.11), and building a second
    // one would decode the Mixbox LUT twice.
    let blend = Arc::new(BlendPass::new(&gpu, cs.as_ref()));
    // The same bargain for the filter pass, which `gpu::merge` runs on tile-sized
    // targets to merge a filter layer into the paint beneath it (§14.11.7).
    let filter = Arc::new(FilterPass::new(&gpu, cs.as_ref()));
    let compositor_pipeline = CompositorPipeline::new(
        &gpu,
        target_format,
        cs.as_ref(),
        substrate.clone(),
        environment.clone(),
        crate::gpu::composite::SharedPasses {
            blend: blend.clone(),
            filter: filter.clone(),
            tile_bgl,
        },
    );
    let compositor = Compositor::new(&compositor_pipeline);
    let transform = TransformRenderer::new(
        &gpu,
        cs.as_ref(),
        selection.clone(),
        zeroes.clone(),
        scratch.clone(),
    );
    let fill = FillRenderer::new(
        &gpu,
        cs.clone(),
        selection.clone(),
        zeroes.clone(),
        scratch.clone(),
    );
    let merge = MergeRenderer::new(&gpu, cs.as_ref(), zeroes, blend, filter, scratch.clone());
    // No pipeline and no layout: a placed image's tiles are computed on the CPU
    // (§23), so this is the color space and the queue and nothing else.
    let place = crate::gpu::PlaceRenderer::new(&gpu, cs.clone());
    // Rebuilt with the rest rather than carried across, like the two stores beside it:
    // a color-space change is a new document (§6.7), so nothing it held is still named.
    let pictures = crate::pictures::PictureStore::new();
    // `passes` and `media` are read off the pipeline that was just built, never
    // assembled beside it: they are the two things `EngineShared` keeps *of* the
    // compositor, and a second source for either is how a rebuild leaves a sibling
    // holding pipelines that no longer exist.
    let shared = EngineShared {
        passes: compositor_pipeline.passes(),
        media: compositor_pipeline.media(),
        output: compositor_pipeline.output(),
        gpu: gpu.clone(),
        target_format,
        color_space: cs.clone(),
        environment: environments,
        scratch,
        apply: ApplyCtx {
            pool,
            stroke,
            assets,
            selection,
            transform,
            fill,
            merge,
            place,
            pictures,
            gpu,
            substrates,
            prepared: None,
        },
    };
    GpuBuilt {
        shared,
        compositor_pipeline,
        compositor,
    }
}

/// Convenience for tests/tools: build an engine on a headless device.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    headless_engine_with(target_format, viewport, ColorSpaceId::Oklab).await
}

/// Headless engine in a chosen color space (§6.7).
pub async fn headless_engine_with(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    color_space: ColorSpaceId,
) -> Result<Engine> {
    let gpu = GpuContext::headless().await?;
    Engine::new_with_color_space(gpu, target_format, viewport, color_space)
}
