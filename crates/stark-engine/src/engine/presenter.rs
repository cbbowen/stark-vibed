//! The compositing half of an engine (§6.3): the attachments a frame is built
//! through, the pipelines and view settings every [`Compositor`] shares, and the memo
//! the draw list is held in.
//!
//! **Why they are one value.** They are used together and by nothing else, and holding
//! them apart from the rest of [`Engine`](super::Engine) is what lets a frame borrow
//! the document and the compositor at the same time. The clone of the whole `DocState`
//! a frame used to pay for was the price of a read that borrowed all of `self`
//! (`Engine::presented`); the fields being disjoint, the borrow checker now allows what
//! was always true.

#[cfg(not(target_arch = "wasm32"))]
use super::render::ProbeTarget;
use super::render::Rendered;
use crate::colorspace::ColorSpace;
use crate::document::DocState;
use crate::gpu::composite::{
    CompositeGroup, CompositeScene, Compositor, CompositorPasses, CompositorPipeline, MediaParams,
    Offscreen, Output, Patch, PreparedPick, ViewSettings,
};
use crate::gpu::environment::Environment;
use crate::gpu::substrate::SubstrateMap;
use crate::projection::Memo;
use crate::view::ViewTransform;
use std::sync::Arc;

use super::draw_list::DrawList;
use stark_model::document::LayerId;
use stark_model::geom::TileRect;

/// Which [`Compositor`] a render's offscreen attachments come from: the screen's own,
/// or the caller's — one per target, since sharing one would resize the other's away.
pub(super) enum Attachments<'a> {
    /// The screen's own, cached across frames ([`Presenter::compositor`]).
    Screen,
    /// The caller's, for a second surface on the same screen — the navigator's
    /// miniature — presented as the screen is (§6.5). Whether they outlive the call is
    /// the caller's choice, since only it knows whether the render repeats — see
    /// [`Offscreen`].
    Surface(&'a mut Offscreen),
    /// The caller's, for a picture bound for a file or the CPU: an export, a thumbnail,
    /// a golden. Rendered [`Output::SDR`] whatever the screen is showing, which keeps
    /// the screen's headroom and gamut out of a file (§6.5, §15.6).
    Export(&'a mut Offscreen),
}

/// See the module docs: the three fields a frame is composited through.
pub(super) struct Presenter {
    /// Compositing state for the **substrate**: the attachments a screen frame is built
    /// through, kept from frame to frame. Anything drawn beside the screen — an export,
    /// the navigator's miniature — brings a [`Compositor`] of its own for the call, so
    /// it never resizes these.
    compositor: Compositor,
    /// Beside the one above rather than inside it because a second [`Compositor`]
    /// borrows it: the view settings the media pass reads have one owner, so two
    /// consumers cannot disagree about the canvas substrate or the lighting.
    pipeline: CompositorPipeline,
    /// The largest of the engine's three memos, and the only one whose value is not a
    /// projection — see [`draw_list`](Self::draw_list).
    draw_cache: Memo<DrawKey, Arc<[CompositeGroup]>>,
    /// What the suite's two readbacks render through, kept between them — see
    /// [`ProbeTarget`] for why the parking is sound here and nowhere else.
    #[cfg(not(target_arch = "wasm32"))]
    probe: ProbeTarget,
}

impl Presenter {
    /// The screen's attachments over the pipeline they answer to, with nothing drawn
    /// yet. The pair comes from one build (`engine::build`), which is what keeps the
    /// attachments and the passes they were made for together.
    pub(super) fn new(compositor: Compositor, pipeline: CompositorPipeline) -> Self {
        Self {
            compositor,
            pipeline,
            draw_cache: Memo::default(),
            #[cfg(not(target_arch = "wasm32"))]
            probe: ProbeTarget::default(),
        }
    }

    /// The probe's kept target, **taken out** for the render that uses it;
    /// [`keep_probe`](Self::keep_probe) puts it back (§9).
    ///
    /// Taken rather than borrowed because a frame borrows the whole engine, so the
    /// attachments it draws into cannot be reached through a field of it at the same
    /// time. Losing it to a panic mid-render costs the next call one allocation and
    /// nothing else.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn take_probe(&mut self) -> ProbeTarget {
        std::mem::take(&mut self.probe)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn keep_probe(&mut self, probe: ProbeTarget) {
        self.probe = probe;
    }

    /// [`DrawList`], **memoized**: the draw list for `key`, built afresh or taken from
    /// the cache.
    ///
    /// **Why the cache earns its place.** Building the list clones a `TilePairHandle`
    /// per visible tile, per layer, plus a `Vec` per layer, and the visible tile count
    /// scales as 1/zoom² — so a zoomed-out multi-layer document pays ~10⁵ atomic bumps
    /// a frame to produce a list identical to the last one.
    ///
    /// **A single-layer list is never cached.** The memo holds one list, so a thumbnail
    /// pass (one layer per row, interleaved with screen frames) would evict the screen's
    /// list N times over — and such a key can never be hit twice anyway, every row
    /// naming a different `only`.
    ///
    /// Takes the key and a borrowed `doc` so the caller can compute one while it still
    /// holds the other. The list comes back as an `Arc` rather than a reference into the
    /// memo, so holding it does not hold a borrow of the presenter.
    pub(super) fn draw_list(
        &self,
        key: DrawKey,
        doc: &DocState,
        cs: &dyn ColorSpace,
    ) -> Arc<[CompositeGroup]> {
        let build = || {
            let walk = DrawList::new(cs, key.visible);
            match key.only {
                Some(id) => walk.layer_alone(doc, id),
                None => walk.document(doc),
            }
            .into()
        };
        if key.only.is_some() {
            return build();
        }
        self.draw_cache.get_or_build(key, build)
    }

    /// The three compositing passes and the draws inside them, encoded and submitted
    /// (§6.3).
    ///
    /// CPU time to *record* them: WebGPU offers no timestamp query on the web, so
    /// nothing here can say what the GPU then spent executing them — the frontend's
    /// `Renderer::gpu_behind` is that signal.
    pub(super) fn present(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        scene: CompositeScene<'_>,
        attachments: Attachments<'_>,
    ) {
        crate::timing::span!("render.composite");
        match attachments {
            Attachments::Screen => self.compositor.render(&self.pipeline, target, view, scene),
            Attachments::Surface(into) | Attachments::Export(into) => into
                .get(&self.pipeline)
                .render(&self.pipeline, target, view, scene),
        }
    }

    /// Prepare one pick's patches — **once for a whole trace**, before any of them are
    /// recorded ([`Compositor::prepare_pick`]).
    pub(super) fn prepare_pick<'a>(
        &mut self,
        views: &[ViewTransform],
        groups: &'a [CompositeGroup],
    ) -> PreparedPick<'a> {
        self.compositor.prepare_pick(&self.pipeline, views, groups)
    }

    /// Record patch `slot` of a prepared pick into the caller's encoder
    /// ([`Compositor::composite_patch`]). Nothing here submits.
    pub(super) fn composite_patch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedPick<'_>,
        slot: usize,
    ) -> Patch {
        self.compositor
            .composite_patch(&self.pipeline, encoder, prepared, slot)
    }

    /// The compiled passes, for the assertion that a sibling's kit is this one's
    /// (`Engine::shared`).
    pub(super) fn passes(&self) -> Arc<CompositorPasses> {
        self.pipeline.passes()
    }

    /// The lighting and the display as they stand — **the live value**, which is the
    /// only thing a sibling may be seeded from (`EngineShared::sibling`).
    pub(super) fn settings(&self) -> ViewSettings {
        self.pipeline.settings()
    }

    /// The largest 2-D texture edge this device will make, asked once by the pipeline:
    /// what caps an export's size and the supersampling (§6.4).
    ///
    /// The device's own limit rather than a literal, because there is no one number —
    /// the frontend requests `wgpu::Limits::default()` while the headless device asks
    /// only for the engine's own minimums
    /// ([`MAX_TEXTURE_DIM_2D`](crate::gpu::context::MAX_TEXTURE_DIM_2D)), so any literal
    /// agrees with one of them and not the other.
    pub(super) fn max_dim(&self) -> u32 {
        self.pipeline.max_dim()
    }

    /// The media parameters and the display the screen is presented on (§6.3, §6.5) —
    /// what [`ObservableState`](super::ObservableState) projects.
    pub(super) fn media(&self) -> MediaParams {
        self.pipeline.media()
    }

    pub(super) fn output(&self) -> Output {
        self.pipeline.output()
    }

    pub(super) fn set_media(&mut self, media: MediaParams) {
        self.pipeline.set_media(media);
    }

    pub(super) fn set_output(&mut self, output: Output) {
        self.pipeline.set_output(output);
    }

    /// Rebind the canvas substrate the media pass samples for relief (§6.4). One bind
    /// group, not a rebuild: the attachments survive it (`CompositorPipeline`).
    pub(super) fn set_substrate(&mut self, substrate: SubstrateMap) {
        self.pipeline.set_substrate(substrate);
    }

    /// Rebind the lighting environment (§6.3), on [`set_substrate`](Self::set_substrate)'s
    /// terms.
    pub(super) fn set_environment(&mut self, environment: Environment) {
        self.pipeline.set_environment(environment);
    }

    /// Drop the cached list, which was built over a document that is being closed. The
    /// next frame builds its own.
    pub(super) fn forget_draws(&mut self) {
        self.draw_cache = Memo::default();
    }

    /// Hand back the screen's blend scratch and instance streams, grown to the document
    /// being closed and never shrunk — the one part of a reset the pool cannot reach
    /// (`Engine::release_closed_document`).
    pub(super) fn release_attachments(&mut self) {
        self.compositor = Compositor::new(&self.pipeline);
    }
}

/// What a built draw list is a function of — the whole of it, which is what makes
/// caching one sound (§6.3, C4). See [`Memo`] for the rule this is one of three keys
/// under.
///
/// Every term is already counted for another reason, and that is the point: nothing
/// here is a new notion of "has it changed", only the existing ones read together.
///
/// - `doc_revision` moves on every **committed** change — a commit, an undo, a
///   merged remote action, a load.
/// - `epoch` moves whenever the document the previews are drawn over is *replaced*:
///   an unlogged drag preview installed or dropped, and a commit. So it is *wider*
///   than `doc_revision` rather than a second, narrower term; both are named because
///   a key names the terms its value depends on, not the smallest set that happens to
///   cover them.
/// - `fold` moves whenever the live fold is built or dropped (`Preview::rebuild`); a
///   rebuild with nothing to drop leaves it alone. A stroke in flight commits nothing
///   and replaces no document, so neither counter above stirs while one is being
///   drawn, and a list keyed without this would hold the frame at the moment the pen
///   went down. It is the term the two roster keys do not need: they project what is
///   *shown*, and the fold is not that (`super::ShownKey`).
/// - `content` is which document is being drawn at all. `Live` and `Committed` differ
///   by exactly the in-flight gesture, so at one instant they are two different lists.
/// - `only` and `visible` are the two arguments that shape the list.
///
/// **What is deliberately absent is the view.** A draw list is built against the
/// *tile rect* a view can reach, not the view, so panning within one tile — or
/// rotating, or supersampling — hits the same key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DrawKey {
    pub(super) doc_revision: u64,
    pub(super) epoch: u64,
    pub(super) fold: u64,
    pub(super) content: Rendered,
    pub(super) only: Option<LayerId>,
    pub(super) visible: Option<TileRect>,
}
