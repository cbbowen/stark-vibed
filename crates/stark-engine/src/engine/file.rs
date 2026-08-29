//! The document's lifecycle: saving it, opening it, and the frontend-provided
//! resources it is replayed against (§8, §6.4, §6.6).
//!
//! The document *is* the action log, so a file is that log plus what it takes to
//! replay it — the brush shapes strokes reference and the canvas substrates they were
//! deposited against, both bundled, because a deposit is stored and no later arrival
//! un-bakes one laid against the wrong substrate. [`Engine::adopt`] is the order those
//! have to arrive in, written once for the three callers that need it.
//!
//! The registries live here for the same reason: a substrate is a replay input first
//! and a rendering input second, and its identity is derived from its bytes rather
//! than asserted alongside them, so a wrong binding cannot be expressed.

use stark_model::DocError;
use std::sync::Arc;

use super::{Authoring, Engine, GpuBuild, GpuKeep, ROOT_LAYER, build_gpu};
use crate::Result;
use crate::colorspace::ColorSpace;
use crate::document::{DocState, LinearTimeline, Timeline, effective_actions};
use crate::gpu::EnvironmentId;
use crate::gpu::substrate::Substrate;
use stark_model::AssetId;
use stark_model::AssetNeed;
use stark_model::ColorSpaceId;
use stark_model::DocumentFile;
use stark_model::SubstrateId;
use stark_model::document::Action;

/// A [`DocumentFile`] whose color space **this build can honour** (§6.7) — the one
/// question every adoption path has to settle before it touches the open document.
///
/// **The one place an untrusted color space enters.** Every `ColorSpaceId` decodes —
/// the enum is unconditional so the save format's indices cannot shift with a build's
/// features (§8, §19) — and what a build may lack is the *implementation*. So this is
/// not a decode failure, and saying so is what lets a frontend offer "this document
/// needs a Mixbox build" instead of "this file is corrupt". It is asked here rather
/// than in [`DocumentFile::from_bytes`] because since the crate split (§2) the decoder
/// cannot answer it: whether a space can be *honoured* is a fact about this build's
/// renderer, and `stark-model` has no `mixbox` feature to consult.
///
/// **A type rather than a call each caller remembers**, and the difference is not
/// theoretical. [`Engine::adopt`] begins by emptying the document and then resolves
/// the space, so the question must be settled *before* it is reached — and it made
/// that argument in a comment, naming the two callers that had settled it. There was
/// a third. `join_collaboration` handed it a file straight off the wire, so a build
/// without Mixbox joining a Mixbox session met the `expect` with the painting still
/// unsaved — on the web, the tab. A comment can be given a third caller; a
/// constructor cannot.
///
/// Borrowing rather than owning, because every caller already holds the file and the
/// validation reads one field: this is a *proof*, not a container.
#[derive(Clone, Copy)]
pub(super) struct ValidatedFile<'a>(&'a DocumentFile);

impl<'a> ValidatedFile<'a> {
    /// Settle the space, or refuse with [`DocError::UnsupportedColorSpace`] — leaving
    /// whatever is open alone, since nothing has been touched yet.
    pub(super) fn new(file: &'a DocumentFile) -> Result<Self> {
        if crate::colorspace::available(file.canvas.color_space) {
            Ok(Self(file))
        } else {
            Err(DocError::UnsupportedColorSpace(file.canvas.color_space).into())
        }
    }
}

impl std::ops::Deref for ValidatedFile<'_> {
    type Target = DocumentFile;

    fn deref(&self) -> &DocumentFile {
        self.0
    }
}

impl Engine {
    /// Snapshot the document as a saveable [`DocumentFile`] (§8), bundling the
    /// brush-shape assets that strokes actually reference (§6.6) and the canvas
    /// substrates the log names (§6.4).
    ///
    /// **What the log names is asked of [`DocumentFile::required_content`]**, which
    /// is the crate's one answer to that question (`content.rs`) and already what
    /// the loader, the joiner and the transport ask. This scanned the log itself
    /// until it didn't: two scans of one log, in two modules, is two things to teach
    /// about a new action that carries an id — and the one that is *not* taught
    /// writes a file that silently fails to bundle it.
    ///
    /// Every substrate the log names travels, not just the one it ends on: the tooth
    /// reads whichever was in force when a stroke was made, so a document that
    /// switched part-way through needs both to replay to the same pixels. `Flat` is
    /// skipped — procedural, no bytes — as is anything whose image never arrived,
    /// which cannot be bundled because it was never held.
    pub fn document_file(&self) -> DocumentFile {
        let mut file = DocumentFile::new(self.timeline.clone_actions());
        file.canvas.color_space = self.shared.color_space.id();
        // Before the scan below, which reads it: the substrate the log *starts* on is
        // named by the container rather than by any action, and is otherwise the one
        // piece of content nothing asks for.
        file.canvas.substrate = self.initial_substrate;
        // One bag, keyed by the need itself, so a store is asked only for the bytes
        // it can decode (§8). The match is over *which store*, not over which bag to
        // push to — the key carries that.
        for need in file.required_content() {
            let bytes = match need {
                AssetNeed::Brush(id) => self.asset_bytes(id),
                AssetNeed::Substrate(_) => need.substrate().and_then(|id| self.substrate_bytes(id)),
                AssetNeed::Picture(id) => self.shared.apply.pictures.bytes(id),
            };
            if let Some(bytes) = bytes {
                file.content.push((need, bytes));
            }
        }
        file
    }

    /// Serialize the document to the compact on-disk container, bundling
    /// everything it names (§8). A file that needs nothing but itself.
    pub fn save_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.document_file().to_bytes()?)
    }

    /// The same, leaving out content the opening app can produce itself — the ids
    /// of the assets it ships with (§8, §12.4).
    ///
    /// Worth it because the bundle dominates the file: a log is fitted paths and
    /// a canvas substrate is megabytes, so a doodle on the built-in rough substrate weighs 2.8
    /// MB of which almost none is the painting. The id stays in the file either
    /// way, so what is left out is looked up rather than guessed at, and bytes
    /// that do not hash to it are refused rather than substituted.
    ///
    /// What it costs is self-containment, which is why it is a separate call and
    /// not a flag on the other one: the result needs an app that still ships the
    /// content, and [`DocumentFile::unbundled_content`] is what the opener has to
    /// settle before replaying. Anything not in `resolvable` is bundled as usual,
    /// so passing an empty slice is [`Engine::save_bytes`].
    pub fn save_bytes_resolvable(&self, resolvable: &[AssetId]) -> Result<Vec<u8>> {
        let mut file = self.document_file();
        // Pictures are *not* dropped against this list, and cannot be: `resolvable`
        // is what the opening app ships with (§12.4), and a picture is by definition
        // something someone brought in — no build ships one. Filtering it anyway
        // would be a rule that can only ever fire on an id collision.
        file.content.retain(|(need, _)| match need {
            AssetNeed::Picture(_) => true,
            other => !resolvable.contains(&other.content()),
        });
        Ok(file.to_bytes()?)
    }

    /// Empty this engine's document and install everything `file` needs **before**
    /// its log can be replayed onto it — the shared preamble of opening a document,
    /// joining a shared session, and running a timelapse.
    ///
    /// The order is the whole content, and every step of it is a replay *input*
    /// rather than a preference:
    ///
    /// - the substrate the log starts from, recorded before `reset_document` seeds the
    ///   empty document with it; replayed `SetSubstrate` actions move it from there
    ///   (§6.4);
    /// - the document's color space, since the channel layouts differ between
    ///   spaces and a stroke replayed through the wrong shaders is a different
    ///   painting (§6.7);
    /// - the brush shapes strokes reference, and the substrates the log names, both
    ///   before any stroke that needs them. A deposit is *stored*: unlike the media
    ///   pass, which re-reads the substrate every frame and rights itself the moment an
    ///   image lands, no later arrival un-bakes a stroke laid against the flat
    ///   stand-in (§6.6, §6.4).
    ///
    /// The three callers had this written out three times and it had already drifted
    /// three ways — the timelapse was missing the initial substrate, so every frame
    /// before the log's first `SetSubstrate` deposited against the wrong substrate; it was
    /// missing the color space too, so a Mixbox document replayed through Oklab's
    /// shaders; and it swallowed a broken brush asset silently where the other two
    /// said so. A sequence whose *order* is the correctness argument is a sequence to
    /// write once.
    ///
    /// A substrate or a shape that fails to install is logged and skipped rather than
    /// fatal: the document still opens, degraded, which is the same bargain either
    /// asset gets.
    pub(super) fn adopt(&mut self, file: ValidatedFile<'_>) {
        self.initial_substrate = file.canvas.substrate;
        self.reset_document();
        if file.canvas.color_space != self.shared.color_space.id() {
            // Infallible because of the argument's *type*: a [`ValidatedFile`] is one
            // whose space this build resolves, and there is no other way to spell one.
            // This line used to be an `expect` under a comment naming the callers that
            // had settled it — see [`ValidatedFile`] for the third caller that had not.
            let cs = crate::colorspace::make(file.canvas.color_space)
                .expect("a ValidatedFile's color space resolves, by construction");
            self.rebuild_gpu_for(cs);
        }
        // Each entry to the store its need names, which is the whole reason the bundle
        // is keyed by one: a substrate handed to the brush store decodes as luminance
        // × alpha rather than channel 0, and nothing downstream can tell (§8).
        for (need, bytes) in &file.content {
            let handed = match need {
                AssetNeed::Brush(_) => self
                    .shared
                    .apply
                    .assets
                    .insert_bytes(bytes)
                    .map(|_| ())
                    .map_err(|e| format!("brush asset: {e}")),
                AssetNeed::Substrate(id) => self
                    .accept_substrate(SubstrateId::Image(*id), bytes)
                    .map(|_| ())
                    .map_err(|e| format!("canvas substrate: {e}")),
                AssetNeed::Picture(id) => self
                    .accept_picture(*id, bytes)
                    .map(|_| ())
                    .map_err(|e| format!("placed picture: {e}")),
            };
            if let Err(e) = handed {
                tracing::warn!("skipping content this document names — {e}");
            }
        }
        // Reachable only from a collaboration join now — [`Engine::load_document`] and
        // the timelapse refuse outright rather than adopt (`DocError::MissingContent`).
        // A joiner is the one caller that legitimately starts short: the actions arrive
        // over the same transport as the blobs, and the waitlist parks a `SetSubstrate`
        // until its substrate lands (§12.4), so this is a statement about ordering in
        // flight rather than about a document that cannot be reproduced.
        let missing = self.unresolved_content(&file);
        if !missing.is_empty() {
            tracing::warn!(
                ?missing,
                "joining a session whose content has not arrived yet; the waitlist                  holds anything that depends on it"
            );
        }
        // The empty document is on the log's initial substrate, so bind it — a timelapse
        // renders its first frame before any action has moved it.
        self.apply_document_substrate();
    }

    /// Replace the document by replaying a loaded file's action log. The full
    /// undo timeline is available afterwards — undo-after-load (§8).
    ///
    /// **Fails, and changes nothing, if the file's content is not all here**
    /// ([`DocError::MissingContent`]). The check is before [`Self::adopt`] rather
    /// than inside it so a refusal leaves the open document alone: half-replacing a
    /// painting is worse than declining to.
    pub fn load_document(&mut self, file: &DocumentFile) -> Result<()> {
        let file = ValidatedFile::new(file)?;
        self.require_content(&file)?;
        self.adopt(file);
        // Replay only the *effective* sequence: a file saved from a shared
        // session is the full log, including `Undo` actions and the actions
        // they suppress (§12.3). A solo load flattens those away.
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
        }
        self.resync_counters(&file.actions);
        // Whatever the replayed log left the document on.
        self.apply_document_substrate();
        // The loaded document *is* this file, unedited — so the baseline is where the
        // replay ended, not where `reset_document` set it before the replay began.
        self.doc_origin = self.doc_revision;
        Ok(())
    }

    /// Decode and load a container produced by [`Engine::save_bytes`].
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let file = DocumentFile::from_bytes(bytes)?;
        self.load_document(&file)
    }

    /// `Err(MissingContent)` if anything `file`'s log names is neither bundled in it
    /// nor already loaded here — the guard in front of every replay that stores pixels.
    fn require_content(&self, file: &DocumentFile) -> Result<()> {
        let missing = self.unresolved_content(file);
        if missing.is_empty() {
            return Ok(());
        }
        Err(DocError::MissingContent(missing).into())
    }

    /// Replay a document, invoking `on_frame` with the rendered image after each
    /// action — a timelapse (§8). Ends with the document fully loaded.
    ///
    /// Native-only, because it reads each frame back with the blocking path. Making
    /// it web-capable means awaiting the readback per frame — a change to this
    /// signature, not to the replay.
    ///
    /// Refuses on unresolved content for the same reason [`Self::load_document`] does,
    /// and it matters more here rather than less: a timelapse renders every
    /// intermediate state, so a missing substrate is baked into every frame it emits.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn replay_timelapse(
        &mut self,
        file: &DocumentFile,
        mut on_frame: impl FnMut(crate::image::RgbaImage),
    ) -> Result<()> {
        let file = ValidatedFile::new(file)?;
        self.require_content(&file)?;
        self.adopt(file);
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
            // Per action, not once before the loop: a replayed `SetSubstrate` moves the
            // substrate the *media pass* samples as well as the one the deposit reads, so
            // a timelapse across a mid-document switch would otherwise go on lighting
            // every later frame through the substrate the piece started on.
            self.apply_document_substrate();
            on_frame(self.render_to_image());
        }
        self.resync_counters(&file.actions);
        Ok(())
    }

    /// Every imported brush asset (id + canonical PNG bytes) — used to seed a
    /// transport session's asset mirror so peers can fetch any brush a future
    /// stroke references (§12.4).
    pub fn all_asset_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        self.shared.apply.assets.all_bytes()
    }

    /// The canonical PNG bytes of one imported brush asset, if loaded — for
    /// seeding a live session's mirror or a second (preview) engine.
    pub fn asset_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.shared.apply.assets.bytes(id)
    }

    /// Whether a brush asset is loaded in this engine.
    pub fn has_asset(&self, id: AssetId) -> bool {
        self.shared.apply.assets.contains(id)
    }

    /// Whether this engine already holds what `need` names, in whichever of the
    /// three stores it belongs to.
    pub fn holds(&self, need: AssetNeed) -> bool {
        match need {
            AssetNeed::Brush(id) => self.has_asset(id),
            AssetNeed::Substrate(_) => need
                .substrate()
                .is_some_and(|id| self.substrate_bytes(id).is_some()),
            AssetNeed::Picture(id) => self.shared.apply.pictures.contains(id),
        }
    }

    /// What `file` needs that neither it bundles nor this engine already holds
    /// (§8, §12.4).
    ///
    /// A lean file leaves out content it expects the opening app to produce — the
    /// assets that ship with it — so this is the bill, and it has to be settled
    /// **before** [`Engine::load_document`] replays the log. A `SetSubstrate` whose
    /// height map is not registered when its strokes replay deposits them through
    /// the flat stand-in, and those pixels are stored (§6.4).
    ///
    /// Empty for a file that bundles everything, which is what
    /// [`Engine::save_bytes`] writes.
    pub fn unresolved_content(&self, file: &DocumentFile) -> Vec<AssetNeed> {
        file.unbundled_content()
            .into_iter()
            .filter(|need| !self.holds(*need))
            .collect()
    }

    /// Start a fresh, empty document in `color_space`, on `substrate`.
    ///
    /// The **only** way to choose a color space, and deliberately so: the channel
    /// layouts differ between spaces, so existing tiles cannot be reinterpreted and
    /// changing it can never preserve a document. Modelling it as a setter hid that
    /// — every caller was really asking for a new document (§6.7).
    ///
    /// Takes `&mut self` rather than being an associated function because
    /// frontend-provided *resources* survive: imported brush assets, and the
    /// registered substrate and environment bytes. Those belong to the app, not to
    /// the document, and re-fetching them on every New would be gratuitous.
    /// Fails with [`DocError::UnsupportedColorSpace`] if this build does not carry
    /// `color_space`, **before** anything is reset — so a refusal leaves the open
    /// document alone, the same bargain [`Self::load_document`] makes. A frontend
    /// whose picker comes from
    /// [`colorspace::all_available`](crate::colorspace::all_available)
    /// never sees it.
    pub fn new_document(
        &mut self,
        color_space: ColorSpaceId,
        substrate: SubstrateId,
    ) -> crate::error::Result<()> {
        let cs = crate::colorspace::make(color_space)
            .ok_or(DocError::UnsupportedColorSpace(color_space))?;
        self.initial_substrate = substrate;
        self.reset_document();
        self.rebuild_gpu_for(cs);
        self.apply_document_substrate();
        Ok(())
    }

    /// The document's current substrate (§6.4). Change it with
    /// [`crate::command::DocCommand::SetSubstrate`].
    pub fn substrate(&self) -> SubstrateId {
        self.document().substrate
    }

    /// The canonical PNG bytes of a loaded image substrate — what a save file bundles
    /// and what a live session serves to a joining peer (§8, §12.4).
    pub fn substrate_bytes(&self, id: SubstrateId) -> Option<Vec<u8>> {
        self.shared.apply.substrates.bytes(id)
    }

    /// What share of a substrate a tip with this tooth — the `give` it settles with
    /// and the `softness` of its contact transition — travelling along `dir`, stands
    /// on (§6.4) — the bearing fraction the tool books its half of a toothed transfer
    /// against.
    ///
    /// Exposed because it is the model's own falsifiable quantity: it is the substrate's
    /// own rise-along-the-travel distribution integrated against the contact gate, so
    /// it can be checked against the map rather than taken on trust
    /// (`tests/tooth.rs`). `dir` is there because contact reads the substrate's slope
    /// *along the travel*, which makes the curve a property of the substrate and the
    /// direction crossing it together. Builds the substrate if this is the first time
    /// it has been asked for.
    ///
    /// At the **document's** scale, since that is the substrate a stroke would actually
    /// bite right now (§6.4) — the same pair `apply` resolves. Asking for a bearing
    /// against a differently-sized substrate than the one in force would be asking about
    /// a substrate nothing is painting on.
    pub fn substrate_bearing(
        &self,
        id: SubstrateId,
        give: f32,
        softness: f32,
        dir: stark_model::geom::Vec2,
    ) -> f32 {
        let substrate = Substrate {
            id,
            scale: self.document().substrate_scale,
        };
        self.shared
            .apply
            .substrates
            .get(&self.shared.gpu, substrate)
            .bearing(give, softness, dir.to_array())
    }

    /// Import a canvas substrate from a height-map PNG, returning the id that names it
    /// (§6.4). The frontend fetches the bytes — the engine embeds none — and this is
    /// how a substrate enters the engine, whether it ships with the app, came out of a
    /// save file, or arrived from a peer.
    ///
    /// **The id is derived from the image, never asserted alongside it.** The
    /// previous `register_substrate(id, bytes)` let a caller bind any name to any
    /// bytes, and nothing downstream could tell a wrong binding from a right one —
    /// which is the joint the tooth's divergence came through, since a substrate that
    /// failed to arrive fell back to `Flat` and baked a flat deposit into tiles that
    /// never heal. Here a mismatch cannot be expressed: ask for `id`, and `id` is
    /// what these bytes *are*.
    ///
    /// Idempotent, and cheap on a repeat — the same image re-imports to the same id.
    /// If it is the substrate in use, it is rebuilt so the bytes take effect at once.
    pub fn import_substrate(&mut self, png_bytes: &[u8]) -> Result<SubstrateId> {
        let (id, canonical) = crate::gpu::substrate::canonicalize(png_bytes)?;
        // `canonicalize` produced these bytes by encoding a field it decoded, so the
        // registry's own decode of them cannot fail; it is the same door the accepting
        // path goes through and there is nothing here to report differently.
        let rebuilt = self
            .shared
            .apply
            .substrates
            .register(&self.shared.gpu, id, canonical)?;
        if rebuilt {
            self.apply_substrate();
        }
        Ok(id)
    }

    /// Import a **picture** (any image PNG), returning the content id a
    /// [`PlaceImage`](stark_model::document::ActionKind::PlaceImage) references it by
    /// (§23).
    ///
    /// [`import_brush`](Engine::import_brush)'s sibling, and the same bargain: the id
    /// names the decoded, capped picture rather than the file bytes, so two people who
    /// encoded the same photograph differently converge on one id and the stored form
    /// reloads to it. Idempotent, and cheap on a repeat — placing the same reference
    /// image twice holds one copy of it.
    ///
    /// A **request**, not a command (§4): it has to answer with the id, because the
    /// action that references it cannot be built until the id exists. That ordering is
    /// also what a shared session depends on — see
    /// [`CollabSession::add_content`](stark_net::CollabSession), which must be told
    /// about the bytes before the commit that names them goes out.
    pub fn import_picture(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.shared.apply.pictures.import(png_bytes)
    }

    /// Take in a picture that arrives already named: out of a save file's bundle, or
    /// fetched for a peer's `PlaceImage` (§8, §12.4, §23).
    ///
    /// [`accept_substrate`](Engine::accept_substrate)'s argument, applied to the third
    /// kind: bytes installed under someone else's id would place a *different
    /// picture* than the log says, so they are refused rather than installed. The
    /// failure it rules out is quieter than a substrate's — no tooth is baked, the wrong
    /// photograph simply appears — but it is the same joint, and the same check closes
    /// it.
    pub fn accept_picture(&self, expected: AssetId, png_bytes: &[u8]) -> Result<()> {
        let actual = self.shared.apply.pictures.insert_bytes(png_bytes)?;
        if actual != expected {
            return Err(DocError::Misnamed {
                expected: AssetNeed::Picture(expected),
                actual: AssetNeed::Picture(actual),
            }
            .into());
        }
        Ok(())
    }

    /// Take in a substrate that arrives already named: out of a save file's bundle, or
    /// fetched for a peer's `SetSubstrate` (§8, §12.4). The bytes are kept verbatim —
    /// they are canonical by construction — and **checked against the id that asked
    /// for them**.
    ///
    /// The check is the point. Bytes installed under someone else's id are the one
    /// way a content-addressed substrate could still deposit the wrong tooth, so they
    /// are refused rather than installed. `import_substrate` needs no equivalent: there
    /// the id comes out of the bytes, so there is nothing to disagree with.
    ///
    /// If this is the substrate the document already moved to while its bytes were in
    /// flight, registering it is also what swaps the flat stand-in for the real
    /// substrate.
    pub fn accept_substrate(
        &mut self,
        expected: SubstrateId,
        png_bytes: &[u8],
    ) -> Result<SubstrateId> {
        let actual = crate::gpu::substrate::identify(png_bytes)?;
        if actual != expected {
            // Named through `for_substrate`, the one place a substrate's procedural
            // `Flat` case is answered. `identify` always derives an image, so only
            // `expected` can be `Flat` — and a caller naming it has brought bytes for
            // a substrate that has none, which is the same refusal said properly.
            return Err(match (
                AssetNeed::for_substrate(expected),
                AssetNeed::for_substrate(actual),
            ) {
                (Some(expected), Some(actual)) => DocError::Misnamed { expected, actual },
                _ => DocError::Asset(
                    "the flat substrate is procedural and has no bytes to install".into(),
                ),
            }
            .into());
        }
        let rebuilt =
            self.shared
                .apply
                .substrates
                .register(&self.shared.gpu, actual, png_bytes.to_vec())?;
        if rebuilt {
            self.apply_substrate();
        }
        Ok(actual)
    }

    /// Bring the GPU-side substrate in line with the document's, rebuilding it if the
    /// document moved to a different one **or laid the same one at a different size**
    /// — after a commit, an undo, a load, or a remote merge. A no-op when unchanged,
    /// which is the common case.
    ///
    /// The pair, through [`DocState::substrate`], because the pair is what a `SubstrateMap` is
    /// built from (§6.4): a scale change with the substrate unmoved has to rebind the
    /// media pass exactly as a substrate change does, and asking with the id alone was
    /// how it would silently not.
    ///
    /// There is deliberately no public `set_substrate`: the substrate is document state
    /// (§6.4), so it changes by logging an action like anything else.
    ///
    /// [`DocState::substrate`]: crate::document::DocState::substrate
    pub(super) fn apply_document_substrate(&mut self) {
        let substrate = self.document().substrate();
        if self
            .shared
            .apply
            .substrates
            .set(&self.shared.gpu, substrate)
        {
            self.apply_substrate();
        }
    }

    /// Rebind the current substrate in the media pass — the only thing that samples
    /// it. No pipeline or pool rebuild, no document reset, and **no reallocation of
    /// the compositor's attachments**: what a swap costs is one bind group, which is
    /// what the binding stamp beside the generation buys (`CompositorPipeline`).
    /// This sentence was true of the pipeline and false of the frame until then.
    fn apply_substrate(&mut self) {
        self.compositor_pipeline
            .set_substrate(self.shared.apply.substrates.current());
    }

    /// The current lighting environment (§6.3).
    pub fn environment(&self) -> EnvironmentId {
        self.shared.environment.id()
    }

    /// Whether `id` is ready — `Neutral` always is; an HDR environment is ready once
    /// its bytes have been [`register_environment`](Self::register_environment)ed.
    pub fn environment_loaded(&self, id: EnvironmentId) -> bool {
        self.shared.environment.is_loaded(id)
    }

    /// Provide (frontend-fetched) HDR bytes for an environment. If it's the one in
    /// use, it's rebuilt so the bytes take effect immediately.
    ///
    /// **The bytes are decoded before they are stored**, by
    /// [`Registry::register`](crate::gpu::registry::Registry::register) — which is
    /// where both resources' door now is, rather than a check each caller had to
    /// remember. An environment is fetched over the network and handed straight in,
    /// so this is the boundary between bytes somebody else wrote and a value the
    /// engine treats as its own: a truncated download or a file that is not an
    /// `.hdr` is refused, the caller can say so, and the canvas keeps the light it
    /// has. Without it the first *use* of the id met a decoder panic on the render
    /// thread — an abort on the web, with the painting unsaved.
    ///
    /// The decode is paid twice on the accepting path (once at the door, once in the
    /// build), and that is the honest price of validating before storing: an HDR is
    /// registered a handful of times in a session, where the build behind it is a mip
    /// chain. Keeping the first decode instead would be several megabytes of float
    /// image held for a build that has already happened — see
    /// [`EnvironmentId::Decoded`](crate::gpu::registry::Resource::Decoded).
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) -> Result<()> {
        let rebuilt = self
            .shared
            .environment
            .register(&self.shared.gpu, id, hdr_bytes)?;
        if rebuilt {
            self.apply_environment();
        }
        Ok(())
    }

    /// Switch the lighting environment. A view setting, so this never touches the
    /// document — it just re-lights the canvas on the next render. Image
    /// environments fall back to the procedural neutral one until their bytes arrive.
    ///
    /// Private: reached through
    /// [`crate::command::ViewCommand::SetEnvironment`].
    pub(super) fn set_environment(&mut self, id: EnvironmentId) {
        if self.shared.environment.set(&self.shared.gpu, id) {
            self.apply_environment();
        }
    }

    /// Rebind the current environment in the media pass.
    fn apply_environment(&mut self) {
        self.compositor_pipeline
            .set_environment(self.shared.environment.current());
    }

    /// Rebuild the GPU subsystems (pool/stroke/compositor) for `id`. Assumes the
    /// document is already empty (no tiles of the old format are referenced).
    /// Takes the *resolved* space rather than an id, which is what keeps this
    /// infallible: every caller has already had to obtain one, so there is no
    /// "unsupported space" case left to handle here or to forget.
    fn rebuild_gpu_for(&mut self, cs: Arc<dyn ColorSpace>) {
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

    /// Reset to an empty document (one root layer) before a load/replay. Also
    /// leaves any shared session: the caller (UI/transport) tears down the
    /// network side; `join_collaboration` re-enables after its reset.
    fn reset_document(&mut self) {
        self.timeline = Timeline::Linear(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER).with_substrate(self.initial_substrate),
        ));
        self.preview.clear();
        self.peers.clear();
        self.committed_changed();
        // The document that just arrived is what later edits are measured against
        // (`Engine::doc_origin`). After `committed_changed`, which moved the counter
        // this is a copy of — a reset is itself a committed change, and the empty
        // document it leaves has not been edited.
        //
        // A load re-states this once its replay is done, since every action of that
        // replay moves the counter again; a join does not, because actions reaching a
        // joiner over the wire *are* edits to a document no file of theirs holds.
        self.doc_origin = self.doc_revision;
        // One assignment: who this client is and what it owes the wire go back to
        // what a fresh engine starts with, which is the same statement the two
        // constructors make rather than a fourth list of fields to keep level.
        self.authoring = Authoring::solo();
        self.session.cancel_stroke();
        self.session.active_layer = ROOT_LAYER;
    }

    /// Commit one already-built action onto the timeline (replays its GPU work).
    fn replay_one(&mut self, action: Action) {
        let ctx = &mut self.shared.apply;
        self.timeline.push(action, ctx);
        // A replayed action is a committed change like any other, and
        // `doc_revision` says so in its own doc — "a commit, an undo, a merged
        // remote action, **a load**". Pushing straight onto the timeline left that
        // false for every action of every load and every timelapse frame.
        //
        // Invisible until something read the counter per frame rather than per edit:
        // the timelapse renders after each action, and with the revision frozen every
        // frame after the first drew the first one's list (C4). Cheap here —
        // `repoint_active_layer` returns on its first line while the layer exists,
        // which through a replay from an empty document it does.
        self.committed_changed();
    }

    /// After loading, advance the Lamport clock past everything in the log so new
    /// edits get fresh, monotonic ids.
    ///
    /// **One counter, where there were two.** The layer counter that stood beside it
    /// is gone with the id shape it served: a layer's id is the id of the action that
    /// minted it (`LayerId`), so resuming the action clock resumes the layer ids too,
    /// and there is nothing left that could be resumed wrongly. What that machinery
    /// cost while it existed is written up in §17.9 — a per-actor counter has to be
    /// recovered from the log at *both* doors a document arrives by, and `AddFilter`
    /// was missing from the list at one of them until the list stopped being a list.
    pub(super) fn resync_counters(&mut self, actions: &[Action]) {
        let max_lamport = actions.iter().map(|a| a.id.lamport).max();
        self.authoring.clock = max_lamport.map_or(0, |m| m + 1);
    }
}
