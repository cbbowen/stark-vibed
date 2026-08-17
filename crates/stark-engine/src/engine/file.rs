//! The document's lifecycle: saving it, opening it, and the frontend-provided
//! resources it is replayed against (§8, §6.4, §6.6).
//!
//! The document *is* the action log, so a file is that log plus what it takes to
//! replay it — the brush shapes strokes reference and the canvas grounds they were
//! deposited against, both bundled, because a deposit is stored and no later arrival
//! un-bakes one laid against the wrong ground. [`Engine::adopt`] is the order those
//! have to arrive in, written once for the three callers that need it.
//!
//! The registries live here for the same reason: a ground is a replay input first
//! and a rendering input second, and its identity is derived from its bytes rather
//! than asserted alongside them, so a wrong binding cannot be expressed.

use stark_model::DocError;
use std::sync::Arc;

use super::{Authoring, Engine, GpuBuild, GpuKeep, ROOT_LAYER, build_gpu};
use crate::Result;
use crate::colorspace::ColorSpace;
use crate::document::{DocState, LinearTimeline, effective_actions};
use crate::gpu::EnvironmentId;
use stark_model::AssetId;
use stark_model::AssetNeed;
use stark_model::ColorSpaceId;
use stark_model::DocumentFile;
use stark_model::SurfaceId;
use stark_model::document::Action;

impl Engine {
    /// Snapshot the document as a saveable [`DocumentFile`] (§8), bundling the
    /// brush-shape assets that strokes actually reference (§6.6) and the canvas
    /// grounds the log names (§6.4).
    ///
    /// **What the log names is asked of [`DocumentFile::required_content`]**, which
    /// is the crate's one answer to that question (`content.rs`) and already what
    /// the loader, the joiner and the transport ask. This scanned the log itself
    /// until it didn't: two scans of one log, in two modules, is two things to teach
    /// about a new action that carries an id — and the one that is *not* taught
    /// writes a file that silently fails to bundle it.
    ///
    /// Every ground the log names travels, not just the one it ends on: the tooth
    /// reads whichever was in force when a stroke was made, so a document that
    /// switched part-way through needs both to replay to the same pixels. `Flat` is
    /// skipped — procedural, no bytes — as is anything whose image never arrived,
    /// which cannot be bundled because it was never held.
    pub fn document_file(&self) -> DocumentFile {
        let mut file = DocumentFile::new(self.timeline.clone_actions());
        file.canvas.color_space = self.shared.color_space.id();
        // Before the scan below, which reads it: the ground the log *starts* on is
        // named by the container rather than by any action, and is otherwise the one
        // piece of content nothing asks for.
        file.canvas.surface = self.initial_surface;
        for need in file.required_content() {
            match need {
                AssetNeed::Brush(id) => {
                    if let Some(bytes) = self.asset_bytes(id) {
                        file.assets.push((id, bytes));
                    }
                }
                AssetNeed::Ground(_) => {
                    if let Some(id) = need.surface()
                        && let Some(bytes) = self.surface_bytes(id)
                    {
                        file.surfaces.push((id, bytes));
                    }
                }
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
    /// a canvas ground is megabytes, so a doodle on the built-in gesso weighs 2.8
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
        let keep = |id: &AssetId| !resolvable.contains(id);
        file.assets.retain(|(id, _)| keep(id));
        file.surfaces
            .retain(|(id, _)| AssetNeed::ground(*id).is_none_or(|n| keep(&n.content())));
        Ok(file.to_bytes()?)
    }

    /// Empty this engine's document and install everything `file` needs **before**
    /// its log can be replayed onto it — the shared preamble of opening a document,
    /// joining a shared session, and running a timelapse.
    ///
    /// The order is the whole content, and every step of it is a replay *input*
    /// rather than a preference:
    ///
    /// - the ground the log starts from, recorded before `reset_document` seeds the
    ///   empty document with it; replayed `SetSurface` actions move it from there
    ///   (§6.4);
    /// - the document's color space, since the channel layouts differ between
    ///   spaces and a stroke replayed through the wrong shaders is a different
    ///   painting (§6.7);
    /// - the brush shapes strokes reference, and the grounds the log names, both
    ///   before any stroke that needs them. A deposit is *stored*: unlike the media
    ///   pass, which re-reads the ground every frame and rights itself the moment an
    ///   image lands, no later arrival un-bakes a stroke laid against the flat
    ///   stand-in (§6.6, §6.4).
    ///
    /// The three callers had this written out three times and it had already drifted
    /// three ways — the timelapse was missing the initial ground, so every frame
    /// before the log's first `SetSurface` deposited against the wrong weave; it was
    /// missing the color space too, so a Mixbox document replayed through Oklab's
    /// shaders; and it swallowed a broken brush asset silently where the other two
    /// said so. A sequence whose *order* is the correctness argument is a sequence to
    /// write once.
    ///
    /// A ground or a shape that fails to install is logged and skipped rather than
    /// fatal: the document still opens, degraded, which is the same bargain either
    /// asset gets.
    pub(super) fn adopt(&mut self, file: &DocumentFile) {
        self.initial_surface = file.canvas.surface;
        self.reset_document();
        if file.canvas.color_space != self.shared.color_space.id() {
            // A `DocumentFile` reaches here from exactly two places, and both have
            // already settled this: one decoded from bytes was refused by
            // [`require_color_space`](Self::require_color_space) if this build cannot
            // honour its space, and one built in memory came from a live `Engine` in
            // this same build, whose space therefore resolves by construction. So the
            // `None` arm is not a case this function declines to handle — it is one
            // that cannot arrive.
            let cs = crate::colorspace::make(file.canvas.color_space)
                .expect("a document whose space this build lacks is refused before adoption");
            self.rebuild_gpu_for(cs);
        }
        for (_, bytes) in &file.assets {
            if let Err(e) = self.shared.apply.assets.insert_bytes(bytes) {
                tracing::warn!("skipping unreadable brush asset: {e}");
            }
        }
        for (id, bytes) in &file.surfaces {
            if let Err(e) = self.accept_surface(*id, bytes) {
                tracing::warn!("skipping a canvas ground this document names: {e}");
            }
        }
        // Reachable only from a collaboration join now — [`Engine::load_document`] and
        // the timelapse refuse outright rather than adopt (`DocError::MissingContent`).
        // A joiner is the one caller that legitimately starts short: the actions arrive
        // over the same transport as the blobs, and the waitlist parks a `SetSurface`
        // until its ground lands (§12.4), so this is a statement about ordering in
        // flight rather than about a document that cannot be reproduced.
        let missing = self.unresolved_content(file);
        if !missing.is_empty() {
            tracing::warn!(
                ?missing,
                "joining a session whose content has not arrived yet; the waitlist                  holds anything that depends on it"
            );
        }
        // The empty document is on the log's initial ground, so bind it — a timelapse
        // renders its first frame before any action has moved it.
        self.apply_document_surface();
    }

    /// Replace the document by replaying a loaded file's action log. The full
    /// undo timeline is available afterwards — undo-after-load (§8).
    ///
    /// **Fails, and changes nothing, if the file's content is not all here**
    /// ([`DocError::MissingContent`]). The check is before [`Self::adopt`] rather
    /// than inside it so a refusal leaves the open document alone: half-replacing a
    /// painting is worse than declining to.
    pub fn load_document(&mut self, file: &DocumentFile) -> Result<()> {
        self.require_content(file)?;
        self.adopt(file);
        // Replay only the *effective* sequence: a file saved from a shared
        // session is the full log, including `Undo` actions and the actions
        // they suppress (§12.3). A solo load flattens those away.
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
        }
        self.resync_counters(&file.actions);
        // Whatever the replayed log left the document on.
        self.apply_document_surface();
        Ok(())
    }

    /// Decode and load a container produced by [`Engine::save_bytes`].
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let file = DocumentFile::from_bytes(bytes)?;
        Self::require_color_space(&file)?;
        self.load_document(&file)
    }

    /// **The one place an untrusted color space enters.** Checked before anything is
    /// adopted, for the reason the magic and the version are: this is the boundary
    /// between bytes someone else wrote and a value the engine treats as its own, and
    /// a refusal here leaves whatever is open alone.
    ///
    /// Every `ColorSpaceId` decodes — the enum is unconditional so the save format's
    /// indices cannot shift with a build's features (§8, §19) — and what a build may
    /// lack is the *implementation*. So this is not a decode failure, and saying so is
    /// what lets a frontend offer "this document needs a Mixbox build" instead of
    /// "this file is corrupt".
    ///
    /// It asks here rather than in [`DocumentFile::from_bytes`], where it used to,
    /// because since the crate split (§2) the decoder cannot answer it: whether a
    /// space can be *honoured* is a fact about this build's renderer, and
    /// `stark-model` has no `mixbox` feature to consult. The property that mattered —
    /// refuse before disturbing anything — is unchanged, because nothing between the
    /// decode and here touches the open document.
    fn require_color_space(file: &DocumentFile) -> Result<()> {
        if crate::colorspace::available(file.canvas.color_space) {
            Ok(())
        } else {
            Err(DocError::UnsupportedColorSpace(file.canvas.color_space).into())
        }
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
    /// intermediate state, so a missing ground is baked into every frame it emits.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn replay_timelapse(
        &mut self,
        file: &DocumentFile,
        mut on_frame: impl FnMut(crate::image::RgbaImage),
    ) -> Result<()> {
        self.require_content(file)?;
        self.adopt(file);
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
            // Per action, not once before the loop: a replayed `SetSurface` moves the
            // ground the *media pass* samples as well as the one the deposit reads, so
            // a timelapse across a mid-document switch would otherwise go on lighting
            // every later frame through the weave the piece started on.
            self.apply_document_surface();
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
    /// two stores it belongs to.
    pub fn holds(&self, need: AssetNeed) -> bool {
        match need {
            AssetNeed::Brush(id) => self.has_asset(id),
            AssetNeed::Ground(_) => need
                .surface()
                .is_some_and(|id| self.surface_bytes(id).is_some()),
        }
    }

    /// What `file` needs that neither it bundles nor this engine already holds
    /// (§8, §12.4).
    ///
    /// A lean file leaves out content it expects the opening app to produce — the
    /// assets that ship with it — so this is the bill, and it has to be settled
    /// **before** [`Engine::load_document`] replays the log. A `SetSurface` whose
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

    /// Start a fresh, empty document in `color_space`, on `surface`.
    ///
    /// The **only** way to choose a color space, and deliberately so: the channel
    /// layouts differ between spaces, so existing tiles cannot be reinterpreted and
    /// changing it can never preserve a document. Modelling it as a setter hid that
    /// — every caller was really asking for a new document (§6.7).
    ///
    /// Takes `&mut self` rather than being an associated function because
    /// frontend-provided *resources* survive: imported brush assets, and the
    /// registered surface and environment bytes. Those belong to the app, not to
    /// the document, and re-fetching them on every New would be gratuitous.
    /// Fails with [`DocError::UnsupportedColorSpace`] if this build does not carry
    /// `color_space`, **before** anything is reset — so a refusal leaves the open
    /// document alone, the same bargain [`Self::load_document`] makes. A frontend
    /// whose picker comes from
    /// [`ColorSpaceId::all_available`](stark_model::ColorSpaceId::all_available)
    /// never sees it.
    pub fn new_document(
        &mut self,
        color_space: ColorSpaceId,
        surface: SurfaceId,
    ) -> crate::error::Result<()> {
        let cs = crate::colorspace::make(color_space)
            .ok_or(DocError::UnsupportedColorSpace(color_space))?;
        self.initial_surface = surface;
        self.reset_document();
        self.rebuild_gpu_for(cs);
        self.apply_document_surface();
        Ok(())
    }

    /// The document's current surface (§6.4). Change it with
    /// [`crate::command::DocCommand::SetSurface`].
    pub fn surface(&self) -> SurfaceId {
        self.document().surface
    }

    /// The canonical PNG bytes of a loaded image ground — what a save file bundles
    /// and what a live session serves to a joining peer (§8, §12.4).
    pub fn surface_bytes(&self, id: SurfaceId) -> Option<Vec<u8>> {
        self.shared.apply.surfaces.bytes(id)
    }

    /// What share of a ground a tip with this `tooth`, travelling along `dir`, stands
    /// on (§6.4) — the bearing fraction the tool books its half of a toothed transfer
    /// against.
    ///
    /// Exposed because it is the model's own falsifiable quantity: it is the ground's
    /// own rise-along-the-travel distribution integrated against the contact gate, so
    /// it can be checked against the map rather than taken on trust
    /// (`tests/tooth.rs`). `dir` is there because contact reads the ground's slope
    /// *along the travel*, which makes the curve a property of the weave and the
    /// direction crossing it together. Builds the surface if this is the first time
    /// it has been asked for.
    pub fn surface_bearing(&self, id: SurfaceId, tooth: f32, dir: stark_model::geom::Vec2) -> f32 {
        self.shared
            .apply
            .surfaces
            .get(&self.shared.gpu, id)
            .bearing(tooth, dir.to_array())
    }

    /// Import a canvas ground from a height-map PNG, returning the id that names it
    /// (§6.4). The frontend fetches the bytes — the engine embeds none — and this is
    /// how a ground enters the engine, whether it ships with the app, came out of a
    /// save file, or arrived from a peer.
    ///
    /// **The id is derived from the image, never asserted alongside it.** The
    /// previous `register_surface(id, bytes)` let a caller bind any name to any
    /// bytes, and nothing downstream could tell a wrong binding from a right one —
    /// which is the joint the tooth's divergence came through, since a ground that
    /// failed to arrive fell back to `Flat` and baked a flat deposit into tiles that
    /// never heal. Here a mismatch cannot be expressed: ask for `id`, and `id` is
    /// what these bytes *are*.
    ///
    /// Idempotent, and cheap on a repeat — the same image re-imports to the same id.
    /// If it is the ground in use, it is rebuilt so the bytes take effect at once.
    pub fn import_surface(&mut self, png_bytes: &[u8]) -> Result<SurfaceId> {
        let (id, canonical) = crate::gpu::surface::canonicalize(png_bytes)?;
        if self
            .shared
            .apply
            .surfaces
            .register(&self.shared.gpu, id, canonical)
        {
            self.apply_surface();
        }
        Ok(id)
    }

    /// Take in a ground that arrives already named: out of a save file's bundle, or
    /// fetched for a peer's `SetSurface` (§8, §12.4). The bytes are kept verbatim —
    /// they are canonical by construction — and **checked against the id that asked
    /// for them**.
    ///
    /// The check is the point. Bytes installed under someone else's id are the one
    /// way a content-addressed ground could still deposit the wrong tooth, so they
    /// are refused rather than installed. `import_surface` needs no equivalent: there
    /// the id comes out of the bytes, so there is nothing to disagree with.
    ///
    /// If this is the ground the document already moved to while its bytes were in
    /// flight, registering it is also what swaps the flat stand-in for the real
    /// weave.
    pub fn accept_surface(&mut self, expected: SurfaceId, png_bytes: &[u8]) -> Result<SurfaceId> {
        let actual = crate::gpu::surface::identify(png_bytes)?;
        if actual != expected {
            return Err(DocError::Asset(format!(
                "ground {expected:?} arrived as {actual:?}; refusing to install it"
            ))
            .into());
        }
        if self
            .shared
            .apply
            .surfaces
            .register(&self.shared.gpu, actual, png_bytes.to_vec())
        {
            self.apply_surface();
        }
        Ok(actual)
    }

    /// Bring the GPU-side surface in line with the document's, rebuilding it if the
    /// document moved to a different one — after a commit, an undo, a load, or a
    /// remote merge. A no-op when unchanged, which is the common case.
    ///
    /// There is deliberately no public `set_surface`: the surface is document state
    /// (§6.4), so it changes by logging an action like anything else.
    pub(super) fn apply_document_surface(&mut self) {
        let id = self.document().surface;
        if self.shared.apply.surfaces.set(&self.shared.gpu, id) {
            self.apply_surface();
        }
    }

    /// Rebind the current surface in the media pass — the only thing that samples
    /// it. No pipeline or pool rebuild, no document reset.
    fn apply_surface(&mut self) {
        self.compositor_pipeline
            .set_surface(self.shared.apply.surfaces.current());
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
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) {
        if self
            .shared
            .environment
            .register(&self.shared.gpu, id, hdr_bytes)
        {
            self.apply_environment();
        }
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
        // replaced below, and a `Surface` is two reference-counted wgpu handles.
        let surface = self.shared.apply.surfaces.current();
        let environment = self.shared.environment.current();
        let built = build_gpu(GpuBuild {
            // What a rebuild does not touch, moved through into the new context —
            // stated as a list rather than as four arguments, because "what survives
            // a color-space change" is the interesting half of this function.
            keep: GpuKeep {
                gpu: self.shared.gpu.clone(),
                assets: self.shared.apply.assets.clone(),
                selection: self.shared.apply.selection.clone(),
                surfaces: self.shared.apply.surfaces.clone(),
                environments: self.shared.environment.clone(),
            },
            target_format: self.shared.target_format,
            cs: &cs,
            surface: &surface,
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
        self.timeline = Box::new(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface),
        ));
        self.preview.clear();
        self.peers.clear();
        self.committed_changed();
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

    /// After loading, advance the id counters past everything in the log so new
    /// edits get fresh, monotonic ids.
    pub(super) fn resync_counters(&mut self, actions: &[Action]) {
        let mut max_lamport = None;
        // Only *this* client's layer ids matter: the id space is partitioned by
        // author (§17.9), so resuming past someone else's counter would
        // skip ids for no reason and, worse, hide the fact that they cannot collide.
        let mut max_ordinal = 0u64;
        for a in actions {
            max_lamport = Some(max_lamport.map_or(a.id.lamport, |m: u64| m.max(a.id.lamport)));
            // Which actions mint an id is asked of [`ActionKind::minted_layers`],
            // beside the variants, rather than answered by a list kept here. The
            // list kept here is how `AddFilter` came to be missed: it is not the
            // same set as the actions that *name* an id, so nothing about a wrong
            // answer shows up until two layers share one.
            for id in a.kind.minted_layers() {
                if id.minted_by(self.actor()) {
                    max_ordinal = max_ordinal.max(id.ordinal());
                }
            }
        }
        self.authoring.clock = max_lamport.map_or(0, |m| m + 1);
        self.authoring.next_layer = max_ordinal + 1;
    }
}
