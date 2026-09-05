//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (§7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (§7) wrap this
//! same core in a later step.
//!
//! # What is where
//!
//! [`Engine`] owns everything and is one type, but it is not one subject, so its
//! `impl` is split across this module's children along the seams it already had
//! section comments for. A child module can reach a private field of a struct its
//! parent defines, so this is a division of the *file* and not of the type: no field
//! moved, and nothing became `pub(crate)` to allow it.
//!
//! - here — the state itself, how a command reaches it ([`Engine::process`]), and
//!   the projection that comes back out ([`Engine::observe`]);
//! - `render` — the compositor's draw list, the screen frame and export (§6.3,
//!   §15.6);
//! - `live` — the preview fold and its per-stroke cache (§17.6, §6.2);
//! - `pick` — the eyedropper (§18.0.2);
//! - `collab` — the action and presence channels of a shared session (§12, §17);
//! - `file` — saving, opening, replay, and the resources a replay is run against
//!   (§8, §6.4, §6.6).

mod collab;
mod file;
mod live;
mod pick;
pub(crate) mod render;

use crate::command::Tool;
use crate::gpu::scratch::ScratchPool;
use stark_model::DocError;
use stark_model::Srgb;
use std::sync::Arc;

pub use collab::PresenceTick;
pub use pick::{PickOptions, PickSource};
pub use render::{Background, ExportPlan, ExportScale, Rendered};

use crate::Result;
use crate::assets::AssetStore;
use crate::colorspace::ColorSpace;
use crate::command::{DocCommand, GestureCommand, InputCommand, PeerCommand, ViewCommand};
use crate::document::{
    ApplyCtx, CanvasBounds, DocState, Layer, LayerContent, LinearTimeline, Timeline,
};
use crate::gpu::channels::Zeroes;
use crate::gpu::{
    BlendPass, Compositor, CompositorPipeline, Environment, EnvironmentId, FillRenderer,
    FilterPass, GpuContext, MergeRenderer, Registry, SelectionRenderer, StrokeRenderer, Substrate,
    SubstrateMap, TilePool, TransformRenderer,
};
use crate::gpu::{MediaParams, Output};
use crate::peer::Peers;
use crate::session::ShapeResult;
use crate::view::{Extent2, ViewTransform};
use stark_model::AssetId;
use stark_model::ColorSpaceId;
use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, BlendMode, Filter, GuideId, LayerId, Parcel,
    PerspectiveGuide, Scaffold, ShapeAction, StrokeRecord,
};
use stark_model::geom::{IVec2, Vec2};
use stark_model::{SubstrateId, SubstrateScale};

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId::ROOT;

/// How much resident tile memory the engine will let **history retention** hold
/// before it starts giving up undo depth (§5).
///
/// `DocState` is cheap to clone and tiles are copy-on-write, so history retention
/// drives GPU memory reclamation for free — but only if something retires history.
/// `history` keeps its snapshots geometrically spaced, so what is retained is
/// `O(log n)` states rather than `O(n)`; each still pins every tile version that has
/// changed since, and on a large canvas a tile pair is ~640 KB.
///
/// **This is a bound on a cost that has not been measured**, in the sense
/// [`MAX_RELEASE_PER_EPOCH`](crate::gpu::TilePool) and the compositor's flush cadence
/// are: 2 GiB is about 3200 tile pairs, comfortably past a large painting's working
/// set and well past what an ordinary session reaches. Raising it costs memory and
/// buys undo depth; the honest way to change it is to measure a session and say so.
///
/// The **default**, not the value — a frontend that knows what it is running on
/// sets its own ([`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand)), and
/// Stark's own offers it as a slider. A default has to be safe on the smallest
/// machine that will meet it and generous on the largest, and where those disagree
/// it errs generous: reaching this at all takes a long session on a big canvas, and
/// the cost of being wrong upwards is memory pressure the browser reports, where
/// being wrong downwards is undo steps silently gone.
///
/// It is a *ceiling on retention*, not on the document. Paint that is on the canvas
/// now is held by the current state and no amount of trimming frees it — see
/// [`Engine::trim_history`] for why that is what [`MIN_UNDO_DEPTH`] guards.
pub const DEFAULT_HISTORY_BUDGET: u64 = 2 << 30;

/// Whether a stroke's commit takes the tiles its live preview already drew, rather
/// than rendering the stroke again at pen-up (`document::PreparedStroke`, §6.2).
///
/// **On**, because the two renders are the same picture to within a level or two and
/// only one of them is paid for while the artist is waiting: a long stroke rendered
/// a second time is a hitch at exactly the moment the incremental repaint exists to
/// remove one. What the other setting buys is not a better picture but an *identical*
/// one — the stroke drawn the single way a file, an undo and a collaborator all draw
/// it, so the drawing reproduces bit for bit (§8, §9) rather than within the seam a
/// cut costs. That is worth offering and not worth defaulting to.
///
/// The **default**, not the value — [`ViewCommand::SetFastCommit`] sets it, and
/// Stark's own settings dialog offers it. Here rather than in the frontend's stored
/// preferences for the reason [`DEFAULT_HISTORY_BUDGET`] is: two defaults for one
/// behaviour is two answers to what Stark does out of the box, and this is a
/// behaviour whose two paths are nearly indistinguishable in pixels — a disagreement
/// would be invisible in everything but [`Engine::strokes_reused`].
pub const DEFAULT_FAST_COMMIT: bool = true;

/// Undo steps the engine will not trim below, however tight memory is.
///
/// **The guard against trimming for nothing.** Resident tiles are held by the
/// *current* document as much as by history, so a session with four full-canvas
/// layers can exceed any budget with almost no history at all — and there, folding
/// the undo stack away frees nothing and costs the user every step they might want
/// back. A floor makes that failure bounded: the worst case is a document that sits
/// over budget with [`MIN_UNDO_DEPTH`] steps of undo, which is the true answer rather
/// than an unbounded march to zero.
const MIN_UNDO_DEPTH: usize = 10;

/// Longest name that will be recorded, in `char`s — the wire's bound, reached for
/// rather than restated.
///
/// The argument is one argument: a name travels, so it is bounded, and nothing about
/// a text field stops a paste from being a megabyte. It is stated where the *wire*
/// can also reach it ([`stark_model::MAX_NAME`]), because a presence frame's
/// name is capped by the same number and the model cannot depend on this crate (§2).
/// Two constants agreeing at 64 would be two things to keep level.
use stark_model::MAX_NAME;

/// The name to record, given what a frontend collected: surrounding whitespace
/// trimmed, length capped, and anything that comes out empty treated as *no name*
/// rather than as a name that happens to be blank.
///
/// One funnel for every source — the panel's field, a script, a peer's command —
/// so "a name is either absent or something you can read" is a property of the
/// model rather than a habit of the UI. The logged action carries the result, so
/// replay reproduces it without re-running these rules.
///
/// Shared by layers and drawing guides: the two are named through different commands
/// — one logged, one view state — and the rule for what a name *is* should not be a
/// property of which command carried it.
///
/// Generic at both ends for that reason too, and only for that reason: the two
/// callers hold their names differently — a logged action carries a `String`,
/// because that is what goes on the wire, while a guide holds an `Arc<str>`, because
/// its list is re-projected at pointer rate — and neither difference is about what a
/// name is. `String: From<String>` is the identity, so the logged path still moves
/// its bytes rather than copying them.
fn normalize_name<T: From<String>>(name: Option<impl AsRef<str>>) -> Option<T> {
    let trimmed = name?;
    let capped: String = trimmed.as_ref().trim().chars().take(MAX_NAME).collect();
    (!capped.is_empty()).then(|| T::from(capped))
}

/// The payload of a setter: a document command whose drag previews by folding the
/// very action its release commits (§21.6). One variant per `DocCommand` /
/// `ViewCommand::Preview*` pair, and both mint their kind through
/// [`Engine::setter_kind`].
enum Setter {
    LayerBlend(LayerId, BlendMode),
    LayerOpacity(LayerId, f32),
    SelectionOpacity(f32),
    Filter(LayerId, Filter),
    MatteRect(LayerId, Vec2, Vec2),
    MattePaint(LayerId, Parcel),
    SubstrateColor(Srgb),
    SubstrateScale(SubstrateScale),
    Translate(LayerId, IVec2),
    Guide(GuideId, PerspectiveGuide),
}

/// A list [`ObservableState`] carries: **shared rather than copied**, because a
/// projection is taken after *every* command — including the pan, zoom and
/// brush-tuning commands that arrive at pointer rate — and almost none of them can
/// move any given list.
///
/// Two properties, and the type exists for both:
///
/// - **Handing one out is a refcount bump**, whatever it holds and however long it
///   is. What that saves depends on the list: the layer roster costs a walk of the
///   whole tree, cloning every name and asking
///   [`merge::plan_at`](crate::document::merge::plan_at) per row, and `Engine` keeps
///   the last one against the counters it is a function of
///   ([`Engine::projected_layers`]) so an unchanged document walks nothing at all.
/// - **Asking "did this move?" is a pointer comparison** — see the [`PartialEq`]
///   impl, which is the half a frontend holding this in a reactive signal actually
///   feels.
///
/// Generic because the argument is about what a *projection* is, not about what any
/// one list holds — a second roster projected from the same `observe()` at the same
/// rate would otherwise be a `Vec` deep-cloned and deep-compared per pointer sample.
///
/// Derefs to `[T]`, so it is read exactly as the `Vec` it replaces was. Building one
/// is `Vec::into`, which happens where the list actually changes and nowhere else.
#[derive(Debug)]
pub struct Projected<T>(Arc<[T]>);

/// Cloning shares; it never copies the elements, so this is deliberately **not**
/// derived — a derived impl would demand `T: Clone` to do what an `Arc` bump does
/// for free, and would invite someone to satisfy it.
impl<T> Clone for Projected<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> Default for Projected<T> {
    fn default() -> Self {
        Self(Vec::new().into())
    }
}

impl<T> std::ops::Deref for Projected<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> From<Vec<T>> for Projected<T> {
    fn from(items: Vec<T>) -> Self {
        Self(items.into())
    }
}

impl<T> FromIterator<T> for Projected<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T: PartialEq> PartialEq for Projected<T> {
    /// **Structural equality, with identity as a fast path.**
    ///
    /// The fast path is the whole point of sharing the list: two projections taken
    /// while the document stood still hold the *same* `Arc`, so the frontend's
    /// "did this slice move?" — asked per memo, per command — is one pointer
    /// comparison instead of a walk of every element.
    ///
    /// The fall-through keeps the answer exact. Identity alone would be sound
    /// (same `Arc` ⇒ same contents, since the contents are immutable once shared)
    /// but conservative: a rebuild that changed nothing would report a change, and a
    /// commit that leaves the tree alone happens on every stroke.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

/// The layer list [`ObservableState`] carries: the whole tree flattened in composite
/// order, shared rather than copied — see [`Projected`] for what that buys and why
/// it is a type.
pub type Layers = Projected<LayerInfo>;

/// The drawing-guide roster, shared on exactly [`Layers`]' argument (§20.5).
pub type Guides = Projected<GuideInfo>;

/// A **one-slot cache**: a value beside the key it was built from, rebuilt only when
/// the key moves (C4). The engine keeps three of these — the layer roster, the guide
/// roster and the compositor's draw list — and this type is the whole of what they
/// have in common.
///
/// **The rule, stated here rather than three times over.** A key must name every term
/// its value is a function of. One term too few and the memo hands back a stale answer
/// that nothing downstream can notice; one too many and it rebuilds for a change the
/// value cannot see, which is only a cost. So where a key cannot be exact it errs
/// *wide*, and each of the three below says where it does.
///
/// **Nothing here counts anything of its own**, and that is what makes a memo sound
/// rather than merely plausible. Every term of every key is a counter something else
/// already maintains for its own reasons — [`Engine::doc_revision`], `Preview::epoch`,
/// `Preview::fold`, [`Engine::guide_epoch`]. There is no invalidation call anywhere,
/// because the key *is* the invalidation; a memo that had to be told it was stale
/// would be one a new mutation path could forget to tell (§1).
///
/// `RefCell` because [`Engine::observe`] takes `&self`: a projection is a *read*, and
/// making it `&mut` to let it memoize would put a mutable borrow of the whole engine
/// on the path every panel takes to draw itself. The draw list is held the same way
/// for a second reason — see [`Engine::draw_list`].
struct Memo<K, V> {
    slot: std::cell::RefCell<Option<(K, V)>>,
}

/// Empty, whatever it holds. Deliberately not derived: a derived impl would demand a
/// `Default` of the key and the value, which neither has and neither needs.
impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            slot: std::cell::RefCell::new(None),
        }
    }
}

impl<K: PartialEq, V: Clone> Memo<K, V> {
    /// What was built from `key`, or `build`'s answer stored against it.
    ///
    /// **The borrow is released before `build` runs**, which is the half of this that
    /// had to be a function rather than three comparisons written out. A build is
    /// arbitrary engine code — the layer walk asks
    /// [`merge::plan_at`](crate::document::merge::plan_at) per row, the draw list
    /// walks every visible tile of every layer — so one that read the memo it was
    /// filling would panic, at run time, on whichever path a test did not take.
    ///
    /// `V: Clone`, and cheaply so at all three call sites: the two rosters hand back
    /// an `Arc` bump ([`Projected`]) and the draw list an `Arc<[CompositeGroup]>`. A
    /// memo whose value is expensive to hand out gives back what it saved.
    ///
    /// [`CompositeGroup`]: crate::gpu::CompositeGroup
    fn get_or_build(&self, key: K, build: impl FnOnce() -> V) -> V {
        if let Some(hit) = self.hit(&key) {
            return hit;
        }
        let value = build();
        *self.slot.borrow_mut() = Some((key, value.clone()));
        value
    }

    /// What is held, if it was built from `key`. Its own function so the borrow ends
    /// where the compiler says it does rather than where a reader hopes it does.
    fn hit(&self, key: &K) -> Option<V> {
        let slot = self.slot.borrow();
        let (cached, value) = slot.as_ref()?;
        (cached == key).then(|| value.clone())
    }
}

/// What a projection off the **shown** document is a function of: the committed state,
/// and the unlogged edit standing in for it (§17.6).
///
/// - `doc_revision` advances whenever the committed document does
///   ([`Engine::committed_changed`]).
/// - `preview` is `Preview::epoch`, which advances whenever the stand-in document is
///   installed, replaced or dropped — `Preview::set_doc` is the only way to move that
///   slot, and it invalidates.
///
/// Neither is new: [`render::DrawKey`] keys the compositor's draw list on these same
/// two, and every golden in the suite depends on that key being complete. A document
/// that could move without moving them would be rendering the wrong picture long
/// before it projected a stale roster.
///
/// **The live fold is absent because `shown` is not the fold.** [`Engine::observe`]
/// reads `Preview::doc` — the unlogged drag slot — and falls back to the committed
/// document; the fold is `Preview::presented`'s business and the renderer's. A stroke
/// in flight bumps `Preview::fold` and neither term here, so a stroke's samples
/// reproject nothing at all. `DrawKey` is where the fold has to be named, and it names
/// it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ShownKey {
    doc_revision: u64,
    preview: u64,
}

/// [`ShownKey`] and the one term a guide roster has that no document does: whether
/// **this client** draws each guide (§20.5).
///
/// Shutting an eye changes what the roster answers while changing nothing about the
/// document at all, so a key built from the document alone would go on handing back a
/// roster whose guides are wrong. See [`Engine::guide_epoch`], which is the exact
/// complement of the revision beside it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GuideKey {
    shown: ShownKey,
    guide_epoch: u64,
}

/// One row of the drawing-guide roster, **as this client sees it** (§20.5).
///
/// The whole reason this type exists rather than the document's own
/// [`PerspectiveGuide`] being projected: a guide is two things kept in two
/// places. Its camera, its name and its place in the roster are document state —
/// logged, saved, replicated, undoable — while its **eye** is per-client view
/// state that is none of those. What a panel row, a hit test and the overlay all
/// want is the two put together, and putting them together is a projection's
/// job.
///
/// The same shape [`LayerInfo`] takes for the same reason, and cloned as cheaply:
/// the camera is `Copy` and the name is an `Arc<str>` bump.
#[derive(Clone, Debug, PartialEq)]
pub struct GuideInfo {
    pub id: GuideId,
    /// What the artist calls it, or `None` for one never named — the panel then
    /// describes it by its place in the roster ("Perspective 2").
    pub name: Option<Arc<str>>,
    /// Whether **this client** draws it — `false` until somebody here asks for it
    /// (§20.5). Never saved, never sent, never undoable; see
    /// [`ViewCommand::SetGuideVisible`](crate::command::ViewCommand).
    pub visible: bool,
    /// The camera and its dressing: everything §20 derives from.
    pub guide: PerspectiveGuide,
}

/// A layer's presentation properties, for the UI's layer panel (§11).
///
/// `Clone` but not `Copy` since it carries the name — an `Arc<str>` bump, so
/// cloning one is still a handful of instructions and `observe()` stays cheap.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: LayerId,
    pub blend: stark_model::document::BlendMode,
    /// Whether this layer clips to the paint beneath it (§14.4).
    pub clip: bool,
    pub opacity: f32,
    pub visible: bool,
    /// The layer carrying this one — i.e. the group it is in — or `None` for one
    /// in the document's own stack (§14.2).
    pub carrier: Option<LayerId>,
    /// How deeply nested it is: 0 in the document's own stack, one more per level
    /// of carrying. What a panel indents by.
    pub depth: usize,
    /// Whether this layer carries others, i.e. whether it is a **group**. A panel
    /// gives one of these a disclosure triangle; nothing else distinguishes it.
    pub is_group: bool,
    /// Whether anything composites beneath it, so its blend mode and its clip do
    /// anything at all (§14.4.3). False on exactly one row — the
    /// bottom of the document — where a mode is the identity and a clip would
    /// erase the layer, and where a panel therefore shows both controls inert.
    pub has_backdrop: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named — in which case it is for the frontend to describe it, since only the
    /// frontend knows how it presents a stack (see [`Layer::name`]).
    ///
    /// [`Layer::name`]: crate::document::Layer::name
    pub name: Option<std::sync::Arc<str>>,
    /// Set when this layer is a **matte** (§15.2) — a frame rather
    /// than paint. `None` for an ordinary paint layer.
    ///
    /// Projected so the frontend can label it, draw its handles, and show that the
    /// brush has nowhere to go while it is selected — all without reaching past
    /// `observe()` into `DocState`.
    pub matte: Option<MatteInfo>,
    /// Set when this layer is a **filter** (§21) — a function of what is composited
    /// beneath it rather than content of its own. `None` for anything else.
    ///
    /// The whole filter, not a name for it: the filter bar's sliders read their
    /// current values off this and send the adjusted filter straight back, so a
    /// filter that grows a knob costs the projection nothing (§21.6).
    pub filter: Option<stark_model::document::Filter>,
    /// Whether the compositor would draw anything beneath this layer **within its
    /// own stack** — which is exactly what a filter layer rewrites (§21.2).
    ///
    /// Not the same question as [`has_backdrop`](Self::has_backdrop), in two ways
    /// that both matter to the one consumer (the filter bar's "nothing below it"
    /// notice). It counts the **carrier's own content**: a group's base composites
    /// at the bottom of the group, so a filter carried onto a painted layer — the
    /// "filter just this layer" gesture — has that paint beneath it even as the
    /// first carried row. And it counts only what would actually **draw**: a
    /// hidden, fully transparent or never-painted layer is culled from the draw
    /// list (`render.rs`), so a filter above nothing but those reaches nothing,
    /// whatever the row order says. `has_backdrop` stays positional because blend
    /// and clip are defined against position (§14.4.3); this one follows the
    /// renderer because a filter's reach *is* the renderer's accumulator.
    pub has_underlay: bool,
    /// The layer this one would **merge down** onto, or `None` when there is no merge
    /// here that leaves the document looking the same (§14.11).
    ///
    /// Projected as the destination rather than as a `bool` because the panel says
    /// what the click will do — the row it folds into — and asking the engine twice
    /// for one answer is how a tooltip ends up describing a different merge from the
    /// one the button performs.
    ///
    /// Unlike every other field here this is a statement about a *pair* of layers, and
    /// it is the only control in the panel that is absent rather than merely inert
    /// when the answer is no: a merge that would change the picture is not a weaker
    /// merge, it is a different edit, and offering it greyed out would suggest the
    /// document is what stands in the way.
    pub merge_down: Option<LayerId>,
    /// A number that changes exactly when this layer's own tiles do, or `None` for a
    /// layer that holds none — [`Layer::content_revision`], projected (§14.6).
    ///
    /// The layer panel's thumbnails are keyed on it. Projected rather than read off
    /// `DocState`, like everything else on this row, so the panel can ask "is the
    /// picture I cached still this layer's picture?" without reaching past `observe()`
    /// — and so the answer is taken at the same instant as the name and the blend chip
    /// beside it rather than from a document that has moved on since.
    ///
    /// **This is the one field that moves on an ordinary stroke.** Every other one
    /// describes the tree, which a stroke leaves alone — so without this the layer list
    /// would compare equal across a commit that only painted, and a thumbnail that did
    /// not notice paint landing on its layer is a wrong picture.
    ///
    /// **A stroke costs one rebuild, at the commit.** Not because the field is cheap
    /// but because a stroke is not in `shown` at all: the live fold is the renderer's
    /// document, this row is projected off the drag slot or the committed state, and
    /// neither term of [`ShownKey`] moves while a pen is down. So the roster is not
    /// even rebuilt between pen-down and pen-up, let alone compared unequal.
    ///
    /// **An unlogged drag that rewrites tiles is the case where it does cost a
    /// sample.** `PreviewTransform` and `PreviewFill` install a fresh `DocState` per
    /// pointer sample, and a fresh [`PaintTiles`] carries a revision it has never
    /// carried before — so this row moves per sample where the tree beside it does
    /// not, and the roster that would otherwise have compared equal does not. That is
    /// the price of reading the field off `shown`, which is what makes a thumbnail
    /// track the drag rather than the state behind it; the alternative is a thumbnail
    /// that is wrong for as long as the hand is down.
    ///
    /// [`PaintTiles`]: crate::document::PaintTiles
    ///
    /// [`Layer::content_revision`]: crate::document::Layer::content_revision
    pub content_revision: Option<u64>,
    /// Where the layer's frame sits on the canvas (§14.12) — what the
    /// pick-and-translate drag adds its delta to
    /// ([`DocCommand::TranslateLayer`](crate::command::DocCommand::TranslateLayer)).
    /// A matte stands in its frame like paint (§15.2); only a filter has no
    /// frame to move, and its row here is always zero.
    pub translation: stark_model::geom::IVec2,
}

impl LayerInfo {
    /// Whether a stroke aimed at this layer would draw anything. Neither a matte nor
    /// a filter has a tile map, so selecting one is legal but painting on it does
    /// nothing (§15.7, §21.4).
    pub fn is_paintable(&self) -> bool {
        self.matte.is_none() && self.filter.is_none()
    }
}

/// A matte layer's geometry and fill, for the frame chrome (§15.7).
#[derive(Clone, Debug, PartialEq)]
pub struct MatteInfo {
    /// The rect the region is defined against, in canvas px — the stored rect
    /// placed by the layer's translation (§14.12), so the chrome reads it where
    /// it shows. For a frame this is
    /// the *hole* — the piece — which is what the handles resize and what export
    /// frames against (§15.6). `None` for a region defined against no rect
    /// ([`MatteRegion::Everything`](stark_model::document::MatteRegion::Everything)): the handle box, the aspect readout and the
    /// export frame all stand down rather than invent one.
    pub rect: Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)>,
    /// The paint the region wears — flat, or a ramp (§15.4, §22.4).
    pub paint: stark_model::document::Parcel,
}

impl MatteInfo {
    /// Width × height of the rect, where there is one.
    pub fn dims(&self) -> Option<(f32, f32)> {
        let (min, max) = self.rect?;
        Some((max.x - min.x, max.y - min.y))
    }
}

/// A cheap, UI-facing projection of engine state (§7). Published to
/// the frontend so it can render chrome reactively without touching pixels.
///
/// `PartialEq` because "reactively" is the whole point: a frontend holding this in
/// a signal marks every subscriber dirty when it publishes, and a projection that
/// cannot be compared leaves it no way to notice that the answer did not move. Every
/// field was already comparable but the two view settings, both plain data
/// ([`ViewTransform`], [`MediaParams`]), so this costs
/// nothing but the derive.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservableState {
    pub can_undo: bool,
    pub can_redo: bool,
    pub is_stroking: bool,
    pub tool: Tool,
    pub view: ViewTransform,
    pub bounds: CanvasBounds,
    /// A counter that changes whenever the **committed** document does — a commit,
    /// an undo, a merged remote action, a load.
    ///
    /// For a frontend that keeps a *rendered* stand-in for the document and has to
    /// know when it went stale: the navigator's miniature (a small `export`) is one
    /// GPU render plus a readback, so it cannot be redone per observation. Nothing
    /// else in this projection answers the question — a second stroke over the same
    /// tiles leaves the bounds, the layer list and the undo flags all exactly as
    /// they were.
    ///
    /// Deliberately *not* bumped by an in-flight gesture or an unlogged drag
    /// preview, both of which change what the canvas shows at pointer rate. Those
    /// are already on screen at full size; a watcher keyed on them would re-render
    /// the miniature per pointer sample to say something the canvas is saying
    /// better. Compare `is_stroking`, which is exactly the in-flight question.
    pub doc_revision: u64,
    /// Whether the committed document has moved since it arrived — since the reset
    /// that made a new one, or since the last action of a load. False for a document
    /// that still stands exactly as the file it came out of, and for the empty one
    /// the app opens on.
    ///
    /// **Not "unsaved".** A frontend that asks before throwing the page away wants
    /// this *and* something only it knows — whether the revision on screen is one it
    /// has since written to a file (`stark-dioxus-frontend`'s `files::unsaved`). The
    /// engine supplies the half that would otherwise be a list of document-replacing
    /// call sites kept by hand (`Engine::doc_origin`).
    pub edited: bool,
    pub active_layer: LayerId,
    /// Layers bottom-to-top. Shared rather than copied — see [`Layers`].
    pub layers: Layers,
    /// Whether a selection is masking the canvas (§6.8) — drives the
    /// "Deselect"/"Invert" affordances and the selection indicator.
    pub has_selection: bool,
    /// A conservative canvas-space bounding box of this client's selected
    /// coverage, or `None` when the selection is unbounded or unknown
    /// ([`Selection::hull`](crate::document::Selection::hull)). What the
    /// transform chrome hangs its handles on; committed-only, like
    /// `has_selection`.
    pub selection_hull: Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)>,
    /// What the next shape gesture will do with the region it encloses — combine
    /// it into the selection one of four ways, or fill it (§18.0.4).
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) the next shape gesture will apply.
    pub selection_feather: f32,
    /// How strongly a **fill** gesture's parcel will land, `0..=1` (§18.0.4).
    pub shape_opacity: f32,
    /// How strongly this client's whole selection mask gates, `0..=1` (§6.8) — the
    /// selection bar's Opacity slider. Live with nothing selected too: there it is
    /// the strength the next region will take, and the whole canvas's until one is
    /// drawn ([`Selection::opacity`](crate::document::Selection::opacity)); a
    /// deselect puts it back to 1.
    pub selection_opacity: f32,
    /// Whether collaborators' selection outlines are drawn (§17.3).
    pub show_peer_selections: bool,
    /// How much resident tile memory history retention may hold before undo depth
    /// is given up, in bytes (§5) — what
    /// [`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand) sets.
    ///
    /// Projected for the reason `tool` and `brush` are: a frontend that has to read
    /// this back off the engine keeps a copy of its own, and a copy seeded from a
    /// default rather than from the engine goes stale the moment anything else moves
    /// it (§4). Stark's own settings dialog reads its slider off this, and its
    /// stored preference is captured from it.
    pub history_budget: u64,
    /// Whether a stroke's commit takes the tiles its preview already drew (§6.2) —
    /// what [`ViewCommand::SetFastCommit`](crate::command::ViewCommand) sets.
    ///
    /// Projected for `history_budget`'s reason: the settings dialog reads its switch
    /// off the engine's own value rather than off a copy that can disagree, and its
    /// stored preference is captured from it.
    pub fast_commit: bool,
    /// The drawing guides (§20.5), **as this client sees them**: the document's
    /// roster with each row carrying whether this client's eye on it is open.
    ///
    /// Projected so the Drawing Guides panel, the edit bar and the hit test read
    /// the engine's roster rather than a shadow of their own — and so that the two
    /// halves of a guide are put together once, here, rather than at each of them
    /// ([`GuideInfo`]).
    pub guides: Guides,

    // --- view settings (per-client, never historized) ---------------------
    //
    // Projected here for the same reason as `tool` and `brush`: a frontend that
    // has to read these back off the engine ends up keeping its own copy, and a
    // copy seeded from `Default` rather than from the engine goes stale the
    // moment anything else changes them (§4).
    /// Media/lighting parameters of the painterly pass (§6.3).
    pub media: crate::gpu::MediaParams,
    /// The display the screen is presented on (§6.5) — what
    /// [`ViewCommand::SetOutput`](crate::command::ViewCommand) set.
    pub output: crate::gpu::Output,
    /// The HDR lighting environment in use (§6.3).
    pub environment: EnvironmentId,

    // --- document properties fixed at creation ----------------------------
    /// The document's color space. Immutable for the document's life — changing
    /// it means starting a new document ([`Engine::new_with_color_space`]).
    pub color_space: ColorSpaceId,
    /// The physical canvas substrate (§6.4).
    pub substrate: SubstrateId,
    /// How large that substrate is laid (§6.4) — what the Lighting panel's scale slider
    /// reads and what it commits against.
    pub substrate_scale: SubstrateScale,
    /// The canvas substrate color, straight sRGB (§15.5). Document
    /// state, not a view setting — projected here so the frontend shows what the
    /// document says rather than a copy of its own that goes stale.
    pub substrate_color: Srgb,

    /// Set once the GPU has failed — a lost device, an out-of-memory, an error no
    /// scope caught (§5). `None` for a healthy device, which is every ordinary
    /// observation.
    ///
    /// **Projected because the document outlives the device.** The engine's state is
    /// an action log in ordinary memory, so a frontend told this has gone can still
    /// write the file — where discovering the same fact by aborting in the readback
    /// path takes the painting with it. What a frontend should do with it is stop
    /// dispatching and offer to save; what it must not do is keep painting, since
    /// nothing after this point reaches a pixel.
    ///
    /// An `Arc` so that this projection stays cheap to clone at pointer rate: the
    /// common value is `None`, and the uncommon one is a refcount bump rather than a
    /// `String`.
    pub gpu_failure: Option<Arc<crate::gpu::DeviceFailure>>,
}

/// The expensive half of an engine: everything a second engine on the same device
/// reuses rather than rebuilding (§11).
///
/// **What "shared" means here is exactly what [`Registry`] means by it** — *the store
/// is shared; the choice is not*. The maps of registered bytes, the decoded substrates
/// and environments, the tile pool, the compiled pipelines and the content-addressed
/// brush assets all sit behind `Arc`s and are genuinely one copy. The *choices* that
/// ride along — which substrate is in use, which environment, the media parameters —
/// are per-engine values that a clone merely **seeds** from the donor, so a sibling
/// opens mirroring the canvas it came from and is free to move from there. That
/// seeding is deliberate and is what lets the brush editor's preview open on the
/// document's own substrate with nothing re-fetched (`CompositorPipeline::sharing`).
///
/// **Why it is a type rather than a constructor's argument list.** Assembled field by
/// field, a renderer added to [`ApplyCtx`] is shared by every engine except the one
/// that constructor builds, and nothing says so until a preview canvas uses it.
/// `ApplyCtx` closes that one level down by being cloned whole; this is the same move
/// one level up, so a thing added here is shared on every path — the new engine, the
/// sibling, and the color-space rebuild.
///
/// It is also what a consumer can *hold*. A preset thumbnail wants the device and the
/// pipelines and nothing else, and this clones for a handful of refcount bumps and
/// outlives whoever it came from — where borrowing a live engine to reach them means
/// the thumbnail rig cannot exist until one does.
#[derive(Clone)]
pub struct EngineShared {
    gpu: GpuContext,
    /// The format every pipeline in `passes` was compiled against. A sibling must
    /// present to the same one — a second substrate that chose differently would fail
    /// validation rather than merely look wrong.
    target_format: wgpu::TextureFormat,
    /// The document's color space (§6.7). Shared because the pipelines below were
    /// built for it: a sibling in a *different* space is not a sibling at all, it is
    /// a rebuild (`rebuild_gpu_for`), which replaces this whole value.
    color_space: Arc<dyn ColorSpace>,
    /// The GPU subsystems an action needs in order to apply itself — the tile pool,
    /// the stroke renderer, the asset store, the selection rasterizer, and the canvas
    /// substrates — held as the `history::Action::Context` (§5).
    ///
    /// Stored rather than built per call. `history`'s `Context` is an owned associated
    /// type, so there is nothing to hand it a borrow of — and building it per call
    /// means cloning all of it on *every* commit, undo, redo and remote merge: tens of
    /// `Arc` bumps plus a `HashMap` allocation each time, for a value that only changes
    /// when the color space is rebuilt.
    ///
    /// `selection` is color-space independent (a mask is one coverage channel whatever
    /// the paint is), so unlike the pool and the stroke renderer it survives a rebuild.
    apply: ApplyCtx,
    /// The working textures and buffers every recording leases (`gpu::scratch`), one
    /// pool for the whole stack. Here as well as inside the renderers that lease from
    /// it, because a rebuild has to carry it across and the renderers it would ask do
    /// not survive one (`GpuKeep`).
    scratch: ScratchPool,
    /// The compiled compositing passes — the ~19 shaders and ~30 pipelines that make
    /// building an engine expensive. A sibling's [`CompositorPipeline`] is built over
    /// these ([`CompositorPipeline::sharing`]), so it pays for its own three view
    /// settings and nothing else.
    passes: Arc<crate::gpu::composite::CompositorPasses>,
    /// The HDR lighting environment and its registered bytes (§6.3). A view setting,
    /// so it is the *store* that is shared and the current id that is seeded.
    environment: Registry<EnvironmentId>,
    /// The media/lighting parameters a sibling opens with (§6.3) — a seed, not a
    /// shared value; see the note on the type.
    media: MediaParams,
    /// The display a sibling opens presenting to (§6.5) — a seed on `media`'s terms,
    /// since a sibling's surface is the same screen's.
    output: Output,
}

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

pub struct Engine {
    /// Everything a sibling engine reuses (§11). Held as one value so that a thing
    /// added to it is shared on every path rather than on the paths somebody
    /// remembered — see [`EngineShared`].
    shared: EngineShared,
    /// Compositing state for the **substrate**: the attachments a screen frame is
    /// built through, kept from frame to frame (`gpu::composite`). Anything drawn
    /// beside the screen — an export, the navigator's miniature — gets a
    /// [`Compositor`] of its own for the call, so it never resizes these.
    compositor: Compositor,
    /// The pipelines, layouts and view settings every `Compositor` shares. Held
    /// beside the one above rather than inside it because a second one borrows it:
    /// the expensive half of compositing is built once, and the view settings the
    /// media pass reads have one owner, so two consumers cannot disagree about the
    /// canvas substrate or the lighting.
    compositor_pipeline: CompositorPipeline,
    /// The substrate the action log starts from, written to `CanvasMeta` and used to
    /// seed the document. Plays the same role as `CanvasMeta::color_space`: it
    /// describes the empty document that the log is replayed onto, and is not
    /// itself a logged change.
    initial_substrate: SubstrateId,
    timeline: Timeline,
    session: crate::session::Session,
    /// Everyone else in the session (§17.4). Empty when solo.
    peers: Peers,
    /// The presence clock: the newest instant a caller has handed in, in seconds on
    /// a monotonic scale.
    ///
    /// `stark-engine` deliberately owns no clock *source* — that is what lets it run on
    /// wasm and native alike — but it does own the *value*. Sampling one per tick and
    /// reading it everywhere else means expiry, publishing and the timestamping of
    /// arriving frames all see the same instant, instead of each call site being free
    /// to hand in its own. `max` because a clock that steps backwards must not
    /// un-expire a peer, whatever the frontend hands in.
    now: f64,
    /// What is being *shown* over the committed document, and the caches that make
    /// showing it affordable: the unlogged drag in flight, the fold of every
    /// in-flight gesture, the settled head of each live stroke, and the epoch that
    /// says when a head has gone stale (§17.6).
    ///
    /// One field rather than four, because they are one thing with one invariant: as
    /// four they can be moved independently, and a call site that drops the drag
    /// preview without bumping the epoch shows a document nothing knows is stale. The
    /// slot cannot move without the epoch moving with it.
    preview: live::Preview,
    /// How much resident tile memory history retention may hold before undo depth
    /// is given up (§5) — [`ViewCommand::SetHistoryBudget`], defaulting to
    /// [`DEFAULT_HISTORY_BUDGET`].
    ///
    /// Per-client and never logged, for the reason the command's doc gives: how much
    /// history a machine can afford is a fact about the machine.
    history_budget: u64,
    /// Whether a stroke's commit takes the tiles its live preview already drew
    /// (§6.2) — [`ViewCommand::SetFastCommit`], defaulting to
    /// [`DEFAULT_FAST_COMMIT`].
    ///
    /// Per-client and never logged, like the budget above and for a related reason:
    /// what it changes is not the document but how *this* client spends the moment
    /// the pointer comes up. A peer receives the stroke as an action either way and
    /// renders it whole, so nothing here reaches anybody else's picture.
    fast_commit: bool,
    /// The compositor's draw list and the key it was built from — the largest of the
    /// three memos, and the only one whose value is not a projection
    /// ([`Engine::draw_list`]).
    draw_cache: Memo<render::DrawKey, std::sync::Arc<[crate::gpu::CompositeGroup]>>,
    /// The layer roster and the key it was built from ([`Engine::projected_layers`]).
    layer_cache: Memo<ShownKey, Layers>,
    /// The guide roster this client sees, on the roster above's terms plus one
    /// ([`Engine::projected_guides`]).
    guide_cache: Memo<GuideKey, Guides>,
    /// Bumped whenever this client opens or shuts a **guide's eye** (§20.5).
    ///
    /// Its own counter beside `doc_revision` because it is the exact complement of
    /// one: the eye is the one thing about a guide that is *not* in the document,
    /// so nothing in the document's revision moves when it does — and the roster
    /// this client sees is a function of both.
    guide_epoch: u64,
    /// Bumped whenever the **committed** document changes — a commit, an undo, a
    /// merged remote action, a load. Projected as
    /// [`ObservableState::doc_revision`], where it is what a frontend showing a
    /// rendered stand-in for the document (the navigator's miniature) watches to
    /// know when that render is out of date.
    ///
    /// Strictly narrower than the preview's epoch, which an unlogged drag also
    /// bumps: a drag moves at pointer rate and is *not* a change to the document,
    /// so a watcher keyed on the epoch would re-render for every sample of one. The
    /// two advance together through [`Engine::committed_changed`].
    doc_revision: u64,
    /// What [`doc_revision`](Self::doc_revision) read when the document now open
    /// *arrived* — the reset that made a new one, or the last action of a load.
    /// Projected as [`ObservableState::edited`], which is the comparison.
    ///
    /// A baseline the engine keeps rather than one a frontend keeps for it, because
    /// the engine is where the list of ways a document can be replaced is complete:
    /// `new_document`, `load_document` and a collaboration join all reach
    /// [`reset_document`](Self::reset_document), and a frontend tracking the same
    /// thing would be enumerating those call sites and missing the next one. What
    /// a frontend does own is the other half of "unsaved" — which revision it last
    /// wrote to a file — and that is a question no engine can answer.
    doc_origin: u64,
    /// How many of this client's stroke commits took the preview's tiles instead of
    /// rendering the stroke again (`PreparedStroke`, §6.2). For tests and
    /// diagnostics, on `live_head_count`'s terms: the two paths are the same pixels
    /// by design, so only a count can say which one ran.
    strokes_reused: u64,
    /// Raw pointer reports of the in-flight stroke, dumped on release under the
    /// `debug-unfrozen` feature so a misfit stroke can be replayed as a test.
    #[cfg(feature = "debug-unfrozen")]
    debug_samples: Vec<crate::command::InputSample>,
    /// Who this client is when it writes to the log, and the counters that keep
    /// its writes unique.
    authoring: Authoring,
}

/// Who this client is when it writes to the log (§17.9), and what it owes the
/// wire (§12.4).
///
/// One struct because they are one thing and they *move* as one thing: an engine is
/// authoring solo, or as some actor in a session, and every field here changes at
/// exactly the moments that identity does — sharing, joining, and the reset that
/// precedes a load. Flat, "a fresh solo session" is stated once in `reset_document`
/// and again in each constructor, and the compiler can hold those to the same *shape*
/// but never to the same values.
struct Authoring {
    actor: ActorId,
    /// This client's Lamport counter: the `lamport` half of every [`ActionId`] it
    /// mints, advanced past everything it has seen (§12.1).
    clock: u64,
    /// Locally-committed actions awaiting broadcast to peers (§12.4), drained by
    /// the transport through [`Engine::take_outbox`].
    ///
    /// `None` when solo, rather than an empty `Vec` beside a flag: "queued actions
    /// that will never be sent" was a state the pair could express and this cannot,
    /// and the presence of the queue *is* the answer to
    /// [`is_shared`](Engine::is_shared). It also decides whether a commit pays to
    /// clone its action at all — a stroke's control-point list is the largest thing
    /// in the log, and a solo session has nowhere to put the copy.
    outbox: Option<Vec<Action>>,
}

/// Whether a captured pointer report opens a stroke or continues one — see
/// [`Engine::note_debug_sample`]. A named pair rather than a `bool`, because a
/// bare `true` at the call site says nothing about which way round it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Capture {
    /// The press: forget the last stroke's samples first.
    Restart,
    /// A move within the stroke in hand.
    Continue,
}

impl Authoring {
    /// A fresh, unshared session: the solo actor, the clock at its origin, nothing
    /// owed to anybody.
    ///
    /// One counter, because a `LayerId` is the id of the action that minted it — so
    /// the clock is the only thing a fresh session has to start and a loaded one has
    /// to resume (§17.9).
    const fn solo() -> Self {
        Self {
            actor: ActorId::SOLO,
            clock: 0,
            outbox: None,
        }
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
            guide_epoch: 0,
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

    /// The canvas offset of a layer's frame (§14.12) — zero for a layer that is
    /// not there, which every action aimed at one is refused as anyway. Off the
    /// **committed** document, which is what a mint reads and what every preview
    /// entry point reads too, so the frame a gesture is converted into is the
    /// frame its commit will carry.
    pub(crate) fn frame_of(&self, layer: LayerId) -> stark_model::geom::IVec2 {
        self.timeline
            .current()
            .layer(layer)
            .map_or(stark_model::geom::IVec2::ZERO, |l| l.translation)
    }

    /// The whole move a translate gesture means (§14.12): `layer`'s subtree —
    /// a group moves as one, and translation does not inherit — with each
    /// member that has a frame to move ([`Layer::is_translatable`]: paint and
    /// mattes) displaced by the same delta. Filters are left out rather than
    /// named-and-refused: a move naming one would answer "not a no-op" forever,
    /// and a drag out and back would log a step that does nothing. Read off the
    /// committed document, like every mint. An absent layer expands to no moves,
    /// which `is_noop_on` then declines to log.
    fn translate_moves(
        &self,
        layer: LayerId,
        to: stark_model::geom::IVec2,
    ) -> Vec<(LayerId, stark_model::geom::IVec2)> {
        use stark_model::document::FRAME_LIMIT;
        // The command's `to` has not been through the funnel yet, and this
        // arithmetic runs before the commit that would clamp it — so hold it
        // here, where the subtraction below would otherwise be the first thing
        // an unbounded value reaches.
        let to = to.clamp(
            stark_model::geom::IVec2::splat(-FRAME_LIMIT),
            stark_model::geom::IVec2::splat(FRAME_LIMIT),
        );
        let doc = self.timeline.current();
        let delta = to - self.frame_of(layer);
        doc.subtree_ids(layer)
            .unwrap_or_default()
            .into_iter()
            .filter(|id| doc.layer(*id).is_some_and(|l| l.is_translatable()))
            .map(|id| (id, self.frame_of(id) + delta))
            .collect()
    }

    /// The action a setter commits — and the very one its preview folds (§21.6).
    ///
    /// **One function, so `preview == committed` (§1.3) is a property of the code
    /// rather than of a setter's two arms agreeing.** Whatever a kind needs beyond its payload
    /// is read here, once, off the committed document: the canvas-to-frame conversion
    /// a matte's rect and paint carry (§14.12), and the subtree a translate expands to.
    fn setter_kind(&self, setter: Setter) -> ActionKind {
        match setter {
            Setter::LayerBlend(id, blend) => ActionKind::SetLayerBlend(id, blend),
            Setter::LayerOpacity(id, opacity) => ActionKind::SetLayerOpacity(id, opacity),
            Setter::SelectionOpacity(opacity) => ActionKind::SetSelectionOpacity(opacity),
            Setter::Filter(id, filter) => ActionKind::SetFilter(id, filter),
            Setter::MatteRect(id, min, max) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMatteRect(id, min - d, max - d)
            }
            Setter::MattePaint(id, paint) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMattePaint(id, paint.translated(-d))
            }
            Setter::SubstrateColor(rgb) => ActionKind::SetSubstrateColor(rgb),
            Setter::SubstrateScale(scale) => ActionKind::SetSubstrateScale(scale),
            Setter::Translate(layer, to) => ActionKind::TranslateLayers {
                moves: self.translate_moves(layer, to),
            },
            Setter::Guide(id, guide) => ActionKind::SetGuide(id, guide),
        }
    }

    /// Apply one input command (§4).
    ///
    /// One-way by construction: nothing comes back. Reads go through
    /// [`Engine::observe`]; anything that must answer is a request (see
    /// [`command`](crate::command)).
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        match command.into() {
            InputCommand::Gesture(c) => self.process_gesture(c),
            InputCommand::Doc(c) => self.process_doc(c),
            InputCommand::View(c) => self.process_view(c),
            InputCommand::Peer(c) => self.process_peer(c),
        }
    }

    /// The press-drag-release lifecycle. One path for both kinds of tool
    /// (§6.8): the selection tools build an op where the brush builds a
    /// stroke, and both preview through the same `preview` DocState.
    fn process_gesture(&mut self, command: GestureCommand) {
        match command {
            GestureCommand::Start {
                tool,
                sample,
                tolerance,
                rope,
            } => {
                if tool.is_selection() {
                    // A marquee or lasso fits no curve, so it has no use for the
                    // tolerance (or the rope); its own decimation is a mask-cost
                    // knob (§6.8).
                    //
                    // What it does need is whether there is a mask to combine with,
                    // which only this side holds: an Add drawn over nothing is a New
                    // (`session::against_selection`). Off the committed document and
                    // read at the press, so the gesture's meaning is fixed before it
                    // has drawn anything.
                    let has_selection = self.document().has_selection(self.actor());
                    let frame = self.frame_of(self.session.active_layer);
                    self.session
                        .start_selection(tool, sample.pos, has_selection, frame);
                } else {
                    let seed = self.authoring.clock;
                    let frame = self.frame_of(self.session.active_layer);
                    self.session
                        .start_stroke(tool, sample, seed, tolerance, rope, frame);
                    self.note_debug_sample(Capture::Restart, sample);
                }
                self.mark_live_stale();
            }
            GestureCommand::To { sample } => {
                // The CPU half of a pointer sample, and the *whole* of what arrives at
                // input rate — the fold and the render it marks stale are paid once a
                // frame instead (`mark_live_stale`). Worth its own row because it is
                // the one phase that grows with stroke length rather than with the
                // tail: the fitter re-solves its unfrozen prefix on every push, and
                // that has measured ~350× the flattening beside it.
                crate::timing::span!("input.fit");
                if self.session.is_selecting() {
                    self.session.selection_to(sample.pos);
                } else {
                    self.session.stroke_to(sample);
                    self.note_debug_sample(Capture::Continue, sample);
                }
                self.mark_live_stale();
            }
            // A held pointer: snap the stroke to the shape it resembles (§6.9). Nothing
            // is committed and nothing is decided about the gesture's end — a snap
            // changes what the *same* drag builds, and the release still commits one
            // stroke either way.
            GestureCommand::Hold => {
                // Built here rather than inside the session, because the two halves
                // of a guide live in two places now (§20.5) and this is the only
                // side holding both. Off the **committed** document, which is what
                // a stroke is drawn over — a guide being dragged elsewhere in the
                // same instant previews without moving what a snap aims at, and a
                // snap that changed underfoot mid-gesture is the surprise §20.5
                // rules out.
                let scaffold = self.scaffold(self.timeline.current());
                if self.session.assist_stroke(&scaffold) {
                    self.mark_live_stale();
                }
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                if self.session.is_selecting() {
                    // One gesture, two things it can commit — which one was decided
                    // when the drag started (§18.0.4).
                    match self.session.end_shape() {
                        Some(ShapeResult::Select(op)) => self.commit(ActionKind::Select(op)),
                        // The layer the drag pinned at the press, not the active
                        // layer now: the op was converted into *that* layer's
                        // frame, and the two must not part (`ShapeResult::Fill`).
                        Some(ShapeResult::Fill {
                            layer,
                            op,
                            translation: frame,
                        }) => self.commit(ActionKind::Fill {
                            layer,
                            op,
                            translation: frame,
                        }),
                        None => {}
                    }
                } else {
                    // Fold first, so what is offered to the commit is the stroke as
                    // it stands at the release — the frame that would have shown the
                    // last few samples, drawn now instead of never. A fold costs the
                    // live tail; the render it saves the commit costs the stroke.
                    self.flush_live();
                    if let Some(rec) = self.session.end_stroke() {
                        self.log_debug_samples();
                        self.commit_stroke(rec);
                    }
                }
                self.mark_live_stale();
            }
            GestureCommand::Cancel => {
                self.session.cancel_stroke();
                self.mark_live_stale();
            }
        }
    }

    /// Per-client state that is published rather than logged (§17.7).
    /// Nothing here enters the history or the save file; it rides the presence
    /// channel so collaborators can see where this client is working.
    fn process_peer(&mut self, command: PeerCommand) {
        match command {
            // Any existing layer, including a matte. `active_layer` is *the
            // selected layer*, not "a paint target" — a frame is selected the same
            // way a paint layer is, which is what lets the frontend have one
            // selection concept instead of two (§15.7). A stroke aimed
            // at a matte then simply draws nothing, refused identically by `apply`
            // and by the preview path.
            PeerCommand::SetActiveLayer(id) => {
                if self.document().contains_layer(id) {
                    self.session.active_layer = id;
                    // The hover mark follows the brush's target (§18.1.10): it
                    // is built against the active layer at fold time, so moving
                    // the selection has to re-lay it there. Free when nothing is
                    // in flight — a clean fold's rebuild is an early return.
                    self.mark_live_stale();
                }
            }
            PeerCommand::SetCursor(pos) => self.session.set_cursor(pos),
            PeerCommand::SetName(name) => self.session.set_name(name),
        }
    }

    /// Document-state mutations: every arm here either commits an action or
    /// navigates the history that holds them.
    fn process_doc(&mut self, command: DocCommand) {
        self.process_doc_inner(command);
        // Every arm changes the document the in-flight previews are drawn over, so
        // the fold is rebuilt once, here, rather than at each of a dozen call sites.
        // Cheap when nothing is in flight (there is nothing to fold) and correct when
        // a peer is mid-stroke while this client edits.
        self.mark_live_stale();
    }

    fn process_doc_inner(&mut self, command: DocCommand) {
        match command {
            // Shared sessions log undo as an action peers can order (§5.4, §12.3);
            // solo falls back to navigation. Redo is an `Undo` of an `Undo`, which
            // is why the two differ only in which pair of timeline methods they
            // name — see [`Self::navigate`].
            DocCommand::Undo => {
                self.navigate(|t| t.undo_as_action(), |t, ctx| t.undo(ctx));
            }
            DocCommand::Redo => {
                self.navigate(|t| t.redo_as_action(), |t, ctx| t.redo(ctx));
            }
            DocCommand::Seek(to) => {
                self.preview.set_doc(None);
                if self.timeline.seek(to, &mut self.shared.apply) {
                    // A scrub crosses layer additions wholesale — dragging to the
                    // start of the log withdraws every one of them — so the selected
                    // layer routinely stops existing here. `committed_changed`
                    // repoints the brush for every such cause at once (§17.9).
                    self.committed_changed();
                    self.apply_document_substrate();
                }
            }
            DocCommand::Select(op) => self.commit(ActionKind::Select(op)),
            DocCommand::InvertSelection => self.commit(ActionKind::InvertSelection),
            DocCommand::SetSelectionOpacity(opacity) => {
                self.commit(self.setter_kind(Setter::SelectionOpacity(opacity)))
            }
            DocCommand::Fill { layer, op } => {
                // The command's op is on the canvas, where every gesture is; the
                // action's is in the layer's frame — the same pair of reads
                // `preview_fill` makes, so preview == committed (§14.12).
                let frame = self.frame_of(layer);
                self.commit(ActionKind::Fill {
                    layer,
                    op: op.translated(-frame.as_vec2()),
                    translation: frame,
                });
            }
            DocCommand::Transform { layer, map } => {
                // A degenerate or non-finite map would be rejected by `apply`
                // anyway (deterministically — §16.1); refusing it
                // here as well keeps a knowably-dead action out of the log.
                // Each family goes to its own action kind — the wire format
                // never carries the routing enum, only the map it named.
                if map.usable() {
                    use stark_model::document::TransformMap;
                    // The map stays stated on the canvas; the frame rides beside
                    // it and `apply` conjugates (§14.12) — the same value
                    // `preview_transform` reads.
                    let frame = self.frame_of(layer);
                    self.commit(match map {
                        TransformMap::Affine(affine) => ActionKind::Transform {
                            layer,
                            affine,
                            translation: frame,
                        },
                        TransformMap::Perspective(map) => ActionKind::TransformPerspective {
                            layer,
                            map,
                            translation: frame,
                        },
                        TransformMap::Warp(map) => ActionKind::TransformWarp {
                            layer,
                            map,
                            translation: frame,
                        },
                    });
                } else {
                    // Nothing is logged, but the gesture's preview still has to be
                    // superseded — `commit`'s bargain, made by hand because the
                    // refusal is about the map rather than about the document.
                    self.preview.set_doc(None);
                }
            }
            DocCommand::TranslateLayer { layer, to } => {
                self.commit(self.setter_kind(Setter::Translate(layer, to)))
            }
            DocCommand::FloatSelection { layer } => {
                // Asked before an action is spent, exactly as `MergeLayerDown`
                // asks its plan (§16.12): the same refusals `apply` makes, off the
                // same committed state, so a command that would no-op logs nothing.
                let frame = self.frame_of(layer);
                let doc = self.timeline.current();
                let offered = doc.layer(layer).and_then(|l| l.tiles()).is_some_and(|t| {
                    let selection = doc.selection_of(self.actor());
                    !selection.is_universal()
                        && crate::document::transform::plan_float(t, &selection, frame).is_some()
                });
                if offered {
                    let action = self.commit_minting(|a| ActionKind::FloatSelection {
                        layer,
                        child: LayerId::new(a, 0),
                        translation: frame,
                    });
                    // The float is what the hand is about to move — and it is
                    // paint, so the next stroke has somewhere to go (`AddLayer`'s
                    // reason).
                    self.arm_active(LayerId::new(action, 0));
                }
            }
            DocCommand::SetSubstrate(id) => {
                self.commit(ActionKind::SetSubstrate(id));
                // Unconditional, and a no-op when the substrate did not move: the
                // registry is brought level with the document rather than with what
                // this command asked for.
                self.apply_document_substrate();
            }
            DocCommand::SetSubstrateScale(scale) => {
                self.commit(self.setter_kind(Setter::SubstrateScale(scale)));
                // The same call for the same reason, and it is the same *state*: a
                // `SubstrateMap` is built from the substrate and its scale together, so laying
                // the substrate larger invalidates the bound substrate exactly as switching
                // it does (`gpu::substrate::Substrate`).
                self.apply_document_substrate();
            }
            DocCommand::AddLayer { carrier, above } => {
                // A freshly added layer becomes the active painting target — but only
                // if it landed and can take a stroke, which is `arm_active`'s whole
                // question.
                let action = self.commit_minting(|a| ActionKind::AddLayer {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                });
                self.arm_active(LayerId::new(action, 0));
            }
            DocCommand::PlaceImage {
                carrier,
                above,
                at,
                name,
                image,
            } => {
                // The active layer, exactly as an `AddLayer` is and for its reason:
                // it is paint, so the next stroke has somewhere to go.
                let action = self.commit_minting(|a| ActionKind::PlaceImage {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    at,
                    name,
                    image,
                });
                self.arm_active(LayerId::new(action, 0));
            }
            DocCommand::AddMatte {
                carrier,
                at,
                region,
                paint,
            } => {
                self.commit_minting(|a| ActionKind::AddMatte {
                    id: LayerId::new(a, 0),
                    carrier,
                    at,
                    region,
                    paint,
                });
                // Deliberately *not* made the active layer, unlike `AddLayer`: a
                // matte has no tile map, so painting on it is refused
                // (§15.7) and arming it as the target would just
                // swallow the user's next stroke.
            }
            DocCommand::AddFilter {
                carrier,
                above,
                filter,
            } => {
                self.commit_minting(|a| ActionKind::AddFilter {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    filter,
                });
                // Deliberately *not* made the active layer, for the reason
                // `AddMatte` is not: a filter has no tile map, so arming it as the
                // paint target would swallow the next stroke (§21.4). The frontend
                // selects it, which is what raises its bar.
            }
            DocCommand::SetFilter(id, filter) => {
                self.commit(self.setter_kind(Setter::Filter(id, filter)))
            }
            DocCommand::SetMatteRect(id, min, max) => {
                self.commit(self.setter_kind(Setter::MatteRect(id, min, max)))
            }
            DocCommand::SetMattePaint(id, paint) => {
                self.commit(self.setter_kind(Setter::MattePaint(id, paint)))
            }
            DocCommand::SetSubstrateColor(rgb) => {
                self.commit(self.setter_kind(Setter::SubstrateColor(rgb)))
            }
            DocCommand::DuplicateLayer(source) => {
                // One minted id per layer of the subtree, paired with the layer it
                // copies, in composite order — the map the action carries (§14.8).
                // The copies are this action's own ids at `k = 0..n`, so the map is
                // only a list of *sources* wearing its positions; it is still written
                // as pairs because that is the shape `apply` reads and the shape the
                // footprint claims a `Layer(src)` from.
                //
                // Through the document's own walk, not a second one here: `apply`
                // declines the action unless `ids` names exactly the subtree
                // `duplicate_layer` walks, so a copy of the traversal in the engine is
                // two walks that must agree — on this client and on every peer.
                if let Some(sources) = self.document().subtree_ids(source) {
                    let action = self.commit_minting(|a| ActionKind::DuplicateLayer {
                        ids: sources
                            .iter()
                            .enumerate()
                            .map(|(k, &src)| (src, LayerId::new(a, k as u32)))
                            .collect(),
                    });
                    // The copy is what you go on to work on.
                    self.arm_active(LayerId::new(action, 0));
                }
            }
            // The subtree travels in the action, read off the document the command
            // was aimed at (§12.6) — see `ActionKind::RemoveLayer`. A layer that is
            // not there mints an empty list and the fold declines it, which is what
            // every other action naming an absent layer does.
            DocCommand::RemoveLayer(id) => {
                let carried = self.document().carried_ids(id).unwrap_or_default();
                self.commit(ActionKind::RemoveLayer { id, carried })
            }
            DocCommand::MergeLayerDown(id) => {
                // Asked here rather than only inside `apply`, so a merge that cannot
                // preserve the document's appearance never reaches the log at all —
                // the same argument `Transform` makes about a degenerate map. `apply`
                // asks again anyway, because a peer's action arrives without passing
                // through here (§14.11).
                if let Some(plan) = crate::document::merge::plan(self.document(), id) {
                    // The frame bake's own refusal, asked here for the reason the
                    // plan is (§14.12.3): a source too large to restate in the
                    // destination's frame is declined by `apply`, and an offer
                    // that outran that would log a dead action and still repoint
                    // the brush below as if it had worked.
                    let shift = self.frame_of(plan.source) - self.frame_of(plan.dest);
                    if shift != stark_model::geom::IVec2::ZERO {
                        let bakeable = self
                            .document()
                            .layer(plan.source)
                            .and_then(|l| l.tiles())
                            .is_none_or(|tiles| {
                                crate::document::transform::plan_paint(
                                    tiles,
                                    &crate::document::selection::Selection::everything(),
                                    stark_model::geom::Affine2::from_translation(shift.as_vec2()),
                                )
                                .is_some()
                            });
                        if !bakeable {
                            return;
                        }
                    }
                    // Read **before** the commit, which is what makes it answerable:
                    // the commit repoints the brush off the layer it is about to fold
                    // away (§17.9), so afterwards there is nothing left to compare.
                    let follow = self.session.active_layer == id;
                    self.commit(ActionKind::MergeLayerDown {
                        source: plan.source,
                        dest: plan.dest,
                    });
                    // The merged layer is where the work now is, so the brush follows
                    // it. The repoint has already put it somewhere that exists; this
                    // says *which* somewhere, because picking the nearest paintable
                    // layer is not the same as picking the paint that just absorbed
                    // what you were working on.
                    if follow {
                        self.arm_active(plan.dest);
                    }
                }
            }
            DocCommand::SetLayerBlend(id, blend) => {
                self.commit(self.setter_kind(Setter::LayerBlend(id, blend)))
            }
            DocCommand::SetLayerClip(id, clip) => self.commit(ActionKind::SetLayerClip(id, clip)),
            DocCommand::SetLayerOpacity(id, opacity) => {
                self.commit(self.setter_kind(Setter::LayerOpacity(id, opacity)))
            }
            DocCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            DocCommand::SetLayerName(id, name) => {
                self.commit(ActionKind::SetLayerName(id, normalize_name(name)))
            }
            DocCommand::MoveLayer { id, carrier, at } => {
                self.commit(ActionKind::MoveLayer { id, carrier, at })
            }

            // The drawing guides (§20.5). A guide's identity is the id of the action
            // that adds it, minted through the same door a layer's is — so there is
            // no counter here and nothing for `resync_counters` to resume past
            // (`GuideId`, §17.9).
            DocCommand::AddGuide { guide, after, name } => {
                self.commit_minting(|a| ActionKind::AddGuide {
                    id: GuideId(a),
                    guide,
                    after,
                    name: normalize_name(name),
                });
            }
            DocCommand::RemoveGuide(id) => self.commit(ActionKind::RemoveGuide(id)),
            DocCommand::SetGuide(id, guide) => {
                self.commit(self.setter_kind(Setter::Guide(id, guide)))
            }
            DocCommand::SetGuideName(id, name) => {
                self.commit(ActionKind::SetGuideName(id, normalize_name(name)))
            }
            DocCommand::MoveGuide { id, after } => self.commit(ActionKind::MoveGuide { id, after }),
        }
    }

    /// Show the document committing `setter` would leave behind, without logging it —
    /// the body every `Preview*` setter arm below shares (§21.6).
    ///
    /// `None` clears the preview, which is what the release of a drag that changed
    /// nothing sends. The sanitize and the fold both happen inside
    /// [`crate::document::apply::preview_of`], which is the point: an arm cannot forget a
    /// step it does not perform. See that function for the two arms that had.
    fn preview_setter(&mut self, setter: Option<Setter>) {
        let kind = setter.map(|s| self.setter_kind(s));
        let actor = self.actor();
        let preview = kind.map(|kind| {
            crate::document::apply::preview_of(
                kind,
                self.timeline.current(),
                actor,
                &mut self.shared.apply,
            )
        });
        self.set_doc_preview(preview);
    }

    /// View-state mutations: nothing here is logged, replicated, or reachable by
    /// undo.
    fn process_view(&mut self, command: ViewCommand) {
        match command {
            ViewCommand::SetTool(tool) => {
                // Switching away mid-gesture abandons it rather than committing a
                // half-dragged marquee.
                self.session.cancel_stroke();
                self.session.tool = tool;
                self.mark_live_stale();
            }
            ViewCommand::SetBrush { brush, color } => {
                // Held here for the reason `PeerFrame::sanitized` holds a peer's:
                // a committed stroke's brush is held by `ActionKind::sanitized`,
                // and a live one is drawn by the same renderer without ever
                // becoming an action, so nothing else would. `preview ==
                // committed` needs both doors (§6.2).
                self.session.brush = brush.sanitized();
                self.session.color = color;
                self.mark_live_stale();
            }
            // Grab-and-drag: content follows the cursor, so the view center moves
            // opposite by the drag delta, carried into canvas units — through the
            // whole map, since a turned or mirrored canvas sends a screen-space drag
            // somewhere else entirely. Every arm here names a mutator rather than
            // writing a view field, so a command carrying a non-finite number is
            // refused by the view rather than stored (see [`ViewTransform`]).
            ViewCommand::Pan { delta } => self.session.view.pan_by(delta),
            ViewCommand::SetRotation(radians) => self.session.view.set_rotation(radians),
            ViewCommand::MirrorH => self.session.view.mirror_screen_h(),
            ViewCommand::CenterOn(point) => self.session.view.center_on(point),
            ViewCommand::ShowPiece(frame) => self.show_piece(frame),
            ViewCommand::Zoom { anchor, factor } => {
                self.session.view.zoom_about(anchor, factor);
            }
            ViewCommand::Pinch {
                anchor,
                to,
                scale,
                turn,
            } => self.session.view.pinch(anchor, to, scale, turn),
            ViewCommand::Resize(viewport) => self.session.view.resize(viewport),
            ViewCommand::SetShapeAction(action) => self.session.shape_action = action,
            ViewCommand::SetSelectionFeather(feather) => {
                self.session.set_selection_feather(feather);
            }
            ViewCommand::SetShapeOpacity(opacity) => self.session.set_shape_opacity(opacity),
            ViewCommand::SetShowPeerSelections(show) => self.session.show_peer_selections = show,
            ViewCommand::SetGuideVisible(id, visible) => {
                // The eye is the one per-client thing about a guide (§20.5), so it
                // moves the session and never the document. The bump is what a
                // frontend's memo on the roster watches: nothing in `doc_revision`
                // moves when an eye does, and without saying so the panel would
                // keep showing the eye it drew last time.
                if self.session.set_guide_visible(id, visible) {
                    self.guide_epoch = self.guide_epoch.wrapping_add(1);
                    self.mark_live_stale();
                }
            }
            ViewCommand::PreviewGuide(drag) => {
                self.preview_setter(drag.map(|(id, guide)| Setter::Guide(id, guide)));
            }
            ViewCommand::PreviewMatteRect(drag) => {
                self.preview_setter(drag.map(|(id, min, max)| Setter::MatteRect(id, min, max)));
            }
            ViewCommand::PreviewSubstrateColor(rgb) => {
                self.preview_setter(rgb.map(Setter::SubstrateColor));
            }
            // The preview moves the *document* the compositor reads, and stops there:
            // no `apply_document_substrate`, so nothing is baked while the hand is on
            // the slider. What that costs: a preview shows the scale in the
            // **light**, since the media pass re-reads the substrate every frame off
            // one uniform, and not in the **tooth**, whose substrate is a stored bake.
            // Paint already down looks right immediately;
            // what the next stroke will bite is right from the commit.
            ViewCommand::PreviewSubstrateScale(scale) => {
                self.preview_setter(scale.map(Setter::SubstrateScale));
            }
            ViewCommand::PreviewParcel(pick) => {
                self.preview_setter(pick.map(|(id, paint)| Setter::MattePaint(id, paint)));
            }
            ViewCommand::PreviewSelectionOpacity(opacity) => {
                self.preview_setter(opacity.map(Setter::SelectionOpacity));
            }
            ViewCommand::PreviewLayerOpacity(set) => {
                self.preview_setter(set.map(|(id, opacity)| Setter::LayerOpacity(id, opacity)));
            }
            ViewCommand::PreviewFilter(set) => {
                self.preview_setter(set.map(|(id, filter)| Setter::Filter(id, filter)));
            }
            ViewCommand::PreviewLayerBlend(set) => {
                self.preview_setter(set.map(|(id, blend)| Setter::LayerBlend(id, blend)));
            }
            ViewCommand::PreviewTransform(t) => {
                let preview = t.and_then(|(layer, map)| self.preview_transform(layer, &map));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewTranslate(set) => {
                self.preview_setter(set.map(|(layer, to)| Setter::Translate(layer, to)));
            }
            ViewCommand::PreviewFill(f) => {
                let preview = f.and_then(|(layer, op)| self.preview_fill(layer, &op));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewHover(report) => match report {
                Some(r) => {
                    // The CPU half of a hover report — the window refit — on its
                    // own row, so the cost of following a resting pointer is
                    // never folded into what painting costs (`input.fit`, §7.1).
                    crate::timing::span!("input.hover");
                    // A report the window declined — sub-tolerance drift under a
                    // resting pen — refolds nothing.
                    if self.session.hover_to(r.sample, r.tolerance, r.reach) {
                        self.mark_live_stale();
                    }
                }
                None => {
                    if self.session.clear_hover() {
                        self.mark_live_stale();
                    }
                }
            },
            ViewCommand::SetMediaParams(params) => self.compositor_pipeline.set_media(params),
            ViewCommand::SetOutput(output) => self.compositor_pipeline.set_output(output),
            ViewCommand::SetEnvironment(id) => self.set_environment(id),
            ViewCommand::SetHistoryBudget(bytes) => self.history_budget = bytes,
            ViewCommand::SetFastCommit(on) => self.fast_commit = on,
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, without the per-sample staleness marks. Interactive samples go
    /// through `GestureCommand::To`, whose marks a frame's `flush_live` services
    /// by rendering the in-flight tail — right for drawing (the user must see
    /// each frame's moves), pointless across a replay where nothing is presented
    /// in between. This renders the stroke exactly once, at commit. Used by the
    /// brush editor's test-stroke replay.
    /// Answers the id of the action it committed, or `None` where the samples held no
    /// stroke — empty, or a hand that never left the first point.
    pub fn replay_stroke(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
    ) -> Option<ActionId> {
        self.replay_stroke_seeded(tool, samples, self.authoring.clock, 0.0)
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the
    /// Lamport clock. Replaying the same samples repeatedly advances the clock
    /// (each replay commits), so the seed — and with it the color dynamics and
    /// dither — changes on every replay. A caller re-rendering *one* stroke to
    /// show the effect of a brush change (the brush editor's preview) wants the
    /// jitter held fixed, so only the edited parameter moves.
    /// `rope` is the §6.11 smoothing string, and it is a parameter here — where
    /// [`Engine::replay_stroke`] pins it to zero — because the brush editor's
    /// preview replays a *recorded hand* (the user's own test stroke) and has to
    /// show what the smoothing slider beside it would do to that hand.
    /// **Answers what it committed**, which is §4's requirement of anything that
    /// mutates and is not a command: this is a batch of inputs ending in a logged,
    /// replicated action, so a caller has to be able to tell a committed stroke from a
    /// refused one. `None` is "these samples held no stroke": none at all, or a hand
    /// that never left its first point.
    ///
    /// Not routed through `GestureCommand` instead, deliberately. The command tier's
    /// payloads are values a frontend builds per event; this takes a borrowed slice a
    /// bench replays in a loop, and making it a command would mean an `Arc<[_]>` per
    /// call to say the same thing. Answering is what §4 actually asks for.
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
        seed: u64,
        rope: f32,
    ) -> Option<ActionId> {
        let mut it = samples.iter();
        let first = it.next()?;
        // Replayed samples are already in canvas space and came from a fit or from a
        // generator, not from a device, so there is no device tolerance to declare.
        let frame = self.frame_of(self.session.active_layer);
        self.session.start_stroke(
            tool,
            *first,
            seed,
            crate::path::DEFAULT_TOLERANCE,
            rope,
            frame,
        );
        for s in it {
            self.session.stroke_to(*s);
        }
        let committed = self.session.end_stroke().map(|rec| {
            let id = self.next_action_id();
            self.commit_with_id(id, ActionKind::CommitStroke(rec));
            id
        });
        self.mark_live_stale();
        committed
    }

    /// The in-flight tow, for the frontend's string overlay (§6.11) — a named
    /// read like [`view`](Self::view), at pointer rate while a smoothing brush
    /// draws. `None` whenever there is no string to show (no stroke, no rope,
    /// or the gesture has snapped to a shape).
    pub fn tow_string(&self) -> Option<crate::tow::TowString> {
        self.session.tow_string()
    }

    /// Whether a hover mark is folded into the shown canvas (§18.1.10) — what a
    /// frontend peeks before spending a [`ViewCommand::PreviewHover`]`(None)`,
    /// so taking the mark down costs a command and a repaint only when there is
    /// one to take down.
    pub fn hover_held(&self) -> bool {
        self.session.hover_held()
    }

    /// What the stroke in flight has snapped to (§6.9), or `None` where there is no
    /// stroke or the hold found nothing.
    ///
    /// A named read like [`view`](Self::view) and [`tow_string`](Self::tow_string),
    /// and it exists for the reason the split in §6.9 exists: the frontend owns the
    /// *dwell* — how long a pause has to be is a fact about a hand and there is no
    /// clock here — while the engine owns what a hold **means**. So whether a hold
    /// found anything is knowable only on this side, and a frontend that wants to
    /// know has to ask.
    ///
    /// Read **before** the gesture's `End`, which is the only moment it answers: what
    /// is committed is the path the shape produced, not the shape, and the assist goes
    /// with the gesture (`assist::AssistShape`).
    pub fn assisted(&self) -> Option<crate::assist::Assisted> {
        self.session.assisted()
    }

    /// What every projection off `shown` is keyed on, read in one place so the two
    /// below cannot come to disagree about which counters say `shown` moved — see
    /// [`ShownKey`] for what the two terms are.
    fn shown_key(&self) -> ShownKey {
        ShownKey {
            doc_revision: self.doc_revision,
            preview: self.preview.epoch(),
        }
    }

    /// The layer roster for `shown`, **rebuilt only when the document it describes
    /// has moved** — the walk in [`observe`](Self::observe) is a pure function of
    /// `shown`, and [`ShownKey`] names exactly what `shown` is a function of.
    fn projected_layers(&self, build: impl FnOnce() -> Vec<LayerInfo>) -> Layers {
        self.layer_cache
            .get_or_build(self.shown_key(), || build().into())
    }

    /// The drawing-guide roster this client sees, keyed on [`GuideKey`] — and it is
    /// here for the property rather than for the cost (§20.5).
    ///
    /// Building the roster is cheap: a handful of rows, a `Copy` camera and an
    /// `Arc` bump apiece. What the memo buys is that an *unchanged* roster hands
    /// back the same `Arc`, so the frontend's "did this move?" stays the pointer
    /// comparison [`Projected`] exists for. Rebuilt fresh per observation it would be
    /// a new allocation every time, and every memo over the roster would fall through
    /// to comparing rows.
    fn projected_guides(&self, build: impl FnOnce() -> Vec<GuideInfo>) -> Guides {
        let key = GuideKey {
            shown: self.shown_key(),
            guide_epoch: self.guide_epoch,
        };
        self.guide_cache.get_or_build(key, || build().into())
    }

    /// What the guide overlay draws and what a snapped stroke is held to, for the
    /// document `doc` — this client's shown guides (§20.5), gathered.
    ///
    /// The one place the two halves of the roster meet on the *rendering* side, as
    /// `GuideInfo` is on the panel's: the document holds the guides,
    /// [`Session::shown_guides`](crate::session::Session::shown_guides) drops the
    /// ones this client has hidden, and everything past here sees only geometry.
    pub(crate) fn scaffold(&self, doc: &DocState) -> Scaffold {
        Scaffold::of(self.session.shown_guides(doc).map(|g| &g.camera))
    }

    /// A snapshot of UI-facing state (§7).
    pub fn observe(&self) -> ObservableState {
        /// Whether the compositor would draw anything at all for `l` — the same
        /// culls `render.rs` applies, asked of the document rather than of a
        /// viewport, so the answer does not change when the artist scrolls. Both
        /// halves are [`Layer`]'s own, which is what keeps this in agreement with
        /// those culls rather than merely alongside them.
        fn contributes(l: &Layer) -> bool {
            l.is_shown() && (l.draws_content() || l.carries.iter().any(contributes))
        }
        let doc = self.timeline.current();
        // The layers and the substrate color are read from the *previewed*
        // document when one is in flight, so the frame's handles track a drag and
        // the color swatch tracks the picker (both live in the preview,
        // §15.7, §15.5) instead of lagging on the committed value — which
        // for the color would leave the panel disagreeing with the canvas it
        // controls, since rendering reads `presented`.
        //
        // Deliberately only those two. `has_selection` must stay committed-only —
        // a marquee drag would otherwise flash the selection bar in and out before
        // anything is selected — and that is asserted by
        // `a_selection_gesture_commits_the_same_op_it_previewed`. A stroke preview
        // changes no presentation property, so it is not consulted here at all.
        let shown = self.preview.doc().unwrap_or(doc);
        // Flattened in **composite order** — each stack bottom-to-top, a group's
        // base before what it carries — with the tree carried alongside as `depth`
        // and `carrier` (§14.6). Flat rather than nested because that
        // is the order a panel draws in and the order the compositor draws in, and
        // one list that means both is one thing to keep in agreement.
        /// What the walk knows about the stack it is currently in, one per depth.
        ///
        /// Three facts rather than three parallel vectors, because they are kept in
        /// step by exactly the same rule — truncating on the way back up is what makes
        /// them per-*stack* rather than per-depth: re-entering depth `d` from deeper is
        /// the same stack and keeps them, while descending to a new `d` starts a fresh
        /// stack.
        struct Cursor<'a> {
            /// Whether this stack has anything drawable beneath the row being visited
            /// — the walk's own answer to "would a filter here reach something"
            /// (§21.2), kept in agreement with the draw list's culls (`render.rs`)
            /// rather than with row order: a hidden, transparent or empty sibling
            /// fills nothing.
            filled: bool,
            /// The row visited before this one **in this stack** — a layer's lower
            /// sibling, which is what it would merge down onto (§14.11).
            below: Option<&'a Layer>,
            /// How far up this stack the row being visited sits, counting from the
            /// bottom — `LayerSite::index` without the search for it.
            index: usize,
        }
        // Rebuilt only when the document it describes has moved; every other
        // observation hands out the last one (see [`Engine::projected_layers`]).
        let layers = self.projected_layers(|| {
            let mut layers: Vec<LayerInfo> = Vec::new();
            // The carrier chain down to the current row: the layer (whose id
            // `LayerInfo::carrier` reports) and whether its own content draws anything —
            // the seed for the stack its carries open, since a base composites at the
            // bottom of its group (§14.1).
            let mut carriers: Vec<(&Layer, bool)> = Vec::new();
            let mut stack: Vec<Cursor<'_>> = Vec::new();
            shown.visit(&mut |l, depth| {
                carriers.truncate(depth);
                if stack.len() > depth {
                    stack.truncate(depth + 1);
                } else {
                    let filled = carriers.last().is_some_and(|&(_, draws)| draws);
                    stack.push(Cursor {
                        filled,
                        below: None,
                        index: 0,
                    });
                }
                let has_underlay = stack[depth].filled;
                stack[depth].filled = stack[depth].filled || contributes(l);
                layers.push(LayerInfo {
                    id: l.id,
                    blend: l.composite.blend,
                    clip: l.composite.clip,
                    opacity: l.composite.opacity,
                    visible: l.visible,
                    carrier: carriers.last().map(|&(c, _)| c.id),
                    depth,
                    is_group: l.is_group(),
                    // Read straight off the traversal: composite order visits the
                    // bottom of the root stack first, and that is the *only* layer
                    // with nothing beneath it (§14.4.3) — every other one has either
                    // a lower sibling or the content of the layer carrying it. So
                    // "has a backdrop" is "is not the first row", and asking the tree
                    // per layer was a search for an answer the walk already gave.
                    has_backdrop: !layers.is_empty(),
                    name: l.name.clone(),
                    matte: match &l.content {
                        // On the canvas, where the chrome lives: the region and
                        // its paint are stated in the layer's frame (§15.2), and
                        // the projection places them the way the compositor does
                        // — the mint takes the same offset back out
                        // (`DocCommand::SetMatteRect`), so the handles never do
                        // frame arithmetic.
                        LayerContent::Matte { region, paint } => {
                            let d = l.translation.as_vec2();
                            Some(MatteInfo {
                                rect: region.translated(d).rect(),
                                paint: paint.translated(d),
                            })
                        }
                        LayerContent::Paint(_) | LayerContent::Filter(_) => None,
                    },
                    filter: l.filter(),
                    has_underlay,
                    // Asked of the *shown* document, like everything else on this row, so
                    // the control tracks a drag preview rather than the value behind it.
                    //
                    // **Read off the walk**, which already knows the two things the
                    // question needs: the lower sibling it visited a moment ago, and the
                    // carrier it descended through. Asking `merge::plan` per row instead
                    // spent a `site_of` — a walk of the whole tree — per layer, which made
                    // this projection quadratic in the layer count (79 µs at 60 layers
                    // against 1.3 µs at 4). That is the search `has_backdrop` above
                    // already refuses to make, for the same reason.
                    merge_down: stack[depth]
                        .below
                        .map(|d| (d, false))
                        .or_else(|| carriers.last().map(|&(c, _)| (c, true)))
                        .and_then(|(dest, dest_is_carrier)| {
                            crate::document::merge::plan_at(&crate::document::merge::MergeSite {
                                source: l,
                                dest,
                                // The destination is the whole backdrop where it carries
                                // the source, and — in the root stack — where the source
                                // sits second from the foot over an unclipped layer, whose
                                // accumulator starts cleared (§14.11.2).
                                backdrop_is_dest: dest_is_carrier
                                    || (depth == 0
                                        && stack[depth].index == 1
                                        && !dest.composite.clip),
                                dest_is_carrier,
                            })
                            .map(|p| p.dest)
                        }),
                    // Off the *shown* document like the rest of the row, so a thumbnail
                    // tracks a drag preview's tiles rather than the committed ones
                    // behind them.
                    content_revision: l.content_revision(),
                    translation: l.translation,
                });
                stack[depth].below = Some(l);
                stack[depth].index += 1;
                carriers.push((l, l.draws_content()));
            });
            layers
        });
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool,
            view: self.session.view,
            bounds: doc.bounds(),
            doc_revision: self.doc_revision,
            edited: self.doc_revision != self.doc_origin,
            active_layer: self.session.active_layer,
            layers,
            has_selection: doc.has_selection(self.actor()),
            selection_hull: doc.selection_of(self.actor()).hull(),
            // Off the *shown* document, unlike the two above: this one previews
            // (`PreviewSelectionOpacity`), and a bar reading the committed number
            // would fight its own slider mid-drag (§6.8).
            selection_opacity: shown.selection_of(self.actor()).opacity(),
            shape_action: self.session.shape_action,
            selection_feather: self.session.selection_feather(),
            shape_opacity: self.session.shape_opacity(),
            show_peer_selections: self.session.show_peer_selections,
            history_budget: self.history_budget,
            fast_commit: self.fast_commit,
            guides: self.projected_guides(|| {
                // Off `shown` — the previewed document — for the reason the layer
                // list and the substrate color are: a guide's drag previews
                // through `PreviewGuide`, and a panel reading the committed
                // roster would show the pose the hand left behind (§20.5).
                shown
                    .guides()
                    .iter()
                    .map(|g| GuideInfo {
                        id: g.id,
                        name: g.name.clone(),
                        visible: self.session.guide_visible(g.id),
                        guide: g.camera,
                    })
                    .collect()
            }),
            media: self.compositor_pipeline.media(),
            output: self.compositor_pipeline.output(),
            environment: self.shared.environment.id(),
            color_space: self.shared.color_space.id(),
            substrate: doc.substrate,
            substrate_scale: doc.substrate_scale,
            substrate_color: shown.substrate_color,
            gpu_failure: self.shared.gpu.health().failure().map(Arc::new),
        }
    }

    /// Whether the GPU is still usable, and what went wrong if not (§5) — the same
    /// fact [`ObservableState::gpu_failure`] projects, as a **request** for a caller
    /// that holds the engine and has no projection to hand.
    ///
    /// The collaboration pump is the caller that wants it: it services peer traffic
    /// without taking an observation each time (§17.5), and a device that has died is
    /// exactly the thing that should stop it applying anything further.
    pub fn gpu_failure(&self) -> Option<crate::gpu::DeviceFailure> {
        self.shared.gpu.health().failure()
    }

    /// The current committed document state.
    pub fn document(&self) -> &DocState {
        self.timeline.current()
    }

    /// Where the history playhead stands and how far it can travel, in actions —
    /// or `None` for a document whose history is not this client's alone to walk
    /// (a shared session). See
    /// [`Timeline::scrub_range`](crate::document::Timeline::scrub_range).
    ///
    /// A **request** rather than a field of [`ObservableState`]: it is asked for
    /// only while a scrubber is on screen, and putting it in the projection would
    /// have every command — every pointer sample of every stroke — pay for a
    /// number nothing else reads.
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        self.timeline.scrub_range()
    }

    /// A caption per action across the whole scrub range, oldest first — what a
    /// scrubber labels its ticks with.
    pub fn scrub_labels(&self) -> Vec<&'static str> {
        self.timeline.scrub_labels()
    }

    /// The GPU context this engine renders with (for substrate/readback setup).
    pub fn gpu(&self) -> &GpuContext {
        self.shared.gpu()
    }

    /// The current pan/zoom view (for mapping pointer input to canvas space).
    pub fn view(&self) -> ViewTransform {
        self.session.view
    }

    /// The current media/lighting parameters (§6.3).
    pub fn media_params(&self) -> crate::gpu::MediaParams {
        self.compositor_pipeline.media()
    }

    /// The display the screen is presented on (§6.5).
    pub fn output(&self) -> Output {
        self.compositor_pipeline.output()
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (§6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.shared.apply.assets.import(png_bytes)
    }

    /// Note that the **committed** document has been replaced: every cached
    /// [`FrozenHead`](live::FrozenHead) built against the old one is stale, and anything
    /// the frontend
    /// derived from it is out of date.
    ///
    /// One call rather than two counters bumped side by side at each of seven sites —
    /// a commit, either half of undo/redo, a merge, a share, a join, a reset —
    /// because "these advance together" is the property that has to hold, and a site
    /// that remembered one and forgot the other would be silent. The preview path
    /// deliberately does *not* come through here: it moves what is drawn without
    /// changing the document (see [`ObservableState::doc_revision`]).
    ///
    /// **Repointing the brush belongs here too**, for the same argument one scope up.
    /// The rule is not "a `RemoveLayer` removes a layer" — an undo of an `AddLayer`
    /// withdraws one, a merge folds one away, a peer's merge arrives having done the
    /// same, and a seek crosses additions wholesale (§17.9). All of them come through
    /// here, and none has to know it:
    /// [`repoint_active_layer`](Self::repoint_active_layer) returns on its first line
    /// when the layer still exists, which is every ordinary commit.
    fn committed_changed(&mut self) {
        self.preview.invalidate();
        self.doc_revision += 1;
        self.repoint_active_layer();
    }

    /// Move the history playhead one step, the way [`DocCommand::Undo`] and
    /// [`DocCommand::Redo`] each do it.
    ///
    /// The two are one operation named twice: a shared session logs the step as an
    /// `Undo` action peers can order (§5.4, §12.3) and a solo one navigates, and redo
    /// is an `Undo` of an `Undo` — so the *only* thing that differs is which pair of
    /// timeline methods is asked. Passing the pair rather than writing the body out
    /// twice is what stops the two drifting: dropping the preview, bumping the
    /// revision on the navigating branch and re-reading the document's substrate
    /// afterwards are all things one arm could have grown and the other not.
    fn navigate(
        &mut self,
        as_action: impl Fn(&Timeline) -> Option<ActionId>,
        step: impl Fn(&mut Timeline, &mut ApplyCtx) -> bool,
    ) {
        self.preview.set_doc(None);
        if let Some(target) = as_action(&self.timeline) {
            // The unconditional door: a step the timeline has just said it can take
            // must land, and a declined `Undo` would report a move the playhead did
            // not make.
            let id = self.next_action_id();
            self.commit_with_id(id, ActionKind::Undo(target));
        } else {
            step(&mut self.timeline, &mut self.shared.apply);
            self.committed_changed();
        }
        // A step across a `SetSubstrate` — or a `SetSubstrateScale` — moves the
        // document's substrate (§6.4).
        self.apply_document_substrate();
    }

    /// Log one action and apply it — **unless the document already reads that way**
    /// ([`is_noop_on`](crate::document::apply::is_noop_on)), in which case nothing is
    /// logged and the drag in flight is still dropped.
    ///
    /// **One door, so an arm of [`process_doc_inner`](Self::process_doc_inner) is one
    /// word.** A second entry point that skipped the no-op question would make which
    /// door an arm reaches for a per-arm judgement nothing checks — and that judgement
    /// has produced a bug before, `SetLayerVisible` logging a step for setting the
    /// value it already held while `SetLayerOpacity` did not. Ruling out the class
    /// costs nothing here, because every kind that would want the unchecked door — a
    /// stroke, a fill, a transform, a selection, a removal, a merge, a move — sits in
    /// `is_noop_on`'s exhaustive `false` arm and answers by construction (CLAUDE.md).
    ///
    /// **The unlogged drag in flight is dropped on every path through here**, once,
    /// rather than at each commit site that remembered to. A drag preview is a whole
    /// document standing in for the committed one (§17.6), so anything that moves the
    /// committed document supersedes it — leaving it up pins the canvas to the last
    /// dragged value and shadows every later edit.
    ///
    /// The declining path drops it too, which is the other half of a setter's
    /// bargain: a slider dragged out and back must log nothing *and* must still
    /// supersede the preview it left up, because a preview is superseded by something
    /// or not at all.
    fn commit(&mut self, kind: ActionKind) {
        // Sanitized before the comparison, not after: `is_noop_on` compares the
        // payload against the one already in the document, and the stored one has
        // been through this funnel. Left raw, a slider released on the value it was
        // pressed on would compare unequal to its own sanitized twin and log an
        // action that changes nothing — the case the check exists to catch.
        let kind = kind.sanitized();
        if crate::document::apply::is_noop_on(&kind, self.document(), self.actor()) {
            self.preview.set_doc(None);
            return;
        }
        // Drawn only once the action is known to be worth logging, so a slider
        // dragged out and back spends no Lamport tick.
        let id = self.next_action_id();
        self.commit_sanitized(id, kind);
    }

    /// [`commit`](Self::commit) with the action id drawn by the caller and **the
    /// no-op question not asked** — the unconditional door.
    ///
    /// The id comes from outside because two callers cannot let this draw it:
    /// [`commit_minting`](Self::commit_minting) has to build the kind *from* the id,
    /// since a layer's id is the id of the action that mints it, and
    /// [`replay_stroke_seeded`](Self::replay_stroke_seeded) answers with it.
    ///
    /// The question is not asked because for these callers a decline would be silent
    /// damage rather than a saved undo step, which is why the exemption is a door and
    /// not a list of kinds. `commit_minting` has already baked the id into the kind's
    /// layer ids, so declining would leave them naming an action that never happened;
    /// [`commit_stroke`](Self::commit_stroke) has an offer of already-rendered tiles
    /// riding the context for exactly this push (`PreparedStroke`, §6.2); and an
    /// `Undo` that declined would leave the playhead where it was
    /// ([`navigate`](Self::navigate)). That none of their kinds *could* answer "no-op"
    /// today is a fact about `is_noop_on`, not a contract they rest on.
    fn commit_with_id(&mut self, id: ActionId, kind: ActionKind) {
        self.commit_sanitized(id, kind.sanitized());
    }

    /// What both doors above open onto: log `kind` under `id` and apply it, with the
    /// payload already through the funnel.
    ///
    /// The precondition is in the name for a reason that is not the saved pass:
    /// `Logged::new` runs the funnel again on the way into the log, deliberately, so
    /// that a footprint is built from what the fold will actually see. What the
    /// precondition buys is that [`commit`](Self::commit) **logs the very value it
    /// compared** — the no-op answer and the stored payload cannot come apart, and a
    /// reader need not know `sanitized` is idempotent to trust that they agree.
    fn commit_sanitized(&mut self, id: ActionId, kind: ActionKind) {
        // Every logged edit, whatever kind — a stroke landing at pen-up, a fill, a
        // layer move. One row rather than one per `ActionKind`, because what a
        // profile is being asked here is "is a commit the hitch the artist felt",
        // and the answer is read against `input.fit` and `frame` beside it. Which
        // *kind* of commit was slow is a question the phases underneath it already
        // answer — `stroke.range` and its parts for a stroke that had to render,
        // where one that took its preview's tiles (`commit_stroke`) has none.
        crate::timing::span!("doc.commit");
        self.preview.set_doc(None);
        // `kind` arrives having been through the **minted** half of the sanitizing
        // funnel (§21.5); `Logged::new` is the "enters state" half. It runs in the two
        // doors above rather than inside the timeline so that the log and the wire
        // carry *what was applied* — the broadcast clone below is taken from this
        // action, so a peer that received the raw kind and cleaned it on arrival would
        // agree about pixels while disagreeing about the log.
        //
        // One call for every kind, rather than a payload-level call per kind that
        // carries a knob — a list every new action-with-a-knob has to be added to.
        let action = Action { id, kind };
        // Cloned only when there is somewhere for the copy to go: a `CommitStroke`
        // carries the stroke's whole fitted control-point list, the largest thing in
        // the log, and a solo session has nowhere to put it.
        let broadcast = self.is_shared().then(|| action.clone());
        let ctx = &mut self.shared.apply;
        self.timeline.push(action, ctx);
        // The committed document is what every in-flight preview is drawn over, so
        // every cached head built against the old one is now stale.
        self.committed_changed();
        if let Some(action) = broadcast
            && let Some(outbox) = self.authoring.outbox.as_mut()
        {
            outbox.push(action);
        }
        // Committing is the only thing that grows the undo stack, so it is the only
        // place retention has to be reconsidered. Deliberately not in
        // `committed_changed`, which an undo also comes through — giving up undo
        // depth *while the user is undoing* is the one moment it would be felt.
        self.trim_history();
    }

    /// Log and apply a stroke, offering the fold the tiles the preview already drew
    /// for it (`PreparedStroke`, §6.2) — what makes pen-up cost a tail rather than a
    /// stroke.
    ///
    /// The offer rides the context for exactly the one push, and the fold accepts it
    /// by taking it: a slot still full afterwards was declined — the record moved
    /// between the last fold and the release, or the base did — and is dropped here
    /// rather than left for a later fold to find.
    ///
    /// **This is the whole of what [`fast_commit`](Self::fast_commit) switches**, and
    /// the reason the setting can be one line: with nothing offered, the fold below
    /// renders the stroke exactly as a replay does, which is what makes the switched-
    /// off path bit-for-bit rather than merely close (`DEFAULT_FAST_COMMIT`).
    fn commit_stroke(&mut self, rec: StrokeRecord) {
        // Taken either way. The tiles describe a stroke that is being committed this
        // instant, so they are no use to the next fold whichever path lands it, and
        // holding them past here would pin a stroke's worth of tiles on the one
        // setting that is about *not* using them.
        let prepared = self.preview.take_prepared();
        let offered = self
            .shared
            .apply
            .offer(prepared.filter(|_| self.fast_commit));
        // The unconditional door, and the one place a decline would cost paint rather
        // than an undo step: `reclaim` would take the offered tiles back safely enough,
        // but a `CommitStroke` that never reached the log is a stroke the artist drew
        // and the document does not have.
        let id = self.next_action_id();
        self.commit_with_id(id, ActionKind::CommitStroke(rec));
        // A slot still full after the push was declined; an empty one was taken.
        if offered && !self.shared.apply.reclaim() {
            self.strokes_reused += 1;
        }
    }

    /// Give up the oldest undo steps if history retention is holding more tile
    /// memory than [`DEFAULT_HISTORY_BUDGET`] allows (§5).
    ///
    /// **Half of what is left, not down to a target.** Halving converges in a few
    /// commits and leaves a cushion, where trimming to exactly the budget would fold
    /// the whole stack away the moment one large layer pushed it over — the same
    /// hysteresis argument, and the same arithmetic, as the tile pool's own surplus
    /// policy (`surplus_to_release`).
    ///
    /// **Asked of the pool, answered by the pool.** What is measured is resident tile
    /// bytes, and what a fold releases is the snapshots between here and there: the
    /// handles they pinned drop, their textures return to the free list, and the
    /// pool's epoch boundary hands the surplus back to the driver. So this does not
    /// free memory itself — it stops history being the reason the pool cannot.
    ///
    /// A shared session declines by construction, because
    /// [`Timeline::forget_oldest`] does: its document is re-materialized from the
    /// whole log (§12.2), so nothing there is foldable. Nothing about this call site
    /// has to know that.
    fn trim_history(&mut self) {
        if self.shared.apply.pool.resident_bytes() <= self.history_budget {
            return;
        }
        // How far back undo can currently travel. `None` is a timeline whose history
        // is not this client's alone to walk, which is also one that cannot be
        // trimmed — the same question, so the same answer.
        let Some((applied, _)) = self.timeline.scrub_range() else {
            return;
        };
        let Some(excess) = applied.checked_sub(MIN_UNDO_DEPTH).filter(|e| *e > 0) else {
            return;
        };
        let forgotten = self.timeline.forget_oldest((applied / 2).min(excess));
        if forgotten > 0 {
            tracing::debug!(
                forgotten,
                remaining = applied - forgotten,
                resident_mb = self.shared.apply.pool.resident_bytes() / (1 << 20),
                "gave up undo depth to release retained tiles",
            );
        }
    }

    /// Keep a raw pointer report of the stroke in hand, so a misfit seen in the app
    /// can be dumped on release and replayed as a test
    /// ([`log_debug_samples`](Self::log_debug_samples)).
    ///
    /// A diagnostic, so a shipping build carries neither the samples nor the field
    /// that would hold them: this is `#[cfg]`, not a runtime `cfg!` around a `Vec`
    /// that exists either way. Keeping the capture behind a *call* rather than an
    /// `#[cfg]` block at each site keeps the gesture arms readable, and stops the two
    /// of them disagreeing about the gate — one gated and one not means a shipping
    /// build accumulates the first sample of every stroke and drops the rest.
    #[cfg(feature = "debug-unfrozen")]
    fn note_debug_sample(&mut self, capture: Capture, sample: crate::command::InputSample) {
        if capture == Capture::Restart {
            self.debug_samples.clear();
        }
        self.debug_samples.push(sample);
    }

    #[cfg(not(feature = "debug-unfrozen"))]
    fn note_debug_sample(&mut self, _capture: Capture, _sample: crate::command::InputSample) {}

    /// Point the active layer at something that exists, preferring a paintable one:
    /// a matte may legitimately be selected, but someone who just lost the layer they
    /// were painting on wants to keep painting, not to land on the frame.
    /// Searches the whole tree, not just the root stack: removing a group takes
    /// carried layers with it, and the replacement may itself be carried by
    /// something (§14.2).
    fn repoint_active_layer(&mut self) {
        if self.document().contains_layer(self.session.active_layer) {
            return;
        }
        let (mut paintable, mut any) = (None, None);
        self.document().visit(&mut |l, _| {
            any = any.or(Some(l.id));
            if paintable.is_none() && l.is_paintable() {
                paintable = Some(l.id);
            }
        });
        if let Some(id) = paintable.or(any) {
            self.session.active_layer = id;
        }
    }

    /// The document's color space id (§6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.shared.color_space.id()
    }

    fn next_action_id(&mut self) -> ActionId {
        let id = ActionId {
            lamport: self.authoring.clock,
            actor: self.actor(),
        };
        self.authoring.clock += 1;
        id
    }

    /// Point the brush at `id` if it landed and can take a stroke.
    ///
    /// **Both halves matter**, and there are four callers (`AddLayer`, `PlaceImage`,
    /// `DuplicateLayer`, `MergeLayerDown`) that would otherwise each choose which to
    /// ask: an unknown carrier adds nothing (§14.8), so arming an id no layer has
    /// leaves the next stroke with nowhere to go — and a matte or a filter has no tile
    /// map, so arming one swallows that stroke instead (§15.7, §21.4).
    ///
    /// A no-op otherwise, deliberately: the brush stays where it was, which is a place
    /// that exists, rather than moving somewhere that cannot be painted on.
    fn arm_active(&mut self, id: LayerId) {
        if self.document().layer(id).is_some_and(|l| l.is_paintable()) {
            self.session.active_layer = id;
        }
    }

    /// Commit an action **whose kind names the layers it mints**, and hand back the
    /// id it was given.
    ///
    /// The door [`LayerId`]'s shape asks for: a layer's id is the id of the action
    /// that minted it, so the kind cannot be built until that id exists — and here it
    /// cannot be built any other way. Peeking at the clock and committing separately
    /// would work exactly as long as nothing committed in between, which is a rule a
    /// call site could forget and no test would notice: the ids would name an action
    /// that never happened, and the layers would still be distinct.
    ///
    /// Everything else goes through [`commit`](Self::commit), which is this with the
    /// id thrown away.
    fn commit_minting(&mut self, build: impl FnOnce(ActionId) -> ActionKind) -> ActionId {
        let id = self.next_action_id();
        let kind = build(id);
        // What the door is for, asked of the kind that came back rather than trusted
        // of the closure that built it: `ActionKind::minted_layers` is the exhaustive
        // list of what an action claims to mint, so a variant that grew a layer and
        // forgot to derive its id from `id` is caught here and not by two peers
        // disagreeing about which layer a stroke landed on.
        //
        // **Both halves of the id**, since `k` is the half that does the work for a
        // duplicate: sharing an action id is what makes the ids this action's, and
        // differing in `k` is what makes them each other's.
        let minted: Vec<LayerId> = kind.minted_layers().collect();
        debug_assert!(
            minted.iter().all(|layer| layer.action == id),
            "{} mints a layer id that is not this action's",
            kind.label(),
        );
        debug_assert!(
            minted
                .iter()
                .enumerate()
                .all(|(i, l)| !minted[..i].contains(l)),
            "{} mints one layer id twice",
            kind.label(),
        );
        self.commit_with_id(id, kind);
        id
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
