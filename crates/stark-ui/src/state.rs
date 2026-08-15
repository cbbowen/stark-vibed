//! The app's shared state handle and the command path everything funnels through.
//!
//! [`dispatch`] is the single seam between the UI and the engine: every mutation
//! goes through it, so repaint, observable refresh and collaboration broadcast
//! happen in one place rather than at each call site (§4).
//!
//! "Goes through it" is enforced by the types rather than asked for in prose. The
//! engine and its projection are held in [`ReadOnly`] handles, which have `read` and
//! `peek` and no `write`, so `&mut Renderer` cannot be obtained outside this module
//! at all — the only doors are [`dispatch`] and [`with_engine`], which publish, and
//! [`with_engine_quiet`], which is named for what it declines to do. The failure this
//! rules out: a panel reaches the engine through the signal, changes something the
//! chrome reads back, and leaves the chrome showing the previous value until an
//! unrelated command refreshes it (§4, §7). Both the canvas ground and the lighting
//! environment are reachable that way, and neither spelling compiles.

use dioxus::dioxus_core::{Subscribers, Task};
use dioxus::prelude::*;

use crate::collab;
use crate::prefs::Prefs;
use crate::render::Renderer;
use stark_core::command::ViewCommand;
use stark_core::document::{
    BrushParams, LayerId, PerspectiveMap, TransformMap, WarpMap, rect_corners,
};
use stark_core::geom::{Affine2, Mat2, Vec2};
use stark_core::{InputCommand, ObservableState};

/// Create one of [`AppState`]'s signals, owned by the **root** scope rather than by
/// the component that declares it.
///
/// A `use_signal` belongs to the scope that ran the hook, and reading it from a scope
/// that is not a descendant is what Dioxus warns about — the value could be dropped
/// while still held. Every signal in `AppState` is reached from detached
/// `spawn_forever` tasks (the renderer init, the collaboration pumps), and those live
/// in [`ScopeId::ROOT`], which is *not* a descendant of `app`. Owning them at the root
/// is the fix the warning itself prescribes: move the value to the common ancestor of
/// everything that uses it.
///
/// Nothing about their lifetime actually changes — `app` is the root component and is
/// never unmounted — so this only makes the ownership Dioxus checks match the
/// ownership the design always assumed.
fn root_signal<T: 'static>(init: impl FnOnce() -> T) -> Signal<T> {
    use_hook(|| Signal::new_in_scope(init(), ScopeId::ROOT))
}

/// A signal handed out with its `write` half kept back: `read` and `peek` work
/// exactly as they do on a [`Signal`], and there is no way through it to the value
/// as `&mut`.
///
/// The inner signal is a private field of a type declared in this module, so *this
/// module* is the whole set of code that can mutate what one of these wraps. That
/// is the point: [`AppState::renderer`] and [`AppState::obs`] are the two pieces of
/// state whose mutation has to be paired with a publish, and the pairing is done
/// once here ([`with_engine`]) rather than remembered at two dozen call sites.
///
/// Not `ReadSignal` (dioxus's own read-only wrapper), for one reason: `ReadSignal`
/// boxes the readable into a `CopyValue` owned by whatever scope constructs it, and
/// these are built in `app`'s scope while being read from the `spawn_forever` tasks
/// that live in `ScopeId::ROOT` — the ownership mismatch `root_signal` exists to
/// avoid. Wrapping the root-owned `Signal` keeps its ownership and adds nothing to
/// the read path.
pub struct ReadOnly<T: 'static>(Signal<T>);

// `Signal` is `Copy` whatever it holds, so these are hand-written: a derive would
// demand `T: Copy` and none of what this wraps is.
impl<T: 'static> Clone for ReadOnly<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: 'static> Copy for ReadOnly<T> {}

impl<T: 'static> Readable for ReadOnly<T> {
    type Target = T;
    type Storage = UnsyncStorage;

    fn try_read_unchecked(&self) -> Result<ReadableRef<'static, Self>, BorrowError> {
        self.0.try_read_unchecked()
    }

    fn try_peek_unchecked(&self) -> Result<ReadableRef<'static, Self>, BorrowError> {
        self.0.try_peek_unchecked()
    }

    fn subscribers(&self) -> Subscribers {
        self.0.subscribers()
    }
}

/// Shared `Copy` handle to the app's signals. Provided once by `app` and read back
/// through `use_context` wherever a component needs to reach the engine.
///
/// Built by [`AppState::new`] rather than field-by-field at the call site, so every
/// signal here goes through [`root_signal`] — the one property that has to hold for
/// all of them, and the kind of thing a hand-written literal drifts out of the moment
/// a field is added.
#[derive(Clone, Copy)]
pub struct AppState {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    ///
    /// Read freely; to *move* engine state, call [`dispatch`] or [`with_engine`]
    /// (see the module note on [`ReadOnly`]).
    pub renderer: ReadOnly<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command — by
    /// [`with_engine`], which is the only thing that can write it.
    pub obs: ReadOnly<Option<ObservableState>>,
    /// Whether the user is holding space.
    pub space_down: Signal<bool>,
    /// Whether a canvas gesture is in flight (a stroke, a selection drag, a pan,
    /// or a run of wheel zooming). The floating chrome fades out while it is set,
    /// handing the screen back to the painting — see
    /// [`chrome_class`](crate::layout::chrome_class).
    pub canvas_active: Signal<bool>,
    /// Whether the brush editor dialog is open (rendered at the app root so its
    /// backdrop escapes the panels' `backdrop-filter` containing blocks).
    pub brush_editor_open: Signal<bool>,
    /// Whether the "Save preset" dialog is open. At the app root for the same reason
    /// as the editor above: a panel is a `backdrop-filter` containing block, so a
    /// dialog rendered inside one would be trapped in its box rather than covering
    /// the window.
    pub preset_save_open: Signal<bool>,
    /// Bumped whenever the brush color is set from **outside** the color picker —
    /// today only by the eyedropper.
    ///
    /// The Color panel's picker holds its own Oklab state (out-of-gamut `a`/`b`
    /// survive there but not in the clamped sRGB it reports), so it is *seeded*
    /// rather than driven, and this is how it is told to re-seed. Deliberately not
    /// "the color changed": a pick inside the field would then drag its own marker
    /// back onto the gamut boundary under the user's cursor.
    pub color_epoch: Signal<u64>,
    /// The eyedropper (§18.0.2).
    pub pick: PickState,
    /// The size ring a brush-tuning drag draws over the canvas (§18.1.9), or `None`
    /// when no size drag is in flight.
    ///
    /// Shared state rather than the gesture's own, for [`PickState::dragging`]'s
    /// reason: the drag is the canvas's, but the ring is a sibling overlay of it — a
    /// `<div>` cannot be drawn from inside a `<canvas>`'s handler without somewhere
    /// for both to read.
    pub brush_ring: Signal<Option<BrushRing>>,
    /// The live brush's stroke-smoothing amount, 0..=1 (§6.11). Frontend state
    /// beside the engine's own `BrushParams`, never inside them: the stored
    /// path already embodies the smoothing, so a field there would be one that
    /// replay reads and ignores. It travels with every whole-brush snapshot
    /// through [`presets::Wearable`](crate::presets::Wearable).
    pub smoothing: Signal<f32>,
    /// The tow string on screen while a smoothing brush draws (§6.11), in the
    /// canvas element's own px — `None` when there is nothing to show. Its own
    /// signal for the reason [`brush_ring`](Self::brush_ring) is: only the
    /// little overlay re-renders at pointer rate, never the chrome.
    pub tow: Signal<Option<TowUi>>,
    /// The drag-and-hold drawing assist (§6.9; `crate::input`).
    pub assist: AssistState,
    /// Minimal mode: the chrome over the canvas drops its words and keeps its marks
    /// (§11).
    ///
    /// A client preference like [`AssistState::enabled`], not document state — it is a
    /// way of *looking* at the application, so it never enters the document and is never
    /// replicated, and a collaborator's copy is theirs to set. It does follow *this*
    /// browser between visits (`crate::prefs`).
    ///
    /// Nothing reads this but the app root, which turns it into a class on `.app-root`;
    /// what the class hides is decided in the stylesheet against the `.label` spans the
    /// components emit. Deliberately not a prop threaded through the tree: every
    /// control would then have to be handed a boolean it does nothing with but pass on,
    /// and a control that forgot to would be the one that kept its word.
    pub minimal: Signal<bool>,
    /// The transform gesture in flight (§16.6, §16.8, §16.9): `Some` while the
    /// user is composing a move/scale/flip, a perspective or a warp of the
    /// selected paint. View state — the engine sees only the previews it
    /// produces and the one commit on "Done".
    pub transform: Signal<Option<TransformUi>>,
    /// The gradient bar's gesture in flight (§22.4): `Some` while a ramp's
    /// axis is being composed — over the selection for a fill, over a matte
    /// for its paint. View state on [`transform`]'s model — the engine sees
    /// only previews and the one commit on "Done".
    pub gradient_bar: Signal<Option<GradientUi>>,
    /// The drawing-guide edit mode (§20.5): `Some` while a guide is selected
    /// for composing — the canvas drags the camera, and the Perspective Guide
    /// bar stands in at the bottom. View state through and through: the engine
    /// sees only the `SetGuides` lists the gestures produce.
    pub guide_edit: Signal<Option<GuideEdit>>,
    /// Whether a [`request_paint`] is already waiting on the next animation frame.
    /// The latch that turns any number of paint requests into one paint per frame.
    /// Read and written only from non-component code (`peek`/`set`), so no
    /// component ever subscribes to it.
    pub paint_queued: Signal<bool>,
    /// Everything to do with a shared drawing (§12).
    pub collab: CollabState,
    /// Timeline mode: scrubbing and playing back the history
    /// (§18.2.4; `crate::panels::timeline`).
    pub timeline: TimelineState,
    /// The custom brush-shape library (§6.6; `crate::shapes`).
    pub shapes: ShapesState,
    /// The brush preset library (`crate::presets`), loaded from `localStorage`
    /// at startup like the shape library.
    pub presets: Signal<Vec<crate::presets::PresetEntry>>,
    /// Rendered preset thumbnails and the offscreen rig that generates them
    /// (`crate::thumbs`).
    pub thumbs: crate::thumbs::ThumbState,
    /// The gradient library and its trace mode (§22; `crate::gradients`),
    /// loaded from `localStorage` at startup like the libraries above.
    pub gradients: crate::gradients::GradientsState,
    /// The ten brushes under the hand (§18.1.8; `crate::slots`).
    pub slots: SlotState,
}

/// The quick-brush rack's signals (§18.1.8), grouped because they are one
/// feature's worth of state and because the second is meaningless without the
/// first: a hold names a slot in the rack.
///
/// Root-owned like everything here, and it has to be — the rack is read from the
/// window's own key handlers (`crate::input::bind_shortcuts`), which are bound
/// once for the life of the page and belong to no component's scope.
#[derive(Clone, Copy)]
pub struct SlotState {
    /// What each digit holds; `None` for a slot nobody has filled. Loaded from
    /// `localStorage` at startup like the shape and preset libraries.
    pub brushes: Signal<crate::slots::Rack>,
    /// The hold in flight — `Some` for exactly as long as a number key is down
    /// or the pen's eraser end is on the glass. The rack's overlay is mounted on
    /// this (`slots::SlotOverlay`), and the release reads the brushes it has to
    /// restore from it.
    pub held: Signal<Option<crate::slots::Held>>,
    /// Whether the rack is kept open with no key held — the Panels menu's
    /// "Quick brushes" (§18.1.8). What it buys is a rack that can be *clicked*,
    /// which is the only way to a slot for a hand with no keyboard under it.
    ///
    /// Not persisted, like the panel stack's own visibility (`layout::PanelLayout`
    /// — which this is not part of, the rack being chrome of its own down the
    /// left rather than a panel in the stack): the menu says what is on screen
    /// now, and none of its entries claims to say what will be next time.
    pub pinned: Signal<bool>,
}

/// A drawing guide selected for composing (§20.5): which entry of the
/// engine's guide list the mode edits, and the per-axis locks constraining
/// the canvas drag.
///
/// The locks live here rather than on the guide because they are *gesture*
/// state — a constraint on the hand for the duration of the mode, not a fact
/// about the guide worth keeping (or, later, saving). Leaving the mode
/// releases them.
#[derive(Clone, Copy, PartialEq)]
pub struct GuideEdit {
    /// Index into the engine's guide list ([`ObservableState::guides`]).
    pub index: usize,
    /// World axes held fixed under the orbit drag: one lock constrains the
    /// drag to turning about that axis, two pin the frame entirely
    /// ([`PerspectiveGuide::dragged`](stark_core::PerspectiveGuide::dragged)).
    pub locked: [bool; 3],
}

/// The drag-and-hold drawing assist's signals (§6.9).
///
/// The engine owns what a hold *means* ([`GestureCommand::Hold`](stark_core::command::GestureCommand));
/// what lives here is the other half — noticing that the pointer has stopped, which is
/// gesture feel and needs a clock the engine deliberately does not have (§7).
#[derive(Clone, Copy)]
pub struct AssistState {
    /// Whether a held pointer snaps the stroke at all. On by default, and off is a
    /// real answer: assist changes what an ordinary stroke does, so somebody who wants
    /// their line left crooked has to be able to say so (`crate::settings`).
    pub enabled: Signal<bool>,
    /// The dwell being watched — `Some` for exactly as long as a paint gesture is in
    /// flight, and clearing it is what stops the watcher below.
    pub dwell: Signal<Option<Dwell>>,
    /// The watcher counting it down. Cancelled and replaced per gesture, like the
    /// collaboration pumps.
    pub task: Signal<Option<Task>>,
}

/// A pointer that has stopped moving, as the assist watcher reads it.
#[derive(Clone, Copy, PartialEq)]
pub struct Dwell {
    /// Where it stopped, in **element (CSS) pixels**.
    ///
    /// Screen space and not canvas space, because holding still is a fact about the
    /// hand: measured on the canvas, the same tremor would count as a hold at one zoom
    /// level and as movement at another, and the threshold would mean something
    /// different every time the view changed.
    pub at: Vec2,
    /// When it got there, on the monotonic clock (`crate::collab::now_seconds`).
    pub since: f64,
    /// Whether this dwell has already been reported.
    ///
    /// A latch rather than an exit, because a gesture may earn several holds: the
    /// pause a pen makes on touching down before the hand starts moving is one, and
    /// giving up there would disarm the assist for the stroke that followed it. So the
    /// watcher latches, and movement past the slop is what clears it.
    pub fired: bool,
}

/// What a brush-tuning drag's size indicator draws (§18.1.9): the size being asked
/// for, and the size the drag started from, about the point it pressed on.
///
/// **Screen px, not canvas px** — deliberately a drawing instruction rather than a
/// statement about the brush. The gesture holds the one zoom it measures against
/// (`input::TuneDrag::zoom`), so converting there means the ring and the radius it
/// reports cannot be scaled by two different numbers; and it leaves the overlay pure
/// layout, with no view to read and nothing to re-render it when the engine writes.
///
/// A circle in canvas space is still a circle on screen at any angle or handedness, so
/// a radius through the zoom is the whole of the transform: this needs no matrix, which
/// is the one thing that makes a `<div>` a fair way to draw it.
#[derive(Copy, Clone, PartialEq)]
pub struct BrushRing {
    /// The press position in page px — where the ring is centred, and the one point a
    /// gesture agrees on however far it has wandered since.
    pub at: Vec2,
    /// The radius the brush had when the drag began, screen px. The reference: without
    /// it the ring says how big the brush is about to be and nothing about whether that
    /// is bigger or smaller than what the last stroke was made with.
    pub was: f32,
    /// The radius being asked for now, screen px.
    pub now: f32,
}

/// The tow string as the overlay draws it (§6.11): the towed tip, the pointer,
/// and the rope, all in the canvas element's own px.
///
/// **Screen px, not canvas px**, on [`BrushRing`]'s own argument: a drawing
/// instruction rather than a statement about the stroke. The conversion happens
/// where the engine is read, against the one view the gesture holds — a stroke
/// cannot outlive its view, since a pinch cancels it — and the overlay stays
/// pure layout.
#[derive(Copy, Clone, PartialEq)]
pub struct TowUi {
    /// Where the mark is being laid — the towed tip.
    pub tip: Vec2,
    /// Where the hand is — the pointer the string runs to.
    pub target: Vec2,
    /// The string's length. `|target − tip|` short of it is slack, and the
    /// overlay shows the difference as sag.
    pub rope: f32,
}

/// The custom brush-shape library's signals (`crate::shapes`). Root-owned:
/// imports are started from the brush editor's modal scope but must survive
/// its close.
#[derive(Clone, Copy)]
pub struct ShapesState {
    /// Library entries, loaded from `localStorage` at startup.
    pub entries: Signal<Vec<crate::shapes::ShapeEntry>>,
    /// A transient line under the shape gallery: import errors, or the
    /// "inverted a dark-on-light image" explanation. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

impl AppState {
    /// Build the app's state. Call once, from the root component.
    pub fn new() -> Self {
        Self {
            renderer: ReadOnly(root_signal(|| None)),
            obs: ReadOnly(root_signal(|| None)),
            space_down: root_signal(|| false),
            canvas_active: root_signal(|| false),
            brush_editor_open: root_signal(|| false),
            preset_save_open: root_signal(|| false),
            color_epoch: root_signal(|| 0),
            pick: PickState {
                scope: root_signal(PickScope::default),
                radius: root_signal(|| 0),
                busy: root_signal(|| false),
                alt_down: root_signal(|| false),
                dragging: root_signal(|| false),
            },
            brush_ring: root_signal(|| None),
            smoothing: root_signal(|| 0.0),
            tow: root_signal(|| None),
            assist: AssistState {
                enabled: root_signal(|| Prefs::default().assist),
                dwell: root_signal(|| None),
                task: root_signal(|| None),
            },
            // Seeded from the preference defaults rather than written out again here:
            // `prefs::load` overwrites both of these at app start, so a default stated
            // in this file would be the one that never applies (`crate::prefs`).
            minimal: root_signal(|| Prefs::default().minimal),
            transform: root_signal(|| None),
            gradient_bar: root_signal(|| None),
            guide_edit: root_signal(|| None),
            paint_queued: root_signal(|| false),
            collab: CollabState {
                session: root_signal(|| None),
                ticket: root_signal(|| None),
                phase: root_signal(collab::CollabPhase::default),
                error: root_signal(|| None),
                peers: root_signal(Vec::new),
                links: root_signal(Vec::new),
                pump: root_signal(|| None),
                presence: root_signal(|| None),
            },
            timeline: TimelineState {
                open: root_signal(|| false),
                playing: root_signal(|| false),
                speed: root_signal(|| 1.0),
                task: root_signal(|| None),
            },
            shapes: ShapesState {
                entries: root_signal(Vec::new),
                notice: root_signal(|| None),
            },
            presets: root_signal(Vec::new),
            thumbs: crate::thumbs::ThumbState {
                cache: root_signal(std::collections::HashMap::new),
                rig: root_signal(|| None),
                busy: root_signal(|| false),
            },
            gradients: crate::gradients::GradientsState {
                entries: root_signal(Vec::new),
                selected: root_signal(|| None),
                armed: root_signal(|| false),
                busy: root_signal(|| false),
                notice: root_signal(|| None),
            },
            slots: SlotState {
                brushes: root_signal(|| [None; crate::slots::COUNT]),
                held: root_signal(|| None),
                pinned: root_signal(|| false),
            },
        }
    }
}

/// The gradient gesture being composed on the shared gradient bar (§22.4):
/// what the ramp lands on, how the composing drag is read, and the drag itself.
///
/// The drag is kept as its two raw points and the axis **derived** per kind —
/// linear reads them as from→to, radial as centre and reach — so switching
/// kinds on the bar reinterprets the drag the hand already made instead of
/// throwing it away.
#[derive(Clone, PartialEq)]
pub struct GradientUi {
    /// What "Done" commits — a fill of the selection, or a matte's paint. One
    /// bar, one catcher, two targets: the interface for laying a ramp is the
    /// same wherever the ramp lands.
    pub target: GradientTarget,
    /// How the drag becomes an axis — the bar's Linear/Radial chips.
    pub kind: GradientAxisKind,
    /// The composing drag, canvas space: anchor and current end. `None` until
    /// the first drag — there is no axis to preview yet.
    pub drag: Option<(Vec2, Vec2)>,
}

/// What the gradient bar's ramp lands on (§22.4).
#[derive(Clone, PartialEq)]
pub enum GradientTarget {
    /// Fill the selection on `layer`. The ramp is read live from the library
    /// (the panel's highlighted row), so a click there re-previews.
    ///
    /// It lays opaque paint through the mask, so how strongly it lands is the
    /// selection's business and there is nothing here to capture (§6.8).
    Fill { layer: LayerId },
    /// Repaint the matte `layer` (§15.4). The ramp rides the target — seeded
    /// from the matte's own paint, so re-composing an old gradient's axis does
    /// not silently swap its colors for the library's current row; a library
    /// click replaces it deliberately.
    Matte {
        layer: LayerId,
        gradient: stark_core::Gradient,
    },
}

/// How a gradient drag is read as an axis (§22.4).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GradientAxisKind {
    Linear,
    Radial,
}

impl GradientUi {
    /// The axis the current drag composes, or `None` before the first drag.
    pub fn axis(&self) -> Option<stark_core::document::GradientAxis> {
        let (from, to) = self.drag?;
        Some(match self.kind {
            GradientAxisKind::Linear => stark_core::document::GradientAxis::Linear { from, to },
            GradientAxisKind::Radial => stark_core::document::GradientAxis::Radial {
                center: from,
                radius: from.distance(to),
            },
        })
    }
}

/// Where a pointer stands relative to the transform widget's ellipse — which
/// decides what a drag starting there does (§16.6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransformRegion {
    /// Strictly inside: dragging translates.
    Inside,
    /// On the rim: dragging turns and scales uniformly — tangential motion is
    /// pure rotation, radial motion pure scale, anything between blends the two.
    Rim,
    /// Outside: dragging stretches and shears along the grab direction, pinning
    /// the perpendicular diameter.
    Outside,
}

/// The transform gesture being composed (§16.6). The widget is
/// an **ellipse**: the image of a reference **circle** under the accumulated
/// linear map, so the widget's shape *is* the transform — it stays a circle
/// exactly as long as the transform is a similarity, and any distortion shows
/// as eccentricity. The affine the engine sees is derived on every change —
/// `x ↦ center + linear·(x − anchor)` — so a long drag is one accumulated
/// transform, never a chain of them, and the preview resamples the committed
/// tiles exactly once ("lossless" until "Done").
///
/// Every shaping gesture left-composes a world-space factor onto `linear`, each
/// solved so that **the grabbed point follows the pointer** within its family:
/// a similarity for the rim, a rank-1 stretch/shear for the outside. Gestures
/// that were never used leave their factors out entirely — a pure move keeps
/// `linear` bit-exactly the identity, which is what keeps it a pure translation
/// through the engine's exactness invariants (§16.4).
#[derive(Clone, Copy, PartialEq)]
pub struct TransformState {
    /// The layer whose selected paint is being transformed.
    pub layer: LayerId,
    /// The reference ellipse's centre — the hull's — in canvas px. Fixed for the
    /// mode's life; the affine pivots here.
    pub anchor: Vec2,
    /// The reference **circle**'s radius, canvas px. A circle, not the hull's
    /// own aspect: the widget's shape carries meaning — a circle says the
    /// accumulated transform is a similarity (rotation, uniform scale,
    /// translation), and any other shape says distortion has been applied.
    /// Encompasses the hull; floored so a hairline selection still mounts a
    /// grabbable widget.
    pub radius: f32,
    /// Where the gesture has carried the centre.
    pub center: Vec2,
    /// The accumulated linear map, applied about the centre.
    pub linear: Mat2,
}

impl TransformState {
    pub fn begin(layer: LayerId, hull: (Vec2, Vec2), min_radius: f32) -> Self {
        let anchor = (hull.0 + hull.1) * 0.5;
        let half = (hull.1 - hull.0) * 0.5;
        Self {
            layer,
            anchor,
            radius: half.max(Vec2::ZERO).length().max(min_radius),
            center: anchor,
            linear: Mat2::IDENTITY,
        }
    }

    /// Whether committing would change nothing — "Done" then skips the commit
    /// rather than spending an undo step on a no-op.
    pub fn is_identity(&self) -> bool {
        self.center == self.anchor && self.linear == Mat2::IDENTITY
    }

    /// The affine this gesture stands for — what the preview shows and "Done"
    /// commits.
    pub fn affine(&self) -> Affine2 {
        if self.linear == Mat2::IDENTITY {
            // The untouched-linear case stays a *pure* translation, not a
            // translation reconstituted through matrix arithmetic.
            return Affine2::from_translation(self.center - self.anchor);
        }
        Affine2::from_mat2_translation(self.linear, self.center - self.linear * self.anchor)
    }

    /// Classify a canvas-space pointer against the widget
    /// (§16.6): pull it back through the linear map into the reference circle's own
    /// space, where the test is a radius. `band` is the rim's grab half-width in
    /// canvas px, converted to circle units by the widget's local radius along
    /// the pointer's direction.
    pub fn region(&self, pointer: Vec2, band: f32) -> TransformRegion {
        let det = self.linear.determinant();
        if det.abs() < 1e-6 {
            // Collapsed to a sliver: everything reads as inside, so the widget
            // can still be moved (the shaping clamps keep this unreachable in
            // practice).
            return TransformRegion::Inside;
        }
        let u = (self.linear.inverse() * (pointer - self.center)) / self.radius;
        let rho = u.length();
        if rho < 1e-6 {
            return TransformRegion::Inside;
        }
        let local_radius = (self.linear * (self.radius * (u / rho))).length();
        let band = band / local_radius.max(1e-3);
        if rho < 1.0 - band {
            TransformRegion::Inside
        } else if rho <= 1.0 + band {
            TransformRegion::Rim
        } else {
            TransformRegion::Outside
        }
    }

    /// An inside drag: translate. `eps` (canvas px) snaps a jiggle back to the
    /// start, so touching the widget without meaning to never resamples.
    pub fn translated(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        Self {
            center: self.center + (to - from),
            ..self
        }
    }

    /// A rim drag: the similarity (rotation + uniform scale about the centre)
    /// that carries the grabbed point `from` exactly to the pointer `to` — the
    /// complex ratio `(to − c)/(from − c)`. Tangential motion is thereby pure
    /// rotation and radial motion pure scale, with no mode to pick.
    pub fn turned_scaled(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        let v0 = from - self.center;
        let v = to - self.center;
        let n = v0.length_squared();
        if n < 1e-6 {
            return self;
        }
        // Keep the widget grabbable: never scale below 5% in one gesture.
        let v = clamp_len(v, 0.05 * n.sqrt());
        let (a, b) = (v.dot(v0) / n, v0.perp_dot(v) / n);
        Self {
            linear: Mat2::from_cols(Vec2::new(a, b), Vec2::new(-b, a)) * self.linear,
            ..self
        }
    }

    /// An outside drag: the rank-1 update `I + (Δ ⊗ d̂)/λ` that carries the
    /// grabbed point exactly to the pointer while **pinning the diameter
    /// perpendicular to the grab** — radial pull scales along the grab
    /// direction, tangential drag shears, and everything on the pinned axis
    /// stays put, which is what makes the gesture predictable.
    pub fn stretched(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        let v0 = from - self.center;
        let lambda = v0.length();
        if lambda < 1e-3 {
            return self;
        }
        let dir = v0 / lambda;
        let mut delta = to - from;
        // Pulling in past the pinned axis would run the determinant through
        // zero (the paint would vanish into a line, and the engine would refuse
        // the commit); floor the radial component at 90% pulled-in.
        let radial = delta.dot(dir) / lambda;
        if radial < -0.9 {
            delta += dir * ((-0.9 - radial) * lambda);
        }
        let g = Mat2::from_cols(
            Vec2::new(1.0 + delta.x * dir.x / lambda, delta.y * dir.x / lambda),
            Vec2::new(delta.x * dir.y / lambda, 1.0 + delta.y * dir.y / lambda),
        );
        Self {
            linear: g * self.linear,
            ..self
        }
    }

    /// Mirror left↔right, about the vertical axis through the centre.
    pub fn flipped_h(self) -> Self {
        Self {
            linear: Mat2::from_diagonal(Vec2::new(-1.0, 1.0)) * self.linear,
            ..self
        }
    }

    /// Mirror top↕bottom, about the horizontal axis through the centre.
    pub fn flipped_v(self) -> Self {
        Self {
            linear: Mat2::from_diagonal(Vec2::new(1.0, -1.0)) * self.linear,
            ..self
        }
    }
}

/// `v`, no shorter than `min` (direction kept; zero stays zero).
fn clamp_len(v: Vec2, min: f32) -> Vec2 {
    let len = v.length();
    if len < min && len > 1e-9 {
        v * (min / len)
    } else {
        v
    }
}

/// The transform mode's whole in-flight state (§16.6, §16.8, §16.9): which of
/// the three families the bar has selected, with that family's own gesture
/// state. One value in one signal, because the mode is *modal* — there is
/// always exactly one family composing, and switching families is an explicit
/// act on the bar (which carries the deformation along when the new family
/// contains the old one exactly, and commits it first when it cannot).
#[derive(Clone, Copy, PartialEq)]
pub enum TransformUi {
    /// The ellipse widget over the whole affine group — `rect` is the hull the
    /// mode was entered around, kept so a switch to a rect-scoped family knows
    /// its source rect.
    Affine {
        rect: (Vec2, Vec2),
        ts: TransformState,
    },
    Perspective(PerspectiveUi),
    Warp(WarpUi),
}

impl TransformUi {
    pub fn layer(&self) -> LayerId {
        match self {
            TransformUi::Affine { ts, .. } => ts.layer,
            TransformUi::Perspective(p) => p.layer,
            TransformUi::Warp(w) => w.layer,
        }
    }

    /// The map this gesture stands for — what the preview shows and "Done"
    /// commits.
    pub fn map(&self) -> TransformMap {
        match self {
            TransformUi::Affine { ts, .. } => TransformMap::Affine(ts.affine()),
            TransformUi::Perspective(p) => TransformMap::Perspective(p.map()),
            TransformUi::Warp(w) => TransformMap::Warp(w.map()),
        }
    }

    /// Whether committing would change nothing — "Done" then skips the commit
    /// rather than spending an undo step on a no-op.
    pub fn is_identity(&self) -> bool {
        match self {
            TransformUi::Affine { ts, .. } => ts.is_identity(),
            TransformUi::Perspective(p) => p.is_identity(),
            TransformUi::Warp(w) => w.is_identity(),
        }
    }
}

/// Where a pointer stands relative to the perspective quad (§16.8) — which
/// decides what a drag starting there does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuadRegion {
    /// On a corner handle: dragging carries that corner exactly, pinning the
    /// other three.
    Corner(usize),
    /// On an edge (named by its two corner indices): dragging shifts the whole
    /// edge — the foreshortening gesture.
    Edge(usize, usize),
    /// Inside the quad, or anywhere else: dragging translates all four
    /// corners together.
    Inside,
    Outside,
}

/// The perspective gesture being composed (§16.8): the image of the source
/// rect is a quad, its corners are the handles, and **the grabbed corner
/// follows the pointer exactly** — the map is defined as "the homography
/// putting the corners where the hand put them", so the widget cannot disagree
/// with the paint. A drag that would cross the quad (a concave or reflected
/// configuration — the map's horizon) holds at the last valid shape rather
/// than letting the paint fly through infinity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PerspectiveUi {
    pub layer: LayerId,
    /// The source rect the map acts on — the hull the mode was entered around.
    pub rect: (Vec2, Vec2),
    /// The corner images, in [`rect_corners`] order (00, 10, 01, 11).
    pub corners: [Vec2; 4],
}

impl PerspectiveUi {
    pub fn begin(layer: LayerId, rect: (Vec2, Vec2)) -> Self {
        Self {
            layer,
            rect,
            // The exact base values — identity is "the corners *are* the
            // rect's", bitwise (§16.4).
            corners: rect_corners(rect.0, rect.1),
        }
    }

    pub fn map(&self) -> PerspectiveMap {
        PerspectiveMap {
            min: self.rect.0,
            max: self.rect.1,
            corners: self.corners,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.corners == rect_corners(self.rect.0, self.rect.1)
    }

    /// Classify a canvas-space pointer: corner handles win, then edges, then
    /// the quad's inside. `grab` and `band` are canvas-px radii (screen px
    /// over the zoom).
    pub fn region(&self, p: Vec2, grab: f32, band: f32) -> QuadRegion {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.corners.iter().enumerate() {
            let d = p.distance(*c);
            if d <= grab && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        if let Some((i, _)) = best {
            return QuadRegion::Corner(i);
        }
        for (a, b) in EDGES {
            if segment_distance(p, self.corners[a], self.corners[b]) <= band {
                return QuadRegion::Edge(a, b);
            }
        }
        if point_in_quad(&self.corners, p) {
            QuadRegion::Inside
        } else {
            QuadRegion::Outside
        }
    }

    /// A corner drag, recomputed from the drag's start: the corner follows the
    /// pointer exactly while the shape stays convex; a pull past validity
    /// holds at `current` (the last valid shape) instead of tearing through
    /// the horizon. `eps` snaps a jiggle back to the start (§16.6).
    pub fn corner_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.corners[i] = start.corners[i] + delta;
        if next.map().usable() { next } else { current }
    }

    /// An edge drag: both of its corners follow together.
    pub fn edge_dragged(
        start: Self,
        current: Self,
        (a, b): (usize, usize),
        delta: Vec2,
        eps: f32,
    ) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.corners[a] = start.corners[a] + delta;
        next.corners[b] = start.corners[b] + delta;
        if next.map().usable() { next } else { current }
    }

    /// An inside (or outside) drag: the whole quad translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for c in &mut next.corners {
            *c += delta;
        }
        if next.map().usable() { next } else { current }
    }
}

/// The quad's edges as corner-index pairs, walking the boundary
/// (corner order is 00, 10, 01, 11, so the boundary is 0 → 1 → 3 → 2).
const EDGES: [(usize, usize); 4] = [(0, 1), (1, 3), (3, 2), (2, 0)];

fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Point-in-convex-quad, corners in (00, 10, 01, 11) order. Same-side test
/// against every boundary edge; orientation-agnostic so it also serves a
/// mid-drag shape the validity clamp has not yet vetoed.
fn point_in_quad(c: &[Vec2; 4], p: Vec2) -> bool {
    let b = [c[0], c[1], c[3], c[2]];
    let mut sign = 0.0f32;
    for i in 0..4 {
        let cross = (b[(i + 1) % 4] - b[i]).perp_dot(p - b[i]);
        if cross.abs() < 1e-9 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// Control points per axis of the warp gesture's mesh. 4×4 spans "gentle bend"
/// to "full puppet" through the smooth interpolation; the engine accepts up to
/// [`stark_core::document::MAX_WARP_GRID`] if a denser UI ever wants one.
pub const WARP_GRID: usize = 4;

/// Where a pointer stands relative to the warp mesh (§16.9).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshRegion {
    /// On a control point: dragging carries that point exactly.
    Point(usize),
    /// On the surface between points: dragging grabs the *paint* — the
    /// control points share the motion so the grabbed surface point follows
    /// the pointer exactly.
    Inside,
    /// Anywhere else: dragging translates the whole mesh.
    Outside,
}

/// The warp gesture being composed (§16.9): a 4×4 control grid over the source
/// rect, smoothly interpolated by the engine's own surface — the mesh the
/// overlay draws is sampled from the very lattice the paint resamples through,
/// so the curves *are* the deformation. Two ways to shape it, both
/// exact-follow: drag a control point, or grab the surface anywhere and the
/// least-norm control move puts that spot of paint under the pointer
/// (`Δpᵢ = Bᵢ·Δ / ΣB²`, with `B` the surface basis at the grab). A drag that
/// would fold the mesh holds at the last valid shape.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WarpUi {
    pub layer: LayerId,
    pub rect: (Vec2, Vec2),
    /// Row-major control points — the images of the rect's uniform grid.
    pub points: [Vec2; WARP_GRID * WARP_GRID],
}

impl WarpUi {
    pub fn begin(layer: LayerId, rect: (Vec2, Vec2)) -> Self {
        // The engine's own base points, not a re-derivation: identity is "the
        // points *are* these values", bitwise (§16.4).
        let base = WarpMap::identity(rect.0, rect.1, WARP_GRID as u32, WARP_GRID as u32);
        let mut points = [Vec2::ZERO; WARP_GRID * WARP_GRID];
        points.copy_from_slice(&base.points);
        Self {
            layer,
            rect,
            points,
        }
    }

    pub fn map(&self) -> WarpMap {
        WarpMap {
            min: self.rect.0,
            max: self.rect.1,
            cols: WARP_GRID as u32,
            rows: WARP_GRID as u32,
            points: self.points.to_vec(),
        }
    }

    pub fn is_identity(&self) -> bool {
        let base = WarpMap::identity(self.rect.0, self.rect.1, WARP_GRID as u32, WARP_GRID as u32);
        self.points[..] == base.points[..]
    }

    /// Classify a canvas-space pointer: the nearest control point within
    /// `grab` wins; otherwise inside the mesh's boundary polygon is a surface
    /// grab; otherwise a whole-mesh translate.
    pub fn region(&self, p: Vec2, grab: f32) -> MeshRegion {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.points.iter().enumerate() {
            let d = p.distance(*c);
            if d <= grab && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        if let Some((i, _)) = best {
            return MeshRegion::Point(i);
        }
        if point_in_polygon(&self.boundary(), p) {
            MeshRegion::Inside
        } else {
            MeshRegion::Outside
        }
    }

    /// The mesh's border control points, walking the boundary clockwise.
    fn boundary(&self) -> Vec<Vec2> {
        let n = WARP_GRID;
        let at = |i: usize, j: usize| self.points[j * n + i];
        let mut out = Vec::with_capacity(4 * (n - 1));
        for i in 0..n - 1 {
            out.push(at(i, 0));
        }
        for j in 0..n - 1 {
            out.push(at(n - 1, j));
        }
        for i in (1..n).rev() {
            out.push(at(i, n - 1));
        }
        for j in (1..n).rev() {
            out.push(at(0, j));
        }
        out
    }

    /// The grid fraction whose surface point is nearest `p`, with the surface
    /// basis there and its squared norm — everything a surface drag needs,
    /// computed once at the press. Coarse scan plus local refinement; the
    /// surface is smooth and unfolded, so nearest-on-a-grid converges fast.
    pub fn grab(&self, p: Vec2) -> (Vec2, [f32; WARP_GRID * WARP_GRID], f32) {
        let map = self.map();
        let mut best = (Vec2::splat(0.5), f32::INFINITY);
        let scan = |from: Vec2, step: f32, best: &mut (Vec2, f32)| {
            for j in -2..=2i32 {
                for i in -2..=2i32 {
                    let t =
                        (from + Vec2::new(i as f32, j as f32) * step).clamp(Vec2::ZERO, Vec2::ONE);
                    let d = map.eval(t).distance_squared(p);
                    if d < best.1 {
                        *best = (t, d);
                    }
                }
            }
        };
        for j in 0..=8 {
            for i in 0..=8 {
                let t = Vec2::new(i as f32 / 8.0, j as f32 / 8.0);
                let d = map.eval(t).distance_squared(p);
                if d < best.1 {
                    best = (t, d);
                }
            }
        }
        let mut step = 1.0 / 16.0;
        for _ in 0..6 {
            let from = best.0;
            scan(from, step, &mut best);
            step *= 0.5;
        }
        let basis = map.basis(best.0);
        let mut b = [0.0f32; WARP_GRID * WARP_GRID];
        b.copy_from_slice(&basis);
        let norm: f32 = b.iter().map(|w| w * w).sum();
        (best.0, b, norm.max(1e-6))
    }

    /// A control-point drag, recomputed from the drag's start; a fold holds at
    /// `current`, the last valid shape.
    pub fn point_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.points[i] = start.points[i] + delta;
        if next.map().usable() { next } else { current }
    }

    /// A surface drag: the least-norm control move that carries the grabbed
    /// surface point exactly with the pointer — the hand holds the paint, not
    /// a handle (§16.9). `basis`/`norm` come from [`grab`](Self::grab) at the
    /// press.
    pub fn surface_dragged(
        start: Self,
        current: Self,
        basis: &[f32; WARP_GRID * WARP_GRID],
        norm: f32,
        delta: Vec2,
        eps: f32,
    ) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for (pt, w) in next.points.iter_mut().zip(basis) {
            *pt += delta * (*w / norm);
        }
        if next.map().usable() { next } else { current }
    }

    /// An outside drag: the whole mesh translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for pt in &mut next.points {
            *pt += delta;
        }
        if next.map().usable() { next } else { current }
    }
}

/// Even-odd point-in-polygon over an arbitrary boundary walk.
fn point_in_polygon(poly: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    fn state() -> TransformState {
        TransformState::begin(
            LayerId(0),
            (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0)),
            10.0,
        )
    }

    #[test]
    fn untouched_gesture_is_the_identity() {
        let ts = state();
        assert!(ts.is_identity());
        assert_eq!(ts.affine(), Affine2::IDENTITY);
    }

    #[test]
    fn translation_alone_keeps_the_linear_part_exact() {
        let ts = state().translated(Vec2::ZERO, Vec2::new(37.5, -12.0), 0.5);
        assert_eq!(ts.linear, Mat2::IDENTITY);
        let a = ts.affine();
        assert_eq!(a.matrix2, Mat2::IDENTITY);
        assert_eq!(a.translation, Vec2::new(37.5, -12.0));
    }

    #[test]
    fn a_sub_epsilon_jiggle_changes_nothing() {
        let ts = state();
        let from = Vec2::new(100.0, 0.0);
        assert!(ts.turned_scaled(from, from + Vec2::splat(0.1), 0.5) == ts);
        assert!(ts.stretched(from, from + Vec2::splat(0.1), 0.5) == ts);
        assert!(ts.translated(from, from + Vec2::splat(0.1), 0.5) == ts);
    }

    #[test]
    fn rim_drag_carries_the_grab_point_to_the_pointer() {
        // Grab east, drag to twice-north: a quarter turn plus a 2× scale.
        let ts = state();
        let from = ts.center + Vec2::new(100.0, 0.0);
        let to = ts.center + Vec2::new(0.0, 200.0);
        let turned = ts.turned_scaled(from, to, 0.5);
        let moved = turned.linear * (from - ts.center);
        assert!((moved - (to - ts.center)).length() < 1e-3, "got {moved:?}");
        assert!(turned.linear.determinant() > 0.0);
    }

    #[test]
    fn outside_drag_pins_the_perpendicular_diameter() {
        // Grab east of the widget and drag: the north–south diameter must not move.
        let ts = state();
        let from = ts.center + Vec2::new(300.0, 0.0);
        let to = from + Vec2::new(80.0, 55.0);
        let stretched = ts.stretched(from, to, 0.5);
        let moved = stretched.linear * (from - ts.center);
        assert!((moved - (to - ts.center)).length() < 1e-3, "got {moved:?}");
        let pinned = stretched.linear * Vec2::new(0.0, 1.0);
        assert!(
            (pinned - Vec2::new(0.0, 1.0)).length() < 1e-6,
            "got {pinned:?}"
        );
    }

    #[test]
    fn flips_are_involutions() {
        let ts = state().flipped_h().flipped_v();
        assert!(!ts.is_identity());
        let back = ts.flipped_v().flipped_h();
        assert!(back.is_identity(), "four mirrors must cancel bit-exactly");
    }

    #[test]
    fn the_reference_is_a_circle_matching_the_hull_ellipses_area() {
        // A 200×100 hull: the circle's area equals the inscribed ellipse's
        // (π·100·50), i.e. r = √(100·50) — not the ellipse itself, because a
        // circle is what says "no distortion yet" (§16.6).
        let r = state().radius;
        assert!((r - 100.0f32.hypot(50.0)).abs() < 1e-3, "got {r}");
    }

    #[test]
    fn regions_classify_by_the_deformed_circle() {
        let ts = state();
        let (c, r) = (ts.center, ts.radius);
        assert_eq!(ts.region(c, 4.0), TransformRegion::Inside);
        assert_eq!(
            ts.region(c + Vec2::new(0.6 * r, 0.0), 4.0),
            TransformRegion::Inside
        );
        assert_eq!(ts.region(c + Vec2::new(r, 0.0), 4.0), TransformRegion::Rim);
        assert_eq!(ts.region(c + Vec2::new(0.0, -r), 4.0), TransformRegion::Rim);
        assert_eq!(
            ts.region(c + Vec2::new(1.6 * r, 0.0), 4.0),
            TransformRegion::Outside
        );

        // Stretch the widget to 2× along x: the rim moves with it.
        let wide = ts.stretched(c + Vec2::new(r, 0.0), c + Vec2::new(2.0 * r, 0.0), 0.5);
        assert_eq!(
            wide.region(c + Vec2::new(2.0 * r, 0.0), 4.0),
            TransformRegion::Rim
        );
        assert_eq!(
            wide.region(c + Vec2::new(r, 0.0), 4.0),
            TransformRegion::Inside
        );
    }
}

#[cfg(test)]
mod gesture_tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0))
    }

    #[test]
    fn a_fresh_perspective_is_the_identity_and_usable() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert!(p.is_identity());
        assert!(p.map().usable());
        assert!(TransformUi::Perspective(p).is_identity());
    }

    #[test]
    fn a_dragged_corner_lands_under_the_pointer() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        let delta = Vec2::new(-30.0, 22.0);
        let dragged = PerspectiveUi::corner_dragged(p, p, 3, delta, 0.5);
        assert_eq!(dragged.corners[3], p.corners[3] + delta);
        assert!(!dragged.is_identity());
        assert!(dragged.map().usable());
    }

    #[test]
    fn a_corner_pulled_across_the_quad_holds_at_the_last_valid_shape() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        // Almost across: still convex, accepted.
        let near = PerspectiveUi::corner_dragged(p, p, 0, Vec2::new(150.0, 60.0), 0.5);
        assert!(near.map().usable());
        // All the way across the opposite corner: the candidate is concave, so
        // the drag holds at `current` rather than folding the map.
        let held = PerspectiveUi::corner_dragged(p, near, 0, Vec2::new(500.0, 300.0), 0.5);
        assert_eq!(held, near);
    }

    #[test]
    fn quad_regions_classify_corners_edges_and_inside() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert_eq!(p.region(p.corners[1], 8.0, 5.0), QuadRegion::Corner(1));
        // Mid-top edge.
        let mid = (p.corners[0] + p.corners[1]) * 0.5;
        assert_eq!(p.region(mid, 8.0, 5.0), QuadRegion::Edge(0, 1));
        assert_eq!(p.region(Vec2::ZERO, 8.0, 5.0), QuadRegion::Inside);
        assert_eq!(
            p.region(Vec2::new(400.0, 400.0), 8.0, 5.0),
            QuadRegion::Outside
        );
    }

    #[test]
    fn a_fresh_warp_is_the_identity_and_usable() {
        let w = WarpUi::begin(LayerId(0), rect());
        assert!(w.is_identity());
        assert!(w.map().usable());
    }

    #[test]
    fn a_dragged_control_point_lands_under_the_pointer() {
        let w = WarpUi::begin(LayerId(0), rect());
        let delta = Vec2::new(14.0, -9.0);
        let dragged = WarpUi::point_dragged(w, w, 5, delta, 0.5);
        assert_eq!(dragged.points[5], w.points[5] + delta);
        assert!(dragged.map().usable());
        assert!(!dragged.is_identity());
    }

    #[test]
    fn a_surface_drag_carries_the_grabbed_paint_exactly() {
        let w = WarpUi::begin(LayerId(0), rect());
        let grab_at = Vec2::new(20.0, -10.0);
        let (t, basis, norm) = w.grab(grab_at);
        let before = w.map().eval(t);
        assert!(before.distance(grab_at) < 1.0, "grab missed: {before:?}");
        let delta = Vec2::new(18.0, 12.0);
        let dragged = WarpUi::surface_dragged(w, w, &basis, norm, delta, 0.5);
        let after = dragged.map().eval(t);
        assert!(
            after.distance(before + delta) < 0.1,
            "the paint under the finger moved {:?}, the finger moved {delta:?}",
            after - before
        );
    }

    #[test]
    fn a_folding_drag_holds_at_the_last_valid_shape() {
        let w = WarpUi::begin(LayerId(0), rect());
        // Drag an interior point far past its neighbour: the mesh would fold.
        let held = WarpUi::point_dragged(w, w, 5, Vec2::new(250.0, 0.0), 0.5);
        assert_eq!(held, w, "a fold must hold, not tear");
    }

    #[test]
    fn mesh_regions_classify_points_surface_and_outside() {
        let w = WarpUi::begin(LayerId(0), rect());
        assert_eq!(w.region(w.points[0], 8.0), MeshRegion::Point(0));
        assert_eq!(w.region(Vec2::new(15.0, 5.0), 8.0), MeshRegion::Inside);
        assert_eq!(w.region(Vec2::new(500.0, 0.0), 8.0), MeshRegion::Outside);
    }

    #[test]
    fn sub_epsilon_jiggles_snap_back_to_the_start() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert_eq!(
            PerspectiveUi::corner_dragged(p, p, 2, Vec2::splat(0.1), 0.5),
            p
        );
        let w = WarpUi::begin(LayerId(0), rect());
        assert_eq!(WarpUi::point_dragged(w, w, 5, Vec2::splat(0.1), 0.5), w);
        assert_eq!(WarpUi::translated(w, w, Vec2::splat(0.1), 0.5), w);
    }
}

/// Timeline mode's signals (§18.2.4), grouped because they are one
/// mode's worth of view state: the mode itself, whether it is playing, and how fast.
///
/// None of this is the *playhead* — that lives in the engine's timeline, where undo
/// and redo already move it, and is read back through
/// [`Engine::scrub_range`](stark_core::Engine::scrub_range). A copy here is exactly
/// the copy that would go stale the moment a stroke, an undo or a load moved the
/// history underneath it.
#[derive(Clone, Copy)]
pub struct TimelineState {
    /// Whether the mode is on — which is to say whether the bar is mounted.
    pub open: Signal<bool>,
    /// Whether playback is running. The source of truth for the transport button;
    /// the loop below reads it every tick and stops when it goes false, so anything
    /// that wants playback to end only has to clear this.
    pub playing: Signal<bool>,
    /// Playback rate, as a multiple of
    /// [`BASE_RATE`](crate::panels::timeline::BASE_RATE).
    pub speed: Signal<f32>,
    /// The playback loop, so a second Play cannot start a second one and closing
    /// the mode can cancel it. Root-owned like everything here: the loop outlives
    /// the bar that started it (the bar unmounts when the mode closes), which is
    /// the case `spawn_forever` exists for.
    pub task: Signal<Option<Task>>,
}

/// The eyedropper's signals (§18.0.2), grouped because they are one
/// feature's worth of view state: the two options a sample is taken with, the latch
/// that keeps Alt+drag from asking for samples faster than the GPU answers them, and
/// the two flags that say where in the gesture we are.
///
/// The options live here rather than in the engine because nothing in the engine
/// reads them between calls — [`Engine::pick_color`](stark_core::Engine::pick_color)
/// is a request and they are its arguments, so a copy projected back through
/// `observe()` would be state with no owner.
#[derive(Clone, Copy)]
pub struct PickState {
    /// Which layers a sample comes off, and whether the canvas stands behind them.
    pub scope: Signal<PickScope>,
    /// Half-width of the averaged square, in canvas px (0 = point sample).
    pub radius: Signal<u32>,
    /// Whether a sample is in flight — see [`crate::input::pick_color`].
    pub busy: Signal<bool>,
    /// Whether Alt is held. Only ever *shown*: the canvas wears the eyedropper
    /// cursor while it is and the options bar comes up, so the modifier announces
    /// itself before it is used — which is the whole reason a modifier binding is
    /// discoverable at all.
    pub alt_down: Signal<bool>,
    /// Whether an Alt+drag is actually sampling. Shared rather than local to the
    /// canvas, unlike `drawing`/`panning`, because the options bar is mounted on
    /// *armed but not yet dragging* and so has to be able to tell the two apart.
    pub dragging: Signal<bool>,
}

/// Which of the eyedropper's three sources the bar has selected (§18.0.2).
///
/// The *choice*, not [`PickSource`](stark_core::PickSource): which layer "this layer"
/// means is resolved against the selected layer at the moment of the sample
/// ([`crate::input::pick_color`]), so the bar cannot be left holding a layer id whose
/// layer has since been deleted — the same reason the radius is a number here and a
/// clamped one in the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PickScope {
    /// Every visible layer — the paint the canvas shows.
    #[default]
    AllLayers,
    /// Every visible layer over the substrate color, so bare canvas and thin paint
    /// answer with what the eye sees rather than with nothing and with the glaze.
    AllLayersAndCanvas,
    /// The selected layer alone, ignoring anything over or under it.
    ThisLayer,
}

/// The shared-session signals, grouped because they share one lifecycle: they are
/// set together when a session starts and cleared together when it ends, and
/// nothing outside [`crate::collab`] should be writing them piecemeal.
#[derive(Clone, Copy)]
pub struct CollabState {
    /// The live session, if any. `!Send` iroh handles live in unsync storage
    /// beside the renderer.
    pub session: Signal<Option<stark_net::CollabSession>>,
    /// The shareable ticket string, while hosting/joined.
    pub ticket: Signal<Option<String>>,
    /// Where the session lifecycle stands (drives the dialog + rail badge).
    pub phase: Signal<collab::CollabPhase>,
    /// The last share/join failure, surfaced in the dialog.
    pub error: Signal<Option<String>>,
    /// Who else is in the session, refreshed by the presence pump
    /// (§17.4). Its own signal rather than a field of `obs`: it changes on every remote
    /// pointer move, and re-running the whole component tree at that rate to move a
    /// cursor would be absurd.
    pub peers: Signal<Vec<crate::render::PeerInfo>>,
    /// How each directly-connected peer is reached — WebRTC, hole-punched UDP,
    /// or an iroh relay — polled off the mesh by the presence pump on a slow
    /// cadence (links change on the order of seconds, not frames). Peers in the
    /// roster but absent here have no direct connection; the mesh forwards
    /// their traffic. Read by the session dialog.
    pub links: Signal<Vec<stark_net::PeerLink>>,
    /// The incoming-event pump for `session`. Its lifetime is tied to the
    /// session's: `collab::install` replaces it, [`crate::collab::leave`]
    /// cancels it.
    pub pump: Signal<Option<Task>>,
    /// The outgoing presence pump — a fixed-cadence loop that drains the engine's
    /// presence latch onto the mesh (§17.5). Separate from `pump`
    /// because it is a *pull* on a timer rather than a reaction to arriving events,
    /// but shares the same lifecycle.
    pub presence: Signal<Option<Task>>,
}

impl CollabState {
    /// Whether a live session exists — i.e. whether anyone is on the other end
    /// of presence-only commands. `peek`: asked from event handlers at pointer
    /// rate, and nothing there should subscribe.
    pub fn active(&self) -> bool {
        self.session.peek().is_some()
    }
}

/// Repaint the canvas surface on the **next animation frame**, coalescing however
/// many requests land before it into one paint.
///
/// This is deliberately not `Renderer::paint` called inline. Paint requests arrive
/// per local pointer sample ([`dispatch`]) *and* per peer gesture frame (the collab
/// pump, ~30 Hz × peers) — with several clients drawing, that sums to hundreds of
/// full-frame composites a second submitted to a display that shows ~60. Nothing
/// bounds the GPU queue, so it grows for as long as the strokes last and input
/// latency climbs with it. The engine's state still integrates per event
/// (coalescing there would drop samples); it is only the *presentation* that is a
/// snapshot, and painting the latest state once per frame shows exactly what
/// painting it per event would have — minus the queue.
pub fn request_paint(state: AppState) {
    let mut queued = state.paint_queued;
    if *queued.peek() {
        return;
    }
    queued.set(true);
    schedule_paint(state);
}

/// One animation-frame hop of [`request_paint`]: paint, unless the GPU has
/// fallen behind — then hold the latch and try again next frame.
///
/// Directly in the rAF callback, not a task the rAF wakes. A woken task
/// resumes in the microtask drain, two scheduler hops after the callback —
/// and a *dioxus* task further waits out a VDOM render of whatever scopes
/// are dirty by then, since the scheduler polls tasks only once no scope is.
/// Any of that slipping past the frame's rendering steps shows the previous
/// frame's canvas for a whole display interval. The callback itself runs in
/// the animation phase, ahead of the rendering steps by definition, so the
/// frame that fires it is the frame that shows it. The closure is a plain
/// move of root-owned signals, so no component scope is involved and there
/// is nothing here to die with one (see the module note on `root_signal`).
fn schedule_paint(state: AppState) {
    crate::platform::on_animation_frame(move || {
        let mut renderer = state.renderer.0;
        let mut guard = renderer.write();
        // The GPU still owes more than a full pipeline of painted frames
        // (`Renderer::gpu_behind`): submitting another would deepen the queue,
        // not the picture. Skip the frame — with the latch *held*, so the next
        // rAF re-checks — and let the work drain. Nothing is lost: mutations
        // keep integrating per event and only mark the fold stale, so the
        // first paint after the drain folds everything accrued at once and
        // shows exactly what the skipped frames would have.
        if guard.as_ref().is_some_and(Renderer::gpu_behind) {
            drop(guard);
            schedule_paint(state);
            return;
        }
        let mut queued = state.paint_queued;
        queued.set(false);
        if let Some(r) = guard.as_mut() {
            r.paint();
        }
    });
}

/// Reach the engine as `&mut`, publish what it now looks like, and repaint —
/// `None` when WebGPU init has yet to hand one over, which every caller may
/// simply let fall through.
///
/// The publish is attached to the *door* rather than to the caller's memory, and
/// that is the whole design: `observe` is what the chrome renders from, so an
/// engine mutation that skips it leaves the chrome asserting something the engine
/// stopped believing — visible as a control that snaps back to its old value and
/// stays there (§4, §7). Two shipped bugs of exactly that shape are why this is a
/// function and `renderer` is a [`ReadOnly`].
///
/// For engine entry points that are not commands — opening a document, joining a
/// session. A command goes through [`dispatch`], which is this plus the broadcast.
pub fn with_engine<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    let (mut renderer, mut obs) = (state.renderer.0, state.obs.0);
    let out = {
        let mut guard = renderer.write();
        let r = guard.as_mut()?;
        let out = f(r);
        // Inside the guard, as `dispatch` has always done it: `obs.set` marks
        // subscribers dirty but renders nothing synchronously, so no reader can
        // observe the renderer mid-borrow.
        obs.set(Some(r.observe()));
        out
    };
    request_paint(state);
    Some(out)
}

/// Reach the engine as `&mut` **without** publishing anything or asking for a
/// frame — for work that cannot change what `observe` projects.
///
/// That is a real and large category, which is why this exists rather than a
/// `dispatch` everywhere: rendering, the readbacks (export, the eyedropper, the
/// navigator's miniature), draining the outbox and the presence tick, and
/// installing asset bytes an action will later name. None of it moves a value the
/// chrome shows, and several run at pointer or frame rate where the `observe` walk
/// and the VDOM diff behind a publish would be pure cost (see [`dispatch_sample`]).
///
/// If a closure here calls `Renderer::process`, it is in the wrong door — that is a
/// command, and commands publish. Callers wanting the frame but not the publish ask
/// for it with [`request_paint`], which several below do.
pub fn with_engine_quiet<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    let mut renderer = state.renderer.0;
    let mut guard = renderer.write();
    let r = guard.as_mut()?;
    Some(f(r))
}

/// Publish the engine's current projection without having mutated anything.
///
/// One caller, and it earns it: the collaboration pump takes the snapshot for the
/// events that commit and skips it for the ones that arrive at presence rate
/// (§17.5), so *which* event happened decides, not the fact of holding the engine.
pub fn publish_observation(state: AppState) {
    let mut obs = state.obs;
    if let Some(snapshot) = state.renderer.peek().as_ref().map(Renderer::observe) {
        obs.0.set(Some(snapshot));
    }
}

/// Hand the freshly-built engine to the app and publish its opening projection.
///
/// The order is the point: the projection is taken from `r` *before* it is moved
/// into the signal, because everything that reads the signal is written on the
/// assumption that a renderer being there means the chrome already describes it.
pub fn publish_renderer(state: AppState, r: Renderer) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    obs.0.set(Some(r.observe()));
    renderer.0.set(Some(r));
}

/// Apply a command, request a repaint, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
pub fn dispatch(state: AppState, command: impl Into<InputCommand>) {
    with_engine(state, |r| r.process(command));
    collab::flush_outbox(state);
}

/// Apply a pointer-rate sample of an in-flight gesture: integrate and repaint,
/// skipping the observable refresh and the outbox flush [`dispatch`] does.
///
/// A mid-gesture sample changes what the canvas shows but nothing the chrome
/// reads: the committed document, the layer list and the undo flags all stand
/// until the gesture ends, so `Engine::observe` would walk the layer tree to
/// produce the same answer — and `obs.set` marks every subscriber dirty
/// regardless, which puts a full VDOM diff of the chrome *ahead* of the rAF
/// registration on every sample (the scheduler polls tasks only once no scope
/// is dirty). Nothing commits mid-gesture either, so there is no outbox to
/// flush; the gesture's End goes through [`dispatch`], which refreshes the
/// observable and broadcasts whatever the commit banked.
pub fn dispatch_sample(state: AppState, command: impl Into<InputCommand>) {
    with_engine_quiet(state, |r| r.process(command));
    request_paint(state);
}

/// Apply a command **without** repainting, refreshing the observable, or
/// broadcasting — for the ones that change nothing this client can see.
///
/// [`PeerCommand::SetCursor`](stark_core::command::PeerCommand::SetCursor) is the
/// case that needs it: it arrives at pointer rate, and drawing our own cursor is the
/// browser's job, so the full `dispatch` would repaint the canvas hundreds of times a
/// second to show nothing. The value still reaches peers, because the presence pump
/// reads it off the engine on its own cadence (§17.5) rather than being
/// pushed from here.
pub fn dispatch_quiet(state: AppState, command: impl Into<InputCommand>) {
    with_engine_quiet(state, |r| r.process(command));
}

/// Resize the surface/engine, then repaint — inline, not [`request_paint`]: the
/// surface was just reconfigured, and a frame of the old size shown until the next
/// rAF is a visible flash. Resize arrives at layout rate, which cannot flood.
///
/// A report that arrives before the renderer exists is *dropped*, and the app start
/// is full of them — the canvas is laid out long before WebGPU init has finished
/// fetching its assets. That is safe only because the last thing init does before
/// publishing the renderer is re-read the element itself
/// ([`Renderer::sync_to_canvas`](crate::render::Renderer::sync_to_canvas)): the size
/// is recovered from the DOM rather than replayed from here, so nothing needs to be
/// queued. Move that call and the viewport goes stale again.
///
/// The paint stays inline *inside* the closure for the reason above; the frame
/// [`with_engine`] then queues is a redundant repaint of the same state, which
/// layout rate can well afford and which keeps the publish where every other
/// mutation has it.
pub fn resize(state: AppState, width: u32, height: u32) {
    with_engine(state, |r| {
        r.resize(width, height);
        r.paint();
    });
}

/// Read the current brush, mutate a copy, and commit it (releasing the `obs`
/// read guard before `dispatch` writes — avoids an AlreadyBorrowed panic).
pub fn update_brush(state: AppState, f: impl FnOnce(&mut BrushParams)) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        f(&mut brush);
        dispatch(state, ViewCommand::SetBrush(brush));
    }
}
