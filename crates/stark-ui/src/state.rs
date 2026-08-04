//! The app's shared state handle and the command path everything funnels through.
//!
//! [`dispatch`] is the single seam between the UI and the engine: every mutation
//! goes through it, so repaint, observable refresh and collaboration broadcast
//! happen in one place rather than at each call site (§4).

use dioxus::dioxus_core::{Task, spawn_forever};
use dioxus::prelude::*;

use crate::collab;
use crate::render::Renderer;
use stark_core::command::ViewCommand;
use stark_core::document::{BrushParams, LayerId};
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
    pub renderer: Signal<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command.
    pub obs: Signal<Option<ObservableState>>,
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
    /// Bumped whenever the brush colour is set from **outside** the colour picker —
    /// today only by the eyedropper.
    ///
    /// The Color panel's picker holds its own Oklab state (out-of-gamut `a`/`b`
    /// survive there but not in the clamped sRGB it reports), so it is *seeded*
    /// rather than driven, and this is how it is told to re-seed. Deliberately not
    /// "the colour changed": a pick inside the field would then drag its own marker
    /// back onto the gamut boundary under the user's cursor.
    pub color_epoch: Signal<u64>,
    /// The eyedropper (§18.0.2).
    pub pick: PickState,
    /// The drag-and-hold drawing assist (§6.9; `crate::input`).
    pub assist: AssistState,
    /// Minimal mode: the chrome over the canvas drops its words and keeps its marks
    /// (§11).
    ///
    /// A client preference like [`AssistState::enabled`], not document state — it is a
    /// way of *looking* at the application, so it is never saved, never replicated, and
    /// a collaborator's copy is theirs to set.
    ///
    /// Nothing reads this but the app root, which turns it into a class on `.app-root`;
    /// what the class hides is decided in the stylesheet against the `.label` spans the
    /// components emit. Deliberately not a prop threaded through the tree: every
    /// control would then have to be handed a boolean it does nothing with but pass on,
    /// and a control that forgot to would be the one that kept its word.
    pub minimal: Signal<bool>,
    /// The transform gesture in flight (§16.6): `Some` while the
    /// user is composing a move/scale/flip of the selected paint. View state —
    /// the engine sees only the previews it produces and the one commit on
    /// "Done".
    pub transform: Signal<Option<TransformState>>,
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
            renderer: root_signal(|| None),
            obs: root_signal(|| None),
            space_down: root_signal(|| false),
            canvas_active: root_signal(|| false),
            brush_editor_open: root_signal(|| false),
            preset_save_open: root_signal(|| false),
            color_epoch: root_signal(|| 0),
            pick: PickState {
                all_layers: root_signal(|| true),
                radius: root_signal(|| 0),
                busy: root_signal(|| false),
                alt_down: root_signal(|| false),
                dragging: root_signal(|| false),
            },
            assist: AssistState {
                enabled: root_signal(|| true),
                dwell: root_signal(|| None),
                task: root_signal(|| None),
            },
            // Off by default: the words are how the chrome is *learned*, and minimal
            // mode is what you turn on once you no longer need them.
            minimal: root_signal(|| false),
            transform: root_signal(|| None),
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
        }
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
    /// Sample the whole visible stack, rather than the selected layer alone.
    pub all_layers: Signal<bool>,
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
    // `spawn_forever`: requests originate in component event handlers and in the
    // collab pump alike, and a paint owed must not die with whichever scope asked
    // for it (see the module note on `root_signal`).
    spawn_forever(async move {
        crate::render::next_frame().await;
        let mut queued = state.paint_queued;
        queued.set(false);
        let mut renderer = state.renderer;
        if let Some(r) = renderer.write().as_mut() {
            r.paint();
        }
    });
}

/// Apply a command, request a repaint, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
pub fn dispatch(state: AppState, command: impl Into<InputCommand>) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    {
        let mut guard = renderer.write();
        if let Some(r) = guard.as_mut() {
            r.process(command);
            obs.set(Some(r.observe()));
        }
    }
    request_paint(state);
    collab::flush_outbox(state);
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
    let mut renderer = state.renderer;
    if let Some(r) = renderer.write().as_mut() {
        r.process(command);
    }
}

/// Resize the surface/engine, then repaint — inline, not [`request_paint`]: the
/// surface was just reconfigured, and a frame of the old size shown until the next
/// rAF is a visible flash. Resize arrives at layout rate, which cannot flood.
pub fn resize(state: AppState, width: u32, height: u32) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.resize(width, height);
        r.paint();
        obs.set(Some(r.observe()));
    }
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
