//! The app's shared state handle and the command path everything funnels through.
//!
//! [`dispatch`] is the single seam between the UI and the engine: every mutation
//! goes through it, so repaint, observable refresh and collaboration broadcast
//! happen in one place rather than at each call site (DESIGN.md §4).

use dioxus::dioxus_core::{Task, spawn_forever};
use dioxus::prelude::*;

use crate::collab;
use crate::render::Renderer;
use stark_core::command::ViewCommand;
use stark_core::document::{BrushParams, LayerId};
use stark_core::geom::{Affine2, Vec2};
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
    /// Bumped whenever the brush colour is set from **outside** the colour picker —
    /// today only by the eyedropper.
    ///
    /// The Color panel's picker holds its own Oklab state (out-of-gamut `a`/`b`
    /// survive there but not in the clamped sRGB it reports), so it is *seeded*
    /// rather than driven, and this is how it is told to re-seed. Deliberately not
    /// "the colour changed": a pick inside the field would then drag its own marker
    /// back onto the gamut boundary under the user's cursor.
    pub color_epoch: Signal<u64>,
    /// The eyedropper (MISSING_FEATURES §0.2).
    pub pick: PickState,
    /// The transform gesture in flight (TRANSFORM_DESIGN.md §6): `Some` while the
    /// user is composing a move/scale/flip of the selected paint. View state —
    /// the engine sees only the previews it produces and the one commit on
    /// "Done".
    pub transform: Signal<Option<TransformState>>,
    /// Whether a [`request_paint`] is already waiting on the next animation frame.
    /// The latch that turns any number of paint requests into one paint per frame.
    /// Read and written only from non-component code (`peek`/`set`), so no
    /// component ever subscribes to it.
    pub paint_queued: Signal<bool>,
    /// Everything to do with a shared drawing (DESIGN.md §12).
    pub collab: CollabState,
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
            color_epoch: root_signal(|| 0),
            pick: PickState {
                all_layers: root_signal(|| true),
                radius: root_signal(|| 0),
                busy: root_signal(|| false),
                alt_down: root_signal(|| false),
                dragging: root_signal(|| false),
            },
            transform: root_signal(|| None),
            paint_queued: root_signal(|| false),
            collab: CollabState {
                session: root_signal(|| None),
                ticket: root_signal(|| None),
                phase: root_signal(collab::CollabPhase::default),
                error: root_signal(|| None),
                peers: root_signal(Vec::new),
                pump: root_signal(|| None),
                presence: root_signal(|| None),
            },
        }
    }
}

/// The transform gesture being composed (TRANSFORM_DESIGN.md §6): the rect the
/// handles started from, the rect they have dragged it to, and the flips applied
/// inside it. The affine the engine sees is *derived* from these on every change,
/// always mapping the original `hull` to the current `rect` — so a long drag is
/// one accumulated transform, never a chain of them, and the preview resamples the
/// committed tiles exactly once ("lossless" until "Done").
#[derive(Clone, Copy, PartialEq)]
pub struct TransformState {
    /// The layer whose selected paint is being transformed.
    pub layer: LayerId,
    /// The canvas-space box the handles were mounted around — the selection hull
    /// at entry (or the painted content's bounds for an unbounded selection).
    pub hull: (Vec2, Vec2),
    /// Where the drags have taken that box, canvas space, normalized min/max.
    /// Kept in the box's own (unrotated) frame: `angle` turns the whole box
    /// afterwards, about its centre.
    pub rect: (Vec2, Vec2),
    /// Mirrors applied within the box: (horizontal, vertical) — i.e. flip.0
    /// mirrors left↔right, flip.1 top↔bottom.
    pub flip: (bool, bool),
    /// Rotation about `rect`'s centre, radians, clockwise on screen (canvas y is
    /// down, so `Affine2::from_angle`, the CSS `rotate()` the box is drawn with,
    /// and the pointer's `atan2` all agree on the sign).
    pub angle: f32,
}

impl TransformState {
    pub fn begin(layer: LayerId, hull: (Vec2, Vec2)) -> Self {
        Self {
            layer,
            hull,
            rect: hull,
            flip: (false, false),
            angle: 0.0,
        }
    }

    /// Whether committing would change nothing — "Done" then skips the commit
    /// rather than spending an undo step on a no-op.
    pub fn is_identity(&self) -> bool {
        self.rect == self.hull && !self.flip.0 && !self.flip.1 && self.angle == 0.0
    }

    /// The affine mapping `hull` onto `rect` with the flips applied — what the
    /// preview shows and "Done" commits.
    pub fn affine(&self) -> Affine2 {
        let (hmin, hmax) = self.hull;
        let (rmin, rmax) = self.rect;
        // One axis of the map: hull span → rect span, mirrored if flipped. A drag
        // that only *moved* the box leaves the spans equal up to float noise, and
        // the scale is snapped to exactly 1 there — a pure move must be a pure
        // translation, which resamples losslessly on the integer grid
        // (TRANSFORM_DESIGN.md §4) instead of softening under a 1±ε scale.
        let axis = |h0: f32, h1: f32, r0: f32, r1: f32, flipped: bool| {
            let hw = (h1 - h0).max(1e-3);
            let mut s = (r1 - r0) / hw;
            if (s - 1.0).abs() < 1e-4 {
                s = 1.0;
            }
            if flipped {
                (-s, r1 + h0 * s)
            } else {
                (s, r0 - h0 * s)
            }
        };
        let (sx, tx) = axis(hmin.x, hmax.x, rmin.x, rmax.x, self.flip.0);
        let (sy, ty) = axis(hmin.y, hmax.y, rmin.y, rmax.y, self.flip.1);
        let mut a = Affine2::from_scale(Vec2::new(sx, sy));
        a.translation = Vec2::new(tx, ty);
        if self.angle == 0.0 {
            return a;
        }
        // The rotation turns the finished box about its own centre, after the
        // scale/flip map — matching what the rotated chrome shows. Gated on
        // exactly zero so an unrotated gesture keeps the pure scale/translate
        // form the exactness invariants rely on (TRANSFORM_DESIGN.md §4).
        let c = (rmin + rmax) * 0.5;
        Affine2::from_translation(c)
            * Affine2::from_angle(self.angle)
            * Affine2::from_translation(-c)
            * a
    }
}

/// The eyedropper's signals (MISSING_FEATURES §0.2), grouped because they are one
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
    /// Who else is in the session, refreshed by the presence pump (PEER_DESIGN.md
    /// §4). Its own signal rather than a field of `obs`: it changes on every remote
    /// pointer move, and re-running the whole component tree at that rate to move a
    /// cursor would be absurd.
    pub peers: Signal<Vec<crate::render::PeerInfo>>,
    /// The incoming-event pump for `session`. Its lifetime is tied to the
    /// session's: `collab::install` replaces it, [`crate::collab::leave`]
    /// cancels it.
    pub pump: Signal<Option<Task>>,
    /// The outgoing presence pump — a fixed-cadence loop that drains the engine's
    /// presence latch onto the mesh (PEER_DESIGN.md §5.1). Separate from `pump`
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
/// reads it off the engine on its own cadence (PEER_DESIGN.md §5.1) rather than being
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
