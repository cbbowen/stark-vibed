//! The chrome that rides the canvas: collaborators' pointers, the brush cursor,
//! the readout a tuning drag draws, the held pick's swatch, and the tow string
//! (§17.4, §18.1.8-§18.1.11, §6.11).
//!
//! **DOM rather than a compositor pass, every one of them**, and always for the
//! same reason: each is chrome, and the one thing chrome must never do is reach
//! an export. A `<div>` the browser lays out cannot, and a pass that wrote into
//! the canvas would have to be remembered about at every place a picture leaves
//! the app.
//!
//! They are collected here because they share that argument and a layer — over
//! the canvas, under the floating chrome — and because what each of them costs
//! is the same question asked five times: *what wakes this at pointer rate?* The
//! answers are worth reading together. Two of them are pure layout, converting
//! to screen px on the way in so they read no view at all
//! (`state::TuneReadout`, `state::TowUi`); one reads the view through a memo
//! ([`PeerCursors`]); one splits its position from its size so only the position
//! moves per report ([`BrushCursor`]).

use dioxus::prelude::*;

use crate::state::{AppState, BrushRing, FlowBar, TuneReadout, use_obs};
use stark_chrome::drags::DragAction;

/// Collaborators' pointers, drawn in each peer's own color (§17.4).
///
/// DOM rather than a compositor pass, on purpose: a cursor is chrome, not artwork —
/// it must never reach an export, and a label beside it is a `<div>` the browser
/// already knows how to lay out. The positions are canvas-space, so they follow the
/// painting under pan and zoom exactly as the paint does.
#[component]
pub fn PeerCursors() -> Element {
    let state = use_context::<AppState>();
    // The view, through one memo — unconditionally, ahead of the early return,
    // like any `use_*`.
    //
    // **Not a `peek` of the renderer**, which is what this was and which left the
    // cursors behind under a pan: the only other thing this component subscribes
    // to is the roster, and the presence pump writes that solely when the
    // roster's revision moves (`collab`). So a collaborator holding still while
    // this client pans left their cursor pinned to a *screen* position — the one
    // thing the doc above says it is not. The peek's reason was sound (the
    // renderer signal is written by every engine door, so subscribing to it would
    // redraw these on every stroke sample) and this is what `use_obs` is for: the
    // memo propagates when the view moves and sleeps through everything else.
    let look = use_obs(state, |o| o.view);
    let peers = (state.collab.peers)();
    if peers.is_empty() {
        return rsx! {};
    }
    let Some(view) = look() else {
        return rsx! {};
    };
    rsx! {
        div { class: "peer-cursors",
            for peer in peers {
                if let Some(canvas) = peer.cursor {
                    {
                        let p = view.canvas_to_screen(canvas);
                        rsx! {
                            div {
                                key: "{peer.actor.0}",
                                class: "peer-cursor",
                                style: "left:{p.x}px; top:{p.y}px; --peer:{peer.css_color()}",
                                div { class: "peer-cursor-dot" }
                                div { class: "peer-cursor-name", "{peer.name}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The brush cursor (§18.1.10): a circle of the live brush's size riding the
/// resting pointer, so the canvas says how much of it the next stroke would take
/// before the stroke is made. The crosshair stays — it is the hotspot; this is
/// the footprint.
///
/// DOM rather than a compositor pass, for [`PeerCursors`]'s reason — it is
/// chrome, and it must never reach an export. The split of who re-renders is the
/// point of its shape: the *position* is its own signal written per pointer
/// report ([`Signals::brush_cursor`](crate::state::Signals::brush_cursor)), so only this component moves at pointer
/// rate, while the *size* is the projection's `brush.radius × view.zoom` through
/// one memo — a bracket tap or a wheel notch resizes the circle where it stands,
/// and a pan, which moves neither factor, wakes nothing here at all.
///
/// A circle, though the brush may be any shape (§6.6) — deliberately, even now
/// that the real thing renders beneath it: the engine folds the mark a drag
/// begun this instant would open (§18.1.10, `ViewCommand::PreviewHover`, fed by
/// the same move handler), and the two halves divide the work — the mark says
/// what the paint would *do*, the ring says how far the brush *reaches*, which
/// a soft tip's mark understates.
#[component]
pub fn BrushCursor() -> Element {
    let state = use_context::<AppState>();
    // The memo ahead of the early returns — a hook, like any `use_*`.
    // The size half comes off the frontend's own brush signal; the projection
    // supplies what only the engine knows (the zoom, the layer, the tool).
    let size = (state.transient)().size;
    let look = use_obs(state, |o| {
        let paintable = o
            .layers
            .iter()
            .any(|l| l.id == o.active_layer && l.is_paintable());
        (o.view.zoom, paintable, o.tool)
    });
    let Some(at) = (state.brush_cursor)() else {
        return rsx! {};
    };
    let Some((zoom, paintable, tool)) = look() else {
        return rsx! {};
    };
    let r = size * zoom;
    // Shown exactly where the crosshair itself promises paint: not over a layer
    // that takes none (the cursor already says not-allowed — §15.7), not under a
    // marquee tool, whose mark is the shape dragged rather than the brush, and
    // not while space arms a pan or a chord arms an act that shadows the brush —
    // the eyedropper, the layer carry. Those announce themselves through the
    // cursor, and the circle must not outbid them; which acts they are is the
    // table's answer rather than a list kept here (`DragAction::shadows_paint`).
    // It stays up through a stroke, where it goes on being true.
    if !paintable
        || tool.is_selection()
        || (state.space_down)()
        || stark_chrome::drags::armed(&state.drags.read(), (state.held_mods)())
            .is_some_and(DragAction::shadows_paint)
        || (state.pick.dragging)()
    {
        return rsx! {};
    }
    // A circle smaller than the crosshair is noise inside it: for a tip this
    // fine the crosshair is the better picture, so the circle waits for a size
    // worth drawing.
    if r < 3.0 {
        return rsx! {};
    }
    rsx! {
        div { class: "brush-cursor",
            div {
                class: "brush-ring-circle",
                style: "left:{at.x - r}px; top:{at.y - r}px; width:{2.0 * r}px; height:{2.0 * r}px",
            }
        }
    }
}

/// The brush-tuning drag's indicator (§18.1.9): the size ring, or the flow bar, or
/// nothing at all — whichever the gesture in flight is asking for.
///
/// DOM rather than a compositor pass, for [`PeerCursors`]'s reason — it is chrome, and
/// the one thing it must never do is reach an export. It is also pure layout: the
/// gesture converts to screen px on its way in (`state::TuneReadout`), so this reads no
/// view and re-renders on nothing but the readout itself.
///
/// **One component for both knobs**, because the state is one value: the drag commits
/// to a single knob, so a second mounted overlay would exist to draw the thing the
/// first one is not drawing, and would have to be told when to keep quiet. Here that is
/// a `match` with no third arm.
#[component]
pub fn TuneReadoutOverlay() -> Element {
    let state = use_context::<AppState>();
    let Some(readout) = (state.tune_readout)() else {
        return rsx! {};
    };
    rsx! {
        div { class: "tune-readout",
            match readout {
                TuneReadout::Size(ring) => size_ring(ring),
                TuneReadout::Flow(bar) => flow_bar(bar),
            }
        }
    }
}

/// The size half: a ring at the radius being asked for, with the radius it started
/// from behind it.
///
/// A circle, though the brush may be any shape (§6.6). Deliberately, for now: what the
/// drag sets is one number, and a ring is the honest picture of one number — an outline
/// of the actual tip would be a picture of the *shape*, which this gesture cannot
/// change, and would say the mark is that crisp when a soft brush's is not.
fn size_ring(ring: BrushRing) -> Element {
    // Both circles are laid out the same way: a box of the diameter, pulled back onto
    // the centre. `left`/`top` rather than a transform, like the peer cursors.
    let circle = |class: &'static str, r: f32| {
        rsx! {
            div {
                class: class,
                style: "left:{ring.at.x - r}px; top:{ring.at.y - r}px; width:{2.0 * r}px; height:{2.0 * r}px",
            }
        }
    };
    rsx! {
        // The old size first, so the one being asked for draws over it.
        {circle("brush-ring-circle was", ring.was)}
        {circle("brush-ring-circle", ring.now)}
    }
}

/// The flow half: a bar on the press, filled to how much paint the brush is laying.
///
/// A *level* rather than a picture of the mark, which is the honest shape for this
/// knob: flow has no length on the canvas to be drawn at, so the bar says where in the
/// range the brush sits and claims nothing about how the stroke will look. Quiet, too —
/// it is the answer to a gesture the hand is in the middle of making, not a control,
/// and the Brush panel's slider is moving in step with it either way.
///
/// **Centred on the press, exactly as the ring is** — the readout has one anchor, and
/// the moment the drag commits to a knob is not a moment for the answer to jump
/// sideways. It can sit on the point because this drag takes the crosshair down
/// (`canvas`), so there is nothing drawn there to collide with, and because the bar
/// holds still while the pointer runs past both its ends — which is what keeps it from
/// reading as a track the pointer is somewhere *on*.
///
/// **How large the bar is, is the stylesheet's** (`state::FlowBar`), which is why the
/// centring is a transform where the ring's is arithmetic: half of a box this side
/// does not have cannot be subtracted from the point it was handed.
fn flow_bar(bar: FlowBar) -> Element {
    let fill = bar.fill.clamp(0.0, 1.0) * 100.0;
    rsx! {
        div { class: "flow-bar", style: "left:{bar.at.x}px; top:{bar.at.y}px",
            div { class: "flow-bar-fill", style: "height:{fill:.1}%" }
        }
    }
}

/// The held touch pick's loupe (§18.1.11): the color the finger is standing on,
/// drawn clear of the finger.
///
/// The gesture's whole readout, and the only feedback it has. The eyedropper's
/// chord binding needs none — a mouse or a pen leaves a cursor on the point and the
/// Color panel in plain view — but a finger is *on* the place it is asking about,
/// with a hand behind it covering most of a tablet's screen. So the answer is put
/// where it can be seen: above the contact by more than a fingertip, which is what
/// [`LOUPE_LIFT`] is.
///
/// DOM rather than a compositor pass, for [`PeerCursors`]'s reason — it is chrome,
/// and the one thing it must never do is reach an export. It reads the **brush**
/// color rather than the sample, deliberately: what the gesture is *for* is loading
/// the brush, so the swatch shows what would be painted with, and a color the
/// readback has not landed yet is honestly still the old one.
#[component]
pub fn PickLoupe() -> Element {
    let state = use_context::<AppState>();
    let color = (state.transient)().color;
    let Some(at) = (state.pick.loupe)() else {
        return rsx! {};
    };
    let [r, g, b] = color;
    // Straight sRGB, which is what a brush color is (`panels::color`).
    let fill = format!(
        "background: rgb({:.1}% {:.1}% {:.1}%);",
        r * 100.0,
        g * 100.0,
        b * 100.0
    );
    // Above the finger, or below it near the top of the window. Flipped rather than
    // clamped: clamping would slide the swatch onto the contact point it exists to
    // stay clear of, which at the top of the canvas is exactly where a hand reaching
    // over the picture puts it.
    let y = if at.y < LOUPE_LIFT * 1.5 {
        at.y + LOUPE_LIFT
    } else {
        at.y - LOUPE_LIFT
    };
    rsx! {
        div { class: "pick-loupe",
            div {
                class: "pick-loupe-well",
                style: "left:{at.x}px; top:{y}px; {fill}",
            }
        }
    }
}

/// How far above the finger the held pick's swatch sits, CSS px.
///
/// A fingertip's contact patch is about 10 mm across and the hand behind it covers
/// everything below and to one side, so the swatch clears the *finger* rather than
/// the contact point — far enough to be read without moving the hand, close enough
/// that it still reads as belonging to it.
const LOUPE_LIFT: f32 = 44.0;

/// The tow string while a smoothing brush draws (§6.11): a hairline from the
/// towed tip — where paint is landing — to the pointer, with a dot under the
/// hand. It **sags while slack and straightens as it tows**, so the state of
/// the mechanism is legible at a glance: the one thing that makes deliberate
/// lag feel like latency is being unexplained, and the sag is the explanation.
///
/// DOM (an SVG overlay) rather than a compositor pass, for [`PeerCursors`]'s
/// reason: it is chrome, and it must never reach an export. Pure layout, too —
/// the gesture converts to screen px on its way in (`input::refresh_tow`,
/// `state::TowUi`), so this reads no view and re-renders on nothing but the
/// string itself.
#[component]
pub fn TowStringOverlay() -> Element {
    let state = use_context::<AppState>();
    let Some(t) = (state.tow)() else {
        return rsx! {};
    };
    let slack = (t.rope - (t.target - t.tip).length()).max(0.0);
    // The sag is the slack made visible: a quadratic dip whose depth is a share
    // of the string the tow is not using — capped, so a fresh stroke's fully
    // slack string drapes rather than plunges. Screen-down whatever the canvas
    // rotation, because the string hangs from the hand, not from the painting.
    let sag = (slack * 0.4).min(t.rope * 0.2);
    let mid = (t.tip + t.target) * 0.5 + stark_model::Vec2::new(0.0, sag);
    let d = format!(
        "M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}",
        t.tip.x, t.tip.y, mid.x, mid.y, t.target.x, t.target.y
    );
    rsx! {
        svg { class: "tow-string",
            path { d }
            circle { cx: "{t.target.x}", cy: "{t.target.y}", r: "2.5" }
        }
    }
}
