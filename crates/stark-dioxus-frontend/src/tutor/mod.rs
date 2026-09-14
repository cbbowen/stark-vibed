//! The guided tour: short lessons that arrive once the artist has earned them (§24).
//!
//! A lesson is attached to a **deed** and a **count** — paint four strokes and the Color
//! panel explains itself — so a tip only reaches somebody who has shown they care about
//! its subject, and each is given once, ever (§24.1). Doing the thing a card describes
//! answers it as well as its button does (§24.3).
//!
//! Deeds are read off [`dispatch`](crate::state::dispatch), the one seam every mutation
//! this user makes goes through (§4, §24.2). A new way to do a thing is counted for free;
//! a collaborator's work, which reaches the engine through `with_engine`, is not counted
//! as yours; and what a `SetBrush` means is decided by what moved
//! ([`lessons::brush_deed`]). The stream cannot say what wrote the brush, which is what
//! [`not_reaching`] brackets say, and [`did`] reports the deeds it does not carry at all.
//!
//! - [`lessons`] — what a deed is, the table of lessons, and which one a ledger owes.
//! - [`reader`] — the reader's state machine as a plain value ([`Tour`]).
//! - [`card`] — the chrome: where a card stands, what it opens, when it comes down.
//!
//! This module is the hook between them and the app. The tour is written on every deed
//! and every pan sample, so it sits in a `CopyValue` no render can subscribe to; what
//! renders read of it — the lesson waiting, the lesson on screen and whether dismissing it
//! brings another — is copied to signals, each written only when it changes ([`publish`]).

mod card;
mod lessons;
mod reader;

pub use card::TutorCard;
pub use lessons::Deed;
pub(crate) use lessons::Row;

use dioxus::prelude::*;
use stark_engine::command::{DocCommand, GestureCommand, InputCommand, ViewCommand};

use crate::platform;
use crate::state::{AppState, root_signal};
use lessons::{LESSONS, Ledger};
use reader::{Effects, Tour};

/// Everything the tour holds, root-owned like the rest of [`AppState`].
#[derive(Clone, Copy)]
pub struct TutorState {
    /// The reader. A `CopyValue` rather than a signal, so no render can subscribe to what
    /// every deed and pan sample writes; changed only through [`step`], except by
    /// [`begin`], which runs before any card is up, and by the bracket depth and the pan
    /// run, which no render reads.
    tour: CopyValue<Tour>,
    /// The lesson waiting for the screen to be free, copied from the tour by [`publish`].
    /// Read by the card's promotion effect.
    due: Signal<Option<usize>>,
    /// The lesson on screen, copied likewise. Read by the card and by
    /// `layout::standing_down` ([`holding_panels`]).
    showing: Signal<Option<usize>>,
    /// Whether dismissing the lesson on screen brings another, copied likewise. Read by the
    /// card's button.
    brings_another: Signal<bool>,
    /// Bumped to make the card measure its anchor again — a window resize.
    epoch: Signal<u64>,
}

impl TutorState {
    /// Its state, root-owned like every other group of signals (`state::root_signal`), and
    /// built here so the fields and the values they open on stay together.
    pub(crate) fn new() -> Self {
        Self {
            // Root-owned for `root_signal`'s reason, and made once, by a hook.
            tour: use_hook(|| CopyValue::new_in_scope(Tour::default(), ScopeId::ROOT)),
            due: root_signal(|| None),
            showing: root_signal(|| None),
            brings_another: root_signal(|| false),
            epoch: root_signal(|| 0),
        }
    }
}

/// Start listening, from what this browser has stored.
///
/// Called at the **end** of app startup, for `files::bind_file_launch`'s reason: the
/// opening preset, color and preferences all reach the engine through
/// [`dispatch`](crate::state::dispatch) too, and none of them is something the user did.
pub fn begin(state: AppState) {
    let ledger = stored();
    let mut tour = state.tutor.tour;
    tour.write().begin(ledger);
}

/// Open (`true`) or close (`false`) a bracket saying the brush is being written by
/// something other than the artist **reaching for one of its controls** (§24.2).
///
/// Three callers, each a different reason: the tuning drag (`input::Tune`, §18.1.9) and
/// the eyedropper (`input::pick_color`, §18.0.2) are the gestures two lessons teach, and
/// `presets::wear` is a whole tool arriving, which adjusts nothing. A bracket left open
/// costs counting, never a wrong card — the failure direction to have.
pub fn not_reaching(state: AppState, open: bool) {
    let mut tour = state.tutor.tour;
    if open {
        tour.write().open_bracket();
    } else {
        tour.write().close_bracket();
    }
}

/// Report a deed the command stream cannot name at all (§24.2).
///
/// [`not_reaching`]'s opposite: that one says a command should not be read, this one that
/// something happened no command describes — a library row clicked, a panel closed, the
/// panel column reached into, a color sampled, the brush editor opened.
pub fn did(state: AppState, deed: Deed) {
    let armed = state.tutor.tour.read().is_armed();
    if armed {
        tally(state, &[deed]);
    }
}

/// Whether a card on screen points into the panel stack, which it then holds up (§24.3).
///
/// Asked by `layout::standing_down`, which decides the stack's fade and its wake slice. A
/// subscribing read, since the caller is a render.
pub fn holding_panels(state: AppState) -> bool {
    (state.tutor.showing)()
        .and_then(|i| LESSONS.get(i))
        .is_some_and(|lesson| lesson.anchor.holds_panels())
}

/// Read one of the user's commands as deeds, and bring due whatever lesson they owe.
///
/// **Called before the command reaches the engine**, and the order is load-bearing:
/// [`lessons::brush_deed`] compares a `SetBrush` with the brush the engine still holds,
/// and the shape assist answers only before the gesture's `End` lands (§24.2).
pub fn observe(state: AppState, command: &InputCommand) {
    let armed = state.tutor.tour.read().is_armed();
    if !armed {
        return;
    }
    let deeds = read(state, command);
    if !deeds.is_empty() {
        tally(state, &deeds);
    }
}

/// Tips were switched off (§24.4): take down the card already up, which was promoted
/// under the old answer ([`Tour::switch_off`]).
///
/// Called by `prefs::set` alone, which both controls — ⚙ → Guidance and the card's
/// "Stop tips" — write through. Switching on needs no step: the card's promotion effect
/// reads the preference.
pub fn switch_off(state: AppState) {
    step(state, Tour::switch_off);
}

/// The deeds `command` reports — usually none, and for a stroke that snapped along a guide
/// three at once ([`stroke`]).
///
/// A `Vec`, and free where it matters: `Vec::new()` does not allocate, and that is what
/// every command at pointer rate gets — a pan sample included, since the run it feeds
/// scores only at its crossing.
fn read(state: AppState, command: &InputCommand) -> Vec<Deed> {
    match command {
        // A commit, not a start: a stroke abandoned by a second finger (§18.1.7) arrives
        // as `Cancel` and left no paint.
        InputCommand::Gesture(GestureCommand::End) => stroke(state),
        // Not the transport's: playback drives the playhead with these very commands
        // (§18.2.4), and the user's own stops playback before it dispatches
        // (`commands::edit_history`).
        InputCommand::Doc(DocCommand::Redo) => one(Deed::Redo, !playing(state)),
        InputCommand::Doc(DocCommand::Undo) => one(Deed::Undo, !playing(state)),
        // Nothing inside a bracket ([`not_reaching`]).
        InputCommand::View(ViewCommand::SetBrush { brush, color }) => {
            // The brush still held: `update_brush` dispatches before it moves the signal.
            let held = *state.brush.peek();
            let tune = *state.transient.peek();
            let tour = state.tutor.tour.read();
            tour.brush_write(&held.params(tune), tune.color, brush, *color)
                .into_iter()
                .collect()
        }
        // Both shapes of pan: one-pointer drags and the last finger of a two-finger gesture
        // arrive as `Pan`, the pair itself as `Pinch`. A `Zoom` is not travel.
        InputCommand::View(ViewCommand::Pan { delta }) => pan(state, delta.length()),
        InputCommand::View(ViewCommand::Pinch { anchor, to, .. }) => {
            pan(state, (*to - *anchor).length())
        }
        // Travel by the miniature, the one emitter of an absolute move
        // ([`Deed::UsedNavigator`]).
        InputCommand::View(ViewCommand::CenterOn(_)) => vec![Deed::UsedNavigator],
        // The Layers panel's button and the command sharing it; an image or a matte
        // arriving carries a command of its own.
        InputCommand::Doc(DocCommand::AddLayer { .. }) => vec![Deed::AddedLayer],
        _ => Vec::new(),
    }
}

/// `deed` if `counts`, as the list [`read`] answers with.
fn one(deed: Deed, counts: bool) -> Vec<Deed> {
    if counts { vec![deed] } else { Vec::new() }
}

/// Whether the timeline transport is moving the playhead right now (§18.2.4).
fn playing(state: AppState) -> bool {
    crate::panels::timeline::is_playing(state)
}

/// Feed a pan sample into the run in flight ([`Tour::pan`]), as the list [`read`] answers
/// with.
fn pan(state: AppState, travel: f32) -> Vec<Deed> {
    let now = platform::now_seconds();
    let mut tour = state.tutor.tour;
    let long = tour.write().pan(now, travel);
    long.into_iter().collect()
}

/// What a committed gesture was: a selection for a mask, nothing for a fill laid through a
/// marquee, and otherwise a stroke plus whatever the shape assist made of it (§6.9).
///
/// **Up to three deeds, and not alternatives.** A stroke that snapped along a vanishing
/// line is a stroke, an assisted stroke and a guided line at once, and each feeds a
/// different lesson; counting only the most specific would stall the two behind it for
/// somebody who works on a grid.
fn stroke(state: AppState) -> Vec<Deed> {
    // Which tool the gesture was is not on the command, but it is on the projection (§6.8).
    let Some(obs) = state.obs.peek().clone() else {
        return Vec::new();
    };
    if obs.tool.is_selection() {
        // The same drag builds a mask or lays a fill, decided when it started (§18.0.4).
        return one(Deed::Selection, obs.shape_action.is_select());
    }
    let mut deeds = vec![Deed::Stroke];
    // The engine rather than the projection: a gesture in flight is not projected, and the
    // assist goes with the gesture once its `End` lands.
    let assisted =
        crate::state::with_engine_quiet(state, |r| r.session.engine().assisted()).flatten();
    if let Some(shape) = assisted {
        deeds.push(Deed::AssistedStroke);
        // A guide this client shows is what makes the grid's aiming reachable (§20.6) —
        // this client's own eye rather than the document's roster (§20.5). Whether this
        // line landed near an axis is not the question.
        let guided = shape == stark_engine::Assisted::Line && obs.guides.iter().any(|g| g.visible);
        if guided {
            deeds.push(Deed::GuidedLine);
        }
    }
    deeds
}

/// Count `deeds` against the tour, judged by the clock and the two settings a count is
/// judged by.
fn tally(state: AppState, deeds: &[Deed]) {
    let now = platform::now_seconds();
    let prefs = *state.prefs.peek();
    step(state, |tour| {
        tour.tally(now, deeds, prefs.chrome_hiding, prefs.tips)
    });
}

/// Run one step of the tour, carry out the [`Effects`] it asks for, and [`publish`] its
/// card.
///
/// The door every change to the card goes through, so what a render reads cannot fall out
/// of step with the tour. The tour's borrow ends before any effect runs.
fn step(state: AppState, f: impl FnOnce(&mut Tour) -> Effects) {
    let mut tour = state.tutor.tour;
    let effects = f(&mut tour.write());
    if effects.save {
        save(state);
    }
    if effects.wake_panels {
        crate::layout::wake_panels(state);
    }
    publish(state);
}

/// Copy the tour's card into the signals renders read, writing each only if it changed: a
/// `set` wakes every reader whatever it is handed, and most steps change none of them.
fn publish(state: AppState) {
    let chrome = state.prefs.peek().chrome_hiding;
    let (card, more) = {
        let tour = state.tutor.tour.read();
        let card = tour.card();
        let more = card
            .showing()
            .is_some_and(|i| tour.brings_another(i, chrome));
        (card, more)
    };
    mirror(state.tutor.due, card.due());
    mirror(state.tutor.showing, card.showing());
    mirror(state.tutor.brings_another, more);
}

/// Write `value` into `signal` only if it differs from what the signal holds.
fn mirror<T: Copy + PartialEq + 'static>(mut signal: Signal<T>, value: T) {
    let held = *signal.peek();
    if held != value {
        signal.set(value);
    }
}

/// What this browser has stored, or an empty ledger — nothing stored and a damaged store
/// both mean somebody new.
fn stored() -> Ledger {
    stark_ui::storage::load_list::<Row>().map_or_else(Ledger::default, Ledger::from_rows)
}

/// Persist the ledger — after every counted deed, so at most one write per coalesced act,
/// and only while the app is in use.
fn save(state: AppState) {
    let rows = state.tutor.tour.read().ledger().rows();
    stark_ui::storage::save_list(&rows);
}
