//! Timeline mode: the history as something to travel through
//! (§18.2.4, §8).
//!
//! The action log is already a timelapse — it is what the document *is*, and
//! replaying it is what loading a file does. What this adds is a way to stand
//! anywhere in it: a bar carrying a scrubber and a transport, so the artist can
//! drag back to the moment a piece went wrong, or watch it arrive at speed.
//!
//! **The playhead is not stored here.** Scrubbing moves the engine timeline's
//! applied/withheld split — the very split undo and redo move one step at a time
//! ([`Timeline::seek`](stark_core::document::Timeline::seek)) — so the mode adds
//! no second notion of "where we are" that could disagree with the undo stack.
//! Two consequences fall out rather than being arranged:
//!
//! - leaving the mode scrubbed back is *being undone by that many steps*, with
//!   Redo (and the scrubber) the way forward again;
//! - painting there truncates the future exactly as painting after an undo does.
//!
//! Painting is blocked while the transport is **playing**, and only then: a
//! commit clears the withheld half, so a stroke laid under a moving playhead
//! would delete the rest of the piece. Paused, a scrub is a place to work from —
//! which is the whole point of a scrubber the artist can reach mid-painting.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use crate::layout::chrome_class;
use crate::platform::sleep_ms;
use crate::state::{AppState, dispatch};
use stark_core::command::{DocCommand, ViewCommand};

/// Actions per second at 1×. Eight is about the rate at which a painting reads as
/// being *made* rather than as a slideshow of states: fast enough that a session's
/// worth of strokes lands in a minute or two, slow enough that a single one can be
/// seen arriving. The speed chips scale it from a quarter of this to four times it.
pub const BASE_RATE: f32 = 8.0;

/// The shortest wait worth asking the browser for. Below one animation frame the
/// timer stops being what paces playback — the browser clamps it, and the rate
/// silently stops responding to the speed control — so past this point the *stride*
/// grows instead of the interval shrinking.
const MIN_TICK_MS: f32 = 16.0;

/// Most ticks a track will draw. Past this they sit closer together than a hairline
/// and stop being marks at all; the fill and the counter still say where the
/// playhead is, and eight hundred `<div>`s to say it worse is not a trade.
const MAX_TICKS: usize = 240;

/// The playback rates the transport offers, as multiples of [`BASE_RATE`].
const SPEEDS: [(f32, &str); 5] = [
    (0.25, "\u{00BC}\u{00D7}"),
    (0.5, "\u{00BD}\u{00D7}"),
    (1.0, "1\u{00D7}"),
    (2.0, "2\u{00D7}"),
    (4.0, "4\u{00D7}"),
];

/// How long to wait between steps, and how many actions to cross each time, for a
/// given speed multiplier.
///
/// One step per tick until a tick would be shorter than a frame; from there the
/// interval is pinned and the stride takes over. Without that, "4×" past the
/// browser's timer floor would be indistinguishable from "1×" — the control would
/// still move and nothing would happen.
fn pace(speed: f32) -> (i32, usize) {
    let per_step = 1000.0 / (BASE_RATE * speed.max(0.01));
    if per_step >= MIN_TICK_MS {
        (per_step.round() as i32, 1)
    } else {
        (
            MIN_TICK_MS as i32,
            (MIN_TICK_MS / per_step).round().max(1.0) as usize,
        )
    }
}

/// Where the playhead stands and how far it can travel, or `None` for a document
/// whose history is not this client's alone to walk (a shared session).
///
/// A document with no engine behind it yet — the first frames of app start — is an
/// *empty* history rather than an unscrubbable one, so the bar says "0 / 0" instead
/// of blaming a session that does not exist.
///
/// `peek`, not `read`: the engine is written on every command, and anything that
/// subscribed to it here would wake on every pointer sample of every stroke. What
/// moves the playhead is a committed change, and that is what the bar watches.
pub fn scrub_range(state: AppState) -> Option<(usize, usize)> {
    match state.renderer.peek().as_ref() {
        Some(r) => r.scrub_range(),
        None => Some((0, 0)),
    }
}

/// Whether playback is running — the one question the canvas asks of this module.
pub fn is_playing(state: AppState) -> bool {
    *state.timeline.playing.peek()
}

/// Enter or leave Timeline mode.
///
/// Leaving stops playback but leaves the playhead exactly where it was put: the
/// mode is a way of *looking* at the history, not a round trip that undoes itself,
/// and a scrub that snapped back on exit would make "go back to where it went wrong
/// and carry on from there" impossible. Where it was left shows up in the Redo
/// affordance, which is what an undo of the same depth would have left too.
pub fn set_open(state: AppState, open: bool) {
    stop(state);
    // A transform gesture composes against the committed document; scrubbing that
    // out from under it would leave the widget pointing at paint that is no longer
    // there, and commit against whatever the playhead landed on.
    let mut transform = state.transform;
    if open && transform.write().take().is_some() {
        dispatch(state, ViewCommand::PreviewTransform(None));
    }
    let mut mode = state.timeline.open;
    mode.set(open);
}

/// Stop playback, if any. Idempotent and cheap when already stopped, because the
/// scrubber calls it on every pointer sample of a drag.
///
/// Both halves matter: clearing `playing` is what the loop watches, and cancelling
/// the task is what stops it waiting out the tick it is already inside.
pub fn stop(state: AppState) {
    let mut playing = state.timeline.playing;
    if *playing.peek() {
        playing.set(false);
    }
    let mut task = state.timeline.task;
    if task.peek().is_some()
        && let Some(t) = task.write().take()
    {
        t.cancel();
    }
}

/// Start playback from the playhead — or from the top, when it is already at the
/// end, since that is the only reading of "play" that does anything there.
fn play(state: AppState) {
    stop(state);
    let Some((at, total)) = scrub_range(state) else {
        return;
    };
    if total == 0 {
        return;
    }
    if at >= total {
        dispatch(state, DocCommand::Seek(0));
    }

    let mut playing = state.timeline.playing;
    playing.set(true);
    // `spawn_forever`: the loop is started from the bar's scope, and the bar
    // unmounts the moment the mode closes — a plain `spawn` would tie playback to
    // the thing it is meant to outlive (see `state::root_signal`).
    let task = spawn_forever(async move {
        loop {
            let (delay, stride) = pace(*state.timeline.speed.peek());
            sleep_ms(delay).await;
            // Read afresh every tick rather than closing over the range: the speed
            // can change mid-play, and so can the log (a load, a merge).
            if !*state.timeline.playing.peek() {
                return;
            }
            let Some((at, total)) = scrub_range(state) else {
                break;
            };
            let to = (at + stride).min(total);
            dispatch(state, DocCommand::Seek(to));
            if to >= total {
                break;
            }
        }
        // Running out of history is the loop's own business to report. A cancel
        // returns above instead, having already been cleared by `stop`.
        let mut playing = state.timeline.playing;
        playing.set(false);
    });
    let mut slot = state.timeline.task;
    slot.set(Some(task));
}

/// The timeline bar: transport, scrubber, and speed (§18.2.4).
///
/// Mounted by the app root only while the mode is on — like the modals, and unlike
/// the other bars, because this one owns hooks and a component may not gain or lose
/// those between renders.
///
/// It watches `doc_revision` and nothing else. The playhead lives in the engine, and
/// the only thing that moves it is a committed change: a scrub, an undo, a stroke, a
/// load. Reading the renderer reactively instead would re-render this on every
/// pointer sample to redraw a track that had not moved.
#[component]
pub fn TimelineBar() -> Element {
    let state = use_context::<AppState>();
    // The revision alone, memoized: `obs` is written per command, but this only
    // notifies when the *committed* document actually moved.
    let revision = use_memo(move || state.obs.read().as_ref().map_or(0, |o| o.doc_revision));
    let range = use_memo(move || {
        let _ = revision();
        scrub_range(state)
    });
    let labels = use_memo(move || {
        let _ = revision();
        match state.renderer.peek().as_ref() {
            Some(r) => r.scrub_labels(),
            None => Vec::new(),
        }
    });

    let close = rsx! {
        button {
            class: "chip timeline-close",
            title: "Leave Timeline mode",
            onclick: move |_| set_open(state, false),
            "\u{2715}"
        }
    };

    // A shared session has no single playhead to move: the document is a function of
    // a log peers are still appending to, so a scrub would be undone by the next
    // arrival (§12.1). Say so, rather than offer a control that does
    // nothing.
    let Some((at, total)) = range() else {
        return rsx! {
            div { class: chrome_class(state, "timeline-bar"),
                span { class: "bar-label", "Timeline" }
                span { class: "timeline-note",
                    "Unavailable while the drawing is shared \u{2014} the history is everyone's."
                }
                {close}
            }
        };
    };

    let playing = (state.timeline.playing)();
    let speed = (state.timeline.speed)();
    let behind = at < total;
    let pct = if total == 0 {
        0.0
    } else {
        at as f32 / total as f32 * 100.0
    };
    // What the playhead has just applied — the step it is *standing on*. Named
    // rather than tooltipped onto the ticks: a mark one pixel wide is not something
    // to hover, and the one caption that is always worth reading is the one for
    // where you are.
    let caption = match at.checked_sub(1) {
        Some(i) => labels.read().get(i).copied().unwrap_or("Step"),
        None => "Empty canvas",
    };
    // One tick per action while they are still marks; past that whole actions are
    // skipped rather than the marks thinned, so the ones drawn stay evenly spaced.
    let step = total.div_ceil(MAX_TICKS).max(1);
    let denom = total.max(1) as f32;

    let play_title = if playing {
        "Pause"
    } else {
        "Play the history from here"
    };
    let play_glyph = if playing {
        "\u{275A}\u{275A}"
    } else {
        "\u{25B6}"
    };
    let count_class = if behind {
        "timeline-count behind"
    } else {
        "timeline-count"
    };
    let count_title = if behind {
        "Painting here discards the steps ahead, as painting after an undo does"
    } else {
        "At the newest step"
    };

    rsx! {
        div { class: chrome_class(state, "timeline-bar"),
            button {
                class: "chip timeline-play",
                title: "{play_title}",
                disabled: total == 0,
                onclick: move |_| if playing { stop(state) } else { play(state) },
                "{play_glyph}"
            }

            div { class: "timeline-track",
                // Purely a ruler: one mark per action, lit behind the playhead. It
                // carries no tooltip because a one-pixel mark is not a hover
                // target — what each step *is* is said once, in the caption, for
                // the step the playhead is on.
                div { class: "timeline-ticks",
                    for i in (0..total).step_by(step) {
                        {
                            let left = (i + 1) as f32 / denom * 100.0;
                            let class = if i < at { "timeline-tick past" } else { "timeline-tick" };
                            rsx! { div { key: "{i}", class, style: "left: {left}%" } }
                        }
                    }
                }
                div { class: "timeline-fill", style: "width: {pct}%" }
                // A range input rather than a hand-rolled track: it is the one
                // control that is width-independent without measuring the DOM, and
                // it brings arrow-key stepping — a finer instrument for finding one
                // stroke than any pair of buttons — for nothing.
                input {
                    class: "timeline-range",
                    r#type: "range",
                    min: "0",
                    max: "{total}",
                    step: "1",
                    value: "{at}",
                    title: "Drag to scrub; \u{2190}/\u{2192} step one action",
                    oninput: move |e| {
                        if let Ok(to) = e.value().parse::<usize>() {
                            // A hand on the scrubber takes the playhead off the
                            // transport rather than fighting it for it.
                            stop(state);
                            dispatch(state, DocCommand::Seek(to));
                        }
                    },
                }
            }

            span { class: count_class, title: "{count_title}",
                span { class: "timeline-position", "{at} / {total}" }
                span { class: "timeline-caption", "{caption}" }
            }

            div { class: "tool-row segmented timeline-speed",
                for (mult, label) in SPEEDS {
                    {
                        let class = if speed == mult { "chip active" } else { "chip" };
                        rsx! {
                            button {
                                key: "{label}",
                                class,
                                title: "Play at {label} speed",
                                onclick: move |_| {
                                    let mut s = state.timeline.speed;
                                    s.set(mult);
                                },
                                "{label}"
                            }
                        }
                    }
                }
            }

            {close}
        }
    }
}
