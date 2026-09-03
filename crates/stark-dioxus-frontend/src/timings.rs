//! The Timing Stats dialog: where this session's time has actually gone
//! (`stark_engine::timing`, §7.1).
//!
//! Off the command search, on the rail's own rule (main.rs): an infrequent command, read
//! when a question comes up rather than kept on screen. It is deliberately **not** a
//! panel — a live frame-rate readout beside the canvas is a thing to watch instead of
//! painting, and the whole point of the histograms is that they keep accruing while
//! nobody is looking.
//!
//! Three things it is built around.
//!
//! - **The engine's phase list is not transcribed here.** The table renders whatever
//!   rows the histograms hold, in name order, so adding a `timing::span!` anywhere in
//!   the pipeline makes a row appear with no edit to this file. A dialog that listed
//!   the phases would be a second copy of the instrumentation, and the copy that
//!   drifted would be this one.
//! - **Nothing here is a rate the app keeps.** [`stark_engine::timing::Timings`] carries
//!   the wall-clock window it covers, and every "per second" and "% of window" below
//!   is arithmetic over that — so the numbers stay honest across a Reset, a
//!   backgrounded tab, and the minutes the dialog was closed.
//! - **The clock's resolution is on screen.** In a browser it is 100 µs at best and a
//!   whole millisecond in Firefox, and without that caption a row reading `0.0 ms`
//!   invites the wrong conclusion. See the module note on `stark_engine::timing`.

use std::time::Duration;

use dioxus::prelude::*;
use stark_engine::timing::{self, Phase};

use crate::icons::{self, icon};
use crate::widgets::Modal;

/// How often the open dialog re-reads the histograms.
///
/// Twice a second: fast enough that a stroke drawn with the dialog open visibly moves
/// the rows, slow enough that reading them — which flushes every recorder behind a
/// write lock — is not itself a thing the profile has to account for. The numbers are
/// cumulative anyway, so a slower tick loses nothing but immediacy.
const REFRESH_MS: i32 = 500;

/// The rows whose *count* is the headline, paired with what that count means.
///
/// The end-to-end story in two numbers, and the reason the dialog is not just a
/// table: `frame` is the rate the canvas actually reached the screen at and
/// `input.sample` is the rate pointer reports actually reached the engine at, and
/// every phase below is read as an explanation of one of those two.
const HEADLINE: [(&str, &str, &str); 2] = [
    ("frame", "Frames", "painted per second"),
    ("input.sample", "Pointer", "samples per second"),
];

/// The Timing Stats dialog, opened from the command search and dismissed by Done or by
/// clicking the backdrop (as the other dialogs are).
#[component]
pub fn TimingModal(on_close: EventHandler<()>) -> Element {
    // Seeded synchronously so the dialog opens with numbers rather than with a blank
    // frame that fills in half a second later — the histograms have been collecting
    // since the page loaded, and there is nothing to wait for.
    let mut stats = use_signal(timing::snapshot);
    // `use_future`, so the tick is owned by *this* scope and stops when the dialog
    // closes. The opposite of what the collaboration pumps need (`state::root_signal`)
    // and for the opposite reason: a poll that outlived its reader would be work done
    // for nobody.
    use_future(move || async move {
        loop {
            crate::platform::sleep_ms(REFRESH_MS).await;
            stats.set(timing::snapshot());
        }
    });

    let Some(t) = stats() else {
        return rsx! {
            Shell { on_close,
                div { class: "timing-empty",
                    "No timing subscriber is installed in this build, so nothing is being measured."
                }
            }
        };
    };

    rsx! {
        Shell { on_close,
            div { class: "timing-headline",
                for (name , label , unit) in HEADLINE {
                    div { class: "timing-headline-cell",
                        div { class: "timing-headline-value",
                            match t.rate(name) {
                                // A phase that has not run has no rate, which is a
                                // different statement from a rate of zero — a session
                                // where nobody has touched the pen has never sampled.
                                Some(r) => format!("{r:.0}"),
                                None => "\u{2014}".to_string(),
                            }
                        }
                        div { class: "timing-headline-label", "{label}" }
                        div { class: "timing-headline-unit", "{unit}" }
                    }
                }
            }

            div { class: "modal-section-label", "PHASES" }
            div { class: "timing-table",
                div { class: "timing-row timing-head",
                    span { class: "timing-name", "Phase" }
                    span { "Count" }
                    span { "Mean" }
                    span { "p99" }
                    span { "Max" }
                    span { "Share" }
                }
                if t.phases.is_empty() {
                    div { class: "timing-empty", "Nothing has run in this window yet." }
                }
                for phase in t.phases.iter() {
                    PhaseRow { key: "{phase.name}", phase: phase.clone(), window: t.window }
                }
            }

            // The caption, and the two facts without which a row above can be
            // misread: what the numbers are averaged over, and how finely the clock
            // behind them can measure at all.
            div { class: "timing-note",
                "Over {duration(t.window)} of wall clock, at a clock resolution of "
                "{duration(t.quantum)}. Share is of that window; nested phases "
                "overlap, so the column sums past 100%."
            }

            div { class: "modal-actions",
                button {
                    class: "btn btn-secondary",
                    // Re-read immediately rather than waiting out the tick: a Reset
                    // whose table keeps its old numbers for half a second reads as a
                    // button that did not work.
                    onclick: move |_| {
                        timing::reset();
                        stats.set(timing::snapshot());
                    },
                    {icon(icons::RESET)}
                    "Reset"
                }
                button {
                    class: "btn btn-primary",
                    onclick: move |_| on_close.call(()),
                    {icon(icons::DONE)}
                    "Done"
                }
            }
        }
    }
}

/// The dialog's chrome, so the "nothing is being measured" case is the same dialog
/// with different contents rather than a second one that has to be kept looking like
/// this one.
#[component]
fn Shell(on_close: EventHandler<()>, children: Element) -> Element {
    rsx! {
        // Wide, for Credits' reason turned around: that dialog is wide to
        // respect text it must not reflow, this one to fit six numeric
        // columns without any of them wrapping.
        Modal { class: "modal-wide", on_close,
            div { class: "modal-title", "Timing Stats" }
            div { class: "modal-subtitle",
                "Where this session's time has gone, measured in this browser. \
                 Every figure is CPU time to prepare the work — what the GPU then \
                 spends executing it is not measurable from here."
            }
            {children}
        }
    }
}

/// One phase.
///
/// A component rather than inline markup for [`crate::credits::Credit`]'s reason: the
/// row is a list entry that only ever grows in number, and one laid out by hand is one
/// that ends up laid out differently.
#[component]
fn PhaseRow(phase: Phase, window: Duration) -> Element {
    // Dotted names group the eye (`stroke.region`, `stroke.loop`), so the prefix is
    // dimmed and the leaf is not — the same shape as an indented tree without
    // claiming a nesting the histograms do not record (see `stark_engine::timing`).
    let (prefix, leaf) = match phase.name.rsplit_once('.') {
        Some((head, tail)) => (Some(format!("{head}.")), tail.to_string()),
        None => (None, phase.name.to_string()),
    };
    let share = window
        .as_secs_f64()
        .gt(&0.0)
        .then(|| phase.total.as_secs_f64() / window.as_secs_f64() * 100.0);

    rsx! {
        div { class: "timing-row",
            span { class: "timing-name",
                if let Some(prefix) = prefix {
                    span { class: "timing-prefix", "{prefix}" }
                }
                "{leaf}"
            }
            span { "{phase.count}" }
            span { {duration(phase.mean)} }
            span { {duration(phase.p99)} }
            span { {duration(phase.max)} }
            span {
                match share {
                    Some(s) => format!("{s:.1}%"),
                    None => "\u{2014}".to_string(),
                }
            }
        }
    }
}

/// A duration at a scale a reader can hold in their head, with a fixed number of
/// significant digits so a column of them lines up.
///
/// The break between microseconds and milliseconds is at one millisecond, which is
/// the unit a frame budget is thought in: above it the ms decimals are the
/// interesting digits, and below it they would all be zero where the same figure in
/// microseconds is exact. The two bands above are for the caption rather than for a
/// phase — a *window* is minutes into a session, and a phase that reached them would
/// have saturated the histogram long before.
fn duration(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns >= 90_000_000_000.0 {
        format!("{:.1} min", ns / 6e10)
    } else if ns >= 1_000_000_000.0 {
        format!("{:.1} s", ns / 1e9)
    } else if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1e6)
    } else if ns >= 1_000.0 {
        format!("{:.0} \u{b5}s", ns / 1e3)
    } else {
        // Not "0 µs": the browser's clock quantizes to 100 µs or worse, so a phase
        // that never crosses it reads as *below the resolution* rather than as free
        // — which is what the dialog's caption is there to let this say in one glyph.
        "\u{2039}\u{2009}tick".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scale breaks are where a reader's units change, and every one of them has
    /// to render *something* — a formatter that fell through to an empty string would
    /// leave a blank cell that reads as "not measured".
    #[test]
    fn every_scale_reads_as_a_quantity() {
        assert_eq!(duration(Duration::from_micros(1500)), "1.50 ms");
        assert_eq!(duration(Duration::from_micros(250)), "250 \u{b5}s");
        assert_eq!(duration(Duration::from_millis(1500)), "1.5 s");
        // A window, which is what the two coarse bands exist for. 12.5 minutes is an
        // ordinary length for a session with the dialog opened once at the end of it.
        assert_eq!(duration(Duration::from_secs(750)), "12.5 min");
        // The break sits exactly at a millisecond, so both sides of it are pinned.
        assert_eq!(duration(Duration::from_micros(1000)), "1.00 ms");
        assert_eq!(duration(Duration::from_micros(999)), "999 \u{b5}s");
        // A browser's clock quantum, which is the figure the caption reports most
        // often and the one the ms break would have rendered as "0.10 ms".
        assert_eq!(duration(Duration::from_micros(100)), "100 \u{b5}s");
        // Under the finest browser clock: shown as under the tick, never as zero.
        assert_eq!(duration(Duration::from_nanos(300)), "\u{2039}\u{2009}tick");
        assert_eq!(duration(Duration::ZERO), "\u{2039}\u{2009}tick");
    }

    /// Both headline rows have to name phases that are actually instrumented, or the
    /// dialog opens with two em-dashes and no way to tell a quiet session from a
    /// renamed span.
    ///
    /// Checked against the *names* rather than against a snapshot, because a snapshot
    /// needs a painted frame and this test has no GPU: what can drift is the string,
    /// and it drifts at the `timing::span!` call site in `state.rs`.
    #[test]
    fn the_headline_names_are_the_ones_the_frontend_opens() {
        let source = include_str!("state.rs");
        for (name, _, _) in HEADLINE {
            assert!(
                source.contains(&format!("timing::span!(\"{name}\")")),
                "the Timing Stats headline reads `{name}`, which nothing in state.rs \
                 opens any more — the row will always be an em-dash",
            );
        }
    }
}
