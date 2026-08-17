//! The gradient library (§22.3): named color ramps, captured off the canvas.
//!
//! A gradient is made by **tracing a line through the painting** — through a
//! palette the artist has already mixed on it — not by placing control points
//! one picker dialog at a time. The trace itself is the Gradients panel's armed
//! mode (`crate::panels::gradients`); what lives here is the library around it:
//! the entries, their persistence, and the capture call that turns a finished
//! trace into one of them through [`Engine::pick_gradient`] (§22.2).
//!
//! Like the brush presets and the shape library, the entries are frontend state
//! that follows *this browser* across documents via `localStorage`, and never
//! enter the document or reach a peer: a gradient is something the artist paints
//! **with**, not part of what they have painted. What consumes one — the
//! gradient fill's parcel (§22.4), a matte's graded paint (§15.4) — embeds the
//! ramp **by value** in the action it commits, the way a stroke embeds the
//! brush color, so a document stays self-contained and this library stays
//! personal.
//!
//! Unlike the preset library there are no built-in entries at all: a gradient's
//! whole story is that it came off *your* canvas, and a panel opening on a
//! stranger's sunset would tell the opposite one. The empty state says how to
//! begin instead.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use stark_engine::color::srgb_to_oklab;
use stark_engine::geom::Vec2;
use stark_engine::gradient::Gradient;
use stark_engine::{PickOptions, PickSource};

use crate::platform::{base64_decode, base64_encode};
use crate::state::AppState;
use crate::storage;

/// One key, namespaced and versioned like the preset library's.
const KEY_GRADIENTS: &str = "stark.gradients.v1";

/// How wide a patch each trace sample averages, in canvas px (radius 2 = 5×5).
///
/// Fixed rather than borrowed from the eyedropper bar's radius: the bar
/// configures a *pick*, and a trace needs the averaging for a different reason
/// — a ramp fitted through single texels would grow stops out of paint grain.
/// The fitter smooths along the trace; this smooths across it.
const TRACE_RADIUS: u32 = 2;

/// A trace shorter than this (canvas px of arc) is a click that wandered, not a
/// gradient — release ends the mode without a capture.
const MIN_TRACE_LEN: f32 = 8.0;

/// One named gradient in the library.
#[derive(Clone, PartialEq)]
pub struct GradientEntry {
    /// Display name; unique in the library ([`next_name`] proposes the first
    /// free "Gradient N", and a capture never overwrites).
    pub name: String,
    pub gradient: Gradient,
}

/// The gradient library's signals (`crate::gradients`). Root-owned like every
/// library here: the capture is spawned detached, and the trace overlay that
/// arms it unmounts the moment the mode ends.
#[derive(Clone, Copy)]
pub struct GradientsState {
    /// Library entries, loaded from `localStorage` at startup.
    pub entries: Signal<Vec<GradientEntry>>,
    /// The entry the next gradient fill uses (§22.4), by name; `None` — or a
    /// name a removal has since orphaned — falls back to the first entry
    /// ([`current`]). Per-session, deliberately: which ramp is in hand is
    /// working state like the brush color, not a fact about the library.
    pub selected: Signal<Option<String>>,
    /// Whether the trace mode is armed: the canvas catcher is mounted and the
    /// next drag across the painting is sampled into a gradient.
    pub armed: Signal<bool>,
    /// Whether a capture's sampling is in flight (between release and answer).
    pub busy: Signal<bool>,
    /// A transient line under the library: "the trace found no paint", storage
    /// trouble. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

/// Make `name` the gradient the next fill uses, and re-preview a composing
/// fill so the canvas answers the click immediately (§22.4).
pub fn select(state: AppState, name: &str) {
    let mut sel = state.gradients.selected;
    sel.set(Some(name.to_string()));
    crate::panels::gradient_bar::refresh(state);
    // …and hand it to a selected gradient-map filter (§21.11): while its bar is
    // up, a row click here is the choosing gesture, the way it is for a
    // composing fill or matte. The two cannot race — `apply_ramp` stands down
    // while the gradient bar is composing.
    crate::panels::filter::apply_ramp(state);
}

/// The gradient a fill would use right now: the selected entry, or the first —
/// so a library with anything in it always answers, and the Gradients panel
/// highlights the same row this resolves to.
pub fn current(state: AppState) -> Option<Gradient> {
    current_name(state).and_then(|name| {
        state
            .gradients
            .entries
            .read()
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.gradient.clone())
    })
}

/// The name [`current`] resolves to, for the panel's highlight.
pub fn current_name(state: AppState) -> Option<String> {
    let entries = state.gradients.entries.read();
    let sel = state.gradients.selected.read();
    sel.as_ref()
        .filter(|n| entries.iter().any(|e| &e.name == *n))
        .cloned()
        .or_else(|| entries.first().map(|e| e.name.clone()))
}

/// Load what this browser has saved. Call once at app start, before the
/// renderer exists — entries are pure data and the panel should be populated
/// on first open.
pub fn load(state: AppState) {
    let Some(entries) = read_storage() else {
        return;
    };
    let mut list = state.gradients.entries;
    list.set(entries);
}

/// Arm or disarm the trace mode. Arming clears the notice — the answer to "no
/// paint found" is another trace, and the line has done its job the moment one
/// starts.
pub fn set_armed(state: AppState, on: bool) {
    if on {
        // One composing mode at a time (`crate::modes`): the trace's catcher is
        // the last of the four to be stacked, so arming while a transform is
        // composing would leave two of them over one pointer. `leave` writes
        // this signal itself rather than calling back through here.
        crate::modes::leave(state);
    }
    let mut armed = state.gradients.armed;
    armed.set(on);
    if on {
        let mut notice = state.gradients.notice;
        notice.set(None);
    }
}

/// Turn a finished trace into a library entry — the capture (§22.2).
///
/// Samples the **composite**, the eyedropper's default: the ramp should hold
/// the paint the canvas shows, not one layer's contribution to it. The engine
/// resamples the path by arc length, picks every sample exactly as an
/// eyedropper pick, and fits the fewest stops that reproduce the run
/// (`Engine::pick_gradient`); what comes back is added under the first free
/// "Gradient N", or the notice says why nothing was.
pub fn capture(state: AppState, path: Vec<Vec2>) {
    let mut busy = state.gradients.busy;
    if *busy.peek() {
        return;
    }
    // Render now and drop the guard before awaiting — the readback future owns
    // what it needs, and the UI re-renders while the browser runs the copy (the
    // same bargain as `input::pick_color`).
    let Some(readback) = crate::state::with_engine_quiet(state, |r| {
        r.pick_gradient(
            &path,
            PickOptions {
                source: PickSource::Composite,
                radius: TRACE_RADIUS,
            },
        )
    }) else {
        return;
    };
    busy.set(true);
    // Detached: the answer must outlive the release gesture that asked for it,
    // and every signal written is root-owned (`state::root_signal`).
    spawn_forever(async move {
        let fitted = readback.await;
        busy.set(false);
        let mut notice = state.gradients.notice;
        let Some(gradient) = fitted else {
            notice.set(Some(
                "The trace found no paint to sample \u{2014} draw it through the painting."
                    .to_string(),
            ));
            return;
        };
        notice.set(None);
        let mut entries = state.gradients.entries;
        {
            let mut list = entries.write();
            let name = next_name(&list);
            list.push(GradientEntry { name, gradient });
        }
        persist(&entries.read());
    });
}

/// Drop one gradient from the library.
pub fn remove(state: AppState, name: &str) {
    let mut entries = state.gradients.entries;
    entries.write().retain(|e| e.name != name);
    persist(&entries.read());
}

/// The first free "Gradient N" name — captures are named by the machinery, so
/// the artist traces twice without a dialog between.
pub fn next_name(entries: &[GradientEntry]) -> String {
    (1..)
        .map(|i| format!("Gradient {i}"))
        .find(|n| !entries.iter().any(|e| &e.name == n))
        .unwrap()
}

/// The CSS that draws an entry's strip: a `linear-gradient(in oklab, …)` whose
/// stops are the gradient's own, as `oklab()` literals.
///
/// CSS interpolating in Oklab between the same stop colors *is*
/// [`Gradient::sample`]'s interpolation, so the strip previews exactly the ramp
/// the engine fitted — the invariant that makes the preview trustworthy, and
/// the reason the colors are handed over in Oklab rather than as hex CSS
/// would re-convert.
pub fn css_strip(g: &Gradient) -> String {
    let stops: Vec<String> = g
        .stops()
        .iter()
        .map(|s| {
            let lab = srgb_to_oklab([s.color[0], s.color[1], s.color[2], 1.0]);
            format!(
                "oklab({:.4} {:.4} {:.4}) {:.2}%",
                lab[0],
                lab[1],
                lab[2],
                s.t * 100.0
            )
        })
        .collect();
    format!("linear-gradient(in oklab to right, {})", stops.join(", "))
}

// --- persistence ----------------------------------------------------------
//
// One `crate::storage` table, a line per entry: `b64(name)|b64(json(stops))`.
// The format and the rule it exists for — one damaged entry is skipped rather
// than poisoning the library — are stated there, once, for all four libraries.
// What is this module's own is the *fields*: the stops are JSON rather than
// postcard because `localStorage` outlives app versions, and deserialization
// funnels through `Gradient`'s own gate, so a tampered line becomes a skipped
// line rather than an unsampleable ramp.

fn persist(entries: &[GradientEntry]) {
    storage::save_table(
        KEY_GRADIENTS,
        "the gradient library",
        entries.iter().filter_map(|e| {
            let json = serde_json::to_string(&e.gradient).ok()?;
            Some(storage::record([
                base64_encode(e.name.as_bytes()).as_str(),
                base64_encode(json.as_bytes()).as_str(),
            ]))
        }),
    );
}

fn read_storage() -> Option<Vec<GradientEntry>> {
    storage::load_table(KEY_GRADIENTS, parse_entry)
}

fn parse_entry(line: &str) -> Option<GradientEntry> {
    let mut fields = line.split(storage::FIELD);
    let name = String::from_utf8(base64_decode(fields.next()?).ok()?).ok()?;
    let gradient = serde_json::from_slice(&base64_decode(fields.next()?).ok()?).ok()?;
    Some(GradientEntry { name, gradient })
}

/// Arc length of a traced polyline, for the "was that a trace or a click"
/// threshold ([`MIN_TRACE_LEN`]).
pub fn trace_length(points: &[Vec2]) -> f32 {
    points.windows(2).map(|w| w[0].distance(w[1])).sum()
}

/// Whether a finished trace is long enough to capture.
pub fn trace_long_enough(points: &[Vec2]) -> bool {
    trace_length(points) >= MIN_TRACE_LEN
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_engine::gradient::GradientStop;

    fn gradient() -> Gradient {
        Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: [1.0, 0.0, 0.0],
            },
            GradientStop {
                t: 1.0,
                color: [0.0, 0.0, 1.0],
            },
        ])
        .unwrap()
    }

    #[test]
    fn the_strip_is_the_gradient_in_css() {
        let css = css_strip(&gradient());
        assert!(css.starts_with("linear-gradient(in oklab to right, oklab("));
        assert!(css.contains("0.00%") && css.contains("100.00%"));
    }

    #[test]
    fn names_count_past_the_holes() {
        let mut entries = vec![
            GradientEntry {
                name: "Gradient 1".into(),
                gradient: gradient(),
            },
            GradientEntry {
                name: "Gradient 3".into(),
                gradient: gradient(),
            },
        ];
        assert_eq!(next_name(&entries), "Gradient 2");
        entries.remove(0);
        assert_eq!(next_name(&entries), "Gradient 1");
    }

    #[test]
    fn a_stored_line_survives_the_round_trip_and_a_bad_one_is_skipped() {
        let entry = GradientEntry {
            name: "Dusk".into(),
            gradient: gradient(),
        };
        let json = serde_json::to_string(&entry.gradient).unwrap();
        let line = format!(
            "{}|{}",
            base64_encode(entry.name.as_bytes()),
            base64_encode(json.as_bytes())
        );
        let back = parse_entry(&line).expect("round trip");
        assert_eq!(back.name, "Dusk");
        assert_eq!(back.gradient, entry.gradient);

        // One stop is not a gradient; the line is dropped by `Gradient`'s own
        // deserialization gate rather than parsed into an unsampleable ramp.
        let bad = format!(
            "{}|{}",
            base64_encode(b"Bad"),
            base64_encode(br#"[{"t":0.5,"color":[0.0,0.0,0.0]}]"#)
        );
        assert!(parse_entry(&bad).is_none());
    }
}
