//! The gradient library (§22.3): named color ramps, captured off the canvas.
//!
//! A gradient is made by **tracing a line through the painting** — through a
//! palette the artist has already mixed on it — not by placing control points
//! one picker dialog at a time. The trace itself is the mode the library
//! pop-out arms (`crate::panels::gradients`); what lives here is the library
//! around it: the entries, their persistence, and the capture call that turns a
//! finished trace into one of them through [`Engine::pick_gradient`](stark_engine::Engine::pick_gradient) (§22.2).
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
//! whole story is that it came off *your* canvas, and a library opening on a
//! stranger's sunset would tell the opposite one. The empty state says how to
//! begin instead.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use stark_engine::{PickOptions, PickSource};
use stark_model::color::srgb_to_oklab;
use stark_model::geom::Vec2;
use stark_model::gradient::Gradient;

use crate::state::{AppState, root_signal};
use stark_ui::storage::{self, Store};

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

/// One named gradient in the library — and, unchanged, one stored entry: both fields
/// are durable, so a second struct to map it onto would be a copy with nothing to say.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GradientEntry {
    /// Display name; unique in the library ([`next_name`] proposes the first
    /// free "Gradient N", and a capture never overwrites).
    pub name: String,
    pub gradient: Gradient,
}

impl storage::Entry for GradientEntry {
    const STORE: Store = Store::Gradients;
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
    /// Whether a capture's sampling is in flight (between release and answer).
    pub busy: Signal<bool>,
    /// A transient line under the library: "the trace found no paint", storage
    /// trouble. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

impl GradientsState {
    /// Its signals, root-owned like every other group of them
    /// (`state::root_signal`); built here rather than in `AppState::new` so the
    /// fields and the values they open on stay in one place.
    pub(crate) fn new() -> Self {
        Self {
            entries: root_signal(Vec::new),
            selected: root_signal(|| None),
            busy: root_signal(|| false),
            notice: root_signal(|| None),
        }
    }
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
/// so a library with anything in it always answers, and the pop-out
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

/// The name [`current`] resolves to, for the pop-out's highlight.
pub fn current_name(state: AppState) -> Option<String> {
    let entries = state.gradients.entries.read();
    let sel = state.gradients.selected.read();
    sel.as_ref()
        .filter(|n| entries.iter().any(|e| &e.name == *n))
        .cloned()
        .or_else(|| entries.first().map(|e| e.name.clone()))
}

/// Load what this browser has saved. Call once at app start, before the
/// renderer exists — entries are pure data and the pop-out should be populated
/// on first open.
pub fn load(state: AppState) {
    let Some(entries) = read_storage() else {
        return;
    };
    let mut list = state.gradients.entries;
    list.set(entries);
}

/// Whether the trace mode is armed: the canvas catcher is mounted and the next
/// drag across the painting is sampled into a gradient (§22.2).
///
/// **Asked of `modes`, not of a flag here**, which is what it was. A trace is one
/// of the four composing modes, and a second place recording that it is live is a
/// second place for "one mode at a time" to be false in
/// (`crate::modes::Composing`). This is the render-time half — subscribing, so a
/// chip lit by it goes out when the mode does.
pub fn armed(state: AppState) -> bool {
    matches!(
        crate::modes::composing(state),
        Some(crate::modes::Composing::GradientTrace)
    )
}

/// Arm or disarm the trace mode. Arming clears the notice — the answer to "no
/// paint found" is another trace, and the line has done its job the moment one
/// starts.
///
/// A gradient bar composing when the trace is armed is **set aside and handed
/// back**, not abandoned (`panels::gradient_bar::suspend`). Every other mode
/// this displaces is put down for good, and rightly — reaching for the
/// transform is choosing to stop composing a fill. Reaching for Trace is not:
/// the trace was armed from the well *on that bar*, to get a ramp to lay with
/// it, and a mode that closed the bar the artist was working in to fetch what
/// they wanted to work with would take the tool away at the moment they had
/// finally chosen it.
pub fn set_armed(state: AppState, on: bool) {
    if on {
        // Suspended *before* the trace is entered, which is what saves the
        // gesture from the `leave` that entering performs — and parked *after*
        // it, since putting a mode down is what clears the stash. So the order
        // here is the one that neither loses the gesture nor leaves it behind
        // after the mode it belongs to has ended.
        let held = crate::panels::gradient_bar::suspend(state);
        // One composing mode at a time, which `enter` is now the whole of: the
        // trace's catcher is the last of the four to be stacked, so arming while
        // a transform is composing would otherwise leave two of them over one
        // pointer (`crate::modes`).
        crate::modes::enter(state, crate::modes::Composing::GradientTrace);
        let mut resume = state.gradient_resume;
        resume.set(held);
        let mut notice = state.gradients.notice;
        notice.set(None);
    } else {
        // The stash is taken out of `leave`'s reach first: the trace is the one
        // mode whose ending hands something back rather than dropping it, and
        // leaving is what clears the stash for every other way out.
        let mut resume = state.gradient_resume;
        let held = resume.write().take();
        crate::modes::leave(state);
        if let Some(ui) = held {
            crate::panels::gradient_bar::resume_from(state, ui);
        }
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
        let name = {
            let mut list = entries.write();
            let name = next_name(&list);
            list.push(GradientEntry {
                name: name.clone(),
                gradient,
            });
            name
        };
        persist(&entries.read());
        // And take it in hand. A trace is a *choosing* gesture — the line was
        // drawn to get this ramp — so the capture lands selected rather than at
        // the foot of the library with something else still the highlighted
        // row. Which is the same click a library row is, and reaches the same
        // consumers through it: the gradient bar the trace was armed from comes
        // back already previewing with it (`gradient_bar::refresh`), a
        // gradient-map filter takes it (§21.11).
        select(state, &name);
    });
}

/// Drop one gradient from the library.
pub fn remove(state: AppState, name: &str) {
    let mut entries = state.gradients.entries;
    entries.write().retain(|e| e.name != name);
    persist(&entries.read());
}

/// Rename one gradient. Trimmed; an empty or unchanged result costs nothing.
///
/// Names are this library's identity — the selection, a removal and the row
/// keys all speak them — so a name another entry already wears is refused, and
/// the notice says so rather than a silent no-op leaving the field's text
/// apparently ignored. A selection pointing at the old name follows the
/// rename: falling back to the first entry would swap the ramp in hand as a
/// side effect of relabelling it.
pub fn rename(state: AppState, from: &str, to: &str) {
    let to = to.trim();
    if to.is_empty() || to == from {
        return;
    }
    let mut entries = state.gradients.entries;
    {
        let mut list = entries.write();
        if list.iter().any(|e| e.name == to) {
            let mut notice = state.gradients.notice;
            notice.set(Some(format!(
                "There is already a gradient named \u{201c}{to}\u{201d} \u{2014} \
                 the name stands unchanged."
            )));
            return;
        }
        let Some(entry) = list.iter_mut().find(|e| e.name == from) else {
            return;
        };
        entry.name = to.to_string();
    }
    persist(&entries.read());
    let mut sel = state.gradients.selected;
    if sel.peek().as_deref() == Some(from) {
        sel.set(Some(to.to_string()));
    }
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
// [`GradientEntry`] is the stored entry, so there is nothing here but the two calls:
// the format and the rule it exists for — one damaged entry is skipped rather than
// poisoning the library — are stated in `stark_ui::storage`, once, for all four
// libraries. What this library leans on it for is `Gradient`'s own deserialization
// gate: a tampered entry is refused there and dropped here, rather than becoming an
// unsampleable ramp.

fn persist(entries: &[GradientEntry]) {
    storage::save_list(entries);
}

fn read_storage() -> Option<Vec<GradientEntry>> {
    storage::load_list()
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
    use stark_model::Srgb;
    use stark_model::gradient::GradientStop;

    fn gradient() -> Gradient {
        Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::new([1.0, 0.0, 0.0]),
            },
            GradientStop {
                t: 1.0,
                color: Srgb::new([0.0, 0.0, 1.0]),
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
    fn a_stored_entry_survives_the_round_trip_and_a_bad_one_is_repaired() {
        let entry = GradientEntry {
            name: "Dusk".into(),
            gradient: gradient(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: GradientEntry = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back.name, "Dusk");
        assert_eq!(back.gradient, entry.gradient);

        // One stop names no ramp, and the load path repairs it into one rather than
        // refusing (§22.1) — so the row survives `storage::load_list` as a flat ramp of
        // its own color, under the name the artist gave it. It used to be dropped, which
        // is silent loss of a row the artist can see and delete.
        let bad = r#"{"name":"Bad","gradient":[{"t":0.5,"color":[0.25,0.5,0.75]}]}"#;
        let back: GradientEntry = serde_json::from_str(bad).expect("a row still reads");
        assert_eq!(back.name, "Bad");
        assert_eq!(back.gradient.sample(0.0), Srgb::new([0.25, 0.5, 0.75]));
        assert_eq!(back.gradient.sample(1.0), Srgb::new([0.25, 0.5, 0.75]));
    }
}
