//! Small reusable controls shared by the panels and the brush editor.

use dioxus::prelude::*;

use crate::icons::{icon, label as label_span};

/// A labelled range control.
///
/// `glyph` is an `Option` because the brush editor's dense parameter list has not been
/// marked yet, **not** because a slider is expected to go without one. A control's mark
/// is the half of it that survives its label, so anything reachable wants one; a `None`
/// here is a row that would be blank if the words were hidden, and is a to-do rather
/// than a decision (see [`crate::icons::SIZE`]).
///
/// Which is exactly why the word is wrapped as hideable ([`crate::icons::label`]) only
/// when there *is* a mark to fall back on. An unmarked slider keeps its word in minimal
/// mode — not as a special case, but because the two facts are one fact here, and a
/// component that reads them off each other cannot be given the wrong pair. The rows
/// still to be marked therefore stay legible in the meantime instead of turning into a
/// column of anonymous tracks.
///
/// `marked` on the row carries that same fact out to the stylesheet, which needs it for
/// a second reason: in minimal mode a marked row folds onto **one line**, its glyph to
/// the left of the track instead of over it, which is where the mode's vertical saving
/// in the panel stack actually comes from. A row that kept its word cannot fold — the
/// words differ in length, so the tracks would start at a ragged left edge — and it
/// does not have to, because the class it would need is the one it does not get.
#[component]
pub fn Slider(
    label: String,
    #[props(default)] glyph: Option<&'static str>,
    min: f32,
    max: f32,
    value: f32,
    oninput: EventHandler<f32>,
) -> Element {
    rsx! {
        div { class: if glyph.is_some() { "slider-row marked" } else { "slider-row" },
            div { class: "slider-label",
                match glyph {
                    Some(glyph) => rsx! { {icon(glyph)} {label_span(&label)} },
                    None => rsx! { "{label}" },
                }
            }
            input {
                class: "slider",
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
                },
            }
        }
    }
}

// --- command dispatch ---

/// End a preview drag: commit the value the previews have been showing, once, and
/// disarm — the release half of the preview-per-sample, commit-once bargain the
/// opacity slider (§14.6), the filter bar (§21.6) and the frame drag all make.
///
/// A no-op when nothing is pending, which is what lets a control wire it to all
/// three of the events that can end a drag (`change`, `pointerup`, `pointercancel`)
/// without spending an undo step per event — `change` alone is not enough, because
/// it is not sent when a drag ends on the value it started on, which would strand a
/// preview with no commit to supersede it. The commit supersedes the preview
/// engine-side, so nothing has to drop it here; and a commit to the value the
/// document already holds is refused there too, so a drag that travelled out and
/// came back logs nothing.
///
/// One function rather than a copy per slider, because the subtleties above are the
/// kind that get re-learned wrong: the next drag control gets them by calling this.
pub fn settle<T: 'static>(
    state: crate::state::AppState,
    mut pending: Signal<Option<T>>,
    commit: impl FnOnce(T) -> stark_core::command::DocCommand,
) {
    let settled = pending.write().take();
    if let Some(value) = settled {
        crate::state::dispatch(state, commit(value));
    }
}
