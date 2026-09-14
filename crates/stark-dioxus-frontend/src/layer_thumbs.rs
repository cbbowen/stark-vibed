//! Rendered thumbnails for the **layers** of the document being painted: one
//! small picture per row of the Layers panel, showing that layer's own paint
//! (§14.6).
//!
//! # What a row's picture is of
//!
//! **The layer alone, with its surroundings dropped.** Blend mode, clip and
//! opacity all say how much of a layer the *document* shows; none of them says
//! anything about what the paint is. So a thumbnail renders through
//! [`Engine::export_view`](stark_engine::Engine::export_view)'s `only`, which drops all three — the same isolate the
//! eyedropper samples through, and settled there first (§18.0.2). Turning a layer
//! down does not turn its paint into a paler paint, and its row must not claim
//! otherwise.
//!
//! **Framed on the document, not on the layer.** Every row shows the same window
//! — the rect an export would write, through the same [`Engine::export_plan`](stark_engine::Engine::export_plan) the
//! navigator asks (§11) — so the rows are mutually comparable and you can see at a
//! glance that the highlights live in the top-left. Framing each row on its own
//! content would fill every thumbnail and make none of them relate to another, and
//! would rescale a row's picture as its layer is painted outward.
//!
//! **Cut out, not on the substrate** (`Background::Transparent`): a row shows where
//! the layer *has* paint, and a substrate under it would fill every thumbnail edge to
//! edge and hide exactly that. The stylesheet puts a flat tint behind it rather than
//! the usual transparency checker — at this size a checker is noise, and it would
//! have to be drawn with the same `background-image` the picture arrives in.
//!
//! # Why these are readbacks when the navigator's miniature is a surface
//!
//! The navigator deleted its readback and gained a `<canvas>` of its own, and the
//! reasons that won there invert here. There is one navigator and there are as
//! many of these as the document has layers, so a surface each means a WebGPU
//! context and a swapchain per row; rows are re-keyed and moved by the drag that
//! reorders them, and a CSS `background-image` survives a node moving where a
//! bound surface is per-node state to rebind. And the frame of latency a surface
//! buys off is worth nothing on a 32-px picture refreshed once per commit.
//!
//! # The cache, and why it costs nothing per stroke
//!
//! Keyed on `(LayerId, content_revision)` — a number that moves exactly when a
//! layer's own tiles do and stands when anything else about it changes
//! (`PaintTiles::revision`). So painting one layer regenerates one thumbnail and
//! every other row is a hit; a slider drag, a rename, a reorder and a run of undo
//! through property edits regenerate nothing at all. Keying on `doc_revision`
//! instead — the obvious choice, and what the navigator can afford because it is
//! one picture — would re-render every row on every commit.
//!
//! **That revision decides what to re-render, never what to show.** A row draws
//! whatever picture we last took of its layer, current or not (`url`); the
//! comparison lives in `wanted`, which decides what to render next. Keying
//! the display on it too is the natural reading and is wrong in one visible way: a
//! commit invalidates the entry a whole settle and a readback before its
//! replacement exists, so the row goes empty in between — once per stroke you
//! commit, and once per stroke a *peer* commits (§12), which is where it was
//! caught, since their work arrives while you are looking at the panel rather than
//! at your own hand.
//!
//! Unlike the brush thumbnails ([`crate::thumbs`]) there is no sibling engine
//! here: these are pictures of the live document, so they render through the
//! *main* renderer and must take its borrow. That is the whole of what makes the
//! pacing below matter.

use dioxus::prelude::*;

use stark_engine::Extent2;
use stark_engine::{ExportScale, LayerInfo};
use stark_model::document::LayerId;
use stark_model::geom::IVec2;

use crate::platform::sleep_ms;
use crate::state::AppState;
use crate::thumb_cache::{Insert, NoRenderer, Pace, Picture, ThumbCache, readback_url};

/// Thumbnail pixel size: 2× the box a row shows it in, so it stays crisp on a
/// dense display. Square, because the document it frames may be any shape and
/// `contain` letterboxes it in the row rather than cropping — a thumbnail that
/// cropped would hide the very corner a layer's paint is in.
const THUMB: u32 = 64;

/// How long a change has to stop arriving before thumbnails are re-rendered.
///
/// The navigator's own settle, and for its reasons — collapse a held undo or a
/// peer's arriving actions into one pass — with a second job here: a stroke
/// commits on pen-up and the hand usually goes straight back down, so this is also
/// what keeps a run of strokes from rendering a thumbnail between each.
///
/// Once per run, before the first row rather than before each: a panel opening on
/// a twenty-layer document would otherwise take twenty settles to fill.
const SETTLE_MS: i32 = 180;

/// How long the generator waits between rows.
///
/// **This is the pacing that keeps the panel off the canvas's back.** The
/// compositor's draw list lives in a single slot keyed on, among other things,
/// which layer is being drawn alone (`DrawKey`), so every thumbnail evicts the
/// screen's list and the next frame rebuilds it — and building it clones a tile
/// handle per visible tile per layer, which is the cost that cache exists to
/// avoid. One row per turn of the event loop keeps that to one rebuild per frame
/// on a document nobody is touching, instead of a burst of them back to back.
const BETWEEN_MS: i32 = 16;

/// The layer thumbnail machinery's signals. All root-owned
/// (`state::root_signal`): generation runs in `spawn_forever` tasks that outlive
/// the panel, which is closed as often as it is open.
#[derive(Clone, Copy)]
pub struct LayerThumbState {
    /// Finished thumbnails: a `data:image/png` URL per layer, filed under the
    /// [`Subject`] it shows.
    ///
    /// Replaced rather than appended, unlike the brush cache: a brush library is a
    /// fixed set of things a user made, while a layer is repainted all day, so
    /// keeping every revision's picture would grow without bound over a session's
    /// painting.
    pictures: ThumbCache<Subject>,
}

impl LayerThumbState {
    /// Its signals, root-owned like every other group of them
    /// (`state::root_signal`); built here rather than in `AppState::new` so the
    /// fields and the values they open on stay in one place.
    pub(crate) fn new() -> Self {
        Self {
            pictures: ThumbCache::new(
                Insert::Replace(|a: &Subject, b: &Subject| shows(a, b.layer)),
                Pace {
                    settle_ms: SETTLE_MS,
                    between_ms: BETWEEN_MS,
                },
            ),
        }
    }
}

/// What a row's picture is of: the key it is filed under.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    pub layer: LayerId,
    /// `LayerInfo::content_revision` at the moment it was rendered.
    pub revision: u64,
    /// `LayerInfo::translation` at the moment it was rendered. Part of the
    /// staleness key beside the revision, because the row is framed on the
    /// *piece* (§14.6): moving a layer moves where its paint sits in that frame
    /// without minting a revision (§14.12.4), and a key of the revision alone
    /// left the row showing the paint where it used to be.
    pub translation: IVec2,
}

/// Whether `subject` is a picture of `layer`, whatever revision it was taken at.
///
/// The one question both the row ([`url`]) and the cache's replacement ask, so the
/// entry a row draws is the entry a fresher render replaces.
fn shows(subject: &Subject, layer: LayerId) -> bool {
    subject.layer == layer
}

/// The thumbnail to draw in `layer`'s row: **the last picture of it we rendered,
/// current or not.** Subscribes, so a row re-renders when a fresher one lands.
///
/// **Deliberately not keyed on the revision, though the regeneration below is.**
/// The instant a layer's tiles move, the picture we hold stops being current — but
/// it does not stop being a picture of that layer, and it is a settle plus a render
/// plus a readback away from being replaced. Serving nothing across that gap blanks
/// the row on every commit, which is unmissable when the commits are a peer's
/// (§12): the panel is the one place their work shows while you are looking
/// elsewhere, and the row strobed once per stroke they finished. This is the
/// statement §14.6 already makes about a *hidden* layer — the last thumbnail stands
/// — held for the same reason, since in both cases the alternative is an empty box
/// claiming the layer is empty.
///
/// `None` for a layer with no tiles at all (a matte, a filter) — said through the
/// same field that would have keyed the picture, `LayerInfo::content_revision`, so
/// the row asks one question rather than two. Those rows fill the slot with their
/// kind mark instead (§14.6), which is the panel's business and not this module's.
pub fn url(state: AppState, layer: &LayerInfo) -> Option<String> {
    // Asked for its `None` alone: a layer that holds no tiles has no picture to
    // show, whatever the cache may still hold under its id.
    layer.content_revision?;
    state
        .layer_thumbs
        .pictures
        .find(|subject| shows(subject, layer.id))
}

/// Whether `l` is worth rendering: it has tiles, it has some, and the isolate
/// would not come back blank.
///
/// **A hidden or fully transparent layer is skipped rather than rendered**, and
/// that is the whole of how the panel keeps its picture. `composite_groups` draws
/// nothing for one (deliberately — a sample must not report paint that is switched
/// off), so rendering would replace a good thumbnail with an empty one at exactly
/// the moment the row's own picture is the only reminder of what is in it. Not
/// rendering leaves the last one standing, and unhiding brings the row up to date
/// on the next pass.
fn worth_rendering(l: &LayerInfo) -> bool {
    l.content_revision.is_some() && l.visible && l.opacity > 0.0
}

/// Every layer worth a thumbnail, as the subject its picture would be of: the cache
/// renders the first it does not hold.
///
/// **This is where the revision is the key** — it decides what to *re-render*, as
/// against what to *show*, which is [`url`] above. A picture that has fallen behind
/// still draws and still counts as stale, and those are not in tension: they are
/// the two halves of replacing a thumbnail without a gap in the row.
///
/// Top-down, which is the order the panel draws in: the rows a user is looking at
/// fill first.
fn wanted(layers: &[LayerInfo]) -> Vec<Subject> {
    layers
        .iter()
        .rev()
        .filter(|l| worth_rendering(l))
        .filter_map(|l| {
            // The picture is a function of the tiles *and* of where the frame
            // puts them in the piece (§14.12), so both key it.
            Some(Subject {
                layer: l.id,
                revision: l.content_revision?,
                translation: l.translation,
            })
        })
        .collect()
}

/// Make sure every layer with a picture to show has an up-to-date thumbnail,
/// rendering the stale ones in the background ([`ThumbCache::refresh`]). Cheap when
/// nothing is stale; safe to call from a render effect.
pub fn refresh(state: AppState) {
    state.layer_thumbs.pictures.refresh(
        move || {
            state
                .obs
                .peek()
                .as_ref()
                .map_or_else(Vec::new, |obs| wanted(&obs.layers))
        },
        // Never render with a hand on the canvas. This render takes the engine's
        // own borrow, so one landing mid-stroke spends its cost exactly where it
        // is least affordable — and `canvas_active` covers strokes, marquees,
        // pans and runs of wheel zoom alike.
        //
        // **Yield until it lifts rather than abandoning the pass**: the effect that
        // starts a pass fires on the layer list moving, and a pan does not move it.
        // And wait *before* the layer is chosen, so one a peer hid or emptied during
        // the stroke is not rendered blank over its last picture.
        async move || {
            while *state.canvas_active.peek() {
                sleep_ms(SETTLE_MS).await;
            }
        },
        async move |subject: &Subject| generate(state, subject.layer).await,
    );
}

/// Render `layer`'s thumbnail as a `data:` URL, `None` when there is nothing to frame
/// or the readback failed. [`NoRenderer`] when there is no renderer to draw with.
async fn generate(state: AppState, layer: LayerId) -> Result<Picture, NoRenderer> {
    // The frame the navigator frames itself against, so an overview and a row's
    // picture cannot come to disagree about where the piece is. Read before the
    // engine borrow, not inside it: `piece_frame` reads `obs`, which the borrow
    // below has no need to hold open.
    let frame = state
        .obs
        .peek()
        .as_ref()
        .and_then(stark_ui::bounds::piece_frame);
    // Everything up to the readback happens under the engine borrow, which must
    // end before the await — `Engine::export_view`'s own bargain.
    //
    // Quiet: a thumbnail is a second render of state the panel is *reading*, and
    // publishing here would ask every chrome component to re-render for a picture
    // none of them is waiting on.
    let asked = crate::state::with_engine_quiet(state, |r| {
        let plan = r
            .session
            .engine()
            .export_plan(frame, ExportScale::Fit(Extent2::new(THUMB, THUMB)))
            .ok()?;
        r.export_layer(layer, &plan).ok()
    })
    .ok_or(NoRenderer)?;
    Ok(match asked {
        Some(readback) => readback_url(readback).await,
        // A renderer, but nothing to frame: an empty canvas with no frame has no
        // picture to be a thumbnail of. Filed as a miss so it is not asked for again.
        None => None,
    })
}

/// Drop cached thumbnails for layers the document no longer has.
///
/// Removal is the one edit that leaves an entry with nothing to serve it, and a
/// deleted layer's id is never reissued (`a_duplicates_ids_are_not_reused_after_a_reload`),
/// so nothing can come to read it — this is memory, not correctness.
pub fn prune(state: AppState, live: &[LayerInfo]) {
    state
        .layer_thumbs
        .pictures
        .retain(|subject| live.iter().any(|l| l.id == subject.layer));
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::BlendMode;

    /// A paint layer at a given content revision. Every other field is a statement
    /// about the tree, and nothing here reads one.
    fn paint(id: u64, revision: u64) -> LayerInfo {
        LayerInfo {
            id: LayerId::solo(id),
            blend: BlendMode::Normal,
            clip: false,
            opacity: 1.0,
            visible: true,
            carrier: None,
            depth: 0,
            is_group: false,
            has_backdrop: true,
            name: None,
            number: Some(1),
            matte: None,
            filter: None,
            has_underlay: true,
            merge_down: None,
            content_revision: Some(revision),
            translation: IVec2::ZERO,
        }
    }

    fn subject(id: u64, revision: u64) -> Subject {
        Subject {
            layer: LayerId::solo(id),
            revision,
            translation: IVec2::ZERO,
        }
    }

    /// The blank flash this split exists to rule out: a commit — a peer's, most
    /// visibly (§12) — moves the layer's revision, and the row must go on showing
    /// the picture we have until the replacement lands rather than emptying itself
    /// for a settle and a readback.
    #[test]
    fn a_commit_does_not_take_the_row_s_picture_away() {
        assert!(
            shows(&subject(1, 7), LayerId::solo(1)),
            "an older revision's picture is still the row's"
        );
        assert!(
            !shows(&subject(2, 7), LayerId::solo(1)),
            "another layer's is not"
        );
    }

    /// And the other half: still shown is not still current. The layer is wanted at
    /// its revision now, which the picture of the last one does not hold.
    #[test]
    fn but_it_is_still_the_row_to_render_next() {
        assert_eq!(wanted(&[paint(1, 8)]), [subject(1, 8)]);
    }

    /// A translate moves no tile and mints no revision (§14.12.4), but the row is
    /// framed on the piece — so it *is* the row to render next, at the frame it
    /// now stands at.
    #[test]
    fn a_translated_layer_is_the_row_to_render_next() {
        let d = IVec2::new(300, -40);
        let moved = LayerInfo {
            translation: d,
            ..paint(1, 8)
        };
        assert_eq!(
            wanted(&[moved]),
            [Subject {
                translation: d,
                ..subject(1, 8)
            }]
        );
    }

    /// A hidden or fully transparent layer is not re-rendered, so the picture it had
    /// when it last showed stands (§14.6). Its tiles moving underneath changes nothing.
    #[test]
    fn a_hidden_layer_asks_for_no_new_picture() {
        let hidden = LayerInfo {
            visible: false,
            ..paint(1, 8)
        };
        let clear = LayerInfo {
            opacity: 0.0,
            ..paint(2, 8)
        };
        assert!(wanted(&[hidden, clear]).is_empty(), "neither is wanted");
    }
}
