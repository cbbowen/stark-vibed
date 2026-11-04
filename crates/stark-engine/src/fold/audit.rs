//! **Every `apply` touches only what its `Footprint` declares** (§12.6), checked on
//! every fold — and every *unfold* — of every debug build.
//!
//! `Materialize::audit` is called from `Logged::apply` and `Logged::inverse` behind
//! `cfg(debug_assertions)`, so every action any test in the workspace folds — or
//! unfolds — is held to its own declaration, in shapes no hand-written vocabulary
//! enumerates.
//!
//! # What it costs
//!
//! One `DocState` clone (cheap by design, §5.1) and one walk of the layer
//! tree comparing tile handles by pointer, both debug-only. `Materialize::AUDITED`
//! keeps a consumer that does not audit from paying even the clone.
//!
//! # The direction it is wrong in
//!
//! A difference the footprint does not cover **panics**; a resource declared but not
//! touched says nothing. That asymmetry is §12.6's own: over-claiming costs the
//! commutation fast path, under-claiming diverges peers silently, and only one of
//! those is worth failing a test over.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(debug_assertions)]
use stark_model::document::Action;
use stark_model::document::{Footprint, LayerId, Prop, Resource};
use stark_model::geom::TileRect;

use crate::document::layer::Layer;
use crate::document::liquify::LiquifyRun;
use crate::document::patch::changed_tiles;
use crate::document::selection::Selection;
use crate::document::state::DocState;

/// Every difference between two states that `footprint` does not declare, described
/// for a human — empty when the fold was honest.
///
/// **The one enumeration of "what can differ"** — `tests/engine/footprint.rs` and the debug
/// fold below both call it, so a `Layer` or `DocState` field added without a case
/// here goes uncovered in one place rather than two.
///
/// `#[doc(hidden)]`: a test hook on a public module, public only because an
/// integration test can reach nothing else.
#[doc(hidden)]
pub fn undeclared(before: &DocState, after: &DocState, footprint: &Footprint) -> Vec<String> {
    differences(before, after)
        .into_iter()
        .filter(|d| !covered(d, &footprint.writes))
        .map(|d| format!("{d:?}"))
        .collect()
}

/// Whether two states are the same document — [`differences`] with nothing to
/// hide behind, which is what `apply::is_noop_on` asks of a fold it has already run.
pub(super) fn changes_nothing(before: &DocState, after: &DocState) -> bool {
    differences(before, after).is_empty()
}

/// Panic unless every difference `action` made — folding or unfolding — lies inside a
/// resource it declared.
#[cfg(debug_assertions)]
pub(super) fn audit(before: &DocState, after: &DocState, action: &Action, footprint: &Footprint) {
    let loose = undeclared(before, after, footprint);
    assert!(
        loose.is_empty(),
        "§12.6: applying {:?} changed state its footprint does not declare.\n\
         undeclared: {}\n\
         declared writes: {:?}\n\
         declared reads: {:?}",
        action.kind.tag(),
        loose.join(", "),
        footprint.writes,
        footprint.reads,
    );
}

/// Whether the declared writes account for one difference — through
/// [`Resource::overlaps`], the *same* predicate the timeline commutes by, for every
/// difference alike: a changed tile is named as the one-tile `Paint` claim it is, so
/// there is no second reading of "covers" for a coarse claim to look finer under.
///
/// Equality would not do: `Resource::Layer` stands for everything about one layer,
/// so a coarse claim has to keep covering every fine difference under it.
fn covered(diff: &Resource, writes: &[Resource]) -> bool {
    writes.iter().any(|w| w.overlaps(diff))
}

/// Every layer in the tree, by id — the traversal flattened so two states can be
/// compared layer for layer regardless of how the tree was reshaped.
fn layers(state: &DocState) -> BTreeMap<LayerId, &Layer> {
    let mut out = BTreeMap::new();
    state.visit(&mut |l, _| {
        out.insert(l.id, l);
    });
    out
}

/// The tree's shape: every layer in composite order with its depth. Two states
/// agreeing here have the same stacks, the same nesting and the same order.
fn shape(state: &DocState) -> Vec<(LayerId, usize)> {
    let mut out = Vec::new();
    state.visit(&mut |l, depth| out.push((l.id, depth)));
    out
}

/// Whether two masks are the same selection. Mask tiles are rasterized afresh rather
/// than rewritten, so handle identity is the content test here too.
fn selections_agree(a: &Selection, b: &Selection) -> bool {
    // Asked of the type rather than assembled here. A chain of accessor comparisons
    // silently covers only the fields it names, so a divergence in the level that
    // decides the outline contour and the invert reflection passes the audit;
    // `Selection::same` is a destructuring `let`, which cannot lose a field.
    a.same(b)
}

/// Everything that differs between two states, as resources.
fn differences(before: &DocState, after: &DocState) -> Vec<Resource> {
    let mut out = Vec::new();
    // The id and the scale as one value, as `Resource::Substrate` is one resource for
    // the two (§6.4).
    if before.substrate() != after.substrate() {
        out.push(Resource::Substrate);
    }
    if before.substrate_color() != after.substrate_color() {
        out.push(Resource::SubstrateColor);
    }
    if shape(before) != shape(after) {
        out.push(Resource::StackOrder);
    }
    // The guide roster is one coarse resource (§20.5), so this compares it as one
    // thing: every guide and the order they sit in.
    if before.guides() != after.guides() {
        out.push(Resource::Guides);
    }

    // Every actor either state has a mask for. An absent entry *is* the unrestricted
    // selection, and `selection_of` already says so, so an actor deselecting reads as
    // a change rather than as a disappearance.
    let actors: BTreeSet<_> = before
        .selections()
        .chain(after.selections())
        .map(|(a, _)| a)
        .collect();
    for actor in actors {
        if !selections_agree(&before.selection_of(actor), &after.selection_of(actor)) {
            out.push(Resource::Selection(actor));
        }
    }

    let (a, b) = (layers(before), layers(after));
    let ids: BTreeSet<LayerId> = a.keys().chain(b.keys()).copied().collect();
    for id in ids {
        let (Some(x), Some(y)) = (a.get(&id), b.get(&id)) else {
            // Arrived or departed. A layer on one side only has no properties to
            // compare — its whole record is the difference.
            out.push(Resource::Existence(id));
            continue;
        };
        // At exactly the granularity [`Prop`] splits a layer into, since that is the
        // granularity undo restores it at.
        for p in Prop::ALL.iter().filter(|p| x.prop(**p) != y.prop(**p)) {
            out.push(Resource::Prop(id, *p));
        }
        // A layer the fold did not touch still holds the *same persistent root*
        // (`Layer::with_tiles`), so the common case is one pointer test rather than
        // a scan of every tile of every layer on every fold in the workspace — which
        // `changed_tiles` makes for every caller, not this one alone.
        if !LiquifyRun::same(x.liquify_run(), y.liquify_run()) {
            out.push(Resource::LiquifyRun(id));
        }
        if let (Some(tx), Some(ty)) = (x.tiles(), y.tiles()) {
            out.extend(
                changed_tiles(tx, ty, TileRect::ALL)
                    .into_iter()
                    .map(|(coord, _)| Resource::Paint(id, TileRect::at(coord))),
            );
        }
    }
    out
}
