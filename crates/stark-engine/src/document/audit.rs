//! **Every `apply` touches only what its `Footprint` declares** (§12.6), checked on
//! every fold — and every *unfold* — of every debug build.
//!
//! This is the first rule in CLAUDE.md's list of the ones that break silently, and
//! until now nothing structural held it. Seven exhaustive matches over `ActionKind`
//! say each action *has* a footprint; none of them says the footprint is the one its
//! `apply` arm honours, and the compiler cannot tell — the two are a walk of a tree
//! and a list of resources. What held the line was a test driving a hand-written
//! vocabulary, which is a sample: the arm it never mints is the arm nothing checks,
//! and the group removal that this module's first run would have caught sat wrong for
//! as long as no row removed a group.
//!
//! So the check moved to where the folding happens. `Materialize::audit` is called
//! from `Logged::apply` behind `cfg(debug_assertions)`, so every action every test in
//! the workspace folds — hundreds of them, in shapes no vocabulary enumerates — is
//! held to its own declaration.
//!
//! # What it costs, and why that is affordable
//!
//! One `DocState` clone (a handful of `Arc` bumps, §5.1) and one walk of the layer
//! tree comparing tile handles by pointer. Both are debug-only: a release fold is
//! exactly what it was, and `Materialize::AUDITED` is what keeps a consumer that does
//! not audit from paying even the clone.
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
use stark_model::geom::TileCoord;

use super::layer::{Layer, LayerContent};
use super::selection::Selection;
use super::state::DocState;

/// One addressable difference between two states, named as the [`Resource`] that owns
/// it — the same vocabulary a footprint speaks, so coverage is a containment test
/// rather than an interpretation.
#[derive(Debug)]
enum Diff {
    /// One tile of one layer's paint changed hands. Handle identity is the change
    /// test, for the reason `patch.rs` diffs by it: a committed tile is never
    /// rewritten in place, so a shared handle *is* an unchanged tile (§5.2).
    Tile(LayerId, TileCoord),
    /// Everything else, named directly.
    Named(Resource),
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Diff::Tile(layer, coord) => write!(f, "tile {coord:?} of {layer:?}"),
            Diff::Named(r) => write!(f, "{r:?}"),
        }
    }
}

/// Every difference between two states that `footprint` does not declare, described
/// for a human — empty when the fold was honest.
///
/// **The one enumeration of "what can differ".** `tests/footprint.rs` kept a second
/// copy so it could drive the vocabulary and check it, and two lists of a struct's
/// fields is one list that goes stale: a `Layer` or `DocState` field added to one and
/// not the other is invisible to whichever forgot it. It calls this now, and so does
/// the debug fold below.
///
/// `#[doc(hidden)]`: this is a test hook on a public module, not part of what the
/// crate offers (`ENGINE_CLEANUP.md` [T]). It is public because an integration test
/// can reach nothing else, and narrow enough that saying so costs one line.
#[doc(hidden)]
pub fn undeclared(before: &DocState, after: &DocState, footprint: &Footprint) -> Vec<String> {
    differences(before, after)
        .into_iter()
        .filter(|d| !covered(d, &footprint.writes))
        .map(|d| d.to_string())
        .collect()
}

/// Whether two states are the same document — [`differences`] with nothing to
/// hide behind, which is what `apply::is_noop_on` asks of a fold it has already run.
pub(super) fn changes_nothing(before: &DocState, after: &DocState) -> bool {
    differences(before, after).is_empty()
}

/// Panic unless every difference `action` made lies inside a resource it declared.
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
/// [`Resource::overlaps`], the *same* predicate the timeline commutes by.
///
/// Asking it a second way here is how a coarse claim comes to look finer than it is:
/// `Resource::Layer` stands for everything about one layer, and a check that only
/// compared resources for equality would report every one of those as undeclared.
fn covered(diff: &Diff, writes: &[Resource]) -> bool {
    match diff {
        Diff::Tile(layer, coord) => writes.iter().any(|w| match w {
            Resource::Paint(l, r) => l == layer && r.contains(*coord),
            Resource::Layer(l) => l == layer,
            _ => false,
        }),
        Diff::Named(r) => writes.iter().any(|w| w.overlaps(r)),
    }
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

/// Which presentation properties differ — at exactly the granularity [`Prop`] splits
/// them into, since that is the granularity undo restores them at.
fn props(a: &Layer, b: &Layer) -> Vec<Prop> {
    Prop::ALL
        .iter()
        .copied()
        .filter(|p| differs(a, b, *p))
        .collect()
}

/// Whether `p` names something these two layers disagree about.
///
/// **A `match`, which is the point.** This was seven `if a.x != b.x` statements, and
/// seven statements are what a new [`Prop`] variant slips past: the function goes on
/// compiling and the audit quietly stops covering the property, in the one checker
/// that carries §12.6 on every fold. Every other reader of `Prop` is already held to
/// the variant list by the compiler — `patch::capture_resource` matches, `Prop::ALL`
/// is generated from the enum's own list, `patch`'s round-trip test matches — and
/// this was the last one that did not.
///
/// Its order is [`Prop::ALL`]'s, which is why [`props`] can collect straight from it.
fn differs(a: &Layer, b: &Layer, p: Prop) -> bool {
    let matte_of = |l: &Layer| match &l.content {
        LayerContent::Matte { region, paint } => Some((*region, paint.clone())),
        LayerContent::Paint(_) | LayerContent::Filter(_) => None,
    };
    match p {
        // Split finer than the struct: `CompositeParams` travels as one value through
        // compositing, but undo restores each of the three on its own — a clip toggle has
        // to commute with a blend change on the same layer (§12.6).
        Prop::Blend => a.composite.blend != b.composite.blend,
        Prop::Clip => a.composite.clip != b.composite.clip,
        Prop::Opacity => a.composite.opacity != b.composite.opacity,
        Prop::Visible => a.visible != b.visible,
        Prop::Name => a.name != b.name,
        Prop::Matte => matte_of(a) != matte_of(b),
        Prop::Filter => a.filter() != b.filter(),
        Prop::Translation => a.translation != b.translation,
    }
}

/// Everything that differs between two states, as resources.
fn differences(before: &DocState, after: &DocState) -> Vec<Diff> {
    let mut out = Vec::new();
    // **The substrate and the scale together**, because `Resource::Substrate` is one
    // resource for the two (§6.4). Comparing only the id left `SetSubstrateScale`
    // with no difference to report, so its footprint was checked against nothing —
    // and it is the very pair whose undo patch was found carrying only half of it.
    if before.substrate != after.substrate || before.substrate_scale != after.substrate_scale {
        out.push(Diff::Named(Resource::Substrate));
    }
    if before.substrate_color != after.substrate_color {
        out.push(Diff::Named(Resource::SubstrateColor));
    }
    if shape(before) != shape(after) {
        out.push(Diff::Named(Resource::StackOrder));
    }
    // The guide roster is one coarse resource (§20.5), so this compares it as one
    // thing: every guide and the order they sit in.
    if before.guides() != after.guides() {
        out.push(Diff::Named(Resource::Guides));
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
            out.push(Diff::Named(Resource::Selection(actor)));
        }
    }

    let (a, b) = (layers(before), layers(after));
    let ids: BTreeSet<LayerId> = a.keys().chain(b.keys()).copied().collect();
    for id in ids {
        let (Some(x), Some(y)) = (a.get(&id), b.get(&id)) else {
            // Arrived or departed. A layer on one side only has no properties to
            // compare — its whole record is the difference.
            out.push(Diff::Named(Resource::Existence(id)));
            continue;
        };
        for p in props(x, y) {
            out.push(Diff::Named(Resource::Prop(id, p)));
        }
        // A layer the fold did not touch still holds the *same persistent root*
        // (`Layer::with_tiles`), so the common case is one pointer test rather than
        // a scan of every tile of every layer on every fold in the workspace.
        if let (Some(tx), Some(ty)) = (x.tiles(), y.tiles())
            && !tx.ptr_eq(ty)
        {
            for (coord, handle) in tx.iter() {
                if !ty.get(coord).is_some_and(|h| h.same(handle)) {
                    out.push(Diff::Tile(id, *coord));
                }
            }
            for coord in ty.keys() {
                if tx.get(coord).is_none() {
                    out.push(Diff::Tile(id, *coord));
                }
            }
        }
    }
    out
}
