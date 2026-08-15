//! **Footprint honesty** (§12.6): every action's `apply` touches only what its
//! `Footprint` declares.
//!
//! This is the first of CLAUDE.md's "rules that are easy to break silently", and
//! it is the one with no natural alarm. A *false* conflict costs the commutation
//! fast path and nothing else. A *missed* one lets the history splice an undone
//! action out past work it actually overlapped — and pixels cannot show which
//! materialization ran, because agreeing on pixels is the whole claim. Peers
//! diverge quietly, and the first symptom is a painting that differs between two
//! people who watched each other make it.
//!
//! Rust's exhaustiveness gets us the *presence* of an arm in `footprint()` for
//! every `ActionKind`. Nothing checks *correspondence* — that the arm names what
//! the matching arm of `apply` goes on to touch. `tests/commute.rs` covers five
//! hand-written scenarios end-to-end; this covers every action kind
//! structurally, by diffing the document across each commit and insisting every
//! difference lands inside a resource the action declared.
//!
//! Two things are deliberately out of scope. `ActionKind::Undo` is resolved by
//! the timeline and never materialized through `apply` (which is *why* its
//! footprint is empty), so a state change around one is a re-materialization
//! rather than an application — `tests/commute.rs` guards that path. And
//! `reads` are not checked: a read is not observable in a state diff, and
//! catching an undeclared one needs a different instrument.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{engine_or_skip, paint};
use stark_core::Engine;
use stark_core::assets::AssetId;
use stark_core::command::{DocCommand, InputCommand};
use stark_core::document::{
    ActionId, ActionKind, ActorId, BlendMode, ColorAdjust, DocState, FillOp, Filter, Layer,
    LayerContent, LayerId, MattePaint, MatteRegion, PerspectiveMap, Place, Prop, Resource,
    Selection, SelectionMode, SelectionOp, SelectionShape, TransformMap, WarpMap, footprint,
    rect_corners,
};
use stark_core::geom::{Affine2, TileCoord, Vec2};
use stark_core::gpu::SurfaceId;

// ---------------------------------------------------------------------------
// The state diff
//
// One entry per *addressable* difference, named as the `Resource` that owns it —
// the same vocabulary the footprint speaks, so coverage is a containment test
// rather than an interpretation.

#[derive(Debug, PartialEq)]
enum Diff {
    /// One tile of one layer's paint changed hands. Handle identity is the
    /// change test, for the reason `patch.rs` diffs by it: a committed tile is
    /// never rewritten in place, so a shared handle *is* an unchanged tile.
    Tile(LayerId, TileCoord),
    /// Everything else, named directly.
    Named(Resource),
}

/// Every layer in the tree, by id — the traversal flattened so two states can be
/// compared layer for layer regardless of how the tree was reshaped.
fn layers(state: &DocState) -> BTreeMap<LayerId, Layer> {
    let mut out = BTreeMap::new();
    state.visit(&mut |l, _| {
        out.insert(l.id, l.clone());
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

fn matte_of(l: &Layer) -> Option<(MatteRegion, MattePaint)> {
    match &l.content {
        LayerContent::Matte { region, paint } => Some((*region, paint.clone())),
        LayerContent::Paint(_) | LayerContent::Filter(_) => None,
    }
}

/// Which presentation properties differ — at exactly the granularity `Prop`
/// splits them into, since that is the granularity undo restores them at.
fn props(a: &Layer, b: &Layer) -> Vec<Prop> {
    let mut out = Vec::new();
    // Split finer than the struct: `CompositeParams` travels as one value through
    // compositing, but undo restores each of the three on its own — a clip toggle has
    // to commute with a blend change on the same layer (§12.6). One place where the
    // grouping deliberately does not propagate.
    if a.composite.blend != b.composite.blend {
        out.push(Prop::Blend);
    }
    if a.composite.clip != b.composite.clip {
        out.push(Prop::Clip);
    }
    if a.composite.opacity != b.composite.opacity {
        out.push(Prop::Opacity);
    }
    if a.visible != b.visible {
        out.push(Prop::Visible);
    }
    if a.name != b.name {
        out.push(Prop::Name);
    }
    if matte_of(a) != matte_of(b) {
        out.push(Prop::Matte);
    }
    if a.filter() != b.filter() {
        out.push(Prop::Filter);
    }
    out
}

/// Whether two masks are the same selection. Mask tiles are rasterized afresh
/// rather than rewritten, so handle identity is the content test here too.
fn selections_agree(a: &Selection, b: &Selection) -> bool {
    a.outside() == b.outside()
        && a.tile_count() == b.tile_count()
        && a.tiles()
            .all(|(c, h)| b.tile(*c).is_some_and(|o| o.same(h)))
}

fn differences(before: &DocState, after: &DocState) -> Vec<Diff> {
    let mut out = Vec::new();
    if before.surface != after.surface {
        out.push(Diff::Named(Resource::Surface));
    }
    if before.background != after.background {
        out.push(Diff::Named(Resource::Background));
    }
    if shape(before) != shape(after) {
        out.push(Diff::Named(Resource::StackOrder));
    }

    // Every actor either state has a mask for. An absent entry *is* the
    // unrestricted selection, and `selection_of` already says so, so an actor
    // deselecting reads as a change rather than as a disappearance.
    let actors: BTreeSet<ActorId> = before
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
            // Arrived or departed. A layer present on one side only has no
            // properties to compare — its whole record is the difference.
            out.push(Diff::Named(Resource::Existence(id)));
            continue;
        };
        for p in props(x, y) {
            out.push(Diff::Named(Resource::Prop(id, p)));
        }
        if let (Some(tx), Some(ty)) = (x.tiles(), y.tiles()) {
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

/// Whether the action's declared writes account for one difference.
///
/// `Existence` is the one place a coarser resource stands in, and deliberately:
/// a removal takes a whole subtree, so it writes the existence of layers it does
/// not name, and `footprint.rs` declares `StackOrder` to cover them. Everything
/// else has to be named exactly — a tile by a rect that contains it, a property
/// by its own `Prop`.
fn covered(diff: &Diff, writes: &[Resource]) -> bool {
    match diff {
        Diff::Tile(layer, coord) => writes
            .iter()
            .any(|w| matches!(w, Resource::Paint(l, r) if l == layer && r.contains(*coord))),
        Diff::Named(r @ Resource::Existence(_)) => {
            writes.contains(r) || writes.contains(&Resource::StackOrder)
        }
        Diff::Named(r) => writes.contains(r),
    }
}

// ---------------------------------------------------------------------------
// The driver

/// The slot of every action kind, in the order `ActionKind` declares them —
/// which is also the order postcard encodes them in (§8), so a slot is a fact
/// about the wire form rather than a number invented here.
///
/// **Exhaustive, with no `_` arm, and that is the whole point of it.** What this
/// replaces was a hand-written list of captions the run had to reach, which could
/// not fail when a *new* kind appeared: an action added to the enum silenced its
/// own coverage rather than failing, which is exactly how `MergeLayerDown`,
/// `AddFilter` and `SetFilter` came to be unchecked here. Adding a variant now
/// stops this file compiling, at this match, three lines from the run that has to
/// drive it — the device `Modulations::all`'s `..`-free destructure already uses
/// for a struct's fields.
fn slot(kind: &ActionKind) -> usize {
    match kind {
        ActionKind::CommitStroke(_) => 0,
        ActionKind::AddLayer { .. } => 1,
        ActionKind::RemoveLayer(_) => 2,
        ActionKind::SetLayerBlend(..) => 3,
        ActionKind::SetLayerOpacity(..) => 4,
        ActionKind::SetLayerVisible(..) => 5,
        ActionKind::MoveLayer { .. } => 6,
        ActionKind::Undo(_) => 7,
        ActionKind::SetSurface(_) => 8,
        ActionKind::Select(_) => 9,
        ActionKind::InvertSelection => 10,
        ActionKind::AddMatte { .. } => 11,
        ActionKind::SetMatteRect(..) => 12,
        ActionKind::SetMattePaint(..) => 13,
        ActionKind::SetBackground(_) => 14,
        ActionKind::Transform { .. } => 15,
        ActionKind::SetLayerName(..) => 16,
        ActionKind::Fill { .. } => 17,
        ActionKind::SetLayerClip(..) => 18,
        ActionKind::TransformPerspective { .. } => 19,
        ActionKind::TransformWarp { .. } => 20,
        ActionKind::DuplicateLayer { .. } => 21,
        ActionKind::AddFilter { .. } => 22,
        ActionKind::SetFilter(..) => 23,
        ActionKind::MergeLayerDown { .. } => 24,
    }
}

/// How many kinds there are — `slot`'s range, bumped with its last arm. [`NAMES`]
/// is indexed by slot and its length is held to this by the type, so a slot added
/// without a name is a compile error rather than a worse failure message.
const KINDS: usize = 25;

/// What to call each slot when the run has missed one. `ActionKind::label`'s own
/// captions, which is what a reader of the failure will go looking for.
const NAMES: [&str; KINDS] = [
    "Stroke",
    "Add layer",
    "Remove layer",
    "Blend mode",
    "Layer opacity",
    "Layer visibility",
    "Reorder layer",
    "Undo",
    "Canvas surface",
    "Select",
    "Invert selection",
    "Add matte",
    "Move frame",
    "Matte paint",
    "Canvas color",
    "Transform",
    "Rename layer",
    "Fill",
    "Clip layer",
    "Perspective",
    "Warp",
    "Duplicate layer",
    "Add filter",
    "Filter",
    "Merge down",
];

/// The action kinds a run actually reached, by [`slot`].
///
/// Collected because a step that logs nothing passes the check trivially, so
/// without this the whole test could rot into vacuity — a command quietly
/// becoming a no-op would *silence* its coverage rather than fail. The run
/// asserts its own reach at the end.
type Seen = BTreeSet<usize>;

/// Run `command` and hold whatever it committed to its own footprint.
///
/// A command that logs *nothing* must also change nothing: there is no footprint
/// to hide behind, and a silent edit outside the log would not replay at all.
fn step(engine: &mut Engine, seen: &mut Seen, what: &str, command: impl Into<InputCommand>) {
    let before = engine.document().clone();
    engine.process(command);
    check(engine, seen, what, &before);
}

fn check(engine: &mut Engine, seen: &mut Seen, what: &str, before: &DocState) {
    let after = engine.document().clone();
    let committed = engine.take_outbox();
    assert!(
        committed.len() <= 1,
        "{what}: this test attributes a diff to one action, but {} were logged",
        committed.len()
    );
    let Some(action) = committed.first() else {
        assert_eq!(
            differences(before, &after),
            Vec::new(),
            "{what}: logged nothing, so it must have changed nothing"
        );
        return;
    };
    seen.insert(slot(&action.kind));
    if matches!(action.kind, ActionKind::Undo(_)) {
        return; // Resolved by the timeline, never applied — see the module note.
    }
    let writes = footprint(action).writes;
    let undeclared: Vec<Diff> = differences(before, &after)
        .into_iter()
        .filter(|d| !covered(d, &writes))
        .collect();
    assert!(
        undeclared.is_empty(),
        "{what}: {:?} changed {undeclared:?}, which its footprint does not declare.\n\
         declared writes: {writes:?}",
        action.kind,
    );
}

/// A stroke is a gesture rather than a command, so it commits on release; the
/// snapshot has to span the whole of it.
fn stroke(engine: &mut Engine, seen: &mut Seen, what: &str, points: &[Vec2]) {
    let before = engine.document().clone();
    paint(engine, [0.8, 0.2, 0.3, 1.0], 12.0, points);
    check(engine, seen, what, &before);
}

fn rect(min: Vec2, max: Vec2) -> SelectionShape {
    SelectionShape::rect_from_corners(min, max)
}

/// Every action kind the engine can be driven to commit, each held to its own
/// footprint. Shared mode so the outbox reports what was logged; the actions are
/// plain appends, so each one is exactly one `apply` over the state before it.
#[test]
fn every_action_touches_only_what_it_declares() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.start_collaboration(ActorId(1));
    engine.take_outbox();
    let root = engine.observe().active_layer;
    let seen = &mut Seen::new();

    // Paint, so later actions have tiles to move, cut and clip against.
    stroke(
        &mut engine,
        seen,
        "stroke",
        &[Vec2::new(20.0, 20.0), Vec2::new(90.0, 70.0)],
    );
    stroke(
        &mut engine,
        seen,
        "second stroke",
        &[Vec2::new(30.0, 60.0), Vec2::new(120.0, 40.0)],
    );

    // Structure.
    step(
        &mut engine,
        seen,
        "add layer",
        DocCommand::AddLayer {
            carrier: None,
            above: Some(root),
        },
    );
    let second = engine.observe().active_layer;
    stroke(
        &mut engine,
        seen,
        "stroke on the second layer",
        &[Vec2::new(140.0, 30.0), Vec2::new(200.0, 90.0)],
    );
    step(
        &mut engine,
        seen,
        "duplicate layer",
        DocCommand::DuplicateLayer(second),
    );
    let copy = engine.observe().active_layer;
    step(
        &mut engine,
        seen,
        "carry a layer",
        DocCommand::MoveLayer {
            id: copy,
            carrier: Some(root),
            at: Place::Top,
        },
    );
    step(
        &mut engine,
        seen,
        "release a layer",
        DocCommand::MoveLayer {
            id: copy,
            carrier: None,
            at: Place::Above(root),
        },
    );

    // Presentation properties.
    step(
        &mut engine,
        seen,
        "blend",
        DocCommand::SetLayerBlend(second, BlendMode::Multiply),
    );
    step(
        &mut engine,
        seen,
        "clip",
        DocCommand::SetLayerClip(second, true),
    );
    step(
        &mut engine,
        seen,
        "opacity",
        DocCommand::SetLayerOpacity(second, 0.4),
    );
    step(
        &mut engine,
        seen,
        "visibility",
        DocCommand::SetLayerVisible(second, false),
    );
    step(
        &mut engine,
        seen,
        "rename",
        DocCommand::SetLayerName(second, Some("wash".into())),
    );
    step(
        &mut engine,
        seen,
        "unname",
        DocCommand::SetLayerName(second, None),
    );

    // The canvas itself.
    step(
        &mut engine,
        seen,
        "background",
        DocCommand::SetBackground([0.2, 0.3, 0.4]),
    );

    // Mattes.
    step(
        &mut engine,
        seen,
        "add matte",
        DocCommand::AddMatte {
            carrier: None,
            at: Place::Top,
            region: MatteRegion::OutsideRect {
                min: Vec2::new(10.0, 10.0),
                max: Vec2::new(180.0, 140.0),
            },
            paint: MattePaint::Solid([0.9, 0.9, 0.9]),
        },
    );
    let matte = engine
        .observe()
        .layers
        .last()
        .expect("the matte is on top")
        .id;
    step(
        &mut engine,
        seen,
        "move matte",
        DocCommand::SetMatteRect(matte, Vec2::new(5.0, 5.0), Vec2::new(190.0, 150.0)),
    );
    step(
        &mut engine,
        seen,
        "recolor matte",
        DocCommand::SetMattePaint(matte, MattePaint::Solid([0.1, 0.1, 0.1])),
    );
    step(
        &mut engine,
        seen,
        "remove the matte",
        DocCommand::RemoveLayer(matte),
    );

    // A ground the document names but holds no bytes for still changes what the
    // log says the paint sat on, which is what the footprint has to claim.
    step(
        &mut engine,
        seen,
        "switch the canvas surface",
        DocCommand::SetSurface(SurfaceId::Image(AssetId([7; 32]))),
    );

    // Selection, and the tools that act through it.
    step(
        &mut engine,
        seen,
        "select a rect",
        DocCommand::Select(SelectionOp::new(
            SelectionMode::Replace,
            rect(Vec2::new(25.0, 25.0), Vec2::new(110.0, 95.0)),
            2.0,
        )),
    );
    step(
        &mut engine,
        seen,
        "union another rect",
        DocCommand::Select(SelectionOp::new(
            SelectionMode::Union,
            rect(Vec2::new(100.0, 20.0), Vec2::new(160.0, 60.0)),
            0.0,
        )),
    );
    step(
        &mut engine,
        seen,
        "invert the selection",
        DocCommand::InvertSelection,
    );
    step(
        &mut engine,
        seen,
        "select again",
        DocCommand::Select(SelectionOp::new(
            SelectionMode::Replace,
            rect(Vec2::new(25.0, 25.0), Vec2::new(110.0, 95.0)),
            1.0,
        )),
    );
    stroke(
        &mut engine,
        seen,
        "stroke through a selection",
        &[Vec2::new(10.0, 50.0), Vec2::new(150.0, 55.0)],
    );
    step(
        &mut engine,
        seen,
        "fill a shape",
        DocCommand::Fill {
            layer: root,
            op: FillOp::new(
                rect(Vec2::new(40.0, 40.0), Vec2::new(80.0, 80.0)),
                1.0,
                [0.2, 0.7, 0.4],
                0.5,
            ),
        },
    );
    step(
        &mut engine,
        seen,
        "fill the selection",
        DocCommand::Fill {
            layer: root,
            op: FillOp::of_selection([0.3, 0.3, 0.8]),
        },
    );

    // The three transform families.
    step(
        &mut engine,
        seen,
        "affine transform",
        DocCommand::Transform {
            layer: root,
            map: TransformMap::Affine(Affine2::from_translation(Vec2::new(24.0, -16.0))),
        },
    );
    step(
        &mut engine,
        seen,
        "perspective transform",
        DocCommand::Transform {
            layer: root,
            map: TransformMap::Perspective(PerspectiveMap {
                min: Vec2::new(20.0, 20.0),
                max: Vec2::new(160.0, 120.0),
                corners: {
                    let mut c = rect_corners(Vec2::new(20.0, 20.0), Vec2::new(160.0, 120.0));
                    c[1] += Vec2::new(0.0, 14.0);
                    c[3] += Vec2::new(0.0, -12.0);
                    c
                },
            }),
        },
    );
    step(
        &mut engine,
        seen,
        "warp transform",
        DocCommand::Transform {
            layer: root,
            map: TransformMap::Warp({
                let mut w = WarpMap::identity(Vec2::new(20.0, 20.0), Vec2::new(160.0, 120.0), 3, 3);
                w.points[4] += Vec2::new(9.0, -7.0);
                w
            }),
        },
    );

    // …and a removal last, which takes a subtree with it.
    step(
        &mut engine,
        seen,
        "carry before removing",
        DocCommand::MoveLayer {
            id: copy,
            carrier: Some(second),
            at: Place::Top,
        },
    );
    step(
        &mut engine,
        seen,
        "remove a group",
        DocCommand::RemoveLayer(second),
    );

    // Filters. A fresh one is neutral by construction, so `AddFilter` has to be
    // followed by a `SetFilter` that dials it: a neutral filter is dropped from
    // the draw list entirely (§21.3), and a step whose action changes no pixel
    // would hold its footprint to a diff of nothing.
    step(
        &mut engine,
        seen,
        "add a filter",
        DocCommand::AddFilter {
            carrier: None,
            above: Some(root),
            filter: Filter::Color(ColorAdjust::NEUTRAL),
        },
    );
    let filter = engine
        .observe()
        .layers
        .last()
        .expect("the filter is on top")
        .id;
    step(
        &mut engine,
        seen,
        "dial the filter",
        DocCommand::SetFilter(
            filter,
            Filter::Color(ColorAdjust {
                saturation: 0.0,
                ..ColorAdjust::NEUTRAL
            }),
        ),
    );

    // …and the merge, which is the largest footprint in the enum: it rewrites the
    // destination's tiles, folds both opacities in, and takes the source layer out
    // of the tree. A fresh painted layer directly over `root`, which is the
    // second-from-the-foot position where the destination *is* the whole backdrop
    // (§14.11.2) — the filter added above sits outside the pair and does not bear
    // on the plan.
    step(
        &mut engine,
        seen,
        "a layer to merge down",
        DocCommand::AddLayer {
            carrier: None,
            above: Some(root),
        },
    );
    let upper = engine.observe().active_layer;
    stroke(
        &mut engine,
        seen,
        "stroke on the layer to be merged",
        &[Vec2::new(50.0, 100.0), Vec2::new(170.0, 120.0)],
    );
    step(
        &mut engine,
        seen,
        "merge down",
        DocCommand::MergeLayerDown(upper),
    );

    // What the run actually reached. Without this the test could rot into
    // vacuity one command at a time: a step that stops committing *silences* its
    // own coverage rather than failing, because `step` waves through an action
    // that was never logged.
    //
    // Every slot has to be reached except `Undo`'s, which this test excludes by
    // construction (module note) — and that one is *asked for* rather than
    // written down, so it stays right if the enum is ever reordered.
    let exempt = slot(&ActionKind::Undo(ActionId {
        lamport: 0,
        actor: ActorId(0),
    }));
    let missed: Vec<&str> = (0..KINDS)
        .filter(|s| *s != exempt && !seen.contains(s))
        .map(|s| NAMES[s])
        .collect();
    assert!(
        missed.is_empty(),
        "these action kinds were never committed, so nothing about them was \
         checked: {missed:?}"
    );
}

/// The diff has to be able to *see* an escape, or the test above passes by being
/// blind. Feeds a deliberately dishonest footprint — one tile column narrower
/// than the stroke actually painted — and insists the check rejects it.
#[test]
fn the_check_rejects_a_footprint_that_under_claims() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.start_collaboration(ActorId(1));
    engine.take_outbox();

    let before = engine.document().clone();
    paint(
        &mut engine,
        [0.8, 0.2, 0.3, 1.0],
        12.0,
        &[Vec2::new(20.0, 20.0), Vec2::new(400.0, 40.0)],
    );
    let after = engine.document().clone();
    let action = engine.take_outbox().pop().expect("the stroke was logged");

    let diffs = differences(&before, &after);
    assert!(
        !diffs.is_empty(),
        "a stroke has to show up as a difference at all"
    );
    let honest = footprint(&action).writes;
    assert!(
        diffs.iter().all(|d| covered(d, &honest)),
        "the real footprint covers the real stroke"
    );

    // Now shrink every claimed rect to a single tile at the origin — what the
    // pre-`TileRect::covering` quantizer produced for a non-finite radius.
    let liar: Vec<Resource> = honest
        .iter()
        .map(|w| match w {
            Resource::Paint(l, _) => Resource::Paint(
                *l,
                stark_core::geom::TileRect {
                    min: (0, 0),
                    max: (0, 0),
                },
            ),
            other => other.clone(),
        })
        .collect();
    assert!(
        diffs.iter().any(|d| !covered(d, &liar)),
        "an under-claimed footprint has to be caught"
    );
}
