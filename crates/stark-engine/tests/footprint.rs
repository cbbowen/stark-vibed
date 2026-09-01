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
//! Rust's exhaustiveness gets us the *presence* of an arm in `compute_footprint()`
//! for every `ActionKind`. Nothing checks *correspondence* — that the arm names what
//! the matching arm of `apply` goes on to touch.
//!
//! **The correspondence is checked inside the fold now**, not here: `Materialize::audit`
//! diffs every debug-build fold against the action's own footprint
//! (`document::audit`), so every action every test in the workspace commits is held
//! to it. What this file adds is **reach** — it drives the engine to commit one of
//! *every kind in the roster*, and asserts at the end that it did, which is a claim
//! about coverage that no amount of folding makes on its own.
//!
//! That it covers *every* kind is `stark_testdata::vocabulary`'s claim rather than
//! this file's: the run collects what it reached by slot and insists at the end on
//! the whole roster, and the roster is held complete by an exhaustive match no new
//! variant can compile past. Written out here as well as in the model's own suite,
//! both copies can miss the same variant — see that module's note for what that
//! cost.
//!
//! `ActionKind::Undo` is deliberately out of scope: it is resolved by the timeline
//! and never materialized through `apply` (which is *why* its footprint is empty),
//! so a state change around one is a re-materialization rather than an application —
//! `tests/commute.rs` guards that path.
//!
//! **`reads` are not checked here**, and cannot be: a read is not observable in a
//! state diff. That instrument is `tests/commute_pairs.rs`, which asks the parent
//! question instead — every pair the vocabulary claims commutes must produce the
//! same document in either order — and so fails on an undeclared read and an
//! undeclared write alike.

mod common;

use stark_model::Srgb;
use std::collections::BTreeSet;

use common::{engine_or_skip, paint};
use stark_engine::Engine;
use stark_engine::command::{DocCommand, InputCommand};
use stark_engine::document::{DocState, undeclared};
use stark_model::AssetId;
use stark_model::document::{
    ActionId, ActionKind, ActorId, BlendMode, ColorAdjust, FillOp, Filter, Footprint, MatteRegion,
    Parcel, PerspectiveGuide, PerspectiveMap, Place, Resource, SelectionMode, SelectionOp,
    SelectionShape, TransformMap, WarpMap, compute_footprint, rect_corners,
};
use stark_model::geom::{Affine2, IVec2, Vec2};
use stark_model::{SubstrateId, SubstrateScale};
use stark_testdata::vocabulary::{KINDS, labels, slot};

// ---------------------------------------------------------------------------
// The driver

/// The action kinds a run actually reached, by their place in the roster.
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
        // The claim the fold's own audit cannot make, because no fold happened: a
        // command that declines to log must also leave the document alone, since an
        // edit outside the log would not replay at all. An empty footprint declares
        // nothing, so *any* difference is undeclared.
        assert_eq!(
            undeclared(before, &after, &Footprint::default()),
            Vec::<String>::new(),
            "{what}: logged nothing, so it must have changed nothing"
        );
        return;
    };
    seen.insert(slot(&action.kind));
    // The diff against the footprint is **not** repeated here. `Materialize::audit`
    // already ran it inside the fold this command performed — on the same two states,
    // through the same `document::undeclared` — and would have panicked (§12.6). What
    // this run contributes is *reach*: it drives every kind in the roster and asserts
    // so below, where the audit contributes depth by holding every fold in the
    // workspace to the same rule.
}

/// A stroke is a gesture rather than a command, so it commits on release; the
/// snapshot has to span the whole of it.
fn stroke(engine: &mut Engine, seen: &mut Seen, what: &str, points: &[Vec2]) {
    let before = engine.document().clone();
    paint(engine, [0.8, 0.2, 0.3], 12.0, points);
    check(engine, seen, what, &before);
}

fn rect(min: Vec2, max: Vec2) -> SelectionShape {
    SelectionShape::rect_from_corners(min, max)
}

/// A small image with something in every channel, including a transparent margin —
/// so a placed layer's tiles are neither uniform nor everywhere opaque.
fn swatch(w: u32, h: u32) -> Vec<u8> {
    let pixels = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            [
                (x * 9) as u8,
                (y * 11) as u8,
                ((x ^ y) * 5) as u8,
                if edge { 0 } else { 255 },
            ]
        })
        .collect();
    stark_assetid::Picture {
        width: w,
        height: h,
        pixels,
    }
    .encode()
    .expect("a well-formed swatch")
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
        DocCommand::SetSubstrateColor(Srgb::new([0.2, 0.3, 0.4])),
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
            paint: Parcel::Solid(Srgb::new([0.9, 0.9, 0.9])),
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
        DocCommand::SetMattePaint(matte, Parcel::Solid(Srgb::new([0.1, 0.1, 0.1]))),
    );
    step(
        &mut engine,
        seen,
        "remove the matte",
        DocCommand::RemoveLayer(matte),
    );

    // An image from outside the document: one action that mints a layer, names it and
    // fills it with paint, so it is the only kind here whose writes span all four
    // resource shapes at once (§23).
    let picture = engine
        .import_picture(&swatch(24, 18))
        .expect("import a picture");
    step(
        &mut engine,
        seen,
        "place an image",
        DocCommand::PlaceImage {
            carrier: None,
            above: Some(root),
            at: IVec2::new(40, 30),
            name: Some("sunset.png".into()),
            image: picture,
        },
    );
    let placed = engine.observe().active_layer;
    step(
        &mut engine,
        seen,
        "remove the placed image",
        DocCommand::RemoveLayer(placed),
    );

    // A substrate the document names but holds no bytes for still changes what the
    // log says the paint sat on, which is what the footprint has to claim.
    step(
        &mut engine,
        seen,
        "switch the canvas substrate",
        DocCommand::SetSubstrate(SubstrateId::Image(AssetId([7; 32]))),
    );

    // And laying it at a different size, which the same resource claims (§6.4): the
    // substrate and how large it is are one fact about what a stroke bites.
    step(
        &mut engine,
        seen,
        "lay the canvas substrate larger",
        DocCommand::SetSubstrateScale(SubstrateScale::new(200)),
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
        "dim the selection",
        DocCommand::SetSelectionOpacity(0.5),
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
                Srgb::new([0.2, 0.7, 0.4]),
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
            op: FillOp::of_selection(Srgb::new([0.3, 0.3, 0.8])),
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

    // The layer frame (§14.12) and the float (§16.12) — the float made on a
    // *translated* source, so the audit covers the mask brought into a nonzero
    // frame, not only the zero-frame shortcut.
    step(
        &mut engine,
        seen,
        "translate a layer",
        DocCommand::TranslateLayer {
            layer: root,
            to: stark_model::geom::IVec2::new(37, -13),
        },
    );
    step(
        &mut engine,
        seen,
        "float the selection",
        DocCommand::FloatSelection { layer: root },
    );
    // Drag the float somewhere its frame differs from the source's, then merge
    // it home — the §14.12.3 bake under the audit, and what hands `root` back
    // as a leaf so the sibling merge below is still offered.
    let float_child = engine.observe().active_layer;
    step(
        &mut engine,
        seen,
        "drag the float",
        DocCommand::TranslateLayer {
            layer: float_child,
            to: stark_model::geom::IVec2::new(-40, 21),
        },
    );
    step(
        &mut engine,
        seen,
        "merge the float home",
        DocCommand::MergeLayerDown(float_child),
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

    // The drawing guides (§20.5). No pixel moves for any of these — a guide is
    // geometry to construct through — which is exactly why they belong in this
    // run: `Resource::Guides` is a resource nothing else claims, and a footprint
    // that forgot to claim it would be invisible to every other check here.
    step(
        &mut engine,
        seen,
        "add a guide",
        DocCommand::AddGuide {
            guide: PerspectiveGuide::default(),
            after: None,
            name: Some("horizon".into()),
        },
    );
    let first = engine.observe().guides.first().expect("the guide").id;
    step(
        &mut engine,
        seen,
        "a second guide",
        DocCommand::AddGuide {
            guide: PerspectiveGuide {
                focal: 640.0,
                ..PerspectiveGuide::default()
            },
            after: Some(first),
            name: None,
        },
    );
    let second_guide = engine.observe().guides.last().expect("the guide").id;
    step(
        &mut engine,
        seen,
        "reshape a guide",
        DocCommand::SetGuide(
            first,
            PerspectiveGuide {
                focal: 1200.0,
                ..PerspectiveGuide::default()
            },
        ),
    );
    step(
        &mut engine,
        seen,
        "rename a guide",
        DocCommand::SetGuideName(first, Some("the substrate".into())),
    );
    step(
        &mut engine,
        seen,
        "reorder the guides",
        DocCommand::MoveGuide {
            id: first,
            after: Some(second_guide),
        },
    );
    step(
        &mut engine,
        seen,
        "remove a guide",
        DocCommand::RemoveGuide(second_guide),
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
        .map(|s| labels()[s])
        .collect();
    assert!(
        missed.is_empty(),
        "these action kinds were never committed, so nothing about them was \
         checked: {missed:?}"
    );
}

/// The diff has to be able to *see* an escape, or every check built on it — the
/// run above, and `Materialize::audit` on every fold in the workspace — passes by
/// being blind. Feeds a deliberately dishonest footprint, one tile column narrower
/// than the stroke actually painted, and insists `undeclared` reports it.
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
        [0.8, 0.2, 0.3],
        12.0,
        &[Vec2::new(20.0, 20.0), Vec2::new(400.0, 40.0)],
    );
    let after = engine.document().clone();
    let action = engine.take_outbox().pop().expect("the stroke was logged");

    // The stroke shows up at all — against a footprint that declares nothing,
    // everything it did is undeclared.
    assert!(
        !undeclared(&before, &after, &Footprint::default()).is_empty(),
        "a stroke has to show up as a difference at all"
    );
    // …and its real footprint covers it. This is the assertion the fold already
    // made when the stroke committed; repeated here so the two halves of the
    // self-test read together.
    let honest = compute_footprint(&action);
    assert_eq!(
        undeclared(&before, &after, &honest),
        Vec::<String>::new(),
        "the real footprint covers the real stroke"
    );

    // Now shrink every claimed rect to a single tile at the origin — what the
    // pre-`TileRect::covering` quantizer produced for a non-finite radius.
    let liar = Footprint {
        reads: honest.reads.clone(),
        writes: honest
            .writes
            .iter()
            .map(|w| match w {
                Resource::Paint(l, _) => Resource::Paint(
                    *l,
                    stark_model::geom::TileRect {
                        min: (0, 0),
                        max: (0, 0),
                    },
                ),
                other => other.clone(),
            })
            .collect(),
    };
    assert!(
        !undeclared(&before, &after, &liar).is_empty(),
        "an under-claimed footprint has to be caught"
    );
}
