//! One of every [`ActionKind`], and the two things the whole vocabulary has to be
//! true of: **the funnel reaches every payload that carries a number** (§21.5), and
//! **it does not move one twice** (§8).
//!
//! # Why this is a file rather than a `mod tests`
//!
//! It is the model's first integration test, and the reason it is one is that what
//! drives it is not about any single item in the crate. [`kinds`] is one of every
//! action the document has a word for; it is held to being *every* one by
//! `stark_testdata::vocabulary`, whose exhaustive match stops compiling when a
//! variant appears and whose roster is this array's declared length.
//!
//! That device is shared with `stark-engine/tests/footprint.rs`, which asks a
//! different question of the same vocabulary (did a *run* reach every kind?) and
//! needs a GPU to ask it — which is why the model's funnel cannot simply be checked
//! beside it. The roster lives once, in `stark_testdata::vocabulary`, the crate both
//! suites already depend on; kept per crate, the two copies drift the first time a
//! variant lands, and that module's own note records what escaped.
//!
//! What is still local is the list itself, and deliberately: an action's payload is
//! the model's vocabulary spelled out, and the run that drives it is three lines
//! below. [`the_list_holds_one_of_every_kind`] pins the two together.

use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, BlendMode, BrushEffect, BrushParams, FillOp, Filter,
    GradientAxis, GradientParcel, GuideId, LayerId, MatteRegion, Parcel, PerspectiveGuide,
    PerspectiveMap, Place, SelectionMode, SelectionOp, SelectionShape, StrokeRecord, ToothParams,
    WarpMap, affine_usable, rect_corners,
};
use stark_model::geom::{Affine2, IVec2, Vec2};
use stark_model::gradient::{Gradient, GradientStop};
use stark_model::path::ControlPoint;
use stark_model::{AssetId, Srgb, SubstrateId, SubstrateScale};
use stark_testdata::vocabulary::{KINDS, labels, slot};

// ---------------------------------------------------------------------------
// One of each

/// One of every kind, with every number it carries taken from `n`.
///
/// **One list, two drivers.** Called with an ordinary value it is a document's worth
/// of plausible actions, which is what idempotence has to be checked on; called with
/// a `NaN` (or an infinity) it is the same document with every float poisoned, which
/// is what the funnel has to be checked against. Writing it once is what keeps the
/// two runs over *the same* set — the failure mode this file exists to end is a list
/// that quietly covers less than it says.
///
/// Payloads that are not numbers — ids, anchors, names, the shape of a warp grid —
/// are held fixed, since poisoning them is a different question and one the encoding
/// answers (§8).
fn kinds(n: f32) -> [ActionKind; KINDS] {
    let id = LayerId::solo(3);
    let other = LayerId::solo(1);
    let action = ActionId {
        lamport: 1,
        actor: ActorId::SOLO,
    };
    let guide = GuideId(action);
    let v = Vec2::splat(n);

    let ramp = || {
        Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::new([n, 0.5, 0.25]),
            },
            GradientStop {
                t: 1.0,
                color: Srgb::new([0.75, n, 1.0]),
            },
        ])
        // A ramp whose stops are not finite is not a ramp at all, and `new` says so
        // (§22.1) — so the poisoned run falls back to a clean ramp on a poisoned
        // *axis*, which is the part of a gradient parcel this file is about.
        .unwrap_or_else(|| {
            Gradient::new(vec![
                GradientStop {
                    t: 0.0,
                    color: Srgb::BLACK,
                },
                GradientStop {
                    t: 1.0,
                    color: Srgb::WHITE,
                },
            ])
            .expect("a two-stop ramp")
        })
    };
    let axis = GradientAxis::Linear { from: v, to: v };

    [
        ActionKind::CommitStroke(StrokeRecord {
            layer: id,
            brush: BrushParams {
                size: n,
                tooth: ToothParams {
                    give: n,
                    softness: n,
                },
                stretch: n,
                drain: n,
                start_taper_length: n,
                effect: BrushEffect::painted([n, 0.5, 0.5]),
                ..BrushParams::default()
            },
            path: vec![ControlPoint::at(Vec2::splat(4.0))],
            seed: 1,
            start: n,
        }),
        ActionKind::AddLayer {
            id,
            carrier: None,
            above: None,
        },
        ActionKind::RemoveLayer {
            id,
            carried: Vec::new(),
        },
        ActionKind::SetLayerBlend(id, BlendMode::Drago { k: n }),
        ActionKind::SetLayerOpacity(id, n),
        ActionKind::SetLayerVisible(id, true),
        ActionKind::MoveLayer {
            id,
            carrier: None,
            at: Place::Top,
        },
        ActionKind::Undo(action),
        ActionKind::SetSubstrate(SubstrateId::Flat),
        ActionKind::SetSubstrateScale(SubstrateScale::new(140)),
        // A **measurable** shape with poisoned scalars, for the reason `AddMatte`
        // below carries a usable region with poisoned paint: a shape's coordinates
        // are gated by `SelectionShape::bounds` refusing them rather than clamped,
        // and are checked as such by `gated_geometry_is_refused_rather_than_repaired`.
        ActionKind::Select(SelectionOp::at(
            SelectionMode::Subtract,
            SelectionShape::Ellipse {
                center: Vec2::splat(8.0),
                radii: Vec2::splat(4.0),
            },
            n,
            n,
        )),
        ActionKind::InvertSelection,
        // The mask's own dial (§6.8), which carries no shape and rasterizes nothing,
        // so `n` is the whole of what it holds — and the funnel is the only thing
        // between a poisoned one and the ceiling every stroke is then capped at.
        ActionKind::SetSelectionOpacity(n),
        // A **usable** region with poisoned paint: the region is gated at `apply`
        // rather than sanitized, and is checked as such by
        // `gated_geometry_is_refused_rather_than_repaired`. What this list asks of
        // an `AddMatte` is the other half — that its paint comes through clean.
        ActionKind::AddMatte {
            id,
            carrier: None,
            at: Place::Bottom,
            region: MatteRegion::Everything,
            paint: Parcel::Gradient(GradientParcel {
                gradient: ramp(),
                axis,
            }),
        },
        ActionKind::SetMatteRect(id, v, v),
        ActionKind::SetMattePaint(id, Parcel::Solid(Srgb::new([n, 0.5, 0.75]))),
        ActionKind::SetSubstrateColor(Srgb::new([n, 0.25, 0.75])),
        ActionKind::Transform {
            layer: id,
            affine: Affine2::from_scale(v),
        },
        ActionKind::SetLayerName(id, Some("wash".into())),
        ActionKind::Fill {
            layer: id,
            op: FillOp::with_paint(
                SelectionShape::Rect {
                    min: Vec2::ZERO,
                    max: Vec2::splat(8.0),
                },
                n,
                Parcel::Gradient(GradientParcel {
                    gradient: ramp(),
                    axis,
                }),
                n,
            ),
        },
        ActionKind::SetLayerClip(id, true),
        ActionKind::TransformPerspective {
            layer: id,
            map: PerspectiveMap {
                min: v,
                max: v,
                corners: rect_corners(v, v),
            },
        },
        ActionKind::TransformWarp {
            layer: id,
            map: WarpMap::identity(v, v, 2, 2),
        },
        ActionKind::DuplicateLayer {
            ids: vec![(id, LayerId::solo(9))],
        },
        ActionKind::AddFilter {
            id,
            carrier: None,
            above: None,
            filter: poisoned_filter(n),
        },
        ActionKind::SetFilter(id, poisoned_filter(n)),
        ActionKind::MergeLayerDown {
            source: id,
            dest: other,
        },
        ActionKind::PlaceImage {
            id,
            carrier: None,
            above: None,
            at: IVec2::new(-3, 9),
            name: Some("sunset.png".into()),
            image: AssetId([4; 32]),
        },
        ActionKind::AddGuide {
            id: guide,
            guide: poisoned_guide(n),
            after: None,
            name: Some("two-point".into()),
        },
        ActionKind::RemoveGuide(guide),
        ActionKind::SetGuide(guide, poisoned_guide(n)),
        ActionKind::SetGuideName(guide, None),
        ActionKind::MoveGuide {
            id: guide,
            after: None,
        },
    ]
}

/// A filter with `n` in every knob it has. `Filter::ALL[0]` is the color adjust,
/// whose fields are the ones that reach a fullscreen pass.
fn poisoned_filter(n: f32) -> Filter {
    match Filter::ALL[0].clone() {
        Filter::Color(mut c) => {
            c.exposure = n;
            c.contrast = n;
            c.saturation = n;
            c.hue = n;
            c.tint = [n, n];
            Filter::Color(c)
        }
        other => other,
    }
}

/// A camera with `n` in every lane the guide pass reads (§20.4).
fn poisoned_guide(n: f32) -> PerspectiveGuide {
    PerspectiveGuide {
        center: Vec2::splat(n),
        focal: n,
        opacity: n,
        ..PerspectiveGuide::default()
    }
}

/// Whether this kind's numbers are **gated at `apply`** rather than clamped on the
/// way in — geometry that cannot be repaired into a different geometry without
/// changing what the author asked for (§16.1, §15.5).
///
/// Exhaustive for [`kinds`]'s reason. A variant added to the enum has to say which
/// side of this line it is on, and both sides are checked below: the clamped ones by
/// [`the_funnel_leaves_no_action_holding_a_number_a_shader_cannot_use`], the gated
/// ones by [`gated_geometry_is_refused_rather_than_repaired`].
fn gated_at_apply(kind: &ActionKind) -> bool {
    matches!(
        kind,
        ActionKind::Transform { .. }
            | ActionKind::TransformPerspective { .. }
            | ActionKind::TransformWarp { .. }
            | ActionKind::SetMatteRect(..)
    )
}

// ---------------------------------------------------------------------------
// The tests

/// [`kinds`] is one of *every* kind, in the roster's order — which is what makes the
/// two runs below exhaustive rather than merely long.
///
/// The array is `[ActionKind; KINDS]`, so its *length* is already the roster's by the
/// type. What is left to say is that it holds each kind **once**, in place, which is
/// what a per-position `slot` says: a list of the right length with one kind twice is
/// a list missing another.
///
/// It also pins the roster's captions to `ActionKind::label`'s own, so a failure in
/// either suite names what the app names, and there is no second set of names to keep
/// in step.
#[test]
fn the_list_holds_one_of_every_kind() {
    let names = labels();
    for (i, kind) in kinds(0.5).iter().enumerate() {
        assert_eq!(slot(kind), i, "slot {i} holds a {} instead", kind.label());
        assert_eq!(
            names[i],
            kind.label(),
            "the roster calls slot {i} something else"
        );
    }
}

/// **Nothing that is not a number survives the funnel** — on every kind, and every
/// payload of every kind.
///
/// Asked of the whole action's `Debug` rather than field by field, deliberately: a
/// per-field list is one a new field can be left out of, and what is being checked
/// is a *class*. `f32`'s `Debug` writes `NaN` and `inf`, so a poisoned float
/// anywhere in the tree — inside a brush, inside a ramp's stop, inside a filter's
/// tint, inside a camera — shows up in the string whatever the shape around it is.
///
/// The gated kinds are exempt here and checked by their own test below, which is the
/// only exemption and is itself an exhaustive match.
#[test]
fn the_funnel_leaves_no_action_holding_a_number_a_shader_cannot_use() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for kind in kinds(bad) {
            if gated_at_apply(&kind) {
                continue;
            }
            let label = kind.label();
            let text = format!("{:?}", kind.sanitized());
            assert!(
                !text.contains("NaN") && !text.contains("inf"),
                "{label} carried {bad} through the funnel: {text}",
            );
        }
    }
}

/// **Sanitizing is idempotent, on every kind.**
///
/// The funnel runs where an action is minted *and* where it enters state
/// (`Logged::new`), so a second pass that moved anything would make every load and
/// every replay a small edit — and goldens are blessed against the first pass.
///
/// Run on ordinary payloads *and* on poisoned ones, because the two exercise
/// different halves: an ordinary action must come through untouched, while a
/// repaired one must not be repaired *again* into something else. The second is
/// where a clamp written as `max` after a divide, or a normalize with a tolerance,
/// tends to keep moving.
#[test]
fn sanitizing_is_idempotent_on_every_kind() {
    for n in [0.5, f32::NAN, f32::INFINITY, -3.0, 1e30] {
        for kind in kinds(n) {
            let once = kind.sanitized();
            let twice = once.clone().sanitized();
            assert_eq!(
                format!("{once:?}"),
                format!("{twice:?}"),
                "sanitizing {} moved on the second pass, at n = {n}",
                once.label(),
            );
        }
    }
}

/// **Every default is already sanitized**, so the funnel moves nothing a fresh
/// document contains.
///
/// This is the property goldens actually rest on. The funnel runs on replay, so a
/// default the funnel nudged would shift every reference image — and unlike the
/// idempotence check above, which says only that the *second* pass is still, this
/// says the first one is too for the values a document starts from.
///
/// Stated over the defaults rather than over [`kinds`], and that is not a
/// weakening: `kinds` takes one number for every field, and no single number is in
/// range for all of them (a tint stops at 0.16, a focal length starts at 1), so a
/// bit-for-bit run over it would be checking the literal chosen here rather than
/// the funnel. Each type's own default is in range **by construction**, which is
/// exactly the claim worth making.
#[test]
fn every_default_is_already_sanitized() {
    assert_eq!(BrushParams::default().sanitized(), BrushParams::default());
    assert_eq!(
        PerspectiveGuide::default().sanitized(),
        PerspectiveGuide::default(),
    );
    for filter in Filter::ALL {
        assert_eq!(filter.clone().sanitized(), filter, "a filter default moved");
        let neutral = filter.neutral();
        assert_eq!(neutral.clone().sanitized(), neutral, "a neutral moved");
    }
    for paint in [
        Parcel::Solid(Srgb::new([0.25, 0.5, 0.75])),
        Parcel::Gradient(GradientParcel {
            gradient: Gradient::new(vec![
                GradientStop {
                    t: 0.0,
                    color: Srgb::BLACK,
                },
                GradientStop {
                    t: 1.0,
                    color: Srgb::WHITE,
                },
            ])
            .expect("a two-stop ramp"),
            axis: GradientAxis::Radial {
                center: Vec2::ZERO,
                radius: 32.0,
            },
        }),
    ] {
        assert_eq!(paint.clone().sanitized(), paint, "a matte paint moved");
    }
    let op = SelectionOp::at(
        SelectionMode::Union,
        SelectionShape::Rect {
            min: Vec2::ZERO,
            max: Vec2::splat(16.0),
        },
        2.0,
        0.5,
    );
    let ActionKind::Select(through) = ActionKind::Select(op.clone()).sanitized() else {
        panic!("a select stays a select")
    };
    assert_eq!(through, op, "a selection op moved");
}

/// The other side of [`gated_at_apply`]: geometry the funnel deliberately leaves
/// alone has to be **refused** by its own predicate instead (§16.1, §15.5).
///
/// A rect, an affine and a mesh have no repaired form — there is no other rectangle
/// that is a fixed version of one nobody can measure — so they are declined at
/// `apply`, deterministically, and every peer declines the same one. What this
/// checks is that the predicate actually says no: an exemption from the funnel with
/// no gate behind it is just an unguarded payload with a comment.
#[test]
fn gated_geometry_is_refused_rather_than_repaired() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for kind in kinds(bad) {
            if !gated_at_apply(&kind) {
                continue;
            }
            let refused = match &kind {
                ActionKind::Transform { affine, .. } => !affine_usable(*affine),
                ActionKind::TransformPerspective { map, .. } => !map.usable(),
                ActionKind::TransformWarp { map, .. } => !map.usable(),
                ActionKind::SetMatteRect(_, min, max) => !MatteRegion::OutsideRect {
                    min: *min,
                    max: *max,
                }
                .usable(),
                other => panic!("{} is exempt from the funnel with no gate", other.label()),
            };
            assert!(
                refused,
                "{} is exempt from the funnel, and its gate admitted {bad}",
                kind.label(),
            );
        }
    }

    // A selection shape is gated the same way, by the one function that can hand
    // its geometry out: `bounds` refuses what it cannot measure, which declines the
    // op at `Selection::plan`, fills nothing at `fill::plan`, and claims the whole
    // layer in a footprint. Driven over all four shapes, because the lasso is the
    // one that used to fold its way past a bad vertex and come back with a tight
    // box (§12.6).
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let v = Vec2::splat(bad);
        let shapes = [
            SelectionShape::Rect {
                min: Vec2::ZERO,
                max: v,
            },
            SelectionShape::Ellipse {
                center: v,
                radii: Vec2::splat(4.0),
            },
            SelectionShape::Lasso(vec![Vec2::ZERO, Vec2::splat(9.0), v]),
            // …including one whose bad vertex is not the extreme in either axis,
            // which is exactly what a `min`/`max` fold steps over.
            SelectionShape::Lasso(vec![Vec2::ZERO, v, Vec2::splat(9.0)]),
        ];
        for shape in shapes {
            assert!(
                shape.bounds().is_none(),
                "{shape:?} handed out a box it cannot measure",
            );
        }
    }

    // …and a usable one is still admitted, so the gates are not simply closed.
    let ok = Vec2::ZERO;
    let wide = Vec2::splat(64.0);
    assert!(affine_usable(Affine2::IDENTITY));
    assert!(
        PerspectiveMap {
            min: ok,
            max: wide,
            corners: rect_corners(ok, wide),
        }
        .usable()
    );
    assert!(WarpMap::identity(ok, wide, 2, 2).usable());
    assert!(MatteRegion::OutsideRect { min: ok, max: wide }.usable());
    assert!(
        SelectionShape::Rect { min: ok, max: wide }
            .bounds()
            .is_some(),
        "an ordinary shape still has a box",
    );
    assert!(
        SelectionShape::Lasso(vec![ok, wide, Vec2::new(0.0, 64.0)])
            .bounds()
            .is_some(),
    );
}

/// The funnel is the **one door**, so what `Logged::new` sanitizes is what reaches a
/// state — not what the caller happened to hand it.
///
/// Stated here rather than in `fold.rs` because it is the only assertion in the
/// crate that ties the vocabulary to the door: the list above proves every kind is
/// clamped by `ActionKind::sanitized`, and this proves that is the function the
/// history actually runs.
#[test]
fn every_action_reaching_a_state_has_been_through_the_funnel() {
    #[derive(Clone)]
    struct Nothing;
    impl stark_model::document::Materialize for Nothing {
        type Ctx = ();
        fn fold(self, _action: &ActionKind2, _ctx: &mut ()) -> Self {
            self
        }
    }
    // The trait's `fold` takes an `Action`; alias it so the impl above reads as the
    // shape it is rather than as a path.
    type ActionKind2 = Action;

    for kind in kinds(f32::NAN) {
        if gated_at_apply(&kind) {
            continue;
        }
        let logged = stark_model::document::Logged::<Nothing>::new(Action {
            id: ActionId {
                lamport: 7,
                actor: ActorId(2),
            },
            kind: kind.clone(),
        });
        let text = format!("{:?}", logged.kind);
        assert!(
            !text.contains("NaN") && !text.contains("inf"),
            "{} reached a state unsanitized: {text}",
            kind.label(),
        );
    }
}

/// **A footprint stays small**, which is what makes [`Footprint::conflicts`]'s
/// nested scan the right shape for it.
///
/// `conflicts` is the hot one — `history` builds the centralizer once per removal
/// and then asks it about *every* later action — and it is `O(reads × writes)` on
/// both sides. That is only fine while the lists are short, so this pins that they
/// are: measured today, every kind claims at most 3 reads (`CommitStroke`, which
/// names the layer it paints on, the author's mask and the substrate that tooths
/// it) and at most 7 writes (`MergeLayerDown`, which names everything about both
/// layers), with `size_of::<Resource>() == 48` and `size_of::<Footprint>() == 48`.
///
/// A `Resource` grew from 32 bytes to 48 when a `LayerId` stopped being a `u64` and
/// became the id of the action that minted it (§17.9) — 24 bytes, which
/// `Resource::Paint`'s `TileRect` then rounds up. That is the price of the collision
/// class the id's shape rules out, paid in a list of at most ten.
///
/// The one kind that scales is `DuplicateLayer`, whose lists grow with the subtree
/// it copies — and that is exactly why `Resource::Layer` exists, collapsing nine
/// resources a layer into one. A new action that needed a dozen resources would not
/// be *wrong*, but it would want that treatment too, and this is where it says so.
///
/// It also settles whether the two `Vec`s should be inline storage: at these
/// lengths the allocations are two per commit, amortized against the GPU work a
/// commit already does, and the scan is over nine elements. Inline storage would
/// trade ~130 bytes per logged action for that, and buy a dependency. Measured, and
/// not worth it — recorded here so the question is not re-opened from intuition.
#[test]
fn a_footprint_stays_small_enough_for_a_nested_scan() {
    use stark_model::document::{Footprint, Resource, compute_footprint};

    const MAX_READS: usize = 3;
    const MAX_WRITES: usize = 7;

    assert_eq!(std::mem::size_of::<Resource>(), 48);
    assert_eq!(std::mem::size_of::<Footprint>(), 48);

    for kind in kinds(0.5) {
        let label = kind.label();
        let f = compute_footprint(&Action {
            id: ActionId {
                lamport: 1,
                actor: ActorId::SOLO,
            },
            kind,
        });
        assert!(
            f.reads.len() <= MAX_READS,
            "{label} reads {} resources; if that is right, it wants the coarse              `Resource::Layer` treatment rather than a longer list",
            f.reads.len(),
        );
        assert!(
            f.writes.len() <= MAX_WRITES,
            "{label} writes {} resources; see above",
            f.writes.len(),
        );
    }
}
