//! Layer groups and clipping (§14) — one feature, tested as one.
//!
//! A layer carries layers; a layer's blend mode, clip and opacity describe how it
//! *together with everything it carries* meets what lies beneath it. Everything here
//! is an invariant of that sentence rather than a look, and they come in two kinds.
//!
//! **The ones that must be exact.** Grouping layers that use no modes must produce
//! the identical picture, to the byte, because "I tidied my layers and my painting
//! changed" is the failure this design exists to prevent — and because the exactness
//! is *structural*: [`CompositeGroup::stack`] collapses such a group back into a plain
//! run at build time, so the compositor is handed the same draw list it would have
//! been handed without the group at all (§14.7 rule 2). A test that tolerated a few
//! least-significant bits here would pass just as happily if that collapse silently
//! stopped happening.
//!
//! **The ones about clipping's algebra.** A clipped layer must contribute no
//! coverage of its own (§14.4.1) — over an opaque backdrop it is exactly an unclipped
//! layer, over nothing it is nothing at all, and there is no middle case where it
//! invents opacity the group did not have. That last one is what the "obvious"
//! implementation gets wrong, so it is tested directly rather than inferred from
//! the two ends.

mod common;

use common::*;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_engine::{Engine, LayerInfo, RgbaImage};
use stark_model::Srgb;
use stark_model::document::{BlendMode, LayerId, MatteRegion, Parcel, Place};
use stark_model::geom::Vec2;

const ROOT: LayerId = LayerId(0);

const WARM: [f32; 3] = [0.90, 0.35, 0.10];
const COOL: [f32; 3] = [0.10, 0.30, 0.85];
const PALE: [f32; 3] = [0.95, 0.90, 0.80];

const H_STROKE: &[Vec2] = &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)];
const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -60.0), Vec2::new(0.0, 60.0)];
/// Far off to one side of both strokes: nothing is painted here on any layer, so it
/// is where "did this layer leak outside its clip?" is asked.
const AWAY: &[Vec2] = &[Vec2::new(-100.0, -100.0), Vec2::new(-70.0, -100.0)];

/// Add a layer at the top of the document and return its id.
fn add_layer(engine: &mut Engine) -> LayerId {
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    engine.observe().active_layer
}

/// Every layer, flattened in composite order (the projection's own order).
fn layers(engine: &Engine) -> Vec<LayerInfo> {
    engine.observe().layers.to_vec()
}

fn info(engine: &Engine, id: LayerId) -> LayerInfo {
    layers(engine)
        .into_iter()
        .find(|l| l.id == id)
        .expect("layer is in the document")
}

/// Assert two renders are the *same image*, byte for byte.
fn assert_identical(a: &RgbaImage, b: &RgbaImage, what: &str) {
    let (frac, worst) = diff_fraction(a, b);
    assert_eq!((frac, worst), (0.0, 0), "{what}");
}

// ---------------------------------------------------------------------------
// Structure: what carrying does to the document.
// ---------------------------------------------------------------------------

/// Carrying is a move, and it moves the whole subtree. Releasing puts it back
/// beside the group rather than inside it (§14.8).
#[test]
fn carry_and_release_move_the_whole_subtree() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let mid = add_layer(&mut engine);
    let top = add_layer(&mut engine);

    // top onto mid, then mid (carrying top) onto root.
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(mid),
        at: Place::Top,
    });
    engine.process(DocCommand::MoveLayer {
        id: mid,
        carrier: Some(ROOT),
        at: Place::Top,
    });

    let ls = layers(&engine);
    assert_eq!(
        ls.iter().map(|l| (l.id, l.depth)).collect::<Vec<_>>(),
        vec![(ROOT, 0), (mid, 1), (top, 2)],
        "composite order is base-first, and depth counts the carrying"
    );
    assert_eq!(info(&engine, mid).carrier, Some(ROOT));
    assert_eq!(info(&engine, top).carrier, Some(mid));
    assert!(info(&engine, ROOT).is_group);

    // Releasing `mid` lifts `top` out with it — a group travels as one.
    engine.process(DocCommand::MoveLayer {
        id: mid,
        carrier: None,
        at: Place::Above(ROOT),
    });
    assert_eq!(info(&engine, mid).carrier, None);
    assert_eq!(
        info(&engine, top).carrier,
        Some(mid),
        "what it carried came with it"
    );
}

/// A layer may not carry its own ancestor. Both halves of a cycle are `MoveLayer`s
/// that conflict on `StackOrder`, so the log's total order serializes them and
/// whichever applies second sees the first's result and declines (§14.8).
#[test]
fn a_layer_cannot_carry_its_own_ancestor() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let top = add_layer(&mut engine);
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    let before = layers(&engine);

    // The closing half of the cycle, and the degenerate self-carry.
    engine.process(DocCommand::MoveLayer {
        id: ROOT,
        carrier: Some(top),
        at: Place::Top,
    });
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(top),
        at: Place::Top,
    });

    let after = layers(&engine);
    assert_eq!(
        after.iter().map(|l| (l.id, l.depth)).collect::<Vec<_>>(),
        before.iter().map(|l| (l.id, l.depth)).collect::<Vec<_>>(),
        "a cycle is declined, leaving the tree exactly as it was"
    );
}

/// The foot of a stack is a place a move can land in — in the document's own stack
/// and inside a group alike (§14.8).
///
/// It is the one position `Place::Above` cannot name, because a stack of `n` layers
/// has `n + 1` places and only `n` siblings to state them against. Both ends are
/// tested because the two take different paths through `move_layer` — the root stack
/// is spliced directly, a group's is spliced inside the carrier — and a panel that
/// can drop a layer behind everything at one depth and not the other would be a
/// panel with a hole in it.
#[test]
fn a_layer_can_be_moved_to_the_foot_of_a_stack() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let mid = add_layer(&mut engine);
    let top = add_layer(&mut engine);

    // Under the root layer, which no `Above` in the document could have said.
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: None,
        at: Place::Bottom,
    });
    assert_eq!(
        layers(&engine).iter().map(|l| l.id).collect::<Vec<_>>(),
        vec![top, ROOT, mid],
        "composite order is bottom-first, so the foot of the stack comes first"
    );
    assert!(
        !info(&engine, top).has_backdrop,
        "and it is now the layer with nothing under it"
    );

    // The same place inside a group: `mid` carries `top`, then `ROOT` is dropped
    // under `top` — the foot of `mid`'s carried stack, not the base of the group.
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(mid),
        at: Place::Top,
    });
    engine.process(DocCommand::MoveLayer {
        id: ROOT,
        carrier: Some(mid),
        at: Place::Bottom,
    });
    assert_eq!(
        layers(&engine)
            .iter()
            .map(|l| (l.id, l.depth))
            .collect::<Vec<_>>(),
        vec![(mid, 0), (ROOT, 1), (top, 1)],
        "under everything the group carries, and `mid` is still the base"
    );
}

/// Removing a group removes what it carries, and undo brings the whole subtree
/// back — a `Layer` owns its subtree, so the patch that restores it restores all of
/// it (§14.8).
#[test]
fn removing_a_group_takes_its_carried_layers_and_undo_returns_them() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let mid = add_layer(&mut engine);
    let top = add_layer(&mut engine);
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(mid),
        at: Place::Top,
    });
    let before = layers(&engine);

    engine.process(DocCommand::RemoveLayer(mid));
    let ids: Vec<LayerId> = layers(&engine).iter().map(|l| l.id).collect();
    assert_eq!(ids, vec![ROOT], "the group went as one");

    engine.process(DocCommand::Undo);
    let after = layers(&engine);
    assert_eq!(
        after
            .iter()
            .map(|l| (l.id, l.depth, l.carrier))
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|l| (l.id, l.depth, l.carrier))
            .collect::<Vec<_>>(),
        "undo restores the shape, not just the layers"
    );
}

/// `has_backdrop` is the one predicate behind both relational controls (§14.4.3): true
/// wherever something composites beneath, false on exactly one row — the bottom of
/// the document — including for a layer carried by it.
#[test]
fn only_the_bottom_of_the_document_has_no_backdrop() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let mid = add_layer(&mut engine);
    let carried = add_layer(&mut engine);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(ROOT),
        at: Place::Top,
    });

    assert!(
        !info(&engine, ROOT).has_backdrop,
        "nothing is under the bottom of the document"
    );
    assert!(
        info(&engine, carried).has_backdrop,
        "a carried layer has its group's base under it"
    );
    assert!(info(&engine, mid).has_backdrop);
}

// ---------------------------------------------------------------------------
// Exactness: organization is free.
// ---------------------------------------------------------------------------

/// **The invariant this design exists for.** Grouping layers that use no modes and
/// no clipping renders the identical picture, to the byte.
///
/// Not "close enough": §14.7 rule 2 collapses such a group back into a plain run at
/// build time, so the compositor receives the same draw list it would have received
/// with no group at all. Any drift here means the collapse stopped happening and the
/// group is being isolated through two extra render passes that are supposed to be
/// provably unnecessary.
#[test]
fn grouping_plain_layers_changes_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    let ungrouped = engine.render_to_image();

    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    let grouped = engine.render_to_image();

    assert_identical(&ungrouped, &grouped, "grouping plain layers must be free");
}

/// Carrying and releasing again is the identity — the structural round trip, byte
/// for byte. What it guards is the whole tree-surgery path: `move_layer`'s remove
/// and splice, the bounds recomputation over the nested stacks, and the collapse
/// that has to fire again on the way back.
#[test]
fn carry_then_release_returns_the_same_picture() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    let before = engine.render_to_image();

    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: None,
        at: Place::Above(ROOT),
    });
    let after = engine.render_to_image();

    assert_identical(&before, &after, "carry then release is the identity");
}

/// Duplicating a group copies the **whole subtree** — one copy per carried layer,
/// nested the way the original is — and lands it beside the group (§14.8). The
/// subtree travels as one for the reason removing it does: the subtree *is* the
/// group (§14.2).
#[test]
fn duplicating_a_group_copies_the_whole_subtree() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let base = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, PALE, 20.0, AWAY);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });
    let before = engine.render_to_image();

    engine.process(DocCommand::DuplicateLayer(base));
    let ls = layers(&engine);
    assert_eq!(ls.len(), 5, "root, the group's two layers, and two copies");
    let base_copy = engine.observe().active_layer;
    let carried_copy = ls
        .iter()
        .find(|l| l.carrier == Some(base_copy))
        .expect("the copy carries a copy")
        .id;
    assert_eq!(
        ls.iter().map(|l| (l.id, l.depth)).collect::<Vec<_>>(),
        vec![
            (ROOT, 0),
            (base, 0),
            (carried, 1),
            (base_copy, 0),
            (carried_copy, 1)
        ],
        "the copied group sits directly above the group it was copied from, \
         nested the same way"
    );

    // Hiding the copy hides everything it carries, so what is left is the document
    // as it stood — the copy is the only difference, byte for byte.
    engine.process(DocCommand::SetLayerVisible(base_copy, false));
    assert_identical(
        &before,
        &engine.render_to_image(),
        "a hidden copy of a group leaves the original picture",
    );
}

/// **The cost of having no pass-through, asserted rather than assumed** (§14.5).
///
/// A group is always isolated, so a blend mode *inside* one blends against the group
/// rather than against what lies under the group — and putting a glow layer into a
/// group therefore changes the picture. This is the one place this model is worse
/// than Photoshop's, and the test exists so that the day someone decides it is a bug
/// and "fixes" it, they have to come here and read why it is not.
///
/// What buys it back is §14.5's other half, which the modes were built for: every mode
/// past `Normal` is a conjugation of addition, so regrouping layers that share a
/// mode is exactly invariant. It is only mixing scopes, as here, that moves.
#[test]
fn grouping_rescopes_an_interior_blend_mode() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let base = add_layer(&mut engine);
    paint(&mut engine, PALE, 44.0, H_STROKE);
    let glow = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::SetLayerBlend(glow, BlendMode::Reinhard));
    let ungrouped = engine.render_to_image();

    // `glow` now combines light with `base` alone, not with everything under it.
    engine.process(DocCommand::MoveLayer {
        id: glow,
        carrier: Some(base),
        at: Place::Top,
    });
    let grouped = engine.render_to_image();

    let (frac, _) = diff_fraction(&ungrouped, &grouped);
    assert!(
        frac > 0.0,
        "a group bounds what its members blend against \u{2014} that is what a group is"
    );
}

/// Hiding a group hides what it carries: the group is the layer, so its visibility
/// is the group's (§14.3).
#[test]
fn hiding_a_group_hides_what_it_carries() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let bare = engine.render_to_image();

    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Top,
    });

    engine.process(DocCommand::SetLayerVisible(ROOT, false));
    let hidden = engine.render_to_image();
    assert_identical(&bare, &hidden, "the carried layer went with its base");
}

// ---------------------------------------------------------------------------
// Clipping: the algebra of §14.4.1.
// ---------------------------------------------------------------------------

/// Over **nothing**, a clipped layer is nothing. The one case where a clip and a
/// blend mode differ at the same place: a mode over an empty backdrop is the
/// identity, a clip erases (§14.4.3).
#[test]
fn clipping_over_an_empty_backdrop_shows_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let bare = engine.render_to_image();

    // A layer over an empty root: there is no paint under it anywhere.
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::SetLayerClip(top, true));
    let clipped = engine.render_to_image();

    assert_identical(
        &bare,
        &clipped,
        "clipped to paint that is not there, so nothing shows",
    );
}

/// Over **solid** paint, a clipped layer is the unclipped one.
///
/// The backdrop has to be genuinely solid for this, which is why the base here is a
/// matte rather than a stroke: coverage is the slab law `1 − exp(−K·α·h)`, so a
/// brushed passage reads about 0.98 even at its heaviest, and a clip legitimately
/// removes that last 2% of the layer above — a difference of ~5/255, which is the
/// clip working rather than failing. A matte lays thickness 8, so `αb ≈ 0.9997`.
///
/// **Why the tolerance is 2 rather than 0.** The two renders take different paths to
/// the same arithmetic: unclipped, the layer draws under fixed-function `over`;
/// clipped, it is isolated and merged by the blend shader. `merge` is written so the
/// two coincide where `αb` is exactly 1 — for `Normal` it takes the premultiplied
/// source verbatim rather than dividing by alpha and multiplying it back
/// (`blend_common.wesl`) — but two things still separate them here: the matte's own
/// `1 − exp(−8)` is not quite 1, and the half-float targets round a shader's
/// `mix`-and-store differently from a fixed-function blend. Both are bounded by a
/// couple of least-significant bits, and neither is what this test is looking for: a
/// clip that wrongly removed a *visible* share of the layer would show up here as
/// tens of LSB, the same size as the 5 a merely-heavy stroke would produce.
#[test]
fn clipping_inside_solid_paint_is_the_unclipped_result() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // A matte whose hole is off-canvas, so it covers the viewport solidly.
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(1e4, 1e4),
            max: Vec2::new(2e4, 2e4),
        },
        paint: Parcel::Solid(Srgb::new([0.5, 0.4, 0.3])),
    });
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 20.0, H_STROKE);
    let unclipped = engine.render_to_image();

    engine.process(DocCommand::SetLayerClip(top, true));
    let clipped = engine.render_to_image();

    let (_, worst) = diff_fraction(&unclipped, &clipped);
    assert!(
        worst <= 2,
        "inside solid paint a clip removes nothing; worst channel diff {worst}"
    );
}

/// A clipped layer contributes **no coverage of its own**: everywhere the backdrop
/// is empty, the result is what the backdrop alone renders — even though the clipped
/// layer has paint there.
///
/// This is the case the natural-but-wrong implementation fails. Scaling the source's
/// alpha by the backdrop's (`αs ← αs·αb`) passes both tests above and still leaks
/// here, because it makes the *result* alpha `αs·αb + αb(1−αs·αb)` rather than `αb`.
#[test]
fn a_clipped_layer_adds_no_coverage_where_the_backdrop_is_empty() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let base_only = engine.render_to_image();

    // Paint on a clipped layer somewhere the base has nothing.
    let top = add_layer(&mut engine);
    paint(&mut engine, PALE, 30.0, AWAY);
    engine.process(DocCommand::SetLayerClip(top, true));
    let clipped = engine.render_to_image();

    assert_identical(
        &base_only,
        &clipped,
        "a clipped layer must not paint where its backdrop is empty \u{2014} \
         not even faintly, and not in the height field",
    );
}

/// Clipping inherits the alpha of the **whole stack below it in its group**, not of
/// the one layer under it (§14.4).
///
/// Three layers in one group. The clipped one paints across a region where only the
/// *base* has paint and the layer immediately below it does not — a nearest-neighbour
/// clip, which is what every other app implements, would cut it away there. So the
/// test asks whether it still contributes, by comparing against the same stack with
/// the clipped layer absent entirely.
#[test]
fn clipping_inherits_the_whole_stack_below_it() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Base: a wide horizontal bar. Middle: a vertical bar crossing it.
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let middle = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: middle,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    let without_top = engine.render_to_image();

    // A third layer, clipped, painting along the whole horizontal bar — including
    // the far end, which `middle` does not reach.
    let top = add_layer(&mut engine);
    paint(&mut engine, PALE, 20.0, H_STROKE);
    engine.process(DocCommand::SetLayerClip(top, true));
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Above(middle),
    });
    let with_top = engine.render_to_image();

    // Well clear of the vertical bar (radius 44 about x = 0), but inside the
    // horizontal one: here the clipped layer inherits the *base's* coverage, two
    // layers down.
    let x = with_top.width / 2 + 70;
    let y = with_top.height / 2;
    assert_ne!(
        with_top.pixel(x, y),
        without_top.pixel(x, y),
        "a clip reads the whole stack below it, not just the layer under it"
    );
}

/// The base's clip points **outward**: it clips the composited group to what lies
/// beneath the *group*, exactly as its blend mode does (§14.4.3).
#[test]
fn clipping_the_base_clips_the_whole_group() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Something under the group to clip to: a narrow horizontal band, so the
    // vertical group below reaches well past it top and bottom.
    paint(&mut engine, WARM, 26.0, H_STROKE);

    // A group of two, both painting the full height of the vertical stroke.
    let base = add_layer(&mut engine);
    paint(&mut engine, COOL, 16.0, V_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, PALE, 8.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });

    let unclipped = engine.render_to_image();
    engine.process(DocCommand::SetLayerClip(base, true));
    let clipped = engine.render_to_image();

    // Beyond the horizontal band the group has nothing to sit on, so the whole
    // group — the carried layer included — is gone there.
    let x = clipped.width / 2;
    let beyond = clipped.height / 2 - 60;
    assert_ne!(
        clipped.pixel(x, beyond),
        unclipped.pixel(x, beyond),
        "past the paint beneath it, a clipped group must not show"
    );
    // And the layer it carries went with it — clipping the base clips the group,
    // not just the base's own paint.
    engine.process(DocCommand::SetLayerVisible(carried, false));
    let without_carried = engine.render_to_image();
    assert_eq!(
        clipped.pixel(x, beyond),
        without_carried.pixel(x, beyond),
        "the carried layer is clipped by its base's clip too"
    );
}

// ---------------------------------------------------------------------------
// The group is functional: its base's blend mode is the group's.
// ---------------------------------------------------------------------------

/// The base's blend mode is the group's, and it applies to the group's *composite*.
/// So a group whose base multiplies darkens what is under the group — including
/// where only the carried layer has paint, which is the whole difference from the
/// base's mode applying to the base alone.
#[test]
fn the_bases_blend_mode_is_the_groups() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, PALE, 90.0, H_STROKE);

    let base = add_layer(&mut engine);
    paint(&mut engine, WARM, 20.0, H_STROKE);
    let carried = add_layer(&mut engine);
    // The carried layer paints somewhere the base does not.
    paint(&mut engine, COOL, 20.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });

    let normal = engine.render_to_image();
    engine.process(DocCommand::SetLayerBlend(base, BlendMode::Multiply));
    let multiplied = engine.render_to_image();

    // A point on the carried layer's stroke, inside the pale bar but away from the
    // base's own stroke: the mode reached it, so it belongs to the group.
    let x = normal.width / 2;
    let y = normal.height / 2 - 30;
    assert_ne!(
        normal.pixel(x, y),
        multiplied.pixel(x, y),
        "the base's mode applies to the whole group, carried layers included"
    );
}

/// Group opacity fades the group as a unit, and is not the same as fading its
/// members: two overlapping members at 50% each show their seam, the group at 50%
/// does not (§14.3).
#[test]
fn group_opacity_fades_the_composite_not_the_members() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let base = ROOT;
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let carried = add_layer(&mut engine);
    // Overlapping the base almost exactly, so "fade each" and "fade the composite"
    // differ by the most they can.
    paint(&mut engine, WARM, 44.0, H_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });

    engine.process(DocCommand::SetLayerOpacity(base, 0.5));
    let group_faded = engine.render_to_image();

    engine.process(DocCommand::SetLayerOpacity(base, 1.0));
    engine.process(DocCommand::SetLayerOpacity(carried, 0.5));
    let member_faded = engine.render_to_image();

    let (frac, _) = diff_fraction(&group_faded, &member_faded);
    assert!(
        frac > 0.0,
        "fading a group is not fading its members one at a time"
    );
}

/// A group's opacity fades its **base** exactly once.
///
/// The test above says the two granularities differ; it does not say by how much, and
/// for a long time the answer was wrong in a way it could not see. `layer_items` tags
/// every item with the layer's opacity — right for a leaf, where that is the only
/// place the slider is applied — and the group built from those items then applied it
/// *again* at the merge. So a group base at 0.5 drew its own paint at 0.25 while
/// everything it carried drew at 0.5. The slider still faded, the two granularities
/// still differed, and nothing in the suite contradicted it.
///
/// What pins it is a case where the two granularities **coincide**: where a group's
/// members do not overlap, fading the composite and fading each member are the same
/// operation, so a group at opacity `a` must render exactly as the same layers
/// ungrouped at opacity `a`. A base faded twice comes out at `a²` and breaks the
/// equality, while every layer it carries stays right — which is what makes this a
/// test of the base in particular rather than of group opacity in general.
#[test]
fn a_groups_opacity_fades_its_base_exactly_once() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    const FADE: f32 = 0.5;

    // The base's paint and the carried layer's, far enough apart that no texel of one
    // is under the other — which is the whole reason the two answers coincide here.
    paint(&mut engine, WARM, 26.0, H_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, COOL, 26.0, AWAY);

    // Grouped: one slider, on the group, faded at its merge.
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerOpacity(ROOT, FADE));
    let grouped = engine.render_to_image();

    // Ungrouped: two sliders, one per layer, folded into their tiles.
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: None,
        at: Place::Above(ROOT),
    });
    engine.process(DocCommand::SetLayerOpacity(carried, FADE));
    let flat = engine.render_to_image();

    // Not to the byte: the grouped path scales the accumulator in the blend pass
    // while the flat one scales each tile in pass A, and the two round differently at
    // half precision. A base faded twice would be out by tens of levels over the whole
    // stroke, not by a bit at its edges.
    let bad = frac_exceeding(&grouped, &flat, 2);
    assert!(
        bad < 0.001,
        "a group of non-overlapping members must fade as one layer would: \
         {:.2}% of the canvas differs by more than 2 levels",
        bad * 100.0,
    );
}

// ---------------------------------------------------------------------------
// The projection the panel is built on.
// ---------------------------------------------------------------------------

/// The active layer survives its group being removed, landing on something that
/// still exists — including when the replacement is itself carried (§14.2).
#[test]
fn the_active_layer_repoints_out_of_a_removed_group() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let mid = add_layer(&mut engine);
    let top = add_layer(&mut engine);
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(mid),
        at: Place::Top,
    });
    engine.process(PeerCommand::SetActiveLayer(top));
    assert_eq!(engine.observe().active_layer, top);

    engine.process(DocCommand::RemoveLayer(mid));
    let obs = engine.observe();
    assert!(
        obs.layers.iter().any(|l| l.id == obs.active_layer),
        "the active layer must exist; got {:?} of {:?}",
        obs.active_layer,
        obs.layers.iter().map(|l| l.id).collect::<Vec<_>>()
    );
}
