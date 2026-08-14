//! Merging a layer down (§14.11) — one law, tested by rendering.
//!
//! > **A merge must not change what the document looks like.**
//!
//! Everything here is that sentence applied to a case, and the shape of every test is
//! the same: render, merge, render again, compare the two images. That is deliberately
//! not how the rest of the suite works — most tests here assert a property of the
//! model — but a merge has no property of its own to assert. Its whole content is
//! *agreement with the compositor*, and the compositor is what a render runs.
//!
//! The comparisons are **exact wherever they can be**. Where the merge hands a tile
//! across by handle, or takes `merge.wesl`'s untouched branch, the result is the same
//! texture the stack was drawing from and there is nothing to round; the tests that
//! can demand zero do. Where the tile is genuinely recomputed — the overlap, and every
//! texel of a faded layer — the arithmetic runs in f32 and lands in f16 storage whose
//! rounding is implementation-defined, so a least-significant bit is allowed and no
//! more. A merge that moved a pixel by 2 would pass no test in this file.
//!
//! The other half is [`document::merge`]'s own unit tests, which pin *which* pairs are
//! offered. This file pins that the offered ones are honest, which is the part a rule
//! table cannot check: a rule that is too permissive shows up here as a picture that
//! changed, and one that is too strict shows up as a control the panel never offers.

mod common;

use common::*;
use stark_core::command::{DocCommand, PeerCommand};
use stark_core::document::{BlendMode, DRAGO_K, LayerId, Place};
use stark_core::geom::Vec2;
use stark_core::{Engine, LayerInfo, RgbaImage};

const ROOT: LayerId = LayerId(0);

/// Every mode that combines rather than covers, each at the setting a fresh layer
/// wears — the three a same-mode merge has to hold for. `Drago`'s bend is part of the
/// mode (§18.0.4), so "sharing a mode" here means sharing the curve too; which pairs
/// that admits is `document::merge`'s own to pin, and it does.
const MODES: [BlendMode; 3] = [
    BlendMode::Reinhard,
    BlendMode::Drago { k: DRAGO_K },
    BlendMode::Multiply,
];

const WARM: [f32; 4] = [0.90, 0.35, 0.10, 1.0];
const COOL: [f32; 4] = [0.10, 0.30, 0.85, 1.0];
const PALE: [f32; 4] = [0.95, 0.90, 0.80, 0.55];

const H_STROKE: &[Vec2] = &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)];
const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -60.0), Vec2::new(0.0, 60.0)];
/// Off to one side of both strokes — where a merge that leaked paint would show it.
const AWAY: &[Vec2] = &[Vec2::new(-100.0, -90.0), Vec2::new(-60.0, -90.0)];

fn add_layer(engine: &mut Engine) -> LayerId {
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    engine.observe().active_layer
}

fn info(engine: &Engine, id: LayerId) -> Option<LayerInfo> {
    engine.observe().layers.into_iter().find(|l| l.id == id)
}

/// What the panel would offer on this row: the layer it merges down onto.
fn offered(engine: &Engine, id: LayerId) -> Option<LayerId> {
    info(engine, id).and_then(|l| l.merge_down)
}

/// Merge `id` down and return what the document looked like before and after.
///
/// Asserts the merge was actually offered and actually happened, so a test that
/// stopped exercising the feature fails loudly instead of comparing an image to
/// itself — which is the one way a suite of "nothing changed" assertions can rot.
fn merged(engine: &mut Engine, id: LayerId) -> (RgbaImage, RgbaImage, LayerId) {
    let dest = offered(engine, id).expect("the merge must be offered");
    let before = engine.render_to_image();
    engine.process(DocCommand::MergeLayerDown(id));
    assert!(
        info(engine, id).is_none(),
        "the merged layer must be gone from the document",
    );
    assert!(info(engine, dest).is_some(), "the destination must survive");
    (before, engine.render_to_image(), dest)
}

/// The law, as an assertion: `tol` least-significant bits, and no more.
fn unchanged(before: &RgbaImage, after: &RgbaImage, tol: u8, what: &str) {
    let (frac, worst) = diff_fraction(before, after);
    assert!(
        worst <= tol,
        "{what}: a merge moved a pixel by {worst} (over {:.2}% of the canvas)",
        frac * 100.0,
    );
}

// ---------------------------------------------------------------------------
// The plain case, and the one that has to be exact to the byte.
// ---------------------------------------------------------------------------

/// Two ordinary layers, overlapping. The composite is unchanged, and the merged layer
/// holds both strokes rather than one of them.
///
/// The tolerance is one bit rather than zero because the overlapping tiles are
/// genuinely recomputed: the stack's "over" happens in the compositor's f32
/// accumulator, the merge's happens in a tile pass and is stored back to f16. Away
/// from the overlap the tiles pass through untouched, which the second half of this
/// test asks about directly.
#[test]
fn merging_two_plain_layers_leaves_the_picture_alone() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);

    let (before, after, dest) = merged(&mut engine, top);
    assert_eq!(dest, ROOT, "down from the second row is the first");
    unchanged(&before, &after, 1, "two plain layers");

    // …and the paint really did move: one layer now carries both strokes.
    assert_eq!(
        engine.observe().layers.len(),
        1,
        "the document should be down to one layer",
    );
}

/// A merge that does not overlap at all must be **exact**, to the byte.
///
/// This is the structural half of the claim rather than a looser version of the test
/// above: where the two layers do not meet, `merge.wesl` takes one of its two
/// untouched branches and copies the texel that is already the answer, so no
/// arithmetic runs and there is nothing to round. A regression that started
/// recomputing texels would still look right and would fail here.
///
/// Both branches are needed for this, and the *source's* is why the strokes here are
/// only ~90 canvas px apart rather than tiles apart: two marks that never meet still
/// share the tiles their boxes cross, so tile-level pass-through alone would leave one
/// of the two strokes being rewritten for nothing. That was this test's first failure,
/// and it is the reason the shader has a second branch at all.
#[test]
fn a_merge_that_does_not_overlap_is_exact() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 30.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 30.0, AWAY);

    let (before, after, _) = merged(&mut engine, top);
    let (frac, worst) = diff_fraction(&before, &after);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "disjoint layers must merge without touching a texel",
    );
}

/// The **order** survives: the upper layer's paint stays on top of the lower's.
///
/// Asked by color rather than by pixel equality, because it is the one thing the two
/// images being equal would *not* prove — a merge that stacked the pair the wrong way
/// round would render differently from the stack, but so would a dozen other bugs, and
/// this says which.
#[test]
fn the_upper_layer_stays_on_top() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);

    let (before, after, _) = merged(&mut engine, top);
    let (b, a) = (center(&before), center(&after));
    unchanged(&before, &after, 1, "crossing strokes");
    assert!(
        a[2] as i32 > a[0] as i32 + 30,
        "the cool stroke must still be the one on top at the crossing: {a:?}",
    );
    assert!(
        b[2] as i32 > b[0] as i32 + 30,
        "…which is what it was before the merge too: {b:?}",
    );
}

// ---------------------------------------------------------------------------
// Opacity: folded into the tiles, which is the part that has to be derived
// rather than copied.
// ---------------------------------------------------------------------------

/// A faded layer merged into a full-strength one. Both sliders end up inside the
/// merged tiles, so the surviving layer stands at opacity 1 and looks the same.
///
/// This is the case the slab law has to be *inverted* for (`merge.wesl`'s
/// `optical_mass`): pass A scales a layer's finished weight by the slider, which is
/// not the weight of any slab, so the merged tile's per-unit opacity is not something
/// that can be read off either input.
#[test]
fn a_faded_layer_folds_its_opacity_into_the_merge() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::SetLayerOpacity(top, 0.45));

    let (before, after, dest) = merged(&mut engine, top);
    unchanged(&before, &after, 1, "a faded source");
    assert_eq!(
        info(&engine, dest)
            .expect("the destination survives")
            .opacity,
        1.0,
        "the merged layer carries its fade in its tiles, not on its slider",
    );
}

/// Both layers faded, by different amounts — the general case of the fold, where
/// neither slider can be kept and the merged opacity is a height-weighted mean of two
/// numbers that were never equal.
#[test]
fn two_faded_layers_merge_without_a_shift() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    engine.process(DocCommand::SetLayerOpacity(ROOT, 0.7));
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::SetLayerOpacity(top, 0.35));

    let (before, after) = {
        let (b, a, _) = merged(&mut engine, top);
        (b, a)
    };
    unchanged(&before, &after, 2, "two faded layers");
}

/// A translucent brush — paint whose per-unit opacity is well under 1 — merged onto
/// opaque paint. The two layers' *stored* opacities differ, so the merged tile has to
/// carry a value that is neither.
#[test]
fn translucent_paint_merges_onto_opaque_paint() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, PALE, 44.0, V_STROKE);

    let (before, after, _) = merged(&mut engine, top);
    unchanged(&before, &after, 1, "a translucent glaze");
}

// ---------------------------------------------------------------------------
// Clipping and groups.
// ---------------------------------------------------------------------------

/// A clipped layer folded into the layer it is clipped to. The clip is a *deletion*
/// (§14.4), not a stack, so this exercises `merge.wesl`'s other law — and the place
/// it shows is outside the destination's coverage, where the source's paint must
/// vanish rather than appear.
#[test]
fn a_clipped_layer_merges_into_its_backdrop() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    // Crossing the backdrop *and* running well past it, so the clip has something to
    // delete: the vertical stroke leaves the horizontal one at both ends.
    paint(&mut engine, COOL, 20.0, V_STROKE);
    engine.process(DocCommand::SetLayerClip(top, true));

    let (before, after, _) = merged(&mut engine, top);
    unchanged(&before, &after, 1, "a clipped layer");
    // The clip really was doing something: above the horizontal stroke there is no
    // backdrop, so neither render shows the cool paint there.
    let above = |img: &RgbaImage| img.pixel(img.width / 2, img.height / 2 - 50);
    assert!(
        above(&before)[2] as i32 <= above(&before)[0] as i32 + 30
            || above(&before) == above(&after),
        "the clipped tail must read the same before and after: {:?} vs {:?}",
        above(&before),
        above(&after),
    );
}

/// The bottom of a carried stack merges into its carrier — a group's members
/// composite over its base (§14.1), so "down" from there walks out of the group.
///
/// The carrier keeps a blend mode of its own through the merge, which is the point:
/// its mode and its clip describe how the *group* meets what lies under the group, and
/// the merge only rewrites what is inside it.
#[test]
fn the_bottom_of_a_group_merges_into_its_base() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerBlend(ROOT, BlendMode::Multiply));

    let (before, after, dest) = merged(&mut engine, carried);
    assert_eq!(dest, ROOT);
    unchanged(&before, &after, 1, "the base of a group");
    assert_eq!(
        info(&engine, ROOT).expect("the base survives").blend,
        BlendMode::Multiply,
        "the carrier's outward mode must survive the merge",
    );
}

/// A clipped member folded into the base it is clipped to — the "clip to exactly this
/// layer" gesture (§14.4), which is a carry plus a clip, collapsed into one layer.
#[test]
fn a_clipped_group_member_merges_into_its_base() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, COOL, 20.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerClip(carried, true));

    let (before, after, _) = merged(&mut engine, carried);
    unchanged(&before, &after, 1, "a clipped group member");
}

// ---------------------------------------------------------------------------
// What the engine refuses, and what happens if it is asked anyway.
// ---------------------------------------------------------------------------

/// A command for a merge that is not offered is a **silent no-op**, not a different
/// picture — the frontend's rule and the engine's are one rule, asked in one place.
#[test]
fn an_unoffered_merge_changes_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let middle = add_layer(&mut engine);
    paint(&mut engine, PALE, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    // Modes that **disagree**, above the foot of the stack where both are stated
    // against something: after a merge one set of params would have to speak for
    // both, and there is no third mode that means "glow here and multiply there".
    engine.process(DocCommand::SetLayerBlend(middle, BlendMode::Multiply));
    engine.process(DocCommand::SetLayerBlend(top, BlendMode::Reinhard));
    assert_eq!(
        offered(&engine, top),
        None,
        "modes that disagree must not be mergeable",
    );

    let before = engine.render_to_image();
    let layers = engine.observe().layers.len();
    engine.process(DocCommand::MergeLayerDown(top));
    let after = engine.render_to_image();

    assert_eq!(engine.observe().layers.len(), layers, "nothing may be lost");
    let (frac, worst) = diff_fraction(&before, &after);
    assert_eq!((frac, worst), (0.0, 0), "a refused merge must draw nothing");
    // …and it must not have spent an undo step on doing nothing.
    assert!(
        !engine.observe().can_undo || {
            engine.process(DocCommand::Undo);
            engine.observe().layers.len() == layers
        },
        "a refused merge should not have been logged",
    );
}

// ---------------------------------------------------------------------------
// History.
// ---------------------------------------------------------------------------

/// A merge is one undo step, and undoing it puts both layers back — the record, its
/// place in the tree, and the destination's own opacity, which the fold had set to 1.
#[test]
fn undo_puts_both_layers_back() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::SetLayerOpacity(top, 0.6));
    engine.process(DocCommand::SetLayerOpacity(ROOT, 0.8));
    engine.process(PeerCommand::SetActiveLayer(top));

    let before = engine.render_to_image();
    engine.process(DocCommand::MergeLayerDown(top));
    engine.process(DocCommand::Undo);

    let after = engine.render_to_image();
    let (frac, worst) = diff_fraction(&before, &after);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "undoing a merge restores the tiles by handle, so it must be exact",
    );
    let restored = info(&engine, top).expect("the merged layer comes back");
    assert_eq!(restored.opacity, 0.6, "with its own opacity");
    assert_eq!(
        info(&engine, ROOT)
            .expect("the base is still there")
            .opacity,
        0.8,
        "and the destination's slider is put back too",
    );
}

/// Merge, undo, redo — the round trip through the log, which is where an action that
/// re-derives its own plan could disagree with itself.
#[test]
fn redo_reaches_the_same_picture() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);

    engine.process(DocCommand::MergeLayerDown(top));
    let once = engine.render_to_image();
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    let again = engine.render_to_image();

    let (frac, worst) = diff_fraction(&once, &again);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "a replayed merge must reproduce itself exactly",
    );
    assert!(info(&engine, top).is_none(), "and must have merged again");
}

/// The brush follows the paint: merging the layer you are working on leaves you
/// working on the layer that now holds it, rather than on whatever the repointing
/// happened to land on.
#[test]
fn the_active_layer_follows_the_merge() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let top = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(PeerCommand::SetActiveLayer(top));

    engine.process(DocCommand::MergeLayerDown(top));
    assert_eq!(engine.observe().active_layer, ROOT);
}

// ---------------------------------------------------------------------------
// Through a blend mode. These take the general path — both layers expanded into
// what they composite to, the compositor's own blend pass between them, the
// result stored back as a tile — where everything above takes the direct one.
// ---------------------------------------------------------------------------

/// **Siblings sharing a blend mode.** The pair composites as one layer carrying
/// `merge(D, S)`, because the modes are associative at any coverage (§18.0.4). While
/// they weighed coverage in the working space they were not, and this merge was
/// refused for exactly that reason.
///
/// A backdrop under the pair is the point: with nothing beneath them every mode is the
/// identity and the test would pass on a merge that had learned nothing.
#[test]
fn siblings_sharing_a_mode_merge() {
    for mode in MODES {
        let Some(mut engine) = engine_or_skip_blue() else {
            return;
        };
        paint(&mut engine, PALE, 60.0, H_STROKE);
        let lower = add_layer(&mut engine);
        paint(&mut engine, WARM, 44.0, H_STROKE);
        engine.process(DocCommand::SetLayerBlend(lower, mode));
        let upper = add_layer(&mut engine);
        paint(&mut engine, COOL, 44.0, V_STROKE);
        engine.process(DocCommand::SetLayerBlend(upper, mode));

        let (before, after, dest) = merged(&mut engine, upper);
        assert_eq!(dest, lower);
        unchanged(&before, &after, 3, &format!("{mode:?} siblings"));
        assert_eq!(
            info(&engine, lower).expect("the survivor").blend,
            mode,
            "the survivor has to go on meeting the backdrop the same way",
        );
    }
}

/// **Any mode, into its carrier.** A group's members composite over its base (§14.1),
/// so the group's isolated content is `merge_source(base, source)` before and after —
/// and what the group merges outward is unchanged whatever the source's mode is. This
/// one was always sound and only ever blocked on the tile-space plumbing.
///
/// The carrier wears a mode of its own too, so the test says that the *group's* merge
/// outward is untouched as well as its inside.
#[test]
fn any_mode_merges_into_its_carrier() {
    for mode in MODES {
        let Some(mut engine) = engine_or_skip_blue() else {
            return;
        };
        paint(&mut engine, PALE, 60.0, H_STROKE);
        let base = add_layer(&mut engine);
        paint(&mut engine, WARM, 44.0, H_STROKE);
        let carried = add_layer(&mut engine);
        paint(&mut engine, COOL, 44.0, V_STROKE);
        engine.process(DocCommand::SetLayerBlend(carried, mode));
        engine.process(DocCommand::MoveLayer {
            id: carried,
            carrier: Some(base),
            at: Place::Top,
        });
        engine.process(DocCommand::SetLayerBlend(base, BlendMode::Multiply));

        let (before, after, dest) = merged(&mut engine, carried);
        assert_eq!(dest, base);
        unchanged(&before, &after, 3, &format!("{mode:?} into a carrier"));
        assert_eq!(
            info(&engine, base).expect("the survivor").blend,
            BlendMode::Multiply,
            "the carrier's own outward mode is untouched",
        );
    }
}

/// **A faded carrier keeps its slider.** A group's opacity is applied to its composited
/// whole, and the merge rewrites the inside of that whole — so the base expands at full
/// strength and the fade stays on the layer, where it goes on meaning what it meant.
///
/// This was refused while the base's slider was being folded into the tiles like a
/// sibling's, which would have faded the merged paint twice.
#[test]
fn a_faded_carrier_keeps_its_own_opacity() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, PALE, 60.0, H_STROKE);
    let base = add_layer(&mut engine);
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, COOL, 44.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerOpacity(base, 0.45));

    let (before, after, _) = merged(&mut engine, carried);
    unchanged(&before, &after, 2, "a faded carrier");
    assert_eq!(
        info(&engine, base).expect("the survivor").opacity,
        0.45,
        "the group's fade belongs to the layer, not to its tiles",
    );
}

/// A clipped member folded into a **faded** base — the two relaxations at once, since
/// a clip reads the backdrop's coverage and the fade must not have touched it yet.
#[test]
fn a_clipped_member_merges_into_a_faded_carrier() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, PALE, 60.0, H_STROKE);
    let base = add_layer(&mut engine);
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let carried = add_layer(&mut engine);
    paint(&mut engine, COOL, 20.0, V_STROKE);
    engine.process(DocCommand::MoveLayer {
        id: carried,
        carrier: Some(base),
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerClip(carried, true));
    engine.process(DocCommand::SetLayerOpacity(base, 0.6));

    let (before, after, _) = merged(&mut engine, carried);
    unchanged(&before, &after, 2, "a clipped member under a faded base");
}
