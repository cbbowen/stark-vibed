//! A **known, unfixed** defect in how layers composite over one another
//! (DESIGN.md §6.3, pass A). The test is `#[ignore]`d because it fails: it is here
//! as a repro, not as a guard.
//!
//! **The claim it makes.** A layer should affect what is beneath it only as much as
//! it is visible at all. It sounds too obvious to test, and it is false today.
//!
//! **Why.** A tile stores paint as a per-unit *opacity* plus an *amount* (§6.1), and
//! neither alone says how much of a fragment is paint — the media pass rightly
//! combines them as a translucent slab, `1 − exp(−K · opacity · amount)`. But pass A
//! composites layers with premultiplied "over" weighted by the per-unit **opacity
//! alone**, so a film with opacity 1 and no thickness — which the media pass draws as
//! nothing over bare canvas — replaces the colour outright over another layer's
//! paint.
//!
//! That state is not exotic. A soft tip deposits opacity as `1 − exp(−op·τ)` and
//! height linearly in `τ` (`stamp_oklab.wesl`), so the opacity saturates long before
//! the height does and every soft brush has a wide fringe of exactly it. The symptom
//! is a ghost of the brush's whole footprint painted over the layer below, visible
//! only where there is something to paint it onto.
//!
//! **The fix, and why it is not here.** Pass A should weigh each layer by its own
//! visible alpha — the same slab law `paint_common.wesl` already uses to stack
//! parcels *within* a layer — and the media pass should then read the accumulated
//! coverage instead of re-deriving it. That was written and it does fix this test and
//! the reported artifact, and for a single layer it is algebraically the identity. It
//! also, unexplained, moves the **smear** path: `golden_smudge_drag`,
//! `golden_self_smear`, `golden_selection_smear` and — the part that matters, because
//! it is an invariant rather than a golden — `a_smear_stroke_previews_as_it_commits`.
//! A preview that no longer matches its own commit is a worse defect than this one,
//! so the change is not landed until that interaction is understood.

mod common;

use common::*;
use stark_core::command::DocCommand;
use stark_core::document::{BrushDynamics, BrushParams};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.10, 0.10, 1.0];
const BLUE: [f32; 4] = [0.10, 0.20, 0.85, 1.0];

/// A very soft, wide tip — the shape whose faint fringe is the whole point.
fn soft(color: [f32; 4], radius: f32) -> BrushParams {
    BrushParams {
        color,
        radius,
        hardness: 0.0,
        drain: 0.0,
        dynamics: BrushDynamics {
            add: 0.6,
            ..Default::default()
        },
        ..Default::default()
    }
}

const UNDER: &[Vec2] = &[Vec2::new(-120.0, 0.0), Vec2::new(120.0, 0.0)];
const OVER: &[Vec2] = &[Vec2::new(-20.0, 0.0), Vec2::new(20.0, 0.0)];

fn max_diff(a: [u8; 4], b: [u8; 4]) -> u32 {
    (0..4)
        .map(|i| (a[i] as i32 - b[i] as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// Where the upper layer is **invisible on bare canvas**, it must leave the layer
/// below it untouched.
///
/// Three renders answer it without having to know anything about brush internals:
/// the canvas alone (what "invisible" looks like), the soft blue stroke alone (where
/// it is invisible), and the two layers together against the red stroke alone (what
/// it did). Every texel where the second is indistinguishable from the first is a
/// texel where the fourth must be indistinguishable from the third.
///
/// Currently reports a shift of ~30 levels across the fringe.
#[test]
#[ignore = "known defect: pass A composites layers by per-unit opacity, not coverage"]
fn an_invisible_layer_does_not_repaint_the_one_below() {
    let Some(mut bare) = engine_or_skip() else {
        return;
    };
    let bare = bare.render_to_image();

    let mut top_alone = engine_or_skip().expect("engine");
    stroke_with(&mut top_alone, soft(BLUE, 60.0), OVER);
    let top_alone = top_alone.render_to_image();

    let mut under = engine_or_skip().expect("engine");
    stroke_with(&mut under, soft(RED, 120.0), UNDER);
    let under_alone = under.render_to_image();
    under.process(DocCommand::AddLayer { above: None });
    stroke_with(&mut under, soft(BLUE, 60.0), OVER);
    let stacked = under.render_to_image();

    let mut checked = 0u32;
    let mut worst = 0u32;
    for y in 0..bare.height {
        for x in 0..bare.width {
            // "Invisible": the blue stroke on its own is the untouched canvas here.
            if max_diff(top_alone.pixel(x, y), bare.pixel(x, y)) > 1 {
                continue;
            }
            checked += 1;
            worst = worst.max(max_diff(stacked.pixel(x, y), under_alone.pixel(x, y)));
        }
    }
    // Sanity: the band actually exists, so a bug that made the stroke vanish
    // entirely cannot pass this by having nothing to check.
    assert!(
        checked > 2_000,
        "expected a real invisible fringe to test, found {checked} texels"
    );
    assert!(
        worst <= 2,
        "a layer invisible over bare canvas moved the paint beneath it by {worst}"
    );
}
