//! How layers cover one another (DESIGN.md §6.3, pass A).
//!
//! **The claim.** A layer affects what is beneath it only as much as it is visible
//! at all. It sounds too obvious to test, and it used to be false: pass A weighed a
//! layer's "over" by its per-unit **opacity** alone, so a film with opacity 1 and no
//! thickness — which the media pass draws as nothing over bare canvas — replaced the
//! colour outright over another layer's paint. Every soft brush deposits exactly that
//! state across its fringe (`stamp_oklab.wesl` saturates opacity as `1 − exp(−op·τ)`
//! while height stays linear in `τ`), so the symptom was a ghost of the brush's whole
//! footprint painted over the layer below.
//!
//! **The law now.** Pass A weighs each layer by its own visible alpha — the slab law
//! `1 − exp(−K·opacity·height)` that `paint_common.wesl` already uses to stack
//! parcels *within* a layer — and the media pass reads the accumulated coverage
//! instead of re-deriving it (`composite.wesl`, `media_common.wesl`). For a single
//! layer that is algebraically the identity.
//!
//! **The smear interaction that stalled the first attempt**, for the record: the
//! dynamics loop composites base tiles into its working region with the *same*
//! `composite` shader, and that region must keep the tile representation itself —
//! per-unit opacity in alpha — because the pickup reads it and the slice writes it
//! back to persistent tiles. Applying the slab law there stored *coverage* as
//! opacity, corrupting smeared paint differently on each side of a piece or freeze
//! cut, which is why the preview stopped matching the commit. The screen path and
//! the region path are now separate entry points (`fs_main` / `fs_raw`).

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
/// Before the coverage weighting this reported a shift of ~30 levels across the
/// fringe.
#[test]
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
