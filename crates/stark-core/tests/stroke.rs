//! Stroke engine tests: the step-2 MVP (command/action split, copy-on-write tiles,
//! and history undo/redo — DESIGN.md §13 build order, step 2).

mod common;

use common::*;
use stark_core::Engine;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, Tool};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A brush that runs the **sequential stamp loop** rather than the swept fast path:
/// it lifts paint off the canvas onto the tool and lays it back down, so what it
/// draws at any point depends on everywhere it has already been (DESIGN.md §6.2).
///
/// That carried state is the whole reason this path is worth testing separately —
/// the swept deposit composes freely, so cutting its path anywhere is trivially
/// safe, while the loop is only cuttable because `gpu::stroke::ToolState` remembers
/// the reservoir and the pickup cadence across the cut.
fn smear_brush(color: [f32; 4], radius: f32) -> BrushParams {
    let mut b = brush(color, radius);
    b.dynamics.lift = 0.6;
    b.dynamics.deposit = 0.5;
    b.dynamics.add = 0.5;
    b
}

fn paint_stroke(engine: &mut Engine) {
    paint(
        engine,
        RED,
        40.0,
        &[
            Vec2::new(-30.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
        ],
    );
}

// Lit paint is never a pure primary, so assert channel *dominance* rather than
// near-saturation (the image-based-lighting media pass legitimately shades and
// desaturates color). The margin must exceed the warm studio tint, not just noise:
// even the neutral near-white PAPER renders red-dominant by ~33 levels under the
// studio HDR, while actual red paint dominates by ~210 (and blue BG by ~180) — so 60
// cleanly separates "lit substrate" from "paint". Tests below self-check this.
fn is_red(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 60 && c[0] as i32 > c[2] as i32 + 60
}
fn is_blue(c: [u8; 4]) -> bool {
    c[2] as i32 > c[0] as i32 + 60 && c[2] as i32 > c[1] as i32 + 60
}
fn center(img: &stark_core::RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

// Every test that reads the live preview's *colour* — either directly, or by holding
// it against what commits — is gated off under `debug-unfrozen`, which repaints the
// live tail magenta by design (see this crate's `Cargo.toml`). The tint has its own
// test, gated the other way.

#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn live_preview_shows_stroke_before_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Build an in-flight stroke without ending it.
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-30.0, 0.0)),
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 0.0)),
    });

    assert!(engine.observe().is_stroking);
    let preview = engine.render_to_image(BG);
    assert!(is_red(center(&preview)), "preview should show the stroke");
}

/// The live preview draws a long stroke in two pieces — a frozen head kept from
/// earlier moves, plus the live tail over it — while the commit draws the whole
/// thing in one pass (DESIGN.md §6.2, `engine::FrozenHead`). Those must agree, or
/// the stroke would visibly change at the moment the pointer is released.
///
/// This is the property that makes cutting the path safe at all: the swept deposit
/// is a definite integral per segment and composes by summing optical depth, so
/// where the path is cut cannot matter. A stroke long enough to actually freeze
/// several times is the point — a short one never exercises the seam.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn a_split_live_preview_matches_the_single_pass_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 12.0)));
    let path: Vec<Vec2> = (0..120)
        .map(|i| {
            let t = i as f32 * 0.05;
            Vec2::new(t * 22.0 - 60.0, (t * 1.1).sin() * 30.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }

    let preview = engine.render_to_image(BG);
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image(BG);

    // Both paths run the same shaders over the same segments; only the number of
    // compositing passes differs, so they agree to accumulated float rounding — a
    // handful of texels at the seam land a level or two apart. What would show if
    // the split were actually wrong is a *visible* discontinuity, so bound the
    // magnitude at zero rather than the count: no texel may be off by 8 levels,
    // however many are off by 2.
    assert_eq!(
        frac_exceeding(&preview, &committed, 8),
        0.0,
        "split preview visibly differs from the single-pass commit ({:.4}% of px over tol 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}

/// The unfrozen-tail tint is a *view* setting: it must change the live preview and
/// nothing else, so a stroke drawn under `debug-unfrozen` still commits in its real
/// colour. The only test here that wants the feature on.
#[cfg(feature = "debug-unfrozen")]
#[test]
fn tinting_the_live_tail_does_not_change_what_commits() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 12.0)));
    let path: Vec<Vec2> = (0..60)
        .map(|i| Vec2::new(i as f32 * 2.0 - 60.0, (i as f32 * 0.1).sin() * 20.0))
        .collect();
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }

    let preview = engine.render_to_image(BG);
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image(BG);
    // Mid-stroke the tail is magenta, so the preview differs from what lands — which
    // also rules out the commit having simply kept the preview's pixels.
    assert!(
        !images_match(&preview, &committed, 8),
        "tint had no visible effect on the preview"
    );
    // And what landed is the stroke's own colour: a tint that reached the stroke
    // *record* would repaint the whole stroke magenta in the single commit pass.
    assert!(
        is_red(center(&committed)),
        "the tint leaked into the committed stroke"
    );
}

/// The same seam check as above, but on a stroke captured from the app that
/// visibly misfits — slow and dense (305 reports over 334px), which is nothing like
/// the synthetic input the other cases use.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn a_real_captured_stroke_previews_as_it_commits() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let path: Vec<Vec2> = stark_testdata::HAIRPIN_STROKE
        .iter()
        .map(|&[x, y]| Vec2::new(x, y))
        .collect();
    engine.process(ViewCommand::SetBrush(brush(RED, 16.0)));
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }
    let preview = engine.render_to_image(BG);
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image(BG);
    assert_eq!(
        frac_exceeding(&preview, &committed, 8),
        0.0,
        "split preview visibly differs from the single-pass commit ({:.4}% of px over tol 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}

/// At **every point during** a stroke, the incrementally-composited preview must
/// equal the one a from-scratch render would give.
///
/// The other seam test only checks the final frame, which is the one moment the
/// incremental path is most likely to be right. This walks a real captured stroke
/// and compares at many points along it, because a frozen head that has gone wrong
/// stays wrong — it is never redrawn — so a fault introduced at sample 200 would be
/// invisible in a check at the end if later spans happen to cover it.
///
/// Re-setting the brush doubles as the control: every command but `StrokeTo` drops
/// the frozen head, so the very next refresh renders the whole stroke in one pass.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_incremental_preview_matches_a_fresh_one_throughout() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let path: Vec<Vec2> = stark_testdata::LOOP_STROKE
        .iter()
        .map(|&[x, y]| Vec2::new(x, y))
        .collect();
    engine.process(ViewCommand::SetBrush(brush(RED, 14.0)));
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for (i, &p) in it.enumerate() {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
        if i % 40 != 0 {
            continue;
        }
        let incremental = engine.render_to_image(BG);
        // The brush it is already using, so nothing about the stroke changes — but
        // the head is dropped, so this repaint is a single whole-stroke pass.
        engine.process(ViewCommand::SetBrush(brush(RED, 14.0)));
        let fresh = engine.render_to_image(BG);
        assert_eq!(
            frac_exceeding(&incremental, &fresh, 8),
            0.0,
            "at sample {i}: incremental preview differs from a fresh render \
             ({:.4}% of px over tol 2)",
            frac_exceeding(&incremental, &fresh, 2) * 100.0,
        );
    }
}

/// Lay down something for a smear brush to pick up: two crossing bands of colour,
/// committed, so the canvas under the test stroke is not blank. A `lift` brush over
/// bare canvas carries nothing and would exercise none of the reservoir.
#[cfg(not(feature = "debug-unfrozen"))] // both its callers are seam tests
fn undercoat(engine: &mut Engine) {
    paint(
        engine,
        [0.1, 0.9, 0.2, 1.0],
        26.0,
        &[Vec2::new(-110.0, -40.0), Vec2::new(110.0, -10.0)],
    );
    paint(
        engine,
        [0.95, 0.85, 0.1, 1.0],
        26.0,
        &[Vec2::new(-90.0, 50.0), Vec2::new(100.0, 20.0)],
    );
}

/// The seam check for the **stamp loop**: a `lift`/`deposit` stroke drawn as a frozen
/// head plus a live tail must land the same pixels as the single pass a commit runs.
///
/// This is a much stronger claim than the swept case. The swept deposit is a definite
/// integral per segment that composes by summing optical depth, so where the path is
/// cut genuinely cannot matter. The loop is *sequential*: each segment reads the
/// canvas the previous one left and the tool the previous one loaded, so cutting it is
/// only sound because [`ToolState`](stark_core::gpu::stroke::ToolState) carries the
/// reservoir — and the travel since the last pickup — across the cut. Get either wrong
/// and the tail draws with a tool that has forgotten where it has been: a visible step
/// in colour at the freeze boundary, trailing back across the stroke.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn a_smear_stroke_previews_as_it_commits() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    undercoat(&mut engine);

    engine.process(ViewCommand::SetBrush(smear_brush(RED, 15.0)));
    let path: Vec<Vec2> = (0..140)
        .map(|i| {
            let t = i as f32 * 0.045;
            Vec2::new(t * 20.0 - 62.0, (t * 1.3).sin() * 34.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }

    let preview = engine.render_to_image(BG);
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image(BG);
    assert_eq!(
        frac_exceeding(&preview, &committed, 8),
        0.0,
        "split smear preview visibly differs from the single-pass commit \
         ({:.4}% of px over tol 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}

/// [`the_incremental_preview_matches_a_fresh_one_throughout`] for the stamp loop.
///
/// Checking only the last frame is far too weak here: a frozen head is never redrawn,
/// so a reservoir handed over wrong at sample 60 stays wrong on the canvas forever,
/// and later spans painting over the top could hide it from a check at the end. What
/// this walks is every handover.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_incremental_smear_preview_matches_a_fresh_one_throughout() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    undercoat(&mut engine);

    engine.process(ViewCommand::SetBrush(smear_brush(RED, 15.0)));
    let path: Vec<Vec2> = stark_testdata::C_STROKE
        .iter()
        .map(|&[x, y]| Vec2::new(x - 160.0, y - 160.0))
        .collect();
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for (i, &p) in it.enumerate() {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
        if i % 30 != 0 {
            continue;
        }
        let incremental = engine.render_to_image(BG);
        // The brush it is already using, so nothing about the stroke changes — but
        // the head is dropped, so this repaint runs the whole loop in one pass.
        engine.process(ViewCommand::SetBrush(smear_brush(RED, 15.0)));
        let fresh = engine.render_to_image(BG);
        assert_eq!(
            frac_exceeding(&incremental, &fresh, 8),
            0.0,
            "at sample {i}: incremental smear preview differs from a fresh render \
             ({:.4}% of px over tol 2)",
            frac_exceeding(&incremental, &fresh, 2) * 100.0,
        );
    }
}

/// An undercoat for a stroke measured in thousands of pixels: an ordinary brush with
/// its one-way load switched off, since the default `drain` runs a stroke dry a few
/// hundred px in and would leave the far end of the band bare.
fn long_band(engine: &mut Engine, points: &[Vec2]) {
    let mut b = brush([0.1, 0.9, 0.2, 1.0], 30.0);
    b.drain = 0.0;
    stroke_with(engine, b, points);
}

/// A stroke far wider than one stamp-loop region still **manipulates paint**.
///
/// This is the regression the piecewise path exists for. The loop works on a 1:1 copy
/// of the canvas under the stroke, and a stroke whose bounding box outgrew
/// `MAX_REGION_DIM` used to degrade to the plain swept deposit — which is not a
/// coarser version of the same brush but a different one: the swept path only ever
/// *adds* paint, so a brush whose whole purpose is to lift it silently stopped doing
/// the one thing it was for. Long strokes and fat tips are exactly where a smear
/// brush earns its keep, so the failure landed on the strokes that wanted it most.
/// Now the stroke is drawn in as many region-sized pieces as it takes.
///
/// The brush here is a pure scrape — it lifts everything and lays nothing back, and
/// its own colour is fully transparent — so the two paths are unmistakable: the loop
/// takes the undercoat away, while the swept deposit has nothing to lay and leaves it
/// exactly where it was. The stroke runs well past the window on both sides; only its
/// bounding box has to be oversized, not the part under test.
#[test]
fn a_stroke_too_wide_for_one_region_still_moves_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let across = [
        Vec2::new(-1200.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1200.0, 0.0),
    ];
    long_band(&mut engine, &across);
    assert!(
        !is_blue(center(&engine.render_to_image(BG))),
        "the undercoat did not land, so there is nothing to scrape"
    );

    let mut scrape = brush([0.0, 0.0, 0.0, 0.0], 22.0);
    scrape.drain = 0.0;
    scrape.dynamics.add = 0.0;
    scrape.dynamics.lift = 1.0;
    stroke_with(&mut engine, scrape, &across);
    assert!(
        is_blue(center(&engine.render_to_image(BG))),
        "a stroke too wide for one region left the paint under it untouched"
    );
}

/// [`a_smear_stroke_previews_as_it_commits`] for a stroke that no longer fits one
/// region — so the commit runs the loop in several pieces while the preview's ranges
/// each run it in one.
///
/// The pieces are cut where the *region* fills up, which has nothing to do with where
/// the fitter freezes, so the two renders cut the same stroke in different places.
/// That is what makes this worth running: a piece boundary that dropped the reservoir
/// — or restarted the pickup cadence — would show as a step in colour under the
/// commit that the preview never draws.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn an_oversized_smear_stroke_previews_as_it_commits() {
    // Wide enough to see a piece boundary: the first piece fills its region around
    // `x = -2000 + MAX_REGION_DIM`, which lands just right of centre.
    let Some(mut engine) = engine_or_skip_sized(stark_core::geom::Extent2 {
        width: 1280,
        height: 256,
    }) else {
        return;
    };
    let path: Vec<Vec2> = (0..220)
        .map(|i| {
            let t = i as f32 / 219.0;
            Vec2::new(t * 2560.0 - 2000.0, (t * 26.0).sin() * 40.0)
        })
        .collect();
    long_band(&mut engine, &path);

    engine.process(ViewCommand::SetBrush(smear_brush(RED, 15.0)));
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }

    let preview = engine.render_to_image(BG);
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image(BG);
    assert_eq!(
        frac_exceeding(&preview, &committed, 8),
        0.0,
        "an oversized smear stroke commits differently than it previewed \
         ({:.4}% of px over tol 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}

#[test]
fn stroke_commit_undo_redo() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_stroke(&mut engine);
    assert!(!engine.observe().is_stroking);
    assert!(engine.observe().can_undo);

    let committed = engine.render_to_image(BG);
    assert!(is_red(center(&committed)), "committed center should be red");
    assert!(
        is_blue(committed.pixel(10, 10)),
        "untouched corner should be background blue"
    );

    engine.process(DocCommand::Undo);
    assert!(engine.observe().can_redo);
    assert!(
        is_blue(center(&engine.render_to_image(BG))),
        "after undo, center should be background"
    );

    engine.process(DocCommand::Redo);
    assert!(
        is_red(center(&engine.render_to_image(BG))),
        "after redo, center should be red again"
    );
}

#[test]
fn stroke_spans_multiple_tiles_via_cow() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_stroke(&mut engine);

    // A radius-40 stroke straddling the canvas origin touches all four tiles
    // around (0,0); copy-on-write should have populated more than one.
    let populated: usize = engine
        .document()
        .layers
        .iter()
        .map(|l| l.tiles.size())
        .sum();
    assert!(
        populated >= 2,
        "stroke across the origin should populate multiple tiles, got {populated}"
    );
}

/// Per-move cost must not grow with the length of the stroke — which is the whole
/// point of the frozen head (DESIGN.md §6.2, `engine::FrozenHead`).
///
/// Run for **both** render paths. The stamp loop is the case that matters: before it
/// carried its reservoir across the freeze boundary it re-rendered the whole stroke
/// every move, and its cost climbed from 0.8 ms to 13 ms over 1200 samples, with the
/// composite behind it going 6.7 ms → 131 ms because the readback waits on all that
/// queued work. A brush people actually paint with is a smear brush, so that was the
/// responsiveness users felt.
///
/// The path is a travelling sine, wide and tall enough to keep growing the tile set
/// the compositor walks. It replaced a spiral, which was the wrong shape to measure
/// with: a spiral's spans get steadily longer in arc as the radius opens out, so the
/// live tail — a fixed number of *control points* — covers more and more distance, and
/// on the stamp loop (whose cost is per flattened segment) that read as growth even
/// though the head was never being repainted. Under a sine the tail's geometry is
/// statistically the same at both ends, so what is left is what the test means to
/// measure. It still catches the regression it exists for: a stroke re-rendered whole
/// would climb from ~150 segments a move to several thousand.
fn measure_per_move_growth(b: BrushParams) -> (f64, f64) {
    let size = stark_core::geom::Extent2 {
        width: 1280,
        height: 800,
    };
    let Ok(mut engine) = pollster::block_on(stark_core::engine::headless_engine(
        wgpu::TextureFormat::Rgba8UnormSrgb,
        size,
    )) else {
        eprintln!("skipping GPU test");
        return (1.0, 1.0);
    };
    engine.process(ViewCommand::SetBrush(b));
    let n = 900usize;
    let path: Vec<Vec2> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            Vec2::new(t * 1180.0 - 590.0, (t * 34.0).sin() * 300.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    let (mut early, mut late) = (0.0f64, 0.0f64);
    let (mut ne, mut nl) = (0u32, 0u32);
    for (i, &p) in it.enumerate() {
        let at = std::time::Instant::now();
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
        // The readback is what makes this a measurement of the *work*, not just of
        // encoding it: `process` only queues, so without waiting here a path that
        // queued ten times the GPU work would look identical.
        let _ = engine.render_to_image(BG);
        let ms = at.elapsed().as_secs_f64() * 1000.0;
        // Skip the middle: the first stretch also pays for warm-up.
        if i < n / 4 {
            early += ms;
            ne += 1;
        } else if i > n * 3 / 4 {
            late += ms;
            nl += 1;
        }
    }
    (early / ne as f64, late / nl as f64)
}

#[test]
fn per_move_cost_does_not_grow_with_stroke_length() {
    for (what, b) in [
        ("swept", brush(RED, 14.0)),
        ("stamp loop", smear_brush(RED, 14.0)),
    ] {
        let (early, late) = measure_per_move_growth(b);
        // Generous, because this is a wall-clock measurement on a shared machine. What
        // it has to catch is *growth*, and the failure it guards against is severalfold.
        assert!(
            late < early * 2.0,
            "{what}: per-move cost grew with stroke length: \
             {early:.2} ms early vs {late:.2} ms late"
        );
    }
}
