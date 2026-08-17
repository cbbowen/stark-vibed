//! Stroke engine tests: the step-2 MVP (command/action split, copy-on-write tiles,
//! and history undo/redo — §13 build order, step 2).
//!
//! What is left here is what *reads the picture*: tests that measure a stroke's width
//! off the render, or ask whether there is paint at a point. The claims of the form
//! "these two renders must agree" — preview against commit, incremental against fresh,
//! oversized against whole — moved to [`corpus.rs`](corpus.rs), which asks them of
//! every stroke in the corpus instead of of the one each was written for.
//!
//! That move was worth making on its own account. Five of the tests that lived here
//! passed a recorded stroke straight to the engine, and the recordings carry the
//! canvas coordinates the pen was actually at — `LOOP_STROKE` sits around `y = 980`,
//! `C_STROKE` around `x = 1310`, against a viewport showing the 256 px about the
//! origin. They had been comparing **blank paper against blank paper** for as long as
//! they had existed, and every one of them passed. The corpus centres its strokes from
//! the data and `every_case_leaves_a_mark` makes an empty case a failure.

mod common;

use common::*;
use stark_engine::Engine;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::document::{BrushParams, Tool};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

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
// near-saturation (the media pass legitimately shades and desaturates color). Under
// the reference light (§6.3) the substrate is achromatic, so the neutral PAPER carries
// no dominance at all while red paint dominates by ~210 and blue BG by ~180 — 60 is
// conservative rather than merely sufficient. The margin is wide because the
// separation it states is the one that matters, not because the number is tight; a
// warm tint alone can push a neutral substrate to ~33 levels of false dominance.
// Tests below self-check this.
fn is_red(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 60 && c[0] as i32 > c[2] as i32 + 60
}
fn is_blue(c: [u8; 4]) -> bool {
    c[2] as i32 > c[0] as i32 + 60 && c[2] as i32 > c[1] as i32 + 60
}
// Every test that reads the live preview's *color* — either directly, or by holding
// it against what commits — is gated off under `debug-unfrozen`, which repaints the
// live tail magenta by design (see this crate's `Cargo.toml`). The tint has its own
// test, gated the other way.

#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn live_preview_shows_stroke_before_commit() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Build an in-flight stroke without ending it.
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-30.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 0.0)),
    });

    assert!(engine.observe().is_stroking);
    let preview = engine.render_to_image();
    assert!(is_red(center(&preview)), "preview should show the stroke");
}

/// The unfrozen-tail tint is a *view* setting: it must change the live preview and
/// nothing else, so a stroke drawn under `debug-unfrozen` still commits in its real
/// color. The only test here that wants the feature on.
#[cfg(feature = "debug-unfrozen")]
#[test]
fn tinting_the_live_tail_does_not_change_what_commits() {
    let Some(mut engine) = engine_or_skip_blue() else {
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
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }

    let preview = engine.render_to_image();
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();
    // Mid-stroke the tail is magenta, so the preview differs from what lands — which
    // also rules out the commit having simply kept the preview's pixels.
    assert!(
        !images_match(&preview, &committed, 8),
        "tint had no visible effect on the preview"
    );
    // And what landed is the stroke's own color: a tint that reached the stroke
    // *record* would repaint the whole stroke magenta in the single commit pass.
    assert!(
        is_red(center(&committed)),
        "the tint leaked into the committed stroke"
    );
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
/// This is what the piecewise path exists for. The loop works on a 1:1 copy of the
/// canvas under the stroke, so a stroke whose bounding box outgrows `MAX_REGION_DIM`
/// has to be cut into as many region-sized pieces as it takes. Degrading to the plain
/// swept deposit instead is not a coarser version of the same brush but a different
/// one: the swept path only ever *adds* paint, so a brush whose whole purpose is to
/// lift it silently stops doing the one thing it is for — and long strokes with fat
/// tips are exactly where a smear brush earns its keep.
///
/// The brush here is a pure scrape — it lifts everything and lays nothing back, and
/// its own color is fully transparent — so the two paths are unmistakable: the loop
/// takes the undercoat away, while the swept deposit has nothing to lay and leaves it
/// exactly where it was. The stroke runs well past the window on both sides; only its
/// bounding box has to be oversized, not the part under test.
#[test]
fn a_stroke_too_wide_for_one_region_still_moves_paint() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let across = [
        Vec2::new(-1200.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1200.0, 0.0),
    ];
    long_band(&mut engine, &across);
    assert!(
        !is_blue(center(&engine.render_to_image())),
        "the undercoat did not land, so there is nothing to scrape"
    );

    let mut scrape = brush([0.0, 0.0, 0.0, 0.0], 22.0);
    scrape.drain = 0.0;
    scrape.dynamics.add = 0.0;
    scrape.dynamics.lift = 1.0;
    stroke_with(&mut engine, scrape, &across);
    assert!(
        is_blue(center(&engine.render_to_image())),
        "a stroke too wide for one region left the paint under it untouched"
    );
}

// --- tapered ends (§6.2) ---------------------------------------

/// A hard, opaque inking brush with the given taper lengths (in radii). Hard on
/// purpose: the tests below *measure the stroke's width* off the rendered image, and
/// a soft tip's falloff would put the edge wherever the dominance threshold happens
/// to cut rather than where the brush is. `drain` off, so nothing but the taper
/// varies along the stroke.
fn inking_brush(start: f32, end: f32) -> BrushParams {
    let mut b = brush(RED, 16.0);
    b.shape = stark_engine::document::BrushShape::Round { hardness: 0.9 };
    b.drain = 0.0;
    b.dynamics.add = 1.0;
    b.start_taper_length = start;
    b.end_taper_length = end;
    b
}

/// A dense straight run across the canvas, the shape a taper is easiest to read on.
fn straight_run() -> Vec<Vec2> {
    (0..=40)
        .map(|i| Vec2::new(i as f32 / 40.0 * 200.0 - 100.0, 0.0))
        .collect()
}

/// The stroke's rendered width where it crosses screen column `x`: how many rows
/// there read as paint.
fn painted_height(img: &stark_engine::RgbaImage, x: u32) -> u32 {
    (0..img.height).filter(|&y| is_red(img.pixel(x, y))).count() as u32
}

/// The taper's whole point: the stroke leaves and enters as a point while its body
/// stays full width. Held against the *same* stroke drawn untapered, so the claim is
/// about the taper rather than about where a soft edge happens to read as paint.
///
/// Canvas x maps to screen x + 128 at the default 1:1 view, so the run from -100 to
/// 100 spans columns 28..228 and its two 80px tapers (5 radii of a 16px tip) cover
/// 28..108 and 148..228.
#[test]
fn a_tapered_brush_draws_a_stroke_pointed_at_both_ends() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let run = straight_run();

    stroke_with(&mut engine, inking_brush(0.0, 0.0), &run);
    let plain = engine.render_to_image();
    engine.process(DocCommand::Undo);
    stroke_with(&mut engine, inking_brush(5.0, 5.0), &run);
    let tapered = engine.render_to_image();

    // Mid-stroke both are clear of either taper, so the taper must cost nothing there
    // — a taper that thinned the whole stroke would be a width knob, not a taper.
    let (mid_plain, mid_taper) = (painted_height(&plain, 128), painted_height(&tapered, 128));
    assert!(
        mid_plain > 20,
        "the control stroke did not land ({mid_plain}px)"
    );
    assert!(
        mid_taper >= mid_plain,
        "the taper thinned the body of the stroke: {mid_taper}px vs {mid_plain}px"
    );

    // A quarter of the way into each taper the stroke must be markedly narrower...
    for (what, x) in [("start", 48u32), ("end", 208)] {
        let (p, t) = (painted_height(&plain, x), painted_height(&tapered, x));
        assert!(p > 20, "the control stroke is not full width at the {what}");
        assert!(
            (t as f32) < 0.6 * p as f32,
            "the {what} taper barely narrows the stroke: {t}px of {p}px"
        );
    }
    // ...and at the very tips, all but gone.
    for (what, x) in [("start", 30u32), ("end", 226)] {
        let t = painted_height(&tapered, x);
        assert!(t <= 4, "the {what} of the stroke is not a point: {t}px");
    }
    // The taper is a *shape*, not an erasure: the stroke still reaches its extremes.
    assert!(
        (28..40).any(|x| painted_height(&tapered, x) > 0),
        "the tapered stroke does not start where it was drawn"
    );
    assert!(
        (216..228).any(|x| painted_height(&tapered, x) > 0),
        "the tapered stroke does not reach its end"
    );
}

/// The taper widens smoothly — no step where it meets the stroke's full-width body,
/// which is the artifact a profile with a slope left at the join would leave, and the
/// one thing that would give the whole effect away.
#[test]
fn the_taper_widens_without_a_step() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // One long taper over the whole run, so every column is inside it and the width
    // profile can be walked end to end.
    stroke_with(&mut engine, inking_brush(12.5, 0.0), &straight_run());
    let img = engine.render_to_image();

    // Stop a radius short of where the stroke ends (canvas x = 100, column 228). The
    // last 16 px are the round **cap**, which must narrow — that is what a round tip
    // is — and narrowing there says nothing about whether the taper has a step in it.
    // Walking further — to 226, say — asks the cap's own curve to be monotone: under
    // the reference light the paint sits against an achromatic ground and the cap
    // reads to its full extent, so its edge stays above the dominance threshold all
    // the way out.
    let widths: Vec<u32> = (30..212).map(|x| painted_height(&img, x)).collect();
    // Monotone up to the body (within the ±1 a rasterized edge rounds by), and never
    // jumping more than a couple of px between adjacent columns.
    for (i, w) in widths.windows(2).enumerate() {
        assert!(
            w[1] + 1 >= w[0],
            "the taper narrows again at column {}: {} then {}",
            30 + i,
            w[0],
            w[1]
        );
        assert!(
            w[1].abs_diff(w[0]) <= 3,
            "the taper steps {} px at column {} — a visible crease",
            w[1].abs_diff(w[0]),
            30 + i
        );
    }
}

/// A tapered brush still **dots**: a click has no length for a taper to run along, so
/// tapping with an inking brush has to leave a full-size mark rather than the
/// invisible speck a taper read literally would give.
#[test]
fn a_tapered_brush_still_dots() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(inking_brush(6.0, 6.0)));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::ZERO),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert!(is_red(center(&img)), "a tapered brush left no dot at all");
    assert!(
        painted_height(&img, 128) > 20,
        "the dot is a speck, not the brush's own size ({}px)",
        painted_height(&img, 128)
    );
}

/// The painted span of screen row/column `at`, as `(first, last)`.
fn painted_span(img: &stark_engine::RgbaImage, along_x: bool, at: u32) -> (u32, u32) {
    let n = if along_x { img.width } else { img.height };
    let px = |i| {
        if along_x {
            img.pixel(i, at)
        } else {
            img.pixel(at, i)
        }
    };
    let painted: Vec<u32> = (0..n).filter(|&i| is_red(px(i))).collect();
    (
        *painted.first().expect("some paint"),
        *painted.last().expect("some paint"),
    )
}

/// **A click leaves a dab, centred where it was pressed** — and so does the same
/// press nudged by a pixel, which is the half of it that is easy to get wrong.
///
/// A swept deposit is a definite integral over travel, so a press that has not moved
/// integrates over nothing. The minimum travel that fixes it (`segments::DAB_TRAVEL`)
/// must be centred on the point rather than swept *from* it: swept from the point, in
/// the `+x` a click's absent tangent falls back to, it is a whole tip's width of
/// travel all on one side, which reads as a short dash pointing right rather than as a
/// dot. And applying only to a stroke with no travel at all, one pixel of movement
/// replaces it with a twentieth of a dab — the dot vanishes the moment the hand moves
/// and comes back a tip's width later.
///
/// So both are asserted here against the *same* threshold: a click is round and
/// centred, and one pixel of movement changes it hardly at all.
#[test]
fn a_click_dabs_where_it_was_pressed_and_a_nudge_keeps_it() {
    let dab = |nudge: f32| {
        let mut engine = engine_or_skip_blue()?;
        engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
        engine.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: InputSample::at(Vec2::ZERO),
            tolerance: DEFAULT_TOLERANCE,
            rope: 0.0,
        });
        if nudge > 0.0 {
            engine.process(GestureCommand::To {
                sample: InputSample::at(Vec2::new(nudge, 0.0)),
            });
        }
        engine.process(GestureCommand::End);
        Some(engine.render_to_image())
    };
    let Some(click) = dab(0.0) else {
        return;
    };
    let Some(nudged) = dab(1.0) else {
        return;
    };

    assert!(is_red(center(&click)), "a click left no mark at all");
    assert!(
        is_red(center(&nudged)),
        "a one-pixel drag left no mark at all"
    );

    // Canvas (0,0) is screen (128,128) at the default 1:1 view.
    for (what, img) in [("a click", &click), ("a one-pixel drag", &nudged)] {
        let (x0, x1) = painted_span(img, true, 128);
        let (y0, y1) = painted_span(img, false, 128);
        let centre = (x0 + x1) as i32 / 2;
        assert!(
            (centre - 128).abs() <= 3,
            "{what}'s mark spans {x0}..{x1}, centred at {centre} rather than on the \
             128 that was pressed"
        );
        let (w, h) = ((x1 - x0) as f32, (y1 - y0) as f32);
        assert!(w / h < 1.25, "{what} left a {w}x{h} dash rather than a dab");
    }
}

#[test]
fn stroke_commit_undo_redo() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint_stroke(&mut engine);
    assert!(!engine.observe().is_stroking);
    assert!(engine.observe().can_undo);

    let committed = engine.render_to_image();
    assert!(is_red(center(&committed)), "committed center should be red");
    assert!(
        is_blue(committed.pixel(10, 10)),
        "untouched corner should be background blue"
    );

    engine.process(DocCommand::Undo);
    assert!(engine.observe().can_redo);
    assert!(
        is_blue(center(&engine.render_to_image())),
        "after undo, center should be background"
    );

    engine.process(DocCommand::Redo);
    assert!(
        is_red(center(&engine.render_to_image())),
        "after redo, center should be red again"
    );
}

#[test]
fn stroke_spans_multiple_tiles_via_cow() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint_stroke(&mut engine);

    // A radius-40 stroke straddling the canvas origin touches all four tiles
    // around (0,0); copy-on-write should have populated more than one.
    let populated: usize = engine
        .document()
        .root()
        .iter()
        .map(|l| l.tiles().map_or(0, |t| t.size()))
        .sum();
    assert!(
        populated >= 2,
        "stroke across the origin should populate multiple tiles, got {populated}"
    );
}

/// Per-move cost must not grow with the length of the stroke — which is the whole
/// point of the frozen head (§6.2, `engine::FrozenHead`).
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
    let size = stark_model::geom::Extent2 {
        width: 1280,
        height: 800,
    };
    let Ok(mut engine) = pollster::block_on(stark_engine::engine::headless_engine(
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
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
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
        let _ = engine.render_to_image();
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
        ("stamp loop", common::corpus::smear_brush(14.0)),
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
