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

#![expect(
    clippy::disallowed_types,
    reason = "a native-only test binary, timing GPU work against a real adapter"
)]

mod common;

use common::*;
use stark_engine::Engine;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::document::BrushParams;
use stark_model::geom::Vec2;

const RED: [f32; 3] = [1.0, 0.0, 0.0];

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
    leads(rgb(c), Lead::Red, MARGIN_LIT)
}
fn is_blue(c: [u8; 4]) -> bool {
    leads(rgb(c), Lead::Blue, MARGIN_LIT)
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
    engine.process(ViewCommand::set_brush(brush(RED, 40.0)));
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
    engine.process(ViewCommand::set_brush(brush(RED, 12.0)));
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
    // Mid-stroke the tail is magenta, so the preview differs from what lands. A
    // commit ordinarily *takes* the preview's tiles (`PreparedStroke`, §6.2), and
    // this is the one render it must not take — it is a render of a different
    // record — so the commit has to have drawn the stroke itself.
    assert!(
        !images_match(&preview, &committed, 8),
        "tint had no visible effect on the preview"
    );
    assert_eq!(
        engine.strokes_reused(),
        0,
        "the commit took a preview whose tail was painted in the diagnostic's color"
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
    let mut b = brush([0.1, 0.9, 0.2], 30.0);
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

    let mut scrape = brush([0.0, 0.0, 0.0], 22.0);
    scrape.drain = 0.0;
    scrape.paint_mut().expect("a paint brush").flow = 0.0;
    scrape.make_wet().dynamics.lift = 1.0;
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
    b.shape = stark_model::document::BrushShape::Round { hardness: 0.9 };
    b.drain = 0.0;
    b.paint_mut().expect("a paint brush").flow = 1.0;
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
    // the reference light the paint sits against an achromatic substrate and the cap
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

/// **The deposit is as smooth along a taper as the outline is** (§6.2,
/// `stamp_common::Sweep::span`).
///
/// The outline's continuity has its own test above; this one reads the *inside* of
/// the mark. A point's exposure to one segment is the footprint integrated over the
/// offsets the point held relative to the tip, and an offset is measured against
/// whatever tip was in force at the moment — `r_start` when the sweep opens,
/// `r_end` when it closes. Divide both ends by the segment's reference radius
/// instead and the two are right only on average: the error is a *step* at every
/// knot rather than a smooth bias, so a taper's exposure ripples at whatever cadence
/// the flattener happened to cut, worst at the point where the ramp is largest and
/// no subdivision can shrink it.
///
/// Read as ink per column rather than as width, because the artifact is in the
/// deposit and not in the edge: the brush is soft and thin enough that the mark
/// stays linear in exposure everywhere, so a ripple in `τ` is a ripple in the
/// picture rather than something saturation flattens out. The taper's own profile is
/// a cubic, so the true column ink is smooth to well under the bound below; the
/// ripple this pins ran to 5.8% of the column mean where the taper's own curvature
/// leaves 1.1%, and the bound sits between them.
#[test]
fn a_tapers_deposit_has_no_ripple_at_the_cut() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Dither off (§6.5): the column sums below are a stand-in for `τ`, and the
    // display's per-pixel half-code of deliberate noise reads as a ~5%-of-mean
    // second difference — right where the artifact this test pins used to sit.
    engine.process(stark_engine::command::ViewCommand::SetMediaParams(
        stark_engine::MediaParams {
            dither: false,
            ..Default::default()
        },
    ));
    // One taper over the whole run, so every column is inside it, and a soft thin
    // tip so the deposit never saturates. `drain` is off in `inking_brush`, so the
    // taper is the only thing varying along the stroke.
    let mut b = inking_brush(12.5, 0.0);
    b.shape = stark_model::document::BrushShape::Round { hardness: 0.2 };
    b.paint_mut().expect("a paint brush").flow = 0.08;
    stroke_with(&mut engine, b, &straight_run());
    let img = engine.render_to_image();

    // How far this column has been moved off bare canvas, summed down the rows. A
    // linear reading, so it tracks `τ` rather than an edge's crossing of a threshold;
    // the substrate is uniform, so one corner texel is what "bare" means everywhere.
    let bare = img.pixel(0, 0);
    let lean = |c: [u8; 4]| c[0] as f32 - c[2] as f32;
    let ink = |x: u32| -> f32 {
        (0..img.height)
            .map(|y| lean(img.pixel(x, y)) - lean(bare))
            .sum::<f32>()
    };
    // Column 28 is the taper's point and 228 the stroke's end; stay a radius clear of
    // both, where the profile is a plain cubic in the arc.
    let profile: Vec<f32> = (46..210).map(ink).collect();
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    assert!(mean > 0.0, "the stroke left no ink to measure");

    // The second difference: a step at a knot shows here where the taper's own
    // curvature barely registers.
    let (mut worst, mut at) = (0.0f32, 0usize);
    for (i, w) in profile.windows(3).enumerate() {
        let d2 = (w[0] - 2.0 * w[1] + w[2]).abs() / mean;
        if d2 > worst {
            worst = d2;
            at = 46 + i + 1;
        }
    }
    assert!(
        worst < 0.02,
        "the taper's ink jumps by {:.1}% of its mean at column {at} — the deposit \
         has a step in it where the flattener cut",
        worst * 100.0,
    );
}

/// **A click paints nothing, and commits nothing.** A swept deposit is a
/// definite integral over travel, and a press that has not moved integrates
/// over nothing — the tool now says so honestly instead of fabricating a
/// minimum (the retired `DAB_TRAVEL` dwell). What tells the artist *before*
/// the press is the brush cursor: the hover's mark previews exactly what a
/// press would lay (§18.1.10), which for a stationary press is nothing.
///
/// Nothing in the picture means nothing in the log, either: a record that
/// cannot deposit would spend an undo step invisibly, so the release declines
/// it — the same answer a marquee click has always been given.
#[test]
fn a_click_paints_nothing_and_commits_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Tapered as well as not: the taper compresses on short strokes rather
    // than gating them, so it must not resurrect a mark here either.
    for b in [brush(RED, 40.0), inking_brush(6.0, 6.0)] {
        engine.process(ViewCommand::set_brush(b));
        let before = engine.render_to_image();
        // The committed document as it stands — `engine_or_skip_blue` already
        // banked a `SetSubstrateColor`, so "nothing committed" is a revision that
        // did not move, never an undo stack that is empty.
        let rev = engine.observe().doc_revision;
        engine.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: InputSample::at(Vec2::ZERO),
            tolerance: DEFAULT_TOLERANCE,
            rope: 0.0,
        });
        engine.process(GestureCommand::End);
        assert!(
            images_match(&before, &engine.render_to_image(), 0),
            "a click changed the canvas"
        );
        assert_eq!(
            engine.observe().doc_revision,
            rev,
            "a click left an undo step with nothing under it"
        );
    }
}

/// A long stroke, drawn with a frame between samples the way the app draws one.
/// Long enough that the fitter freezes spans behind the pointer, so the preview is
/// a kept head plus a live tail rather than one range — the case a commit that
/// re-rendered the stroke would pay for in full.
///
/// Gated with the two tests that use it: `debug-unfrozen` tints the live tail, so
/// every preview-vs-commit test is off under it (see the module note) and these
/// helpers would be dead code in that build.
#[cfg(not(feature = "debug-unfrozen"))]
fn draw_long_stroke(engine: &mut Engine) {
    engine.process(ViewCommand::set_brush(brush(RED, 14.0)));
    let mut it = long_wave().into_iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(it.next().unwrap()),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
        let _ = engine.render_to_image();
    }
}

#[cfg(not(feature = "debug-unfrozen"))]
fn long_wave() -> Vec<Vec2> {
    (0..200)
        .map(|i| {
            let t = i as f32 / 200.0;
            Vec2::new(t * 230.0 - 115.0, (t * 9.0).sin() * 40.0)
        })
        .collect()
}

/// **The commit is the preview** (`PreparedStroke`, §6.2). Releasing the pointer
/// takes the tiles the last fold drew rather than rendering the stroke again, so the
/// hitch at pen-up costs the live tail and not the stroke's length.
///
/// Held against a fresh engine *replaying* the same samples, which is the path that
/// folds no preview and so renders the whole stroke in one range at commit: what
/// lands live is the head-plus-tail render, and the two must agree to within the
/// corpus's seam bound (`corpus.rs`) — the cut is an f16 store of the head, not a
/// different stroke. Both engines seed the stroke from the same clock, so the
/// comparison is of the render paths alone.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn a_stroke_commits_the_tiles_it_previewed() {
    let (Some(mut live), Some(mut replayed)) = (engine_or_skip_blue(), engine_or_skip_blue())
    else {
        return;
    };
    draw_long_stroke(&mut live);
    assert!(
        live.live_head_count() == 1,
        "the stroke should have a kept head before it is released"
    );
    let preview = live.render_to_image();
    live.process(GestureCommand::End);
    assert_eq!(
        live.strokes_reused(),
        1,
        "the commit rendered the stroke again instead of taking the preview's tiles"
    );
    let committed = live.render_to_image();
    assert!(
        images_match(&preview, &committed, 0),
        "what landed is not, to the bit, what was previewed"
    );

    replayed.process(ViewCommand::set_brush(brush(RED, 14.0)));
    let samples: Vec<InputSample> = long_wave().into_iter().map(InputSample::at).collect();
    replayed.replay_stroke(Tool::Brush, &samples);
    assert_eq!(
        replayed.strokes_reused(),
        0,
        "a replay folds no preview, so there is nothing for its commit to take"
    );
    let fresh = replayed.render_to_image();
    assert!(
        images_match(&committed, &fresh, 4),
        "the previewed render and a whole render of the same stroke disagree by          more than the seam bound: worst {} levels",
        diff_fraction(&committed, &fresh).1
    );
}

/// **Fast commit off is bit-for-bit** (`DEFAULT_FAST_COMMIT`, §6.2). With the
/// setting cleared the commit is offered nothing, so it renders the stroke the one
/// way a replay, a file, an undo and a collaborator all render it — and *exactly*
/// that way, which is the whole of what the setting buys and the reason the bound
/// here is zero where its neighbour's is the seam.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn a_stroke_committed_without_fast_commit_is_the_replay() {
    let (Some(mut live), Some(mut replayed)) = (engine_or_skip_blue(), engine_or_skip_blue())
    else {
        return;
    };
    live.process(ViewCommand::SetFastCommit(false));
    draw_long_stroke(&mut live);
    live.process(GestureCommand::End);
    assert_eq!(
        live.strokes_reused(),
        0,
        "the commit took the preview's tiles with fast commit switched off"
    );
    let committed = live.render_to_image();

    replayed.process(ViewCommand::set_brush(brush(RED, 14.0)));
    let samples: Vec<InputSample> = long_wave().into_iter().map(InputSample::at).collect();
    replayed.replay_stroke(Tool::Brush, &samples);

    assert!(
        images_match(&committed, &replayed.render_to_image(), 0),
        "fast commit is off and the stroke still did not land the replay's pixels: \
         worst {} levels",
        diff_fraction(&committed, &replayed.render_to_image()).1
    );
}

/// The setting reaches the frontend as the engine's own value, so a dialog reading
/// it back cannot disagree with what the engine believes (§4).
#[test]
fn fast_commit_is_projected() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert_eq!(
        engine.observe().fast_commit,
        stark_engine::DEFAULT_FAST_COMMIT,
        "a fresh engine does not report the default the frontend stores"
    );
    engine.process(ViewCommand::SetFastCommit(false));
    assert!(!engine.observe().fast_commit);
    engine.process(ViewCommand::SetFastCommit(true));
    assert!(engine.observe().fast_commit);
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
        "untouched corner should be substrate blue"
    );

    engine.process(DocCommand::Undo);
    assert!(engine.observe().can_redo);
    assert!(
        is_blue(center(&engine.render_to_image())),
        "after undo, center should be the substrate"
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
///
/// **Answers `None` where there is no GPU**, rather than a pair — and the difference
/// is the whole reason this does not build its own device. It used to, and returned
/// `(1.0, 1.0)` when the build failed: the caller's `late < early * 2.0` then read
/// `1.0 < 2.0` and the test reported `ok` having measured nothing, on a machine with
/// no adapter at all. Going through [`common::engine_or_skip_sized`] is what puts it
/// back under `STARK_ALLOW_NO_GPU` with every other test — a missing GPU is a
/// *failure* unless the skip was asked for (CLAUDE.md), and this was the one place
/// that had quietly opted out.
fn measure_per_move_growth(b: BrushParams) -> Option<(f64, f64)> {
    let size = stark_model::geom::Extent2 {
        width: 1280,
        height: 800,
    };
    let mut engine = common::engine_or_skip_sized(size)?;
    engine.process(ViewCommand::set_brush(b));
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
    Some((early / ne as f64, late / nl as f64))
}

#[test]
fn per_move_cost_does_not_grow_with_stroke_length() {
    for (what, b) in [
        ("swept", brush(RED, 14.0)),
        ("stamp loop", common::corpus::smear_brush(14.0)),
    ] {
        let Some((early, late)) = measure_per_move_growth(b) else {
            return;
        };
        // Generous, because this is a wall-clock measurement on a shared machine. What
        // it has to catch is *growth*, and the failure it guards against is severalfold.
        assert!(
            late < early * 2.0,
            "{what}: per-move cost grew with stroke length: \
             {early:.2} ms early vs {late:.2} ms late"
        );
    }
}

/// The supersampled resolve's whole claim (§6.2), measured off the tiles: the
/// **visible** edge of a heavy, hard stroke spans about a pixel.
///
/// The slab law re-sharpens the pixel footprint's τ ramp — `1 − exp(−K·m)` hugs
/// saturation until `K·m` falls to order 1 — so before the resolve this brush drew
/// its 10–90% transition across a fifth of a px: a step the tile grid can only
/// alias. A texel-spaced read cannot see that (interpolating 1 px samples inflates
/// any sub-texel step to look ~0.8 px wide), so the profile is reconstructed at
/// eighth-px resolution instead: the same stroke at eight sub-texel placements,
/// each rim texel's visible alpha pinned to its distance from that stroke's own
/// centreline. Before the resolve this measured ~0.2 px and fails the floor.
#[test]
fn a_heavy_hard_strokes_visible_edge_spans_a_px() {
    use stark_model::document::{BrushShape, LayerId};
    use stark_shaders::mirror::paint_common::OPACITY_K;

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let mut b = brush(RED, 12.0);
    b.shape = BrushShape::Round { hardness: 1.0 };
    b.drain = 0.0;
    b.paint_mut().expect("a paint brush").flow = 2.5;

    // Eight horizontal strokes, each 30 px apart plus an eighth of a px: stacking
    // their rims by distance-to-centreline samples one profile at 1/8 px. The
    // spacing clears the tip's whole reach (12 px + the filter's rim) plus the
    // measurement window, so no stroke's window reads its neighbour's paint.
    let phases = 8;
    let mut profile: Vec<(f32, f32)> = Vec::new();
    for j in 0..phases {
        let y_c = 24.3 + 30.0 * j as f32 + j as f32 / phases as f32;
        replay_with(
            &mut engine,
            b,
            &[Vec2::new(40.0, y_c), Vec2::new(216.0, y_c)],
        );
        // The upper rim: texel centres from well outside the tip to its core.
        for y in (y_c - 16.0) as i32..(y_c - 6.0) as i32 {
            let w = paint_at(&engine, LayerId::ROOT, Vec2::new(128.0, y as f32))
                .map_or(0.0, |(h, op)| 1.0 - (-(OPACITY_K * op * h)).exp());
            profile.push((y as f32 + 0.5 - y_c, w));
        }
    }
    profile.sort_by(|a, b| a.0.total_cmp(&b.0));

    let wmax = profile.iter().fold(0.0f32, |m, &(_, w)| m.max(w));
    assert!(
        wmax > 0.95,
        "the stroke's interior is not saturated (w = {wmax}), so this test \
         would measure a soft ramp rather than the slab law's edge"
    );

    // First crossing of each level, walking in from outside — the profile rises
    // monotonically across the rim, and the 1/8 px sampling makes interpolation
    // between neighbours exact to the reconstruction's own resolution.
    let cross = |level: f32| -> f32 {
        let lv = level * wmax;
        for pair in profile.windows(2) {
            let (u0, w0) = pair[0];
            let (u1, w1) = pair[1];
            if w0 <= lv && w1 > lv {
                return u0 + (u1 - u0) * (lv - w0) / (w1 - w0);
            }
        }
        panic!("the profile never crosses {level} of its own maximum");
    };
    let width = cross(0.9) - cross(0.1);
    assert!(
        (0.6..=1.8).contains(&width),
        "the visible 10–90% edge spans {width:.2} px; under 0.6 the slab law's \
         re-sharpening is back (the resolve is not engaging), over 1.8 the edge \
         has gone soft beyond the pixel footprint"
    );
}
