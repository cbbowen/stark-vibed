//! **The battery** (§9): the invariants every stroke in [`common::corpus`] must obey.
//!
//! The corpus and the battery are separate on purpose. A case there says *what to
//! draw* and what it is the only cover for; a check here says *what must be true*, and
//! is immediately asked of every stroke already in the list. Fuse the two — a stroke
//! written for one bug, checking the one property that bug broke — and the blind spots
//! come out shaped exactly like the combinations nobody happened to draw (see the
//! corpus's own header).
//!
//! Each case gets one `#[test]`, so a failure names the stroke, and the checks inside
//! it **accumulate**: a run reports every invariant the case broke rather than the
//! first, because "the frozen head is wrong" and "the picture moved" are different
//! diagnoses and seeing both at once is most of the work.

mod common;

use common::corpus::{CASES, Case, held_down, lifted};
use common::*;
use stark_engine::command::Tool;

use stark_engine::RgbaImage;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::SubstrateId;
use stark_model::document::BrushParams;
use stark_model::geom::{TILE_SIZE, Vec2};

/// One `#[test]` per case, plus the check that the two lists agree.
///
/// The pairing is by name and it is checked rather than assumed: adding a `Case` to
/// the corpus and forgetting to list it here would otherwise be a stroke that is
/// described, documented, and never drawn — the exact failure the corpus exists to
/// stop, reintroduced one level up.
macro_rules! battery {
    ($($case:ident),* $(,)?) => {
        $(
            #[test]
            fn $case() {
                run(stringify!($case));
            }
        )*

        const COVERED: &[&str] = &[$(stringify!($case)),*];

        #[test]
        fn every_case_in_the_corpus_is_run() {
            let missing: Vec<&str> = CASES
                .iter()
                .map(|c| c.name)
                .filter(|n| !COVERED.contains(n))
                .collect();
            assert!(
                missing.is_empty(),
                "these corpus cases have no test in the battery: {missing:?} — \
                 add them to `battery!`"
            );
        }
    };
}

battery!(
    line,
    curve,
    hairpin,
    pressure_ramp,
    taper,
    stamp_arc,
    pen_stamp,
    tooth_arc,
    smear,
    smear_taper,
    bleed,
    wide_smear,
    oversized_smear,
);

/// How many points along a stroke the incremental preview is held against a fresh
/// render. A frozen head is never redrawn, so a head handed over wrong at sample 60
/// stays wrong on the canvas forever and later spans painting over it can hide the
/// damage from a check at the end — which is why this walks the stroke rather than
/// looking once. Bounded because each stop costs two full renders.
const CHECKPOINTS: usize = 5;

/// How much finer the refinement check fits the same path. See [`check_refinement`].
const REFINEMENT: f32 = 4.0;

/// What counts as *visibly* moved: twelve levels out of 255, a difference nobody has
/// to squint at. Fixed globally rather than per case so that every case's `Tol::refine`
/// and `Tol::lift` are the same measurement and the corpus reads as one table.
const VISIBLE_LEVELS: u8 = 12;

/// Per-channel levels a case may move when the whole of it is drawn half a tile further
/// along and viewed from half a tile further along. See [`check_translation`].
///
/// **Global where `Tol`'s bounds are per case, and for `Tol`'s own reason.** Those are
/// per case because the answers differ by two orders of magnitude and the differences
/// are the interesting part. Here they do not differ: every case in the corpus comes in
/// at 1 to 3 levels, swept and sequential alike, which is f16 storage rounding. A column
/// of near-identical numbers would read as a table of measurements and be a table of
/// noise. The headroom above 3 is for another adapter's rounding, not for a case with
/// something to say.
const TRANSLATION_LEVELS: u8 = 4;

fn run(name: &str) {
    let case = CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no corpus case named {name:?}"));
    let Some((mut engine, brush)) = case.open() else {
        return;
    };
    let mut report = Report::new(case);

    let committed = check_the_stroke_as_it_is_drawn(case, &mut engine, brush, &mut report);
    check_repeat_render(&mut engine, &committed, &mut report);
    check_undo_redo(&mut engine, &committed, &mut report);
    check_save_load(case, &engine, &committed, &mut report);
    check_refinement(case, brush, &committed, &mut report);
    check_lift_off(case, brush, &mut report);
    check_translation(case, brush, &committed, &mut report);

    report.finish();

    // Last, and deliberately: the golden is a *description* of a correct render, and
    // describing one that has already been shown to break an invariant is noise. If
    // the run gets this far the stroke is internally consistent, and the only question
    // left is whether it still looks like what was blessed.
    assert_golden(&format!("corpus_{name}"), &committed, case.tol.golden);
}

// --- the checks ---------------------------------------------------------------

/// Draw the stroke the way the app does — sample by sample, with the preview rendered
/// as it goes — and return the committed image.
///
/// Two invariants come out of the one pass, because both are about the same thing:
/// **where the stroke is cut must not matter**.
///
/// * *Incremental == fresh*, at [`CHECKPOINTS`] points along the way. The live preview
///   is a frozen head kept from earlier moves plus a live tail over it; re-setting the
///   brush (to the brush it is already using, so nothing about the stroke changes)
///   drops the head, so the very next repaint renders the whole stroke in one pass.
/// * *Preview == commit*, at the release. The last frame previewed is what lands, or
///   the stroke visibly changes at the moment the pointer comes up (§1.3).
///
/// For the swept path this is nearly free — a segment's deposit is a definite integral
/// that composes by summing optical depth, so the cut genuinely cannot matter. For the
/// stamp loop it is the strongest claim in the suite: each segment reads the canvas the
/// previous one left and the tool the previous one loaded, and cutting it is only
/// sound because `gpu::stroke::ToolState` carries the reservoir, and the travel since
/// the last pickup, across the cut.
fn check_the_stroke_as_it_is_drawn(
    case: &Case,
    engine: &mut stark_engine::Engine,
    brush: BrushParams,
    report: &mut Report,
) -> RgbaImage {
    let samples = case.samples();
    let (first, rest) = samples.split_first().expect("a case draws something");

    engine.process(ViewCommand::SetBrush(brush));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: *first,
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });

    let stride = rest.len().div_ceil(CHECKPOINTS).max(1);
    for (i, s) in rest.iter().enumerate() {
        engine.process(GestureCommand::To { sample: *s });
        // The preview's color is only meaningful with the tail untinted; under
        // `debug-unfrozen` the live tail is repainted magenta by design, which is what
        // that feature is for and has its own test in `stroke.rs`.
        if cfg!(feature = "debug-unfrozen") || i % stride != 0 {
            continue;
        }
        let incremental = engine.render_to_image();
        engine.process(ViewCommand::SetBrush(brush));
        let fresh = engine.render_to_image();
        report.check(
            &format!("incremental preview at sample {i} vs a fresh render"),
            &incremental,
            &fresh,
            case.tol.seam,
        );
    }

    let preview = engine.render_to_image();
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();
    if !cfg!(feature = "debug-unfrozen") {
        report.check(
            "the last previewed frame vs what committed",
            &preview,
            &committed,
            case.tol.seam,
        );
    }
    committed
}

/// Rendering the same document twice gives the same pixels.
///
/// Trivial-looking, and it is the check that catches a compositing pass which
/// *accumulates* instead of resolving — a tile that is blended into its target rather
/// than written, an apron applied twice. Such a bug is invisible to every other check
/// here, all of which compare one render against another render.
fn check_repeat_render(
    engine: &mut stark_engine::Engine,
    committed: &RgbaImage,
    report: &mut Report,
) {
    let again = engine.render_to_image();
    report.check("a second render of the same document", committed, &again, 0);
}

/// Undo then redo restores the exact pixels.
///
/// History retention is what drives GPU tile reclamation (§1), so this is where a tile
/// recycled while a history entry still refers to it shows up — as the *next* stroke's
/// paint appearing inside this one.
fn check_undo_redo(engine: &mut stark_engine::Engine, committed: &RgbaImage, report: &mut Report) {
    engine.process(DocCommand::Undo);
    let undone = engine.render_to_image();
    engine.process(DocCommand::Redo);
    let redone = engine.render_to_image();
    report.check("undo then redo", committed, &redone, 0);
    if images_match(committed, &undone, 0) {
        report.note("undo left the stroke on the canvas");
    }
}

/// The document round-trips through its file (§8): saved, loaded into an engine that
/// has never seen this stroke, and replayed to the same pixels **exactly**.
///
/// This is the one idea the whole design rests on — pixels are a cached view of a
/// replayable action log — stated as a test, and it is a stronger claim than it looks:
/// the loaded engine reaches these pixels by re-running the actions through the same
/// renderer, so anything the live path carries in memory and the file does not carry
/// (a brush asset, a chosen substrate, the reservoir's initial state) breaks it.
fn check_save_load(
    case: &Case,
    engine: &stark_engine::Engine,
    committed: &RgbaImage,
    report: &mut Report,
) {
    let bytes = match engine.save_bytes() {
        Ok(b) => b,
        Err(e) => return report.note(&format!("the document would not serialize: {e}")),
    };
    // `open` again rather than a bare engine: `load_bytes` replaces the *document*, so
    // the undercoat it paints is discarded, but the case's **view** state — media
    // params, which are not document state (§4) — has to be in place for the render to
    // be comparable.
    let Some((mut loaded, _)) = case.open() else {
        return;
    };
    if let Err(e) = loaded.load_bytes(&bytes) {
        return report.note(&format!("the document would not load: {e}"));
    }
    report.check(
        "save → load → replay",
        committed,
        &loaded.render_to_image(),
        0,
    );
}

/// **The same gesture, described more precisely, must draw the same mark.**
///
/// The input tolerance is what a caller declares about its own precision — one canvas
/// px is a mouse at 1:1 — and the fit is priced against it (`path::KNOT_COST`), so the
/// same pointer reports at a quarter of it buy several times as many control points.
/// This is not a synthetic knob: it is what changes when the user zooms in, and a
/// stroke that redrew itself differently at 4× would be a bug a painter would find
/// before any test did.
///
/// Note what it does *not* vary. `flatten_tolerance` is a function of the brush alone,
/// so refining this does not sweep the flattening budget — it moves the fitted curve,
/// and with it **where the curve gets cut into segments**. That is still the property
/// worth pinning, and §6.2's rule that a segment's deposit be additive in `τ` (or of
/// the form `1 − exp(−k·τ)`) is exactly what makes moving the cuts harmless. What no
/// test here can reach is the budget itself, since it is internal to the renderer; the
/// nearest cover is each case's golden, which is why the corpus bothers to paint a
/// pressure ramp and a stamp around a curve at all.
///
/// It is a convergence check, not an equality. `line` is fitted exactly at any
/// tolerance and moves by nothing at all; a path with real curvature genuinely shifts
/// under a better fit, and its bound records how much. `pressure_ramp` used to sit
/// with `line` and no longer does — see its bound for what the curvature ridge costs
/// a stroke that has no curvature of its own.
///
/// **This is the one check measured by area rather than by worst texel**, and the two
/// other checks' preference for the maximum is exactly why. A seam is a step: loud in
/// the worst pixel, invisible in the average, so a bound on the maximum is the only one
/// that sees it. A convergence failure is the opposite — a staircase of radii, or a
/// stroke whose weight drifts with its segment count, covers *substrate*.
///
/// The maximum is also unusable here, for two separate reasons. A threshold in the
/// pipeline flips a handful of texels either side under any change at all: the tooth
/// gate (§6.4) is a comparison against the substrate, so a texel sitting on the line lands
/// differently for reasons that say nothing about convergence — six such pixels put
/// `tooth_arc`'s worst at 131 levels. And on a hard-edged stroke over a path with real
/// curvature, refining the tolerance genuinely *moves the curve*: the outline shifts a
/// fraction of a pixel and a hard edge answers with 200 levels along its whole length.
/// Bounds set to accommodate either would catch nothing. What the share catches is the
/// thing worth catching — how much of the picture is still moving.
fn check_refinement(case: &Case, brush: BrushParams, committed: &RgbaImage, report: &mut Report) {
    let Some((mut fine, _)) = case.open() else {
        return;
    };
    case.paint(&mut fine, brush, DEFAULT_TOLERANCE / REFINEMENT);
    let refined = fine.render_to_image();
    let moved = frac_exceeding(committed, &refined, VISIBLE_LEVELS) * 100.0;
    if moved > case.tol.refine {
        report.note(&format!(
            "fitted {REFINEMENT}× finer: {moved:.3}% of the viewport moved past \
             {VISIBLE_LEVELS} levels, allowed {:.3}%",
            case.tol.refine,
        ));
    }
}

/// **The pen coming off the tablet must not redraw the stroke it is leaving.**
///
/// A tablet keeps reporting through the release: the last handful of samples carry the
/// pressure down to nothing across a fraction of a pixel of tip travel
/// ([`lifted`]). Every one of them is a piece of path with no length, and §6.2 says
/// what a piece of path with no length deposits — a segment's contribution is a
/// definite integral over travel, so it is exactly zero, whatever the pen was doing
/// while it happened. Appending the release to a gesture must therefore draw the same
/// mark as not appending it, for **every** stroke in the corpus.
///
/// It is worth stating as an invariant rather than as one case's golden because the
/// release is not a kind of stroke — it is the last inch of *every* stroke a tablet
/// ever draws, and what it costs depends on what it is appended to. A swept line pays
/// for it in a blunted end; a stamp brush pays again through the reservoir, whose
/// exchange cadence is quoted in a radius the release is shrinking; a tapered brush
/// pays a third time, since the taper is measured from the stroke's ends and the
/// release moves one of them.
///
/// Measured by area rather than by worst texel, for the same reason as
/// [`check_refinement`]: a release does not put a step in one pixel, it quietly
/// changes the width of the last stretch of the mark.
///
/// **Against [`held_down`] rather than against the committed stroke**, which is what
/// makes it a check about the release and not about the six-tenths of a pixel the tip
/// slides while it happens. That travel is real, the fit is entitled to answer it, and
/// on the cases described by three or four reports its answer is most of the viewport —
/// so the control carries the same tail with the tip still down, and the difference is
/// the release alone. The committed render is not the reference here for the same
/// reason it *is* the reference everywhere else: this is the one check whose input is
/// not the case's own.
fn check_lift_off(case: &Case, brush: BrushParams, report: &mut Report) {
    let samples = case.samples();
    let draw = |input: &[InputSample]| {
        let (mut engine, _) = case.open()?;
        case.paint_input(&mut engine, brush, DEFAULT_TOLERANCE, input);
        Some(engine.render_to_image())
    };
    let (Some(released), Some(down)) = (draw(&lifted(&samples)), draw(&held_down(&samples))) else {
        return;
    };
    let moved = frac_exceeding(&down, &released, VISIBLE_LEVELS) * 100.0;
    if moved > case.tol.lift {
        report.note(&format!(
            "the pen lifted off the tablet: {moved:.3}% of the viewport moved past \
             {VISIBLE_LEVELS} levels, allowed {:.3}%",
            case.tol.lift,
        ));
    }
}

/// **Where the tile grid falls under a mark must not change the mark.**
///
/// The whole case — undercoat, substrate, stroke — is laid down half a tile further along
/// *both* axes, and the view is moved by exactly the same amount, so the same mark falls
/// on the same screen pixels. The two renders must agree.
///
/// This is §6.4 asked from the outside. Every pass that writes tiles must be a pure
/// function of canvas position, which is what makes a tile's apron bit-identical to its
/// neighbour's interior without a copy pass; `tests/seam.rs` checks that from the
/// inside, one pass at a time, on a synthetic pair of tiles. What it cannot reach is the
/// *composition* — a stroke's deposit, its reservoir exchanges, its bleed stencil, the
/// apron refreshes between them and the compositor reading the result, all landing on a
/// different set of tiles with the mark at a different phase against their boundaries.
///
/// **Half a tile is the offset that asks the question.** A whole tile is invariant for a
/// trivial reason: every tile-relative quantity is unchanged, so the render would agree
/// even if the pipeline were riddled with tile-local state. Half a tile puts every
/// boundary through the middle of where it used to be, so a mark that crossed one now
/// crosses two, spans are cut in different places, and a tile that was interior becomes
/// an edge. `TILE_SIZE` is 254, so the offset is 127.
///
/// Measured by **worst texel**, unlike its two neighbours, and the measurements are why:
/// every case comes in at 1 to 3 levels, which is f16 storage rounding and nothing else.
/// There is no haze to average over — a fault here is a seam, which is loud in the
/// maximum and quiet in the area.
fn check_translation(case: &Case, brush: BrushParams, committed: &RgbaImage, report: &mut Report) {
    let offset = Vec2::splat((TILE_SIZE / 2) as f32);
    let Some((mut moved, _)) = case.open_at(offset) else {
        return;
    };
    case.paint_input(
        &mut moved,
        brush,
        DEFAULT_TOLERANCE,
        &case.samples_at(offset),
    );
    let translated = moved.render_to_image();

    // Whether this case's picture is nailed to the **canvas** rather than to the
    // gesture. A substrate with relief is a height field in canvas coordinates (§6.4): the
    // tooth gate reads it where the paint lands and the media pass lights it where it
    // lies, so the grain under a mark is not the grain 127 px along. Such a case is not
    // translation invariant and must not be.
    //
    // Asked of the **document** rather than declared per case, so the exemption cannot
    // go stale — `Flat` is the one substrate with no relief, and a case that stops using
    // linen stops being exempt in the same edit.
    if moved.document().substrate != SubstrateId::Flat {
        // Checked in the opposite direction, because an exemption nothing tests is an
        // exemption that quietly widens. A woven case that came through translation
        // *unchanged* would mean the grain is no longer being read at canvas position,
        // which is the §6.4 violation this whole check is about — arriving as a case
        // that suddenly passes.
        let (_, worst) = diff_fraction(committed, &translated);
        if worst <= VISIBLE_LEVELS {
            report.note(
                "a woven substrate survived being moved half a tile: either the grain \
                 has stopped being read at canvas position (§6.4), or this case has \
                 stopped painting on one and no longer earns its exemption",
            );
        }
        return;
    }

    report.check(
        "the case drawn half a tile over, viewed from half a tile over",
        committed,
        &translated,
        TRANSLATION_LEVELS,
    );
}

// --- reporting ----------------------------------------------------------------

/// Collects every failed check so one run answers the whole question.
struct Report {
    case: &'static str,
    what: &'static str,
    failures: Vec<String>,
}

impl Report {
    fn new(case: &Case) -> Self {
        Self {
            case: case.name,
            what: case.what,
            failures: Vec::new(),
        }
    }

    /// `a` and `b` must agree within `tol` at **every** texel. A bound on the worst
    /// pixel rather than on how many are off: the failures worth catching here are
    /// steps and seams — a contiguous band where the stroke visibly changes — and
    /// those are loud in the maximum and quiet in the average.
    fn check(&mut self, what: &str, a: &RgbaImage, b: &RgbaImage, tol: u8) {
        if a.width != b.width || a.height != b.height {
            self.failures.push(format!(
                "{what}: size {}x{} vs {}x{}",
                a.width, a.height, b.width, b.height
            ));
            return;
        }
        let (_, worst) = diff_fraction(a, b);
        if worst <= tol {
            return;
        }
        self.failures.push(format!(
            "{what}: worst {worst} levels (tol {tol}); {:.3}% of px over tol, \
             {:.3}% over 2",
            frac_exceeding(a, b, tol) * 100.0,
            frac_exceeding(a, b, 2) * 100.0,
        ));
    }

    fn note(&mut self, what: &str) {
        self.failures.push(what.to_string());
    }

    fn finish(self) {
        if self.failures.is_empty() {
            return;
        }
        let list = self
            .failures
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "corpus case `{}` broke {} of the battery's checks:\n{list}\n\nthis case covers: {}",
            self.case,
            self.failures.len(),
            self.what,
        );
    }
}

/// Every case must actually paint something.
///
/// The whole battery compares renders against other renders, and a case whose stroke
/// missed the viewport, or whose brush laid nothing, would pass all of it — bare paper
/// equals bare paper, round-trips exactly, and converges beautifully. This is the one
/// check that looks at the picture and asks whether there is a stroke in it, and it is
/// what keeps a silently-empty case from reading as coverage.
#[test]
fn every_case_leaves_a_mark() {
    for case in CASES {
        let Some((mut engine, brush)) = case.open() else {
            return;
        };
        let before = engine.render_to_image();
        case.paint(&mut engine, brush, DEFAULT_TOLERANCE);
        let after = engine.render_to_image();
        let (moved, worst) = diff_fraction(&before, &after);
        assert!(
            worst > 8 && moved > 0.001,
            "corpus case `{}` painted nothing worth looking at: {:.4}% of px moved, \
             worst {worst} levels",
            case.name,
            moved * 100.0,
        );
    }
}

/// A case's samples are its own: nothing in the battery may depend on the corpus
/// handing back the same `Vec` twice, since every check re-draws the stroke.
#[test]
fn a_case_replays_its_own_input() {
    for case in CASES {
        let (a, b) = (case.samples(), case.samples());
        assert_eq!(a.len(), b.len(), "case `{}` is not reproducible", case.name);
        assert!(
            a.iter().zip(&b).all(same_sample),
            "case `{}` is not reproducible",
            case.name
        );
    }
}

fn same_sample((a, b): (&InputSample, &InputSample)) -> bool {
    a.pos == b.pos && a.pressure == b.pressure && a.tilt == b.tilt && a.time == b.time
}
