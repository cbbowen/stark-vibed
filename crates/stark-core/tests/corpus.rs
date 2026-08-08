//! **The battery** (§9): the invariants every stroke in [`common::corpus`] must obey.
//!
//! The corpus and the battery are separate on purpose. A case there says *what to
//! draw* and what it is the only cover for; a check here says *what must be true*, and
//! is immediately asked of every stroke already in the list. The suite used to fuse
//! the two — a stroke was written for one bug and checked the one property that bug
//! broke — and the cost of that was a set of blind spots shaped exactly like the
//! combinations nobody had drawn (see the corpus's own header).
//!
//! Each case gets one `#[test]`, so a failure names the stroke, and the checks inside
//! it **accumulate**: a run reports every invariant the case broke rather than the
//! first, because "the frozen head is wrong" and "the picture moved" are different
//! diagnoses and seeing both at once is most of the work.

mod common;

use common::corpus::{CASES, Case};
use common::*;

use stark_core::RgbaImage;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, Tool};
use stark_core::path::DEFAULT_TOLERANCE;

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
    dab,
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

/// What counts as *visibly* moved under refinement: twelve levels out of 255, a
/// difference nobody has to squint at. Fixed globally rather than per case so that
/// every case's `Tol::refine` is the same measurement and the corpus reads as one
/// table of convergence.
const REFINE_LEVELS: u8 = 12;

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
    engine: &mut stark_core::Engine,
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
    });

    let stride = rest.len().div_ceil(CHECKPOINTS).max(1);
    for (i, s) in rest.iter().enumerate() {
        engine.process(GestureCommand::To { sample: *s });
        // The preview's colour is only meaningful with the tail untinted; under
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
    engine: &mut stark_core::Engine,
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
fn check_undo_redo(engine: &mut stark_core::Engine, committed: &RgbaImage, report: &mut Report) {
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
/// (a brush asset, a chosen ground, the reservoir's initial state) breaks it.
fn check_save_load(
    case: &Case,
    engine: &stark_core::Engine,
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
/// It is a convergence check, not an equality. The straight cases (`line`,
/// `pressure_ramp`, `dab`) are fitted exactly at any tolerance and move by nothing at
/// all; a path with real curvature genuinely shifts under a better fit, and its bound
/// records how much.
///
/// **This is the one check measured by area rather than by worst texel**, and the two
/// other checks' preference for the maximum is exactly why. A seam is a step: loud in
/// the worst pixel, invisible in the average, so a bound on the maximum is the only one
/// that sees it. A convergence failure is the opposite — a staircase of radii, or a
/// stroke whose weight drifts with its segment count, covers *ground*.
///
/// The maximum is also unusable here, for two separate reasons. A threshold in the
/// pipeline flips a handful of texels either side under any change at all: the tooth
/// gate (§6.4) is a comparison against the weave, so a texel sitting on the line lands
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
    let moved = frac_exceeding(committed, &refined, REFINE_LEVELS) * 100.0;
    if moved > case.tol.refine {
        report.note(&format!(
            "fitted {REFINEMENT}× finer: {moved:.3}% of the viewport moved past \
             {REFINE_LEVELS} levels, allowed {:.3}%",
            case.tol.refine,
        ));
    }
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
