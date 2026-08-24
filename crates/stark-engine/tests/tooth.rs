//! The deposition tooth (§6.4): the canvas's own substrate gating how much of the
//! brush's paint lands.
//!
//! Two claims, and neither is "the mark gets patchy" — dither would do that.
//!
//! The first is that the patches are a **level set of one field**: take the tip's give away
//! and the texels that still take paint are the ones that already took it, because
//! both are reading the substrate's rise along the travel through a rising threshold
//! (the follow limit, `paint_common.wesl`). That is what makes successive strokes in
//! one direction register with each other instead of stacking independent noise, and
//! it is what [`turning_the_tooth_up_takes_a_level_set_away`] measures.
//!
//! The second is that the field is the **slope of the substrate along the stroke**, not
//! its height: a dragged tip is pressed up by substrate that rises to meet it and left
//! bridging substrate that falls away, so it catches the near face of every bump and
//! skips the lee side. That makes the mark a property of the stroke as well as the
//! substrate — run the same line the other way and the ink lands on the other side of
//! every feature ([`a_stroke_catches_on_the_faces_it_meets`]), which is precisely
//! what a height threshold cannot do.
//!
//! Everything here paints on **Rough**: an irregular substrate, whose rise distribution
//! is a smooth spread rather than the handful of discrete slopes a regular substrate
//! gives, so a level-set claim about it says something.
//!
//! The model's own arithmetic — the sign of the rise, its null cases, and the knob's
//! ends — is pinned without a GPU in `gpu::substrate`'s unit tests, on a substrate of
//! ramps built for the purpose.

mod common;

use common::*;
use stark_engine::command::DocCommand;
use stark_model::SubstrateId;
use stark_model::document::{BrushDynamics, BrushParams, BrushShape};
use stark_model::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// The contact transition every brush here is drawn through, and every bearing here is
/// asked at (§6.4).
///
/// **The tests' own, deliberately not `BrushParams::DEFAULT_TOOTH_SOFTNESS`.** What the
/// shipped default is set to is taste — which tool the app opens on — and it gets
/// retuned; what this file measures is the deposition model, and a change of taste must
/// not be able to move a level-set claim about the grain. `gpu::substrate::tooth`'s unit
/// tests keep a `NARROW` of their own for the same reason, and it is the same number:
/// the bundled substrates' interquartile rise, which is what makes the gate a threshold
/// on the rise rather than a tone across it.
const BAND: f32 = 0.06;

/// A brush with a given tooth — the tip's **give** against the substrate, so 1 is the
/// solid mark and 0 the driest (`BrushParams::tooth_give`) — and nothing else that
/// varies along a stroke: no drain, no taper, and — deliberately — **no size
/// mapping**, so every stroke here covers exactly the same substrate and only the
/// gate differs.
fn toothed(give: f32) -> BrushParams {
    BrushParams {
        drain: 0.0,
        tooth_give: give,
        tooth_softness: BAND,
        shape: BrushShape::Round { hardness: 0.9 },
        modulation: Default::default(),
        dynamics: BrushDynamics {
            // Low flow: the tooth decides how much of the parcel lands, and at a flow
            // that saturates the slab law every texel reads as solid whatever fraction
            // it received. A dry-brush mark is the case this axis is *for*.
            flow: 0.35,
            ..Default::default()
        },
        ..brush(RED, 26.0)
    }
}

/// The stroke every test here draws: a straight run across the middle, long enough
/// to cross several of the substrate's features.
fn run() -> [Vec2; 2] {
    [Vec2::new(-140.0, 0.0), Vec2::new(140.0, 0.0)]
}

/// The rough substrate imported into `engine`, and the id that names it.
///
/// The id comes *out of* the height map (§6.4), so it cannot be named before the
/// bytes are in hand — which is the property that stops a substrate going quietly
/// missing, and the reason every test here imports rather than asserting a name.
fn rough(engine: &mut stark_engine::Engine) -> SubstrateId {
    engine
        .import_substrate(&stark_testdata::assets::rough())
        .expect("the rough height map imports")
}

/// An engine on the rough substrate. Without a real height map a substrate is `Flat`,
/// whose relief is 0, and every test here would silently be testing nothing.
fn rough_engine() -> Option<stark_engine::Engine> {
    let mut engine = engine_or_skip()?;
    let id = rough(&mut engine);
    engine.process(DocCommand::SetSubstrate(id));
    Some(engine)
}

/// Which texels the stroke moved, and by how much in total.
fn mark(before: &stark_engine::RgbaImage, after: &stark_engine::RgbaImage) -> (Vec<bool>, f64) {
    let mut hit = Vec::with_capacity(before.pixels.len() / 4);
    let mut ink = 0.0;
    for (a, b) in before
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.pixels.as_chunks::<4>().0)
    {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap_or(0);
        hit.push(d > 12);
        ink += d as f64;
    }
    (hit, ink)
}

/// Draw one stroke on a clean canvas and measure it, leaving the canvas as it was.
fn one_mark(engine: &mut stark_engine::Engine, b: BrushParams) -> (Vec<bool>, f64) {
    let before = engine.render_to_image();
    stroke_with(engine, b, &run());
    let out = mark(&before, &engine.render_to_image());
    engine.process(DocCommand::Undo);
    out
}

/// `tooth = 0` leaves the mark **solid on a substrate that would otherwise break it**:
/// every texel a toothed setting reaches is reached here too, and many it does not.
///
/// The stronger claim — that the deposit at `tooth = 0` is the float it always was,
/// to the bit — is not tested here because there is nothing in-process to compare it
/// against. What carries it is the **golden suite**, every one of which paints at
/// `tooth = 0` and none of which moved when the axis landed. That is the same
/// evidence §6.4 cites for the original tooth having been inert, read the other way
/// round.
#[test]
fn no_tooth_leaves_a_substrate_that_has_tooth_alone() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let (solid, solid_ink) = one_mark(&mut engine, toothed(1.0));
    let (broken, _) = one_mark(&mut engine, toothed(0.45));

    assert!(solid_ink > 0.0, "the test drew nothing");
    let missed = broken
        .iter()
        .zip(&solid)
        .filter(|(b, s)| **b && !**s)
        .count();
    assert_eq!(
        missed, 0,
        "{missed} texels took paint through the tooth that the un-toothed brush \
         skipped — `tooth = 0` is not the top of the level-set family"
    );
}

/// A smooth canvas has no tooth to catch on, whatever the brush says — and that is
/// structural, not a check: `SubstrateMap::relief` is 0 on `Flat`, which zeroes the uv
/// scale the shader gates on before the brush's number is ever consulted.
///
/// It is what keeps the axis orthogonal to the golden suite, nearly all of which
/// paints on `Flat` to isolate some other feature (§6.4).
#[test]
fn a_smooth_canvas_has_no_tooth_whatever_the_brush_says() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let before = engine.render_to_image();
    stroke_with(&mut engine, toothed(1.0), &run());
    let smooth = engine.render_to_image();
    engine.process(DocCommand::Undo);
    stroke_with(&mut engine, toothed(0.1), &run());
    let toothy = engine.render_to_image();

    assert_eq!(
        diff_fraction(&smooth, &toothy),
        (0.0, 0),
        "on a flat substrate the tooth has nothing to bite and must change no pixel"
    );
    assert!(mark(&before, &smooth).1 > 0.0, "the test drew nothing");
}

/// The headline: a substrate with tooth breaks the mark up, and turning the knob up
/// takes more of it away.
#[test]
fn a_substrate_with_tooth_breaks_the_mark_up() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let (solid, solid_ink) = one_mark(&mut engine, toothed(1.0));
    let (broken, broken_ink) = one_mark(&mut engine, toothed(0.45));

    let solid_n = solid.iter().filter(|h| **h).count();
    let broken_n = broken.iter().filter(|h| **h).count();
    assert!(
        solid_n > 0,
        "the un-toothed stroke covered nothing — the substrate is not registered"
    );
    assert!(
        broken_n < solid_n,
        "a toothed brush should skip part of the substrate: {broken_n} of {solid_n} texels"
    );
    assert!(
        broken_n > solid_n / 20,
        "…but not all of it: {broken_n} of {solid_n} texels is not a broken mark, it is no mark"
    );
    assert!(
        broken_ink < solid_ink,
        "and it should lay less paint overall: {broken_ink} vs {solid_ink}"
    );
}

// --- the leading edge (§6.4) ------------------------------------------------------

/// Draw the same line in the given direction and measure how much ink landed on each
/// texel, leaving the canvas as it was. `+1` runs left to right, `-1` right to left,
/// over exactly the same texels.
///
/// Per-texel ink rather than [`mark`]'s hit set, because what the leading edge moves is
/// *where within the envelope* the paint sits, and a threshold only registers that
/// where it happens to straddle the line.
fn one_run(engine: &mut stark_engine::Engine, b: BrushParams, way: f32) -> Vec<f64> {
    let before = engine.render_to_image();
    let [a, z] = run();
    let path = if way > 0.0 { [a, z] } else { [z, a] };
    stroke_with(engine, b, &path);
    let after = engine.render_to_image();
    engine.process(DocCommand::Undo);
    before
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.pixels.as_chunks::<4>().0)
        .map(|(p, q)| {
            p.iter()
                .zip(q)
                .map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max()
                .unwrap_or(0) as f64
        })
        .collect()
}

/// How far apart two marks are, as a share of the ink in them: `Σ|a−b| / Σ(a+b)/2`.
/// 0 is the same mark texel for texel, 2 is two marks with no texel in common.
fn apart(a: &[f64], b: &[f64]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    let ink: f64 = a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).sum();
    if ink > 0.0 { diff / ink } else { 0.0 }
}

/// **The headline claim of the leading edge.** A brush stroke is not a stamp: the tip
/// rides up onto the substrate it is about to meet, so it catches on the near face of
/// every bump and bridges the lee side behind it. Run the same line the other way and
/// the near faces become the far ones — so the *same* brush on the *same* substrate
/// leaves a materially different mark.
///
/// That is the whole of what separates this from a height threshold, which cannot tell
/// the two runs apart at all, and it is why the mark reads as a brush dragged over
/// tooth rather than as a screen printed through it.
///
/// Measured as disagreement between the two marks rather than as a shift, because a
/// shift is not what happens: each bump keeps its paint, on the side of itself the tip
/// arrived from.
#[test]
fn a_stroke_catches_on_the_faces_it_meets() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let there = one_run(&mut engine, toothed(0.45), 1.0);
    let back = one_run(&mut engine, toothed(0.45), -1.0);
    let (there_ink, back_ink) = (there.iter().sum::<f64>(), back.iter().sum::<f64>());
    assert!(
        there_ink > 0.0 && back_ink > 0.0,
        "the test drew nothing: {there_ink} and {back_ink}"
    );

    // Neither run may be a wholesale loss of the mark — the anticipation moves contact
    // between the faces of the grain, it does not throw it away. The two runs read the
    // same substrate through the same curve, only from opposite sides.
    let ratio = back_ink / there_ink;
    assert!(
        (0.85..1.18).contains(&ratio),
        "reversing the stroke changed how much paint landed by more than the faces it \
         moved between can account for: {back_ink} vs {there_ink}"
    );
    // The two marks share an envelope; what separates them is which side of each
    // feature took the paint. An eighth of the ink sitting somewhere else is already
    // far past anything the renderer varies by on its own — the same stroke drawn
    // twice is identical to the bit — and the slope gate moves several times that.
    let moved = apart(&there, &back);
    assert!(
        moved > 0.12,
        "the two directions laid all but {:.1}% of their ink on the same texels — the \
         tooth is reading the substrate under the tip rather than the substrate ahead of it",
        moved * 100.0
    );
}

/// …and at full give there is no slope to read, so the direction stops mattering.
/// The guard that makes it so is structural — the shaders return a gate of exactly
/// 1.0 at `tooth = 0` before the map is read, and the follow limit dives past any
/// encodable fall well before the knob gets there, so the approach is continuous —
/// which is what lets every golden in the suite go on painting at `tooth = 0` while
/// this axis exists.
///
/// Measured against the same yardstick as the test above rather than asserted texel for
/// texel: a reversed stroke is not the bit-identical draw call, because its arc length
/// runs the other way and its segments start from the other end, so the sweep frame
/// rounds differently along it. That residual is the renderer's, it predates this axis,
/// and it is two orders below what the anticipation moves.
#[test]
fn with_no_tooth_the_direction_stops_mattering() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let there = one_run(&mut engine, toothed(1.0), 1.0);
    let back = one_run(&mut engine, toothed(1.0), -1.0);

    assert!(there.iter().sum::<f64>() > 0.0, "the test drew nothing");
    let moved = apart(&there, &back);
    assert!(
        moved < 0.005,
        "an un-toothed brush moved {:.2}% of its ink by being drawn the other way",
        moved * 100.0
    );
}

/// **The bearing-curve claim.** For strokes drawn one way, the tooth is a threshold
/// on one rise field, so the texels a stiffer tip keeps are a *subset* of the ones a
/// softer one kept — the marks are nested level sets, not independent noise.
///
/// This is the property that separates a substrate from a dither, and it is the reason
/// overpainting registers: two strokes the same way at the same tooth catch on the
/// same faces, so the second one lands *on* the first rather than filling its gaps.
///
/// The tolerance is for the soft transition band (`BrushParams::tooth_softness`) and for the 12-
/// level threshold `mark` reads coverage at: a texel sitting inside the band at both
/// settings can land either side of that line. What cannot survive by accident is
/// 95% agreement across thousands of texels.
#[test]
fn turning_the_tooth_up_takes_a_level_set_away() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let (shallow, _) = one_mark(&mut engine, toothed(0.65));
    let (deep, _) = one_mark(&mut engine, toothed(0.4));

    let deep_n = deep.iter().filter(|h| **h).count();
    let escaped = deep
        .iter()
        .zip(&shallow)
        .filter(|(d, s)| **d && !**s)
        .count();
    assert!(
        deep_n > 200,
        "too little mark left to say anything: {deep_n}"
    );
    assert!(
        (escaped as f64) < 0.05 * deep_n as f64,
        "the deeper tooth kept {escaped} of {deep_n} texels the shallower one skipped — \
         the two marks are not level sets of one substrate"
    );
}

/// The tooth reaches the stamp loop too, and through the *same* function.
///
/// Worth its own test because the two paths compute the gate in different shaders
/// from different bindings, and the failure mode is the one §6.2 has been bitten by
/// before: a brush whose behaviour changes because some unrelated axis moved it onto
/// the other path. Nudging `lift` off zero must not change what the substrate does.
///
/// Measured with a **hysteresis pair** rather than one hit threshold, because the
/// two marks legitimately differ a few percent in *amount* — the loop's lift takes
/// back a share of the `add` it lays — and the slope gate leaves much of the substrate
/// mid-contact, so any single threshold has a large borderline population that a few
/// percent flips. A texel one path calls 11 levels and the other 13 is agreement in
/// substance; what the gate reading *differently* looks like is solid ink on one path
/// where the other left bare canvas.
///
/// **The pair is a fraction of the mark's own contrast, not a level count**, and that
/// is not fussiness. A level count is a claim about the light (§6.3): change the
/// lighting and the ink-to-substrate swing changes with it — the mark's strong ink runs
/// to 190 levels under the reference light and to a few tens under a warm studio one —
/// so fixed thresholds slide down into the rim, where a borderline population lives
/// that neither path is claiming anything about. A pair quoted against the swing it is
/// a share of cannot be moved that way. Half of the mark's strong ink is solid; a
/// twentieth of it is bare.
#[test]
fn the_tooth_reads_the_same_on_both_render_paths() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    // `lift` is what decides the path (`dynamics_setup`), and a lift with nothing to
    // pick up off bare canvas leaves the `add` paint as the only thing landing — so
    // the two strokes differ in which shader ran and in nothing else that shows.
    let swept = toothed(0.5);
    let looped = BrushParams {
        dynamics: BrushDynamics {
            lift: 0.05,
            ..swept.dynamics
        },
        ..swept
    };
    let a = one_run(&mut engine, swept, 1.0);
    let b = one_run(&mut engine, looped, 1.0);

    // The mark's own scale: the 90th percentile of every non-zero delta either path
    // laid, which is strong ink without being the single darkest texel.
    let strong = {
        let mut v: Vec<f64> = a.iter().chain(&b).copied().filter(|d| *d > 0.0).collect();
        v.sort_by(f64::total_cmp);
        assert!(!v.is_empty(), "neither path laid anything");
        v[(v.len() - 1) * 9 / 10]
    };
    let (solid, blank) = (0.5 * strong, 0.05 * strong);
    let material = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| **x > solid || **y > solid)
        .count();
    let disagree = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (**x > solid && **y < blank) || (**y > solid && **x < blank))
        .count();
    assert!(material > 200, "too little mark to compare: {material}");
    // The slack is for the tip's hardness shoulder, where τ falls off a cliff and the
    // two paths' different prefix machinery can land a rim texel a fraction of a texel
    // apart. A path whose gate differed would flip a third of the mark, not a rim.
    // Measured at 3.4% of 7091 material texels.
    assert!(
        (disagree as f64) < 0.05 * material as f64,
        "the two paths disagree about the substrate on {disagree} of {material} texels (solid > {solid:.0}, blank < {blank:.0})"
    );
    let (ink_a, ink_b) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    let ratio = ink_b / ink_a;
    assert!(
        (0.8..1.25).contains(&ratio),
        "and about how much they laid through it: {ink_b} vs {ink_a}"
    );
}

/// The substrate is the **document's**, so a stroke deposits through the tooth that was
/// in force where it sits in the log — not through whatever the canvas was switched
/// to afterwards (§6.4).
///
/// Asserted through a save/load round trip, because that is the only thing that
/// actually re-runs the earlier strokes: a live `SetSubstrate` leaves their tiles
/// alone, so it could not tell a correct engine from one holding a single "current
/// substrate" the whole log renders against. On load the log is replayed in order onto
/// an empty document, and a stroke laid before the switch has to come back with its
/// tooth and one laid after it without.
///
/// This is what the substrate registry moving onto the apply context buys, and why
/// `SetSubstrate` was a logged action from the start rather than a view setting — §6.4
/// says so in as many words, several years of commits before there was anything to
/// read it.
#[test]
fn a_stroke_keeps_the_substrate_it_was_painted_on() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    let before = engine.render_to_image();
    // One stroke on the toothed substrate, then the canvas is switched to smooth and a
    // second is laid across the other half.
    stroke_with(
        &mut engine,
        toothed(0.45),
        &[Vec2::new(-140.0, -40.0), Vec2::new(140.0, -40.0)],
    );
    engine.process(DocCommand::SetSubstrate(SubstrateId::Flat));
    stroke_with(
        &mut engine,
        toothed(0.45),
        &[Vec2::new(-140.0, 40.0), Vec2::new(140.0, 40.0)],
    );
    let live = engine.render_to_image();
    assert!(mark(&before, &live).1 > 0.0, "the test drew nothing");

    let bytes = engine.save_bytes().expect("serialize");
    let Some(mut loaded) = engine_or_skip() else {
        return;
    };
    // Nothing hands the substrate to this engine: the *file* carries it (§6.4, §8).
    // That is the second half of what makes the round trip meaningful — a file that
    // named its substrate and left the image to the reader would replay these strokes
    // on the flat stand-in, and pixels cannot say whether a substrate was reproduced or
    // merely defaulted to.
    loaded.load_bytes(&bytes).expect("deserialize + replay");

    assert_eq!(
        diff_fraction(&live, &loaded.render_to_image()),
        (0.0, 0),
        "the replay did not put each stroke back on the substrate it was painted on",
    );
}

// --- the toothed transfer, and whether it still conserves ------------------------

/// **A toothed transfer must still conserve paint**, and this is the test the
/// exposure-scaling formulation exists to pass.
///
/// The tooth gates a *transfer*, and a transfer has two halves computed in two
/// different dispatches: the canvas gives up `1 − keep` and takes `dep` of the tool's
/// load, while the tool takes and gives the exact complements. Gate one half and not
/// the other and the books stop closing. Gating the **exposure** rather than the
/// shares is what keeps them complementary — `exchange_at` is solved once at `g·e`,
/// so its four fractions still sum to what went in.
///
/// **Measured on a finite glob, and that shape is the point.** A smear inside a field
/// is the obvious test and it is nearly blind here: lift and deposit run at the same
/// place, so a tool that both takes and gives too much is wrong twice in opposite
/// directions and the canvas ink barely moves. (Checked — that version of this test
/// passes with the tool's half of the gate deleted.) A pre-`charge`d tool with **no
/// lift** has only one direction to be wrong in: it starts holding a known amount and
/// the canvas is the only place that amount can go, so a tool that gives up more than
/// the canvas accepts destroys the difference, and one that gives up less than the
/// canvas takes mints it. Run long enough to empty, the total delivered is the charge
/// whatever the substrate does — the tooth decides *where* it lands, not how much there
/// was.
#[test]
fn a_toothed_transfer_delivers_the_whole_glob() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    // **Lighting flat, dither off.** The measure below is a stand-in for mass, and a
    // lit render is not one: the media pass takes its normals from the height gradient
    // (§6.3), so the same paint laid bumpily catches more light than the same paint
    // laid smooth — which is the whole visible point of the tooth, and here it is the
    // contaminant. At `height_strength = 0` and `substrate_strength = 0` the pass is
    // the reference identity and the deviation from bare tracks visible alpha, which
    // tracks mass. The display dither is the same story one layer down: it makes
    // sub-code ink *register* (§6.5), which is its whole visible point — and here it
    // would inflate exactly the faint tail this test needs to see stay faint.
    engine.process(stark_engine::command::ViewCommand::SetMediaParams(
        stark_engine::MediaParams {
            height_strength: 0.0,
            substrate_strength: 0.0,
            dither: false,
            ..Default::default()
        },
    ));
    // `add = 0`, no lift: the only paint in play is the pre-loaded glob, and the only
    // place it can go is the canvas. A long stroke at a brisk deposit rate, so the
    // tool is empty well before the end either way and the total delivered is the
    // whole charge rather than however far each got.
    let glob = |give: f32| BrushParams {
        tooth_give: give,
        tooth_softness: BAND,
        dynamics: BrushDynamics {
            flow: 0.0,
            lift: 0.0,
            deposit: 0.6,
            charge: 0.25,
            ..Default::default()
        },
        ..BrushParams {
            drain: 0.0,
            shape: BrushShape::Round { hardness: 0.9 },
            modulation: Default::default(),
            ..brush([1.0, 0.0, 0.0, 0.25], 30.0)
        }
    };
    let path = [Vec2::new(-150.0, 0.0), Vec2::new(150.0, 0.0)];

    let bare = engine.render_to_image();
    stroke_with(&mut engine, glob(1.0), &path);
    let (plain, plain_ink) = far_share(&bare, &engine.render_to_image());
    engine.process(DocCommand::Undo);
    stroke_with(&mut engine, glob(0.4), &path);
    let (toothed, toothed_ink) = far_share(&bare, &engine.render_to_image());

    assert!(
        plain_ink > 0.0 && toothed_ink > 0.0,
        "a glob laid nothing at all"
    );
    // The un-toothed tool empties inside the first third and never really reaches
    // here — measured at 1.1% of its ink against the toothed one's 15.6%. Quoted
    // against *that*, not as a level of its own: `far_share` weights each texel by how
    // far it moved the pixel, so how much of a faint tail registers at all is a
    // property of the light the suite paints under (§6.3), and this bound was set
    // under a different one. What the test needs is that the two reaches are not the
    // same reach.
    assert!(
        plain * 4.0 < toothed,
        "the un-toothed glob was still going past the two-thirds mark ({plain:.3} against the toothed {toothed:.3}) — it is not emptying fast enough for this test to say anything"
    );
    // The toothed one trades at reduced contact, so the same charge lasts longer and
    // some of it is still being laid out there. If the tool drained at the *ungated*
    // rate while the canvas accepted only the gated share, it would run out in the
    // same place as the un-toothed one and the difference would simply be gone.
    assert!(
        toothed > 0.025,
        "the toothed glob ran out where the un-toothed one did ({toothed:.3} of its          ink past the two-thirds mark) — the tool is draining at the ungated rate          while the canvas accepts the gated share, so the difference is destroyed"
    );
}

/// How much of a mark's ink lands past the two-thirds mark of the stroke — where the
/// glob got to before it ran out — and the mark's total, to catch a test measuring
/// nothing.
///
/// A *distribution* measure rather than a total, and that is deliberate. Total ink
/// cannot answer a conservation question here: the slab law is concave in thickness,
/// so the same mass spread thinner measures as more, and the tooth changes exactly
/// how spread out the glob is. Where the glob dies is immune to that — it is set by
/// the rate the tool drains at, which is the half of the transfer under test.
fn far_share(before: &stark_engine::RgbaImage, after: &stark_engine::RgbaImage) -> (f64, f64) {
    let w = before.width as usize;
    let (mut far, mut all) = (0.0, 0.0);
    for (i, (a, b)) in before
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.pixels.as_chunks::<4>().0)
        .enumerate()
    {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap_or(0) as f64;
        all += d;
        if i % w > w * 2 / 3 {
            far += d;
        }
    }
    (if all > 0.0 { far / all } else { 0.0 }, all)
}

/// The bearing fraction is the mean of the gate over the substrate, so it has to *be*
/// that: monotone down in the tooth, exactly 1 where there is nothing to bite.
///
/// It is a CPU mirror of a GPU function (`SubstrateMap::bearing` against
/// `paint_common.wesl::tooth_gate`), and a mirror is only as good as what notices it
/// drifting. What notices is the conservation test above; this one pins the shape so
/// a failure there points at the right half.
#[test]
fn the_bearing_fraction_tracks_the_substrate() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let east = Vec2::new(1.0, 0.0);
    let soft = BAND;
    let flat = engine.substrate_bearing(SubstrateId::Flat, 0.5, soft, east);
    assert_eq!(flat, 1.0, "a smooth substrate is full contact at any tooth");

    let substrate = rough(&mut engine);
    let at = |g| engine.substrate_bearing(substrate, g, soft, east);
    assert_eq!(at(1.0), 1.0, "full give is full contact, exactly");
    let (a, b, c) = (at(0.75), at(0.5), at(0.25));
    assert!(
        1.0 > a && a > b && b > c && c > 0.0,
        "contact must fall as the tip loses the give to follow the substrate: {a} {b} {c}"
    );
}

/// The bearing curve is tabulated per direction, so it has to be **continuous** in
/// one: the table is sampled at sixteen angles and read between them, and a tool that
/// stepped as the pen turned would book a different share of a smooth arc from one
/// segment to the next and print the step.
///
/// Also that the direction is *carried* at all — a lookup that ignored it would come
/// back bit-identical from every angle, which is what this would catch if the
/// interpolation were ever short-circuited.
#[test]
fn the_bearing_curve_is_continuous_in_the_direction() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let substrate = rough(&mut engine);
    // A full turn in fine steps: no adjacent pair may jump by more than a small
    // fraction of the curve's own value, and the walk must close on itself.
    let at = |engine: &mut stark_engine::Engine, turns: f32| {
        let a = turns * std::f32::consts::TAU;
        engine.substrate_bearing(substrate, 0.45, BAND, Vec2::new(a.cos(), a.sin()))
    };
    let samples: Vec<f32> = (0..=128)
        .map(|i| at(&mut engine, i as f32 / 128.0))
        .collect();
    let first = samples[0];
    assert!(first > 0.0 && first < 1.0, "nothing to say at {first}");
    assert!(
        (samples[128] - first).abs() < 1e-6,
        "the direction table does not close on itself: {first} vs {}",
        samples[128]
    );
    let jump = samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        jump < 0.02 * first,
        "the bearing steps by {jump} between neighbouring directions (of {first}) — \
         the table is being read without interpolation"
    );
}

/// **`preview == committed` with a tooth** (§1.3). The live path renders a frozen
/// head and then a tail over it, and that is what the commit lands (`PreparedStroke`,
/// §6.2); a replay renders the whole stroke in one range. A gate that is a pure
/// function of canvas position factors out of the sum over segments, so the two must
/// agree — this is where that claim is checked, since the suite's existing
/// split-preview test runs on a substrate with no relief.
#[test]
fn a_toothed_live_preview_matches_the_commit() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    engine.process(stark_engine::command::ViewCommand::SetBrush(toothed(0.45)));
    let path: Vec<Vec2> = (0..120)
        .map(|i| {
            let t = i as f32 * 0.05;
            Vec2::new(t * 22.0 - 60.0, (t * 1.1).sin() * 30.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(stark_engine::command::GestureCommand::Start {
        tool: stark_engine::command::Tool::Brush,
        sample: stark_engine::command::InputSample::at(*it.next().unwrap()),
        tolerance: stark_engine::path::DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for &p in it {
        engine.process(stark_engine::command::GestureCommand::To {
            sample: stark_engine::command::InputSample::at(p),
        });
    }
    engine.process(stark_engine::command::GestureCommand::End);
    assert_eq!(
        engine.strokes_reused(),
        1,
        "the commit did not take the preview"
    );
    let committed = engine.render_to_image();
    let rendered = whole_render(&mut engine);

    assert_eq!(
        frac_exceeding(&committed, &rendered, 11),
        0.0,
        "a toothed live preview visibly differs from the stroke rendered whole \
         ({:.4}% of px over 2)",
        frac_exceeding(&committed, &rendered, 2) * 100.0,
    );
}

/// The same, on the **stamp loop** — which is the path the shipped presets take
/// (`Hard Round` lifts and deposits), and the one where a live tail carries state:
/// the tool reservoir is threaded from the frozen head into the tail, while the
/// whole render runs the stroke from an empty tool in one go.
#[test]
fn a_toothed_smear_previews_as_it_commits() {
    let Some(mut engine) = rough_engine() else {
        return;
    };
    // Something to smear, so the tool has a load to carry across the cut.
    stroke_with(
        &mut engine,
        BrushParams {
            drain: 0.0,
            modulation: Default::default(),
            ..brush(RED, 60.0)
        },
        &[Vec2::new(-140.0, 0.0), Vec2::new(0.0, 0.0)],
    );
    let smear = BrushParams {
        tooth_give: 0.5,
        tooth_softness: BAND,
        dynamics: BrushDynamics {
            flow: 0.6,
            lift: 0.6,
            deposit: 0.95,
            ..Default::default()
        },
        ..BrushParams {
            drain: 0.0,
            shape: BrushShape::Round { hardness: 0.9 },
            modulation: Default::default(),
            ..brush(RED, 26.0)
        }
    };
    engine.process(stark_engine::command::ViewCommand::SetBrush(smear));
    let path: Vec<Vec2> = (0..100)
        .map(|i| {
            let t = i as f32 * 0.06;
            Vec2::new(t * 24.0 - 130.0, (t * 1.3).sin() * 26.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(stark_engine::command::GestureCommand::Start {
        tool: stark_engine::command::Tool::Brush,
        sample: stark_engine::command::InputSample::at(*it.next().unwrap()),
        tolerance: stark_engine::path::DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for &p in it {
        engine.process(stark_engine::command::GestureCommand::To {
            sample: stark_engine::command::InputSample::at(p),
        });
    }
    engine.process(stark_engine::command::GestureCommand::End);
    assert_eq!(
        engine.strokes_reused(),
        2,
        "the commit did not take the preview"
    );
    let committed = engine.render_to_image();
    let rendered = whole_render(&mut engine);

    assert_eq!(
        frac_exceeding(&committed, &rendered, 11),
        0.0,
        "a toothed smear previews differently from the stroke rendered whole \
         ({:.4}% of px over 2)",
        frac_exceeding(&committed, &rendered, 2) * 100.0,
    );
}
