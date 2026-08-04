//! The deposition tooth (§6.4): the canvas's own ground gating how much of the
//! brush's paint lands.
//!
//! The interesting claim is not "the mark gets patchy" — dither would do that. It is
//! that the patches are a **level set of the ground**: turn the tooth up and the
//! texels that still take paint are the ones that already took it, because both are
//! reading the same height field through a rising threshold (the bearing-area curve,
//! `paint_common.wesl`). That is what makes successive strokes register with each
//! other instead of stacking independent noise, and it is what
//! [`turning_the_tooth_up_takes_a_level_set_away`] measures.
//!
//! Everything here paints on **Gesso**: an irregular ground, whose bearing curve is
//! a smooth spread rather than the handful of discrete levels a regular weave gives,
//! so a level-set claim about it says something.

mod common;

use common::*;
use stark_core::command::DocCommand;
use stark_core::document::{BrushDynamics, BrushParams, BrushShape};
use stark_core::geom::Vec2;
use stark_core::gpu::SurfaceId;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A brush with a given tooth, and nothing else that varies along a stroke: no
/// drain, no taper, and — deliberately — **no size mapping**, so every stroke here
/// covers exactly the same ground and only the gate differs.
fn toothed(tooth: f32) -> BrushParams {
    BrushParams {
        drain: 0.0,
        tooth,
        shape: BrushShape::Round { hardness: 0.9 },
        modulation: Default::default(),
        dynamics: BrushDynamics {
            // Low flow: the tooth decides how much of the parcel lands, and at a flow
            // that saturates the slab law every texel reads as solid whatever fraction
            // it received. A dry-brush mark is the case this axis is *for*.
            add: 0.35,
            ..Default::default()
        },
        ..brush(RED, 26.0)
    }
}

/// The stroke every test here draws: a straight run across the middle, long enough
/// to cross several of the ground's features.
fn run() -> [Vec2; 2] {
    [Vec2::new(-140.0, 0.0), Vec2::new(140.0, 0.0)]
}

/// An engine on the gesso ground, with the height map actually registered — without
/// the bytes a surface falls back to `Flat`, whose relief is 0, and every test here
/// would silently be testing nothing.
fn gesso_engine() -> Option<stark_core::Engine> {
    let mut engine = engine_or_skip()?;
    engine.register_surface(SurfaceId::Gesso, stark_testdata::assets::gesso());
    engine.process(DocCommand::SetSurface(SurfaceId::Gesso));
    Some(engine)
}

/// Which texels the stroke moved, and by how much in total.
fn mark(before: &stark_core::RgbaImage, after: &stark_core::RgbaImage) -> (Vec<bool>, f64) {
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
fn one_mark(engine: &mut stark_core::Engine, b: BrushParams) -> (Vec<bool>, f64) {
    let before = engine.render_to_image();
    stroke_with(engine, b, &run());
    let out = mark(&before, &engine.render_to_image());
    engine.process(DocCommand::Undo);
    out
}

/// `tooth = 0` leaves the mark **solid on a ground that would otherwise break it**:
/// every texel a toothed setting reaches is reached here too, and many it does not.
///
/// The stronger claim — that the deposit at `tooth = 0` is the float it always was,
/// to the bit — is not tested here because there is nothing in-process to compare it
/// against. What carries it is the **golden suite**, every one of which paints at
/// `tooth = 0` and none of which moved when the axis landed. That is the same
/// evidence §6.4 cites for the original tooth having been inert, read the other way
/// round.
#[test]
fn no_tooth_leaves_a_ground_that_has_tooth_alone() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    let (solid, solid_ink) = one_mark(&mut engine, toothed(0.0));
    let (broken, _) = one_mark(&mut engine, toothed(0.55));

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
/// structural, not a check: `Surface::relief` is 0 on `Flat`, which zeroes the uv
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
    stroke_with(&mut engine, toothed(0.0), &run());
    let smooth = engine.render_to_image();
    engine.process(DocCommand::Undo);
    stroke_with(&mut engine, toothed(0.9), &run());
    let toothy = engine.render_to_image();

    assert_eq!(
        diff_fraction(&smooth, &toothy),
        (0.0, 0),
        "on a flat ground the tooth has nothing to bite and must change no pixel"
    );
    assert!(mark(&before, &smooth).1 > 0.0, "the test drew nothing");
}

/// The headline: a ground with tooth breaks the mark up, and turning the knob up
/// takes more of it away.
#[test]
fn a_ground_with_tooth_breaks_the_mark_up() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    let (solid, solid_ink) = one_mark(&mut engine, toothed(0.0));
    let (broken, broken_ink) = one_mark(&mut engine, toothed(0.55));

    let solid_n = solid.iter().filter(|h| **h).count();
    let broken_n = broken.iter().filter(|h| **h).count();
    assert!(
        solid_n > 0,
        "the un-toothed stroke covered nothing — the ground is not registered"
    );
    assert!(
        broken_n < solid_n,
        "a toothed brush should skip part of the ground: {broken_n} of {solid_n} texels"
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

/// **The bearing-curve claim.** The tooth is a threshold on one height field, so the
/// texels a deeper threshold keeps are a *subset* of the ones a shallower one kept —
/// the marks are nested level sets, not independent noise.
///
/// This is the property that separates a ground from a dither, and it is the reason
/// overpainting registers: two strokes at the same tooth catch on the same peaks, so
/// the second one lands *on* the first rather than filling its gaps.
///
/// The tolerance is for the soft transition band (`TOOTH_SOFTNESS`) and for the 12-
/// level threshold `mark` reads coverage at: a texel sitting inside the band at both
/// settings can land either side of that line. What cannot survive by accident is
/// 95% agreement across thousands of texels.
#[test]
fn turning_the_tooth_up_takes_a_level_set_away() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    let (shallow, _) = one_mark(&mut engine, toothed(0.35));
    let (deep, _) = one_mark(&mut engine, toothed(0.6));

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
         the two marks are not level sets of one ground"
    );
}

/// The tooth reaches the stamp loop too, and through the *same* function.
///
/// Worth its own test because the two paths compute the gate in different shaders
/// from different bindings, and the failure mode is the one §6.2 has been bitten by
/// before: a brush whose behaviour changes because some unrelated axis moved it onto
/// the other path. Nudging `lift` off zero must not change what the ground does.
#[test]
fn the_tooth_reads_the_same_on_both_render_paths() {
    let Some(mut engine) = gesso_engine() else {
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
    let (a, ink_a) = one_mark(&mut engine, swept);
    let (b, ink_b) = one_mark(&mut engine, looped);

    let a_n = a.iter().filter(|h| **h).count();
    let disagree = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert!(a_n > 200, "too little mark to compare: {a_n}");
    assert!(
        (disagree as f64) < 0.1 * a_n as f64,
        "the two paths disagree about the ground on {disagree} of {a_n} texels"
    );
    let ratio = ink_b / ink_a;
    assert!(
        (0.8..1.25).contains(&ratio),
        "and about how much they laid through it: {ink_b} vs {ink_a}"
    );
}

/// The ground is the **document's**, so a stroke deposits through the tooth that was
/// in force where it sits in the log — not through whatever the canvas was switched
/// to afterwards (§6.4).
///
/// Asserted through a save/load round trip, because that is the only thing that
/// actually re-runs the earlier strokes: a live `SetSurface` leaves their tiles
/// alone, so it could not tell a correct engine from one holding a single "current
/// surface" the whole log renders against. On load the log is replayed in order onto
/// an empty document, and a stroke laid before the switch has to come back with its
/// tooth and one laid after it without.
///
/// This is what the surface registry moving onto the apply context buys, and why
/// `SetSurface` was a logged action from the start rather than a view setting — §6.4
/// says so in as many words, several years of commits before there was anything to
/// read it.
#[test]
fn a_stroke_keeps_the_ground_it_was_painted_on() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    let before = engine.render_to_image();
    // One stroke on the toothed ground, then the canvas is switched to smooth and a
    // second is laid across the other half.
    stroke_with(
        &mut engine,
        toothed(0.55),
        &[Vec2::new(-140.0, -40.0), Vec2::new(140.0, -40.0)],
    );
    engine.process(DocCommand::SetSurface(SurfaceId::Flat));
    stroke_with(
        &mut engine,
        toothed(0.55),
        &[Vec2::new(-140.0, 40.0), Vec2::new(140.0, 40.0)],
    );
    let live = engine.render_to_image();
    assert!(mark(&before, &live).1 > 0.0, "the test drew nothing");

    let bytes = engine.save_bytes().expect("serialize");
    let Some(mut loaded) = engine_or_skip() else {
        return;
    };
    loaded.register_surface(SurfaceId::Gesso, stark_testdata::assets::gesso());
    loaded.load_bytes(&bytes).expect("deserialize + replay");

    assert_eq!(
        diff_fraction(&live, &loaded.render_to_image()),
        (0.0, 0),
        "the replay did not put each stroke back on the ground it was painted on",
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
/// whatever the ground does — the tooth decides *where* it lands, not how much there
/// was.
#[test]
fn a_toothed_transfer_delivers_the_whole_glob() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    // **Lighting flat.** The measure below is a stand-in for mass, and a lit render is
    // not one: the media pass takes its normals from the height gradient (§6.3), so
    // the same paint laid bumpily catches more light than the same paint laid smooth —
    // which is the whole visible point of the tooth, and here it is the contaminant.
    // At `height_strength = 0` and `surface_strength = 0` the pass is the reference
    // identity and the deviation from bare tracks visible alpha, which tracks mass.
    engine.process(stark_core::command::ViewCommand::SetMediaParams(
        stark_core::MediaParams {
            height_strength: 0.0,
            surface_strength: 0.0,
            ..Default::default()
        },
    ));
    // `add = 0`, no lift: the only paint in play is the pre-loaded glob, and the only
    // place it can go is the canvas. A long stroke at a brisk deposit rate, so the
    // tool is empty well before the end either way and the total delivered is the
    // whole charge rather than however far each got.
    let glob = |tooth: f32| BrushParams {
        tooth,
        dynamics: BrushDynamics {
            add: 0.0,
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
    stroke_with(&mut engine, glob(0.0), &path);
    let (plain, plain_ink) = far_share(&bare, &engine.render_to_image());
    engine.process(DocCommand::Undo);
    stroke_with(&mut engine, glob(0.6), &path);
    let (toothed, toothed_ink) = far_share(&bare, &engine.render_to_image());

    assert!(
        plain_ink > 0.0 && toothed_ink > 0.0,
        "a glob laid nothing at all"
    );
    // The un-toothed tool empties inside the first third and never reaches here.
    assert!(
        plain < 0.01,
        "the un-toothed glob was still going past the two-thirds mark ({plain:.3}) —          it is not emptying fast enough for this test to say anything"
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
fn far_share(before: &stark_core::RgbaImage, after: &stark_core::RgbaImage) -> (f64, f64) {
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

/// The bearing fraction is the mean of the gate over the ground, so it has to *be*
/// that: monotone down in the tooth, exactly 1 where there is nothing to bite.
///
/// It is a CPU mirror of a GPU function (`Surface::bearing` against
/// `paint_common.wesl::tooth_gate`), and a mirror is only as good as what notices it
/// drifting. What notices is the conservation test above; this one pins the shape so
/// a failure there points at the right half.
#[test]
fn the_bearing_fraction_tracks_the_ground() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let flat = engine.surface_bearing(SurfaceId::Flat, 0.5);
    assert_eq!(flat, 1.0, "a smooth ground is full contact at any tooth");

    engine.register_surface(SurfaceId::Gesso, stark_testdata::assets::gesso());
    let mut at = |t| engine.surface_bearing(SurfaceId::Gesso, t);
    assert_eq!(at(0.0), 1.0, "no tooth is full contact, exactly");
    let (a, b, c) = (at(0.25), at(0.5), at(0.75));
    assert!(
        1.0 > a && a > b && b > c && c > 0.0,
        "contact must fall as the tip reaches for higher ground: {a} {b} {c}"
    );
}

/// **`preview == committed` with a tooth** (§1.3). The live path renders a frozen
/// head and then a tail over it; the commit renders the whole stroke in one range.
/// A gate that is a pure function of canvas position factors out of the sum over
/// segments, so the two must agree — this is where that claim is checked, since the
/// suite's existing split-preview test runs on a ground with no relief.
#[test]
fn a_toothed_live_preview_matches_the_commit() {
    let Some(mut engine) = gesso_engine() else {
        return;
    };
    engine.process(stark_core::command::ViewCommand::SetBrush(toothed(0.55)));
    let path: Vec<Vec2> = (0..120)
        .map(|i| {
            let t = i as f32 * 0.05;
            Vec2::new(t * 22.0 - 60.0, (t * 1.1).sin() * 30.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(stark_core::command::GestureCommand::Start {
        tool: stark_core::document::Tool::Brush,
        sample: stark_core::command::InputSample::at(*it.next().unwrap()),
        tolerance: stark_core::path::DEFAULT_TOLERANCE,
    });
    for &p in it {
        engine.process(stark_core::command::GestureCommand::To {
            sample: stark_core::command::InputSample::at(p),
        });
    }
    let preview = engine.render_to_image();
    engine.process(stark_core::command::GestureCommand::End);
    let committed = engine.render_to_image();

    assert_eq!(
        frac_exceeding(&preview, &committed, 11),
        0.0,
        "a toothed live preview visibly differs from its commit ({:.4}% of px over 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}

/// The same, on the **stamp loop** — which is the path the shipped presets take
/// (`Hard Round` lifts and deposits), and the one where a live tail carries state:
/// the tool reservoir is threaded from the frozen head into the tail, while the
/// commit runs the whole stroke from an empty tool in one go.
#[test]
fn a_toothed_smear_previews_as_it_commits() {
    let Some(mut engine) = gesso_engine() else {
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
        tooth: 0.5,
        dynamics: BrushDynamics {
            add: 0.6,
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
    engine.process(stark_core::command::ViewCommand::SetBrush(smear));
    let path: Vec<Vec2> = (0..100)
        .map(|i| {
            let t = i as f32 * 0.06;
            Vec2::new(t * 24.0 - 130.0, (t * 1.3).sin() * 26.0)
        })
        .collect();
    let mut it = path.iter();
    engine.process(stark_core::command::GestureCommand::Start {
        tool: stark_core::document::Tool::Brush,
        sample: stark_core::command::InputSample::at(*it.next().unwrap()),
        tolerance: stark_core::path::DEFAULT_TOLERANCE,
    });
    for &p in it {
        engine.process(stark_core::command::GestureCommand::To {
            sample: stark_core::command::InputSample::at(p),
        });
    }
    let preview = engine.render_to_image();
    engine.process(stark_core::command::GestureCommand::End);
    let committed = engine.render_to_image();

    assert_eq!(
        frac_exceeding(&preview, &committed, 11),
        0.0,
        "a toothed smear previews differently from its commit ({:.4}% of px over 2)",
        frac_exceeding(&preview, &committed, 2) * 100.0,
    );
}
