//! Filter layers — the third kind of layer (§21).
//!
//! A filter holds no tiles and no region: it is a function of what its own stack
//! has already composited. So these cover the two halves of that sentence — that it
//! *is* a layer (it takes opacity and visibility, it undoes, it takes no paint), and
//! that "its own stack" is exactly what it reaches (a filter in a group leaves the
//! rest of the document alone, which is what makes carrying a filter onto a layer
//! the whole of "filter just this one").
//!
//! Several of these are about a filter doing **nothing**, and that is deliberate. A
//! pass that runs over every texel of the frame has no coverage to hide behind, so
//! the cases where it must be the exact identity — neutral, hidden, nothing beneath
//! it — are the ones where a mistake is a whole-picture change with nothing on
//! screen to say where it came from. Clipping a *point* filter joins them (§21.4.1):
//! there the identity is the claim itself, since a clip can only take away what a
//! filter said about coverage and a point filter said nothing.

mod common;

use common::*;
use stark_engine::command::{DocCommand, PeerCommand, ViewCommand};
use stark_engine::document::{ChromaticAberration, ColorAdjust, Filter, LayerId, Place};
use stark_engine::{Engine, RgbaImage};
use stark_model::geom::Vec2;
use stark_model::gradient::{Gradient, GradientStop};

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const STROKE: &[Vec2] = &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];

/// The neutral filter — what `AddFilter` lands.
const NEUTRAL: Filter = Filter::Color(ColorAdjust::NEUTRAL);

/// Every color drained away. The sharpest filter to test with: it is visible on
/// any painting, it is checkable without knowing the render's exact numbers (the
/// three channels simply have to agree), and it is the one setting whose *correct*
/// answer differs from the naive one — dropping Oklab chroma keeps lightness where
/// it was, which a weighted RGB average does not.
const GREY: Filter = Filter::Color(ColorAdjust {
    saturation: 0.0,
    ..ColorAdjust::NEUTRAL
});

/// The color drained away and one put back: a greyscale **toned** to a single
/// color, which is the tint's own defining claim (§21.5). Asymmetric on purpose —
/// `a` and `b` different, and of different signs — so the direction it produces
/// cannot be reached by swapping the two axes or by flipping either.
const TONED: Filter = Filter::Color(ColorAdjust {
    saturation: 0.0,
    tint: [0.10, -0.05],
    ..ColorAdjust::NEUTRAL
});

/// A hard chromatic dispersion along the suite's stroke: wide enough that the
/// fringes span several rendered pixels, aimed down the axis the stroke runs
/// (§21.10), so a scan of the stroke's own row crosses both of them.
const FRINGE: Filter = Filter::Chromatic(ChromaticAberration {
    spread: 12.0,
    angle: 0.0,
});

/// How far the filter moved each pixel's red-minus-blue separation, scanned along
/// the stroke's row: `(min, max)` of `(R−B)_after − (R−B)_before` across it.
///
/// This is the observable that tells a *spectrum pulled apart* from a picture
/// merely smeared: dispersion shifts the red and blue ends of the picture opposite
/// ways along the axis, so one flank of the stroke gains red-over-blue and the
/// other loses it — the swing must reach well clear of zero **in both signs**. A
/// blur, a uniform shift, or a tint moves the separation everywhere the same way.
/// Scanning for the extremes rather than sampling named pixels keeps the claim
/// about the physics rather than about where exactly the brush's soft edge fell.
fn separation_swing(before: &RgbaImage, after: &RgbaImage) -> (i32, i32) {
    let y = before.height / 2;
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for x in 0..before.width {
        let (b, a) = (before.pixel(x, y), after.pixel(x, y));
        let d = (a[0] as i32 - a[2] as i32) - (b[0] as i32 - b[2] as i32);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    (lo, hi)
}

/// Whether a pixel is achromatic — the three channels within a level or two of one
/// another, which is what a saturation of zero has to produce.
fn is_grey(c: [u8; 4]) -> bool {
    let (lo, hi) = (
        c[0].min(c[1]).min(c[2]) as i32,
        c[0].max(c[1]).max(c[2]) as i32,
    );
    hi - lo <= 3
}

/// A rendered pixel's Oklab chroma, as the `(a, b)` pair the filter's own knobs
/// move — the space the claim is made in, so the assertion is about the adjustment
/// rather than about how sRGB happens to encode it.
fn chroma_ab(c: [u8; 4]) -> [f32; 2] {
    let lin = |i: usize| stark_model::color::srgb_to_linear(c[i] as f32 / 255.0);
    let lab = stark_model::color::linear_srgb_to_oklab([lin(0), lin(1), lin(2)]);
    [lab[1], lab[2]]
}

/// A ramp from positioned sRGB stops — the test's shorthand for what a trace
/// captures (§22.2).
fn ramp(stops: &[(f32, [f32; 3])]) -> Gradient {
    Gradient::new(
        stops
            .iter()
            .map(|&(t, color)| GradientStop { t, color })
            .collect(),
    )
    .expect("a valid test ramp")
}

/// Add a filter into `carrier`'s stack (the document's own when `None`) and hand
/// back its id. The engine mints the id, so the new filter is the layer the
/// projection *gained* — found by diffing against the ids that existed before,
/// because "the topmost filter" is somebody else's filter the moment a document
/// holds two (the trap the AddFilterButton fell into once).
fn add_filter(engine: &mut Engine, carrier: Option<LayerId>, filter: Filter) -> LayerId {
    let before: Vec<LayerId> = engine.observe().layers.iter().map(|l| l.id).collect();
    engine.process(DocCommand::AddFilter {
        carrier,
        above: None,
        filter,
    });
    engine
        .observe()
        .layers
        .iter()
        .find(|l| l.filter.is_some() && !before.contains(&l.id))
        .map(|l| l.id)
        .expect("the filter landed")
}

/// A painting with a red stroke across the middle, and nothing else.
fn painted() -> Option<Engine> {
    let mut engine = engine_or_skip()?;
    paint(&mut engine, RED, 22.0, STROKE);
    Some(engine)
}

/// The core claim: a filter rewrites the paint composited beneath it, without being
/// paint itself.
#[test]
fn a_filter_recolors_what_is_beneath_it() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    assert!(
        red_dominant(center(&before)),
        "the stroke should be red before the filter: {:?}",
        center(&before),
    );

    add_filter(&mut engine, None, GREY);
    let after = engine.render_to_image();
    assert!(
        is_grey(center(&after)),
        "a saturation of zero should leave the stroke achromatic: {:?}",
        center(&after),
    );
}

/// **Chroma, and nothing else.** Draining the color leaves *perceived lightness*
/// where it was, which is the whole reason the adjustment runs in Oklab.
///
/// Measured as Oklab `L`, and the distinction is the test rather than a detail of
/// it: a saturated red and the grey a correct desaturation gives it have very
/// different relative luminance — the grey is far brighter in `Y` — so an assertion
/// on luminance would fail on the right answer and pass on `dot(rgb, luma)`, which
/// is exactly the naive desaturation this is here to rule out.
#[test]
fn desaturating_keeps_the_lightness_it_found() {
    let Some(mut engine) = painted() else { return };
    let before = center(&engine.render_to_image());
    add_filter(&mut engine, None, GREY);
    let after = center(&engine.render_to_image());

    let lightness = |c: [u8; 4]| {
        let lin = |i: usize| stark_model::color::srgb_to_linear(c[i] as f32 / 255.0);
        stark_model::color::linear_srgb_to_oklab([lin(0), lin(1), lin(2)])[0]
    };
    let (was, now) = (lightness(before), lightness(after));
    // Loose, because the trip out to light and back through the media pass's tonemap
    // is not an identity. What it rules out is the ~0.1 slide in `L` a
    // luminance-weighted average produces on a saturated red, which is well outside.
    assert!(
        (was - now).abs() < 0.04,
        "desaturation moved Oklab L from {was:.3} to {now:.3} \
         (before {before:?}, after {after:?})",
    );
}

/// **The tint is the color a grey becomes.** That sentence is the whole definition
/// of the knob (§21.5), and it is a claim about *where in the adjustment the offset
/// lands*: last, after the rotation and the gain, so that an achromatic texel — which
/// arrives at the origin of the `(a, b)` plane and is left there by both — comes out
/// holding the tint itself.
///
/// Worth a render rather than a unit test on the struct, because the ordering the
/// claim rests on exists only in the shader, and the pair of knobs that would break
/// it are exactly the two the panel draws around it: hue and saturation are a
/// rotation and a scale, and *either* applied after the tint would turn the color
/// under the pointer into some other color. Checked as a direction in Oklab and not
/// as an RGB triple: the media pass's tonemap moves the magnitude and must be allowed
/// to, while the hue it lands on is the filter's alone.
#[test]
fn a_tint_is_the_color_a_grey_becomes() {
    let Some(mut engine) = painted() else { return };
    add_filter(&mut engine, None, GREY);
    let grey = center(&engine.render_to_image());
    assert!(is_grey(grey), "the setup should be achromatic: {grey:?}");

    let toned = add_filter(&mut engine, None, TONED);
    let got = center(&engine.render_to_image());
    assert!(
        !is_grey(got),
        "a tint over a greyscale should put a color back: {got:?}",
    );

    let Filter::Color(c) = TONED else {
        unreachable!("TONED is a color filter")
    };
    let ab = chroma_ab(got);
    let want = c.tint;
    let mag = |v: [f32; 2]| v[0].hypot(v[1]);
    // The direction, as the cosine between the two — one number that catches an axis
    // swap, a sign flip on either axis, and a rotation applied on the wrong side of
    // the offset, none of which a per-component tolerance would.
    let cos = (ab[0] * want[0] + ab[1] * want[1]) / (mag(ab) * mag(want)).max(1e-6);
    assert!(
        cos > 0.9,
        "the toned grey points at {ab:?}, not at the tint {want:?} (cos {cos:.3})",
    );
    // And it arrived at roughly the strength asked for — loose in both directions,
    // because the trip out through the media pass is not an identity, but tight
    // enough to rule out a tint that reached the frame scaled by the saturation gain
    // (which is 0 here) or doubled by being applied twice.
    let (got_c, want_c) = (mag(ab), mag(want));
    assert!(
        got_c > want_c * 0.4 && got_c < want_c * 1.6,
        "the toned grey has chroma {got_c:.3}, the tint {want_c:.3}",
    );

    // And it is the filter's, not the painting's: dropped, the grey comes back.
    engine.process(DocCommand::RemoveLayer(toned));
    assert!(
        is_grey(center(&engine.render_to_image())),
        "removing the tint should leave the greyscale it found",
    );
}

/// **A neutral filter is the exact identity.** This is what makes adding one a step
/// you take before deciding what it does rather than an edit in itself — and it is
/// checked to the byte, because the draw list drops a neutral filter entirely
/// (§21.3) and anything less than bit-equality would mean it had not.
#[test]
fn a_neutral_filter_changes_no_pixel() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    add_filter(&mut engine, None, NEUTRAL);
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "a neutral filter is not the identity: {:?}",
        diff_fraction(&before, &after),
    );
}

/// Hiding a filter, or fading it to nothing, puts the picture back exactly. The two
/// are one test because they are one claim about presentation: a filter's opacity is
/// its strength (§21.4), and zero of it is the same nothing as hidden.
#[test]
fn a_hidden_or_faded_filter_changes_no_pixel() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    let id = add_filter(&mut engine, None, GREY);
    assert!(
        !images_match(&before, &engine.render_to_image(), 0),
        "the filter has to be doing something for this test to mean anything",
    );

    engine.process(DocCommand::SetLayerVisible(id, false));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a hidden filter still changed the picture",
    );

    engine.process(DocCommand::SetLayerVisible(id, true));
    engine.process(DocCommand::SetLayerOpacity(id, 0.0));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a filter at zero strength still changed the picture",
    );
}

/// **A filter reaches exactly as far as its own stack** (§21.2) — which is what
/// makes "filter just this layer" the single gesture of carrying the filter onto it,
/// with no scoping mode to invent.
///
/// Two strokes on two layers, the filter carried onto the lower one: the layer it is
/// in must change and the other must not.
#[test]
fn a_carried_filter_reaches_only_its_own_group() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Bottom layer: a stroke on the left. Then a second layer with a stroke on the
    // right, well clear of it.
    let lower = engine.observe().active_layer;
    paint(
        &mut engine,
        RED,
        22.0,
        &[Vec2::new(-90.0, 0.0), Vec2::new(-40.0, 0.0)],
    );
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    paint(
        &mut engine,
        RED,
        22.0,
        &[Vec2::new(40.0, 0.0), Vec2::new(90.0, 0.0)],
    );
    let before = engine.render_to_image();

    // Carried onto the lower layer, so the two become a group and the filter sees
    // only what that group has composited.
    add_filter(&mut engine, Some(lower), GREY);
    let after = engine.render_to_image();

    let (x_lo, x_hi) = (before.width / 4, before.width * 3 / 4);
    let y = before.height / 2;
    assert!(
        red_dominant(before.pixel(x_lo, y)) && red_dominant(before.pixel(x_hi, y)),
        "both strokes should start red",
    );
    assert!(
        is_grey(after.pixel(x_lo, y)),
        "the carried filter should have drained the layer it is in: {:?}",
        after.pixel(x_lo, y),
    );
    assert!(
        red_dominant(after.pixel(x_hi, y)),
        "…and left the layer above the group alone: {:?}",
        after.pixel(x_hi, y),
    );
}

/// A filter with nothing beneath it **in its own stack** is the identity: at the
/// foot of the document there is nothing composited yet, and a stack whose lower
/// layers are all hidden reaches the filter with an empty accumulator (§21.2).
#[test]
fn a_filter_with_nothing_beneath_it_changes_no_pixel() {
    let Some(mut engine) = painted() else { return };
    let paint_layer = engine.observe().active_layer;
    let before = engine.render_to_image();

    // At the foot of the document.
    let id = add_filter(&mut engine, None, GREY);
    engine.process(DocCommand::MoveLayer {
        id,
        carrier: None,
        at: Place::Bottom,
    });
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a filter under everything still changed the picture",
    );

    // …and above nothing but a hidden layer, which the draw list culls exactly as
    // it culls an empty one.
    engine.process(DocCommand::MoveLayer {
        id,
        carrier: None,
        at: Place::Top,
    });
    engine.process(DocCommand::SetLayerVisible(paint_layer, false));
    let hidden = engine.render_to_image();
    engine.process(DocCommand::SetLayerVisible(paint_layer, true));
    let shown = engine.render_to_image();
    assert!(
        !images_match(&hidden, &shown, 0),
        "hiding the only painted layer has to change the picture",
    );
    // What matters: with the paint hidden, the grade must not spring onto the
    // bare canvas — the canvas with everything hidden reads the same whether the
    // grey filter is above it or removed.
    engine.process(DocCommand::SetLayerVisible(paint_layer, false));
    let with_filter = engine.render_to_image();
    engine.process(DocCommand::RemoveLayer(id));
    assert!(
        images_match(&with_filter, &engine.render_to_image(), 0),
        "a filter above nothing but a hidden layer still changed the picture",
    );
}

/// A filter never carries (§21.2): dropping a layer onto one — and adding a layer
/// into one — is refused **by the engine**, deterministically, so no document can
/// hold the arrangement in which a filter's reach would be an empty group. The
/// refusal is state's, not a frontend rule, so replay and peers agree.
#[test]
fn a_filter_refuses_carried_layers() {
    let Some(mut engine) = painted() else { return };
    let paint_layer = engine.observe().active_layer;
    let before = engine.render_to_image();
    let id = add_filter(&mut engine, None, NEUTRAL);

    // A move onto the filter is declined outright…
    engine.process(DocCommand::MoveLayer {
        id: paint_layer,
        carrier: Some(id),
        at: Place::Top,
    });
    let obs = engine.observe();
    let carrier_of = |target: LayerId| {
        obs.layers
            .iter()
            .find(|l| l.id == target)
            .expect("layer projected")
            .carrier
    };
    assert_eq!(
        carrier_of(paint_layer),
        None,
        "a filter accepted a carried layer",
    );
    drop(obs);

    // …and so is adding a new layer into the filter's (nonexistent) group.
    engine.process(DocCommand::AddLayer {
        carrier: Some(id),
        above: None,
    });
    assert!(
        engine
            .observe()
            .layers
            .iter()
            .all(|l| l.carrier != Some(id)),
        "AddLayer attached a child to a filter",
    );

    // The declined gestures changed nothing on screen either.
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a refused carry still changed the picture",
    );
}

/// The bar's "nothing below it" note reads `has_underlay`, and that answer has to
/// agree with the renderer (§21.2): a filter carried onto a painted layer *is*
/// reaching that layer's paint — the "filter just this layer" gesture — and a
/// filter above nothing but a hidden layer is reaching nothing, whatever the row
/// order says.
#[test]
fn the_reach_note_agrees_with_the_renderer() {
    let Some(mut engine) = painted() else { return };
    let paint_layer = engine.observe().active_layer;

    let underlay_of = |engine: &Engine, id: LayerId| {
        engine
            .observe()
            .layers
            .iter()
            .find(|l| l.id == id)
            .expect("filter projected")
            .has_underlay
    };

    // Carried onto a painted layer: the carrier's own content is beneath it.
    let carried = add_filter(&mut engine, Some(paint_layer), GREY);
    assert!(
        underlay_of(&engine, carried),
        "a filter carried onto painted content reaches it (its bar must not say \
         'nothing below it' while the canvas is visibly graded)",
    );
    engine.process(DocCommand::RemoveLayer(carried));

    // Above the paint in the root stack: reaches it. Hide the paint: reaches
    // nothing, exactly as the draw list culls it.
    let top = add_filter(&mut engine, None, GREY);
    assert!(
        underlay_of(&engine, top),
        "paint beneath, in the same stack"
    );
    engine.process(DocCommand::SetLayerVisible(paint_layer, false));
    assert!(
        !underlay_of(&engine, top),
        "a hidden layer fills nothing beneath a filter — the sliders would change \
         no pixel, and the bar has to say why",
    );
}

/// A filter takes no paint, refused **by the engine** rather than by a frontend
/// rule, so a replayed or remote log agrees (§21.4). It may still be selected, like
/// a matte — one selection concept, not two.
#[test]
fn a_filter_can_be_selected_but_takes_no_paint() {
    let Some(mut engine) = painted() else { return };
    let id = add_filter(&mut engine, None, NEUTRAL);
    engine.process(PeerCommand::SetActiveLayer(id));

    let obs = engine.observe();
    assert_eq!(obs.active_layer, id, "a filter can be the selected layer");
    let info = obs
        .layers
        .iter()
        .find(|l| l.id == id)
        .expect("the filter is projected");
    assert!(!info.is_paintable(), "a filter takes no paint");
    assert_eq!(info.filter, Some(NEUTRAL), "the filter itself is projected");

    let before = engine.render_to_image();
    paint(&mut engine, [0.1, 0.1, 0.9, 1.0], 30.0, STROKE);
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a stroke aimed at a filter layer laid paint",
    );
}

/// Retuning a filter undoes, and undoing back past the add removes it — the
/// ordinary layer guarantees, which a filter gets for free by being a layer.
#[test]
fn a_filter_undoes() {
    let Some(mut engine) = painted() else { return };
    let plain = engine.render_to_image();
    let id = add_filter(&mut engine, None, NEUTRAL);
    engine.process(DocCommand::SetFilter(id, GREY));
    let grey = engine.render_to_image();
    assert!(!images_match(&plain, &grey, 0), "the filter did nothing");

    engine.process(DocCommand::Undo);
    assert!(
        images_match(&plain, &engine.render_to_image(), 0),
        "undoing the adjustment left it applied",
    );
    engine.process(DocCommand::Redo);
    assert!(
        images_match(&grey, &engine.render_to_image(), 0),
        "redo did not put the adjustment back",
    );

    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Undo);
    assert!(
        !engine.observe().layers.iter().any(|l| l.id == id),
        "undoing past the add left the filter layer behind",
    );
}

/// A slider drag previews per pointer move and logs **once**, on release — the same
/// bargain the frame drag and the opacity slider make (§21.6), and the one this
/// feature could least do without, since a color adjustment is judged by looking.
#[test]
fn dragging_a_filter_previews_without_logging() {
    let Some(mut engine) = painted() else { return };
    let id = add_filter(&mut engine, None, NEUTRAL);
    let steps = engine.scrub_range().expect("solo timeline").1;

    let mut shown = None;
    for i in 1..=8 {
        let filter = Filter::Color(ColorAdjust {
            saturation: 1.0 - i as f32 / 8.0,
            ..ColorAdjust::NEUTRAL
        });
        engine.process(ViewCommand::PreviewFilter(Some((id, filter.clone()))));
        shown = Some(filter);
    }
    assert_eq!(
        engine.scrub_range().expect("solo timeline").1,
        steps,
        "previewing logged an action",
    );
    // The projection reports the *previewed* filter, which is what keeps the
    // sliders and the canvas agreeing under the pointer.
    let previewed = engine.render_to_image();
    let obs = engine.observe();
    let info = obs.layers.iter().find(|l| l.id == id).expect("projected");
    assert_eq!(info.filter, shown, "observe() should report the preview");
    drop(obs);

    engine.process(DocCommand::SetFilter(id, shown.expect("eight samples")));
    assert_eq!(
        engine.scrub_range().expect("solo timeline").1,
        steps + 1,
        "the whole drag should cost exactly one action",
    );
    assert!(
        images_match(&previewed, &engine.render_to_image(), 0),
        "the commit rendered something other than what the preview showed",
    );
}

/// Chromatic aberration **parts the spectrum, both ways** (§21.10). Across the
/// stroke, the separation the filter adds between the red and blue channels must
/// swing to opposite signs on the two flanks — red spilling one way and blue the
/// other is what dispersion *is*, and it is exactly what the three-shifted-copies
/// shortcut this filter refuses would also show, so the same check covers the
/// integral's ordering. See [`separation_swing`] for why this observable and not a
/// named pixel's hue.
#[test]
fn chromatic_aberration_parts_the_spectrum_both_ways() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    add_filter(&mut engine, None, FRINGE);
    let after = engine.render_to_image();
    assert!(
        !images_match(&before, &after, 0),
        "the dispersion has to change the picture for this test to mean anything",
    );

    let (lo, hi) = separation_swing(&before, &after);
    assert!(
        hi > 15 && lo < -15,
        "dispersion should push red-vs-blue separation both ways across the \
         stroke, got a swing of {lo}..{hi}",
    );
}

/// **Deep inside flat paint the gather is the identity** — the partition of unity
/// that §21.10 leans on: every channel's weights are normalized by their own sum,
/// so where all the taps land on the same paint the integral provably returns it.
/// The dispersion runs *along* the stroke, so every tap under the centre sits on
/// the stroke's spine — the flattest paint the suite can offer — while the stroke's
/// ends still prove the pass did something. A tolerance rather than bytes: the
/// identity is exact in the linear-light algebra, and the trip out to light and
/// back is not.
#[test]
fn chromatic_aberration_is_the_identity_deep_inside_flat_paint() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    add_filter(
        &mut engine,
        None,
        Filter::Chromatic(ChromaticAberration {
            spread: 8.0,
            angle: 0.0,
        }),
    );
    let after = engine.render_to_image();
    assert!(
        !images_match(&before, &after, 0),
        "the dispersion has to change the picture somewhere (the stroke's ends)",
    );

    let (b, a) = (center(&before), center(&after));
    let worst = b
        .iter()
        .zip(a.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap();
    assert!(
        worst <= 5,
        "flat paint under the gather moved by {worst} levels \
         (before {b:?}, after {a:?}) — the weights no longer sum to one",
    );
}

/// The chromatic filter in a **pigment** document (§21.10, §6.7): every tap decodes
/// through Mixbox's polynomial with its residual, and the summed light re-enters
/// through the inverse LUT once. The physics claim is the same as the Oklab test's;
/// what this covers is the per-tap leg no Oklab test touches.
#[cfg(feature = "mixbox")]
#[test]
fn chromatic_aberration_parts_the_spectrum_in_pigment_too() {
    let Some(mut engine) = engine_or_skip_with(stark_engine::colorspace::ColorSpaceId::Mixbox)
    else {
        return;
    };
    paint(&mut engine, RED, 22.0, STROKE);
    let before = engine.render_to_image();
    add_filter(&mut engine, None, FRINGE);
    let after = engine.render_to_image();
    assert!(
        !images_match(&before, &after, 0),
        "the dispersion has to change the pigment picture too",
    );
    let (lo, hi) = separation_swing(&before, &after);
    assert!(
        hi > 15 && lo < -15,
        "dispersion in pigment should part red and blue both ways across the \
         stroke, got a swing of {lo}..{hi}",
    );
}

/// **Clipping a point filter changes nothing, to the byte** (§21.4.1).
///
/// The claim is not "close enough": a clip is inert wherever the filter has no
/// opinion about coverage, and a point filter has none — it writes back the alpha it
/// read and copies the height across (§21.3.1). So the two renders are not two paths
/// that agree, they are the same path, and bytes are what says so. A tolerance here
/// would pass just as happily on a clipped branch that recomputed the texel some
/// other way, which is the thing this rules out.
///
/// Asserted on the **greyscale** rather than on a gentle grade for the reason
/// [`GREY`] is the suite's sharpest filter: a clip that suppressed the adjustment
/// where coverage is partial — the plausible wrong implementation — would show up on
/// the stroke's soft edge, which is most of the pixels this filter touches.
#[test]
fn clipping_a_point_filter_changes_nothing() {
    let Some(mut engine) = painted() else { return };
    let bare = engine.render_to_image();
    let id = add_filter(&mut engine, None, GREY);
    let open = engine.render_to_image();
    assert!(
        !images_match(&bare, &open, 0),
        "the filter has to be doing something for this test to mean anything",
    );

    engine.process(DocCommand::SetLayerClip(id, true));
    assert!(
        engine
            .observe()
            .layers
            .iter()
            .find(|l| l.id == id)
            .is_some_and(|l| l.clip),
        "state refused the clip, so the render below proves nothing",
    );
    assert!(
        images_match(&open, &engine.render_to_image(), 0),
        "clipping a point filter changed the picture",
    );
}

/// Whether `(x, y)` is canvas the painting never reached — **and is clear of the
/// paint's own edge by [`CLEAR`]**.
///
/// The bare test itself is exact: the substrate is a pure function of canvas position
/// (§6.4), so a pixel the painting left untouched is bit-identical to the same pixel
/// with no painting at all. The *margin* is what the rendered byte cannot say on its
/// own. A frame pixel is a downsample of the supersampled accumulator (§6.4), so one
/// that rounds to the substrate may still cover a whisker of paint — coverage of a
/// few thousandths, which a filter is entitled to recolor and which lands as a single
/// level. Measured, that band is one pixel wide at the rim of a soft stroke; the
/// margin is two, so what the caller gets back is a statement about **coverage**
/// rather than about rounding.
fn is_bare(painted: &RgbaImage, unpainted: &RgbaImage, x: u32, y: u32) -> bool {
    let (w, h) = (painted.width as i32, painted.height as i32);
    (-CLEAR..=CLEAR).all(|dy| {
        (-CLEAR..=CLEAR).all(|dx| {
            let (nx, ny) = (
                (x as i32 + dx).clamp(0, w - 1),
                (y as i32 + dy).clamp(0, h - 1),
            );
            painted.pixel(nx as u32, ny as u32) == unpainted.pixel(nx as u32, ny as u32)
        })
    })
}

/// How far [`is_bare`] holds a pixel clear of anything the painting touched.
const CLEAR: i32 = 2;

/// **A clipped gather stays inside the paint it filters** (§21.4.1, §21.10).
///
/// The chromatic filter is the one kind with an opinion about coverage: it carries
/// coverage and height along with the light it displaces, which is what lets a fringe
/// be seen past a stroke's edge — and is exactly what a clip refuses. Both halves are
/// asserted, because either alone is a different bug:
///
/// - unclipped, the fringe **does** reach bare canvas (else the test is vacuous);
/// - clipped, every texel the paint does not cover comes through **byte for byte**,
///   while the stroke itself is still filtered.
///
/// "Bare canvas" is read off the paint-free render rather than guessed at from a
/// color — see [`is_bare`], which is also where the one subtlety lives. That makes
/// the mask a fact about the render rather than a threshold to tune, and it is why
/// the comparison can be exact: where coverage is zero a clipped filter writes the
/// zero it read, so the accumulator, and then the frame, is unchanged.
#[test]
fn a_clipped_gather_stays_inside_the_paint() {
    let Some(mut empty) = engine_or_skip() else {
        return;
    };
    let unpainted = empty.render_to_image();
    let Some(mut engine) = painted() else { return };
    let bare = engine.render_to_image();

    let id = add_filter(&mut engine, None, FRINGE);
    let open = engine.render_to_image();
    engine.process(DocCommand::SetLayerClip(id, true));
    let shut = engine.render_to_image();

    // The two questions, asked pixel by pixel over the canvas the paint never
    // reached: did the unclipped fringe land here, and did the clipped one stay away?
    let (mut spilled, mut leaked) = (0u32, 0u32);
    for y in 0..bare.height {
        for x in 0..bare.width {
            if !is_bare(&bare, &unpainted, x, y) {
                continue;
            }
            spilled += u32::from(open.pixel(x, y) != bare.pixel(x, y));
            leaked += u32::from(shut.pixel(x, y) != bare.pixel(x, y));
        }
    }
    assert!(
        spilled > 0,
        "the unclipped fringe never left the paint, so the clip has nothing to bound",
    );
    assert_eq!(
        leaked, 0,
        "a clipped filter wrote {leaked} texels of bare canvas: coverage came out \
         other than it went in",
    );
    assert!(
        !images_match(&bare, &shut, 0),
        "the clipped filter changed nothing at all, so it is not the clip under test",
    );
}

/// A filter survives the round trip through the **file** (§8). Worth its own test
/// rather than being left to `save_load.rs`'s strokes: `AddFilter` and `SetFilter`
/// are the first actions to carry a `Filter`, postcard writes no field names and no
/// lengths, and a layout mistake there decodes into a different adjustment rather
/// than into an error.
#[test]
fn a_filter_survives_save_and_load() {
    let Some(mut engine) = painted() else { return };
    let id = add_filter(&mut engine, None, NEUTRAL);
    // Set through the command rather than added already-dialled, so both of the two
    // new actions are in the log this saves.
    engine.process(DocCommand::SetFilter(
        id,
        // Every field distinct, and the tint's two distinct from each other: postcard
        // writes no names, so a pair read in the wrong order — or read off the end of
        // the struct it was appended to — decodes as a different adjustment rather
        // than as an error.
        Filter::Color(ColorAdjust {
            exposure: 0.75,
            contrast: 1.4,
            saturation: 0.35,
            hue: 0.6,
            tint: [0.09, -0.04],
        }),
    ));
    // A second filter of the **other kind** rides the same log: `Chromatic` is an
    // appended enum variant (§8), and appended is exactly the layout mistake class
    // this test exists to catch — a variant misnumbered on either side decodes as
    // a *different filter* rather than as an error.
    let chroma = add_filter(
        &mut engine,
        None,
        Filter::Chromatic(ChromaticAberration::NEUTRAL),
    );
    engine.process(DocCommand::SetFilter(
        chroma,
        Filter::Chromatic(ChromaticAberration {
            spread: 9.5,
            angle: -1.2,
        }),
    ));
    // And a third of the third kind: the gradient map is the first filter whose
    // payload has a *length* (a stop list) and an `Option` around it — two more
    // shapes postcard writes without names, each a fresh way for a layout mistake
    // to decode as a different ramp. Three stops, every number distinct.
    let map = add_filter(&mut engine, None, Filter::GradientMap(None));
    engine.process(DocCommand::SetFilter(
        map,
        Filter::GradientMap(Some(ramp(&[
            (0.0, [0.12, 0.34, 0.56]),
            (0.4, [0.9, 0.62, 0.21]),
            (1.0, [0.05, 0.77, 0.43]),
        ]))),
    ));
    let before = engine.render_to_image();
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    assert!(
        images_match(&before, &loaded.render_to_image(), 0),
        "save \u{2192} load must reproduce the filtered picture identically",
    );
    // …and the settings come back as settings, not merely as the same pixels —
    // every filter, in stack order, kind and numbers alike.
    let filters = |e: &Engine| -> Vec<Filter> {
        e.observe()
            .layers
            .iter()
            .filter_map(|l| l.filter.clone())
            .collect()
    };
    let back = filters(&loaded);
    assert_eq!(back.len(), 3, "all three filter layers came back");
    assert_eq!(back, filters(&engine));
}

/// **A filter's id is a minted id**, so the counter a load resumes has to count it
/// (§17.9). The sibling of `layers.rs`'s `a_duplicates_ids_are_not_reused_after_a_
/// reload`, and here for the reason that one exists: two layers sharing an id is
/// not a wrong picture but a wrong *document* — every lookup finds whichever comes
/// first, so painting, renaming and deleting all reach a row nobody pointed at.
///
/// The id space is partitioned by author, so a solo document's ordinals *are* its
/// ids; a filter added last therefore leaves the highest one, which is exactly the
/// case a resync that does not know `AddFilter` mints gets wrong.
#[test]
fn a_filters_id_is_not_reused_after_a_reload() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    add_filter(&mut engine, None, NEUTRAL);
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    let existing: Vec<LayerId> = loaded.observe().layers.iter().map(|l| l.id).collect();
    assert_eq!(existing.len(), 2, "the filter came back with the log");

    loaded.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let after: Vec<LayerId> = loaded.observe().layers.iter().map(|l| l.id).collect();
    let mut unique = after.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        after.len(),
        "two layers share one id: {after:?}",
    );
}

/// **The gradient map's index is Oklab `L`, and its lerp is `Gradient::sample`'s**
/// (§21.11) — both pinned at once by the one ramp with a closed-form answer: the
/// black→white ramp maps every color to `(L, 0, 0)`, which is exactly what the
/// color filter's saturation-0 setting produces. Two different filters, two
/// different code paths (a chroma gain against a stop walk), one picture — a wrong
/// index (luminance, or un-saturated `L`), a wrong interpolation space, or a
/// mis-packed stop lane all break the agreement, and nothing else has to be known
/// about the render to check it.
#[test]
fn a_black_to_white_gradient_map_is_the_lightness_preserving_greyscale() {
    let Some(mut engine) = painted() else { return };
    let id = add_filter(&mut engine, None, GREY);
    let desaturated = engine.render_to_image();

    engine.process(DocCommand::SetFilter(
        id,
        Filter::GradientMap(Some(ramp(&[(0.0, [0.0; 3]), (1.0, [1.0; 3])]))),
    ));
    let mapped = engine.render_to_image();
    assert!(
        images_match(&desaturated, &mapped, 1),
        "the black\u{2192}white map should be the saturation-0 greyscale: {:?}",
        diff_fraction(&desaturated, &mapped),
    );
}

/// **A gradient map repaints; coverage stays put** (§21.11, §21.3.1). A ramp whose
/// two stops are the same red maps *every* lightness to red — the sharpest way to
/// see the repaint — while the bare canvas around the stroke must not change by a
/// byte: the pass writes color, not coverage, so paint that is not there cannot
/// be graded into being.
#[test]
fn a_gradient_map_repaints_the_paint_and_only_the_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, [0.1, 0.1, 0.9, 1.0], 22.0, STROKE);
    let before = engine.render_to_image();
    let off = (before.width / 2, before.height / 8); // well clear of the stroke
    assert!(
        !red_dominant(center(&before)),
        "the stroke starts blue: {:?}",
        center(&before),
    );

    add_filter(
        &mut engine,
        None,
        Filter::GradientMap(Some(ramp(&[
            (0.0, [0.85, 0.1, 0.1]),
            (1.0, [0.85, 0.1, 0.1]),
        ]))),
    );
    let after = engine.render_to_image();
    assert!(
        red_dominant(center(&after)),
        "an all-red ramp should repaint the stroke red: {:?}",
        center(&after),
    );
    assert_eq!(
        before.pixel(off.0, off.1),
        after.pixel(off.0, off.1),
        "bare canvas must come through a gradient map untouched",
    );
}

/// **A rampless gradient map is the exact identity** (§21.11) — it is the kind's
/// neutral, which is what a freshly added one holds, and §21.3's byte-level
/// neutral rule applies to it exactly as to a unity gain: the draw list drops it
/// rather than trusting a round trip.
#[test]
fn a_rampless_gradient_map_changes_no_pixel() {
    let Some(mut engine) = painted() else { return };
    let before = engine.render_to_image();
    let id = add_filter(&mut engine, None, Filter::GradientMap(None));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a gradient map with no ramp is not the identity",
    );

    // Dialling a ramp in must change the picture, and Neutral must put it back —
    // the same out-and-back every other kind's bar makes.
    engine.process(DocCommand::SetFilter(
        id,
        Filter::GradientMap(Some(ramp(&[
            (0.0, [0.1, 0.1, 0.6]),
            (1.0, [1.0, 0.9, 0.4]),
        ]))),
    ));
    assert!(
        !images_match(&before, &engine.render_to_image(), 0),
        "the ramp has to grade the picture for the second half to mean anything",
    );
    engine.process(DocCommand::SetFilter(id, Filter::GradientMap(None)));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "putting the ramp back to none did not restore the picture",
    );
}

/// The gradient map in a **pigment** document (§21.11, §6.7): the mapped color
/// re-enters through the inverse LUT with the residual recomputed — the leg that,
/// missing, renders a mapped black as `#383838` (the very defect §6.7 records).
/// An all-red ramp onto a blue stroke exercises a saturated answer the polynomial
/// alone cannot store.
#[cfg(feature = "mixbox")]
#[test]
fn a_gradient_map_works_in_a_pigment_document() {
    let Some(mut engine) = engine_or_skip_with(stark_engine::colorspace::ColorSpaceId::Mixbox)
    else {
        return;
    };
    paint(&mut engine, [0.1, 0.1, 0.9, 1.0], 22.0, STROKE);
    add_filter(
        &mut engine,
        None,
        Filter::GradientMap(Some(ramp(&[
            (0.0, [0.85, 0.1, 0.1]),
            (1.0, [0.85, 0.1, 0.1]),
        ]))),
    );
    let after = center(&engine.render_to_image());
    assert!(
        red_dominant(after),
        "an all-red ramp should repaint pigment paint red too: {after:?}",
    );
}

/// The **pigment** path (§6.7). A filter in a Mixbox document takes a different
/// road entirely — out through Mixbox's polynomial to light, back through its
/// inverse LUT, with the latent residual carried on both legs — and the residual is
/// exactly the half that has gone missing here before. Nothing in an Oklab test
/// touches it.
#[cfg(feature = "mixbox")]
#[test]
fn a_filter_works_in_a_pigment_document() {
    let Some(mut engine) = engine_or_skip_with(stark_engine::colorspace::ColorSpaceId::Mixbox)
    else {
        return;
    };
    paint(&mut engine, RED, 22.0, STROKE);
    let before = center(&engine.render_to_image());
    assert!(red_dominant(before), "the stroke should be red: {before:?}");

    add_filter(&mut engine, None, GREY);
    let after = center(&engine.render_to_image());
    assert!(
        is_grey(after),
        "a saturation of zero should leave the stroke achromatic in pigment too: \
         {after:?}",
    );
}
