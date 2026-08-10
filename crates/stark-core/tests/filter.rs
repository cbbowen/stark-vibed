//! Filter layers — the third kind of layer (§21).
//!
//! A filter holds no tiles and no region: it is a function of what its own stack
//! has already composited. So these cover the two halves of that sentence — that it
//! *is* a layer (it takes opacity and visibility, it undoes, it takes no paint), and
//! that "its own stack" is exactly what it reaches (a filter in a group leaves the
//! rest of the document alone, which is what makes carrying a filter onto a layer
//! the whole of "filter just this one").
//!
//! Three of these are about a filter doing **nothing**, and that is deliberate. A
//! pass that runs over every texel of the frame has no coverage to hide behind, so
//! the cases where it must be the exact identity — neutral, hidden, nothing beneath
//! it — are the ones where a mistake is a whole-picture change with nothing on
//! screen to say where it came from.

mod common;

use common::*;
use stark_core::Engine;
use stark_core::command::{DocCommand, PeerCommand, ViewCommand};
use stark_core::document::{ColorAdjust, Filter, LayerId, Place};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const STROKE: &[Vec2] = &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];

/// The neutral filter — what `AddFilter` lands.
const NEUTRAL: Filter = Filter::Color(ColorAdjust::NEUTRAL);

/// Every colour drained away. The sharpest filter to test with: it is visible on
/// any painting, it is checkable without knowing the render's exact numbers (the
/// three channels simply have to agree), and it is the one setting whose *correct*
/// answer differs from the naive one — dropping Oklab chroma keeps lightness where
/// it was, which a weighted RGB average does not.
const GREY: Filter = Filter::Color(ColorAdjust {
    saturation: 0.0,
    ..ColorAdjust::NEUTRAL
});

/// Whether a pixel is achromatic — the three channels within a level or two of one
/// another, which is what a saturation of zero has to produce.
fn is_grey(c: [u8; 4]) -> bool {
    let (lo, hi) = (
        c[0].min(c[1]).min(c[2]) as i32,
        c[0].max(c[1]).max(c[2]) as i32,
    );
    hi - lo <= 3
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
fn a_filter_recolours_what_is_beneath_it() {
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

/// **Chroma, and nothing else.** Draining the colour leaves *perceived lightness*
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
        let lin = |i: usize| stark_core::color::srgb_to_linear(c[i] as f32 / 255.0);
        stark_core::color::linear_srgb_to_oklab([lin(0), lin(1), lin(2)])[0]
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
/// feature could least do without, since a colour adjustment is judged by looking.
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
        engine.process(ViewCommand::PreviewFilter(Some((id, filter))));
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
        Filter::Color(ColorAdjust {
            exposure: 0.75,
            contrast: 1.4,
            saturation: 0.35,
            hue: 0.6,
        }),
    ));
    let before = engine.render_to_image();
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    assert!(
        images_match(&before, &loaded.render_to_image(), 0),
        "save \u{2192} load must reproduce the filtered picture identically",
    );
    // …and the settings come back as settings, not merely as the same pixels.
    let obs = loaded.observe();
    let back = obs
        .layers
        .iter()
        .find_map(|l| l.filter)
        .expect("the filter layer came back");
    assert_eq!(
        back,
        engine
            .observe()
            .layers
            .iter()
            .find_map(|l| l.filter)
            .unwrap()
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
    let Some(mut engine) = engine_or_skip_with(stark_core::colorspace::ColorSpaceId::Mixbox) else {
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
