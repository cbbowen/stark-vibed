//! How layers cover one another (§6.3, pass A).
//!
//! **The claim.** A layer affects what is beneath it only as much as it is visible
//! at all. It sounds too obvious to test, and it used to be false: pass A weighed a
//! layer's "over" by its per-unit **opacity** alone, so a film with opacity 1 and no
//! thickness — which the media pass draws as nothing over bare canvas — replaced the
//! colour outright over another layer's paint. Every soft brush deposits exactly that
//! state across its fringe (`stamp.wesl` saturates opacity as `1 − exp(−op·τ)`
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
use stark_core::document::{BrushDynamics, BrushParams, BrushShape};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.10, 0.10, 1.0];
const BLUE: [f32; 4] = [0.10, 0.20, 0.85, 1.0];

/// A very soft, wide tip — the shape whose faint fringe is the whole point.
fn soft(color: [f32; 4], radius: f32) -> BrushParams {
    BrushParams {
        color,
        radius,
        shape: BrushShape::Round { hardness: 0.0 },
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
    under.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
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
        worst <= 3,
        "a layer invisible over bare canvas moved the paint beneath it by {worst}"
    );
}

/// A render at a size other than the **surface's** builds its own pass-A
/// attachments instead of resizing the cached pair (`Compositor::render`) — which is
/// what keeps an export, or the navigator's per-edit miniature, from evicting the
/// screen's set and forcing it to be rebuilt on the very next frame.
///
/// The two paths therefore have to be equivalent, or what a file gets would depend
/// on how large the window happened to be when it was written. Both engines here
/// export the same 320×224 view of the same painting; only the surface they were
/// built on differs, so one goes through its own attachments and the other through
/// the cached ones. The **Multiply** layer is deliberate: a non-normal blend mode is
/// what brings the scratch pair the blend passes bounce through into it, and that
/// pair is transient for the same reason the targets are.
#[test]
fn an_off_size_render_matches_one_at_the_surfaces_own_size() {
    use stark_core::command::ViewCommand;
    use stark_core::document::{BlendMode, MatteRegion};
    use stark_core::geom::Extent2;
    use stark_core::{Background, ExportScale, Offscreen, Rendered};

    // The piece: a frame whose rect is exactly the exported size, centred on the
    // canvas origin, so both exports render the identical view.
    const HALF: Vec2 = Vec2::new(160.0, 112.0);
    let exported = Extent2::new(320, 224);

    let built = |viewport| -> Option<_> {
        let mut engine = engine_or_skip_sized(viewport)?;
        paint(
            &mut engine,
            RED,
            18.0,
            &[
                Vec2::new(-90.0, -60.0),
                Vec2::new(70.0, 20.0),
                Vec2::new(-40.0, 80.0),
            ],
        );
        // A second layer, multiplied over the first.
        engine.process(DocCommand::AddLayer {
            carrier: None,
            above: None,
        });
        let top = engine.observe().active_layer;
        engine.process(DocCommand::SetLayerBlend(top, BlendMode::Multiply));
        paint(
            &mut engine,
            BLUE,
            26.0,
            &[Vec2::new(-80.0, 40.0), Vec2::new(90.0, -30.0)],
        );
        engine.process(DocCommand::AddMatte {
            carrier: None,
            above: None,
            region: MatteRegion::OutsideRect {
                min: -HALF,
                max: HALF,
            },
            color: [0.0, 0.0, 0.0],
        });
        let frame = engine.observe().layers.last().expect("matte").id;
        // The view is the export's, so nothing about the surface's size can reach
        // the picture other than through the attachments this is testing.
        engine.process(ViewCommand::CenterOn(Vec2::ZERO));
        let image = pollster::block_on(
            engine
                .export(
                    &mut Offscreen::default(),
                    Some(frame),
                    ExportScale::Factor(1.0),
                    Background::Substrate,
                    Rendered::Committed,
                )
                .expect("export"),
        );
        Some(image)
    };

    // Its own attachments (the surface is smaller than the export)…
    let Some(transient) = built(Extent2::new(140, 100)) else {
        return;
    };
    // …and the cached ones (the surface *is* the export's size).
    let Some(cached) = built(exported) else {
        return;
    };

    assert_eq!((transient.width, transient.height), (320, 224));
    let (frac, worst) = diff_fraction(&transient, &cached);
    assert!(
        worst <= 2,
        "an off-size render came out differently from one at the surface's own size \
         ({frac:.4} of pixels differ, worst {worst})"
    );
}

/// A **kept** [`Offscreen`] renders the same picture a fresh one would, across
/// everything that changes what it was built against: a new size, a swapped
/// environment, and a colour-space rebuild — which does not mutate the compositing
/// pipeline but replaces it (§6.7).
///
/// This is the navigator's arrangement: one slot, reused for the life of the app,
/// one render per edit. So "reused" has to mean "reused *or rebuilt*", and the
/// comparison against a fresh slot is what tells the two apart — a kept slot that
/// failed to notice a change would composite through attachments of the wrong size or
/// belonging to a pipeline that is gone, and a fresh one never can.
#[test]
fn a_kept_offscreen_renders_what_a_fresh_one_would() {
    #[cfg(feature = "mixbox")]
    use stark_core::ColorSpaceId;
    use stark_core::command::ViewCommand;
    use stark_core::document::MatteRegion;
    use stark_core::geom::Extent2;
    use stark_core::{Background, EnvironmentId, ExportScale, Offscreen, Rendered};

    let Some(mut engine) = engine_or_skip_sized(Extent2::new(200, 150)) else {
        return;
    };
    // The one slot, reused by every `kept` render below.
    let mut kept = Offscreen::default();

    // Paint something with edges at an angle — what a composite through
    // wrong-sized attachments would resample and soften — and frame it, so the
    // export renders a fixed rect rather than the window.
    let framed = |engine: &mut stark_core::Engine| {
        paint(
            engine,
            RED,
            16.0,
            &[
                Vec2::new(-70.0, -45.0),
                Vec2::new(60.0, 15.0),
                Vec2::new(-30.0, 50.0),
            ],
        );
        engine.process(DocCommand::AddMatte {
            carrier: None,
            above: None,
            region: MatteRegion::OutsideRect {
                min: Vec2::new(-80.0, -60.0),
                max: Vec2::new(80.0, 60.0),
            },
            color: [0.0, 0.0, 0.0],
        });
    };
    let shot = |engine: &mut stark_core::Engine, into: &mut Offscreen, scale: f32| {
        let frame = engine.observe().layers.last().expect("matte").id;
        pollster::block_on(
            engine
                .export(
                    into,
                    Some(frame),
                    ExportScale::Factor(scale),
                    Background::Substrate,
                    Rendered::Committed,
                )
                .expect("export"),
        )
    };
    let same = |a: &stark_core::RgbaImage, b: &stark_core::RgbaImage, what: &str| {
        assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
        let (frac, worst) = diff_fraction(a, b);
        assert!(
            worst <= 2,
            "{what}: a kept slot drew differently from a fresh one \
             ({frac:.4} of pixels differ, worst {worst})"
        );
    };

    framed(&mut engine);
    let first = shot(&mut engine, &mut kept, 1.0);
    assert_eq!((first.width, first.height), (160, 120));

    // A second size through the same slot: its attachments have to follow, or the
    // composite comes back resampled through the first size.
    let grown = shot(&mut engine, &mut kept, 2.0);
    assert_eq!((grown.width, grown.height), (320, 240));
    same(
        &grown,
        &shot(&mut engine, &mut Offscreen::default(), 2.0),
        "resized",
    );

    // Back down again — the same slot, shrinking.
    same(
        &shot(&mut engine, &mut kept, 1.0),
        &first,
        "back to the first size",
    );

    // The lighting is bound *into* the media bind group the slot is holding. The
    // harness starts on the studio HDR, so switching to the procedural reference
    // light is a real swap.
    engine.process(ViewCommand::SetEnvironment(EnvironmentId::Neutral));
    same(
        &shot(&mut engine, &mut kept, 1.0),
        &shot(&mut engine, &mut Offscreen::default(), 1.0),
        "after a lighting swap",
    );

    // And a new document in the other colour space, which rebuilds the pipeline the
    // slot's attachments belong to. Same surface, so nothing else moves with it.
    //
    // Only where there *is* another one: without the `mixbox` feature Oklab is the
    // whole set, and "switching" to the space already open would rebuild nothing, so
    // the claim has nothing left to make rather than a weaker version of itself.
    #[cfg(feature = "mixbox")]
    {
        let surface = engine.surface();
        engine
            .new_document(ColorSpaceId::Mixbox, surface)
            .expect("the `mixbox` feature is on in this build");
        framed(&mut engine);
        same(
            &shot(&mut engine, &mut kept, 1.0),
            &shot(&mut engine, &mut Offscreen::default(), 1.0),
            "after a colour-space rebuild",
        );
    }
}
