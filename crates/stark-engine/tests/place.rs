//! Placing an image brought in from outside the document (§23).
//!
//! The claims worth pinning are the ones that make an import feel like *paint* rather
//! than like a pasted rectangle, and they are all consequences of the tiles being built
//! on the CPU from the paint representation (`gpu::place`):
//!
//! - the pixels land where they were put, unresampled and the right way up;
//! - a transparent source pixel lays nothing, so a cut-out is a cut-out;
//! - it is one undo step, because it is one action (§23);
//! - it replays — save and reload reproduce it to the byte, which for this path is a
//!   stronger claim than for any other, since nothing here goes through a shader.

mod common;

use common::*;
use stark_assetid::Picture;
use stark_engine::Engine;
use stark_engine::command::DocCommand;
use stark_model::geom::{IVec2, Vec2};

/// A 64×64 swatch: opaque red in the left half, opaque green in the right, and a
/// **fully transparent** 8 px margin all round.
///
/// One fixture rather than three, because the three questions this file asks are about
/// one image: where its pixels landed, whether its colors survived, and whether its
/// transparent part laid anything.
const SWATCH: u32 = 64;
const MARGIN: u32 = 8;

fn swatch() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((SWATCH * SWATCH * 4) as usize);
    for y in 0..SWATCH {
        for x in 0..SWATCH {
            let clear = x < MARGIN || y < MARGIN || x >= SWATCH - MARGIN || y >= SWATCH - MARGIN;
            let left = x < SWATCH / 2;
            pixels.extend_from_slice(&match (clear, left) {
                (true, _) => [0, 0, 0, 0],
                (false, true) => [220, 30, 25, 255],
                (false, false) => [30, 200, 60, 255],
            });
        }
    }
    Picture {
        width: SWATCH,
        height: SWATCH,
        pixels,
    }
    .encode()
    .expect("a well-formed swatch")
}

/// Import the swatch and place it centred on the canvas origin, which the default
/// view puts at the middle of the viewport.
///
/// Two steps, because a picture is content-addressed like a brush shape (§23): the
/// import answers with the id, and the action references it.
fn place(engine: &mut Engine) {
    let id = engine.import_picture(&swatch()).expect("import the swatch");
    engine.process(DocCommand::PlaceImage {
        carrier: None,
        above: None,
        at: IVec2::splat(-(SWATCH as i32) / 2),
        name: Some("swatch.png".into()),
        image: id,
    });
}

/// Whether a pixel reads as green paint — [`red_dominant`]'s counterpart, and needed
/// here because the swatch's two halves are what say the image did not arrive mirrored.
fn green_dominant(c: [u8; 4]) -> bool {
    c[1] as i32 > c[0] as i32 + 30 && c[1] as i32 > c[2] as i32 + 30
}

/// **The pixels land where they were put, and stay the way round they were.**
///
/// Placement is in whole canvas pixels precisely so nothing is resampled (§23), so this
/// is not a tolerance test: the left half of the image has to be red on the left of the
/// canvas and the right half green on the right. A transposed row index, a flipped
/// origin or an off-by-one in the tile walk all show up here as the two halves in the
/// wrong places.
#[test]
fn a_placed_image_lands_where_it_was_put() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    place(&mut engine);
    let img = engine.render_to_image();

    let (cx, cy) = (img.width / 2, img.height / 2);
    let quarter = SWATCH / 4;
    assert!(
        red_dominant(img.pixel(cx - quarter, cy)),
        "the image's left half is red, and it is on the left: {:?}",
        img.pixel(cx - quarter, cy),
    );
    assert!(
        green_dominant(img.pixel(cx + quarter, cy)),
        "…and its right half is green, on the right: {:?}",
        img.pixel(cx + quarter, cy),
    );
}

/// **A transparent source pixel lays no paint**, so a cut-out is a cut-out rather than
/// a rectangle with a faint ghost round it.
///
/// The margin is checked one pixel inside the image's own bounds, which is the
/// interesting place: outside the image nothing could have been written whatever the
/// code did, while inside it the texel was visited, converted and stored — and an
/// implementation that laid `−ln(1)` worth of paint there would leave a tile that reads
/// as painted to `bounds` and to the compositor.
#[test]
fn a_transparent_margin_leaves_the_canvas_bare() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let bare = engine.render_to_image();
    place(&mut engine);
    let placed = engine.render_to_image();

    let (cx, cy) = (placed.width / 2, placed.height / 2);
    let edge = SWATCH / 2 - 1;
    for (x, y) in [
        (cx - edge, cy),
        (cx + edge - 1, cy),
        (cx, cy - edge),
        (cx, cy + edge - 1),
    ] {
        assert_eq!(
            placed.pixel(x, y),
            bare.pixel(x, y),
            "({x}, {y}) is inside the image's margin and must be untouched canvas",
        );
    }
}

/// **One action, so one undo step** — the whole reason a placement mints its own layer
/// rather than being spelled as an add, a rename and a fill (§23). Undo has to take the
/// layer, its name and its paint together and leave the document exactly as it was.
#[test]
fn undoing_a_placement_takes_the_layer_with_it() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let before = engine.render_to_image();
    let layers_before = engine.observe().layers.len();

    place(&mut engine);
    assert_eq!(
        engine.observe().layers.len(),
        layers_before + 1,
        "the placement adds exactly one layer",
    );

    engine.process(DocCommand::Undo);
    assert_eq!(
        engine.observe().layers.len(),
        layers_before,
        "one undo takes the whole placement back",
    );
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "…and leaves the canvas byte-identical to what it was",
    );
}

/// The layer arrives **named after what it came from**, and is the paint target: an
/// artist who just placed a reference photograph is looking at the layer they will work
/// over (§23).
#[test]
fn the_placed_layer_is_named_and_is_the_paint_target() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    place(&mut engine);
    let obs = engine.observe();
    let layer = obs
        .layers
        .iter()
        .find(|l| l.id == obs.active_layer)
        .expect("the placed layer is the active one");
    assert_eq!(layer.name.as_deref(), Some("swatch.png"));
    assert!(
        layer.matte.is_none() && layer.filter.is_none(),
        "it is paint"
    );
}

/// **The picture is content, so placing it twice costs one copy** (§23).
///
/// This is what content-addressing buys over carrying the pixels in the action, and it
/// is worth a test rather than a comment: two placements of one photograph are two
/// actions naming one id, so the bundle holds it once and a peer that already has it
/// fetches nothing. The bundle is where that is observable without reaching into the
/// store.
#[test]
fn placing_one_picture_twice_bundles_it_once() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    place(&mut engine);
    place(&mut engine);

    let file = engine.document_file();
    assert_eq!(
        file.actions
            .iter()
            .filter(|a| matches!(a.kind, stark_model::document::ActionKind::PlaceImage { .. }))
            .count(),
        2,
        "two placements",
    );
    assert_eq!(file.pictures.len(), 1, "…and one picture between them");
}

/// A placement whose picture this session does not hold **adds the layer and leaves it
/// empty**, rather than silently succeeding or dropping the action.
///
/// Reaching this is a caller that skipped the bill — the loader refuses a document
/// whose content is unresolved, and the transport parks the action until its blob lands
/// (§12.4) — so what is pinned here is that the failure is *visible in the document*:
/// the layer the action minted exists, because withdrawing it would leave the action
/// half-applied and the minted id spent on nothing.
#[test]
fn a_placement_whose_picture_is_missing_leaves_an_empty_layer() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let before = engine.render_to_image();
    let layers_before = engine.observe().layers.len();

    engine.process(DocCommand::PlaceImage {
        carrier: None,
        above: None,
        at: IVec2::ZERO,
        name: Some("absent.png".into()),
        image: stark_model::AssetId([3; 32]),
    });

    assert_eq!(
        engine.observe().layers.len(),
        layers_before + 1,
        "the layer the action minted is there",
    );
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "…and nothing was painted onto it",
    );
}

/// **Save and reload reproduce it exactly** (§8) — the replay claim, which is stronger
/// here than anywhere else in the suite: these tiles come from CPU arithmetic rather
/// than from a pass, so "identical" means every byte, not every byte within a
/// tolerance, and it would hold across two different adapters as well.
///
/// It is also the one test that exercises the PNG round trip the action serializes
/// through, end to end: encode on save, decode on load, and the same picture out.
#[test]
fn a_placed_image_survives_a_save_and_reload() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    place(&mut original);
    // Paint over it, so the reload is asked to reproduce a placement *and* the stroke
    // that followed — which is what makes this a replay test rather than a decode one.
    paint(
        &mut original,
        [0.9, 0.9, 0.2],
        14.0,
        &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)],
    );
    let before = original.render_to_image();
    let bytes = original.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");

    assert!(
        images_match(&before, &loaded.render_to_image(), 0),
        "a reloaded placement must reproduce identical pixels",
    );
    // …and the layer came back as a layer, name and all, rather than as pixels merged
    // into whatever was beneath.
    assert!(
        loaded
            .observe()
            .layers
            .iter()
            .any(|l| l.name.as_deref() == Some("swatch.png")),
        "the placed layer survives the reload as itself",
    );
    // The file *is* the history (§8), so undo reaches back through the reload: two
    // steps take the stroke and then the placement, and the canvas is bare again.
    let bare = {
        let mut fresh = engine_or_skip_blue().expect("adapter available");
        fresh.render_to_image()
    };
    loaded.process(DocCommand::Undo);
    loaded.process(DocCommand::Undo);
    assert!(
        images_match(&bare, &loaded.render_to_image(), 0),
        "undo after load takes the placement back with the stroke",
    );
}
