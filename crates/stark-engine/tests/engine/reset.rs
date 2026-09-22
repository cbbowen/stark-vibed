//! What a document reset gives back (§5, §23).

use crate::common::palette::RED;
use crate::common::*;
use stark_assetid::Picture;
use stark_engine::Engine;
use stark_engine::command::{DocCommand, Edit};
use stark_engine::testing::{AllocSource, TilePool};
use stark_model::document::{FillOp, SelectionShape};
use stark_model::geom::{IVec2, TILE_SIZE, Vec2};
use stark_model::{AssetId, AssetNeed, ColorSpaceId, Srgb, SubstrateId};

/// An opaque square picture of one color, `tint` apart from its siblings so each
/// imports under an id of its own.
fn picture(tint: u8) -> Vec<u8> {
    const SIDE: u32 = 64;
    Picture {
        width: SIDE,
        height: SIDE,
        pixels: [220, tint, 25, 255].repeat((SIDE * SIDE) as usize),
    }
    .encode()
    .expect("a well-formed picture")
}

/// Import `png` and place it centred on the canvas origin, answering its id.
fn place(engine: &mut Engine, png: &[u8]) -> AssetId {
    let image = engine.import_picture(png).expect("import the picture");
    engine.process(DocCommand::PlaceImage {
        carrier: None,
        above: None,
        at: IVec2::splat(-32),
        name: None,
        image,
    });
    image
}

/// **A New gives back what the closed document held**, in the space it was already in.
///
/// Same-space, the pipelines are kept (§6.7) and nothing else is rebuilt, so the reset
/// has to hand back the rest itself: every tile the closed document owned would
/// otherwise sit idle on the pool's free list — counted against the history budget,
/// so a blank page gives up undo depth for memory it cannot free (§5) — and every
/// picture it placed stays decoded for a document that no longer names it (§23).
#[test]
fn a_new_document_gives_back_the_closed_one_s_tiles_and_pictures() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let layer = engine.observe().active_layer;
    let half = 2.0 * TILE_SIZE as f32;
    engine.process(DocCommand::Set(Edit::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(Vec2::splat(-half), Vec2::splat(half)),
            0.0,
            Srgb::new(RED),
            1.0,
        ),
    }));
    let image = place(&mut engine, &picture(30));
    // The draw list holds tile handles as well, and a frame builds one.
    let _ = engine.render_to_image();
    assert!(engine.resident_tile_bytes() > 0, "the fill owned no tiles");

    engine
        .new_document(engine.color_space(), SubstrateId::default())
        .expect("the space it is already in");
    assert_eq!(
        engine.resident_tile_bytes(),
        0,
        "the closed document's tiles are still resident"
    );
    assert!(
        !engine.holds(AssetNeed::Picture(image)),
        "the closed document's picture is still held"
    );
}

/// A sibling's document is its own, and so are its pictures (§11, §23): a store shared
/// with the donor would keep every picture the donor's closed documents placed alive
/// for as long as the sibling lives.
#[test]
fn a_sibling_does_not_hold_the_donor_s_pictures() {
    let Some(mut donor) = engine_or_skip() else {
        return;
    };
    let sibling = Engine::new_sharing(&donor, SIZE);
    let image = place(&mut donor, &picture(30));
    assert!(donor.holds(AssetNeed::Picture(image)));
    assert!(
        !sibling.holds(AssetNeed::Picture(image)),
        "the sibling shares the donor's pictures"
    );
}

/// Releasing the idle textures leaves the ones in use alone: a handle still out keeps
/// its texture and returns it to the free list as ever.
#[test]
fn releasing_the_idle_textures_keeps_the_ones_in_use() {
    const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    let Some(engine) = engine_or_skip() else {
        return;
    };
    let pool = TilePool::new(engine.gpu().clone(), [COLOR]);
    let held = pool.acquire_tex(COLOR, AllocSource::Unknown);
    let one = pool.resident_bytes();
    drop(
        (0..3)
            .map(|_| pool.acquire_tex(COLOR, AllocSource::Unknown))
            .collect::<Vec<_>>(),
    );
    assert_eq!(pool.free_count(COLOR), 3);

    pool.release_idle();
    assert_eq!(pool.free_count(COLOR), 0, "an idle texture was kept");
    assert_eq!(
        pool.resident_bytes(),
        one,
        "the texture in use was released"
    );
    drop(held);
    assert_eq!(
        pool.free_count(COLOR),
        1,
        "the held texture did not come back"
    );
}

/// **A load keeps the pictures its log names, and only those** — in the space the
/// engine is in and across a change of space alike.
///
/// A picture held here satisfies the load's content check without being bundled
/// (`unresolved_content`), so the reset in front of the replay has to carry it: a
/// placement replayed without its picture is an empty layer (§23). A picture the log
/// does not name belongs to the closed document and goes with it.
#[test]
fn a_load_keeps_the_pictures_its_log_names_and_no_others() {
    for space in [ColorSpaceId::Oklab, ColorSpaceId::Mixbox] {
        if !stark_engine::available(space) {
            continue;
        }
        let Some(mut engine) = engine_or_skip() else {
            return;
        };
        let named = place(&mut engine, &picture(30));
        let unnamed = engine.import_picture(&picture(90)).expect("import");
        let mut file = engine.document_file();
        file.content
            .retain(|(need, _)| !matches!(need, AssetNeed::Picture(_)));
        file.canvas.color_space = space;

        engine
            .load_document(&file)
            .expect("the log's picture is held here");
        assert!(
            engine.holds(AssetNeed::Picture(named)),
            "{space:?}: the picture the log places was dropped"
        );
        assert!(
            painted(&engine.render_to_image(), Vec2::ZERO),
            "{space:?}: the placement replayed without its picture"
        );
        assert!(
            !engine.holds(AssetNeed::Picture(unnamed)),
            "{space:?}: a picture no log names outlived the document"
        );
    }
}
