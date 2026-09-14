//! The desk's doors on a real engine: what `Desk::replace` refuses, what a join
//! skips, where it leaves the view, and what `Desk::install` checks (§8, §12.4, §15.6).
//!
//! GPU tests. A missing adapter fails them unless `STARK_ALLOW_NO_GPU=1`
//! (`stark_engine::testing::or_skip`).

use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::{Engine, EngineError, Extent2, Identity, Transfer};
use stark_model::document::{ActorId, MatteRegion, Parcel, Place};
use stark_model::geom::Vec2;
use stark_model::{AssetId, AssetNeed, DocError, DocumentFile, Srgb};
use stark_ui::desk::{Desk, Replacement};

const SIZE: Extent2 = Extent2 {
    width: 256,
    height: 256,
};

/// An engine of its own — its own document and asset stores — on this binary's one
/// device. `None` for a permitted skip.
fn engine() -> Option<Engine> {
    let ctx = stark_engine::testing::shared_context(|| {
        stark_engine::testing::or_skip(
            pollster::block_on(stark_engine::GpuContext::headless()),
            "the desk's GPU tests",
        )
    })?;
    Some(Engine::new(
        ctx.clone(),
        wgpu::TextureFormat::Rgba8Unorm,
        SIZE,
    ))
}

fn desk() -> Option<Desk> {
    Some(Desk::new(engine()?, Transfer::Srgb))
}

/// A small grey field that decodes as a stamp and as a height map alike; `seed` picks
/// which, so two seeds are two ids.
fn gray_png(seed: u8) -> Vec<u8> {
    stark_assetid::Canonical {
        width: 16,
        height: 16,
        texels: (0..=255u8).map(|i| i.wrapping_mul(seed)).collect(),
    }
    .encode()
    .expect("a small grey field encodes")
}

/// A document moved onto an image substrate, with its bundle left out — what a lean
/// file and a host's snapshot both are — and the need that leaves owed.
fn lean_on_a_substrate() -> Option<(DocumentFile, AssetNeed)> {
    let mut donor = engine()?;
    let substrate = donor
        .import_substrate(&gray_png(3))
        .expect("a grey field is a height map");
    donor.process(DocCommand::SetSubstrate(substrate));
    let mut file = donor.document_file();
    file.content.clear();
    let owed = file.unbundled_content();
    let need = AssetNeed::for_substrate(substrate).expect("an image substrate has bytes");
    assert_eq!(owed, [need], "the lean document owes its substrate");
    Some((file, need))
}

fn is_misnamed<T>(result: &Result<T, EngineError>) -> bool {
    matches!(
        result,
        Err(EngineError::Document(DocError::Misnamed { .. }))
    )
}

/// **An open whose owed bytes are not what the id names is refused before the engine
/// replaces anything**: the view and the document stay as they were.
#[test]
fn an_open_owed_the_wrong_bytes_is_refused_and_moves_nothing() {
    let (Some((file, need)), Some(mut desk)) = (lean_on_a_substrate(), desk()) else {
        return;
    };
    // Somewhere a replacement would visibly move: off the origin, one commit in.
    desk.engine_mut().process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    desk.engine_mut()
        .process(ViewCommand::CenterOn(Vec2::new(300.0, -200.0)));
    let before = desk.engine().observe();

    let wrong = gray_png(5);
    let refused = desk.replace(Replacement::Open(&file), &[(need, &wrong)]);

    assert!(is_misnamed(&refused), "refused as misnamed: {refused:?}");
    let after = desk.engine().observe();
    assert_eq!(after.view, before.view, "a refusal leaves the view");
    assert_eq!(
        after.doc_revision, before.doc_revision,
        "a refusal leaves the document"
    );
}

/// **The same bytes through a join are skipped, not refused**: the promise is left to
/// the peer fetch (`stark_net::Joined::owed`), and the caller is told which.
#[test]
fn a_join_owed_the_wrong_bytes_goes_on_and_names_what_it_skipped() {
    let (Some((file, need)), Some(mut desk)) = (lean_on_a_substrate(), desk()) else {
        return;
    };
    let wrong = gray_png(5);
    let identity = Identity::new(ActorId(2), 0);

    let replaced = desk
        .replace(Replacement::Join(&file, identity), &[(need, &wrong)])
        .expect("a join goes on without owed content");

    assert!(
        matches!(
            replaced.skipped.as_slice(),
            [(skipped, EngineError::Document(DocError::Misnamed { .. }))] if *skipped == need
        ),
        "the substrate is the one skip: {:?}",
        replaced.skipped
    );
    assert!(
        !desk.engine().holds(need),
        "nothing was installed under its id"
    );
}

/// **An open frames the piece**: a document with a matte frame arrives with the frame
/// on screen, and framing moved the view and nothing else the projection holds.
#[test]
fn an_open_puts_the_view_on_the_frame() {
    let (Some(mut donor), Some(mut desk)) = (engine(), desk()) else {
        return;
    };
    // Far from the origin, where every fresh view starts.
    let (min, max) = (Vec2::new(2000.0, 1000.0), Vec2::new(2400.0, 1300.0));
    donor.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect { min, max },
        paint: Parcel::Solid(Srgb::new([0.0, 0.0, 0.0])),
    });
    let file = donor.document_file();

    let replaced = desk
        .replace(Replacement::Open(&file), &[])
        .expect("a document that owes nothing opens");

    let view = desk.engine().view();
    let middle = view.canvas_to_screen((min + max) * 0.5);
    let centre = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    assert!(
        middle.distance(centre) < 1.0,
        "the frame's middle is on screen at {middle}, not the centre {centre}"
    );
    let (shown_min, shown_max) = view.visible_bounds();
    assert!(
        shown_min.cmple(min).all() && shown_max.cmpge(max).all(),
        "the whole frame is on screen: {shown_min}..{shown_max} around {min}..{max}"
    );
    assert!(replaced.skipped.is_empty(), "an open skips nothing");
    assert_eq!(
        replaced.seen,
        desk.engine().observe(),
        "the projection handed back is the framed document's"
    );
}

/// **A brush install checks its id**, as a substrate's and a picture's do: bytes
/// under someone else's id are refused, and bytes under their own go in.
#[test]
fn a_brush_install_refuses_bytes_under_another_id() {
    let Some(mut desk) = desk() else {
        return;
    };
    let png = gray_png(3);
    let own = stark_assetid::coverage(&png)
        .expect("a grey field is a stamp")
        .id();
    let other = AssetId([7; 32]);
    assert_ne!(own, other);

    let refused = desk.install(AssetNeed::Brush(other), &png);
    assert!(is_misnamed(&refused), "refused as misnamed: {refused:?}");
    assert!(
        !desk.engine().has_asset(other),
        "nothing is held as `other`"
    );

    desk.install(AssetNeed::Brush(own), &png)
        .expect("bytes under their own id install");
    assert!(desk.engine().has_asset(own));
}
