//! **If a footprint says two actions commute, they must actually commute** (§12.6)
//! — the other half of footprint honesty, asked over pairs of the whole vocabulary.
//!
//! `tests/footprint.rs` proves the **writes** half structurally: every difference a
//! run makes lands inside a resource its action declared. Its own header concedes
//! what it cannot reach — *"`reads` are not checked: a read is not observable in a
//! state diff, and catching an undeclared one needs a different instrument."* This
//! is that instrument, and it is stronger than the reads check it stands in for,
//! because it asks the question §12.6 actually cares about rather than a proxy for
//! it.
//!
//! An undeclared **read** and an undeclared **write** fail identically here:
//! `Footprint::conflicts` tests writes against reads in both directions, so either
//! omission makes a pair claim to commute. And "commute" has one meaning that can be
//! checked — *applying them in either order produces the same document* — which is a
//! statement about pixels, so it is asked in pixels.
//!
//! # Why the actions are minted once and re-stamped
//!
//! A command run twice does not produce the same action twice: a stroke's `seed` is
//! the document clock at the press, and `deposit_jitter` defaults to 1% everywhere
//! (§6.2), so two strokes from one gesture spelled twice differ in every texel they
//! touch. So each action is minted **once** and the same payload is replayed into two
//! fresh documents.
//!
//! The log is totally ordered by `(lamport, actor)`, which is the whole point of it —
//! feeding one pair into two peers in two arrival orders gives one materialization,
//! by design (§12.1). So the two orders are made by swapping the **lamports**: the
//! payload is identical and only the order key moves.
//!
//! # What a failure means
//!
//! That the two pictures differ is not itself a bug — plenty of action pairs
//! genuinely do not commute. The bug is a pair that differs *while its footprint says
//! it commutes*, because that is what the history's splice trusts: an undo shifts its
//! target past everything it commutes with instead of replaying, and pixels cannot
//! show which materialization ran. `tests/commute.rs` exercises that splice on five
//! hand-written scenarios; this asks the underlying claim of the vocabulary at large.
//!
//! # What it costs
//!
//! One test of ~20 s, and quadratic in its table: every commuting pair builds two
//! documents and renders both. That is the price of the coverage and it is worth
//! knowing before adding a row — the two levers that mattered were keeping the
//! strokes small and synthesizing the substrate rather than decoding the bundled
//! one, which alone was the difference between 20 seconds and nearly three minutes.

mod common;

use common::{engine_or_skip, images_match};
use stark_engine::command::{DocCommand, InputCommand, PeerCommand};
use stark_engine::{Engine, RgbaImage};
use stark_model::document::{
    Action, ActionId, ActorId, BlendMode, ColorAdjust, FillOp, Filter, LayerId, MatteRegion,
    Parcel, Place, SelectionMode, SelectionOp, SelectionShape, TransformMap, compute_footprint,
};
use stark_model::geom::{Affine2, Vec2};
use stark_model::io::DocumentFile;
use stark_model::{Srgb, SubstrateScale};

/// The actor every minted action is authored by.
///
/// One actor throughout, deliberately. A selection is per-author (§17.3), so two
/// actors would make every stroke/selection pair commute trivially and quietly
/// remove the most interesting rows from the table.
const ACTOR: ActorId = ActorId(1);

const A: LayerId = LayerId(0);

/// A document both runs start from: two layers with paint on them, a selection, and
/// a picture placed — enough that the pairs below have something to act on and
/// something to disturb.
fn base() -> Option<DocumentFile> {
    let mut e = engine_or_skip()?;
    e.start_collaboration(ACTOR);
    // A substrate with real relief, so the deposition tooth has something to read
    // (§6.4). On the flat builtin a stroke is not a function of the substrate at
    // all, and the pairs most worth asking about would be vacuous.
    //
    // Synthesized rather than `stark_testdata::assets::rough()`, and that is a cost
    // decision this table has to make: every peer replays the log, so the substrate
    // is decoded once per engine, and this test builds a couple of hundred of them.
    // The bundled rough map is 2.6 MB and took the run from 20 seconds to nearly
    // three minutes on its own. What the tooth needs is *relief*, not resolution.
    let grain = e
        .import_substrate(&grain())
        .expect("the synthesized height map imports");
    e.process(DocCommand::SetSubstrate(grain));
    e.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    for (layer, y) in [(A, 90.0f32), (LayerId(1), 150.0)] {
        e.process(PeerCommand::SetActiveLayer(layer));
        common::paint(
            &mut e,
            [0.8, 0.3, 0.2],
            10.0,
            &[Vec2::new(20.0, y), Vec2::new(140.0, y)],
        );
    }
    e.process(PeerCommand::SetActiveLayer(A));
    let _ = e.take_outbox();
    Some(e.document_file())
}

/// A small, high-contrast grayscale height map — a canvas substrate with real grain
/// and a trivial decode (see [`base`]).
///
/// Deterministic: a fixed hash over the coordinates, so every run of this test
/// tooths against the same relief and a failure is reproducible. Irregular rather
/// than a regular weave, for the reason `stark_testdata::assets::rough` is the one
/// the tooth's own tests use — a regular substrate's bearing curve is a few discrete
/// levels, which is a weaker thing to gate a deposit by.
fn grain() -> Vec<u8> {
    const DIM: u32 = 64;
    let mut pixels = Vec::with_capacity((DIM * DIM) as usize);
    for y in 0..DIM {
        for x in 0..DIM {
            let mut h = u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(y);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            pixels.push((h & 0xFF) as u8);
        }
    }
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, DIM, DIM);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(&pixels).expect("png body");
    writer.finish().expect("png finish");
    out
}

/// A fresh document at [`base`], ready to be handed actions.
fn peer(file: &DocumentFile) -> Option<Engine> {
    let mut e = engine_or_skip()?;
    e.join_collaboration(file, ACTOR);
    let _ = e.take_outbox();
    Some(e)
}

/// The action a command commits, minted once so the pair below replays one payload
/// rather than re-deriving it (see the module note on `seed`).
fn mint(file: &DocumentFile, command: impl Into<InputCommand>) -> Option<Action> {
    let mut e = peer(file)?;
    e.process(command);
    e.take_outbox().into_iter().next()
}

/// The stroke a gesture commits — a command cannot spell one.
fn mint_stroke(file: &DocumentFile, y: f32) -> Option<Action> {
    let mut e = peer(file)?;
    let mut brush = common::brush([0.2, 0.4, 0.9], 6.0);
    // A biting tooth, so the substrate actually reaches the paint: with the inert
    // default (`give` at 1.0) a stroke is not a function of the substrate at all and
    // the most interesting row in the table would be vacuous.
    brush.tooth.give = 0.25;
    // Short and narrow on purpose: this table is quadratic in its rows and every
    // stroke row is rendered twice per pair, so the stroke is sized to be a stroke
    // and no larger.
    common::stroke_with(&mut e, brush, &[Vec2::new(30.0, y), Vec2::new(96.0, y)]);
    e.take_outbox().into_iter().next()
}

/// `action` under a chosen order key — the payload untouched.
fn at(action: &Action, lamport: u64) -> Action {
    Action {
        id: ActionId {
            lamport,
            actor: ACTOR,
        },
        kind: action.kind.clone(),
    }
}

/// The document that results from applying `first` and then `second`.
fn ordered(file: &DocumentFile, first: &Action, second: &Action) -> Option<RgbaImage> {
    let mut e = peer(file)?;
    e.merge_remote(at(first, 100));
    e.merge_remote(at(second, 101));
    Some(e.render_to_image())
}

/// One of most things the vocabulary can do, as `(name, action)`.
///
/// Not every kind: the ones left out are the ones that cannot appear in a *pair* at
/// all. `Undo` is resolved by the timeline and never materialized (§5.4); every guide
/// edit writes the one coarse `Resource::Guides`, so guides only ever conflict with
/// each other and a guide's identity is its own action id, which re-stamping would
/// move. What is here is everything that paints, gates, restructures or presents.
fn vocabulary(file: &DocumentFile) -> Vec<(&'static str, Action)> {
    let ramp = Parcel::Solid(Srgb::new([0.2, 0.5, 0.9]));
    let mut out = Vec::new();
    let mut push = |name: &'static str, action: Option<Action>| {
        if let Some(a) = action {
            out.push((name, a));
        }
    };

    push("stroke-high", mint_stroke(file, 60.0));
    push("stroke-low", mint_stroke(file, 200.0));
    push(
        "fill",
        mint(
            file,
            DocCommand::Fill {
                layer: A,
                op: FillOp::new(
                    SelectionShape::rect_from_corners(Vec2::splat(40.0), Vec2::splat(120.0)),
                    2.0,
                    Srgb::new([0.1, 0.7, 0.3]),
                    0.8,
                ),
            },
        ),
    );
    push(
        "select",
        mint(
            file,
            DocCommand::Select(SelectionOp::new(
                SelectionMode::Replace,
                SelectionShape::rect_from_corners(Vec2::splat(30.0), Vec2::splat(200.0)),
                3.0,
            )),
        ),
    );
    push("invert-selection", mint(file, DocCommand::InvertSelection));
    push(
        "selection-opacity",
        mint(file, DocCommand::SetSelectionOpacity(0.4)),
    );
    push(
        "substrate-scale",
        mint(
            file,
            DocCommand::SetSubstrateScale(SubstrateScale::new(200)),
        ),
    );
    push(
        "substrate-color",
        mint(
            file,
            DocCommand::SetSubstrateColor(Srgb::new([0.9, 0.85, 0.7])),
        ),
    );
    push(
        "blend",
        mint(file, DocCommand::SetLayerBlend(A, BlendMode::Multiply)),
    );
    push("opacity", mint(file, DocCommand::SetLayerOpacity(A, 0.5)));
    push("visible", mint(file, DocCommand::SetLayerVisible(A, false)));
    push(
        "clip",
        mint(file, DocCommand::SetLayerClip(LayerId(1), true)),
    );
    push(
        "rename",
        mint(file, DocCommand::SetLayerName(A, Some("wash".into()))),
    );
    push(
        "add-layer",
        mint(
            file,
            DocCommand::AddLayer {
                carrier: None,
                above: None,
            },
        ),
    );
    push(
        "remove-layer",
        mint(file, DocCommand::RemoveLayer(LayerId(1))),
    );
    push("duplicate", mint(file, DocCommand::DuplicateLayer(A)));
    push(
        "move-layer",
        mint(
            file,
            DocCommand::MoveLayer {
                id: A,
                carrier: None,
                at: Place::Top,
            },
        ),
    );
    push(
        "transform",
        mint(
            file,
            DocCommand::Transform {
                layer: A,
                map: TransformMap::Affine(Affine2::from_translation(Vec2::new(12.0, -7.0))),
            },
        ),
    );
    push(
        "add-matte",
        mint(
            file,
            DocCommand::AddMatte {
                carrier: None,
                at: Place::Bottom,
                region: MatteRegion::Everything,
                paint: ramp,
            },
        ),
    );
    push(
        "add-filter",
        mint(
            file,
            DocCommand::AddFilter {
                carrier: None,
                above: None,
                filter: Filter::Color(ColorAdjust {
                    exposure: 0.7,
                    ..ColorAdjust::NEUTRAL
                }),
            },
        ),
    );
    out
}

/// **Every pair the vocabulary claims commutes, actually commutes.**
///
/// Both orders, pixel for pixel, tolerance zero — a commuting pair produces the
/// *same document*, not a similar one, because agreeing on pixels is the whole claim
/// (§12.6).
///
/// The pairs whose footprints conflict are skipped rather than asserted about: those
/// are the ones the history already refuses to splice, so what they do in two orders
/// is not a promise anyone made. The count of each is reported, so a change that
/// quietly made everything conflict — which would make this pass vacuously — shows up
/// as the commuting count collapsing.
#[test]
fn a_pair_that_claims_to_commute_does() {
    let Some(file) = base() else {
        return;
    };
    let vocab = vocabulary(&file);
    assert!(
        vocab.len() >= 18,
        "the table lost rows: {} actions minted",
        vocab.len(),
    );

    let (mut commuting, mut conflicting) = (0usize, 0usize);
    for (i, (a_name, a)) in vocab.iter().enumerate() {
        for (b_name, b) in vocab.iter().skip(i + 1) {
            if compute_footprint(a).conflicts(&compute_footprint(b)) {
                conflicting += 1;
                continue;
            }
            commuting += 1;
            let (Some(ab), Some(ba)) = (ordered(&file, a, b), ordered(&file, b, a)) else {
                return;
            };
            assert!(
                images_match(&ab, &ba, 0),
                "{a_name} and {b_name} are declared to commute, and do not.\n\
                 {a_name} footprint: {:?}\n{b_name} footprint: {:?}",
                compute_footprint(a),
                compute_footprint(b),
            );
        }
    }

    // A vocabulary that conflicted with itself everywhere would pass the loop above
    // having checked nothing, so the shape of the table is asserted too.
    assert!(
        commuting >= 40,
        "only {commuting} pairs claim to commute ({conflicting} conflict); \
         this test is close to vacuous",
    );
}
