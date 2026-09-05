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
//! Quadratic in its table, and seconds rather than the minutes it was: measure before
//! quoting a number, but it is ~5 s on the box these words were last checked on.
//!
//! **The engines were the cost, and it was `Engine::new`, not the replay.**
//! Constructing one is ~140 ms of pipelines; joining the seven-action base log into
//! one that already exists is ~2 ms, and a render is ~1.5 ms. Six documents per
//! commuting pair at 184 pairs is ~1100 of them, which was the whole 150 s. So the
//! table builds **one** engine and re-joins it per trial — `join_collaboration`
//! replaces the document wholesale, and that a re-joined peer draws what a fresh one
//! draws is asserted at the end of the test rather than assumed.
//!
//! What remains is the joins, so the substrate is still synthesized rather than
//! decoded from the bundled 2.6 MB map ([`base`]) — a join decodes the file's content
//! — and the strokes are still kept small.
//!
//! It runs unignored, which at ~150 s it could not. That matters more here than the
//! seconds: this is the only instrument for the *reads* half of footprint honesty
//! (`tests/footprint.rs` proves the writes half and says so), and a correctness gate
//! nothing runs is one that reports `ok` having checked nothing — the failure
//! CLAUDE.md names twice. The `.config/nextest.toml` window is what it runs behind:
//! ~5 s is a real adapter's number, and CI renders on lavapipe.

mod common;

use common::{engine_or_skip, images_match};
use stark_engine::command::{DocCommand, InputCommand, PeerCommand};
use stark_engine::{Engine, RgbaImage};
use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, BlendMode, ColorAdjust, FillOp, Filter, LayerId,
    MatteRegion, Parcel, Place, SelectionMode, SelectionOp, SelectionShape, TransformMap,
    compute_footprint,
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

const A: LayerId = LayerId::ROOT;

/// The layer that **carries** [`INNER`] — a group, so the table can ask what a
/// removal that takes a subtree commutes with (§14.2).
const GROUP: LayerId = LayerId::solo(1);

/// A layer inside [`GROUP`], painted on. The pair `remove-group` × anything naming
/// this is the one a footprint can get wrong without any pixel saying so: a removal
/// that declared only its own id and `StackOrder` was judged to commute with a
/// stroke in here, and the fast-path undo then put the pre-stroke subtree back.
const INNER: LayerId = LayerId::solo(2);

/// A document both runs start from: two layers with paint on them, one of them
/// carrying a third that is also painted, a selection, and a picture placed — enough
/// that the pairs below have something to act on and something to disturb.
///
/// Takes `e` **before anything else has**: it shares the engine's current document
/// rather than joining a file, so an engine already carrying a painting would put it
/// in the base.
fn base(e: &mut Engine) -> DocumentFile {
    e.start_collaboration(ACTOR);
    // A substrate with real relief, so the deposition tooth has something to read
    // (§6.4). On the flat builtin a stroke is not a function of the substrate at
    // all, and the pairs most worth asking about would be vacuous.
    //
    // Synthesized rather than `stark_testdata::assets::rough()`, and that is a cost
    // decision this table has to make: a join decodes the file's content, and this
    // test makes upwards of a thousand of them. The bundled rough map is 2.6 MB and
    // cost more than the rest of the table put together. What the tooth needs is
    // *relief*, not resolution.
    let grain = e
        .import_substrate(&grain())
        .expect("the synthesized height map imports");
    e.process(DocCommand::SetSubstrate(grain));
    e.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    for (layer, y) in [(A, 90.0f32), (LayerId::solo(1), 150.0)] {
        e.process(PeerCommand::SetActiveLayer(layer));
        common::paint(
            e,
            [0.8, 0.3, 0.2],
            10.0,
            &[Vec2::new(20.0, y), Vec2::new(140.0, y)],
        );
    }
    // A third layer, carried by the second, painted on — which is what makes
    // `GROUP` a group and lets the table hold a subtree removal against an edit
    // inside the subtree.
    e.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    e.process(DocCommand::MoveLayer {
        id: INNER,
        carrier: Some(GROUP),
        at: Place::Top,
    });
    e.process(PeerCommand::SetActiveLayer(INNER));
    common::paint(
        e,
        [0.2, 0.5, 0.8],
        10.0,
        &[Vec2::new(30.0, 200.0), Vec2::new(150.0, 200.0)],
    );
    e.process(PeerCommand::SetActiveLayer(A));
    let _ = e.take_outbox();
    e.document_file()
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

/// Put `e` on `file` as [`ACTOR`], ready to be handed actions.
///
/// A re-join, not a construction: a joiner replaces the document wholesale, and
/// `Engine::new` is two orders of magnitude dearer than the replay (see the module
/// note on cost). Draining the outbox is what makes the peer's *own* commits the only
/// thing a caller can take out of it.
fn peer(e: &mut Engine, file: &DocumentFile) {
    e.join_collaboration(file, ACTOR)
        .expect("join a session this build can render");
    let _ = e.take_outbox();
}

/// The action a command commits, minted once so the pair below replays one payload
/// rather than re-deriving it (see the module note on `seed`).
fn mint(e: &mut Engine, file: &DocumentFile, command: impl Into<InputCommand>) -> Option<Action> {
    peer(e, file);
    e.process(command);
    e.take_outbox().into_iter().next()
}

/// [`mint_stroke`] aimed at a **named layer** rather than at whichever one the peer
/// opens on — what a row inside a group needs, since the point of it is the layer.
fn mint_on(e: &mut Engine, file: &DocumentFile, layer: LayerId, path: &[Vec2]) -> Option<Action> {
    peer(e, file);
    e.process(PeerCommand::SetActiveLayer(layer));
    let mut brush = common::brush([0.9, 0.6, 0.1], 6.0);
    brush.tooth.give = 0.25;
    common::stroke_with(e, brush, path);
    e.take_outbox().into_iter().next()
}

/// The stroke a gesture commits — a command cannot spell one.
fn mint_stroke(e: &mut Engine, file: &DocumentFile, y: f32) -> Option<Action> {
    peer(e, file);
    let mut brush = common::brush([0.2, 0.4, 0.9], 6.0);
    // A biting tooth, so the substrate actually reaches the paint: with the inert
    // default (`give` at 1.0) a stroke is not a function of the substrate at all and
    // the most interesting row in the table would be vacuous.
    brush.tooth.give = 0.25;
    // Short and narrow on purpose: this table is quadratic in its rows and every
    // stroke row is rendered twice per pair, so the stroke is sized to be a stroke
    // and no larger.
    common::stroke_with(e, brush, &[Vec2::new(30.0, y), Vec2::new(96.0, y)]);
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
fn ordered(e: &mut Engine, file: &DocumentFile, first: &Action, second: &Action) -> RgbaImage {
    peer(e, file);
    e.merge_remote(at(first, 100));
    e.merge_remote(at(second, 101));
    e.render_to_image()
}

/// The **inverse** half of the commutation claim: `first` applied, `second` applied
/// over it, then `first` undone — beside the canonical materialization of the log
/// that leaves behind.
///
/// This is the property the fast path actually rests on, and it is *not* the one
/// [`ordered`] checks. `history`'s `Centralizer` contract is about the inverse:
///
/// > if `Centralizer::for_action(a).commutes(b)`, then for all `s`,
/// > `a.inverse(b.apply(a.apply(s)))` must be equivalent to `b.apply(s)`.
///
/// Swapping two actions and getting the same picture does not imply it, and a group
/// removal is the case that shows the gap: it wipes the paint whichever order it runs
/// in, so the swap agrees — while the splice restores the subtree from a record taken
/// before the stroke inside it, so the paint comes back and a canonical replay says
/// it should not. That pair is why this exists. It is the shape every under-declared
/// footprint produces, and [`ordered`] alone reports it as fine.
///
/// The undo is an explicit `ActionKind::Undo` naming `first` rather than
/// `DocCommand::Undo`, because the table keeps one actor throughout (see [`ACTOR`])
/// and a local undo would take the *later* of the two.
fn spliced_and_canonical(
    e: &mut Engine,
    file: &DocumentFile,
    first: &Action,
    second: &Action,
) -> (RgbaImage, RgbaImage) {
    peer(e, file);
    let first = at(first, 100);
    e.merge_remote(first.clone());
    e.merge_remote(at(second, 101));
    e.merge_remote(Action {
        id: ActionId {
            lamport: 102,
            actor: ACTOR,
        },
        kind: ActionKind::Undo(first.id),
    });
    // The spliced log, taken before the re-join below empties the document.
    let log = e.document_file();
    let spliced = e.render_to_image();
    // A peer joining the same log rewinds nothing and splices nothing, so what it
    // draws is what the log *means* — the comparison §12.6 makes convergence out of,
    // and the one `commute.rs` holds its own scenarios to. It is the same engine
    // because a join replaces the document wholesale; that it is equivalent to a
    // fresh one is asserted at the end of the test rather than assumed.
    //
    // **Joining as [`ACTOR`], not as a stranger.** A selection is per-author (§17.3)
    // and its outline is chrome drawn for its *owner* — `show_peer_selections` is off
    // by default — so a canonical peer with an actor of its own would render every
    // selection row without the outline the spliced engine draws, and report a
    // difference of 216 levels that is about who is looking rather than about what
    // the log says. The materialization is identical either way; only the viewpoint
    // has to match.
    peer(e, &log);
    (spliced, e.render_to_image())
}

/// One of most things the vocabulary can do, as `(name, action)`.
///
/// Not every kind: the ones left out are the ones that cannot appear in a *pair* at
/// all. `Undo` is resolved by the timeline and never materialized (§5.4); every guide
/// edit writes the one coarse `Resource::Guides`, so guides only ever conflict with
/// each other and a guide's identity is its own action id, which re-stamping would
/// move. What is here is everything that paints, gates, restructures or presents.
fn vocabulary(e: &mut Engine, file: &DocumentFile) -> Vec<(&'static str, Action)> {
    let ramp = Parcel::Solid(Srgb::new([0.2, 0.5, 0.9]));
    let mut out = Vec::new();
    let mut push = |name: &'static str, action: Option<Action>| {
        if let Some(a) = action {
            out.push((name, a));
        }
    };

    // **One stroke, not two.** A stroke is the most expensive row — it is rendered
    // twice for every commuting pair it appears in — and the second one was here to
    // ask whether two strokes on disjoint tiles commute, which
    // `commute.rs::undo_and_redo_splice_past_disjoint_peer_strokes` already asks
    // end-to-end. This table's job is breadth across the vocabulary; where a pair is
    // covered better elsewhere, it does not need to pay for it here as well.
    push("stroke", mint_stroke(e, file, 60.0));
    push(
        "fill",
        mint(
            e,
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
            e,
            file,
            DocCommand::Select(SelectionOp::new(
                SelectionMode::Replace,
                SelectionShape::rect_from_corners(Vec2::splat(30.0), Vec2::splat(200.0)),
                3.0,
            )),
        ),
    );
    push(
        "invert-selection",
        mint(e, file, DocCommand::InvertSelection),
    );
    push(
        "selection-opacity",
        mint(e, file, DocCommand::SetSelectionOpacity(0.4)),
    );
    push(
        "substrate-scale",
        mint(
            e,
            file,
            DocCommand::SetSubstrateScale(SubstrateScale::new(200)),
        ),
    );
    push(
        "substrate-color",
        mint(
            e,
            file,
            DocCommand::SetSubstrateColor(Srgb::new([0.9, 0.85, 0.7])),
        ),
    );
    push(
        "blend",
        mint(e, file, DocCommand::SetLayerBlend(A, BlendMode::Multiply)),
    );
    push(
        "opacity",
        mint(e, file, DocCommand::SetLayerOpacity(A, 0.5)),
    );
    push(
        "visible",
        mint(e, file, DocCommand::SetLayerVisible(A, false)),
    );
    push(
        "clip",
        mint(e, file, DocCommand::SetLayerClip(LayerId::solo(1), true)),
    );
    push(
        "rename",
        mint(e, file, DocCommand::SetLayerName(A, Some("wash".into()))),
    );
    push(
        "add-layer",
        mint(
            e,
            file,
            DocCommand::AddLayer {
                carrier: None,
                above: None,
            },
        ),
    );
    // A **group** removal, which takes `INNER` with it: the row whose footprint has
    // to name the whole subtree, and the reason the three rows after it exist.
    push(
        "remove-group",
        mint(e, file, DocCommand::RemoveLayer(GROUP)),
    );
    // Edits *inside* that group. Each of these names a layer the removal never
    // names, so each is a pair the old footprint declared commuting — a stroke,
    // a property, and the layer's own departure by a second route.
    push(
        "paint-in-group",
        mint_on(
            e,
            file,
            INNER,
            &[Vec2::new(40.0, 205.0), Vec2::new(120.0, 195.0)],
        ),
    );
    push(
        "opacity-in-group",
        mint(e, file, DocCommand::SetLayerOpacity(INNER, 0.4)),
    );
    push(
        "rename-in-group",
        mint(
            e,
            file,
            DocCommand::SetLayerName(INNER, Some("inner".into())),
        ),
    );
    push("duplicate", mint(e, file, DocCommand::DuplicateLayer(A)));
    push(
        "move-layer",
        mint(
            e,
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
            e,
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
            e,
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
            e,
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
    // **One engine for the whole table**, re-joined per trial. Every document below
    // is a `join_collaboration` away, and that is the difference between minutes and
    // seconds here (see the module note on cost). That a re-joined peer is
    // indistinguishable from a fresh one is asserted at the end, not assumed.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let file = base(&mut engine);
    let vocab = vocabulary(&mut engine, &file);
    assert!(
        vocab.len() >= 20,
        "the table lost rows: {} actions minted",
        vocab.len(),
    );

    let (mut commuting, mut conflicting) = (0usize, 0usize);
    // The last pair checked, kept for the reuse assertion below.
    let mut last: Option<(&str, &str, &Action, &Action, RgbaImage)> = None;
    for (i, (a_name, a)) in vocab.iter().enumerate() {
        for (b_name, b) in vocab.iter().skip(i + 1) {
            if compute_footprint(a).conflicts(&compute_footprint(b)) {
                conflicting += 1;
                continue;
            }
            commuting += 1;
            let ab = ordered(&mut engine, &file, a, b);
            let ba = ordered(&mut engine, &file, b, a);
            assert!(
                images_match(&ab, &ba, 0),
                "{a_name} and {b_name} are declared to commute, and do not.\n\
                 {a_name} footprint: {:?}\n{b_name} footprint: {:?}",
                compute_footprint(a),
                compute_footprint(b),
            );
            // …and the half the fast path actually rests on, in both directions,
            // since either of the pair may be the one undone
            // ([`spliced_and_canonical`]).
            for (x_name, x, y_name, y) in [(a_name, a, b_name, b), (b_name, b, a_name, a)] {
                let (spliced, canonical) = spliced_and_canonical(&mut engine, &file, x, y);
                assert!(
                    images_match(&spliced, &canonical, 0),
                    "undoing {x_name} past {y_name} — declared to commute — does not \
                     match a canonical replay of the same log.\n\
                     {x_name} footprint: {:?}\n{y_name} footprint: {:?}",
                    compute_footprint(x),
                    compute_footprint(y),
                );
            }
            last = Some((a_name, b_name, a, b, ab));
        }
    }

    // A vocabulary that conflicted with itself everywhere would pass the loop above
    // having checked nothing, so the shape of the table is asserted too.
    assert!(
        commuting >= 35,
        "only {commuting} pairs claim to commute ({conflicting} conflict); \
         this test is close to vacuous",
    );

    // **The premise the whole table rests on, asked of the thing that depends on it.**
    // Every document above came out of re-joining one engine, which is only the same
    // as building one per trial if a join leaves nothing of the last document behind.
    // So the last pair is asked again of an engine that has rendered nothing: a
    // difference here would mean every comparison above was between two states of one
    // dirty engine rather than between two materializations of a log.
    let (a_name, b_name, a, b, ab) = last.expect("a commuting pair was checked");
    let mut fresh = engine_or_skip().expect("the device this run already built an engine on");
    assert!(
        images_match(&ab, &ordered(&mut fresh, &file, a, b), 0),
        "{a_name} then {b_name} renders differently on the re-joined engine than on a \
         fresh one — this test's engine reuse is unsound, and every result above with it",
    );
}
