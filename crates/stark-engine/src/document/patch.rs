//! [`unapply`]: the implementation of `Action::inverse` (§12.6) —
//! remove one action's effect from a state by restoring what it wrote from the
//! state it was originally applied to.
//!
//! The two states handed in are *not* adjacent: the history calls this while
//! shifting an undone action past later actions it commutes with, so `state`
//! contains their work too. Commutation is exactly what makes the restore
//! sound — nothing in between touched the action's footprint, so inside it
//! every difference belongs to the action, and outside it nothing may be
//! touched (see [`tile_diff`]). Restored tile values are `Arc` handles shared
//! with `previous`, so removal re-renders nothing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rpds::Vector;

use super::layer::{Layer, LayerContent};
use super::selection::Selection;
use super::state::{DocState, Guide, LayerSite};
use crate::gpu::tile::TilePairHandle;
use stark_model::Srgb;
use stark_model::document::Filter;
use stark_model::document::{Action, ActorId};
use stark_model::document::{BlendMode, LayerId, MatteRegion};
use stark_model::document::{Footprint, Prop, Resource};
use stark_model::geom::{TileCoord, TileRect};
use stark_model::{SubstrateId, SubstrateScale};

/// One restorable write. Each variant covers exactly one [`footprint
/// resource`](stark_model::document::Resource), never more — a commuting action in
/// between may own any *other* resource of the same layer, and restoring it
/// would clobber that edit.
enum PatchOp {
    /// A layer's tile entries: `None` means the entry did not exist.
    Tiles {
        layer: LayerId,
        tiles: Vec<(TileCoord, Option<TilePairHandle>)>,
    },
    /// The layer did not exist (undoes an add).
    Absent(LayerId),
    /// The layer existed, as this record, at this place in the tree (undoes a
    /// remove). The record carries whatever it carried — a `Layer` owns its
    /// subtree — so removing a group and undoing it restores the group
    /// (§14.8).
    Present {
        site: LayerSite,
        layer: Layer,
    },
    /// The layer tree had this **shape** (undoes a move): every layer in
    /// composite order, each with the layer carrying it.
    ///
    /// One op restoring the whole structure, exactly as the flat order it
    /// replaces did. It has to be the whole shape rather than a per-layer
    /// carrier, because a move can change several layers' relative positions at
    /// once and `StackOrder` is the single coarse resource that covers all of
    /// them (`footprint.rs`).
    Structure(Vec<(LayerId, Option<LayerId>)>),
    Blend(LayerId, BlendMode),
    Clip(LayerId, bool),
    Opacity(LayerId, f32),
    Visible(LayerId, bool),
    Name(LayerId, Option<Arc<str>>),
    /// Where the layer's frame sat on the canvas (§14.12).
    Translation(LayerId, stark_model::geom::IVec2),
    /// A matte's region and color together — one footprint resource.
    Matte(LayerId, MatteRegion, stark_model::document::Parcel),
    /// A filter layer's settings (§21) — one footprint resource, because the action
    /// that writes them carries the filter entire.
    Filter(LayerId, Filter),
    Selection(ActorId, Selection),
    /// The canvas substrate **and the size it is laid at** — one op, because
    /// `Resource::Substrate` is one resource for the two (§6.4): the tooth reads the
    /// substrate's rise over a reach in canvas px, so which substrate and how large
    /// it is laid decide a deposit together.
    ///
    /// Carrying only the id restored only the id, so undoing a `SetSubstrateScale`
    /// through the commuting splice put the substrate back and left the *scale* where
    /// the undone action had set it — a document holding a scale its own log no
    /// longer contains, and a later stroke toothed against it. The two fields the
    /// resource names are the two fields the op carries.
    Substrate(SubstrateId, SubstrateScale),
    SubstrateColor(Srgb),
    /// The **whole drawing-guide roster** (§20.5): every guide and the order they
    /// were arranged in.
    ///
    /// One op for all five guide actions, because `Resource::Guides` is one
    /// resource for all five — the same correspondence `Structure` keeps with
    /// `StackOrder`, and cheap for the same kind of reason it is not there: the
    /// roster is a persistent vector, so capturing it whole is an `Arc` bump
    /// rather than a walk.
    Guides(Vector<Guide>),
}

impl PatchOp {
    /// `state` with this op's value written back — the other half of
    /// [`capture_resource`], and the reason both are per-op rather than per-action:
    /// what a resource *is* decides how it is recorded and how it is put back, and
    /// the two belong next to each other rather than in two matches over
    /// `ActionKind`.
    fn restore(&self, state: &DocState) -> DocState {
        match self {
            PatchOp::Tiles { layer, tiles } => {
                match state.layer(*layer).and_then(|l| l.tiles()).cloned() {
                    Some(mut map) => {
                        for (coord, handle) in tiles {
                            map = match handle {
                                Some(handle) => map.insert(*coord, handle.clone()),
                                None => map.remove(coord),
                            };
                        }
                        state.map_layer(*layer, |l| l.with_tiles(map))
                    }
                    None => state.clone(),
                }
            }
            PatchOp::Absent(id) => state.remove_layer(*id),
            PatchOp::Present { site, layer } => state.restore_layer(site, layer.clone()),
            PatchOp::Structure(shape) => restore_structure(state, shape),
            PatchOp::Blend(id, v) => state.set_layer_blend(*id, *v),
            PatchOp::Clip(id, v) => state.set_layer_clip(*id, *v),
            PatchOp::Opacity(id, v) => state.set_layer_opacity(*id, *v),
            PatchOp::Visible(id, v) => state.set_layer_visible(*id, *v),
            PatchOp::Name(id, v) => state.set_layer_name(*id, v.clone()),
            PatchOp::Translation(id, v) => state.translate_layers(&[(*id, *v)]),
            // The *value*, not the rect: a region restored through its rect could
            // not round-trip `Everything`, which has none.
            PatchOp::Matte(id, region, paint) => state
                .set_matte_region(*id, *region)
                .set_matte_paint(*id, paint.clone()),
            PatchOp::Filter(id, filter) => state.set_filter(*id, filter.clone()),
            PatchOp::Selection(actor, selection) => state.with_selection(*actor, selection.clone()),
            PatchOp::Substrate(id, scale) => state.with_substrate(*id).with_substrate_scale(*scale),
            PatchOp::SubstrateColor(rgb) => state.with_substrate_color(*rgb),
            PatchOp::Guides(guides) => state.with_guides(guides.clone()),
        }
    }
}

/// `state` with every resource `action` writes restored to the value it holds
/// in `previous` — the state the action was originally applied to. Valid
/// exactly when every action applied between the two commutes with `action`
/// (the `Action::inverse` contract, which the history's commutation gate
/// upholds).
pub fn unapply(
    action: &Action,
    footprint: &Footprint,
    previous: &DocState,
    state: &DocState,
) -> DocState {
    StatePatch::capture(action, footprint, previous, state).restore(state)
}

/// The written resources of one action, with the values they held in a chosen
/// state — apply with [`restore`](StatePatch::restore) to put those values back.
struct StatePatch {
    ops: Vec<PatchOp>,
}

impl StatePatch {
    /// The patch that rewrites what `action` writes back to the values `to`
    /// holds, diffed against `from` so untouched entries cost nothing.
    ///
    /// **Driven by the footprint's own write list**, resource for resource, rather
    /// than by a second match on [`ActionKind`](stark_model::document::ActionKind). That is
    /// what makes "a patch restores exactly what the action declared" true by
    /// construction instead of by inspection. As parallel matches in two files there
    /// is nothing but prose between them, and Rust's exhaustiveness gets the
    /// *presence* of an arm, never its *correspondence*; driven by the write list, a
    /// kind whose footprint grows a resource grows the op that puts it back in the
    /// same edit.
    ///
    /// `Undo` falls out rather than needing an arm: it is never materialized, which
    /// is why its footprint is empty, so it captures nothing.
    fn capture(
        action: &Action,
        footprint: &Footprint,
        to: &DocState,
        from: &DocState,
    ) -> StatePatch {
        let mut ops = Vec::new();
        // **Existence first**, whatever order a footprint happens to list its writes
        // in. A layer has to be back in the tree before anything can put its tiles,
        // its properties or the tree's shape right — `restore_structure` in
        // particular arranges records it does not create, and drops a layer the shape
        // names but the state has lost. Partitioning here rather than relying on the
        // footprints being written that way is what keeps that from being a rule
        // `footprint.rs` has to remember.
        // The **cached** write list the `Logged` carries, handed down through
        // `Materialize::unfold`. Derived here instead, it was a fresh derivation per
        // `inverse` — once per cached state per shift, and for a `TransformWarp` a
        // whole fine-lattice solve — which is the cost `Logged` holds a footprint to
        // avoid.
        //
        // `action` is therefore unread: a patch is built from *what was written*, and
        // the footprint is the whole of that. It stays in the signature because
        // `Materialize::unfold` hands it down and a caller reading this line should
        // see that ignoring it is the design rather than an omission.
        let _ = action;
        let (existence, rest): (Vec<&Resource>, Vec<&Resource>) = footprint
            .writes
            .iter()
            .partition(|r| matches!(r, Resource::Existence(_)));
        for resource in existence.into_iter().chain(rest) {
            capture_resource(resource, to, from, &mut ops);
        }
        StatePatch { ops }
    }

    /// Write the captured values back into `state`.
    fn restore(&self, state: &DocState) -> DocState {
        let mut state = state.clone();
        for op in &self.ops {
            state = op.restore(&state);
        }
        state
    }
}

/// Record what `to` holds for one written resource — the map from the
/// [`Footprint`] vocabulary to the [`PatchOp`] that
/// puts that resource back, and the whole of the correspondence [`StatePatch::capture`]
/// rests on.
///
/// Silence is a real answer here and it means "nothing of this action's to restore":
/// a layer absent on both sides (the action no-oped), a `Matte` prop on something
/// that is not a matte, a `Filter` prop on something that is not a filter.
fn capture_resource(resource: &Resource, to: &DocState, from: &DocState, ops: &mut Vec<PatchOp>) {
    match resource {
        Resource::Paint(layer, rect) => tile_diff(*layer, *rect, to, from, ops),
        Resource::Existence(id) => match (to.site_of(*id), from.contains_layer(*id)) {
            (None, true) => ops.push(PatchOp::Absent(*id)),
            (Some(site), false) => ops.push(PatchOp::Present {
                site,
                layer: to.layer(*id).expect("sited layer exists").clone(),
            }),
            // Present or absent on both sides: the action no-oped.
            _ => {}
        },
        Resource::Prop(id, prop) => {
            let Some(l) = to.layer(*id) else {
                return;
            };
            ops.push(match prop {
                Prop::Blend => PatchOp::Blend(*id, l.composite.blend),
                Prop::Clip => PatchOp::Clip(*id, l.composite.clip),
                Prop::Opacity => PatchOp::Opacity(*id, l.composite.opacity),
                Prop::Visible => PatchOp::Visible(*id, l.visible),
                Prop::Name => PatchOp::Name(*id, l.name.clone()),
                Prop::Matte => match &l.content {
                    LayerContent::Matte { region, paint } => {
                        PatchOp::Matte(*id, *region, paint.clone())
                    }
                    LayerContent::Paint(_) | LayerContent::Filter(_) => return,
                },
                Prop::Filter => match l.filter() {
                    Some(f) => PatchOp::Filter(*id, f),
                    None => return,
                },
                Prop::Translation => PatchOp::Translation(*id, l.translation),
            });
        }
        // The coarse claim expands into the fine ones it stands for, so a footprint
        // that *writes* a whole layer restores a whole layer.
        //
        // **Five kinds write one**, and this arm is on the undo path of every one of
        // them: `AddLayer`/`AddFilter`, `PlaceImage`, `AddMatte`, `DuplicateLayer`
        // (one per copy) and `RemoveLayer` (one per layer in the subtree). It is the
        // claim every action that mints a layer makes over the id it minted, and the
        // one a removal makes over each id it took away — see `compute_footprint`.
        //
        // The subtree case is why `restore_layer` refuses an id already present. A
        // group removal claims the group *and* each of its descendants, and the
        // group's own `Existence` op puts the whole subtree back before the
        // descendants' ops are read; each of those would otherwise insert a layer that
        // is already there. Nine ops per claimed layer is also what this costs, which
        // is the price of expanding rather than restoring the layer whole.
        Resource::Layer(id) => {
            capture_resource(&Resource::Existence(*id), to, from, ops);
            capture_resource(&Resource::Paint(*id, TileRect::ALL), to, from, ops);
            for prop in Prop::ALL {
                capture_resource(&Resource::Prop(*id, prop), to, from, ops);
            }
        }
        Resource::StackOrder => ops.push(PatchOp::Structure(structure(to))),
        Resource::Selection(actor) => ops.push(PatchOp::Selection(*actor, to.selection_of(*actor))),
        Resource::Substrate => ops.push(PatchOp::Substrate(to.substrate, to.substrate_scale)),
        Resource::SubstrateColor => ops.push(PatchOp::SubstrateColor(to.substrate_color)),
        Resource::Guides => ops.push(PatchOp::Guides(to.guides().clone())),
    }
}

/// The shape of `state`'s layer tree: every layer in composite order, paired
/// with the layer carrying it (§14.8).
///
/// Composite order matters — a carrier is always recorded before anything it
/// carries — so [`restore_structure`] can rebuild top-down without a sort.
fn structure(state: &DocState) -> Vec<(LayerId, Option<LayerId>)> {
    let mut out = Vec::new();
    let mut stack: Vec<Option<LayerId>> = Vec::new();
    state.visit(&mut |layer, depth| {
        stack.truncate(depth);
        out.push((layer.id, stack.last().copied().flatten()));
        stack.push(Some(layer.id));
    });
    out
}

/// `state` rebuilt into `shape`, keeping each layer's **current record** — its
/// tiles, its name, its opacity.
///
/// That is the whole point of restoring a shape rather than a snapshot: the two
/// states this runs between are not adjacent, and a commuting action in the gap
/// may have painted on a layer or renamed it. Only the tree's shape belongs to
/// the move being undone, so only the shape is put back.
fn restore_structure(state: &DocState, shape: &[(LayerId, Option<LayerId>)]) -> DocState {
    // Every layer, stripped of what it carries — the tree is rebuilt from
    // `shape`, so the old nesting must not travel along inside the records.
    let mut records: HashMap<LayerId, Layer> = HashMap::new();
    state.visit(&mut |l, _| {
        records.insert(l.id, l.with_carries(Vector::new()));
    });

    let mut children: HashMap<Option<LayerId>, Vec<LayerId>> = HashMap::new();
    let mut placed: HashSet<LayerId> = HashSet::new();
    for (id, carrier) in shape {
        // `placed` also guards the rebuild against a shape that names a layer
        // twice, which would otherwise recurse forever.
        if records.contains_key(id) && placed.insert(*id) {
            children.entry(*carrier).or_default().push(*id);
        }
    }
    // Anything the shape predates keeps stacking on top of the root. With a
    // commuting suffix — no structural edits, which is what `StackOrder`
    // guarantees — this arm is empty.
    state.visit(&mut |l, _| {
        if !placed.contains(&l.id) {
            children.entry(None).or_default().push(l.id);
        }
    });

    fn build(
        carrier: Option<LayerId>,
        children: &HashMap<Option<LayerId>, Vec<LayerId>>,
        records: &HashMap<LayerId, Layer>,
    ) -> Vector<Layer> {
        let mut out = Vector::new();
        for id in children.get(&carrier).into_iter().flatten() {
            let record = records[id].with_carries(build(Some(*id), children, records));
            out = out.push_back(record);
        }
        out
    }
    state.with_layers(build(None, &children, &records))
}

/// Record `to`'s value for every tile entry of `layer` **within `rect`** that
/// differs between the two states. Handle identity is change detection:
/// committed tiles are never rewritten in place (see [`TilePairHandle::same`]).
///
/// The rect bound is load-bearing, not an optimization: in the `inverse` use
/// the two states are not adjacent — commuting actions applied in between have
/// changed this layer *outside* the action's footprint, and "restoring" those
/// entries would erase their work. Inside the footprint the gate guarantees
/// every difference belongs to this action alone.
fn tile_diff(
    layer: LayerId,
    rect: TileRect,
    to: &DocState,
    from: &DocState,
    ops: &mut Vec<PatchOp>,
) {
    let tiles_of = |state: &DocState| state.layer(layer).and_then(|l| l.tiles()).cloned();
    let (Some(to_tiles), Some(from_tiles)) = (tiles_of(to), tiles_of(from)) else {
        // Absent layer or matte on either side: the action no-oped on paint.
        return;
    };
    let mut tiles = Vec::new();
    // **Walk whichever side is smaller**: the rect the action declared, or the maps
    // themselves. A stroke claims the handful of tiles it painted while the layer
    // under it may hold thousands, and this runs once per cached state in a shift
    // window — so scanning the map for a rect of four tiles was tens of thousands of
    // lookups to find four. `TileRect::ALL` saturates its count, so a whole-layer
    // claim (a transform, a removal) still takes the map walk, which is the smaller
    // side there.
    //
    // Both walks answer identically: a coord outside `rect` is skipped by one and
    // never visited by the other.
    let span = rect.count();
    if span < to_tiles.size() as u64 + from_tiles.size() as u64 {
        for coord in rect.coords() {
            let (old, new) = (to_tiles.get(&coord), from_tiles.get(&coord));
            match (old, new) {
                (_, Some(h)) if !old.is_some_and(|o| o.same(h)) => {
                    tiles.push((coord, old.cloned()));
                }
                // Arrived where the earlier state had none: restoring means removing.
                (Some(h), None) => tiles.push((coord, Some(h.clone()))),
                _ => {}
            }
        }
    } else {
        for (coord, handle) in from_tiles.iter() {
            if !rect.contains(*coord) {
                continue;
            }
            let old = to_tiles.get(coord);
            if !old.is_some_and(|h| h.same(handle)) {
                tiles.push((*coord, old.cloned()));
            }
        }
        for (coord, handle) in to_tiles.iter() {
            if rect.contains(*coord) && from_tiles.get(coord).is_none() {
                tiles.push((*coord, Some(handle.clone())));
            }
        }
    }
    if !tiles.is_empty() {
        ops.push(PatchOp::Tiles { layer, tiles });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::compute_footprint;

    /// [`unapply`] with the footprint its caller would have carried.
    ///
    /// Production hands the cached one down from the `Logged` (`Materialize::unfold`);
    /// a test builds a bare `Action`, so this is where it is derived. One helper so
    /// the derivation is not written out at every call — and so a test cannot
    /// accidentally hand `unapply` a footprint that is not this action's, which is
    /// the one way the new parameter could be got wrong.
    fn undo(action: &Action, before: &DocState, after: &DocState) -> DocState {
        unapply(action, &compute_footprint(action), before, after)
    }

    /// The action folded into `state` — **the fold itself**, through the half of
    /// [`apply`](super::super::apply) that is a `DocState` call and nothing else (§4).
    ///
    /// Written out here it was a third statement of the same mutation: a match over
    /// `ActionKind` in a test, with nothing tying it to the real one, so a sanitize
    /// or a refusal added to an arm left this whole module round-tripping a fold
    /// nobody runs. That is the hazard [`StatePatch::capture`] argues against one
    /// level up, and it was sitting here.
    ///
    /// A GPU kind answers `None` and there is nothing here to render it with, which
    /// is what makes "this suite drives the ctx-free half" a fact rather than a
    /// convention.
    fn fold(action: &Action, state: &DocState) -> DocState {
        apply_pure(&action.kind, state.clone(), action.id.actor)
            .expect("this suite drives the ctx-free half of the fold")
    }
    use super::super::apply::apply_pure;
    use super::super::audit::undeclared;
    use stark_model::document::Place;
    use stark_model::document::{ActionId, ActionKind, ActionTag, GuideId};
    use stark_model::{AssetId, SubstrateId, SubstrateScale};

    const A: LayerId = LayerId::ROOT;
    const B: LayerId = LayerId::solo(1);
    const C: LayerId = LayerId::solo(2);

    fn act(kind: ActionKind) -> Action {
        Action {
            id: ActionId {
                lamport: 7,
                actor: ActorId(1),
            },
            kind,
        }
    }

    /// The tree's shape, flattened — what a structural assertion compares.
    fn shape(state: &DocState) -> Vec<(LayerId, usize)> {
        let mut out = Vec::new();
        state.visit(&mut |l, depth| out.push((l.id, depth)));
        out
    }

    /// Everything a [`Prop`] can name about one layer, as one comparable value —
    /// so a round trip is asserted on all of them at once and a property the
    /// comparison forgot cannot pass by looking unchanged.
    type Props = (
        super::super::layer::CompositeParams,
        bool,
        Option<Arc<str>>,
        Option<Filter>,
        Option<(MatteRegion, stark_model::document::Parcel)>,
        stark_model::geom::IVec2,
    );

    fn props(l: &Layer) -> Props {
        let matte = match &l.content {
            LayerContent::Matte { region, paint } => Some((*region, paint.clone())),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        };
        (
            l.composite,
            l.visible,
            l.name.clone(),
            l.filter(),
            matte,
            l.translation,
        )
    }

    /// Three layers in the root stack, bottom to top: A, B, C.
    fn flat() -> DocState {
        DocState::with_layer(A)
            .insert_layer(B, None, Some(A))
            .insert_layer(C, None, Some(B))
    }

    /// Every property an action can write, put back by the patch that declared it.
    ///
    /// Driven off [`Prop::ALL`] rather than a list of its own, so a property that
    /// grows a resource cannot quietly go untested: the coverage and the enum are
    /// the same list, and the match below has no `_` arm.
    #[test]
    fn every_property_round_trips() {
        use stark_model::document::{BlendMode, ColorAdjust, Filter};
        use stark_model::document::{MatteRegion, Parcel};
        use stark_model::geom::Vec2;

        let rect = MatteRegion::OutsideRect {
            min: Vec2::ZERO,
            max: Vec2::splat(64.0),
        };
        // One of each kind of layer, so a `Matte` or a `Filter` property has
        // somewhere to land.
        let before = flat()
            .insert_matte(
                LayerId::solo(3),
                None,
                Place::Top,
                rect,
                Parcel::Solid(Srgb::new([0.2, 0.4, 0.6])),
            )
            .insert_filter(
                LayerId::solo(4),
                None,
                None,
                Filter::Color(ColorAdjust::NEUTRAL),
            );

        for prop in Prop::ALL {
            let (kind, target) = match prop {
                Prop::Blend => (ActionKind::SetLayerBlend(B, BlendMode::Multiply), B),
                Prop::Clip => (ActionKind::SetLayerClip(B, true), B),
                Prop::Opacity => (ActionKind::SetLayerOpacity(B, 0.25), B),
                Prop::Visible => (ActionKind::SetLayerVisible(B, false), B),
                Prop::Name => (ActionKind::SetLayerName(B, Some("wash".into())), B),
                Prop::Matte => (
                    ActionKind::SetMattePaint(LayerId::solo(3), Parcel::Solid(Srgb::new([1.0; 3]))),
                    LayerId::solo(3),
                ),
                Prop::Filter => (
                    ActionKind::SetFilter(
                        LayerId::solo(4),
                        Filter::Color(ColorAdjust {
                            saturation: 0.0,
                            ..ColorAdjust::NEUTRAL
                        }),
                    ),
                    LayerId::solo(4),
                ),
                Prop::Translation => (
                    ActionKind::TranslateLayers {
                        moves: vec![(B, stark_model::geom::IVec2::new(508, -254))],
                    },
                    B,
                ),
            };
            let action = act(kind);
            let after = fold(&action, &before);
            let was = props(before.layer(target).expect("target exists"));
            let now = props(after.layer(target).expect("target exists"));
            // …the action really did change something, or the round trip below is
            // vacuous.
            assert_ne!(
                now, was,
                "{prop:?} was a no-op, so its round trip would prove nothing",
            );

            let back = undo(&action, &before, &after);
            let restored = props(back.layer(target).expect("target survives the restore"));
            assert_eq!(restored, was, "{prop:?} did not come back");
        }
    }

    /// **The two states are not adjacent** — the whole subtlety of `Action::inverse`
    /// (§12.6). The history calls this while shifting an undone action past later
    /// ones it commutes with, so the state handed in carries their work too and the
    /// restore must leave it standing.
    ///
    /// Here `B`'s opacity is undone across a rename of `C` that commutes with it. A
    /// patch restoring more than its footprint would take the rename with it.
    #[test]
    fn a_restore_leaves_a_commuting_edit_alone() {
        let before = flat();
        let action = act(ActionKind::SetLayerOpacity(B, 0.25));
        let after = before
            .set_layer_opacity(B, 0.25)
            .set_layer_name(C, Some("sky".into()));

        let back = undo(&action, &before, &after);
        assert_eq!(
            back.layer(B).expect("B").composite.opacity,
            1.0,
            "the undone action's own write has to be restored",
        );
        assert_eq!(
            back.layer_name(C),
            Some("sky"),
            "a commuting edit in the gap has to survive the restore",
        );
    }

    /// Adding a layer is undone by removing it; removing one is undone by putting
    /// the record back **where it was**, subtree and all (§14.8).
    #[test]
    fn existence_round_trips_in_both_directions() {
        let before = flat();
        let added = before.insert_layer(LayerId::solo(9), None, Some(A));
        let back = undo(
            &act(ActionKind::AddLayer {
                id: LayerId::solo(9),
                carrier: None,
                above: Some(A),
            }),
            &before,
            &added,
        );
        assert_eq!(shape(&back), shape(&before), "the add came back out");

        // A group, removed and restored: the subtree travels as one.
        let grouped = flat().move_layer(C, Some(B), Place::Top);
        let removed = grouped.remove_layer(B);
        assert!(removed.layer(C).is_none(), "the subtree went with its base");
        let back = undo(
            &act(ActionKind::RemoveLayer {
                id: B,
                carried: vec![C],
            }),
            &grouped,
            &removed,
        );
        assert_eq!(
            shape(&back),
            shape(&grouped),
            "the group came back at its own site, carrying what it carried",
        );
    }

    /// A move is undone through the tree's whole **shape**, which is the only thing
    /// `StackOrder` can restore — and the rebuild keeps each layer's *current*
    /// record, so a name that arrived in the gap survives it.
    #[test]
    fn a_move_restores_the_shape_and_keeps_current_records() {
        let before = flat();
        let action = act(ActionKind::MoveLayer {
            id: C,
            carrier: Some(A),
            at: Place::Top,
        });
        let after = before
            .move_layer(C, Some(A), Place::Top)
            .set_layer_name(C, Some("sky".into()));

        let back = undo(&action, &before, &after);
        assert_eq!(shape(&back), shape(&before), "the move came back out");
        assert_eq!(
            back.layer_name(C),
            Some("sky"),
            "the rebuild keeps the record it finds, not the one the shape was taken with",
        );
    }

    /// The canvas's own two resources, which belong to the document rather than to
    /// any layer.
    #[test]
    fn the_canvas_round_trips() {
        let before = flat().with_substrate_color(Srgb::new([0.5, 0.5, 0.5]));
        let after = before.with_substrate_color(Srgb::new([0.1, 0.2, 0.3]));
        let back = undo(
            &act(ActionKind::SetSubstrateColor(Srgb::new([0.1, 0.2, 0.3]))),
            &before,
            &after,
        );
        assert_eq!(back.substrate_color, before.substrate_color);

        let after = before.with_substrate(SubstrateId::Flat);
        let back = undo(
            &act(ActionKind::SetSubstrate(SubstrateId::Flat)),
            &before,
            &after,
        );
        assert_eq!(back.substrate, before.substrate);
    }

    /// `Undo` is resolved by the timeline and never materialized, which is *why* its
    /// footprint is empty — so its patch restores nothing at all. With the capture
    /// driven by the footprint (§4) that falls out rather than needing an arm, and
    /// this is what says so.
    #[test]
    fn an_undo_action_captures_nothing() {
        let before = flat();
        let after = before.set_layer_opacity(B, 0.25);
        let back = undo(
            &act(ActionKind::Undo(ActionId {
                lamport: 1,
                actor: ActorId(1),
            })),
            &before,
            &after,
        );
        assert_eq!(
            back.layer(B).expect("B").composite.opacity,
            0.25,
            "an empty patch has to leave the state exactly as it found it",
        );
    }

    /// `restore_structure` guards a shape that names a layer twice — a crafted log,
    /// which the rebuild would otherwise recurse on forever. The guard is the
    /// `placed` set, and nothing else exercised it.
    #[test]
    fn a_shape_naming_a_layer_twice_is_survived() {
        let state = flat();
        let doubled = vec![(A, None), (B, None), (B, None), (C, None)];
        let back = restore_structure(&state, &doubled);
        let ids: Vec<LayerId> = shape(&back).into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 3, "each layer lands exactly once");
        for id in [A, B, C] {
            assert!(ids.contains(&id), "{id:?} survived");
        }
    }

    // -----------------------------------------------------------------------
    // The whole ctx-free half of the fold, round-tripped without an adapter

    const MATTE: LayerId = LayerId::solo(3);
    const FILTER: LayerId = LayerId::solo(4);
    /// An id `furnished` does not hold, for the four kinds that mint a layer.
    const FRESH: LayerId = LayerId::solo(9);
    /// The copy a `DuplicateLayer` names, likewise unheld.
    const COPY: LayerId = LayerId::solo(10);

    /// A guide's identity is the adding action's own id (§20.5), so three of them
    /// are three action ids.
    const fn guide(lamport: u64) -> GuideId {
        GuideId(ActionId {
            lamport,
            actor: ActorId(1),
        })
    }
    const G1: GuideId = guide(1);
    const G2: GuideId = guide(2);
    const G3: GuideId = guide(3);

    /// [`flat`] with one of everything the ctx-free fold can bite on: a matte, a
    /// filter layer and a two-row guide roster.
    ///
    /// The furniture is what keeps [`sample`] honest. A `SetFilter` on a document
    /// with no filter layer, a `SetMatteRect` with no matte, a `MoveGuide` with no
    /// roster are all *no-ops* — they would round-trip perfectly and prove nothing,
    /// which is the way this test would rot if nothing watched for it. The run
    /// asserts that exactly one kind came out inert.
    fn furnished() -> DocState {
        use stark_model::document::{ColorAdjust, Parcel, PerspectiveGuide};
        use stark_model::geom::Vec2;
        flat()
            .insert_matte(
                MATTE,
                None,
                Place::Bottom,
                MatteRegion::OutsideRect {
                    min: Vec2::ZERO,
                    max: Vec2::splat(64.0),
                },
                Parcel::Solid(Srgb::new([0.2, 0.4, 0.6])),
            )
            .insert_filter(FILTER, None, None, Filter::Color(ColorAdjust::NEUTRAL))
            .insert_guide(G1, PerspectiveGuide::default(), None, None)
            .insert_guide(
                G2,
                PerspectiveGuide::default(),
                Some(G1),
                Some("horizon".into()),
            )
    }

    /// One action of every kind in the roster, each payload aimed at [`furnished`].
    ///
    /// **Exhaustive over [`ActionTag`], with no `_` arm**, which is the whole reason
    /// it is keyed by the tag rather than written as a list: the roster is the
    /// model's own and a kind added later stops this compiling until it has a sample
    /// (§8, §17.9).
    ///
    /// It holds the GPU kinds too, and deliberately. Which half of the fold a kind
    /// belongs to is not restated here — `apply_pure` answers `None` for the ones
    /// that need a renderer and the run below skips exactly those, so an arm moved
    /// across that line moves here with it and no second list can disagree.
    fn sample(tag: ActionTag) -> ActionKind {
        use stark_model::document::{
            BrushParams, ColorAdjust, FillOp, Parcel, PerspectiveGuide, PerspectiveMap,
            SelectionMode, SelectionOp, SelectionShape, StrokeRecord, WarpMap, rect_corners,
        };
        use stark_model::geom::{Affine2, IVec2, Vec2};

        let box_ = || SelectionShape::Rect {
            min: Vec2::ZERO,
            max: Vec2::splat(8.0),
        };
        let desaturated = || {
            Filter::Color(ColorAdjust {
                saturation: 0.0,
                ..ColorAdjust::NEUTRAL
            })
        };
        match tag {
            // The renderer's half. Never folded here — the payloads are the smallest
            // well-formed ones, since what they are for is to make this list one of
            // *every* kind rather than of the ones a GPU-free test can drive.
            ActionTag::CommitStroke => ActionKind::CommitStroke(StrokeRecord {
                layer: B,
                brush: BrushParams::default(),
                path: Vec::new(),
                seed: 1,
                start: 0.0,
                frame: IVec2::ZERO,
            }),
            ActionTag::FloatSelection => ActionKind::FloatSelection {
                layer: B,
                child: FRESH,
                frame: IVec2::ZERO,
            },
            ActionTag::PlaceImage => ActionKind::PlaceImage {
                id: FRESH,
                carrier: None,
                above: None,
                at: IVec2::ZERO,
                name: None,
                image: AssetId([4; 32]),
            },
            ActionTag::Select => {
                ActionKind::Select(SelectionOp::at(SelectionMode::Subtract, box_(), 0.0, 1.0))
            }
            ActionTag::InvertSelection => ActionKind::InvertSelection,
            ActionTag::Transform => ActionKind::Transform {
                layer: B,
                affine: Affine2::IDENTITY,
                frame: IVec2::ZERO,
            },
            ActionTag::TransformPerspective => ActionKind::TransformPerspective {
                layer: B,
                map: PerspectiveMap {
                    min: Vec2::ZERO,
                    max: Vec2::splat(32.0),
                    corners: rect_corners(Vec2::ZERO, Vec2::splat(32.0)),
                },
                frame: IVec2::ZERO,
            },
            ActionTag::TransformWarp => ActionKind::TransformWarp {
                layer: B,
                map: WarpMap::identity(Vec2::ZERO, Vec2::splat(32.0), 2, 2),
                frame: IVec2::ZERO,
            },
            ActionTag::MergeLayerDown => ActionKind::MergeLayerDown { source: C, dest: B },
            ActionTag::Fill => ActionKind::Fill {
                layer: B,
                op: FillOp::new(box_(), 0.0, Srgb::new([0.3, 0.6, 0.9]), 1.0),
                frame: IVec2::ZERO,
            },
            // Away from wherever `B` stands (zero, in `furnished()`), so the fold
            // moves and the round trip proves something.
            ActionTag::TranslateLayers => ActionKind::TranslateLayers {
                moves: vec![(B, IVec2::new(300, -40))],
            },

            // The ctx-free half, each aimed to actually move `furnished()` — see
            // there for why an inert sample would be worse than a failing one.
            ActionTag::AddLayer => ActionKind::AddLayer {
                id: FRESH,
                carrier: None,
                above: Some(A),
            },
            // `B` is a leaf, so it carries nothing and the action's subtree is the
            // empty one the state agrees with (§12.6).
            ActionTag::RemoveLayer => ActionKind::RemoveLayer {
                id: B,
                carried: Vec::new(),
            },
            ActionTag::DuplicateLayer => ActionKind::DuplicateLayer {
                ids: vec![(B, COPY)],
            },
            ActionTag::MoveLayer => ActionKind::MoveLayer {
                id: C,
                carrier: Some(A),
                at: Place::Top,
            },
            ActionTag::SetLayerBlend => ActionKind::SetLayerBlend(B, BlendMode::Multiply),
            ActionTag::SetLayerClip => ActionKind::SetLayerClip(B, true),
            ActionTag::SetLayerOpacity => ActionKind::SetLayerOpacity(B, 0.25),
            ActionTag::SetLayerVisible => ActionKind::SetLayerVisible(B, false),
            ActionTag::SetLayerName => ActionKind::SetLayerName(B, Some("wash".into())),
            // The one kind that is identity by design: resolved at the timeline and
            // never materialized, which is why its footprint is empty.
            ActionTag::Undo => ActionKind::Undo(ActionId {
                lamport: 1,
                actor: ActorId(1),
            }),
            ActionTag::SetSelectionOpacity => ActionKind::SetSelectionOpacity(0.5),
            // Away from `DEFAULT_SUBSTRATE`, which is `Flat` — naming that one would
            // be a no-op on a fresh document.
            ActionTag::SetSubstrate => {
                ActionKind::SetSubstrate(SubstrateId::Image(AssetId([7; 32])))
            }
            ActionTag::SetSubstrateScale => ActionKind::SetSubstrateScale(SubstrateScale::new(140)),
            ActionTag::SetSubstrateColor => {
                ActionKind::SetSubstrateColor(Srgb::new([0.1, 0.2, 0.3]))
            }
            ActionTag::AddMatte => ActionKind::AddMatte {
                id: FRESH,
                carrier: None,
                at: Place::Top,
                region: MatteRegion::Everything,
                paint: Parcel::Solid(Srgb::WHITE),
            },
            ActionTag::SetMatteRect => {
                ActionKind::SetMatteRect(MATTE, Vec2::splat(-8.0), Vec2::splat(24.0))
            }
            ActionTag::SetMattePaint => {
                ActionKind::SetMattePaint(MATTE, Parcel::Solid(Srgb::new([1.0, 0.0, 0.5])))
            }
            ActionTag::AddFilter => ActionKind::AddFilter {
                id: FRESH,
                carrier: None,
                above: None,
                filter: desaturated(),
            },
            ActionTag::SetFilter => ActionKind::SetFilter(FILTER, desaturated()),
            ActionTag::AddGuide => ActionKind::AddGuide {
                id: G3,
                guide: PerspectiveGuide::default(),
                after: Some(G2),
                name: None,
            },
            ActionTag::RemoveGuide => ActionKind::RemoveGuide(G1),
            ActionTag::SetGuide => ActionKind::SetGuide(
                G1,
                PerspectiveGuide {
                    center: Vec2::splat(37.0),
                    focal: 512.0,
                    ..PerspectiveGuide::default()
                },
            ),
            ActionTag::SetGuideName => ActionKind::SetGuideName(G1, Some("vanishing".into())),
            ActionTag::MoveGuide => ActionKind::MoveGuide {
                id: G1,
                after: Some(G2),
            },
        }
    }

    /// **Fold, then unfold, and the document is exactly where it started** — over
    /// every kind the ctx-free half of `apply` can answer, with no adapter in the
    /// room.
    ///
    /// The three claims are all asked through [`undeclared`], which is the one
    /// enumeration of what can differ between two states (§12.6) — so a `DocState`
    /// field that grows is a field this test compares without being told:
    ///
    /// - the fold **did something**, or the round trip proves nothing. `Undo` is the
    ///   one kind that is identity by design, so the inert set is asserted to be
    ///   exactly that rather than each sample being trusted to bite;
    /// - the fold touched **only what its footprint declares** — the rule
    ///   `Materialize::audit` holds every fold in the workspace to, and which every
    ///   run of it until now needed a GPU to reach at all (`tests/footprint.rs`);
    /// - the unfold put back **everything**, asked against an empty footprint, which
    ///   declares no writes and so reports any surviving difference at all.
    ///
    /// It is the round trip the tests above make one kind at a time, made over the
    /// vocabulary — possible only because `apply_pure` exists to be called. That
    /// function's own note has what was standing in for it here.
    #[test]
    fn every_pure_kind_folds_and_unfolds_exactly() {
        let before = furnished();
        let actor = ActorId(1);
        let mut inert = Vec::new();
        for tag in ActionTag::ALL {
            let kind = sample(*tag);
            assert_eq!(kind.tag(), *tag, "the sample list is keyed by its own tag");
            let what = tag.label();
            // The renderer's half declines, and this is the only place the split is
            // consulted — see `sample`.
            let Some(after) = apply_pure(&kind, before.clone(), actor) else {
                continue;
            };
            let action = act(kind);
            let footprint = compute_footprint(&action);
            if undeclared(&before, &after, &Footprint::default()).is_empty() {
                inert.push(what);
            }
            assert_eq!(
                undeclared(&before, &after, &footprint),
                Vec::<String>::new(),
                "{what}: the fold changed state its footprint does not declare",
            );
            let back = unapply(&action, &footprint, &before, &after);
            assert_eq!(
                undeclared(&before, &back, &Footprint::default()),
                Vec::<String>::new(),
                "{what}: the unfold left the document somewhere other than it started",
            );
        }
        assert_eq!(
            inert,
            ["Undo"],
            "only `Undo` folds to identity; anything else here has stopped testing \
             what it says it does",
        );
    }
}
