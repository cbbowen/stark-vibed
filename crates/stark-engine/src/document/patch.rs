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

use super::action::{Action, ActorId};
use super::filter::Filter;
use super::footprint::{Prop, Resource, footprint};
use super::layer::{BlendMode, Layer, LayerContent, LayerId, MatteRegion};
use super::selection::Selection;
use super::state::{DocState, LayerSite};
use crate::geom::{TileCoord, TileRect};
use crate::gpu::SurfaceId;
use crate::gpu::tile::TilePairHandle;

/// One restorable write. Each variant covers exactly one [`footprint
/// resource`](super::footprint::Resource), never more — a commuting action in
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
    /// A matte's region and color together — one footprint resource.
    Matte(LayerId, MatteRegion, super::layer::MattePaint),
    /// A filter layer's settings (§21) — one footprint resource, because the action
    /// that writes them carries the filter entire.
    Filter(LayerId, Filter),
    Selection(ActorId, Selection),
    Surface(SurfaceId),
    Background([f32; 3]),
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
            // The *value*, not the rect: a region restored through its rect could
            // not round-trip `Everything`, which has none.
            PatchOp::Matte(id, region, paint) => state
                .set_matte_region(*id, *region)
                .set_matte_paint(*id, paint.clone()),
            PatchOp::Filter(id, filter) => state.set_filter(*id, filter.clone()),
            PatchOp::Selection(actor, selection) => state.with_selection(*actor, selection.clone()),
            PatchOp::Surface(id) => state.with_surface(*id),
            PatchOp::Background(rgb) => state.with_background(*rgb),
        }
    }
}

/// `state` with every resource `action` writes restored to the value it holds
/// in `previous` — the state the action was originally applied to. Valid
/// exactly when every action applied between the two commutes with `action`
/// (the `Action::inverse` contract, which the history's commutation gate
/// upholds).
pub fn unapply(action: &Action, previous: &DocState, state: &DocState) -> DocState {
    StatePatch::capture(action, previous, state).restore(state)
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
    /// than by a second match on [`ActionKind`](super::action::ActionKind). That is
    /// what makes "a patch restores exactly what the action declared" true by
    /// construction instead of by inspection: the two used to be parallel matches in
    /// two files with nothing but prose between them, and Rust's exhaustiveness gets
    /// the *presence* of an arm, never its *correspondence*. A kind whose footprint
    /// grows a resource now grows the op that puts it back, in the same edit.
    ///
    /// `Undo` falls out rather than needing an arm: it is never materialized, which
    /// is why its footprint is empty, so it captures nothing.
    fn capture(action: &Action, to: &DocState, from: &DocState) -> StatePatch {
        let mut ops = Vec::new();
        // **Existence first**, whatever order a footprint happens to list its writes
        // in. A layer has to be back in the tree before anything can put its tiles,
        // its properties or the tree's shape right — `restore_structure` in
        // particular arranges records it does not create, and drops a layer the shape
        // names but the state has lost. Partitioning here rather than relying on the
        // footprints being written that way is what keeps that from being a rule
        // `footprint.rs` has to remember.
        let writes = footprint(action).writes;
        let (existence, rest): (Vec<&Resource>, Vec<&Resource>) = writes
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
/// [`Footprint`](super::footprint::Footprint) vocabulary to the [`PatchOp`] that
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
            });
        }
        // The coarse claim expands into the fine ones it stands for, so a footprint
        // that *writes* a whole layer restores a whole layer.
        //
        // No action writes one today — `Resource::Layer` is how `DuplicateLayer` and
        // `MergeLayerDown` state what they **read** — so this arm is unreached. It is
        // written out rather than left to an `unreachable!` because a patch that
        // answered "nothing" for a declared write is the §12.6 hazard read backwards,
        // and an `unreachable!` is exactly how that would arrive: as a panic in the
        // undo path, on the day someone finds a use for the coarse resource.
        Resource::Layer(id) => {
            capture_resource(&Resource::Existence(*id), to, from, ops);
            capture_resource(&Resource::Paint(*id, TileRect::ALL), to, from, ops);
            for prop in Prop::ALL {
                capture_resource(&Resource::Prop(*id, prop), to, from, ops);
            }
        }
        Resource::StackOrder => ops.push(PatchOp::Structure(structure(to))),
        Resource::Selection(actor) => ops.push(PatchOp::Selection(*actor, to.selection_of(*actor))),
        Resource::Surface => ops.push(PatchOp::Surface(to.surface)),
        Resource::Background => ops.push(PatchOp::Background(to.background)),
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
    if !tiles.is_empty() {
        ops.push(PatchOp::Tiles { layer, tiles });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::action::{ActionId, ActionKind};
    use crate::document::layer::Place;
    use crate::gpu::SurfaceId;

    const A: LayerId = LayerId(0);
    const B: LayerId = LayerId(1);
    const C: LayerId = LayerId(2);

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
        Option<(MatteRegion, super::super::layer::MattePaint)>,
    );

    fn props(l: &Layer) -> Props {
        let matte = match &l.content {
            LayerContent::Matte { region, paint } => Some((*region, paint.clone())),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        };
        (l.composite, l.visible, l.name.clone(), l.filter(), matte)
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
        use crate::document::layer::{MattePaint, MatteRegion};
        use crate::document::{BlendMode, ColorAdjust, Filter};
        use crate::geom::Vec2;

        let rect = MatteRegion::OutsideRect {
            min: Vec2::ZERO,
            max: Vec2::splat(64.0),
        };
        // One of each kind of layer, so a `Matte` or a `Filter` property has
        // somewhere to land.
        let before = flat()
            .insert_matte(
                LayerId(3),
                None,
                Place::Top,
                rect,
                MattePaint::Solid([0.2, 0.4, 0.6]),
            )
            .insert_filter(LayerId(4), None, None, Filter::Color(ColorAdjust::NEUTRAL));

        for prop in Prop::ALL {
            let (kind, target) = match prop {
                Prop::Blend => (ActionKind::SetLayerBlend(B, BlendMode::Multiply), B),
                Prop::Clip => (ActionKind::SetLayerClip(B, true), B),
                Prop::Opacity => (ActionKind::SetLayerOpacity(B, 0.25), B),
                Prop::Visible => (ActionKind::SetLayerVisible(B, false), B),
                Prop::Name => (ActionKind::SetLayerName(B, Some("wash".into())), B),
                Prop::Matte => (
                    ActionKind::SetMattePaint(LayerId(3), MattePaint::Solid([1.0; 3])),
                    LayerId(3),
                ),
                Prop::Filter => (
                    ActionKind::SetFilter(
                        LayerId(4),
                        Filter::Color(ColorAdjust {
                            saturation: 0.0,
                            ..ColorAdjust::NEUTRAL
                        }),
                    ),
                    LayerId(4),
                ),
            };
            let action = act(kind);
            // Applied by hand: these arms of `apply` are pure `DocState` calls, so
            // the GPU context it nominally takes has nothing to do here — which is
            // what lets this whole module be tested without an adapter.
            let after = match &action.kind {
                ActionKind::SetLayerBlend(id, v) => before.set_layer_blend(*id, *v),
                ActionKind::SetLayerClip(id, v) => before.set_layer_clip(*id, *v),
                ActionKind::SetLayerOpacity(id, v) => before.set_layer_opacity(*id, *v),
                ActionKind::SetLayerVisible(id, v) => before.set_layer_visible(*id, *v),
                ActionKind::SetLayerName(id, v) => {
                    before.set_layer_name(*id, v.as_deref().map(Into::into))
                }
                ActionKind::SetMattePaint(id, v) => before.set_matte_paint(*id, v.clone()),
                ActionKind::SetFilter(id, v) => before.set_filter(*id, v.clone()),
                other => unreachable!("{other:?} is not a property write"),
            };
            let was = props(before.layer(target).expect("target exists"));
            let now = props(after.layer(target).expect("target exists"));
            // …the action really did change something, or the round trip below is
            // vacuous.
            assert_ne!(
                now, was,
                "{prop:?} was a no-op, so its round trip would prove nothing",
            );

            let back = unapply(&action, &before, &after);
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

        let back = unapply(&action, &before, &after);
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
        let added = before.insert_layer(LayerId(9), None, Some(A));
        let back = unapply(
            &act(ActionKind::AddLayer {
                id: LayerId(9),
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
        let back = unapply(&act(ActionKind::RemoveLayer(B)), &grouped, &removed);
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

        let back = unapply(&action, &before, &after);
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
        let before = flat().with_background([0.5, 0.5, 0.5]);
        let after = before.with_background([0.1, 0.2, 0.3]);
        let back = unapply(
            &act(ActionKind::SetBackground([0.1, 0.2, 0.3])),
            &before,
            &after,
        );
        assert_eq!(back.background, before.background);

        let after = before.with_surface(SurfaceId::Flat);
        let back = unapply(
            &act(ActionKind::SetSurface(SurfaceId::Flat)),
            &before,
            &after,
        );
        assert_eq!(back.surface, before.surface);
    }

    /// `Undo` is resolved by the timeline and never materialized, which is *why* its
    /// footprint is empty — so its patch restores nothing at all. With the capture
    /// driven by the footprint (§4) that falls out rather than needing an arm, and
    /// this is what says so.
    #[test]
    fn an_undo_action_captures_nothing() {
        let before = flat();
        let after = before.set_layer_opacity(B, 0.25);
        let back = unapply(
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
}
