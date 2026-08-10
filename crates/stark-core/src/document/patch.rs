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

use super::action::{Action, ActionKind, ActorId};
use super::filter::Filter;
use super::footprint::{Resource, footprint};
use super::layer::{BlendMode, Layer, LayerId, MatteRegion};
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
    /// A matte's region and colour together — one footprint resource.
    Matte(LayerId, MatteRegion, [f32; 3]),
    /// A filter layer's settings (§21) — one footprint resource, because the action
    /// that writes them carries the filter entire.
    Filter(LayerId, Filter),
    Selection(ActorId, Selection),
    Surface(SurfaceId),
    Background([f32; 3]),
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
    /// holds, diffed against `from` so untouched entries cost nothing. Walks
    /// only the action's own footprint.
    fn capture(action: &Action, to: &DocState, from: &DocState) -> StatePatch {
        let actor = action.id.actor;
        let mut ops = Vec::new();
        match &action.kind {
            ActionKind::CommitStroke(rec) => {
                tile_diff(rec.layer, paint_rect(action, rec.layer), to, from, &mut ops);
            }
            // A fill writes only paint (it carries no mask along, unlike a
            // transform), so restoring it is the stroke's own diff over its rect.
            ActionKind::Fill { layer, .. } => {
                tile_diff(*layer, paint_rect(action, *layer), to, from, &mut ops);
            }
            ActionKind::Transform { layer, .. }
            | ActionKind::TransformPerspective { layer, .. }
            | ActionKind::TransformWarp { layer, .. } => {
                tile_diff(*layer, paint_rect(action, *layer), to, from, &mut ops);
                ops.push(PatchOp::Selection(actor, to.selection_of(actor)));
            }
            ActionKind::AddLayer { id, .. }
            | ActionKind::AddMatte { id, .. }
            | ActionKind::AddFilter { id, .. }
            | ActionKind::RemoveLayer(id) => {
                match (to.site_of(*id), from.contains_layer(*id)) {
                    (None, true) => ops.push(PatchOp::Absent(*id)),
                    (Some(site), false) => ops.push(PatchOp::Present {
                        site,
                        layer: to.layer(*id).expect("sited layer exists").clone(),
                    }),
                    // Present or absent on both sides: the action no-oped.
                    _ => {}
                }
            }
            // Every layer the copy brought into being, gone again. Removing the
            // root takes the rest of the subtree with it, so the later ops are
            // no-ops — but the copies are what this action *wrote*, and a patch
            // that named only the root would be claiming the others were
            // somebody else's.
            ActionKind::DuplicateLayer { ids } => {
                for (_, copy) in ids {
                    // Absent in `to` and present in `from` is the only direction a
                    // duplicate can go: the state it was applied to predates every
                    // one of these ids.
                    if to.site_of(*copy).is_none() && from.contains_layer(*copy) {
                        ops.push(PatchOp::Absent(*copy));
                    }
                }
            }
            ActionKind::MoveLayer { .. } => ops.push(PatchOp::Structure(structure(to))),
            ActionKind::SetLayerBlend(id, _) => {
                if let Some(l) = to.layer(*id) {
                    ops.push(PatchOp::Blend(*id, l.blend));
                }
            }
            ActionKind::SetLayerClip(id, _) => {
                if let Some(l) = to.layer(*id) {
                    ops.push(PatchOp::Clip(*id, l.clip));
                }
            }
            ActionKind::SetLayerOpacity(id, _) => {
                if let Some(l) = to.layer(*id) {
                    ops.push(PatchOp::Opacity(*id, l.opacity));
                }
            }
            ActionKind::SetLayerVisible(id, _) => {
                if let Some(l) = to.layer(*id) {
                    ops.push(PatchOp::Visible(*id, l.visible));
                }
            }
            ActionKind::SetLayerName(id, _) => {
                if let Some(l) = to.layer(*id) {
                    ops.push(PatchOp::Name(*id, l.name.clone()));
                }
            }
            ActionKind::SetMatteRect(id, _, _) | ActionKind::SetMatteColor(id, _) => {
                if let Some(l) = to.layer(*id)
                    && let super::layer::LayerContent::Matte { region, color } = &l.content
                {
                    ops.push(PatchOp::Matte(*id, *region, *color));
                }
            }
            ActionKind::SetFilter(id, _) => {
                if let Some(f) = to.layer(*id).and_then(|l| l.filter()) {
                    ops.push(PatchOp::Filter(*id, f));
                }
            }
            ActionKind::Select(_) | ActionKind::InvertSelection => {
                ops.push(PatchOp::Selection(actor, to.selection_of(actor)));
            }
            ActionKind::SetSurface(_) => ops.push(PatchOp::Surface(to.surface)),
            ActionKind::SetBackground(_) => ops.push(PatchOp::Background(to.background)),
            // Never materialized (resolved into its target's effectiveness).
            ActionKind::Undo(_) => {}
        }
        StatePatch { ops }
    }

    /// Write the captured values back into `state`.
    fn restore(&self, state: &DocState) -> DocState {
        let mut state = state.clone();
        for op in &self.ops {
            state = match op {
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
                        None => state,
                    }
                }
                PatchOp::Absent(id) => state.remove_layer(*id),
                PatchOp::Present { site, layer } => state.restore_layer(site, layer.clone()),
                PatchOp::Structure(shape) => restore_structure(&state, shape),
                PatchOp::Blend(id, v) => state.set_layer_blend(*id, *v),
                PatchOp::Clip(id, v) => state.set_layer_clip(*id, *v),
                PatchOp::Opacity(id, v) => state.set_layer_opacity(*id, *v),
                PatchOp::Visible(id, v) => state.set_layer_visible(*id, *v),
                PatchOp::Name(id, v) => state.set_layer_name(*id, v.clone()),
                PatchOp::Matte(id, region, color) => {
                    let (min, max) = region.rect();
                    state
                        .set_matte_rect(*id, min, max)
                        .set_matte_color(*id, *color)
                }
                PatchOp::Filter(id, filter) => state.set_filter(*id, *filter),
                PatchOp::Selection(actor, selection) => {
                    state.with_selection(*actor, selection.clone())
                }
                PatchOp::Surface(id) => state.with_surface(*id),
                PatchOp::Background(rgb) => state.with_background(*rgb),
            };
        }
        state
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

/// The tile rect an action's footprint claims on `layer` — the region the
/// commutation gate guarantees nothing in between has touched.
///
/// Asked of the footprint itself rather than re-derived, so the region a restore
/// rewrites is the very region the action declared; the two cannot drift.
///
/// **Empty** when the footprint claims no paint on this layer, and that is the
/// safe answer rather than a fallback: an action that did not declare paint here
/// did not write paint here, so there is nothing of its to put back. Claiming
/// everything instead would restore tiles *outside* the action's footprint —
/// exactly the tiles a commuting action in the gap may own — which is the one
/// thing [`tile_diff`]'s rect bound exists to prevent.
fn paint_rect(action: &Action, layer: LayerId) -> TileRect {
    footprint(action)
        .writes
        .iter()
        .find_map(|r| match r {
            Resource::Paint(l, rect) if *l == layer => Some(*rect),
            _ => None,
        })
        .unwrap_or(TileRect::EMPTY)
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
