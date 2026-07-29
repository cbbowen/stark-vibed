//! [`unapply`]: the implementation of `Action::inverse` (DESIGN.md §12.6) —
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

use std::sync::Arc;

use rpds::Vector;

use super::action::{Action, ActionKind, ActorId};
use super::footprint::{Resource, TileRect, footprint};
use super::layer::{BlendMode, Layer, LayerId, MatteRegion};
use super::selection::Selection;
use super::state::DocState;
use crate::geom::TileCoord;
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
    /// The layer existed, as this record, at this stack index (undoes a remove).
    Present {
        index: usize,
        layer: Layer,
    },
    /// The stack sat in this order (undoes a move).
    Order(Vec<LayerId>),
    Blend(LayerId, BlendMode),
    Opacity(LayerId, f32),
    Visible(LayerId, bool),
    Name(LayerId, Option<Arc<str>>),
    /// A matte's region and colour together — one footprint resource.
    Matte(LayerId, MatteRegion, [f32; 3]),
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
            ActionKind::Transform { layer, .. } => {
                tile_diff(*layer, paint_rect(action, *layer), to, from, &mut ops);
                ops.push(PatchOp::Selection(actor, to.selection_of(actor)));
            }
            ActionKind::AddLayer { id, .. }
            | ActionKind::AddMatte { id, .. }
            | ActionKind::RemoveLayer(id) => match (to.layer_index(*id), from.layer_index(*id)) {
                (None, Some(_)) => ops.push(PatchOp::Absent(*id)),
                (Some(index), None) => ops.push(PatchOp::Present {
                    index,
                    layer: to.layer_at(index).clone(),
                }),
                // Present or absent on both sides: the action no-oped.
                _ => {}
            },
            ActionKind::MoveLayer { .. } => {
                ops.push(PatchOp::Order(to.layers.iter().map(|l| l.id).collect()));
            }
            ActionKind::SetLayerBlend(id, _) => {
                if let Some(i) = to.layer_index(*id) {
                    ops.push(PatchOp::Blend(*id, to.layer_at(i).blend));
                }
            }
            ActionKind::SetLayerOpacity(id, _) => {
                if let Some(i) = to.layer_index(*id) {
                    ops.push(PatchOp::Opacity(*id, to.layer_at(i).opacity));
                }
            }
            ActionKind::SetLayerVisible(id, _) => {
                if let Some(i) = to.layer_index(*id) {
                    ops.push(PatchOp::Visible(*id, to.layer_at(i).visible));
                }
            }
            ActionKind::SetLayerName(id, _) => {
                if let Some(i) = to.layer_index(*id) {
                    ops.push(PatchOp::Name(*id, to.layer_at(i).name.clone()));
                }
            }
            ActionKind::SetMatteRect(id, _, _) | ActionKind::SetMatteColor(id, _) => {
                if let Some(i) = to.layer_index(*id)
                    && let super::layer::LayerContent::Matte { region, color } =
                        &to.layer_at(i).content
                {
                    ops.push(PatchOp::Matte(*id, *region, *color));
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
                PatchOp::Tiles { layer, tiles } => match state.layer_index(*layer) {
                    Some(index) => {
                        let record = state.layer_at(index);
                        match record.tiles() {
                            Some(map) => {
                                let mut map = map.clone();
                                for (coord, handle) in tiles {
                                    map = match handle {
                                        Some(handle) => map.insert(*coord, handle.clone()),
                                        None => map.remove(coord),
                                    };
                                }
                                let record = record.with_tiles(map);
                                state.with_layer_at(index, record)
                            }
                            None => state,
                        }
                    }
                    None => state,
                },
                PatchOp::Absent(id) => state.remove_layer(*id),
                PatchOp::Present { index, layer } => {
                    let mut layers = Vector::new();
                    for (i, l) in state.layers.iter().enumerate() {
                        if i == *index {
                            layers = layers.push_back(layer.clone());
                        }
                        layers = layers.push_back(l.clone());
                    }
                    if *index >= state.layers.len() {
                        layers = layers.push_back(layer.clone());
                    }
                    state.with_layers(layers)
                }
                PatchOp::Order(ids) => {
                    let mut layers = Vector::new();
                    for id in ids {
                        if let Some(i) = state.layer_index(*id) {
                            layers = layers.push_back(state.layer_at(i).clone());
                        }
                    }
                    // Anything the order predates keeps stacking on top; with a
                    // commuting suffix (no structural edits) this arm is empty.
                    for l in state.layers.iter() {
                        if !ids.contains(&l.id) {
                            layers = layers.push_back(l.clone());
                        }
                    }
                    state.with_layers(layers)
                }
                PatchOp::Blend(id, v) => state.set_layer_blend(*id, *v),
                PatchOp::Opacity(id, v) => state.set_layer_opacity(*id, *v),
                PatchOp::Visible(id, v) => state.set_layer_visible(*id, *v),
                PatchOp::Name(id, v) => state.set_layer_name(*id, v.clone()),
                PatchOp::Matte(id, region, color) => {
                    let (min, max) = region.rect();
                    state
                        .set_matte_rect(*id, min, max)
                        .set_matte_color(*id, *color)
                }
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

/// The tile rect an action's footprint claims on `layer` — the region the
/// commutation gate guarantees nothing in between has touched.
fn paint_rect(action: &Action, layer: LayerId) -> TileRect {
    footprint(action)
        .writes
        .iter()
        .find_map(|r| match r {
            Resource::Paint(l, rect) if *l == layer => Some(*rect),
            _ => None,
        })
        .unwrap_or(TileRect::ALL)
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
    let tiles_of = |state: &DocState| {
        state
            .layer_index(layer)
            .and_then(|i| state.layer_at(i).tiles().cloned())
    };
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
