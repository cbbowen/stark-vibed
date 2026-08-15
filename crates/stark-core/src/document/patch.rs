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
