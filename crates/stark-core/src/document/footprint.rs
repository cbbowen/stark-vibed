//! Action footprints: which parts of the document an action reads and writes
//! (§12.6).
//!
//! Two actions **commute** — applying them in either order produces the same
//! state — when neither writes anything the other reads or writes. The
//! history uses that (via the [`Centralizer`](history::Centralizer) impl
//! below) to service an undo by shifting the undone action out of the
//! materialization instead of replaying everything after it: strokes on
//! different layers commute, strokes on the same layer commute when their
//! padded extents don't touch, a rename commutes with nearly everything.
//!
//! Footprints are **conservative**: a stroke claims the whole tile-aligned
//! bounding box of its path (padded past any reach of the tip), a transform
//! claims its entire layer, and every structural edit claims the shared stack
//! order. A false conflict only costs the fast path; a missed one would
//! silently diverge peers, so every `apply` in `action.rs` must read only what
//! its kind's footprint declares and write only what it declares — that
//! locality is what makes the splice sound (see `timeline.rs`).

use super::action::{Action, ActionKind, ActorId, BrushParams, StrokeRecord};
use super::layer::LayerId;
use crate::geom::{TILE_SIZE, Vec2};

/// An inclusive tile-coordinate rectangle. `min > max` on either axis is the
/// empty rect (overlapping nothing).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileRect {
    pub min: (i32, i32),
    pub max: (i32, i32),
}

impl TileRect {
    /// The whole infinite canvas — what a transform claims of its layer.
    pub const ALL: TileRect = TileRect {
        min: (i32::MIN, i32::MIN),
        max: (i32::MAX, i32::MAX),
    };

    pub fn intersects(&self, other: &TileRect) -> bool {
        self.min.0 <= self.max.0
            && other.min.0 <= other.max.0
            && self.min.0 <= other.max.0
            && other.min.0 <= self.max.0
            && self.min.1 <= other.max.1
            && other.min.1 <= self.max.1
    }

    pub fn contains(&self, c: crate::geom::TileCoord) -> bool {
        self.min.0 <= c.x && c.x <= self.max.0 && self.min.1 <= c.y && c.y <= self.max.1
    }
}

/// A per-layer property, at the granularity undo needs to restore it: each
/// variant is overwritten wholesale by the actions that write it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Prop {
    Blend,
    /// Whether the layer clips to the paint beneath it (§14.4).
    /// Its own resource beside `Blend` rather than folded into it: the two are
    /// applied at the same step but written by different actions, and a clip
    /// toggle has to commute with a blend change on the same layer.
    Clip,
    Opacity,
    Visible,
    Name,
    /// A matte's region *and* colour — split no finer because no action writes
    /// one without the freedom to write the other.
    Matte,
}

/// One addressable piece of document state.
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// A layer's painted tiles within a tile rect.
    Paint(LayerId, TileRect),
    /// A layer's presence in the document at all. Every action that targets a
    /// layer reads this (they all no-op on an absent layer, and that no-op is
    /// order-dependent against add/remove).
    Existence(LayerId),
    /// One presentation property of a layer.
    Prop(LayerId, Prop),
    /// The **shape of the whole layer tree** — every stack's order and who
    /// carries whom (§14.8). One coarse resource: two concurrent
    /// restructures genuinely don't commute, and structural edits are rare
    /// enough that finer granularity would buy nothing.
    ///
    /// Nesting rides on it unchanged, which is the point. It is also what makes
    /// the carry-your-own-ancestor case safe without tree-CRDT machinery: two
    /// halves of a cycle conflict here, so the log's total order serializes them
    /// and the second one to apply sees the first's result and declines.
    StackOrder,
    /// An actor's selection mask (§17.3).
    Selection(ActorId),
    /// The canvas surface (§6.4).
    Surface,
    /// The substrate colour (§15.5).
    Background,
}

impl Resource {
    fn overlaps(&self, other: &Resource) -> bool {
        match (self, other) {
            (Resource::Paint(a, ra), Resource::Paint(b, rb)) => a == b && ra.intersects(rb),
            _ => self == other,
        }
    }
}

/// What an action touches: resources it only reads, and resources it writes
/// (which may also be read — a written resource conflicts with everything, so
/// listing it once under `writes` covers both).
#[derive(Clone, Debug, Default)]
pub struct Footprint {
    pub reads: Vec<Resource>,
    pub writes: Vec<Resource>,
}

impl Footprint {
    /// Whether the two actions may fail to commute: any write here overlapping
    /// any read *or* write there (and vice versa). Reads never conflict with
    /// reads.
    pub fn conflicts(&self, other: &Footprint) -> bool {
        let hits =
            |xs: &[Resource], ys: &[Resource]| xs.iter().any(|x| ys.iter().any(|y| x.overlaps(y)));
        hits(&self.writes, &other.writes)
            || hits(&self.writes, &other.reads)
            || hits(&self.reads, &other.writes)
    }
}

/// A [`Footprint`] is the [`history::Centralizer`] of an [`Action`]
/// (§12.6): the history builds it **once** per removal and asks it about each
/// later action, which is what lets `try_remove_action_with` shift an undone
/// action past everything it commutes with instead of replaying it.
///
/// Disjoint footprints satisfy the centralizer contract — neither action reads
/// or writes anything the other writes, so applying them in either order
/// produces the same state, and [`Action`]'s `inverse` restricted to this
/// footprint removes exactly this action's effect. A false conflict only costs
/// the fast path (the contract permits false negatives, never false positives).
impl<'a> history::Centralizer<'a, Action> for Footprint {
    fn for_action(action: &'a Action) -> Self {
        footprint(action)
    }

    fn commutes(&self, other: &Action) -> bool {
        !self.conflicts(&footprint(other))
    }
}

/// Padding around a stroke's control-point bounding box, in canvas px: the
/// farthest any of the tip's marks can land from the fitted centerline.
///
/// The B-spline stays inside its control points' convex hull, so the bbox
/// bounds the centerline exactly; the tip then reaches at most `radius`
/// scaled by √2 for a square stamp swept at an angle (1.5 covers it), and the
/// renderer refreshes a `TILE_APRON` of duplicated neighbor pixels past its
/// marks (the +4 covers that with slack).
fn stroke_pad(brush: &BrushParams) -> f32 {
    brush.radius * 1.5 + 4.0
}

/// The tile-aligned reach of a stroke: everything its render may read or write.
fn stroke_rect(rec: &StrokeRecord) -> TileRect {
    let mut min = (f32::INFINITY, f32::INFINITY);
    let mut max = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in &rec.path {
        min = (min.0.min(p.pos.x), min.1.min(p.pos.y));
        max = (max.0.max(p.pos.x), max.1.max(p.pos.y));
    }
    if min.0 > max.0 {
        // An empty path touches nothing.
        return TileRect {
            min: (1, 1),
            max: (0, 0),
        };
    }
    let pad = stroke_pad(&rec.brush);
    let tile = |v: f32| ((v / TILE_SIZE as f32).floor().clamp(-1e9, 1e9)) as i32;
    TileRect {
        min: (tile(min.0 - pad), tile(min.1 - pad)),
        max: (tile(max.0 + pad), tile(max.1 + pad)),
    }
}

/// The conservative footprint of an action, mirroring exactly what its arm of
/// [`Action`]'s `apply` touches. `Undo` has an empty footprint because it is
/// never materialized — the timeline resolves it into the effectiveness of its
/// target instead.
pub fn footprint(action: &Action) -> Footprint {
    let actor = action.id.actor;
    match &action.kind {
        ActionKind::CommitStroke(rec) => Footprint {
            reads: vec![Resource::Existence(rec.layer), Resource::Selection(actor)],
            writes: vec![Resource::Paint(rec.layer, stroke_rect(rec))],
        },
        // Both anchors are read — the sibling to insert above and the layer whose
        // stack to insert into — because either being absent changes where the
        // layer lands (§14.8).
        ActionKind::AddLayer { id, carrier, above } => Footprint {
            reads: [*carrier, *above]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::Existence(*id), Resource::StackOrder],
        },
        ActionKind::AddMatte {
            id, carrier, above, ..
        } => Footprint {
            reads: [*carrier, *above]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::Existence(*id), Resource::StackOrder],
        },
        // A removal takes the whole subtree, so it writes the existence of layers
        // it does not name. `StackOrder` is the coarse resource that covers them:
        // anything touching another layer's place in the tree conflicts here
        // already, and a property edit on a carried layer commutes with the
        // removal only in the sense that the restore puts the subtree back
        // wholesale — which is exactly what `PatchOp::Present` does.
        ActionKind::RemoveLayer(id) => Footprint {
            reads: Vec::new(),
            writes: vec![
                Resource::Existence(*id),
                Resource::StackOrder,
                Resource::Paint(*id, TileRect::ALL),
            ],
        },
        ActionKind::MoveLayer { id, carrier, above } => Footprint {
            reads: [Some(*id), *carrier, *above]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::StackOrder],
        },
        ActionKind::SetLayerBlend(id, _) => prop_write(*id, Prop::Blend),
        ActionKind::SetLayerClip(id, _) => prop_write(*id, Prop::Clip),
        ActionKind::SetLayerOpacity(id, _) => prop_write(*id, Prop::Opacity),
        ActionKind::SetLayerVisible(id, _) => prop_write(*id, Prop::Visible),
        ActionKind::SetLayerName(id, _) => prop_write(*id, Prop::Name),
        ActionKind::SetMatteRect(id, _, _) => prop_write(*id, Prop::Matte),
        ActionKind::SetMatteColor(id, _) => prop_write(*id, Prop::Matte),
        ActionKind::Select(_) | ActionKind::InvertSelection => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Selection(actor)],
        },
        ActionKind::SetSurface(_) => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Surface],
        },
        ActionKind::SetBackground(_) => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Background],
        },
        ActionKind::Transform { layer, .. } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, TileRect::ALL),
                Resource::Selection(actor),
            ],
        },
        // The rect-scoped transforms (§16.8, §16.9) cut only inside their rect
        // and paste only inside the map's image, so unlike the whole-plane
        // affine they can claim an honest box: the union of the two, padded a
        // tile for the apron reach. An unusable warp (whose image is unknown)
        // falls back to the whole layer — it will be rejected by `apply`, and
        // a too-big footprint is the safe direction.
        ActionKind::TransformPerspective { layer, map } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(
                    *layer,
                    gated_rect((map.min, map.max), Some(map.image_aabb())),
                ),
                Resource::Selection(actor),
            ],
        },
        ActionKind::TransformWarp { layer, map } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, gated_rect((map.min, map.max), map.image_aabb())),
                Resource::Selection(actor),
            ],
        },
        // A fill reads the mask that bounds it and writes the paint its region
        // reaches — the same shape of footprint a stroke has. A fill bounded only
        // by the selection has no analytic box, so it claims the whole layer, the
        // conservative answer a transform gives for the same reason.
        ActionKind::Fill { layer, op } => Footprint {
            reads: vec![Resource::Existence(*layer), Resource::Selection(actor)],
            writes: vec![Resource::Paint(*layer, fill_rect(op))],
        },
        ActionKind::Undo(_) => Footprint::default(),
    }
}

/// The tile-aligned reach of a rect-scoped transform: its source rect unioned
/// with its image bound, padded one tile so apron rewrites are covered.
/// `None` for the image (an unusable map) claims everything.
fn gated_rect(rect: (Vec2, Vec2), image: Option<(Vec2, Vec2)>) -> TileRect {
    let Some(image) = image else {
        return TileRect::ALL;
    };
    let lo = rect.0.min(image.0);
    let hi = rect.1.max(image.1);
    let tile = |v: f32| ((v / TILE_SIZE as f32).floor().clamp(-1e9, 1e9)) as i32;
    TileRect {
        min: (tile(lo.x) - 1, tile(lo.y) - 1),
        max: (tile(hi.x) + 1, tile(hi.y) + 1),
    }
}

/// The tile-aligned reach of a fill: everything its pass may read or write.
fn fill_rect(op: &super::fill::FillOp) -> TileRect {
    let Some((lo, hi)) = super::fill::fill_bounds(op) else {
        return TileRect::ALL;
    };
    let tile = |v: f32| ((v / TILE_SIZE as f32).floor().clamp(-1e9, 1e9)) as i32;
    TileRect {
        min: (tile(lo.x), tile(lo.y)),
        max: (tile(hi.x), tile(hi.y)),
    }
}

fn prop_write(id: LayerId, prop: Prop) -> Footprint {
    Footprint {
        reads: vec![Resource::Existence(id)],
        writes: vec![Resource::Prop(id, prop)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::action::{ActionId, BrushParams, Tool};
    use crate::geom::Vec2;
    use crate::path::ControlPoint;

    fn act(actor: u64, kind: ActionKind) -> Action {
        Action {
            id: ActionId {
                lamport: 1,
                actor: ActorId(actor),
            },
            kind,
        }
    }

    fn stroke(actor: u64, layer: LayerId, from: Vec2, to: Vec2, radius: f32) -> Action {
        let point = |pos| ControlPoint {
            pos,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        };
        act(
            actor,
            ActionKind::CommitStroke(StrokeRecord {
                layer,
                tool: Tool::Brush,
                brush: BrushParams {
                    radius,
                    ..BrushParams::default()
                },
                path: vec![point(from), point(to)],
                seed: 0,
            }),
        )
    }

    fn commutes(a: &Action, b: &Action) -> bool {
        !footprint(a).conflicts(&footprint(b))
    }

    #[test]
    fn strokes_on_different_layers_commute() {
        let a = stroke(1, LayerId(0), Vec2::ZERO, Vec2::splat(100.0), 16.0);
        let b = stroke(2, LayerId(1), Vec2::ZERO, Vec2::splat(100.0), 16.0);
        assert!(commutes(&a, &b));
    }

    #[test]
    fn distant_strokes_on_one_layer_commute_and_near_ones_conflict() {
        let a = stroke(1, LayerId(0), Vec2::ZERO, Vec2::splat(60.0), 16.0);
        let far = stroke(
            2,
            LayerId(0),
            Vec2::splat(2000.0),
            Vec2::splat(2100.0),
            16.0,
        );
        let near = stroke(2, LayerId(0), Vec2::splat(80.0), Vec2::splat(300.0), 16.0);
        assert!(commutes(&a, &far));
        assert!(!commutes(&a, &near));
    }

    #[test]
    fn rename_commutes_with_strokes_but_not_with_removal() {
        let name = act(1, ActionKind::SetLayerName(LayerId(0), Some("wash".into())));
        let paint = stroke(2, LayerId(0), Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let remove = act(2, ActionKind::RemoveLayer(LayerId(0)));
        let other_name = act(2, ActionKind::SetLayerName(LayerId(0), None));
        assert!(commutes(&name, &paint));
        assert!(!commutes(&name, &remove));
        assert!(!commutes(&name, &other_name));
    }

    #[test]
    fn selection_gates_only_its_author() {
        let select = act(1, ActionKind::InvertSelection);
        let own = stroke(1, LayerId(0), Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let other = stroke(2, LayerId(0), Vec2::splat(500.0), Vec2::splat(600.0), 8.0);
        assert!(!commutes(&select, &own));
        assert!(commutes(&select, &other));
    }

    #[test]
    fn structural_edits_conflict_with_each_other() {
        let add = act(
            1,
            ActionKind::AddLayer {
                id: LayerId(7),
                carrier: None,
                above: None,
            },
        );
        let mv = act(
            2,
            ActionKind::MoveLayer {
                id: LayerId(3),
                carrier: None,
                above: None,
            },
        );
        assert!(!commutes(&add, &mv));
    }

    /// Clipping and the blend mode are applied at the same step but are separate
    /// resources, so setting one commutes with setting the other
    /// (§14.8) — while two clip toggles on one layer do not.
    #[test]
    fn clip_commutes_with_blend_but_not_with_itself() {
        let clip = act(1, ActionKind::SetLayerClip(LayerId(0), true));
        let blend = act(
            2,
            ActionKind::SetLayerBlend(LayerId(0), crate::document::BlendMode::Multiply),
        );
        let unclip = act(2, ActionKind::SetLayerClip(LayerId(0), false));
        let elsewhere = act(2, ActionKind::SetLayerClip(LayerId(1), true));
        assert!(commutes(&clip, &blend));
        assert!(commutes(&clip, &elsewhere));
        assert!(!commutes(&clip, &unclip));
    }

    /// Carrying a layer is a `MoveLayer`, so it conflicts with every other
    /// structural edit through `StackOrder` — which is what serializes the two
    /// halves of a would-be cycle (§14.8).
    #[test]
    fn carrying_conflicts_with_the_reverse_carry() {
        let a_onto_b = act(
            1,
            ActionKind::MoveLayer {
                id: LayerId(0),
                carrier: Some(LayerId(1)),
                above: None,
            },
        );
        let b_onto_a = act(
            2,
            ActionKind::MoveLayer {
                id: LayerId(1),
                carrier: Some(LayerId(0)),
                above: None,
            },
        );
        assert!(!commutes(&a_onto_b, &b_onto_a));
    }
}
