//! Layers (§5.1, §15.2, §14, §21). A layer is a sparse map of painted tiles, a
//! **matte** — a procedural region filled with a flat color — or a **filter**,
//! which is a function of what is composited beneath it; plus its presentation
//! properties, plus the layers it **carries**.
//!
//! A layer stacks with premultiplied "over" unless its [`BlendMode`] says
//! otherwise or it is clipped ([`SetLayerClip`](super::ActionKind::SetLayerClip),
//! [`Prop::Clip`](super::Prop::Clip)), in which case the compositor isolates it
//! and merges it through the mode (§18.0.4). A layer that carries others is a
//! **group** — there is no separate group type — and the same isolation, recursed,
//! is what composites it (§14.7).

use serde::{Deserialize, Serialize};

use super::action::{ActionId, ActorId};
use crate::geom::Vec2;

/// Stable identifier for a layer within a document: **the action that minted it,
/// and which of that action's layers this is**.
///
/// An [`ActionId`] is the log's total-order key `(lamport, actor)` and so already
/// globally unique, which is what keeps two peers adding a layer at the same moment
/// from minting the same id — a collision no pixel could resolve (§17.9). `k` is
/// which of the action's layers this is, assigned by the author in the order
/// `Layer::visit` walks and **carried** in that action's own map, so every peer
/// reads the same `k` whatever its own tree looks like.
///
/// [`ROOT`](Self::ROOT) is the one id no action mints: every peer must agree on the
/// root layer, which predates every action.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    carbonite::Schema,
)]
pub struct LayerId {
    /// The action that minted this layer.
    pub action: ActionId,
    /// Which of that action's layers — `0` for the four kinds that mint one, the
    /// subtree position for a [`DuplicateLayer`](super::ActionKind::DuplicateLayer).
    pub k: u32,
}

impl LayerId {
    /// The root layer, which every document has before any action runs.
    ///
    /// **A reserved `k`, not a reserved action.** The lamport clock starts at zero,
    /// so `ActionId { lamport: 0, actor: SOLO }` is an ordinary solo document's first
    /// action and cannot be spent on a sentinel. `u32::MAX` is the `k` no mint
    /// produces: the four single-layer kinds pass `0`, and a duplicate's is a position
    /// in the subtree it copies.
    pub const ROOT: LayerId = LayerId {
        action: ActionId {
            lamport: 0,
            actor: ActorId::SOLO,
        },
        k: u32::MAX,
    };

    /// The id of `action`'s `k`th layer.
    pub const fn new(action: ActionId, k: u32) -> Self {
        Self { action, k }
    }

    /// Whether `actor` minted this layer — the author of the action it came from.
    pub fn minted_by(self, actor: ActorId) -> bool {
        self.action.actor == actor
    }

    /// The id a **solo** author's action at `lamport` mints for its first layer.
    ///
    /// Not a test affordance: `ActorId::SOLO` authors every action in a document that
    /// has never been shared (§12.3), so this is an id such a document really mints.
    pub const fn solo(lamport: u64) -> Self {
        Self::new(
            ActionId {
                lamport,
                actor: ActorId::SOLO,
            },
            0,
        )
    }
}

impl std::fmt::Display for LayerId {
    /// `lamport.actor.k` — the id as a stable, unique string, for a frontend that
    /// needs a list key two layers cannot share (§11).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}",
            self.action.lamport, self.action.actor.0, self.k
        )
    }
}

/// Where in a stack a layer lands — the anchor half of a structural move
/// (§14.8).
///
/// [`Above`](Self::Above) names a sibling, which covers every landing place but one:
/// **under the bottom layer**, which has no sibling below it to be named after.
/// [`Bottom`](Self::Bottom) is that place.
///
/// The variant order is **not** load-bearing: variants are matched by *name* (§8), so
/// a case may be added wherever it reads best.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub enum Place {
    /// On top of the stack, over everything already in it.
    Top,
    /// Directly above this sibling — or on top, if it is not in this stack.
    Above(LayerId),
    /// At the foot of the stack, under everything already in it.
    Bottom,
}

impl Place {
    /// The sibling this place is stated against, if any — what a footprint has to
    /// name as read (§12.6), since where the move lands depends on where that
    /// layer is.
    pub fn anchor(self) -> Option<LayerId> {
        match self {
            Place::Above(id) => Some(id),
            Place::Top | Place::Bottom => None,
        }
    }
}

impl From<Option<LayerId>> for Place {
    /// The two-state anchor insertion takes: a named sibling, or the top of the
    /// stack.
    fn from(above: Option<LayerId>) -> Self {
        match above {
            Some(id) => Place::Above(id),
            None => Place::Top,
        }
    }
}

/// How a layer combines with the layers below it (§18.0.4).
///
/// Every mode past `Normal` combines the two layers' **light** rather than covering
/// one with the other. The numbers in a layer are light that has already been through
/// a tone curve, so the honest combination is: undo the curve, add, re-apply it. Each
/// mode is that sentence with a different curve `T`:
///
/// ```text
///     f(a, b) = T(T⁻¹(a) + T⁻¹(b))
/// ```
///
/// Being a conjugation of addition is the whole guarantee: every mode is commutative
/// and associative with a neutral element, so three glowing layers give the same
/// result in any order and regrouping them changes nothing.
///
/// **The guarantee holds at any coverage**: a layer's coverage weighs it in the space
/// where its own blend function is affine — emission for the emissive modes, light
/// itself for [`Multiply`](Self::Multiply) — not in the working space, since applying
/// a curve to a coverage-averaged color is not the same as averaging the curve
/// (§18.0.4).
///
/// The combination happens in **CIE XYZ normalized to the display white**: linear in
/// light, non-negative for every real color (which is what makes the curves
/// well-defined), and `1` means the same thing on all three axes. In RGB the result
/// would depend on the display's primaries; in Oklab or in pigment concentrations it
/// would be adding things that are not light.
///
/// **A mode may carry its own parameters**, and [`Drago`](Self::Drago) does. They live
/// on the variant, which is the one shape in which a parameter cannot be stated for a
/// mode that has none — and it keeps the merge's "the two layers agree about how they
/// meet the backdrop" (`document::merge`) meaning something once a mode is a family of
/// curves: two `Drago`s with different `k` are two different functions, and `!=` says
/// so.
///
/// A new mode may go wherever it reads best, and a parameterized one may gain a knob:
/// variants and fields are matched by *name* (§8).
///
/// See `blend_common.wesl` for the derivations and `Compositor` for the isolation
/// pass that makes per-layer blending possible at all.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum BlendMode {
    /// Premultiplied "over": the layer sits on top of what is below it.
    #[default]
    Normal,
    /// **Glow** — addition conjugated by the Reinhard tonemap `T(x) = x/(1+x)`,
    /// which collapses to
    ///
    /// ```text
    ///     f(a, b) = (a + b − 2ab) / (1 − ab)
    /// ```
    ///
    /// Reinhard's curve is asymptotic, so this mode **cannot blow out**: stack a
    /// hundred glow layers and the result approaches white without ever clipping. The
    /// one to reach for on glazes, mist, rim light and bloom.
    Reinhard,
    /// **Radiance** — addition conjugated by Drago's log curve
    /// `T(x) = k·log(1 + x/k)`, which collapses to
    ///
    /// ```text
    ///     f(a, b) = k·log(e^{a/k} + e^{b/k} − 1)
    /// ```
    ///
    /// A log curve has no asymptote, so unlike [`Reinhard`](Self::Reinhard) this one
    /// *does* push past display white where two strong lights coincide — and that
    /// overflow is the point. The composite targets are half-float, so the excess
    /// survives into the media pass and comes back through its highlight roll-off
    /// (§6.3) as a bloom rather than being clipped at the blend. Reach for it on
    /// flame, specular hits, anything meant to read as *brighter than the paper*.
    ///
    /// `k` is **the layer's own** bend: large `k` tends to plain addition, so two
    /// lights reach the roll-off sooner; small `k` tends to `max`, so the brighter of
    /// the two simply wins. [`DRAGO_K`] is where it starts and [`DRAGO_K_RANGE`] is
    /// how far it goes. Every setting is still a conjugation of addition, so no `k` a
    /// painter picks costs the guarantee the mode exists for.
    Drago { k: f32 },
    /// **Multiply** — the same construction read the other way round, with
    /// `T(x) = e^{-x}`, which collapses to
    ///
    /// ```text
    ///     f(a, b) = a·b
    /// ```
    ///
    /// The quantity being added is **optical density**, so this is Beer-Lambert: what
    /// two stacked glazes do to the light passing through them. Everything the
    /// emissive modes guarantee still holds, dualised, except that the neutral element
    /// is **white** rather than black — and the output is in `[0,1]` by construction,
    /// so this is the one mode that never reaches the media pass's highlight roll-off.
    ///
    /// One consequence to know about. The blend sees the layer stack, not the
    /// **substrate** — the paper is composited in pass B, after all blending
    /// (`media_common.wesl`) — so a glaze laid on bare canvas leaves the paper's own
    /// color untouched instead of tinting it. On white paper that is exactly right,
    /// white being multiply's identity; on a toned substrate it is a divergence from
    /// what a real glaze would do.
    Multiply,
}

/// The bend a [`BlendMode::Drago`] layer **starts at**, in units of display white —
/// what the picker hands out and what the panel's Bend slider rests on. Large `k`
/// The bend a [`BlendMode::Drago`] layer **starts at**, in units of display white —
/// what the picker hands out and what the panel's Bend slider rests on. Large `k`
/// tends to plain addition, small `k` tends to `max`.
///
/// A **default** rather than the value, since the curve is per layer — but it is the
/// setting the goldens and the docs' worked example are written against.
pub const DRAGO_K: f32 = 0.6;

/// How far [`BlendMode::Drago`]'s bend may be taken — the span a frontend's slider
/// covers and the span [`BlendMode::sanitized`] holds a log entry to.
///
/// The ends are where the mode stops changing rather than round numbers: at `0.125`
/// the curve has arrived at "the brighter one wins", and at `4.0` it is straight over
/// the whole display range, which is plain addition waiting to clip.
///
/// Bounded because a blend is a fullscreen pass with no coverage to hide behind, and
/// `k = 0` is a division by zero in `emission` that would take every texel of the
/// frame with it. A file or a peer reaches [`BlendMode::sanitized`] without passing
/// through a slider, which is the case the bound is actually for.
pub const DRAGO_K_RANGE: (f32, f32) = (0.125, 4.0);

impl BlendMode {
    /// Every mode **at its default setting**, in the order a frontend should offer
    /// them: `Normal` first, then increasingly emphatic light, then the one that takes
    /// light away.
    ///
    /// A list of modes, not of settings of them — which is why a picker built from it
    /// selects its current row with [`same_mode`](Self::same_mode) rather than `==`,
    /// so a tuned `k` is never quietly reset by re-picking the mode it belongs to.
    pub const ALL: [BlendMode; 4] = [
        Self::Normal,
        Self::Reinhard,
        Self::Drago { k: DRAGO_K },
        Self::Multiply,
    ];

    /// What this mode is called: the painter-facing name, not the tonemap's.
    /// `Multiply` is the exception — there the operation's name is already the term of
    /// art, and renaming it to match its neighbours would invent a synonym.
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Reinhard => "Glow",
            Self::Drago { .. } => "Radiance",
            Self::Multiply => "Multiply",
        }
    }

    /// Whether these are the **same mode**, whatever either has it set to — what a
    /// picker's rows are selected by, since a picker offers a mode and not a setting
    /// of one.
    ///
    /// Distinct from `==`, which is the question the compositor and the merge ask,
    /// where two bends really are two different functions.
    pub fn same_mode(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    /// The curve bend the blend pass's uniform carries — this layer's for
    /// [`Drago`](Self::Drago), and [`DRAGO_K`] for every mode whose shader path never
    /// reads it (`blend_common.wesl` branches on the mode first).
    pub fn drago_k(self) -> f32 {
        match self {
            Self::Drago { k } => k,
            _ => DRAGO_K,
        }
    }

    /// The same mode with every parameter finite and in range — the funnel a mode
    /// passes through on its way into the document, applied where the action is minted
    /// (`Engine::process`), so the log records what was applied, and where a mode
    /// enters state (`DocState::set_layer_blend`), because a loaded file or a remote
    /// peer reaches state without passing through `process`.
    ///
    /// A non-finite `k` falls back to [`DRAGO_K`] rather than to a bound: `NaN` says
    /// nothing about which end was meant.
    #[must_use]
    pub fn sanitized(self) -> Self {
        match self {
            Self::Drago { k } => Self::Drago {
                k: crate::sanitize::finite_in(k, DRAGO_K, DRAGO_K_RANGE),
            },
            Self::Normal | Self::Reinhard | Self::Multiply => self,
        }
    }

    /// Whether this mode composites under plain premultiplied "over".
    ///
    /// The compositor's fast path: a run of consecutive `Normal` layers needs no
    /// isolation and draws straight into the accumulator (§6.3).
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
}

/// The region a matte layer fills (§15.2).
///
/// A region is a coverage field over the *infinite* plane, so what matters is its
/// value at infinity — which is what makes the frame case (fill everywhere except a
/// rect) expressible at all, and expressible without a mask.
///
/// Its geometry is stated **in the layer's frame** (§14.12): the layer's `translation`
/// places it on the canvas, which is what lets a matte answer `TranslateLayers` with
/// the same property write a paint layer does. `SetMatteRect` writes this frame's
/// coordinates; converting a canvas-space gesture is the command tier's business.
///
/// It is stored as **geometry, not a rasterized mask**: the fill is evaluated
/// analytically from a signed distance at canvas position, as `selection.wesl` does
/// (§6.8). That costs no tiles, stays exact at any zoom, keeps the log to four floats,
/// and — being a pure function of canvas position — satisfies the §6.4 seam invariant
/// for free.
///
/// Two variants because two are built; §15.9 (P4) is where the `SelectionOp` algebra
/// widens this to comic gutters, lasso mattes and frame-from-selection.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum MatteRegion {
    /// Everything *outside* this rect — the frame / mat board. In the layer's
    /// frame (§14.12); see the enum docs.
    OutsideRect { min: Vec2, max: Vec2 },
    /// The whole plane — a backing / underpainting, made to sit at the bottom of
    /// the stack (§15.5). It has no rect: it frames nothing, so it
    /// defines no export rect and mounts no handles — the coverage is the whole
    /// of what it says.
    Everything,
}

impl MatteRegion {
    /// The rect this region is defined against, in the layer's frame (canvas-px
    /// units, §14.12) — for [`OutsideRect`](Self::OutsideRect) the *hole*, the piece,
    /// which is what export frames against (§15.6). `None` for a region not defined
    /// against one: an [`Everything`](Self::Everything) matte frames nothing, and
    /// every consumer of the rect has a real answer for that.
    pub fn rect(&self) -> Option<(Vec2, Vec2)> {
        match self {
            Self::OutsideRect { min, max } => Some((*min, *max)),
            Self::Everything => None,
        }
    }

    /// Whether this region may be applied at all: its rect, if it has one, is
    /// measurable. Deterministic, so peers and replays agree about rejection —
    /// exactly [`TransformMap::usable`](super::transform::TransformMap::usable)'s
    /// contract.
    ///
    /// **Refused rather than clamped**, which is why this is a predicate and not a
    /// `sanitized`: there is no repaired version of a rectangle nobody can measure,
    /// and rounding it to the origin would silently reframe the piece, which the
    /// export rect, the aspect readout and the handle box all read (§16.1).
    pub fn usable(&self) -> bool {
        match self {
            Self::OutsideRect { min, max } => min.is_finite() && max.is_finite(),
            Self::Everything => true,
        }
    }

    /// The same region with its rect replaced (the frame drag's commit) — a
    /// no-op on a region that has none, matching `SetMatteRect`'s no-op on a
    /// layer that is not a matte: the action names a property this region does
    /// not have.
    #[must_use]
    pub fn with_rect(&self, min: Vec2, max: Vec2) -> Self {
        match self {
            Self::OutsideRect { .. } => Self::OutsideRect { min, max },
            Self::Everything => Self::Everything,
        }
    }

    /// The same region shifted whole by `by` (§14.12): what places a frame-stated
    /// rect on the canvas, and — negated — a canvas-space gesture into the frame at
    /// the mint. [`Everything`](Self::Everything) has no position and rides through.
    #[must_use]
    pub fn translated(&self, by: Vec2) -> Self {
        match self {
            Self::OutsideRect { min, max } => Self::OutsideRect {
                min: *min + by,
                max: *max + by,
            },
            Self::Everything => Self::Everything,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A variant is identified by its **name**, not its position — so [`Place`] may
    /// gain a case anywhere, and a saved `MoveLayer` still means the move it meant
    /// (§8). `Old` is the hazard made concrete: the same three cases in a different
    /// order, which must read back exactly.
    #[test]
    fn a_place_is_read_by_variant_name_not_position() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Place")]
        enum Old {
            Bottom,
            Above(LayerId),
            Top,
        }

        let id = LayerId::solo(0x1234_5678);
        let read = |old: &Old| {
            carbonite::from_slice::<Place>(&carbonite::to_vec(old).expect("encodes"))
                .expect("an order this build does not declare still reads")
        };

        assert_eq!(read(&Old::Top), Place::Top);
        assert_eq!(read(&Old::Bottom), Place::Bottom);
        assert_eq!(read(&Old::Above(id)), Place::Above(id));
    }

    /// The same for [`BlendMode`], where the stakes are a picture: a mode read as the
    /// wrong one recomposites every layer that used it. `Drago` is the sharp case —
    /// it carries a payload, and must arrive as itself, bend and all.
    #[test]
    fn a_mode_is_read_by_variant_name_bend_and_all() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "BlendMode")]
        enum Old {
            Multiply,
            Drago { k: f32 },
            Normal,
            Reinhard,
        }

        let read = |old: &Old| {
            carbonite::from_slice::<BlendMode>(&carbonite::to_vec(old).expect("encodes"))
                .expect("a declaration order this build does not use still reads")
        };

        assert_eq!(read(&Old::Normal), BlendMode::Normal);
        assert_eq!(read(&Old::Reinhard), BlendMode::Reinhard);
        assert_eq!(read(&Old::Multiply), BlendMode::Multiply);
        assert_eq!(
            read(&Old::Drago { k: DRAGO_K }),
            BlendMode::Drago { k: DRAGO_K },
            "the payload travels with the name, not with an index",
        );
    }

    /// A picker asks [`BlendMode::same_mode`] and the compositor asks `==`, and the
    /// two must give different answers about two bends of the same mode — that is the
    /// whole reason both exist.
    #[test]
    fn a_bend_is_the_same_mode_but_not_the_same_value() {
        let (a, b) = (BlendMode::Drago { k: 0.4 }, BlendMode::Drago { k: 1.2 });
        assert!(a.same_mode(b), "both are Radiance");
        assert_ne!(a, b, "…and they are not the same curve");
        assert!(!a.same_mode(BlendMode::Reinhard));
        assert_eq!(a.label(), b.label(), "one row in the picker, so one name");
        // Every mode in the list is the row it selects, which is what makes the
        // picker's `find(|m| m.label() == …)` and its `same_mode` agree.
        for mode in BlendMode::ALL {
            assert_eq!(
                BlendMode::ALL.iter().filter(|m| m.same_mode(mode)).count(),
                1,
                "{} names more than one row",
                mode.label(),
            );
        }
    }

    /// A bend from a file or a peer is brought back into range, and an unusable one
    /// falls back to the default rather than to a bound — [`BlendMode::sanitized`]'s
    /// contract, which the fullscreen blend pass has no coverage to hide behind.
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "the funnel replaces its input with this constant itself, so the assertion is identity rather than proximity"
    )]
    fn a_bend_is_sanitized_into_range() {
        let k = |m: BlendMode| m.sanitized().drago_k();
        assert_eq!(k(BlendMode::Drago { k: 0.0 }), DRAGO_K_RANGE.0);
        assert_eq!(k(BlendMode::Drago { k: -3.0 }), DRAGO_K_RANGE.0);
        assert_eq!(k(BlendMode::Drago { k: 1e9 }), DRAGO_K_RANGE.1);
        assert_eq!(k(BlendMode::Drago { k: f32::NAN }), DRAGO_K);
        assert_eq!(k(BlendMode::Drago { k: f32::INFINITY }), DRAGO_K);
        // A setting already in range is left exactly alone — a sanitizer that nudged
        // would make every load a small edit.
        assert_eq!(k(BlendMode::Drago { k: 0.3 }), 0.3);
        // And the default is in range, or the picker would hand out a value the very
        // funnel it passes through would change.
        assert_eq!(BlendMode::ALL[2].sanitized(), BlendMode::ALL[2]);
        // The modes without parameters have nothing to sanitize and are untouched.
        for mode in [BlendMode::Normal, BlendMode::Reinhard, BlendMode::Multiply] {
            assert_eq!(mode.sanitized(), mode);
            assert_eq!(mode.drago_k(), DRAGO_K, "the uniform still needs a number");
        }
    }

    /// The same for [`MatteRegion`], where reading the wrong variant is the whole
    /// picture: a frame read as [`Everything`](MatteRegion::Everything) floods the
    /// canvas with the mat board's paint, and an [`Everything`](MatteRegion::Everything)
    /// read as a frame stands a backing where none was placed (§15.5).
    #[test]
    fn a_matte_region_is_read_by_variant_name_not_position() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "MatteRegion")]
        enum Old {
            Everything,
            OutsideRect { min: Vec2, max: Vec2 },
        }

        let read = |old: &Old| {
            carbonite::from_slice_static::<MatteRegion>(
                &carbonite::to_vec_static(old).expect("encodes"),
            )
            .expect("a declaration order this build does not use still reads")
        };

        assert_eq!(read(&Old::Everything), MatteRegion::Everything);
        let (min, max) = (Vec2::new(-12.5, 8.0), Vec2::new(640.0, 480.0));
        assert_eq!(
            read(&Old::OutsideRect { min, max }),
            MatteRegion::OutsideRect { min, max },
            "the frame travels with the name, not with an index",
        );
    }
}
