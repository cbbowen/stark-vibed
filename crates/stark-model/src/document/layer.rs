//! Layers (§5.1, §15.2, §14, §21). A layer is a sparse, persistent map of painted
//! tiles, a **matte** — a procedural region filled with a flat color — or a
//! **filter**, which is a function of what is composited beneath it rather than
//! content of its own; plus its presentation properties, plus the layers it
//! **carries**.
//!
//! A layer stacks with premultiplied "over" unless its [`BlendMode`] says
//! otherwise or it is clipped ([`SetLayerClip`](super::ActionKind::SetLayerClip),
//! [`Prop::Clip`](super::Prop::Clip)), in which case the compositor
//! isolates it and merges it through the mode (§18.0.4). A
//! layer that carries others is a **group** — there is no separate group type —
//! and the same isolation, recursed, is what composites it (§14.7).

use serde::{Deserialize, Serialize};

use super::action::{ActionId, ActorId};
use crate::geom::Vec2;

/// Stable identifier for a layer within a document: **the action that minted it,
/// and which of that action's layers this is**.
///
/// Two peers adding a layer at the same moment must not mint the same id — the log
/// would then hold two different layers under one, which `layer_index` resolves to
/// whichever comes first, and no pixel says which peer's it was (§17.9). That is the
/// convergence failure this shape rules out rather than guards against: an
/// [`ActionId`] is already the log's total-order key `(lamport, actor)`, so it is
/// already globally unique, and an id built from one cannot collide with an id built
/// from another. There is no counter, nothing to resync when a log is picked back up,
/// and no re-share rule to remember.
///
/// [`GuideId`](super::GuideId) is the same answer without a `k`, since one `AddGuide`
/// mints exactly one guide where `DuplicateLayer` mints one per layer of a subtree.
/// `k` is which of the action's layers this is, assigned by the author in the order
/// `Layer::visit` walks and **carried** in that action's own map. Carried rather than
/// re-derived at each peer: every peer then reads the same `k` off the log whatever
/// its own tree looks like, which is what `DuplicateLayer`'s doc insists on.
///
/// [`ROOT`](Self::ROOT) is the one id no action mints, and it has to be: every peer
/// must agree on the root layer, which predates every action.
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
    /// **A reserved `k`, not a reserved action.** The lamport clock starts at zero, so
    /// `ActionId { lamport: 0, actor: SOLO }` is a perfectly ordinary first action of a
    /// solo document and cannot be spent on a sentinel.
    ///
    /// `u32::MAX` is the `k` no mint produces. The four single-layer kinds pass `0`;
    /// a duplicate's is a position in the subtree it copies, and a subtree of `2³² − 1`
    /// layers is not a bound anybody imposed but a document larger than the address
    /// space — a `Layer` is a persistent map of tile handles and several presentation
    /// fields, so four billion of them is terabytes of them. The check that matters is
    /// the one at the door (`Engine::commit_minting`), which asserts the ids an action
    /// mints are distinct and its own; this sentinel only has to sit outside what a
    /// document can reach, and it does by nine orders.
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
    /// Not a test affordance: `ActorId::SOLO` is the author of every action in a
    /// document that has never been shared (§12.3), so this is the id such a document
    /// really does mint — which is what makes it a usable stand-in for one, and what
    /// makes a test that names a layer this way name a layer that could exist.
    pub const fn solo(lamport: u64) -> Self {
        Self::new(
            ActionId {
                lamport,
                actor: ActorId::SOLO,
            },
            0,
        )
    }

    /// When this layer was minted, on the author's Lamport clock — what an unnamed
    /// layer is labelled by (§11).
    ///
    /// A *display* number and nothing else, which is why it is not called an ordinal:
    /// nothing resumes from it, and it is neither dense nor unique across authors. It
    /// is monotone within one author's layers, which is the whole of what a label
    /// needs to be.
    pub fn minted_at(self) -> u64 {
        self.action.lamport
    }
}

impl std::fmt::Display for LayerId {
    /// `lamport.actor.k` — the id as a stable, unique string.
    ///
    /// For a frontend that needs one: a DOM row is keyed by its layer so the browser
    /// can tell a reordered list from a rebuilt one (§11), and a key that two layers
    /// could share would animate one row into another. Every field, in the order that
    /// makes the common case short: a solo document's ids read `0.0.0`, `3.0.0`, and
    /// only a duplicate or a peer's layer grows the tail.
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
/// [`Above`](Self::Above) names a sibling, and a stack of `n` layers has `n + 1`
/// places to land in, so naming siblings covers all of them but one: the place
/// **under the bottom layer**, which has no sibling below it to be named after.
/// [`Bottom`](Self::Bottom) is that place. Without it a panel could offer every
/// drop position in a stack except its foot — and "put this behind everything"
/// is not an exotic move, it is where a background goes.
///
/// The variant order is **not** load-bearing: variants are matched by *name* (§8), so
/// a case may be added wherever it reads best.
/// `a_place_is_read_by_variant_name_not_position` is what keeps that honest.
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
/// Everything past `Normal` combines the two layers' **light** rather than covering
/// one with the other — and none of it is Screen. Screen is `a + b − ab`, which is
/// what falls out of inverting a multiply; it describes no physical process, and it
/// crushes the top of the range into a flat, chalky white that is the giveaway of a
/// digital glow.
///
/// Ours are derived the other way round. Two lights *add* — that is the only thing
/// light does — but the numbers in a layer are not light, they are light that has
/// already been through a tone curve on its way to being displayable. So the honest
/// combination is: undo the curve, add, re-apply it. Every mode here is that same
/// sentence with a different curve `T`:
///
/// ```text
///     f(a, b) = T(T⁻¹(a) + T⁻¹(b))
/// ```
///
/// Being a conjugation of addition is not a technicality — it is the whole
/// guarantee. Each mode is commutative and associative with a neutral element, so
/// three glowing layers give the same result in any order and regrouping them
/// changes nothing, exactly as three real lamps would. Screen happens to share those
/// properties (it is addition conjugated by `1 − e^{-x}`'s cousin), which is *why*
/// it survived; these are what you get when the curve is chosen for how light
/// actually rolls off instead of for algebraic convenience.
///
/// **The guarantee holds at any coverage**, which took getting right: a layer's
/// coverage weighs it in the space where its blend function is affine, not in the
/// working space, because applying a curve to a coverage-averaged color is not the
/// same as averaging the curve. Weighed in the working space instead, stacking order
/// matters by up to 20 levels wherever a stroke is less than solid. See
/// `blend_common.wesl`'s `combined_light`, and §18.0.4 for what that costs.
///
/// [`Reinhard`](Self::Reinhard) and [`Drago`](Self::Drago) are the emissive half:
/// they add light and their identity is black. [`Multiply`](Self::Multiply) is the
/// subtractive half — the same construction with `T(x) = e^{-x}`, which makes the
/// added quantity optical density and the identity white. That is the *whole* of
/// what changes between the two halves; the family is one idea, not two. Each half
/// does weigh coverage in a different space — emission for the emissive modes, light
/// itself for `Multiply` — but that is the same rule read twice: **the space where
/// the mode's own blend function is affine**, which is the only space a weighted
/// average commutes with it in.
///
/// The combination happens in **CIE XYZ normalized to the display white**, not in
/// the working color space and not in RGB: XYZ is linear in light, its components
/// are non-negative for every real color (which is what makes the curves
/// well-defined), and normalizing by the white point puts an in-gamut color's
/// components in `[0,1]` — so "1" means the same thing on all three axes. Blending
/// in RGB instead would make the result depend on the display's primaries; blending
/// in Oklab or in pigment concentrations would be adding things that are not light.
///
/// **A mode may carry its own parameters**, and [`Drago`](Self::Drago) does. They live
/// on the variant rather than in a settings struct beside it, because that is the one
/// shape in which a parameter cannot be stated for a mode that has none: there is no
/// `k` on a `Multiply` layer to be edited, saved, replicated and silently ignored. It
/// is also what keeps the merge's "the two layers agree about how they meet the
/// backdrop" (`document::merge`) meaning that once a mode is a family of curves — two
/// `Drago`s with different `k` are two different functions, and `!=` says so.
///
/// A new mode may go wherever it reads best, and a parameterized one may gain a knob:
/// variants and fields are matched by *name* (§8), so neither disturbs the modes in
/// saved files. `a_mode_is_read_by_variant_name_bend_and_all` holds that.
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
    /// Reinhard's curve is asymptotic: no finite amount of light reaches 1. So this
    /// mode **cannot blow out** — stack a hundred glow layers and the result
    /// approaches white without ever clipping, and detail survives everywhere. That
    /// makes it the one to reach for on glazes, mist, rim light and bloom, where
    /// Screen's flat white is exactly the failure.
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
    /// (§6.3) as a genuine bloom with a filmic shoulder, rather than being
    /// clipped at the blend. Reach for it on flame, specular hits, anything meant to
    /// read as *brighter than the paper*.
    ///
    /// `k` sets how quickly the curve bends, and it is **the layer's own**: large
    /// `k` tends to plain addition, so two lights reach the roll-off sooner and a
    /// flame reads hotter; small `k` tends to `max`, so the brighter of the two
    /// simply wins and coincident lights barely add at all. [`DRAGO_K`] is where it
    /// starts and [`DRAGO_K_RANGE`] is how far it goes.
    ///
    /// It is the first blend parameter, and it is on the variant for the reason [the
    /// enum's docs](Self) give. That it is a *curve* being chosen rather than an
    /// amount being dialled is what makes it worth having at all: every setting is
    /// still a conjugation of addition, so the whole family is commutative and
    /// associative — a `k` a painter picks cannot cost them the guarantee the mode
    /// exists for.
    Drago { k: f32 },
    /// **Multiply** — the same construction read the other way round, with
    /// `T(x) = e^{-x}`, which collapses to
    ///
    /// ```text
    ///     f(a, b) = a·b
    /// ```
    ///
    /// The quantity being added is **optical density**, so this is Beer-Lambert:
    /// what two stacked filters, two glazes, or two sheets of stained glass do to
    /// the light passing through them. It is the mode Screen is an inversion *of* —
    /// and of the two it is the one that describes something real, which is why this
    /// is here and Screen is not.
    ///
    /// Everything the emissive modes guarantee still holds, dualised: commutative
    /// and associative, so a stack of glazes is order-independent, but the neutral
    /// element is **white** rather than black. Glaze over bare paper and nothing
    /// happens; glaze over black and nothing shows. Because it runs in normalized
    /// XYZ rather than in RGB, the darkening is a statement about light rather than
    /// about the display's primaries — two saturated glazes cross without the dead
    /// channel that an RGB multiply produces when one primary happens to be near
    /// zero.
    ///
    /// The one mode here that *removes* light, and so the one that never reaches the
    /// media pass's highlight roll-off: its output is in `[0,1]` by construction.
    ///
    /// One consequence to know about. The blend sees the layer stack, not the
    /// **substrate** — the paper is composited in pass B, after all blending
    /// (`media_common.wesl`) — so a glaze laid on bare canvas leaves the paper's own
    /// color untouched instead of tinting it. On white paper that is exactly right,
    /// white being multiply's identity, and it is why the mode reads correctly to a
    /// painter by default. On a toned substrate it is a divergence from what a real
    /// glaze would do, and the fix is not here: it is for the substrate to become the
    /// bottom of the stack rather than a step of the media pass.
    Multiply,
}

/// The bend a [`BlendMode::Drago`] layer **starts at**, in units of display white —
/// what the picker hands out and what the panel's Bend slider rests on. Large `k`
/// tends to plain addition, small `k` tends to `max`.
///
/// Chosen so the two light modes are a genuine choice rather than two settings of one.
/// Take two half-lit layers: [`BlendMode::Reinhard`] gives 0.667, Screen gives 0.75,
/// plain addition gives 1.0 (clipped), and this gives 0.769 — so Glow reads distinctly
/// softer than the mode everyone already knows and Radiance distinctly hotter, across
/// the whole range rather than only at the extremes. At the top, two whites come out
/// at ≈1.36, well into the media pass's highlight roll-off.
///
/// A **default** rather than the value, since the curve is per layer — but a mode's
/// resting setting is the one it is judged by, and this is the one the goldens and the
/// docs' worked example are written against.
pub const DRAGO_K: f32 = 0.6;

/// How far [`BlendMode::Drago`]'s bend may be taken — the span a frontend's slider
/// covers and the span [`BlendMode::sanitized`] holds a log entry to.
///
/// The ends are where the mode stops changing rather than round numbers. At `0.125`
/// two half-lit layers give 0.586 against `max`'s 0.5, so the curve has arrived at
/// "the brighter one wins" and a smaller `k` would only make `e^{y/k}` bigger for
/// nothing. At `4.0` they give 0.944 against addition's 1.0, so it has arrived at the
/// other end; past it the log is straight over the whole display range and Radiance
/// is just a clip waiting to happen.
///
/// Bounded at all for the reason [`ColorAdjust`](super::ColorAdjust)'s knobs are: a
/// blend is a fullscreen pass with no coverage to hide behind, and `k = 0` is a
/// division by zero in `emission` that would take every texel of the frame with it.
/// A file or a peer reaches [`BlendMode::sanitized`] without passing through a
/// slider, which is the case the bound is actually for.
pub const DRAGO_K_RANGE: (f32, f32) = (0.125, 4.0);

impl BlendMode {
    /// Every mode **at its default setting**, in the order a frontend should offer
    /// them: `Normal` first, then increasingly emphatic light, then the one that
    /// takes light away.
    ///
    /// A list of modes, not of settings of them — which is why a picker built from it
    /// selects its current row with [`same_mode`](Self::same_mode) rather than `==`,
    /// and why choosing `Radiance` on a layer that is already `Radiance` is not a
    /// thing the picker can do (so a tuned `k` is never quietly reset by re-picking
    /// the mode it belongs to). It is [`Filter::ALL`](super::Filter::ALL)'s
    /// neutral-settings list read for a smaller enum.
    pub const ALL: [BlendMode; 4] = [
        Self::Normal,
        Self::Reinhard,
        Self::Drago { k: DRAGO_K },
        Self::Multiply,
    ];

    /// What this mode is called. The painter-facing name, not the tonemap's — the
    /// curve is how it is *built*, not what it is *for*.
    ///
    /// `Multiply` is the exception that proves it: there the operation's name and the
    /// painter's name are the same word, and it has been that word in every paint
    /// program for thirty years. Renaming it "Glaze" to match its neighbours would be
    /// inventing a synonym for a term of art nobody needs translated.
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
    /// Distinct from `==`, and both are wanted: this is the question the *frontend*
    /// asks, while `==` is the question the compositor and the merge ask, where two
    /// bends really are two different functions and answering "same mode" would fold
    /// a layer into a curve that is not its own.
    pub fn same_mode(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    /// The curve bend the blend pass's uniform carries — this layer's for
    /// [`Drago`](Self::Drago), and [`DRAGO_K`] for every mode whose shader path never
    /// reads it (`blend_common.wesl` branches on the mode first).
    ///
    /// A plain `f32` rather than an `Option`, because the uniform has one field and
    /// no way to spell "absent": an `Option` here would only be unwrapped to the same
    /// number at both call sites, one of which is the merge and one the compositor.
    /// A live value, so the two cannot drift.
    pub fn drago_k(self) -> f32 {
        match self {
            Self::Drago { k } => k,
            _ => DRAGO_K,
        }
    }

    /// The same mode with every parameter finite and in range — the funnel a mode
    /// passes through on its way into the document, exactly as
    /// [`Filter::sanitized`](super::Filter::sanitized) is for a filter, and applied
    /// in the same two places for the same two reasons: where the action is minted
    /// (`Engine::process`), so the log records what was applied, and where a mode
    /// enters state (`DocState::set_layer_blend`), because a loaded file or a remote
    /// peer reaches state without passing through `process`.
    ///
    /// A non-finite `k` falls back to [`DRAGO_K`] rather than to a bound, on
    /// [`ColorAdjust::sanitized`](super::ColorAdjust::sanitized)'s argument: `NaN`
    /// says nothing about which end was meant, and the default is the one answer that
    /// cannot make a picture worse.
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
    /// isolation and draws straight into the accumulator, so an ordinary document
    /// costs exactly what it did before blend modes existed (§6.3).
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
}

/// The region a matte layer fills (§15.2).
///
/// A region is a coverage field over the *infinite* plane, so what matters is its
/// value at infinity — which is what makes the frame case (fill everywhere except
/// a rect) expressible at all, and expressible without a mask.
///
/// Its geometry is stated **in the layer's frame** (§14.12), like every fact a
/// paint action states about geometry: the layer's `translation` places it on the
/// canvas, added by the compositor, the export framing and the projection on the
/// way out — which is what lets a matte answer `TranslateLayers` with the same
/// property write a paint layer does. `SetMatteRect` writes this frame's
/// coordinates; the canvas-space handles are the command tier's business, converted
/// where the gesture becomes an action, exactly as a fill's shape is.
///
/// It is stored as **geometry, not a rasterized mask**: the fill is evaluated
/// analytically from a signed distance at canvas position, exactly as
/// `selection.wesl` does (§6.8). That costs no tiles (a 4000² frame would
/// otherwise be ~16 MB of mask and could trip `MAX_SELECTION_TILES`), stays exact
/// at any zoom, keeps the log to four floats, and — being a pure function of
/// canvas position — satisfies the §6.4 seam invariant for free.
///
/// Two variants, because two are built — the frame, and the §15.2 table's third row,
/// the whole-plane backing ([`Everything`](Self::Everything), §15.5's "opaque
/// underpainting"). This is still the seam where the `SelectionOp` algebra lands
/// (§15.9, P4), bringing comic gutters, lasso mattes and frame-from-selection at once;
/// per §1, no variant appears here before it does something.
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
    /// The rect this region is defined against, in the layer's frame
    /// (canvas-px units, §14.12) — for
    /// [`OutsideRect`](Self::OutsideRect) the *hole*, the piece, which is what
    /// export frames against (§15.6). `None` for a region that is not defined
    /// against one: an [`Everything`](Self::Everything) matte frames nothing,
    /// and every consumer of the rect (export, the aspect readout, the handle
    /// box) has a real answer for that — fall back or stand down — rather than
    /// a made-up rectangle.
    pub fn rect(&self) -> Option<(Vec2, Vec2)> {
        match self {
            Self::OutsideRect { min, max } => Some((*min, *max)),
            Self::Everything => None,
        }
    }

    /// Whether this region may be applied at all: its rect, if it has one, is
    /// measurable. Deterministic, so peers and replays agree about rejection —
    /// exactly [`TransformMap::usable`](super::transform::TransformMap::usable)'s
    /// contract, and here for its reason.
    ///
    /// **Refused rather than clamped**, which is the whole of why this is a
    /// predicate and not a `sanitized`. A frame is a rectangle the artist placed;
    /// there is no other rectangle that is a repaired version of one nobody can
    /// measure, and rounding it to the origin would silently reframe the piece —
    /// the export rect, the aspect readout and the handle box all read this. The
    /// same argument §16.1 makes for an unusable affine, which is also why the
    /// matte's *paint* is sanitized where its *geometry* is gated: a color out of
    /// range has an obvious nearest legal value and a rect does not.
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
    /// rect on the canvas, and — negated — a canvas-space gesture into the frame
    /// at the mint, [`Parcel::translated`](super::Parcel::translated)'s pair.
    /// [`Everything`](Self::Everything) has no position and rides through, the
    /// reading that same pair gives a solid.
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
    /// (§8).
    ///
    /// `Old` is the hazard made concrete: the same three cases in a different order,
    /// written by a build that declared them that way. It must read back exactly.
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
    /// wrong one recomposites every layer that used it.
    ///
    /// `Drago` is the sharp case: it carries a payload and sits in the middle of the
    /// declaration order, so here it is written from the far end of the enum and must
    /// still arrive as itself, bend and all.
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
    ///
    /// The rect variant is written from the far end of the declaration and must
    /// still arrive as itself, both corners and all.
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
