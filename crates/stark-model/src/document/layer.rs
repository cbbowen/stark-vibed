//! Layers (§5.1, §15.2, §14, §21). A layer is a sparse, persistent map of painted
//! tiles, a **matte** — a procedural region filled with a flat color — or a
//! **filter**, which is a function of what is composited beneath it rather than
//! content of its own; plus its presentation properties, plus the layers it
//! **carries**.
//!
//! A layer stacks with premultiplied "over" unless its [`BlendMode`] says
//! otherwise or it is [`clip`](Layer::clip)ped, in which case the compositor
//! isolates it and merges it through the mode (§18.0.4). A
//! layer that carries others is a **group** — there is no separate group type —
//! and the same isolation, recursed, is what composites it (§14.7).

use serde::{Deserialize, Serialize};

use super::action::ActorId;
use crate::geom::Vec2;

/// Stable identifier for a layer within a document.
///
/// Ids are **minted from the author**, not from a shared counter — see
/// [`LayerId::mint`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

impl LayerId {
    /// Mint the id for `actor`'s `n`th layer (§17.9).
    ///
    /// Two peers adding a layer at the same moment must not mint the same id. A
    /// counter resynced from the log does exactly that — both peers see `n` layers,
    /// both mint `n + 1`, and the log ends up holding two different layers under one
    /// id, which `layer_index` resolves to whichever comes first. That is a genuine
    /// convergence failure, so the id space is partitioned by author instead: a
    /// mixed 32-bit fold of the actor in the high half, the per-actor counter in the
    /// low.
    ///
    /// [`ActorId::SOLO`] maps to high half 0, so a document that was never shared
    /// keeps the small, readable ids it always had — including the root layer's
    /// `LayerId(0)`, which every peer must agree on because it predates any actor.
    pub fn mint(actor: ActorId, n: u64) -> Self {
        let hi = if actor == ActorId::SOLO {
            0
        } else {
            // Never 0: that is SOLO's space, and colliding with it would clash with
            // the layers a document had before it was ever shared.
            mix32(actor.0).max(1)
        };
        LayerId((u64::from(hi) << 32) | (n & 0xFFFF_FFFF))
    }

    /// The per-actor counter this id was minted from — the inverse of the low half
    /// of [`mint`](Self::mint).
    pub fn ordinal(self) -> u64 {
        self.0 & 0xFFFF_FFFF
    }

    /// Whether this id was minted by `actor`, so the engine can resume that actor's
    /// counter from a log without also resuming everyone else's.
    pub fn minted_by(self, actor: ActorId) -> bool {
        self.0 >> 32 == Self::mint(actor, 0).0 >> 32
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
/// **The variant order is load-bearing: this must stay wire-compatible with
/// `Option<LayerId>`.** Postcard writes an `Option` as `0` for `None` / `1` for `Some`
/// and an enum as its variant index, so `Top` and `Above` occupy exactly the `None`
/// and `Some` discriminants and `Bottom` is an *appended* third variant — the one
/// shape §8 allows without a format break. `place_encodes_as_an_option_layer_id`
/// below is what keeps the claim honest.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// splitmix64's finalizer, folded to 32 bits: decorrelates the bits an
/// endpoint-derived [`ActorId`] takes verbatim from a public key.
fn mix32(x: u64) -> u32 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 32) as u32
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
/// **A mode may carry its own parameters**, and [`Drago`](Self::Drago) is the first
/// that does. They live on the variant rather than beside it, in a struct of blend
/// settings a layer would carry alongside its mode, because that is the one shape in
/// which a parameter cannot be stated for a mode that has none: there is no `k` on a
/// `Multiply` layer to be edited, saved, replicated and silently ignored, and no way
/// for a mode and its settings to disagree about which mode they are. It is also what
/// makes the merge's "the two layers agree about how they meet the backdrop"
/// (`document::merge`) keep meaning that once a mode is a family of curves rather than
/// one — two `Drago`s with different `k` are two different functions, and `!=` already
/// says so.
///
/// **Appended only, payloads included**: postcard encodes an enum by index and a
/// variant's fields in order (§8), so a variant inserted above an existing one would
/// rename every layer's mode in every saved file, and a field inserted into an
/// existing variant would misread the ones after it.
///
/// See `blend_common.wesl` for the derivations and `Compositor` for the isolation
/// pass that makes per-layer blending possible at all.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// painter by default. On a toned ground it is a divergence from what a real
    /// glaze would do, and the fix is not here: it is for the substrate to become the
    /// bottom of the stack rather than a step of the media pass.
    Multiply,
}

/// The bend a [`BlendMode::Drago`] layer **starts at**, in units of display white —
/// what the picker hands out and what the panel's Bend slider rests on. Large `k`
/// tends to plain addition, small `k` tends to `max`.
///
/// Chosen so the two light modes are a genuine choice rather than two settings of
/// one. Take two half-lit layers: [`BlendMode::Reinhard`] gives 0.667, Screen gives
/// 0.75, plain addition gives 1.0 (clipped), and this gives 0.769. So Glow reads
/// distinctly softer than the mode everyone already knows and Radiance reads
/// distinctly hotter, across the whole range instead of only at the extremes — which
/// is what a value near 0.35 gave, and the reason it is not that. At the top, two
/// whites come out at ≈1.36, well into the media pass's highlight roll-off.
///
/// It is a **default** rather than the value now that the curve is per layer, and it
/// keeps its argument: a mode's resting setting is the one it is judged by, and this
/// is the one the goldens and the docs' worked example are written against.
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
    pub fn sanitized(self) -> Self {
        match self {
            Self::Drago { k } => Self::Drago {
                k: if k.is_finite() {
                    k.clamp(DRAGO_K_RANGE.0, DRAGO_K_RANGE.1)
                } else {
                    DRAGO_K
                },
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
/// It is stored as **geometry, not a rasterized mask**: the fill is evaluated
/// analytically from a signed distance at canvas position, exactly as
/// `selection.wesl` does (§6.8). That costs no tiles (a 4000² frame would
/// otherwise be ~16 MB of mask and could trip `MAX_SELECTION_TILES`), stays exact
/// at any zoom, keeps the log to four floats, and — being a pure function of
/// canvas position — satisfies the §6.4 seam invariant for free.
///
/// Two variants, because two are built — the frame, and the §15.2 table's third
/// row, the whole-plane ground ([`Everything`](Self::Everything), §15.5's
/// "opaque underpainting"). This is still the seam where the `SelectionOp`
/// algebra lands (§15.9, P4), bringing comic gutters, lasso mattes and
/// frame-from-selection at once. Per this codebase's own precedent (§1 —
/// `drag` and `wetness` were deleted rather than kept inert, and `bleed` and
/// `tooth` came back only once each had a model), no variant appears here before
/// it does something.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MatteRegion {
    /// Everything *outside* this canvas-space rect — the frame / mat board.
    OutsideRect { min: Vec2, max: Vec2 },
    /// The whole plane — a ground / underpainting, made to sit at the bottom of
    /// the stack (§15.5). It has no rect: it frames nothing, so it
    /// defines no export rect and mounts no handles — the coverage is the whole
    /// of what it says.
    Everything,
}

impl MatteRegion {
    /// The rect this region is defined against, in canvas px — for
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

    /// The same region with its rect replaced (the frame drag's commit) — a
    /// no-op on a region that has none, matching `SetMatteRect`'s no-op on a
    /// layer that is not a matte: the action names a property this region does
    /// not have.
    pub fn with_rect(&self, min: Vec2, max: Vec2) -> Self {
        match self {
            Self::OutsideRect { .. } => Self::OutsideRect { min, max },
            Self::Everything => Self::Everything,
        }
    }
}

/// What a matte is filled with (§15.4, §22): one flat color, or a
/// gradient read from canvas position — the same ramp the fill lays (§22.4),
/// embedded by value the same way.
///
/// No opacity of its own in either variant: a matte's transparency *is* its
/// layer opacity (§15.3), and its paint is a full-strength coat — which is why
/// the solid keeps three channels, not four, and the gradient carries no
/// per-unit opacity where the fill's parcel does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MattePaint {
    /// One color everywhere. Straight sRGB, like [`BrushParams::color`],
    /// converted to working-space channels at composite time.
    ///
    /// [`BrushParams::color`]: crate::document::BrushParams::color
    Solid([f32; 3]),
    /// A color ramp along an axis (§22.4): interpolated per fragment in the
    /// working space, so an Oklab document's matte matches the library strip
    /// and a Mixbox document's is a pigment ramp — a graded wash, not a screen
    /// gradient.
    Gradient {
        gradient: crate::gradient::Gradient,
        axis: super::fill::GradientAxis,
    },
}

impl MattePaint {
    /// The color a one-swatch summary shows: the solid itself, or the ramp's
    /// start — the stop the axis anchors on.
    pub fn swatch(&self) -> [f32; 3] {
        match self {
            Self::Solid(c) => *c,
            Self::Gradient { gradient, .. } => gradient.sample(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Place`] rides a logged action **byte-for-byte as an `Option<LayerId>`**:
    /// postcard encodes an `Option` as a `0`/`1` discriminant and an enum as its
    /// variant index, so the two `Option`-shaped cases hold those indices and anything
    /// further is appended (§8).
    ///
    /// Asserted rather than reasoned about, because the failure is silent in exactly
    /// the way §8 warns: reorder the variants and every `MoveLayer` in every saved
    /// document decodes as a *different* move, with nothing in the file able to notice.
    #[test]
    fn place_encodes_as_an_option_layer_id() {
        let id = LayerId(0x1234_5678_9ABC_DEF0);
        assert_eq!(
            postcard::to_allocvec(&Place::Top).expect("encodes"),
            postcard::to_allocvec(&None::<LayerId>).expect("encodes"),
            "Top must occupy None's discriminant"
        );
        assert_eq!(
            postcard::to_allocvec(&Place::Above(id)).expect("encodes"),
            postcard::to_allocvec(&Some(id)).expect("encodes"),
            "Above must occupy Some's discriminant, payload and all"
        );
        // And the new case is appended, so nothing that existed had to move.
        assert_eq!(
            postcard::to_allocvec(&Place::Bottom).expect("encodes"),
            vec![2u8],
            "Bottom is the third variant"
        );
    }

    /// [`BlendMode`]'s parameterless modes still encode as a bare variant index, and
    /// [`BlendMode::Drago`] as its index followed by its payload — the shape §8
    /// promises for an enum, asserted here for the same reason the test above is.
    ///
    /// This is the fact the "appended only, payloads included" rule protects, and the
    /// one a reader would otherwise have to take on trust: a field added *before* `k`
    /// in a later revision of the variant would decode every saved bend as something
    /// else, silently.
    #[test]
    fn a_mode_encodes_as_its_index_and_its_payload() {
        let bytes = |m: BlendMode| postcard::to_allocvec(&m).expect("encodes");
        assert_eq!(bytes(BlendMode::Normal), vec![0u8]);
        assert_eq!(bytes(BlendMode::Reinhard), vec![1u8]);
        assert_eq!(bytes(BlendMode::Multiply), vec![3u8]);
        let drago = bytes(BlendMode::Drago { k: DRAGO_K });
        assert_eq!(drago[0], 2, "Radiance kept the index it shipped with");
        assert_eq!(
            &drago[1..],
            &postcard::to_allocvec(&DRAGO_K).expect("encodes")[..],
            "and `k` follows it, as the variant's only field",
        );
        assert_eq!(
            postcard::from_bytes::<BlendMode>(&drago).expect("decodes"),
            BlendMode::Drago { k: DRAGO_K },
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
}
