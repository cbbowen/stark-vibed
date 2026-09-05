//! The brush (§6.2, §6.6): what a stroke is *made with*, as opposed to what a
//! stroke *is*.
//!
//! Everything here is configuration a [`StrokeRecord`](super::action::StrokeRecord)
//! carries — the tip's shape and orientation, the four flux axes that decide how
//! it meets paint already on the canvas, the color jitter, the tapers, and the
//! mapping from pen input to all of the above. None of it is an action; it is the
//! payload one action kind happens to hold, which is why it lives beside
//! `action.rs` rather than inside it.
//!
//! Three layers, and the dependency runs one way through them: `modulation` — what
//! the pen may take away — knows nothing of `effect` — what the swept extent then
//! does to the canvas — and this file, the tip that carries both, reads each.
//!
//! Two properties hold across all three, and most of the design follows from them:
//!
//! - **A modulation can only ever scale a parameter down** — [`Modulation`] for what
//!   that bound buys.
//! - **Every number here decides stored pixels**, so replay, goldens and peers have
//!   to agree on it to the last bit (§12.1). That is why the response curve is
//!   rational rather than a `powf`, and why the values that arrive from files,
//!   presets and peers are clamped on the way in rather than trusted.

use serde::{Deserialize, Serialize};

use crate::sanitize::{at_least_zero, clamp01, finite_or};

mod effect;
mod modulation;

// The layers' own public types, lifted so `document::brush` still means one thing
// from outside — `document`'s re-export list is untouched.
pub use effect::{
    BrushDynamics, BrushEffect, ColorDynamics, EraseEffect, LiquifyEffect, NoiseKind, PaintEffect,
    WetEffect,
};
use modulation::Modulations;
pub use modulation::{
    BrushModulations, EraseModulations, LiquifyModulations, ModSource, Modulation,
    PaintModulations, PenState, WetModulations,
};

/// The brush tip shape (§6.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum BrushShape {
    /// Procedural soft disc. Edge softness in [0, 1]: 0 = very soft, 1 = the hard
    /// edge (supported, and what a shipped preset uses) — meaningful only here,
    /// since it is what shapes this tip's falloff.
    Round { hardness: f32 },
    /// A sampled coverage mask, referenced by content id (an imported image).
    Stamp(crate::AssetId),
}

impl Default for BrushShape {
    fn default() -> Self {
        Self::Round {
            hardness: Self::DEFAULT_HARDNESS,
        }
    }
}

impl BrushShape {
    /// The round tip's hardness fallback: used both as [`Default`] and when a
    /// `Stamp` asset fails to resolve and rendering falls back to the round tip
    /// (§6.6), which by construction has no hardness of its own to fall
    /// back on.
    pub const DEFAULT_HARDNESS: f32 = 0.5;

    /// The same tip with its hardness inside the `[0, 1]` it is quoted in — see
    /// [`BrushParams::sanitized`]. A `Stamp` carries no number to hold.
    pub fn sanitized(self) -> Self {
        match self {
            Self::Round { hardness } => Self::Round {
                hardness: clamp01(hardness),
            },
            Self::Stamp(id) => Self::Stamp(id),
        }
    }
}

/// What sets the brush shape's orientation as it sweeps along the stroke
/// (§6.6). The swept-depth integral runs along the stroke's travel direction, so the
/// shape is looked up in a per-orientation prefix-τ texture indexed by the *relative*
/// angle between the shape's native axis and the travel direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default, carbonite::Schema)]
pub enum OrientationSource {
    /// The shape's native axis tracks the stroke tangent — the relative angle is always
    /// 0, so the extent always faces along the motion (the historical behaviour).
    #[default]
    FollowStroke,
    /// The shape stays pinned to the pen's orientation (the tilt azimuth) in canvas
    /// space; as the stroke curves under a fixed pen the extent angle stays put,
    /// like a calligraphy nib.
    Pen,
}

/// The brush's two knobs against the canvas substrate's tooth (§6.4) — one
/// pair, because they are the two halves of one contact model: how far the tip
/// settles, and how sharply it stops. The *substrate* is document state
/// ([`SubstrateId`](crate::SubstrateId)) — a pencil and a loaded brush on the same
/// canvas see the same grain — which is why only these two knobs live on the brush.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct ToothParams {
    /// How much **give** this tool has, in [0, 1]: 1 = infinite give — the tip
    /// follows every fall, the substrate does not break the mark up at all, and
    /// this is the default; 0 = no give at all — it rides the very tops of the
    /// grain, so the mark is what a dry brush leaves.
    ///
    /// **The give, not its inverse, and that is what makes the pen mapping mean
    /// something.** A [`Modulation`] can only ever scale a parameter *down*, so a knob
    /// quoted as the depth of the bite would have made light pressure the solid mark
    /// and a hard press the dry one — backwards for the one mapping this axis exists
    /// for. Quoted as the give, pressure reads the way a hand expects:
    /// [`BrushModulations::tooth_give`] mapped to pressure is the charcoal, barely
    /// touching the paper it prints the peaks alone, and borne down it presses past
    /// the falls it was bridging and the grain fills in.
    ///
    /// It does cost the slider its usual direction — the *interesting* end of this one
    /// is the left. That is the trade, and the mapping is worth more than the habit.
    ///
    /// What it scales is the exposure the stroke presents per unit swept optical
    /// depth, gated per texel by whether the substrate clears the level this tool
    /// settles to (`paint_common.wesl::tooth_gate`). Inert on a `Flat` canvas
    /// whatever it says, because the substrate's rise is 0 there — so the axis is
    /// orthogonal to every golden that paints on `Flat`.
    ///
    /// `#[serde(default = "…")]` and not a bare `#[serde(default)]`: 0 is *maximum*
    /// tooth at this end of the knob, so a file that does not mention the field has
    /// to be told the full give it meant rather than handed a zero (§8).
    #[serde(default = "ToothParams::default_give")]
    pub give: f32,
    /// The **width of the contact transition**, in the rise's own units — height per
    /// reach of travel (§6.4). The band of rise, either side of the follow limit
    /// [`give`](Self::give) sets, over which a texel goes from taking none of the
    /// tip's paint to taking all of it.
    ///
    /// A hard threshold — 0 — is a binary indicator per texel: correct in the mean,
    /// and at canvas resolution it aliases into hard-edged speckle that reads as
    /// dither rather than as tooth. Too wide and the faces are smeared into a flat
    /// grey and the grain stops reading at all.
    /// [`DEFAULT_SOFTNESS`](Self::DEFAULT_SOFTNESS) is the bundled substrates' own
    /// interquartile rise, so the transition spans the grain's natural variation —
    /// which is what a paint that *sits on* the substrate wants. A charcoal or a
    /// soft graphite does not sit on the substrate: the stick crumbles into the
    /// valleys instead of spanning them, so its contact comes on gradually over
    /// several times that band, and a knob is the only way to say so.
    ///
    /// **A width, so it has no ceiling this crate owns** — the same reading as the
    /// flow and the drain ([`BrushParams::sanitized`]). Past about twice the
    /// encodable rise range the whole distribution is inside the band and the gate is
    /// a flat scale factor; that is where the frontend's slider stops, not where the
    /// quantity stops meaning something.
    ///
    /// `#[serde(default = "…")]` rather than a bare `#[serde(default)]`, because a
    /// file saved before this field meant the constant the shader used to carry —
    /// and 0, which is what a plain default hands back, is the hard threshold and
    /// not that (§8).
    #[serde(default = "ToothParams::default_softness")]
    pub softness: f32,
}

impl Default for ToothParams {
    fn default() -> Self {
        Self {
            give: Self::DEFAULT_GIVE,
            softness: Self::DEFAULT_SOFTNESS,
        }
    }
}

impl ToothParams {
    /// The give a brush has when it does not say ([`give`](Self::give)): all of it,
    /// so the tip follows every fall and the substrate breaks nothing up — the mark
    /// a brush made before the tooth existed, to the bit.
    ///
    /// Named rather than spelled `1.0` at the places that need it, because *which*
    /// end of this knob is the inert one is the fact worth being able to look up.
    pub const DEFAULT_GIVE: f32 = 1.0;

    /// [`DEFAULT_GIVE`](Self::DEFAULT_GIVE) as a function, for `#[serde(default = "…")]`.
    fn default_give() -> f32 {
        Self::DEFAULT_GIVE
    }

    /// The contact transition a brush gets when it does not say
    /// ([`softness`](Self::softness)) — the bundled substrates' own interquartile
    /// rise, so the band spans the grain's natural variation.
    ///
    /// **The one place this number is written.** It was `paint_common.wesl`'s
    /// `const TOOTH_SOFTNESS` until the knob existed; the shader takes it as a
    /// uniform now and declares nothing, which is what keeps the default from being
    /// a host transcription of a shader constant (§6.10).
    pub const DEFAULT_SOFTNESS: f32 = 0.5;

    /// [`DEFAULT_SOFTNESS`](Self::DEFAULT_SOFTNESS) as a function, for
    /// `#[serde(default = "…")]` — which takes a path to call and cannot name a
    /// constant.
    fn default_softness() -> f32 {
        Self::DEFAULT_SOFTNESS
    }

    /// Both knobs held to what they are: the give to its quoted `[0, 1]`, the
    /// softness a finite non-negative width — [`BrushParams::sanitized`]'s rules,
    /// stated where the fields live.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            give: clamp01(finite_or(self.give, d.give)),
            softness: at_least_zero(self.softness, d.softness),
        }
    }
}

/// Brush configuration: what a stroke's record carries — the shape of the
/// tip, how the swept extent builds, and the effect it has on the canvas.
/// Every field here is read whatever the [`effect`](Self::effect); a knob only
/// one effect consumes lives inside that effect's own variant, the pigment
/// ([`PaintEffect::color`]) above all.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct BrushParams {
    /// How large the mark is, in canvas pixels at full pressure: **the disc the
    /// mark fits in**, for every shape (§6.6). Not a radius — a stamp is
    /// normalized to this reach whatever its own aspect, which is what lets one
    /// number mean the same thing across a round tip and an asset.
    pub size: f32,
    /// Brush tip shape (§6.6).
    pub shape: BrushShape,
    /// How far the extent **elongates along the brush's facing axis** (§6.6), in
    /// `[0, 1)`: the tip is stretched by [`elongation`](Self::elongation)
    /// `s = 1/(1 − stretch)` along that axis and left alone across it, so 0 is the
    /// extent the shape draws and 0.5 is one twice as long as it is wide.
    ///
    /// **The axis is [`orientation`](Self::orientation)'s**, not a second direction to
    /// set. That is the whole of why this is one number: the brush already says which
    /// way it faces, and a tip elongates along the way it faces. With
    /// [`OrientationSource::Pen`] the axis is the pen's tilt azimuth, so a
    /// [`ModSource::Tilt`] mapping onto this is the pencil — lean the pen and the
    /// contact patch draws out along the lean, exactly as a real conical tip's does,
    /// which is also why the mapping is a reciprocal (a cone leaning at `θ` contacts
    /// over `1/cos θ`) rather than a straight ramp. Held there it is a chisel nib, off
    /// a round tip and with no stamp asset at all.
    ///
    /// The renderer never stretches the *mask*: a swept integral of a stretched
    /// extent is the unstretched one read at another angle, over another travel,
    /// with a factor on the result (§6.6), so the prefix-τ volume the brush already
    /// binds is the volume this reads. What it does cost is extent area — the tip
    /// reaches `s` times as far along its axis, so the tiles a segment touches, and
    /// the dynamics loop's dispatch over them, grow with it. That is what
    /// [`MAX_ELONGATION`](Self::MAX_ELONGATION) bounds.
    #[serde(default)]
    pub stretch: f32,
    /// Length of the stroke's **leading taper** — the run over which the tip widens
    /// from a point to its full [`size`](Self::size) — in *units of `size`*,
    /// so 4.0 means four brush sizes of taper (§6.2). 0 = no taper: the
    /// stroke starts at full width, which is the historical behaviour.
    ///
    /// In radii rather than canvas px so a brush keeps its *look* as it is resized:
    /// scale the tip up and the whole mark scales with it, instead of a taper that
    /// was the shape of the stroke at radius 8 becoming a blunt nub at radius 80.
    /// This is what lets a brush mimic an inker's entry stroke.
    #[serde(default)]
    pub start_taper_length: f32,
    /// Length of the stroke's **trailing taper**, in units of
    /// [`size`](Self::size) — [`start_taper_length`](Self::start_taper_length)
    /// measured back from the end of the stroke, for the exit of an inked line.
    ///
    /// Together the two are held to the stroke's own length: if they would overlap
    /// they are scaled down in proportion, so a short flick is a small pointed mark
    /// rather than a sliver (see `gpu::stroke::segments::Taper`).
    #[serde(default)]
    pub end_taper_length: f32,
    /// Reservoir depletion per **radius** travelled: the stroke thins as paint runs
    /// out (§6.2). 0 = inexhaustible — which is what a pen, a charcoal
    /// stick, or an ordinary digital brush wants; a physical loaded brush wants a
    /// small positive value, and 1 is a tool bone dry one radius past the press.
    ///
    /// In radii rather than canvas px for the reason the tapers are
    /// ([`start_taper_length`](Self::start_taper_length)), and it is the stronger
    /// case of the two: [`size`](Self::size) is meant to be a pure *scale* on
    /// the mark, and a falloff quoted in canvas px is exactly what that scale does
    /// not carry — enlarge such a brush and it runs dry a fraction of the way into
    /// its own tip, which is not a bigger version of anything.
    /// [`drain_px`](Self::drain_px) is where it becomes the per-px rate every render
    /// path reads. Shared by both effects: an eraser runs dry as honestly as a
    /// loaded brush does.
    pub drain: f32,
    /// The brush's two knobs against the canvas substrate's tooth (§6.4).
    #[serde(default)]
    pub tooth: ToothParams,
    /// The deposit jitter (§6.2) — color dynamics' sibling for the *amount*: every
    /// texel of a stroke scales the exposure it presents by a factor uniform in
    /// `(1 − ε, 1 + ε)`, `ε` in `[0, 1]` (past 1 the gate would go negative, which
    /// is not a stronger setting but a meaningless one), keyed on the canvas texel
    /// and the stroke's seed. What it buys is freedom from banding: what the
    /// exchange loop's iterative accumulation would pile into coherent bands lands
    /// as per-texel dither, because neighbouring texels accumulate at decorrelated
    /// phases. 0 is the exact gate 1 — bit-identical to the unjittered deposit.
    ///
    /// Outside the [`effect`](Self::effect) because it is not one effect's knob: it
    /// gates the exposure *every* path presents — the swept fast path, the stamp
    /// loop and the erase pass alike — so it sits beside [`tooth`](Self::tooth),
    /// the other per-texel factor on what a stroke presents.
    #[serde(default = "BrushParams::default_jitter")]
    pub jitter: f32,
    /// What orients the shape as it sweeps (§6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`.
    #[serde(default)]
    pub orientation: OrientationSource,
    /// What a stroke of this brush **does**: paint, or erase. The tool's identity,
    /// and each variant carries the knobs that exist only under it — see
    /// [`BrushEffect`].
    #[serde(default)]
    pub effect: BrushEffect,
    /// The pen mappings every brush has, whatever its effect — the tip's own
    /// (§6.2). [`BrushModulations::PRESSURE_SIZE`] by default: the pressure →
    /// radius scaling, held here as data so a preset can drop it or aim it
    /// elsewhere. An effect's own rates are mapped where they live
    /// ([`PaintEffect::modulation`], [`EraseEffect::modulation`]).
    ///
    /// `#[serde(default = "…")]` and not a bare `#[serde(default)]`, for
    /// [`ToothParams::give`]'s reason (§8): [`BrushModulations`]'s own `Default` is
    /// **no mapping at all**, and a bare attribute would hand a file that predates
    /// this field the empty set — which is not what its absence meant. Before the
    /// mappings were data, the pressure → radius scaling was wired into the segment
    /// generator, so every such stroke replays at full width with its taper gone:
    /// a silent, pixel-changing read of an older log, which is the one thing the
    /// format's name-based reconciliation exists to prevent.
    #[serde(default = "BrushModulations::pressure_size")]
    pub modulation: BrushModulations,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            size: 16.0,
            shape: BrushShape::default(),
            stretch: 0.0,
            start_taper_length: 0.0,
            end_taper_length: 0.0,
            drain: 0.0,
            tooth: ToothParams::default(),
            jitter: Self::DEFAULT_JITTER,
            orientation: OrientationSource::default(),
            effect: BrushEffect::default(),
            modulation: BrushModulations::PRESSURE_SIZE,
        }
    }
}

impl BrushParams {
    /// The [`PaintEffect`] in force, on a brush that lays plain paint — the
    /// reading every consumer of a paint-only knob goes through, so "this brush
    /// does not have that knob" is a `None` rather than a number that lies.
    pub fn paint(&self) -> Option<&PaintEffect> {
        match &self.effect {
            BrushEffect::Paint(p) => Some(p),
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`paint`](Self::paint), writable.
    pub fn paint_mut(&mut self) -> Option<&mut PaintEffect> {
        match &mut self.effect {
            BrushEffect::Paint(p) => Some(p),
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The [`WetEffect`] in force, on a brush that works wet paint —
    /// [`paint`](Self::paint) for the loop's own knobs.
    pub fn wet(&self) -> Option<&WetEffect> {
        match &self.effect {
            BrushEffect::Wet(w) => Some(w),
            BrushEffect::Paint(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`wet`](Self::wet), writable.
    pub fn wet_mut(&mut self) -> Option<&mut WetEffect> {
        match &mut self.effect {
            BrushEffect::Wet(w) => Some(w),
            BrushEffect::Paint(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The brush turned wet in place, and its effect handed back — the editor's
    /// own gesture when a flux slider is first raised on a plain brush, and the
    /// shorthand a test builds a smearing brush with.
    ///
    /// What both kinds hold carries over — the pigment, the opacity and the flow
    /// (each with its mapping), the color dynamics — and the fluxes start at
    /// zero with `add` at its full share, so wetting a brush and touching
    /// nothing lays the paint it always laid. A brush already wet is handed back
    /// untouched; an eraser keeps its opacity and its flow — the rate means the
    /// same thing on every effect — and takes the default wet brush for the
    /// rest.
    pub fn make_wet(&mut self) -> &mut WetEffect {
        self.effect = match self.effect {
            BrushEffect::Wet(w) => BrushEffect::Wet(w),
            BrushEffect::Paint(p) => BrushEffect::Wet(WetEffect {
                color: p.color,
                opacity: p.opacity,
                flow: p.flow,
                dynamics: BrushDynamics::default(),
                color_dynamics: p.color_dynamics,
                modulation: WetModulations {
                    flow: p.modulation.flow,
                    opacity: p.modulation.opacity,
                    ..WetModulations::default()
                },
            }),
            BrushEffect::Erase(e) => BrushEffect::Wet(WetEffect {
                opacity: e.opacity,
                flow: e.flow,
                modulation: WetModulations {
                    flow: e.modulation.flow,
                    opacity: e.modulation.opacity,
                    ..WetModulations::default()
                },
                ..WetEffect::default()
            }),
            // A liquify brush shares nothing the wet effect could keep — no
            // pigment, no ceiling, and its one rate means "follow", not "lay" —
            // so it takes the default wet brush whole, as the eraser nearly does.
            BrushEffect::Liquify(_) => BrushEffect::Wet(WetEffect::default()),
        };
        self.wet_mut().expect("just made wet")
    }

    /// The [`EraseEffect`] in force, on a brush that erases — [`paint`](Self::paint)
    /// from the other side.
    pub fn erase(&self) -> Option<&EraseEffect> {
        match &self.effect {
            BrushEffect::Erase(e) => Some(e),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`erase`](Self::erase), writable.
    pub fn erase_mut(&mut self) -> Option<&mut EraseEffect> {
        match &mut self.effect {
            BrushEffect::Erase(e) => Some(e),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The [`LiquifyEffect`] in force, on a brush that drags the picture —
    /// [`paint`](Self::paint) from the fourth side (§6.13).
    pub fn liquify(&self) -> Option<&LiquifyEffect> {
        match &self.effect {
            BrushEffect::Liquify(l) => Some(l),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Erase(_) => None,
        }
    }

    /// [`liquify`](Self::liquify), writable.
    pub fn liquify_mut(&mut self) -> Option<&mut LiquifyEffect> {
        match &mut self.effect {
            BrushEffect::Liquify(l) => Some(l),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Erase(_) => None,
        }
    }

    /// The pigment a stroke of this brush lays — the laying effect's own color,
    /// and `None` on an eraser, which lays nothing a color could be a property
    /// of (§6.12). The one spelling of "whichever effect carries the color",
    /// so a consumer cannot ask the paint side alone and silently read black
    /// off a wet brush.
    pub fn pigment(&self) -> Option<[f32; 3]> {
        match &self.effect {
            BrushEffect::Paint(p) => Some(p.color),
            BrushEffect::Wet(w) => Some(w.color),
            // Neither lays anything a color could be a property of: the eraser
            // removes (§6.12), the liquify brush moves what is already
            // colored (§6.13).
            BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The color dynamics a stroke of this brush jitters with: the laying
    /// effect's, and the inactive default on an eraser — which has no color to
    /// wander (§6.12). By value because [`ColorDynamics`] is small and `Copy`,
    /// and a borrow would force every eraser call site through a `static`
    /// default.
    pub fn color_dynamics(&self) -> ColorDynamics {
        match &self.effect {
            BrushEffect::Paint(p) => p.color_dynamics,
            BrushEffect::Wet(w) => w.color_dynamics,
            BrushEffect::Erase(_) | BrushEffect::Liquify(_) => ColorDynamics::default(),
        }
    }

    /// The steepest response across **every** active pen mapping — the tip's and
    /// the effect's own — which is what the flattener buys segments against
    /// (`gpu::stroke::budget::flatten_tolerance`). One derivation, so a target
    /// added to either half cannot be missing from the budget.
    pub fn max_slope(&self) -> f32 {
        self.modulation.max_slope().max(self.effect.max_slope())
    }

    /// The deposit jitter a brush gets when it does not say
    /// ([`jitter`](Self::jitter)), sized between two floors. It must clear the f16
    /// tile quantum (relative ≈ 2⁻¹¹ ≈ 0.05%) by a wide margin, so the exchange
    /// loop's stores land at decorrelated phases of the f16 lattice instead of
    /// ratcheting a region coherently into bands — 1% is ~20 quanta of displacement
    /// wherever a gradient is gentle enough to band at all. And it must stay under what reads as texture: at ±1% a strong deposit
    /// shifts by at most a couple of 8-bit levels per texel, at the threshold of
    /// visibility.
    pub const DEFAULT_JITTER: f32 = 0.01;

    /// [`DEFAULT_JITTER`](Self::DEFAULT_JITTER) as a function, for
    /// `#[serde(default = "…")]`.
    fn default_jitter() -> f32 {
        Self::DEFAULT_JITTER
    }

    /// The two taper lengths in **canvas px**: the stored lengths (in radii) scaled
    /// by [`size`](Self::size). Negative or non-finite lengths read as 0 — the
    /// fields arrive from files, presets and peers, and a taper is a length.
    pub fn taper_px(&self) -> (f32, f32) {
        // `f32::max` returns the non-NaN operand, so this also normalizes NaN to 0.
        let r = self.size.max(0.0);
        let px = |len: f32| {
            let px = len.max(0.0) * r;
            if px.is_finite() { px } else { 0.0 }
        };
        (px(self.start_taper_length), px(self.end_taper_length))
    }

    /// The drain falloff in **canvas px⁻¹**: the stored rate (per radius, see
    /// [`drain`](Self::drain)) over [`size`](Self::size) — [`taper_px`](Self::taper_px)
    /// for the reciprocal quantity, and guarding itself the same way, because the
    /// number arrives from files, presets and peers.
    ///
    /// A radius of zero has no reciprocal, so it reads as **inexhaustible** rather
    /// than as the infinity a shader would turn into a NaN falloff. That is the
    /// honest answer as well as the safe one: a tip with no width lays nothing, and
    /// what lays nothing cannot run out.
    pub fn drain_px(&self) -> f32 {
        // `f32::max` returns the non-NaN operand, so a NaN in either field lands on
        // the `is_finite` fallback below by way of a NaN quotient — as does 0/0.
        let px = self.drain.max(0.0) / self.size.max(0.0);
        if px.is_finite() { px } else { 0.0 }
    }

    /// Whether either end of a stroke with this brush tapers.
    pub fn tapers(&self) -> bool {
        let (start, end) = self.taper_px();
        start > 0.0 || end > 0.0
    }

    /// The furthest the extent may be drawn out along its facing axis — what
    /// [`elongation`](Self::elongation) saturates at, and so the factor by which the
    /// worst-case tip outgrows its own radius.
    ///
    /// A bound on *area*, which is why there is one at all: every tile the stretched
    /// tip reaches is a tile the stroke is rasterized into and the dynamics loop
    /// dispatches over, so `s` prices the stroke roughly linearly. Eight is already a
    /// pen laid almost flat; past it the mark stops reading as a wider stroke and
    /// starts reading as a smear the length of the tip.
    pub const MAX_ELONGATION: f32 = 8.0;

    /// The stretch knob's own top: the value at which
    /// [`elongation`](Self::elongation) reaches [`MAX_ELONGATION`](Self::MAX_ELONGATION)
    /// and the knob stops meaning anything (§6.6).
    ///
    /// Here rather than inline at the frontend's stretch slider, because a slider is
    /// only one of the ways a value reaches this field — a file and a peer are two
    /// more, and neither passes through a panel. `MAX_FLOW`'s own doc makes this
    /// argument for the *drag* bindings; the wire is the third way.
    pub const MAX_STRETCH: f32 = 1.0 - 1.0 / Self::MAX_ELONGATION;

    /// The same brush with every number a number, and every number that has a
    /// documented range inside it — the funnel a brush passes through on its way
    /// into the document, exactly as [`Filter::sanitized`](super::Filter::sanitized)
    /// is for a filter (§21.5) and for the same two reasons.
    ///
    /// **It clamps only where this crate already states a range.** The wet axes
    /// (`add` included), the tooth's *give*, either effect's *opacity*, the hardness and the color are
    /// quoted in `[0, 1]` by their own field docs, and so is the deposit
    /// [`jitter`](Self::jitter), whose gate goes negative past 1; the stretch
    /// saturates at [`MAX_STRETCH`](Self::MAX_STRETCH) by construction. Everything
    /// else — the radius, the flows, the drain, the charge, the tapers, the color
    /// dynamics, the tooth's *softness* — is required to be a finite, non-negative
    /// number and nothing more, because the ceilings those have are a *frontend's*
    /// slider ends rather than facts about the quantity, and clamping a document to
    /// one this crate does not own would rewrite brushes that were never wrong.
    ///
    /// Every guard this replaces stays where it is. `taper_px`, `drain_px`,
    /// `elongation` and `stroke_rect` defend themselves against values that never
    /// came through here, which is what keeps an extent honest for a record built
    /// by hand in a test or arriving down a path this funnel does not cover (§12.6).
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            size: at_least_zero(self.size, d.size),
            shape: self.shape.sanitized(),
            // Bounded at the knob's own saturation point rather than at 1: past
            // `MAX_STRETCH` the reciprocal is already pinned, so a larger value
            // stored is a number that cannot mean what it says.
            stretch: finite_or(self.stretch, d.stretch).clamp(0.0, Self::MAX_STRETCH),
            start_taper_length: at_least_zero(self.start_taper_length, d.start_taper_length),
            end_taper_length: at_least_zero(self.end_taper_length, d.end_taper_length),
            drain: at_least_zero(self.drain, d.drain),
            tooth: self.tooth.sanitized(),
            // In `[0, 1]` by the field's own doc: the gate `1 + 2ε·centered` is
            // positive for every ε ≤ 1 and meaningless past it.
            jitter: clamp01(finite_or(self.jitter, d.jitter)),
            orientation: self.orientation,
            effect: self.effect.sanitized(),
            modulation: self.modulation.sanitized(),
        }
    }

    /// [`stretch`](Self::stretch) as the factor the extent is drawn out by along
    /// the facing axis: `s = 1/(1 − stretch)`, clamped to
    /// [`MAX_ELONGATION`](Self::MAX_ELONGATION).
    ///
    /// **Exactly 1 at `stretch = 0`**, which is the whole reason the knob is quoted as
    /// the reciprocal's argument rather than as `s` itself: a brush that never heard of
    /// stretch — and one whose modulation is sitting at a zero floor because the pen is
    /// upright or there is no pen — takes the renderer's identity path bit for bit.
    ///
    /// Takes the modulated knob rather than reading [`stretch`](Self::stretch), because
    /// what a [`Modulation`] scales is the knob and not the factor: scaling `s` towards
    /// 0 would *shrink* the tip across its axis at a low tilt, where scaling the knob
    /// walks `s` back to 1 and leaves the shape alone.
    ///
    /// `min`-then-`max` rather than `clamp`, for `clamp01`'s reason and with more
    /// riding on it: `clamp` returns the NaN where these return the other operand, and
    /// the NaN would reach a lane the shaders divide by.
    #[expect(
        clippy::manual_clamp,
        reason = "min-then-max is the NaN policy; `clamp` returns the NaN these reject"
    )]
    pub fn elongation(stretch: f32) -> f32 {
        // Bounded before the divide rather than clamped after it, so a knob past 1 —
        // or a negative one, which is not a squash but no stretch at all — lands on a
        // real factor instead of on an infinity or an inside-out tip.
        //
        // `min` first and `max` second, and that order is the NaN policy: `f32::min`
        // and `f32::max` return the non-NaN operand (the argument at
        // `clamp01`), so this way a NaN knob falls out as the *identity* and the
        // other way it would fall out as the widest extent the brush can ask for.
        1.0 / (1.0 - stretch).min(1.0).max(1.0 / Self::MAX_ELONGATION)
    }
}

#[cfg(test)]
mod tests {
    use super::modulation::pen;
    use super::*;

    /// **A brush from before the mappings were data still tapers with pressure.**
    ///
    /// The save format reconciles a file's schema against today's types by *name*
    /// (§8), so a field this build added arrives from its serde default — and that
    /// default is the file's only way of saying what its absence meant. For
    /// [`BrushParams::modulation`] the absence meant the pressure → radius scaling
    /// the segment generator used to carry, not [`BrushModulations`]'s own
    /// `Default`, which is no mapping at all.
    ///
    /// The two disagree by a whole taper: `mod_factor(None, ..)` is exactly 1, so
    /// the empty set replays every such stroke at full width. Nothing else in the
    /// suite can see it — the field is present on everything this build writes.
    #[test]
    fn a_brush_from_before_the_mappings_were_data_still_follows_pressure() {
        /// `BrushParams` as it stood before `modulation` — spelled with the names
        /// the real type carries, since names are what carbonite reconciles on, and
        /// carrying only the fields that have no default of their own.
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "BrushParams")]
        struct OldBrush {
            size: f32,
            shape: BrushShape,
            drain: f32,
        }

        let bytes = carbonite::to_vec_static(&OldBrush {
            size: 24.0,
            shape: BrushShape::default(),
            drain: 0.5,
        })
        .expect("encode the old shape");
        let back =
            carbonite::from_slice_static::<BrushParams>(&bytes).expect("an older shape loads");

        assert_eq!(back.size, 24.0, "what both shapes share comes through");
        assert_eq!(
            back.modulation,
            BrushModulations::PRESSURE_SIZE,
            "an absent modulation means the mapping the generator used to carry",
        );
        // Said as the picture rather than as the value: what the wrong default
        // costs is a stroke that no longer thins as the hand lifts.
        assert!(
            back.modulation.size(pen(0.25)) < back.modulation.size(pen(1.0)),
            "a lighter press must still make a narrower stroke",
        );
    }

    /// The ceiling is a target on every effect that has a ceiling, and the
    /// brush says so as one bit (`opacity_modulated`) — the bit the renderer
    /// routes on. Turning a brush wet keeps the mapping, like the flow's.
    #[test]
    fn a_mapped_ceiling_is_a_fact_about_the_brush() {
        let steep = Modulation {
            source: ModSource::Pressure,
            floor: 0.0,
            curve: 1.0,
        };
        let mut b = BrushParams::default();
        assert!(!b.effect.opacity_modulated());
        b.paint_mut()
            .expect("the default brush paints")
            .modulation
            .opacity = Some(steep);
        assert!(b.effect.opacity_modulated());
        // Counted by the flattener's budget like every other target.
        assert!(
            b.max_slope() > 1.0,
            "a steep ceiling response costs segments"
        );

        b.make_wet();
        assert_eq!(b.wet().expect("wet").modulation.opacity, Some(steep));
        assert!(b.effect.opacity_modulated());

        b.effect = BrushEffect::Erase(EraseEffect {
            modulation: EraseModulations {
                opacity: Some(steep),
                ..EraseModulations::default()
            },
            ..EraseEffect::default()
        });
        assert!(b.effect.opacity_modulated());
        b.make_wet();
        assert_eq!(b.wet().expect("wet").modulation.opacity, Some(steep));

        b.effect = BrushEffect::Liquify(LiquifyEffect::default());
        assert!(
            !b.effect.opacity_modulated(),
            "a warp has no ceiling to drive"
        );
    }

    /// The unmodulated brush and every plain linear mapping cost the flattener
    /// nothing: `attribute / 1.0` is the budget it always had.
    #[test]
    fn a_linear_brush_pays_no_extra_flattening() {
        assert_eq!(BrushParams::default().max_slope(), 1.0);
        assert_eq!(BrushModulations::PRESSURE_SIZE.max_slope(), 1.0);
        // A steep one does pay, and the bill is bounded (`MIN_BIAS`) — and it
        // reaches the brush's own slope through either half: the tip's mappings
        // or the effect's.
        let steep = Modulation {
            source: ModSource::Tilt,
            floor: 0.0,
            curve: -1.0,
        };
        let mut through_effect = BrushParams::default();
        through_effect
            .paint_mut()
            .expect("the default brush paints")
            .modulation
            .flow = Some(steep);
        let mut through_tip = BrushParams::default();
        through_tip.modulation.stretch = Some(steep);
        for slope in [through_effect.max_slope(), through_tip.max_slope()] {
            assert!(
                (1.0..=9.0 + 1e-4).contains(&slope),
                "an extreme curve should cost something, and a bounded something: {slope}"
            );
        }
    }

    /// **Nothing that is not a number survives the funnel**, on any field.
    ///
    /// Driven off a poison list applied to every field in turn rather than one
    /// assertion each, so a field added to the brush has an obvious place to be
    /// added and no way to be quietly exempt — the device
    /// `a_view_never_stores_a_number_it_cannot_use` uses for the view's mutators,
    /// and for the same reason: what is being checked is a *class*.
    #[test]
    fn a_sanitized_brush_holds_no_number_a_shader_cannot_use() {
        /// The default brush's paint effect, writable — what a poke at a
        /// paint-only knob reaches through.
        fn paint(b: &mut BrushParams) -> &mut PaintEffect {
            b.paint_mut().expect("the default brush paints")
        }
        /// The brush turned wet and its effect handed back — what a poke at a
        /// wet-only knob reaches through.
        fn wet(b: &mut BrushParams) -> &mut WetEffect {
            b.effect = BrushEffect::Wet(WetEffect::default());
            b.wet_mut().expect("just made wet")
        }
        /// A pen mapping with `f` in one of its two knobs — what a poke at a
        /// [`Modulation`] reaches through. The other knob is left at a value the
        /// funnel would not move, so a failure names the knob it is about.
        fn response(floor: f32, curve: f32) -> Option<Modulation> {
            Some(Modulation {
                source: ModSource::Pressure,
                floor,
                curve,
            })
        }
        type Poke = (&'static str, fn(&mut BrushParams, f32));
        let pokes: [Poke; 26] = [
            ("radius", |b, f| b.size = f),
            ("drain", |b, f| b.drain = f),
            ("erase.opacity", |b, f| {
                b.effect = BrushEffect::Erase(EraseEffect {
                    opacity: f,
                    ..EraseEffect::default()
                })
            }),
            ("erase.flow", |b, f| {
                b.effect = BrushEffect::Erase(EraseEffect {
                    flow: f,
                    ..EraseEffect::default()
                })
            }),
            ("tooth.give", |b, f| b.tooth.give = f),
            ("tooth.softness", |b, f| b.tooth.softness = f),
            ("stretch", |b, f| b.stretch = f),
            ("start_taper", |b, f| b.start_taper_length = f),
            ("end_taper", |b, f| b.end_taper_length = f),
            ("color.r", |b, f| paint(b).color[0] = f),
            ("paint.opacity", |b, f| paint(b).opacity = f),
            ("paint.flow", |b, f| paint(b).flow = f),
            ("wet.flow", |b, f| wet(b).flow = f),
            ("wet.opacity", |b, f| wet(b).opacity = f),
            ("wet.add", |b, f| wet(b).dynamics.add = f),
            ("wet.lift", |b, f| wet(b).dynamics.lift = f),
            ("wet.deposit", |b, f| wet(b).dynamics.deposit = f),
            ("wet.charge", |b, f| wet(b).dynamics.charge = f),
            ("wet.bleed", |b, f| wet(b).dynamics.bleed = f),
            ("jitter.amplitude.0", |b, f| {
                paint(b).color_dynamics.amplitude[0] = f
            }),
            ("jitter.amplitude.1", |b, f| {
                paint(b).color_dynamics.amplitude[1] = f
            }),
            ("jitter.amplitude.2", |b, f| {
                paint(b).color_dynamics.amplitude[2] = f
            }),
            ("jitter.frequency", |b, f| {
                paint(b).color_dynamics.frequency[0] = f
            }),
            ("hardness", |b, f| {
                b.shape = BrushShape::Round { hardness: f }
            }),
            // Both halves of the mapping tree, since they are sanitized through
            // two different owners: the tip's own targets and the effect's.
            ("mod.floor", |b, f| b.modulation.size = response(f, 0.0)),
            ("mod.curve", |b, f| {
                paint(b).modulation.flow = response(0.0, f)
            }),
        ];
        // Every stored number, gathered per effect — the shared fields, the rate
        // either effect carries, and whichever effect's own knobs this brush has.
        let numbers = |b: &BrushParams| {
            let mut v = vec![
                b.size,
                b.drain,
                b.tooth.give,
                b.tooth.softness,
                b.stretch,
                b.start_taper_length,
                b.end_taper_length,
                b.effect.flow(),
                b.effect.opacity(),
            ];
            v.extend(b.color_dynamics().frequency);
            v.extend(b.color_dynamics().amplitude);
            if let Some(w) = b.wet() {
                v.extend([
                    w.dynamics.add,
                    w.dynamics.lift,
                    w.dynamics.deposit,
                    w.dynamics.charge,
                    w.dynamics.bleed,
                ]);
            }
            v
        };
        let unit = |b: &BrushParams| {
            let mut v = vec![b.tooth.give, b.effect.opacity()];
            if let Some(w) = b.wet() {
                v.extend([
                    w.dynamics.add,
                    w.dynamics.lift,
                    w.dynamics.deposit,
                    w.dynamics.bleed,
                ]);
            }
            v
        };
        // The colors are neither of the two lists above: a brush color is extended
        // sRGB, so it may be negative and may pass 1 — what it may not be is
        // unbounded or a `NaN` (§6.5, `Srgb`).
        let colors = |b: &BrushParams| {
            let mut v = vec![];
            v.extend(b.paint().map(|p| p.color).unwrap_or_default());
            v.extend(b.wet().map(|w| w.color).unwrap_or_default());
            v
        };
        // Every pen mapping the brush holds, from both owners: the tip's targets
        // and whichever effect is in force. Its two knobs are neither of the lists
        // above — a floor is a share and a curve is a *signed* bias — so each is
        // checked against the range `Modulation::sanitized` quotes, which is what
        // the panel draws its sliders in.
        let responses = |b: &BrushParams| {
            let mut v: Vec<Modulation> = b.modulation.all().into_iter().flatten().collect();
            match &b.effect {
                BrushEffect::Paint(p) => v.extend(p.modulation.all().into_iter().flatten()),
                BrushEffect::Wet(w) => v.extend(w.modulation.all().into_iter().flatten()),
                BrushEffect::Erase(e) => v.extend(e.modulation.all().into_iter().flatten()),
                BrushEffect::Liquify(l) => v.extend(l.modulation.all().into_iter().flatten()),
            }
            v
        };
        for (name, poke) in pokes {
            for f in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9.0, 1e30] {
                let mut brush = BrushParams::default();
                poke(&mut brush, f);
                let clean = brush.sanitized();
                // Every number is a number…
                for v in numbers(&clean) {
                    assert!(v.is_finite(), "{name} = {f} left a non-finite brush");
                    assert!(v >= 0.0, "{name} = {f} left {v}, which is negative");
                }
                // …the ones this crate quotes in [0, 1] are in it…
                for v in unit(&clean) {
                    assert!((0.0..=1.0).contains(&v), "{name} = {f} escaped [0, 1]");
                }
                // …a color is finite and bounded, in either direction…
                for v in colors(&clean) {
                    let bound = crate::Srgb::EXTENT;
                    assert!((-bound..=bound).contains(&v), "{name} = {f} left {v}");
                }
                // …and the stretch cannot outrun its own saturation point.
                assert!((0.0..=BrushParams::MAX_STRETCH).contains(&clean.stretch));
                // A jitter amplitude is an offset in the color space this crate
                // defines, so it carries `Srgb`'s ceiling rather than a slider's
                // (`ColorDynamics::amplitude`) — an unbounded one reaches an f16
                // tile as an infinity.
                for a in clean.color_dynamics().amplitude {
                    assert!((0.0..=crate::Srgb::EXTENT).contains(&a), "{name} = {f}");
                }
                // …and a pen mapping is stored in the range the sliders show, so a
                // panel cannot be asked to draw a handle at an impossible position.
                for m in responses(&clean) {
                    assert!((0.0..=1.0).contains(&m.floor), "{name} = {f}: {m:?}");
                    assert!((-1.0..=1.0).contains(&m.curve), "{name} = {f}: {m:?}");
                }
                // Idempotent, or a load would be a small edit every time.
                assert_eq!(clean.sanitized(), clean, "{name} = {f}");
            }
        }
        // An ordinary brush comes through **bit for bit**: this runs on replay, so
        // anything it nudged would move a golden.
        let ordinary = BrushParams {
            size: 40.0,
            stretch: 0.5,
            tooth: ToothParams {
                give: 0.25,
                // Past the default band, and legitimately so: a soft stick spans
                // the grain rather than sitting on it, and the ceiling is a
                // slider's.
                softness: 0.3,
            },
            effect: BrushEffect::Wet(WetEffect {
                opacity: 0.85,
                flow: 2.5, // past the frontend's slider, and legitimately so
                dynamics: BrushDynamics {
                    add: 0.7,
                    lift: 1.0,
                    bleed: 0.95,
                    ..BrushDynamics::default()
                },
                ..WetEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(ordinary.sanitized(), ordinary);
        // …an ordinary plain-paint brush too, its flow past the slider included…
        let plain = BrushParams {
            effect: BrushEffect::Paint(PaintEffect {
                opacity: 0.85,
                flow: 2.5,
                ..PaintEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(plain.sanitized(), plain);
        // …and so does an ordinary eraser, its own rate past 1 included.
        let eraser = BrushParams {
            effect: BrushEffect::Erase(EraseEffect {
                opacity: 0.5,
                flow: 2.0,
                ..EraseEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(eraser.sanitized(), eraser);
    }

    /// **Flow is one knob with one meaning on every effect** (§6.2): the overall
    /// rate, read and written through whichever effect is in force — and carried
    /// whole across [`BrushParams::make_wet`], so wetting a brush moves no
    /// slider. What a wet brush *is* — its axes — is untouched by it.
    #[test]
    fn flow_is_the_one_overall_rate_whatever_the_effect() {
        // Paint → Wet: the rate and its pen mapping carry, the axes start at the
        // defaults, so the brush lays the paint it always laid.
        let mut b = BrushParams::default();
        b.effect.set_flow(1.7);
        assert_eq!(b.effect.flow(), 1.7);
        let w = b.make_wet();
        assert_eq!(w.flow, 1.7, "wetting a brush must not move the Flow slider");
        assert_eq!(w.dynamics, BrushDynamics::default());
        assert_eq!(b.effect.flow(), 1.7, "…and the enum reads the same knob");
        // The slider scales the overall rate, never the axes: a blend brush
        // stays a blend brush at any flow.
        let d = &mut b.wet_mut().expect("wet").dynamics;
        d.add = 0.0;
        d.lift = 0.5;
        b.effect.set_flow(0.3);
        let w = b.wet().expect("wet");
        assert_eq!((w.dynamics.add, w.dynamics.lift), (0.0, 0.5));
        assert_eq!(w.flow, 0.3);
        // Erase → Wet: the rate carries there too — flow means the same thing
        // on both sides of that switch now.
        let mut e = BrushParams {
            effect: BrushEffect::Erase(EraseEffect {
                opacity: 0.5,
                flow: 2.0,
                ..EraseEffect::default()
            }),
            ..BrushParams::default()
        };
        let w = e.make_wet();
        assert_eq!((w.flow, w.opacity), (2.0, 0.5));
        // Liquify is the one effect whose rate has a ceiling this crate owns
        // (`LiquifyEffect::strength`, §6.13), so the setter holds it where the other
        // three pass the number through — and lands a NaN where `sanitized` lands
        // one, since two doors onto a field that disagree about NaN is the thing
        // holding a range by hand costs you.
        let mut warp = BrushParams {
            effect: BrushEffect::Liquify(LiquifyEffect::default()),
            ..BrushParams::default()
        };
        for (wrote, held) in [(2.5, 1.0), (-1.0, 0.0), (f32::NAN, 1.0), (0.4, 0.4)] {
            warp.effect.set_flow(wrote);
            assert_eq!(warp.effect.flow(), held, "set_flow({wrote})");
        }
    }

    /// **`radius` is a pure scale on the mark**, which is the whole of why `drain` is
    /// quoted per radius (§6.2): enlarge the tip and the stroke has to run dry
    /// proportionally further along, not at the same canvas distance.
    ///
    /// Stated on the *reach* — the travel at which the load reaches zero,
    /// `1/drain_px` — because that is the length the falloff actually draws, and the
    /// claim is that it grows with the tip instead of standing still. A per-canvas-px
    /// `drain` fails this at every radius but the one it was tuned at.
    #[test]
    fn a_bigger_brush_runs_dry_proportionally_further() {
        let at = |radius: f32| BrushParams {
            size: radius,
            drain: 0.25,
            ..BrushParams::default()
        };
        for radius in [1.0, 8.0, 16.0, 100.0, 1000.0] {
            let reach = 1.0 / at(radius).drain_px();
            assert!(
                (reach - 4.0 * radius).abs() <= 1e-3 * radius,
                "0.25 per radius must reach four radii at radius {radius}, not {reach} px"
            );
        }
        // A tip with no width cannot run out, so the reciprocal that does not exist
        // reads as inexhaustible rather than as an infinity the shader would turn into
        // a NaN falloff — the same class the poison test above rules out for the
        // stored fields, asked of the derived one.
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let b = BrushParams {
                size: radius,
                ..at(16.0)
            };
            assert_eq!(b.drain_px(), 0.0, "radius {radius} must drain nothing");
        }
        // And zero stays zero at every size, so a brush that never heard of drain
        // takes the shader's identity path whatever it is scaled to.
        for radius in [1.0, 16.0, 1000.0] {
            let b = BrushParams {
                drain: 0.0,
                ..at(radius)
            };
            assert_eq!(b.drain_px(), 0.0);
        }
    }

    /// A liquify brush survives the save format's own encode/decode whole — the
    /// schema-by-name reconciliation (§8) carries the new variant and both its
    /// fields, mapping included.
    #[test]
    fn a_liquify_brush_round_trips_through_the_save_format() {
        let b = BrushParams {
            effect: BrushEffect::Liquify(LiquifyEffect {
                strength: 0.65,
                modulation: LiquifyModulations {
                    strength: Some(Modulation::linear(ModSource::Pressure)),
                },
            }),
            ..BrushParams::default()
        };
        let bytes = carbonite::to_vec_static(&b).expect("encode a liquify brush");
        let back = carbonite::from_slice_static::<BrushParams>(&bytes).expect("decode it back");
        assert_eq!(back, b);
    }

    /// [`BrushParams::taper_px`]'s own guard, [`drain_px`](BrushParams::drain_px)'s
    /// sibling: **a negative or non-finite length reads as 0**, and so does any
    /// length scaled by a radius that is not a radius.
    ///
    /// The claim is really about `f32::max`'s NaN policy — it returns the *other*
    /// operand, which is what makes the guard land on 0 where `clamp` would carry
    /// the NaN through. A taper is a distance, and the fields arrive from files,
    /// presets and peers; the sanitizer is not the only door
    /// ([`BrushParams::sanitized`] says so), so this is the guard that has to hold.
    ///
    /// [`tapers`](BrushParams::tapers) rides on it: an unmeasurable length must not
    /// put the segment generator on the tapered path with no ends to taper between.
    #[test]
    fn a_taper_nobody_can_measure_is_no_taper() {
        let at = |size: f32, start: f32, end: f32| BrushParams {
            size,
            start_taper_length: start,
            end_taper_length: end,
            ..BrushParams::default()
        };
        // An ordinary taper is the plain product, in canvas px.
        assert_eq!(at(16.0, 2.0, 0.5).taper_px(), (32.0, 8.0));
        assert!(at(16.0, 2.0, 0.0).tapers(), "one end is a taper");
        assert!(at(16.0, 0.0, 0.5).tapers(), "so is the other");
        assert!(!at(16.0, 0.0, 0.0).tapers(), "and neither is not");
        // A length that is not a length.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -4.0] {
            assert_eq!(at(16.0, bad, bad).taper_px(), (0.0, 0.0), "length {bad}");
            assert!(!at(16.0, bad, bad).tapers(), "length {bad} drew a taper");
            // One bad end does not take the good one with it: the two are
            // separate distances, not one quantity.
            assert_eq!(at(16.0, bad, 0.5).taper_px(), (0.0, 8.0), "length {bad}");
            assert!(at(16.0, bad, 0.5).tapers());
        }
        // A radius that is not one, against a taper that is: nothing to scale by,
        // so nothing to taper over.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -4.0, 0.0] {
            assert_eq!(at(bad, 2.0, 0.5).taper_px(), (0.0, 0.0), "radius {bad}");
            assert!(!at(bad, 2.0, 0.5).tapers(), "radius {bad} drew a taper");
        }
    }
}
