//! The brush as **this frontend** carries it: the **durable** half
//! ([`BrushConfig`] — what the tool *is*) and the **transient** half
//! ([`Transient`] — the hand's own state: the size and flow it is working the
//! tool at, and the color it is painting in). Two types, because they are two
//! different kinds of state with different owners: a preset stores both, a
//! quick slot stores a [`Transient`] beside a preset's *name*, and the live
//! pair sits in two signals (`AppState::brush`, `AppState::transient`) so a
//! tuning drag or an eyedropper sweep at pointer rate wakes nothing that only
//! shows the tool.
//!
//! The engine's [`BrushParams`] is shaped for what a stroke's record needs —
//! the shared tip knobs and *the* effect in force, with each effect carrying
//! exactly the knobs that exist under it (§6.2, §6.12). The frontend
//! needs a different shape, because a brush is *edited* across those lines:
//! every effect stays configured while only one is in force, so switching
//! between Paint, Wet and Erase forgets nothing — the color above all, which
//! survives every switch by not being the tool's at all
//! ([`Transient::color`], §18.1.8) — and the
//! stroke-smoothing **feel** (§6.11) travels with the brush even though the
//! stored path already embodies it and the record must not carry it.
//!
//! **Shared knobs are stored once.** Paint and Wet are the two *laying* kinds,
//! and everything they agree on — the opacity ceiling, the flow
//! mapping, the color dynamics — is one field here, not a copy per effect the
//! switch would have to keep reconciled. What is left of wet is genuinely wet's
//! alone ([`WetDynamics`]: the axes and their mappings), and
//! [`params`](BrushConfig::params) assembles the model's per-effect structs
//! from the shared fields and the transient at the boundary. The one
//! consequence worth naming: [`effect`](BrushConfig::effect) is the user's own
//! choice and **nothing here changes it behind their back** — a flux slider
//! edits the wet half whether or not wet is in force, exactly as the erase
//! half has always been editable in waiting.
//!
//! The engine never sees either type — [`params`](BrushConfig::params) is the
//! one projection down, and `state::update_brush` the one door that sends it.
//!
//! # Durable and transient
//!
//! The line between the halves is what a hand changes its mind about. The
//! **size** and the **flow** are adjusted all day without the tool becoming a
//! different tool — they are the two knobs on the Brush panel, the two the
//! tuning drag moves (§18.1.9), the two a number key remembers. The flow earns
//! its place there by being the same *kind* of knob as the size on every
//! effect: the overall rate of whatever the tool does, never a part of what it
//! is (§6.2 — on a wet brush it scales the smear and the mint together, where
//! the `add` axis that says *whether* the tool lays its own paint is durable).
//! So the transient applies to **whichever effect is in force** — one rate,
//! carried across the switch exactly as the size always was: an eraser picked
//! up mid-session bites at the flow the hand was just painting at, which is
//! what "the hand's intensity" means. The **color** is the third of the
//! hand's knobs, with one rule of its own ([`Transient`]'s doc): it never
//! arrives with a tool. Everything else — shape, tapers,
//! dynamics, the effect and its opacity, the feel — is [`BrushConfig`]:
//! what the tool *is*, which a preset owns and nothing else stores.
//!
//! Neither half holds the other, on purpose: "the same tool at another size"
//! is now plain equality on [`BrushConfig`] (`presets::same_tool`), and a
//! snapshot that accidentally froze the tune into the tool — the bug the old
//! `with_transient` shuffle existed to paper over — is unrepresentable.

use serde::{Deserialize, Serialize};
use stark_model::document::{
    BrushDynamics, BrushEffect, BrushModulations, BrushParams, BrushShape, ColorDynamics,
    EraseEffect, EraseModulations, LiquifyEffect, LiquifyModulations, Modulation,
    OrientationSource, PaintEffect, PaintModulations, ToothParams, WetEffect, WetModulations,
};

/// Which effect a stroke of the brush has — the user's own choice, beside the
/// configurations it chooses between. A plain switch rather than the model's
/// [`BrushEffect`] sum, because here the knobs it would carry live on
/// [`BrushConfig`] where the *other* effects' survive too.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushEffectType {
    /// Lay paint — the swept, antialiased deposit (§6.2).
    #[default]
    Paint,
    /// Lay paint and work what is already there — the wet-mixing loop (§6.2).
    Wet,
    /// Remove visible opacity (§6.12).
    Erase,
    /// Drag the picture itself — the paint under the tip follows the travel as
    /// a warp of the field (§6.13).
    Liquify,
}

/// The **wet-only** half of the natural-media tool (§6.2): the axes
/// `BrushDynamics` carries, beside their own pen mappings — everything a wet
/// brush is that a paint brush is not. The knobs the two kinds share live once
/// on [`BrushConfig`] itself — the flow above all, which for a wet brush scales
/// this whole half rather than being a member of it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WetDynamics {
    /// The brush's own paint's share of the mix ([`BrushDynamics::add`]) —
    /// durable, unlike the flow that scales it: a blender is a tool that adds
    /// nothing, at any flow.
    pub add: f32,
    /// Canvas paint lifted onto the tool per step ([`BrushDynamics::lift`]).
    pub lift: f32,
    /// Tool paint laid back per step ([`BrushDynamics::deposit`]).
    pub deposit: f32,
    /// Paint pre-loaded onto the tool ([`BrushDynamics::charge`]).
    pub charge: f32,
    /// Canvas paint diffusing under the tip ([`BrushDynamics::bleed`]).
    pub bleed: f32,
    /// The add's pen mapping (`WetModulations::add`).
    pub add_modulation: Option<Modulation>,
    /// The lift's pen mapping (`WetModulations::lift`).
    pub lift_modulation: Option<Modulation>,
    /// The deposit's pen mapping (`WetModulations::deposit`).
    pub deposit_modulation: Option<Modulation>,
    /// The bleed's pen mapping (`WetModulations::bleed`).
    pub bleed_modulation: Option<Modulation>,
}

impl Default for WetDynamics {
    /// The fluxes at zero and `add` at its full share ([`BrushDynamics::default`]),
    /// so switching a fresh brush to Wet lays exactly the paint the shared flow
    /// says — the same paint Paint was laying.
    fn default() -> Self {
        Self {
            add: 1.0,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
            add_modulation: None,
            lift_modulation: None,
            deposit_modulation: None,
            bleed_modulation: None,
        }
    }
}

/// The **transient** half of a brush — the hand's own state, adjusted all day
/// without the tool becoming a different tool (see the module doc): the size,
/// the flow, and the color. A value of its own, not a view of the config: the
/// live one rides `AppState::transient`, a preset keeps one beside its
/// [`BrushConfig`], and a quick slot keeps one beside the name of the preset
/// it is bound to (`slots::QuickBrush`, §18.1.8).
///
/// The color is the transient with one rule the other two knobs do not have:
/// **it never arrives with a tool.** A preset or a slot put on keeps the color
/// the hand already held (`presets::wear`, the one door every swap comes
/// through), and a slot decides "did the hold change anything?" with the color
/// set aside ([`same_tune`](Self::same_tune)) — a color picked mid-hold is the
/// Color panel's act, not the number's to keep. So the color a stored tune
/// carries is simply the color the hand held when the snapshot was taken;
/// nothing reads it back.
///
/// Serde because the rack and the preset store keep it; no `#[serde(default)]`,
/// since a stored entry that lacks part of its tune is a damaged entry and not
/// a tune at some size.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transient {
    /// Stamp radius in canvas pixels at full pressure (`BrushParams::size`).
    pub size: f32,
    /// The overall rate of whichever effect is in force (`BrushEffect::flow`,
    /// §6.2): how much a pass lays, how hard a wet pass works the canvas,
    /// how fast an eraser's bite builds, how hard a liquify stroke drags —
    /// clamped to the strength's own 1 there ([`BrushConfig::params`]).
    pub flow: f32,
    /// The **hand's** color (§18.1.8), and the laying side's pigment — one
    /// home: the Color panel and the eyedropper write it whatever effect is in
    /// force, a fill lays it even while the eraser is held (the projection
    /// sends it beside the params — `ViewCommand::SetBrush`), and Paint and
    /// Wet cannot disagree about it because neither carries a copy. Straight
    /// sRGB, components in [0, 1].
    pub color: [f32; 3],
}

impl Default for Transient {
    /// The engine's own default brush's tune, so the untouched pair projects
    /// `BrushParams::default()` exactly (the test below holds it to that).
    /// Black included — the app's own opening color is the state's to choose
    /// (`AppState::new`), not this type's.
    fn default() -> Self {
        let d = BrushParams::default();
        Self {
            size: d.size,
            flow: d.effect.flow(),
            color: d.pigment().unwrap_or([0.0; 3]),
        }
    }
}

impl Transient {
    /// Whether two tunes agree on **what a number keeps**: the size and the
    /// flow, with the color set aside — the color is the Color panel's, so a
    /// slot must neither rebind for a color picked mid-hold (`slots::Held`)
    /// nor go out over one (`presets::same_brush`).
    pub fn same_tune(&self, other: &Transient) -> bool {
        self.size == other.size && self.flow == other.flow
    }
}

/// The **erase-only** half (§6.12): what an eraser *is*, beyond the tip —
/// its ceiling and its pen response. Its rate is not here, being the
/// transient's ([`Transient::flow`]); the model's `EraseEffect` is assembled
/// from the two at the boundary ([`BrushConfig::params`]).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EraseConfig {
    /// How much of the visible opacity a saturated stroke removes
    /// (`EraseEffect::opacity`) — the eraser's own ceiling, apart from the
    /// laying side's ([`BrushConfig::opacity`]) because removing and laying
    /// are different intents a hand sets separately.
    pub opacity: f32,
    /// The erase flow's pen mapping (`EraseModulations::flow`) — the eraser's
    /// own, where the laying side shares [`BrushConfig::flow_modulation`]: how
    /// *this* tool responds to the pen is part of what it is.
    pub flow_modulation: Option<Modulation>,
}

impl Default for EraseConfig {
    /// The plain full eraser (`EraseEffect::default`), less the rate that is
    /// the transient's.
    fn default() -> Self {
        Self {
            opacity: 1.0,
            flow_modulation: None,
        }
    }
}

/// The **liquify-only** half (§6.13): the drag's pen response. The strength
/// itself is the transient flow, clamped to its quoted 1 at the projection —
/// so the one knob the effect has left to own is how the pen drives it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiquifyConfig {
    /// The strength's pen mapping (`LiquifyModulations::strength`).
    pub strength_modulation: Option<Modulation>,
}

/// The **durable** half of a brush, as edited: the shared tip knobs, the
/// laying side's shared knobs, the wet-, erase- and liquify-only halves, the
/// switch that says which effect is in force, and the feel — everything the
/// tool *is*, and nothing about how hard or how large the hand is working it
/// (that is [`Transient`], which [`params`](Self::params) takes). See the
/// module doc.
///
/// Serde, because this is what the preset library stores. `#[serde(default)]`
/// on the container: the store skips a damaged entry outright, so a field a
/// stored brush lacks falling back to the default is strictly more of it
/// surviving.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushConfig {
    /// Brush tip shape (§6.6).
    pub shape: BrushShape,
    /// Elongation along the facing axis (`BrushParams::stretch`).
    pub stretch: f32,
    /// Leading taper, in radii (`BrushParams::start_taper_length`).
    pub start_taper_length: f32,
    /// Trailing taper, in radii (`BrushParams::end_taper_length`).
    pub end_taper_length: f32,
    /// Reservoir depletion per radius travelled (`BrushParams::drain`).
    pub drain: f32,
    /// The brush's two knobs against the substrate's tooth (§6.4).
    pub tooth: ToothParams,
    /// The per-texel deposit dither (`BrushParams::jitter`, §6.2).
    pub jitter: f32,
    /// What orients the shape as it sweeps (§6.6).
    pub orientation: OrientationSource,
    /// The tip's own pen mappings (`BrushParams::modulation`).
    pub modulation: BrushModulations,
    /// Which effect a stroke has **now** — the user's own switch, and nothing
    /// else moves it. The other effects keep their configuration below, which
    /// is the whole reason this type exists.
    pub effect: BrushEffectType,
    /// The laying side's opacity ceiling (`PaintEffect::opacity`,
    /// `WetEffect::opacity`) — shared, one field for the two laying kinds. The
    /// eraser's removal ceiling is its own ([`erase`](Self::erase)). The
    /// pigment is not beside it any more: the color is the hand's
    /// ([`Transient::color`]), not the tool's.
    pub opacity: f32,
    /// The laying flow's pen mapping — shared by Paint and Wet, since the rate
    /// it scales ([`Transient::flow`]) means the same thing on both (§6.2).
    /// The eraser's own response is [`EraseConfig::flow_modulation`], the
    /// liquify drag's [`LiquifyConfig::strength_modulation`]: the rate is the
    /// hand's, but how a tool responds to the pen is the tool's.
    pub flow_modulation: Option<Modulation>,
    /// Color dynamics (§6.2) — the laying side's, shared like the pigment.
    pub color_dynamics: ColorDynamics,
    /// The wet-only fluxes and their mappings — held whether or not wet is in
    /// force, so a flux tuned while painting is waiting when the user switches.
    pub wet: WetDynamics,
    /// The erase-only half — held whether or not it is in force.
    pub erase: EraseConfig,
    /// The liquify-only half (§6.13) — held whether or not it is in force,
    /// like the eraser's.
    pub liquify: LiquifyConfig,
    /// Stroke smoothing, 0..=1 (§6.11) — the knob, not the rope. The rope
    /// is derived at gesture start (`input::rope`), because the knob is
    /// denominated in the hand's own screen px and only a live view converts
    /// it. Frontend-only: the stored path already embodies the smoothing, so a
    /// field on [`BrushParams`] would be one that replay reads and ignores.
    ///
    /// Clamped on the way in from storage because the range **is** the number's
    /// meaning: a hand-edited store must not be able to hand the fitter a rope
    /// it cannot use.
    #[serde(deserialize_with = "stored_smoothing")]
    pub smoothing: f32,
}

/// [`BrushConfig::smoothing`]'s gate — see the field.
fn stored_smoothing<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    Ok(f32::deserialize(d)?.clamp(0.0, 1.0))
}

impl Default for BrushConfig {
    /// The engine's own default brush's durable half, unsmoothed — and
    /// [`params`](Self::params) of this at [`Transient::default`] **is**
    /// `BrushParams::default()`, which a test below keeps true: the brush the
    /// app shows before anything is applied is the one the session already
    /// holds.
    fn default() -> Self {
        let d = BrushParams::default();
        let p = match d.effect {
            BrushEffect::Paint(p) => p,
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => {
                PaintEffect::default()
            }
        };
        Self {
            shape: d.shape,
            stretch: d.stretch,
            start_taper_length: d.start_taper_length,
            end_taper_length: d.end_taper_length,
            drain: d.drain,
            tooth: d.tooth,
            jitter: d.jitter,
            orientation: d.orientation,
            modulation: d.modulation,
            effect: BrushEffectType::Paint,
            opacity: p.opacity,
            flow_modulation: p.modulation.flow,
            color_dynamics: p.color_dynamics,
            wet: WetDynamics::default(),
            erase: EraseConfig::default(),
            liquify: LiquifyConfig::default(),
            smoothing: 0.0,
        }
    }
}

impl BrushConfig {
    /// The engine's view of this brush at `t`: the shared knobs and the
    /// transient assembled into the effect in force — the inactive effects and
    /// the feel deliberately do not survive the projection, and the hand's
    /// color rides down *beside* the result (`ViewCommand::SetBrush`), not
    /// inside an erasing brush.
    ///
    /// The transient is a parameter rather than a pair of fields, because it is
    /// not this type's to hold (the module doc): `t.size` is the tip, and
    /// `t.flow` is the rate of **whichever effect is in force** — the one knob
    /// with one meaning the flow/add split bought (§6.2), clamped to the
    /// liquify strength's own load-bearing 1 where that is the effect.
    ///
    /// Written out field by field with no `..` on purpose, the effect structs
    /// included: a field added to `BrushParams` or to an effect fails to
    /// compile here, which is what keeps this type from silently dropping a
    /// knob the engine grew.
    pub fn params(&self, t: Transient) -> BrushParams {
        BrushParams {
            size: t.size,
            shape: self.shape,
            stretch: self.stretch,
            start_taper_length: self.start_taper_length,
            end_taper_length: self.end_taper_length,
            drain: self.drain,
            tooth: self.tooth,
            jitter: self.jitter,
            orientation: self.orientation,
            modulation: self.modulation,
            effect: match self.effect {
                BrushEffectType::Paint => BrushEffect::Paint(PaintEffect {
                    color: t.color,
                    opacity: self.opacity,
                    flow: t.flow,
                    color_dynamics: self.color_dynamics,
                    modulation: PaintModulations {
                        flow: self.flow_modulation,
                    },
                }),
                BrushEffectType::Wet => BrushEffect::Wet(WetEffect {
                    color: t.color,
                    opacity: self.opacity,
                    flow: t.flow,
                    dynamics: BrushDynamics {
                        add: self.wet.add,
                        lift: self.wet.lift,
                        deposit: self.wet.deposit,
                        charge: self.wet.charge,
                        bleed: self.wet.bleed,
                    },
                    color_dynamics: self.color_dynamics,
                    modulation: WetModulations {
                        flow: self.flow_modulation,
                        add: self.wet.add_modulation,
                        lift: self.wet.lift_modulation,
                        deposit: self.wet.deposit_modulation,
                        bleed: self.wet.bleed_modulation,
                    },
                }),
                BrushEffectType::Erase => BrushEffect::Erase(EraseEffect {
                    opacity: self.erase.opacity,
                    flow: t.flow,
                    modulation: EraseModulations {
                        flow: self.erase.flow_modulation,
                    },
                }),
                BrushEffectType::Liquify => BrushEffect::Liquify(LiquifyEffect {
                    // `min`, not `clamp`: the quoted 1 is a renderer invariant
                    // (`LiquifyEffect::strength`), the slider already stops
                    // there ([`max_flow`](Self::max_flow)), and `f32::min`
                    // hands a NaN's place to the 1 where `clamp` would keep it.
                    strength: t.flow.min(1.0),
                    modulation: LiquifyModulations {
                        strength: self.liquify.strength_modulation,
                    },
                }),
            },
        }
    }

    /// The effect's **opacity** — the ceiling on what a saturated stroke does
    /// (`BrushEffect::opacity`), read off whichever side is in force. A liquify
    /// brush has none (`BrushEffect::opacity`'s own arm says why), so it reads
    /// as the identity and the editor hides the dial.
    pub fn opacity(&self) -> f32 {
        match self.effect {
            BrushEffectType::Paint | BrushEffectType::Wet => self.opacity,
            BrushEffectType::Erase => self.erase.opacity,
            BrushEffectType::Liquify => 1.0,
        }
    }

    /// Write the effect's opacity — [`opacity`](Self::opacity)'s other half.
    /// A no-op while liquify is in force, which has no such knob and no dial
    /// shown to write it.
    pub fn set_opacity(&mut self, opacity: f32) {
        match self.effect {
            BrushEffectType::Paint | BrushEffectType::Wet => self.opacity = opacity,
            BrushEffectType::Erase => self.erase.opacity = opacity,
            BrushEffectType::Liquify => {}
        }
    }

    /// Where the in-force effect's rate slider stops — one range for a knob
    /// reachable three ways (the panel's Flow slider, the editor's row, the
    /// tuning drag), or the drag would quietly go somewhere a slider cannot
    /// show. The liquify strength's top is its quoted — and load-bearing — 1
    /// (`LiquifyEffect::strength`); every other rate's ceiling is the slider's
    /// own (`panels::brush::MAX_FLOW`).
    pub fn max_flow(&self) -> f32 {
        match self.effect {
            BrushEffectType::Liquify => 1.0,
            _ => crate::panels::brush::MAX_FLOW,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The promise [`BrushConfig::default`] makes: before anything is applied,
    /// the brush the chrome shows is the brush the session already holds.
    #[test]
    fn the_default_config_projects_the_default_params() {
        assert_eq!(
            BrushConfig::default().params(Transient::default()),
            BrushParams::default()
        );
    }

    /// Switching away and back forgets nothing — the reason this type exists.
    /// The laying knobs survive a spell of erasing untouched, the erase side
    /// keeps its own edits, and the wet fluxes wait through both.
    #[test]
    fn the_inactive_effect_survives_the_switch() {
        let mut c = BrushConfig {
            opacity: 0.85,
            ..BrushConfig::default()
        };
        c.wet.lift = 0.4;
        let held = c;
        c.effect = BrushEffectType::Erase;
        c.set_opacity(0.25);
        assert_eq!(
            (c.opacity, c.wet),
            (held.opacity, held.wet),
            "erase edits must not reach the laying side",
        );
        c.effect = BrushEffectType::Paint;
        assert_eq!(c.opacity(), 0.85, "the laying opacity waited");
        assert_eq!(c.erase.opacity, 0.25, "…and the erase side keeps its own");
        c.effect = BrushEffectType::Wet;
        assert_eq!(c.wet.lift, 0.4, "…and the wet fluxes waited too");
    }

    /// The shared laying knobs really are shared: what Paint projects and what
    /// Wet projects agree on the pigment — the hand's — the opacity, the flow
    /// and the color dynamics, with nothing to reconcile — the wet projection
    /// differs from the paint one only by the fluxes.
    #[test]
    fn paint_and_wet_project_the_same_shared_knobs() {
        let mut c = BrushConfig {
            opacity: 0.7,
            ..BrushConfig::default()
        };
        let t = Transient {
            size: 40.0,
            flow: 1.3,
            color: [0.9, 0.2, 0.1],
        };
        c.effect = BrushEffectType::Paint;
        let BrushEffect::Paint(p) = c.params(t).effect else {
            panic!("paint in force projects Paint");
        };
        c.effect = BrushEffectType::Wet;
        let BrushEffect::Wet(w) = c.params(t).effect else {
            panic!("wet in force projects Wet");
        };
        assert_eq!(
            (p.color, p.opacity, p.flow, p.color_dynamics),
            (w.color, w.opacity, w.flow, w.color_dynamics),
        );
        assert_eq!(p.color, [0.9, 0.2, 0.1], "…and the pigment is the hand's");
        assert_eq!(
            w.dynamics.add, 1.0,
            "at the full add share, the shared flow lays the same paint wet",
        );
    }

    /// [`Transient::same_tune`] is the slot's question — did the hold change
    /// what a number keeps? — so the color is set aside: a color picked
    /// mid-hold is the Color panel's act (§18.1.8).
    #[test]
    fn the_tune_comparison_sets_the_color_aside() {
        let a = Transient::default();
        let recolored = Transient {
            color: [0.9, 0.1, 0.2],
            ..a
        };
        let resized = Transient { size: 80.0, ..a };
        assert!(a.same_tune(&recolored), "color is not the number's to keep");
        assert!(!a.same_tune(&resized), "the size is");
    }

    /// The transient is the rate of **whichever effect is in force** — one
    /// knob carried across the switch, exactly as the size is (§6.2). The
    /// liquify strength alone clamps to its quoted 1, which is a renderer
    /// invariant and where its slider stops ([`BrushConfig::max_flow`]).
    #[test]
    fn the_transient_drives_every_effects_rate() {
        let mut c = BrushConfig::default();
        let t = Transient {
            size: 40.0,
            flow: 2.0,
            ..Transient::default()
        };
        for effect in [
            BrushEffectType::Paint,
            BrushEffectType::Wet,
            BrushEffectType::Erase,
        ] {
            c.effect = effect;
            let p = c.params(t);
            assert_eq!(p.size, 40.0);
            assert_eq!(p.effect.flow(), 2.0, "{effect:?} takes the hand's rate");
        }
        c.effect = BrushEffectType::Liquify;
        assert_eq!(
            c.params(t).effect.flow(),
            1.0,
            "the strength's 1 is load-bearing (§6.13)",
        );
        assert_eq!(c.params(Transient { flow: 0.4, ..t }).effect.flow(), 0.4);
    }
}
