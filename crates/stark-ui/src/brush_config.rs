//! The brush as **this frontend** carries it: [`BrushConfig`], and the
//! **transient** half of it a quick slot keeps of its own ([`Transient`]).
//!
//! The engine's [`BrushParams`] is shaped for what a stroke's record needs —
//! the shared tip knobs and *the* effect in force, with each effect carrying
//! exactly the knobs that exist under it (§6.2, §6.12). The frontend
//! needs a different shape, because a brush is *edited* across those lines:
//! every effect stays configured while only one is in force, so switching
//! between Paint, Wet and Erase forgets nothing — the color above all, which
//! the Color panel writes whatever the hand holds (§18.1.8) — and the
//! stroke-smoothing **feel** (§6.11) travels with the brush even though the
//! stored path already embodies it and the record must not carry it.
//!
//! **Shared knobs are stored once.** Paint and Wet are the two *laying* kinds,
//! and everything they agree on — the pigment, the opacity ceiling, the flow
//! and its pen mapping, the color dynamics — is one field here, not a copy per
//! effect the switch would have to keep reconciled. The flow can be shared
//! because it means the same thing on both: the overall rate, which on a wet
//! brush scales the whole loop rather than being one of its axes (§6.2).
//! What is left of wet is genuinely wet's alone ([`WetDynamics`]: the axes and
//! their mappings), and [`params`](BrushConfig::params) assembles the model's
//! per-effect structs from the shared fields at the boundary. The one
//! consequence worth naming: [`effect`](BrushConfig::effect) is the user's own
//! choice and **nothing here changes it behind their back** — a flux slider
//! edits the wet half whether or not wet is in force, exactly as the erase
//! half has always been editable in waiting.
//!
//! One type for all of it, so a whole-brush snapshot that lost its feel or its
//! inactive half is unrepresentable: the live brush (`AppState::brush`) and the
//! preset library (`crate::presets`) both traffic in this. The engine never sees
//! it — [`params`](BrushConfig::params) is the one projection down, and
//! `state::update_brush` the one door that sends it.
//!
//! # Durable and transient
//!
//! The brush has two halves, and the line between them is what a hand changes
//! its mind about. The **size** and the **flow** are adjusted all day without the
//! tool becoming a different tool — they are the two knobs on the Brush panel, the
//! two the tuning drag moves (§18.1.9), the two a number key remembers. That is
//! the **transient** half, and the flow earns its place there by being the same
//! *kind* of knob as the size on every effect: the overall rate of whatever the
//! tool does, never a part of what it is (§6.2 — on a wet brush it scales the
//! smear and the mint together, where the `add` axis that says *whether* the
//! tool lays its own paint is durable). Everything else — shape, tapers,
//! dynamics, the effect and its opacity, the feel — is what the tool *is*: the
//! **durable** half, which a preset owns and nothing else stores. A preset carries both halves, so
//! clicking one puts on the tool at the size and flow it was saved at; a quick
//! slot carries a preset's *name* beside a [`Transient`] of its own, so the tool
//! on a number is looked up live and an edit to the preset reaches every number
//! bound to it (§18.1.8, `crate::slots`).
//!
//! The transient half is a type and the durable half is not, on purpose: the
//! in-force flow still has two homes — the laying side's
//! ([`flow`](BrushConfig::flow), which Paint and Wet share) and the eraser's own
//! (`EraseEffect::flow`) — so [`Transient`] is a *view* of the brush, read off
//! it and written back ([`transient`](BrushConfig::transient),
//! [`set_transient`](BrushConfig::set_transient)), and "the durable half" is
//! the brush with that view set aside (`presets::same_tool`).

use serde::{Deserialize, Serialize};
use stark_model::document::{
    BrushDynamics, BrushEffect, BrushModulations, BrushParams, BrushShape, ColorDynamics,
    EraseEffect, LiquifyEffect, Modulation, OrientationSource, PaintEffect, PaintModulations,
    ToothParams, WetEffect, WetModulations,
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

/// The **transient** half of a brush: the size, and the flow of the effect in
/// force — the two knobs a hand adjusts without changing its mind about the tool
/// (see the module doc). What a quick slot keeps of its own beside the name of
/// the preset it is bound to (`slots::QuickBrush`, §18.1.8).
///
/// A view of a [`BrushConfig`] rather than a piece of it: read off with
/// [`BrushConfig::transient`], written back with [`BrushConfig::set_transient`].
/// Serde because the rack stores it; no `#[serde(default)]`, since a stored slot
/// that lacks half of its tune is a damaged entry and not a slot at some size.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transient {
    /// Stamp radius in canvas pixels at full pressure ([`BrushConfig::size`]).
    pub size: f32,
    /// The in-force effect's source rate ([`BrushConfig::flow`]).
    pub flow: f32,
}

/// A whole brush, as edited: the shared tip knobs, the laying side's shared
/// knobs, the wet-only and erase-only halves, the switch that says which effect
/// is in force, and the feel. See the module doc.
///
/// Serde, because this is what the preset library stores. `#[serde(default)]`
/// on the container: the store skips a damaged entry outright, so a field a
/// stored brush lacks falling back to the default is strictly more of it
/// surviving.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushConfig {
    /// Stamp radius in canvas pixels at full pressure (`BrushParams::size`).
    /// Transient — see the module doc.
    pub size: f32,
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
    /// The **hand's** color (§18.1.8), and the laying side's pigment — one
    /// home: the Color panel and the eyedropper write it whatever
    /// [`effect`](Self::effect) says, a preset or a slot leaves it alone on the
    /// way in, a fill lays it even while the eraser is held (the projection
    /// sends it beside the params — `ViewCommand::SetBrush`), and Paint and Wet
    /// cannot disagree about it because neither carries a copy.
    pub color: [f32; 3],
    /// The laying side's opacity ceiling (`PaintEffect::opacity`,
    /// `WetEffect::opacity`) — shared, like the pigment. The eraser's removal
    /// ceiling is its own ([`erase`](Self::erase)).
    pub opacity: f32,
    /// The laying side's flow — one **overall rate** for Paint and Wet
    /// (`PaintEffect::flow`, `WetEffect::flow`): how hard a pass works, which
    /// for a wet brush scales the whole loop — smear and mint together — and
    /// for a paint brush the laying that is the whole of what it does (§6.2).
    /// Shared because it means the same thing on both, so switching kinds
    /// cannot jump the slider. The transient half's flow while a laying effect
    /// is in force; the eraser's rate is its own.
    pub flow: f32,
    /// The laying flow's pen mapping — shared for the reason the rate is.
    pub flow_modulation: Option<Modulation>,
    /// Color dynamics (§6.2) — the laying side's, shared like the pigment.
    pub color_dynamics: ColorDynamics,
    /// The wet-only fluxes and their mappings — held whether or not wet is in
    /// force, so a flux tuned while painting is waiting when the user switches.
    pub wet: WetDynamics,
    /// The erasing effect's configuration — held whether or not it is in force.
    /// Its `flow` is the transient half's flow while erase is in force.
    pub erase: EraseEffect,
    /// The liquify effect's configuration (§6.13) — held whether or not it is
    /// in force, like the eraser's. Its `strength` is the transient half's flow
    /// while liquify is in force.
    pub liquify: LiquifyEffect,
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
    /// The engine's own default brush, unsmoothed — and
    /// [`params`](Self::params) of this **is** `BrushParams::default()`, which
    /// a test below keeps true: the brush the app shows before anything is
    /// applied is the one the session already holds.
    fn default() -> Self {
        let d = BrushParams::default();
        let p = match d.effect {
            BrushEffect::Paint(p) => p,
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => {
                PaintEffect::default()
            }
        };
        Self {
            size: d.size,
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
            color: p.color,
            opacity: p.opacity,
            flow: p.flow,
            flow_modulation: p.modulation.flow,
            color_dynamics: p.color_dynamics,
            wet: WetDynamics::default(),
            erase: EraseEffect::default(),
            liquify: LiquifyEffect::default(),
            smoothing: 0.0,
        }
    }
}

impl BrushConfig {
    /// The engine's view of this brush: the shared knobs assembled into the
    /// effect in force — the inactive effects and the feel deliberately do not
    /// survive the projection, and the hand's color rides down *beside* the
    /// result (`ViewCommand::SetBrush`), not inside an erasing brush.
    ///
    /// Written out field by field with no `..` on purpose, the effect structs
    /// included: a field added to `BrushParams` or to an effect fails to
    /// compile here, which is what keeps this type from silently dropping a
    /// knob the engine grew.
    pub fn params(&self) -> BrushParams {
        BrushParams {
            size: self.size,
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
                    color: self.color,
                    opacity: self.opacity,
                    flow: self.flow,
                    color_dynamics: self.color_dynamics,
                    modulation: PaintModulations {
                        flow: self.flow_modulation,
                    },
                }),
                BrushEffectType::Wet => BrushEffect::Wet(WetEffect {
                    color: self.color,
                    opacity: self.opacity,
                    flow: self.flow,
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
                BrushEffectType::Erase => BrushEffect::Erase(self.erase),
                BrushEffectType::Liquify => BrushEffect::Liquify(self.liquify),
            },
        }
    }

    /// The hand's color — [`color`](Self::color), whatever effect is in force
    /// (§18.1.8). Kept as a method so the Color panel's sites read the rule
    /// rather than a field that happens to hold it.
    pub fn color(&self) -> [f32; 3] {
        self.color
    }

    /// The effect's **source rate** — `BrushEffect::flow`, read off whichever
    /// side is in force: the laying side's shared rate, or the eraser's own.
    pub fn flow(&self) -> f32 {
        match self.effect {
            BrushEffectType::Paint | BrushEffectType::Wet => self.flow,
            BrushEffectType::Erase => self.erase.flow,
            BrushEffectType::Liquify => self.liquify.strength,
        }
    }

    /// Write the effect's source rate — [`flow`](Self::flow)'s other half.
    pub fn set_flow(&mut self, flow: f32) {
        match self.effect {
            BrushEffectType::Paint | BrushEffectType::Wet => self.flow = flow,
            BrushEffectType::Erase => self.erase.flow = flow,
            BrushEffectType::Liquify => self.liquify.strength = flow,
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

    /// The transient half — the size, and the in-force effect's flow (see the
    /// module doc). What a quick slot keeps of its own.
    pub fn transient(&self) -> Transient {
        Transient {
            size: self.size,
            flow: self.flow(),
        }
    }

    /// Write the transient half back: the size, and the flow of the effect in
    /// force. The eraser's own rate is not touched while a laying effect is in
    /// force, and the other way round — each is its side's, waiting for the
    /// switch.
    pub fn set_transient(&mut self, t: Transient) {
        self.size = t.size;
        self.set_flow(t.flow);
    }

    /// This brush at another size and flow — [`set_transient`](Self::set_transient)
    /// by value, for the sites that build one: a quick slot resolved against its
    /// preset (`slots::resolve`), and the "same tool" test that sets the halves
    /// apart (`presets::same_tool`).
    pub fn with_transient(mut self, t: Transient) -> Self {
        self.set_transient(t);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The promise [`BrushConfig::default`] makes: before anything is applied,
    /// the brush the chrome shows is the brush the session already holds.
    #[test]
    fn the_default_config_projects_the_default_params() {
        assert_eq!(BrushConfig::default().params(), BrushParams::default());
    }

    /// Switching away and back forgets nothing — the reason this type exists.
    /// The laying knobs survive a spell of erasing untouched, the erase side
    /// keeps its own edits, and the wet fluxes wait through both.
    #[test]
    fn the_inactive_effect_survives_the_switch() {
        let mut c = BrushConfig {
            color: [0.3, 0.5, 0.7],
            opacity: 0.85,
            ..BrushConfig::default()
        };
        c.wet.lift = 0.4;
        let held = c;
        c.effect = BrushEffectType::Erase;
        c.set_opacity(0.25);
        c.set_flow(2.0);
        assert_eq!(
            (c.color, c.opacity, c.flow, c.wet),
            (held.color, held.opacity, held.flow, held.wet),
            "erase edits must not reach the laying side",
        );
        c.effect = BrushEffectType::Paint;
        assert_eq!(c.opacity(), 0.85, "the laying opacity waited");
        assert_eq!(c.erase.opacity, 0.25, "…and the erase side keeps its own");
        c.effect = BrushEffectType::Wet;
        assert_eq!(c.wet.lift, 0.4, "…and the wet fluxes waited too");
    }

    /// The shared laying knobs really are shared: what Paint projects and what
    /// Wet projects agree on the pigment, the opacity, the flow and the color
    /// dynamics, with nothing to reconcile — the wet projection differs from
    /// the paint one only by the fluxes.
    #[test]
    fn paint_and_wet_project_the_same_shared_knobs() {
        let mut c = BrushConfig {
            color: [0.9, 0.2, 0.1],
            opacity: 0.7,
            flow: 1.3,
            ..BrushConfig::default()
        };
        c.effect = BrushEffectType::Paint;
        let BrushEffect::Paint(p) = c.params().effect else {
            panic!("paint in force projects Paint");
        };
        c.effect = BrushEffectType::Wet;
        let BrushEffect::Wet(w) = c.params().effect else {
            panic!("wet in force projects Wet");
        };
        assert_eq!(
            (p.color, p.opacity, p.flow, p.color_dynamics),
            (w.color, w.opacity, w.flow, w.color_dynamics),
        );
        assert_eq!(
            w.dynamics.add, 1.0,
            "at the full add share, the shared flow lays the same paint wet",
        );
    }

    /// The transient half is the in-force effect's flow and nothing of the
    /// other's: a slot's tune written onto a paint brush leaves the eraser's
    /// own rate for the switch to find, which is what keeps the halves from
    /// undoing what the switch promises above.
    #[test]
    fn the_transient_half_is_the_in_force_flow() {
        let mut c = BrushConfig::default();
        c.erase.flow = 2.0;
        c.set_transient(Transient {
            size: 40.0,
            flow: 0.5,
        });
        assert_eq!(
            c.transient(),
            Transient {
                size: 40.0,
                flow: 0.5
            }
        );
        assert_eq!(c.flow, 0.5, "paint is in force, so the laying flow moved");
        assert_eq!(c.erase.flow, 2.0, "the eraser keeps its own");
        c.effect = BrushEffectType::Erase;
        assert_eq!(c.transient().flow, 2.0, "…and it is the transient half now");
    }
}
