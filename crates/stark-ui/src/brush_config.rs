//! The brush as **this frontend** carries it: [`BrushConfig`].
//!
//! The engine's [`BrushParams`] is shaped for what a stroke's record needs —
//! the shared tip knobs and *the* effect in force, with each effect carrying
//! exactly the knobs that exist under it (§6.2, §6.12). The frontend
//! needs a different shape, because a brush is *edited* across those lines:
//! both effects stay configured while only one is in force, so toggling Paint ↔
//! Erase forgets nothing — the color above all, which the Color panel writes
//! whatever the hand holds (§18.1.8) — and the stroke-smoothing **feel**
//! (§6.11) travels with the brush even though the stored path already
//! embodies it and the record must not carry it.
//!
//! One type for all of it, so a whole-brush snapshot that lost its feel or its
//! inactive half is unrepresentable: the live brush (`AppState::brush`), the
//! preset library (`crate::presets`) and the quick-brush rack (`crate::slots`)
//! all traffic in this. The engine never sees it — [`params`](BrushConfig::params)
//! is the one projection down, and `state::update_brush` the one door that
//! sends it.

use serde::{Deserialize, Serialize};
use stark_model::document::{
    BrushEffect, BrushModulations, BrushParams, BrushShape, EraseEffect, OrientationSource,
    PaintEffect, ToothParams,
};

/// Which effect a stroke of the brush has — the toggle's own state, beside the
/// two configurations it chooses between. A plain switch rather than the
/// model's [`BrushEffect`] sum, because here the knobs it would carry live on
/// [`BrushConfig`] where the *other* effect's survive too.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushEffectType {
    /// Lay and work paint (§6.2).
    #[default]
    Paint,
    /// Remove visible opacity (§6.12).
    Erase,
}

/// A whole brush, as edited: the shared tip knobs beside **both** effects, the
/// switch that says which is in force, and the feel. See the module doc.
///
/// Serde, because this is what the preset library and the quick-brush rack
/// store — one stored shape for one type, so the two libraries cannot come to
/// disagree about what a stored brush is. `#[serde(default)]` on the container:
/// the stores skip a damaged entry outright, so a field a stored brush lacks
/// falling back to the default is strictly more of it surviving.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushConfig {
    /// Stamp radius in canvas pixels at full pressure (`BrushParams::size`).
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
    /// Which effect a stroke has **now**. The other one keeps its configuration
    /// below, which is the whole reason this type exists.
    pub effect: BrushEffectType,
    /// The painting effect's configuration — held whether or not it is in
    /// force. Its `color` is the **hand's** color (§18.1.8): the Color
    /// panel and the eyedropper write it whatever [`effect`](Self::effect)
    /// says, a preset or a slot leaves it alone on the way in, and a fill lays
    /// it even while the eraser is held (the projection sends it beside the
    /// params — `ViewCommand::SetBrush`).
    pub paint: PaintEffect,
    /// The erasing effect's configuration — held whether or not it is in force.
    pub erase: EraseEffect,
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
        let paint = match d.effect {
            BrushEffect::Paint(p) => p,
            BrushEffect::Erase(_) => PaintEffect::default(),
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
            paint,
            erase: EraseEffect::default(),
            smoothing: 0.0,
        }
    }
}

impl BrushConfig {
    /// The engine's view of this brush: the shared knobs and the effect in
    /// force — the inactive effect and the feel deliberately do not survive the
    /// projection, and the hand's color rides down *beside* the result
    /// (`ViewCommand::SetBrush`), not inside an erasing brush.
    ///
    /// Written out field by field with no `..` on purpose: a field added to
    /// `BrushParams` fails to compile here, which is what keeps this type from
    /// silently dropping a knob the engine grew.
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
                BrushEffectType::Paint => BrushEffect::Paint(self.paint),
                BrushEffectType::Erase => BrushEffect::Erase(self.erase),
            },
        }
    }

    /// The hand's color — [`paint`](Self::paint)'s own, whatever effect is in
    /// force (§18.1.8). Named so a reader of the Color panel's sites sees
    /// the rule rather than a reach into the paint effect.
    pub fn color(&self) -> [f32; 3] {
        self.paint.color
    }

    /// The effect's **source rate** — `BrushEffect::flow`, read off whichever
    /// side is in force.
    pub fn flow(&self) -> f32 {
        match self.effect {
            BrushEffectType::Paint => self.paint.dynamics.flow,
            BrushEffectType::Erase => self.erase.flow,
        }
    }

    /// Write the effect's source rate — [`flow`](Self::flow)'s other half.
    pub fn set_flow(&mut self, flow: f32) {
        match self.effect {
            BrushEffectType::Paint => self.paint.dynamics.flow = flow,
            BrushEffectType::Erase => self.erase.flow = flow,
        }
    }

    /// The effect's **opacity** — the ceiling on what a saturated stroke does
    /// (`BrushEffect::opacity`), read off whichever side is in force.
    pub fn opacity(&self) -> f32 {
        match self.effect {
            BrushEffectType::Paint => self.paint.opacity,
            BrushEffectType::Erase => self.erase.opacity,
        }
    }

    /// Write the effect's opacity — [`opacity`](Self::opacity)'s other half.
    pub fn set_opacity(&mut self, opacity: f32) {
        match self.effect {
            BrushEffectType::Paint => self.paint.opacity = opacity,
            BrushEffectType::Erase => self.erase.opacity = opacity,
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
        assert_eq!(BrushConfig::default().params(), BrushParams::default());
    }

    /// Toggling away and back forgets nothing — the reason this type exists.
    #[test]
    fn the_inactive_effect_survives_the_toggle() {
        let mut c = BrushConfig {
            paint: PaintEffect::colored([0.3, 0.5, 0.7]),
            ..BrushConfig::default()
        };
        let held = c.paint;
        c.effect = BrushEffectType::Erase;
        c.set_opacity(0.25);
        assert_eq!(c.paint, held, "erase edits must not reach the paint side");
        c.effect = BrushEffectType::Paint;
        assert_eq!(c.params().effect, BrushEffect::Paint(held));
        assert_eq!(c.erase.opacity, 0.25, "…and the erase side keeps its own");
    }
}
