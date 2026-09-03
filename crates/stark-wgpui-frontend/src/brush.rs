//! The one brush this frontend carries: **Hard Round**, as the web frontend's
//! preset table ships it (§6.2).
//!
//! Written out as a [`BrushParams`] rather than reached for. The durable/transient
//! split a preset has over there — what the tool *is*, beside the size, flow and
//! color the hand is working it at — is that frontend's own shape, and the engine
//! takes only the assembled parameters. One brush needs no split; a frontend that
//! grew a second one would grow its own.

use stark_model::document::{
    BrushDynamics, BrushEffect, BrushModulations, BrushParams, BrushShape, ColorDynamics,
    ModSource, Modulation, NoiseKind, WetEffect, WetModulations,
};

/// The color in hand, which is not the brush's: a wet brush carries pigment, an
/// eraser would not, and the engine takes the two in one command
/// (`ViewCommand::SetBrush`). No alpha — minted paint is per-unit opaque and "how
/// much shows" is the effect's opacity (§6.1).
pub const INK: [f32; 3] = [0.0, 0.0, 0.0];

/// The stroke smoothing the preset asks for, 0..=1 (§6.11). Mapped to a string
/// length by [`canvas`](crate::canvas), which is the only side holding the view it
/// has to be converted through.
pub const SMOOTHING: f32 = 0.15;

/// Hard Round: a nearly-hard round tip on the wet-mixing loop, ridden at flow 3.
///
/// The wet axes are quoted low because of that flow — they are per pass at the
/// neutral flow (§6.2), so at flow 3 the exchange lands where a single saturated
/// pass would.
///
/// Both pen mappings are inert under a mouse and deliberately kept anyway: a mouse
/// reports pressure 1, so the flow rides at its full rate and the size at its full
/// width. They are what the brush *is*, and dropping them would make this a
/// different tool the moment a tablet was plugged in.
pub fn hard_round() -> BrushParams {
    BrushParams {
        size: 100.0,
        shape: BrushShape::Round { hardness: 0.98 },
        drain: 0.1,
        effect: BrushEffect::Wet(WetEffect {
            color: INK,
            opacity: 1.0,
            flow: 3.0,
            dynamics: BrushDynamics {
                lift: 0.1,
                deposit: 0.37,
                bleed: 0.08,
                ..BrushDynamics::default()
            },
            color_dynamics: ColorDynamics {
                noise: NoiseKind::Simplex,
                frequency: [0.05, 0.1],
                amplitude: [0.0, 0.025, 0.05],
            },
            modulation: WetModulations {
                flow: Some(Modulation::linear(ModSource::Pressure)),
                ..WetModulations::default()
            },
        }),
        modulation: BrushModulations {
            size: Some(Modulation {
                source: ModSource::Pressure,
                floor: 0.8,
                curve: 0.0,
            }),
            ..BrushModulations::default()
        },
        ..BrushParams::default()
    }
}
