//! What the brush editor shows (§6.2, §11.2): every parameter it offers, the range
//! each is offered over, and which group it belongs to.
//!
//! The dialog itself is each frontend's — a web modal over a stylesheet, a native
//! panel over measured rectangles — but *which knobs a brush has* is not, and a
//! second copy of that list is two apps that disagree about what a brush is. So the
//! rows are here and the markup is there.
//!
//! Three kinds of thing live here:
//!
//! - **[`ModRow`]** — a parameter the pen can drive, with the mapping slot that
//!   belongs to it. Adding a target to any of the model's modulation tables and not
//!   here fails to compile at [`ModRow::slot`], which is why the addressing is a
//!   `match` returning a slot rather than a field name written twice.
//! - **[`Knob`]** — every other track: a word, a range, and where the value lives.
//!   The ranges are the interesting half — each is either the model's own bound or a
//!   ceiling the *slider* owns, and the note on each says which.
//! - **[`Section`] and [`Row`]** — the groups, and what is in one for a given brush.
//!   A liquify brush has no opacity ceiling and no color dynamics; a paint brush has
//!   no fluxes. Which rows vanish is a fact about the engine (§6.12, §6.13), so it is
//!   answered once.
//!
//! The **test stroke** is here too ([`default_stroke`], [`reference_stroke`]), for
//! the same reason: a preview whose stroke ran a different way in the two apps would
//! be two previews of two brushes.

use stark_engine::ViewTransform;
use stark_engine::command::InputSample;
use stark_model::document::{
    BrushEffect, BrushParams, BrushShape, ModSource, Modulation, NoiseKind, OrientationSource,
    PenState,
};
use stark_model::geom::Vec2;
use stark_model::{ColorSpaceId, SubstrateId};
use strum::{EnumCount as _, VariantArray as _};

use crate::brush_config::{BrushConfig, BrushEffectType, MAX_RADIUS, MIN_RADIUS, Transient};
use crate::icons::Icon;

/// The longest taper the editor offers, in radii (`BrushParams::start_taper_length`).
///
/// A slider's end rather than a bound on the quantity: past twenty radii the taper is
/// longer than any stroke a hand makes at that size, so the knob would only be walking
/// towards a mark that is all point.
pub const MAX_TAPER: f32 = 20.0;

/// The widest contact transition the editor offers, in the rise's own units
/// (`BrushParams::tooth_softness`, §6.4).
///
/// Also a slider's end, and also not arbitrary: the rise a substrate map can carry
/// spans ±`RISE_LIMIT` = 0.25, so a band of 0.5 already covers the whole of it — every
/// texel is somewhere inside the transition, the gate is a flat scale on the deposit,
/// and the grain has stopped reading. Past that the knob only walks towards a half.
pub const MAX_TOOTH_SOFTNESS: f32 = 1.0;

/// The three wet fluxes' ceiling: λ diverges at 1 (§6.2), so the slider stops short.
const MAX_FLUX: f32 = 0.95;

// --- what the pen can drive -----------------------------------------------

/// The parameters the pen can drive (§6.2) — one variant per modulation target the
/// brush carries, and the addressing for the row's own mapping.
///
/// It carries everything about a row that differs: its word, its range, where its
/// base value lives on the brush, and which mapping slot belongs to it. That is what
/// lets a frontend's row take a `ModRow` and nothing else, and it is why the rows
/// cannot drift out of step with the engine's set — adding a target to any of the
/// modulation tables (`BrushModulations`, `PaintModulations`, `EraseModulations`)
/// and not here fails to compile at [`Self::slot`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, strum::VariantArray, strum::EnumCount)]
pub enum ModRow {
    Size,
    Opacity,
    Flow,
    Stretch,
    ToothGive,
    Add,
    Lift,
    Deposit,
    Bleed,
}

/// Every modulatable row, for a caller that wants the set rather than one of them —
/// **derived**, so a tenth row cannot be laid out by a section and missing from here.
///
/// An array rather than [`strum::VariantArray`]'s own slice, because a frontend keeps
/// one control per row and builds the run with `MOD_ROWS.map(…)`, which a slice does
/// not offer. Its order is the declaration order, which is what makes
/// [`ModRow::index`] an infallible seat number rather than a search.
pub const MOD_ROWS: [ModRow; ModRow::COUNT] = {
    let mut rows = [ModRow::Size; ModRow::COUNT];
    let mut i = 0;
    while i < ModRow::COUNT {
        rows[i] = ModRow::VARIANTS[i];
        i += 1;
    }
    rows
};

impl ModRow {
    /// Where this row sits in [`MOD_ROWS`] — the seat a frontend's control for it is
    /// kept in.
    ///
    /// `as usize` rather than a search, and the two agree by construction: the roster
    /// *is* the declaration order ([`MOD_ROWS`]), so there is no answer to be wrong
    /// and no arm for a new row to be missing from. What was a `position().expect()`
    /// in a frontend — a panic on the frame a forgotten row's dialog opened — is now
    /// nothing at all.
    pub fn index(self) -> usize {
        self as usize
    }

    /// The word on the row, which is also the word the section already used for the
    /// parameter. Takes the brush because the Flow row *is* the in-force effect's
    /// rate, and the liquify effect's rate is not a flow of anything: it is how hard
    /// the paint follows (§6.13).
    pub fn label(self, b: &BrushConfig) -> &'static str {
        match self {
            Self::Size => "Size",
            Self::Opacity => "Opacity",
            Self::Flow => match b.effect {
                BrushEffectType::Liquify => "Strength",
                _ => "Flow",
            },
            Self::Stretch => "Stretch",
            Self::ToothGive => "Tooth give",
            Self::Add => "Add",
            Self::Lift => "Lift",
            Self::Deposit => "Deposit",
            Self::Bleed => "Bleed",
        }
    }

    /// The glyph beside the row's word, where the parameter has one it wears
    /// everywhere else — the size's and the flow's, which the Brush panel keeps, and
    /// the opacity's, which the layer and selection panels show against their own.
    pub fn glyph(self) -> Option<Icon> {
        match self {
            Self::Size => Some(crate::icons::SIZE),
            Self::Flow => Some(crate::icons::FLOW),
            Self::Opacity => Some(crate::icons::OPACITY),
            _ => None,
        }
    }

    /// The base slider's range, for the brush being edited.
    ///
    /// Takes both halves because one row's top is not a constant: the Flow row's is
    /// the in-force effect's, and `Stretch`'s asks the engine about the tip the
    /// transient sizes.
    pub fn range(self, b: &BrushConfig, t: Transient) -> (f32, f32) {
        match self {
            Self::Size => (MIN_RADIUS, MAX_RADIUS),
            // A ceiling: the fraction of a full stroke (§6.2, §6.12).
            Self::Opacity => (0.0, 1.0),
            // The in-force effect's own range (`BrushConfig::max_flow`) — the liquify
            // strength stops at its quoted 1, the rates at the slider's own top.
            Self::Flow => (0.0, b.max_flow()),
            // The knob is `1 − 1/s`, so its own top is an infinitely long tip. Two
            // things stop it short, and the smaller wins: the elongation saturates at
            // `MAX_ELONGATION`, past which the slider stops meaning anything (§6.6) —
            // and the *renderer* cannot draw a tip reaching further than one region
            // holds, which for a large brush bites first (`stark_engine::max_stretch`,
            // §6.2).
            //
            // Asking the engine rather than restating its arithmetic is the whole
            // point: a tip past that limit does not draw a coarser stroke, it silently
            // stops lifting and depositing altogether. A slider that offered one would
            // be offering a broken brush, and no note beside it would make that better
            // than not offering it.
            Self::Stretch => (0.0, stark_engine::max_stretch(&b.params(t))),
            // Full range, and it reads right-to-left: 1 is all the give there is, so
            // the substrate gates nothing, and 0 is the driest tip (§6.4). Quoted that
            // way round for the pen's sake — see `BrushParams::tooth_give`.
            Self::ToothGive => (0.0, 1.0),
            // The full share (`BrushDynamics::add`): 1 is a wet brush laying exactly
            // what a paint brush at the same flow would.
            Self::Add => (0.0, 1.0),
            Self::Lift | Self::Deposit | Self::Bleed => (0.0, MAX_FLUX),
        }
    }

    /// Where the row stands now.
    pub fn get(self, b: &BrushConfig, t: Transient) -> f32 {
        match self {
            Self::Size => t.size,
            // The ceiling of whichever effect is in force — the laying side's or the
            // eraser's own (`BrushConfig::opacity`).
            Self::Opacity => b.opacity(),
            // The overall rate of whichever effect is in force (§6.2, §6.12) — the
            // transient's, like the size beside it.
            Self::Flow => t.flow,
            Self::Stretch => b.stretch,
            Self::ToothGive => b.tooth.give,
            Self::Add => b.wet.add,
            Self::Lift => b.wet.lift,
            Self::Deposit => b.wet.deposit,
            Self::Bleed => b.wet.bleed,
        }
    }

    /// The three wet-only rows write the wet half directly: the configuration holds
    /// every effect, so a write racing the pen's eraser end (§18.1.8) lands on the
    /// remembered wet half instead of being dropped. The effect switch is the user's
    /// own and never moves under an edit (`BrushConfig::effect`).
    pub fn set(self, b: &mut BrushConfig, t: &mut Transient, v: f32) {
        match self {
            Self::Size => t.size = v,
            Self::Opacity => b.set_opacity(v),
            Self::Flow => t.flow = v,
            Self::Stretch => b.stretch = v,
            Self::ToothGive => b.tooth.give = v,
            Self::Add => b.wet.add = v,
            Self::Lift => b.wet.lift = v,
            Self::Deposit => b.wet.deposit = v,
            Self::Bleed => b.wet.bleed = v,
        }
    }

    /// Whether moving this row changes what the tool *is*, rather than how hard the
    /// hand is working it (§18.1.8).
    ///
    /// The two that answer `false` are the transient's, and they are exactly the two
    /// the Brush panel keeps: working a brush at another size is the same tool, so a
    /// preset's name stays on it.
    pub fn durable(self) -> bool {
        !matches!(self, Self::Size | Self::Flow)
    }

    /// Where this row's mapping lives on the brush: the tip's own table
    /// (`BrushModulations`), or the effect's — which for Flow is whichever effect is
    /// in force, that being the row's whole point (§6.12).
    pub fn slot(self, b: &mut BrushConfig) -> &mut Option<Modulation> {
        match self {
            Self::Size => &mut b.modulation.size,
            Self::Stretch => &mut b.modulation.stretch,
            Self::ToothGive => &mut b.modulation.tooth_give,
            Self::Flow => match b.effect {
                BrushEffectType::Paint | BrushEffectType::Wet => &mut b.flow_modulation,
                BrushEffectType::Erase => &mut b.erase.flow_modulation,
                BrushEffectType::Liquify => &mut b.liquify.strength_modulation,
            },
            // The laying side's or the eraser's, like the dial itself. A liquify brush
            // shows no Opacity row at all (§6.13), so its arm is never reached; the
            // laying side's slot is what the dial would write if it were.
            Self::Opacity => match b.effect {
                BrushEffectType::Erase => &mut b.erase.opacity_modulation,
                BrushEffectType::Paint | BrushEffectType::Wet | BrushEffectType::Liquify => {
                    &mut b.opacity_modulation
                }
            },
            Self::Add => &mut b.wet.add_modulation,
            Self::Lift => &mut b.wet.lift_modulation,
            Self::Deposit => &mut b.wet.deposit_modulation,
            Self::Bleed => &mut b.wet.bleed_modulation,
        }
    }

    /// The mapping this row carries, if any.
    pub fn of(self, b: &BrushConfig) -> Option<Modulation> {
        match self {
            Self::Size => b.modulation.size,
            Self::Stretch => b.modulation.stretch,
            Self::ToothGive => b.modulation.tooth_give,
            Self::Flow => match b.effect {
                BrushEffectType::Paint | BrushEffectType::Wet => b.flow_modulation,
                BrushEffectType::Erase => b.erase.flow_modulation,
                BrushEffectType::Liquify => b.liquify.strength_modulation,
            },
            Self::Opacity => match b.effect {
                BrushEffectType::Erase => b.erase.opacity_modulation,
                BrushEffectType::Paint | BrushEffectType::Wet | BrushEffectType::Liquify => {
                    b.opacity_modulation
                }
            },
            Self::Add => b.wet.add_modulation,
            Self::Lift => b.wet.lift_modulation,
            Self::Deposit => b.wet.deposit_modulation,
            Self::Bleed => b.wet.bleed_modulation,
        }
    }
}

/// Set (or clear) a row's mapping source, keeping the shape it already had — so
/// switching pressure → tilt is one edit rather than three.
pub fn set_source(b: &mut BrushConfig, row: ModRow, source: Option<ModSource>) {
    let held = row.of(b);
    *row.slot(b) = source.map(|source| Modulation {
        source,
        ..held.unwrap_or(Modulation::linear(source))
    });
}

/// The word a pen source wears on its chip.
pub fn source_label(s: ModSource) -> &'static str {
    match s {
        ModSource::Pressure => "Pressure",
        ModSource::Tilt => "Tilt",
    }
}

/// The word a noise kind wears on its chip.
pub fn noise_label(kind: NoiseKind) -> &'static str {
    match kind {
        NoiseKind::Simplex => "Simplex",
        NoiseKind::White => "White",
        NoiseKind::Voronoi => "Voronoi",
        NoiseKind::Mosaic => "Mosaic",
    }
}

/// The four noise fields, in the order the chips offer them.
pub const NOISE_KINDS: [NoiseKind; 4] = [
    NoiseKind::Simplex,
    NoiseKind::White,
    NoiseKind::Voronoi,
    NoiseKind::Mosaic,
];

/// The two pen axes, in the order the chips offer them.
pub const SOURCES: [ModSource; 2] = [ModSource::Pressure, ModSource::Tilt];

/// How many points [`curve_points`] samples the response at.
pub const CURVE_N: usize = 25;

/// The mapping's response as a polyline in the unit square: input left → right, the
/// factor it multiplies the parameter by bottom → top, both in 0..=1.
///
/// Sampled from [`Modulation::factor`] itself rather than redrawn from the formula,
/// so the picture cannot disagree with the renderer — including about the floor and
/// about the clamps. Both sources are fed the same sweep, which is what makes one plot
/// serve either.
///
/// In the unit square rather than in px because the box it is drawn into is the
/// frontend's: an SVG viewbox and a painted path want different numbers for one curve.
pub fn curve_points(m: Modulation) -> Vec<(f32, f32)> {
    (0..CURVE_N)
        .map(|i| {
            let x = i as f32 / (CURVE_N - 1) as f32;
            (
                x,
                m.factor(PenState {
                    pressure: x,
                    tilt: x,
                }),
            )
        })
        .collect()
}

// --- every other track ----------------------------------------------------

/// Which of the color space's three channels a wander amplitude is about (§6.7).
///
/// A closed set rather than an index, because there is no fourth: `ColorDynamics`
/// carries exactly three amplitudes and every space this app renders in has exactly
/// three channels. An index could name a channel that does not exist, and used to —
/// with three different wrong answers, since the reader clamped, the writer clamped
/// onto a channel the caller had not named, and the label's wildcard read "Blue ↔
/// yellow" for all of them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, strum::VariantArray, strum::EnumCount)]
pub enum Channel {
    First,
    Second,
    Third,
}

impl Channel {
    /// Where the channel sits in `ColorDynamics::amplitude` — the space's own order
    /// (§6.7), which is what [`Knob::label`] names it by.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Which lookup axis a wander frequency is about: across the stroke, then along it.
///
/// [`Channel`]'s argument for the other indexed family — `ColorDynamics::frequency`
/// is a pair, and a third axis is not a thing a stroke has.
#[derive(Copy, Clone, Debug, PartialEq, Eq, strum::VariantArray, strum::EnumCount)]
pub enum Axis {
    Across,
    Along,
}

impl Axis {
    /// Where the axis sits in `ColorDynamics::frequency`.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// A parameter with no pen mapping — its word, its range, and where the value lives.
///
/// Everything the pen *can* drive is a [`ModRow`] instead. The split is the model's
/// rather than the layout's: a knob is here because the engine reads no modulation for
/// it, and the two that look like omissions are argued for where the rows are laid out
/// ([`Section::rows`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Knob {
    /// How abruptly a round tip's coverage falls to nothing (§6.2). The procedural
    /// tip's alone — a stamp's edge is its image's.
    Hardness,
    /// The run over which the tip widens from a point, in **radii** — so a taper keeps
    /// its shape as the brush is resized (§6.2).
    StartTaper,
    EndTaper,
    /// The towed tip (§6.11). The one knob here that never reaches the engine: the
    /// stored path already embodies it, so the amount is the frontend's own
    /// (`BrushConfig::smoothing`).
    Smoothing,
    /// How finely a liquify drag is stepped (§6.13). A cost dial, not a rate, which is
    /// why it carries no pen chip.
    Quality,
    /// How *abruptly* the tip meets the grain (§6.4) — the other half of contact, and a
    /// different question from the give beside it.
    ToothSoftness,
    /// The per-texel deposit dither (§6.2).
    Jitter,
    /// Depletion per radius travelled — the stroke runs dry (§6.2).
    Drain,
    /// How far one color channel wanders, in the channel's own units. Which channel
    /// is the *color space's*, so the word depends on the space (§6.7).
    Amplitude(Channel),
    /// How fast the color wanders along one lookup axis.
    Frequency(Axis),
    /// The finite glob pre-loaded on the tool (the palette knife, §6.2).
    Charge,
}

/// The knobs that are one row and one variant each — everything but the two families
/// that carry a member.
const PLAIN: [Knob; 9] = [
    Knob::Hardness,
    Knob::StartTaper,
    Knob::EndTaper,
    Knob::Smoothing,
    Knob::Quality,
    Knob::ToothSoftness,
    Knob::Jitter,
    Knob::Drain,
    Knob::Charge,
];

/// Every knob the editor can show, so a frontend that keeps one control per knob has
/// a list to build them from.
///
/// The two families are **swept** rather than spelled out — a control can hang off
/// every member of a closed set, which is what [`Channel`] and [`Axis`] being closed
/// bought. A fourth channel would arrive here, in the sections that lay it out and in
/// the frontends' runs of controls together, instead of compiling clean and panicking
/// the moment the dialog opened.
pub const KNOBS: [Knob; PLAIN.len() + Channel::COUNT + Axis::COUNT] = {
    let mut knobs = [Knob::Hardness; PLAIN.len() + Channel::COUNT + Axis::COUNT];
    let mut i = 0;
    while i < PLAIN.len() {
        knobs[i] = PLAIN[i];
        i += 1;
    }
    let mut c = 0;
    while c < Channel::COUNT {
        knobs[i] = Knob::Amplitude(Channel::VARIANTS[c]);
        i += 1;
        c += 1;
    }
    let mut a = 0;
    while a < Axis::COUNT {
        knobs[i] = Knob::Frequency(Axis::VARIANTS[a]);
        i += 1;
        a += 1;
    }
    knobs
};

impl Knob {
    /// The word on the row. Takes the space because the three color-dynamics channels
    /// are the *space's* channels, and Mixbox's are pigments rather than a lightness
    /// and two opponents (§6.7).
    pub fn label(self, space: ColorSpaceId) -> &'static str {
        match self {
            Self::Hardness => "Hardness",
            Self::StartTaper => "Start taper (radii)",
            Self::EndTaper => "End taper (radii)",
            Self::Smoothing => "Smoothing",
            Self::Quality => "Quality",
            Self::ToothSoftness => "Tooth softness",
            Self::Jitter => "Jitter",
            Self::Drain => "Drain",
            Self::Amplitude(channel) => channel_label(space, channel),
            Self::Frequency(Axis::Across) => "Scale \u{2192} across stroke",
            Self::Frequency(Axis::Along) => "Scale \u{2192} along stroke",
            Self::Charge => "Charge",
        }
    }

    /// The glyph beside the word, where the parameter has one it wears elsewhere.
    pub fn glyph(self) -> Option<Icon> {
        match self {
            Self::Hardness => Some(crate::icons::HARDNESS),
            _ => None,
        }
    }

    /// The track's range. Each is either the model's own bound or a ceiling the
    /// *slider* owns; which of the two it is, is the interesting half of every row.
    pub fn range(self) -> (f32, f32) {
        match self {
            // Fractions, both ends load-bearing.
            Self::Hardness | Self::Quality | Self::Smoothing => (0.0, 1.0),
            Self::StartTaper | Self::EndTaper => (0.0, MAX_TAPER),
            Self::ToothSoftness => (0.0, MAX_TOOTH_SOFTNESS),
            // The field runs to 1 (`BrushParams::jitter`); past strong grain the gate
            // is only noise, so the *slider* stops at a fifth. A ceiling the model does
            // not own belongs to the slider's end rather than to the quantity.
            Self::Jitter => (0.0, 0.2),
            // Dry half a radius past the press is already a stub. In radii, so the top
            // means the same thing at every brush size (§6.2) — quoted per canvas px it
            // did not, and the same setting was a gentle fade on a small tip and a stub
            // on a large one.
            Self::Drain => (0.0, 0.5),
            // An offset in the color space's own units, so half its extent is already a
            // color that has wandered off the one that was picked.
            Self::Amplitude(_) => (0.0, 0.5),
            Self::Frequency(_) => (0.0, 8.0),
            Self::Charge => (0.0, 2.0),
        }
    }

    /// Where the knob stands now.
    ///
    /// A stamp has no hardness of its own, so that arm answers the fallback the
    /// renderer would use if the asset failed to resolve (§6.6) — the row is not laid
    /// out for a stamp at all ([`Section::rows`]), so nothing shows it.
    pub fn get(self, b: &BrushConfig) -> f32 {
        match self {
            Self::Hardness => match b.shape {
                BrushShape::Round { hardness } => hardness,
                BrushShape::Stamp(_) => BrushShape::DEFAULT_HARDNESS,
            },
            Self::StartTaper => b.start_taper_length,
            Self::EndTaper => b.end_taper_length,
            Self::Smoothing => b.smoothing,
            Self::Quality => b.liquify.quality,
            Self::ToothSoftness => b.tooth.softness,
            Self::Jitter => b.jitter,
            Self::Drain => b.drain,
            Self::Amplitude(c) => b.color_dynamics.amplitude[c.index()],
            Self::Frequency(a) => b.color_dynamics.frequency[a.index()],
            Self::Charge => b.wet.charge,
        }
    }

    /// Move it. Hardness writes the *procedural* tip, which is what makes the row an
    /// edit of the shape rather than of a number beside it.
    pub fn set(self, b: &mut BrushConfig, v: f32) {
        match self {
            Self::Hardness => b.shape = BrushShape::Round { hardness: v },
            Self::StartTaper => b.start_taper_length = v,
            Self::EndTaper => b.end_taper_length = v,
            Self::Smoothing => b.smoothing = v,
            Self::Quality => b.liquify.quality = v,
            Self::ToothSoftness => b.tooth.softness = v,
            Self::Jitter => b.jitter = v,
            Self::Drain => b.drain = v,
            Self::Amplitude(c) => b.color_dynamics.amplitude[c.index()] = v,
            Self::Frequency(a) => b.color_dynamics.frequency[a.index()] = v,
            Self::Charge => b.wet.charge = v,
        }
    }
}

/// What the three color-dynamics channels are called in `space` — the noise offsets
/// the *space's* own channels (§6.2, §6.7), so the words follow the document.
///
/// Exhaustive in the channel, which is what [`Channel`] being a closed set is for: the
/// wildcard that used to close this match spelled the third channel's word for
/// anything a caller passed, so an out-of-range index read as "Blue ↔ yellow" rather
/// than failing.
fn channel_label(space: ColorSpaceId, channel: Channel) -> &'static str {
    match (space, channel) {
        (ColorSpaceId::Mixbox, Channel::First) => "Pigment 1",
        (ColorSpaceId::Mixbox, Channel::Second) => "Pigment 2",
        (ColorSpaceId::Mixbox, Channel::Third) => "Pigment 3",
        (_, Channel::First) => "Lightness",
        (_, Channel::Second) => "Green \u{2194} red",
        (_, Channel::Third) => "Blue \u{2194} yellow",
    }
}

// --- the groups, and what is in one ---------------------------------------

/// A sentence the dialog says when the brush is in a state that needs one.
///
/// Each is a knob that has stopped meaning what its track says, and each is shown only
/// when it actually has: a note that is always there is a note nobody reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Note {
    /// Stretch is on, and it is drawing the tip out along the *travel* rather than
    /// across it — a coherent thing to ask for, and not the one people reach for this
    /// slider wanting. So say which axis is in force rather than second-guess the
    /// setting.
    StretchAlongStroke,
    /// The stretch slider stopped short of where it stops on a smaller brush.
    StretchCapped,
    /// The tooth is turned down on a canvas that has no tooth to catch on.
    SmoothCanvas,
}

impl Note {
    pub fn text(self) -> &'static str {
        match self {
            Self::StretchAlongStroke => {
                "Stretching along the stroke, so the mark gets heavier rather than wider. \
                 Switch to Pen angle for a tip that broadens as the pen leans."
            }
            Self::StretchCapped => {
                "This tip is too big to draw out any further \u{2014} a stroke that lifts and \
                 deposits works over a copy of the canvas beneath it, and that has a size \
                 limit. Lower Size to stretch it more."
            }
            Self::SmoothCanvas => {
                "This canvas is smooth, so there is no tooth to catch on. Pick a substrate in \
                 the Lighting panel."
            }
        }
    }
}

/// One thing a section lays out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Row {
    /// The stamp gallery: the procedural tip, the shipped shapes and the library. It
    /// sits with the brush rather than with the presets, because what a shape *is* is
    /// the tool and a preset is a way of arriving at one.
    Shapes,
    /// The two chips that aim the footprint (§6.6).
    Orientation,
    /// The four chips that say what a stroke of the brush *does* (§6.2, §6.12).
    Effects,
    /// The four chips that pick the noise field.
    Noise,
    /// A plain track.
    Knob(Knob),
    /// A track with a pen-mapping chip hung off the end.
    Mod(ModRow),
    /// A sentence, when the brush is in the state that needs it.
    Note(Note),
}

/// The four groups, by what they affect.
///
/// `Hash` because a frontend keeps a *set* of them — which are folded, which have
/// their "Show more" open — and a set keyed by anything else would be a second name
/// for a group.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    /// The footprint the stroke sweeps along the path.
    Tip,
    /// What the stroke *does* with it — named for the effect in force, which is what
    /// makes the switch at the top of it read as the group's own subject.
    Effect,
    /// Where the color wanders as it is laid.
    Color,
    /// Canvas paint on the move.
    Wet,
}

/// The groups in the order the dialog stacks them.
pub const SECTIONS: [Section; 4] = [Section::Tip, Section::Effect, Section::Color, Section::Wet];

/// The runtime facts a section's rows depend on that the brush does not carry.
///
/// Two, and both are the *document's* rather than the tool's: which color space the
/// channels are in (§6.7), and whether the canvas has a tooth to catch on (§6.4). A
/// brush is edited against a document, and these are the two places that shows.
#[derive(Copy, Clone, Debug)]
pub struct Shown {
    pub brush: BrushConfig,
    pub tune: Transient,
    pub space: ColorSpaceId,
    pub substrate: SubstrateId,
}

impl Section {
    /// The name the group wears. The effect group is named for the effect in force:
    /// the sections below it come and go with that switch, so a title saying which is
    /// what makes the coming and going read as an answer rather than as a glitch.
    pub fn title(self, b: &BrushConfig) -> &'static str {
        match self {
            Self::Tip => "Tip",
            Self::Effect => match b.effect {
                BrushEffectType::Paint => "Paint",
                BrushEffectType::Wet => "Wet",
                BrushEffectType::Erase => "Erase",
                BrushEffectType::Liquify => "Liquify",
            },
            Self::Color => "Color dynamics",
            Self::Wet => "Wet",
        }
    }

    /// The sentence under the name — what the group is about, said once so a person
    /// reading the dialog for the first time is not deducing it from the knobs.
    pub fn desc(self, b: &BrushConfig) -> &'static str {
        match self {
            Self::Tip => "The footprint the stroke sweeps along the path.",
            Self::Effect => match b.effect {
                BrushEffectType::Paint => {
                    "The brush's own paint: how much goes down and how far it lasts."
                }
                BrushEffectType::Wet => {
                    "The paint mixes with what is on the canvas: lift, deposit, bleed."
                }
                BrushEffectType::Erase => {
                    "The stroke removes what the eye sees, instead of laying paint."
                }
                BrushEffectType::Liquify => {
                    "The stroke drags the picture with it \u{2014} paint warps instead of mixing."
                }
            },
            Self::Color => {
                "The color wanders across the brush and along the stroke, following a noise field."
            }
            Self::Wet => "Canvas paint on the move \u{2014} smudge, knife, blur.",
        }
    }

    /// The mark beside the name. Each says what the group is *about* — the same job the
    /// sentence does, except that the sentence is inside the fold and the mark is not:
    /// a shut section is a word on a line, and four words in a column are read one at a
    /// time where four marks are read at once.
    pub fn glyph(self) -> Icon {
        match self {
            Self::Tip => crate::icons::TIP,
            Self::Effect => crate::icons::PAINT,
            Self::Color => crate::icons::COLOR,
            Self::Wet => crate::icons::WET,
        }
    }

    /// The name this group wears in the markup, and the one a guided-tour selector
    /// finds it by (§24.3). Stable across a rename on screen.
    pub fn key(self) -> &'static str {
        match self {
            Self::Tip => "tip",
            Self::Effect => "paint",
            Self::Color => "color",
            Self::Wet => "wet",
        }
    }

    /// Whether the group is laid out at all for this brush.
    ///
    /// Pigment wander is a property of *laying* pigment, so the whole color group is
    /// the laying side's (§6.12, §6.13) — an eraser or a liquify brush shows no rows
    /// that reach nothing. The fluxes are the wet effect's own, so that group goes with
    /// the chip that names it.
    pub fn mounted(self, b: &BrushConfig) -> bool {
        match self {
            Self::Tip | Self::Effect => true,
            Self::Color => lays(b),
            Self::Wet => b.effect == BrushEffectType::Wet,
        }
    }

    /// What the group lays out, in order, for the brush as it stands.
    pub fn rows(self, shown: &Shown) -> Vec<Row> {
        let b = &shown.brush;
        let mut rows = Vec::new();
        match self {
            Self::Tip => {
                rows.push(Row::Shapes);
                // Orientation is what aims the footprint (§6.6), and there are two ways
                // for that to matter: a non-round tip has a silhouette to turn, and
                // **any** tip that stretches has an axis to draw out along. A round tip
                // that does neither is the one case where the chips would decide
                // nothing, so it is the one case that does not show them.
                if !is_round(b) || b.stretch > 0.0 {
                    rows.push(Row::Orientation);
                }
                rows.push(Row::Mod(ModRow::Size));
                // How far the footprint is drawn out along the axis above (§6.6).
                // Pointed at Tilt with "Pen angle" this is the pencil: lean the pen and
                // the contact patch elongates along the lean, the way a real conical
                // tip's does. Held at a value with no mapping it is a chisel nib.
                rows.push(Row::Mod(ModRow::Stretch));
                if b.stretch > 0.0 && b.orientation == OrientationSource::FollowStroke {
                    rows.push(Row::Note(Note::StretchAlongStroke));
                }
                // Said only when the slider actually stopped short: below ~110 px the
                // whole range is there and there is nothing to explain.
                if stark_engine::max_stretch(&b.params(shown.tune)) < BrushParams::MAX_STRETCH {
                    rows.push(Row::Note(Note::StretchCapped));
                }
                if is_round(b) {
                    rows.push(Row::Knob(Knob::Hardness));
                }
                rows.push(Row::Knob(Knob::StartTaper));
                rows.push(Row::Knob(Knob::EndTaper));
                rows.push(Row::Knob(Knob::Smoothing));
            }
            Self::Effect => {
                // Chips rather than a dial, because it is the tool's identity and not an
                // amount — the sections below come and go with it, which a slider
                // position would not say.
                rows.push(Row::Effects);
                // The effect's ceiling (§6.2, §6.12), whichever it is: the fraction of a
                // full stroke this stroke lays — or, erasing, removes. A liquify brush
                // has no such ceiling — scrubbing keeps carrying (§6.13) — so the row is
                // not shown rather than shown and vetoed.
                if b.effect != BrushEffectType::Liquify {
                    rows.push(Row::Mod(ModRow::Opacity));
                }
                // The effect's overall rate (§6.2). Not a wet axis: what the tool *does*
                // per unit of this is the Wet group's business.
                rows.push(Row::Mod(ModRow::Flow));
                if b.effect == BrushEffectType::Liquify {
                    rows.push(Row::Knob(Knob::Quality));
                }
                // How far the tip settles into the canvas's own tooth (§6.4): at 1 it
                // follows every fall and the mark is solid; turned *down* the paint
                // catches on the substrate's peaks and skips its valleys, which is what a
                // dry brush leaves.
                rows.push(Row::Mod(ModRow::ToothGive));
                rows.push(Row::Knob(Knob::ToothSoftness));
                // The substrate is the *document's*, not the brush's — a pencil and a
                // loaded brush on one canvas see one tooth — so on a smooth canvas this
                // knob has nothing to bite and says so, rather than moving and changing
                // nothing.
                if b.tooth.give < 1.0 && shown.substrate == SubstrateId::Flat {
                    rows.push(Row::Note(Note::SmoothCanvas));
                }
                rows.push(Row::Knob(Knob::Jitter));
                // Not behind a fold, because it is the only knob that decides whether a
                // tool runs out.
                rows.push(Row::Knob(Knob::Drain));
            }
            Self::Color => {
                rows.push(Row::Noise);
                for channel in Channel::VARIANTS {
                    rows.push(Row::Knob(Knob::Amplitude(*channel)));
                }
                // The two lookup axes live only while some channel is active: at zero
                // amplitude they scale nothing.
                if b.color_dynamics.is_active() {
                    for axis in Axis::VARIANTS {
                        rows.push(Row::Knob(Knob::Frequency(*axis)));
                    }
                }
            }
            Self::Wet => {
                // The source axis (§6.2): how much of the brush's own paint is in the
                // mix, as a share the shared Flow scales. At 0 the tool only works what
                // is there — the blender.
                rows.push(Row::Mod(ModRow::Add));
                // The three fluxes a palette knife is built out of, and the three most
                // worth mapping onto the pen: a knife that lifts with pressure and lays
                // back with tilt is two of these chips (§6.2).
                rows.push(Row::Mod(ModRow::Lift));
                rows.push(Row::Mod(ModRow::Deposit));
                rows.push(Row::Mod(ModRow::Bleed));
            }
        }
        rows
    }

    /// The rarely-touched rows, behind the group's own "Show more".
    pub fn more(self, _shown: &Shown) -> Vec<Row> {
        match self {
            Self::Wet => vec![Row::Knob(Knob::Charge)],
            Self::Tip | Self::Effect | Self::Color => Vec::new(),
        }
    }

    /// Whether the group starts open. The everyday two do; the specialised two are a
    /// word on a line until they are asked for.
    pub fn open_by_default(self) -> bool {
        matches!(self, Self::Tip | Self::Effect)
    }
}

/// Whether the effect in force lays pigment at all — what gates the color group and
/// the opacity ceiling on the amount laid (§6.12, §6.13).
fn lays(b: &BrushConfig) -> bool {
    matches!(b.effect, BrushEffectType::Paint | BrushEffectType::Wet)
}

fn is_round(b: &BrushConfig) -> bool {
    matches!(b.shape, BrushShape::Round { .. })
}

// --- the test stroke ------------------------------------------------------

/// The test stroke's fixed RGB (straight sRGB): a warm gold, so it reads clearly over
/// the red reference stroke beneath it — the preview is about the brush's *behaviour*,
/// not its color. Only the color is forced; the effect's own opacity still applies.
pub const PREVIEW_STROKE_COLOR: [f32; 3] = [0.852, 0.645, 0.125];

/// The reference stroke's fixed RGB — a plain, opaque red.
pub const REFERENCE_COLOR: [f32; 3] = [0.82, 0.15, 0.12];

/// Fixed jitter seed for the previewed test stroke. Every edit re-strokes, and a
/// stroke's seed is normally the document clock — which advances with each replay's
/// commit, re-rolling the color dynamics and dither each time and hiding the parameter
/// change behind fresh noise. Pinning it means only the edited setting moves between
/// renders. Arbitrary value; it just never changes.
pub const PREVIEW_STROKE_SEED: u64 = 0x5747_1CED_57A2_4B11;

/// The seeded test stroke: an S-curve **down** the preview surface with a pressure
/// bell (light → full → light) and a forward tilt that ramps in — so pressure- and
/// tilt-modulated settings visibly shape the stroke even for mouse users.
///
/// Downward because the preview is a tall column, and because it is the direction a
/// hand draws a test stroke in: the run is along the long axis, and the S's swing is
/// across the short one.
///
/// `w` and `h` are the preview surface's size, in the px `view` is denominated in.
pub fn default_stroke(w: f32, h: f32, view: ViewTransform) -> Vec<InputSample> {
    const N: usize = 64;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let x = w * 0.5 + (t * std::f32::consts::TAU).sin() * w * 0.26;
            let y = h * 0.06 + t * h * 0.88;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: (t * std::f32::consts::PI).sin().clamp(0.08, 1.0),
                // Lean along the (mostly +y) travel direction, growing over the stroke,
                // so tilt→deposit reads as a knife laying down more and more.
                tilt: Vec2::new(0.0, 0.65 * t),
                time: (t * 0.7) as f64,
            }
        })
        .collect()
}

/// The fixed reference stroke laid on the preview canvas before any test stroke: a
/// hard-edged, opaque red band across the middle, committed once at init so the user
/// can see how the brush being edited interacts with paint already on the canvas
/// (smudge, drag, bleed, …).
///
/// Across, because the test stroke runs down: the two have to *cross*, or the brush
/// never meets the paint it is meant to be shown moving. It runs off both edges so the
/// crossing is never near an end of it.
pub fn reference_stroke(w: f32, h: f32, view: ViewTransform) -> Vec<InputSample> {
    const N: usize = 8;
    let y = h * 0.5;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let x = w * -0.25 + t * w * 1.5;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: 1.0,
                ..Default::default()
            }
        })
        .collect()
}

/// The brush the reference stroke is laid with: plain `add` paint, no dynamics, no
/// drain — a clean, unchanging target for the brush being edited to work.
pub fn reference_brush() -> BrushParams {
    BrushParams {
        size: 75.0,
        shape: BrushShape::Round { hardness: 0.9 },
        drain: 0.0,
        effect: BrushEffect::painted(REFERENCE_COLOR),
        ..BrushParams::default()
    }
}

/// The test stroke a preview replays, and what a hand on the preview canvas does to
/// it (§6.2, §11.2).
///
/// Six values and five transitions, and both frontends carried the same six until
/// this type did: which samples are replayed, whether they are the artist's own
/// rather than the seeded default, the samples of a stroke in flight, whether a hand
/// is down, and whether a stroke is *committed* on the preview document and so has to
/// be undone before the next replay. What a frontend keeps is one of these — a
/// `Signal` on the web, a field natively — and the engine calls around it, which are
/// the only half a toolkit shows in.
///
/// **A tap is not a stroke**, which is the rule the type exists to state once. The
/// engine says so too (`Engine::replay_stroke_seeded` answers `None` for a hand that
/// never left its first point), and getting it wrong is not cosmetic: the press has
/// already undone the committed stroke, so a tap marked committed makes the *next*
/// edit's undo reach past it into the reference band beneath, and two taps empty the
/// preview canvas with nothing on screen saying why.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TestStroke {
    /// What is replayed, in the preview document's canvas space.
    samples: Vec<InputSample>,
    /// Whether [`samples`](Self::samples) is the artist's own rather than the seeded
    /// default — the one thing a resize must not lay over.
    drawn: bool,
    /// The samples of a stroke in flight on the preview canvas.
    rec: Vec<InputSample>,
    /// Whether a hand is on the preview canvas right now.
    drawing: bool,
    /// Whether a committed test stroke is on the preview document.
    committed: bool,
}

impl TestStroke {
    /// Lay the seeded stroke for a `w`×`h` preview surface under `view`
    /// ([`default_stroke`]) — what opening the dialog and the Reset button both do.
    pub fn seed(&mut self, w: f32, h: f32, view: ViewTransform) {
        self.samples = default_stroke(w, h, view);
        self.drawn = false;
    }

    /// Re-lay the seeded stroke onto a surface that has changed size, so the default
    /// keeps running the length of the column instead of ending short of it.
    ///
    /// A stroke the artist drew is left exactly where they drew it: it is theirs, and
    /// it is in canvas space, so it survives the resize untouched.
    pub fn relay_after_resize(&mut self, w: f32, h: f32, view: ViewTransform) {
        if !self.drawn {
            self.seed(w, h, view);
        }
    }

    /// Begin a stroke the artist is drawing, at `first`. **Answers whether the caller
    /// owes the preview document an undo** — the committed test stroke has to come off
    /// before a new one goes on, and only the frontend holds the engine to say so.
    #[must_use]
    pub fn begin(&mut self, first: InputSample) -> bool {
        self.rec = vec![first];
        self.drawing = true;
        // Taken, not read: the stroke is off the document the moment the caller acts
        // on this answer, so leaving the flag up would have the next replay undo one
        // stroke too many.
        std::mem::take(&mut self.committed)
    }

    /// Extend the stroke in flight. Ignored when there is none, so a stray move
    /// between a cancel and the next press records nothing.
    pub fn extend(&mut self, sample: InputSample) {
        if self.drawing {
            self.rec.push(sample);
        }
    }

    /// End it, and answer whether it **became** the test stroke.
    ///
    /// `false` for a hand that never left its first point (see the type's doc) and for
    /// a release with no stroke under it; the caller replays what was already there
    /// either way, and must not mark anything committed off the back of it.
    ///
    /// Two samples is the test because it is the one the frontend can make: what the
    /// engine actually refuses is a *fit* that painted nothing
    /// (`Session::end_stroke`), and `GestureCommand::End` answers nothing back. So a
    /// hand that moved less than the fitter's tolerance is the one case left where
    /// this says yes and the document holds no stroke — narrower than the tap it
    /// replaces by every gesture that is a single point, and not closable from here.
    #[must_use]
    pub fn end(&mut self) -> bool {
        if !self.drawing {
            return false;
        }
        self.drawing = false;
        let rec = std::mem::take(&mut self.rec);
        if rec.len() < 2 {
            return false;
        }
        self.samples = rec;
        self.drawn = true;
        self.committed = true;
        true
    }

    /// Abandon the stroke in flight — a cancelled pointer. The caller restores the
    /// last one by replaying, which is what leaves nothing to say here about
    /// [`needs_undo`](Self::needs_undo).
    pub fn cancel(&mut self) {
        self.drawing = false;
        self.rec.clear();
    }

    /// Say what a replay did: `true` where it committed a stroke the next one has to
    /// undo, `false` where the samples held none.
    pub fn replayed(&mut self, committed: bool) {
        self.committed = committed;
    }

    /// What to replay.
    pub fn samples(&self) -> &[InputSample] {
        &self.samples
    }

    /// Whether a committed test stroke stands on the preview document, and so has to
    /// be undone before the next replay.
    pub fn needs_undo(&self) -> bool {
        self.committed
    }

    /// Whether a hand is on the preview canvas — what stands a replay down, since
    /// what is on screen then is the stroke being drawn and replacing it mid-gesture
    /// would take it out from under the pointer.
    pub fn drawing(&self) -> bool {
        self.drawing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(brush: BrushConfig) -> Shown {
        Shown {
            brush,
            tune: Transient::default(),
            space: ColorSpaceId::Oklab,
            substrate: SubstrateId::Flat,
        }
    }

    /// Every row a section lays out is one the brush can actually answer: a `Mod` row
    /// reads and writes its slot, a `Knob` row its field. Stated as a round trip rather
    /// than by eye, because a track that moves and changes nothing is exactly what the
    /// editor must not offer.
    #[test]
    fn every_row_reads_back_what_it_wrote() {
        for effect in [
            BrushEffectType::Paint,
            BrushEffectType::Wet,
            BrushEffectType::Erase,
            BrushEffectType::Liquify,
        ] {
            // A stretched brush, so the Tip group lays out its orientation chips and
            // both of its notes are reachable.
            let config = BrushConfig {
                effect,
                stretch: 0.3,
                ..BrushConfig::default()
            };
            let s = shown(config);
            for section in SECTIONS.into_iter().filter(|s| s.mounted(&config)) {
                for row in section.rows(&s).into_iter().chain(section.more(&s)) {
                    match row {
                        Row::Knob(knob) => {
                            let (lo, hi) = knob.range();
                            let want = lo + 0.25 * (hi - lo);
                            let mut b = config;
                            knob.set(&mut b, want);
                            assert!(
                                (knob.get(&b) - want).abs() < 1e-6,
                                "{knob:?} did not read back what it wrote"
                            );
                        }
                        Row::Mod(m) => {
                            let (lo, hi) = m.range(&config, s.tune);
                            let want = lo + 0.25 * (hi - lo);
                            let (mut b, mut t) = (config, s.tune);
                            m.set(&mut b, &mut t, want);
                            assert!(
                                (m.get(&b, t) - want).abs() < 1e-6,
                                "{m:?} did not read back what it wrote"
                            );
                        }
                        Row::Shapes
                        | Row::Orientation
                        | Row::Effects
                        | Row::Noise
                        | Row::Note(_) => {}
                    }
                }
            }
        }
    }

    /// Every modulatable row's slot is the one its reader answers from — the pair a
    /// hand-written `match` per direction is exactly the place to get wrong.
    #[test]
    fn a_mapping_is_read_from_the_slot_it_was_written_to() {
        for effect in [
            BrushEffectType::Paint,
            BrushEffectType::Wet,
            BrushEffectType::Erase,
            BrushEffectType::Liquify,
        ] {
            for row in MOD_ROWS {
                let mut b = BrushConfig {
                    effect,
                    ..BrushConfig::default()
                };
                // Cleared first rather than asserted empty: the default brush's
                // radius already follows pressure, which is the app's opening brush
                // rather than a state this test gets to choose.
                set_source(&mut b, row, None);
                assert_eq!(row.of(&b), None, "{row:?} cleared");
                set_source(&mut b, row, Some(ModSource::Tilt));
                assert_eq!(
                    row.of(&b).map(|m| m.source),
                    Some(ModSource::Tilt),
                    "{row:?} under {effect:?} read back a different slot than it wrote"
                );
                // Switching source keeps the shape, which is the whole reason
                // `set_source` exists rather than three writes at the call site.
                if let Some(m) = row.slot(&mut b) {
                    m.floor = 0.4;
                }
                set_source(&mut b, row, Some(ModSource::Pressure));
                assert_eq!(row.of(&b).map(|m| m.floor), Some(0.4));
                set_source(&mut b, row, None);
                assert_eq!(row.of(&b), None);
            }
        }
    }

    /// A liquify brush has no opacity ceiling and no color dynamics (§6.13), and an
    /// eraser has no color dynamics either (§6.12) — so neither is offered a row that
    /// reaches nothing.
    #[test]
    fn an_effect_without_a_knob_is_not_offered_one() {
        let liquify = BrushConfig {
            effect: BrushEffectType::Liquify,
            ..BrushConfig::default()
        };
        let s = shown(liquify);
        assert!(!Section::Color.mounted(&liquify));
        assert!(!Section::Wet.mounted(&liquify));
        assert!(
            !Section::Effect
                .rows(&s)
                .contains(&Row::Mod(ModRow::Opacity))
        );
        assert!(Section::Effect.rows(&s).contains(&Row::Knob(Knob::Quality)));

        let erase = BrushConfig {
            effect: BrushEffectType::Erase,
            ..BrushConfig::default()
        };
        let s = shown(erase);
        assert!(!Section::Color.mounted(&erase));
        assert!(!Section::Wet.mounted(&erase));
        // The eraser *does* have a ceiling of its own, so its row stays.
        assert!(
            Section::Effect
                .rows(&s)
                .contains(&Row::Mod(ModRow::Opacity))
        );
        assert!(!Section::Effect.rows(&s).contains(&Row::Knob(Knob::Quality)));
    }

    /// Hardness is the procedural tip's alone: a stamp's edge is its image's, so the
    /// row is not laid out at all rather than laid out and ignored.
    #[test]
    fn a_stamp_is_offered_no_hardness() {
        let round = BrushConfig::default();
        assert!(
            Section::Tip
                .rows(&shown(round))
                .contains(&Row::Knob(Knob::Hardness))
        );
        let stamp = BrushConfig {
            shape: BrushShape::Stamp(stark_model::AssetId([7; 32])),
            ..BrushConfig::default()
        };
        let rows = Section::Tip.rows(&shown(stamp));
        assert!(!rows.contains(&Row::Knob(Knob::Hardness)));
        // ...but a stamp always has a silhouette to turn, so it always gets the chips.
        assert!(rows.contains(&Row::Orientation));
    }

    /// Every knob a section can lay out is in [`KNOBS`] — the roster a frontend builds
    /// its controls from, which is only useful if nothing can be shown that is not in
    /// it. A knob added to a section and not to the roster is a track with no control
    /// behind it, and nothing else would say so.
    #[test]
    fn the_roster_holds_every_knob_a_section_can_show() {
        for effect in [
            BrushEffectType::Paint,
            BrushEffectType::Wet,
            BrushEffectType::Erase,
            BrushEffectType::Liquify,
        ] {
            for shape in [
                BrushShape::Round { hardness: 0.5 },
                BrushShape::Stamp(stark_model::AssetId([3; 32])),
            ] {
                for stretch in [0.0, 0.3] {
                    let mut config = BrushConfig {
                        effect,
                        shape,
                        stretch,
                        ..BrushConfig::default()
                    };
                    // With a channel awake, so the two frequency rows are laid out too.
                    config.color_dynamics.amplitude[0] = 0.1;
                    let s = shown(config);
                    for section in SECTIONS.into_iter().filter(|s| s.mounted(&config)) {
                        for row in section.rows(&s).into_iter().chain(section.more(&s)) {
                            if let Row::Knob(knob) = row {
                                assert!(
                                    KNOBS.contains(&knob),
                                    "{knob:?} is laid out but is not in the roster"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every modulatable row sits in the seat its own [`ModRow::index`] names — the
    /// claim that lets a frontend keep one control per row and reach it without a
    /// search, and so without a panic for a row the roster had lost.
    ///
    /// It cannot fail while [`MOD_ROWS`] is derived from the declaration order, which
    /// is exactly what it is here to say: it is what would fail first if the roster
    /// were ever written out by hand again.
    #[test]
    fn every_row_sits_in_the_seat_its_index_names() {
        for row in MOD_ROWS {
            assert_eq!(MOD_ROWS[row.index()], row);
        }
    }

    /// **A tap is not a stroke**, and the stroke that was showing survives one.
    ///
    /// The bug this is here for cost the whole test canvas: opening the gesture had
    /// already undone the committed stroke, so marking a tap committed made the *next*
    /// edit's undo reach past it into the reference band — and two taps in a row left
    /// the canvas empty with nothing on screen saying why.
    #[test]
    fn a_tap_does_not_become_the_test_stroke() {
        let mut stroke = TestStroke::default();
        stroke.seed(
            300.0,
            500.0,
            ViewTransform::identity(stark_engine::Extent2::new(300, 500)),
        );
        let seeded = stroke.samples().len();
        stroke.replayed(true);

        // A press takes the committed stroke off, which is what the caller is told.
        assert!(stroke.begin(InputSample::default()), "the replay had one");
        assert!(!stroke.needs_undo(), "…and it is off the document now");
        assert!(!stroke.end(), "one sample is not a stroke");
        assert_eq!(
            stroke.samples().len(),
            seeded,
            "the stroke that was there stays"
        );
        assert!(
            !stroke.needs_undo(),
            "and nothing is claimed committed for a later undo to reach past"
        );

        // Two recorded samples are a gesture the caller may replay, which is what
        // becomes the test stroke — see [`TestStroke::end`] on why the count is the
        // test a frontend can make.
        assert!(!stroke.begin(InputSample::default()), "nothing to undo");
        stroke.extend(InputSample::default());
        assert!(stroke.end());
        assert_eq!(stroke.samples().len(), 2, "and it becomes the test stroke");
        assert!(stroke.needs_undo());
    }

    /// A resize re-lays the *seeded* stroke and leaves the artist's own alone — it is
    /// theirs, and it is in canvas space, so it survives untouched.
    #[test]
    fn a_resize_re_lays_only_the_stroke_nobody_drew() {
        let view = ViewTransform::identity(stark_engine::Extent2::new(300, 500));
        let mut stroke = TestStroke::default();
        stroke.seed(300.0, 200.0, view);
        let short = stroke.samples().to_vec();
        stroke.relay_after_resize(300.0, 500.0, view);
        assert_ne!(stroke.samples(), short, "the default follows the column");

        assert!(!stroke.begin(InputSample::default()));
        stroke.extend(InputSample::default());
        assert!(stroke.end());
        let drawn = stroke.samples().to_vec();
        stroke.relay_after_resize(300.0, 900.0, view);
        assert_eq!(
            stroke.samples(),
            drawn,
            "a hand's own stroke is not re-laid"
        );
    }

    /// A cancelled pointer abandons the stroke in flight without laying it down, and
    /// without claiming anything for the next replay to undo.
    #[test]
    fn a_cancelled_stroke_lays_nothing_down() {
        let mut stroke = TestStroke::default();
        stroke.seed(
            300.0,
            500.0,
            ViewTransform::identity(stark_engine::Extent2::new(300, 500)),
        );
        let seeded = stroke.samples().to_vec();
        assert!(!stroke.begin(InputSample::default()));
        stroke.extend(InputSample::default());
        stroke.cancel();
        assert!(!stroke.drawing());
        assert!(!stroke.end(), "there is no stroke left to end");
        assert_eq!(stroke.samples(), seeded);
    }

    /// The curve the plot draws is the one the renderer applies, sampled over the unit
    /// square — so a picture that left the box would be a picture of a different
    /// mapping.
    #[test]
    fn the_plotted_curve_stays_in_the_unit_square() {
        for source in SOURCES {
            for floor in [0.0, 0.5, 1.0] {
                for curve in [-1.0, 0.0, 1.0] {
                    let m = Modulation {
                        source,
                        floor,
                        curve,
                    };
                    let pts = curve_points(m);
                    assert_eq!(pts.len(), CURVE_N);
                    for (x, y) in pts {
                        assert!((0.0..=1.0).contains(&x), "input {x} left the box");
                        assert!((0.0..=1.0).contains(&y), "factor {y} left the box");
                    }
                }
            }
        }
    }
}
