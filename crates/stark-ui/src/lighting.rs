//! What the Lighting panel offers: which environments it lists, and the five
//! continuous knobs it draws (§6.3, §6.4, §6.5).
//!
//! The rows are here and the markup is each frontend's, which is
//! [`brush_editor`](crate::brush_editor)'s split one panel over. Two of these were a
//! literal in a web `Slider` and a named constant in the native shelf whose doc said
//! it was "the same ceiling the web panel's slider carries" — a comment where a `use`
//! should have been.
//!
//! **The bytes are not here and must not come**, which is the one genuine difference
//! between the two: a browser fetches an environment's HDR through `asset!`, a native
//! binary carries it through `include_bytes!`, and §11.2's N7 argues both. Each
//! frontend keeps its own resolver; only the *list* is shared.

use stark_engine::{EnvironmentId, MediaParams, ObservableState};
use stark_model::SubstrateScale;

use crate::icons::Icon;
use crate::prefs::Hdr;

/// The selectable lighting environments, in display order (§6.3).
///
/// `Neutral` leads because it is the reference light — the achromatic one you switch
/// to in order to judge colour; the HDRs are the room you paint in.
pub const ENVIRONMENTS: &[(EnvironmentId, &str)] = &[
    (EnvironmentId::Neutral, "Neutral"),
    (EnvironmentId::Ferndale, "Ferndale studio"),
    (EnvironmentId::BloemHill, "Bloem hill"),
    (EnvironmentId::KloofendalOvercast, "Kloofendal overcast"),
    (EnvironmentId::QwantaniDusk, "Qwantani dusk"),
];

/// What the app lights the canvas with on startup: the achromatic reference light,
/// which is also what the engine boots on.
///
/// Paint reads as its own colour under it — the media pass is an identity there,
/// normalized by the irradiance a flat canvas receives (§6.3) — so what you mix is
/// what you see, and a studio HDR is a deliberate switch into a room. Named because a
/// frontend's startup fetches its bytes if it has any; `Neutral` is procedural, so
/// today that fetch is skipped.
pub const DEFAULT_ENVIRONMENT: EnvironmentId = EnvironmentId::Neutral;

/// The panel's continuous knobs, in the order it draws them.
///
/// Ordered light-first: the two relief strengths and the gloss describe how the paint
/// meets the light, the scale describes what it is lying on, and the headroom is about
/// the display rather than the picture at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray, strum::EnumCount)]
pub enum Dial {
    /// How strongly the paint's own relief tilts the light (§6.3).
    Impasto,
    /// How strongly the canvas substrate's relief does.
    Texture,
    /// How glossy the paint is.
    Gloss,
    /// How large the substrate is laid (§6.4) — **document state**, unlike the three
    /// above, which are a view setting.
    Scale,
    /// How far above SDR white this display is driven, where the platform will not
    /// say (§6.5). This client's own preference, not the document's and not the
    /// engine's.
    Headroom,
}

impl Dial {
    /// Which seat this dial has, so a view holding one state per dial can index
    /// rather than search. Exhaustive, so a sixth dial does not compile until it
    /// says where it sits — where a `position().expect()` would have panicked at
    /// the moment the panel opened.
    pub fn index(self) -> usize {
        match self {
            Dial::Impasto => 0,
            Dial::Texture => 1,
            Dial::Gloss => 2,
            Dial::Scale => 3,
            Dial::Headroom => 4,
        }
    }

    /// The mark the control wears.
    pub fn glyph(self) -> Icon {
        match self {
            Dial::Impasto => crate::icons::IMPASTO,
            Dial::Texture => crate::icons::TEXTURE,
            Dial::Gloss => crate::icons::GLOSS,
            Dial::Scale => crate::icons::SUBSTRATE_SCALE,
            Dial::Headroom => crate::icons::HDR,
        }
    }

    /// The one word a caption gives it.
    pub fn label(self) -> &'static str {
        match self {
            Dial::Impasto => "Impasto",
            Dial::Texture => "Texture",
            Dial::Gloss => "Gloss",
            Dial::Scale => "Scale",
            Dial::Headroom => "Headroom",
        }
    }

    /// What it says on hover, where a frontend has hover to say it on — the label and
    /// then what the knob is actually *for*, since a one-word caption cannot carry it.
    pub fn tip(self) -> &'static str {
        match self {
            Dial::Impasto => {
                "Impasto \u{2014} how strongly the paint's own relief catches the light"
            }
            Dial::Texture => "Texture \u{2014} how strongly the canvas weave shows through",
            Dial::Gloss => "Gloss \u{2014} how wet the paint reads under the light",
            Dial::Scale => "Scale \u{2014} how large the canvas surface is laid",
            Dial::Headroom => "Headroom \u{2014} how far above white this display is driven",
        }
    }

    /// The ends of its track.
    ///
    /// Each is either the model's own bound or a ceiling this control owns, and the
    /// second kind is why one copy of this table rather than one per frontend.
    pub fn range(self) -> (f32, f32) {
        match self {
            Dial::Impasto | Dial::Texture => (0.0, 1.0),
            // A ceiling this control owns: past a third the specular lobe is a mirror
            // rather than paint.
            Dial::Gloss => (0.0, 0.35),
            Dial::Scale => (SubstrateScale::MIN as f32, SubstrateScale::MAX as f32),
            Dial::Headroom => (Hdr::MIN_HEADROOM, Hdr::MAX_HEADROOM),
        }
    }

    /// The step a track moves in. The substrate's is the *lattice its own value lands
    /// on* ([`SubstrateScale::STEP`]), so a track cannot offer a position `new` would
    /// move the handle off.
    pub fn step(self) -> f32 {
        match self {
            Dial::Impasto | Dial::Texture | Dial::Gloss => 0.01,
            Dial::Scale => SubstrateScale::STEP as f32,
            Dial::Headroom => 0.1,
        }
    }

    /// Where the dial stands, off the engine's projection and this client's record —
    /// never off a copy kept beside it, which would go stale under an undo or a load
    /// (§4).
    pub fn read(self, o: Option<&ObservableState>, hdr: Hdr) -> f32 {
        let media = o.map_or_else(MediaParams::default, |o| o.media);
        match self {
            Dial::Impasto => media.height_strength,
            Dial::Texture => media.substrate_strength,
            Dial::Gloss => media.specular,
            Dial::Scale => o
                .map_or(SubstrateScale::NATURAL, |o| o.substrate_scale)
                .percent() as f32,
            Dial::Headroom => hdr.clamped_headroom(),
        }
    }

    /// How a readout prints it: a percentage for the substrate, a multiple for the
    /// headroom, two places otherwise.
    pub fn readout(self, v: f32) -> String {
        match self {
            Dial::Scale => format!("{v:.0}%"),
            Dial::Headroom => format!("{v:.1}\u{00d7}"),
            _ => format!("{v:.2}"),
        }
    }
}

/// Which dials this frame shows.
///
/// The headroom is the only conditional one, and both halves of its condition are
/// real: there is nothing to set with the switch off, and nothing to *guess* on a
/// display that states its own.
pub fn dials(hdr: Hdr, hdr_capable: bool, display_headroom: Option<f32>) -> Vec<Dial> {
    <Dial as strum::VariantArray>::VARIANTS
        .iter()
        .copied()
        .filter(|dial| {
            *dial != Dial::Headroom || (hdr.on && hdr_capable && display_headroom.is_none())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::VariantArray;

    /// The reference light leads, and it is what the app opens on — the two are one
    /// claim, and a list reordered without the constant following would break it
    /// silently.
    #[test]
    fn the_reference_light_leads_and_is_the_default() {
        assert_eq!(ENVIRONMENTS.first().map(|r| r.0), Some(DEFAULT_ENVIRONMENT));
    }

    /// Every environment is named once. A duplicate name in a drop-down is two rows a
    /// user cannot tell apart.
    #[test]
    fn every_environment_is_named_once() {
        let mut names: Vec<_> = ENVIRONMENTS.iter().map(|r| r.1).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two environments share a name");
    }

    /// Every dial says something on hover, and says it about itself — what a column
    /// with no captions stands on. The label leads the tip so the two cannot come to
    /// call one knob two things.
    #[test]
    fn every_dial_says_what_it_does() {
        for dial in Dial::VARIANTS {
            assert!(!dial.label().is_empty(), "{dial:?} has no caption");
            assert!(
                dial.tip().starts_with(dial.label()),
                "{dial:?}'s hover does not lead with its own name: {:?}",
                dial.tip()
            );
        }
    }

    /// A track a hand cannot move back off is the failure this table exists to rule
    /// out, so every range is non-empty and every step fits inside it.
    #[test]
    fn every_track_is_one_a_hand_can_move() {
        for dial in Dial::VARIANTS {
            let (lo, hi) = dial.range();
            assert!(lo < hi, "{dial:?} has an empty track: {lo}..={hi}");
            assert!(dial.step() > 0.0, "{dial:?} steps by nothing");
            assert!(
                dial.step() <= hi - lo,
                "{dial:?} steps further than its own track"
            );
        }
    }

    /// The headroom is the one dial that comes and goes, and it wants all three of its
    /// conditions — the switch on, the display able, and the platform silent.
    #[test]
    fn the_headroom_dial_needs_all_three_of_its_conditions() {
        // `Hdr::default` is *on* — previewing an export is the exception, not the
        // session — so the off case has to be built rather than defaulted.
        let on = Hdr::default();
        let off = Hdr { on: false, ..on };
        assert!(dials(on, true, None).contains(&Dial::Headroom));
        assert!(
            !dials(on, true, Some(2.0)).contains(&Dial::Headroom),
            "a display that states its own is not guessed at"
        );
        assert!(
            !dials(on, false, None).contains(&Dial::Headroom),
            "nothing to drive on an SDR display"
        );
        assert!(
            !dials(off, true, None).contains(&Dial::Headroom),
            "nothing to set with the switch off"
        );
        // The other four are unconditional.
        assert_eq!(dials(off, false, None).len(), Dial::VARIANTS.len() - 1);
    }

    /// The seat and the roster are one order. They are two statements, and a view
    /// indexes by the first into a list built from the second.
    #[test]
    fn every_dial_sits_in_the_seat_its_index_names() {
        for (i, dial) in Dial::VARIANTS.iter().enumerate() {
            assert_eq!(dial.index(), i, "{dial:?} names a seat it does not sit in");
        }
    }
}
