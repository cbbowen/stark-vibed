//! The standing preferences a client keeps between visits — what a settings dialog
//! sets (§11).
//!
//! A setting is the one kind of state that is neither the artwork nor the tool in your
//! hand: a standing choice about how Stark behaves for **this client**, set once and
//! then left alone. It is never written into the document and never sent to peers.
//!
//! Here rather than in a frontend because it is a [`Record`]: one serde struct, one
//! key, one format that reconciles by name (§25.6). What *reads* it is each
//! frontend's own — the dialog that sets it, the signals it seeds, the commands the
//! engine half becomes — and that half stays there.
//!
//! # The contract for adding a setting
//!
//! A field on [`Prefs`], a default in its `Default`, and a control in whichever
//! frontend offers it. `#[serde(default)]` on the struct is what makes that safe
//! across versions: a field added later reads as its default out of every value
//! stored before it existed, and one removed is ignored.

use serde::{Deserialize, Serialize};

use crate::storage::{Record, Store};

/// What the floating chrome does while the canvas is in hand — this browser's own
/// choice ([`Prefs::chrome_hiding`], the ⚙ dialog's APPEARANCE section).
///
/// There were two mechanisms here and no switch: every floating container fades while
/// a gesture is in flight, and the panel *stack* then stays down afterwards until the
/// pointer reaches into its column. Which of those is wanted turns out to be a fact
/// about the hardware rather than a matter of taste — a pen on a tablet crosses the
/// panel column on the way to everything, so the wake gesture costs it nothing and it
/// gets the whole window back; a mouse reaching for a slider between strokes pays the
/// reach every time. The one thing that could not stay is the assumption.
///
/// Three states rather than two switches, because the third combination does not
/// exist: a stack that stays down after a gesture it never faded for is a panel
/// vanishing at the moment the artist stopped painting, which is nobody's preference.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ChromeHiding {
    /// Never get out of the way: the chrome is where it was put, whatever the hand is
    /// doing.
    Never,
    /// Fade for the length of a canvas gesture and come straight back when it ends.
    WhilePainting,
    /// Fade for the gesture, and leave the panel stack down afterwards until the
    /// pointer reaches into its column (§11). The default, and what Stark did before
    /// there was anything to choose.
    #[default]
    AfterPainting,
}

impl ChromeHiding {
    /// Whether a canvas gesture takes the chrome with it.
    pub fn fades(self) -> bool {
        !matches!(self, Self::Never)
    }

    /// Whether the panel stack stays down once the gesture has ended.
    ///
    /// Public because it is not only the fade's question: it decides whether
    /// a frontend's `sleep_panels` does anything at all, and the tour asks it about the one
    /// lesson whose whole subject is this gesture (§24).
    pub fn sleeps(self) -> bool {
        matches!(self, Self::AfterPainting)
    }

    /// The name this is stored and chosen by — one vocabulary for the store and the
    /// dialog, so a row of the dialog cannot come to mean a value nothing reads.
    pub fn key(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::WhilePainting => "while-painting",
            Self::AfterPainting => "after-painting",
        }
    }
}

impl From<String> for ChromeHiding {
    /// A name this build does not know reads as the default — a preference written by
    /// a later version, or a damaged one. Lenient on purpose: [`Prefs`] is one JSON blob, so an enum that refused an unknown variant would take every
    /// *other* setting down with it rather than costing its own.
    fn from(name: String) -> Self {
        [Self::Never, Self::WhilePainting, Self::AfterPainting]
            .into_iter()
            .find(|c| c.key() == name)
            .unwrap_or_default()
    }
}

impl From<ChromeHiding> for String {
    fn from(c: ChromeHiding) -> Self {
        c.key().to_string()
    }
}

/// The Lighting panel's HDR switch and the headroom slider under it (§6.5). Off,
/// the canvas is drawn as an export would be. `headroom` stands in where the
/// platform will not say how bright the display goes (the web says only *whether*
/// it is HDR). Kept whether or not the current surface can show HDR.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Hdr {
    pub on: bool,
    /// Times SDR white; a stored value outside the slider's range reads back clamped.
    pub headroom: f32,
}

impl Hdr {
    pub const MIN_HEADROOM: f32 = 1.0;
    /// Three stops over white — past what a consumer display drives above its SDR
    /// white.
    pub const MAX_HEADROOM: f32 = 8.0;

    /// The headroom as this record stores it, held to the slider's range.
    pub fn clamped_headroom(self) -> f32 {
        if self.headroom.is_finite() {
            self.headroom.clamp(Self::MIN_HEADROOM, Self::MAX_HEADROOM)
        } else {
            Self::default().headroom
        }
    }
}

impl Default for Hdr {
    fn default() -> Self {
        Self {
            // On: previewing an export is the exception, not the session.
            on: true,
            // One stop over white — what every HDR panel drives; a brighter one
            // leaves headroom unused rather than clipping.
            headroom: 2.0,
        }
    }
}

/// Every preference the ⚙ dialog sets, in the form they are stored in.
///
/// `#[serde(default)]` is what makes the struct extensible across versions — see
/// the module comment.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Whether holding a stroke still snaps it to a line or an ellipse (§6.9).
    pub assist: bool,
    /// Whether the chrome over the canvas drops its words and keeps its marks.
    pub minimal: bool,
    /// Whether the chrome gets out of the way while you paint, and for how long
    /// (§11). Not a `bool`, and the one preference here that is not: the two
    /// mechanisms behind it are three states, not four (`layout::ChromeHiding`).
    pub chrome_hiding: ChromeHiding,
    /// Whether collaborators' selections are outlined alongside your own (§17.3).
    pub show_peer_selections: bool,
    /// Whether the guided tour offers its lessons (§24).
    ///
    /// The switch alone lives here. What the tour has *learned* about this browser —
    /// the tally of deeds, the lessons already given — is a table of its own
    /// (`crate::tutor`), because it is a record of what happened rather than a
    /// choice anybody made, and because turning the tour off and on again must not
    /// be a way to lose it.
    pub tips: bool,
    /// How much GPU memory undo history may hold before the oldest steps are given
    /// up, in bytes (§5) — the one preference that is about the *machine* rather
    /// than about how Stark behaves, which is why it is the one with a slider.
    pub history_budget: u64,
    /// Whether a stroke's commit keeps the pixels its preview already drew, rather
    /// than drawing the stroke again when the pen lifts (§6.2).
    pub fast_commit: bool,
    /// The HDR switch and headroom (§6.5) — set from the Lighting panel rather than
    /// the dialog, stored here because it is a standing choice like every other row.
    pub hdr: Hdr,
}

impl Record for Prefs {
    const STORE: Store = Store::Prefs;
}

impl Default for Prefs {
    /// The app's defaults, and the authority on them: every signal these seed is
    /// overwritten by a frontend's own `prefs::load` at startup, so a value written
    /// anywhere else would
    /// be the one that never applies.
    fn default() -> Self {
        Self {
            // On, because the assist is most of the value of a hold and somebody
            // who wants their line left crooked can find the switch.
            assist: true,
            // Off, because the words are how the chrome is *learned*; minimal mode
            // is what you turn on once you no longer need them.
            minimal: false,
            // What Stark did before there was a choice, so the default is not a new
            // opinion — it is the behavior every existing browser already has, and
            // the ones that stored their preferences before this field existed read
            // as it (`ChromeHiding::default`).
            chrome_hiding: ChromeHiding::default(),
            // Off, because a second contour over the artwork is paid for on every
            // frame you look at it (§17.3).
            show_peer_selections: false,
            // On, because the tour exists for the artist who has not found the
            // switch yet, and it is the one preference whose default decides whether
            // anybody it is for ever sees it. It costs a newcomer five cards across
            // their first few sessions and nothing after that (§24).
            tips: true,
            // The engine's own default, not a second opinion about it: a value
            // written here would be the one that actually applies, and then there
            // would be two answers to what Stark does out of the box. `load_engine`
            // pushes this back in at startup, so the engine's constant has to be
            // what a browser that has never stored anything gets.
            history_budget: stark_engine::DEFAULT_HISTORY_BUDGET,
            // The engine's own default, for the budget's reason above: two answers
            // to what Stark does out of the box is one too many, and this is the
            // pair where a disagreement would be near-invisible — both settings
            // paint the same stroke.
            fast_commit: stark_engine::DEFAULT_FAST_COMMIT,
            hdr: Hdr::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stored name of a hiding mode is what the dialog offers and what
    /// [`ChromeHiding::from`] reads back, and a name from a version that knows more
    /// modes than this one reads as the default rather than refusing — which would
    /// take every *other* preference in the blob down with it ([`Prefs`]).
    #[test]
    fn a_chrome_mode_round_trips_through_its_stored_name() {
        for mode in [
            ChromeHiding::Never,
            ChromeHiding::WhilePainting,
            ChromeHiding::AfterPainting,
        ] {
            assert_eq!(ChromeHiding::from(mode.key().to_string()), mode);
        }
        assert_eq!(
            ChromeHiding::from("hide-on-tuesdays".to_string()),
            ChromeHiding::default(),
        );
        // The default is what Stark did before there was a choice, which is what keeps
        // a browser that stored its preferences before this field existed where it was.
        assert!(ChromeHiding::default().sleeps());
        assert!(ChromeHiding::default().fades());
        assert!(!ChromeHiding::Never.fades(), "never means never");
        assert!(
            !ChromeHiding::WhilePainting.sleeps(),
            "the stack comes straight back",
        );
    }

    /// A stored headroom outside the slider's range reads back inside it.
    #[test]
    fn a_stored_headroom_is_held_to_the_slider() {
        let wild = |headroom: f32| Hdr { on: true, headroom }.clamped_headroom();
        assert_eq!(wild(0.25), Hdr::MIN_HEADROOM);
        assert_eq!(wild(64.0), Hdr::MAX_HEADROOM);
        assert_eq!(wild(f32::NAN), Hdr::default().headroom);
        assert_eq!(wild(3.0), 3.0);
        // A record stored before the field existed reads as the default.
        let old: Prefs = serde_json::from_str("{}").expect("an empty record reads");
        assert_eq!(old.hdr, Hdr::default());
    }
}
