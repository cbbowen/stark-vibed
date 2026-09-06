//! The brush preset library: named [`BrushConfig`] snapshots, and the ones the app
//! ships (§6.2, §18.1.8).
//!
//! A preset is a whole brush **except the painting color**: applying one keeps the
//! hand's RGB while everything else — including the effect's own opacity — comes from
//! the preset. Both halves of the brush, that is: the durable one, which is what the
//! tool *is*, and the transient one it was saved at ([`Transient`]).
//!
//! What is here is the part with no app in it: the entry, the shipped table, the
//! arithmetic over a library (is this name taken, what an overwrite would mean), and
//! the stored form. What *applies* a preset is each frontend's — it writes signals,
//! reports a deed to a tour, dispatches a command — and stays there.
//!
//! The shipped table is here rather than in a frontend because it is the definition
//! of the app's tools, and there is one app. It takes its two bundled *shapes* as a
//! parameter ([`shipped`]), for the reason a content id cannot be written down: an id
//! is the hash of the bytes, so it is not knowable until they have been imported.

use stark_model::document::{
    BrushModulations, BrushShape, ColorDynamics, ModSource, Modulation, NoiseKind,
    OrientationSource, ToothParams,
};

use crate::brush_config::{
    BrushConfig, BrushEffectType, EraseConfig, LiquifyConfig, Transient, WetDynamics,
};
use crate::slots;
use crate::storage::{self, Store};

/// One named preset in the library.
#[derive(Clone, PartialEq)]
pub struct PresetEntry {
    /// Display name; unique in the library (saving under a taken name
    /// overwrites, and a name the app has already taken is refused).
    pub name: String,
    /// The tool applied by clicking the preset — the durable half.
    pub brush: BrushConfig,
    /// The size and flow it was saved at — the transient half, carried so that
    /// clicking a preset puts the tool on at the tune it was kept at, while a
    /// quick slot substitutes a tune of its own (`slots::resolve`).
    pub transient: Transient,
    /// The digit this preset **ships on**, if any — how a fresh quick-brush rack
    /// is filled (§18.1.8; `slots::seed_defaults`).
    ///
    /// Not the binding itself: a slot names its preset from the rack's side
    /// (`slots::QuickBrush`), and this says only where the app puts a tool on the
    /// keyboard the first time somebody sees one — which is why only the app's
    /// own presets declare it and everything the user saves has `None`. The rack
    /// is the user's to arrange after that, and a preset does not follow its
    /// shipped digit around once it has been moved off it.
    pub slot: Option<usize>,
    /// Whether this is one of the app's own presets rather than the user's.
    ///
    /// Never stored: it is *provenance*, set where the entry is made — true in
    /// a frontend's `default_presets`, false where a [`StoredPreset`] becomes one — so it cannot
    /// disagree with where the entry actually came from. What it decides is the row's
    /// last control (a lock, not a trash) and whether storage ever sees the entry.
    pub builtin: bool,
}

/// The two bundled shapes the app's own presets reach for, resolved to content ids.
///
/// [`Default`] is the round tip for both, which is what a session that has not
/// imported the bundled images yet gets — see [`shipped`].
#[derive(Default)]
pub struct BuiltinShapes {
    pub bristles: BrushShape,
    pub pencil: BrushShape,
}

/// The presets the app ships, with their one dependency on a running engine passed
/// in: the two stamps, each a content id that cannot be written down until its bytes
/// have been imported.
///
/// The first entry is also the brush the app opens on, so it leads with the everyday
/// one. A shape that has not arrived degrades to the round tip **for that session
/// only**, since nothing here is persisted; the next start resolves it properly
/// rather than inheriting a bad one.
///
/// The `slot` each declares is what fills a fresh quick-brush rack, and it is
/// independent of the order here: the eraser is last in the list and first on the
/// keyboard, because a list is read top-down and a rack is reached by the digit under
/// the finger.
///
/// Taking the shapes as a parameter is what makes the list testable — which matters more than it looks. The
/// invariants below (every entry shipped, every slot distinct and on the rack)
/// are the kind that a sixth preset breaks silently: the rack would quietly drop
/// or overwrite one, and nothing on screen would say which.
pub fn shipped(shapes: BuiltinShapes) -> Vec<PresetEntry> {
    // One constructor for the lot, so "everything the app ships is a built-in and
    // sits on a digit" is a property of the list rather than five copies of two
    // fields that could drift apart. `smoothing` is the §6.11 amount, part of
    // what a tool *is*: the inker leans on the string, the pencil keeps every
    // tremor because tremor is what a pencil is for — its own argument rather
    // than a field in the definitions below, so no definition can forget to say.
    // `size` and `flow` are the transient half every preset carries
    // (`PresetEntry::transient`) — arguments for the same reason, since a
    // definition cannot say them anywhere else now that the config is the
    // durable half alone.
    let shipped = |name: &str,
                   slot: Option<usize>,
                   smoothing: f32,
                   size: f32,
                   flow: f32,
                   brush: BrushConfig| PresetEntry {
        name: name.to_string(),
        brush: BrushConfig { smoothing, ..brush },
        transient: Transient {
            size,
            flow,
            // The color a preset's tune carries is never applied ([`wear`]
            // keeps the hand's), so the definitions do not say one.
            ..Transient::default()
        },
        slot,
        builtin: true,
    };
    vec![
        shipped(
            "Hard Round",
            Some(1),
            0.15,
            100.0,
            3.0,
            BrushConfig {
                drain: 0.1,
                shape: BrushShape::Round { hardness: 0.98 },
                effect: BrushEffectType::Wet,
                opacity: 1.0,
                flow_modulation: Some(Modulation::linear(ModSource::Pressure)),
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                // The axes are per pass at the neutral flow (§6.2), and this
                // brush rides at flow 3, so they are quoted low: at full crank
                // the exchange lands where the old per-pass fractions did
                // (1 − (1 − a)³), and a light touch — the flow mapping above —
                // now smears less as well as laying less.
                wet: WetDynamics {
                    lift: 0.1,
                    deposit: 0.37,
                    bleed: 0.08,
                    ..WetDynamics::default()
                },
                modulation: BrushModulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    ..BrushModulations::default()
                },
                ..BrushConfig::default()
            },
        ),
        shipped(
            "Bristles",
            None,
            0.15,
            100.0,
            3.0,
            BrushConfig {
                shape: shapes.bristles,
                effect: BrushEffectType::Wet,
                opacity: 1.0,
                flow_modulation: Some(Modulation::linear(ModSource::Pressure)),
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                // Quoted low for the flow-3 ride, the Hard Round's argument.
                wet: WetDynamics {
                    lift: 0.1,
                    deposit: 0.37,
                    bleed: 0.17,
                    ..WetDynamics::default()
                },
                modulation: BrushModulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    ..BrushModulations::default()
                },
                ..BrushConfig::default()
            },
        ),
        // An inking pen: hard, opaque, never runs dry, and pointed at both ends —
        // a short entry taper and a longer exit, which is the asymmetry a hand
        // makes when it lands a line and flicks off it.
        shipped(
            "Pen",
            Some(2),
            0.5,
            18.0,
            1.0,
            BrushConfig {
                shape: BrushShape::Round { hardness: 1.0 },
                start_taper_length: 5.0,
                end_taper_length: 11.0,
                ..BrushConfig::default()
            },
        ),
        // A pencil: a broken-edged stamp held at the pen's own angle, laying
        // very little paint per pass so the tooth of the paper shows and a
        // second pass over the same line reads as darker.
        //
        // The one preset that maps the pen somewhere other than the default
        // (§6.2), because a pencil is the tool where the two axes visibly do
        // different jobs: **tilt draws the tip out** — laying a pencil over is how
        // you get a shading stroke out of the same point — and **pressure darkens
        // it**, which is the graphite the lead gives up rather than the shape of
        // the contact patch.
        //
        // Tilt drives `stretch` and not `size`, and the difference is the whole
        // point of the axis (§6.6). A leaned cone does not contact a bigger
        // circle, it contacts a longer *ellipse* — drawn out along the lean and
        // no wider across it — so a size mapping made a leaned pencil a fat round
        // marker, broadening the mark just as much whichever way the hand went.
        // Stretched, the same lean broadens a stroke dragged across the lean and
        // merely darkens one dragged along it, which is what the hand expects
        // from the pencil it is imitating and what it could not ask for before.
        shipped(
            "Pencil",
            Some(3),
            0.0,
            30.0,
            0.4,
            BrushConfig {
                shape: shapes.pencil,
                jitter: 0.1,
                tooth: ToothParams {
                    give: 1.0,
                    softness: 0.5,
                },
                orientation: OrientationSource::Pen,
                stretch: 0.75,
                modulation: BrushModulations {
                    stretch: Some(Modulation {
                        source: ModSource::Tilt,
                        floor: 0.0,
                        curve: -0.5,
                    }),
                    tooth_give: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.5,
                        curve: 0.0,
                    }),
                    ..Default::default()
                },
                opacity: 1.0,
                opacity_modulation: Some(Modulation::linear(ModSource::Pressure)),
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::White,
                    frequency: [0.5, 0.0],
                    amplitude: [0.01, 0.0, 0.0],
                },
                ..Default::default()
            },
        ),
        shipped(
            "Airbrush",
            Some(4),
            0.1,
            500.0,
            0.1,
            BrushConfig {
                shape: BrushShape::Round { hardness: 0.5 },
                modulation: BrushModulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.6,
                        curve: 0.0,
                    }),
                    ..BrushModulations::default()
                },
                opacity: 1.0,
                flow_modulation: Some(Modulation::linear(ModSource::Pressure)),
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                ..BrushConfig::default()
            },
        ),
        shipped(
            "Blender",
            Some(5),
            0.1,
            100.0,
            1.0,
            BrushConfig {
                tooth: ToothParams {
                    give: 0.5,
                    ..ToothParams::default()
                },
                shape: BrushShape::Round { hardness: 0.8 },
                effect: BrushEffectType::Wet,
                opacity: 1.0,
                // The Flow slider scales the *blend* — what the flow/add split
                // was for. What makes it a blender is `add = 0` below, which no
                // slider moves.
                wet: WetDynamics {
                    add: 0.0,
                    lift: 0.25,
                    deposit: 0.75,
                    bleed: 0.25,
                    lift_modulation: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    bleed_modulation: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.0,
                        curve: 0.0,
                    }),
                    ..WetDynamics::default()
                },
                ..BrushConfig::default()
            },
        ),
        // The eraser the pen's other end starts life holding (§18.1.8). An
        // eraser is the `Erase` effect (§6.12) — it removes what the eye
        // sees, so a half-pressure pass really is a lighter erase — and the
        // hand's flow is its rate, which is where the pressure mapping points:
        // light touch feathers the coverage in, borne down it walks to the
        // full bite.
        shipped(
            "Soft Eraser",
            Some(slots::ERASER),
            0.0,
            80.0,
            1.0,
            BrushConfig {
                shape: BrushShape::Round { hardness: 0.25 },
                effect: BrushEffectType::Erase,
                erase: EraseConfig {
                    opacity_modulation: Some(Modulation::linear(ModSource::Pressure)),
                    ..Default::default()
                },
                modulation: BrushModulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    ..BrushModulations::default()
                },
                ..BrushConfig::default()
            },
        ),
        shipped(
            "Hard Eraser",
            None,
            0.0,
            40.0,
            // Enough that one pass saturates the bite to the tip's very
            // shoulder — the hard edge the name promises.
            2.0,
            BrushConfig {
                shape: BrushShape::Round { hardness: 0.95 },
                effect: BrushEffectType::Erase,
                erase: EraseConfig {
                    opacity_modulation: Some(Modulation::linear(ModSource::Pressure)),
                    ..Default::default()
                },
                modulation: BrushModulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    ..Default::default()
                },
                ..BrushConfig::default()
            },
        ),
        // The liquify brush (§6.13): the stroke drags the picture with it —
        // structure warps along the travel instead of mixing toward a mean. The
        // tip holds its width, and what the pen drives is the *grip*: a light
        // touch nudges the paint, borne down it keeps pace with the hand — so
        // the one mapping points at strength, not size. A soft-ish shoulder is
        // what makes the drag read as a field bending rather than a disc
        // shearing against its rim.
        shipped(
            "Liquify",
            None,
            0.2,
            90.0,
            1.0,
            BrushConfig {
                shape: BrushShape::Round { hardness: 0.0 },
                effect: BrushEffectType::Liquify,
                liquify: LiquifyConfig {
                    strength_modulation: Some(Modulation::linear(ModSource::Pressure)),
                    ..Default::default()
                },
                modulation: BrushModulations::default(),
                ..BrushConfig::default()
            },
        ),
    ]
}

/// Populate the library signal with the **user's** presets from storage. Called
/// once at app start, before the renderer exists, so what this browser saved is
/// on screen right away.
///
/// Whether `name` belongs to one of the app's own presets — which the Save
/// dialog asks before offering to write it, since a name taken by a built-in is
/// not free to overwrite (a frontend's `save_current` refuses it) and a second row under
/// the same name would make "the preset called X" ambiguous.
pub fn is_builtin(entries: &[PresetEntry], name: &str) -> bool {
    entries.iter().any(|e| e.builtin && e.name == name)
}

/// The first free "Preset N" name — the default for a save with the name field
/// left empty.
pub fn next_name(entries: &[PresetEntry]) -> String {
    (1..)
        .map(|i| format!("Preset {i}"))
        .find(|n| !entries.iter().any(|e| &e.name == n))
        .unwrap()
}

/// The library's entry called `name`, if it still has one.
pub fn find<'a>(entries: &'a [PresetEntry], name: &str) -> Option<&'a PresetEntry> {
    entries.iter().find(|e| e.name == name)
}

/// Whether the live brush is still **this tool** — the preset in every durable
/// knob. Nothing is set aside any more: with the transient — the color
/// included — a value of its own, [`BrushConfig`] *is* the durable half and
/// this is plain equality, which is the whole of what the split bought this
/// test. What the preset rows light on: a tool put on and then
/// sized up is the same tool at another size, so the row stays lit, while the
/// smoothing amount is one of the knobs (§6.11) and a preset worn and then
/// smoothed differently is no longer that preset. The effect's own opacity is
/// durable too — part of what the tool does (§6.2) — so the row goes out
/// the moment it moves.
///
/// Exact equality, on purpose: the row says the brush still *is*
/// the preset, not that it resembles one. Kept as a named function rather
/// than an `==` at the call sites, because it is the *rule* the rows light on
/// and the place its docs live.
pub fn same_tool(current: &BrushConfig, preset: &BrushConfig) -> bool {
    current == preset
}

/// Whether two brushes are **the same brush** — equal in both halves, the
/// painting color alone set aside ([`Transient::same_tune`]), since a frontend's `apply`
/// deliberately leaves it. The
/// stricter of the two tests, for the questions a size counts in: whether a
/// quick slot's brush is the one in hand (`slots::SlotOverlay`), and whether
/// writing the brush back over its preset would change anything
/// ([`Overwrite::Unchanged`]), since a preset carries the transient half too.
pub fn same_brush(a: &(BrushConfig, Transient), b: &(BrushConfig, Transient)) -> bool {
    same_tool(&a.0, &b.0) && a.1.same_tune(&b.1)
}

/// What "Overwrite preset" can do with the brush in hand. The brush editor's
/// button is drawn from this — dead in every arm but one, with the arm as its
/// tooltip — and a frontend's `overwrite_in_hand` acts on that one arm. A pure rule over the
/// library, the name in hand (a frontend's `preset_in_hand`) and the brush, so a
/// test can reach it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Overwrite {
    /// The brush was taken from no preset the library still has.
    Nothing,
    /// The preset in hand is one of the app's own, which is not on offer: the
    /// next start rebuilds it from its definition, so the write would not last
    /// (see a frontend's `save_current`).
    Builtin(String),
    /// The preset in hand already *is* this brush (color aside): nothing to write.
    /// Judged with the transient half counted ([`same_brush`]): a preset keeps a
    /// size and a flow, so a brush moved only in those has something to write.
    Unchanged(String),
    /// Replace the named preset's snapshot with the brush in hand.
    Ready(String),
}

/// Decide [`Overwrite`] for the brush in hand — both halves of it, since a
/// preset carries both.
pub fn overwrite(
    entries: &[PresetEntry],
    in_hand: Option<&str>,
    brush: &(BrushConfig, Transient),
) -> Overwrite {
    let Some(entry) = in_hand.and_then(|name| find(entries, name)) else {
        return Overwrite::Nothing;
    };
    let name = entry.name.clone();
    if entry.builtin {
        Overwrite::Builtin(name)
    } else if same_brush(brush, &(entry.brush, entry.transient)) {
        Overwrite::Unchanged(name)
    } else {
        Overwrite::Ready(name)
    }
}

// --- persistence ----------------------------------------------------------
//
// One entry per **user** preset, through `stark_ui::storage`, which is where the format
// and the skip-a-damaged-entry rule live. What is this module's own is which entries
// are written and which two fields are not:
//
// The app's own presets are filtered out on the way in and out, which is what makes
// them updatable at all — there is one copy of a built-in, in `default_presets`, so
// there is nothing stale for a new version to disagree with. Neither of their two
// extra fields is stored: `slot` belongs to the definition and `builtin` is
// provenance, and a stored copy of either would be a second opinion about a question
// the code already answers.

/// One stored preset: a name and a whole brush — both halves, the tool and the
/// tune it was saved at.
///
/// A type of its own rather than [`PresetEntry`] with two fields skipped, because
/// provenance is the point: everything storage holds is the user's by definition, so
/// `builtin: false` is settled where the entry is *made* rather than carried in the
/// record and trusted. No `#[serde(default)]` on `transient`: a stored preset
/// that lacks its tune is a damaged entry, which the store already skips.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StoredPreset {
    name: String,
    brush: BrushConfig,
    transient: Transient,
}

impl storage::Entry for StoredPreset {
    const STORE: Store = Store::Presets;
}

pub fn persist(entries: &[PresetEntry]) {
    let stored: Vec<StoredPreset> = entries
        .iter()
        .filter(|e| !e.builtin)
        .map(|e| StoredPreset {
            name: e.name.clone(),
            brush: e.brush,
            transient: e.transient,
        })
        .collect();
    storage::save_list(&stored);
}

/// What this browser has saved, or `None` where it has saved nothing (or
/// storage is unavailable). Either way the app's own presets arrive separately,
/// so there is no "never seeded" state left to tell apart from an empty one.
pub fn read_storage() -> Option<Vec<PresetEntry>> {
    Some(
        storage::load_list::<StoredPreset>()?
            .into_iter()
            .map(|e| PresetEntry {
                name: e.name,
                brush: e.brush,
                transient: e.transient,
                slot: None,
                builtin: false,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list as the app ships it, with the built-in stamps standing in — the
    /// one thing a live canvas would supply, and nothing here depends on which
    /// shapes they are.
    fn table() -> Vec<PresetEntry> {
        shipped(BuiltinShapes::default())
    }

    #[test]
    fn everything_the_app_ships_says_so() {
        // The flag decides two things at once — the row's lock, and whether
        // storage ever sees the entry — so an entry that forgot it would be
        // deletable *and* would persist, and the next start would then have two
        // presets of the same name.
        assert!(table().iter().all(|e| e.builtin));
    }

    #[test]
    fn every_shipped_preset_has_a_distinct_home_on_the_rack() {
        // `slots::seed_defaults` writes each preset to its own digit, so a
        // repeat would silently mean "whichever came last" and a slot past the
        // end would be dropped — neither of them visible anywhere.
        let mut seen = Vec::new();
        for entry in table() {
            let Some(slot) = entry.slot else {
                continue;
            };
            assert!(slot < slots::COUNT, "{} ships on {slot}", entry.name);
            assert!(!seen.contains(&slot), "two presets ship on {slot}");
            seen.push(slot);
        }
    }

    #[test]
    fn names_are_unique() {
        // Every lookup in this module is by name (`apply`, `remove`,
        // `is_builtin`), and the panel keys its rows on one.
        let mut names: Vec<String> = table().into_iter().map(|e| e.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
    }

    #[test]
    fn the_shipped_eraser_erases() {
        // What makes it an eraser at all (§6.12) — there is no eraser *tool* to
        // check for, only these two numbers: the dial that routes the stroke
        // through the erase pass, and the transient flow that is that pass's
        // rate now that the rate is the hand's (§6.2).
        let e = table()
            .into_iter()
            .find(|e| e.slot == Some(slots::ERASER))
            .expect("an eraser ships on the pen's own slot");
        assert_eq!(
            e.brush.effect,
            BrushEffectType::Erase,
            "the pen's tail must erase, not paint"
        );
        assert!(
            e.brush.erase.opacity > 0.0,
            "an eraser at no strength does nothing"
        );
        assert!(
            e.transient.flow > 0.0,
            "flow is the erase pass's rate; at zero the eraser would do nothing"
        );
    }

    #[test]
    fn the_startup_brush_is_not_the_eraser() {
        // `apply_first` opens the app on the library's first entry, and the list
        // order is free to differ from the rack's — so this is the one place the
        // two orders have to be checked against each other.
        let first = table().into_iter().next().expect("the app ships presets");
        assert_ne!(first.slot, Some(slots::ERASER));
    }

    /// The line between the halves, as the preset rows light on it: every
    /// durable knob counts, and neither half of the hand's state can — the
    /// transient, color included, is a different value entirely now
    /// ([`same_tool`] takes only configs).
    #[test]
    fn the_same_tool_is_the_preset_at_any_size_and_flow() {
        let preset = BrushConfig {
            drain: 0.2,
            ..BrushConfig::default()
        };
        let mut thinned = preset;
        thinned.set_opacity(0.4);
        let smoothed = BrushConfig {
            smoothing: 0.5,
            ..preset
        };
        let erasing = BrushConfig {
            effect: BrushEffectType::Erase,
            ..preset
        };

        assert!(!same_tool(&thinned, &preset), "opacity is the tool's");
        assert!(!same_tool(&smoothed, &preset), "the feel is the tool's");
        assert!(!same_tool(&erasing, &preset), "the effect is the tool");
        // The stricter test counts the tune — its color set aside, which is
        // the Color panel's (`Transient::same_tune`).
        let tune = Transient::default();
        let resized = Transient { size: 80.0, ..tune };
        let recolored = Transient {
            color: [0.9, 0.1, 0.2],
            ..tune
        };
        assert!(same_brush(&(preset, recolored), &(preset, tune)));
        assert!(!same_brush(&(preset, resized), &(preset, tune)));
    }

    #[test]
    fn overwrite_answers_for_the_preset_in_hand() {
        // The rule the brush editor's button is drawn from: dead with no name in
        // hand or a name the library no longer has, dead on the app's own (the
        // write would not outlast the next start), dead where there is nothing
        // to write, and live only for the user's own preset the brush has moved
        // off — by something a preset carries, which the color is not. The size
        // is: a preset keeps the transient half, so a resize is a write.
        let mut entries = table();
        let mine = BrushConfig {
            drain: 0.2,
            ..BrushConfig::default()
        };
        let tune = Transient {
            size: 33.0,
            ..Transient::default()
        };
        entries.push(PresetEntry {
            name: "Mine".into(),
            brush: mine,
            transient: tune,
            slot: None,
            builtin: false,
        });
        let builtin = entries[0].name.clone();
        let edited = (mine, Transient { size: 34.0, ..tune });
        let recolored = Transient {
            color: [0.9, 0.1, 0.2],
            ..tune
        };

        assert_eq!(overwrite(&entries, None, &edited), Overwrite::Nothing);
        assert_eq!(
            overwrite(&entries, Some("Gone"), &edited),
            Overwrite::Nothing
        );
        assert_eq!(
            overwrite(&entries, Some(&builtin), &edited),
            Overwrite::Builtin(builtin)
        );
        assert_eq!(
            overwrite(&entries, Some("Mine"), &(mine, tune)),
            Overwrite::Unchanged("Mine".into())
        );
        assert_eq!(
            overwrite(&entries, Some("Mine"), &(mine, recolored)),
            Overwrite::Unchanged("Mine".into())
        );
        assert_eq!(
            overwrite(&entries, Some("Mine"), &edited),
            Overwrite::Ready("Mine".into())
        );
    }
}
