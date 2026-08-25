//! The brush preset library: named [`BrushParams`] snapshots, shown as a
//! section at the foot of the Brush panel.
//!
//! A preset is a whole brush **except the painting color**: applying one keeps
//! the current RGB (color belongs to the Color panel) while everything else —
//! including the brush's own opacity, `color[3]` — comes from the preset
//! ([`wear`]).
//!
//! # Two kinds of entry, one list
//!
//! - **The app's own** ([`default_presets`]) — the everyday brush, a tapered
//!   inking pen, a pencil on one of the bundled stamps, an airbrush, and an
//!   eraser. These are **code, not data**: built fresh on every start and never
//!   written to storage, so improving one reaches every browser on its next
//!   visit rather than only the browsers that have never run Stark. While the
//!   engine's own parameters are still moving under them — a dynamics axis
//!   rescaled, a mapping added — that is the difference between a default that
//!   is *current* and one that is a fossil of whichever version first ran here.
//! - **The user's own** — everything the Save button puts there, which is all
//!   `localStorage` holds.
//!
//! The rows differ by one control: the user's wear a trash, the app's wear a
//! lock. The lock is not a rule the panel enforces, it is a fact it reports —
//! there is nothing stored behind a built-in to remove, and next start would
//! rebuild it from the same source in any case.
//!
//! A browser carrying the *old* scheme's persisted copies of the built-ins drops
//! them the first time the two lists are merged ([`install_builtins`]), and its
//! storage is rewritten without them. The app's definition wins, which is the
//! whole point: a stored copy is a staler one.
//!
//! Like the shape library (`crate::shapes`), the user's half is frontend state
//! that follows this browser across documents via `localStorage` — and degrades
//! to a per-session library where storage is unavailable.

use dioxus::prelude::*;

use stark_model::document::{
    BrushDynamics, BrushParams, BrushShape, ColorDynamics, ModSource, Modulation, Modulations,
    NoiseKind, OrientationSource,
};

use crate::builtins;
use crate::slots;
use crate::state::{AppState, update_brush};
use crate::storage::{self, Store};

/// A brush as the frontend carries it: the engine's own parameters plus the
/// **feel** the frontend owns — today just the stroke-smoothing amount
/// (§6.11).
///
/// One type rather than a field beside a field, so a whole-brush snapshot that
/// lost its feel is unrepresentable: the preset library, the quick-brush rack
/// (`crate::slots`) and a hold's swap all traffic in this. The feel is not on
/// [`BrushParams`] because the stored path already embodies the smoothing — a
/// field there would be one that replay reads and ignores, and the log's encoding makes
/// appending it a wire-version bump (§8).
///
/// Serde, because this is what both the preset library and the quick-brush rack store —
/// one stored shape for one type, so the two libraries cannot come to disagree about
/// what a stored brush is.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Wearable {
    pub params: BrushParams,
    /// Stroke smoothing, 0..=1 (§6.11) — the knob, not the rope. The rope is
    /// derived at gesture start (`input::rope`), because the knob is
    /// denominated in the hand's own screen px and only a live view converts
    /// it.
    ///
    /// Defaulted so a brush stored before §6.11 reads as unsmoothed, and clamped on
    /// the way in because the range **is** the number's meaning: a hand-edited store
    /// must not be able to hand the fitter a rope it cannot use.
    #[serde(default, deserialize_with = "stored_smoothing")]
    pub smoothing: f32,
}

/// [`Wearable::smoothing`]'s gate — see the field.
fn stored_smoothing<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    use serde::Deserialize;
    Ok(f32::deserialize(d)?.clamp(0.0, 1.0))
}

/// The live brush as a [`Wearable`]: the engine's parameters beside the feel
/// this frontend holds. `None` before the engine exists.
pub fn worn(state: AppState) -> Option<Wearable> {
    let params = state.obs.peek().as_ref().map(|o| o.brush)?;
    Some(Wearable {
        params,
        smoothing: *state.smoothing.peek(),
    })
}

/// One named preset in the library.
#[derive(Clone, PartialEq)]
pub struct PresetEntry {
    /// Display name; unique in the library (saving under a taken name
    /// overwrites, and a name the app has already taken is refused).
    pub name: String,
    /// The snapshot applied by clicking the preset.
    pub brush: Wearable,
    /// The digit this preset **ships on**, if any — how a fresh quick-brush rack
    /// is filled (§18.1.8; `slots::seed_defaults`).
    ///
    /// Deliberately not a live binding: the rack holds brush *snapshots*, not
    /// references back to here, so editing a slot does not edit the preset and
    /// vice versa. This says only where the app puts a tool on the keyboard the
    /// first time somebody sees one, which is why only the app's own presets
    /// declare it and everything the user saves has `None`.
    pub slot: Option<usize>,
    /// Whether this is one of the app's own presets rather than the user's.
    ///
    /// Never stored: it is *provenance*, set where the entry is made — true in
    /// [`default_presets`], false where a [`StoredPreset`] becomes one — so it cannot
    /// disagree with where the entry actually came from. What it decides is the row's
    /// last control (a lock, not a trash) and whether storage ever sees the entry.
    pub builtin: bool,
}

/// The presets the app ships, rebuilt on every start. The first entry is also the
/// brush the app opens on ([`apply_first`]), so it leads with the everyday one.
///
/// A preset that reaches for a shape bundled with the app names it through
/// `stamp`, which resolves the name to the content id the engine imported the
/// bytes under (`crate::builtins`) — the only way to write one down, since a
/// [`BrushShape::Stamp`] *is* a content id. One that has not arrived degrades to
/// the round tip **for this session only**, since nothing here is persisted; the
/// next start resolves it properly rather than inheriting a bad one.
///
/// The `slot` each declares is what fills a fresh rack, and it is independent of
/// the order here: the eraser is last in the list and first on the keyboard,
/// because a list is read top-down and a rack is reached by the digit under the
/// finger.
fn default_presets(state: AppState) -> Vec<PresetEntry> {
    shipped_presets(BuiltinShapes::for_app(state).unwrap_or_default())
}

#[derive(Default)]
struct BuiltinShapes {
    bristles: BrushShape,
    pencil: BrushShape,
}

impl BuiltinShapes {
    pub fn for_app(state: AppState) -> Option<Self> {
        Some(BuiltinShapes {
            bristles: builtins::shape(state, builtins::BRISTLES)?,
            pencil: builtins::shape(state, builtins::PENCIL)?,
        })
    }
}

/// [`default_presets`] with its one dependency on the running engine passed in:
/// the pencil's stamp, which is a content id and so cannot be written down until
/// its bytes have been imported (`crate::builtins`).
///
/// Split out because that is *all* the app's presets need from a live canvas, and
/// saying so makes the list testable — which matters more than it looks. The
/// invariants below (every entry shipped, every slot distinct and on the rack)
/// are the kind that a sixth preset breaks silently: the rack would quietly drop
/// or overwrite one, and nothing on screen would say which.
fn shipped_presets(shapes: BuiltinShapes) -> Vec<PresetEntry> {
    // One constructor for the lot, so "everything the app ships is a built-in and
    // sits on a digit" is a property of the list rather than five copies of two
    // fields that could drift apart. `smoothing` is the §6.11 amount, part of
    // what a tool *is*: the inker leans on the string, the pencil keeps every
    // tremor because tremor is what a pencil is for.
    let shipped =
        |name: &str, slot: Option<usize>, smoothing: f32, brush: BrushParams| PresetEntry {
            name: name.to_string(),
            brush: Wearable {
                params: brush,
                smoothing,
            },
            slot,
            builtin: true,
        };
    vec![
        shipped(
            "Hard Round",
            Some(1),
            0.15,
            BrushParams {
                size: 100.0,
                drain: 0.1,
                shape: BrushShape::Round { hardness: 0.98 },
                dynamics: BrushDynamics {
                    flow: 3.0,
                    lift: 0.25,
                    deposit: 0.75,
                    bleed: 0.25,
                    ..BrushDynamics::default()
                },
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                modulation: Modulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    flow: Some(Modulation::linear(ModSource::Pressure)),
                    ..Modulations::default()
                },
                ..BrushParams::default()
            },
        ),
        shipped(
            "Bristles",
            None,
            0.15,
            BrushParams {
                size: 100.0,
                shape: shapes.bristles,
                dynamics: BrushDynamics {
                    flow: 3.0,
                    lift: 0.25,
                    deposit: 0.75,
                    bleed: 0.5,
                    ..BrushDynamics::default()
                },
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                modulation: Modulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    flow: Some(Modulation::linear(ModSource::Pressure)),
                    ..Modulations::default()
                },
                ..BrushParams::default()
            },
        ),
        // An inking pen: hard, opaque, never runs dry, and pointed at both ends —
        // a short entry taper and a longer exit, which is the asymmetry a hand
        // makes when it lands a line and flicks off it.
        shipped(
            "Pen",
            Some(2),
            0.5,
            BrushParams {
                size: 18.0,
                shape: BrushShape::Round { hardness: 0.95 },
                start_taper_length: 5.0,
                end_taper_length: 11.0,
                dynamics: BrushDynamics {
                    flow: 0.45,
                    ..BrushDynamics::default()
                },
                ..BrushParams::default()
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
            BrushParams {
                size: 30.0,
                shape: shapes.pencil,
                jitter: 0.1,
                tooth_give: 1.0,
                tooth_softness: 0.5,
                orientation: OrientationSource::Pen,
                stretch: 0.75,
                modulation: Modulations {
                    stretch: Some(Modulation {
                        source: ModSource::Tilt,
                        floor: 0.0,
                        curve: -0.5,
                    }),
                    flow: Some(Modulation::linear(ModSource::Pressure)),
                    tooth_give: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.5,
                        curve: 0.0,
                    }),
                    ..Modulations::default()
                },
                dynamics: BrushDynamics {
                    flow: 0.4,
                    ..BrushDynamics::default()
                },
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::White,
                    frequency: [0.5, 0.0],
                    amplitude: [0.01, 0.0, 0.0],
                },
                ..BrushParams::default()
            },
        ),
        shipped(
            "Airbrush",
            Some(4),
            0.1,
            BrushParams {
                size: 500.0,
                shape: BrushShape::Round { hardness: 0.5 },
                modulation: Modulations {
                    size: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.6,
                        curve: 0.0,
                    }),
                    flow: Some(Modulation::linear(ModSource::Pressure)),
                    ..Modulations::default()
                },
                dynamics: BrushDynamics {
                    flow: 0.1,
                    ..BrushDynamics::default()
                },
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                ..BrushParams::default()
            },
        ),
        shipped(
            "Blender",
            Some(5),
            0.1,
            BrushParams {
                size: 100.0,
                tooth_give: 0.5,
                shape: BrushShape::Round { hardness: 0.8 },
                dynamics: BrushDynamics {
                    flow: 0.0,
                    lift: 0.25,
                    deposit: 0.75,
                    bleed: 0.25,
                    ..BrushDynamics::default()
                },
                modulation: Modulations {
                    lift: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.8,
                        curve: 0.0,
                    }),
                    bleed: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.0,
                        curve: 0.0,
                    }),
                    ..Modulations::default()
                },
                ..BrushParams::default()
            },
        ),
        // The eraser the pen's other end starts life holding (§18.1.8). An
        // eraser is `erase` up (§6.12) — it removes what the eye sees, so a
        // half-pressure pass really is a lighter erase — with Flow as its rate,
        // which is where the pressure mapping points: light touch feathers the
        // coverage in, borne down it walks to the full bite.
        shipped(
            "Soft Eraser",
            Some(slots::ERASER),
            0.0,
            BrushParams {
                size: 80.0,
                shape: BrushShape::Round { hardness: 0.25 },
                erase: 1.0,
                dynamics: BrushDynamics {
                    flow: 1.0,
                    ..BrushDynamics::default()
                },
                modulation: Modulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    flow: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.0,
                        curve: 1.0,
                    }),
                    ..Modulations::default()
                },
                ..BrushParams::default()
            },
        ),
        shipped(
            "Hard Eraser",
            None,
            0.0,
            BrushParams {
                size: 40.0,
                shape: BrushShape::Round { hardness: 0.95 },
                erase: 1.0,
                dynamics: BrushDynamics {
                    // Enough that one pass saturates the bite to the tip's very
                    // shoulder — the hard edge the name promises.
                    flow: 2.0,
                    ..BrushDynamics::default()
                },
                modulation: Modulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    flow: Some(Modulation {
                        source: ModSource::Pressure,
                        floor: 0.25,
                        curve: 0.0,
                    }),
                    ..Modulations::default()
                },
                ..BrushParams::default()
            },
        ),
    ]
}

/// Populate the library signal with the **user's** presets from storage. Called
/// once at app start, before the renderer exists, so what this browser saved is
/// on screen right away.
///
/// The app's own presets are not here: they name bundled shapes by content id,
/// and an id is not knowable until those bytes have been imported. They join the
/// list in [`install_builtins`] once the canvas is up.
pub fn load(state: AppState) {
    let mut entries = state.presets;
    if let Some(list) = read_storage() {
        entries.set(list);
    }
}

/// Put the app's own presets at the head of the library. Called once the canvas
/// is up and the bundled brush shapes are in it, which is later than [`load`] on
/// purpose — see there.
///
/// Every start, not only a first one, and that is the point of the whole
/// arrangement: the definitions in [`default_presets`] are the only copy, so an
/// improved default reaches a browser that has been running Stark for months
/// rather than only a browser that has never run it. While the engine's own
/// parameters are still moving, a default that cannot be updated is a default
/// that is wrong.
///
/// A stored preset whose name collides with one of the app's is **dropped**. That is
/// how a browser holding persisted copies of the built-ins sheds them — and how one
/// that saved over a built-in loses that edit, which is the same act read from the
/// other side. Storage is rewritten without them, once, and only if something actually
/// went, so an ordinary start writes nothing.
pub fn install_builtins(state: AppState) {
    let mut list = default_presets(state);
    let mut entries = state.presets;
    let stored: Vec<PresetEntry> = entries.peek().iter().cloned().collect();
    let kept: Vec<PresetEntry> = stored
        .iter()
        .filter(|u| !list.iter().any(|d| d.name == u.name))
        .cloned()
        .collect();
    let dropped = kept.len() != stored.len();
    list.extend(kept);
    entries.set(list);
    if dropped {
        persist(&entries.read());
    }
}

/// Whether `name` belongs to one of the app's own presets — which the Save
/// dialog asks before offering to write it, since a name taken by a built-in is
/// not free to overwrite ([`save_current`] refuses it) and a second row under
/// the same name would make "the preset called X" ambiguous.
pub fn is_builtin(entries: &[PresetEntry], name: &str) -> bool {
    entries.iter().any(|e| e.builtin && e.name == name)
}

/// Make `name`'s preset the live brush — the painting color stays, everything
/// else comes from the preset ([`wear`]).
pub fn apply(state: AppState, name: &str) {
    let entry = state
        .presets
        .read()
        .iter()
        .find(|e| e.name == name)
        .cloned();
    let Some(entry) = entry else { return };
    // The one act in the app that means *the artist chose a different tool from the
    // library*, and the command it is about to make cannot say so — a quick slot
    // emits the same one (§18.1.8, §24.2). Reported after the lookup, so a row that
    // is not there counts as nothing.
    crate::tutor::did(state, crate::tutor::Deed::AppliedPreset);
    // And it is the same act a held number is listening for (§18.1.8): a tool
    // chosen from the library counts as the hold's change even where it moves
    // nothing, which is exactly the case of filling a slot with the brush already
    // in hand. Said here rather than in [`wear`], which the hold uses itself.
    slots::claim(state);
    wear(state, entry.brush);
}

/// Put `brush` on: make it the live brush, **keeping the painting color** and
/// resolving its stamp.
///
/// The rule this module's docs state, as a function, because it is not only the
/// preset library's any more — the quick-brush rack (`crate::slots`) swaps
/// brushes in and out through it too, in both directions. Stated once, so "a
/// tool is everything but the color" cannot come to mean two things: a slot
/// that changed the color under the hand on the way in, or handed back the old
/// one on the way out, would make the color a property of which key was last
/// pressed.
///
/// The RGB kept is the *live* one, so it survives every swap; the brush's own
/// opacity (`color[3]`, a material property — §6.1) rides along with the tool,
/// as does everything else.
///
/// A stamp shape whose bytes are no longer anywhere (removed from the shape
/// library, unseen by this document) falls back to the round tip rather than
/// pointing at an asset the engine would silently substitute.
pub fn wear(state: AppState, wearable: Wearable) {
    // A whole tool arriving is not an adjustment of the one you had — not even in
    // the case where it differs in nothing but its size, which the tour would
    // otherwise read as somebody reaching for the size slider (§24.2). This is the
    // one door every swap comes through, in both directions, which is what makes it
    // the place to say so.
    crate::tutor::not_reaching(state, true);
    let Wearable {
        params: mut brush,
        smoothing,
    } = wearable;
    brush.shape = match brush.shape {
        BrushShape::Stamp(id) => crate::shapes::ensure(state, id)
            .map(BrushShape::Stamp)
            .unwrap_or_default(),
        round @ BrushShape::Round { .. } => round,
    };
    // The feel rides the same swap as the parameters (§6.11): it is part of
    // what the tool *is*, so a slot or a preset that changed it hands it over
    // and a release hands it back.
    let mut amount = state.smoothing;
    amount.set(smoothing.clamp(0.0, 1.0));
    update_brush(state, move |b| {
        let rgb = [b.color[0], b.color[1], b.color[2]];
        *b = brush;
        b.color[..3].copy_from_slice(&rgb);
    });
    crate::tutor::not_reaching(state, false);
}

/// Put the app on its startup brush: the first preset in the library, or — for a
/// browser whose library is empty — nothing at all, leaving the engine's own
/// default brush in place. Called once the renderer is up, unlike [`load`]: the
/// library is frontend state, but applying one is a command, and there is no
/// engine to take one before then.
pub fn apply_first(state: AppState) {
    let first = state.presets.peek().first().map(|e| e.name.clone());
    if let Some(name) = first {
        apply(state, &name);
    }
}

/// Snapshot the live brush under `name` and persist. Saving under a name one of
/// the *user's* presets already has overwrites that preset in place — updating,
/// not duplicating, which keeps names unique and rows stable.
///
/// A name one of the **app's** presets holds is refused outright, and the dialog
/// says so before the button is reachable ([`is_builtin`]). Overwriting is not
/// on offer — the next start would rebuild the built-in from source and the work
/// would be gone — and adding a second row under the same name would make "the
/// preset called Pen" two different brushes to every lookup in this module.
/// Refusing here as well as in the dialog because this is the function that
/// would have to be right if a second caller ever appeared.
pub fn save_current(state: AppState, name: String) {
    let brush = worn(state);
    let Some(brush) = brush else { return };
    // Its own statement, not inline in the `if`: a `peek` guard in a condition
    // stays borrowed through the body, and the body of the next one writes the
    // very signal being read.
    let refused = is_builtin(&state.presets.peek(), &name);
    if refused {
        return;
    }
    let mut entries = state.presets;
    {
        let mut list = entries.write();
        match list.iter_mut().find(|e| e.name == name) {
            Some(e) => e.brush = brush,
            None => list.push(PresetEntry {
                name,
                brush,
                // The user's own: no home on the rack, and storage's to keep.
                slot: None,
                builtin: false,
            }),
        }
    }
    persist(&entries.read());
}

/// Drop one of the user's presets from the library. The live brush is untouched
/// — it stops matching a library entry, nothing more.
///
/// The app's own presets survive by name, which is belt and braces: their rows
/// wear a lock rather than a trash, so nothing offers this. It costs one term to
/// say it here too, and the alternative is a list that could be left showing a
/// built-in the next start would silently bring back.
pub fn remove(state: AppState, name: &str) {
    let mut entries = state.presets;
    entries.write().retain(|e| e.builtin || e.name != name);
    persist(&entries.read());
}

/// The first free "Preset N" name — the default for a save with the name field
/// left empty.
pub fn next_name(entries: &[PresetEntry]) -> String {
    (1..)
        .map(|i| format!("Preset {i}"))
        .find(|n| !entries.iter().any(|e| &e.name == n))
        .unwrap()
}

/// The library's name for `brush`, where some preset still *is* it — what the
/// quick-brush rack labels a slot by (`slots::SlotOverlay`), since a slot holds
/// a snapshot rather than a reference back to the preset it came from.
///
/// [`matches`] is the test, so this answers on exactly the terms the preset rows
/// light on: a slot tuned away from what it was given gets no name back, which is
/// the truth — it is not that preset any more.
pub fn name_of(entries: &[PresetEntry], brush: &Wearable) -> Option<String> {
    entries
        .iter()
        .find(|e| matches(brush, &e.brush))
        .map(|e| e.name.clone())
}

/// Whether the live brush *is* this preset — everything but the painting color
/// (RGB), which [`apply`] deliberately leaves alone. Exact equality on purpose:
/// the row highlights until any knob moves off the preset, then goes out. The
/// smoothing amount is one of the knobs (§6.11): a preset worn and then
/// smoothed differently is no longer that preset.
pub fn matches(current: &Wearable, preset: &Wearable) -> bool {
    let mut p = *preset;
    p.params.color[..3].copy_from_slice(&current.params.color[..3]);
    p == *current
}

// --- persistence ----------------------------------------------------------
//
// One entry per **user** preset, through `crate::storage`, which is where the format
// and the skip-a-damaged-entry rule live. What is this module's own is which entries
// are written and which two fields are not:
//
// The app's own presets are filtered out on the way in and out, which is what makes
// them updatable at all — there is one copy of a built-in, in `default_presets`, so
// there is nothing stale for a new version to disagree with. Neither of their two
// extra fields is stored: `slot` belongs to the definition and `builtin` is
// provenance, and a stored copy of either would be a second opinion about a question
// the code already answers.

/// One stored preset: a name and a whole brush ([`Wearable`], which the rack stores
/// too).
///
/// A type of its own rather than [`PresetEntry`] with two fields skipped, because
/// provenance is the point: everything storage holds is the user's by definition, so
/// `builtin: false` is settled where the entry is *made* rather than carried in the
/// record and trusted.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredPreset {
    name: String,
    brush: Wearable,
}

impl storage::Entry for StoredPreset {
    const STORE: Store = Store::Presets;
}

fn persist(entries: &[PresetEntry]) {
    let stored: Vec<StoredPreset> = entries
        .iter()
        .filter(|e| !e.builtin)
        .map(|e| StoredPreset {
            name: e.name.clone(),
            brush: e.brush,
        })
        .collect();
    storage::save_list(&stored);
}

/// What this browser has saved, or `None` where it has saved nothing (or
/// storage is unavailable). Either way the app's own presets arrive separately,
/// so there is no "never seeded" state left to tell apart from an empty one.
fn read_storage() -> Option<Vec<PresetEntry>> {
    Some(
        storage::load_list::<StoredPreset>()?
            .into_iter()
            .map(|e| PresetEntry {
                name: e.name,
                brush: e.brush,
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
    fn shipped() -> Vec<PresetEntry> {
        shipped_presets(BuiltinShapes::default())
    }

    #[test]
    fn everything_the_app_ships_says_so() {
        // The flag decides two things at once — the row's lock, and whether
        // storage ever sees the entry — so an entry that forgot it would be
        // deletable *and* would persist, and the next start would then have two
        // presets of the same name.
        assert!(shipped().iter().all(|e| e.builtin));
    }

    #[test]
    fn every_shipped_preset_has_a_distinct_home_on_the_rack() {
        // `slots::seed_defaults` writes each preset to its own digit, so a
        // repeat would silently mean "whichever came last" and a slot past the
        // end would be dropped — neither of them visible anywhere.
        let mut seen = Vec::new();
        for entry in shipped() {
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
        let mut names: Vec<String> = shipped().into_iter().map(|e| e.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
    }

    #[test]
    fn the_shipped_eraser_erases() {
        // What makes it an eraser at all (§6.12) — there is no eraser *tool* to
        // check for, only these two numbers: the dial that routes the stroke
        // through the erase pass, and the flow that is that pass's one rate.
        let e = shipped()
            .into_iter()
            .find(|e| e.slot == Some(slots::ERASER))
            .expect("an eraser ships on the pen's own slot");
        assert!(
            e.brush.params.erase > 0.0,
            "the pen's tail must erase, not paint"
        );
        assert!(
            e.brush.params.dynamics.flow > 0.0,
            "flow is the erase pass's rate; at zero the eraser would do nothing"
        );
    }

    #[test]
    fn the_startup_brush_is_not_the_eraser() {
        // `apply_first` opens the app on the library's first entry, and the list
        // order is free to differ from the rack's — so this is the one place the
        // two orders have to be checked against each other.
        let first = shipped().into_iter().next().expect("the app ships presets");
        assert_ne!(first.slot, Some(slots::ERASER));
    }
}
