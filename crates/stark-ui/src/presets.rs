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
use crate::platform::{base64_decode, base64_encode};
use crate::slots;
use crate::state::{AppState, update_brush};
use crate::storage;

/// One key, namespaced like the shape library's; versioned so a future format
/// change can migrate rather than mis-parse.
const KEY_PRESETS: &str = "stark.presets.v1";

/// A brush as the frontend carries it: the engine's own parameters plus the
/// **feel** the frontend owns — today just the stroke-smoothing amount
/// (§6.11).
///
/// One type rather than a field beside a field, so a whole-brush snapshot that
/// lost its feel is unrepresentable: the preset library, the quick-brush rack
/// (`crate::slots`) and a hold's swap all traffic in this. The feel is not on
/// [`BrushParams`] because the stored path already embodies the smoothing — a
/// field there would be one that replay reads and ignores, and postcard makes
/// appending it a wire-version bump (§8).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Wearable {
    pub params: BrushParams,
    /// Stroke smoothing, 0..=1 (§6.11) — the knob, not the rope. The rope is
    /// derived at gesture start (`input::rope`), because the knob is
    /// denominated in the hand's own screen px and only a live view converts
    /// it.
    pub smoothing: f32,
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
    /// [`default_presets`], false in [`parse_entry`] — so it cannot disagree with
    /// where the entry actually came from. What it decides is the row's last
    /// control (a lock, not a trash) and whether storage ever sees the entry.
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
    shipped_presets(builtins::shape(state, builtins::FLAT_TIP).unwrap_or_default())
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
fn shipped_presets(pencil: BrushShape) -> Vec<PresetEntry> {
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
                radius: 100.0,
                drain: 0.001,
                shape: BrushShape::Round { hardness: 0.95 },
                dynamics: BrushDynamics {
                    add: 3.0,
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
        // An inking pen: hard, opaque, never runs dry, and pointed at both ends —
        // a short entry taper and a longer exit, which is the asymmetry a hand
        // makes when it lands a line and flicks off it.
        shipped(
            "Pen",
            Some(2),
            0.5,
            BrushParams {
                radius: 18.0,
                shape: BrushShape::Round { hardness: 0.95 },
                start_taper_length: 5.0,
                end_taper_length: 11.0,
                dynamics: BrushDynamics {
                    add: 0.45,
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
                // The sharpened point itself now, where the radius used to stand
                // for the *widest* mark the tilt mapping would scale back from.
                radius: 15.0,
                shape: pencil,
                // Which aims the stretch, as well as turning the stamp: the axis a
                // tip is drawn out along is the axis it faces (§6.6).
                orientation: OrientationSource::Pen,
                // Laid flat the contact patch runs four points long. Linear from a
                // zero floor, so a mouse — which reports no tilt at all — gets the
                // sharpened point and nothing else, which is the honest reading of
                // "the pen is upright" rather than a case to floor away.
                stretch: 0.75,
                modulation: Modulations {
                    stretch: Some(Modulation {
                        source: ModSource::Tilt,
                        floor: 0.0,
                        curve: -0.5,
                    }),
                    flow: Some(Modulation::linear(ModSource::Pressure)),
                    ..Modulations::default()
                },
                dynamics: BrushDynamics {
                    add: 0.4,
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
                radius: 500.0,
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
                    add: 0.1,
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
        // The eraser the pen's other end starts life holding (§18.1.8).
        shipped(
            "Soft Eraser",
            Some(slots::ERASER),
            0.0,
            BrushParams {
                radius: 80.0,
                shape: BrushShape::Round { hardness: 0.25 },
                dynamics: BrushDynamics {
                    add: 0.0,
                    lift: 1.0,
                    bleed: 0.25,
                    ..BrushDynamics::default()
                },
                modulation: Modulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    lift: Some(Modulation {
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
                radius: 40.0,
                shape: BrushShape::Round { hardness: 0.95 },
                dynamics: BrushDynamics {
                    add: 0.0,
                    lift: 1.0,
                    ..BrushDynamics::default()
                },
                modulation: Modulations {
                    size: Some(Modulation::linear(ModSource::Pressure)),
                    lift: Some(Modulation {
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
// One storage key holding one line per **user** preset:
// `b64(name)|b64(json(brush))|b64(json(feel))`. Line-oriented and
// field-delimited like the shape library, so a single damaged entry is skipped
// rather than poisoning the whole library. The brush itself is JSON rather than
// the save file's postcard because localStorage outlives app versions: JSON is
// self-describing, so a `BrushParams` field added later (with
// `#[serde(default)]`) still reads every stored preset instead of dropping the
// lot. The feel (§6.11) is its own trailing field for the same reason
// run the other way: a line saved before it exists simply has two fields, and
// absent means zero.
//
// The app's own presets are filtered out on the way in and out, which is what
// makes them updatable at all: there is one copy of a built-in, in
// `default_presets`, so there is nothing stale for a new version to disagree
// with. Neither of their two extra fields is written — `slot` belongs to the
// definition and `builtin` is provenance, and a stored copy of either would be
// a second opinion about a question the code already answers.

/// The feel half of a stored line (§6.11) — its own JSON object rather than a
/// bare number, so a future feel field reads old lines through
/// `#[serde(default)]` exactly as `BrushParams` does.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredFeel {
    #[serde(default)]
    pub(crate) smoothing: f32,
}

/// The two brush fields of a stored line, shared with the rack's persistence
/// (`crate::slots`): the same wire shape, so the two libraries cannot come to
/// disagree about what a stored brush is.
pub(crate) fn encode_wearable(w: &Wearable) -> Option<String> {
    let brush = serde_json::to_string(&w.params).ok()?;
    let feel = serde_json::to_string(&StoredFeel {
        smoothing: w.smoothing,
    })
    .ok()?;
    Some(storage::record([
        base64_encode(brush.as_bytes()).as_str(),
        base64_encode(feel.as_bytes()).as_str(),
    ]))
}

/// Read a [`Wearable`] back off a line's remaining fields. The feel field is
/// optional — a line saved before §6.11 has none, and absent means zero.
pub(crate) fn decode_wearable<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Wearable> {
    let params = serde_json::from_slice(&base64_decode(fields.next()?).ok()?).ok()?;
    let smoothing = fields
        .next()
        .and_then(|f| base64_decode(f).ok())
        .and_then(|b| serde_json::from_slice::<StoredFeel>(&b).ok())
        .map_or(0.0, |f| f.smoothing.clamp(0.0, 1.0));
    Some(Wearable { params, smoothing })
}

fn persist(entries: &[PresetEntry]) {
    storage::save_table(
        KEY_PRESETS,
        "the brush presets",
        entries.iter().filter(|e| !e.builtin).filter_map(|e| {
            Some(storage::record([
                base64_encode(e.name.as_bytes()).as_str(),
                encode_wearable(&e.brush)?.as_str(),
            ]))
        }),
    );
}

/// What this browser has saved, or `None` where it has saved nothing (or
/// storage is unavailable). Either way the app's own presets arrive separately,
/// so there is no "never seeded" state left to tell apart from an empty one.
fn read_storage() -> Option<Vec<PresetEntry>> {
    storage::load_table(KEY_PRESETS, parse_entry)
}

/// One stored line. Everything storage holds is the user's by definition — the
/// app's own are never written — so provenance is settled here rather than
/// carried in the line and trusted.
fn parse_entry(line: &str) -> Option<PresetEntry> {
    let mut fields = line.split(storage::FIELD);
    let name = String::from_utf8(base64_decode(fields.next()?).ok()?).ok()?;
    let brush = decode_wearable(&mut fields)?;
    Some(PresetEntry {
        name,
        brush,
        slot: None,
        builtin: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list as the app ships it, with the pencil's stamp standing in — the
    /// one thing a live canvas would supply, and nothing here depends on which
    /// shape it is.
    fn shipped() -> Vec<PresetEntry> {
        shipped_presets(BrushShape::default())
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
    fn the_shipped_eraser_takes_paint_and_gives_none_back() {
        // What makes it an eraser at all (§6.2) — there is no eraser *tool* to
        // check for, only these three numbers.
        let e = shipped()
            .into_iter()
            .find(|e| e.slot == Some(slots::ERASER))
            .expect("an eraser ships on the pen's own slot");
        assert_eq!(
            e.brush.params.dynamics.add, 0.0,
            "an eraser lays no paint of its own"
        );
        assert!(e.brush.params.dynamics.lift > 0.0, "an eraser lifts");
        assert_eq!(
            e.brush.params.dynamics.deposit, 0.0,
            "and never lays back what it lifted"
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
