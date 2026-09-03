//! The brush preset library: named [`BrushConfig`] snapshots, shown as a
//! section at the foot of the Brush panel.
//!
//! A preset is a whole brush **except the painting color**: applying one keeps
//! the current RGB (color belongs to the Color panel — [`wear`] writes the
//! hand's back over whatever the stored tune carries) while everything else —
//! including the effect's own opacity (§6.2) — comes from the preset.
//! Both halves of the brush, that is: the durable half, which is
//! what the tool *is* and which nothing but a preset stores, and the transient
//! half — the size and flow it was saved at (`brush_config::Transient`). The
//! quick-brush rack (`crate::slots`) holds no brush of its own: a slot names a
//! preset here and keeps a transient half beside the name, so the tool on a
//! number is whatever the preset is *now* (§18.1.8). That is what makes an
//! overwrite of a preset reach every number bound to it — and what
//! [`same_tool`] is for, the test that sets the transient half aside.
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
//! - **The user's own** — everything the brush editor's two save buttons put
//!   there ("Overwrite preset", "Save new preset"), which is all `localStorage`
//!   holds.
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
    BrushModulations, BrushShape, ColorDynamics, ModSource, Modulation, NoiseKind,
    OrientationSource, ToothParams,
};

use crate::builtins;
use crate::slots;
use crate::state::{AppState, update_brush};
use crate::storage::{self, Store};
use stark_chrome::brush_config::{
    BrushConfig, BrushEffectType, EraseConfig, LiquifyConfig, Transient, WetDynamics,
};

/// The live brush, snapshotted: both halves — the tool ([`BrushConfig`]) and
/// the tune it is being worked at (`Transient`) — copies of the two signals,
/// with nothing to assemble.
pub fn worn(state: AppState) -> (BrushConfig, Transient) {
    (*state.brush.peek(), *state.transient.peek())
}

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
                shape: BrushShape::Round { hardness: 0.35 },
                effect: BrushEffectType::Liquify,
                liquify: LiquifyConfig {
                    strength_modulation: Some(Modulation::linear(ModSource::Pressure)),
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
    wear(state, entry.brush, entry.transient, Some(entry.name));
}

/// Put `brush` on at `tune`: make the pair the live brush, **keeping the
/// painting color** and resolving the stamp — and record which preset it came
/// `from`, which is what
/// [`Signals::preset_in_hand`](crate::state::Signals::preset_in_hand) takes.
///
/// The rule this module's docs state, as a function, because it is not only the
/// preset library's any more — the quick-brush rack (`crate::slots`) swaps
/// brushes in and out through it too, in both directions. Stated once, so "a
/// tool is everything but the color" cannot come to mean two things: a slot
/// that changed the color under the hand on the way in, or handed back the old
/// one on the way out, would make the color a property of which key was last
/// pressed.
///
/// The RGB kept is the *live* one, so it survives every swap; the effect's own
/// opacity (`BrushEffect::opacity`, part of what the tool does — §6.2) rides
/// along with the tool, as does everything else.
///
/// `from` is the caller's to say rather than looked up here, because every
/// caller knows it better than a lookup by snapshot could: a preset row clicked
/// ([`apply`]) has an exact name even where two entries hold the same brush, a
/// quick slot names its preset outright (`slots::QuickBrush`), and a hold ending
/// (`slots::release`) puts back a brush that may have been edited away from its
/// preset and must not forget which one that was.
///
/// A stamp shape whose bytes are no longer anywhere (removed from the shape
/// library, unseen by this document) falls back to the round tip rather than
/// pointing at an asset the engine would silently substitute.
pub fn wear(state: AppState, brush: BrushConfig, tune: Transient, from: Option<String>) {
    // A whole tool arriving is not an adjustment of the one you had — not even in
    // the case where it differs in nothing but its size, which the tour would
    // otherwise read as somebody reaching for the size slider (§24.2). This is the
    // one door every swap comes through, in both directions, which is what makes it
    // the place to say so.
    crate::tutor::not_reaching(state, true);
    let mut brush = brush;
    brush.smoothing = brush.smoothing.clamp(0.0, 1.0);
    brush.shape = match brush.shape {
        BrushShape::Stamp(id) => crate::shapes::ensure(state, id)
            .map(BrushShape::Stamp)
            .unwrap_or_default(),
        round @ BrushShape::Round { .. } => round,
    };
    update_brush(state, move |b, t| {
        // The hand's color survives the swap — a tool is everything but it
        // (§18.1.8), and this door is where that rule lives now that the
        // color rides the transient: the whole configuration moves as one,
        // the feel (§6.11), the inactive effect and the tune included, and
        // the one field written back is the color the hand already held.
        let rgb = t.color;
        *b = brush;
        *t = tune;
        t.color = rgb;
    });
    crate::tutor::not_reaching(state, false);
    let mut in_hand = state.preset_in_hand;
    in_hand.set(from);
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
    let (brush, transient) = worn(state);
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
            Some(e) => {
                e.brush = brush;
                e.transient = transient;
            }
            None => list.push(PresetEntry {
                name: name.clone(),
                brush,
                transient,
                // The user's own: no home on the rack, and storage's to keep.
                slot: None,
                builtin: false,
            }),
        }
    }
    persist(&entries.read());
    // The brush in hand now *is* this preset, whichever it was taken from.
    let mut in_hand = state.preset_in_hand;
    in_hand.set(Some(name));
}

/// Drop one of the user's presets from the library. The live brush is untouched
/// — it stops matching a library entry, nothing more. Every quick slot bound to
/// the preset is emptied with it (`slots::unbind`): a slot is a name and a tune,
/// and a name the library no longer answers to holds nothing.
///
/// The app's own presets survive by name, which is belt and braces: their rows
/// wear a lock rather than a trash, so nothing offers this. It costs one term to
/// say it here too, and the alternative is a list that could be left showing a
/// built-in the next start would silently bring back.
pub fn remove(state: AppState, name: &str) {
    let mut entries = state.presets;
    // Its own statement, for the `peek` rule — and before anything moves, since a
    // built-in that stays must keep its slots too.
    let refused = is_builtin(&entries.peek(), name);
    if refused {
        return;
    }
    entries.write().retain(|e| e.name != name);
    persist(&entries.read());
    slots::unbind(state, name);
    // A brush taken from a preset that is gone descends from nothing the library
    // has: no name to show for it, and nothing left to overwrite. Bound before the
    // `if`: a `peek` in the condition stays borrowed through a body that writes
    // the same signal.
    let mut in_hand = state.preset_in_hand;
    let was_in_hand = in_hand.peek().as_deref() == Some(name);
    if was_in_hand {
        in_hand.set(None);
    }
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
/// painting color alone set aside ([`Transient::same_tune`]), since [`apply`]
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
/// tooltip — and [`overwrite_in_hand`] acts on that one arm. A pure rule over the
/// library, the name in hand ([`Signals::preset_in_hand`](crate::state::Signals::preset_in_hand)) and the brush, so a
/// test can reach it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Overwrite {
    /// The brush was taken from no preset the library still has.
    Nothing,
    /// The preset in hand is one of the app's own, which is not on offer: the
    /// next start rebuilds it from its definition, so the write would not last
    /// (see [`save_current`]).
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

/// Write the brush in hand back over the preset it was taken from — the brush
/// editor's "Overwrite preset". A no-op unless [`overwrite`] says
/// [`Overwrite::Ready`]: the button is dead in every other case, and nothing else
/// may do what the button would not.
pub fn overwrite_in_hand(state: AppState) {
    let brush = worn(state);
    // Its own statement: the guards are `peek`s, and `save_current` writes the
    // very signals being read.
    let verdict = overwrite(
        &state.presets.peek(),
        state.preset_in_hand.peek().as_deref(),
        &brush,
    );
    if let Overwrite::Ready(name) = verdict {
        save_current(state, name);
    }
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

/// One stored preset: a name and a whole brush — both halves, the tool and the
/// tune it was saved at.
///
/// A type of its own rather than [`PresetEntry`] with two fields skipped, because
/// provenance is the point: everything storage holds is the user's by definition, so
/// `builtin: false` is settled where the entry is *made* rather than carried in the
/// record and trusted. No `#[serde(default)]` on `transient`: a stored preset
/// that lacks its tune is a damaged entry, which the store already skips.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredPreset {
    name: String,
    brush: BrushConfig,
    transient: Transient,
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
            transient: e.transient,
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
        // through the erase pass, and the transient flow that is that pass's
        // rate now that the rate is the hand's (§6.2).
        let e = shipped()
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
        let first = shipped().into_iter().next().expect("the app ships presets");
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
        let mut entries = shipped();
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
