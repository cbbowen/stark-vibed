//! The user's brush preset library: named [`BrushParams`] snapshots, shown as a
//! collapsible section at the foot of the Brush panel.
//!
//! Like the shape library (`crate::shapes`), the presets are frontend state that
//! follows this browser across documents via `localStorage` — and degrade to a
//! per-session library where storage is unavailable. A browser that has never
//! stored a library is seeded with the built-in tools that used to be hard-coded
//! chips in the brush editor: the everyday brush, and a tapered inking pen.
//! Deleting them is respected — only a browser with *no* stored library (not an
//! emptied one) is re-seeded.
//!
//! A preset is a whole brush **except the painting colour**: applying one keeps
//! the current RGB (colour belongs to the Color panel) while everything else —
//! including the brush's own opacity, `color[3]` — comes from the preset.

use dioxus::prelude::*;

use stark_core::document::{BrushDynamics, BrushParams, BrushShape, ColorDynamics, NoiseKind};

use crate::platform::{base64_decode, base64_encode};
use crate::state::{AppState, update_brush};

/// One key, namespaced like the shape library's; versioned so a future format
/// change can migrate rather than mis-parse.
const KEY_PRESETS: &str = "stark.presets.v1";

/// One named preset in the library.
#[derive(Clone, PartialEq)]
pub struct PresetEntry {
    /// Display name; unique in the library (saving under a taken name overwrites).
    pub name: String,
    /// The snapshot applied by clicking the preset.
    pub brush: BrushParams,
}

/// The library a fresh browser starts with. The first entry is also the brush the
/// app opens on ([`apply_first`]), so it leads with the everyday one.
fn default_presets() -> Vec<PresetEntry> {
    vec![
        PresetEntry {
            name: "Hard Round".to_string(),
            brush: BrushParams {
                radius: 100.0,
                shape: BrushShape::Round { hardness: 0.95 },
                dynamics: BrushDynamics {
                    add: 1.0,
                    lift: 0.6,
                    deposit: 0.95,
                    ..BrushDynamics::default()
                },
                color_dynamics: ColorDynamics {
                    noise: NoiseKind::Simplex,
                    frequency: [0.05, 0.1],
                    amplitude: [0.0, 0.025, 0.05],
                },
                ..BrushParams::default()
            },
        },
        // An inking pen: hard, opaque, never runs dry, and pointed at both ends —
        // a short entry taper and a longer exit, which is the asymmetry a hand
        // makes when it lands a line and flicks off it.
        PresetEntry {
            name: "Pen".to_string(),
            brush: BrushParams {
                radius: 18.0,
                shape: BrushShape::Round { hardness: 0.92 },
                drain: 0.0,
                start_taper_length: 5.0,
                end_taper_length: 11.0,
                dynamics: BrushDynamics {
                    add: 1.0,
                    ..BrushDynamics::default()
                },
                ..BrushParams::default()
            },
        },
    ]
}

/// Populate the library signal from storage, seeding the defaults into a
/// browser that has never stored a library. Called once at app start.
pub fn load(state: AppState) {
    let mut entries = state.presets;
    match read_storage() {
        Some(list) => entries.set(list),
        None => {
            let defaults = default_presets();
            persist(&defaults);
            entries.set(defaults);
        }
    }
}

/// Make `name`'s preset the live brush. The painting colour (RGB) stays — a
/// preset is a tool, not a colour choice — while the preset's own opacity
/// (`color[3]`, the editor's Opacity slider) rides along. A stamp shape whose
/// bytes are no longer anywhere (removed from the shape library, unseen by this
/// document) falls back to the round tip rather than pointing at an asset the
/// engine would silently substitute.
pub fn apply(state: AppState, name: &str) {
    let entry = state
        .presets
        .read()
        .iter()
        .find(|e| e.name == name)
        .cloned();
    let Some(entry) = entry else { return };
    let mut brush = entry.brush;
    brush.shape = match brush.shape {
        BrushShape::Stamp(id) => crate::shapes::ensure(state, id)
            .map(BrushShape::Stamp)
            .unwrap_or_default(),
        round @ BrushShape::Round { .. } => round,
    };
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

/// Snapshot the live brush under `name` and persist. Saving under a name the
/// library already has overwrites that preset in place — updating, not
/// duplicating, which keeps names unique and rows stable.
pub fn save_current(state: AppState, name: String) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    let Some(brush) = brush else { return };
    let mut entries = state.presets;
    {
        let mut list = entries.write();
        match list.iter_mut().find(|e| e.name == name) {
            Some(e) => e.brush = brush,
            None => list.push(PresetEntry { name, brush }),
        }
    }
    persist(&entries.read());
}

/// Drop a preset from the library. The live brush is untouched — it stops
/// matching a library entry, nothing more.
pub fn remove(state: AppState, name: &str) {
    let mut entries = state.presets;
    entries.write().retain(|e| e.name != name);
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

/// Whether the live brush *is* this preset — everything but the painting colour
/// (RGB), which [`apply`] deliberately leaves alone. Exact equality on purpose:
/// the row highlights until any knob moves off the preset, then goes out.
pub fn matches(current: &BrushParams, preset: &BrushParams) -> bool {
    let mut p = *preset;
    p.color[..3].copy_from_slice(&current.color[..3]);
    p == *current
}

// --- persistence ----------------------------------------------------------
//
// One storage key holding one line per preset: `b64(name)|b64(json(brush))`.
// Line-oriented and field-delimited like the shape library, so a single
// damaged entry is skipped rather than poisoning the whole library. The brush
// itself is JSON rather than the save file's postcard because localStorage
// outlives app versions: JSON is self-describing, so a `BrushParams` field
// added later (with `#[serde(default)]`) still reads every stored preset
// instead of dropping the lot.

fn persist(entries: &[PresetEntry]) {
    let Some(store) = storage() else { return };
    let text: Vec<String> = entries
        .iter()
        .filter_map(|e| {
            let json = serde_json::to_string(&e.brush).ok()?;
            Some(format!(
                "{}|{}",
                base64_encode(e.name.as_bytes()),
                base64_encode(json.as_bytes())
            ))
        })
        .collect();
    if store.set_item(KEY_PRESETS, &text.join("\n")).is_err() {
        // Quota, most likely. The library still works for this session.
        tracing::warn!("could not persist the brush presets (storage full or unavailable)");
    }
}

/// `None` when this browser has never stored a library (or storage is
/// unavailable) — the caller seeds the defaults. An emptied library comes back
/// as `Some(vec![])`, a state to respect rather than re-seed.
fn read_storage() -> Option<Vec<PresetEntry>> {
    let text = storage()?.get_item(KEY_PRESETS).ok()??;
    Some(text.lines().filter_map(parse_entry).collect())
}

fn parse_entry(line: &str) -> Option<PresetEntry> {
    let mut fields = line.split('|');
    let name = String::from_utf8(base64_decode(fields.next()?).ok()?).ok()?;
    let brush = serde_json::from_slice(&base64_decode(fields.next()?).ok()?).ok()?;
    Some(PresetEntry { name, brush })
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}
