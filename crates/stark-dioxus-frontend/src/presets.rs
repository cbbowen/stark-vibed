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
//! [`same_tool`](stark_chrome::presets::same_tool) is for, the test that sets the transient half aside.
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

use stark_model::document::BrushShape;

use crate::builtins;
use crate::slots;
use crate::state::{AppState, update_brush};
use stark_chrome::brush_config::{BrushConfig, Transient};
use stark_chrome::presets::{
    self, BuiltinShapes, Overwrite, PresetEntry, is_builtin, overwrite, persist, read_storage,
};

/// The live brush, snapshotted: both halves — the tool ([`BrushConfig`]) and
/// the tune it is being worked at (`Transient`) — copies of the two signals,
/// with nothing to assemble.
pub fn worn(state: AppState) -> (BrushConfig, Transient) {
    (*state.brush.peek(), *state.transient.peek())
}

/// The app's shipped presets, resolved against *this* canvas — the one thing
/// `stark_chrome::presets::shipped` cannot do for itself.
///
/// A preset reaching for a bundled shape names it by content id, and an id is the
/// hash of the bytes, so it is not knowable until they have been imported
/// (`crate::builtins`). A shape that has not arrived leaves the round tip in its
/// place **for this session only**, since nothing here is persisted; the next start
/// resolves it properly rather than inheriting a bad one.
fn default_presets(state: AppState) -> Vec<PresetEntry> {
    presets::shipped(builtins_for(state).unwrap_or_default())
}

fn builtins_for(state: AppState) -> Option<BuiltinShapes> {
    Some(BuiltinShapes {
        bristles: builtins::shape(state, builtins::BRISTLES)?,
        pencil: builtins::shape(state, builtins::PENCIL)?,
    })
}

/// Read this browser's saved presets into the library. Called at app start, before
/// the canvas exists, so the library is on screen right away.
///
/// The app's own are not here: they name bundled shapes by content id, which is not
/// knowable until those bytes have been imported. They join in [`install_builtins`]
/// once the canvas is up.
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
