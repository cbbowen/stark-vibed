//! What the **window** hears: the keyboard, the pen's other end, and the
//! browser's context menu.
//!
//! Bound to the window rather than to any element, and each for its own reason.
//! The shortcuts answer whatever has focus, including `document.body`, where the
//! browser leaves it after a clicked button unmounts itself. The pen's eraser end
//! is a hold like a number key's and so belongs to no one surface (§18.1.8). And
//! the context menu is refused outright, because a pen held still is this app's
//! own gesture (§6.9) and Windows reads the same hold as a right-click.
//!
//! **Holds live here, not in the chord table.** Anything that owns both edges of
//! its key — space's pan, the digit rack, Alt's eyedropper — is this module's,
//! because the table has no vocabulary for a release (`crate::commands`, §25.3).

use super::*;

/// Bind the app's keyboard shortcuts, once, for the life of the page.
///
/// On the window rather than on the app's root element — see
/// [`platform::on_window_key`] for why an element cannot hold them.
///
/// Only the **keydown** side is withheld from a field being typed into. Keyup is
/// what disarms `space_down` and corrects `held_mods`, and focus can move between a
/// press and its release — a click into the rename field with space held — so a
/// guarded keyup would leave the pan armed with nothing to release it. Nothing is
/// given up by letting it through: on keyup there is no default action left to
/// cancel, since a character is inserted on the press.
pub fn bind_shortcuts(state: AppState) {
    on_window_key("keydown", move |e| {
        if !e.on_text_entry() {
            handle_keydown(state, &e);
        }
    });
    on_window_key("keyup", move |e| handle_keyup(state, &e));
    // The one event that takes a key away without ever sending its keyup: focus
    // leaving the window. A number held across an Alt+Tab would otherwise hold
    // its brush for the rest of the session, with the key that ends it now
    // belonging to another window (`slots::release_all`).
    on_window_blur(move || slots::release_all(state));
}

/// Bind the pen's eraser end to its brush slot, once, for the life of the page
/// (§18.1.8).
///
/// The pointer half of [`bind_shortcuts`], and deliberately shaped like it: the
/// tail of the stylus is a **hold**, exactly as a number key is, so it is bound
/// once at the window rather than being armed by whichever surface happens to be
/// pressed. That is what lets it reach past the canvas — dragging Size or Flow
/// with the eraser tunes *the eraser*, and eraser-clicking a preset assigns it to
/// the eraser — for the same reason holding `3` while dragging Size tunes slot 3.
/// Armed by each surface instead, it would work on the surfaces somebody
/// remembered, and the list of the ones they did not is the kind nobody keeps
/// complete.
///
/// The two tests are deliberately **not** the same one:
///
/// - The **press** has to really be the eraser ([`is_eraser_event`]), or the tip
///   would arm the eraser's slot and every ordinary stroke would erase.
/// - The **release** is any pen leaving the glass. A stylus has one contact, so a
///   tip release cannot coexist with the tail being down, and a driver that
///   reports the release without the eraser bit still ends the hold — where the
///   stricter test would leave the brush swapped with nothing left to swap it
///   back. [`slots::release`] is a no-op unless an eraser hold is in flight, so
///   asking too often costs nothing where asking too rarely costs the session.
///
/// A *finger's* release is left alone on purpose: a palm settling on the glass
/// mid-erase would otherwise hand the brush back under a pen that never moved.
pub fn bind_pen(state: AppState) {
    on_window_pointer("pointerdown", move |e| {
        if is_eraser_event(&e) {
            slots::hold(state, slots::ERASER, Grip::Eraser);
        }
    });

    // Both edges, because a cancel is a release the browser made on your behalf —
    // a gesture the system took over, a tab switched away from mid-stroke.
    for kind in ["pointerup", "pointercancel"] {
        on_window_pointer(kind, move |e| {
            if e.pen {
                slots::release(state, slots::ERASER, Grip::Eraser);
            }
        });
    }
}

/// Refuse the browser's context menu, once, for the life of the page.
///
///
/// A pen held still is a **gesture** here, not a request for a menu: the drawing
/// assist snaps a stroke to the shape it resembles after 0.45s of dwell (§6.9,
/// [`DWELL`]), which is inside the half-second Windows spends deciding that a
/// held stylus meant a right-click. So the menu arrives on top of the assist,
/// over the canvas, mid-stroke. The same hold ends the same way on a slider
/// being dragged, a preset row, a layer being reordered, a transform handle —
/// every drag long enough to be deliberate.
///
/// Bound at the window rather than per surface, [`bind_pen`]'s argument exactly:
/// the surfaces where this is unwanted are all of them, and a handler per surface
/// would work on the ones somebody remembered. The right button is a tool in the
/// navigator's miniature and means nothing anywhere else, so there is no reading
/// of a press this takes away.
///
/// The one exception is a text field, where the browser's menu is the only cut,
/// copy and paste the app offers — the same carve-out the shortcuts make for the
/// same reason ([`platform::KeyEvent::on_text_entry`]).
pub fn bind_context_menu() {
    on_window_event("contextmenu", |e| {
        if !e.on_text_entry() {
            e.prevent_default();
        }
    });
}

/// Whether a key event went to a control that owns its own keystrokes — a text
/// field, a `<select>`, a contenteditable region — is
/// [`platform::KeyEvent::on_text_entry`], asked of the DOM at the moment of the
/// keystroke so it cannot fall out of step with focus.
///
/// Declining a keystroke there is what hands the field the browser's own editing
/// bindings: Ctrl+Z undoes the *text* rather than the document, and Ctrl+A selects
/// the text rather than the canvas, purely because nothing calls `prevent_default`
/// on them.
///
/// That is the *only* way a widget can opt out. `e.stop_propagation()` in an
/// element's own `onkeydown` will not do it: dioxus-web reads `prevent_default`
/// off a handled event but never calls `stopPropagation` on the underlying DOM
/// event, so propagation is halted inside the virtual tree only and the real event
/// reaches the window regardless.
fn handle_keydown(state: AppState, e: &platform::KeyEvent) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            let mut space_down = state.space_down;
            space_down.set(true);
            // Space arms the pan: a hover mark left standing would promise
            // paint the press will not make (§18.1.10). Self-guarding, so the
            // key's auto-repeat costs a peek and nothing else.
            clear_hover_mark(state);
            e.prevent_default();
        }
        // Alt on its own focuses the browser's menu bar on Windows and Linux, which
        // would take the keyboard away the moment the eyedropper is reached for.
        Key::Alt => e.prevent_default(),
        _ => {}
    }

    let m = e.modifiers();
    track_mods(state, m);
    // The quick-brush rack, claimed before the chord table is consulted so a
    // future row on a digit could never shadow it. A digit is not a row there:
    // it is a *hold*, owning both edges of its key (§18.1.8); it reads the
    // physical row so a layout that types `&é"'` on it still has a rack; and
    // Shift is deliberately tolerated — on most layouts it is what the digit
    // row types under, and a hand resting on it should not silently disarm the
    // rack — where the table's chords are exact. `slots::hold` ignores a press
    // while a hold is in flight, which is what makes the key's own auto-repeat
    // harmless, and it is what counts a digit pressed twice in a beat
    // (`slots::Taps`), so nothing here keeps time. Alt is not tolerated: bare
    // Alt is the eyedropper's, and only a bare digit is ours.
    if !accel(m)
        && !m.contains(Modifiers::ALT)
        && let Some(slot) = slots::of_code(&e.code())
    {
        slots::hold(state, slot, Grip::Key);
        e.prevent_default();
        return;
    }
    // Everything else a keydown may simply *mean* is a chord row in the command
    // registry (`crate::commands`), and the claim on a matched chord is uniform:
    // `prevent_default` whether or not the act was accepted, because the
    // browser's own Ctrl+A would select the page's text, and a refusal that
    // let that through would answer a declined command with a highlighted
    // user interface.
    if let Some(command) = commands::find(state, e) {
        command.run(state);
        e.prevent_default();
    }
}

fn handle_keyup(state: AppState, e: &platform::KeyEvent) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            let mut space_down = state.space_down;
            space_down.set(false);
            e.prevent_default();
        }
        _ => {}
    }
    // The rack's release, named by the slot it lets go of — so a hand rolling
    // from 3 to 4 and off 4 first does not end the hold 3 still has (§18.1.8).
    // Unguarded by `KeyEvent::on_text_entry` like the two above, and for the same
    // reason: focus can move between a press and its release, and a release that
    // never arrived would leave the brush swapped.
    if let Some(slot) = slots::of_code(&e.code()) {
        slots::release(state, slot, Grip::Key);
    }
    track_mods(state, e.modifiers());
}

/// Record which modifiers are held, so the resting cursor can say what a press
/// would do — the drag table's advertisement half (`drags::armed`, §18.0.2).
///
/// Read off the event's **modifier set** rather than off the keys themselves: a
/// keystroke that arrives after a modifier was pressed or released while the
/// window was not focused then corrects the triple, instead of leaving it stuck on
/// a press whose release never came. Written only on a change, since every write
/// re-renders the canvas component.
fn track_mods(state: AppState, m: Modifiers) {
    let now = drags::Mods::of(m);
    let mut held = state.held_mods;
    if *held.peek() != now {
        held.set(now);
        // These modifiers arm an act that reads the *shown* canvas back —
        // the eyedropper's sample (`Engine::pick_colors`), the layer carry's
        // hit test (`Engine::pick_layer`) — so the hover mark has to leave the
        // screen with the same keystroke, or a press could read the hypothesis
        // back as paint: the wrong color for one and the wrong layer for the
        // other (§18.1.10). Asked of the table rather than named here, so a
        // rebinding moves it (§25.5).
        if drags::armed(&state.drags.peek(), now).is_some_and(drags::DragAction::shadows_paint) {
            clear_hover_mark(state);
        }
    }
}
