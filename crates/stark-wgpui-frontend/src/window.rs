//! Where the window was, kept between runs (§11.2, N1).
//!
//! The one record that is **this frontend's alone**. A browser has nothing like it —
//! a page is where the user put the tab, and the web app's `Store::Visible` keeps
//! what is *inside* the window rather than the window itself. It is a row of the
//! shared registry all the same, because the registry's job is that a key is written
//! down once, and a second enum here would be a second place for one to collide from.
//!
//! It is also the cheapest end-to-end proof that the store works: quit, reopen, and
//! the window is where it was.

use serde::{Deserialize, Serialize};
use stark_chrome::storage::{self, Record, Store};
use wgpui::{App, Bounds, WindowBounds, px, size};

/// The window's place, in logical pixels.
///
/// Four numbers and a flag rather than a `WindowBounds`: that type is wgpui's, and
/// a record's shape is a thing files are written in — it outlives the toolkit
/// version that happens to be linked, and reconciling it by name (§25.6) is what
/// lets a field be added later. `#[serde(default)]` so a record written before a
/// field existed still reads.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Placement {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    /// Restored as maximized, with the four above as the size to restore *to* — which
    /// is what `WindowBounds::Maximized` already means, so nothing is lost by keeping
    /// one record instead of two.
    maximized: bool,
}

impl Record for Placement {
    const STORE: Store = Store::Window;
}

impl Default for Placement {
    /// Only reached for a *field* a stored record predates: an absent record is
    /// `None` and takes [`opening`]'s centred default instead, which is a different
    /// answer and a better one — a first run should be centred on the display it
    /// opens on, and this type has no idea where that is.
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1280.0,
            height: 800.0,
            maximized: false,
        }
    }
}

/// The window's opening bounds: where it was last, or centred if this is a first run
/// or the store has nothing to say.
///
/// **wgpui 0.3.4 reads only the size out of this.** Its `create_window` builds the
/// winit attributes with `with_inner_size` and no `with_position`, and it treats
/// `Maximized` and `Fullscreen` as `Windowed` — so a reopened window comes back the
/// right *size* wherever the OS chose to cascade it, and a maximized one comes back
/// restored. What is written here is complete and correct; what reads it is not, and
/// the gap is one `with_position` plus a `set_maximized` in a file this workspace
/// already patches (`vendor/wgpui/VENDORING.md`).
///
/// Left stating the whole placement rather than trimmed to what works, because the
/// record is what a *file* holds: trimming it would mean every window that ever
/// stored a position losing it the day the toolkit learns to read one.
pub fn opening(cx: &mut App) -> WindowBounds {
    match storage::load::<Placement>() {
        Some(p) => {
            let bounds = Bounds {
                origin: wgpui::point(px(p.x), px(p.y)),
                size: size(px(p.width), px(p.height)),
            };
            if p.maximized {
                WindowBounds::Maximized(bounds)
            } else {
                WindowBounds::Windowed(bounds)
            }
        }
        // `Bounds::centered` needs the app to know which display it is centring on,
        // which is why this is not the `Default` above.
        None => WindowBounds::Windowed(Bounds::centered(None, size(px(1280.), px(800.)), cx)),
    }
}

/// Remember where the window is, on the way out.
///
/// [`WindowBounds`] rather than [`Window::bounds`](wgpui::Window::bounds), and the
/// difference is the whole of why: a maximized window's `bounds` is the screen it
/// fills, while this carries the size it would *restore* to. Saving the first would
/// mean unmaximizing to a window the size of the display.
///
/// Called once, when the window is closing. A run that is killed rather than closed
/// keeps its old placement, which is the same bargain every other record makes about
/// a write that never happened.
pub fn remember(bounds: WindowBounds) {
    let (rect, maximized) = match bounds {
        WindowBounds::Windowed(b) => (b, false),
        // Fullscreen restores to a window, not to fullscreen: coming back into a mode
        // with no chrome and no way out but a key is not what "where it was" should
        // mean, and the restore rect is right there either way.
        WindowBounds::Maximized(b) => (b, true),
        WindowBounds::Fullscreen(b) => (b, false),
    };
    storage::save(&Placement {
        x: f32::from(rect.origin.x),
        y: f32::from(rect.origin.y),
        width: f32::from(rect.size.width),
        height: f32::from(rect.size.height),
        maximized,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record round-trips, and a value written before a field existed reads with
    /// that field's default rather than being refused — the property `#[serde(default)]`
    /// buys and the reason every record in the registry has it (§25.6).
    #[test]
    fn a_placement_survives_a_field_it_predates() {
        let p = Placement {
            x: 12.0,
            y: 34.0,
            width: 800.0,
            height: 600.0,
            maximized: true,
        };
        let json = serde_json::to_string(&p).expect("a placement encodes");
        let back: Placement = serde_json::from_str(&json).expect("and decodes");
        assert_eq!((back.x, back.y), (12.0, 34.0));
        assert!(back.maximized);

        let older: Placement =
            serde_json::from_str(r#"{"x":1.0,"y":2.0,"width":3.0,"height":4.0}"#)
                .expect("a record written before `maximized` still reads");
        assert!(!older.maximized, "and reads as the default it was given");
    }
}
