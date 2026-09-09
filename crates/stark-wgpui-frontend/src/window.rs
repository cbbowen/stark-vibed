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
use stark_ui::storage::{self, Record, Store};
use wgpui::{App, Bounds, WindowBounds, px, size};

/// The window's place, in logical pixels.
///
/// Four numbers and a flag rather than a `WindowBounds`: that type is wgpui's, and
/// a record's shape is a thing files are written in — it outlives the toolkit
/// version that happens to be linked, and reconciling it by name (§25.6) is what
/// lets a field be added later. `#[serde(default)]` so a record written before a
/// field existed still reads.
///
/// The four numbers go through `storage::finite` for the reason `Prefs` does: JSON
/// cannot spell a non-finite number, so one written as `null` would make the *whole*
/// record unreadable and lose the flag beside it. What arrives instead is the NaN it
/// was written from, and [`usable`](Self::usable) is where that stops — a window has
/// nowhere to be put at a NaN, so this record's repair is to decline to answer rather
/// than to answer with a number of its own.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Placement {
    #[serde(with = "storage::finite")]
    x: f32,
    #[serde(with = "storage::finite")]
    y: f32,
    #[serde(with = "storage::finite")]
    width: f32,
    #[serde(with = "storage::finite")]
    height: f32,
    /// Restored as maximized, with the four above as the size to restore *to* — which
    /// is what `WindowBounds::Maximized` already means, so nothing is lost by keeping
    /// one record instead of two.
    maximized: bool,
}

impl Placement {
    /// Whether this is a rectangle a window can actually be put at.
    ///
    /// The funnel [`Hdr::clamped_headroom`](stark_ui::prefs::Hdr::clamped_headroom) is
    /// for the settings: the one place that knows what these numbers are *for*, and so
    /// the one that can say what a value outside them means. Here the answer is
    /// nothing — a stored rectangle is the only thing this record has to say, and a
    /// degenerate one is worse than not having said it, since [`opening`]'s centred
    /// fallback is a good answer and a zero-width window is not.
    fn usable(&self) -> bool {
        [self.x, self.y, self.width, self.height]
            .iter()
            .all(|v| v.is_finite())
            && self.width > 0.0
            && self.height > 0.0
    }
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
/// Honoured whole — position, size and maximized state — over a vendored patch:
/// upstream wgpui 0.3.4 read only the size out of this and treated `Maximized` as
/// `Windowed` (`vendor/wgpui/VENDORING.md`, patch 3). What it still does not do is
/// *report* a restore rect, which is [`remember`]'s problem rather than this one's.
pub fn opening(cx: &mut App) -> WindowBounds {
    // A stored rectangle a window cannot be put at is the same answer as none — the
    // centred fallback below is a good one (`Placement::usable`).
    match storage::load::<Placement>().filter(Placement::usable) {
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
    // **A maximized window keeps the rect it already had.** `WindowBounds::Maximized`
    // documents its bounds as the size to restore *to*, but wgpui fills all three
    // variants from the window's current frame — so what arrives here for a maximized
    // window is the screen it fills. Storing that would mean un-maximizing, next run,
    // to a window the size of the display. What the restore rect actually is, is the
    // one already on file: the last size this window was *not* maximized at.
    let previous = storage::load::<Placement>();
    let (rect, maximized) = match bounds {
        WindowBounds::Windowed(b) => (Some(b), false),
        WindowBounds::Maximized(_) => (None, true),
        // Fullscreen comes back as a window: returning into a mode with no chrome and
        // no way out but a key is not what "where it was" should mean. Its frame is
        // the display's too, so it keeps the stored rect for the same reason.
        WindowBounds::Fullscreen(_) => (None, false),
    };
    let previous = previous.filter(Placement::usable).unwrap_or_default();
    storage::save(&Placement {
        x: rect.map_or(previous.x, |b| f32::from(b.origin.x)),
        y: rect.map_or(previous.y, |b| f32::from(b.origin.y)),
        width: rect.map_or(previous.width, |b| f32::from(b.size.width)),
        height: rect.map_or(previous.height, |b| f32::from(b.size.height)),
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

    /// The same property stated once for every field rather than for the one that was
    /// added last — the gate the other records go through in the web frontend's
    /// `records`, which cannot see this one because a page has no window to remember.
    #[test]
    fn every_field_of_a_placement_may_be_absent() {
        stark_ui::storage::every_field_may_be_absent(&Placement::default(), &[]);
    }

    /// A rectangle JSON cannot spell **costs no other field**, and then declines to be
    /// a window.
    ///
    /// `serde_json` writes a non-finite float as `null`, and a `null` that would not
    /// read back as an `f32` takes the whole record with it — so a window that once
    /// reported a NaN frame would lose its `maximized` flag too, on every launch
    /// after. What arrives instead is the NaN, and [`Placement::usable`] is where it
    /// stops: `opening` centres, which is the right answer for "the store has nothing
    /// to say".
    #[test]
    fn a_rectangle_no_window_can_sit_at_costs_no_other_field() {
        let broken = Placement {
            x: f32::NAN,
            ..Placement::default()
        };
        let json = serde_json::to_string(&broken).expect("a placement encodes");
        assert!(
            json.contains("\"x\":null"),
            "the write is what it always was: {json}",
        );
        let back: Placement = serde_json::from_str(&json).expect("and the record still reads");
        assert!(back.x.is_nan());
        assert!(!back.usable(), "but it is not somewhere to put a window");
        assert!(
            !Placement {
                width: 0.0,
                ..Placement::default()
            }
            .usable(),
            "nor is a window with no width",
        );
        assert!(Placement::default().usable());
    }
}
