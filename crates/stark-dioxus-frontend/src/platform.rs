//! **The only module that touches the browser** (§11).
//!
//! Split by target rather than by item. `platform/web.rs` is the browser, compiled for
//! wasm32 only; `platform/stub.rs` answers the same signatures for the host build, which
//! is what tests and clippy compile. This file holds what both share: the types that
//! carry a browser's report across without its vocabulary ([`RawPointer`],
//! [`Coalesced`], [`ElementBox`]), and [`LocalStore`].
//!
//! `web-sys`, `js-sys` and `wasm-bindgen` are wasm-only dependencies, so a browser call
//! written anywhere else fails the host build at the line that wrote it. Two tests cover
//! what that build cannot see: `tests/no_web_outside_platform.rs`, for a cfg-gated call
//! outside `web.rs`, and `tests/platform_parity.rs`, for a stub that has drifted from
//! its web twin.
//!
//! What crosses is the smallest thing its callers use, so the policy stays with them:
//! this module says what the browser reported, never what the app should do about it.

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(not(target_arch = "wasm32"))]
mod stub;
#[cfg(not(target_arch = "wasm32"))]
pub use stub::*;

/// A pointer event as `input::tail_says` reads it — which end of a stylus faces
/// the glass, and whether it is still over the page at all. Everything that
/// policy needs, and nothing it does not.
///
/// The two button fields are both here because the two halves of a press report
/// differently, and the policy that reads them is the caller's (§18.1.8).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RawPointer {
    /// Whether the device is a stylus. A `bool` rather than the DOM's
    /// `pointerType` string, because a pen is the only kind anything here asks
    /// about and a `String` would allocate on every window pointer event.
    pub pen: bool,
    /// The button that *changed*, per Pointer Events; −1 on a move.
    pub button: i16,
    /// The buttons still down.
    pub buttons: u16,
    /// Whether the report names no element being **entered** (`relatedTarget`).
    /// On a `pointerout` that is the whole difference between a pen leaving the
    /// digitizer's range — or the page — and a pointer crossing from one
    /// element to its neighbour, which fires the same event. Vacuously true on
    /// every other kind, where nothing is being left.
    pub entered_nothing: bool,
}

/// One report the browser folded into a delivered `pointermove`, in the target
/// element's own CSS pixels.
#[derive(Clone, Copy, Debug)]
pub struct Coalesced {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    /// `performance.now()`'s clock, in seconds.
    pub time: f64,
}

/// One element's box on screen, in CSS px from the viewport's top-left.
///
/// The whole rectangle, where `element_boxes` keeps only the two numbers a column
/// drag needs. The caller is the guided tour's card ([`anchor_box`]), which has to
/// put itself *beside* a thing rather than order a list of them, so it needs the
/// horizontal half as well.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ElementBox {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl ElementBox {
    /// The box's right edge — where a card placed to its right starts.
    pub fn right(&self) -> f32 {
        self.left + self.width
    }

    /// The box's bottom edge — what a card hung off a box that sits on the foot of
    /// the window is lined up with (`tutor::Side::RightAtBottom`).
    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    /// The horizontal middle of the box.
    pub fn mid_x(&self) -> f32 {
        self.left + self.width * 0.5
    }

    /// The vertical middle of the box.
    pub fn mid_y(&self) -> f32 {
        self.top + self.height * 0.5
    }
}

/// This browser as a [`Backend`](stark_ui::storage::Backend): `localStorage` for
/// the rows, IndexedDB for the bytes.
///
/// Shared rather than per target, because each method forwards to the platform call of
/// the same name. Installed once at startup ([`crate::run`]); off wasm those calls are
/// the stubs, so a host-run test gets a store that forgets — the same case as a private
/// window.
pub struct LocalStore;

impl stark_ui::storage::Backend for LocalStore {
    fn get(&self, key: &str) -> Option<String> {
        local_get(key)
    }

    fn set(&self, key: &str, value: &str) -> bool {
        local_set(key, value)
    }

    fn remove(&self, key: &str) {
        local_remove(key);
    }

    fn blob_get_many<'a>(
        &'a self,
        keys: &'a [String],
    ) -> stark_ui::storage::Stored<'a, Vec<Option<Vec<u8>>>> {
        Box::pin(blob_get_many(keys))
    }

    fn blob_put<'a>(
        &'a self,
        key: &'a str,
        bytes: &'a [u8],
    ) -> stark_ui::storage::Stored<'a, bool> {
        Box::pin(blob_put(key, bytes))
    }

    fn blob_delete<'a>(&'a self, key: &'a str) -> stark_ui::storage::Stored<'a, ()> {
        Box::pin(blob_delete(key))
    }
}
