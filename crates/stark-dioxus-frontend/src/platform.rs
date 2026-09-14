//! The only module that touches the browser (§11).
//!
//! `platform/web.rs` is compiled for wasm32 only; `platform/stub.rs` answers the same
//! signatures for the host build. `tests/no_web_outside_platform.rs` catches a cfg-gated
//! browser call elsewhere, and `tests/platform_parity.rs` a stub that has drifted from
//! `web.rs`. This module reports what the browser said; policy stays with the callers.

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(not(target_arch = "wasm32"))]
mod stub;
#[cfg(not(target_arch = "wasm32"))]
pub use stub::*;

/// The fields of a pointer event `input::tail_says` reads to tell which end of a
/// stylus faces the glass (§18.1.8).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RawPointer {
    /// Whether the device is a stylus; a `bool` so a window pointer event allocates
    /// no `pointerType` string.
    pub pen: bool,
    /// The button that *changed*, per Pointer Events; −1 on a move.
    pub button: i16,
    /// The buttons still down.
    pub buttons: u16,
    /// `relatedTarget` is null. On a `pointerout` this separates a pen leaving range
    /// (or the page) from a pointer crossing into a neighbouring element, which fires
    /// the same event.
    pub entered_nothing: bool,
}

/// One report the browser folded into a delivered `pointermove`, in page px.
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
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ElementBox {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl ElementBox {
    pub fn right(&self) -> f32 {
        self.left + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    pub fn mid_x(&self) -> f32 {
        self.left + self.width * 0.5
    }

    pub fn mid_y(&self) -> f32 {
        self.top + self.height * 0.5
    }
}

/// This browser as a [`Backend`](stark_ui::storage::Backend): `localStorage` for
/// the rows, IndexedDB for the bytes. Installed once by [`crate::run`]; off wasm it
/// forwards to the stubs, so a host test gets a store that forgets.
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
    ) -> stark_ui::storage::Stored<'a, Result<Vec<stark_ui::storage::BlobRead>, String>> {
        Box::pin(blob_get_many(keys))
    }

    fn blob_put<'a>(
        &'a self,
        key: &'a str,
        bytes: &'a [u8],
    ) -> stark_ui::storage::Stored<'a, Result<(), String>> {
        Box::pin(blob_put(key, bytes))
    }

    fn blob_delete<'a>(&'a self, key: &'a str) -> stark_ui::storage::Stored<'a, ()> {
        Box::pin(blob_delete(key))
    }
}
