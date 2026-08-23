//! The off-wasm half of [`crate::platform`]: one stand-in per browser call, so
//! the crate still compiles for the host.
//!
//! **These are what `cargo test` and `cargo clippy --workspace --all-targets`
//! link**, which is worth being exact about, because it is easy to read the
//! host build as coverage and it is the opposite: every one of the browser
//! calls next door is untested logic, and a test that reaches anything through
//! this module is exercising the answer below rather than the one a browser
//! gives. What the host build buys is the boundary — a browser call written into
//! a panel stops it, at the line that wrote it (see [`super`]'s doc) — and it
//! buys nothing else.
//!
//! So the answers here are chosen to be *honest about being nothing*: an empty
//! list, a `None`, a `false`, a size of one pixel. The one exception is
//! [`Canvas::surface_target`], which cannot answer honestly and says so by
//! panicking — reaching it off the web is a bug rather than a fallback.
//!
//! Undocumented by design, item for item: every one of these is the second half
//! of something documented in [`super`], and a doc comment here would be a second
//! description to keep in step with the first. What is worth saying about the
//! pair is said there — which is the argument for collecting them rather than
//! leaving each one under the doc it does not get.

use dioxus::prelude::*;

use super::*;

impl Canvas {
    pub fn laid_out_size(&self) -> (u32, u32) {
        (1, 1)
    }

    pub fn set_buffer_size(&self, _width: u32, _height: u32) {}

    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        unimplemented!("stark-ui targets the web; there is no native surface backend")
    }
}

impl KeyEvent {
    pub fn key(&self) -> Key {
        Key::Unidentified
    }

    pub fn code(&self) -> String {
        String::new()
    }

    pub fn modifiers(&self) -> Modifiers {
        Modifiers::empty()
    }

    pub fn on_text_entry(&self) -> bool {
        false
    }

    pub fn prevent_default(&self) {}
}

impl WindowEvent {
    pub fn on_text_entry(&self) -> bool {
        false
    }

    pub fn prevent_default(&self) {}
}

#[derive(Clone)]
pub struct Canvas;

pub struct KeyEvent;

pub struct WindowEvent;

pub async fn sleep_ms(_ms: i32) {}

pub fn panel_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

pub fn stack_scroll() -> Option<(f32, f32, f32)> {
    None
}

pub fn set_stack_scroll(_top: f32) {}

pub fn layer_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

pub fn guide_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

pub fn anchor_box(_selector: &str) -> Option<ElementBox> {
    None
}

pub fn viewport_height() -> f32 {
    0.0
}

pub fn on_window_key(_kind: &str, _handler: impl FnMut(KeyEvent) + 'static) {}

pub fn on_window_event(_kind: &str, _handler: impl FnMut(WindowEvent) + 'static) {}

pub fn on_window_pointer(_kind: &str, _handler: impl FnMut(RawPointer) + 'static) {}

pub fn raw_pointer(_e: &Event<PointerData>) -> Option<RawPointer> {
    None
}

pub fn event_time(_e: &Event<PointerData>) -> f64 {
    0.0
}

pub fn coalesced(_e: &Event<PointerData>) -> Option<Vec<Coalesced>> {
    None
}

pub fn on_window_blur(_handler: impl FnMut() + 'static) {}

pub fn on_animation_frame(_handler: impl FnOnce() + 'static) {}

pub fn capture_pointer(_e: &Event<PointerData>) {}

pub fn pointer_fraction(_e: &Event<PointerData>) -> Option<(f32, f32)> {
    None
}

pub fn select_all(_e: &Event<MountedData>) {}

pub fn focus(_e: &Event<MountedData>) {}

pub fn focus_stays_within(_root: Option<&Event<MountedData>>, _e: &Event<FocusData>) -> bool {
    false
}

pub fn canvas_of(_e: &Event<MountedData>) -> Option<Canvas> {
    None
}

pub fn canvas_by_id(_id: &str) -> Canvas {
    Canvas
}

pub async fn next_frame() {}

pub fn device_pixel_ratio() -> f32 {
    1.0
}

pub fn now_seconds() -> f64 {
    0.0
}

pub fn url_fragment() -> Option<String> {
    None
}

pub fn url_with_fragment(fragment: &str) -> String {
    format!("#{fragment}")
}

pub fn set_url_fragment(_fragment: Option<&str>) {}

pub fn copy_to_clipboard(_text: &str) {}

pub fn local_get(_key: &str) -> Option<String> {
    None
}

pub fn local_set(_key: &str, _value: &str) -> bool {
    false
}

pub fn local_remove(_key: &str) {}

pub async fn blob_get_many(keys: &[String]) -> Vec<Option<Vec<u8>>> {
    keys.iter().map(|_| None).collect()
}

pub async fn blob_put(_key: &str, _bytes: &[u8]) -> bool {
    false
}

pub async fn blob_delete(_key: &str) {}

pub fn download_bytes(_bytes: &[u8], _filename: &str, _mime: &str) -> Result<(), String> {
    Ok(())
}

pub fn pick_file(_accept: &str, _on_file: impl Fn(String, Vec<u8>) + 'static) {}

pub fn on_file_launch(_on_file: impl Fn(String, Vec<u8>) + 'static) {}

pub async fn normalize_shape_image(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    Ok((bytes, false))
}

pub async fn normalize_substrate_image(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    Ok(bytes)
}

pub async fn decode_image(_bytes: Vec<u8>) -> Result<(u32, u32, Vec<u8>), String> {
    Err("no image decoder off the web".to_string())
}

pub fn on_window_paste(_handler: impl Fn(Vec<u8>) + 'static) {}
