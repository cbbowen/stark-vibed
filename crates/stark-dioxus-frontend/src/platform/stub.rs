//! The host build's half of [`crate::platform`], so the crate compiles off wasm. Items
//! are documented in `web.rs`.
//!
//! Every answer says nothing happened — an empty list, `None`, `false`, or an `Err` for
//! any act a caller would otherwise report as done, so a host test cannot mistake a stub
//! for a browser. [`Canvas::surface_target`] and [`canvas_by_id`] cannot answer at all,
//! and panic.

use dioxus::prelude::*;
use stark_ui::assets::Decoded;
use stark_ui::storage::BlobRead;

use super::{Coalesced, ElementBox, RawPointer};

#[derive(Clone)]
pub struct Canvas;

pub struct KeyEvent;

pub struct WindowEvent;

impl Canvas {
    pub fn laid_out_size(&self) -> (u32, u32) {
        (1, 1)
    }

    pub fn set_buffer_size(&self, _width: u32, _height: u32) {}

    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        unimplemented!("stark-dioxus-frontend targets the web; there is no native surface backend")
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

pub fn install_panic_hook() {}

pub fn install_tracing() {}

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

pub fn page_position(_e: &Event<PointerData>) -> Option<(f32, f32)> {
    None
}

pub fn coalesced(_e: &Event<PointerData>) -> Option<Vec<Coalesced>> {
    None
}

pub fn on_window_blur(_handler: impl FnMut() + 'static) {}

pub fn on_before_unload(_confirm: impl Fn() -> bool + 'static) {}

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
    unimplemented!("stark-dioxus-frontend targets the web; there is no document off it")
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

pub async fn blob_get_many(_keys: &[String]) -> Result<Vec<BlobRead>, String> {
    Err("no blob store off the web".to_string())
}

pub async fn blob_put(_key: &str, _bytes: &[u8]) -> Result<(), String> {
    Err("no blob store off the web".to_string())
}

pub async fn blob_delete(_key: &str) {}

pub fn download_bytes(_bytes: &[u8], _filename: &str, _mime: &str) -> Result<(), String> {
    Err("no browser to download through".to_string())
}

pub fn pick_file(_accept: &str, _on_file: impl Fn(String, Vec<u8>) + 'static) {}

pub fn on_file_launch(_on_file: impl Fn(String, Vec<u8>) + 'static) {}

pub async fn normalize_shape_image(_bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    Err("no image decoder off the web".to_string())
}

pub async fn normalize_substrate_image(_bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    Err("no image decoder off the web".to_string())
}

pub async fn decode_image(_bytes: Vec<u8>) -> Result<Decoded, String> {
    Err("no image decoder off the web".to_string())
}

pub fn on_window_paste(_handler: impl Fn(Vec<u8>) + 'static) {}
