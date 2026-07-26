//! The two places the frontend touches the browser directly, each with an off-wasm
//! no-op so the crate still compiles for the host (which is what `cargo test` and
//! clippy exercise — see `render::canvas_target`).

use dioxus::prelude::*;

/// Resolve after `ms` milliseconds (so a settle animation can finish before the order is
/// committed). Browser `setTimeout` on web; a no-op off-wasm.
#[cfg(target_arch = "wasm32")]
pub async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let _ = web_sys::window()
            .expect("window")
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn sleep_ms(_ms: i32) {}

/// Capture the pointer for the element under `e`, so the in-progress drag keeps
/// streaming move/up events to it while the button is held — even after the pointer
/// leaves the element. The capture releases automatically on pointer-up, which is
/// guaranteed to be delivered to the capturing element.
#[cfg(target_arch = "wasm32")]
pub fn capture_pointer(e: &Event<PointerData>) {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    if let Some(ev) = e.try_as_web_event()
        && let Some(target) = ev
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
    {
        let _ = target.set_pointer_capture(ev.pointer_id());
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn capture_pointer(_e: &Event<PointerData>) {}
