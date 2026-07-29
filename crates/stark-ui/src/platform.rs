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

/// Route the window's `kind` events ("keydown" / "keyup") to `handler`.
///
/// The shortcuts hang off the **window** rather than off an element, so they keep
/// working whatever has focus. Bound to an element they go quiet the moment a
/// clicked button is unmounted by its own command — "Deselect", the brush editor's
/// "Done" — because the browser then falls focus back to `document.body`, which is
/// *outside* the app's tree, so nothing reaches the handler until something inside
/// takes focus again.
///
/// Registered once for the life of the page (see [`crate::input::bind_shortcuts`]),
/// so the closure is `forget`ten rather than kept around for removal — the same
/// bargain [`pick_file`] makes.
#[cfg(target_arch = "wasm32")]
pub fn on_window_key(kind: &str, handler: impl FnMut(web_sys::KeyboardEvent) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(handler);
    let _ = window.add_event_listener_with_callback(kind, cb.as_ref().unchecked_ref());
    cb.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_key(_kind: &str, _handler: impl FnMut(web_sys::KeyboardEvent) + 'static) {}

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

/// Select all the text in the element `e` was mounted on — a no-op unless it is a
/// text field.
///
/// Dioxus's `MountedData` can focus an element but has nothing to say about the
/// selection inside it, so this reaches for the DOM node directly, the same route
/// [`capture_pointer`] takes.
#[cfg(target_arch = "wasm32")]
pub fn select_all(e: &Event<MountedData>) {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    if let Some(field) = e
        .try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
    {
        field.select();
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn select_all(_e: &Event<MountedData>) {}

/// Hand `bytes` to the browser as a file download named `filename`.
///
/// A Blob behind an object URL, clicked through a synthetic `<a download>` — the
/// only way to write a file from a page without a server. The object URL is
/// revoked immediately after the click: the download has already taken a
/// reference to the blob by then, and leaving it alive pins the whole buffer (a
/// full-resolution PNG) for the life of the document.
#[cfg(target_arch = "wasm32")]
pub fn download_bytes(bytes: &[u8], filename: &str, mime: &str) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    // `Uint8Array::from` copies into the JS heap, which the Blob then owns — the
    // borrow here does not have to outlive the call.
    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::of1(&array.buffer());
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(|_| "could not build the blob".to_string())?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| "could not create an object URL".to_string())?;

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let anchor = document
        .create_element("a")
        .ok()
        .and_then(|e| e.dyn_into::<web_sys::HtmlAnchorElement>().ok())
        .ok_or("could not create the download link")?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn download_bytes(_bytes: &[u8], _filename: &str, _mime: &str) -> Result<(), String> {
    Ok(())
}

/// Ask the user for a file and hand its bytes to `on_bytes`.
///
/// A hidden `<input type=file>` clicked programmatically: a page cannot open a
/// file picker any other way, and the click must happen inside the user gesture
/// that asked for it, so this is called straight from the menu handler rather
/// than from a task.
///
/// The closures are `forget`ten rather than dropped — the input and its reader
/// outlive this call by design (the user may sit in the picker for a minute), and
/// dropping the `Closure` would invalidate the JS callback before it fires.
#[cfg(target_arch = "wasm32")]
pub fn pick_file(accept: &str, on_bytes: impl Fn(Vec<u8>) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(input) = document
        .create_element("input")
        .ok()
        .and_then(|e| e.dyn_into::<web_sys::HtmlInputElement>().ok())
    else {
        return;
    };
    input.set_type("file");
    input.set_accept(accept);

    // The handler outlives this call and is re-entered per selected file, so it is
    // shared rather than moved: `Fn` is not `Clone`, and the inner (per-file)
    // closure needs its own handle.
    let on_bytes = std::rc::Rc::new(on_bytes);
    let input_for_change = input.clone();
    let on_change = Closure::<dyn FnMut()>::new(move || {
        let Some(file) = input_for_change.files().and_then(|f| f.get(0)) else {
            return;
        };
        let Ok(reader) = web_sys::FileReader::new() else {
            return;
        };
        let reader_for_load = reader.clone();
        let on_bytes = on_bytes.clone();
        let on_load = Closure::<dyn FnMut()>::new(move || {
            if let Ok(buffer) = reader_for_load.result()
                && let Some(buffer) = buffer.dyn_ref::<js_sys::ArrayBuffer>()
            {
                on_bytes(js_sys::Uint8Array::new(buffer).to_vec());
            }
        });
        reader.set_onload(Some(on_load.as_ref().unchecked_ref()));
        on_load.forget();
        let _ = reader.read_as_array_buffer(&file);
    });
    input.set_onchange(Some(on_change.as_ref().unchecked_ref()));
    on_change.forget();
    input.click();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn pick_file(_accept: &str, _on_bytes: impl Fn(Vec<u8>) + 'static) {}
