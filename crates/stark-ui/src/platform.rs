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

/// Every element matching `selector`: the identity it wears in `attr`, and its
/// `(top, height)` in client px. Empty off-wasm, and empty before the elements have
/// mounted.
///
/// In DOM order, which a caller must **not** read as the order they appear in — the
/// panel stack's children are a fixed sequence ordered by a flex `order`
/// ([`crate::layout::PanelStack`]). Hence the attribute: the identity travels *with*
/// the box, off the same element, so a caller matches on identity rather than on
/// position. Matched by position an element would be measured through its
/// neighbour's box in silence, because a box is a plausible box whichever element it
/// came from (§11).
///
/// The two callers are the two drags that reorder a column by dropping a thing into
/// it: the panel stack ([`panel_boxes`]) and the layer tree ([`layer_boxes`]). Both
/// measure once at grab time and derive everything after from the live pointer, so
/// there is no cached geometry to fall out of date.
#[cfg(target_arch = "wasm32")]
fn element_boxes(selector: &str, attr: &str) -> Vec<(String, f32, f32)> {
    use wasm_bindgen::JsCast;
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return Vec::new();
    };
    let Ok(nodes) = doc.query_selector_all(selector) else {
        return Vec::new();
    };
    (0..nodes.length())
        .filter_map(|i| nodes.item(i))
        .filter_map(|n| n.dyn_into::<web_sys::Element>().ok())
        .map(|el| {
            let r = el.get_bounding_client_rect();
            (
                el.get_attribute(attr).unwrap_or_default(),
                r.top() as f32,
                r.height() as f32,
            )
        })
        .collect()
}

/// The panel stack's panels, each under its `data-panel` id — see [`element_boxes`].
#[cfg(target_arch = "wasm32")]
pub fn panel_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".panel-stack > .panel", "data-panel")
}
#[cfg(not(target_arch = "wasm32"))]
pub fn panel_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

/// The layer panel's rows, each under its `data-layer` id — see [`element_boxes`].
///
/// The whole entry is measured (`.layer-item`, indent included) rather than the row
/// inside it, because that is the box a drag opens a slot the size of: the margin
/// between two entries is the gap the drop lands in.
#[cfg(target_arch = "wasm32")]
pub fn layer_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".layer-item[data-layer]", "data-layer")
}
#[cfg(not(target_arch = "wasm32"))]
pub fn layer_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

/// The guides panel's rows, each under its `data-guide` position — see
/// [`element_boxes`]. A *position* rather than an id, because a guide has none; it
/// holds for the length of one gesture, which is all it is asked to.
#[cfg(target_arch = "wasm32")]
pub fn guide_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".guide-row[data-guide]", "data-guide")
}
#[cfg(not(target_arch = "wasm32"))]
pub fn guide_boxes() -> Vec<(String, f32, f32)> {
    Vec::new()
}

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

/// Route the window's `kind` events to `handler` in the **capture** phase, as the
/// base [`web_sys::Event`].
///
/// Untyped on purpose, unlike its two neighbours: `contextmenu` is a `MouseEvent`
/// in some engines and a `PointerEvent` in the ones that have adopted the newer
/// spec, and everything a refusal needs — `target` and `prevent_default` — is on
/// the base interface. Casting to the wrong subclass to reach nothing extra would
/// be a way to fail on one browser for no gain.
///
/// Capture for [`on_window_pointer`]'s reason: the browser decides what to do
/// once the event has finished propagating, so a listener the tree could silence
/// with `stopPropagation` is one that stops working the day something downstream
/// does.
#[cfg(target_arch = "wasm32")]
pub fn on_window_event(kind: &str, handler: impl FnMut(web_sys::Event) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::Event)>::new(handler);
    let _ = window.add_event_listener_with_callback_and_bool(
        kind,
        cb.as_ref().unchecked_ref(),
        true, // useCapture
    );
    cb.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_event(_kind: &str, _handler: impl FnMut(web_sys::Event) + 'static) {}

/// Route the window's `kind` pointer events ("pointerdown", "pointerup", …) to
/// `handler`, in the **capture** phase.
///
/// The pointer counterpart to [`on_window_key`], and it exists for the same
/// reason: what it binds is not any one surface's business. The pen's eraser end
/// holds a brush slot for as long as it is down (§18.1.8) whatever it is pressed
/// against — the canvas, a slider, a preset row — exactly as a held number key
/// does, and a listener per surface would be a list nobody could keep complete.
///
/// **Capture, not bubble**, and that is load-bearing on the press: the swap has
/// to be in force before the surface's own handler runs, or the canvas would open
/// its stroke on the brush the eraser displaced. Capture runs window-inward, so
/// this is ahead of every handler in the tree; bubble would be behind all of them.
/// It also cannot be silenced — `stopPropagation` downstream is too late to
/// prevent something that has already run, which matters most for the release,
/// where a listener that could be skipped would leave the brush swapped.
#[cfg(target_arch = "wasm32")]
pub fn on_window_pointer(kind: &str, handler: impl FnMut(web_sys::PointerEvent) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(handler);
    let _ = window.add_event_listener_with_callback_and_bool(
        kind,
        cb.as_ref().unchecked_ref(),
        true, // useCapture
    );
    cb.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_pointer(_kind: &str, _handler: impl FnMut(web_sys::PointerEvent) + 'static) {}

/// Call `handler` whenever the window loses focus.
///
/// The counterpart to [`on_window_key`], and it exists because keyup is not the
/// only way a held key ends: focus leaving the window — Alt+Tab, a click into the
/// browser's own chrome — takes the keyboard away and the release is delivered
/// somewhere else entirely. Anything armed on a keydown therefore needs a second
/// way to be disarmed, or it stays armed for the rest of the session
/// (`crate::slots`).
///
/// `blur` on the window rather than `visibilitychange` on the document: it is
/// focus that decides where a keyup goes, and a window can be fully visible with
/// the keyboard somewhere else.
///
/// Registered once for the life of the page, so the closure is `forget`ten like
/// the key handlers'.
#[cfg(target_arch = "wasm32")]
pub fn on_window_blur(mut handler: impl FnMut() + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut()>::new(move || handler());
    let _ = window.add_event_listener_with_callback("blur", cb.as_ref().unchecked_ref());
    cb.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_blur(_handler: impl FnMut() + 'static) {}

/// Run `handler` **inside** the next animation-frame callback.
///
/// This is not the same as a task awaiting a promise the rAF resolves
/// ([`crate::render::next_frame`]): a woken task only resumes in the microtask
/// drain, behind the scheduler's other work — for a dioxus task that includes a
/// VDOM render of whatever scopes are dirty by then. The callback itself runs in
/// the frame's animation phase, ahead of the browser's rendering steps by
/// definition, so work that must land in the frame that woke it belongs here.
///
/// One-shot: the closure frees itself after the call, so a registration per
/// frame leaks nothing (unlike the `forget`ten window listeners above, which
/// are bound once for the life of the page).
#[cfg(target_arch = "wasm32")]
pub fn on_animation_frame(handler: impl FnOnce() + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    // The callback's DOMHighResTimeStamp argument is dropped: callers time
    // nothing off it today, and the engine keeps its own clock.
    let cb = Closure::once_into_js(move |_: f64| handler());
    let _ = window.request_animation_frame(cb.unchecked_ref());
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_animation_frame(_handler: impl FnOnce() + 'static) {}

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

/// The `<canvas>` element a mount event fired on, for binding a WebGPU surface to it
/// — the navigator's miniature (`panels::navigator`).
///
/// Off the event rather than by id through the document, unlike
/// [`crate::render::canvas_element`]: this element's whole significance is that it
/// has *just* appeared and a surface has to be bound to this one rather than to
/// whichever element an id happens to resolve to. A panel that closes and reopens
/// mounts a new canvas, and binding the new surface to the old element would leave
/// the miniature blank with nothing to say why.
#[cfg(target_arch = "wasm32")]
pub fn canvas_of(e: &Event<MountedData>) -> Option<web_sys::HtmlCanvasElement> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    e.try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn canvas_of(_e: &Event<MountedData>) -> Option<web_sys::HtmlCanvasElement> {
    None
}

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

/// Ask the user for a file and hand its name and bytes to `on_file`.
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
pub fn pick_file(accept: &str, on_file: impl Fn(String, Vec<u8>) + 'static) {
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
    let on_file = std::rc::Rc::new(on_file);
    let input_for_change = input.clone();
    let on_change = Closure::<dyn FnMut()>::new(move || {
        let Some(file) = input_for_change.files().and_then(|f| f.get(0)) else {
            return;
        };
        let Ok(reader) = web_sys::FileReader::new() else {
            return;
        };
        let name = file.name();
        let reader_for_load = reader.clone();
        let on_file = on_file.clone();
        let on_load = Closure::<dyn FnMut()>::new(move || {
            if let Ok(buffer) = reader_for_load.result()
                && let Some(buffer) = buffer.dyn_ref::<js_sys::ArrayBuffer>()
            {
                on_file(name.clone(), js_sys::Uint8Array::new(buffer).to_vec());
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
pub fn pick_file(_accept: &str, _on_file: impl Fn(String, Vec<u8>) + 'static) {}

/// Hand `on_file` whatever file the OS launched the app with — the other end of
/// the manifest's `file_handlers` (§11, [`crate::files::bind_file_launch`]).
///
/// Reflection rather than typed bindings, unlike everything else in this module:
/// neither `launchQueue` nor the `FileSystemFileHandle` it yields is in `web-sys`
/// (the handle is, but behind the `web_sys_unstable_apis` cfg, which is a
/// `RUSTFLAGS` change for the whole build). A browser without the API leaves the
/// lookup undefined and this returns having promised nothing, which is the same
/// shape as a browser that simply never launches with a file.
///
/// Setting the consumer is what *delivers* a launch: the browser queues the
/// params from before the page had any say, so this must not be called until the
/// handler can act on them. The closure is `forget`ten for the same reason
/// [`pick_file`]'s are — it outlives this call by design, and may fire more than
/// once (`focus-existing`).
#[cfg(target_arch = "wasm32")]
pub fn on_file_launch(on_file: impl Fn(String, Vec<u8>) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::closure::Closure;

    /// `obj.name` as a callable, or `None` if it is missing or not one.
    fn method(obj: &JsValue, name: &str) -> Option<js_sys::Function> {
        js_sys::Reflect::get(obj, &JsValue::from_str(name))
            .ok()?
            .dyn_into()
            .ok()
    }

    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(queue) = js_sys::Reflect::get(&window, &JsValue::from_str("launchQueue")) else {
        return;
    };
    let Some(set_consumer) = method(&queue, "setConsumer") else {
        return;
    };

    // Shared, not moved: the consumer may be re-entered per launch, and each of
    // those spawns a task that needs its own handle.
    let on_file = std::rc::Rc::new(on_file);
    let consumer = Closure::<dyn FnMut(JsValue)>::new(move |params: JsValue| {
        let files = js_sys::Reflect::get(&params, &JsValue::from_str("files"))
            .unwrap_or(JsValue::UNDEFINED);
        let Ok(files) = files.dyn_into::<js_sys::Array>() else {
            return;
        };
        // The first only. Opening a document *replaces* the canvas (§8), so a
        // second file would be a painting nobody ever sees — which is why the
        // manifest asks for `single-client` rather than a window per file.
        let Some(handle) = files.iter().next() else {
            return;
        };
        let on_file = on_file.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let Some(get_file) = method(&handle, "getFile") else {
                return;
            };
            let Ok(promise) = get_file.call0(&handle) else {
                return tracing::error!("could not read the launched file");
            };
            let Ok(promise) = promise.dyn_into::<js_sys::Promise>() else {
                return;
            };
            let Ok(file) = wasm_bindgen_futures::JsFuture::from(promise).await else {
                // Permission for the handle is the usual reason: a launch grants
                // read access, but a stale handle replayed later may not have it.
                return tracing::error!("the launched file could not be opened");
            };
            let Ok(file) = file.dyn_into::<web_sys::File>() else {
                return;
            };
            let name = file.name();
            let Ok(buffer) = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await else {
                return tracing::error!(name, "the launched file could not be read");
            };
            let Some(buffer) = buffer.dyn_ref::<js_sys::ArrayBuffer>() else {
                return;
            };
            on_file(name, js_sys::Uint8Array::new(buffer).to_vec());
        });
    });
    let _ = set_consumer.call1(&queue, consumer.as_ref().unchecked_ref());
    consumer.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_file_launch(_on_file: impl Fn(String, Vec<u8>) + 'static) {}

/// Normalize an image into a brush-shape PNG, using the browser as the decoder —
/// any format the browser can display can be imported (JPEG, WebP, GIF, …).
///
/// Two normalizations beyond transcoding:
/// - **Downscale** so the longest edge is at most the engine's shape cap
///   (1024 px, `stark_core::assets::MAX_SHAPE_DIM`). The engine would cap it
///   anyway; doing it here keeps library entries small in `localStorage`.
/// - **Dark-on-light inversion.** The engine reads coverage as
///   `luminance × alpha` (white paints, black doesn't) — but scanned or drawn
///   brush images are usually dark ink on light paper, which would import as a
///   solid rectangle with a shape-shaped hole. If the image's border ring is
///   mostly covered, it's read as ink-on-paper and the luminance is inverted.
///
/// Returns the PNG bytes and whether the inversion fired (so the UI can say so).
#[cfg(target_arch = "wasm32")]
pub async fn normalize_shape_image(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("no window")?;
    let array = js_sys::Uint8Array::from(bytes.as_slice());
    let parts = js_sys::Array::of1(&array.buffer());
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)
        .map_err(|_| "could not wrap the image bytes".to_string())?;
    let promise = window
        .create_image_bitmap_with_blob(&blob)
        .map_err(|_| "image decoding unavailable".to_string())?;
    let bitmap: web_sys::ImageBitmap = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "not an image the browser can decode".to_string())?
        .dyn_into()
        .map_err(|_| "unexpected decode result".to_string())?;

    let (sw, sh) = (bitmap.width(), bitmap.height());
    if sw == 0 || sh == 0 {
        return Err("the image is empty".to_string());
    }
    let cap = stark_core::assets::MAX_SHAPE_DIM;
    let scale = (cap as f64 / sw.max(sh) as f64).min(1.0);
    let w = ((sw as f64 * scale) as u32).max(1);
    let h = ((sh as f64 * scale) as u32).max(1);

    let document = window.document().ok_or("no document")?;
    let canvas: web_sys::HtmlCanvasElement = document
        .create_element("canvas")
        .ok()
        .and_then(|e| e.dyn_into().ok())
        .ok_or("could not create a canvas")?;
    canvas.set_width(w);
    canvas.set_height(h);
    let ctx: web_sys::CanvasRenderingContext2d = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into().ok())
        .ok_or("no 2d context")?;
    ctx.draw_image_with_image_bitmap_and_dw_and_dh(&bitmap, 0.0, 0.0, w as f64, h as f64)
        .map_err(|_| "could not draw the image".to_string())?;

    let data = ctx
        .get_image_data(0.0, 0.0, w as f64, h as f64)
        .map_err(|_| "could not read the pixels".to_string())?;
    let mut px = data.data().0;

    // Mirror of the engine's coverage read (assets.rs::decode_coverage):
    // coverage = luminance × alpha, in [0, 255].
    let coverage = |px: &[u8], i: usize| -> u32 {
        let lum = (77 * px[i] as u32 + 150 * px[i + 1] as u32 + 29 * px[i + 2] as u32) >> 8;
        lum * px[i + 3] as u32 / 255
    };
    let (mut border_sum, mut border_n) = (0u64, 0u64);
    let mut ring = |x: u32, y: u32, px: &[u8]| {
        border_sum += coverage(px, ((y * w + x) * 4) as usize) as u64;
        border_n += 1;
    };
    for x in 0..w {
        ring(x, 0, &px);
        ring(x, h - 1, &px);
    }
    for y in 1..h.saturating_sub(1) {
        ring(0, y, &px);
        ring(w - 1, y, &px);
    }
    let inverted = border_sum / border_n.max(1) > 127;
    if inverted {
        for p in px.as_chunks_mut::<4>().0 {
            let lum = ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8;
            let inv = 255 - lum;
            (p[0], p[1], p[2]) = (inv, inv, inv);
        }
        let data =
            web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&px), w, h)
                .map_err(|_| "could not rebuild the pixels".to_string())?;
        ctx.put_image_data(&data, 0.0, 0.0)
            .map_err(|_| "could not write the pixels".to_string())?;
    }

    let url = canvas
        .to_data_url_with_type("image/png")
        .map_err(|_| "could not encode the PNG".to_string())?;
    let b64 = url
        .strip_prefix("data:image/png;base64,")
        .ok_or("unexpected data URL")?;
    Ok((base64_decode(b64)?, inverted))
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn normalize_shape_image(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    Ok((bytes, false))
}

/// Standard base64 (with padding) — small, so data URLs and `localStorage`
/// blobs stay dependency-free.
pub fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (chunk[0] as u32) << 16
            | (*chunk.get(1).unwrap_or(&0) as u32) << 8
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Standard-alphabet base64 → bytes, the inverse of [`base64_encode`].
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        lookup[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = lookup[c as usize];
        if v == 255 {
            return Err("invalid base64".to_string());
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}
