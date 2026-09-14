//! The browser's half of [`crate::platform`], compiled for wasm32 only.

use dioxus::prelude::*;
use stark_ui::assets::Decoded;
use stark_ui::storage::BlobRead;

use super::{Coalesced, ElementBox, RawPointer};

/// Report a panic on the console. Without this a panic in wasm surfaces as an
/// `unreachable` trap, with its message lost.
pub fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Route `tracing` to the browser console and to the timing histograms (§7.1).
///
/// The console layer is filtered with `TimingFilter::<false>`, the exact complement of
/// the timing layer's filter: `tracing_wasm` calls `performance.mark`/`measure` per
/// span, and unfiltered it would flood the devtools timeline with every frame's spans.
pub fn install_tracing() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    let console = tracing_wasm::WASMLayer::new(tracing_wasm::WASMLayerConfig::default())
        .with_filter(stark_engine::timing::TimingFilter::<false>);
    let subscriber = Registry::default()
        .with(stark_engine::timing::layer())
        .with(console);
    // A second install would leave the first histograms still collecting.
    if tracing::subscriber::set_global_default(subscriber).is_err() {
        web_sys::console::warn_1(&"stark: a tracing subscriber was already installed".into());
    }
}

/// A `<canvas>` element, exposing only what `render` needs.
#[derive(Clone)]
pub struct Canvas(web_sys::HtmlCanvasElement);

impl Canvas {
    /// The element's (not the window's) laid-out size in CSS px, at least 1.
    pub fn laid_out_size(&self) -> (u32, u32) {
        (
            self.0.client_width().max(1) as u32,
            self.0.client_height().max(1) as u32,
        )
    }

    /// Resize the drawing buffer, which the stylesheet's layout size does not set.
    pub fn set_buffer_size(&self, width: u32, height: u32) {
        self.0.set_width(width);
        self.0.set_height(height);
    }

    /// What `wgpu` binds a surface to. There is no native backend (§11): the stub
    /// panics, because reaching it off the web is a bug rather than a fallback.
    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        wgpu::SurfaceTarget::Canvas(self.0.clone())
    }
}

/// A key press or release at the window, translated to dioxus's [`Key`] and
/// [`Modifiers`].
pub struct KeyEvent(web_sys::KeyboardEvent);

impl KeyEvent {
    pub fn key(&self) -> Key {
        use std::str::FromStr;
        Key::from_str(&self.0.key()).unwrap_or(Key::Unidentified)
    }

    /// The physical key, independent of keyboard layout (§18.1.8).
    pub fn code(&self) -> String {
        self.0.code()
    }

    pub fn modifiers(&self) -> Modifiers {
        let mut m = Modifiers::empty();
        if self.0.alt_key() {
            m.insert(Modifiers::ALT);
        }
        if self.0.ctrl_key() {
            m.insert(Modifiers::CONTROL);
        }
        if self.0.meta_key() {
            m.insert(Modifiers::META);
        }
        if self.0.shift_key() {
            m.insert(Modifiers::SHIFT);
        }
        m
    }

    /// See [`on_text_entry`].
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }

    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
}

/// A window event the app only refuses, such as the context menu.
pub struct WindowEvent(web_sys::Event);

impl WindowEvent {
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }

    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
}

/// Resolve after `ms` milliseconds, via `setTimeout`.
pub async fn sleep_ms(ms: i32) {
    called_back(|window, wake| {
        window.set_timeout_with_callback_and_timeout_and_arguments_0(wake, ms)
    })
    .await;
}

/// Await the browser calling the function `schedule` hands it.
///
/// If scheduling fails the reason is logged and the await never finishes: resolving at
/// once would turn every `loop { sleep_ms(..) }` into a spin that never yields.
async fn called_back(
    schedule: impl Fn(&web_sys::Window, &js_sys::Function) -> Result<i32, wasm_bindgen::JsValue>,
) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let scheduled = match web_sys::window() {
            Some(window) => schedule(&window, &resolve).map_err(|e| reason(&e)),
            None => Err("no window".to_string()),
        };
        if let Err(reason) = scheduled {
            tracing::error!(
                reason,
                "could not schedule a callback; the task awaiting it stops"
            );
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Every element matching `selector`: the identity it wears in `attr`, and its
/// `(top, height)` in client px. Empty before the elements have mounted.
///
/// In DOM order, which is **not** display order — the panel stack is ordered by flex
/// `order` ([`crate::layout::PanelStack`]). Callers must match on the identity, not
/// the position, or they silently measure a neighbour's box (§11).
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
pub fn panel_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".panel-stack > .panel", "data-panel")
}

/// The panel stack's `(scrollTop, scrollHeight, clientHeight)` in CSS px (§11), or
/// `None` before it has mounted.
pub fn stack_scroll() -> Option<(f32, f32, f32)> {
    let el = web_sys::window()?
        .document()?
        .query_selector(".panel-stack")
        .ok()
        .flatten()?;
    Some((
        el.scroll_top() as f32,
        el.scroll_height() as f32,
        el.client_height() as f32,
    ))
}

/// Scroll the panel stack to `top`. The browser clamps it, and the `scroll` event it
/// raises is what tells the rail (`layout::PanelScrollbar`) where it ended up.
pub fn set_stack_scroll(top: f32) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.query_selector(".panel-stack").ok().flatten())
    {
        el.set_scroll_top(top.max(0.0).round() as i32);
    }
}

/// The layer panel's entries, each under its `data-layer` id — see [`element_boxes`].
///
/// The whole `.layer-item` is measured rather than the row inside it: the margin
/// between two entries is the gap a drop lands in.
pub fn layer_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".layer-item[data-layer]", "data-layer")
}

/// The guides panel's rows, each under its `data-guide` position — see
/// [`element_boxes`]. A guide has no id, so the position holds for one gesture only.
pub fn guide_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".guide-row[data-guide]", "data-guide")
}

/// The box of the first element matching `selector` (built by `tutor::Anchor`), or
/// `None` where nothing matches — a closed panel, which is not a failure.
pub fn anchor_box(selector: &str) -> Option<ElementBox> {
    let doc = web_sys::window().and_then(|w| w.document())?;
    let el = doc.query_selector(selector).ok().flatten()?;
    let r = el.get_bounding_client_rect();
    Some(ElementBox {
        left: r.left() as f32,
        top: r.top() as f32,
        width: r.width() as f32,
        height: r.height() as f32,
    })
}

/// The window's inner height in CSS px, `0.0` where there is no window.
///
/// Only for placements CSS cannot express because they branch, such as whether a
/// pop-out hangs down or rises (`panels::popout`).
pub fn viewport_height() -> f32 {
    web_sys::window()
        .and_then(|w| w.inner_height().ok())
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32
}

/// Route the window's `kind` events ("keydown" / "keyup") to `handler`.
///
/// On the window, not an element: when a clicked button unmounts itself, the browser
/// moves focus to `document.body`, outside the app's tree, and an element listener
/// would go quiet.
///
/// Registered once for the life of the page, so the closure is `forget`ten. The other
/// window listeners below follow the same rule.
pub fn on_window_key(kind: &str, mut handler: impl FnMut(KeyEvent) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e| handler(KeyEvent(e)));
    let _ = window.add_event_listener_with_callback(kind, cb.as_ref().unchecked_ref());
    cb.forget();
}

/// Route the window's `kind` events to `handler` in the **capture** phase, so no
/// `stopPropagation` downstream can silence it.
///
/// Untyped on purpose: `contextmenu` is a `MouseEvent` in some engines and a
/// `PointerEvent` in others, and `target` and `prevent_default` are on the base
/// interface.
pub fn on_window_event(kind: &str, mut handler: impl FnMut(WindowEvent) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::Event)>::new(move |e| handler(WindowEvent(e)));
    let _ = window.add_event_listener_with_callback_and_bool(
        kind,
        cb.as_ref().unchecked_ref(),
        true, // useCapture
    );
    cb.forget();
}

/// Route the window's `kind` pointer events ("pointerdown", "pointerup", …) to
/// `handler`, in the **capture** phase.
///
/// On the window because the eraser end holds a brush slot whatever surface it is
/// over (§18.1.8). Capture is load-bearing: the swap must be in force before any
/// handler in the tree opens a stroke, and a release no `stopPropagation` can skip
/// never leaves the brush swapped.
pub fn on_window_pointer(kind: &str, mut handler: impl FnMut(RawPointer) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e| handler(raw_of(&e)));
    let _ = window.add_event_listener_with_callback_and_bool(
        kind,
        cb.as_ref().unchecked_ref(),
        true, // useCapture
    );
    cb.forget();
}

fn raw_of(e: &web_sys::PointerEvent) -> RawPointer {
    RawPointer {
        pen: e.pointer_type() == "pen",
        button: e.button(),
        buttons: e.buttons(),
        entered_nothing: e.related_target().is_none(),
    }
}

/// The [`RawPointer`] of a dioxus pointer event. `None` where there is no raw event,
/// which callers read as "not a pen".
pub fn raw_pointer(e: &Event<PointerData>) -> Option<RawPointer> {
    use dioxus::web::WebEventExt;
    e.try_as_web_event().map(|raw| raw_of(&raw))
}

/// A pointer event's timestamp in seconds, on `performance.now()`'s monotonic clock
/// ([`InputSample::time`](stark_engine::command::InputSample::time)). Zero when the raw
/// event is out of reach; only differences matter to the fitter.
pub fn event_time(e: &Event<PointerData>) -> f64 {
    use dioxus::web::WebEventExt;
    e.try_as_web_event()
        .map(|raw| raw.time_stamp() / 1000.0)
        .unwrap_or(0.0)
}

/// A pointer event's position in page px, fraction included.
///
/// Every pointer position the canvas reads is in page px, from here or from
/// [`coalesced`], so the two agree. web-sys binds `pageX`/`pageY` as integers (the
/// double is behind `web_sys_unstable_apis`), and whole px are coarser than a fit's
/// tolerance.
pub fn page_position(e: &Event<PointerData>) -> Option<(f32, f32)> {
    use dioxus::web::WebEventExt;
    e.try_as_web_event().map(|raw| page_of(&raw))
}

/// [`page_position`] off a raw event, falling back to the integer getter.
fn page_of(e: &web_sys::MouseEvent) -> (f32, f32) {
    use wasm_bindgen::JsValue;

    thread_local! {
        static KEYS: (JsValue, JsValue) = (JsValue::from_str("pageX"), JsValue::from_str("pageY"));
    }
    let read = |key: &JsValue, whole: i32| {
        js_sys::Reflect::get(e, key)
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::from(whole)) as f32
    };
    KEYS.with(|(x, y)| (read(x, e.page_x()), read(y, e.page_y())))
}

/// Every report the browser folded into a delivered `pointermove`, oldest first.
///
/// The browser delivers about one `pointermove` per frame; the coalesced list carries
/// the pen's full rate. The delivered event equals the list's last entry, so read one
/// or the other, never both. No element's box is measured, which could force a layout
/// per move. `None` where there is no list (a synthetic event).
pub fn coalesced(e: &Event<PointerData>) -> Option<Vec<Coalesced>> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;

    let raw = e.try_as_web_event()?;
    let list = raw.get_coalesced_events();
    (list.length() > 0).then(|| {
        list.iter()
            .filter_map(|v| v.dyn_into::<web_sys::PointerEvent>().ok())
            .map(|c| {
                let (x, y) = page_of(&c);
                Coalesced {
                    x,
                    y,
                    pressure: c.pressure(),
                    tilt_x: c.tilt_x() as f32,
                    tilt_y: c.tilt_y() as f32,
                    time: c.time_stamp() / 1000.0,
                }
            })
            .collect()
    })
}

/// Whether `target` is a control that owns its own keystrokes — a text field, a
/// `<select>`, a contenteditable region.
///
/// Asked of the event's target rather than a focus flag: a field that unmounts while
/// focused never fires its blur, and a stuck flag would kill every shortcut.
fn on_text_entry(target: &web_sys::EventTarget) -> bool {
    use wasm_bindgen::JsCast;

    let Some(el) = target.dyn_ref::<web_sys::HtmlElement>() else {
        return false;
    };
    el.is_content_editable()
        || match el.tag_name().as_str() {
            "TEXTAREA" | "SELECT" => true,
            // Non-text inputs: Ctrl+Z over a slider still means the document.
            "INPUT" => !matches!(
                el.unchecked_ref::<web_sys::HtmlInputElement>()
                    .type_()
                    .as_str(),
                "button" | "checkbox" | "color" | "file" | "radio" | "range" | "reset" | "submit"
            ),
            _ => false,
        }
}

/// Call `handler` whenever the window loses focus.
///
/// When focus leaves the window (Alt+Tab, the browser's chrome) the keyup is delivered
/// elsewhere, so anything armed on a keydown must also disarm here (`crate::slots`).
/// `blur` rather than `visibilitychange`: a visible window can lack the keyboard.
pub fn on_window_blur(handler: impl FnMut() + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut()>::new(handler);
    let _ = window.add_event_listener_with_callback("blur", cb.as_ref().unchecked_ref());
    cb.forget();
}

/// Ask the browser to confirm before the page goes away (reload, closed tab, followed
/// link, back button), whenever `confirm` answers true.
///
/// The browser owns the prompt's wording and whether it shows at all; a page can only
/// object. It objects twice: `preventDefault` for the current spec, a non-empty
/// `returnValue` for older engines. Browsers skip the prompt on a page never
/// interacted with.
pub fn on_before_unload(confirm: impl Fn() -> bool + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::<dyn FnMut(web_sys::BeforeUnloadEvent)>::new(
        move |e: web_sys::BeforeUnloadEvent| {
            if !confirm() {
                return;
            }
            e.prevent_default();
            e.set_return_value("This painting has changes that are not in a file yet.");
        },
    );
    let _ = window.add_event_listener_with_callback("beforeunload", cb.as_ref().unchecked_ref());
    cb.forget();
}

/// Run `handler` **inside** the next animation-frame callback, ahead of the browser's
/// rendering steps, so its work lands in that frame.
///
/// Unlike [`next_frame`], whose woken task resumes in the microtask drain behind the
/// scheduler's other work, a dioxus VDOM render included. The closure frees itself
/// after the call, so registering one per frame leaks nothing.
pub fn on_animation_frame(handler: impl FnOnce() + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    let cb = Closure::once_into_js(move |_: f64| handler());
    let _ = window.request_animation_frame(cb.unchecked_ref());
}

/// Capture the pointer for `e`'s target, so moves and the release reach it even after
/// the pointer leaves. The capture ends by itself on pointer-up.
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

/// Where a pointer event landed as a fraction of its target's box — `(0, 0)` top-left,
/// `(1, 1)` bottom-right, unclamped. `None` for a target with no box.
///
/// Under [`capture_pointer`] the target stays the element that took the press. Fractions
/// keep a control's size the stylesheet's alone (`panels::color`).
pub fn pointer_fraction(e: &Event<PointerData>) -> Option<(f32, f32)> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    let ev = e.try_as_web_event()?;
    let rect = ev
        .target()
        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())?
        .get_bounding_client_rect();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    Some((
        ((ev.client_x() as f64 - rect.left()) / rect.width()) as f32,
        ((ev.client_y() as f64 - rect.top()) / rect.height()) as f32,
    ))
}

/// Select all the text in the element `e` was mounted on; a no-op unless it is a
/// text field. `MountedData` has no selection API.
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

/// Focus the element `e` was mounted on.
pub fn focus(e: &Event<MountedData>) {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    if let Some(el) = e
        .try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = el.focus();
    }
}

/// Whether the focus a `focusout` moved is still inside the element `root` was
/// mounted on. `focusout` bubbles from any child, including a trigger losing focus to
/// the field its dropdown just mounted. Focus moved to nothing reads as having left.
pub fn focus_stays_within(root: Option<&Event<MountedData>>, e: &Event<FocusData>) -> bool {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    let Some(root) = root.and_then(|r| r.try_as_web_event()) else {
        return false;
    };
    e.try_as_web_event()
        .and_then(|ev| ev.related_target())
        .and_then(|t| t.dyn_into::<web_sys::Node>().ok())
        .is_some_and(|n| root.contains(Some(&n)))
}

/// The `<canvas>` element a mount event fired on (`panels::navigator`).
///
/// Off the event, not by id: a panel that reopens mounts a new canvas, and a surface
/// bound to the old element would stay blank.
pub fn canvas_of(e: &Event<MountedData>) -> Option<Canvas> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    e.try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .map(Canvas)
}

/// The canvas the app rendered under `id`, looked up once.
///
/// Panics if it is not there, which is a bug in the markup; the stub panics for
/// [`Canvas::surface_target`]'s reason.
pub fn canvas_by_id(id: &str) -> Canvas {
    use wasm_bindgen::JsCast;
    Canvas(
        web_sys::window()
            .expect("window")
            .document()
            .expect("document")
            .get_element_by_id(id)
            .expect("canvas element present")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("element is a canvas"),
    )
}

/// Await one animation frame, so layout is reflected before the canvas is measured.
/// Resumes in the microtask drain: fine for setup, too late for a paint (see
/// [`on_animation_frame`]).
pub async fn next_frame() {
    called_back(|window, wake| window.request_animation_frame(wake)).await;
}

/// Physical pixels per CSS pixel. `1.0` where unreadable, non-finite or not positive,
/// since it is used as a divisor.
pub fn device_pixel_ratio() -> f32 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0)
}

/// Seconds on a **monotonic** clock, which `stark-engine` does not own (§17.5).
///
/// Every use is a duration (`PEER_TIMEOUT`, `HEARTBEAT`, …), and a wall clock that
/// steps would stall the heartbeat or expire the roster. `Date.now()` only where there
/// is no `performance`.
pub fn now_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or_else(js_sys::Date::now, |p| p.now())
        / 1000.0
}

/// The page URL's fragment, without its leading `#`.
pub fn url_fragment() -> Option<String> {
    let hash = web_sys::window()?.location().hash().ok()?;
    Some(hash.strip_prefix('#').unwrap_or(&hash).to_string())
}

/// This page's address with `fragment` after the `#`, or just the fragment where
/// there is no location to read.
pub fn url_with_fragment(fragment: &str) -> String {
    let Some(location) = web_sys::window().map(|w| w.location()) else {
        return format!("#{fragment}");
    };
    format!(
        "{}{}{}#{fragment}",
        location.origin().unwrap_or_default(),
        location.pathname().unwrap_or_default(),
        location.search().unwrap_or_default()
    )
}

/// Set (or, with `None`, clear) the page URL's fragment, via `replaceState` so tab
/// history is untouched.
pub fn set_url_fragment(fragment: Option<&str>) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let url = match fragment {
        Some(f) => format!("#{f}"),
        // An empty replaceState URL would keep the current hash.
        None => {
            let location = window.location();
            format!(
                "{}{}",
                location.pathname().unwrap_or_default(),
                location.search().unwrap_or_default()
            )
        }
    };
    if let Ok(history) = window.history()
        && let Err(e) = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url))
    {
        tracing::warn!("failed to update URL fragment: {e:?}");
    }
}

/// Put `text` on the system clipboard, fire-and-forget: a denied permission is not
/// reported.
pub fn copy_to_clipboard(text: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.navigator().clipboard().write_text(text);
    }
}

/// `localStorage` under `key`. Format and failure policy live in
/// [`stark_ui::storage`].
pub fn local_get(key: &str) -> Option<String> {
    web_sys::window()?
        .local_storage()
        .ok()
        .flatten()?
        .get_item(key)
        .ok()
        .flatten()
}

/// Store `value` under `key`. `false` if it could not be written — no store, or
/// no room in it.
pub fn local_set(key: &str, value: &str) -> bool {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .is_some_and(|store| store.set_item(key, value).is_ok())
}

pub fn local_remove(key: &str) {
    if let Some(store) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = store.remove_item(key);
    }
}

// --- the blob store --------------------------------------------------------
//
// Bytes go to IndexedDB, not `localStorage`: it is quota'd against the disk rather
// than a few MB of text, and does its I/O off the painting thread (§25.6).

/// The database, its version, and its one object store.
///
/// One store, namespaced by key prefix (`stark.shapes/<hex>`): a store per record
/// would need a version bump, which every other open tab must be talked through.
const BLOB_DB: (&str, u32, &str) = ("stark", 1, "blobs");

/// The database, opened for this call, its object store created if missing.
///
/// Opened per call rather than held, because a live handle blocks another tab's upgrade.
async fn blob_db() -> Result<Connection, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::Closure;

    let (name, version, store) = BLOB_DB;
    let request = web_sys::window()
        .ok_or("no window")?
        .indexed_db()
        .map_err(|e| reason(&e))?
        .ok_or("no IndexedDB in this browser")?
        .open_with_u32(name, version)
        .map_err(|e| reason(&e))?;
    // An object store can only be created inside the upgrade.
    let upgrade = Closure::once_into_js(move |event: web_sys::Event| {
        if let Some(opened) = fired_at::<web_sys::IdbOpenDbRequest>(&event)
            && let Ok(value) = opened.result()
            && let Ok(db) = value.dyn_into::<web_sys::IdbDatabase>()
        {
            let _ = db.create_object_store(store);
        }
    });
    request.set_onupgradeneeded(Some(upgrade.unchecked_ref()));
    blob_opened(&request)
        .await
        .map_err(|e| reason(&e))?
        .dyn_into()
        .map(Connection)
        .map_err(|_| "the open resolved to something other than a database".to_string())
}

/// A connection to the blob store, closed when dropped. `close` lets transactions
/// already started finish, so it may be dropped as soon as the requests are issued.
struct Connection(web_sys::IdbDatabase);

impl std::ops::Deref for Connection {
    type Target = web_sys::IdbDatabase;

    fn deref(&self) -> &web_sys::IdbDatabase {
        &self.0
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Hang a future off the open `request`, now, for [`blob_pending`]'s reason, resolved
/// with the database.
///
/// A blocked open rejects rather than wait for other tabs, but stays queued, so a
/// database delivered after that is closed on arrival. One delivered in time closes
/// itself when another tab asks to upgrade.
fn blob_opened(request: &web_sys::IdbOpenDbRequest) -> wasm_bindgen_futures::JsFuture {
    use std::cell::Cell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::Closure;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let refused = Rc::new(Cell::new(false));
        let blocked = Closure::once_into_js({
            let (refused, reject) = (Rc::clone(&refused), reject.clone());
            move |_: web_sys::Event| {
                refused.set(true);
                let why = "another tab holds the blob store open at an older version";
                let _ = reject.call1(&JsValue::NULL, &JsValue::from_str(why));
            }
        });
        let opened = Closure::once_into_js(move |event: web_sys::Event| {
            let result = fired_at::<web_sys::IdbRequest>(&event)
                .and_then(|opened| opened.result().ok())
                .unwrap_or(JsValue::UNDEFINED);
            if let Some(db) = result.dyn_ref::<web_sys::IdbDatabase>() {
                if refused.get() {
                    db.close();
                    return;
                }
                let yield_to_upgrade = Closure::once_into_js(|event: web_sys::Event| {
                    if let Some(db) = fired_at::<web_sys::IdbDatabase>(&event) {
                        db.close();
                    }
                });
                db.set_onversionchange(Some(yield_to_upgrade.unchecked_ref()));
            }
            let _ = resolve.call1(&JsValue::NULL, &result);
        });
        let failed = Closure::once_into_js(move |event: web_sys::Event| {
            let error = fired_at::<web_sys::IdbRequest>(&event)
                .and_then(|opened| opened.error().ok().flatten());
            let why = error.map_or_else(
                || JsValue::from_str("the open failed with no reason given"),
                JsValue::from,
            );
            let _ = reject.call1(&JsValue::NULL, &why);
        });
        request.set_onblocked(Some(blocked.unchecked_ref()));
        request.set_onsuccess(Some(opened.unchecked_ref()));
        request.set_onerror(Some(failed.unchecked_ref()));
    });
    wasm_bindgen_futures::JsFuture::from(promise)
}

/// Hang a future off an IndexedDB request, **now**: the handlers are attached before
/// this returns, not when the future is first awaited.
///
/// Not an `async fn`, whose body would not run until awaited: a caller awaiting several
/// requests in turn would attach a later handler after its event had fired, and hang.
/// A failure rejects with the request's `DOMException`.
///
/// **No handler holds the request**; each reads it off its event. `once_into_js` frees
/// a handler only by running it, so one that never runs would leak the request and its
/// connection.
fn blob_pending(request: web_sys::IdbRequest) -> wasm_bindgen_futures::JsFuture {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::Closure;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let done = Closure::once_into_js({
            let reject = reject.clone();
            move |event: web_sys::Event| {
                // Rejected rather than resolved `undefined`, which a `get` reads as
                // nothing stored.
                let _ = match fired_at::<web_sys::IdbRequest>(&event)
                    .and_then(|done| done.result().ok())
                {
                    Some(value) => resolve.call1(&JsValue::NULL, &value),
                    None => {
                        let why = JsValue::from_str("the request finished with no result to read");
                        reject.call1(&JsValue::NULL, &why)
                    }
                };
            }
        });
        let failed = Closure::once_into_js(move |event: web_sys::Event| {
            let error = fired_at::<web_sys::IdbRequest>(&event)
                .and_then(|failed| failed.error().ok().flatten());
            let why = error.map_or_else(
                || JsValue::from_str("the request failed with no reason given"),
                JsValue::from,
            );
            let _ = reject.call1(&JsValue::NULL, &why);
        });
        request.set_onsuccess(Some(done.unchecked_ref()));
        request.set_onerror(Some(failed.unchecked_ref()));
    });
    wasm_bindgen_futures::JsFuture::from(promise)
}

/// Hang a future off `tx` finishing, now, for [`blob_pending`]'s reason — rejected with
/// the transaction's `DOMException` if it aborts.
///
/// **Only this says a write landed.** A full disk aborts the transaction at commit,
/// after every request in it succeeded.
fn blob_committed(tx: &web_sys::IdbTransaction) -> wasm_bindgen_futures::JsFuture {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::Closure;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let complete = Closure::once_into_js(move |_: web_sys::Event| {
            let _ = resolve.call0(&JsValue::NULL);
        });
        // The transaction off the event, for `blob_pending`'s reason.
        let abort = Closure::once_into_js(move |event: web_sys::Event| {
            // Null after an `abort()` called with no error behind it.
            let error = fired_at::<web_sys::IdbTransaction>(&event).and_then(|tx| tx.error());
            let why = error.map_or_else(
                || JsValue::from_str("the write was aborted with no reason given"),
                JsValue::from,
            );
            let _ = reject.call1(&JsValue::NULL, &why);
        });
        tx.set_oncomplete(Some(complete.unchecked_ref()));
        tx.set_onabort(Some(abort.unchecked_ref()));
    });
    wasm_bindgen_futures::JsFuture::from(promise)
}

/// What an IndexedDB event was fired at, as the type its handler was attached to.
fn fired_at<T: wasm_bindgen::JsCast>(event: &web_sys::Event) -> Option<T> {
    use wasm_bindgen::JsCast;

    event.target()?.dyn_into().ok()
}

/// Each of `keys` read in order — `None` where nothing is stored, the reason where one
/// would not read — or why the store could not be read at all.
///
/// **All requests are issued in one transaction before any is awaited.** A transaction
/// survives a microtask checkpoint but not a turn of the event loop, so issuing each
/// request after the previous resolved can find it closed.
pub async fn blob_get_many(keys: &[String]) -> Result<Vec<BlobRead>, String> {
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let db = blob_db().await?;
    let store = db
        .transaction_with_str(name)
        .and_then(|tx| tx.object_store(name))
        .map_err(|e| reason(&e))?;

    let pending: Vec<_> = keys
        .iter()
        .map(|key| {
            store
                .get(&JsValue::from_str(key))
                .map(blob_pending)
                .map_err(|e| reason(&e))
        })
        .collect();
    // Every request is issued, so the connection may close: it waits for them.
    drop(db);
    let mut out = Vec::with_capacity(keys.len());
    for request in pending {
        let read = match request {
            Ok(request) => request.await.map_err(|e| reason(&e)).and_then(stored_bytes),
            Err(why) => Err(why),
        };
        out.push(read);
    }
    Ok(out)
}

/// What a `get` resolved with, as bytes: `undefined` is nothing stored under the key.
/// Anything else that is not bytes is an error, since only an absence costs a library
/// its row.
fn stored_bytes(value: wasm_bindgen::JsValue) -> BlobRead {
    use wasm_bindgen::JsCast;

    if value.is_undefined() {
        return Ok(None);
    }
    value
        .dyn_into::<js_sys::Uint8Array>()
        .map(|array| Some(array.to_vec()))
        .map_err(|_| "what is stored under the key is not bytes".to_string())
}

/// Store `bytes` under `key`, or say why they did not land — the store's
/// `DOMException` (`QuotaExceededError` for a full disk), or [`blob_opened`]'s
/// blocked open.
pub async fn blob_put(key: &str, bytes: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let db = blob_db().await?;
    let tx = db
        .transaction_with_str_and_mode(name, web_sys::IdbTransactionMode::Readwrite)
        .map_err(|e| reason(&e))?;
    let committed = blob_committed(&tx);
    // `Uint8Array::from` copies into the JS heap, so the borrow ends here.
    let value = js_sys::Uint8Array::from(bytes);
    tx.object_store(name)
        .and_then(|store| store.put_with_key(&value, &JsValue::from_str(key)))
        .map_err(|e| reason(&e))?;
    drop(db);
    committed.await.map(|_| ()).map_err(|e| reason(&e))
}

/// Drop whatever is stored under `key`. Silent on failure: there is nothing to do
/// about a delete that did not take.
pub async fn blob_delete(key: &str) {
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let Ok(db) = blob_db().await else { return };
    let Ok(store) = db
        .transaction_with_str_and_mode(name, web_sys::IdbTransactionMode::Readwrite)
        .and_then(|tx| tx.object_store(name))
    else {
        return;
    };
    let Ok(request) = store.delete(&JsValue::from_str(key)) else {
        return;
    };
    let deleted = blob_pending(request);
    drop(db);
    let _ = deleted.await;
}

/// Hand `bytes` to the browser as a file download named `filename`, through a
/// synthetic `<a download>` on an object URL.
///
/// The URL is revoked right after the click: the download already holds the blob, and
/// a live URL would pin the whole buffer for the life of the document.
pub fn download_bytes(bytes: &[u8], filename: &str, mime: &str) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    // Copied into the JS heap, which the Blob then owns.
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

/// Ask the user for a file and hand its name and bytes to `on_file`.
///
/// Clicks a detached `<input type=file>`. The click must happen inside the user
/// gesture that asked for it, so call this straight from the handler, not from a task.
/// The change closure is `forget`ten, since the picker outlives this call.
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

    // Shared: `Fn` is not `Clone`, and each file's task needs its own handle.
    let on_file = std::rc::Rc::new(on_file);
    let input_for_change = input.clone();
    let on_change = Closure::<dyn FnMut()>::new(move || {
        let Some(file) = input_for_change.files().and_then(|f| f.get(0)) else {
            return;
        };
        let on_file = on_file.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let name = file.name();
            match read_file(file).await {
                Ok(bytes) => on_file(name, bytes),
                Err(reason) => tracing::error!(name, reason, "the chosen file could not be read"),
            }
        });
    });
    input.set_onchange(Some(on_change.as_ref().unchecked_ref()));
    on_change.forget();
    input.click();
}

/// A file's bytes, or the browser's reason for not reading them.
async fn read_file(file: web_sys::File) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;

    let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
        .await
        .map_err(|e| reason(&e))?;
    let buffer = buffer
        .dyn_ref::<js_sys::ArrayBuffer>()
        .ok_or("`File.arrayBuffer` resolved to something other than an ArrayBuffer")?;
    Ok(js_sys::Uint8Array::new(buffer).to_vec())
}

/// A thrown or rejected value as readable text. A `DOMException` is an `Error`, whose
/// `toString` is its name and message.
fn reason(e: &wasm_bindgen::JsValue) -> String {
    use wasm_bindgen::JsCast;

    match e.dyn_ref::<js_sys::Error>() {
        Some(error) => String::from(error.to_string()),
        None => e.as_string().unwrap_or_else(|| format!("{e:?}")),
    }
}

/// Hand `on_file` whatever file the OS launched the app with — the manifest's
/// `file_handlers` (§11, [`crate::files::bind_file_launch`]).
///
/// Through reflection: `launchQueue` is not in `web-sys`, and the file handle is only
/// behind `web_sys_unstable_apis`, a whole-build `RUSTFLAGS` change. A browser without
/// the API does nothing.
///
/// Setting the consumer delivers launches already queued, so call this only once the
/// handler can act. It may fire more than once (`focus-existing`).
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

    // Shared: each launch spawns a task that needs its own handle.
    let on_file = std::rc::Rc::new(on_file);
    let consumer = Closure::<dyn FnMut(JsValue)>::new(move |params: JsValue| {
        let files = js_sys::Reflect::get(&params, &JsValue::from_str("files"))
            .unwrap_or(JsValue::UNDEFINED);
        let Ok(files) = files.dyn_into::<js_sys::Array>() else {
            return;
        };
        // The first only: opening a document replaces the canvas (§8), so the
        // manifest asks for `single-client`.
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
                // Usually permission: a stale handle replayed later may have lost it.
                return tracing::error!("the launched file could not be opened");
            };
            let Ok(file) = file.dyn_into::<web_sys::File>() else {
                return;
            };
            let name = file.name();
            match read_file(file).await {
                Ok(bytes) => on_file(name, bytes),
                Err(reason) => {
                    tracing::error!(name, reason, "the launched file could not be read");
                }
            }
        });
    });
    let _ = set_consumer.call1(&queue, consumer.as_ref().unchecked_ref());
    consumer.forget();
}

/// Decode an image in any format the browser displays into a brush-shape PNG, and
/// whether it was inverted. The browser only decodes and resamples;
/// `stark_ui::assets::shape_png` decides what the pixels mean, so both frontends agree.
pub async fn normalize_shape_image(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    stark_ui::assets::shape_png(decode_to_canvas(bytes, stark_ui::assets::SHAPE_CAP).await?)
}

/// Decode an image in any format the browser displays into a canvas-substrate PNG
/// (§6.4), decided by `stark_ui::assets::substrate_png` as for
/// [`normalize_shape_image`].
pub async fn normalize_substrate_image(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    stark_ui::assets::substrate_png(decode_to_canvas(bytes, stark_ui::assets::SUBSTRATE_CAP).await?)
}

/// Decode `bytes` through the browser into straight RGBA8, its longest edge brought
/// within `cap`.
///
/// `drawImage` scales without materializing the full-size buffer (a 48 MP photo is
/// 190 MB of RGBA). `getImageData` is specified as un-premultiplied sRGB, the form
/// [`Decoded`] and [`Picture`](stark_assetid::Picture) are defined in.
async fn decode_to_canvas(bytes: Vec<u8>, cap: u32) -> Result<Decoded, String> {
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
    // The size is the frontends' shared rule, even though the resampler is not.
    let (width, height) = stark_ui::assets::fit(sw, sh, cap);

    let document = window.document().ok_or("no document")?;
    let canvas: web_sys::HtmlCanvasElement = document
        .create_element("canvas")
        .ok()
        .and_then(|e| e.dyn_into().ok())
        .ok_or("could not create a canvas")?;
    canvas.set_width(width);
    canvas.set_height(height);
    let ctx: web_sys::CanvasRenderingContext2d = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into().ok())
        .ok_or("no 2d context")?;
    let (dw, dh) = (f64::from(width), f64::from(height));
    ctx.draw_image_with_image_bitmap_and_dw_and_dh(&bitmap, 0.0, 0.0, dw, dh)
        .map_err(|_| "could not draw the image".to_string())?;
    let data = ctx
        .get_image_data(0.0, 0.0, dw, dh)
        .map_err(|_| "could not read the pixels".to_string())?;
    Ok(Decoded {
        width,
        height,
        rgba: data.data().0,
    })
}

/// Decode an image to place (§23), capped at
/// [`MAX_PICTURE_DIM`](stark_assetid::MAX_PICTURE_DIM), which `stark_assetid` applies
/// anyway; capping here spares the full-size buffer.
pub async fn decode_image(bytes: Vec<u8>) -> Result<Decoded, String> {
    decode_to_canvas(bytes, stark_assetid::MAX_PICTURE_DIM).await
}

/// Hand `handler` the bytes of the first image on the clipboard whenever one is
/// pasted into the page, except into a text field ([`on_text_entry`]).
///
/// The `paste` event rather than `navigator.clipboard.read()`: the event needs no
/// permission, while the async read prompts in Chrome and lacks image support in some
/// engines.
pub fn on_window_paste(handler: impl Fn(Vec<u8>) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    // Shared: each paste spawns a task that needs its own handle.
    let handler = std::rc::Rc::new(handler);
    let cb =
        Closure::<dyn FnMut(web_sys::ClipboardEvent)>::new(move |e: web_sys::ClipboardEvent| {
            if e.target().is_some_and(|t| on_text_entry(&t)) {
                return;
            }
            let Some(data) = e.clipboard_data() else {
                return;
            };
            let items = data.items();
            // The first image only: a clipboard often carries one image several times
            // over (a PNG and an HTML fragment naming it).
            let file = (0..items.length())
                .filter_map(|i| items.get(i))
                .filter(|item| item.kind() == "file" && item.type_().starts_with("image/"))
                .find_map(|item| item.get_as_file().ok().flatten());
            let Some(file) = file else {
                return;
            };
            // Only once there is an image, so a text paste still reaches its handler.
            e.prevent_default();
            let handler = handler.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let name = file.name();
                match read_file(file).await {
                    Ok(bytes) => handler(bytes),
                    Err(reason) => {
                        tracing::error!(name, reason, "the pasted image could not be read");
                    }
                }
            });
        });
    let _ = window.add_event_listener_with_callback("paste", cb.as_ref().unchecked_ref());
    cb.forget();
}
