//! The browser's half of [`crate::platform`]: every call that needs `web-sys`,
//! compiled for wasm32 only. `stub.rs` answers the same signatures off wasm, and
//! `tests/platform_parity.rs` holds the two to one surface.

use dioxus::prelude::*;
use stark_ui::assets::Decoded;

use super::{Coalesced, ElementBox, RawPointer};

/// Report a panic on the console. Without this a panic in wasm surfaces as an
/// `unreachable` trap, with its message lost.
pub fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Point `tracing` at both of the places this app reads it: the browser console, and
/// the timing histograms behind `timings::TimingModal` (`stark_engine::timing`, §7.1).
///
/// Two filtered layers over one registry rather than
/// `tracing_wasm::set_as_global_default`, because `tracing_wasm`'s layer calls
/// `performance.mark` and `performance.measure` on every span it is shown: unfiltered,
/// it would report every frame's timing spans into the devtools timeline, sixty times a
/// second. `TimingFilter::<false>` is the exact complement of the timing layer's own
/// filter, so no event is dropped and none is handled twice.
pub fn install_tracing() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    let console = tracing_wasm::WASMLayer::new(tracing_wasm::WASMLayerConfig::default())
        .with_filter(stark_engine::timing::TimingFilter::<false>);
    let subscriber = Registry::default()
        .with(stark_engine::timing::layer())
        .with(console);
    // A second install would be a second set of histograms with the first still
    // collecting, so it is refused. `run` installs once per page, so only a code change
    // can reach this.
    if tracing::subscriber::set_global_default(subscriber).is_err() {
        web_sys::console::warn_1(&"stark: a tracing subscriber was already installed".into());
    }
}

/// The page's painting surface, as everything outside this module sees it: the three
/// things `render` does with a `<canvas>`, and no `web_sys` type.
#[derive(Clone)]
pub struct Canvas(web_sys::HtmlCanvasElement);

impl Canvas {
    /// The element's laid-out size in CSS pixels (≥1). Measures the *element*, not
    /// the window, so an embedded or sub-window canvas works.
    pub fn laid_out_size(&self) -> (u32, u32) {
        (
            self.0.client_width().max(1) as u32,
            self.0.client_height().max(1) as u32,
        )
    }

    /// Resize the drawing buffer — the pixels behind the element, which the
    /// stylesheet's layout size does not set.
    pub fn set_buffer_size(&self, width: u32, height: u32) {
        self.0.set_width(width);
        self.0.set_height(height);
    }

    /// What `wgpu` binds a surface to.
    ///
    /// stark-dioxus-frontend is a web app (§11), so the surface is always the page's
    /// canvas. The crate still *compiles* for the host — that is what `cargo test` and
    /// clippy exercise — but there is no native windowing backend behind it, and
    /// reaching here off the web is a bug rather than a fallback.
    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        wgpu::SurfaceTarget::Canvas(self.0.clone())
    }
}

/// A key press or release at the window, as `input::bind_shortcuts` reads it.
///
/// The translation into the app's own vocabulary — dioxus's [`Key`] and
/// [`Modifiers`] — happens here rather than at the call site, because it is a fact
/// about the DOM event and not about what the shortcut means.
pub struct KeyEvent(web_sys::KeyboardEvent);

impl KeyEvent {
    /// The pressed key, in the same typed vocabulary the rsx! handlers read.
    pub fn key(&self) -> Key {
        use std::str::FromStr;
        Key::from_str(&self.0.key()).unwrap_or(Key::Unidentified)
    }

    /// The physical key, layout-independent — what the quick-brush rack reads, so a
    /// digit is a digit whatever the layout types on it (§18.1.8).
    pub fn code(&self) -> String {
        self.0.code()
    }

    /// The modifier set held during the event.
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

    /// Whether this went to a control that owns its own keystrokes — see
    /// [`on_text_entry`].
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }

    /// Take the browser's own action away from this event.
    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
}

/// A window event this app only wants to refuse — today the context menu, whose
/// two questions are "was it over a text field" and "stop it".
pub struct WindowEvent(web_sys::Event);

impl WindowEvent {
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }

    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
}

/// Resolve after `ms` milliseconds (so a settle animation can finish before the order is
/// committed). Browser `setTimeout` on web; a no-op off-wasm.
pub async fn sleep_ms(ms: i32) {
    called_back(|window, wake| {
        window.set_timeout_with_callback_and_timeout_and_arguments_0(wake, ms)
    })
    .await;
}

/// Await the browser calling the function `schedule` hands it.
///
/// Where there is no window or `schedule` fails, which a page's main thread never
/// meets, the reason is logged and the await never finishes. The task waiting on it
/// stops there; resolving at once instead would turn every `loop { sleep_ms(..) }` into
/// a spin that never yields to the browser.
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
/// The three callers are the three drags that reorder a column by dropping a thing
/// into it: the panel stack ([`panel_boxes`]), the layer tree ([`layer_boxes`]) and
/// the guide list ([`guide_boxes`]). All three go through one gesture
/// (`panels::reorder`), which measures once at grab time and derives everything
/// after from the live pointer, so there is no cached geometry to fall out of date.
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

/// The panel stack's scroll geometry: how far it is scrolled, how tall its content
/// is, and how much of it is showing — all in CSS px (§11).
///
/// Read off the DOM rather than computed from the panels' own heights, and it has to
/// be: the column's height is whatever eight panels of content come to, half of it
/// grown by lists the artist filled, and the one place that number exists is the
/// element the browser laid out. `None` before the stack has mounted, and off wasm.
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

/// Scroll the panel stack to `top` — what dragging its rail's thumb does
/// (`layout::PanelScrollbar`).
///
/// Setting the element's own `scrollTop` rather than keeping a scroll position of our
/// own: the browser clamps it, the wheel and the rail then agree by construction, and
/// the `scroll` event it raises is what tells the rail where it ended up.
pub fn set_stack_scroll(top: f32) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.query_selector(".panel-stack").ok().flatten())
    {
        el.set_scroll_top(top.max(0.0).round() as i32);
    }
}

/// The layer panel's rows, each under its `data-layer` id — see [`element_boxes`].
///
/// The whole entry is measured (`.layer-item`, indent included) rather than the row
/// inside it, because that is the box a drag opens a slot the size of: the margin
/// between two entries is the gap the drop lands in.
pub fn layer_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".layer-item[data-layer]", "data-layer")
}

/// The guides panel's rows, each under its `data-guide` position — see
/// [`element_boxes`]. A *position* rather than an id, because a guide has none; it
/// holds for the length of one gesture, which is all it is asked to.
pub fn guide_boxes() -> Vec<(String, f32, f32)> {
    element_boxes(".guide-row[data-guide]", "data-guide")
}

/// The box of the **first** element matching `selector`, or `None` where nothing
/// matches.
///
/// First rather than all, because the caller is pointing at *a* thing and a
/// selector that matched two would mean the thing had no single place to be pointed
/// at — see `tutor::Anchor`, which is what builds the selectors and is the
/// only place they are written.
///
/// `None` for "not on screen" is the answer the caller wants and not a failure: a
/// panel the user closed under the card is exactly this, and a card that draws
/// nothing is better than one that points at the corner of the window.
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

/// The window's inner height in CSS px — `0.0` where there is no window.
///
/// The one viewport figure a placement asks Rust for rather than the stylesheet.
/// Everything else about staying on screen is a `max-width` or a `max-height`, and
/// `calc` knows `100vh` without being told (`anchor::room_below`); what a declaration
/// cannot do is *branch*, and choosing whether a pop-out hangs from its row or rises
/// from it is a branch (`panels::popout`).
pub fn viewport_height() -> f32 {
    web_sys::window()
        .and_then(|w| w.inner_height().ok())
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32
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
/// The pointer counterpart to [`on_window_key`], and it exists for the same
/// reason: what it binds is not any one surface's business. The pen's eraser end
/// holds a brush slot for as long as it faces the glass (§18.1.8) whatever it is
/// hovering over or pressed against — the canvas, a slider, a preset row —
/// exactly as a held number key does, and a listener per surface would be a list
/// nobody could keep complete.
///
/// **Capture, not bubble**, and that is load-bearing on the press: the swap has
/// to be in force before the surface's own handler runs, or the canvas would open
/// its stroke on the brush the eraser displaced. Capture runs window-inward, so
/// this is ahead of every handler in the tree; bubble would be behind all of them.
/// It also cannot be silenced — `stopPropagation` downstream is too late to
/// prevent something that has already run, which matters most for the release,
/// where a listener that could be skipped would leave the brush swapped.
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

/// The button fields of a raw pointer event.
fn raw_of(e: &web_sys::PointerEvent) -> RawPointer {
    RawPointer {
        pen: e.pointer_type() == "pen",
        button: e.button(),
        buttons: e.buttons(),
        entered_nothing: e.related_target().is_none(),
    }
}

/// The same, off a **dioxus** pointer event — the one the tree's own handlers get.
///
/// `None` off wasm, where there is no raw event to unwrap; every caller reads that
/// as "not a pen", which is the right answer on a platform with no pens.
pub fn raw_pointer(e: &Event<PointerData>) -> Option<RawPointer> {
    use dioxus::web::WebEventExt;
    e.try_as_web_event().map(|raw| raw_of(&raw))
}

/// A pointer event's own timestamp in seconds — `performance.now()`'s clock,
/// monotonic and shared by every event on the page, which is what
/// [`InputSample::time`](stark_engine::command::InputSample::time) needs.
///
/// Zero when the raw event is out of reach (off wasm), matching the field's
/// default. The fitter keys its time channel to the first sample it sees, so only
/// differences matter and the origin is free.
pub fn event_time(e: &Event<PointerData>) -> f64 {
    use dioxus::web::WebEventExt;
    e.try_as_web_event()
        .map(|raw| raw.time_stamp() / 1000.0)
        .unwrap_or(0.0)
}

/// Every report the browser folded into a delivered `pointermove`, oldest first,
/// in the target element's own CSS pixels.
///
/// The browser delivers roughly one `pointermove` per animation frame and folds
/// the reports it withheld — most of what a 120–240 Hz pen produces — into the
/// delivered event's *coalesced* list. Reading that list is what gets the full
/// input rate to the fitter; reading only the event caps every stroke at display
/// rate, whatever the device resolved.
///
/// Mapped through the target's bounding rect (measured once per delivered event)
/// because the entries' client coordinates are the ones the spec guarantees. The
/// delivered event's own data equals the list's last entry, so nothing is reported
/// twice. `None` where there is no list — off wasm, or a synthetic event — and the
/// caller falls back to the event itself.
pub fn coalesced(e: &Event<PointerData>) -> Option<Vec<Coalesced>> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;

    let raw = e.try_as_web_event()?;
    let rect = raw
        .target()
        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())?
        .get_bounding_client_rect();
    let list = raw.get_coalesced_events();
    (list.length() > 0).then(|| {
        list.iter()
            .filter_map(|v| v.dyn_into::<web_sys::PointerEvent>().ok())
            .map(|c| Coalesced {
                x: (c.client_x() as f64 - rect.left()) as f32,
                y: (c.client_y() as f64 - rect.top()) as f32,
                pressure: c.pressure(),
                tilt_x: c.tilt_x() as f32,
                tilt_y: c.tilt_y() as f32,
                time: c.time_stamp() / 1000.0,
            })
            .collect()
    })
}

/// Whether `target` is a control that owns its own keystrokes — a text field, a
/// `<select>`, a contenteditable region.
///
/// Asked of the *event's target*, which for a key event is what has focus, rather
/// than of a flag the fields set on focus and clear on blur: a field that unmounts
/// while focused — commit-and-close on a rename — never fires its blur, and a flag
/// left stuck on would kill every shortcut for the rest of the session. The DOM is
/// asked at the moment of the keystroke, so it cannot fall out of step.
///
/// Two events ask it, which is why it is one function: the keyboard stands aside
/// for a field so the browser's own editing bindings work there (Ctrl+Z undoes the
/// *text*, purely because nothing calls `prevent_default` on it), and the context
/// menu stands aside for the same field because the browser's menu is the only cut,
/// copy and paste this app offers.
fn on_text_entry(target: &web_sys::EventTarget) -> bool {
    use wasm_bindgen::JsCast;

    let Some(el) = target.dyn_ref::<web_sys::HtmlElement>() else {
        return false;
    };
    el.is_content_editable()
        || match el.tag_name().as_str() {
            "TEXTAREA" | "SELECT" => true,
            // Sliders, checkboxes and color wells are not text entry. They want
            // arrows and space from the browser, but Ctrl+Z over one still means
            // the document — there is no text there for it to mean anything else.
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

/// Ask the browser to confirm before the page goes away, whenever `confirm`
/// answers true.
///
/// The one browser affordance for a document that lives in a tab: a reload, a closed
/// tab, a followed link and the back button all reach the same `beforeunload`, and
/// taking it is how an unsaved painting survives a mis-click.
///
/// **Everything about the prompt is the browser's** — its wording, its buttons, and
/// whether it appears at all. All a page can say is *that* it objects, which is why
/// this takes a predicate and returns nothing. It says it twice, because engines
/// disagree about which way counts: `preventDefault` is what the current spec reads,
/// a non-empty `returnValue` what older ones read. No major browser shows that string,
/// so all it has to be is *not empty*; it is a sentence for any engine that still does.
///
/// A browser also declines to prompt at all until the page has been interacted with,
/// which is a condition this app meets by the time it has anything to lose.
///
/// Registered once for the life of the page, so the closure is `forget`ten like the
/// listeners above.
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

/// Run `handler` **inside** the next animation-frame callback.
///
/// This is not the same as a task awaiting a promise the rAF resolves
/// ([`next_frame`]): a woken task only resumes in the microtask
/// drain, behind the scheduler's other work — for a dioxus task that includes a
/// VDOM render of whatever scopes are dirty by then. The callback itself runs in
/// the frame's animation phase, ahead of the browser's rendering steps by
/// definition, so work that must land in the frame that woke it belongs here.
///
/// One-shot: the closure frees itself after the call, so a registration per
/// frame leaks nothing (unlike the `forget`ten window listeners above, which
/// are bound once for the life of the page).
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

/// Capture the pointer for the element under `e`, so the in-progress drag keeps
/// streaming move/up events to it while the button is held — even after the pointer
/// leaves the element. The capture releases automatically on pointer-up, which is
/// guaranteed to be delivered to the capturing element.
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

/// Where a pointer event landed, as a fraction of its target element's box —
/// `(0, 0)` the top-left corner, `(1, 1)` the bottom-right, unclamped past the
/// edges. `None` off wasm, or for a target with no box to measure.
///
/// Read off the event's own target, which under pointer capture
/// ([`capture_pointer`]) stays the element that took the press — so a drag
/// keeps measuring the box it started in wherever the pointer goes. Fractions
/// rather than px so a control's geometry is the stylesheet's alone: the Oklab
/// picker's plane is one size in the Color panel and another in the pop-outs
/// and the Lighting panel, and no Rust constant has to mirror any of them
/// (`panels::color`).
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

/// Select all the text in the element `e` was mounted on — a no-op unless it is a
/// text field.
///
/// Dioxus's `MountedData` can focus an element but has nothing to say about the
/// selection inside it, so this reaches for the DOM node directly, the same route
/// [`capture_pointer`] takes.
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

/// Focus the element `e` was mounted on — how the command search's field takes
/// the keyboard the moment its palette opens (`rail::CommandSearch`). The DOM
/// node directly, for [`select_all`]'s reason.
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
/// mounted on — the question a dropdown holding a text field must ask before
/// reading the event as dismissal. `focusout` bubbles from *any* child losing
/// focus, including the trigger losing it to the field the open just mounted,
/// and a dropdown that closed on that would dismiss itself in the act of
/// opening. Focus gone to nothing at all (the window, a click on the canvas)
/// arrives with no related target and reads as having left, which it has.
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

/// The `<canvas>` element a mount event fired on, for binding a WebGPU surface to it
/// — the navigator's miniature (`panels::navigator`).
///
/// Off the event rather than by id through the document, unlike
/// [`canvas_by_id`]: this element's whole significance is that it
/// has *just* appeared and a surface has to be bound to this one rather than to
/// whichever element an id happens to resolve to. A panel that closes and reopens
/// mounts a new canvas, and binding the new surface to the old element would leave
/// the miniature blank with nothing to say why.
pub fn canvas_of(e: &Event<MountedData>) -> Option<Canvas> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    e.try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .map(Canvas)
}

/// The canvas the app rendered into the DOM under `id` — the main painting
/// canvas, or the brush editor's preview canvas.
///
/// By id through the document, unlike [`canvas_of`]: these two are placed by the
/// app itself and are looked up once, where a mounted element's whole significance
/// is that it has *just* appeared.
///
/// Panics if it is not there, which is a bug in the markup rather than a state to
/// handle. Off wasm there is no document, and reaching here is the same bug — see
/// [`Canvas::surface_target`].
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

/// Await one animation frame, so a layout pass (and any just-applied stylesheet)
/// is reflected before the canvas is measured.
///
/// Not the same as [`on_animation_frame`]: this *suspends* until the frame, where
/// that one runs work inside the callback. A task woken here resumes in the
/// microtask drain, which is the right place for setup and the wrong one for a
/// paint (see that function's note).
pub async fn next_frame() {
    called_back(|window, wake| window.request_animation_frame(wake)).await;
}

/// How many physical pixels the display packs into a CSS pixel — what
/// `input::input_resolution` prices a pointer's tolerance against.
///
/// `1.0` where it cannot be read, and where it is not finite or not positive: a
/// resolution is a divisor, and a bad one would make the fitting tolerance
/// meaningless rather than merely wrong.
pub fn device_pixel_ratio() -> f32 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0)
}

/// Seconds on a **monotonic** clock — the clock `stark-engine` deliberately does not
/// own (§17.5).
///
/// `performance.now()` rather than `Date.now()`, because every use of this is a
/// *duration*: `PEER_TIMEOUT`, `HEARTBEAT`, `GESTURE_TIMEOUT`, `GESTURE_RESYNC`.
/// Nothing compares it across clients, and nothing needs an epoch. A wall clock can
/// step — an NTP correction, a user changing the system time — and a step backwards
/// stops the heartbeat coming due until every peer drops this client, while one
/// forwards expires the whole roster in a single tick.
///
/// `performance.now()` is missing only in environments with no `performance` at
/// all, where `Date.now()` is the best available answer.
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

/// Reflect (or, with `None`, clear) the page URL's fragment. `replaceState`, so
/// joining and leaving a session do not pollute tab history.
pub fn set_url_fragment(fragment: Option<&str>) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let url = match fragment {
        Some(f) => format!("#{f}"),
        // Rebuild path + query without a fragment (an empty replaceState URL
        // would keep the current one, hash included).
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

/// Put `text` on the system clipboard. Fire-and-forget: the returned promise is
/// dropped, and a browser that denies the permission just leaves the readonly
/// field on screen to select by hand.
pub fn copy_to_clipboard(text: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.navigator().clipboard().write_text(text);
    }
}

/// What this browser has stored under `key`, per origin — the raw half of
/// [`stark_ui::storage`], which is where the format and the failure policy live.
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

/// Drop whatever is stored under `key`. Only [`stark_ui::storage::drop_retired`] calls
/// this, and it says how long either of them is worth keeping.
pub fn local_remove(key: &str) {
    if let Some(store) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = store.remove_item(key);
    }
}

// --- the blob store --------------------------------------------------------
//
// The raw half of [`stark_ui::storage`]'s second door. `localStorage` above is *text*,
// and a few megabytes of it per origin shared across every record this browser keeps
// — so bytes go to IndexedDB instead, which is quota'd against the disk, and which
// does its reading and writing off the thread the canvas paints on (§25.6), which is
// why everything here is `async`.

/// The database, its version, and its one object store.
///
/// **One store, with the record's namespace on the key** (`stark.shapes/<hex>`) —
/// which is what `localStorage` already does with its `stark.`-prefixed names. An
/// object store can only be created inside an `upgradeneeded`, so a store *per
/// record* would put a version bump behind every feature that ever wants to keep
/// bytes, and a version bump is a migration every other open tab has to be talked
/// through. A prefix is none of that.
const BLOB_DB: (&str, u32, &str) = ("stark", 1, "blobs");

/// The database, opened for this call, its object store created if this origin has
/// never had one.
///
/// Opened per call rather than held: a live handle blocks another tab's upgrade, and
/// the calls here are a startup read and the odd import — not something in a loop.
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
    // An object store can only be created inside the upgrade, so this closure is the
    // whole of the schema.
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

/// A connection to the blob store, closed when dropped.
///
/// `close` lets the transactions already started on a connection finish, so dropping
/// this once the requests are issued releases it as soon as they are done, on every
/// way out of the call.
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
/// **A blocked open rejects** rather than wait for every other tab to close the store.
/// It stays queued all the same, so a database it delivers after that is closed on
/// arrival. One delivered in time closes itself when another tab asks to upgrade.
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

/// Hang a future off an IndexedDB request, **now** — the handlers are attached before
/// this returns, and awaiting the result is a separate step.
///
/// That split is the whole reason this is not one `async fn`. An `async fn` body does
/// not run until it is awaited, so a caller issuing several requests and awaiting
/// them in turn would attach the second request's handler *after* its success event
/// had already fired, and wait on it forever. Here the handlers are on before the
/// caller can yield, so a batch may be started in one pass and collected in another —
/// which is also what keeps a batch inside one transaction (see [`blob_get_many`]).
///
/// The API is event-based rather than promise-based, so the one-shot handlers are
/// wrapped in a promise built here. A failure rejects with the request's
/// `DOMException`, so the caller can say which failure it was.
///
/// **No handler holds the request**; each reads it off its event. `once_into_js` frees
/// a handler only by running it, and through the request the one that never runs would
/// keep its result and its connection alive for good.
fn blob_pending(request: web_sys::IdbRequest) -> wasm_bindgen_futures::JsFuture {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::Closure;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let done = Closure::once_into_js(move |event: web_sys::Event| {
            let value = fired_at::<web_sys::IdbRequest>(&event)
                .and_then(|done| done.result().ok())
                .unwrap_or(JsValue::UNDEFINED);
            let _ = resolve.call1(&JsValue::NULL, &value);
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
/// **Only this says a write landed.** A request succeeds when the store accepts it; a
/// full disk aborts the transaction at commit, after every request in it succeeded.
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

/// The bytes stored under each of `keys`, in that order — `None` where this browser
/// has nothing under one, or could not read it — or why the store could not be read
/// at all.
///
/// **One transaction, all the requests issued before any of them is awaited.** A
/// transaction stays alive across a microtask checkpoint but not across a turn of the
/// event loop, so issuing request *n+1* only after *n* has resolved is the shape that
/// works right up until it does not. Starting them all first makes the whole batch
/// one exchange with the store and takes the question off the table.
pub async fn blob_get_many(keys: &[String]) -> Result<Vec<Option<Vec<u8>>>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let db = blob_db().await?;
    let store = db
        .transaction_with_str(name)
        .and_then(|tx| tx.object_store(name))
        .map_err(|e| reason(&e))?;

    let pending: Vec<_> = keys
        .iter()
        .map(|key| store.get(&JsValue::from_str(key)).ok().map(blob_pending))
        .collect();
    // Every request is issued, so the connection may close: it waits for them.
    drop(db);
    let mut out = Vec::with_capacity(keys.len());
    for request in pending {
        let bytes = match request {
            Some(request) => request
                .await
                .ok()
                .and_then(|value| value.dyn_into::<js_sys::Uint8Array>().ok())
                .map(|array| array.to_vec()),
            None => None,
        };
        out.push(bytes);
    }
    Ok(out)
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
    // `Uint8Array::from` copies into the JS heap, so the borrow does not have to
    // outlive the call — the same bargain `download_bytes` makes below.
    let value = js_sys::Uint8Array::from(bytes);
    tx.object_store(name)
        .and_then(|store| store.put_with_key(&value, &JsValue::from_str(key)))
        .map_err(|e| reason(&e))?;
    drop(db);
    committed.await.map(|_| ()).map_err(|e| reason(&e))
}

/// Drop whatever is stored under `key`. Silent either way: the caller has already
/// forgotten it, and there is nothing to do about a delete that did not take.
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

/// Hand `bytes` to the browser as a file download named `filename`.
///
/// A Blob behind an object URL, clicked through a synthetic `<a download>` — the
/// only way to write a file from a page without a server. The object URL is
/// revoked immediately after the click: the download has already taken a
/// reference to the blob by then, and leaving it alive pins the whole buffer (a
/// full-resolution PNG) for the life of the document.
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

/// Ask the user for a file and hand its name and bytes to `on_file`.
///
/// A hidden `<input type=file>` clicked programmatically: a page cannot open a
/// file picker any other way, and the click must happen inside the user gesture
/// that asked for it, so this is called straight from the menu handler rather
/// than from a task.
///
/// The closure is `forget`ten rather than dropped — the input outlives this call by
/// design (the user may sit in the picker for a minute), and dropping the `Closure`
/// would invalidate the JS callback before it fires.
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
    // shared rather than moved: `Fn` is not `Clone`, and each file's task needs its
    // own handle.
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

/// The bytes of a file the page was handed — picked, launched or pasted — or the
/// browser's reason for not reading them.
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

/// A thrown or rejected value as a person reads it.
///
/// A `DOMException` (`NotReadableError`, `QuotaExceededError`, …) is an `Error`, whose
/// `toString` is its name and message.
fn reason(e: &wasm_bindgen::JsValue) -> String {
    use wasm_bindgen::JsCast;

    match e.dyn_ref::<js_sys::Error>() {
        Some(error) => String::from(error.to_string()),
        None => e.as_string().unwrap_or_else(|| format!("{e:?}")),
    }
}

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

/// Normalize an image into a brush-shape PNG, using the browser as the decoder —
/// any format the browser can display can be imported (JPEG, WebP, GIF, …).
///
/// **The browser's half only**: decode, and downscale to the size
/// `stark_ui::assets::fit` asks for. What the pixels then mean — that a light
/// border is paper with ink on it, and the inversion that follows — is the crate's,
/// so the two frontends cannot come to read one file two ways.
///
/// Returns the PNG bytes and whether the inversion fired (so the UI can say so).
pub async fn normalize_shape_image(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), String> {
    stark_ui::assets::shape_png(decode_to_canvas(bytes, stark_ui::assets::SHAPE_CAP).await?)
}

/// Normalize an image into a **canvas-substrate PNG**, using the browser as the
/// decoder — any format it can display can become a substrate (JPEG, WebP, TIFF, …).
///
/// [`normalize_shape_image`]'s sibling, and the same split: the browser decodes and
/// resamples, `stark_ui::assets::substrate_png` decides. What it decides — grey by
/// luminance, no inversion, alpha composited over white rather than multiplied in —
/// is the whole of what a substrate is as against a stamp (§6.4), and is stated there.
///
/// The result still goes through `Engine::import_substrate`, which decodes it again
/// and hashes what it finds: this makes a substrate *possible*, and the id still comes
/// out of the bytes.
pub async fn normalize_substrate_image(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    stark_ui::assets::substrate_png(decode_to_canvas(bytes, stark_ui::assets::SUBSTRATE_CAP).await?)
}

/// Decode `bytes` through the browser into straight RGBA8, its longest edge brought
/// within `cap` — the first half of every import here.
///
/// The browser resamples because `drawImage` is the one step in the chain that scales
/// without first materializing the full-size buffer: a 48-megapixel photograph is
/// 190 MB of RGBA before anything has looked at it. `getImageData` is specified as
/// un-premultiplied sRGB, the form [`Decoded`] and
/// [`Picture`](stark_assetid::Picture) are both defined in.
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
    // The *size* is the shared rule's, even though the resampling is not: two
    // frontends asking their own resampler for two different sizes would be a
    // divergence that did not have to exist (`stark_ui::assets`).
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

/// Decode an image to place (§23), capped at the identity contract's
/// [`MAX_PICTURE_DIM`](stark_assetid::MAX_PICTURE_DIM) — which `stark_assetid` applies
/// anyway, so the cap here only spares the full-size buffer.
pub async fn decode_image(bytes: Vec<u8>) -> Result<Decoded, String> {
    decode_to_canvas(bytes, stark_assetid::MAX_PICTURE_DIM).await
}

/// Hand `handler` the bytes of the first image on the clipboard whenever one is
/// pasted into the page.
///
/// The **`paste` event** rather than `navigator.clipboard.read()`, and the difference
/// matters: the event is delivered inside the user's own gesture and needs no
/// permission, where the async read prompts in Chrome and is unimplemented for images
/// in some engines. A paste the page never sees is a feature that works for some people.
///
/// Nothing is handed over for a paste into a **text field** — a layer being renamed, the
/// session name — which is [`on_text_entry`]'s question asked of the event's target, the
/// same way the keyboard shortcuts ask it. Pasting a screenshot while typing a layer
/// name should type nothing and place nothing.
///
/// Bound once for the life of the page, so the closure is `forget`ten like the window
/// key handlers'.
pub fn on_window_paste(handler: impl Fn(Vec<u8>) + 'static) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    let Some(window) = web_sys::window() else {
        return;
    };
    // Shared rather than moved: the listener is re-entered per paste, and each of those
    // spawns a task that needs its own handle — `pick_file`'s bargain.
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
            // The first image, and only the first: a paste is one gesture, and a clipboard
            // carrying an image usually carries it several times over (a PNG *and* an HTML
            // fragment naming it), so taking every entry would place the same picture twice.
            let file = (0..items.length())
                .filter_map(|i| items.get(i))
                .filter(|item| item.kind() == "file" && item.type_().starts_with("image/"))
                .find_map(|item| item.get_as_file().ok().flatten());
            let Some(file) = file else {
                return;
            };
            // Only now, once there is an image to place: an ordinary text paste has to
            // reach whatever would have handled it.
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
