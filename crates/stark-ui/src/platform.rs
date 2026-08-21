//! **The only module that touches the browser**, each call with an off-wasm
//! counterpart so the crate still compiles for the host — which is what
//! `cargo test` and `cargo clippy --workspace --all-targets` exercise.
//!
//! "The only module" is a claim the compiler now checks. `web-sys`, `js-sys`,
//! `wasm-bindgen` and `wasm-bindgen-futures` are declared under
//! `[target.'cfg(target_arch = "wasm32")'.dependencies]`, and a crate that is not a
//! *direct* dependency is not in the extern prelude — so off wasm, `web_sys::` does
//! not resolve anywhere, in this file or any other. It used to be prose, and four
//! other modules had quietly stopped obeying it: `render` held an
//! `HtmlCanvasElement`, `input` downcast to `PointerEvent` and `KeyboardEvent`,
//! `collab` read `window()` five times, `storage` opened the store itself.
//!
//! That is the whole of what gating buys, and it is worth being exact about: it
//! does **not** make the host build cheaper, because `dioxus` pulls `web-sys` in
//! transitively whatever this crate declares. What it buys is that the boundary
//! cannot rot — a browser call written into a panel stops the host build, at the
//! line that wrote it, instead of being found later by someone reading for it.
//!
//! # What crosses the boundary
//!
//! Types the browser's own vocabulary would otherwise leak through: [`Canvas`],
//! [`KeyEvent`], [`WindowEvent`], [`RawPointer`], [`Coalesced`]. Each is the
//! smallest thing its callers actually use, so the *policy* stays with them — this
//! module says what the browser reported, never what the app should do about it.
//! [`RawPointer`] is the clearest case: it carries the two button fields and the
//! pointer type, and `input::is_eraser_event` is what reads a pen's tail out of
//! them (§18.1.8).

use dioxus::prelude::*;

// --- what crosses the boundary ---------------------------------------------

/// The page's painting surface, as everything outside this module sees it.
///
/// `render::Renderer` held a `web_sys::HtmlCanvasElement` and reached for its
/// methods directly; this is the three things it actually did with one. Off wasm
/// it is a unit, which is honest — there is no canvas, and [`Self::surface_target`]
/// says so the way `render` always did.
#[derive(Clone)]
#[cfg(target_arch = "wasm32")]
pub struct Canvas(web_sys::HtmlCanvasElement);
#[derive(Clone)]
#[cfg(not(target_arch = "wasm32"))]
pub struct Canvas;

impl Canvas {
    /// The element's laid-out size in CSS pixels (≥1). Measures the *element*, not
    /// the window, so an embedded or sub-window canvas works.
    #[cfg(target_arch = "wasm32")]
    pub fn laid_out_size(&self) -> (u32, u32) {
        (
            self.0.client_width().max(1) as u32,
            self.0.client_height().max(1) as u32,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn laid_out_size(&self) -> (u32, u32) {
        (1, 1)
    }

    /// Resize the drawing buffer — the pixels behind the element, which the
    /// stylesheet's layout size does not set.
    #[cfg(target_arch = "wasm32")]
    pub fn set_buffer_size(&self, width: u32, height: u32) {
        self.0.set_width(width);
        self.0.set_height(height);
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_buffer_size(&self, _width: u32, _height: u32) {}

    /// What `wgpu` binds a surface to.
    ///
    /// stark-ui is a web app (§11), so the surface is always the page's canvas. The
    /// crate still *compiles* for the host — that is what `cargo test` and clippy
    /// exercise — but there is no native windowing backend behind it, and reaching
    /// here off the web is a bug rather than a fallback.
    #[cfg(target_arch = "wasm32")]
    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        wgpu::SurfaceTarget::Canvas(self.0.clone())
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn surface_target(&self) -> wgpu::SurfaceTarget<'static> {
        unimplemented!("stark-ui targets the web; there is no native surface backend")
    }
}

/// A key press or release at the window, as `input::bind_shortcuts` reads it.
///
/// The translation into the app's own vocabulary — dioxus's [`Key`] and
/// [`Modifiers`] — happens here rather than at the call site, because it is a fact
/// about the DOM event and not about what the shortcut means.
#[cfg(target_arch = "wasm32")]
pub struct KeyEvent(web_sys::KeyboardEvent);
#[cfg(not(target_arch = "wasm32"))]
pub struct KeyEvent;

impl KeyEvent {
    /// The pressed key, in the same typed vocabulary the rsx! handlers read.
    #[cfg(target_arch = "wasm32")]
    pub fn key(&self) -> Key {
        use std::str::FromStr;
        Key::from_str(&self.0.key()).unwrap_or(Key::Unidentified)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn key(&self) -> Key {
        Key::Unidentified
    }

    /// The physical key, layout-independent — what the quick-brush rack reads, so a
    /// digit is a digit whatever the layout types on it (§18.1.8).
    #[cfg(target_arch = "wasm32")]
    pub fn code(&self) -> String {
        self.0.code()
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn code(&self) -> String {
        String::new()
    }

    /// The modifier set held during the event.
    #[cfg(target_arch = "wasm32")]
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn modifiers(&self) -> Modifiers {
        Modifiers::empty()
    }

    /// Whether this went to a control that owns its own keystrokes — see
    /// [`on_text_entry`].
    #[cfg(target_arch = "wasm32")]
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_text_entry(&self) -> bool {
        false
    }

    /// Take the browser's own action away from this event.
    #[cfg(target_arch = "wasm32")]
    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn prevent_default(&self) {}
}

/// A window event this app only wants to refuse — today the context menu, whose
/// two questions are "was it over a text field" and "stop it".
#[cfg(target_arch = "wasm32")]
pub struct WindowEvent(web_sys::Event);
#[cfg(not(target_arch = "wasm32"))]
pub struct WindowEvent;

impl WindowEvent {
    #[cfg(target_arch = "wasm32")]
    pub fn on_text_entry(&self) -> bool {
        self.0.target().is_some_and(|t| on_text_entry(&t))
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_text_entry(&self) -> bool {
        false
    }

    #[cfg(target_arch = "wasm32")]
    pub fn prevent_default(&self) {
        self.0.prevent_default();
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn prevent_default(&self) {}
}

/// A pointer event's raw button fields — everything `input::is_eraser_event`
/// needs, and nothing it does not.
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
/// The three callers are the three drags that reorder a column by dropping a thing
/// into it: the panel stack ([`panel_boxes`]), the layer tree ([`layer_boxes`]) and
/// the guide list ([`guide_boxes`]). All three go through one gesture
/// (`panels::reorder`), which measures once at grab time and derives everything
/// after from the live pointer, so there is no cached geometry to fall out of date.
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

/// The panel stack's scroll geometry: how far it is scrolled, how tall its content
/// is, and how much of it is showing — all in CSS px (§11).
///
/// Read off the DOM rather than computed from the panels' own heights, and it has to
/// be: the column's height is whatever eight panels of content come to, half of it
/// grown by lists the artist filled, and the one place that number exists is the
/// element the browser laid out. `None` before the stack has mounted, and off wasm.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn stack_scroll() -> Option<(f32, f32, f32)> {
    None
}

/// Scroll the panel stack to `top` — what dragging its rail's thumb does
/// (`layout::PanelScrollbar`).
///
/// Setting the element's own `scrollTop` rather than keeping a scroll position of our
/// own: the browser clamps it, the wheel and the rail then agree by construction, and
/// the `scroll` event it raises is what tells the rail where it ended up.
#[cfg(target_arch = "wasm32")]
pub fn set_stack_scroll(top: f32) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.query_selector(".panel-stack").ok().flatten())
    {
        el.set_scroll_top(top.max(0.0).round() as i32);
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn set_stack_scroll(_top: f32) {}

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

/// One element's box on screen, in CSS px from the viewport's top-left.
///
/// The whole rectangle, where [`element_boxes`] keeps only the two numbers a column
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

/// The box of the **first** element matching `selector`, or `None` where nothing
/// matches.
///
/// First rather than all, because the caller is pointing at *a* thing and a
/// selector that matched two would mean the thing had no single place to be pointed
/// at — see [`crate::tutor::Anchor`], which is what builds the selectors and is the
/// only place they are written.
///
/// `None` for "not on screen" is the answer the caller wants and not a failure: a
/// panel the user closed under the card is exactly this, and a card that draws
/// nothing is better than one that points at the corner of the window.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn anchor_box(_selector: &str) -> Option<ElementBox> {
    None
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
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_key(_kind: &str, _handler: impl FnMut(KeyEvent) + 'static) {}

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
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_event(_kind: &str, _handler: impl FnMut(WindowEvent) + 'static) {}

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
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_pointer(_kind: &str, _handler: impl FnMut(RawPointer) + 'static) {}

/// The button fields of a raw pointer event.
#[cfg(target_arch = "wasm32")]
fn raw_of(e: &web_sys::PointerEvent) -> RawPointer {
    RawPointer {
        pen: e.pointer_type() == "pen",
        button: e.button(),
        buttons: e.buttons(),
    }
}

/// The same, off a **dioxus** pointer event — the one the tree's own handlers get.
///
/// `None` off wasm, where there is no raw event to unwrap; every caller reads that
/// as "not a pen", which is the right answer on a platform with no pens.
#[cfg(target_arch = "wasm32")]
pub fn raw_pointer(e: &Event<PointerData>) -> Option<RawPointer> {
    use dioxus::web::WebEventExt;
    e.try_as_web_event().map(|raw| raw_of(&raw))
}
#[cfg(not(target_arch = "wasm32"))]
pub fn raw_pointer(_e: &Event<PointerData>) -> Option<RawPointer> {
    None
}

/// A pointer event's own timestamp in seconds — `performance.now()`'s clock,
/// monotonic and shared by every event on the page, which is what
/// [`InputSample::time`](stark_engine::InputSample::time) needs.
///
/// Zero when the raw event is out of reach (off wasm), matching the field's
/// default. The fitter keys its time channel to the first sample it sees, so only
/// differences matter and the origin is free.
#[cfg(target_arch = "wasm32")]
pub fn event_time(e: &Event<PointerData>) -> f64 {
    use dioxus::web::WebEventExt;
    e.try_as_web_event()
        .map(|raw| raw.time_stamp() / 1000.0)
        .unwrap_or(0.0)
}
#[cfg(not(target_arch = "wasm32"))]
pub fn event_time(_e: &Event<PointerData>) -> f64 {
    0.0
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
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn coalesced(_e: &Event<PointerData>) -> Option<Vec<Coalesced>> {
    None
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
#[cfg(target_arch = "wasm32")]
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
/// ([`next_frame`]): a woken task only resumes in the microtask
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
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn pointer_fraction(_e: &Event<PointerData>) -> Option<(f32, f32)> {
    None
}

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

/// Focus the element `e` was mounted on — how the command search's field takes
/// the keyboard the moment its palette opens (`main::CommandSearch`). The DOM
/// node directly, for [`select_all`]'s reason.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn focus(_e: &Event<MountedData>) {}

/// Whether the focus a `focusout` moved is still inside the element `root` was
/// mounted on — the question a dropdown holding a text field must ask before
/// reading the event as dismissal. `focusout` bubbles from *any* child losing
/// focus, including the trigger losing it to the field the open just mounted,
/// and a dropdown that closed on that would dismiss itself in the act of
/// opening. Focus gone to nothing at all (the window, a click on the canvas)
/// arrives with no related target and reads as having left, which it has.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn focus_stays_within(_root: Option<&Event<MountedData>>, _e: &Event<FocusData>) -> bool {
    false
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
#[cfg(target_arch = "wasm32")]
pub fn canvas_of(e: &Event<MountedData>) -> Option<Canvas> {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    e.try_as_web_event()
        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .map(Canvas)
}
#[cfg(not(target_arch = "wasm32"))]
pub fn canvas_of(_e: &Event<MountedData>) -> Option<Canvas> {
    None
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
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn canvas_by_id(_id: &str) -> Canvas {
    Canvas
}

/// Await one animation frame, so a layout pass (and any just-applied stylesheet)
/// is reflected before the canvas is measured.
///
/// Not the same as [`on_animation_frame`]: this *suspends* until the frame, where
/// that one runs work inside the callback. A task woken here resumes in the
/// microtask drain, which is the right place for setup and the wrong one for a
/// paint (see that function's note).
#[cfg(target_arch = "wasm32")]
pub async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .expect("window")
            .request_animation_frame(&resolve)
            .expect("request_animation_frame");
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn next_frame() {}

/// How many physical pixels the display packs into a CSS pixel — what
/// `input::input_resolution` prices a pointer's grain against.
///
/// `1.0` where it cannot be read, and where it is not finite or not positive: a
/// resolution is a divisor, and a bad one would make the fitting tolerance
/// meaningless rather than merely wrong.
#[cfg(target_arch = "wasm32")]
pub fn device_pixel_ratio() -> f32 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0)
}
#[cfg(not(target_arch = "wasm32"))]
pub fn device_pixel_ratio() -> f32 {
    1.0
}

/// Seconds on a **monotonic** clock — the clock `stark-engine` deliberately does not
/// own (§17.5).
///
/// `performance.now()` rather than `Date.now()`, because every use of this is a
/// *duration*: `PEER_TIMEOUT`, `HEARTBEAT`, `GESTURE_TIMEOUT`, `GESTURE_RESYNC`.
/// Nothing compares it across clients, and nothing needs an epoch. A wall clock
/// stepping — an NTP correction, a user changing the system time — broke those
/// durations in both directions: backwards, `now - pub_at` went negative, the
/// heartbeat stopped coming due and every peer dropped this client after six
/// seconds of apparent silence; forwards, the whole roster expired in a single
/// tick.
///
/// `performance.now()` is missing only in environments with no `performance` at
/// all, where `Date.now()` is the best available answer.
#[cfg(target_arch = "wasm32")]
pub fn now_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or_else(js_sys::Date::now, |p| p.now())
        / 1000.0
}
#[cfg(not(target_arch = "wasm32"))]
pub fn now_seconds() -> f64 {
    0.0
}

/// The page URL's fragment, without its leading `#`.
#[cfg(target_arch = "wasm32")]
pub fn url_fragment() -> Option<String> {
    let hash = web_sys::window()?.location().hash().ok()?;
    Some(hash.strip_prefix('#').unwrap_or(&hash).to_string())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn url_fragment() -> Option<String> {
    None
}

/// This page's address with `fragment` after the `#`, or just the fragment where
/// there is no location to read.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn url_with_fragment(fragment: &str) -> String {
    format!("#{fragment}")
}

/// Reflect (or, with `None`, clear) the page URL's fragment. `replaceState`, so
/// joining and leaving a session do not pollute tab history.
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
pub fn set_url_fragment(_fragment: Option<&str>) {}

/// Put `text` on the system clipboard. Fire-and-forget: the returned promise is
/// dropped, and a browser that denies the permission just leaves the readonly
/// field on screen to select by hand.
#[cfg(target_arch = "wasm32")]
pub fn copy_to_clipboard(text: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.navigator().clipboard().write_text(text);
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn copy_to_clipboard(_text: &str) {}

/// What this browser has stored under `key`, per origin — the raw half of
/// [`crate::storage`], which is where the format and the failure policy live.
#[cfg(target_arch = "wasm32")]
pub fn local_get(key: &str) -> Option<String> {
    web_sys::window()?
        .local_storage()
        .ok()
        .flatten()?
        .get_item(key)
        .ok()
        .flatten()
}
#[cfg(not(target_arch = "wasm32"))]
pub fn local_get(_key: &str) -> Option<String> {
    None
}

/// Store `value` under `key`. `false` if it could not be written — no store, or
/// no room in it.
#[cfg(target_arch = "wasm32")]
pub fn local_set(key: &str, value: &str) -> bool {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .is_some_and(|store| store.set_item(key, value).is_ok())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn local_set(_key: &str, _value: &str) -> bool {
    false
}

/// Drop whatever is stored under `key`. Only [`crate::storage::drop_retired`] calls
/// this, and it says how long either of them is worth keeping.
#[cfg(target_arch = "wasm32")]
pub fn local_remove(key: &str) {
    if let Some(store) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = store.remove_item(key);
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn local_remove(_key: &str) {}

// --- the blob store --------------------------------------------------------
//
// The raw half of [`crate::storage`]'s second door. `localStorage` above is *text*,
// and a few megabytes of it per origin shared across every record this browser keeps
// — so bytes go to IndexedDB instead, which is quota'd against the disk, and which
// does its reading and writing off the thread the canvas paints on (§25.6).
//
// Everything here is `async` for that last reason and not by taste: the store this
// replaces was synchronous, and re-encoding a shape library on the main thread is
// the cost that made the replacement worth doing.

/// The database, its version, and its one object store.
///
/// **One store, with the record's namespace on the key** (`stark.shapes/<hex>`) —
/// which is what `localStorage` already does with its `stark.`-prefixed names. An
/// object store can only be created inside an `upgradeneeded`, so a store *per
/// record* would put a version bump behind every feature that ever wants to keep
/// bytes, and a version bump is a migration every other open tab has to be talked
/// through. A prefix is none of that.
#[cfg(target_arch = "wasm32")]
const BLOB_DB: (&str, u32, &str) = ("stark", 1, "blobs");

/// The database, opened for this call, its object store created if this origin has
/// never had one.
///
/// Opened per call rather than held: a live handle blocks another tab's upgrade, and
/// the calls here are a startup read and the odd import — not something in a loop.
#[cfg(target_arch = "wasm32")]
async fn blob_db() -> Option<web_sys::IdbDatabase> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::Closure;

    let (name, version, store) = BLOB_DB;
    let request = web_sys::window()?
        .indexed_db()
        .ok()??
        .open_with_u32(name, version)
        .ok()?;
    // An object store can only be created inside the upgrade, so this closure is the
    // whole of the schema. `once_into_js` hands ownership to JS, which is what a
    // handler that fires at most once wants — there is nothing here to keep alive.
    let upgrade = Closure::once_into_js(move |event: web_sys::Event| {
        let Some(target) = event.target() else { return };
        let opened = target.unchecked_into::<web_sys::IdbOpenDbRequest>();
        if let Ok(value) = opened.result()
            && let Ok(db) = value.dyn_into::<web_sys::IdbDatabase>()
        {
            let _ = db.create_object_store(store);
        }
    });
    request.set_onupgradeneeded(Some(upgrade.unchecked_ref()));
    blob_pending((*request).clone()).await.ok()?.dyn_into().ok()
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
/// The API is event-based rather than promise-based, so the pair of one-shot handlers
/// is wrapped in a promise built here. A channel crate for two closures would be the
/// larger dependency.
#[cfg(target_arch = "wasm32")]
fn blob_pending(request: web_sys::IdbRequest) -> wasm_bindgen_futures::JsFuture {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::Closure;

    let promise = js_sys::Promise::new(&mut move |resolve, reject| {
        let done = request.clone();
        let ok = Closure::once_into_js(move |_: web_sys::Event| {
            let value = done.result().unwrap_or(JsValue::UNDEFINED);
            let _ = resolve.call1(&JsValue::NULL, &value);
        });
        let failed = Closure::once_into_js(move |_: web_sys::Event| {
            let _ = reject.call0(&JsValue::NULL);
        });
        request.set_onsuccess(Some(ok.unchecked_ref()));
        request.set_onerror(Some(failed.unchecked_ref()));
    });
    wasm_bindgen_futures::JsFuture::from(promise)
}

/// The bytes stored under each of `keys`, in that order — `None` where this browser
/// has nothing under one, or could not read it.
///
/// **One transaction, all the requests issued before any of them is awaited.** A
/// transaction stays alive across a microtask checkpoint but not across a turn of the
/// event loop, so issuing request *n+1* only after *n* has resolved is the shape that
/// works right up until it does not. Starting them all first makes the whole batch
/// one exchange with the store and takes the question off the table.
#[cfg(target_arch = "wasm32")]
pub async fn blob_get_many(keys: &[String]) -> Vec<Option<Vec<u8>>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let nothing = || keys.iter().map(|_| None).collect();
    let Some(db) = blob_db().await else {
        return nothing();
    };
    let Ok(store) = db
        .transaction_with_str(name)
        .and_then(|tx| tx.object_store(name))
    else {
        return nothing();
    };

    let pending: Vec<_> = keys
        .iter()
        .map(|key| store.get(&JsValue::from_str(key)).ok().map(blob_pending))
        .collect();
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
    out
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn blob_get_many(keys: &[String]) -> Vec<Option<Vec<u8>>> {
    keys.iter().map(|_| None).collect()
}

/// Store `bytes` under `key`. `false` if they did not land — no store, or no room in
/// it.
///
/// A full disk surfaces as this request's own error rather than as a short write,
/// which is what makes awaiting the request the answer to "did it land" — and what
/// lets `crate::storage` keep saying so in one line.
#[cfg(target_arch = "wasm32")]
pub async fn blob_put(key: &str, bytes: &[u8]) -> bool {
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let Some(db) = blob_db().await else {
        return false;
    };
    let Ok(store) = db
        .transaction_with_str_and_mode(name, web_sys::IdbTransactionMode::Readwrite)
        .and_then(|tx| tx.object_store(name))
    else {
        return false;
    };
    // `Uint8Array::from` copies into the JS heap, so the borrow does not have to
    // outlive the call — the same bargain `download_bytes` makes above.
    let value = js_sys::Uint8Array::from(bytes);
    let Ok(request) = store.put_with_key(&value, &JsValue::from_str(key)) else {
        return false;
    };
    blob_pending(request).await.is_ok()
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn blob_put(_key: &str, _bytes: &[u8]) -> bool {
    false
}

/// Drop whatever is stored under `key`. Silent either way: the caller has already
/// forgotten it, and there is nothing to do about a delete that did not take.
#[cfg(target_arch = "wasm32")]
pub async fn blob_delete(key: &str) {
    use wasm_bindgen::JsValue;

    let (_, _, name) = BLOB_DB;
    let Some(db) = blob_db().await else { return };
    let Ok(store) = db
        .transaction_with_str_and_mode(name, web_sys::IdbTransactionMode::Readwrite)
        .and_then(|tx| tx.object_store(name))
    else {
        return;
    };
    if let Ok(request) = store.delete(&JsValue::from_str(key)) {
        let _ = blob_pending(request).await;
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn blob_delete(_key: &str) {}

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
///   (1024 px, `stark_model::MAX_SHAPE_DIM`). The engine would cap it
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
    let cap = stark_model::MAX_SHAPE_DIM;
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

/// Decode an image into straight RGBA8, **using the browser as the decoder** — so
/// every format it can display can be placed (JPEG, PNG, WebP, AVIF, GIF, …).
///
/// The same route [`normalize_shape_image`] takes and for the same reason: shipping a
/// decoder per format would be a second, smaller answer to a question the platform
/// already answers completely (§23). What comes back is what a `PlaceImage` carries —
/// `getImageData` is specified as **un-premultiplied** sRGB, which is exactly the form
/// [`Picture`](stark_assetid::Picture) is defined in, so nothing has to be undone on
/// either side.
///
/// Downscaled so the longest edge is at most the identity contract's cap
/// ([`MAX_PICTURE_DIM`](stark_assetid::MAX_PICTURE_DIM)) — which `stark_assetid::picture`
/// would apply anyway, so this is an optimization and not the rule. Here because the
/// *browser* is the only thing in the chain that can resample
/// without first materializing the full-size buffer: a 48-megapixel phone photograph
/// is 190 MB of RGBA before anything has looked at it, and `drawImage` never allocates
/// it at all.
#[cfg(target_arch = "wasm32")]
pub async fn decode_image(bytes: Vec<u8>) -> Result<(u32, u32, Vec<u8>), String> {
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
    let cap = stark_assetid::MAX_PICTURE_DIM;
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
    Ok((w, h, data.data().0))
}
#[cfg(not(target_arch = "wasm32"))]
pub async fn decode_image(_bytes: Vec<u8>) -> Result<(u32, u32, Vec<u8>), String> {
    Err("no image decoder off the web".to_string())
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
#[cfg(target_arch = "wasm32")]
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
                let Ok(buffer) = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await
                else {
                    return tracing::error!("the pasted image could not be read");
                };
                let Some(buffer) = buffer.dyn_ref::<js_sys::ArrayBuffer>() else {
                    return;
                };
                handler(js_sys::Uint8Array::new(buffer).to_vec());
            });
        });
    let _ = window.add_event_listener_with_callback("paste", cb.as_ref().unchecked_ref());
    cb.forget();
}
#[cfg(not(target_arch = "wasm32"))]
pub fn on_window_paste(_handler: impl Fn(Vec<u8>) + 'static) {}

/// The standard base64 alphabet, and its inverse.
///
/// The inverse is built **at compile time**: [`base64_decode`] used to fill a
/// 256-byte table on every call, which is once per stored entry every time a
/// library is loaded — a table that is the same table each time, from an alphabet
/// that is a constant.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const INVERSE: [u8; 256] = {
    let mut table = [255u8; 256];
    let mut i = 0;
    while i < ALPHABET.len() {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// Standard base64 (with padding) — small, so data URLs and `localStorage`
/// blobs stay dependency-free.
pub fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = ALPHABET;
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
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = INVERSE[c as usize];
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
