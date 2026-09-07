//! Windows: the window procedure both readers share, and the choice between them.
//!
//! # Two APIs, one seam
//!
//! Windows offers three ways to hear a stylus and this crate speaks two of them.
//! **Windows Ink** ([`ink`]) reads the pointer messages the window already receives —
//! no driver, every digitizer, and the sub-frame report history that keeps a stroke at
//! the hand's rate rather than the display's. **Wintab** ([`wintab`]) is the
//! vendor-standard API a tablet driver brings with it, and it is here for one reason
//! that matters more than any of Ink's advantages: with *Use Windows Ink* unchecked in
//! a Wacom control panel — which many artists do, because it kills
//! press-and-hold-for-right-click — the Ink path reports **no pressure at all**, and
//! nothing tells the user why.
//!
//! (The third is `RealTimeStylus`, the legacy Ink API: a COM apartment, an async
//! plugin on a thread of its own, and Microsoft's own guidance that there is no reason
//! to use it when the pointer messages will do.)
//!
//! **Wintab wins when it opens.** A context opens only where a vendor driver is
//! installed, and where one is, that driver is the authority on that tablet — it knows
//! the pressure curve the user set, and it answers whether or not the Ink box is
//! ticked. Ink is what every other machine gets, which is every machine without a
//! tablet driver and every tablet the driver does not claim.
//!
//! # What the two share, and why that is this file
//!
//! Everything except how a report is read. Both need the same claim, the same queue,
//! the same wake, and the same answer to the question below — so a reader is a
//! *source of poses* and nothing more, and none of the hard parts are written twice.
//!
//! # The question: a stylus arrives more than once
//!
//! Windows delivers a pen twice over — as pointer messages, and again as the mouse
//! messages it synthesizes for compatibility — and an app that read both would paint
//! every stroke twice, once with real pressure and once at full. The usual fix is a
//! last-device-wins latch, which is a guess.
//!
//! There is no guess here. Windows synthesizes the mouse messages **because
//! `DefWindowProc` was reached**, so a message this file answers itself makes none.
//! The double input is not arbitrated; it is never created.
//!
//! The cost is that a swallowed message reaches nothing else either — not a button,
//! not the title bar — which is what [`Claim`] is for: the frontend publishes the
//! rectangle where a press is *paint*, and everywhere else the message is passed on
//! and the chrome goes on working. [`suppressed`] is that rule in one place, for both
//! readers.
//!
//! # winit is in this path too, and changes both answers
//!
//! winit 0.30 handles `WM_POINTERDOWN`, `WM_POINTERUPDATE` and `WM_POINTERUP` itself,
//! for **every** pointer type rather than only for touch, and consumes them: it turns
//! each into a `WindowEvent::Touch` — which wgpui does not read — and returns zero
//! without reaching `DefWindowProc`. So:
//!
//! **A pen message this file passes on goes to `DefWindowProc` directly** rather than
//! down the subclass chain. The compatibility mouse message a stylus needs in order to
//! press a button is one Windows makes only when `DefWindowProc` is reached, and winit
//! is what stops it being reached. Skipping winit for those messages is not a liberty
//! taken with the chain — it is the only way the chrome hears a pen at all. Touch is
//! untouched: nothing here runs unless the pointer is a stylus.
//!
//! **A message this file answers has to wake the window itself.** wgpui's event loop
//! genuinely sleeps when idle and every wake source it has is an OS event — so a
//! contact whose messages are all answered here would deliver a whole stroke to the
//! queue and never a frame to draw it in. [`wake`] is the missing edge.

use std::rc::Rc;
use std::sync::{Mutex, MutexGuard, PoisonError};

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{ClientToScreen, RDW_INTERNALPAINT, RedrawWindow};
use windows_sys::Win32::System::Performance::QueryPerformanceFrequency;
use windows_sys::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    DefWindowProcW, GetMessageExtraInfo, WM_LBUTTONDBLCLK, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MOUSEMOVE, WM_POINTERCAPTURECHANGED, WM_POINTERDOWN, WM_POINTERUP, WM_POINTERUPDATE,
};

use crate::model::{Claim, Phase, Pose, Report};

mod ink;
mod trace;
mod wintab;

use trace::trace;

/// How many reports may wait for a frame that is not coming.
///
/// The queue drains once a frame and a digitizer reports every few milliseconds, so
/// this is only ever reached when the frame loop has stopped — a modal file dialog, a
/// window being dragged. What it bounds is the memory that would otherwise grow for as
/// long as that lasts. **Motion is what gets dropped and never a press or a lift**, so
/// a gesture that overruns loses detail and still opens and closes.
const QUEUE_LIMIT: usize = 4096;

/// Which subclass on the window is ours. Any value distinguishes it from another
/// library's; this one is `STRK`.
const SUBCLASS_ID: usize = 0x5354_524B;

/// The mark Windows puts in a message's extra information when it made that message
/// from a stylus or a finger, and the mask that reads it.
///
/// Undocumented in the sense of having no constant in any header, and load-bearing
/// anyway: it is the only thing that distinguishes the mouse message a pen caused from
/// one a mouse caused, and telling them apart is what lets a Wintab stroke suppress
/// its own shadow without suppressing the actual mouse.
const PEN_SIGNATURE: u32 = 0xFF51_5700;
const SIGNATURE_MASK: u32 = 0xFFFF_FF00;
/// The bit that says the signed message came from a finger rather than a nib.
const FROM_TOUCH: u32 = 0x80;

/// Where the poses come from. Settled once, at [`Backend::attach`].
enum Reader {
    /// The pointer messages ([`ink`]).
    Ink,
    /// A Wintab context ([`wintab`]), which is preferred wherever one opens.
    Wintab(wintab::Context),
}

struct State {
    claim: Claim,
    queue: Vec<Report>,
    /// Whether a contact **on the canvas** is in hand, and so whether the messages a
    /// stylus is making belong to this file rather than to the chrome.
    ///
    /// Latched at the press and not asked again: a stroke that starts on the canvas
    /// and wanders over a panel is still one stroke, and re-testing the rectangle
    /// halfway through would hand the rest of it to the mouse.
    owned: bool,
    /// The converse latch: a contact that began **outside** the claim, which the
    /// chrome is handling and which must go on reaching it even where it strays over
    /// the canvas — a slider drag that overshoots is the case.
    deferred: bool,
    /// Which pointer the contact is, where the reader is Ink and several may be in
    /// flight at once. Wintab reports one stylus and has no use for it.
    pointer: Option<u32>,
    /// Whether a stylus is over the tablet at all, as the reader last heard it.
    ///
    /// **What replaces the message signature under Wintab.** Windows marks the mouse
    /// messages *it* synthesizes from a stylus, and [`from_pen`] reads that mark — but
    /// a tablet driver moving the system cursor itself is under no obligation to set
    /// it, and at least one does not. So the question a mouse message is asked is not
    /// only *were you made by a pen* but *is a pen on the tablet right now*, which the
    /// packets answer directly.
    stylus_near: bool,
    /// Where the stylus was last seen, for a lift whose own pose cannot be read.
    ///
    /// **Not the queue's last entry**, which is the obvious place to look and the
    /// wrong one: the frontend drains the queue every frame, so by the time a lift
    /// arrives the queue is usually empty and reading it would land the closing sample
    /// at the window's origin.
    last: Option<Pose>,
    ink: ink::Cache,
    packets: wintab::Packets,
}

struct Shared {
    /// Performance-counter ticks per second, or zero where the counter is unusable —
    /// in which case timestamps come off a millisecond clock instead.
    hz: f64,
    reader: Reader,
    state: Mutex<State>,
}

/// Lock the state, taking a poisoned lock as it stands.
///
/// **A window procedure may not panic**: unwinding out of an `extern "system"` call is
/// an abort at best, and the frame it would tear through belongs to winit. There is
/// nothing here an unwind could have left half-written anyway — the state is a queue,
/// a rectangle and some flags — so the poisoned guard is exactly as good as a fresh
/// one.
///
/// A `Mutex` around state that never leaves one thread, which is deliberate rather
/// than overlooked: a window procedure runs on the thread that owns the window and
/// nowhere else, so this is uncontended every time and costs an atomic. What it buys
/// is the absence of `RefCell`'s panic on a re-entrant borrow — and a panic is the one
/// thing that must not happen here.
fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    shared.state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Queue a report, dropping motion rather than growing without bound.
///
/// The pose is remembered whether or not it is queued, so a gesture that overran the
/// cap still lifts where the hand actually left off.
fn push(state: &mut State, phase: Phase, pose: Pose) {
    state.last = Some(pose);
    if state.queue.len() >= QUEUE_LIMIT && phase == Phase::Move {
        return;
    }
    state.queue.push(Report { phase, pose });
}

/// Ask Windows for a frame, the way winit's own `request_redraw` asks for one.
///
/// The same call with the same flags, which is the point of choosing it: what has to
/// happen is winit's `WM_PAINT` arm running and sending `RedrawRequested`, and the
/// surest way to reach it is the call winit reaches it with.
unsafe fn wake(hwnd: HWND) {
    // SAFETY: `hwnd` is this window, and both pointer arguments are the nulls that
    // mean "the whole window" — which is what winit passes.
    unsafe { RedrawWindow(hwnd, std::ptr::null(), 0, RDW_INTERNALPAINT) };
}

/// What one message came to.
struct Outcome {
    /// Whether anything reached the queue, and so whether a frame is owed.
    queued: bool,
    /// Whether Windows should be told the message was handled — which is what stops it
    /// synthesizing a mouse message from this report.
    swallow: bool,
}

impl Outcome {
    /// Nothing happened: no report, and the message is still somebody else's.
    const PASS: Self = Self {
        queued: false,
        swallow: false,
    };

    /// A report was taken and the message goes no further.
    const TAKEN: Self = Self {
        queued: true,
        swallow: true,
    };
}

/// Whether a stylus at `at` is this file's to answer rather than the chrome's.
///
/// **One rule, both readers, and it is the whole of the arbitration.** A contact in
/// hand is ours until it lifts; a contact the chrome is handling stays the chrome's
/// even where it strays over the canvas; and everything else — a hover, or the first
/// message of a press that has not been read yet — is decided by where it is.
///
/// That last clause is what makes the ordering not matter. A mouse message Windows
/// synthesized from a stylus may reach this window before the reader has seen the
/// contact that caused it, so a rule that waited for [`State::owned`] to be latched
/// would let the first press of every stroke through as a mouse click. Asking the
/// rectangle instead answers the same for both orders.
fn suppressed(state: &State, at: [f32; 2]) -> bool {
    if state.owned {
        return true;
    }
    if state.deferred {
        return false;
    }
    state.claim.takes(at)
}

/// Whether a mouse message is a stylus's shadow rather than a mouse.
///
/// Two answers, because one of them is not always available. Windows marks the
/// messages *it* synthesizes ([`from_pen`]), and that mark is exact — but a tablet
/// driver that moves the system cursor itself need not set it, and at least one does
/// not, which is how a full-pressure stroke gets drawn underneath a pen one. Where the
/// reader knows a stylus is over the tablet, that is the better answer anyway: it is
/// about the hardware rather than about who made the message.
unsafe fn is_shadow(state: &State) -> bool {
    // SAFETY: asked of the message this thread is dispatching.
    state.stylus_near || unsafe { from_pen() }
}

/// Whether the message being handled was made from a stylus rather than by a mouse.
///
/// `GetMessageExtraInfo` answers for the message the thread is currently dispatching,
/// which is exactly the one a window procedure has in hand — so this is asked inside
/// the procedure and is meaningless anywhere else.
unsafe fn from_pen() -> bool {
    // SAFETY: no arguments, and the answer is about the message this thread is
    // dispatching — which is the one this procedure was called with.
    let extra = unsafe { GetMessageExtraInfo() } as u32;
    extra & SIGNATURE_MASK == PEN_SIGNATURE && extra & FROM_TOUCH == 0
}

/// A mouse message's position, which it carries in `lParam` as two signed words of
/// client-area px.
fn mouse_at(lparam: LPARAM) -> [f32; 2] {
    let low = (lparam & 0xffff) as u16 as i16;
    let high = ((lparam >> 16) & 0xffff) as u16 as i16;
    [f32::from(low), f32::from(high)]
}

/// Where the window's client area begins on the screen, so a screen position can keep
/// its fraction on the way to a client one.
///
/// `ScreenToClient` is the usual call and cannot be used here: it works on whole
/// pixels, and rounding is what both readers go to some trouble to avoid.
unsafe fn client_origin(hwnd: HWND) -> Option<[f32; 2]> {
    let mut origin = POINT { x: 0, y: 0 };
    // SAFETY: `hwnd` is this window and the point is a live local, in and out.
    if unsafe { ClientToScreen(hwnd, &mut origin) } == 0 {
        return None;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a screen coordinate; f32 is exact well past any display"
    )]
    Some([origin.x as f32, origin.y as f32])
}

/// When a report was made, in seconds on a monotonic clock of this crate's own.
///
/// The performance counter where there is one, because it is what carries the
/// sub-frame spacing that makes a report history worth reading — a run of reports all
/// stamped with the same millisecond is a run the fitter cannot tell the speed of.
///
/// `millis` is the coarse clock the same report also carries, and is what a machine
/// with no usable counter falls back to. Coarse is not the same as useless: a fitter
/// given a run of equal times sees a stroke made in no time at all, and one given no
/// times at all sees the same — so the fallback is the difference between a
/// velocity-driven brush behaving badly and behaving not at all.
#[expect(
    clippy::cast_precision_loss,
    reason = "a tick count; the loss is below a nanosecond for centuries of uptime"
)]
fn seconds(shared: &Shared, ticks: u64, millis: u32) -> f64 {
    if shared.hz > 0.0 && ticks > 0 {
        return ticks as f64 / shared.hz;
    }
    f64::from(millis) / 1000.0
}

/// The subclass, and the handle the window is holding a raw pointer to.
///
/// `Rc` rather than `Arc` because nothing here crosses a thread: the window procedure
/// runs on the window's own thread, and the frontend that drains the queue is the one
/// that made the window. An `Arc` would be claiming otherwise, which the reader's raw
/// function pointers make untrue anyway.
struct Hook {
    hwnd: HWND,
    /// The very pointer handed to `SetWindowSubclass` as its reference word, kept so
    /// the strong count it stands for can be given back.
    raw: *const Shared,
    shared: Rc<Shared>,
}

impl Drop for Hook {
    fn drop(&mut self) {
        // SAFETY: `subclass_proc` and `SUBCLASS_ID` are the pair installed in
        // `Backend::attach`, and `hwnd` is the window it was installed on. Removing a
        // subclass that is no longer there — because the window has already been
        // destroyed — is a `FALSE` return and nothing else.
        unsafe { RemoveWindowSubclass(self.hwnd, Some(subclass_proc), SUBCLASS_ID) };
        if let Reader::Wintab(context) = &self.shared.reader {
            // SAFETY: the context this backend opened, closed once, here.
            unsafe { context.close() };
        }
        // SAFETY: `raw` came from `Rc::into_raw` in `attach` and has been given to
        // nobody but the window. The subclass was removed on the line above, on the
        // thread that dispatches its messages, so no call can be in flight or begin.
        drop(unsafe { Rc::from_raw(self.raw) });
    }
}

pub struct Backend {
    hook: Option<Hook>,
}

impl Backend {
    pub fn attach(window: &impl HasWindowHandle) -> Self {
        let Some(hwnd) = hwnd_of(window) else {
            return Self { hook: None };
        };
        let mut ticks: i64 = 0;
        // SAFETY: the out parameter is a live local. The call cannot fail on any
        // Windows this binary runs on, and a zero is read as "no counter" rather than
        // trusted.
        let ok = unsafe { QueryPerformanceFrequency(&mut ticks) };
        // Wintab first, and Ink where no context opens — see the module note.
        let reader = if forced_ink() {
            trace!("STARK_PEN=ink, so the pointer messages it is");
            Reader::Ink
        } else {
            // SAFETY: `hwnd` is a live window, which is all opening a context needs.
            match unsafe { wintab::Context::open(hwnd) } {
                Some(context) => Reader::Wintab(context),
                None => {
                    trace!("no wintab context, falling back to the pointer messages");
                    Reader::Ink
                }
            }
        };
        let shared = Rc::new(Shared {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a tick rate is ~10^7; f64 is exact to 2^53"
            )]
            hz: if ok != 0 && ticks > 0 {
                ticks as f64
            } else {
                0.0
            },
            reader,
            state: Mutex::new(State {
                claim: Claim::NONE,
                queue: Vec::new(),
                owned: false,
                deferred: false,
                pointer: None,
                stylus_near: false,
                last: None,
                ink: ink::Cache::default(),
                packets: wintab::Packets::default(),
            }),
        });
        let raw = Rc::into_raw(Rc::clone(&shared));
        // SAFETY: `raw` is a live `Rc` allocation whose strong count this call takes
        // over; `Hook::drop` removes the subclass and gives that count back, and
        // nothing else ever reads the word. The procedure has the signature
        // `SUBCLASSPROC` requires.
        let installed =
            unsafe { SetWindowSubclass(hwnd, Some(subclass_proc), SUBCLASS_ID, raw as usize) };
        if installed == 0 {
            // SAFETY: the count taken above was not taken over after all — the window
            // holds no copy of `raw`, so this is the only owner left.
            drop(unsafe { Rc::from_raw(raw) });
            return Self { hook: None };
        }
        Self {
            hook: Some(Hook { hwnd, raw, shared }),
        }
    }

    pub fn attached(&self) -> bool {
        self.hook.is_some()
    }

    pub fn claim(&self, claim: Claim) {
        if let Some(hook) = &self.hook {
            lock(&hook.shared).claim = claim;
        }
    }

    pub fn drain(&self, out: &mut Vec<Report>) {
        if let Some(hook) = &self.hook {
            let before = out.len();
            out.append(&mut lock(&hook.shared).queue);
            if out.len() > before {
                trace!("frontend drained {} report(s)", out.len() - before);
            }
        }
    }
}

/// Whether the environment has asked for the Ink reader outright.
///
/// **A diagnostic, not a setting.** Which API a machine should use is a question this
/// crate answers by trying, and there is no user-facing switch on purpose — but the
/// two readers are hard to tell apart from the outside, and when a stylus misbehaves
/// the first thing worth knowing is whether the other one misbehaves too. An
/// environment variable is how that question gets asked without a dialog, a record and
/// a command that three of the four frontends would have nothing to do with.
///
/// Only this direction: automatic already prefers Wintab, so forcing it would ask for
/// what it does anyway.
fn forced_ink() -> bool {
    std::env::var_os("STARK_PEN").is_some_and(|value| value.eq_ignore_ascii_case("ink"))
}

fn hwnd_of(window: &impl HasWindowHandle) -> Option<HWND> {
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::Win32(handle) => Some(handle.hwnd.get()),
        _ => None,
    }
}

/// The chained window procedure.
///
/// Reached for every message the window gets, so the cheap rejection comes first: a
/// message number no branch below wants is forwarded before anything is read or
/// locked.
unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    data: usize,
) -> LRESULT {
    let chain = || {
        // SAFETY: forwarding the message this procedure was called with, to whatever
        // stands behind this subclass — which is what a subclass that declines to
        // handle something is required to do.
        unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
    };
    let pointer_message = matches!(
        msg,
        WM_POINTERDOWN | WM_POINTERUPDATE | WM_POINTERUP | WM_POINTERCAPTURECHANGED
    );
    let mouse_message = matches!(
        msg,
        WM_MOUSEMOVE | WM_LBUTTONDOWN | WM_LBUTTONUP | WM_LBUTTONDBLCLK
    );
    if !pointer_message && !mouse_message && !wintab::might_be_packet(msg) {
        return chain();
    }
    // SAFETY: `data` is the reference word given to `SetWindowSubclass` in `attach`,
    // which is an `Rc<Shared>` whose strong count the window holds. `Hook::drop`
    // removes this subclass before giving that count back, and it does so on this same
    // thread — the one Windows dispatches messages on — so the allocation is live for
    // the whole of this call and the shared borrow cannot outlive it.
    let shared = unsafe { &*(data as *const Shared) };

    if let Reader::Wintab(context) = &shared.reader
        && msg == context.proximity_message()
    {
        wintab::proximity(shared, lparam);
        return chain();
    }
    if let Reader::Wintab(context) = &shared.reader
        && msg == context.packet_message()
    {
        // SAFETY: the context this backend opened, asked for the packets the message
        // says it has.
        let outcome = unsafe { wintab::take(shared, context, hwnd) };
        if outcome.queued {
            // SAFETY: `hwnd` is this window.
            unsafe { wake(hwnd) };
        }
        return chain();
    }
    let outcome = if pointer_message {
        // SAFETY: a pointer id out of a pointer message; every call inside reads into
        // locals and is checked.
        unsafe { ink::pointer(shared, hwnd, msg, wparam) }
    } else if !mouse_message {
        // A Wintab notification that is not the packet one — a context opening, a
        // cursor changing. **Not a mouse message**, which is what the branch below
        // would have read it as: it would have taken two words of `lParam` that mean
        // something else entirely for a position, and asked whether to swallow a
        // message on the strength of them.
        trace!("wintab notification {msg:#06x}, passed on");
        Outcome::PASS
    } else {
        // A mouse message matters only where a *stylus* made it and a reader is
        // already carrying that stylus. Asking the signature first is also what keeps
        // an actual mouse out of all of this.
        //
        // SAFETY: asked of the message this procedure was called with.
        let at = mouse_at(lparam);
        let state = lock(shared);
        // SAFETY: asked of the message this procedure was called with.
        let shadow = unsafe { is_shadow(&state) };
        let swallow = shadow && suppressed(&state, at);
        if msg != WM_MOUSEMOVE {
            trace!(
                "mouse msg {msg:#06x} at {at:?} shadow={shadow} near={} owned={} swallow={swallow}",
                state.stylus_near, state.owned
            );
        }
        drop(state);
        Outcome {
            queued: false,
            swallow,
        }
    };

    if outcome.queued {
        // SAFETY: `hwnd` is this window.
        unsafe { wake(hwnd) };
    }
    if outcome.swallow {
        // Handled here, so `DefWindowProc` is never reached and Windows synthesizes no
        // mouse message from this report. That is the whole of the double-input fix
        // (the module note).
        return 0;
    }
    if pointer_message {
        // **Not `chain`**: winit stands behind this subclass and would consume the
        // message without letting `DefWindowProc` synthesize anything, which is what
        // would leave a pen unable to press a button (the module note).
        //
        // SAFETY: the message this procedure was called with, handed to the default
        // handler — which is what the chain would have reached had winit not been in
        // it.
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }
    chain()
}
