//! Windows Ink, read off the pointer messages the window already receives.
//!
//! # Why these messages and not one of the other two
//!
//! Windows offers three ways to hear a stylus. **Wintab** is the vendor-standard one
//! and the only one that still reports pressure when a Wacom control panel has "Use
//! Windows Ink" unchecked — it belongs beside this file, behind a preference, and is
//! not written yet. **`RealTimeStylus`** is the legacy Ink API: a COM apartment, an
//! async plugin on a thread of its own, and Microsoft's own guidance that there is no
//! reason to use it when the pointer messages will do. These are the pointer
//! messages, and they are the cheapest of the three by some distance: they arrive on
//! the thread winit is already pumping, they need no driver, and
//! `GetPointerPenInfoHistory` is the same sub-frame report list the web frontend
//! reads out of `getCoalescedEvents`.
//!
//! # The subclass, and what returning zero means
//!
//! A window procedure is chained onto the window winit made (`SetWindowSubclass`),
//! which is the only way to see a message a toolkit does not forward. Every message
//! but a pen's is passed straight along.
//!
//! For a pen's, the return value is the whole design. Windows synthesizes mouse
//! messages from a stylus **because `DefWindowProc` was reached** — so a pointer
//! message this file answers itself generates no mouse message at all, and the double
//! input that every drawing app fights is not arbitrated, it is never created.
//! `Claim` decides which presses get that treatment: inside the rectangle the
//! frontend published, a contact is taken and the mouse path never hears about it;
//! outside it, nothing is touched and the pen goes on driving the chrome through the
//! same synthesized mouse messages it always did.
//!
//! Ownership is latched at the **press**, not per message: a stroke that starts on
//! the canvas and wanders over a panel is still one stroke, and asking the rectangle
//! again halfway through would hand the rest of it to the mouse.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows_sys::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::ClientToScreen;
use windows_sys::Win32::System::Performance::QueryPerformanceFrequency;
use windows_sys::Win32::UI::Input::Pointer::{
    GetPointerDeviceRects, GetPointerPenInfo, GetPointerPenInfoHistory, GetPointerType,
    POINTER_INFO, POINTER_PEN_INFO,
};
use windows_sys::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, PEN_FLAG_INVERTED, PEN_MASK_PRESSURE, PEN_MASK_TILT_X, PEN_MASK_TILT_Y,
    POINTER_INPUT_TYPE, PT_PEN, WM_POINTERCAPTURECHANGED, WM_POINTERDOWN, WM_POINTERUP,
    WM_POINTERUPDATE,
};

use crate::model::{Claim, Phase, Pose, Rect, Report, agrees, map_device};

/// The `pressure` field's full scale, per `POINTER_PEN_INFO`.
const PRESSURE_FULL: f32 = 1024.0;

/// The `tiltX`/`tiltY` fields' full scale in degrees, per `POINTER_PEN_INFO` — a
/// right angle, which is what makes the quotient the same number the web frontend
/// hands the engine (`tiltX / 90`).
const TILT_FULL: f32 = 90.0;

/// How many reports may wait for a frame that is not coming.
///
/// The queue drains once a frame and a digitizer reports every few milliseconds, so
/// this is only ever reached when the frame loop has stopped — a modal file dialog,
/// a window being dragged. What it bounds is the memory that would otherwise grow for
/// as long as that lasts. **Motion is what gets dropped and never a press or a
/// lift**, so a gesture that overruns loses detail and still opens and closes.
const QUEUE_LIMIT: usize = 4096;

/// The largest report history a single message is believed to carry.
///
/// A cap on an allocation sized by a number the platform hands back, which is the
/// only reason it exists. A 240 Hz stylus under a 60 Hz frame puts four reports in a
/// message and a slow frame perhaps a few dozen.
const HISTORY_LIMIT: u32 = 512;

/// Which subclass on the window is ours. Any value distinguishes it from another
/// library's; this one is `STRK`.
const SUBCLASS_ID: usize = 0x5354_524B;

/// A pointer device's two rectangles, kept for as long as the same device keeps
/// reporting.
///
/// Cached because they are a property of the hardware and the desktop layout rather
/// than of a report, and asked for again whenever a different device speaks — which
/// is also what picks up a display rearrangement, since a device that has been
/// re-mapped reports through a new handle.
struct Device {
    handle: HANDLE,
    device: Rect,
    display: Rect,
}

struct State {
    claim: Claim,
    queue: Vec<Report>,
    /// The pointer whose contact this file took at the press, if any.
    owned: Option<u32>,
    device: Option<Device>,
    /// Where the stylus was last seen, for a lift whose own pose cannot be read.
    ///
    /// **Not the queue's last entry**, which is the obvious place to look and the
    /// wrong one: the frontend drains the queue every frame, so by the time a lift
    /// arrives the queue is usually empty and reading it would land the closing
    /// sample at the window's origin.
    last: Option<Pose>,
    /// Whether the high-resolution mapping is still believed (see [`pose_of`]).
    trust_device: bool,
    /// Reused between messages so a report list costs no allocation.
    infos: Vec<POINTER_PEN_INFO>,
}

struct Shared {
    /// Performance-counter ticks per second, or zero where the counter is unusable —
    /// in which case timestamps come off the millisecond clock instead.
    hz: f64,
    state: Mutex<State>,
}

/// Lock the state, taking a poisoned lock as it stands.
///
/// **A window procedure may not panic**: unwinding out of an `extern "system"` call
/// is an abort at best, and the frame it would tear through belongs to winit. There
/// is nothing here an unwind could have left half-written anyway — the state is a
/// queue, a rectangle and two `Option`s — so the poisoned guard is exactly as good as
/// a fresh one.
fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    shared.state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The subclass, and the `Arc` the window is holding a raw pointer to.
struct Hook {
    hwnd: HWND,
    /// The very pointer handed to `SetWindowSubclass` as its reference word, kept so
    /// the strong count it stands for can be given back.
    raw: *const Shared,
    shared: Arc<Shared>,
}

impl Drop for Hook {
    fn drop(&mut self) {
        // SAFETY: `subclass_proc` and `SUBCLASS_ID` are the pair installed in
        // `Backend::attach`, and `hwnd` is the window it was installed on. Removing a
        // subclass that is no longer there — because the window has already been
        // destroyed — is a `FALSE` return and nothing else.
        unsafe { RemoveWindowSubclass(self.hwnd, Some(subclass_proc), SUBCLASS_ID) };
        // SAFETY: `raw` came from `Arc::into_raw` in `attach` and has been given to
        // nobody but the window. The subclass was removed on the line above, on the
        // thread that dispatches its messages, so no call can be in flight or begin.
        drop(unsafe { Arc::from_raw(self.raw) });
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
        // Windows this binary runs on, and a zero is read as "no counter" below
        // rather than trusted.
        let ok = unsafe { QueryPerformanceFrequency(&mut ticks) };
        let shared = Arc::new(Shared {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a tick rate is ~10^7; f64 is exact to 2^53"
            )]
            hz: if ok != 0 && ticks > 0 {
                ticks as f64
            } else {
                0.0
            },
            state: Mutex::new(State {
                claim: Claim::NONE,
                queue: Vec::new(),
                owned: None,
                device: None,
                last: None,
                trust_device: true,
                infos: Vec::new(),
            }),
        });
        let raw = Arc::into_raw(Arc::clone(&shared));
        // SAFETY: `raw` is a live `Arc` allocation whose strong count this call takes
        // over; `Hook::drop` removes the subclass and gives that count back, and
        // nothing else ever reads the word. The procedure has the signature
        // `SUBCLASSPROC` requires.
        let installed =
            unsafe { SetWindowSubclass(hwnd, Some(subclass_proc), SUBCLASS_ID, raw as usize) };
        if installed == 0 {
            // SAFETY: the count taken above was not taken over after all — the window
            // holds no copy of `raw`, so this is the only owner left.
            drop(unsafe { Arc::from_raw(raw) });
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
            out.append(&mut lock(&hook.shared).queue);
        }
    }
}

fn hwnd_of(window: &impl HasWindowHandle) -> Option<HWND> {
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::Win32(handle) => Some(handle.hwnd.get()),
        _ => None,
    }
}

/// The chained window procedure.
///
/// Reached for every message the window gets, so the cheap rejection comes first: all
/// but four message numbers are forwarded before anything is read or locked.
unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    data: usize,
) -> LRESULT {
    let pass = || {
        // SAFETY: forwarding the message this procedure was called with, to whatever
        // stands behind this subclass — which is what a subclass that declines to
        // handle something is required to do.
        unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
    };
    if !matches!(
        msg,
        WM_POINTERDOWN | WM_POINTERUPDATE | WM_POINTERUP | WM_POINTERCAPTURECHANGED
    ) {
        return pass();
    }
    // SAFETY: `data` is the reference word given to `SetWindowSubclass` in `attach`,
    // which is an `Arc<Shared>` whose strong count the window holds. `Hook::drop`
    // removes this subclass before giving that count back, and it does so on this
    // same thread — the one Windows dispatches messages on — so the allocation is
    // live for the whole of this call and the shared borrow cannot outlive it.
    let shared = unsafe { &*(data as *const Shared) };
    let id = pointer_id(wparam);
    // SAFETY: a pointer id out of a pointer message. A stale one answers `FALSE` and
    // is read as "not a pen", which is the safe direction: the message is forwarded.
    if !unsafe { is_pen(id) } {
        return pass();
    }
    // SAFETY: `hwnd` is this window and `id` a live pointer; every call inside reads
    // into locals and is checked.
    if unsafe { take(shared, hwnd, msg, id) } {
        // Handled here, so `DefWindowProc` is never reached and Windows synthesizes
        // no mouse message from this stylus report. That is the whole of the
        // double-input fix (the module note).
        return 0;
    }
    pass()
}

/// The pointer id in a pointer message's `wParam` — its low word.
fn pointer_id(wparam: WPARAM) -> u32 {
    (wparam & 0xffff) as u32
}

/// Whether `id` names a stylus rather than a finger or a mouse.
unsafe fn is_pen(id: u32) -> bool {
    let mut kind: POINTER_INPUT_TYPE = 0;
    // SAFETY: the out parameter is a live local.
    unsafe { GetPointerType(id, &mut kind) != 0 && kind == PT_PEN }
}

/// Read one pointer message, queue what it reported, and answer whether this file
/// handled it — which is to say, whether Windows should synthesize no mouse message
/// from it.
unsafe fn take(shared: &Shared, hwnd: HWND, msg: u32, id: u32) -> bool {
    let mut state = lock(shared);
    match msg {
        WM_POINTERDOWN => {
            // A press that brings the window forward is the platform's to handle:
            // swallowing it would activate nothing and leave the window unfocused
            // under a stroke. So the first press lands as a mouse press like any
            // other, and the pen takes over from the next one.
            //
            // SAFETY: no arguments.
            if unsafe { GetForegroundWindow() } != hwnd {
                return false;
            }
            let mut info = zeroed_info();
            // SAFETY: `id` is a live pointer and the out parameter is a live local.
            if unsafe { GetPointerPenInfo(id, &mut info) } == 0 {
                return false;
            }
            // SAFETY: `info` was filled by the call above and `hwnd` is this window.
            let Some(pose) = (unsafe { pose_of(shared, &mut state, hwnd, &info) }) else {
                return false;
            };
            if !state.claim.takes(pose.position) {
                return false;
            }
            state.owned = Some(id);
            push(&mut state, Phase::Down, pose);
            true
        }
        WM_POINTERUPDATE if state.owned == Some(id) => {
            // Taken out of the state so the poses can be pushed back into the queue
            // beside it; put back at the end, which is what keeps the buffer.
            let mut infos = std::mem::take(&mut state.infos);
            // SAFETY: `id` is a live pointer this file owns the contact of.
            unsafe { read_history(id, &mut infos) };
            for info in &infos {
                // SAFETY: every entry was filled by `read_history` on the line above,
                // and `hwnd` is this window.
                if let Some(pose) = unsafe { pose_of(shared, &mut state, hwnd, info) } {
                    push(&mut state, Phase::Move, pose);
                }
            }
            infos.clear();
            state.infos = infos;
            true
        }
        WM_POINTERUP | WM_POINTERCAPTURECHANGED if state.owned == Some(id) => {
            state.owned = None;
            let mut info = zeroed_info();
            // SAFETY: the out parameter is a live local. A pointer that has already
            // gone answers `FALSE`, which is why the lift below does not depend on
            // this succeeding.
            let pose = if unsafe { GetPointerPenInfo(id, &mut info) } != 0 {
                // SAFETY: `info` was filled by the call above.
                unsafe { pose_of(shared, &mut state, hwnd, &info) }
            } else {
                None
            };
            // **A lift is queued whether or not the pose could be read.** The
            // frontend closes its gesture on this and on nothing else, and a stroke
            // that never closes is worse than one that closes where it last was.
            let pose = pose.or(state.last).unwrap_or_default();
            push(&mut state, Phase::Up, pose);
            // A capture that was taken away still has to reach the rest of the chain:
            // this file is reporting the loss, not consuming the notice of it.
            msg == WM_POINTERUP
        }
        _ => false,
    }
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

fn zeroed_info() -> POINTER_PEN_INFO {
    // SAFETY: `POINTER_PEN_INFO` is a plain C struct of integers, handles and points,
    // for which every field is valid at zero. It is an out parameter in every use.
    unsafe { std::mem::zeroed() }
}

/// Fill `infos` with every report Windows folded into the current message, **oldest
/// first**.
///
/// This is the reason a frame's worth of stylus motion is not one straight line.
/// Windows delivers roughly one pointer message per frame and hangs the reports it
/// withheld — most of what a 240 Hz stylus produces — off it as a history. Reading
/// only the message caps every stroke at display rate whatever the hardware resolved,
/// which is the same trap `getCoalescedEvents` exists to keep the web frontend out
/// of.
unsafe fn read_history(id: u32, infos: &mut Vec<POINTER_PEN_INFO>) {
    infos.clear();
    let mut count: u32 = 0;
    // SAFETY: a null buffer with a live count, which is how this call is asked how
    // many entries it has.
    if unsafe { GetPointerPenInfoHistory(id, &mut count, std::ptr::null_mut()) } == 0 {
        count = 0;
    }
    // A history of one is the message's own report, which is what a device slower
    // than the frame rate gives — and what a failed count above leaves.
    count = count.clamp(1, HISTORY_LIMIT);
    infos.resize(count as usize, zeroed_info());
    // SAFETY: the buffer has `count` entries and `count` is what the call is told it
    // has; it writes no more, and updates `count` to what it wrote.
    if unsafe { GetPointerPenInfoHistory(id, &mut count, infos.as_mut_ptr()) } == 0 {
        infos.clear();
        return;
    }
    infos.truncate(count as usize);
    // Windows hands the history back newest first, and a stroke is drawn in the order
    // the hand made it.
    infos.reverse();
}

/// Turn one report into a [`Pose`], in the units the engine measures a stroke in.
unsafe fn pose_of(
    shared: &Shared,
    state: &mut State,
    hwnd: HWND,
    info: &POINTER_PEN_INFO,
) -> Option<Pose> {
    let pointer = info.pointerInfo;
    // The rounded reading, in screen px. Always available, and the thing the
    // high-resolution mapping below is checked against.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a screen coordinate; f32 is exact well past any display"
    )]
    let rounded = [
        pointer.ptPixelLocationRaw.x as f32,
        pointer.ptPixelLocationRaw.y as f32,
    ];
    // SAFETY: `hwnd` is this window; the point is a live local.
    let origin = unsafe { client_origin(hwnd) }?;
    // SAFETY: `pointer` is a report Windows filled, so its `sourceDevice` is a handle
    // this process may ask about.
    let screen = unsafe { refined(state, &pointer, rounded) };
    Some(Pose {
        position: [screen[0] - origin[0], screen[1] - origin[1]],
        pressure: if info.penMask & PEN_MASK_PRESSURE != 0 {
            #[expect(clippy::cast_precision_loss, reason = "0..=1024, exact in an f32")]
            let p = info.pressure as f32 / PRESSURE_FULL;
            p.clamp(0.0, 1.0)
        } else {
            // A stylus that reports no pressure draws like a mouse rather than like
            // nothing (§6.2: a mouse is always pressed home).
            1.0
        },
        tilt: [
            axis_tilt(info.penMask, PEN_MASK_TILT_X, info.tiltX),
            axis_tilt(info.penMask, PEN_MASK_TILT_Y, info.tiltY),
        ],
        time: stamp(shared, &pointer),
        inverted: info.penFlags & PEN_FLAG_INVERTED != 0,
    })
}

/// One tilt axis as a fraction of a right angle, or flat where the device says
/// nothing.
fn axis_tilt(mask: u32, bit: u32, degrees: i32) -> f32 {
    if mask & bit == 0 {
        return 0.0;
    }
    #[expect(clippy::cast_precision_loss, reason = "-90..=90, exact in an f32")]
    let t = degrees as f32 / TILT_FULL;
    t.clamp(-1.0, 1.0)
}

/// The high-resolution screen position if it can be had and believed, and the rounded
/// one otherwise.
///
/// The digitizer resolves far below the screen, and Windows says so twice: once as a
/// whole pixel and once as a raw reading in the device's own units, with the two
/// rectangles that relate them. Taking the second is what makes
/// `stark_ui::input::PEN_RESOLUTION`'s half-pixel claim true rather than aspirational.
///
/// **It is checked against the first, once, and given up for good if they
/// disagree.** The mapping rests on reading two rectangles the way this file believes
/// they are meant, across every digitizer anybody plugs in — and the failure it would
/// otherwise produce is a stroke that lands somewhere the cursor is not, which is far
/// worse than a stroke fitted to whole pixels.
unsafe fn refined(state: &mut State, pointer: &POINTER_INFO, rounded: [f32; 2]) -> [f32; 2] {
    if !state.trust_device {
        return rounded;
    }
    // SAFETY: `sourceDevice` is the handle this report came from.
    let Some(device) = (unsafe { device_rects(state, pointer.sourceDevice) }) else {
        return rounded;
    };
    #[expect(
        clippy::cast_precision_loss,
        reason = "a HIMETRIC device coordinate; f32 is exact past any digitizer's extent"
    )]
    let raw = [
        pointer.ptHimetricLocationRaw.x as f32,
        pointer.ptHimetricLocationRaw.y as f32,
    ];
    let Some(mapped) = map_device(raw, device.0, device.1) else {
        return rounded;
    };
    if agrees(mapped, rounded) {
        return mapped;
    }
    state.trust_device = false;
    rounded
}

/// The reporting device's two rectangles, cached for as long as the same device keeps
/// reporting.
unsafe fn device_rects(state: &mut State, handle: HANDLE) -> Option<(Rect, Rect)> {
    if let Some(cached) = &state.device
        && cached.handle == handle
    {
        return Some((cached.device, cached.display));
    }
    let mut device = zeroed_rect();
    let mut display = zeroed_rect();
    // SAFETY: `handle` came out of a pointer report and both out parameters are live
    // locals.
    if unsafe { GetPointerDeviceRects(handle, &mut device, &mut display) } == 0 {
        return None;
    }
    let device = rect_of(device);
    let display = rect_of(display);
    state.device = Some(Device {
        handle,
        device,
        display,
    });
    Some((device, display))
}

fn zeroed_rect() -> RECT {
    RECT {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a screen or device extent; f32 is exact well past either"
)]
fn rect_of(r: RECT) -> Rect {
    Rect {
        left: r.left as f32,
        top: r.top as f32,
        right: r.right as f32,
        bottom: r.bottom as f32,
    }
}

/// Where the window's client area begins on the screen, so a screen position can keep
/// its fraction on the way to a client one.
///
/// `ScreenToClient` is the usual call and cannot be used here: it works on whole
/// pixels, and rounding is what this file has just gone to some trouble to avoid.
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
/// sub-frame spacing that makes the history worth reading — a run of reports all
/// stamped with the same millisecond is a run the fitter cannot tell the speed of.
/// The millisecond clock is the fallback and is still monotonic.
fn stamp(shared: &Shared, pointer: &POINTER_INFO) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a tick count; the loss is below a nanosecond for centuries of uptime"
    )]
    if shared.hz > 0.0 && pointer.PerformanceCount > 0 {
        return pointer.PerformanceCount as f64 / shared.hz;
    }
    f64::from(pointer.dwTime) / 1000.0
}
