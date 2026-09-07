//! Windows Ink: poses read off the pointer messages the window already receives.
//!
//! The default reader, and the one that needs nothing installed — a Surface pen, a
//! Huion in HID mode, any Windows-certified digitizer. What it cannot do is report
//! pressure when a Wacom control panel has *Use Windows Ink* unchecked, which is what
//! [`super::wintab`] is for.
//!
//! `GetPointerPenInfoHistory` is why a frame's worth of motion here is not one
//! straight line: Windows delivers roughly one pointer message per frame and hangs the
//! reports it withheld — most of what a 240 Hz stylus produces — off it as a history.
//! It is the same list the web frontend reads out of `getCoalescedEvents`, and reading
//! only the message would cap every stroke at display rate whatever the hardware
//! resolved.

use windows_sys::Win32::Foundation::{HANDLE, HWND, RECT, WPARAM};
use windows_sys::Win32::UI::Input::Pointer::{
    GetPointerDeviceRects, GetPointerPenInfo, GetPointerPenInfoHistory, GetPointerType,
    POINTER_INFO, POINTER_PEN_INFO,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, PEN_FLAG_INVERTED, PEN_MASK_PRESSURE, PEN_MASK_TILT_X, PEN_MASK_TILT_Y,
    POINTER_INPUT_TYPE, PT_PEN, WM_POINTERCAPTURECHANGED, WM_POINTERDOWN, WM_POINTERUP,
    WM_POINTERUPDATE,
};

use super::trace::trace;
use super::{Outcome, Reader, Shared, State, client_origin, lock, push, seconds, suppressed};
use crate::model::{Phase, Pose, Rect, agrees, map_device};

/// The `pressure` field's full scale, per `POINTER_PEN_INFO`.
const PRESSURE_FULL: f32 = 1024.0;

/// The `tiltX`/`tiltY` fields' full scale in degrees, per `POINTER_PEN_INFO` — a right
/// angle, which is what makes the quotient the same number the web frontend hands the
/// engine (`tiltX / 90`).
const TILT_FULL: f32 = 90.0;

/// The largest report history a single message is believed to carry.
///
/// A cap on an allocation sized by a number the platform hands back, which is the only
/// reason it exists. A 240 Hz stylus under a 60 Hz frame puts four reports in a message
/// and a slow frame perhaps a few dozen.
const HISTORY_LIMIT: u32 = 512;

/// A pointer device's two rectangles, kept for as long as the same device keeps
/// reporting.
///
/// Cached because they are a property of the hardware and the desktop layout rather
/// than of a report, and asked for again whenever a different device speaks — which is
/// also what picks up a display rearrangement, since a device that has been re-mapped
/// reports through a new handle.
struct Device {
    handle: HANDLE,
    device: Rect,
    display: Rect,
}

/// What this reader keeps between messages.
pub struct Cache {
    device: Option<Device>,
    /// Whether the high-resolution mapping is still believed (see [`refined`]).
    trust_device: bool,
    /// Reused between messages so a report list costs no allocation.
    infos: Vec<POINTER_PEN_INFO>,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            device: None,
            trust_device: true,
            infos: Vec::new(),
        }
    }
}

/// Answer one pointer message.
///
/// Runs whichever reader is in use: under Wintab it takes no poses and only decides
/// whether the message is suppressed, since the pointer messages are then the stylus's
/// *shadow* rather than its report, and letting them reach `DefWindowProc` is what
/// would put a second stroke under the first.
pub unsafe fn pointer(shared: &Shared, hwnd: HWND, msg: u32, wparam: WPARAM) -> Outcome {
    let id = pointer_id(wparam);
    // SAFETY: a pointer id out of a pointer message. A stale one answers `FALSE` and
    // is read as "not a pen", which is the safe direction: the message is forwarded.
    let pen = unsafe { is_pen(id) };
    if msg != WM_POINTERUPDATE {
        trace!("pointer msg {msg:#06x} id={id} pen={pen}");
    }
    if !pen {
        return Outcome::PASS;
    }
    if !matches!(shared.reader, Reader::Ink) {
        // SAFETY: `id` names a live stylus; the pose is read to place it and nothing
        // is queued from it.
        return unsafe { shadow(shared, hwnd, id) };
    }
    // SAFETY: `hwnd` is this window and `id` a live pointer; every call inside reads
    // into locals and is checked.
    unsafe { take(shared, hwnd, msg, id) }
}

/// A pointer message under a reader that is not this one: swallowed where the contact
/// is ours, so that no mouse message is made from it, and passed on everywhere else so
/// that one is.
unsafe fn shadow(shared: &Shared, hwnd: HWND, id: u32) -> Outcome {
    let mut info = zeroed_info();
    // SAFETY: the out parameter is a live local.
    if unsafe { GetPointerPenInfo(id, &mut info) } == 0 {
        return Outcome::PASS;
    }
    let mut state = lock(shared);
    // SAFETY: `info` was filled by the call above and `hwnd` is this window.
    let Some(pose) = (unsafe { pose_of(shared, &mut state, hwnd, &info) }) else {
        return Outcome::PASS;
    };
    Outcome {
        queued: false,
        swallow: suppressed(&state, pose.position),
    }
}

/// Read one pointer message, queue what it reported, and answer whether this reader
/// handled it.
unsafe fn take(shared: &Shared, hwnd: HWND, msg: u32, id: u32) -> Outcome {
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
                return Outcome::PASS;
            }
            let mut info = zeroed_info();
            // SAFETY: `id` is a live pointer and the out parameter is a live local.
            if unsafe { GetPointerPenInfo(id, &mut info) } == 0 {
                return Outcome::PASS;
            }
            // SAFETY: `info` was filled by the call above and `hwnd` is this window.
            let Some(pose) = (unsafe { pose_of(shared, &mut state, hwnd, &info) }) else {
                return Outcome::PASS;
            };
            trace!(
                "ink down at {:?} claim={:?} takes={}",
                pose.position,
                state.claim,
                state.claim.takes(pose.position)
            );
            if !state.claim.takes(pose.position) {
                // The chrome's press, and its whole contact: latched so that the rest
                // of it goes on reaching the chrome even where it strays over the
                // canvas.
                state.deferred = true;
                return Outcome::PASS;
            }
            state.owned = true;
            state.pointer = Some(id);
            push(&mut state, Phase::Down, pose);
            Outcome::TAKEN
        }
        WM_POINTERUPDATE if state.pointer == Some(id) => {
            // Taken out of the state so the poses can be pushed back into the queue
            // beside it; put back at the end, which is what keeps the buffer.
            let mut infos = std::mem::take(&mut state.ink.infos);
            // SAFETY: `id` is a live pointer this reader owns the contact of.
            unsafe { read_history(id, &mut infos) };
            for info in &infos {
                // SAFETY: every entry was filled by `read_history` on the line above,
                // and `hwnd` is this window.
                if let Some(pose) = unsafe { pose_of(shared, &mut state, hwnd, info) } {
                    push(&mut state, Phase::Move, pose);
                }
            }
            infos.clear();
            state.ink.infos = infos;
            Outcome::TAKEN
        }
        WM_POINTERUP | WM_POINTERCAPTURECHANGED if state.pointer == Some(id) => {
            state.owned = false;
            state.pointer = None;
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
            // **A lift is queued whether or not the pose could be read.** The frontend
            // closes its gesture on this and on nothing else, and a stroke that never
            // closes is worse than one that closes where it last was.
            let pose = pose.or(state.last).unwrap_or_default();
            push(&mut state, Phase::Up, pose);
            // A capture that was taken away still has to reach the rest of the chain:
            // this file is reporting the loss, not consuming the notice of it.
            Outcome {
                queued: true,
                swallow: msg == WM_POINTERUP,
            }
        }
        WM_POINTERUP | WM_POINTERCAPTURECHANGED => {
            // Somebody else's contact ending — the chrome's, if it was deferred.
            state.deferred = false;
            Outcome::PASS
        }
        _ => Outcome::PASS,
    }
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

fn zeroed_info() -> POINTER_PEN_INFO {
    // SAFETY: `POINTER_PEN_INFO` is a plain C struct of integers, handles and points,
    // for which every field is valid at zero. It is an out parameter in every use.
    unsafe { std::mem::zeroed() }
}

/// Fill `infos` with every report Windows folded into the current message, **oldest
/// first**.
unsafe fn read_history(id: u32, infos: &mut Vec<POINTER_PEN_INFO>) {
    infos.clear();
    let mut count: u32 = 0;
    // SAFETY: a null buffer with a live count, which is how this call is asked how many
    // entries it has.
    if unsafe { GetPointerPenInfoHistory(id, &mut count, std::ptr::null_mut()) } == 0 {
        count = 0;
    }
    // A history of one is the message's own report, which is what a device slower than
    // the frame rate gives — and what a failed count above leaves.
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
    let screen = unsafe { refined(&mut state.ink, &pointer, rounded) };
    Some(Pose {
        position: [screen[0] - origin[0], screen[1] - origin[1]],
        pressure: if info.penMask & PEN_MASK_PRESSURE != 0 {
            trace_pressure(info);
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
        time: seconds(shared, pointer.PerformanceCount, pointer.dwTime),
        inverted: info.penFlags & PEN_FLAG_INVERTED != 0,
    })
}

/// Say what the device reported, now and then, so a stroke's worth of readings is a
/// line or two rather than a thousand.
fn trace_pressure(info: &POINTER_PEN_INFO) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NTH: AtomicU32 = AtomicU32::new(0);
    if NTH.fetch_add(1, Ordering::Relaxed).is_multiple_of(32) {
        trace!(
            "ink pose mask={:#06x} pressure={} tilt=({}, {})",
            info.penMask, info.pressure, info.tiltX, info.tiltY
        );
    }
}

/// One tilt axis as a fraction of a right angle, or flat where the device says nothing.
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
/// **It is checked against the first, once, and given up for good if they disagree.**
/// The mapping rests on reading two rectangles the way this file believes they are
/// meant, across every digitizer anybody plugs in — and the failure it would otherwise
/// produce is a stroke that lands somewhere the cursor is not, which is far worse than
/// a stroke fitted to whole pixels.
unsafe fn refined(cache: &mut Cache, pointer: &POINTER_INFO, rounded: [f32; 2]) -> [f32; 2] {
    if !cache.trust_device {
        return rounded;
    }
    // SAFETY: `sourceDevice` is the handle this report came from.
    let Some(device) = (unsafe { device_rects(cache, pointer.sourceDevice) }) else {
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
    cache.trust_device = false;
    rounded
}

/// The reporting device's two rectangles, cached for as long as the same device keeps
/// reporting.
unsafe fn device_rects(cache: &mut Cache, handle: HANDLE) -> Option<(Rect, Rect)> {
    if let Some(cached) = &cache.device
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
    cache.device = Some(Device {
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
