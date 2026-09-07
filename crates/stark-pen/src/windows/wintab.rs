//! Wintab: poses read off the packets a tablet driver posts to this window.
//!
//! # Why this exists beside a working Ink reader
//!
//! One row of the comparison decides it. With *Use Windows Ink* unchecked in a Wacom
//! control panel — which many artists do, because it is what turns off
//! press-and-hold-for-right-click and the ripple — the pointer messages carry **no
//! pressure at all**, and nothing tells the user why. Wintab is the vendor's own API
//! and answers either way, at the driver's full resolution and through the pressure
//! curve the user set in that same control panel.
//!
//! # Messages, not polling
//!
//! A context opened with `CXO::MESSAGES` posts a packet notification
//! ([`Context::packet_message`]) to the window that owns it, so this reader arrives on
//! the same window procedure the Ink one does and needs no timer and no thread. That is what makes the two interchangeable: everything
//! above them — the queue, the claim, the wake, the arbitration — does not know which
//! is running.
//!
//! Each message is a nudge rather than a payload: the whole queue is drained on every
//! one, so a burst that arrived between two frames is taken in a burst.
//!
//! # The library is loaded, never linked
//!
//! `Wintab32.dll` belongs to a *driver*, not to Windows. Linking it at build time
//! would make a tablet driver a build requirement and a machine without one unable to
//! compile this crate. It is loaded by name at startup instead, and its absence is the
//! ordinary case: [`Context::open`] answers `None` and the Ink reader takes over.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::HWND;
use wintab_lite::{AXIS, CXO, DVC, LOGCONTEXT, Packet, WTI, WTPKT};

use super::trace::trace;
use super::{Outcome, Shared, State, client_origin, lock, push};
use crate::model::{Phase, Pose, map_axis, tilt_from_orientation};

/// The message number a context is **asked** to post its packets at.
///
/// Wintab does not have one message number, it has a *base*: a context reports at
/// `lcMsgBase + n`, and a default context's base is whatever the driver felt like —
/// zero, in the one this crate read. So the base is stated rather than inherited, and
/// this is the conventional one: below `WM_APP` and above `WM_USER`, where winit's own
/// private messages are registered ones an order of magnitude higher, so nothing else
/// in this window's path wants the same number.
///
/// What the driver actually settled on comes back in the context it writes, and
/// [`Context::packet_message`] is that rather than this.
const WT_DEFBASE: u32 = wintab_lite::WT::PACKET;

/// How many message numbers a context may report at, from its base.
///
/// The pre-filter in the window procedure runs before the context is in hand, so it
/// asks the coarse question — *could this be Wintab's?* — and the exact one is asked
/// after. Wintab defines eight notifications above the base.
const WT_MESSAGES: u32 = 8;

/// Whether `msg` could be a packet notification, asked before the context is reachable.
pub fn might_be_packet(msg: u32) -> bool {
    (WT_DEFBASE..=WT_DEFBASE + WT_MESSAGES).contains(&msg)
}

/// How many packets a single message drains at most.
///
/// A bound on one stack buffer rather than a rate: the message arrives per packet, so
/// reaching this means the queue had backed up behind a frame that did not come, and
/// the next message takes the rest.
const DRAIN_LIMIT: usize = 256;

/// The largest queue to ask the driver for, in packets.
///
/// **Asking is not free, and this is the trap it sets.** A refused `WTQueueSizeSet`
/// does not leave the queue as it was — it *destroys* it, and a context whose
/// enlargement was refused has no queue at all: the notifications go on arriving and
/// every packet behind them is lost. Which is exactly what it looks like from the
/// outside — hundreds of "a packet is ready" messages and nothing to read.
///
/// So the ask is a descent rather than a request ([`grow_queue`]): halve until one is
/// accepted, because the last refusal is what has to be recovered from.
const QUEUE_PACKETS: i32 = 128;

/// The smallest queue worth having. Below this the descent gives up and leaves the
/// context with whatever the last attempt made of it.
const QUEUE_FLOOR: i32 = 8;

/// Which notification says the stylus has come into or gone out of range.
///
/// `WT_PROXIMITY`, whose low word is non-zero on the way in and zero on the way out.
/// Read for one purpose — see [`super::suppressed`]: a driver's own mouse messages do
/// not always carry the mark that says a stylus made them, so *is a stylus over the
/// tablet right now* is the question that replaces it.
const WT_PROXIMITY_OFFSET: u32 = 5;

type WtOpen = unsafe extern "C" fn(HWND, *mut LOGCONTEXT, i32) -> *mut c_void;
type WtClose = unsafe extern "C" fn(*mut c_void) -> i32;
type WtInfo = unsafe extern "C" fn(u32, u32, *mut c_void) -> u32;
type WtPacketsGet = unsafe extern "C" fn(*mut c_void, i32, *mut c_void) -> i32;
type WtQueueSizeSet = unsafe extern "C" fn(*mut c_void, i32) -> i32;

/// The scratch a drain reuses, so a message costs no allocation.
pub type Packets = Vec<Packet>;

/// An open Wintab context, and what is needed to read from it.
///
/// Every field is settled at [`open`](Self::open) and read-only afterwards, which is
/// why this lives beside the lock rather than inside it.
pub struct Context {
    close: WtClose,
    packets_get: WtPacketsGet,
    ctx: *mut c_void,
    /// What the driver settled on as this context's message base, read back out of the
    /// context it wrote rather than assumed to be what it was asked for.
    msg_base: u32,
    /// The tablet's own range on each axis, as an origin and a signed extent, and the
    /// screen range each maps onto — the second with its vertical **turned over**,
    /// because Wintab measures up the tablet and a window measures down the glass.
    tablet: [(f32, f32); 2],
    screen: [(f32, f32); 2],
    /// What the driver reports at rest and at full force, so a reading can be put on
    /// `0..=1`.
    pressure: (f32, f32),
    /// The packet units of a full turn and of a right angle, for the two orientation
    /// axes — derived from what the device declares rather than assumed, since the
    /// tenth-of-a-degree both usually use is a convention and not a rule.
    azimuth_turn: f32,
    altitude_right: f32,
}

impl Context {
    /// Open a context on `hwnd`, or answer `None` where there is no Wintab to open one
    /// with.
    ///
    /// `None` is the ordinary answer on most machines — no tablet driver, so no DLL —
    /// and is not an error anywhere: the caller falls back to the Ink reader.
    pub unsafe fn open(hwnd: HWND) -> Option<Self> {
        // Leaked deliberately, and for the process's life. Every function pointer
        // below is only valid while the library is mapped, and there is no moment at
        // which unloading a tablet driver's DLL out from under an open context would
        // be a good idea — so the library is never dropped and the pointers are
        // therefore never dangling.
        //
        // SAFETY: loading a library by name runs its initialization routine, which for
        // a tablet driver's own DLL is what it is for.
        let lib = Box::leak(Box::new(
            unsafe { libloading::Library::new("Wintab32.dll") }.ok()?,
        ));

        // SAFETY: each name is a documented export of `Wintab32.dll` with the
        // signature declared above, taken from `WINTAB.H`. A name the DLL does not
        // have answers `Err` and this gives up rather than calling anything.
        let (open, close, info, packets_get) = unsafe {
            (
                *lib.get::<WtOpen>(b"WTOpenA\0").ok()?,
                *lib.get::<WtClose>(b"WTClose\0").ok()?,
                *lib.get::<WtInfo>(b"WTInfoA\0").ok()?,
                *lib.get::<WtPacketsGet>(b"WTPacketsGet\0").ok()?,
            )
        };
        // Not every driver exports this one, and a smaller queue costs detail rather
        // than correctness — so it is optional in a way the four above are not.
        //
        // SAFETY: as above.
        let queue_size = unsafe { lib.get::<WtQueueSizeSet>(b"WTQueueSizeSet\0") }
            .ok()
            .map(|symbol| *symbol);

        // **Start from the default system context, and override only the output.** Its
        // input range is the part of the tablet the driver's own control panel has
        // mapped to the screen, which is a setting this crate has no business
        // re-deciding; what it does decide is the units the packets come back in.
        let mut ctx = LOGCONTEXT::default();
        // SAFETY: the out parameter is a live local of the type this category returns.
        if unsafe { info(WTI::DEFSYSCTX as u32, 0, cast(&mut ctx)) } == 0 {
            return None;
        }
        ctx.lcOptions |= CXO::MESSAGES;
        // **Stated, not inherited.** `CXO::MESSAGES` says *post me the packets* and
        // `lcMsgBase` says what number to post them at — and a default context's base
        // is whatever the driver put there, which in the one this crate read was zero.
        // A context that reports at message zero is one whose packets never arrive.
        ctx.lcMsgBase = WT_DEFBASE;
        ctx.lcPktData = WTPKT::all();
        // Everything absolute. `Packet` is laid out for exactly this and says so.
        ctx.lcPktMode = WTPKT::empty();
        ctx.lcMoveMask = WTPKT::all();
        ctx.lcBtnUpMask = ctx.lcBtnDnMask;
        // **The context's own output transform is not used**, and that is a decision
        // made against a driver rather than a preference. Asked to map the tablet onto
        // the screen with a *negative* vertical extent — the documented way to say
        // "and turn it over", since Wintab measures up the tablet and a window
        // measures down the glass — one driver accepted the value, reported it back
        // unchanged, and then pinned every packet's y at the origin. A constant.
        //
        // So the packets come back in the tablet's own units, which is the identity
        // transform and the one thing every driver can be asked for, and the mapping
        // onto the screen is [`crate::model::map_axis`] — which is a dozen lines, is
        // unit-tested including the flip, and cannot be quietly declined.
        //
        // Native units are also finer than any sub-pixel scheme worth asking for: this
        // tablet reports some 52000 of them across, against 8760 screen px.
        ctx.lcOutOrgXYZ = ctx.lcInOrgXYZ;
        ctx.lcOutExtXYZ = ctx.lcInExtXYZ;

        // SAFETY: `hwnd` is a live window and the context is a live local the call
        // reads and writes back.
        let handle = unsafe { open(hwnd, &mut ctx, 1) };
        if handle.is_null() {
            trace!("wintab is present but would not open a context");
            return None;
        }
        if let Some(set) = queue_size {
            // SAFETY: the context just opened, and the descent recovers from its own
            // refusals — which is the whole reason it is a descent.
            let settled = unsafe { grow_queue(set, handle) };
            trace!("wintab queue size settled at {settled}");
        }

        // The device the context is *on*, rather than the first one plugged in: the
        // two are the same on a machine with one tablet and are not on a machine with
        // two, and reading the wrong device's pressure range is how a stroke saturates
        // at a fraction of full force.
        //
        // SAFETY: each out parameter is a live local of the type its index returns.
        let (pressure, azimuth_turn, altitude_right) = unsafe { axes(info, ctx.lcDevice) };
        trace!(
            "wintab open: msg_base={:#06x} device={} pressure={pressure:?} \
             azimuth_turn={azimuth_turn} altitude_right={altitude_right}",
            ctx.lcMsgBase, ctx.lcDevice
        );
        trace!(
            "wintab granted: in_org={:?} in_ext={:?} out_org={:?} out_ext={:?} \
             sys_org={:?} sys_ext={:?}",
            (ctx.lcInOrgXYZ.x, ctx.lcInOrgXYZ.y),
            (ctx.lcInExtXYZ.x, ctx.lcInExtXYZ.y),
            (ctx.lcOutOrgXYZ.x, ctx.lcOutOrgXYZ.y),
            (ctx.lcOutExtXYZ.x, ctx.lcOutExtXYZ.y),
            (ctx.lcSysOrgXY.x, ctx.lcSysOrgXY.y),
            (ctx.lcSysExtXY.x, ctx.lcSysExtXY.y),
        );
        #[expect(
            clippy::cast_precision_loss,
            reason = "a tablet or screen extent; f32 is exact well past either"
        )]
        let mapping = {
            let (org, ext) = (ctx.lcInOrgXYZ, ctx.lcInExtXYZ);
            let (sys_org, sys_ext) = (ctx.lcSysOrgXY, ctx.lcSysExtXY);
            (
                [(org.x as f32, ext.x as f32), (org.y as f32, ext.y as f32)],
                [
                    (sys_org.x as f32, sys_ext.x as f32),
                    // Bottom edge, counting upward: the flip, stated as the signed
                    // extent `map_axis` was written to take.
                    ((sys_org.y + sys_ext.y) as f32, -sys_ext.y as f32),
                ],
            )
        };
        Some(Self {
            close,
            packets_get,
            ctx: handle,
            msg_base: ctx.lcMsgBase,
            tablet: mapping.0,
            screen: mapping.1,
            pressure,
            azimuth_turn,
            altitude_right,
        })
    }

    /// The message this context posts a packet notification at.
    pub fn packet_message(&self) -> u32 {
        self.msg_base
    }

    /// The message it posts when the stylus comes into or leaves range.
    pub fn proximity_message(&self) -> u32 {
        self.msg_base + WT_PROXIMITY_OFFSET
    }

    /// Close the context. Called once, from the hook's drop.
    pub unsafe fn close(&self) {
        // SAFETY: the context this type opened, closed once.
        unsafe { (self.close)(self.ctx) };
    }
}

/// Enlarge the context's packet queue, recovering from a refusal.
///
/// Answers the size that was accepted, or zero where none was — which is a context
/// that will report notifications and hold nothing, and is worth saying out loud
/// because it looks from every other angle like a driver that has stopped sending.
unsafe fn grow_queue(set: WtQueueSizeSet, ctx: *mut c_void) -> i32 {
    let mut want = QUEUE_PACKETS;
    while want >= QUEUE_FLOOR {
        // SAFETY: the context this type opened, and a size it may refuse.
        if unsafe { set(ctx, want) } != 0 {
            return want;
        }
        want /= 2;
    }
    0
}

/// What the reporting device declares about the three axes this crate reads.
///
/// **Derived rather than assumed.** Pressure is commonly 0..1023 and the orientation
/// axes commonly tenths of a degree, but those are conventions rather than guarantees,
/// and a driver that reports 0..8191 through an assumed 1023 would saturate at an
/// eighth of full force — which reads exactly like a pressure curve that is far too
/// steep, and is the kind of wrong that is easy to mistake for a preference.
/// **`lcDevice` is not always an index**, which is the reason this asks twice. The
/// device categories are multiplexed — the constant names the first device and the
/// rest follow one at a time — but a driver may leave a default context's `lcDevice`
/// as a "whichever device" sentinel rather than a number, and adding *that* to the
/// constant is an overflow rather than a question. Asking is therefore cheaper than
/// deciding: a category that answers a degenerate pressure range was the wrong one,
/// and device zero is what a machine with one tablet has anyway.
///
/// A device that answers nothing on either try reports no range at all, which
/// [`pose_of`] reads as full pressure — a stylus that draws like a mouse rather than
/// one that draws like nothing.
unsafe fn axes(info: WtInfo, device: u32) -> ((f32, f32), f32, f32) {
    let first = WTI::DEVICES as u32;
    let asked = first.checked_add(device);
    for category in asked.into_iter().chain(std::iter::once(first)) {
        let mut pressure = AXIS::default();
        // SAFETY: the out parameter is a live local of the type this index returns.
        unsafe { info(category, DVC::NPRESSURE as u32, cast(&mut pressure)) };
        if pressure.axMax <= pressure.axMin {
            continue;
        }
        let mut orientation = [AXIS::default(); 3];
        // SAFETY: the out parameter is three of them, which is what this index returns.
        unsafe {
            info(
                category,
                DVC::ORIENTATION as u32,
                orientation.as_mut_ptr().cast(),
            )
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "axis bounds are small integers; f32 is exact well past them"
        )]
        let span = |axis: &AXIS| (axis.axMin as f32, axis.axMax as f32);
        let (pressure_min, pressure_max) = span(&pressure);
        let (azimuth_min, azimuth_max) = span(&orientation[0]);
        let (_, altitude_max) = span(&orientation[1]);
        return (
            (pressure_min, pressure_max),
            // The span *is* the turn: azimuth is cyclic, so its greatest value is
            // the direction it started from come back round.
            azimuth_max - azimuth_min,
            altitude_max,
        );
    }
    ((0.0, 0.0), 0.0, 0.0)
}

/// Note that the stylus has come into or gone out of range.
///
/// The low word of a proximity notification's `lParam` is non-zero on the way in.
pub fn proximity(shared: &Shared, lparam: isize) {
    let near = lparam & 0xffff != 0;
    trace!("wintab proximity near={near}");
    let mut state = lock(shared);
    state.stylus_near = near;
    if !near {
        // A stylus out of range is holding nothing, whatever the last packet implied.
        state.deferred = false;
    }
}

/// Drain the context's queue and queue what it reported.
pub unsafe fn take(shared: &Shared, context: &Context, hwnd: HWND) -> Outcome {
    let mut state = lock(shared);
    let mut packets = std::mem::take(&mut state.packets);
    packets.resize(DRAIN_LIMIT, Packet::default());
    // SAFETY: the buffer has `DRAIN_LIMIT` entries and that is what the call is told
    // it has; it writes no more, and answers how many it wrote.
    let read = unsafe {
        (context.packets_get)(
            context.ctx,
            DRAIN_LIMIT as i32,
            packets.as_mut_ptr().cast::<c_void>(),
        )
    };
    let read = read.max(0) as usize;
    if read > 0 {
        trace!("wintab drained {read} packet(s)");
    }
    let mut queued = false;
    for packet in packets.iter().take(read) {
        // SAFETY: `hwnd` is this window.
        let Some(pose) = (unsafe { pose_of(context, hwnd, packet) }) else {
            continue;
        };
        queued |= step(&mut state, packet, pose);
    }
    packets.clear();
    state.packets = packets;
    Outcome {
        queued,
        // The message is a notification, not an input event: nothing downstream is
        // going to act on it, and answering it as handled would be claiming something
        // this file did not do. What suppression there is happens on the mouse and
        // pointer messages the same stylus also makes.
        swallow: false,
    }
}

/// Turn one packet into a queued report, if it is one — answering whether anything was
/// queued.
///
/// Where the gesture's edges come from. A Wintab packet says where the stylus is and
/// whether its tip switch is closed; a press, a move and a lift are the *changes* in
/// that, which is why this is a step over state rather than a mapping.
fn step(state: &mut State, packet: &Packet, pose: Pose) -> bool {
    // The tip switch, or any pressure at all where a driver reports the two
    // differently: a packet carrying force is a nib on the glass whatever the button
    // bitmask says, and of the two mistakes an unopened stroke is the worse one.
    let contact = packet.pkButtons.0 & 1 != 0 || packet.pkNormalPressure > 0;
    trace_packet(packet, pose, contact);
    // A packet is itself the proof that a stylus is over the tablet.
    state.stylus_near = true;
    let held = state.owned || state.deferred;
    match (held, contact) {
        (false, true) => {
            if state.claim.takes(pose.position) {
                state.owned = true;
                push(state, Phase::Down, pose);
                true
            } else {
                // The chrome's contact, and the whole of it: latched so that the rest
                // goes on reaching the chrome even where it strays over the canvas.
                state.deferred = true;
                false
            }
        }
        (true, true) if state.owned => {
            push(state, Phase::Move, pose);
            true
        }
        (true, false) => {
            let was = state.owned;
            state.owned = false;
            state.deferred = false;
            if was {
                push(state, Phase::Up, pose);
            }
            was
        }
        // A hover, or the motion of a contact the chrome is carrying. Remembered
        // either way, so that a lift whose own packet is missed has somewhere to land.
        _ => {
            state.last = Some(pose);
            false
        }
    }
}

/// Say what a packet held, now and then, so a stroke's worth is a line or two rather
/// than a thousand — and always for a press or a lift, which are the two that decide
/// whether a gesture happened at all.
fn trace_packet(packet: &Packet, pose: Pose, contact: bool) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NTH: AtomicU32 = AtomicU32::new(0);
    static WAS: AtomicU32 = AtomicU32::new(0);
    let edge = WAS.swap(u32::from(contact), Ordering::Relaxed) != u32::from(contact);
    if edge || NTH.fetch_add(1, Ordering::Relaxed).is_multiple_of(32) {
        trace!(
            "wintab packet buttons={:#010x} contact={contact} raw_pressure={} \
             raw_xy=({}, {}) pose=({:.1}, {:.1}) p={:.3} tilt=({:.2}, {:.2})",
            packet.pkButtons.0,
            packet.pkNormalPressure,
            packet.pkXYZ.x,
            packet.pkXYZ.y,
            pose.position[0],
            pose.position[1],
            pose.pressure,
            pose.tilt[0],
            pose.tilt[1]
        );
    }
}

/// Turn one packet into a [`Pose`], in the units the engine measures a stroke in.
unsafe fn pose_of(context: &Context, hwnd: HWND, packet: &Packet) -> Option<Pose> {
    // SAFETY: `hwnd` is this window.
    let origin = unsafe { client_origin(hwnd) }?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a tablet coordinate; f32 is exact well past any digitizer's range"
    )]
    let raw = [packet.pkXYZ.x as f32, packet.pkXYZ.y as f32];
    let screen = [
        map_axis(raw[0], context.tablet[0], context.screen[0])?,
        map_axis(raw[1], context.tablet[1], context.screen[1])?,
    ];

    let (min, max) = context.pressure;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a pressure reading; f32 is exact past any driver's range"
    )]
    let raw = packet.pkNormalPressure as f32;
    let pressure = map_axis(raw, (min, max - min), (0.0, 1.0))
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    #[expect(
        clippy::cast_precision_loss,
        reason = "an orientation reading; f32 is exact past any driver's range"
    )]
    let azimuth = packet.pkOrientation.orAzimuth as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "an orientation reading; f32 is exact past any driver's range"
    )]
    let altitude = packet.pkOrientation.orAltitude as f32;
    let tilt = if context.azimuth_turn > 0.0 && context.altitude_right > 0.0 {
        tilt_from_orientation(
            azimuth / context.azimuth_turn * std::f32::consts::TAU,
            altitude.abs() / context.altitude_right * std::f32::consts::FRAC_PI_2,
        )
    } else {
        // A stylus with no orientation to report lies flat in the engine's terms,
        // which is what a mouse reports too.
        [0.0, 0.0]
    };

    Some(Pose {
        position: [screen[0] - origin[0], screen[1] - origin[1]],
        pressure,
        tilt,
        time: f64::from(packet.pkTime) / 1000.0,
        inverted: packet.pkStatus.contains(wintab_lite::TPS::INVERT),
    })
}

/// The `void *` Wintab takes for every out parameter.
fn cast<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast::<c_void>()
}
