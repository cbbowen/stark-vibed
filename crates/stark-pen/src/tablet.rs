//! The one type a frontend holds, and the one seam a backend fills.

use raw_window_handle::HasWindowHandle;

use crate::model::{Claim, Report};

#[cfg(not(windows))]
use crate::none::Backend;
#[cfg(windows)]
use crate::win32::Backend;

/// A stylus, attached to one window.
///
/// Held for as long as the window is, and dropped before it: the drop is what takes
/// the backend back out of the window's message path, and a window outliving its
/// tablet is the ordinary shutdown order rather than a thing to guard against.
///
/// **Every method takes `&self`**, including the two that plainly mutate. That is not
/// interior mutability for its own sake: a backend hears from the platform on the
/// platform's schedule and not on the caller's, so the state is behind a lock either
/// way — and a `&mut self` here would put a borrow of the tablet in the middle of a
/// frontend's frame, where it would collide with the view it is publishing a claim
/// *about*.
pub struct Tablet(Backend);

impl Tablet {
    /// Attach to `window`, or to nothing if this platform has no backend or the
    /// window will not name itself.
    ///
    /// **Never fails.** A missing tablet is the ordinary case — most machines have
    /// no digitizer, and every non-Windows host has no backend at all — so it is a
    /// state rather than an error: the frontend goes on with the mouse path it
    /// already had, and `attached` is there for anything that wants to say so out
    /// loud.
    #[must_use]
    pub fn attach(window: &impl HasWindowHandle) -> Self {
        Self(Backend::attach(window))
    }

    /// Whether a backend is in place. False on every host but Windows.
    ///
    /// Not the same question as *is a pen plugged in*, which nothing here asks: the
    /// pointer messages this crate reads arrive when a stylus is used and never
    /// otherwise, so a machine with no digitizer is one that reports nothing rather
    /// than one that has to be detected.
    #[must_use]
    pub fn attached(&self) -> bool {
        self.0.attached()
    }

    /// Say where a stylus press opens a gesture, and whether it may at all.
    ///
    /// Published every frame from the layout the frontend actually produced, for the
    /// reason the frontend's own `panel::Regions` gives: a rectangle derived twice is
    /// a rectangle that can disagree with itself, and the half that would be wrong
    /// here is the half that decides whether a button can be pressed.
    pub fn claim(&self, claim: Claim) {
        self.0.claim(claim);
    }

    /// Take everything the stylus has reported since the last call, oldest first,
    /// appending to `out`.
    ///
    /// **Appending rather than returning**, so a frontend can keep one buffer and
    /// spend no allocation per frame — this runs on the frame path, and at a
    /// digitizer's rate there is a report every few milliseconds.
    ///
    /// A drain per frame is not a rate limit on the samples: the reports made between
    /// two frames are all here, each with its own timestamp and pose. That is the
    /// same bargain the web frontend makes with `getCoalescedEvents`, and it is what
    /// gets the hand's full detail to the fitter rather than the display's.
    pub fn drain(&self, out: &mut Vec<Report>) {
        self.0.drain(out);
    }
}
