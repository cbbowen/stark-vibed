//! The backend for a platform that has none: it attaches to nothing and reports
//! nothing.
//!
//! Here so that [`Tablet`](crate::Tablet) has one shape everywhere and a frontend
//! needs no `cfg` of its own. A chrome on such a host is not degraded, it is where it
//! was: the mouse path fills a full-pressure sample exactly as it did before this
//! crate existed, and an empty drain is what says so.

use raw_window_handle::HasWindowHandle;

use crate::model::{Claim, Report};

pub struct Backend;

impl Backend {
    pub fn attach(_window: &impl HasWindowHandle) -> Self {
        Self
    }

    pub fn attached(&self) -> bool {
        false
    }

    pub fn claim(&self, _claim: Claim) {}

    /// Leaves `out` as it found it — **it does not clear it**, which is the same
    /// contract the real backend keeps: a caller that appends from two sources gets
    /// both.
    pub fn drain(&self, _out: &mut Vec<Report>) {}
}
