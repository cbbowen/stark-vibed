//! The raw pointer reports of the stroke in hand, kept so a misfit seen in the app can
//! be dumped on release and replayed as a test (§6.2).
//!
//! **The whole of the `debug-unfrozen` gate lives here.** A shipping build carries
//! neither the samples nor the field that would hold them — `#[cfg]`, not a runtime
//! `cfg!` around a `Vec` that exists either way — and the two shapes are written side
//! by side so a caller never asks which build it is in.

use crate::command::InputSample;

/// Whether a captured pointer report opens a stroke or continues one. A named pair
/// rather than a `bool`, because a bare `true` at the call site says nothing about
/// which way round it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Capture {
    /// The press: forget the last stroke's samples first.
    Restart,
    /// A move within the stroke in hand.
    Continue,
}

#[cfg(feature = "debug-unfrozen")]
pub(super) struct DebugCapture {
    samples: Vec<InputSample>,
}

#[cfg(feature = "debug-unfrozen")]
impl DebugCapture {
    /// Nothing captured yet.
    pub(super) fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub(super) fn note(&mut self, capture: Capture, sample: InputSample) {
        if capture == Capture::Restart {
            self.samples.clear();
        }
        self.samples.push(sample);
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed, which
    /// the density policy and the freezing both key off — and no synthetic curve stands
    /// in for a real hand.
    pub(super) fn dump(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        // Positions *and* the pen channels: pressure sizes the brush and tilt steers
        // it, both fitted as their own least-squares channels, so a capture without
        // them cannot reproduce a fault in either.
        let mut lit = String::from("&[");
        for (i, s) in self.samples.iter().enumerate() {
            if i > 0 {
                lit.push(',');
            }
            lit.push_str(&format!(
                "[{:.2},{:.2},{:.3},{:.3},{:.3}]",
                s.pos.x, s.pos.y, s.pressure, s.tilt.x, s.tilt.y
            ));
        }
        lit.push(']');
        tracing::info!(
            samples = self.samples.len(),
            "raw stroke [x,y,pressure,tiltx,tilty]: {lit}"
        );
        self.samples.clear();
    }
}

#[cfg(not(feature = "debug-unfrozen"))]
pub(super) struct DebugCapture;

#[cfg(not(feature = "debug-unfrozen"))]
impl DebugCapture {
    pub(super) fn new() -> Self {
        Self
    }

    pub(super) fn note(&mut self, _capture: Capture, _sample: InputSample) {}

    pub(super) fn dump(&mut self) {}
}
