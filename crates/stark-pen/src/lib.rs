//! Stylus input for a native frontend: pressure, tilt, and the reports a digitizer
//! makes between two frames (§6.2, §11).
//!
//! # What this crate is for
//!
//! The engine has wanted a pen since it was written — [`InputSample`] carries
//! `pressure` and `tilt`, and the path fitter solves them on the same least-squares
//! ride as the geometry (`stark_engine::path::fit`). The web frontend fills them from
//! `PointerEvent`, coalesced reports and all. The native one could not: winit 0.30
//! reports a mouse and nothing else, so `stark-wgpui-frontend` was stating full
//! pressure and no tilt on every sample and the whole of §6.2's dynamics loop was
//! reachable there only through a knob.
//!
//! So this is the *second* answer to a question the web frontend already answered,
//! which is what makes it the shape it is: what comes out is normalized the way the
//! browser normalizes it, because the thing downstream is the same engine (§11.2).
//!
//! # How it works, and the one thing to know before reading a backend
//!
//! **A pen press this crate takes never reaches the frontend as a mouse event.**
//! Windows delivers a stylus twice — once as pointer messages and again, for
//! compatibility, as synthesized mouse messages — and an app that read both would
//! paint every stroke twice, once with real pressure and once at full. The usual fix
//! is a last-device-wins latch, which is a guess that can be wrong. This crate does
//! not guess: the platform generates the mouse messages *because* the app declined
//! the pointer message, so a backend that handles one suppresses the other by
//! construction, and there is no state anywhere that could disagree with itself.
//!
//! The cost is that the pen then reaches nothing else either — not a button, not the
//! title bar. Hence [`Claim`]: the frontend publishes the region where a press is
//! paint, and everywhere else the platform's own compatibility path is left alone and
//! the chrome goes on working exactly as it did. That rectangle is the entire
//! interface between "which device is this" and "what does a press mean here".
//!
//! # Platforms
//!
//! Windows only, through the Windows Ink pointer messages (`WM_POINTER*`), which is
//! the API a digitizer speaks with no vendor driver installed at all. Every other
//! host gets a backend that reports nothing, so a frontend needs no `cfg` of its own
//! and a mouse goes on working as it always did.
//!
//! Two gaps worth naming rather than discovering. **Wintab is not here**: an artist
//! who has unchecked "Use Windows Ink" in a Wacom control panel gets no pressure from
//! this crate, because with that box clear the driver reports none through Ink — the
//! backend would be a second one beside `win32`, behind a preference. **macOS is not
//! here either**: `NSEvent` carries pressure, tilt and rotation on a tablet event and
//! nobody has written that backend.
//!
//! [`InputSample`]: https://docs.rs/stark-engine

#[forbid(unsafe_code)]
mod model;

#[cfg(not(windows))]
#[forbid(unsafe_code)]
mod none;

#[forbid(unsafe_code)]
mod tablet;

// **The one module that may contain `unsafe`**, and the only one — a window
// procedure and six `extern "system"` calls are what a pen is on Windows, so this
// crate is the single place in the tree that does not take the workspace's
// `forbid(unsafe_code)` (its own `Cargo.toml` says why). Every module beside it
// carries its own `#[forbid(unsafe_code)]` above, which is the compiler keeping the
// confinement rather than a convention asking for it.
#[cfg(windows)]
mod win32;

pub use model::{Claim, DEVICE_AGREEMENT_PX, Phase, Pose, Rect, Report, agrees, map_device};
pub use tablet::Tablet;
