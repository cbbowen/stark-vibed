//! The colours the suite paints in, each defined once.
//!
//! `RED` was declared in twenty-two files at six values, and nothing said which
//! differences were meant. This file does not claim to know either: **every value is
//! exactly what the files using it were written with** — goldens were blessed on
//! them — and the names say what tells the values apart, not why they are apart. Two
//! reds stay where they were, `seam.rs`'s and the corpus's, as does any colour one
//! file alone paints; each says so beside its `const`, the way [`super::leads`]'s
//! margins do.

/// The primaries: the widest channel separation there is, so what most of the suite
/// probes by dominance is painted in them.
pub const RED: [f32; 3] = [1.0, 0.0, 0.0];
pub const GREEN: [f32; 3] = [0.0, 1.0, 0.0];

/// Paint at [`PAPER`](super::PAPER)'s level: the layer, matte, export, filter and
/// save/load suites' red, the green most files pair with whichever red they use,
/// and the blue the fill, gradient and picker suites set against them.
pub const RED_SOFT: [f32; 3] = [0.85, 0.1, 0.1];
pub const GREEN_SOFT: [f32; 3] = [0.1, 0.8, 0.2];
pub const BLUE_SOFT: [f32; 3] = [0.1, 0.2, 0.8];

/// [`RED_SOFT`] with a touch more green: the red the picker and gradient suites read
/// back as numbers.
pub const RED_WARM: [f32; 3] = [0.85, 0.12, 0.1];

/// A step under the primaries: what the collaboration suites give their first and
/// third peers, with [`GREEN_SOFT`] for the second.
pub const RED_VIVID: [f32; 3] = [0.9, 0.1, 0.1];
pub const BLUE_VIVID: [f32; 3] = [0.1, 0.2, 0.9];

/// What a matte is painted in by the tests that read its frame and nothing else.
pub const BLACK: [f32; 3] = [0.0, 0.0, 0.0];

/// Two saturated lights that overlap in the middle of the canvas — warm and cool so
/// the overlap is unmistakably a *combination* rather than either one of them — and
/// a pale third for the layer that has to read against both.
pub const WARM: [f32; 3] = [0.90, 0.35, 0.10];
pub const COOL: [f32; 3] = [0.10, 0.30, 0.85];
pub const PALE: [f32; 3] = [0.95, 0.90, 0.80];
