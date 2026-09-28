//! The shared test harness, for the modules of the `engine` test binary: a headless
//! engine, the marks a test lays on it, what it reads back, and the goldens (§9).
//!
//! Split by concern and re-exported whole, so a call site writes
//! `use crate::common::*;` and does not have to know which of these files a helper
//! is in — [`engine`] builds them, [`gesture`] paints, [`probe`] reads back,
//! [`compare`] holds two renders against each other and [`golden`] against a blessed
//! one. [`corpus`] is the stroke list the battery runs (`tests/engine/corpus.rs`),
//! [`collab`] a pair of joined engines, [`palette`] the colours.

pub mod collab;
pub mod compare;
pub mod corpus;
pub mod engine;
pub mod gesture;
pub mod golden;
pub mod palette;
pub mod probe;

/// Every module that takes `common::*` gets the engine's diagnostic reads with it.
pub use stark_engine::testing::Probe;

pub use compare::*;
pub use engine::*;
pub use gesture::*;
pub use golden::*;
pub use probe::*;
