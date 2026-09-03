//! The shape of the quick-brush rack: hold a number, paint with that brush (§18.1.8).
//!
//! Two constants so far, and they are here because the **preset table** names one of
//! them: an app preset declares the digit it ships on ([`PresetEntry::slot`]), so the
//! rack's arithmetic and the list of tools cannot live on opposite sides of a crate
//! boundary. The rack itself — what a slot holds, how a held number swaps the live
//! brush and gives it back — is still a frontend's, and joins this module at its own
//! stage (§11.2).
//!
//! [`PresetEntry::slot`]: crate::presets::PresetEntry::slot

/// How many quick brushes there are — one per digit.
pub const COUNT: usize = 10;

/// The slot the pen's other end holds, and the one the `0` key holds: the digit past
/// the nine, which is where a tenth of anything goes on a keyboard.
pub const ERASER: usize = 0;
