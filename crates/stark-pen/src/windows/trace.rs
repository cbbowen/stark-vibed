//! A running commentary on what the stylus is doing, behind `STARK_PEN_TRACE=1`.
//!
//! **Diagnostic scaffolding, and meant to be removable.** It is here because this is
//! the one part of the tree a test cannot reach: the readers below talk to a driver
//! through a window procedure, on hardware CI does not have and the author may not
//! have either, and the failure they produce — "pressure does not work" — is the same
//! sentence for a dozen different causes. A trace turns one stroke into an answer.
//!
//! Silent unless asked, and asked once: the variable is read on the first line that
//! would print and remembered, so a stroke that is not being traced costs an atomic
//! load per report.

use std::sync::OnceLock;

/// Whether the environment asked for a commentary.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("STARK_PEN_TRACE").is_some())
}

/// Say something, if anybody asked.
macro_rules! trace {
    ($($arg:tt)*) => {
        if $crate::windows::trace::on() {
            eprintln!("[pen] {}", format_args!($($arg)*));
        }
    };
}

pub(crate) use trace;
