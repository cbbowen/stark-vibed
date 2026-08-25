//! The document's vocabulary as **data**, for the two suites that drive it.
//!
//! It used to be the roster itself: a `LABELS` array, a `KINDS` count and an
//! exhaustive `slot` match, written out here because two crates' tests needed a list
//! they could iterate and neither could see the other's tests. That list was the
//! model's own, spelled a second time — and the copies drifted the first time a
//! variant landed after them. `SetSelectionOpacity` (§6.8) got the arm the compiler
//! demanded and was left out of every list those arms index, in both crates at once,
//! and both suites went on passing having never driven the new kind.
//!
//! The roster is [`ActionTag`] now, in the model, built from one macro list beside
//! the enum it describes — so the enum, the list and the captions cannot part. What
//! is left here is the *view* the suites want: an index per kind, and a count to
//! declare an array with.
//!
//! Nothing here can go stale, because nothing here is a list.

use stark_model::document::{ActionKind, ActionTag};

/// How many kinds there are — the length of the model's own roster, and so the
/// length of the one-of-each array `stark-model`'s `action_kinds.rs` drives.
pub const KINDS: usize = ActionTag::ALL.len();

/// Every kind's caption, in roster order.
///
/// A view of [`ActionTag::ALL`] rather than a second list: a caption that disagreed
/// with the model's was the thing this module existed to prevent, and now it cannot
/// be written down here to disagree.
pub fn labels() -> [&'static str; KINDS] {
    let mut out = [""; KINDS];
    for (slot, tag) in out.iter_mut().zip(ActionTag::ALL) {
        *slot = tag.label();
    }
    out
}

/// The slot of an action kind — its place in the roster.
///
/// The exhaustiveness that used to live here lives on `ActionKind::tag`, which is
/// where it belongs: one match, beside the enum, that a new variant cannot compile
/// past. This is the index that match's answer sits at.
pub fn slot(kind: &ActionKind) -> usize {
    kind.tag().index()
}
