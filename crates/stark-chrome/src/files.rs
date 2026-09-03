//! What a document *file* is called, and whether there is anything to lose (§8, §15.6).
//!
//! Two facts, and both are a client's rather than the document's — which is why they
//! are here rather than in the engine. The extension names the format on disk; the
//! rule below answers "would closing this lose work", which no document can notice
//! happening to it.

/// Extension for the native (replayable) document format.
///
/// One spelling, shared: a frontend that told a dropped *document* from a dropped
/// *picture* (§23.4) with a second copy of this word would be one edit away from
/// disagreeing with the file it had just written.
pub const DOC_EXT: &str = "stark";

/// The default name a new file is offered under.
///
/// Here for [`DOC_EXT`]'s reason — it carries the extension, so two frontends
/// spelling it apart would be two spellings of the format.
pub fn default_name() -> String {
    format!("painting.{DOC_EXT}")
}

/// Whether the document holds committed work that has not left the client as a file.
///
/// Two halves, from two places that each know only their own:
///
/// - the **engine** says whether the document has been edited since it was last
///   replaced (`ObservableState::edited`), which is how a fresh canvas and a
///   just-opened file both come out clean without a frontend having to know every
///   way a document can be replaced;
/// - the **client** says which revision it last wrote out, which is not a thing a
///   document can notice happening to it.
///
/// Committed only, deliberately. A stroke in flight and an uncommitted drag preview
/// both move what is on screen without moving `doc_revision`, and a gesture is over
/// before a hand can reach a window's close button.
pub fn unsaved(edited: bool, doc_revision: u64, written_revision: u64) -> bool {
    edited && doc_revision != written_revision
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canvas nobody has touched has nothing to lose, whatever revision it is on —
    /// which is the whole design of the guard this feeds: one that objects to closing
    /// an untouched document is one a user learns to dismiss without reading, and it
    /// would then be no guard at all on the day it was right.
    #[test]
    fn an_untouched_document_has_nothing_to_lose() {
        assert!(!unsaved(false, 7, 0));
    }

    /// Written at the revision on screen is clean; a stroke after it is not.
    #[test]
    fn a_write_settles_the_revision_it_wrote() {
        assert!(!unsaved(true, 7, 7));
        assert!(unsaved(true, 8, 7));
    }

    /// A revision *behind* the write is still dirty rather than clean: undoing past
    /// what was saved is a document the file does not hold either.
    #[test]
    fn undoing_past_a_write_is_not_saved_either() {
        assert!(unsaved(true, 6, 7));
    }
}
