//! The timeline abstraction (DESIGN.md §5).
//!
//! `Session`/`Engine` only ever see the [`Timeline`] trait, so the storage
//! strategy can change without touching them. [`LinearTimeline`] is the
//! single-user implementation (a thin wrapper over `history::History`); the
//! multi-peer `ReplicatedTimeline` slots in at the same seam in step 7.

use history::{History, Version};

use super::action::{Action, ApplyCtx};
use super::state::DocState;

/// A versioned document: the source of the current [`DocState`] plus undo/redo.
pub trait Timeline {
    /// The current document state (`O(1)`).
    fn current(&self) -> &DocState;

    /// Commit an action, advancing the timeline. Clears any redo stack.
    fn push(&mut self, action: Action, ctx: &mut ApplyCtx);

    /// Step back one action if possible; returns whether anything was undone.
    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool;

    /// Re-apply the most recently undone action; returns whether anything was redone.
    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool;

    fn can_undo(&self) -> bool;
    fn can_redo(&self) -> bool;
}

/// Single-user timeline: a linear undo/redo stack over `history::History`.
pub struct LinearTimeline {
    history: History<Action>,
    /// Actions popped by `undo`, awaiting `redo`. Cleared on a fresh `push`.
    redo: Vec<Action>,
}

impl LinearTimeline {
    pub fn new(initial: DocState) -> Self {
        Self {
            history: History::new(initial),
            redo: Vec::new(),
        }
    }

    /// All committed actions, oldest to newest (the basis of the save format,
    /// DESIGN.md §8).
    pub fn actions(&self) -> impl Iterator<Item = &Action> {
        self.history.actions()
    }
}

impl Timeline for LinearTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }

    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        self.redo.clear();
        // Infallible apply (DESIGN.md §5) — no error to handle.
        self.history.push_action_with(action, ctx);
    }

    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self.history.pop_action_with(ctx) {
            Some(action) => {
                self.redo.push(action);
                true
            }
            None => false,
        }
    }

    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self.redo.pop() {
            Some(action) => {
                self.history.push_action_with(action, ctx);
                true
            }
            None => false,
        }
    }

    fn can_undo(&self) -> bool {
        self.history.last_version() != Version::default()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}
