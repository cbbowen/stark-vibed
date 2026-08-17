//! How an action meets a state (§5) — the shape of the fold, not the state.
//!
//! # Why this module exists
//!
//! The history crate materializes a log by folding [`Action`]s over a state, and it
//! asks for that through `history::Action`. Both the trait and — after the split
//! (§2) — the state are outside this crate: the trait is `history`'s, and the state
//! is `stark-engine`'s `DocState`, a map of GPU tile handles that by definition
//! cannot live here. An `impl history::Action for Action` naming `DocState` is
//! therefore an impl of a foreign trait for a foreign type in one crate and of a
//! foreign trait for a *local* type it cannot name in the other. Neither is legal.
//!
//! [`Logged<S>`] is the way through, and it is not a workaround. It is local to this
//! crate, so the generic impl below is allowed; and what it makes explicit is exactly
//! the division the split is about. **This crate owns the fact that a log folds, and
//! which actions commute; the other crate owns what the fold produces.** An action
//! knows what it means without knowing what a pixel is — which is the founding
//! sentence of the whole design, now written as a trait bound.
//!
//! # Using it
//!
//! `stark-engine` implements [`Materialize`] for its `DocState` and stores a
//! `history::History<Logged<DocState>>`. Nothing else changes: [`Footprint`] is
//! still the centralizer, so undo still splices an action out past everything it
//! commutes with (§12.6) rather than replaying the log after it.

use std::marker::PhantomData;

use super::action::Action;
use super::footprint::{Footprint, footprint};

/// A state an action log can be folded into.
///
/// The implementor decides what "applying" means — for `stark-engine` that is tiles
/// rasterized on the GPU; for a test or a future headless consumer it could be far
/// less. What this crate insists on is only the shape: folding is total (an action
/// that cannot be honoured leaves the state alone, §4), and cloning is cheap enough
/// that history can keep snapshots of it (§5.1).
pub trait Materialize: Clone {
    /// Whatever applying needs and the state does not carry — renderers, a device,
    /// an asset store. `()` for a consumer that needs nothing.
    type Ctx;

    /// Apply `action`, producing the next state.
    ///
    /// Returns a state rather than a `Result` on purpose: an action that cannot be
    /// honoured — a stroke on a missing layer, a transform that exceeds the tile
    /// caps — is **declined deterministically** by returning the state unchanged
    /// (§4). Every peer declines it identically, which is what makes refusal a fact
    /// about the log rather than about one client's luck.
    fn fold(self, action: &Action, ctx: &mut Self::Ctx) -> Self;

    /// Restore what `action` changed, given the state as it stood before it.
    ///
    /// The default clones the whole previous state, which is always correct. An
    /// implementor that can restore only the part its footprint names makes history
    /// surgery cheaper (§12.6) without changing what it means.
    fn unfold(&mut self, action: &Action, previous: &Self) {
        let _ = action;
        self.clone_from(previous);
    }
}

/// An [`Action`] paired with the state it is to be folded into — the local type that
/// carries the `history::Action` impl.
///
/// Transparent in every way that matters: it holds nothing but the action, and
/// [`Deref`](std::ops::Deref) gives it straight back. Its only job is to be *this
/// crate's* type, so the impl below can be written at all.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Logged<S: Materialize>(pub Action, PhantomData<S>);

impl<S: Materialize> Logged<S> {
    pub fn new(action: Action) -> Self {
        Self(action, PhantomData)
    }

    /// The action back out, dropping the state it was addressed to.
    pub fn into_action(self) -> Action {
        self.0
    }
}

impl<S: Materialize> From<Action> for Logged<S> {
    fn from(action: Action) -> Self {
        Self::new(action)
    }
}

impl<S: Materialize> std::ops::Deref for Logged<S> {
    type Target = Action;

    fn deref(&self) -> &Action {
        &self.0
    }
}

impl<S: Materialize> history::Action for Logged<S> {
    type State = S;
    type Context = S::Ctx;
    type Centralizer<'a> = Footprint;
    // Applying an action never fails: GPU work reports failure through wgpu's device
    // error callbacks rather than return values, and an action that cannot be
    // honoured is declined by leaving the state alone (see [`Materialize::fold`]).
    type Error = std::convert::Infallible;

    fn apply(&self, state: S, ctx: &mut S::Ctx) -> Result<S, Self::Error> {
        Ok(state.fold(&self.0, ctx))
    }

    fn inverse(&self, previous_state: &S, state: &mut S) {
        state.unfold(&self.0, previous_state);
    }
}

impl<'a, S: Materialize> history::Centralizer<'a, Logged<S>> for Footprint {
    fn for_action(action: &'a Logged<S>) -> Self {
        footprint(&action.0)
    }

    fn commutes(&self, other: &Logged<S>) -> bool {
        !self.conflicts(&footprint(&other.0))
    }
}
