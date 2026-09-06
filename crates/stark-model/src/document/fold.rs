//! How an action meets a state (§5) — the shape of the fold, not the state.
//!
//! `history` folds a log through its own `history::Action` trait, and the state is
//! `stark-engine`'s `DocState` — both foreign to this crate, so the impl cannot be
//! written directly. [`Logged<S>`] is the local type that carries it, and the bound it
//! introduces is the division itself: **this crate owns that a log folds and which
//! actions commute; the other owns what the fold produces.**
//!
//! `stark-engine` implements [`Materialize`] for its `DocState` and stores a
//! `history::History<Logged<DocState>>`. [`Footprint`] remains the centralizer, so
//! undo splices an action out past everything it commutes with (§12.6) rather than
//! replaying the log after it.

use std::marker::PhantomData;

use super::action::Action;
use super::footprint::{Footprint, compute_footprint};

/// A state an action log can be folded into.
///
/// The implementor decides what "applying" means. What this crate insists on is the
/// shape: folding is total — an action that cannot be honoured leaves the state alone
/// (§4) — and cloning is cheap enough that history can keep snapshots (§5.1).
pub trait Materialize: Clone {
    /// Whatever applying needs and the state does not carry — renderers, a device,
    /// an asset store. `()` for a consumer that needs nothing.
    type Ctx;

    /// Apply `action`, producing the next state.
    ///
    /// A state rather than a `Result`: an action that cannot be honoured — a stroke on
    /// a missing layer, a transform past the tile caps — is **declined
    /// deterministically** by returning the state unchanged (§4), so every peer
    /// declines it identically.
    fn fold(self, action: &Action, ctx: &mut Self::Ctx) -> Self;

    /// Restore what `action` changed, given the state as it stood before it.
    ///
    /// The default clones the whole previous state, which is always correct; an
    /// implementor that restores only what its footprint names makes history surgery
    /// cheaper (§12.6) without changing what it means. `footprint` is the one the
    /// [`Logged`] already carries, so it cannot disagree with the one the fold claimed.
    fn unfold(&mut self, action: &Action, footprint: &Footprint, previous: &Self) {
        let _ = (action, footprint);
        self.clone_from(previous);
    }

    /// Check, in debug builds only, that folding `action` changed nothing outside what
    /// its [`Footprint`] declared (§12.6) — the one rule here nothing structural
    /// holds, since the compiler cannot say a footprint is the one its `apply` arm
    /// honours.
    ///
    /// A no-op by default, and free by default: the caller clones the previous state
    /// only when an implementor has something to compare. Debug-only because the
    /// comparison walks the layer tree per action, and a violation is a bug in this
    /// crate's tables rather than a state a shipped build should survive.
    ///
    /// **The gate is on the call, not on this declaration**, so the trait has one shape
    /// in every profile — a `[profile.*.package.stark-model]` override would otherwise
    /// make `audit` vanish from the trait on one side of the crate boundary. An
    /// implementor may gate *its* override; `DocState` does.
    fn audit(_before: &Self, _after: &Self, _action: &Action, _footprint: &Footprint) {}

    /// Whether [`audit`](Self::audit) has anything to say, and so whether the fold
    /// keeps the previous state to hand it. `false` by default, so a no-op audit costs
    /// no clone.
    const AUDITED: bool = false;
}

/// An [`Action`] paired with the state it is to be folded into — the local type that
/// carries the `history::Action` impl, and the **one door onto the history**.
/// [`Deref`](std::ops::Deref) gives the action straight back.
///
/// **It sanitizes** ([`ActionKind::sanitized`](super::ActionKind::sanitized)). Every
/// action reaching a state comes through here — a local commit, a replay from a file,
/// a peer's action merged into the replicated log: the "enters state" half of the
/// funnel §21.5 describes, with `Engine::commit` the "is minted" half. Peers still
/// converge, sanitizing being a pure idempotent function of the action.
///
/// **It computes the footprint once.** `history` asks one centralizer about every
/// later action in the log, so rederiving per comparison would make an undo across a
/// `TransformWarp` quadratic — each comparison a fresh 57×57 lattice solve for an
/// answer that cannot change.
#[derive(Clone, Debug)]
pub struct Logged<S: Materialize> {
    action: Action,
    /// Computed at construction — see the type's note. Not `pub`, and not
    /// recomputable from outside, so it cannot drift from the action beside it.
    footprint: Footprint,
    _state: PhantomData<S>,
}

impl<S: Materialize> Logged<S> {
    pub fn new(action: Action) -> Self {
        let action = Action {
            id: action.id,
            kind: action.kind.sanitized(),
        };
        // After sanitizing: a footprint claims what the fold will touch, and the
        // fold sees the sanitized action. Built from the raw one, a clamp that
        // pulls a value down would leave the claim disagreeing with the pass in
        // the direction §12.6 cannot survive.
        let footprint = compute_footprint(&action);
        Self {
            action,
            footprint,
            _state: PhantomData,
        }
    }

    /// What this action reads and writes (§12.6) — computed once, at construction.
    pub fn footprint(&self) -> &Footprint {
        &self.footprint
    }

    /// The action back out, dropping the state it was addressed to.
    pub fn into_action(self) -> Action {
        self.action
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
        &self.action
    }
}

impl<S: Materialize> history::Action for Logged<S> {
    type State = S;
    type Context = S::Ctx;
    type Centralizer<'a> = &'a Footprint;
    // Applying an action never fails: GPU work reports failure through wgpu's device
    // error callbacks rather than return values, and an action that cannot be
    // honoured is declined by leaving the state alone (see [`Materialize::fold`]).
    type Error = std::convert::Infallible;

    fn apply(&self, state: S, ctx: &mut S::Ctx) -> Result<S, Self::Error> {
        // Kept only where the implementor audits, so a state that does not pays no
        // clone (see [`Materialize::audit`]).
        #[cfg(debug_assertions)]
        let before = S::AUDITED.then(|| state.clone());
        let after = state.fold(&self.action, ctx);
        #[cfg(debug_assertions)]
        if let Some(before) = before {
            S::audit(&before, &after, &self.action, &self.footprint);
        }
        Ok(after)
    }

    fn inverse(&self, previous_state: &S, state: &mut S) {
        state.unfold(&self.action, &self.footprint, previous_state);
    }
}

/// Borrowed from the action rather than rebuilt: both halves read the footprint each
/// `Logged` already carries (see the type's note). `commutes` is the hot one — it runs
/// once per later action, per removal.
impl<'a, S: Materialize> history::Centralizer<'a, Logged<S>> for &'a Footprint {
    fn for_action(action: &'a Logged<S>) -> Self {
        action.footprint()
    }

    fn commutes(&self, other: &Logged<S>) -> bool {
        !self.conflicts(other.footprint())
    }
}
