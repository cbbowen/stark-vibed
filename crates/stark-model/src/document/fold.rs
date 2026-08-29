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
use super::footprint::{Footprint, compute_footprint};

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
    ///
    /// **`footprint` is handed in rather than derived**, and it is the one the
    /// [`Logged`] already carries. An implementor that restores by write list is
    /// asking exactly the question this type was given a cached answer to, and the
    /// engine's did derive it again — once per `inverse`, which is once per cached
    /// state per shift. Passing it makes the cache reach its second consumer and
    /// takes away the call site's chance to be wrong (see [`Logged`]'s note).
    fn unfold(&mut self, action: &Action, footprint: &Footprint, previous: &Self) {
        let _ = (action, footprint);
        self.clone_from(previous);
    }

    /// Check, in debug builds only, that folding `action` changed nothing outside
    /// what its [`Footprint`] declared — the rule §12.6 opens with, asked of every
    /// fold rather than of a table.
    ///
    /// **The rule is the one thing here that nothing structural holds.** Seven
    /// exhaustive matches over `ActionKind` say every action *has* a footprint; none
    /// of them says the footprint is the one its `apply` arm honours, and the
    /// compiler cannot: the two are a walk of the tree and a list of resources.
    /// `stark-engine/tests/footprint.rs` asked it of a hand-driven vocabulary, which
    /// is a sample. This asks it of every action every test in the workspace folds.
    ///
    /// A no-op by default, and by default it also costs nothing: the caller only
    /// clones the previous state when an implementor has something to compare. A
    /// consumer whose state is a counter has no business paying for this.
    ///
    /// Debug-only because the comparison is a walk of the layer tree per action,
    /// which is fine for a test and not for a release fold — and because a violation
    /// is a bug in *this* crate's tables, not a state a shipped build should try to
    /// survive.
    ///
    /// **The gate is on the call, not on this declaration.** Both live here
    /// unconditionally so the trait has *one shape in every profile*: gated, its
    /// members would depend on `debug_assertions` at the definition, and the
    /// arrangement would hold only for as long as every crate that names this
    /// trait is compiled under one profile. A `[profile.*.package.stark-model]`
    /// override, or any split that reaches one side of the boundary and not the
    /// other, would report itself as "`audit` is not a member of trait
    /// `Materialize`" from a line nobody edited. An implementor is free to gate
    /// *its* override — `DocState` does — and falls back to this no-op where the
    /// gate is off, which is the same release fold either way.
    fn audit(_before: &Self, _after: &Self, _action: &Action, _footprint: &Footprint) {}

    /// Whether [`audit`](Self::audit) has anything to say — and so whether the fold
    /// should keep the previous state to hand it.
    ///
    /// `false` by default so the clone is not paid for a no-op audit. `DocState`'s
    /// clone is a handful of `Arc` bumps (§5.1), which is what makes turning it on
    /// affordable there.
    const AUDITED: bool = false;
}

/// An [`Action`] paired with the state it is to be folded into — the local type that
/// carries the `history::Action` impl.
///
/// Transparent in every way that matters: [`Deref`](std::ops::Deref) gives the
/// action straight back. Its first job is to be *this crate's* type, so the impl
/// below can be written at all. Its second is to be the **one door onto the
/// history**, which is what lets the two things below happen exactly once per
/// action instead of once per question asked about it.
///
/// # What the door does
///
/// **It sanitizes** ([`ActionKind::sanitized`](super::ActionKind::sanitized)). Every action reaching a state
/// comes through here — a local commit, a replay from a file, a peer's action
/// merged into the replicated log — and this is the "enters state" half of the
/// funnel §21.5 describes, with `Engine::commit` the "is minted" half. Peers still
/// converge because sanitizing is a pure, idempotent function of the action: two
/// peers handed the same log materialize the same state whether or not the log
/// itself was ever cleaned.
///
/// **It computes the footprint once.** `history` builds a centralizer once per
/// removal and then asks it about *each* later action
/// (`History::try_remove_action_with`), so `Centralizer::commutes` used to
/// rebuild the other action's footprint on every comparison — two `Vec`
/// allocations always, a walk of the whole control-point list for a stroke, and
/// for a `TransformWarp` an entire fine-lattice solve (`WarpMap::image_aabb`,
/// 57×57 nodes at an 8×8 grid). An undo across a warp was quadratic in the log for
/// an answer that cannot change: a footprint is a pure function of an action, and
/// the action is not moving. Held here, it is paid once at push.
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
        // After sanitizing, deliberately: a footprint is a claim about what the
        // fold will touch, and the fold sees the sanitized action. Built from the
        // raw one it could claim a box the run never writes — harmless — or, where
        // a clamp pulls a value *down*, disagree with the pass in the direction
        // §12.6 cannot survive.
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
        // Kept only where the implementor audits, so a state that does not is folded
        // exactly as it was before this existed (see [`Materialize::audit`]).
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

/// Borrowed from the action rather than rebuilt: both halves read the footprint
/// each `Logged` already carries, which is the whole of the fix described on the
/// type. `commutes` is the hot one — it runs once per later action, per removal.
impl<'a, S: Materialize> history::Centralizer<'a, Logged<S>> for &'a Footprint {
    fn for_action(action: &'a Logged<S>) -> Self {
        action.footprint()
    }

    fn commutes(&self, other: &Logged<S>) -> bool {
        !self.conflicts(other.footprint())
    }
}
