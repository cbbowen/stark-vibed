//! Two data structures the engine's read side is built from, and the counter that
//! keys them: a list shared rather than copied ([`Projected`]), a one-slot cache
//! rebuilt when its key moves ([`Memo`]), and the [`Revision`] a key's terms are.
//!
//! Nothing here names an engine type — that is what lets the rule on [`Memo`] be
//! stated once and read without the engine open. The keys themselves live beside
//! what they key (`engine::observe`, `engine::render`).

use std::sync::Arc;

/// A list [`ObservableState`](crate::ObservableState) carries: **shared rather than copied**, because a
/// projection is taken after *every* command — including the pan, zoom and
/// brush-tuning commands that arrive at pointer rate — and almost none of them can
/// move any given list.
///
/// Two properties, and the type exists for both:
///
/// - **Handing one out is a refcount bump**, whatever it holds and however long it
///   is. What that saves depends on the list: the layer roster costs a walk of the
///   whole tree, cloning every name and asking
///   [`merge::plan_at`](crate::document::merge::plan_at) per row, and `Engine` keeps
///   the last one against the counters it is a function of
///   ([`Engine::projected_layers`](crate::Engine::projected_layers)) so an unchanged
///   document walks nothing at all.
/// - **Asking "did this move?" is a pointer comparison** — see the [`PartialEq`]
///   impl, which is the half a frontend holding this in a reactive signal actually
///   feels.
///
/// Generic because the argument is about what a *projection* is, not about what any
/// one list holds — a second roster projected from the same `observe()` at the same
/// rate would otherwise be a `Vec` deep-cloned and deep-compared per pointer sample.
///
/// Derefs to `[T]`, so it is read exactly as the `Vec` it replaces was. Building one
/// is `Vec::into`, which happens where the list actually changes and nowhere else.
#[derive(Debug)]
pub struct Projected<T>(Arc<[T]>);

/// Cloning shares; it never copies the elements, so this is deliberately **not**
/// derived — a derived impl would demand `T: Clone` to do what an `Arc` bump does
/// for free, and would invite someone to satisfy it.
impl<T> Clone for Projected<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> Default for Projected<T> {
    fn default() -> Self {
        Self(Vec::new().into())
    }
}

impl<T> std::ops::Deref for Projected<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> From<Vec<T>> for Projected<T> {
    fn from(items: Vec<T>) -> Self {
        Self(items.into())
    }
}

impl<T> FromIterator<T> for Projected<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T: PartialEq> PartialEq for Projected<T> {
    /// **Structural equality, with identity as a fast path.**
    ///
    /// The fast path is the whole point of sharing the list: two projections taken
    /// while the document stood still hold the *same* `Arc`, so the frontend's
    /// "did this slice move?" — asked per memo, per command — is one pointer
    /// comparison instead of a walk of every element.
    ///
    /// The fall-through keeps the answer exact. Identity alone would be sound
    /// (same `Arc` ⇒ same contents, since the contents are immutable once shared)
    /// but conservative: a rebuild that changed nothing would report a change, and a
    /// commit that leaves the tree alone happens on every stroke.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

/// A **one-slot cache**: a value beside the key it was built from, rebuilt only when
/// the key moves (C4). The engine keeps three of these — the layer roster, the guide
/// roster and the compositor's draw list — and this type is the whole of what they
/// have in common.
///
/// **The rule, stated here rather than three times over.** A key must name every term
/// its value is a function of. One term too few and the memo hands back a stale answer
/// that nothing downstream can notice; one too many and it rebuilds for a change the
/// value cannot see, which is only a cost. So where a key cannot be exact it errs
/// *wide*, and each key says where it does.
///
/// **Nothing here counts anything of its own**, and that is what makes a memo sound
/// rather than merely plausible. Every term of every key is a counter something else
/// already maintains for its own reasons — `Engine::doc_revision`, `Preview::epoch`,
/// `Preview::fold`, `Engine::guide_epoch` — and [`Revision`] is what such a counter
/// is. There is no invalidation call anywhere, because the key *is* the
/// invalidation; a memo that had to be told it was stale would be one a new mutation
/// path could forget to tell (§1).
///
/// `RefCell` because [`Engine::observe`](crate::Engine::observe) takes `&self`: a projection is a *read*, and
/// making it `&mut` to let it memoize would put a mutable borrow of the whole engine
/// on the path every panel takes to draw itself. The draw list is held the same way
/// for a second reason — see [`Engine::draw_list`](crate::Engine::draw_list).
pub(crate) struct Memo<K, V> {
    slot: std::cell::RefCell<Option<(K, V)>>,
}

/// Empty, whatever it holds. Deliberately not derived: a derived impl would demand a
/// `Default` of the key and the value, which neither has and neither needs.
impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            slot: std::cell::RefCell::new(None),
        }
    }
}

impl<K: PartialEq, V: Clone> Memo<K, V> {
    /// What was built from `key`, or `build`'s answer stored against it.
    ///
    /// **The borrow is released before `build` runs**, which is the half of this that
    /// had to be a function rather than three comparisons written out. A build is
    /// arbitrary engine code — the layer walk asks
    /// [`merge::plan_at`](crate::document::merge::plan_at) per row, the draw list
    /// walks every visible tile of every layer — so one that read the memo it was
    /// filling would panic, at run time, on whichever path a test did not take.
    ///
    /// `V: Clone`, and cheaply so at all three call sites: the two rosters hand back
    /// an `Arc` bump ([`Projected`]) and the draw list an `Arc<[CompositeGroup]>`. A
    /// memo whose value is expensive to hand out gives back what it saved.
    ///
    /// [`CompositeGroup`]: crate::gpu::CompositeGroup
    pub(crate) fn get_or_build(&self, key: K, build: impl FnOnce() -> V) -> V {
        if let Some(hit) = self.hit(&key) {
            return hit;
        }
        let value = build();
        *self.slot.borrow_mut() = Some((key, value.clone()));
        value
    }

    /// What is held, if it was built from `key`. Its own function so the borrow ends
    /// where the compiler says it does rather than where a reader hopes it does.
    fn hit(&self, key: &K) -> Option<V> {
        let slot = self.slot.borrow();
        let (cached, value) = slot.as_ref()?;
        (cached == key).then(|| value.clone())
    }
}

/// A counter that exists to be a term of a [`Memo`] key: bumped by whatever owns it
/// when the thing it stands for has moved, compared by the key, read by nothing
/// else. What a bump *means* is the owner's — `Preview::epoch` is "the document
/// under the previews was replaced", `Engine::guide_epoch` is "an eye opened or
/// shut" — and this holds only the arithmetic, once, where four counters had it
/// four ways.
///
/// Wrapping rather than checked: a key's job is to differ from the value it was
/// last compared against, and 2⁶⁴ bumps between two comparisons is not a case a
/// panic would be reporting on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Revision(u64);

impl Revision {
    /// Move on: no key built before this call compares equal to one built after.
    pub(crate) fn bump(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }

    /// The count as a bare number, for a key that keeps one (`render::DrawKey`).
    pub(crate) fn get(self) -> u64 {
        self.0
    }
}
