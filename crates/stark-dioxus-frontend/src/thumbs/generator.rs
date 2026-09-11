//! The loop both thumbnail generators run ([`crate::thumbs`], [`crate::layer_thumbs`]):
//! a cache of `data:` URLs by key, and at most one background run at a time filling
//! it with whatever its caller says is missing.
//!
//! What differs stays with each caller: which key is next, how one renders, how a
//! finished picture joins the cache ([`Insert`]) and how a run is spaced ([`Pace`]).

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use crate::platform::sleep_ms;
use crate::state::root_signal;

/// A generator's cache, and whether a run is filling it.
///
/// Root-owned (`state::root_signal`) and run in `spawn_forever` tasks, so a panel
/// closing or re-rendering cannot end a run.
pub struct Generated<K: 'static> {
    cache: Signal<Vec<(K, String)>>,
    busy: Signal<bool>,
    insert: Insert<K>,
    pace: Pace,
}

// Hand-written, here and on `Insert`: a derive would demand `K: Copy`.
impl<K: 'static> Clone for Generated<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: 'static> Copy for Generated<K> {}

/// How a finished picture joins its cache.
pub enum Insert<K> {
    /// Beside every entry held: for a key that is the whole of what its picture
    /// shows, so no later render supersedes one.
    Append,
    /// Over the entry `same` pairs it with, else beside the rest: one entry per slot,
    /// for a subject whose pictures go stale.
    Replace(fn(&K, &K) -> bool),
}

impl<K> Clone for Insert<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for Insert<K> {}

/// How a run spaces its work. Zero does not yield at all.
#[derive(Clone, Copy)]
pub struct Pace {
    /// Before the run's first scan.
    pub settle_ms: i32,
    /// After each picture is filed, before the next scan.
    pub between_ms: i32,
}

impl Pace {
    /// Back to back.
    pub const EAGER: Self = Self {
        settle_ms: 0,
        between_ms: 0,
    };
}

impl<K: PartialEq + Clone + 'static> Generated<K> {
    /// An empty cache, idle.
    pub fn new(insert: Insert<K>, pace: Pace) -> Self {
        Self {
            cache: root_signal(Vec::new),
            busy: root_signal(|| false),
            insert,
            pace,
        }
    }

    /// `f` over the cache, subscribing, so a view re-renders when a picture lands.
    /// Scoped to `f` so that no borrow of the cache can be held across an await.
    pub fn with<R>(self, f: impl FnOnce(&[(K, String)]) -> R) -> R {
        f(&self.cache.read())
    }

    /// `f` over the cache, without subscribing.
    pub fn with_peek<R>(self, f: impl FnOnce(&[(K, String)]) -> R) -> R {
        f(&self.cache.peek())
    }

    /// Drop the entries `keep` refuses. Writes, and so wakes every view of the cache,
    /// only when one goes.
    pub fn retain(self, mut keep: impl FnMut(&K) -> bool) {
        let mut cache = self.cache;
        if cache.peek().iter().all(|(key, _)| keep(key)) {
            return;
        }
        cache.write().retain(|(key, _)| keep(key));
    }

    /// Start a run unless one is going or `next` has nothing: file what `render`
    /// makes of each key `next` gives, until it gives none.
    ///
    /// A run rescans after each picture, so it picks up whatever a second run would
    /// have. `render` answers `None` when there is nothing to render with (no
    /// renderer yet, a lost device), which ends the run; the caller refreshes again
    /// when one lands. A render that fails for its own reasons answers an empty URL,
    /// filed like any picture so that `next` moves past its key.
    pub fn refresh(
        self,
        next: impl Fn() -> Option<K> + 'static,
        mut render: impl AsyncFnMut(&K) -> Option<String> + 'static,
    ) {
        if !should_start(*self.busy.peek(), || next().is_some()) {
            return;
        }
        let mut busy = self.busy;
        busy.set(true);
        spawn_forever(async move {
            let mut wait = self.pace.settle_ms;
            loop {
                if wait > 0 {
                    sleep_ms(wait).await;
                }
                let Some(key) = next() else { break };
                let Some(url) = render(&key).await else { break };
                let mut cache = self.cache;
                if !file(&mut cache.write(), key, url, self.insert) {
                    tracing::warn!(
                        "a thumbnail key does not compare equal to itself; \
                         skipping the rest of the thumbnails"
                    );
                    break;
                }
                wait = self.pace.between_ms;
            }
            let mut busy = self.busy;
            busy.set(false);
        });
    }
}

/// Whether a refresh starts a run: never beside a running one, and never for
/// nothing. `pending` is not asked while one runs, so a busy refresh scans nothing.
fn should_start(busy: bool, pending: impl FnOnce() -> bool) -> bool {
    !busy && pending()
}

/// Whether `cache` has filed `key`. A `next` rule has to ask on these terms for
/// [`file()`]'s guard to cover it.
pub fn holds<K: PartialEq>(cache: &[(K, String)], key: &K) -> bool {
    cache.iter().any(|(filed, _)| filed == key)
}

/// File `url` under `key` by `policy`.
fn insert<K>(cache: &mut Vec<(K, String)>, key: K, url: String, policy: Insert<K>) {
    let slot = match policy {
        Insert::Append => None,
        Insert::Replace(same) => cache.iter().position(|(filed, _)| same(filed, &key)),
    };
    match slot {
        Some(i) => cache[i] = (key, url),
        None => cache.push((key, url)),
    }
}

/// [`insert`], then whether the cache now [`holds`] `key`. It always does unless the
/// key is unequal to itself (a NaN), and then `next` would hand it back forever, so
/// `false` ends the run.
fn file<K: PartialEq + Clone>(
    cache: &mut Vec<(K, String)>,
    key: K,
    url: String,
    policy: Insert<K>,
) -> bool {
    let filed = key.clone();
    insert(cache, key, url, policy);
    holds(cache, &filed)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// The shape of both callers' `next`: the first wanted key not yet filed.
    fn first_missing(wanted: &[u32], cache: &[(u32, String)]) -> Option<u32> {
        wanted.iter().copied().find(|key| !holds(cache, key))
    }

    #[test]
    fn a_refresh_while_busy_starts_nothing() {
        let asked = Cell::new(false);
        let pending = || {
            asked.set(true);
            true
        };
        assert!(!should_start(true, pending));
        assert!(!asked.get(), "a busy refresh scans nothing");
        assert!(should_start(false, || true));
        assert!(!should_start(false, || false));
    }

    #[test]
    fn a_miss_is_filed_so_it_is_not_asked_for_again() {
        let mut cache = Vec::new();
        assert_eq!(first_missing(&[1, 2], &cache), Some(1));
        assert!(file(&mut cache, 1, String::new(), Insert::Append));
        assert_eq!(first_missing(&[1, 2], &cache), Some(2));
    }

    #[test]
    fn a_key_unequal_to_itself_ends_the_run() {
        let mut cache = Vec::new();
        assert!(!file(&mut cache, f32::NAN, String::new(), Insert::Append));
    }

    /// The brush cache: a snapshot's picture never goes stale, so another brush's
    /// never displaces it.
    #[test]
    fn append_keeps_every_picture() {
        let mut cache = Vec::new();
        insert(&mut cache, 1, "a".to_owned(), Insert::Append);
        insert(&mut cache, 2, "b".to_owned(), Insert::Append);
        assert_eq!(cache, [(1, "a".to_owned()), (2, "b".to_owned())]);
    }

    /// The layer cache: a fresher picture of a layer replaces its last, so the cache
    /// holds one per layer however often each is repainted, and another layer's
    /// picture stands.
    #[test]
    fn replace_keeps_one_picture_per_slot() {
        let same_slot: Insert<(u32, u64)> = Insert::Replace(|a, b| a.0 == b.0);
        let mut cache = Vec::new();
        insert(&mut cache, (1, 7), "old".to_owned(), same_slot);
        insert(&mut cache, (2, 1), "other".to_owned(), same_slot);
        insert(&mut cache, (1, 8), "new".to_owned(), same_slot);
        assert_eq!(
            cache,
            [((1, 8), "new".to_owned()), ((2, 1), "other".to_owned())]
        );
    }
}
