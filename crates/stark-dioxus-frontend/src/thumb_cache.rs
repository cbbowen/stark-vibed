//! The shell both thumbnail caches run on ([`crate::thumbs`], [`crate::layer_thumbs`]):
//! pictures by key, and at most one background run at a time rendering the wanted
//! keys the cache does not hold yet.
//!
//! What differs stays with each caller: which keys are wanted, what a run waits for
//! before it chooses one, how a key renders, how a picture joins the cache
//! ([`Insert`]) and how a run is spaced ([`Pace`]).

use std::sync::atomic::{AtomicBool, Ordering};

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_engine::RgbaImage;

use crate::cards::{Mime, bytes_url};
use crate::platform::sleep_ms;
use crate::state::root_signal;

/// What a key rendered to: a `data:` URL, or `None` for a render that failed for its
/// own reasons. A miss is filed like a picture, so the key is not chosen again.
pub type Picture = Option<String>;

/// A render found nothing to render with: no renderer yet, or a lost device. It ends
/// the run without filing anything, and the caller refreshes again when one lands.
#[derive(Debug)]
pub struct NoRenderer;

/// A thumbnail cache, and whether a run is filling it.
///
/// Root-owned (`state::root_signal`) and run in `spawn_forever` tasks, so a panel
/// closing or re-rendering cannot end a run.
pub struct ThumbCache<K: 'static> {
    cache: Signal<Vec<(K, Picture)>>,
    busy: Signal<bool>,
    insert: Insert<K>,
    pace: Pace,
}

// Hand-written, here and on `Insert`: a derive would demand `K: Copy`.
impl<K: 'static> Clone for ThumbCache<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: 'static> Copy for ThumbCache<K> {}

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
    /// Before the run's first choice.
    pub settle_ms: i32,
    /// After each picture is filed, before the next choice.
    pub between_ms: i32,
}

impl Pace {
    /// Back to back.
    pub const EAGER: Self = Self {
        settle_ms: 0,
        between_ms: 0,
    };
}

impl<K: PartialEq + 'static> ThumbCache<K> {
    /// An empty cache, idle.
    ///
    /// Builds its signals with hooks, so call it unconditionally in a component body.
    pub fn new(insert: Insert<K>, pace: Pace) -> Self {
        Self {
            cache: root_signal(Vec::new),
            busy: root_signal(|| false),
            insert,
            pace,
        }
    }

    /// The picture filed under the first key `which` accepts, subscribing, so a view
    /// re-renders when one lands. `None` both before a render and after a miss.
    pub fn find(self, mut which: impl FnMut(&K) -> bool) -> Option<String> {
        self.cache
            .read()
            .iter()
            .find(|(key, _)| which(key))
            .and_then(|(_, picture)| picture.clone())
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

    /// Start a run unless one is going or the cache holds every key `wanted` lists.
    ///
    /// Each turn of a run waits out its [`Pace`] and then `ready`, chooses the first
    /// wanted key not held ([`pick`]), and files what `render` makes of it, until no
    /// key is left or `render` answers [`NoRenderer`]. `wanted` (most wanted first) is
    /// asked afresh for every choice, so a run picks up whatever a second run would
    /// have, and nothing awaits between choosing a key and starting its render.
    pub fn refresh(
        self,
        wanted: impl Fn() -> Vec<K> + 'static,
        ready: impl AsyncFnMut() + 'static,
        render: impl AsyncFnMut(&K) -> Result<Picture, NoRenderer> + 'static,
    ) {
        if *self.busy.peek() || self.next(wanted()).is_none() {
            return;
        }
        let busy = Busy::claim(self.busy);
        spawn_forever(async move {
            let _busy = busy;
            let file = |key: K, picture: Picture| {
                let mut cache = self.cache;
                insert(&mut cache.write(), key, picture, self.insert);
            };
            run(self.pace, ready, || self.next(wanted()), render, file).await;
        });
    }

    /// [`pick`] against the live cache, without subscribing.
    fn next(self, wanted: Vec<K>) -> Option<K> {
        pick(&self.cache.peek(), wanted, self.insert)
    }
}

/// A run's claim on its cache's `busy` flag, released however the run ends.
struct Busy(Signal<bool>);

impl Busy {
    fn claim(mut busy: Signal<bool>) -> Self {
        busy.set(true);
        Self(busy)
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        // A run dropped with the runtime finds its signal already gone.
        if let Ok(mut busy) = self.0.try_write() {
            *busy = false;
        }
    }
}

/// One run: until `choose` has nothing or `render` has no renderer, wait, then
/// `ready`, then choose a key, render it and file the picture.
async fn run<K>(
    pace: Pace,
    mut ready: impl AsyncFnMut(),
    mut choose: impl FnMut() -> Option<K>,
    mut render: impl AsyncFnMut(&K) -> Result<Picture, NoRenderer>,
    mut file: impl FnMut(K, Picture),
) {
    let mut wait = pace.settle_ms;
    loop {
        if wait > 0 {
            sleep_ms(wait).await;
        }
        // Before the choice, so the key chosen is one that is still wanted once the
        // wait is over.
        ready().await;
        let Some(key) = choose() else { return };
        let Ok(picture) = render(&key).await else {
            return;
        };
        file(key, picture);
        wait = pace.between_ms;
    }
}

/// The first key of `wanted` that `cache` does not hold.
///
/// This is what ends a run: a filed key stays held, so each is chosen once. Two keys
/// would break that and are skipped instead: one not equal to itself (a NaN), which no
/// entry can hold, and under [`Insert::Replace`] one whose slot an earlier wanted key
/// fills, which would take turns with that key replacing each other.
fn pick<K: PartialEq>(cache: &[(K, Picture)], mut wanted: Vec<K>, policy: Insert<K>) -> Option<K> {
    let i = wanted.iter().enumerate().position(|(i, key)| {
        if !is_reflexive(key) {
            warn_unequal_key();
            return false;
        }
        let shadowed = match policy {
            Insert::Append => false,
            Insert::Replace(same) => wanted[..i].iter().any(|earlier| same(earlier, key)),
        };
        !shadowed && !holds(cache, key)
    })?;
    Some(wanted.swap_remove(i))
}

/// Whether `cache` has filed `key`.
fn holds<K: PartialEq>(cache: &[(K, Picture)], key: &K) -> bool {
    cache.iter().any(|(filed, _)| filed == key)
}

#[expect(clippy::eq_op, reason = "false exactly for a key holding a NaN")]
fn is_reflexive<K: PartialEq>(key: &K) -> bool {
    key == key
}

/// Once per session: a NaN key is a standing fact about a brush, and every scan meets it.
fn warn_unequal_key() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!("a thumbnail key is not equal to itself, so it gets no thumbnail");
    }
}

/// File `picture` under `key` by `policy`.
fn insert<K>(cache: &mut Vec<(K, Picture)>, key: K, picture: Picture, policy: Insert<K>) {
    let slot = match policy {
        Insert::Append => None,
        Insert::Replace(same) => cache.iter().position(|(filed, _)| same(filed, &key)),
    };
    match slot {
        Some(i) => cache[i] = (key, picture),
        None => cache.push((key, picture)),
    }
}

/// A readback as a PNG `data:` URL, or `None` when the readback or the encode failed.
pub async fn readback_url(
    readback: impl Future<Output = stark_engine::Result<RgbaImage>>,
) -> Option<String> {
    // Not logged: a readback fails when the GPU does, and the canvas reports that
    // through `ObservableState::gpu_failure` (§5).
    let image = readback.await.ok()?;
    match image.to_png() {
        Ok(png) => Some(bytes_url(Mime::Png, &png)),
        Err(error) => {
            tracing::warn!(%error, "could not encode a thumbnail as PNG");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::future::poll_fn;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use super::*;

    /// Choose and file until the choice runs out, as a run does, returning the keys
    /// in the order chosen. Bounded, so a loop that never ends fails rather than hangs.
    fn drain<K: PartialEq + Clone>(
        cache: &mut Vec<(K, Picture)>,
        wanted: &[K],
        policy: Insert<K>,
    ) -> Vec<K> {
        let mut chosen = Vec::new();
        while let Some(key) = pick(cache, wanted.to_vec(), policy) {
            chosen.push(key.clone());
            assert!(
                chosen.len() <= wanted.len(),
                "a run chose more keys than are wanted"
            );
            insert(cache, key, None, policy);
        }
        chosen
    }

    #[test]
    fn pick_returns_the_first_key_not_held() {
        let cache = [(1, Some("a".to_owned())), (3, None)];
        assert_eq!(pick(&cache, vec![1, 3, 2, 4], Insert::Append), Some(2));
    }

    #[test]
    fn pick_skips_a_key_not_equal_to_itself() {
        assert_eq!(pick(&[], vec![f32::NAN, 1.0], Insert::Append), Some(1.0));
        assert_eq!(pick(&[], vec![f32::NAN], Insert::Append), None);
    }

    #[test]
    fn pick_returns_nothing_when_every_key_is_held() {
        let cache = [(1, Some("a".to_owned())), (2, None)];
        assert_eq!(pick(&cache, vec![2, 1, 2], Insert::Append), None);
    }

    #[test]
    fn an_appending_run_chooses_each_key_once() {
        let mut cache = vec![(2, Some("held".to_owned()))];
        let chosen = drain(&mut cache, &[1, 2, 3, 1], Insert::Append);
        assert_eq!(chosen, [1, 3]);
    }

    /// Two wanted keys in one slot would replace each other forever; the first stands.
    #[test]
    fn a_replacing_run_chooses_each_key_once() {
        let same_slot: Insert<(u32, u64)> = Insert::Replace(|a, b| a.0 == b.0);
        let mut cache = vec![((1, 7), Some("stale".to_owned()))];
        let chosen = drain(&mut cache, &[(1, 8), (2, 1), (1, 9)], same_slot);
        assert_eq!(chosen, [(1, 8), (2, 1)]);
    }

    /// The layer cache: a fresher picture of a layer replaces its last, so the cache
    /// holds one per layer however often each is repainted, and another layer's
    /// picture stands.
    #[test]
    fn replace_keeps_one_picture_per_slot() {
        let same_slot: Insert<(u32, u64)> = Insert::Replace(|a, b| a.0 == b.0);
        let mut cache = Vec::new();
        insert(&mut cache, (1, 7), Some("old".to_owned()), same_slot);
        insert(&mut cache, (2, 1), Some("other".to_owned()), same_slot);
        insert(&mut cache, (1, 8), Some("new".to_owned()), same_slot);
        assert_eq!(
            cache,
            [
                ((1, 8), Some("new".to_owned())),
                ((2, 1), Some("other".to_owned()))
            ]
        );
    }

    /// What the layer cache's `ready` (the canvas-active wait) relies on: a key is
    /// chosen only once `ready` has returned, every turn.
    #[test]
    fn a_run_chooses_no_key_until_ready_returns() {
        let tokens = Cell::new(0);
        let log = RefCell::new(Vec::new());
        let mut keys = vec![1];
        let ready = async || {
            poll_fn(|_| match tokens.get() {
                0 => Poll::Pending,
                n => {
                    tokens.set(n - 1);
                    Poll::Ready(())
                }
            })
            .await;
            log.borrow_mut().push("ready".to_owned());
        };
        let choose = || {
            let key = keys.pop();
            log.borrow_mut().push(format!("choose {key:?}"));
            key
        };
        let render = async |key: &u32| Ok::<_, NoRenderer>(Some(key.to_string()));
        let file = |key: u32, _: Picture| log.borrow_mut().push(format!("file {key}"));
        let mut turns = pin!(run(Pace::EAGER, ready, choose, render, file));
        let mut cx = Context::from_waker(Waker::noop());

        assert!(
            turns.as_mut().poll(&mut cx).is_pending(),
            "ready is pending"
        );
        assert!(
            log.borrow().is_empty(),
            "nothing chosen before ready: {log:?}"
        );
        tokens.set(1);
        assert!(
            turns.as_mut().poll(&mut cx).is_pending(),
            "the second ready is pending"
        );
        tokens.set(1);
        assert!(turns.as_mut().poll(&mut cx).is_ready(), "no key left");
        assert_eq!(
            *log.borrow(),
            ["ready", "choose Some(1)", "file 1", "ready", "choose None"]
        );
    }
}
