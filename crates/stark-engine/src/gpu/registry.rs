//! Frontend-provided GPU resources: registered bytes, which one is in use, and the
//! live object built from it (§6.4, §6.6).
//!
//! The engine embeds no image bytes. The frontend fetches them at runtime and hands
//! them over, which means every such resource has the same three-part shape — a map
//! of registered bytes, the id currently in use, and the GPU object built for that
//! id — and the same two operations: *register bytes* (rebuild if they are the ones
//! in use) and *switch* (rebuild if it actually changed). Written twice, for the
//! canvas substrate and the lighting environment, those were the same six lines with
//! different nouns.
//!
//! Each resource keeps one **builtin** id that needs no bytes at all — `Flat` for
//! substrates, the procedural `Neutral` for environments — which is also the fallback
//! when an id's bytes have not arrived yet.
//!
//! # What is registered and what is built are two keys
//!
//! Usually the same one. The canvas substrate is why they are separate: a substrate is
//! baked *per scale* (§6.4) — the rise a tip meets is measured over a reach in canvas
//! px, so how large the substrate is laid changes the map that gets built from it — while
//! the height map those bakes come from is one PNG whatever scale it is laid at. So
//! bytes are filed under [`Resource::Content`] and built objects under the resource
//! itself, and registering a substrate's bytes readies every scale of it at once.
//!
//! # The store is shared; the *choice* is not
//!
//! The bytes and the built objects live behind an `Arc`, and `Clone` hands out a
//! sibling registry over the same store with its own current id. That is what lets a
//! second engine on the same device ([`Engine::new_sharing`]) reuse every decode this
//! one has paid for — a substrate's height map is a PNG decode plus two whole-image CPU
//! passes, an environment is an HDR decode plus a full mip chain — while still being
//! free to stand on a different substrate. Which id is *in use* is the one part of the
//! shape that is genuinely per-engine: it mirrors that engine's document (§6.4) or
//! its view (§6.3), and siblings agreeing on it would be a bug, not a feature.
//!
//! # Builds run outside the lock, and the cache is bounded
//!
//! A substrate's build is `pack_substrate` — a separable blur plus sixteen projections
//! per texel over up to 2048² — and an upload, and it is reached from the *apply* path
//! whenever a replay crosses a `SetSubstrate`. Run under the store's lock, every
//! sibling's `current()` waited behind it. So [`Registry::get`] takes the lock twice:
//! once to miss and clone out the registration, once to insert what it built. Two
//! callers missing the same id in the gap both build it; the second's is dropped at
//! insert — wasted work, not a wrong answer, since both came from the same bytes.
//!
//! Built objects are held to [`BUILT_BUDGET`] by evicting the least recently asked
//! for, **except the ids registries stand on**, which are pinned for as long as one
//! does. That is what keeps `current()`'s "always built" a structural fact rather than
//! a race against the trim.
//!
//! [`Engine::new_sharing`]: crate::Engine::new_sharing

use crate::unpoisoned;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::gpu::context::GpuContext;
use stark_model::DocError;

/// How many bytes of built objects a store holds before it evicts the least recently
/// used unpinned one — `gpu::scratch`'s `POOL_BUDGET` shape, and the same figure:
/// sixteen 2048² substrate bakes, where `SubstrateScale` admits seventy-six rungs and
/// the cache used to keep every one a document ever crossed for the registry's life.
/// Only what [`Resource::resident_bytes`] reports is counted.
const BUILT_BUDGET: u64 = 256 << 20;

/// A resource the frontend supplies bytes for.
pub trait Resource: Copy + Eq + Hash + std::fmt::Debug {
    /// The live GPU object built from the bytes. `Clone` because the registry hands
    /// out owned handles — the objects are reference-counted wgpu views plus a few
    /// scalars, so a clone is a handful of atomic bumps.
    type Gpu: Clone;

    /// **What the frontend registers bytes under** — see the module note. The
    /// lighting environment is its own content and says so; a substrate is its
    /// `SubstrateId`, so every scale it may be laid at shares one registration.
    type Content: Copy + Eq + Hash + std::fmt::Debug;

    /// **Whatever decoding the bytes once produces that every build from them can
    /// share**, kept beside them by `Registered`.
    ///
    /// Which resources want one is decided by the ratio the module note sets up: the
    /// canvas substrate registers one height map and bakes a `SubstrateMap` per scale
    /// it is laid at (§6.4), so the decode is paid once and the bakes read it. The
    /// lighting environment builds once per registration and its decode is a
    /// multi-megabyte float image only the mip chain ever reads, so it keeps `()` and
    /// decodes inside [`build`](Self::build) — the memory is the reason, and it is a
    /// per-resource answer rather than one this trait should pick.
    type Decoded;

    /// Decode registered bytes, or say why not.
    ///
    /// **This is the door.** It runs in [`Registry::register`] before anything is
    /// stored, so bytes that will not decode are refused where a caller can report
    /// it rather than at the first *use*, which for both of today's resources is on
    /// the render path — an abort on the web with the painting unsaved (§5).
    ///
    /// A [`DocError`] rather than a string, so a resource whose decoder already has a
    /// typed refusal keeps it: a substrate's is `AssetError`, the format's own
    /// identity contract (§19), and flattening that to a sentence at the door only to
    /// have the caller wrap it again lost the one arm anybody would match on.
    fn decode(bytes: &[u8]) -> std::result::Result<Self::Decoded, DocError>;

    /// The registered bytes this id builds from.
    fn content(self) -> Self::Content;

    /// Whether this id needs no registered bytes — the procedural default, which is
    /// also what an id with missing bytes falls back to.
    fn is_builtin(self) -> bool;

    /// Build the GPU object. `registered` is `None` for a builtin id, and also when a
    /// non-builtin id's bytes have not been registered yet.
    fn build(self, gpu: &GpuContext, registered: Option<Registered<'_, Self>>) -> Self::Gpu;

    /// What a built object holds on the device, for [`BUILT_BUDGET`].
    ///
    /// Zero by default, which is "not counted, so never evicted": a resource that
    /// builds once per registration and is dropped by the `register` that replaces its
    /// bytes has nothing for a budget to bound. The substrate, built per scale, says.
    fn resident_bytes(_gpu: &Self::Gpu) -> u64 {
        0
    }
}

/// What a build is given for a registered id: the bytes, and the decode of them
/// [`Registry::register`] already paid for.
///
/// Both, because which one a build reads is the resource's business — see
/// [`Resource::Decoded`].
pub struct Registered<'a, R: Resource> {
    pub bytes: &'a [u8],
    pub decoded: &'a R::Decoded,
}

/// One registration: the bytes as they arrived, and their decode.
///
/// The bytes are kept verbatim because they are what a save file bundles and what a
/// peer is served (§8, §12.4) — a re-encode would be a different byte string under
/// the same content id.
struct Entry<R: Resource> {
    bytes: Vec<u8>,
    decoded: R::Decoded,
}

/// One built object, with what the budget and the eviction order need of it.
struct Built<G> {
    gpu: G,
    bytes: u64,
    /// The store tick this was last built or asked for.
    last: u64,
}

/// The shared half: registered bytes and the objects built from them, one map each.
struct Store<R: Resource> {
    /// Behind an `Arc` so a build can read a registration with the lock released.
    bytes: HashMap<R::Content, Arc<Entry<R>>>,
    /// Everything built, keyed by id; the id in use is always present.
    ///
    /// A **cache**, not a set of live resources, and it exists for the canvas
    /// substrate's sake. Once the deposition tooth reads the substrate (§6.4), a stroke
    /// replayed from before a `SetSubstrate` has to deposit against the substrate it was
    /// actually painted on rather than the one in use now — so [`Registry::get`]
    /// can be asked for any id at any time, and re-decoding a multi-megabyte PNG
    /// every time the log crosses that boundary is not a thing to do on an undo
    /// step. Bounded by [`BUILT_BUDGET`] over what the objects report.
    built: HashMap<R, Built<R::Gpu>>,
    /// How many registries stand on each id. A pinned id is never evicted, so
    /// `current()` finds it built — the invariant, held structurally.
    pinned: HashMap<R, usize>,
    /// The sum of `built`'s `bytes`, kept so the budget check is a compare.
    built_bytes: u64,
    tick: u64,
}

impl<R: Resource> Store<R> {
    /// The object for `id` if it is built, marking it most recently used.
    fn hit(&mut self, id: R) -> Option<R::Gpu> {
        self.tick += 1;
        let built = self.built.get_mut(&id)?;
        built.last = self.tick;
        Some(built.gpu.clone())
    }

    /// File `obj` under `id` — or, if a concurrent build got there first, keep that
    /// one and drop `obj` (the module note). Returns whichever the map now holds.
    fn insert(&mut self, id: R, obj: R::Gpu) -> R::Gpu {
        if let Some(first) = self.hit(id) {
            return first;
        }
        let bytes = R::resident_bytes(&obj);
        self.built_bytes += bytes;
        self.built.insert(
            id,
            Built {
                gpu: obj.clone(),
                bytes,
                last: self.tick,
            },
        );
        self.trim();
        obj
    }

    /// Drop every build of `content` — they were built from bytes now replaced.
    fn forget_content(&mut self, content: R::Content) {
        let mut freed = 0;
        self.built.retain(|id, built| {
            let keep = id.content() != content;
            if !keep {
                freed += built.bytes;
            }
            keep
        });
        self.built_bytes -= freed;
    }

    /// Hold `built` to [`BUILT_BUDGET`], least recently used first, pinned ids never.
    fn trim(&mut self) {
        if self.built_bytes <= BUILT_BUDGET {
            return;
        }
        let candidates: Vec<(R, u64, u64, bool)> = self
            .built
            .iter()
            .map(|(id, b)| (*id, b.last, b.bytes, self.pinned.contains_key(id)))
            .collect();
        let ages: Vec<(u64, u64, bool)> = candidates
            .iter()
            .map(|(_, last, bytes, pinned)| (*last, *bytes, *pinned))
            .collect();
        for i in evict_order(&ages, self.built_bytes, BUILT_BUDGET) {
            let (id, _, bytes, _) = candidates[i];
            self.built.remove(&id);
            self.built_bytes -= bytes;
        }
    }

    fn pin(&mut self, id: R) {
        *self.pinned.entry(id).or_insert(0) += 1;
    }

    fn unpin(&mut self, id: R) {
        if let Some(n) = self.pinned.get_mut(&id) {
            *n -= 1;
            if *n == 0 {
                self.pinned.remove(&id);
            }
        }
    }
}

/// Which entries a store at `total` bytes evicts to get under `budget`: by
/// ascending `last`, skipping pinned, stopping as soon as the budget holds — the
/// whole policy as arithmetic, with no GPU in it. Each entry is
/// `(last, bytes, pinned)`; the answer indexes them.
///
/// A store over budget on pinned entries alone evicts nothing: the budget bounds the
/// cache, not what the registries standing on it need.
fn evict_order(entries: &[(u64, u64, bool)], total: u64, budget: u64) -> Vec<usize> {
    let mut by_age: Vec<usize> = (0..entries.len()).filter(|&i| !entries[i].2).collect();
    by_age.sort_unstable_by_key(|&i| entries[i].0);
    let mut held = total;
    let mut out = Vec::new();
    for i in by_age {
        if held <= budget {
            break;
        }
        held = held.saturating_sub(entries[i].1);
        out.push(i);
    }
    out
}

/// The registered bytes for a resource, the id in use, and the objects built so far.
///
/// `Clone` is a **sibling**, not a copy: the clone shares this registry's store —
/// every registered byte string and every built object, past and future — and takes
/// its own current id, starting where this one stands (see the module note).
pub struct Registry<R: Resource> {
    store: Arc<Mutex<Store<R>>>,
    id: R,
}

impl<R: Resource> Clone for Registry<R> {
    fn clone(&self) -> Self {
        self.store().pin(self.id);
        Self {
            store: Arc::clone(&self.store),
            id: self.id,
        }
    }
}

impl<R: Resource> Drop for Registry<R> {
    fn drop(&mut self) {
        self.store().unpin(self.id);
    }
}

impl<R: Resource> Registry<R> {
    /// A registry holding only `id`, which must therefore be a builtin (nothing has
    /// been registered yet).
    pub fn new(gpu: &GpuContext, id: R) -> Self {
        debug_assert!(id.is_builtin(), "a fresh registry can only hold a builtin");
        let obj = id.build(gpu, None);
        let mut store = Store {
            bytes: HashMap::new(),
            built: HashMap::new(),
            pinned: HashMap::from([(id, 1)]),
            built_bytes: 0,
            tick: 0,
        };
        store.insert(id, obj);
        Self {
            store: Arc::new(Mutex::new(store)),
            id,
        }
    }

    /// The one place this registry's lock is taken, poisoned or not ([`unpoisoned`]).
    ///
    /// What it guards is a byte map and a build cache — derived state, rebuildable
    /// from the bytes — so another thread's panic is a reason to serve a cold entry,
    /// not to take the renderer down with it.
    fn store(&self) -> std::sync::MutexGuard<'_, Store<R>> {
        unpoisoned(self.store.lock())
    }

    /// Which resource is in use.
    pub fn id(&self) -> R {
        self.id
    }

    /// The live GPU object — the one `id()` names.
    pub fn current(&self) -> R::Gpu {
        self.store()
            .hit(self.id)
            .expect("the id in use is pinned, so always built")
    }

    /// The object for **any** `id`, built on demand and cached — without changing
    /// which one is in use.
    ///
    /// This is what a replay asks: the substrate a stroke deposits against is the one
    /// the document was on *at that point in the log* (§6.4), which is a question
    /// about the action being applied, not about what the compositor is showing.
    /// Also how [`register`](Self::register) and [`set`](Self::set) rebuild the
    /// object they have just invalidated: they call this and drop the result, rather
    /// than reaching for a returnless `ensure` twin that nothing would keep in step
    /// with this one.
    ///
    /// The build runs with the lock released (the module note).
    pub fn get(&self, gpu: &GpuContext, id: R) -> R::Gpu {
        let registered = {
            let mut store = self.store();
            if let Some(hit) = store.hit(id) {
                return hit;
            }
            store.bytes.get(&id.content()).cloned()
        };
        if registered.is_none() && !id.is_builtin() {
            tracing::warn!(id = ?id, "no registered bytes; falling back to the builtin");
        }
        let obj = id.build(
            gpu,
            registered.as_deref().map(|e| Registered {
                bytes: &e.bytes,
                decoded: &e.decoded,
            }),
        );
        self.store().insert(id, obj)
    }

    /// Whether `id` is ready to use: builtins always are, everything else once its
    /// bytes have been [`register`](Self::register)ed.
    pub fn is_loaded(&self, id: R) -> bool {
        id.is_builtin() || self.store().bytes.contains_key(&id.content())
    }

    /// The registered bytes for `content`, if any — what a save file bundles and what
    /// a peer is served (§8, §12.4). `None` for a builtin, which has none by
    /// definition, and for content whose image has not arrived.
    pub fn bytes(&self, content: R::Content) -> Option<Vec<u8>> {
        self.store().bytes.get(&content).map(|e| e.bytes.clone())
    }

    /// Provide bytes for `content`, **decoding them here** ([`Resource::decode`]).
    /// `Err` is bytes this resource cannot read, and nothing is stored for them.
    /// `Ok(true)` means the live object was rebuilt, which happens exactly when the id
    /// in use is built from that content — the caller then has to rebind it wherever
    /// it is sampled.
    ///
    /// The decode is done before the store is touched so that a refusal changes
    /// nothing: a half-registered content id whose bytes will not decode is the state
    /// that turns a bad download into a fallback nobody asked for, silently, one
    /// render later.
    ///
    /// Every id standing on this content is rebuilt, siblings' included — they are
    /// pinned, and a pinned id is always built. A sibling still keeps its stale
    /// *binding* until it next rebinds, exactly the exposure two independent
    /// registries had, since neither saw the other's bytes arrive at all.
    pub fn register(
        &self,
        gpu: &GpuContext,
        content: R::Content,
        bytes: Vec<u8>,
    ) -> std::result::Result<bool, DocError> {
        let decoded = R::decode(&bytes)?;
        let standing: Vec<R> = {
            let mut store = self.store();
            store
                .bytes
                .insert(content, Arc::new(Entry { bytes, decoded }));
            // Whatever was built from this content was built from the *old* bytes —
            // usually the builtin fallback, standing in while the fetch was in flight.
            // Dropping it is what makes `get` return the real thing from now on. Every
            // build of it, since one height map bakes a substrate per scale (see the
            // module note): the bytes are one key and the objects are another.
            store.forget_content(content);
            store
                .pinned
                .keys()
                .copied()
                .filter(|id| id.content() == content)
                .collect()
        };
        for id in &standing {
            self.get(gpu, *id);
        }
        Ok(self.id.content() == content)
    }

    /// Switch to `id`. Returns `true` if it changed (and so was rebuilt); switching
    /// to what is already in use is a no-op, not a rebuild.
    pub fn set(&mut self, gpu: &GpuContext, id: R) -> bool {
        if id == self.id {
            return false;
        }
        {
            let mut store = self.store();
            store.unpin(self.id);
            store.pin(id);
        }
        self.id = id;
        self.get(gpu, id);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`evict_order`] is the whole eviction policy: oldest first, never a pinned
    /// entry, and it stops the moment the budget holds.
    #[test]
    fn eviction_takes_the_oldest_unpinned_until_the_budget_holds() {
        // (last, bytes, pinned)
        let entries = [
            (5, 10, false),
            (1, 10, true), // the oldest, but stood on
            (2, 10, false),
            (3, 10, false),
            (4, 10, false),
        ];
        let total: u64 = entries.iter().map(|e| e.1).sum();
        assert!(
            evict_order(&entries, total, total).is_empty(),
            "under budget, nothing goes"
        );
        assert_eq!(
            evict_order(&entries, total, 45),
            vec![2],
            "one over: the oldest unpinned alone"
        );
        assert_eq!(
            evict_order(&entries, total, 20),
            vec![2, 3, 4],
            "three over: ascending age, the pinned one skipped"
        );
        assert_eq!(
            evict_order(&entries, total, 0),
            vec![2, 3, 4, 0],
            "every unpinned entry, and never the pinned one"
        );
    }

    /// A store over budget on pinned entries alone evicts nothing, and a store with
    /// nothing built has nothing to say.
    #[test]
    fn pinned_entries_are_never_evicted() {
        let pinned = [(1, 100, true), (2, 100, true)];
        assert!(evict_order(&pinned, 200, 50).is_empty());
        assert!(evict_order(&[], 0, 0).is_empty());
    }
}
