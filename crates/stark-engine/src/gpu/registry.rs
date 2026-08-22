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
//! [`Engine::new_sharing`]: crate::Engine::new_sharing

use crate::unpoisoned;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::gpu::context::GpuContext;

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

    /// The registered bytes this id builds from.
    fn content(self) -> Self::Content;

    /// Whether this id needs no registered bytes — the procedural default, which is
    /// also what an id with missing bytes falls back to.
    fn is_builtin(self) -> bool;

    /// Build the GPU object. `bytes` is `None` for a builtin id, and also when a
    /// non-builtin id's bytes have not been registered yet.
    fn build(self, gpu: &GpuContext, bytes: Option<&[u8]>) -> Self::Gpu;
}

/// The shared half: registered bytes and the objects built from them, one map each.
struct Store<R: Resource> {
    bytes: HashMap<R::Content, Vec<u8>>,
    /// Everything built, keyed by id; the id in use is always present.
    ///
    /// A **cache**, not a set of live resources, and it exists for the canvas
    /// substrate's sake. Once the deposition tooth reads the substrate (§6.4), a stroke
    /// replayed from before a `SetSubstrate` has to deposit against the substrate it was
    /// actually painted on rather than the one in use now — so [`Registry::get`]
    /// can be asked for any id at any time, and re-decoding a multi-megabyte PNG
    /// every time the log crosses that boundary is not a thing to do on an undo
    /// step. Bounded by the number of distinct ids a document ever names — for a
    /// substrate, by the number of *(substrate, scale)* pairs it ever names, which is what
    /// `SubstrateScale`'s ladder is there to keep small.
    built: HashMap<R, R::Gpu>,
}

impl<R: Resource> Store<R> {
    /// The object for `id`, built on demand and cached.
    fn get(&mut self, gpu: &GpuContext, id: R) -> R::Gpu {
        if let Some(built) = self.built.get(&id) {
            return built.clone();
        }
        let bytes = self.bytes.get(&id.content());
        if bytes.is_none() && !id.is_builtin() {
            tracing::warn!(id = ?id, "no registered bytes; falling back to the builtin");
        }
        let obj = id.build(gpu, bytes.map(|b| b.as_slice()));
        self.built.insert(id, obj.clone());
        obj
    }
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
        Self {
            store: Arc::clone(&self.store),
            id: self.id,
        }
    }
}

impl<R: Resource> Registry<R> {
    /// A registry holding only `id`, which must therefore be a builtin (nothing has
    /// been registered yet).
    pub fn new(gpu: &GpuContext, id: R) -> Self {
        debug_assert!(id.is_builtin(), "a fresh registry can only hold a builtin");
        Self {
            store: Arc::new(Mutex::new(Store {
                bytes: HashMap::new(),
                built: HashMap::from([(id, id.build(gpu, None))]),
            })),
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
            .built
            .get(&self.id)
            .expect("the id in use is always built")
            .clone()
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
    pub fn get(&self, gpu: &GpuContext, id: R) -> R::Gpu {
        self.store().get(gpu, id)
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
        self.store().bytes.get(&content).cloned()
    }

    /// Provide bytes for `content`. Returns `true` if the live object was rebuilt,
    /// which happens exactly when the id in use is built from that content — the
    /// caller then has to rebind it wherever it is sampled.
    ///
    /// A sibling standing on the same content keeps its stale binding until it next
    /// rebinds — exactly the exposure two independent registries had, since neither
    /// saw the other's bytes arrive at all.
    pub fn register(&self, gpu: &GpuContext, content: R::Content, bytes: Vec<u8>) -> bool {
        let mut store = self.store();
        store.bytes.insert(content, bytes);
        // Whatever was built from this content was built from the *old* bytes —
        // usually the builtin fallback, standing in while the fetch was in flight.
        // Dropping it is what makes `get` return the real thing from now on. Every
        // build of it, since one height map bakes a substrate per scale (see the module
        // note): a `retain` rather than a `remove`, because the bytes are one key and
        // the objects are another.
        store.built.retain(|id, _| id.content() != content);
        if self.id.content() != content {
            return false;
        }
        let id = self.id;
        store.get(gpu, id);
        true
    }

    /// Switch to `id`. Returns `true` if it changed (and so was rebuilt); switching
    /// to what is already in use is a no-op, not a rebuild.
    pub fn set(&mut self, gpu: &GpuContext, id: R) -> bool {
        if id == self.id {
            return false;
        }
        self.id = id;
        self.store().get(gpu, id);
        true
    }
}
