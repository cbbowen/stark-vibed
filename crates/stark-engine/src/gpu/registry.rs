//! Frontend-provided GPU resources: registered bytes, which one is in use, and the
//! live object built from it (§6.4, §6.6).
//!
//! The engine embeds no image bytes. The frontend fetches them at runtime and hands
//! them over, which means every such resource has the same three-part shape — a map
//! of registered bytes, the id currently in use, and the GPU object built for that
//! id — and the same two operations: *register bytes* (rebuild if they are the ones
//! in use) and *switch* (rebuild if it actually changed). Written twice, for the
//! canvas surface and the lighting environment, those were the same six lines with
//! different nouns.
//!
//! Each resource keeps one **builtin** id that needs no bytes at all — `Flat` for
//! surfaces, the procedural `Neutral` for environments — which is also the fallback
//! when an id's bytes have not arrived yet.
//!
//! # The store is shared; the *choice* is not
//!
//! The bytes and the built objects live behind an `Arc`, and `Clone` hands out a
//! sibling registry over the same store with its own current id. That is what lets a
//! second engine on the same device ([`Engine::new_sharing`]) reuse every decode this
//! one has paid for — a ground's height map is a PNG decode plus two whole-image CPU
//! passes, an environment is an HDR decode plus a full mip chain — while still being
//! free to stand on a different ground. Which id is *in use* is the one part of the
//! shape that is genuinely per-engine: it mirrors that engine's document (§6.4) or
//! its view (§6.3), and siblings agreeing on it would be a bug, not a feature.
//!
//! [`Engine::new_sharing`]: crate::Engine::new_sharing

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

    /// Whether this id needs no registered bytes — the procedural default, which is
    /// also what an id with missing bytes falls back to.
    fn is_builtin(self) -> bool;

    /// Build the GPU object. `bytes` is `None` for a builtin id, and also when a
    /// non-builtin id's bytes have not been registered yet.
    fn build(self, gpu: &GpuContext, bytes: Option<&[u8]>) -> Self::Gpu;
}

/// The shared half: registered bytes and the objects built from them, one map each.
struct Store<R: Resource> {
    bytes: HashMap<R, Vec<u8>>,
    /// Everything built, keyed by id; the id in use is always present.
    ///
    /// A **cache**, not a set of live resources, and it exists for the canvas
    /// surface's sake. Once the deposition tooth reads the surface (§6.4), a stroke
    /// replayed from before a `SetSurface` has to deposit against the ground it was
    /// actually painted on rather than the one in use now — so [`Registry::get`]
    /// can be asked for any id at any time, and re-decoding a multi-megabyte PNG
    /// every time the log crosses that boundary is not a thing to do on an undo
    /// step. Bounded by the number of distinct ids a document ever names, which is
    /// the size of the id enum.
    built: HashMap<R, R::Gpu>,
}

impl<R: Resource> Store<R> {
    /// The object for `id`, built on demand and cached.
    fn get(&mut self, gpu: &GpuContext, id: R) -> R::Gpu {
        if let Some(built) = self.built.get(&id) {
            return built.clone();
        }
        let bytes = self.bytes.get(&id);
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

    fn store(&self) -> std::sync::MutexGuard<'_, Store<R>> {
        self.store.lock().expect("registry store poisoned")
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
    /// This is what a replay asks: the surface a stroke deposits against is the one
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
        id.is_builtin() || self.store().bytes.contains_key(&id)
    }

    /// The registered bytes for `id`, if any — what a save file bundles and what a
    /// peer is served (§8, §12.4). `None` for a builtin, which has none by
    /// definition, and for an id whose image has not arrived.
    pub fn bytes(&self, id: R) -> Option<Vec<u8>> {
        self.store().bytes.get(&id).cloned()
    }

    /// Provide bytes for `id`. Returns `true` if the live object was rebuilt, which
    /// happens exactly when `id` is the one in use — the caller then has to rebind
    /// it wherever it is sampled.
    ///
    /// A sibling whose current id is also `id` keeps its stale binding until it next
    /// rebinds — exactly the exposure two independent registries had, since neither
    /// saw the other's bytes arrive at all.
    pub fn register(&self, gpu: &GpuContext, id: R, bytes: Vec<u8>) -> bool {
        let mut store = self.store();
        store.bytes.insert(id, bytes);
        // Whatever was built for this id was built from the *old* bytes — usually the
        // builtin fallback, standing in while the fetch was in flight. Dropping it is
        // what makes `get` return the real thing from now on.
        store.built.remove(&id);
        if id != self.id {
            return false;
        }
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
