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

use std::collections::HashMap;
use std::hash::Hash;

use crate::gpu::context::GpuContext;

/// A resource the frontend supplies bytes for.
pub trait Resource: Copy + Eq + Hash + std::fmt::Debug {
    /// The live GPU object built from the bytes.
    type Gpu;

    /// Whether this id needs no registered bytes — the procedural default, which is
    /// also what an id with missing bytes falls back to.
    fn is_builtin(self) -> bool;

    /// Build the GPU object. `bytes` is `None` for a builtin id, and also when a
    /// non-builtin id's bytes have not been registered yet.
    fn build(self, gpu: &GpuContext, bytes: Option<&[u8]>) -> Self::Gpu;
}

/// The registered bytes for a resource, the id in use, and the objects built so far.
pub struct Registry<R: Resource> {
    bytes: HashMap<R, Vec<u8>>,
    id: R,
    /// Everything built, keyed by id; `built[&id]` is the one in use and is always
    /// present.
    ///
    /// A **cache**, not a set of live resources, and it exists for the canvas
    /// surface's sake. Once the deposition tooth reads the surface (§6.4), a stroke
    /// replayed from before a `SetSurface` has to deposit against the ground it was
    /// actually painted on rather than the one in use now — so [`get`](Self::get)
    /// can be asked for any id at any time, and re-decoding a multi-megabyte PNG
    /// every time the log crosses that boundary is not a thing to do on an undo
    /// step. Bounded by the number of distinct ids a document ever names, which is
    /// the size of the id enum.
    built: HashMap<R, R::Gpu>,
}

impl<R: Resource> Registry<R> {
    /// A registry holding only `id`, which must therefore be a builtin (nothing has
    /// been registered yet).
    pub fn new(gpu: &GpuContext, id: R) -> Self {
        debug_assert!(id.is_builtin(), "a fresh registry can only hold a builtin");
        Self {
            bytes: HashMap::new(),
            id,
            built: HashMap::from([(id, id.build(gpu, None))]),
        }
    }

    /// Which resource is in use.
    pub fn id(&self) -> R {
        self.id
    }

    /// The live GPU object — the one `id()` names.
    pub fn current(&self) -> &R::Gpu {
        self.built
            .get(&self.id)
            .expect("the id in use is always built")
    }

    /// The object for **any** `id`, built on demand and cached — without changing
    /// which one is in use.
    ///
    /// This is what a replay asks: the surface a stroke deposits against is the one
    /// the document was on *at that point in the log* (§6.4), which is a question
    /// about the action being applied, not about what the compositor is showing.
    /// Also how [`register`](Self::register) and [`set`](Self::set) rebuild the
    /// object they have just invalidated: they call this and drop the reference. It
    /// used to be a second method, `ensure`, which was this one without the return —
    /// the same four lines, kept in step by nothing.
    pub fn get(&mut self, gpu: &GpuContext, id: R) -> &R::Gpu {
        if !self.built.contains_key(&id) {
            let obj = self.make(gpu, id);
            self.built.insert(id, obj);
        }
        &self.built[&id]
    }

    /// Whether `id` is ready to use: builtins always are, everything else once its
    /// bytes have been [`register`](Self::register)ed.
    pub fn is_loaded(&self, id: R) -> bool {
        id.is_builtin() || self.bytes.contains_key(&id)
    }

    /// The registered bytes for `id`, if any — what a save file bundles and what a
    /// peer is served (§8, §12.4). `None` for a builtin, which has none by
    /// definition, and for an id whose image has not arrived.
    pub fn bytes(&self, id: R) -> Option<&[u8]> {
        self.bytes.get(&id).map(|b| b.as_slice())
    }

    /// Provide bytes for `id`. Returns `true` if the live object was rebuilt, which
    /// happens exactly when `id` is the one in use — the caller then has to rebind
    /// it wherever it is sampled.
    pub fn register(&mut self, gpu: &GpuContext, id: R, bytes: Vec<u8>) -> bool {
        self.bytes.insert(id, bytes);
        // Whatever was built for this id was built from the *old* bytes — usually the
        // builtin fallback, standing in while the fetch was in flight. Dropping it is
        // what makes `get` return the real thing from now on.
        self.built.remove(&id);
        if id != self.id {
            return false;
        }
        self.get(gpu, id);
        true
    }

    /// Switch to `id`. Returns `true` if it changed (and so was rebuilt); switching
    /// to what is already in use is a no-op, not a rebuild.
    pub fn set(&mut self, gpu: &GpuContext, id: R) -> bool {
        if id == self.id {
            return false;
        }
        self.id = id;
        self.get(gpu, id);
        true
    }

    fn make(&self, gpu: &GpuContext, id: R) -> R::Gpu {
        let bytes = self.bytes.get(&id);
        if bytes.is_none() && !id.is_builtin() {
            tracing::warn!(id = ?id, "no registered bytes; falling back to the builtin");
        }
        id.build(gpu, bytes.map(|b| b.as_slice()))
    }
}
