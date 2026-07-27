//! Frontend-provided GPU resources: registered bytes, which one is in use, and the
//! live object built from it (DESIGN.md §6.4, §6.6).
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

/// The registered bytes for a resource, the id in use, and the object built for it.
pub struct Registry<R: Resource> {
    bytes: HashMap<R, Vec<u8>>,
    id: R,
    current: R::Gpu,
}

impl<R: Resource> Registry<R> {
    /// A registry holding only `id`, which must therefore be a builtin (nothing has
    /// been registered yet).
    pub fn new(gpu: &GpuContext, id: R) -> Self {
        debug_assert!(id.is_builtin(), "a fresh registry can only hold a builtin");
        Self {
            bytes: HashMap::new(),
            id,
            current: id.build(gpu, None),
        }
    }

    /// Which resource is in use.
    pub fn id(&self) -> R {
        self.id
    }

    /// The live GPU object.
    pub fn current(&self) -> &R::Gpu {
        &self.current
    }

    /// Whether `id` is ready to use: builtins always are, everything else once its
    /// bytes have been [`register`](Self::register)ed.
    pub fn is_loaded(&self, id: R) -> bool {
        id.is_builtin() || self.bytes.contains_key(&id)
    }

    /// Provide bytes for `id`. Returns `true` if the live object was rebuilt, which
    /// happens exactly when `id` is the one in use — the caller then has to rebind
    /// it wherever it is sampled.
    pub fn register(&mut self, gpu: &GpuContext, id: R, bytes: Vec<u8>) -> bool {
        self.bytes.insert(id, bytes);
        if id != self.id {
            return false;
        }
        self.rebuild(gpu);
        true
    }

    /// Switch to `id`. Returns `true` if it changed (and so was rebuilt); switching
    /// to what is already in use is a no-op, not a rebuild.
    pub fn set(&mut self, gpu: &GpuContext, id: R) -> bool {
        if id == self.id {
            return false;
        }
        self.id = id;
        self.rebuild(gpu);
        true
    }

    fn rebuild(&mut self, gpu: &GpuContext) {
        let bytes = self.bytes.get(&self.id);
        if bytes.is_none() && !self.id.is_builtin() {
            tracing::warn!(id = ?self.id, "no registered bytes; falling back to the builtin");
        }
        self.current = self.id.build(gpu, bytes.map(|b| b.as_slice()));
    }
}
