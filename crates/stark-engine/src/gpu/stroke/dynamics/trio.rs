//! The loop's working **channel trio** (§6.1, §6.7): colour, aux, and the residual a
//! pigment space adds, checked out of the scratch pool as one thing.
//!
//! [`Channels`](crate::device::channels::Channels) is the same trio over pooled *tiles*.
//! This one is over [`scratch`](crate::device::scratch) leases, because everything here is
//! storage-written working memory at a region's size rather than a tile's, and its
//! format is the shader's rather than the tile pool's.
//!
//! **Each channel is created at the format of the binding that writes it** (§6.10). The
//! formats agree today, so a texture built at a sibling's was not a wrong picture — but
//! it is one declaration standing in for another, and nothing would report the day they
//! diverge except `create_texture` on whichever texture was the one still named
//! correctly.

use stark_shaders::Binding;

use crate::device::channels::Targets;
use crate::device::scratch::{Key, LeasedTex, SubmitScope};

/// How long a trio's leases live: to the end of the run, or to the submit of the piece
/// being recorded ([`SubmitScope::take_run`], [`SubmitScope::take_piece`]).
#[derive(Clone, Copy)]
pub(super) enum Life {
    Run,
    Piece,
}

impl Life {
    fn take(self, scope: &mut SubmitScope, key: Key) -> LeasedTex {
        match self {
            Self::Run => scope.take_run(key),
            Self::Piece => scope.take_piece(key),
        }
    }
}

/// Three leased channel textures: the two every build has and the residual (§6.7),
/// decided once from the colour space rather than per channel.
///
/// Each carries its texture beside its view, because a pass binds a view and a copy
/// addresses a texture, and the loop does both to the same allocation — the region is
/// storage-written by the kernels and cut into tiles by copy.
pub(super) struct Trio {
    pub(super) color: LeasedTex,
    pub(super) aux: LeasedTex,
    /// `Some` exactly where the colour space has a residual — the shader's own
    /// `@if(resid)` gate, which is also what decides whether `at[2]` is ever read.
    pub(super) resid: Option<LeasedTex>,
}

impl Trio {
    /// Check a trio out at `size` and `usage`, each channel at the format **its own**
    /// declaration in `at` names (§6.10), labelled by `labels` in the same order.
    pub(super) fn take(
        scope: &mut SubmitScope,
        life: Life,
        size: (u32, u32),
        usage: wgpu::TextureUsages,
        at: [Binding; 3],
        labels: [&'static str; 3],
        resid: bool,
    ) -> Self {
        let keys = std::array::from_fn(|i: usize| Key {
            size,
            format: at[i].storage_format(),
            usage,
            label: labels[i],
        });
        Self::take_at(scope, life, keys, resid)
    }

    /// [`take`](Self::take) against keys the caller built — for a trio whose channels
    /// are copied out to keys that have to be the *same* ones, which is easiest to
    /// mean when both ends ask one function for them ([`tool`](super::tool)).
    pub(super) fn take_at(
        scope: &mut SubmitScope,
        life: Life,
        keys: [Key; 3],
        resid: bool,
    ) -> Self {
        Self {
            color: life.take(scope, keys[0]),
            aux: life.take(scope, keys[1]),
            resid: resid.then(|| life.take(scope, keys[2])),
        }
    }

    /// The views a pass binds or attaches, so the loop's scratch reaches the shared
    /// attachment and count rules (`gpu::channels`).
    pub(super) fn targets(&self) -> Targets<'_> {
        Targets {
            color: &self.color.view,
            aux: &self.aux.view,
            resid: self.resid_view(),
        }
    }

    /// The residual's view, where there is one.
    pub(super) fn resid_view(&self) -> Option<&wgpu::TextureView> {
        self.resid.as_ref().map(|l| &l.view)
    }

    /// The residual's texture, where there is one — what a copy addresses.
    pub(super) fn resid_tex(&self) -> Option<&wgpu::Texture> {
        self.resid.as_ref().map(|l| &l.tex)
    }
}
