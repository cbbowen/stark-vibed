//! The **channel trio** — color, aux, and the residual a pigment space adds — as
//! one thing rather than as three parallel values (§6.1, §6.7).
//!
//! Every renderer here writes a tile's channels, and each of them used to spell the
//! trio its own way: `merge::Trio`, `transform::Parcel`, `blend::Trio`, plus the
//! bare `(TexHandle, TexHandle, Option<TexHandle>)` tuples the fill and the stroke
//! passed around. Four spellings of one shape, four sets of accessors, and — the
//! part that actually cost something — the residual's `Option` threaded by hand at
//! about a dozen sites:
//!
//! ```text
//! if resid { entries.push(desc::load_tex(8, frag)) }      // the layout
//! 2 + usize::from(resid.is_some())                        // the attachment count
//! self.resid_format.map(|f| pool.acquire_tex(f, source))  // the acquire
//! ```
//!
//! **The failure mode that makes this worth a type is a missing attachment on a
//! pigment document**, which is a validation error rather than a wrong pixel, and
//! which the Oklab-only half of the suite cannot see. Written out per call site, the
//! rule "a document's targets all have a residual or none of them do" is a
//! convention; here it is [`ChannelFormats`], decided once from the color space and
//! carried, so a trio cannot be built half-residual.

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, TexHandle, TilePairHandle, TilePool};

/// The formats a document's tiles and offscreen targets carry — the color space's
/// answer, resolved once (§6.7).
///
/// `resid` is `Some` for every target of a pigment document and `None` for every
/// target of a colorimetric one. It is never a per-call-site choice, which is the
/// whole reason the three travel together.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct ChannelFormats {
    pub(crate) color: wgpu::TextureFormat,
    pub(crate) aux: wgpu::TextureFormat,
    /// The **residual** channel (§6.7), or `None` in a space that has none. Oklab
    /// has no residual, and giving it one would be eight bytes a texel of zeroes on
    /// the default space's tiles plus a third attachment through every pass.
    pub(crate) resid: Option<wgpu::TextureFormat>,
}

impl ChannelFormats {
    pub(crate) fn of(space: &dyn ColorSpace) -> Self {
        Self {
            color: space.color_format(),
            aux: space.aux_format(),
            resid: space.resid_format(),
        }
    }

    /// Whether this space carries a residual (§6.7).
    pub(crate) fn has_resid(&self) -> bool {
        self.resid.is_some()
    }

    /// How many attachments a pass over these channels declares — 2 without a
    /// residual, 3 with. [`Targets::count`]'s other half: one is what the formats
    /// say, the other what a caller brought, and comparing them is the whole of
    /// "did this caller bring the right trio".
    pub(crate) fn count(&self) -> usize {
        2 + usize::from(self.has_resid())
    }

    /// The trio as render-pipeline color targets, each replacing what it writes —
    /// what a pass computing whole texels declares.
    pub(crate) fn targets(&self) -> Vec<Option<wgpu::ColorTargetState>> {
        self.blended(None)
    }

    /// [`Self::targets`] with an explicit blend state on all three.
    ///
    /// The residual composites through the **color's** blend, never the aux's: it is
    /// premultiplied by the same coverage and covers by the same rule, being the rest
    /// of the same color (§6.7). A caller that needs the aux blended differently — as
    /// pass A does, additive on height — spells its three out rather than passing one
    /// here, and that difference is then visible at the call site.
    pub(crate) fn blended(
        &self,
        blend: Option<wgpu::BlendState>,
    ) -> Vec<Option<wgpu::ColorTargetState>> {
        let mut out = vec![
            desc::blended_target(self.color, blend),
            desc::blended_target(self.aux, blend),
        ];
        if let Some(f) = self.resid {
            out.push(desc::blended_target(f, blend));
        }
        out
    }
}

/// One tile's three channel textures, **owned** — a destination a pass writes, or a
/// scratch parcel it writes and a later pass reads.
///
/// The shape [`TilePairHandle`] is built from, before it becomes one.
pub(crate) struct Channels {
    pub(crate) color: TexHandle,
    pub(crate) aux: TexHandle,
    pub(crate) resid: Option<TexHandle>,
}

impl Channels {
    /// Check a trio out of the pool.
    ///
    /// The one place the residual's `Option` decides an acquire, where it used to be
    /// four copies of `self.resid_format.map(|f| pool.acquire_tex(f, source))` — one
    /// per renderer, each of which had to remember that the third texture is
    /// conditional and that it takes the same [`AllocSource`] as the other two.
    pub(crate) fn acquire(pool: &TilePool, formats: ChannelFormats, source: AllocSource) -> Self {
        Self {
            color: pool.acquire_tex(formats.color, source),
            aux: pool.acquire_tex(formats.aux, source),
            resid: formats.resid.map(|f| pool.acquire_tex(f, source)),
        }
    }

    /// The views a pass binds or attaches.
    pub(crate) fn targets(&self) -> Targets<'_> {
        Targets {
            color: self.color.view(),
            aux: self.aux.view(),
            resid: self.resid.as_ref().map(TexHandle::view),
        }
    }

    /// Hand the trio to the document as one tile (§5.2).
    pub(crate) fn into_tile(self) -> TilePairHandle {
        TilePairHandle::new(self.color, self.aux, self.resid)
    }
}

impl TilePairHandle {
    /// This tile's channels as a pass reads or writes them.
    ///
    /// Here rather than in `tile.rs` so the trio's shape stays in one file: a tile
    /// *is* one of these, and the stroke path's scratch and destination tiles were
    /// spelling out `(color_view(), aux_view(), resid_view())` at every render pass
    /// to say so.
    pub(crate) fn targets(&self) -> Targets<'_> {
        Targets {
            color: self.color_view(),
            aux: self.aux_view(),
            resid: self.resid_view(),
        }
    }
}

/// The channel trio **borrowed** — what a pass actually reads or writes, whether it
/// came from a [`Channels`], a viewport-sized offscreen set, or a tile the document
/// already holds.
#[derive(Copy, Clone)]
pub(crate) struct Targets<'a> {
    pub(crate) color: &'a wgpu::TextureView,
    pub(crate) aux: &'a wgpu::TextureView,
    pub(crate) resid: Option<&'a wgpu::TextureView>,
}

impl<'a> Targets<'a> {
    /// The color attachments in target order, with `resid` at location 2 where the
    /// space has one — the order every pipeline over these channels declares.
    ///
    /// Returned as a fixed array plus [`Self::count`] rather than a `Vec`, so the
    /// caller slices (`&attachments[..t.count()]`). These are built once per render
    /// pass encoded, and a render pass per tile is the common case, which is the rate
    /// `ScopedResources` exists to keep allocations off (§6.2).
    pub(crate) fn attachments(
        &self,
        ops: wgpu::Operations<wgpu::Color>,
    ) -> [Option<wgpu::RenderPassColorAttachment<'a>>; 3] {
        [
            Some(desc::attach(self.color, ops)),
            Some(desc::attach(self.aux, ops)),
            self.resid.map(|v| desc::attach(v, ops)),
        ]
    }

    /// How many of [`Self::attachments`] are real — 2 without a residual, 3 with.
    pub(crate) fn count(&self) -> usize {
        2 + usize::from(self.resid.is_some())
    }
}
