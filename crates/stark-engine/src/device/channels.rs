//! The **channel trio** — color, aux, and the residual a pigment space adds — as
//! one thing rather than as three parallel values (§6.1, §6.7).
//!
//! Every renderer here writes a tile's channels, and the residual's `Option` is what
//! costs: it gates the layout entry, the attachment count and the acquire alike.
//!
//! **The failure mode that makes this worth a type is a missing attachment on a pigment
//! document**, which is a validation error rather than a wrong pixel, and which the
//! Oklab-only half of the suite cannot see. [`ChannelFormats`] decides it once from the
//! color space and carries it, so a trio cannot be built half-residual.

use crate::colorspace::ColorSpace;
use crate::device::context::GpuContext;
use crate::device::desc;
use crate::device::tile::{AllocSource, TexHandle, TilePairHandle, TilePool};

/// A tile's color channels, in every space (§6.1, §6.7): the latent premultiplied by
/// per-unit opacity. The tile pool is sized for this and [`TILE_AUX_FORMAT`], so a
/// space does not choose them.
pub(crate) const TILE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// A tile's height channel — the amount of paint (§6.1).
pub(crate) const TILE_AUX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The formats a document's tiles and offscreen targets carry — the color space's
/// answer, resolved once (§6.7).
///
/// `resid` is `Some` for every target of a pigment document and `None` for every target
/// of a colorimetric one — never a per-call-site choice, which is why the three travel
/// together.
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
            color: TILE_COLOR_FORMAT,
            aux: TILE_AUX_FORMAT,
            // The rest of the same color, so the color's format and its decode:
            // premultiplied by the same opacity, which it holds in its own alpha
            // because fixed-function "over" reads each target's own (§6.7).
            resid: space.has_resid().then_some(TILE_COLOR_FORMAT),
        }
    }

    /// Whether this space carries a residual (§6.7).
    pub(crate) fn has_resid(&self) -> bool {
        self.resid.is_some()
    }

    /// What one texel of this trio costs, across all of its targets — 10 bytes in
    /// Oklab (`Rgba16Float` + `R16Float`), 18 with a residual (§6.7).
    ///
    /// The one place the residual's *size* is counted, so a budget over viewport-sized
    /// attachments (`composite::resolve`) cannot silently assume the colorimetric space.
    pub(crate) fn bytes_per_px(&self) -> u64 {
        let of = |f: wgpu::TextureFormat| u64::from(crate::device::desc::texel_bytes(f));
        of(self.color) + of(self.aux) + self.resid.map_or(0, of)
    }

    /// The trio as render-pipeline color targets, each replacing what it writes —
    /// what a pass computing whole texels declares.
    pub(crate) fn targets(&self) -> Vec<Option<wgpu::ColorTargetState>> {
        self.blended(None)
    }

    /// [`Self::targets`] with an explicit blend state on all three. A deposit, which
    /// blends the aux differently, takes [`Self::deposit_targets`].
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

    /// The trio as a **deposit** lands it (§6.1, §6.7): color premultiplied-over,
    /// `aux` additive because height is an amount, and the residual over by the
    /// color's rule, being the rest of the same color.
    ///
    /// `aux` is the tile's own format or the wide one a stroke accumulates in.
    /// Location 2 is `None` without a residual — a hole [`desc::targets_for`] keeps or
    /// drops by what the entry point says it writes (§6.10), never a count sliced here.
    pub(crate) fn deposit_targets(
        &self,
        aux: wgpu::TextureFormat,
    ) -> [Option<wgpu::ColorTargetState>; 3] {
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        [
            desc::blended_target(self.color, over),
            desc::blended_target(aux, Some(wgpu::BlendState::ADDITIVE)),
            self.resid.and_then(|f| desc::blended_target(f, over)),
        ]
    }
}

/// One tile's three channel textures, **owned** — a destination a pass writes, or a
/// scratch parcel it writes and a later pass reads.
///
/// The shape [`TilePairHandle`] is built from, before it becomes one.
///
/// `Clone` because [`Self::scratch`] hands one copy to the recording and one to the
/// caller; the textures are `Arc`-backed, so both copies name the same trio.
#[derive(Clone)]
pub(crate) struct Channels {
    pub(crate) color: TexHandle,
    pub(crate) aux: TexHandle,
    pub(crate) resid: Option<TexHandle>,
}

impl Channels {
    /// Check a trio out of the pool — **the one place the residual's `Option` decides
    /// an acquire**, and where the third texture is held to the same [`AllocSource`] as
    /// the other two.
    pub(crate) fn acquire(pool: &TilePool, formats: ChannelFormats, source: AllocSource) -> Self {
        Self {
            color: pool.acquire_tex(formats.color, source),
            aux: pool.acquire_tex(formats.aux, source),
            resid: formats.resid.map(|f| pool.acquire_tex(f, source)),
        }
    }

    /// Check a trio out of the pool **as scratch**, registered with the recording
    /// that names it in the same breath.
    ///
    /// The pairing is the whole point. Nothing in a recorded encoder has run, so a
    /// scratch trio released before its submit is handed straight back out — to a pass
    /// in the very same encoder, which overwrites it before the earlier pass ever reads
    /// it (`gpu::scratch`). Acquiring and holding here cannot come apart.
    ///
    /// Distinct from [`Self::acquire`], which is what a **destination** takes: that trio
    /// is returned to the caller and becomes a tile of the document, so its life is the
    /// document's and not the recording's.
    pub(crate) fn scratch(
        scope: &mut crate::device::scratch::SubmitScope,
        pool: &TilePool,
        formats: ChannelFormats,
        source: AllocSource,
    ) -> Self {
        let trio = Self::acquire(pool, formats, source);
        scope.hold(trio.clone());
        trio
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
    /// Here rather than in `tile.rs` so the trio's shape stays in one file: a tile *is*
    /// one of these.
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
    /// The trio as a pass's attachments, `resid` at location 2 where the space has one
    /// — the order every pipeline over these channels declares — each with `ops`.
    ///
    /// [`desc::Attachments`] reads the length off the last attached location, so the
    /// residual's absence is a hole rather than a count the caller has to slice to.
    pub(crate) fn with(&self, ops: wgpu::Operations<wgpu::Color>) -> desc::Attachments<'a> {
        desc::Attachments::with(ops, [Some(self.color), Some(self.aux), self.resid])
    }
}

/// The trio as a pass writes it, cleared, the residual at location 2 where the space
/// has one.
impl<'a> From<Targets<'a>> for desc::Attachments<'a> {
    fn from(t: Targets<'a>) -> Self {
        t.with(desc::CLEAR)
    }
}

/// The 1×1 stand-ins a pass binds where a tile does not exist — one per persistent
/// tile channel.
///
/// **"There is no tile here" is one question, and this is its one answer.** The fill,
/// the transform's combine and the stroke's integrate all bind these and read them
/// through clamped loads, so the same shader code reads a real tile and a hole
/// (§6.8's pattern). Built once and cloned into each renderer — a `TextureView` is an
/// `Arc` handle, so a clone is a bump.
#[derive(Clone)]
pub(crate) struct Zeroes {
    pub(crate) color: wgpu::TextureView,
    pub(crate) aux: wgpu::TextureView,
    /// The residual channel's stand-in, in a space that has one (§6.7). Bare canvas
    /// has no residual for the same reason it has no color, and this is what the
    /// clamped loads read there.
    pub(crate) resid: Option<wgpu::TextureView>,
}

impl Zeroes {
    /// The stand-ins as a trio a pass binds.
    pub(crate) fn targets(&self) -> Targets<'_> {
        Targets {
            color: &self.color,
            aux: &self.aux,
            resid: self.resid.as_ref(),
        }
    }

    /// `targets`, or these stand-ins where there are none — **the one answer to "a
    /// tile, or the 1×1 zeroes"** (§6.8's pattern).
    ///
    /// Whether a trio exists is one question, so it has one answer, and the residual's
    /// presence rides along with the other two rather than being decided again (§6.7).
    pub(crate) fn or<'a>(&'a self, targets: Option<Targets<'a>>) -> Targets<'a> {
        targets.unwrap_or_else(|| self.targets())
    }

    pub(crate) fn new(ctx: &GpuContext, formats: ChannelFormats) -> Self {
        Self {
            color: desc::zero_texture(ctx, formats.color, "stark zero color"),
            aux: desc::zero_texture(ctx, formats.aux, "stark zero aux"),
            resid: formats
                .resid
                .map(|f| desc::zero_texture(ctx, f, "stark zero resid")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tile::SCRATCH_AUX_FORMAT;

    /// A deposit lands the residual in the color's format and by the color's blend, at
    /// location 2 exactly where the space has one; the aux is the caller's, additive.
    #[test]
    fn a_deposit_lands_the_residual_as_the_rest_of_the_color() {
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let check = |space: &dyn ColorSpace| {
            let formats = ChannelFormats::of(space);
            let targets = formats.deposit_targets(SCRATCH_AUX_FORMAT);
            let state = |i: usize| targets[i].as_ref().map(|t| (t.format, t.blend));
            assert_eq!(state(0), Some((TILE_COLOR_FORMAT, over)));
            assert_eq!(
                state(1),
                Some((SCRATCH_AUX_FORMAT, Some(wgpu::BlendState::ADDITIVE)))
            );
            assert_eq!(
                state(2),
                space.has_resid().then_some((TILE_COLOR_FORMAT, over))
            );
        };
        check(&crate::colorspace::OkLabColorSpace);
        #[cfg(feature = "mixbox")]
        check(&crate::colorspace::MixboxColorSpace);
    }
}
