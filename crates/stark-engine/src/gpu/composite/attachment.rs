//! The **owned** viewport-sized render targets: one attachment, and the channel trio
//! made of three (§6.1, §6.7).
//!
//! The third ownership beside `channels.rs`'s pooled
//! [`Channels`](crate::device::channels::Channels) and borrowed [`Targets`]: the one
//! whose members free their memory when they are replaced.

use crate::view::Extent2;

use crate::device::channels::{ChannelFormats, Targets};
use crate::device::desc;

/// A viewport-sized offscreen render target — pass A's channels, the blend
/// scratch, the supersampled target, the focal blur's planes — that **returns its
/// memory when it is replaced** rather than merely releasing its handle
/// ([`gpu::scratch`](crate::device::scratch)).
///
/// These are the largest allocations the application makes: a whole set is rebuilt
/// whenever the target changes size or the zoom crosses a supersampling threshold
/// (`TargetSized::ensure_targets`), budgeted by `resolve`'s `MAX_SUPERSAMPLED_BYTES`
/// at up to 224 MiB a set. On the web, dropping the view frees none of it — the
/// texture is left to a collector that cannot see the GPU memory behind it — so a
/// window-resize drag, which reports a new size every animation frame, strands whole
/// sets at a rate and takes the GPU process down with every device on it. Hence the
/// `destroy()` in `Drop`, safe because WebGPU defers the real free until in-flight
/// work naming the texture completes (`gpu::scratch`).
pub(super) struct Attachment {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

/// What a target a pass renders into and a later pass samples is used for.
pub(super) const RENDERED: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

impl Attachment {
    /// A target a pass renders into and a later pass samples.
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        Self::with_usage(device, size, format, RENDERED, label)
    }

    /// [`Self::new`] for a target used some other way — the focal blur's planes,
    /// which a compute pass also stores into (§21.12).
    pub(super) fn with_usage(
        device: &wgpu::Device,
        size: Extent2,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            usage,
            ..desc::sampled_texture(label, format, size)
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { tex, view }
    }

    /// What a pass attaches, and what a bind group naming this target reads.
    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The texture itself, for a copy out of it.
    pub(super) fn texture(&self) -> &wgpu::Texture {
        &self.tex
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        self.tex.destroy();
    }
}

/// One set of channel targets — color, aux, and (in a space that has one) the
/// residual — owned rather than borrowed, as [`Targets`] is the borrowed view of.
pub(super) struct Trio {
    pub(super) color: Attachment,
    pub(super) aux: Attachment,
    pub(super) resid: Option<Attachment>,
}

impl Trio {
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        labels: (&str, &str, &str),
        formats: ChannelFormats,
    ) -> Self {
        Self::with_usage(device, size, labels, formats, RENDERED)
    }

    /// [`Self::new`] for a trio used some other way too — the eyedropper's, which is
    /// copied out of (§18.0.2).
    pub(super) fn with_usage(
        device: &wgpu::Device,
        size: Extent2,
        labels: (&str, &str, &str),
        formats: ChannelFormats,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let make = |format, label| Attachment::with_usage(device, size, format, usage, label);
        Self {
            color: make(formats.color, labels.0),
            aux: make(formats.aux, labels.1),
            // A pigment document isolates its residual alongside its concentrations:
            // the blend reads both to work out what light the layer carried
            // (§6.7), so a level that isolated only the color would hand the pass
            // a mixture and none of the correction that makes it a color.
            resid: formats.resid.map(|f| make(f, labels.2)),
        }
    }

    pub(super) fn targets(&self) -> Targets<'_> {
        Targets {
            color: self.color.view(),
            aux: self.aux.view(),
            resid: self.resid.as_ref().map(Attachment::view),
        }
    }
}
