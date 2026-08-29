//! The **owned** viewport-sized render targets: one attachment, and the channel trio
//! made of three (§6.1, §6.7).
//!
//! `channels.rs` holds the trio for pooled tiles ([`Channels`](super::super::channels::Channels))
//! and for borrows ([`Targets`](super::super::channels::Targets)); this is the third
//! ownership, the one whose members free their memory when they are replaced. It was
//! written twice — once as `blend`'s `Trio`, once as the first three fields of
//! `media`'s `Offscreen`, each with its own `targets()` — and the second of those had
//! to pass its three around behind a disabled lint. One shape, one name.
//!
//! Here rather than in `composite.rs` because neither is "the part no single pass
//! owns", which is what that file says is left in it: four sibling passes reached
//! *up* through `super::Attachment` to get at a general resource type.

use stark_model::geom::Extent2;

use super::super::channels::{ChannelFormats, Targets};

/// A viewport-sized offscreen render target — pass A's channels, the blend
/// scratch, the supersampled target — that **returns its memory when it is
/// replaced** rather than merely releasing its handle
/// ([`ScopedResources`](crate::gpu::submit::ScopedResources)).
///
/// These are the largest allocations the application makes: a whole set is rebuilt
/// whenever the target changes size or the zoom crosses a supersampling threshold
/// ([`Compositor::ensure_targets`]), budgeted by `resolve`'s
/// `MAX_SUPERSAMPLED_BYTES` at up to 224 MiB a set. On the web, dropping the view
/// frees none of it: it releases the JS handle and leaves the texture to a collector
/// that cannot see the GPU memory behind it, so nothing reclaims it until that
/// collector happens to run. Survivable at a zoom notch, and fatal at a *rate* — a
/// window-resize drag reports a new size every animation frame, so a second of
/// dragging strands a second's worth of whole sets at once and the GPU process dies
/// with every device on it.
///
/// So the texture is kept beside its view and `destroy()`d here, which is safe for
/// the reason `gpu::submit` gives: WebGPU defers the real free until the in-flight
/// work naming it completes. Handing back both halves would have let each call site
/// arrange this by hand; a target that cannot be built without it is what keeps the
/// next one from being the attachment that forgets.
pub(super) struct Attachment {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Attachment {
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { tex, view }
    }

    /// What a pass attaches, and what a bind group naming this target reads.
    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.view
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
        let make = |format, label| Attachment::new(device, size, format, label);
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
