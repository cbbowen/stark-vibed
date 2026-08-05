//! The eyedropper: sampling a colour off the canvas (§18.0.2).
//!
//! A **request** rather than a command — it has to answer — and one that goes
//! through the *same* draw list rendering does ([`Engine::composite_groups`]), so
//! what it reports is the paint the screen is drawing rather than a second opinion
//! about it. What it deliberately does not sample is the media pass's output: that
//! lights the paint, tonemaps it and encodes sRGB, so picking it up would load the
//! brush with a colour the palette never mixed.

use super::Engine;
use super::render::visible_tiles;
use crate::document::LayerId;
use crate::geom::{Extent2, ViewTransform};

/// Which layers an eyedropper sample is taken from (§18.0.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PickSource {
    /// Every visible layer, composited — the colour the canvas shows.
    #[default]
    Composite,
    /// Every visible layer *over the substrate* (§15.5): the same stack, with the
    /// canvas colour standing in wherever the paint does not cover.
    ///
    /// The one source that answers on bare canvas, and the only one whose answer can
    /// be a colour no layer holds — a glaze over the ground is a mixture of the two.
    /// That is what it is for: matching what the eye sees at a point rather than what
    /// is stored there, which is the question being asked when the paint is thin.
    CompositeOverSubstrate,
    /// One layer alone: the colour that layer would have if it were the only one in
    /// the document. What "sample the current layer" has to mean, since a glaze on
    /// top of somebody else's underpainting is not the same paint as the two mixed.
    Layer(LayerId),
}

/// How an eyedropper sample is taken (§18.0.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct PickOptions {
    pub source: PickSource,
    /// Half-width of the averaged square, in **canvas** px: 0 samples the single
    /// canvas pixel under the point, 1 averages 3×3, 2 averages 5×5 — the prior
    /// art's point / N×N sampler.
    ///
    /// Canvas px rather than screen px, so what a sample covers is a property of the
    /// painting and not of how far the artist happens to be zoomed in.
    pub radius: u32,
}

/// Largest eyedropper radius, in canvas px — a 65×65 average. Not a taste limit but
/// a bound on what one sample may cost: the sampled square is rendered and read
/// back, so an unbounded radius is an unbounded render.
const MAX_PICK_RADIUS: u32 = 32;

/// Below this summed opacity a sampled patch holds no paint worth calling a colour,
/// and dividing by it would amplify float noise into an arbitrary hue.
const PICK_MIN_OPACITY: f32 = 1e-3;

/// A sampled patch summed channel-wise, with the texel count the two means below
/// divide by.
///
/// Both of them start here and differ only in what they divide by — the summed
/// opacity or the count — which is the whole of what separates the two pick modes.
/// Stating the shared half once is what keeps that the *only* difference.
fn sum_texels(texels: &[f32]) -> ([f32; 4], usize) {
    let patch = texels.as_chunks::<4>().0;
    let mut sum = [0.0f32; 4];
    for t in patch {
        for (s, v) in sum.iter_mut().zip(t) {
            *s += v;
        }
    }
    (sum, patch.len())
}

/// The mean **unpremultiplied** channels of a sampled patch, or `None` where there
/// is no paint in it.
///
/// The composite is premultiplied by opacity (§6.1), so summing and
/// dividing by the summed opacity *is* the opacity-weighted mean: a texel carrying
/// more paint counts for more and a bare one counts for nothing. That is what lets a
/// radius wider than the stroke still report the stroke's colour rather than a wash
/// of it fading into empty canvas.
fn mean_channels(texels: &[f32]) -> Option<[f32; 4]> {
    let (sum, _) = sum_texels(texels);
    if !sum[3].is_finite() || sum[3] <= PICK_MIN_OPACITY {
        return None;
    }
    Some([sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3], 1.0])
}

/// The mean channels of a sampled patch **composited over the substrate** `bg` — the
/// colour the canvas shows there rather than the paint's own (`PickSource::CompositeOverSubstrate`).
///
/// The same `over` the media pass runs, in the same latent channels and the same
/// order (`over_substrate` in `media_common.wesl`): `bg·(1−a) + c`, with the
/// composite's premultiplied colour standing in for `c·a`. Sharing the operation
/// rather than restating it is what keeps this mode from becoming a second opinion
/// about a colour the screen has already decided.
///
/// It is a **plain** mean where [`mean_channels`] is an opacity-weighted one, and
/// that difference *is* the mode: with the ground behind it every texel is opaque, so
/// a patch half-covered by a stroke reads as the mixture of paint and canvas an eye
/// sees there instead of reporting the stroke alone. `over` is linear in each texel,
/// so compositing every texel and then averaging is exactly the arithmetic below.
fn mean_over_substrate(texels: &[f32], bg: [f32; 4]) -> Option<[f32; 4]> {
    let (sum, count) = sum_texels(texels);
    if count == 0 || !sum.iter().all(|v| v.is_finite()) {
        return None;
    }
    let n = count as f32;
    // Clamped because it is about to be read as "how much of the patch the ground
    // shows through": the composite cannot exceed full coverage, and a float that
    // lands a hair over 1 would subtract substrate rather than none of it.
    let bare = 1.0 - (sum[3] / n).clamp(0.0, 1.0);
    Some([
        sum[0] / n + bg[0] * bare,
        sum[1] / n + bg[1] * bare,
        sum[2] / n + bg[2] * bare,
        1.0,
    ])
}
impl Engine {
    /// Sample the canvas colour at `at` — the eyedropper (§18.0.2).
    ///
    /// A **request**, not a command: it has to answer, so it stays a direct method
    /// beside `save_bytes` rather than joining [`InputCommand`](crate::InputCommand),
    /// whose whole property is that nothing comes back (§4).
    ///
    /// What it samples is the **raw layer channels**, not the composited, *lit*
    /// result the screen shows, and that is the decision the feature turns on. The
    /// media pass lights the paint, tonemaps it and encodes sRGB, so picking its
    /// output would load the brush with a colour the palette never mixed — and in a
    /// Mixbox document (§6.7) with a display colour rather than the pigment
    /// mixture, which would make picking the mix back up impossible. That is the
    /// entire reason pigment mixing is worth having.
    ///
    /// `None` where the sampled patch holds no paint: the substrate is the ground,
    /// not something a brush picks up, so bare canvas answers "nothing here" rather
    /// than quietly loading the brush with the paper colour. The one source that
    /// answers anyway is [`PickSource::CompositeOverSubstrate`], where the ground is
    /// what was asked for.
    ///
    /// Renders immediately and returns a future for the **readback**, the only
    /// asynchronous part — the same shape as [`Engine::export`], and for the same
    /// reason: an `async fn` would hold `&mut self` across an await during which the
    /// frontend re-renders and tries to read the engine.
    pub fn pick_color(
        &mut self,
        at: crate::geom::Vec2,
        options: PickOptions,
    ) -> impl std::future::Future<Output = Option<[f32; 3]>> + use<> {
        let radius = options.radius.min(MAX_PICK_RADIUS);
        let size = Extent2::new(2 * radius + 1, 2 * radius + 1);
        // Centred on the canvas *pixel* the point falls in rather than on the point
        // itself: pass A samples tile textures bilinearly, so a fractional offset
        // would blend neighbouring texels and a "point sample" would answer with a
        // colour that is at neither of them. Snapping puts every fragment on a texel
        // centre, so radius 0 reports exactly the texel under the cursor.
        let view = ViewTransform {
            center: crate::geom::Vec2::new(at.x.floor() + 0.5, at.y.floor() + 0.5),
            zoom: 1.0,
            // Axis-aligned with the *canvas*: the sampled square is a patch of the
            // painting, so which way the easel is turned cannot change which texels
            // fall in it.
            rotation: 0.0,
            flip_h: false,
            viewport: size,
        };
        // The *presented* document, so a sample agrees with what is on screen —
        // including a collaborator's stroke that has not committed yet, and the
        // substrate colour mid-drag on the picker that sets it (§15.5).
        let (groups, ground) = {
            let doc = self.presented();
            let only = match options.source {
                PickSource::Composite | PickSource::CompositeOverSubstrate => None,
                PickSource::Layer(id) => Some(id),
            };
            // Read here rather than in the future, because it is document state and
            // the future deliberately does not borrow the engine. That the other two
            // sources have no ground is why this is a third *source* rather than a
            // flag on the other two: asking one layer for its own colour and asking
            // what the canvas shows are different questions, and only the second one
            // has a substrate in it.
            let ground = matches!(options.source, PickSource::CompositeOverSubstrate)
                .then(|| self.color_space.rgb_to_channels(doc.background));
            (
                self.composite_groups(doc, only, visible_tiles(view)),
                ground,
            )
        };

        let (color_format, aux_format) = self.compositor_pipeline.channel_formats();
        // `read_rgba16f` decodes four halves per texel. Both colour spaces store the
        // colour channels that way (§6.1); a new one that did not would
        // have to say so here rather than silently mis-decoding.
        debug_assert_eq!(color_format, wgpu::TextureFormat::Rgba16Float);
        let color = self.offscreen_target(
            "stark pick color",
            color_format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        // Written by pass A and never read: the composite pipeline has two targets,
        // and the height it accumulates says how *much* paint is there, not what
        // colour it is.
        let aux = self.offscreen_target(
            "stark pick aux",
            aux_format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        self.compositor.composite_channels(
            &self.compositor_pipeline,
            &color.create_view(&wgpu::TextureViewDescriptor::default()),
            &aux.create_view(&wgpu::TextureViewDescriptor::default()),
            view,
            &groups,
        );

        // Captured, not read through `self`: the future deliberately does not borrow
        // the engine (see `export`). The colour space is an `Arc`, so carrying the
        // channels→RGB conversion into it costs a refcount bump.
        let gpu = self.gpu.clone();
        let color_space = self.color_space.clone();
        async move {
            let texels = crate::gpu::readback::read_rgba16f(&gpu, &color, size).await;
            let mean = match ground {
                Some(bg) => mean_over_substrate(&texels, bg),
                None => mean_channels(&texels),
            };
            mean.map(|c| color_space.channels_to_rgb(c))
        }
    }
}
