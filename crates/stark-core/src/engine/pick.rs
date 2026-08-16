//! The eyedropper: sampling a color off the canvas (§18.0.2).
//!
//! A **request** rather than a command — it has to answer — and one that goes
//! through the *same* draw list rendering does ([`Engine::composite_groups`]), so
//! what it reports is the paint the screen is drawing rather than a second opinion
//! about it. What it deliberately does not sample is the media pass's output: that
//! lights the paint, tonemaps it and encodes sRGB, so picking it up would load the
//! brush with a color the palette never mixed.

use super::Engine;
use super::render::visible_tiles;
use crate::document::LayerId;
use crate::geom::{Extent2, TileRect, ViewTransform};
use crate::gpu::channels::Targets;

/// Which layers an eyedropper sample is taken from (§18.0.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PickSource {
    /// Every visible layer, composited — the color the canvas shows.
    #[default]
    Composite,
    /// Every visible layer *over the substrate* (§15.5): the same stack, with the
    /// canvas color standing in wherever the paint does not cover.
    ///
    /// The one source that answers on bare canvas, and the only one whose answer can
    /// be a color no layer holds — a glaze over the ground is a mixture of the two.
    /// That is what it is for: matching what the eye sees at a point rather than what
    /// is stored there, which is the question being asked when the paint is thin.
    CompositeOverSubstrate,
    /// One layer alone: the color that layer would have if it were the only one in
    /// the document. What "sample the current layer" has to mean, since a glaze on
    /// top of somebody else's underpainting is not the same paint as the two mixed.
    ///
    /// **Its composite params are dropped, opacity included** (§14.4.3). All three
    /// describe how the layer meets what is beneath it, which is precisely what this
    /// source is asked to ignore: a blend mode and a clip decide how much of the paint
    /// survives its surroundings, and the opacity slider decides how much of the layer
    /// the *document* shows. None of them says what the paint **is**, so turning a
    /// layer down reports the same color rather than a paler one — which is the
    /// property a painter needs, since the reason to sample a faded layer is usually
    /// to go on painting with what is already on it.
    ///
    /// **Except at zero**, where the layer contributes nothing at all and this answers
    /// `None` like bare canvas. That is not the same statement made fainter: every
    /// setting above zero is a layer that is there and turned down, and zero is a
    /// layer that is switched off.
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

/// Below this summed opacity a sampled patch holds no paint worth calling a color,
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
/// radius wider than the stroke still report the stroke's color rather than a wash
/// of it fading into empty canvas.
fn mean_channels(texels: &[f32]) -> Option<[f32; 4]> {
    let (sum, _) = sum_texels(texels);
    if !sum[3].is_finite() || sum[3] <= PICK_MIN_OPACITY {
        return None;
    }
    Some([sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3], 1.0])
}

/// The mean channels of a sampled patch **composited over the substrate** `bg` — the
/// color the canvas shows there rather than the paint's own (`PickSource::CompositeOverSubstrate`).
///
/// The same `over` the media pass runs, in the same latent channels and the same
/// order (`over_substrate` in `media_common.wesl`): `bg·(1−a) + c`, with the
/// composite's premultiplied color standing in for `c·a`. Sharing the operation
/// rather than restating it is what keeps this mode from becoming a second opinion
/// about a color the screen has already decided.
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

/// The view one sampled patch is rendered through: `size` canvas px, upright,
/// centred on the canvas **pixel** `at` falls in.
///
/// Snapped to the pixel rather than centred on the point itself: pass A samples
/// tile textures bilinearly, so a fractional offset would blend neighbouring texels
/// and a "point sample" would answer with a color that is at neither of them.
/// Snapping puts every fragment on a texel centre, so radius 0 reports exactly the
/// texel under the cursor.
///
/// Axis-aligned with the *canvas*, and unrotated: the sampled square is a patch of
/// the painting, so which way the easel is turned cannot change which texels fall
/// in it.
fn patch_view(at: crate::geom::Vec2, size: Extent2) -> ViewTransform {
    ViewTransform {
        center: crate::geom::Vec2::new(at.x.floor() + 0.5, at.y.floor() + 0.5),
        zoom: 1.0,
        rotation: 0.0,
        flip_h: false,
        viewport: size,
    }
}

/// The tiles every patch of a trace can reach between them — one cull for the whole
/// batch, so the draw list is built once instead of per point.
///
/// `None` — cull nothing — the moment any single patch is unmeasurable, which is
/// [`visible_tiles`]'s own contract carried across the batch rather than re-derived:
/// an optimization that cannot see one of its inputs has to do nothing about all of
/// them. Folding the *bounds* instead would have let a non-finite point vanish into
/// a `min`/`max` and quietly shrink the answer.
fn patch_cull(points: &[crate::geom::Vec2], size: Extent2) -> Option<TileRect> {
    let mut union: Option<TileRect> = None;
    for &at in points {
        let rect = visible_tiles(patch_view(at, size))?;
        union = Some(match union {
            Some(u) => u.union(rect),
            None => rect,
        });
    }
    union
}

impl Engine {
    /// Sample the canvas color at `at` — the eyedropper (§18.0.2).
    ///
    /// A **request**, not a command: it has to answer, so it stays a direct method
    /// beside `save_bytes` rather than joining [`InputCommand`](crate::InputCommand),
    /// whose whole property is that nothing comes back (§4).
    ///
    /// What it samples is the **raw layer channels**, not the composited, *lit*
    /// result the screen shows, and that is the decision the feature turns on. The
    /// media pass lights the paint, tonemaps it and encodes sRGB, so picking its
    /// output would load the brush with a color the palette never mixed — and in a
    /// Mixbox document (§6.7) with a display color rather than the pigment
    /// mixture, which would make picking the mix back up impossible. That is the
    /// entire reason pigment mixing is worth having.
    ///
    /// `None` where the sampled patch holds no paint: the substrate is the ground,
    /// not something a brush picks up, so bare canvas answers "nothing here" rather
    /// than quietly loading the brush with the paper color. The one source that
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
        let fut = self.pick_colors(std::slice::from_ref(&at), options);
        async move { fut.await.into_iter().next().flatten() }
    }

    /// Sample a gradient off the canvas: colors along a traced path, fitted to
    /// stops — the eyedropper generalized from a point to a line (§22.2).
    ///
    /// The trace is how a gradient is *made* here: the artist draws a line
    /// through paint they have already mixed, and the machinery of control
    /// points is this method's problem. The path — canvas-space, as traced — is
    /// resampled evenly by arc length ([`crate::gradient::resample`]), each
    /// sample is picked exactly as [`Engine::pick_color`] picks (same sources,
    /// same patch mean, same raw-channels-not-lit rule — in a Mixbox document
    /// the ramp is of pigment mixtures), and [`crate::gradient::fit`] reduces
    /// the run to the fewest stops that reproduce it within a perceptual
    /// tolerance.
    ///
    /// Samples over bare canvas answer nothing (the `pick_color` rule), and the
    /// trace simply proceeds without them: a stroke gap crossed mid-trace does
    /// not inject the paper into the ramp. `None` when fewer than two samples
    /// found paint — there is no gradient in an empty trace.
    pub fn pick_gradient(
        &mut self,
        path: &[crate::geom::Vec2],
        options: PickOptions,
    ) -> impl std::future::Future<Output = Option<crate::gradient::Gradient>> + use<> {
        let samples = crate::gradient::resample(path);
        let points: Vec<crate::geom::Vec2> = samples.iter().map(|&(_, p)| p).collect();
        let fut = self.pick_colors(&points, options);
        async move {
            let colors = fut.await;
            let run: Vec<(f32, [f32; 3])> = samples
                .iter()
                .zip(colors)
                .filter_map(|(&(t, _), c)| c.map(|c| (t, c)))
                .collect();
            crate::gradient::fit(&run)
        }
    }

    /// The shared sampling machinery under [`Engine::pick_color`] and
    /// [`Engine::pick_gradient`]: one flush, then one rendered patch per point,
    /// then **one** buffer map for the lot. One implementation on purpose — the
    /// gradient's promise is that every sample is exactly an eyedropper pick,
    /// and two copies of this logic is how that promise would quietly break.
    pub(crate) fn pick_colors(
        &mut self,
        points: &[crate::geom::Vec2],
        options: PickOptions,
    ) -> impl std::future::Future<Output = Vec<Option<[f32; 3]>>> + use<> {
        // The pick samples `presented`, whose fold is rebuilt lazily — flush, so a
        // sample mid-stroke agrees with what the next paint would show.
        self.flush_live();
        let radius = options.radius.min(MAX_PICK_RADIUS);
        let size = Extent2::new(2 * radius + 1, 2 * radius + 1);
        let only = match options.source {
            PickSource::Composite | PickSource::CompositeOverSubstrate => None,
            PickSource::Layer(id) => Some(id),
        };
        // Read here rather than in the future, because it is document state and
        // the future deliberately does not borrow the engine. That the other two
        // sources have no ground is why this is a third *source* rather than a
        // flag on the other two: asking one layer for its own color and asking
        // what the canvas shows are different questions, and only the second one
        // has a substrate in it.
        let ground = matches!(options.source, PickSource::CompositeOverSubstrate).then(|| {
            let bg = self.presented().background;
            (
                self.color_space.rgb_to_channels(bg),
                self.color_space.rgb_to_resid(bg),
            )
        });

        let formats = self.compositor_pipeline.channel_formats();
        // The readback decodes four halves per texel. Both color spaces store the
        // color channels that way (§6.1); a new one that did not would
        // have to say so here rather than silently mis-decoding.
        debug_assert_eq!(formats.color, wgpu::TextureFormat::Rgba16Float);

        // **One draw list for the whole trace**, culled to the union of every patch
        // rather than to each in turn (§6.3).
        //
        // A gradient capture samples up to `gradient::MAX_SAMPLES` points, and the
        // list is the same list every time: `composite_groups` walks the entire layer
        // tree, and the only thing the point changed was the cull. So this was a tree
        // walk per sample to build a hundred-odd copies of one answer. The union is a
        // *wider* cull than any single patch's, which is sound in the only direction
        // that matters — a cull may name tiles a pass then draws nothing for, never
        // omit one it needed (see [`visible_tiles`]) — and the trace is bounded, so
        // the extra tiles are the ones between the samples, which the trace crosses
        // anyway.
        //
        // The *presented* document, so a sample agrees with what is on screen —
        // including a collaborator's stroke that has not committed yet, and the
        // substrate color mid-drag on the picker that sets it (§15.5).
        let groups = {
            let doc = self.presented();
            self.composite_groups(doc, only, patch_cull(points, size))
        };

        let mut colors = Vec::with_capacity(points.len());
        let mut resids = Vec::with_capacity(points.len());
        for &at in points {
            let view = patch_view(at, size);
            let color = self.offscreen_target(
                "stark pick color",
                formats.color,
                size,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            // Written by pass A and never read: the height it accumulates says how *much*
            // paint is there, not what color it is.
            let aux = self.offscreen_target(
                "stark pick aux",
                formats.aux,
                size,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            // The residual, and unlike the aux it **is** read back: in a pigment space it
            // is half the color, so an eyedropper that sampled only the concentrations
            // would report the polynomial's nearest reachable color — black as `#383838`
            // — which is precisely the defect this channel exists to fix (§6.7).
            let resid = formats.resid.map(|f| {
                self.offscreen_target(
                    "stark pick resid",
                    f,
                    size,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                )
            });
            // Named rather than inlined into the call: a `Targets` borrows its three
            // views, so they have to outlive it.
            let default = wgpu::TextureViewDescriptor::default();
            let (color_view, aux_view) = (color.create_view(&default), aux.create_view(&default));
            let resid_view = resid.as_ref().map(|t| t.create_view(&default));
            self.compositor.composite_channels(
                &self.compositor_pipeline,
                Targets {
                    color: &color_view,
                    aux: &aux_view,
                    resid: resid_view.as_ref(),
                },
                view,
                &groups,
            );
            colors.push(color);
            resids.push(resid);
        }

        // Captured, not read through `self`: the future deliberately does not borrow
        // the engine (see `export`). The color space is an `Arc`, so carrying the
        // channels→RGB conversion into it costs a refcount bump.
        let gpu = self.gpu.clone();
        let color_space = self.color_space.clone();
        async move {
            // A readback that fails is the GPU failing underneath it (§5), and a
            // pick has somewhere to put that already: "nothing here" is what this
            // answers for bare canvas, and it is the honest answer for a device that
            // can no longer be read. One `None` per point, so the alignment callers
            // zip against holds whatever happened. What *reports* the failure is
            // `ObservableState::gpu_failure` — the eyedropper going quiet is a
            // symptom, and a symptom is the wrong place to learn the cause.
            let nothing = |n: usize| vec![None; n];
            let refs: Vec<&wgpu::Texture> = colors.iter().collect();
            let Ok(texel_sets) = crate::gpu::readback::read_many_rgba16f(&gpu, &refs, size).await
            else {
                return nothing(colors.len());
            };
            // Whether a residual exists is a color-space property, so it is
            // all-or-none across the batch and the two lists stay index-aligned.
            let resid_refs: Vec<&wgpu::Texture> = resids.iter().flatten().collect();
            let resid_sets = if resid_refs.is_empty() {
                None
            } else {
                match crate::gpu::readback::read_many_rgba16f(&gpu, &resid_refs, size).await {
                    Ok(sets) => Some(sets),
                    // Not `None`: that means "this space has no residual", and taking
                    // it here would report a pigment color as though its residual were
                    // zero — a wrong color rather than no color (§6.7).
                    Err(_) => return nothing(colors.len()),
                }
            };
            texel_sets
                .iter()
                .enumerate()
                .map(|(i, texels)| {
                    let mean = match ground {
                        Some((bg, _)) => mean_over_substrate(texels, bg),
                        None => mean_channels(texels),
                    };
                    // The residual takes the **same** two means, unchanged. It can,
                    // because the residual target's alpha is a duplicate of the
                    // color's: `sum[3]` is the same coverage sum either way, so the
                    // opacity weighting and the over-substrate blend are already the
                    // right ones for it (§6.7).
                    let mean_resid = resid_sets.as_ref().and_then(|sets| match ground {
                        Some((_, bg_resid)) => mean_over_substrate(
                            &sets[i],
                            [bg_resid[0], bg_resid[1], bg_resid[2], 1.0],
                        ),
                        None => mean_channels(&sets[i]),
                    });
                    mean.map(|c| {
                        let r = mean_resid.unwrap_or([0.0; 4]);
                        color_space.channels_to_rgb(c, [r[0], r[1], r[2]])
                    })
                })
                .collect()
        }
    }
}
