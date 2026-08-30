//! Asking the canvas what is at a point: the eyedropper's color (§18.0.2) and
//! the pick-and-translate drag's layer (§16.11).
//!
//! Both are **requests** rather than commands — they have to answer — and both
//! go through the *same* draw list rendering does
//! ([`Engine::composite_groups`]), so what they report is the paint the screen is
//! drawing rather than a second opinion about it. What the eyedropper
//! deliberately does not sample is the media pass's output: that lights the
//! paint, tonemaps it and encodes sRGB, so picking it up would load the brush
//! with a color the palette never mixed.
//!
//! The two share their machinery down to the readback, and differ only in which
//! axis the batch runs along: [`Engine::pick_colors`] renders one source at many
//! points, [`Engine::pick_layer`] many sources at one point. Each ends in a
//! single buffer map, because the map is the latency.

use super::Engine;
use crate::gpu::channels::Targets;
use crate::view::ViewTransform;
use stark_model::document::LayerId;
use stark_model::geom::{Extent2, TileRect};

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
    /// be a color no layer holds — a glaze over the substrate is a mixture of the two.
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
    /// The interior of the group that carries `layer` (§14.2): the layer, its
    /// siblings, and the carrier's own content, composited exactly as the group
    /// composites internally — members keep their modes and clips, since a
    /// sibling's blend against the base is part of what the group *is*. What is
    /// dropped is the carrier's own outward params, by [`Layer`](Self::Layer)'s
    /// argument: they say how the group meets what is beneath it, and beneath it
    /// is exactly what this source excludes. For a layer in the root stack the
    /// interior is the whole document and this answers as [`Composite`](Self::Composite)
    /// does — the root "group" has no substrate in it, so bare canvas still
    /// answers nothing.
    ///
    /// `below` cuts the interior above the layer: only the layer itself, members
    /// beneath it, and the carrier's base content answer, as though the members
    /// above had been switched off.
    Group { layer: LayerId, below: bool },
    /// Every visible layer at or beneath `layer` — the document as the screen
    /// would show it with everything above the layer switched off, **over the
    /// substrate** (§15.5). Ancestors of a nested layer are included as far as
    /// they reach beneath it: their bases, the members below the chain, and
    /// their own outward params applied to that partial whole, which is exactly
    /// what hiding the layers above would leave on screen.
    ///
    /// The substrate is in the question, as it is for
    /// [`CompositeOverSubstrate`](Self::CompositeOverSubstrate): "what would I
    /// see here without the layers over this one" is asked of a canvas, not of
    /// paint floating in the void — so bare canvas answers with the substrate.
    Below(LayerId),
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
/// that difference *is* the mode: with the substrate behind it every texel is opaque, so
/// a patch half-covered by a stroke reads as the mixture of paint and canvas an eye
/// sees there instead of reporting the stroke alone. `over` is linear in each texel,
/// so compositing every texel and then averaging is exactly the arithmetic below.
fn mean_over_substrate(texels: &[f32], bg: [f32; 4]) -> Option<[f32; 4]> {
    let (sum, count) = sum_texels(texels);
    if count == 0 || !sum.iter().all(|v| v.is_finite()) {
        return None;
    }
    let n = count as f32;
    // Clamped because it is about to be read as "how much of the patch the substrate
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
fn patch_view(at: stark_model::geom::Vec2, size: Extent2) -> ViewTransform {
    ViewTransform {
        center: stark_model::geom::Vec2::new(at.x.floor() + 0.5, at.y.floor() + 0.5),
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
/// [`ViewTransform::visible_tiles`]'s own contract carried across the batch rather than re-derived:
/// an optimization that cannot see one of its inputs has to do nothing about all of
/// them. Folding the *bounds* instead would have let a non-finite point vanish into
/// a `min`/`max` and quietly shrink the answer.
fn patch_cull(points: &[stark_model::geom::Vec2], size: Extent2) -> Option<TileRect> {
    let mut union: Option<TileRect> = None;
    for &at in points {
        let rect = patch_view(at, size).visible_tiles()?;
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
    /// beside `save_bytes` rather than joining [`InputCommand`](crate::command::InputCommand),
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
    /// `None` where the sampled patch holds no paint: the substrate is the backdrop,
    /// not something a brush picks up, so bare canvas answers "nothing here" rather
    /// than quietly loading the brush with the paper color. The one source that
    /// answers anyway is [`PickSource::CompositeOverSubstrate`], where the substrate is
    /// what was asked for.
    ///
    /// Renders immediately and returns a future for the **readback**, the only
    /// asynchronous part — the same shape as [`Engine::export`], and for the same
    /// reason: an `async fn` would hold `&mut self` across an await during which the
    /// frontend re-renders and tries to read the engine.
    pub fn pick_color(
        &mut self,
        at: stark_model::geom::Vec2,
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
    /// resampled evenly by arc length ([`stark_model::gradient::resample`]), each
    /// sample is picked exactly as [`Engine::pick_color`] picks (same sources,
    /// same patch mean, same raw-channels-not-lit rule — in a Mixbox document
    /// the ramp is of pigment mixtures), and [`stark_model::gradient::fit`] reduces
    /// the run to the fewest stops that reproduce it within a perceptual
    /// tolerance.
    ///
    /// Samples over bare canvas answer nothing (the `pick_color` rule), and the
    /// trace simply proceeds without them: a stroke gap crossed mid-trace does
    /// not inject the paper into the ramp. `None` when fewer than two samples
    /// found paint — there is no gradient in an empty trace.
    pub fn pick_gradient(
        &mut self,
        path: &[stark_model::geom::Vec2],
        options: PickOptions,
    ) -> impl std::future::Future<Output = Option<stark_model::gradient::Gradient>> + use<> {
        let samples = stark_model::gradient::resample(path);
        let points: Vec<stark_model::geom::Vec2> = samples.iter().map(|&(_, p)| p).collect();
        let fut = self.pick_colors(&points, options);
        async move {
            let colors = fut.await;
            let run: Vec<(f32, [f32; 3])> = samples
                .iter()
                .zip(colors)
                .filter_map(|(&(t, _), c)| c.map(|c| (t, c)))
                .collect();
            stark_model::gradient::fit(&run)
        }
    }

    /// The shared sampling machinery under [`Engine::pick_color`] and
    /// [`Engine::pick_gradient`]: one flush, then one rendered patch per point,
    /// then **one** buffer map for the lot. One implementation on purpose — the
    /// gradient's promise is that every sample is exactly an eyedropper pick,
    /// and two copies of this logic is how that promise would quietly break.
    pub(crate) fn pick_colors(
        &mut self,
        points: &[stark_model::geom::Vec2],
        options: PickOptions,
    ) -> impl std::future::Future<Output = Vec<Option<[f32; 3]>>> + use<> {
        // The pick samples `presented`, whose fold is rebuilt lazily — flush, so a
        // sample mid-stroke agrees with what the next paint would show.
        self.flush_live();
        let radius = options.radius.min(MAX_PICK_RADIUS);
        let size = Extent2::new(2 * radius + 1, 2 * radius + 1);
        // Read here rather than in the future, because it is document state and
        // the future deliberately does not borrow the engine. That the other
        // sources have no substrate is why the substrate rides the *source* rather
        // than a flag beside it: asking paint for its own color and asking what
        // the canvas shows are different questions, and only the second kind has
        // a substrate in it — `CompositeOverSubstrate` and `Below` are both that
        // kind.
        let substrate = matches!(
            options.source,
            PickSource::CompositeOverSubstrate | PickSource::Below(_)
        )
        .then(|| {
            let l = self
                .shared
                .color_space
                .rgb_to_latent(self.presented().substrate_color);
            // Opaque, like the substrate the screen composites over (`render.rs`).
            ([l.lat[0], l.lat[1], l.lat[2], 1.0], l.res)
        });

        // Both color spaces store the color channels as four halves (§6.1). A new
        // one that did not is caught by `read_many_rgba16f` itself, which is the side
        // that does the decoding and so the side that owns the claim.
        let formats = self.compositor_pipeline.channel_formats();

        // **One draw list for the whole trace**, culled to the union of every patch
        // rather than to each in turn (§6.3).
        //
        // A gradient capture samples up to `gradient::MAX_SAMPLES` points, and the
        // list is the same list every time: `composite_groups` walks the entire layer
        // tree, and the only thing the point changes is the cull — so a list per
        // sample is a tree walk to build a hundred-odd copies of one answer. The union
        // is a
        // *wider* cull than any single patch's, which is sound in the only direction
        // that matters — a cull may name tiles a pass then draws nothing for, never
        // omit one it needed (see `ViewTransform::visible_tiles`) — and the trace is bounded, so
        // the extra tiles are the ones between the samples, which the trace crosses
        // anyway.
        //
        // The *presented* document, so a sample agrees with what is on screen —
        // including a collaborator's stroke that has not committed yet, and the
        // substrate color mid-drag on the picker that sets it (§15.5).
        let groups = {
            let doc = self.presented();
            self.pick_groups(doc, options.source, patch_cull(points, size))
        };

        // **Every point's view first, then every point's draws, then one submit.**
        // Each patch is its own view of the same document, and the view uniform is a
        // queue write — so with one uniform the only thing that could order two
        // patches was a submit between them, and a gradient trace of
        // `gradient::MAX_SAMPLES` points was that many round trips. The views are
        // slots now (`composite::ViewBindings`), so what orders them is the offset a
        // patch binds and the whole trace is one encoder.
        let views: Vec<crate::view::ViewTransform> =
            points.iter().map(|&at| patch_view(at, size)).collect();
        self.compositor
            .write_views(&self.compositor_pipeline, &views);
        // The draws, once for the trace. Every patch is the same document under the
        // same cull — `pick_groups` took the union of them above — so this is one
        // plan, and it has to be one *upload*: what the streams write into is one
        // buffer apiece on the compositor (`Compositor::prepare_pick`). Kept across
        // the loop and past the submit, which is the other half of what it is for.
        //
        // `None` only for an empty trace, where the loop below draws nothing either.
        let prepared = views.first().map(|&first| {
            self.compositor
                .prepare_pick(&self.compositor_pipeline, first, &groups)
        });

        let mut encoder =
            self.shared
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("stark pick encoder"),
                });
        let mut colors = Vec::with_capacity(points.len());
        let mut resids = Vec::with_capacity(points.len());
        // The `Option` is unwrapped once, around the loop, rather than per slot: "a
        // non-empty trace prepared above" is a fact about this call, so it is a shape
        // here instead of an `expect` inside.
        if let Some(prepared) = &prepared {
            for slot in 0..views.len() {
                let (color, resid) = self.composite_patch(
                    &mut encoder,
                    prepared,
                    slot,
                    size,
                    formats,
                    ("stark pick color", "stark pick aux", "stark pick resid"),
                );
                colors.push(color);
                resids.push(resid);
            }
        }
        self.shared.gpu.queue.submit([encoder.finish()]);
        // After the submit, never before: the blend scratch inside destroys its
        // textures on drop, and a recorded encoder is not in-flight work.
        drop(prepared);

        // Captured, not read through `self`: the future deliberately does not borrow
        // the engine (see `export`). The color space is an `Arc`, so carrying the
        // channels→RGB conversion into it costs a refcount bump.
        let gpu = self.shared.gpu.clone();
        let color_space = self.shared.color_space.clone();
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
                    let mean = match substrate {
                        Some((bg, _)) => mean_over_substrate(texels, bg),
                        None => mean_channels(texels),
                    };
                    // The residual takes the **same** two means, unchanged. It can,
                    // because the residual target's alpha is a duplicate of the
                    // color's: `sum[3]` is the same coverage sum either way, so the
                    // opacity weighting and the over-substrate blend are already the
                    // right ones for it (§6.7).
                    let mean_resid = resid_sets.as_ref().and_then(|sets| match substrate {
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

    /// Composite one patch of a prepared plan into fresh attachments, returning the
    /// two a readback can name.
    ///
    /// **Shared by both pick paths**, which is what the module header means by their
    /// sharing machinery down to the readback and differing only in which axis the
    /// batch runs along. Written out at each, the copies drift over what the
    /// attachments are for: one gives `aux` `COPY_SRC` for a texture nothing reads
    /// back, the other gives `resid` `COPY_SRC` in the path that drops it.
    ///
    /// The usages here are the strict reading. `color` and `resid` are read back (the
    /// eyedropper wants both; a pigment space's residual is just more colour
    /// channels), so both get `COPY_SRC`. `aux` never is — pass A writes it and the
    /// height it accumulates says how *much* paint is there, not what colour it is —
    /// so it gets none, in both paths now.
    ///
    /// `TEXTURE_BINDING` on all three because some filters sample the accumulator.
    fn composite_patch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &crate::gpu::composite::PreparedPick<'_>,
        slot: usize,
        size: Extent2,
        formats: crate::gpu::channels::ChannelFormats,
        labels: (&str, &str, &str),
    ) -> (wgpu::Texture, Option<wgpu::Texture>) {
        let shared = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let readable = shared | wgpu::TextureUsages::COPY_SRC;
        let color = self.offscreen_target(labels.0, formats.color, size, readable);
        let aux = self.offscreen_target(labels.1, formats.aux, size, shared);
        let resid = formats
            .resid
            .map(|f| self.offscreen_target(labels.2, f, size, readable));
        // Named rather than inlined into the call: a `Targets` borrows its three
        // views, so they have to outlive it.
        let default = wgpu::TextureViewDescriptor::default();
        let (color_view, aux_view) = (color.create_view(&default), aux.create_view(&default));
        let resid_view = resid.as_ref().map(|t| t.create_view(&default));
        self.compositor.composite_channels(
            &self.compositor_pipeline,
            encoder,
            Targets {
                color: &color_view,
                aux: &aux_view,
                resid: resid_view.as_ref(),
            },
            prepared,
            slot,
        );
        (color, resid)
    }
}

/// How much of a layer's own coverage a press has to land on for that layer to
/// count as the one under the pointer (§16.11).
///
/// Not [`PICK_MIN_OPACITY`]'s near-zero floor, which asks a different question:
/// that one asks whether there is enough paint here to divide by without
/// amplifying float noise into a hue, and this one asks whether *this is the
/// layer the artist is pointing at*. The outermost texel of a soft brush's
/// feather is paint by the first question and is not what the hand meant by the
/// second, so the threshold sits where a texel starts to read as covered rather
/// than where it stops being empty.
///
/// A quarter rather than a half because the paint it has to answer for is not
/// only opaque: a glaze laid at low flow covers honestly and never reaches half
/// (§6.1's slab law), and a hit test that could not grab a glaze would be one the
/// thinnest passages of a painting fall out of.
const LAYER_HIT_COVERAGE: f32 = 0.25;

/// Every paint layer the canvas actually shows, **topmost first** — the order a
/// hit test walks, so the first answer it finds is the last one drawn.
///
/// Reverse composite order, which inside a group means its members before the
/// carrier's own content: members composite *over* the base they are carried on
/// (§14.2), so coming back down the stack they are met first.
///
/// A hidden layer — or one turned all the way down — takes its whole subtree with
/// it. That is the part the single-layer branch of `composite_groups` cannot do
/// on its own: it asks only about the layer named, which is right for "sample
/// this layer" (§18.0.2) and wrong here, because the members of a hidden group
/// are not on the screen to be pointed at.
fn shown_paint_layers(stack: &rpds::Vector<crate::document::Layer>, out: &mut Vec<LayerId>) {
    for layer in stack.iter().rev() {
        if !layer.is_shown() {
            continue;
        }
        shown_paint_layers(&layer.carries, out);
        // A tile map is exactly what makes a layer answerable here: a matte and a
        // filter have none (§15.2, §21), and neither is paint a press can pick up
        // and carry — a frame moves by its own handles, and a filter has no
        // content of its own to move.
        if layer.tiles().is_some() {
            out.push(layer.id);
        }
    }
}

impl Engine {
    /// Which layer's paint the canvas shows at `at` — the hit test the
    /// pick-and-translate drag opens with (§16.11).
    ///
    /// A **request** for [`Engine::pick_color`]'s reason (it has to answer), and
    /// built out of the same parts: one small render per candidate and **one**
    /// buffer map for the lot. What differs is which axis the batch runs along —
    /// the eyedropper renders one source at many points, this renders many
    /// sources at one point — so the cost is a tree walk plus a 1×1 pass for each
    /// layer that has a tile under the pointer at all, and a single readback.
    ///
    /// The answer is the **topmost** layer whose own coverage there reaches
    /// [`LAYER_HIT_COVERAGE`], or `None` where the press landed on nothing the
    /// canvas is showing.
    ///
    /// Coverage of the layer *alone*, exactly as [`PickSource::Layer`] takes a
    /// color: the composite params say how a layer meets what is beneath it, and
    /// "which paint is under my finger" is a question about the paint. Two
    /// consequences are worth stating, because they are the ones a user could
    /// notice. A layer turned down to 10% still answers where its paint is solid
    /// — it is *there*, faintly, and the alternative is paint you can see and
    /// cannot grab. And a **clipped** layer answers over its whole extent rather
    /// than only where the paint beneath it lets it through (§14.4); the carrying
    /// group is what bounds a clip, and a hit test that re-derived that would be a
    /// second opinion about compositing rather than a question about paint.
    ///
    /// Renders immediately and returns a future for the readback — the shape
    /// every request that reads the GPU back wears here, and for
    /// [`Engine::export`]'s reason: an `async fn` would hold `&mut self` across
    /// an await during which the frontend re-renders and tries to read the
    /// engine.
    ///
    /// [`Engine::export`]: crate::Engine::export
    pub fn pick_layer(
        &mut self,
        at: stark_model::geom::Vec2,
    ) -> impl std::future::Future<Output = Option<LayerId>> + use<> {
        // The hit test reads `presented`, whose fold is rebuilt lazily — flush,
        // so a press made mid-stroke is answered against what the screen shows.
        self.flush_live();
        let size = Extent2::new(1, 1);
        let view = patch_view(at, size);
        // For a single texel, a cull that cannot be measured is not an
        // optimization declining to help ([`patch_cull`]'s stance): it is the
        // point itself being unaddressable — non-finite, or past the far edge of
        // the tile grid — and there is no paint out there to point at.
        let lists: Vec<(LayerId, Vec<crate::gpu::CompositeGroup>)> = match view.visible_tiles() {
            None => Vec::new(),
            Some(cull) => {
                let doc = self.presented();
                let mut candidates = Vec::new();
                shown_paint_layers(doc.root(), &mut candidates);
                candidates
                    .into_iter()
                    .filter_map(|id| {
                        // An empty list means the layer draws nothing in this tile
                        // — no tile there, hidden, or switched off — so it is out
                        // before it costs a pass.
                        let groups = self.composite_groups(doc, Some(id), Some(cull));
                        (!groups.is_empty()).then_some((id, groups))
                    })
                    .collect()
            }
        };

        // Only the color target's alpha is read, at the stride
        // `read_many_rgba16f` pins.
        let formats = self.compositor_pipeline.channel_formats();
        // **One view for every candidate**, since a hit test asks the same patch of
        // every layer in turn — so one slot, written once.
        //
        // A submit per candidate all the same, unlike the eyedropper's trace. Each
        // candidate has its *own* draw list, so each needs its own upload, and the
        // streams an upload writes are one buffer apiece on the compositor: two
        // uploads before one submit would leave both candidates drawing the second's
        // tile origins (`Compositor::prepare_pick`). What the eyedropper batches is
        // many patches of *one* list, which is the case that can be.
        self.compositor
            .write_views(&self.compositor_pipeline, &[view]);
        let mut hits: Vec<(LayerId, wgpu::Texture)> = Vec::with_capacity(lists.len());
        for (id, groups) in &lists {
            let prepared = self
                .compositor
                .prepare_pick(&self.compositor_pipeline, view, groups);
            let mut encoder =
                self.shared
                    .gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("stark hit encoder"),
                    });
            // The residual comes back and is dropped: what this path reads is coverage,
            // and in a pigment space the residual is more colour channels rather than
            // any part of that.
            let (color, _resid) = self.composite_patch(
                &mut encoder,
                &prepared,
                0,
                size,
                formats,
                ("stark hit color", "stark hit aux", "stark hit resid"),
            );
            self.shared.gpu.queue.submit([encoder.finish()]);
            // After the submit: the scratch inside destroys its textures on drop.
            drop(prepared);
            hits.push((*id, color));
        }

        let gpu = self.shared.gpu.clone();
        async move {
            if hits.is_empty() {
                return None;
            }
            let refs: Vec<&wgpu::Texture> = hits.iter().map(|(_, t)| t).collect();
            // A readback that fails is the GPU failing underneath it (§5), and
            // "nothing under the pointer" is the honest answer for a device that
            // can no longer be read — the eyedropper's stance, for its reason:
            // `ObservableState::gpu_failure` is what *reports* the failure, and a
            // hit test going quiet is a symptom rather than the cause.
            let Ok(sets) = crate::gpu::readback::read_many_rgba16f(&gpu, &refs, size).await else {
                return None;
            };
            hits.iter().zip(sets).find_map(|((id, _), texels)| {
                let coverage = *texels.first_chunk::<4>()?.get(3)?;
                (coverage.is_finite() && coverage >= LAYER_HIT_COVERAGE).then_some(*id)
            })
        }
    }
}
