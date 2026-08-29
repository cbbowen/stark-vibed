//! The brush engine: **swept-segment** stroke rasterization with copy-on-write
//! tiles (§6.2, §5.2, §6.6, §6.7).
//!
//! Rather than stamping discrete dabs, each short segment of the flattened curve
//! is drawn as one oriented quad whose coverage is the brush *swept* along it —
//! the path integral of the extent. Because alpha-"over" is additive in
//! optical depth `τ = −ln(1−α)`, the swept depth of a segment is a difference of
//! the brush's precomputed prefix-τ texture (`prefix(u) − prefix(u−d)`), and the
//! premultiplied-over blend across overlapping segment quads sums those depths
//! *exactly* — reconstructing the continuous stroke with no banding and no
//! double-counting at the joints.
//!
//! That is the plain **add** fast path: extent → cleared scratch tile →
//! integrate over the base into a fresh CoW tile. A brush that also moves paint
//! already on the canvas (`lift` / `deposit` / `charge` / `bleed`, §6.2) instead runs
//! the sequential swept-exchange loop in `dynamics.wesl`; one that erases
//! (`BrushEffect::Erase`, §6.12) runs the same sweep into a per-stroke
//! accumulator and turns it on the base's visible opacity (`erase.rs`).
//! `dynamics_setup` decides which path a record takes.
//!
//! The renderer is parameterized by a [`ColorSpace`] (formats, blends, channel
//! mapping, shader). It holds only immutable GPU objects plus `Arc`-backed
//! handles, so it is cheap to `Clone` and can live in the `Action::Context` (§5).

use std::sync::Arc;

use crate::assets::AssetStore;
use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::Zeroes;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle, TilePool};
use crate::noise::NOISE_TILE_PX;
use crate::unpoisoned;
use stark_model::Srgb;
use stark_model::document::StrokeRecord;

mod accum;
mod budget;
mod dynamics;
mod erase;
mod incremental;
mod region;
mod segments;
mod swept;
mod tips;

use crate::gpu::scratch::ScratchPool;
use budget::MAX_REGION_DIM;
use dynamics::{DynamicsKit, StrokePath, build_dynamics_kit, dynamics_setup};
use erase::{EraseKit, build_erase_kit};
use swept::{SweptKit, build_swept_kit};
use tips::TipCache;

// The module's substrate, re-exported so callers name `gpu::stroke::X` rather than the
// file X happens to live in — the split below is about where a maintainer reads, not
// about what the engine depends on.
pub use budget::{max_stretch, max_tip_reach};
pub(crate) use incremental::{StrokeCarry, StrokeSpans, ToolState};
// Not part of the module's public surface: the engine calls it, nothing outside the
// crate does, and keeping it crate-visible is what lets its doc comment point at the
// `segments` internals the rule is actually about.
pub(crate) use incremental::safe_frozen;

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,

    /// The two render paths' GPU objects, each built by the module that dispatches
    /// them. Symmetric on purpose: one path behind a type and the other's pipelines
    /// loose among the caches would make a struct documented as holding "only
    /// immutable GPU objects" name one path and not the other.
    swept: SweptKit,
    /// The sequential stamp loop (§6.2), used when the brush manipulates existing
    /// paint (`lift` / `deposit` / `charge` / `bleed` — the four axes `dynamics_setup`
    /// gates on).
    dynamics: DynamicsKit,
    /// The erase pass (§6.12), used when the brush erases
    /// (`BrushEffect::Erase`) — the swept extent turned on the base's visible
    /// opacity. Built over the swept kit's own layouts, so the two share bind
    /// groups.
    erase: EraseKit,

    /// What a brush resolves to, and the lazily-baked caches behind it (§6.6) — the
    /// prefix-τ volume both paths integrate against, the coverage mask the reservoir
    /// weights by, and the color-dynamics field. **One of the three mutable things
    /// here** (the others are [`scratch`](Self::scratch) and
    /// [`complained`](Self::complained)), which is why it is a type of its own rather
    /// than five fields: the sentence above about immutable objects is then true of
    /// everything else without qualification.
    tips: TipCache,

    /// The stamp loop's pooled working textures (§6.2): checked out per fold, handed
    /// back after each submit. Shared across clones like the caches in `tips`, so
    /// the live fold and the commit that replaces it draw from one free list.
    scratch: ScratchPool,

    /// The base bound where a stroke reaches a tile the layer does not have yet
    /// (§6.8's pattern). The integrate reads it through clamped loads, so bare canvas
    /// needs no tile of its own — where a real pooled tile would be acquired and
    /// cleared on every pointer move, whether or not the stroke touched anything
    /// unpainted.
    zeroes: Zeroes,

    /// Selection masks (§6.8): the per-tile mask bound into the integrate
    /// pass, and the region gather the stamp loop reads. Color-space independent, so
    /// it is handed in rather than rebuilt with the rest of this renderer.
    selection: SelectionRenderer,

    /// The seed of the last stroke this renderer complained about — could not draw
    /// ([`StrokePath::TipTooLarge`]), or could draw only with segments shortened to
    /// fit the region (`StrokePlan::shortened`) — so it says so once per stroke
    /// rather than once per pointer move.
    ///
    /// The gate is re-asked on every render, and deliberately so — it is a pure
    /// function of the brush, which is what lets a live tail and its commit agree for
    /// free ([`dynamics_setup`]). But its *answer* is a property of the record, so
    /// repeating it per frame turns one undrawable brush into an unbounded stream of
    /// `error!` and buries whatever else the log was carrying. One seed is enough to
    /// collapse that: a gesture renders one stroke at a time, so remembering the last
    /// is remembering the one being drawn.
    complained: Arc<std::sync::Mutex<Option<u64>>>,
}

/// Everything a stroke is drawn *against*, as opposed to the stroke itself.
///
/// [`StrokeRenderer`] holds only immutable GPU objects — pipelines, layouts, the
/// prefix-τ cache — so the mutable scene is handed in per call. These four travel
/// together through every entry point ([`StrokeRenderer::render`],
/// [`render_range`](StrokeRenderer::render_range), and both paths underneath), so
/// they are one parameter rather than four repeated at each hop.
#[derive(Copy, Clone)]
pub struct StrokeScene<'a> {
    pub pool: &'a TilePool,
    pub assets: &'a AssetStore,
    /// The layer's committed tiles: what the stroke composites over.
    pub base: &'a TileMap,
    /// The selection in force, which gates the deposit (§6.8).
    pub selection: &'a Selection,
    /// The canvas substrate the document was on when this stroke was made (§6.4) —
    /// the substrate whose tooth gates how much of the brush's own paint lands
    /// (`BrushParams::tooth_give`).
    ///
    /// Handed in per call, like everything else here, rather than held on the
    /// renderer: it is *document* state, and a renderer that cached it would answer
    /// a replayed stroke with whatever the compositor happens to be showing. That is
    /// the shape the deleted `StrokeRenderer::set_substrate` had (§6.4).
    pub substrate: &'a crate::gpu::substrate::SubstrateMap,
}

impl StrokeRenderer {
    /// `tile_bgl` is pass A's layout over a tile's channels, handed in for the same
    /// reason the selection renderer and the zeroes are: the stamp loop composites
    /// document tiles into its working region through `composite.wesl` (§6.2), and
    /// the group it binds is the one the *tile* caches, which answers to exactly one
    /// layout ([`tile_bind_group_layout`](crate::gpu::composite::tile_bind_group_layout)).
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: Arc<dyn ColorSpace>,
        selection: SelectionRenderer,
        zeroes: Zeroes,
        tile_bgl: wgpu::BindGroupLayout,
        scratch: ScratchPool,
    ) -> Self {
        // Composition, not construction: each path's objects are built by the module
        // that uses them, and the brush textures both paths resolve live with their
        // caches. What is left here is the pair of them plus the scene-independent
        // things a renderer is handed.
        // One compile of `stamp.wesl`, lent to the two kits that draw the swept
        // extent through it (`swept::stamp_module`).
        let stamp = swept::stamp_module(&ctx.device, color_space.as_ref());
        let swept = build_swept_kit(&ctx.device, color_space.as_ref(), &stamp);
        let dynamics = build_dynamics_kit(ctx, color_space.as_ref(), tile_bgl);
        let erase = build_erase_kit(&ctx.device, color_space.as_ref(), &swept, &stamp);

        Self {
            ctx: ctx.clone(),
            color_space,
            swept,
            dynamics,
            erase,
            tips: TipCache::new(ctx),
            scratch,
            zeroes,
            selection,
            complained: Arc::default(),
        }
    }

    /// Render `rec` over `base`, gated by `selection`, returning a copy-on-write tile
    /// map.
    ///
    /// The selection is applied at the *end* of each path — the integrate pass's
    /// merge on the fast path, the deposit's write-back in the stamp loop — rather
    /// than by clipping the extent. That keeps one rule for both paths (a texel
    /// receives the mask's fraction of whatever the stroke did there) and is what
    /// makes a feathered selection fade a stroke out instead of scaling its optical
    /// depth, which for an opaque brush would barely fade at all (§6.8).
    pub fn render(&self, scene: StrokeScene<'_>, rec: &StrokeRecord) -> TileMap {
        self.render_range(scene, rec, StrokeSpans::whole(rec), None)
            .0
    }

    /// Render just `spans` of `rec` over `base`, resuming the brush from `tool` — the
    /// state the preceding range left it in — and returning what a range picking up
    /// where this one stops must resume from ([`StrokeCarry`]).
    ///
    /// This is what makes a live stroke cost its *tail* rather than its whole length
    /// (§6.2). On the swept path it is sound because the deposit is a
    /// definite integral over each segment and composes by summing optical depth, so
    /// cutting the path at a span boundary and compositing the two halves in order
    /// gives the same result as one pass — the same property that lets adaptive
    /// flattening choose segment lengths freely. On the stamp loop it is sound because
    /// [`ToolState`] carries the only thing the loop threads between segments that is
    /// not already on the canvas. Adjacent ranges share exactly one flattened point
    /// (`path::flatten_spans`), so their segments tile with no gap and no overlap.
    pub(crate) fn render_range(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&ToolState>,
    ) -> (TileMap, StrokeCarry) {
        // Every stroke render, live tail or commit, on either path — the row the two
        // path rows below are read against. Its *count* is the one number that says
        // how much of this is the live preview: a gesture renders its tail on every
        // frame it survives, and its commit then takes that render rather than
        // adding one (`document::PreparedStroke`) — so a replay, a redo or a peer's
        // action is what a commit's entry here means.
        crate::timing::span!("stroke.range");
        // Which path the stroke takes — and how finely it flattens — is decided from
        // the record, never from the piece in hand. A live tail and the commit that
        // eventually replaces it have to make the same choice, or releasing the pointer
        // would visibly redraw the stroke. See `dynamics_setup`.
        let Some(_) = self.tips.resolve(scene.assets, &rec.brush) else {
            if self.complain_once(rec.seed) {
                tracing::warn!(
                    "brush stamp asset is unavailable; deferring stroke until it is loaded",
                );
            }
            // **Deferred, not finished.** `spans.dist()` is where the range *began*,
            // which is still where the stroke has got to — nothing was drawn, so the
            // arc clock did not move. The flag is what stops a caller freezing the
            // range on the strength of that: see `StrokeCarry::deferred`.
            return (scene.base.clone(), StrokeCarry::deferred(spans.dist()));
        };
        let plan = dynamics_setup(&rec.brush);
        // Both halves of "is this range resuming, and will it be resumed" — read once
        // here, where `spans` and the record are both in hand, rather than in each of
        // the three paths that has cross-piece state (`Resume`).
        let resume = incremental::Resume::of(rec, &spans, tool);
        match plan.path {
            StrokePath::Loop => {
                // The fit was bought with shorter segments (`budget::fit_len`) —
                // correct geometry, but a real cost, since the loop exchanges once
                // per segment. Said once per stroke, like the error below and for
                // its reason: the cap is a pure function of the brush.
                if let Some(s) = &plan.shortened
                    && self.complain_once(rec.seed)
                {
                    tracing::warn!(
                        radius = rec.brush.size,
                        wanted_px = s.wanted,
                        got_px = s.got,
                        "brush tip nearly fills a dynamics region: segments \
                         shortened to fit, so this stroke costs ~{:.1}x the stamps \
                         its brush budgeted",
                        s.wanted / s.got,
                    );
                }
                self.render_dynamic(scene, rec, spans, resume, plan.tol)
            }
            StrokePath::Swept => self.render_swept(scene, rec, spans, plan.tol, resume),
            StrokePath::Erase => self.render_erase(scene, rec, spans, resume, plan.tol),
            StrokePath::TipTooLarge => {
                // An error, not a warning: what lands is not a rougher version of the
                // stroke that was asked for but a different brush — the swept deposit
                // only ever adds paint, so `lift`, `deposit` and `charge` all silently
                // do nothing. It is the one degradation left: stroke *length* is
                // handled by drawing the stroke in pieces (§6.2) and an oversized
                // *segment* by shortening it (`budget::fit_len`), so only a tip whose
                // own extent overflows the region lands here.
                //
                // **No brush this app can build does.** The frontier is published as
                // [`max_tip_reach`](budget::max_tip_reach) and the frontend clamps
                // every brush to it (`stark-ui`'s `state::hold_the_tip_drawable`), so
                // reaching this arm means a record came from somewhere that did not —
                // a peer, or a file written by another build — and is not being
                // honoured.
                //
                // Said once per stroke, not once per render: the gate is a pure
                // function of the brush and so answers the same way every pointer move
                // ([`complained`](Self::complained)).
                if self.complain_once(rec.seed) {
                    tracing::error!(
                        radius = rec.brush.size,
                        max_region_dim = MAX_REGION_DIM,
                        "brush tip too large for one dynamics region: falling back to \
                         the swept deposit, so this stroke's lift/deposit/charge do \
                         nothing",
                    );
                }
                self.render_swept(scene, rec, spans, plan.tol, resume)
            }
        }
    }

    /// Whether this is the first render of stroke `seed` to find something worth
    /// complaining about — see [`complained`](Self::complained).
    ///
    /// Deliberately not a set: a gesture draws one stroke at a time, so the last seed
    /// is the one being drawn, and a replay that alternates between two undrawable
    /// records is welcome to say so twice. What this rules out is the unbounded case,
    /// which is one stroke shouting on every pointer move.
    fn complain_once(&self, seed: u64) -> bool {
        let mut last = unpoisoned(self.complained.lock());
        if *last == Some(seed) {
            return false;
        }
        *last = Some(seed);
        true
    }

    /// Acquire a persistent tile: the color space's `color` + `aux` formats, paired.
    ///
    /// The pool hands out textures, not tiles (see [`TexHandle`](crate::gpu::tile::TexHandle)).
    /// Pairing them here is what keeps the two formats coming from the color space
    /// actually in use rather than from a constant in the pool: a hardcoded `R16Float`
    /// aux matches every space currently built, and would panic on the first one that
    /// chose otherwise (§6.7).
    fn acquire_tile(&self, pool: &TilePool, source: AllocSource) -> TilePairHandle {
        TilePairHandle::new(
            pool.acquire_tex(self.color_space.color_format(), source),
            pool.acquire_tex(self.color_space.aux_format(), source),
            self.color_space
                .resid_format()
                .map(|f| pool.acquire_tex(f, source)),
        )
    }

    /// Resolve the constants both render paths read for one stroke — see
    /// [`StrokeConstants`] for why they are resolved here rather than at each path.
    fn stroke_constants(
        &self,
        rec: &StrokeRecord,
        substrate: &crate::gpu::substrate::SubstrateMap,
        selection: &crate::document::Selection,
    ) -> StrokeConstants {
        // The pigment is the paint effect's own (§6.2); an erase stroke has none,
        // and zero is safe because every lane derived from it feeds passes the
        // erase path never binds. This is the boundary where the stored sRGB
        // becomes the display color the conversion is defined on.
        let rgb = Srgb::new(rec.brush.paint().map_or([0.0; 3], |p| p.color));
        let l = self.color_space.rgb_to_latent(rgb);
        let (nfreq, namp) = noise_uniform(rec);
        StrokeConstants {
            channels: l.lat,
            resid: [l.res[0], l.res[1], l.res[2], 0.0],
            // Clamped where the record becomes numbers, the color's rule: the
            // ceiling is quoted in [0, 1] (`BrushEffect::opacity`) and a wire
            // value past it is nonsense, not a stronger setting. The mask's
            // opacity is the ceiling's other factor (§6.8), folded in here so it
            // reaches every place the dial does — the integrate, the erase, the
            // loop's mint and the charge — through the one number.
            opacity: rec.brush.effect.opacity().clamp(0.0, 1.0) * selection.opacity(),
            substrate_uv_scale: substrate.relief * substrate.uv_scale,
            tooth_softness: rec.brush.tooth.softness,
            nfreq,
            namp,
            noise_seed: noise_seed(rec.seed),
            // Clamped where the record becomes numbers, like the color above: the
            // gate is `1 + 2ε·centered` with `centered ∈ (−½, ½)`, so any ε ≤ 1
            // keeps it positive and a wire value beyond that is nonsense, not a
            // stronger setting.
            jitter_eps: rec.brush.jitter.clamp(0.0, 1.0),
            jitter_seed: jitter_seed(rec.seed),
        }
    }
}

/// The per-stroke quantities **both** render paths read, resolved once from the record
/// and the scene (§6.2).
///
/// Nothing here is path-specific. The swept sweep puts them in a `TileXform` and the
/// stamp loop in a `Stamp` slot, but they are the same numbers — and they have to be,
/// because which path a brush takes is decided by `dynamics_setup` from axes that have
/// nothing to do with any of this: nudge `deposit` off zero and the same color, the
/// same flow and the same substrate must still lay the same paint.
///
/// That is not hypothetical. `tests/dynamics.rs`'s
/// `a_glaze_lands_the_same_whether_or_not_the_stamp_loop_runs` exists because the two
/// paths once disagreed by 157 levels, and both halves of the disagreement were
/// quantities of exactly this kind, derived twice and drifted apart. Resolving them in
/// one place is what makes the agreement structural rather than a matter of two files
/// happening to contain the same line.
struct StrokeConstants {
    /// The brush's own color in the working space. No per-unit opacity beside it:
    /// the paint a brush lays is opaque per unit, faded only by the `drain` —
    /// which both paths apply per fragment from the fragment's own arc length,
    /// never per segment.
    channels: [f32; 3],
    /// The same color's **residual** (§6.7) in `.xyz`; `.w` unused. Zero in a space
    /// with no residual, and zero because that space's channels above are already the
    /// whole color — both paths write this lane unconditionally, since the uniform it
    /// lands in is one Rust struct across both shader variants.
    resid: [f32; 4],
    /// The **stroke's ceiling**: the effect's opacity (`BrushEffect::opacity`)
    /// times the selection mask's (`Selection::opacity`, §6.8) — the ceiling on
    /// what a saturated stroke does, whichever effect and whichever path. Each
    /// path applies it where its own law can hold it exactly — the swept
    /// integrate and the erase pass scale the whole accumulated extent per
    /// stroke, the stamp loop scales what it mints (§6.2) — but the number is one
    /// number, resolved here so the paths cannot disagree about what the dial
    /// said, and so the mask's dimming cannot be honoured by one and not the
    /// other. The mask's *per-texel* coverage is the same ceiling's third factor,
    /// and rides in the mask each pass binds.
    opacity: f32,
    /// Canvas px → substrate-tile uv (§6.4). Zero on a substrate with no relief — a `Flat`
    /// canvas, or one whose bytes have not arrived — which sends the tooth to exactly
    /// 1 and leaves the deposit bit-for-bit what it was before the tooth existed.
    substrate_uv_scale: f32,
    /// The width of the tooth's contact transition, in the rise's own units
    /// (`BrushParams::tooth_softness`, §6.4).
    ///
    /// Here and not on the segment, unlike the tooth's *depth* beside it: the depth is
    /// a modulation target, because bearing down presses the tip; the width is what
    /// the tip is made of, and a charcoal stick does not go harder under the hand.
    tooth_softness: f32,
    /// The color-dynamics lookup (§6.2): per-axis frequency (across the stroke, along
    /// it) + 1/NOISE_TILE_PX, and per-channel amplitude. Inactive jitter zeroes
    /// frequency *and* amplitude, so with the zero tile bound the shader's early-out
    /// keeps the deposit bit-identical.
    nfreq: [f32; 4],
    namp: [f32; 4],
    /// The seed the stroke's own noise tile is baked from ([`noise_seed`]) — the
    /// field `TipCache::noise_view` binds. Not a uniform lane, but resolved here
    /// beside the lookup it feeds for the same reason as the rest: the two paths
    /// must bind one field.
    noise_seed: u32,
    /// The deposit jitter (§6.2): the gate's half-range
    /// (`BrushParams::jitter`, clamped) and the stroke's own seed for it
    /// ([`jitter_seed`]). Resolved here like the color-dynamics lookup
    /// above and for the same reason: the loop's `add` axis is the swept path's
    /// deposit, and the two must read one gate.
    jitter_eps: f32,
    jitter_seed: u32,
}

/// The stroke's color-dynamics uniform pair — (per-axis frequency (across the
/// stroke, along it) + 1/NOISE_TILE_PX, per-channel amplitude) — shared by the
/// sweep's `TileXform` and the dynamics loop's `Stamp` slots so both paths jitter
/// identically. Inactive jitter zeroes frequency *and* amplitude, so with the
/// zero tile bound the shader's early-out keeps the deposit bit-identical.
fn noise_uniform(rec: &StrokeRecord) -> ([f32; 4], [f32; 4]) {
    let cd = rec.brush.color_dynamics();
    let (freq, amp) = if cd.is_active() {
        (cd.frequency, cd.amplitude)
    } else {
        ([0.0; 2], [0.0; 3])
    };
    (
        [freq[0], freq[1], 1.0 / NOISE_TILE_PX, 0.0],
        [amp[0], amp[1], amp[2], 0.0],
    )
}

/// The `n`-th draw of the stroke's splitmix64 stream. Every per-stroke
/// randomness is its own draw off the record's `seed`, so each is deterministic
/// from the record (replay and live == committed hold, §6.2) and independent of
/// the others.
fn draw(seed: u64, n: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(n));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The seed the stroke's color-dynamics tile is baked from (`noise.rs`): the
/// stream's first draw, folded to the u32 the bake's hash takes. A field of its
/// own per stroke is what keeps strokes side by side from laying one pattern.
fn noise_seed(seed: u64) -> u32 {
    (draw(seed, 1) >> 32) as u32
}

/// The deposit jitter's per-stroke seed: the stream's third draw, folded the same
/// way. A draw of its own, so the gate's pattern is independent of the tile's,
/// and a fresh one per stroke, which is what keeps repeated glazes averaging out
/// instead of compounding one texture.
fn jitter_seed(seed: u64) -> u32 {
    (draw(seed, 3) >> 32) as u32
}
