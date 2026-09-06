//! What a stroke is allowed to cost: the cadences the swept-exchange loop runs at,
//! the ceilings on one region-sized piece of it, and the flattening budget those two
//! together buy (§6.2).
//!
//! These are the numbers a person actually tunes, and they are only meaningful
//! against one another — so they live together, with the measurements and the dead
//! ends that fixed each one recorded on the constant itself. [`flatten_budget`] is
//! where they are spent: it is the single place a brush's settings become a segment
//! length, which is what makes a live tail and the commit that replaces it cut the
//! same path (§1.3).
//!
//! Nothing here touches the GPU. It is float arithmetic over a [`BrushParams`], which
//! is what lets the segment-budget tests pin it exactly (`segments::tests`).

use stark_model::document::{BrushDynamics, BrushParams, BrushShape};
use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX};

use super::dynamics::BLEED_TRAVEL_QUANTUM;

/// The optical depth one full pass of an opaque tip lays over a point — the τ
/// ceiling `assets::build_prefix_tau` clamps to.
///
/// Every exchange in the stamp loop is a rate *per unit optical depth*, because
/// that is the currency the swept integral is denominated in and the only one both
/// sides can agree on (§6.2). But τ ≈ 7 for a single pass, so read
/// literally a `lift` of 0.5 would strip 99% of the canvas in one pass. Dividing
/// the rates through by this makes an axis mean a fraction **per pass of the tip**
/// — hardness-independent, and what a 0..1 knob is expected to mean.
pub(super) const TAU_PER_PASS: f32 = 6.9;
/// Region edge (canvas px) the chunker aims to keep a piece inside. A stroke that
/// wants more is drawn in as many pieces as it takes
/// ([`chunk_segments`](super::region::chunk_segments)), so this bounds the loop's
/// transient GPU memory rather than deciding which strokes the loop can draw at all
/// (§6.2).
///
/// At 2048² that is ~67 MB for a piece: color and aux are both `Rgba16Float`, so
/// each is 2048² × 8 B = 32 MiB. And it really is *per piece* rather than per stroke,
/// because a piece's region is a `ScratchPool` lease its `SubmitScope` releases at the
/// flush that submits it (`gpu::scratch`), so the next piece takes the same memory.
///
/// **A target, not a ceiling** — [`MAX_REGION_DIM`] is the ceiling, and the two are
/// different numbers for a reason. Cutting is by *segment*, so a piece can be made to
/// fit this only while a single segment does; a brush whose tip alone wants more gets
/// a piece the size of its tip and pays for it. Raising this instead would let an
/// ordinary long stroke grow its pieces to the ceiling too, which is the same
/// megabytes bought for nobody: a 10 px tip crossing the canvas draws exactly as well
/// in 67 MB pieces as in 1 GB ones.
pub(super) const REGION_BUDGET_DIM: u32 = 2048;

/// **The largest region a piece may ever allocate** — the hard ceiling, where
/// [`REGION_BUDGET_DIM`] is the target the chunker aims at (§6.2).
///
/// A region is a texture, so this is the device's texture limit and nothing else is
/// available to be. It is reached only by the floor no cutting gets under — one
/// segment of one brush — which is why it also sets the ceiling on a brush's reach
/// ([`max_tip_reach`]).
///
/// **Paid only by the stroke that asks.** A tip needing more than
/// [`REGION_BUDGET_DIM`] gets one segment per piece, so its region is the size of its
/// own extent rather than of this constant: ~124 MB for the widest brush the editor
/// offers drawn along its facing axis, and ~500 MB for the same brush drawn at 45°,
/// where an axis-aligned box around a long diagonal tip is at its worst. Transient,
/// freed per piece, and only for a brush somebody deliberately built at the extreme.
pub(super) const MAX_REGION_DIM: u32 = crate::gpu::context::MAX_TEXTURE_DIM_2D;

// A region is a texture, so the ceiling may never be set past what the device was
// asked for. Written down even though the line above derives it from exactly that
// constant, because the failure it guards against is somebody replacing that
// derivation with a literal — which is the natural thing to do when raising one of
// the two, and which `create_texture` would then report as a validation error from
// inside the render path rather than as a number being wrong.
const _: () = assert!(
    MAX_REGION_DIM <= crate::gpu::context::MAX_TEXTURE_DIM_2D,
    "a region would not fit the texture limit the device was asked for",
);
// And the chunker's target has to sit inside the ceiling it aims below, or every
// ordinary piece would be measured against a bound no texture can honour.
const _: () = assert!(
    REGION_BUDGET_DIM <= MAX_REGION_DIM,
    "the region budget overruns the ceiling it is a target for",
);
/// Cap on the **segments** one piece dispatches. Reached only by a stroke fine enough
/// to fill a whole region with them, and it cuts a new piece rather than coarsening
/// anything.
///
/// It bounds the stamp uniform buffer, but not one slot per segment: `dynamics_plan`
/// also emits a bleed slot per crossing of the bleed cadence — up to
/// `dynamics::bleed::MAX_BLEED_FIRES_PER_SEGMENT` of them — and the pen-up settle, so a piece plans
/// at most `(1 + MAX_BLEED_FIRES_PER_SEGMENT) · MAX_STAMPS + 1` slots. At
/// `dynamics::plan::STAMP_STRIDE` apiece that is ~17.8 MB, which is why the
/// factor is worth stating and not worth chunking around: making the cut count planned
/// slots would couple `chunk_segments` to the bleed cadence to save a few megabytes it
/// does not need. Only a bleeding brush pays any of it, and only one whose segments
/// outrun its own cadence pays more than double.
///
/// The figure moved with the cadence: it was ~9.4 MB when a firing carried half a
/// radius, and halving [`BLEED_TRAVEL_QUANTUM`] doubled the fires a segment may
/// contribute and so this bound with it.
pub(super) const MAX_STAMPS: usize = 4096;
/// How far the tool may travel per exchange, as a fraction of the brush radius
/// (§6.2) — which, since the tool exchanges once per *segment*, is simply a cap on the
/// flattened segment length for a dynamics brush (see [`flatten_budget`]).
///
/// **Quoted at one transfer rate.** This is the travel for `lift = deposit = 0.95`;
/// [`exchange_travel`] scales it by how fast the brush being drawn actually trades,
/// because that — not the travel — is what the error is first order in. A gentler brush
/// is not being given a tolerance, it is being charged its own price.
///
/// A property of the exchange loop rather than of the tip, so nothing about a shape's
/// coverage mask should change it. What it bounds is the pair of mean-field
/// approximations either side of the transfer — `bake` gives the canvas a reservoir
/// frozen at the segment's entry, `exchange` gives the tool a canvas frozen at the same
/// instant — and halving it halves that error, cleanly, with no knee to sit on.
///
/// Why the error is a visible bug rather than a tolerance, why no reformulation of the
/// pair kernel avoids it, and why the gain from the sliding kernel was banked as
/// accuracy instead of spent here: **§6.2**.
/// `golden_drained_brush_length_independent` is what pins it.
const RESERVOIR_EXCHANGE_STEP: f32 = 0.125;

/// The shortest segment worth cutting, in canvas px — the floor under every length
/// cap here. It is also the least [`fit_len`] must afford for the stamp loop to run
/// at all: a fit below it means the tip *alone* overflows the region, which is
/// [`dynamics_setup`](super::dynamics::dynamics_setup)'s refusal rather than a
/// flattening problem.
pub(super) const MIN_SEGMENT_LEN: f32 = 0.5;

/// The margin [`fit_len`] holds over a segment's `max_len` when it prices the travel:
/// the **sagitta**, which `coverage_bounds` adds to *both* sides of the chord's box
/// before a region is measured, and which `max_len` says nothing about.
///
/// Nothing else. `max_len` caps the segment's arc length (`path::FlattenTolerance`), so
/// the chord under it is already paid for — it is shorter. What is left is the bow, and
/// it is bounded by the turn one segment may bend through
/// (`path::MAX_HALF_TURN_SIN` = 0.2, the sine of the half-turn). Writing `u` for that
/// sine, an edge of arc length `L` has chord `L·u/asin u` and sagitta
/// `L·(1 − √(1 − u²))/(2·asin u)`, so the box the region must hold is
///
/// ```text
/// (chord + 2·sagitta) / L  =  (u + 1 − √(1 − u²)) / asin u  ≤  1.09360  at u = 0.2
/// ```
///
/// rising monotonically to that worst case and tending to 1 as the edge straightens.
/// 1.095 rounds it up.
///
/// It read 1.1 while `max_len` capped the *chord*, and that number was covering two
/// terms rather than one: the arc over the chord (0.68% at the same backstop) as well
/// as the bow. The first was never the region's to pay — it was a flattener that
/// measured `dist` along chords while everything downstream walked arcs, and 1.1 was
/// the compensating constant. With the flattener fixed the term is gone; what stayed
/// is the half that was always real. Against the chord the same pair needed
/// `1 + 2·sagitta/chord = 1.10102`, so 1.1 was in fact a hair *short* of it — nobody
/// saw that, because reaching the backstop takes a segment bending ~23° when
/// `FLATTEN_TOLERANCE.angle` admits 5.7°.
const ARC_MARGIN: f32 = 1.095;

/// The longest `max_len` one segment of `b` can flatten at and still fit a
/// [`MAX_REGION_DIM`]-bounded region (§6.2) — negative when the tip's own extent
/// overflows the region before any travel is priced in.
///
/// `chunk_segments` can cut a stroke as fine as a single segment, but no finer: the
/// reservoir pickup reduces over the whole tip at once, so the region can never be
/// smaller than one extent. What the extent holds beyond the tip is the segment's
/// travel, and that is the one knob subdivision still has — so [`flatten_budget`]
/// spends it, shortening segments until one fits, and only a brush whose *minimal*
/// segment overflows is refused. Shorter segments are never wrong, only more
/// numerous: the exchange step they set is a first-order discretization that
/// tightens as they shrink ([`exchange_travel`]).
///
/// Bounded rather than measured, since it has to hold for segments that do not
/// exist yet: radius peaks at the brush's own (pressure only scales it down), and a
/// coverage box of a given extent spans at most one tile more than it covers,
/// whichever tile boundary it happens to straddle — so the budget is the largest
/// whole-tile block whose texture (apron ring included, `Covered::rect`) fits the
/// region.
///
/// **A pure function of the brush**, like everything in this file: a live tail and
/// the commit that replaces it cap the same brush to the same segment length.
pub(super) fn fit_len(b: &BrushParams) -> f32 {
    (region_extent_budget() - tip_extent(tip_reach(b)) - bleed_reach(b)) / ARC_MARGIN
}

/// The extent one region can hold, in canvas px: the largest whole-tile block whose
/// texture — the apron ring [`Covered::rect`](super::region::Covered::rect) adds
/// included — fits [`MAX_REGION_DIM`].
fn region_extent_budget() -> f32 {
    ((MAX_REGION_DIM - TILE_TEX) / TILE_SIZE * TILE_SIZE) as f32
}

/// How far from the centreline `b`'s tip can deposit, in canvas px.
///
/// The tip's radius bounds every shape's reach exactly — nothing a canonical mask
/// can paint lies outside the disc inscribed in its square
/// ([`Sweep::reach`](super::segments::Sweep)) — drawn out along its facing axis by
/// the brush's **own** elongation and not by any one segment's (§6.6). A modulation
/// can only scale either knob down ([`Modulation`](stark_model::document::Modulation)),
/// so the brush's value bounds every segment's.
fn tip_reach(b: &BrushParams) -> f32 {
    b.size.max(0.5) * BrushParams::elongation(b.stretch)
}

/// The canvas extent a tip reaching `reach` occupies, apron included — the part of
/// the region no shortening can give back.
fn tip_extent(reach: f32) -> f32 {
    2.0 * (reach + TILE_APRON as f32)
}

/// The extra floor a bleeding brush's segment carries: its firings' windows reach up
/// to one quantum back past the segment they fire after (`chunk_segments`).
///
/// Measured against the tip's own radius rather than its reach, which is what the
/// cadence is denominated in — so a stretched tip does not pay for it twice.
fn bleed_reach(b: &BrushParams) -> f32 {
    if axes(b).bleed > 0.0 {
        BLEED_TRAVEL_QUANTUM * b.size.max(0.5)
    } else {
        0.0
    }
}

/// The four dynamics axes as the budget prices them: the wet effect's, and all
/// zero on a plain paint brush or an eraser — which have none to price. Neither
/// of those paths needs a region at all, but the caps here are also *published*
/// limits an editor clamps any brush against ([`max_tip_reach`], [`max_stretch`]),
/// so they have to answer for every effect — and zero axes is the relaxed answer
/// the swept paths earn.
fn axes(b: &BrushParams) -> stark_model::document::BrushDynamics {
    b.wet().map_or(
        stark_model::document::BrushDynamics {
            add: 0.0,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
        },
        |w| w.dynamics,
    )
}

// What used to stand here: `manipulates_paint`, the rate predicate that sent a
// stroke down the stamp loop — four `> 0.0`s whose float complement a NaN axis
// satisfied from neither side, which is why it had to be one function with two
// callers. The [`BrushEffect::Wet`](stark_model::document::BrushEffect) split
// retired the question: the loop is the variant's, and a variant has no number
// for two spellings to disagree over (§6.2).

/// **The largest tip reach — `size × elongation`, canvas px — the stamp loop can
/// draw for a brush settled like `b`** (§6.2). A brush past it loses its dynamics
/// altogether ([`StrokePath::TipTooLarge`](super::dynamics::StrokePath)), because
/// the region is a single texture bounded by
/// [`MAX_TEXTURE_DIM_2D`](crate::gpu::context::MAX_TEXTURE_DIM_2D) and one tip has
/// to fit inside it whole.
///
/// **This is a limit an editor is expected to clamp against, not one a stroke is
/// expected to discover.** It is the exact frontier [`fit_len`] refuses at — the
/// same arithmetic inverted, and `the_published_reach_limit_is_the_gates_frontier`
/// is what keeps the two from drifting — so a brush built inside it is drawable and
/// one built outside it is not, with nothing in between for a caller to guess at.
///
/// Around 3936 canvas px for a non-bleeding brush, which is past `MAX_RADIUS × 7.8`:
/// wide enough that the editor gives up nothing until the very top of its size
/// slider. It was ~887 while a region was capped at 2048.
///
/// It reads `b` for everything *except* the two knobs it bounds: a bleeding brush
/// gets less, because its firings reach back past the segment they follow. Handed
/// the brush being edited, it answers for that brush.
pub fn max_tip_reach(b: &BrushParams) -> f32 {
    // `fit_len(b) ≥ MIN_SEGMENT_LEN`, solved for the reach.
    let spare = region_extent_budget() - bleed_reach(b) - MIN_SEGMENT_LEN * ARC_MARGIN;
    (spare / 2.0 - TILE_APRON as f32).max(0.0)
}

/// **The largest `stretch` knob a brush otherwise settled like `b` may carry and
/// still be drawn by the stamp loop** — [`max_tip_reach`] expressed in the units the
/// editor's slider actually moves in (§6.6).
///
/// [`BrushParams::MAX_STRETCH`] whenever the reach cap is out of the knob's own
/// reach, which is every brush up to about a 492 px radius: the knob tops out at an
/// elongation of [`MAX_ELONGATION`](BrushParams::MAX_ELONGATION), so a tip cannot
/// spend its way past the region until it is nearly as wide as the editor allows.
/// Only the top of the size slider trades at all, and only a little.
///
/// **The engine answers this rather than the editor deriving it**, because the
/// derivation does not survive being written twice. `elongation` is `1/(1 − knob)`
/// clamped at the top, so inverting it round-trips to within an ulp and not to the
/// bit — and an ulp on the wrong side is a brush the gate refuses and the slider
/// offered.
///
/// So the answer is settled by **asking the gate**, not by a second expression that
/// ought to agree with it: [`max_tip_reach`] only supplies the starting guess, and
/// the knob is stepped down until [`fit_len`] — the very call
/// [`dynamics_setup`](super::dynamics::dynamics_setup) makes — accepts the brush.
/// A predicate written as `size · elongation ≤ cap` instead is the same inequality
/// in a different association order, and it disagreed with the gate by one ulp at a
/// 500 px tip the moment the region grew; `the_offered_stretch_is_always_drawable`
/// is what caught it and what keeps it caught.
pub fn max_stretch(b: &BrushParams) -> f32 {
    let drawable = |knob: f32| {
        fit_len(&BrushParams {
            stretch: knob,
            ..*b
        }) >= MIN_SEGMENT_LEN
    };
    if drawable(BrushParams::MAX_STRETCH) {
        return BrushParams::MAX_STRETCH;
    }
    // The closed-form answer, then walked down by ulps onto the right side of the
    // gate. `fit_len` is monotone in the knob and this starts a hair above the
    // frontier, so it is a short walk; `knob > 0.0` bounds it regardless, an
    // unstretched tip being the smallest extent the brush has.
    let mut knob = (1.0 - b.size.max(0.5) / max_tip_reach(b)).clamp(0.0, BrushParams::MAX_STRETCH);
    while knob > 0.0 && !drawable(knob) {
        knob = f32::from_bits(knob.to_bits() - 1);
    }
    knob
}

/// The travel the brush's **own** budget puts on one segment, before the region has
/// its say: [`exchange_travel`] priced at the brush's size, floored at
/// [`MIN_SEGMENT_LEN`]. What [`flatten_budget`] spends, and the number the
/// shortening warning quotes against what [`fit_len`] left of it.
pub(super) fn dynamics_len(b: &BrushParams) -> f32 {
    // The wet effect's own overall rate; 1 — the neutral pass — where there is
    // no wet effect to have one, so the zero axes price alone as they always did.
    let flow = b.wet().map_or(1.0, |w| w.flow);
    (exchange_travel(axes(b), flow) * b.size).max(MIN_SEGMENT_LEN)
}

/// The bound every liquify step is held under (§6.13): the Lipschitz constant of
/// one segment's displacement, `strength · travel · slope / radius`, may not exceed
/// this. Under 1 the step `x ↦ x − v(x)` is a bijection of the plane — a
/// contraction's fixed point is unique, so every point has exactly one preimage —
/// which is what makes the stroke's map a homeomorphism and its composition one.
/// Half rather than a hair under 1, so the curvature of a bent sweep, the drain and
/// the quadrature's own error all fit inside the same margin.
const WARP_CONTRACTION: f32 = 0.5;

/// The plateau the liquify profile may not exceed (§6.13): the radius fraction the
/// paint keeps full pace out to, before the shoulder falls to the rim. A brush's
/// hardness names it, capped here so the shoulder is never narrower than a fifth
/// of the radius — the largest slope [`liquify_profile_slope`] then prices is
/// finite, and a tip that asked for a step profile gets the hardest warp that is
/// still a warp rather than a tear.
pub(super) const LIQUIFY_MAX_PLATEAU: f32 = 0.8;

/// The plateau a brush's liquify profile runs at (§6.13): its round tip's
/// hardness, capped by [`LIQUIFY_MAX_PLATEAU`]. A stamp has no hardness to read,
/// and no coverage the follow reads either — a textured tip would put the texture's
/// every step into the field, which is the tear the plateau cap exists to prevent —
/// so it warps as a round tip of middling hardness.
pub(super) fn liquify_plateau(shape: &BrushShape) -> f32 {
    let hardness = match shape {
        BrushShape::Round { hardness } => hardness.clamp(0.0, 1.0),
        BrushShape::Stamp(_) => 0.5,
    };
    hardness.min(LIQUIFY_MAX_PLATEAU)
}

/// The largest slope of the liquify follow profile per radius (§6.13): a
/// smoothstep shoulder over `[plateau, 1]` peaks at `1.5 / (1 − plateau)`
/// (`dynamics.wesl::warp_profile`). What the contraction budget divides by.
pub(super) fn liquify_profile_slope(plateau: f32) -> f32 {
    1.5 / (1.0 - plateau.min(LIQUIFY_MAX_PLATEAU))
}

/// The travel a liquify brush's own budget puts on one segment (§6.13) — the
/// longest step that is still a contraction — floored at [`MIN_SEGMENT_LEN`]. What
/// [`flatten_budget`] spends for the liquify path, and the number the shortening
/// warning quotes, exactly as [`dynamics_len`] is for the loop.
///
/// The step's displacement is `strength · exposure · travel` along the tangent, so
/// its gradient is bounded by `strength · travel` times the profile's slope per px
/// ([`liquify_profile_slope`] over the radius), plus what the tangent's own turn
/// contributes on a bent sweep ([`MAX_TIP_TURN`] per radius, the flattener's cap)
/// and what the drain's falloff does along the arc. [`WARP_CONTRACTION`] over that
/// sum is the travel.
///
/// Priced off the brush's own strength, not the modulated one, for
/// [`exchange_travel`]'s reason: a modulation only ever scales the axis down, so
/// the brush's value bounds every segment's. **Infinite at strength 0** — a drag
/// that moves nothing has no step to be a contraction, and the flattener's base
/// budget governs; a caller comparing against it asks `is_finite` first.
///
/// At the floor the guarantee lapses: a tip a couple of px wide at the hardest
/// plateau asks for a step under half a px, and gets the floor instead. That is a
/// tip whose whole shoulder is under a texel, where the field could not have
/// resolved a fold anyway.
pub(super) fn liquify_len(b: &BrushParams) -> f32 {
    let s = b.liquify().map_or(0.0, |l| l.strength.clamp(0.0, 1.0));
    if s <= 0.0 {
        return f32::INFINITY;
    }
    let radius = b.size.max(0.5);
    let slope_per_radius =
        liquify_profile_slope(liquify_plateau(&b.shape)) + MAX_TIP_TURN + b.drain_px() * radius;
    (WARP_CONTRACTION * radius / (s * slope_per_radius)).max(MIN_SEGMENT_LEN)
}

/// How far a liquify run may let its displacement accumulate before it re-bases
/// (§6.13), canvas px: a bound on `|d|` over any tile of the run, and so on how far
/// outside a piece's region its base composite has to reach. Past it a stroke
/// materializes the picture and starts the field afresh from a segment boundary —
/// one extra resample per this much of accumulated drag, where the first design
/// paid one per quarter-radius.
///
/// **Under the model's reach contract, with room for the margins**
/// ([`LiquifyEffect::REACH_PX`](stark_model::document::LiquifyEffect::REACH_PX)):
/// the base a piece composites is its region grown by the reach the piece can
/// accumulate, plus a texel for the bilinear tap, and the region itself is inside
/// the stroke's own padded rect. The assertion below is what keeps the footprint
/// from quietly under-claiming the reads (§12.6).
pub(super) const LIQUIFY_REACH_CAP: f32 = 512.0;

/// The margin the base composite carries past what the reach bounds, in texels:
/// one for the bilinear tap's far texel, and one of slack for the rounding of a
/// rect to whole texels.
pub(super) const LIQUIFY_BASE_SLACK: u32 = 2;

const _: () = assert!(
    LIQUIFY_REACH_CAP + LIQUIFY_BASE_SLACK as f32 + 4.0
        <= stark_model::document::LiquifyEffect::REACH_PX,
    "a liquify stroke could read past the reach its footprint declares (§12.6)",
);

/// Region edge the chunker aims to keep a **liquify** piece inside (§6.13):
/// [`REGION_BUDGET_DIM`] less the base composite's growth on both sides, so a
/// piece's base — the largest texture the path allocates — lands at the loop's
/// own budget.
pub(super) const LIQUIFY_REGION_BUDGET_DIM: u32 =
    REGION_BUDGET_DIM - 2 * (LIQUIFY_REACH_CAP as u32 + LIQUIFY_BASE_SLACK);

const _: () = assert!(
    LIQUIFY_REGION_BUDGET_DIM >= 2 * TILE_TEX,
    "a liquify piece must be able to hold a tile and its ring",
);

/// Cap on `radius · |curvature|`: how fat the tip may be relative to the turn it is
/// swept through before the segment goes back to being straight (§6.2).
///
/// Both shaders sweep a curved segment by **unrolling** the annulus about its centre
/// of curvature into the straight travel frame, which treats a canvas point as sliding
/// through the tip frame along a line of constant lateral offset. It does not: the
/// true track is an arc of radius `ρ`, so a point out at the extent's shoulder is
/// off that line by `≈ r²/2R`, i.e. **`radius · |curvature| / 2` as a fraction of the
/// tip radius**. That is the constant's real job. The annular sector the swept path
/// rasterizes also folds over itself once `radius ≥ |R|`, but that bound (1.0) is five
/// times looser and never the one that bites.
///
/// 0.1 holds the lateral error to 5% of the tip. It was 0.5 — 25% — which the plain
/// swept deposit absorbed (its segments overlap heavily and the error is smooth) but
/// the dynamics loop did not: there the same offset picks the wrong reservoir texel to
/// serve a canvas texel, and because the loop is sequential the error compounds down
/// the stroke into crescent seams at the reservoir cadence, worst where the tool is
/// dragging paint with nothing left to `add` over them.
pub(super) const MAX_TIP_TURN: f32 = 0.1;

/// The two sides of a binding fit cap, in canvas px — what the budget stood at per
/// segment before the region floor ([`dynamics_len`], [`liquify_len`]) and what the
/// floor left of it ([`fit_len`]).
pub(super) struct Shortened {
    pub(super) wanted: f32,
    pub(super) got: f32,
}

/// [`flatten_budget`]'s tolerance alone, for the tests that want only that.
#[cfg(test)]
pub(super) fn flatten_tolerance(b: &BrushParams) -> crate::path::FlattenTolerance {
    flatten_budget(b).0
}

/// The flatten budget with its reason: the tolerance, and what the region floor took
/// off it — `None` when the fit cost nothing, which is every brush whose full-length
/// segment already fits. The two are one answer because the second is a fact about
/// the `min` the first takes; read off the tolerance where it is taken rather than
/// re-derived from the two lengths, so the price quoted is the price paid.
pub(super) fn flatten_budget(
    b: &BrushParams,
) -> (crate::path::FlattenTolerance, Option<Shortened>) {
    let mut tol = crate::path::FLATTEN_TOLERANCE;
    // Use a more relaxed tolerance for larger brushes.
    tol.position = tol.position.max(0.01 * b.size);
    // The `attribute` bound is a step in the **pen's** own units, so it prices a brush
    // quantity correctly only while the two are proportional — 2% of pressure being 2%
    // of radius. A modulation puts a curve between them (§6.2), and a steep one turns
    // that 2% into as much as 18% of the parameter, which draws a ramp as a staircase
    // since a segment sweeps at one value of everything. So the budget is charged the
    // curve's own slope, bounded by construction (`document::MIN_BIAS`) precisely so
    // this bill is.
    //
    // Exactly 1 for the unmodulated brush and for every plain linear mapping,
    // including the default pressure → size, so those brushes are unaffected to the
    // bit.
    tol.attribute /= b.max_slope();
    // The tightest arc this tip may be swept along (§6.2). Both the
    // flattener and the segment generator get it from here, so an edge too tight to
    // sweep as an arc is priced as a chord as well as drawn as one.
    //
    // Against the tip's **stretched** reach, not its radius (§6.6). Every reason the
    // cap exists is about the extent rather than about the number that names it: the
    // swept sector stays a simple polygon only while the inner rim clears the centre of
    // curvature, and the reservoir's crescent seams are a misplacement measured across
    // the tip. An extent drawn out `s` times reaches `s` times as far, so it may bend
    // `s` times less. The brush's own elongation and not a segment's, for the reason
    // every bound here is stated against `b`: a modulation only ever scales the knob
    // down, so this one bounds them all.
    tol.max_arc_curvature = MAX_TIP_TURN / (b.size * BrushParams::elongation(b.stretch)).max(0.5);
    // **`drain` is deliberately not bought here.** A `0.02 / drain` px cap per segment
    // dominates everything else (at `drain = 0.02`, one segment per pixel), and it buys
    // nothing: the falloff is not a per-segment constant, since both paths evaluate it
    // from the fragment's own arc length. The amount laid is exactly independent of how
    // the path was cut, so there is nothing for a length cap to bound
    // (`generate_segments_in`).
    // The stamp loop exchanges once per segment, so the segment length *is* the step
    // at which the tool reloads and drains — and unlike the canvas side, which the
    // prefix-τ integral makes exact at any length, that step is a plain first-order
    // discretization of a coupled ODE. [`RESERVOIR_EXCHANGE_STEP`] is what keeps it
    // fine enough. The cap also bounds the snapshot scratch, which is sized by the
    // longest segment.
    // The two effects that run the region machinery each price their own step —
    // the exchange's mean-field freeze for wet, the contraction of one field step
    // for liquify (§6.13) — and both then pay the region floor below.
    let own_len = if b.wet().is_some() {
        Some(dynamics_len(b))
    } else if b.liquify().is_some() {
        Some(liquify_len(b))
    } else {
        None
    };
    let Some(own) = own_len else {
        return (tol, None);
    };
    tol.max_len = tol.max_len.min(own);
    // The region floor's price (§6.2): a tip so wide that a full-length
    // segment's extent would overflow [`MAX_REGION_DIM`] gets shorter segments
    // instead of losing its dynamics — the same trade `chunk_segments` makes
    // along the stroke, made along the segment. Never taken below
    // [`MIN_SEGMENT_LEN`]: a fit under the floor means the tip alone
    // overflows, which is `dynamics_setup`'s refusal, and capping here would
    // flatten dust for a loop that cannot run.
    let fit = fit_len(b);
    if fit < MIN_SEGMENT_LEN {
        return (tol, None);
    }
    // What the budget stood at before the floor — the brush's own length, nothing
    // above capping `max_len` — but read here rather than recomputed, so the
    // comparison is against whatever the min actually was. Infinite for a liquify
    // brush at strength 0 ([`liquify_len`]): no step error for a cap to bound, so
    // nothing was shortened and nothing warns.
    let wanted = tol.max_len;
    tol.max_len = wanted.min(fit);
    let shortened = (fit < wanted && wanted.is_finite()).then_some(Shortened { wanted, got: fit });
    (tol, shortened)
}

/// `λ = ln(1 − axis) / TAU_PER_PASS ≤ 0` — the transfer rate an axis becomes in the
/// shader's terms (§6.2), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped
/// clean). Dividing by [`TAU_PER_PASS`] is what makes an axis read as a fraction
/// **per pass of the tip** rather than per unit optical depth. Zero is "no
/// transfer".
///
/// The one definition, on purpose: the plan fills every slot's λ lanes from it,
/// and [`exchange_travel`] prices the flattening budget off the same clamp
/// ([`ln_keep`]). The flattener charging exactly the rates the shader will run is
/// what the exchange-step bound rests on, so it cannot rest on two closures — one
/// here, one in the plan — agreeing by comment.
pub(super) fn lambda(axis: f32) -> f32 {
    ln_keep(axis) / TAU_PER_PASS
}

/// `ln(1 − axis) ≤ 0`, clamped away from −∞ — the shared core of [`lambda`] and of
/// the transfer magnitude [`exchange_travel`] prices.
fn ln_keep(axis: f32) -> f32 {
    (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0)
}

/// How far the tool may travel per segment, in radii — [`RESERVOIR_EXCHANGE_STEP`]
/// scaled by how fast *this* brush actually trades.
///
/// The error the step bounds is a first-order splitting error, and what it is first
/// order in is not the travel but the **transfer the segment completes**: the pair
/// relaxes at `k_lift + k_deposit` per unit optical depth, so a segment's progress
/// through it is `(k_lift + k_deposit) · τ · lr`. Holding *that* fixed rather than `lr`
/// is what makes one constant mean the same thing to every brush.
///
/// The rate falls out in closed form. Each axis enters the shader as
/// `λ = ln(1 − axis) / TAU_PER_PASS` ([`lambda`]), so `(k_lift + k_deposit) · τ` is
/// just `−ln((1 − lift)(1 − deposit))` — the `τ` cancels, and there is no calibration
/// hiding in it.
///
/// Two things the pricing gets right that a cruder one would not:
///
/// * **`charge` is not a rate.** It sets the load the tool *starts* with, and a brush
///   that charges but neither lifts nor deposits has `k = 0`: `exchange_at` takes its
///   no-trading branch and the only thing reaching the canvas is `add`, which is linear
///   in exposure and therefore exact at any segment length. Such a brush must not pay
///   the full cap for a transfer that never happens.
/// * **It is continuous in the rates, not a boolean.** Priced on "has a non-zero axis"
///   alone, every brush is charged as the most extreme one, and a tip that lifts a
///   tenth of a pass costs the same per pixel as a full smear.
///
/// The budget is calibrated so that `lift = deposit = 0.95` — the repro's brush, and
/// about as hard as the transfer gets — comes out at exactly
/// [`RESERVOIR_EXCHANGE_STEP`], leaving every golden that uses it untouched. A gentler
/// brush earns its relaxation and nothing else changes.
///
/// Priced off the brush's own rates, not the modulated ones. A modulation only ever
/// scales an axis down (`document::Modulation`), which lowers the transfer a segment
/// completes and so the error the step bounds — the brush is charged its worst case
/// and every segment of every stroke it draws comes in under it. `flow` is the
/// brush's own overall rate (`WetEffect::flow`) for the same reason: the plan
/// scales every λ by the segment's modulated flow, which is never above it, so
/// charging the brush's puts every segment under the price.
fn exchange_travel(d: BrushDynamics, flow: f32) -> f32 {
    // [`ln_keep`] — the very clamp [`lambda`] hands the shader, so the flattener
    // prices the rates it will actually run (an axis at 1.0 is `−∞` otherwise).
    let rate_of = |axis: f32| -ln_keep(axis);
    // `bleed` is deliberately *not* in this sum: it fires on its own travel cadence
    // with the window's exposure ([`BLEED_TRAVEL_QUANTUM`]), so segment length does
    // not set its step and shortening segments buys it nothing.
    //
    // The flow multiplies the whole sum because it multiplies both λs (§6.2).
    // Past 1 it can only walk the step down to the reference floor below — the
    // same "only ever a relaxation" clamp that already catches an axis at 1.
    let rate = flow.max(0.0) * (rate_of(d.lift) + rate_of(d.deposit));
    if rate <= 0.0 {
        return MAX_EXCHANGE_TRAVEL;
    }
    // **Only ever a relaxation.** Rates above the reference are left at the reference
    // step rather than priced below it. Partly because the scaling has only been
    // measured across the band where a brush is usable — `lift = 1.0` is clamped to
    // `λ = −20` in the shader anyway (`dynamics.rs`), so past a point the axis stops
    // meaning what the rule reads it as — and partly because this is a *cost* change:
    // clamping here is what makes it incapable of charging any brush more than it
    // already pays, so no setting can regress on either axis.
    (RESERVOIR_EXCHANGE_STEP * EXCHANGE_REFERENCE_RATE / rate)
        .clamp(RESERVOIR_EXCHANGE_STEP, MAX_EXCHANGE_TRAVEL)
}

/// The transfer rate [`RESERVOIR_EXCHANGE_STEP`] is quoted at: `lift = deposit = 0.95`,
/// i.e. `−ln(0.05 · 0.05)`.
const EXCHANGE_REFERENCE_RATE: f32 = 5.991_465;

/// Ceiling on the travel per segment however slowly the brush trades, in radii.
///
/// Not an accuracy bound — a structural one. A segment carries **one** tip orientation
/// and one curvature (§6.6), the snapshot scratch is sized by the longest of them, and
/// the sweep's own arc approximation is only good while the segment is short next to
/// the tip. None of those care how fast paint changes hands.
const MAX_EXCHANGE_TRAVEL: f32 = 1.0;

/// Ceiling on the extent cell, in texels. [`extent_cell`]'s own law reaches 10
/// at the 500 px radius cap, so this is headroom against a future cap rather than a
/// number any brush hits today — it exists so a degenerate input cannot ask the cell
/// scratch for a stencil coarser than the shoulder argument was ever measured at.
const EXTENT_CELL_MAX: f32 = 16.0;

/// The **extent cell** (§6.2): the edge, in canvas texels, of the square over which
/// the coarse deposit may evaluate the exchange laws *once* and apply the result to
/// every texel inside — 1 meaning the exact per-texel kernel and nothing else.
///
/// The bound is the tip's **shoulder** — the width of a round tip's coverage falloff,
/// `3·(1−hardness)·radius` for the `1 − |y|^h` profile family — because that is the
/// finest feature the extent-domain fields can carry: the prefix-τ differences, the
/// baked reservoir means and the exchange solves the cell hoists are all smooth at the
/// scale the coverage itself varies. A quarter of the shoulder puts at least four
/// cells across the falloff; the `0.02·radius` term keeps the cell a fixed small
/// fraction of the tip where the shoulder is generous. Both constants are the
/// stroke-space march round's, kept because they were *measured* there: the ripple a
/// coarse cell prints stayed at the no-coarsening floor under the shoulder bound
/// (0.62 vs 0.58 levels rms column-mean) and broke it under a radius-only bound
/// (1.04), and a radius-scaled cell over a shoulderless tip was exactly the
/// stroke-end spike regression of 2026-08-07.
///
/// Two properties are load-bearing rather than tuning:
///
/// * **A hard tip earns no coarsening.** `hardness = 1` has no shoulder, so the min
///   is 0 and the cell is 1 — the exact kernel, bit-for-bit, by construction. A
///   `Stamp` mask can be arbitrarily hard, so it is treated as the sharpest case and
///   never coarsened at all.
/// * **A pure function of the brush shape and the segment's radius**, like every
///   other number in this file — a live tail and its commit pick the same cell for
///   the same segment, which is what `preview == committed` (§1.3) needs from it.
///
/// The threshold is 2: a cell must *beat* two texels before the coarse path engages,
/// because below that the hoist pass costs more than the ~4× it saves — which also
/// means the whole bench sweep at radius ≤ 100 (where `0.02·r ≤ 2`) stays on the
/// exact kernel, dispatch for dispatch.
///
/// The **shoulder** — the width of the tip's coverage falloff per unit radius — is
/// [`shoulder_per_radius`], shared with the taper's subdivision, which leans on the
/// same fact from the other side: a feature narrower than a quarter of the shoulder
/// is one the coverage cannot show.
pub(super) fn extent_cell(shape: &BrushShape, radius: f32) -> u32 {
    let shoulder = shoulder_per_radius(shape) * radius;
    let cell = (0.02 * radius).min(0.25 * shoulder);
    if cell <= 2.0 {
        1
    } else {
        cell.min(EXTENT_CELL_MAX) as u32
    }
}

/// The width of the tip's coverage falloff — its **shoulder** — per unit radius:
/// `3·(1−hardness)` for the round tip's `1 − |y|^h` profile family, and 0 for a
/// `Stamp`, which may be arbitrarily hard and is treated as the sharpest case.
///
/// The one definition, used from both sides of the same fact: features narrower than
/// a fraction of the shoulder are ones the coverage cannot carry. [`extent_cell`]
/// spends that as *coarsening* (the cell the coarse deposit may evaluate at), the
/// taper's subdivision as *smoothness* (the radius step a segment boundary may take
/// without printing, `segments::taper::Taper`).
pub(super) fn shoulder_per_radius(shape: &BrushShape) -> f32 {
    match shape {
        BrushShape::Round { hardness } => 3.0 * (1.0 - hardness.clamp(0.0, 1.0)),
        BrushShape::Stamp(_) => 0.0,
    }
}

/// The supersampling factor the swept deposit renders its parcel at when
/// [`supersample_scale`] engages: 2 per axis, so the integrate resolves each canvas
/// texel from a 2×2 block of finished subsample parcels (§6.2).
pub(super) const SUPERSAMPLE: u32 = 2;

/// The 10–90% span of `1 − exp(−x)`, `ln 9` — what converts an interior optical
/// mass into the *visible* width of the edge a linear τ ramp draws through the slab
/// law (§6.1): the transition occupies `ln 9 / (K·m)` of the ramp.
const EDGE_10_90: f32 = 2.197_224_6;

/// The narrowest visible edge the 1× render is allowed to draw before the sweep
/// supersamples: under ¾ px, the slab law has re-sharpened the pixel footprint's
/// ~px τ ramp into a step the tile grid can only alias.
const SUPERSAMPLE_EDGE_PX: f32 = 0.75;

/// How finely the swept deposit rasterizes `b`'s parcel: [`SUPERSAMPLE`] where the
/// stroke's **visible** edge would come out sharper than the pixel grid, 1 — the
/// plain path, bit for bit — everywhere else (§6.2).
///
/// The pixel footprint's box filter makes **τ** cross a rim as a ~px ramp, but what
/// the eye sees is the parcel's visible alpha `1 − exp(−K·m)`, and the exponential
/// re-sharpens the ramp: the 10–90% transition occupies `ln 9 / (K·m_interior)` of
/// it, so a heavy stroke renders a fraction of a px of visible edge from a full px
/// of coverage. No per-segment correction can widen it back — a shape other than
/// §6.2's two makes the stroke depend on where the flattener cut — so the correct
/// pixel is the **average of the finished parcel**, taken after the cross-segment
/// composition: the sweep rasterizes at 2× and the integrate box-resolves what the
/// slab law produced (`integrate.wesl`). This gate is what decides who pays for
/// that, from the numbers already in hand:
///
///   * the interior mass of one nominal pass, `flow · TAU_PER_PASS` — drain, tooth
///     and modulation only ever scale it down, so the brush's own value bounds it;
///   * the τ ramp's width: the tip's shoulder in px, floored at the box filter's
///     one px — a `Stamp` may be arbitrarily hard, so it gets the floor alone.
///
/// A **pure function of the brush**, like everything in this file: a live tail,
/// its commit and a replay make the same choice, which is what lets the carried
/// parcel resume across pieces at one resolution (§1.3). Only the paint effect
/// answers with 2: the wet loop's exposure is paired point-for-point with its bake
/// rows and the erase reads a different law — both stay 1×.
pub(super) fn supersample_scale(b: &BrushParams) -> u32 {
    let Some(p) = b.paint() else { return 1 };
    let mass = stark_shaders::mirror::paint_common::OPACITY_K * p.flow * TAU_PER_PASS;
    let ramp = (shoulder_per_radius(&b.shape) * b.size.max(0.5)).max(1.0);
    // `mass ≤ 0` (a brush laying nothing) divides to ∞ or NaN, and neither is
    // under the threshold — the comparison answers 1 without a guard.
    if ramp * EDGE_10_90 / mass < SUPERSAMPLE_EDGE_PX {
        SUPERSAMPLE
    } else {
        1
    }
}

/// The hardness a round tip actually **bakes** at `radius` px (§6.6): the brush's
/// own, floored so the shoulder above never falls under one canvas px. A falloff
/// narrower than a px is content past the tile grid's Nyquist — it can only shimmer —
/// so a hard edge keeps the ~px antialiased rim every selection mask's feather
/// already has ("floored at one, which *is* the antialiased hard edge", §6.8), and a
/// tip too small to carry even that comes out as soft as its own footprint.
///
/// Beside [`shoulder_per_radius`] because it is the same fact bounded from the other
/// side. Deliberately **not** fed back into the budgets built on the nominal
/// hardness (the taper's subdivision, [`extent_cell`]): the nominal shoulder is the
/// narrower of the two, so those bounds only over-provide for a floored tip.
pub(super) fn effective_hardness(hardness: f32, radius: f32) -> f32 {
    hardness.min(1.0 - 1.0 / (3.0 * radius.max(0.5)))
}

#[cfg(test)]
mod tests {
    use super::*;
    // --- the extent cell ------------------------------------------------

    /// The property the whole coarse deposit rests on: **a tip with no shoulder earns
    /// no coarsening**, at any size. `hardness = 1` and every `Stamp` mask take the
    /// exact per-texel kernel bit-for-bit — not approximately, structurally: the cell
    /// is 1, so the host never even dispatches the coarse pipelines. This is the
    /// 2026-08-07 stroke-end spike regression, pinned as arithmetic.
    #[test]
    fn a_shoulderless_tip_is_never_coarsened() {
        let stamp = BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        for radius in [8.0f32, 100.0, 250.0, 500.0, 4000.0] {
            assert_eq!(extent_cell(&BrushShape::Round { hardness: 1.0 }, radius), 1);
            assert_eq!(extent_cell(&stamp, radius), 1);
        }
    }

    /// A softer tip is never resolved *finer* than a harder one of the same size —
    /// the law is monotone in the shoulder, so there is no hardness at which
    /// softening a brush makes it more expensive.
    #[test]
    fn a_softer_tip_never_gets_a_finer_cell() {
        for radius in [50.0f32, 250.0, 500.0] {
            let mut last = u32::MAX;
            for h in [0.0f32, 0.25, 0.5, 0.8, 0.95, 0.99, 1.0] {
                let cell = extent_cell(&BrushShape::Round { hardness: h }, radius);
                assert!(
                    cell <= last,
                    "radius {radius}: hardness {h} got cell {cell}, harder was {last}",
                );
                last = cell;
            }
        }
    }

    /// Where the bench sweep actually lands, pinned so a retune is a deliberate act:
    /// every radius up to 100 floors to the exact kernel (the `0.02·r` term is ≤ 2
    /// there), the 250/500 lines coarsen under it, and a nearly-hard wide tip is
    /// bounded by its shoulder instead.
    #[test]
    fn the_cell_law_lands_where_the_bench_reads_it() {
        let soft = BrushShape::Round { hardness: 0.5 };
        for radius in [8.0f32, 30.0, 100.0] {
            assert_eq!(
                extent_cell(&soft, radius),
                1,
                "radius {radius} must stay exact"
            );
        }
        assert_eq!(extent_cell(&soft, 250.0), 5);
        assert_eq!(extent_cell(&soft, 500.0), 10);
        // Shoulder-bound: at hardness 0.99 a 500 px tip's shoulder is 15 px, so the
        // quarter-shoulder term (3.75) undercuts the 10 the radius term would give.
        assert_eq!(extent_cell(&BrushShape::Round { hardness: 0.99 }, 500.0), 3);
        // At least four cells across the shoulder wherever the shoulder binds.
        for h in [0.9f32, 0.95, 0.99] {
            let shoulder = 3.0 * (1.0 - h) * 500.0;
            let cell = extent_cell(&BrushShape::Round { hardness: h }, 500.0);
            assert!(
                cell as f32 * 4.0 <= shoulder || cell == 1,
                "hardness {h}: cell {cell} puts fewer than 4 cells across the \
                 {shoulder} px shoulder",
            );
        }
    }

    // --- the hardness floor ---------------------------------------------

    /// [`effective_hardness`]'s whole claim, read through the shoulder it floors: the
    /// baked tip's shoulder is never under a canvas px, a brush the floor does not
    /// bind keeps its hardness exactly, and the floor relaxes monotonically as the
    /// tip grows — so resizing a hard brush never makes its edge *softer* in px.
    #[test]
    fn the_baked_shoulder_never_falls_under_a_px() {
        for radius in [0.1f32, 0.5, 2.0, 16.0, 100.0, 500.0] {
            let mut last = 0.0f32;
            for h in [0.0f32, 0.3, 0.7, 0.9, 0.99, 1.0] {
                let eff = effective_hardness(h, radius);
                let shoulder =
                    shoulder_per_radius(&BrushShape::Round { hardness: eff }) * radius.max(0.5);
                assert!(
                    shoulder >= 1.0 - 1e-5,
                    "radius {radius}, hardness {h}: baked shoulder is {shoulder} px",
                );
                assert!(eff <= h, "the floor must never harden a brush");
                assert!(eff >= last, "the floor must stay monotone in hardness");
                last = eff;
            }
        }
        // Wherever the tip can carry a px of shoulder, the hardness is untouched.
        assert_eq!(effective_hardness(0.9, 100.0), 0.9);
        assert_eq!(effective_hardness(0.5, 2.0), 0.5);
    }

    // --- the supersample gate --------------------------------------------

    /// Who pays for the supersampled resolve (§6.2), pinned at the corners: a
    /// heavy hard tip gates on, a soft or light one stays on the plain path bit
    /// for bit, and the two non-paint effects never answer 2 — the wet loop's
    /// exposure is paired with its bake rows and the erase reads a different law.
    #[test]
    fn only_a_visibly_sharp_paint_edge_supersamples() {
        let heavy_hard = |hardness: f32, flow: f32, size: f32| {
            let mut b = BrushParams {
                shape: BrushShape::Round { hardness },
                size,
                ..BrushParams::default()
            };
            b.paint_mut().expect("a paint brush").flow = flow;
            b
        };
        // A hard tip at real flow: the 1× visible edge is a fraction of a px.
        assert_eq!(supersample_scale(&heavy_hard(1.0, 2.5, 10.0)), SUPERSAMPLE);
        // The same flow through a soft tip: the shoulder already spans px.
        assert_eq!(supersample_scale(&heavy_hard(0.5, 2.5, 20.0)), 1);
        // A hard tip laying almost nothing: the slab law never saturates, so the
        // τ ramp *is* the visible edge.
        assert_eq!(supersample_scale(&heavy_hard(1.0, 0.1, 10.0)), 1);
        // A brush laying nothing at all divides to ∞ and must still answer 1.
        assert_eq!(supersample_scale(&heavy_hard(1.0, 0.0, 10.0)), 1);
        // A stamp mask may be arbitrarily hard, so it is treated as the sharpest
        // case: the box filter's px is its whole ramp.
        let mut stamp = heavy_hard(1.0, 2.5, 10.0);
        stamp.shape = BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        assert_eq!(supersample_scale(&stamp), SUPERSAMPLE);
        // The other two effects keep their paths whatever their rates say.
        let mut wet = heavy_hard(1.0, 2.5, 10.0);
        wet.make_wet();
        assert_eq!(supersample_scale(&wet), 1);
        let mut erase = heavy_hard(1.0, 2.5, 10.0);
        erase.effect = stark_model::document::BrushEffect::Erase(Default::default());
        assert_eq!(supersample_scale(&erase), 1);
    }
}
