//! The dispatch plan: what the loop is asked to do, worked out entirely on the CPU
//! (§6.2).
//!
//! **Nothing here names `wgpu`.** It is float arithmetic over a [`StrokeRecord`] and
//! the [`Segment`]s it was flattened into, producing [`Stamp`] slots and the workgroup
//! counts that go with them, so every property below is testable without an adapter.
//! The one GPU *type* that reaches in is the canvas
//! [`SubstrateMap`](crate::gpu::substrate), and only for `bearing` — CPU arithmetic
//! over statistics that happen to live on the struct owning the texture.
//!
//! Every slot is a pure function of the record and the piece's own geometry, so replay
//! is deterministic (§12.1).

use stark_model::document::StrokeRecord;
use stark_model::geom::Vec2;

use super::super::budget::extent_cell;
use super::super::region::{coverage_bounds, segment_end};
use super::super::segments::{BleedFire, Segment, Stretch, Sweep};
use super::bleed::bleed_stencil;
use crate::gpu::substrate::TexelsPerPx;
// The `Stamp` uniform, generated from `dynamics_common.wesl`'s own declaration
// (`stark-shaders-build`): the shader decides how the lanes are read, so it is the
// only place they are written down (§6.10).
use stark_shaders::mirror::dynamics::CELL_BORDER;
use stark_shaders::mirror::dynamics_common::{Stamp, TILE_WG};

/// One slot of the sequential swept-exchange loop (§6.2), and the dispatches it
/// stands for — `K` being the step vocabulary of the path it is planned for,
/// [`WetStep`] or [`WarpStep`].
pub(super) struct LoopDispatch<K> {
    pub(super) slot: Stamp,
    /// The slot's extent work in **region texels** — what the `deposit` covers, and
    /// the `snapshot` that rides in `exchange`'s grid. Over the slot's own coverage box
    /// rather than a piece-wide square, so an axis-aligned sweep pays for the ~4·r²
    /// texels its extent can reach rather than the ~10·r² a diagonal one might need.
    ///
    /// Texels, not workgroups: each dispatch divides by **its own** kernel's declared
    /// `@workgroup_size` ([`Kernel::dispatch`](super::kit::Kernel::dispatch), §6.10),
    /// so no host expression states a grid a shader already declares.
    pub(super) extent: (u32, u32),
    pub(super) step: K,
}

/// What one slot of a **wet** plan asks the loop for (§6.2). Each is a different
/// sequence over the same uniform, and only one touches the tool: the reservoir
/// ping-pong advances on a [`Segment`](WetStep::Segment) and on nothing else.
#[derive(Clone, Copy)]
pub(super) enum WetStep {
    /// A stretch of painting: `bake` → `exchange` (+ `snapshot`) → `deposit`.
    Segment {
        /// This segment's index among the piece's, which is also its instance in the
        /// ceiling lane's buffer ([`segment_instance`](super::super::swept)) — taken
        /// off the walk that emits the slot rather than counted a second time while
        /// recording.
        index: u32,
        /// The cells `cell_hoist` must cover when this slot takes the **coarse
        /// deposit** (§6.2) — `Some` exactly when [`extent_cell`] beat 1 for this
        /// segment's tip. `None` is the exact per-texel `deposit`, and the two steps
        /// below carry no cell at all, so the lateral flux and the pen-up never see
        /// one.
        cells: Option<(u32, u32)>,
    },
    /// A dedicated **bleed slot**: a quad whose sweep is one firing of the bleed
    /// cadence's travel window, with every vertical rate and the source zeroed.
    /// `snapshot` + `deposit` alone — the tool plays no part, so the reservoir
    /// ping-pong is left exactly where the previous segment put it.
    Bleed,
    /// The pen-up: `snapshot` → `bake` → `settle`. At most one per plan, and always
    /// its last slot — the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`).
    Settle,
}

/// What one slot of a **liquify** plan asks of the field (§6.13).
#[derive(Clone, Copy)]
pub(super) enum WarpStep {
    /// One segment of a liquify stroke: `snapshot_field` (over the whole square — the
    /// gather pulls from upstream of any one texel's own reach) → `warp`, composing
    /// the segment's step into the run's field. Nothing of the picture and nothing of
    /// the tool.
    Warp,
    /// The one resample of a liquify piece: `warp_apply` over the whole region, the
    /// base composite through the composed field into the region's channels. Always
    /// the last slot of a [`liquify_plan`], and the only slot whose rect is the region
    /// rather than a sweep's box.
    Apply,
}

/// **The neutral slot**: what a [`Stamp`] member means when the slot kind building it
/// leaves that member alone.
///
/// Zero everywhere except the seven lanes whose neutral value is **1**, six of which
/// are *scales*, where a zeroed lane does not say "none of this" but "none of the thing
/// it multiplies": `tooth_bearing` at 0 books the tool's half of every transfer against
/// no substrate at all — infinite tooth, not absent tooth; `cell_px` at 0 is no deposit
/// grid rather than the exact one; `stretch` zeroed is a tip of no width every prefix
/// difference is divided by (§6.6); `opacity` and `opacity_mod` at 0 are the strongest
/// ceiling there is where 1 is the identity branch (§6.2). The seventh, `tooth_give`,
/// runs the other way (`BrushParams::tooth_give`): zeroed it is the driest tip there is,
/// where 1 is a tip that follows every fall — which is what keeps a bleed slot's lateral
/// flux clear of the substrate (`dynamics.wesl::substrate_tooth`).
///
/// **Spelled out member by member on purpose.** A `..Default::default()` would let a
/// lane the shader grows join the zeros in silence, and half of what is below is not
/// zero; the compile error a new member raises here is where it gets classified.
const NEUTRAL: Stamp = Stamp {
    start: [0.0; 2],
    dir: [0.0; 2],
    radius: 0.0,
    travel_radii: 0.0,
    radius_ramp: 0.0,
    lambda_lift: 0.0,
    lambda_deposit: 0.0,
    lambda_bleed: 0.0,
    curvature: 0.0,
    drag: 0.0,
    brush_lat: [0.0; 3],
    opacity: 1.0,
    brush_res: [0.0; 3],
    add: 0.0,
    noise_freq: [0.0; 3],
    arc_at_start: 0.0,
    noise_amp: [0.0; 3],
    drain: 0.0,
    stretch: Stretch::NONE.lanes(),
    orientation: 0.0,
    substrate_num: TexelsPerPx::NONE.num,
    rect_origin: [0; 2],
    cell_anchor: [0; 2],
    tooth_give: 1.0,
    tooth_softness: 0.0,
    tooth_bearing: 1.0,
    // A divisor: `NONE` gates on the zero numerator above and keeps 1 here, so a missed
    // guard cannot divide by zero (§6.4).
    substrate_den: TexelsPerPx::NONE.den,
    cell_px: 1,
    bleed_reach: 0,
    canvas_origin: [0; 2],
    jitter_eps: 0.0,
    jitter_seed: 0,
    opacity_mod: 1.0,
    opacity_ramp: 0.0,
    add_ramp: 0.0,
    tooth_give_ramp: 0.0,
    drag_ramp: 0.0,
    ceiling_lane: 0,
    base_offset: [0; 2],
};

/// The [`Stamp`] lanes every slot in a plan fills the same way, resolved once, so
/// each builder below lists only what its kind differs by.
struct SlotCommon<'a> {
    /// The stroke's own constants: `c` outright, and the color-dynamics lookup that
    /// fills `noise_freq` and `noise_amp`. Borrowed rather than copied out, so a slot
    /// and the swept path's `TileXform` are demonstrably reading one resolution of
    /// them.
    k: &'a super::super::StrokeConstants,
    /// The region rectangle's top-left in canvas px — the map from a region texel to
    /// the absolute canvas one the deposit jitter (§6.2) and the substrate tap (§6.4)
    /// are keyed on, and what every slot's own coordinates are measured from. Integral
    /// by the cell grid's own argument ([`cell_geometry`]'s assert), and the one thing
    /// here that is not a stroke constant: the piece's placement, which `k` cannot know.
    region_origin: Vec2,
}

impl SlotCommon<'_> {
    /// The lanes every slot fills the same way — the stroke's color and the substrate
    /// map — over [`NEUTRAL`].
    fn stamp(&self) -> Stamp {
        Stamp {
            brush_lat: self.k.channels,
            // The other half of the same color (§6.7) — filled wherever `brush_lat` is,
            // and zero in a space whose three channels already say everything. The
            // shader declares three where `k` keeps four, its `.w` being unused.
            brush_res: [self.k.resid[0], self.k.resid[1], self.k.resid[2]],
            // The paint effect's ceiling (§6.2), off `k` like the color: one
            // resolution for every slot kind and for the swept path's integrate.
            // Only a painting segment's mint reads it; a bleed or settle slot
            // carries it inertly, the jitter's argument one field down.
            opacity: self.k.opacity,
            // …and whether the pen drives it, which is likewise the stroke's. A
            // bleed or settle slot carries the flag inertly with the dial.
            ceiling_lane: u32::from(self.k.ceiling_lane),
            // Off `k` like the color: the swept path's `TileXform` reads the same pair,
            // and both key it on the canvas texel.
            substrate_num: self.k.substrate.num,
            substrate_den: self.k.substrate.den,
            // A stroke constant like the color above it, and off `k` for the same
            // reason: it is the one number the tooth needs that the pen never moves,
            // so every slot kind carries the same width and the swept path's
            // `TileXform` is demonstrably reading the same resolution of it.
            tooth_softness: self.k.tooth_softness,
            // The deposit jitter (§6.2), off `k` by the tooth width's argument: one
            // gate for every slot kind and for the swept path. Inert on a bleed slot
            // — its rates are zero, so the jittered exposure decides nothing — and
            // filled anyway, because an exception here would be a lane two slot
            // kinds disagree about.
            jitter_eps: self.k.jitter_eps,
            jitter_seed: self.k.jitter_seed,
            // The jitter's and the substrate tap's shared domain: the absolute canvas
            // grid, integral, so this loop can key a hash on canvas position at all
            // (§6.4).
            canvas_origin: [self.region_origin.x as i32, self.region_origin.y as i32],
            ..NEUTRAL
        }
    }

    /// [`Self::stamp`] plus the **sweep's own frame** — the ten lanes a stretch of
    /// travel fills identically whether it is painting, bleeding or warping.
    ///
    /// A bleed window's `radius_ramp` is 0 by construction (`bleed_fires`: a firing
    /// diffuses at the one radius its stencil was solved against), so the ramp is
    /// shared here exactly rather than by coincidence.
    fn swept(&self, sw: &Sweep, rect: &Rect) -> Stamp {
        Stamp {
            start: (sw.start - self.region_origin).to_array(),
            dir: sw.dir.to_array(),
            radius: sw.radius,
            travel_radii: sw.length / sw.radius,
            radius_ramp: sw.radius_ramp,
            curvature: sw.curvature,
            orientation: sw.orient,
            stretch: sw.stretch.lanes(),
            rect_origin: rect.origin,
            arc_at_start: sw.dist,
            ..self.stamp()
        }
    }

    /// [`Self::swept`] plus what a slot that lays the brush's own `add` paint carries:
    /// the color-dynamics jitter (§6.2), and the bearing fraction it books the tool's
    /// half of the transfer against (§6.4).
    ///
    /// `lambda_bleed` stays 0: the lateral flux runs only on the dedicated bleed
    /// slots, so between firings the canvas takes the no-bleed path bit-for-bit
    /// (§6.2).
    fn painting(&self, sw: &Sweep, rect: &Rect, bearing: f32) -> Stamp {
        let (nfreq, namp) = (self.k.nfreq, self.k.namp);
        Stamp {
            // Three lanes of a four-lane lookup on either side: the shader folds
            // 1/NOISE_TILE_PX into `.z` and has no use for a fourth.
            noise_freq: [nfreq[0], nfreq[1], nfreq[2]],
            noise_amp: [namp[0], namp[1], namp[2]],
            tooth_bearing: bearing,
            ..self.swept(sw, rect)
        }
    }
}

/// A stretch of painting (§6.2): the swept exchange over the segment's own coverage
/// box, at the rates the segment resolved.
fn segment_stamp(
    common: &SlotCommon<'_>,
    s: &Segment,
    rect: &Rect,
    grid: &CellGrid,
    bearing: f32,
) -> Stamp {
    let paint = &s.paint;
    Stamp {
        // The exchange rates as the segment resolved them, the flow already on the
        // exponent (`generate_segments_in`, §6.2) — nothing here scales, as nothing
        // here scales `add`.
        lambda_lift: paint.lambda_lift,
        lambda_deposit: paint.lambda_deposit,
        drain: common.k.drain_px,
        // The `add` rate as the segment resolved it — mappings and the wet flow already
        // folded (`generate_segments_in`), the pen's drive included (§6.2) — with **no
        // further gain here**, exactly as `stamp.wesl` takes it. A gain would make the
        // same slider mean two different amounts of paint depending on whether some
        // *other* axis happened to be non-zero, and none is needed: a pass of the tip
        // is `TAU_PER_PASS ≈ 6.9` of exposure, so a rate of 1 lays 6.9 of height, which
        // the slab law reads as 0.999 coverage. The effect's opacity is *not* folded in
        // either — the raw rate is what the ceiling's prefix-difference law must
        // accumulate (§6.2, `dynamics.wesl::lay_parcel`), and the ceiling rides the
        // slot's own lane.
        add: paint.add,
        // The ceiling's factor as the segment resolved it, beside the rate it caps —
        // the same pair the swept path reads off its instance, ramp included, so a
        // texel's factor is the same number on either path.
        opacity_mod: paint.opacity,
        opacity_ramp: paint.opacity_ramp,
        add_ramp: paint.add_ramp,
        tooth_give_ramp: paint.tooth_give_ramp,
        tooth_give: paint.tooth_give,
        cell_px: grid.edge,
        cell_anchor: grid.anchor,
        // No `bleed_reach` and no `lambda_bleed`: the lateral flux runs only on the
        // dedicated firings, so a painting segment takes the no-bleed path bit-for-bit
        // (§6.2). Both are [`NEUTRAL`]'s zero.
        ..common.painting(&s.sweep, rect, bearing)
    }
}

/// A dedicated **bleed slot**: a quad whose sweep is the firing's travel window, with
/// every vertical rate and the source zeroed — the dispatch is the identity everywhere
/// except the lateral flux. The noise lanes are zeroed too, so the deposit skips its
/// color-jitter taps.
///
/// Everything a painting segment carries and this does not is [`NEUTRAL`]'s zero,
/// which is what the slot *means*: λ_lift = 0 so the canvas keeps everything,
/// λ_deposit = 0 so the uninvolved tool lays nothing, no drain because nothing is laid,
/// no `add` because this is not a stretch of painting, and no tooth because there is no
/// `add` for the substrate to gate.
fn bleed_stamp(common: &SlotCommon<'_>, fire: &BleedFire, rect: &Rect) -> Stamp {
    let w = &fire.window;
    // The stencil this firing diffuses with: how far it reaches, and how hard it
    // relaxes to get there. Both come out of the diffusivity the axis asks for — see
    // [`bleed_stencil`], which is where the axis's whole meaning is.
    let (reach, lambda_bleed) = bleed_stencil(fire.bleed, w.radius, w.length);
    Stamp {
        // The stencil's longest tap — the only slot that carries one, and a whole
        // number by construction (`bleed_stencil` rounds it to a texel).
        bleed_reach: reach as i32,
        // The rate that lands this window's exposure on the blend its reach needs — not
        // `lambda(axis)`, which is the vertical rates' mapping and would make the axis a
        // rate rather than a diffusivity. A firing whose modulated axis has fallen to
        // zero still dispatches: λ = 0 is the identity, and the plan stays a pure
        // function of the segmentation.
        lambda_bleed,
        ..common.swept(w, rect)
    }
}

/// The pen-up (`dynamics.wesl::settle`): the last segment's tip **standing** where the
/// pen left the page, on the same uniform. Zero travel is what makes the shared
/// `segment_frame`/`outside_sweep` reduce to the tip's own extent and `snapshot` copy
/// exactly the texels the settle will write.
fn settle_stamp(common: &SlotCommon<'_>, s: &Segment, rect: &Rect, tan: Vec2) -> Stamp {
    let (sw, paint) = (&s.sweep, &s.paint);
    Stamp {
        // Where the travel ended, facing the way the pen was going ([`settle_tangent`])
        // — the two lanes a standing tip cannot take off the sweep it is the end of.
        start: (segment_end(sw) - common.region_origin).to_array(),
        dir: tan.to_array(),
        // A pen-up is a break of contact, not a stretch of it: no travel, and with it
        // no ramp and no curvature, the frame being a standing tip. The arc is the
        // segment's far end rather than its start, for the same reason.
        travel_radii: 0.0,
        radius_ramp: 0.0,
        curvature: 0.0,
        arc_at_start: sw.dist + sw.length,
        // The rates are the *last* segment's — where the pen was when it left the page
        // — and its λs carry that segment's flow, as a painting slot's do.
        lambda_lift: paint.lambda_lift,
        lambda_deposit: paint.lambda_deposit,
        drain: common.k.drain_px,
        // The last segment's tooth: the settle delivers what the pass still owed,
        // through the same substrate the pass was laying through. What the valleys do
        // not take stays on the discarded tool (§6.4).
        tooth_give: paint.tooth_give,
        // No `add`: the source is a rate per unit of travel, and there is none. No bleed
        // reach and no λ_bleed either: that axis carries no reservoir, every firing
        // having applied its window as the tip passed, so a break of contact strands
        // nothing to finish. All three are [`NEUTRAL`]'s zero.
        //
        // The bearing is the neutral 1: the tool is not written back at pen-up, and the
        // settle's own gate is per texel, from the substrate. The color channels are
        // filled as a segment slot's are, though the settle lays the tool's *carried*
        // paint.
        ..common.painting(sw, rect, 1.0)
    }
}

/// One segment of a **liquify** stroke (§6.13): the segment's step composed into the
/// run's field, and not one rate a deposit would run on.
fn warp_stamp(common: &SlotCommon<'_>, s: &Segment, rect: &Rect, base_offset: [i32; 2]) -> Stamp {
    let paint = &s.paint;
    Stamp {
        // The one rate a warp slot carries: the modulated strength, and how it changes
        // across the segment.
        drag: paint.drag,
        drag_ramp: paint.drag_ramp,
        base_offset,
        // The drag runs dry as a deposit would (§6.2): the drain at the texel's own
        // arc, which is what the sweep's arc length beside it is for.
        drain: common.k.drain_px,
        // Every rate a deposit would run on stays [`NEUTRAL`]'s zero — a warp slot
        // lays nothing, lifts nothing, bleeds nothing — and the noise lanes with them:
        // the effect has no color for a jitter to wander (§6.13). The tooth too, since a
        // field gated by the substrate's every step is a tear rather than a warp.
        ..common.swept(&s.sweep, rect)
    }
}

/// What a plan is built *against*, as opposed to the segments it is built *from*:
/// where the piece's region sits, the budget its segments were cut at, and the stroke
/// constants every slot is filled from. A slot's geometry is only meaningful relative
/// to these.
pub(super) struct PlanCtx<'a> {
    pub(super) rec: &'a StrokeRecord,
    /// The budget `rec` was flattened at, handed down from [`StrokePlan::of`](crate::gpu::stroke::path::StrokePlan::of) rather
    /// than recomputed — one place answers what a stroke's segments are. Only the
    /// pen-up frame reads it ([`settle_tangent`]), which re-flattens an extent's
    /// worth of the record and must cut it exactly as the segments in hand were cut.
    pub(super) tol: crate::path::FlattenTolerance,
    /// The region rectangle's top-left in canvas px — what every slot's coordinates
    /// are measured from, since the shader never learns where the piece sits.
    pub(super) region_origin: Vec2,
    /// Everything both render paths read off the record and the scene
    /// ([`StrokeConstants`](super::super::StrokeConstants)) — the color a slot's `c` is, the
    /// substrate map its `i` carries, and the color-dynamics lookup for `f`–`h`.
    pub(super) consts: &'a super::super::StrokeConstants,
    pub(super) substrate: &'a crate::gpu::substrate::SubstrateMap,
}

/// The margin, in canvas px, a dispatch rect is grown by each side so a fragment
/// sampling just outside its own texel still lands inside the rect.
///
/// Bounded by the region's own low margin, which is `2·TILE_APRON` (`Covered::rect`
/// takes the touched tiles' origin less an apron, and the touched set is floored from
/// bounds already grown by one). That is what keeps a rect origin non-negative, and so
/// every `rect_origin + tn` the bleed ladder derives off `lib::sample::load1`'s lower
/// clamp.
const RECT_MARGIN: f32 = 1.5;
const _: () = assert!(
    RECT_MARGIN <= 2.0 * stark_model::geom::TILE_APRON as f32,
    "a rect origin could go negative, and the bleed ladder's taps with it"
);

/// The snapshot square's pool quantum: [`snapshot_square`] rounds the measured maximum
/// up to a multiple of this.
///
/// For the scratch pool's sake alone: the raw maximum drifts a few texels per fold as
/// the tail's geometry evolves, which would make nearly every checkout a miss
/// ([`ScratchPool`](crate::gpu::scratch)). The round-up moves no pixel — stores and
/// reads are gated by the slot rects and the sweep test, and the texels it adds fall
/// outside both.
const SNAPSHOT_QUANTUM: u32 = 64;

/// One slot's dispatch rectangle over a canvas-space coverage box: its integral origin
/// in region texels and its extent, from which the workgroup counts follow. Texels the
/// rounding adds beyond the box read zero exposure and fall out of `deposit`
/// untouched.
fn dispatch_rect(lo: Vec2, hi: Vec2, region_origin: Vec2) -> Rect {
    let lo = lo - region_origin - Vec2::splat(RECT_MARGIN);
    let hi = hi - region_origin + Vec2::splat(RECT_MARGIN);
    let origin = Vec2::new(lo.x.floor(), lo.y.floor());
    Rect {
        // Integral by that floor, and carried as the integers the shader declares
        // (`Stamp::rect_origin`) rather than as floats it would have to round back.
        origin: [origin.x as i32, origin.y as i32],
        w: ((hi.x - origin.x).ceil() as u32) + 1,
        h: ((hi.y - origin.y).ceil() as u32) + 1,
    }
}

/// One slot's dispatch rectangle.
struct Rect {
    /// The rect's top-left in region texels — the `rect_origin` a slot carries.
    origin: [i32; 2],
    /// Its extent in texels.
    w: u32,
    h: u32,
}

impl Rect {
    /// The texels a tile kernel actually **scans** over this rect: it rounded up to
    /// whole workgroups of `TILE_WG`, the side `dynamics_common.wesl` declares (§6.10).
    ///
    /// The one place the host still turns texels into a grid and back, and it is the
    /// inverse that needs it: a rounding texel past the rect reads zero exposure and
    /// falls out of `deposit` untouched, but it still names a cell, which is what
    /// [`cell_geometry`] counts over.
    fn scanned(&self) -> (u32, u32) {
        (
            self.w.next_multiple_of(TILE_WG),
            self.h.next_multiple_of(TILE_WG),
        )
    }
}

/// The snapshot scratch's square for a piece: **the largest rect the piece will
/// actually dispatch**, rounded up to [`SNAPSHOT_QUANTUM`].
///
/// **A maximum, not a bound**: it is taken over the very rects the plan dispatches,
/// so no assertion in the render path has to hold two derivations of this number
/// together.
fn snapshot_square(rects: &[Rect]) -> u32 {
    // A floor of one workgroup, so an empty plan — which cannot happen, a piece holding
    // at least one segment — still names a texture the device will create.
    rects
        .iter()
        .fold(TILE_WG, |m, r| m.max(r.w).max(r.h))
        .next_multiple_of(SNAPSHOT_QUANTUM)
}

/// The cell scratch's square for a piece, in cells: enough for any rect the piece's
/// snapshot scratch admits at the finest cell the coarse path runs (2), plus the
/// [`CELL_BORDER`] ring on each side. Both sides of the relation go through this
/// function — [`cell_geometry`] asserts against it, `DynamicsRun::cell_scratch`
/// allocates from it — so no slot a plan can build reads cells the scratch lacks.
pub(super) fn cell_scratch_size(dsize: u32) -> u32 {
    dsize.div_ceil(2) + 2 + (2 * CELL_BORDER) as u32
}

/// Cells the hoist must write to cover `span` texels from region texel `base`
/// (already offset by the anchor), border included — the window
/// `deposit_coarse`'s taps are asserted to sit inside.
fn cell_span(base: i32, span: i32, c: i32) -> u32 {
    ((base + span - 1).div_euclid(c) - base.div_euclid(c) + 1) as u32 + (2 * CELL_BORDER) as u32
}

/// The coarse deposit's per-slot geometry (§6.2) — everything one slot's extent cell
/// implies, derived together so the lane the shader reads and the grid the hoist is
/// dispatched over cannot come apart.
struct CellGrid {
    /// The cell's edge in texels (§6.2). 1 is the exact per-texel deposit, which is
    /// also [`NEUTRAL`]'s value: the exact kernels never read the lane.
    edge: i32,
    /// The cell grid's canvas anchor: the region's canvas origin modulo the edge, so
    /// cell boundaries are pinned to canvas texels whatever region surrounds them
    /// (§6.4). Zero whenever [`edge`](Self::edge) is 1.
    anchor: [i32; 2],
    /// The `cell_hoist` workgroup counts covering every cell the slot's texel grid can
    /// touch — `None` when the slot keeps the exact kernel and dispatches no hoist.
    hoist: Option<(u32, u32)>,
}

/// The [`CellGrid`] for a slot dispatching over `rect` at extent cell `cell`.
///
/// The cells are counted over the texels the deposit will actually scan — the
/// dispatch rect *as rounded up to whole workgroups*, clamped to the snapshot
/// scratch — because a rounding texel past the rect still passes the shader's bounds
/// checks and looks its cell up; a cell it can name must be one the hoist wrote, or
/// the deposit reads whatever the previous segment left in the scratch. Each texel
/// names four, so the count carries a [`CELL_BORDER`] ring on each side.
fn cell_geometry(cell: u32, region_origin: Vec2, rect: &Rect, dsize: u32) -> CellGrid {
    if cell <= 1 {
        return CellGrid {
            edge: 1,
            anchor: [0; 2],
            hoist: None,
        };
    }
    let c = cell as i32;
    // The anchor is congruence arithmetic, so it needs the origin to *be* the
    // integer it is (a tile origin less the apron): a fractional origin would put
    // the grid off the canvas texels everything above says it is on.
    debug_assert!(
        region_origin.x.fract() == 0.0 && region_origin.y.fract() == 0.0,
        "a region origin must sit on a canvas texel for the cell grid to anchor to it",
    );
    // Restating `RegionRect::origin`'s own guarantee rather than establishing it — see
    // there for why a fractional origin cannot arrive, and why it would be a seam if
    // one did.
    let anchor = [
        (region_origin.x as i32).rem_euclid(c),
        (region_origin.y as i32).rem_euclid(c),
    ];
    let scanned = rect.scanned();
    let cells = |origin: i32, a: i32, span: u32| cell_span(origin + a, span.min(dsize) as i32, c);
    let cx = cells(rect.origin[0], anchor[0], scanned.0);
    let cy = cells(rect.origin[1], anchor[1], scanned.1);
    // Arithmetic rather than a real risk: a rect spans at most `dsize` texels, so at
    // a cell of `c ≥ 2` it names at most
    // `ceil(dsize/c) + 1 + 2·CELL_BORDER ≤ dsize.div_ceil(2) + 3` cells, and
    // `cell_scratch_size` is that plus one. Debug-only because a panic mid-render is a
    // worse failure than what it guards.
    let fit = cell_scratch_size(dsize);
    debug_assert!(
        cx <= fit && cy <= fit,
        "a {cx}x{cy}-cell hoist overruns the {fit}-cell scratch",
    );
    CellGrid {
        edge: c,
        anchor,
        hoist: Some((cx, cy)),
    }
}

/// What one slot of the plan is built from — the vocabulary [`slot_sources`] walks in.
enum SlotSource<'a> {
    /// A segment and its index among the piece's, which the plan hands on as
    /// [`WetStep::Segment::index`].
    Segment(usize, &'a Segment),
    Bleed(&'a BleedFire),
    /// The pen-up, which is the last segment read a different way — a standing tip
    /// rather than a stretch of travel.
    Settle(&'a Segment),
}

impl SlotSource<'_> {
    /// The canvas box this slot's dispatch has to cover.
    fn bounds(&self) -> (Vec2, Vec2) {
        match self {
            SlotSource::Segment(_, s) => coverage_bounds(&s.sweep),
            SlotSource::Bleed(f) => coverage_bounds(&f.window),
            // The tip's own extent rather than a swept box — a pen-up is a standing
            // tip, whose half-extent is the tip's `reach` (`Sweep::reach`). It cannot
            // be the largest box in the piece: a segment's box is this square grown by
            // its travel, and this is the last segment's.
            SlotSource::Settle(s) => {
                let end = segment_end(&s.sweep);
                let reach = Vec2::splat(s.sweep.reach);
                (end - reach, end + reach)
            }
        }
    }
}

/// **What a piece's plan dispatches, in the order it dispatches them**: each segment,
/// then the bleed firings that land after it, and the pen-up last.
///
/// The one statement of that order — the rects are measured over this list and the
/// slots built by zipping the two, so the passes cannot disagree about which rect
/// belongs to which slot, and the tests that ask what the scratch is sized against ask
/// this rather than a copy of it.
///
/// `first` is the piece's own first segment among the range's, because a firing's
/// `after` indexes the **range**: a piece takes a narrowed slice of the firings, never
/// a re-keyed copy of them.
fn slot_sources<'a>(
    segments: &'a [Segment],
    fires: &'a [BleedFire],
    first: usize,
    settle: bool,
) -> Vec<SlotSource<'a>> {
    // Drained in step with the walk, which is only correct because `bleed_fires` emits
    // them in segment order. Cheap to state, and the alternative — a firing silently
    // landing in the wrong piece of the plan — is not something a pixel would show.
    debug_assert!(
        fires.is_sorted_by_key(|f| f.after),
        "bleed firings must arrive in segment order",
    );
    let mut sources = Vec::with_capacity(segments.len() + fires.len() + usize::from(settle));
    let mut pending = fires.iter().peekable();
    for (si, s) in segments.iter().enumerate() {
        sources.push(SlotSource::Segment(si, s));
        while let Some(fire) = pending.next_if(|f| f.after == first + si) {
            sources.push(SlotSource::Bleed(fire));
        }
    }
    // The pen-up (`dynamics.wesl::settle`), as one more slot on the same uniform: the
    // tip standing at the stroke's last point with **zero travel**.
    if let Some(s) = settle.then(|| segments.last()).flatten() {
        sources.push(SlotSource::Settle(s));
    }
    sources
}

/// The rect each of `sources` dispatches over. Split out so the fit the scratch is
/// sized by can be exercised without an adapter (`tests`).
fn rects_for(sources: &[SlotSource<'_>], region_origin: Vec2) -> Vec<Rect> {
    sources
        .iter()
        .map(|src| {
            let (lo, hi) = src.bounds();
            dispatch_rect(lo, hi, region_origin)
        })
        .collect()
}

/// The plan's slots in dispatch order, and the snapshot square they fit. The square
/// rides with the slots because it is derived from them ([`snapshot_square`]): there
/// is no way to allocate a scratch for a plan other than the one that measured it.
///
/// Generic in the step vocabulary, so neither record loop can be handed the other's
/// plan.
pub(super) struct Plan<K> {
    pub(super) slots: Vec<LoopDispatch<K>>,
    pub(super) dsize: u32,
}

/// Build the swept-exchange dispatch plan (§6.2): one `snapshot` + `deposit` pair per
/// flattened segment (the canvas-side exchange, swept through the prefix-τ integral),
/// each followed by the tool's own `exchange`. λ = ln(1 − axis) makes every rate
/// exponential in exposure, so the exchange composes exactly across overlapping
/// segment quads — the continuous path integral, independent of any spacing.
///
/// Every painting dispatch is a segment: the tool exchanges once per segment rather
/// than on a cadence of its own, so there is no interval state to carry between
/// ranges. The bleed cadence and the pen-up ride as their own [`WetStep`]s.
///
/// `fires` is the range's firings narrowed to this piece and `first` the piece's own
/// first segment among the range's, which is what [`slot_sources`] pairs them by.
pub(super) fn dynamics_plan(
    ctx: &PlanCtx<'_>,
    segments: &[Segment],
    fires: &[BleedFire],
    first: usize,
    settle: bool,
) -> Plan<WetStep> {
    let &PlanCtx {
        rec,
        region_origin,
        consts,
        substrate,
        ..
    } = ctx;
    // What share of the substrate a tip with this tooth, going this way, stands on —
    // per segment, because the tooth's *give* is modulated per segment (§6.2) and the
    // direction is the segment's own. The canvas side of the exchange asks the
    // substrate ahead of each texel; the tool has none of its own and books against
    // this mean, which is what makes a toothed smear conserve
    // (`SubstrateMap::bearing`). The mean has to be taken through the very gate the
    // canvas side evaluates, or the two halves of the transfer stop agreeing — hence
    // the transition width the slot lanes carry too.
    let bearing =
        |give: f32, dir: Vec2| substrate.bearing(give, consts.tooth_softness, dir.to_array());
    let common = SlotCommon {
        k: consts,
        region_origin,
    };

    // ---- Pass one: what the plan dispatches, in order ([`slot_sources`]).
    let sources = slot_sources(segments, fires, first, settle);

    // ---- Pass two: the rect each of those dispatches over, and the scratch square
    // that is their maximum. Computed once and carried, so the rect that sizes the
    // scratch and the rect that is dispatched are the same number rather than two
    // measurements an assertion in the render path has to hold together.
    let rects = rects_for(&sources, region_origin);
    let dsize = snapshot_square(&rects);

    // ---- Pass three: the slots themselves, zipped to the rects measured for them.
    let slots = sources
        .iter()
        .zip(&rects)
        .map(|(src, rect)| {
            let extent = (rect.w, rect.h);
            match src {
                SlotSource::Segment(si, s) => {
                    let sw = &s.sweep;
                    // The tangent at the segment's **midpoint**, along the arc rather
                    // than the chord: a curved segment's canvas side sees a heading
                    // that turns across the sweep, and the midpoint's error is second
                    // order where either endpoint's would be first.
                    let (_, mid_dir) =
                        crate::path::arc_at(sw.start, sw.dir, sw.curvature, sw.length * 0.5);
                    // The extent cell this segment's deposit may evaluate the exchange
                    // at (§6.2): a pure function of the brush shape and the segment's
                    // own radius (`extent_cell`), so a live tail and its commit pick
                    // the same cell.
                    let grid = cell_geometry(
                        extent_cell(&rec.brush.shape, sw.radius),
                        region_origin,
                        rect,
                        dsize,
                    );
                    LoopDispatch {
                        extent,
                        step: WetStep::Segment {
                            index: *si as u32,
                            cells: grid.hoist,
                        },
                        slot: segment_stamp(
                            &common,
                            s,
                            rect,
                            &grid,
                            bearing(s.paint.tooth_give, mid_dir),
                        ),
                    }
                }
                SlotSource::Bleed(fire) => LoopDispatch {
                    extent,
                    // No cells: a bleed slot keeps the exact deposit whatever the tip,
                    // since the ladder's flux pairs need both threads of a pair to read
                    // per-texel exposures, and the firings are rare and small next to
                    // the painting they cut.
                    step: WetStep::Bleed,
                    slot: bleed_stamp(&common, fire, rect),
                },
                SlotSource::Settle(s) => LoopDispatch {
                    extent,
                    // No cells either: the settle is one dispatch at the end of a
                    // stroke — nothing to amortize — and its p-norm handover is exactly
                    // the smooth structure a cell would staircase, so it stays exact
                    // whatever the tip.
                    step: WetStep::Settle,
                    // The frame comes off the *record* — see [`settle_tangent`].
                    // `segments` is one piece of one range, and a lookback that walked
                    // it would stop wherever that cut fell, so a live tail and its
                    // commit would settle at different angles.
                    slot: settle_stamp(&common, s, rect, settle_tangent(rec, ctx.tol, segments)),
                },
            }
        })
        .collect();
    Plan { slots, dsize }
}

/// Build the liquify dispatch plan (§6.13): one warp slot per flattened segment,
/// in stroke order — each composing its step into the run's field — and, last, the
/// one `warp_apply` slot that resamples the whole region through the composed
/// field. No bleed cadence (the effect has no such axis) and no pen-up (the field
/// is complete as the tip passes; a break of contact strands nothing).
///
/// `base_offset` is where the region sits in the base composite, and `region` the
/// region's own extent; only the apply slot reads either.
///
/// The snapshot square is measured over the **warp** slots alone: the apply slot
/// reads no snapshot, and its rect — the region — would otherwise size a scratch
/// nothing dispatches over.
pub(super) fn liquify_plan(
    ctx: &PlanCtx<'_>,
    segments: &[Segment],
    base_offset: [i32; 2],
    region: (u32, u32),
) -> Plan<WarpStep> {
    let &PlanCtx {
        region_origin,
        consts,
        ..
    } = ctx;
    let common = SlotCommon {
        k: consts,
        region_origin,
    };
    // The same three-pass shape as [`dynamics_plan`], through the same walk with
    // nothing for it to interleave: no bleed cadence and no pen-up, so the sources are
    // the segments. The rects are measured over them and the slots zip to those, so
    // the rect that sizes the scratch is the rect that dispatches.
    let sources = slot_sources(segments, &[], 0, false);
    let rects = rects_for(&sources, region_origin);
    let dsize = snapshot_square(&rects);
    let mut plan: Vec<LoopDispatch<WarpStep>> = Vec::with_capacity(sources.len() + 1);
    plan.extend(segments.iter().zip(&rects).map(|(s, rect)| LoopDispatch {
        extent: (rect.w, rect.h),
        step: WarpStep::Warp,
        slot: warp_stamp(&common, s, rect, base_offset),
    }));
    // The resample, over the region: its rect is the region itself, at the region's
    // own origin.
    plan.push(LoopDispatch {
        extent: region,
        step: WarpStep::Apply,
        slot: Stamp {
            base_offset,
            ..common.stamp()
        },
    });
    Plan { slots: plan, dsize }
}

/// The travel direction the pen-up settle measures `owed` and `received` along: the
/// chord over the **last extent's worth of path**, rather than the last segment's
/// own tangent.
///
/// The last segment's tangent cannot be trusted, and that is a property of real input
/// rather than a rare edge case: a hand pauses before it lifts, so a pen-up arrives as
/// a cluster of samples at almost one point, and the flattener turns those into edges
/// whose chord is a rounding error and whose direction is therefore arbitrary. Nothing
/// else in the loop notices — a segment of no length deposits nothing — but the settle
/// takes a whole tip's worth of exchange from that one frame, and its
/// `min(owed, received)` lens is elongated *along* it, so a wrong direction lands a
/// tip-shaped disc across the stroke instead of along it.
///
/// One radius is the window because it is the extent of the thing being settled, so
/// this is the direction the tip was travelling over precisely the stretch of canvas
/// the settle acts on.
///
/// **Measured on the record, not on the segments in hand**, which is why it takes a
/// `rec`: the slice a plan is built from is one *piece* of one *range*, so a lookback
/// that walked it would stop at whichever cut came first, and a live tail would settle
/// at a different angle from its commit — a `preview == committed` break (§1.3) in the
/// one place it cannot be repainted. It walks the curve's own polyline rather than
/// back along the last segment's arc, the way
/// [`bleed_fires`](super::bleed::bleed_fires) does, because the last segments *are*
/// the degenerate ones (`span_end` prices a span boundary without subdividing
/// anything, so finding them costs no polyline).
///
/// `segments` is still read for the two things only it knows: the radius of the tip
/// being settled, and a click's frame — a lone control point is not a curve, and
/// `generate_segments_in` gives its dab a real direction where the path has none.
fn settle_tangent(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    segments: &[Segment],
) -> Vec2 {
    let radius = segments.last().map_or(1.0, |s| s.sweep.radius);
    // A click's frame: the dab's own direction, which is deliberate rather than
    // fitted. `generate_segments_in` sweeps it symmetrically about the point pressed,
    // so which direction it is cannot matter — but it has to be *a* direction.
    let fallback = || segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.sweep.dir);
    let last = crate::path::span_count(rec.path.len());
    if last == 0 {
        return fallback();
    }
    let tip = crate::path::span_end(&rec.path, last - 1);
    // The first span boundary a radius or more back from the tip, measured on chords
    // — which under-estimate arc length, so the span this admits genuinely holds an
    // extent's worth of path behind it. Walking boundaries keeps the flatten below
    // proportional to the *radius* rather than to the length of the stroke.
    let mut from = 0;
    for k in (0..last).rev() {
        let cut = if k == 0 {
            rec.path[0].pos
        } else {
            crate::path::span_end(&rec.path, k - 1)
        };
        if (tip - cut).length() >= radius {
            from = k;
            break;
        }
    }
    let pts = crate::path::flatten_spans(&rec.path, from..last, 0.0, tol);
    let Some(tail) = pts.last() else {
        return fallback();
    };
    // Back one radius of **travel**, not of displacement — the window is an extent's
    // worth of path, and a stroke that curls back on itself still spent that path. The
    // polyline carries its own arc-length accumulator, so this is a comparison rather
    // than a second summation.
    let back = pts
        .iter()
        .rev()
        .find(|p| tail.dist - p.dist >= radius)
        .unwrap_or(&pts[0]);
    let v = tail.pos - back.pos;
    let len = v.length();
    if len > 1e-4 {
        v / len
    } else {
        // A stroke with no travel at all — every knot on one spot. There is no
        // direction in the path to find, so the dab's stands.
        fallback()
    }
}

#[cfg(test)]
mod tests {
    use super::super::bleed::bleed_fires;
    use super::*;
    use crate::gpu::stroke::StrokeSpans;
    use crate::gpu::stroke::budget::flatten_budget;
    use crate::gpu::stroke::region::cover;
    use crate::gpu::stroke::segments::generate_segments_in;
    use crate::gpu::stroke::segments::testing::{record, run, seg, seg_between, smearing};
    use stark_model::geom::Vec2;

    /// The coarse deposit scans the grid [`Rect::scanned`] rounds to.
    ///
    /// Every dispatch divides its extent by its own kernel's declaration (§6.10) and
    /// needs nothing said here. What is left is the **inverse**: [`cell_geometry`]
    /// counts the cells `deposit_coarse` can name over the rect *as that kernel rounds
    /// it up*, so the rounding has to be the one it declares. Device-free.
    #[test]
    fn the_coarse_deposit_scans_the_grid_the_cells_are_counted_over() {
        assert_eq!(
            stark_shaders::dynamics(stark_shaders::Resid::Without)
                .deposit_coarse
                .workgroup_size,
            [TILE_WG, TILE_WG, 1],
            "`cell_geometry` rounds a rect by `TILE_WG`, which `deposit_coarse` no \
             longer declares",
        );
    }

    // --- the neutral slot ---------------------------------------------------

    /// [`SlotCommon::swept`] shares the frame's `radius_ramp` across all three slot
    /// kinds it serves, which is exact only because a **bleed window does not ramp**: a
    /// firing diffuses at the one radius [`bleed_stencil`] solved its reach and rate
    /// against, and a rim moving under the sweep would diffuse at a width that stencil
    /// was not built for (§6.2).
    ///
    /// Stated here because the sharing is silent if it ever stops being true — a bleed
    /// slot would start carrying its crossing segment's ramp and nothing would say so.
    #[test]
    fn a_bleed_window_never_ramps() {
        let mut segments = run(200, 0.2, 30.0);
        for (i, s) in segments.iter_mut().enumerate() {
            s.sweep.radius_ramp = 0.4 - 0.001 * i as f32;
        }
        let (fires, _) = bleed_fires(0.5, &segments);
        assert!(!fires.is_empty(), "no firing to check");
        assert!(
            fires.iter().all(|f| f.window.radius_ramp == 0.0),
            "a firing's window ramps, so a bleed slot no longer shares the swept frame",
        );
    }

    /// [`NEUTRAL`] is neutral *in the shader's terms*, which for seven members is not
    /// zero. Six are **scales**, where a zeroed lane says "none of the thing it
    /// multiplies": a `tooth_bearing` of 0 is infinite tooth rather than absent tooth,
    /// a `cell_px` of 0 no deposit grid rather than the exact one, a zeroed `stretch` a
    /// tip of no width every prefix difference is divided by (§6.6), a zeroed `opacity`
    /// or `opacity_mod` the strongest ceiling there is where 1 is the identity branch
    /// (§6.2). The seventh, `tooth_give`, runs the other way
    /// (`BrushParams::tooth_give`): zeroed it is the driest tip there is.
    ///
    /// The spelled-out const is what makes a new member raise a compile error rather
    /// than take a zero; this is what says the zero would have been wrong.
    #[test]
    fn the_neutral_stamp_is_neutral_rather_than_zeroed() {
        let d = NEUTRAL;
        assert_eq!(d.tooth_bearing, 1.0, "the neutral bearing must be 1, not 0");
        assert_eq!(d.cell_px, 1, "the neutral cell must be 1 — exact — not 0");
        assert_eq!(
            d.tooth_give, 1.0,
            "the neutral give must be 1 — full contact — not 0, which is the dry mark"
        );
        assert_eq!(
            d.stretch,
            [1.0, 0.0, 1.0],
            "the neutral stretch must be the identity map, not zeroes"
        );
        assert_eq!(
            d.opacity, 1.0,
            "the neutral opacity must be 1 — the identity ceiling — not 0"
        );
        assert_eq!(
            d.opacity_mod, 1.0,
            "the neutral ceiling factor must be 1 — the pen taking nothing — not 0"
        );
        // The substrate pitch's denominator is a divisor: `TexelsPerPx::NONE` gates on
        // its zero numerator, and keeps 1 below it so a missed guard cannot divide by 0.
        assert_eq!(
            d.substrate_den, 1,
            "the neutral substrate pitch must be `NONE`"
        );
        // And everything else is zero, which for the rest of the stamp *is* neutral —
        // stated as the complement of the members above so a new one has to be
        // classified rather than silently joining whichever list it was written near.
        let z = Stamp {
            tooth_bearing: 1.0,
            cell_px: 1,
            tooth_give: 1.0,
            stretch: [1.0, 0.0, 1.0],
            opacity: 1.0,
            opacity_mod: 1.0,
            substrate_den: 1,
            ..Default::default()
        };
        assert_eq!(
            bytemuck::bytes_of(&d),
            bytemuck::bytes_of(&z),
            "a member of the neutral stamp is neither zero nor one of the values above",
        );
    }

    /// **The cell grid is anchored to the canvas, not to the region** (§6.4). Where a
    /// cell boundary falls must be a property of the canvas position and the brush
    /// alone: region origins differ per piece and per live fold, and a grid that moved
    /// with them would break tile aprons against neighbour interiors and
    /// `preview == committed` in one stroke. This replays the shader's own index
    /// arithmetic against [`cell_geometry`]'s anchor for a spread of origins.
    #[test]
    fn cell_boundaries_are_canvas_anchored_whatever_the_region_origin() {
        let rect = Rect {
            origin: [0; 2],
            w: 64,
            h: 64,
        };
        let boundaries = |origin: f32| -> Vec<i32> {
            let grid = cell_geometry(5, Vec2::new(origin, origin), &rect, 64);
            assert!(
                grid.hoist.is_some(),
                "a cell of 5 must take the coarse path"
            );
            let (c, a) = (5i32, grid.anchor[0]);
            // The canvas texels where the shader's cell index steps: its region
            // texel is `canvas − origin`, its cell `floor((rt + anchor) / c)`.
            let idx = |canvas: i32| (canvas - origin as i32 + a).div_euclid(c);
            (-63..64).filter(|x| idx(*x) != idx(*x - 1)).collect()
        };
        let want = boundaries(0.0);
        for origin in [-640.0f32, -5.0, 3.0, 17.0, 999.0] {
            assert_eq!(
                boundaries(origin),
                want,
                "cell boundaries moved with the region origin {origin}",
            );
        }
    }

    /// **Every cell the coarse deposit taps is one the hoist wrote** (§6.2). The
    /// bilinear read names four cells per texel — `floor((rt + anchor + ½)/c − ½)`
    /// and its upper neighbours — against a scratch that starts one cell *below* the
    /// rect's first (`cell_base`), and the hoist grid has to cover both ends of that
    /// or the deposit reads whatever the previous segment left in the scratch — not a
    /// wrong color but a stale one, from another segment's tip.
    ///
    /// So this replays the shader's own tap arithmetic in f32, for every texel a
    /// dispatch can scan, and asserts the window sits inside [`cell_span`]'s count.
    #[test]
    fn the_bilinear_read_never_taps_a_cell_the_hoist_skipped() {
        let dsize = 64;
        for cell in [2u32, 3, 5, 10, 16] {
            for region_origin in [-640.0f32, -5.0, 0.0, 3.0, 17.0, 999.0] {
                for rect_origin in [0i32, 1, 7, 33] {
                    let rect = Rect {
                        origin: [rect_origin; 2],
                        w: 24,
                        h: 40,
                    };
                    let grid = cell_geometry(cell, Vec2::splat(region_origin), &rect, dsize);
                    assert!(
                        grid.hoist.is_some(),
                        "a cell above 1 must take the coarse path"
                    );
                    let (c, a) = (cell as i32, grid.anchor[0]);
                    // The shader's `cell_base`: the rect's first cell, less the border.
                    let base = (rect_origin + a).div_euclid(c) - CELL_BORDER;
                    // The texels the deposit scans: its rect as rounded to whole
                    // workgroups, clamped to the snapshot square — what `cells` counts.
                    let span = rect.scanned().0.min(dsize) as i32;
                    let count = cell_span(rect_origin + a, span, c) as i32;
                    for t in 0..span {
                        let rt = rect_origin + t;
                        // `cell_tap`'s continuous cell coordinate, in the shader's f32.
                        let u = ((rt + a) as f32 + 0.5) / cell as f32 - 0.5;
                        let lo = u.floor() as i32 - base;
                        assert!(
                            lo >= 0 && lo + 1 < count,
                            "cell {cell}, region origin {region_origin}, rect origin \
                             {rect_origin}: texel {rt} taps cells {lo}..={} of a \
                             {count}-cell hoist",
                            lo + 1,
                        );
                    }
                }
            }
        }
    }

    // --- the pen-up frame --------------------------------------------------

    /// The segments of `range` of `rec`, at the budget the loop would flatten it with.
    fn segments_of(rec: &StrokeRecord, range: std::ops::Range<usize>) -> Vec<Segment> {
        generate_segments_in(
            rec,
            flatten_budget(&rec.brush, 0.0).0,
            StrokeSpans { range, dist: 0.0 },
        )
        .0
    }

    /// The piece-local answer [`settle_tangent`] must *not* give: walk back a radius
    /// through **the segments in hand**. Kept here as the foil the test below measures
    /// against, since it is what makes a stroke's fade-out cap turn at pen-up.
    fn piece_local_tangent(segments: &[Segment], end: Vec2) -> Vec2 {
        let radius = segments.last().map_or(1.0, |s| s.sweep.radius);
        let mut back = end;
        let mut acc = 0.0;
        for s in segments.iter().rev() {
            back = s.sweep.start;
            acc += s.sweep.length;
            if acc >= radius {
                break;
            }
        }
        let v = end - back;
        let len = v.length();
        if len > 1e-4 {
            v / len
        } else {
            segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.sweep.dir)
        }
    }

    /// **The claim [`settle_tangent`] is built on**: the pen-up frame is a function of
    /// the record, not of where the renderer happened to cut the stroke into ranges and
    /// pieces.
    ///
    /// A live tail starts at a span boundary while the commit renders the whole stroke
    /// from zero, and both run the settle. A frame measured over "the segments in
    /// hand" therefore spans a shorter window for the tail, and on a curving stroke
    /// that is a different direction — a visible turn of the fade-out cap at pen-up,
    /// which is a `preview == committed` break (§1.3) in the one place it cannot be
    /// repainted.
    ///
    /// Checked at every cut point, since the interesting ones leave the tail shorter
    /// than the tip being settled.
    #[test]
    fn the_settle_frame_does_not_depend_on_where_the_stroke_was_cut() {
        // A circle of radius 200 under a 60 px tip: an extent's worth of path turns
        // through 60/200 ≈ 17°, so a lookback that comes up short points somewhere
        // measurably different.
        let curve: Vec<Vec2> = (0..=16)
            .map(|i| {
                let t = i as f32 / 16.0 * 1.2;
                Vec2::new(200.0 * t.sin(), 200.0 * (1.0 - t.cos()))
            })
            .collect();
        let rec = record(smearing(60.0), &curve);
        let all = crate::path::span_count(rec.path.len());
        let whole = segments_of(&rec, 0..all);
        let want = settle_tangent(&rec, flatten_budget(&rec.brush, 0.0).0, &whole);

        let mut ever_differed = false;
        for cut in 1..all {
            let tail = segments_of(&rec, cut..all);
            if tail.is_empty() {
                continue;
            }
            let got = settle_tangent(&rec, flatten_budget(&rec.brush, 0.0).0, &tail);
            assert!(
                (got - want).length() < 1e-4,
                "cutting at span {cut} moved the settle frame from {want:?} to {got:?}",
            );
            // And the piece-local foil really does move on these cuts, so the
            // assertion above is not passing because the case is uninteresting.
            let last = whole.last().expect("segments");
            let last = &last.sweep;
            let end = crate::path::arc_at(last.start, last.dir, last.curvature, last.length).0;
            let stale = piece_local_tangent(&tail, end);
            ever_differed |= (stale - piece_local_tangent(&whole, end)).length() > 1e-2;
        }
        assert!(
            ever_differed,
            "no cut left a tail short enough to move the piece-local frame — the test \
             proves nothing",
        );
    }

    /// [`settle_tangent`] must survive the way a real pen-up arrives: a hand pauses
    /// before it lifts, so the last samples cluster at one point and the flattener
    /// turns them into edges whose chord is a rounding error and whose direction is
    /// therefore arbitrary.
    ///
    /// Reading the record's own polyline is what makes that structural rather than a
    /// rule to remember: knots piled on one spot contribute nothing to the chord over
    /// the last radius, so they cannot steer it whatever direction the fitter gave
    /// the edges between them.
    #[test]
    fn the_settle_frame_ignores_a_paused_hands_arbitrary_last_edges() {
        // A straight drag along +y, then the pause: four knots on the stopping point.
        let mut pts: Vec<Vec2> = (0..20).map(|i| Vec2::new(0.0, i as f32 * 2.0)).collect();
        let stop = Vec2::new(0.0, 38.0);
        pts.extend([stop; 4]);
        let rec = record(smearing(12.0), &pts);
        let all = crate::path::span_count(rec.path.len());
        let segs = segments_of(&rec, 0..all);

        let tan = settle_tangent(&rec, flatten_budget(&rec.brush, 0.0).0, &segs);
        assert!(
            (tan - Vec2::new(0.0, 1.0)).length() < 1e-2,
            "the settle frame followed the paused hand: {tan:?}"
        );
    }

    /// A click has no travel, and — since the `DAB_TRAVEL` dwell was retired — no
    /// segments either: nothing is exchanged, so there is nothing for a settle to
    /// have a frame for. The paused-hand test above is where the settle frame's
    /// real mechanism lives.
    #[test]
    fn a_click_exchanges_nothing_to_settle() {
        let rec = record(smearing(10.0), &[Vec2::new(4.0, -7.0)]);
        let segs = segments_of(&rec, 0..crate::path::span_count(rec.path.len()));
        assert!(segs.is_empty(), "a click swept segments with no travel");
    }

    // --- the dispatch rects fit the scratch --------------------------------

    /// **Every dispatch grid a piece builds fits the snapshot scratch that piece
    /// sized.**
    ///
    /// That a *rect* fits is no claim at all: [`snapshot_square`] takes the maximum of
    /// the very rects the plan dispatches. The claim the shaders' bounds checks rest
    /// on is one step further out — a dispatch is rounded **up to whole 8×8
    /// workgroups**, so the texels a slot scans reach `groups·8`, past its own rect,
    /// and stay inside the scratch only because [`SNAPSHOT_QUANTUM`] is itself a
    /// multiple of 8.
    ///
    /// The stress is in the fractional origins: a rect floors its origin and rounds
    /// its far edge outward, so the worst case straddles texel boundaries at both
    /// ends. Curvature is swept too, since a bent sweep bows a sagitta out of its box.
    #[test]
    fn every_dispatch_grid_fits_the_scratch_its_piece_sized() {
        for &radius in &[0.5f32, 1.0, 7.3, 40.0, 120.0] {
            for &length in &[0.0f32, 0.37, 4.0, 60.0] {
                for &kappa in &[0.0f32, 0.004, -0.02] {
                    for &frac in &[0.0f32, 0.499, 0.5, 0.999] {
                        let start = Vec2::new(frac, -frac);
                        let mut s = seg(start, Vec2::new(1.0, 0.0), length, radius, 0.0);
                        // No flattened edge turns further than `2·asin(0.2)` ≈ 0.40
                        // (`path::fit_arc`), and `path::arc_at` is defined only near that.
                        s.sweep.curvature = kappa.clamp(-0.4 / length, 0.4 / length);
                        let segments = [s];
                        // With the pen-up, whose square is built from the tip alone
                        // rather than from the swept box.
                        let sources = slot_sources(&segments, &[], 0, true);
                        // Region origins that put the box at both ends of the region.
                        let (lo, _) = coverage_bounds(&s.sweep);
                        for &origin in &[Vec2::ZERO, lo.floor(), Vec2::new(-13.7, 91.2)] {
                            let rects = rects_for(&sources, origin);
                            let dsize = snapshot_square(&rects);
                            for r in &rects {
                                let (sx, sy) = r.scanned();
                                assert!(
                                    sx <= dsize && sy <= dsize,
                                    "a {}x{} rect scans {sx}x{sy} texels of a {dsize} \
                                     scratch (radius {radius}, length {length}, \
                                     curvature {kappa}, frac {frac})",
                                    r.w,
                                    r.h,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// A bleed window can be the largest extent in its piece — it sweeps up to a
    /// quarter-radius where the piece's own segments may be sub-pixel — so the scratch
    /// has to be sized with the firings in it, not just the segments. The maximum is
    /// only over what the walk collected, so a walk that forgot the firings would size
    /// the scratch short and go on asserting nothing. Asked of [`slot_sources`], which
    /// is the walk the plan itself runs.
    #[test]
    fn the_scratch_is_sized_with_the_bleed_windows_in_it() {
        // Long enough to cross the 0.25 · 30 px cadence, cut far finer than it.
        let segments = run(200, 0.2, 30.0);
        let (fires, _) = bleed_fires(0.5, &segments);
        assert!(!fires.is_empty(), "no firing to size against");
        // The widest rect, **before** [`SNAPSHOT_QUANTUM`] rounds it. Asked of the raw
        // maximum on purpose: the round-up is a pool-hit concession, and on this case
        // it is generous enough to swallow the difference and let a walk that forgot
        // the firings pass. What is being pinned is that the walk collects them.
        let widest = |f: &[BleedFire]| {
            rects_for(&slot_sources(&segments, f, 0, false), Vec2::ZERO)
                .iter()
                .fold(0, |m, r| m.max(r.w).max(r.h))
        };
        let (with, without) = (widest(&fires), widest(&[]));
        assert!(
            with > without,
            "a firing's window did not widen the scratch ({without} -> {with})"
        );
        // And the square the piece allocates holds the wider of the two.
        assert!(
            snapshot_square(&rects_for(
                &slot_sources(&segments, &fires, 0, false),
                Vec2::ZERO
            )) >= with,
        );
    }

    /// **The bleed ladder's no-flux wall at the region edge, as region arithmetic.**
    ///
    /// A ladder tap moves height only where the pair's mobility `min(w_t, w_n)` is
    /// nonzero, and a `w` is nonzero only where the swept exposure is — inside the
    /// tip's disc about the arc, the box [`coverage_bounds`] measures. So the partner
    /// of every flux is a texel whose own `deposit` thread ran *exactly when* that box
    /// sits inside the region: a sweep reaching past it would shed height into a thread
    /// that returned at the region bounds check, and §6.1's conservation would go with
    /// it. `bleed_weight` does not test the region, and does not need to.
    ///
    /// The rect origin's own non-negativity is not here: it follows from
    /// [`RECT_MARGIN`]'s bound, stated where that constant is.
    #[test]
    fn a_sweeps_coverage_box_sits_inside_the_region_dispatched_over_it() {
        let cases: Vec<(&str, Vec<Segment>)> = vec![
            (
                "a dot",
                vec![seg_between(
                    Vec2::new(10.0, 10.0),
                    Vec2::new(10.5, 10.0),
                    4.0,
                )],
            ),
            (
                "a sub-pixel drag, cut the way a slow hand is fitted",
                run(40, 0.39, 22.0),
            ),
            (
                "a fat tip, whose reach runs past its own endpoints",
                vec![seg_between(
                    Vec2::new(500.3, 500.7),
                    Vec2::new(505.0, 500.0),
                    120.0,
                )],
            ),
            (
                "across the origin, into negative tiles",
                vec![seg_between(
                    Vec2::new(-300.4, -140.6),
                    Vec2::new(220.0, 90.0),
                    37.0,
                )],
            ),
            ("a long run, several tiles wide", run(60, 9.0, 30.0)),
        ];
        // The margin is a **worst case over where the sweep falls in its tile** — a box
        // sitting well inside one has margin to spare — so every case is walked a whole
        // tile period, off the texel grid at both ends of it.
        let shifts: Vec<f32> = (0..stark_model::geom::TILE_SIZE)
            .step_by(13)
            .flat_map(|i| [i as f32, i as f32 + 0.499])
            .collect();
        for (what, segments) in cases {
            for shift in &shifts {
                // Bent too, so a sagitta bows out of every box.
                for (bend, kappa) in [("straight", 0.0f32), ("bent", 0.004)] {
                    let segments: Vec<Segment> = segments
                        .iter()
                        .map(|s| {
                            let mut s = *s;
                            // Held to a turn a flattened edge can take (`path::fit_arc`).
                            s.sweep.curvature = kappa.min(0.4 / s.sweep.length);
                            s.sweep.start += Vec2::new(*shift, -*shift);
                            s
                        })
                        .collect();
                    let (fires, _) = bleed_fires(0.5, &segments);
                    let rect = cover(&segments, &fires)
                        .rect()
                        .expect("a piece is always a region");
                    let sources = slot_sources(&segments, &fires, 0, true);
                    // The last texel index each axis admits: `deposit` returns at
                    // `rt >= rdim`, and `bleed_weight`'s `t` runs the snapshot square.
                    let last = Vec2::new(rect.w as f32 - 1.0, rect.h as f32 - 1.0);
                    for src in &sources {
                        let (lo, hi) = src.bounds();
                        let (lo, hi) = (lo - rect.origin, hi - rect.origin);
                        assert!(
                            lo.x >= 0.0 && lo.y >= 0.0 && hi.x <= last.x && hi.y <= last.y,
                            "{what} ({bend}, shifted {shift}): a slot's coverage box \
                             spans {lo:?}..{hi:?} of a {}x{} region — flux can reach a \
                             texel whose own thread returned",
                            rect.w,
                            rect.h,
                        );
                    }
                }
            }
        }
    }
}
