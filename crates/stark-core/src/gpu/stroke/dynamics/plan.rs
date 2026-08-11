//! The dispatch plan: what the loop is asked to do, worked out entirely on the CPU
//! (§6.2).
//!
//! **Nothing here names `wgpu`.** It is float arithmetic over a [`StrokeRecord`] and
//! the [`Segment`]s it was flattened into, producing [`Stamp`] slots and the workgroup
//! counts that go with them — the same virtue `budget.rs` claims for itself, and for
//! the same payoff: the properties that actually matter here are testable without an
//! adapter. Which windows the bleed cadence fires, that they are independent of where
//! the stroke was cut, that a rect fits the scratch its piece sized, that every named
//! field lands in the lane the shader reads it from — all of it is pinned below, on
//! any machine.
//!
//! The one GPU *type* that reaches in is the canvas [`Surface`](crate::gpu::surface),
//! and only for `bearing` — plain CPU arithmetic over the ground's statistics, which
//! happens to live on the struct that also owns its texture.
//!
//! Every slot is a pure function of the record and the piece's own geometry, in plain
//! CPU float math, so replay is deterministic (§12.1).

use crate::document::StrokeRecord;
use crate::geom::Vec2;

use super::super::budget::{
    BLEED_TRAVEL_QUANTUM, MAX_BLEED_FIRES_PER_SEGMENT, TAU_PER_PASS, bleed_stencil, footprint_cell,
};
use super::super::segments::{Segment, coverage_bounds};
// The `Stamp` uniform, generated from `dynamics.wesl`'s own declaration at build
// time (`stark-shaders/build/mirror.rs`) — lanes, offsets, and the documentation of
// what each lane holds, which is now on the generated fields.
//
// It used to be written out here as well, nine `[f32; 4]` fields against the shader's
// nine `vec4`s, with a second copy of the lane map in the doc comments and nothing
// checking either half. Both halves drifted: this one still described `e.zw` as the
// midpoint `exchange` samples the canvas at, some time after the shader had stopped
// reading the lane at all. The shader decides how the lanes are *read*, so it is now
// the only place they are written down.
//
// Every slot is still a pure function of the `StrokeRecord` and the piece's own
// geometry, computed in plain CPU float math, so replay is deterministic (§12.1).
use stark_shaders::mirror::dynamics::Stamp;

/// One slot's window into the stamp buffer, and the `min_binding_size` its layout
/// declares — both of which have to be `Stamp`'s own size, so they are taken from it
/// rather than written down.
pub(super) const SLOT: usize = std::mem::size_of::<Stamp>();

/// One slot of the sequential swept-exchange loop (§6.2), and the dispatches it
/// stands for.
pub(super) struct LoopDispatch {
    pub(super) slot: Stamp,
    /// Workgroup counts for the slot's footprint work — the `deposit`, and the
    /// `snapshot` that rides in `exchange`'s grid. The slot's own coverage box
    /// rather than the piece-wide worst-case square, so an axis-aligned sweep pays for
    /// the ~4·r² texels its footprint can reach instead of the ~10·r² a diagonal one
    /// might have needed.
    pub(super) groups: (u32, u32),
    /// Workgroup counts for the `cell_hoist` grid when this slot takes the **coarse
    /// deposit** (§6.2) — `Some` exactly when [`footprint_cell`] beat 1 for this
    /// segment's tip, which only a painting segment's can. `None` is the exact
    /// per-texel `deposit`, bit-for-bit the kernel every slot ran before the coarse
    /// path existed; bleed and settle slots are always `None`, so the lateral flux
    /// and the pen-up never see a cell at all.
    pub(super) cell_groups: Option<(u32, u32)>,
    pub(super) kind: SlotKind,
}

/// Which of the loop's three dispatch shapes a slot takes (§6.2).
///
/// A tag rather than a pair of flags because the three are genuinely different
/// sequences over the same uniform, and only one of them touches the tool: the
/// reservoir ping-pong advances on a [`Segment`](SlotKind::Segment) and on nothing
/// else, which is easier to see as an arm than as an early `continue` plus a tail
/// block indexed past the end of the loop.
pub(super) enum SlotKind {
    /// A stretch of painting: `bake` → `exchange` (+ `snapshot`) → `deposit`.
    Segment,
    /// A dedicated **bleed slot**: a quad whose sweep is one firing of the bleed
    /// cadence's travel window, with every vertical rate and the source zeroed.
    /// Dispatched as `snapshot` + `deposit` alone — the tool plays no part, so there
    /// is nothing to bake or exchange, and the reservoir ping-pong is left
    /// exactly where the previous segment put it.
    Bleed,
    /// The pen-up: `snapshot` → `bake` → `settle`. At most one per plan, and always
    /// its last slot — the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`).
    Settle,
}

/// The [`Stamp`] lanes every slot in a plan fills the same way, resolved once — so
/// the three slot kinds below list only what actually differs between them, which
/// is the whole of what makes a bleed slot or a settle slot readable against a
/// painting segment.
struct SlotCommon<'a> {
    /// The stroke's own constants: `c` outright, and the colour-dynamics lookup that
    /// fills `f`, `g.xyz` and `h.xy`. Borrowed rather than copied out, so a slot and
    /// the swept path's `TileXform` are demonstrably reading one resolution of them.
    k: &'a super::super::StrokeConstants,
    /// `i.yzw`: the region texel → weave map, with the piece's origin already
    /// folded into the bias. Only `i.x` — how deep this slot's tip bites — varies.
    ///
    /// The one lane here that is not a stroke constant: the bias is where the *piece*
    /// sits, which `k` cannot know.
    weave: [f32; 3],
}

impl SlotCommon<'_> {
    /// The lanes every slot fills the same way — the stroke's colour and the weave
    /// map — over the neutral value of everything a slot kind may leave alone.
    ///
    /// A slot kind then names only what it actually differs by, which is the whole of
    /// what makes a bleed or settle slot readable against a painting segment.
    fn slot(&self) -> Slot {
        Slot {
            channels: self.k.channels,
            // The other half of the same colour (§6.7) — filled wherever `channels` is,
            // and zero in a space whose three channels already say everything.
            resid: self.k.resid,
            weave_scale: self.weave[0],
            weave_bias: Vec2::new(self.weave[1], self.weave[2]),
            ..Slot::default()
        }
    }

    /// [`Self::slot`] plus the colour-dynamics jitter, for a slot that lays the
    /// brush's own `add` paint: the shared field, this slot's arc length, and the
    /// bearing fraction it books the tool's half of the transfer against.
    ///
    /// `lambda_bleed` stays 0, which is what every such slot wants: the lateral flux
    /// runs only on the dedicated bleed slots, so between firings the canvas takes the
    /// no-bleed path bit-for-bit (§6.2).
    fn painting(&self, dist: f32, bearing: f32) -> Slot {
        let (namp, noff) = (self.k.namp, self.k.noff);
        Slot {
            noise_freq: self.k.nfreq,
            noise_amp: [namp[0], namp[1], namp[2]],
            noise_off: [noff[0], noff[1]],
            dist,
            bearing,
            ..self.slot()
        }
    }
}

/// One dispatch's uniform in **named fields**, packed into [`Stamp`]'s nine `vec4`
/// lanes by [`Slot::pack`] — the one place on this side of the boundary that knows
/// which lane is which.
///
/// The lanes are `vec4`s because that is what a uniform wants, and `dynamics.wesl`
/// long ago stopped reading them as such: every consumer there goes through a named
/// accessor (`radius()`, `travel_px()`, `lift_rate()`), so the shader's lane map lives
/// beside the declaration that decides it. This is the same move on the host, and it
/// is overdue for the same reason. Three slot kinds filled nine lanes at three sites
/// with wholly different meanings per component — 108 positional floats, and nothing
/// checking one of them. The generated `offset_of` assertions pin where a *lane*
/// starts, not what lives inside it, so `lambda(lift)` and `lambda(deposit)` written
/// the wrong way round was a silent wrong picture.
///
/// That it drifts is on the record: the note above the `Stamp` import remembers a
/// host-side copy that "still described `e.zw` as the midpoint `exchange` samples the
/// canvas at, some time after the shader had stopped reading the lane at all".
///
/// [`Default`] is the neutral slot — every rate off, and `bearing` at the 1 that
/// leaves an exchange alone — so each kind below lists only what it differs by.
#[derive(Clone, Copy)]
struct Slot {
    /// The sweep's start in region px, and the unit travel tangent it leaves along.
    start: Vec2,
    dir: Vec2,
    /// The tip's radius in region px, and its travel as a multiple of that radius —
    /// 0 on a settle, which is a break of contact rather than a stretch of it.
    radius: f32,
    travel_radii: f32,
    /// `λ = ln(1 − axis) ≤ 0`, clamped away from −∞. Zero is "no transfer".
    lambda_lift: f32,
    lambda_deposit: f32,
    /// The brush's own colour channels + per-unit opacity. **Undrained**.
    channels: [f32; 4],
    /// The same colour's **residual** (§6.7) in `.xyz`; `.w` unused. Undrained like
    /// `channels`, and zero in a space that has no residual to carry.
    resid: [f32; 4],
    /// The dispatch rect's top-left in region texels, integral.
    rect_origin: Vec2,
    /// Shape orientation in turns ∈ [0, 1) — picks the prefix-τ slice (§6.6).
    orient: f32,
    /// The `drain` falloff per canvas px.
    drain: f32,
    /// The `add` source rate per unit exposure, **undrained** like the opacity.
    add: f32,
    /// Signed curvature of the sweep (1/region px); 0 is a straight one.
    curvature: f32,
    /// The bleed stencil's longest tap in texels — nonzero **only** on a bleed slot.
    bleed_reach: f32,
    /// The colour-dynamics lookup (§6.2): frequency per axis + 1/NOISE_TILE_PX,
    /// per-channel amplitude, and the per-stroke translation. All zero = no jitter.
    noise_freq: [f32; 4],
    noise_amp: [f32; 3],
    noise_off: [f32; 2],
    /// Arc length at the slot's start (canvas px) — the noise's third axis.
    dist: f32,
    /// The tooth's bearing fraction: the share of the ground a tip with this `tooth`,
    /// going this way, stands on (§6.4). What the *tool* books its half of the
    /// transfer against, having no ground of its own. 1 where there is nothing to bite.
    bearing: f32,
    /// The lateral canvas diffusion rate (≤ 0) — nonzero **only** on a bleed slot.
    lambda_bleed: f32,
    /// How little give this slot's tip has (0 = the ground gates nothing), over the
    /// region texel → weave map `uv = rt · weave_scale + weave_bias` (§6.4).
    tooth: f32,
    weave_scale: f32,
    weave_bias: Vec2,
    /// The footprint cell's edge in texels (§6.2) — 1 is the exact per-texel deposit,
    /// which is also the neutral value: the exact kernels never read the lane.
    cell: f32,
    /// The cell grid's canvas anchor: the region's canvas origin modulo the cell,
    /// so cell boundaries are pinned to canvas texels whatever region surrounds them
    /// (§6.4). Zero whenever `cell` is 1.
    cell_anchor: Vec2,
}

impl Default for Slot {
    /// Zero everywhere except `bearing`, whose neutral value is **1** — the share of
    /// the ground a tip stands on where there is nothing to bite, and what leaves an
    /// exchange alone if one runs. A zeroed bearing would silently book the tool's
    /// half of every transfer against no ground at all, which is not "no tooth" but
    /// "infinite tooth", so it is the one field a derived `Default` would get wrong.
    fn default() -> Self {
        Self {
            start: Vec2::ZERO,
            dir: Vec2::ZERO,
            radius: 0.0,
            travel_radii: 0.0,
            lambda_lift: 0.0,
            lambda_deposit: 0.0,
            channels: [0.0; 4],
            resid: [0.0; 4],
            rect_origin: Vec2::ZERO,
            orient: 0.0,
            drain: 0.0,
            add: 0.0,
            curvature: 0.0,
            bleed_reach: 0.0,
            noise_freq: [0.0; 4],
            noise_amp: [0.0; 3],
            noise_off: [0.0; 2],
            dist: 0.0,
            bearing: 1.0,
            lambda_bleed: 0.0,
            tooth: 0.0,
            weave_scale: 0.0,
            weave_bias: Vec2::ZERO,
            // 1, not 0, for the same reason the bearing is 1: the neutral cell is
            // "exact", and a slot that never reads the lane should still carry the
            // value that means what it does.
            cell: 1.0,
            cell_anchor: Vec2::ZERO,
        }
    }
}

impl Slot {
    /// The lane packing — `dynamics.wesl`'s `struct Stamp`, read the other way round.
    /// Every field above lands here exactly once.
    fn pack(&self) -> Stamp {
        Stamp {
            a: [self.start.x, self.start.y, self.dir.x, self.dir.y],
            b: [
                self.radius,
                self.travel_radii,
                self.lambda_lift,
                self.lambda_deposit,
            ],
            c: self.channels,
            d: [
                self.rect_origin.x,
                self.rect_origin.y,
                self.orient,
                self.drain,
            ],
            // `.w` is the last of the lane that carried the midpoint `exchange` used
            // to sample the canvas at; it walks the texel's own track now.
            e: [self.add, self.curvature, self.bleed_reach, 0.0],
            f: self.noise_freq,
            g: [
                self.noise_amp[0],
                self.noise_amp[1],
                self.noise_amp[2],
                self.dist,
            ],
            h: [
                self.noise_off[0],
                self.noise_off[1],
                self.bearing,
                self.lambda_bleed,
            ],
            i: [
                self.tooth,
                self.weave_scale,
                self.weave_bias.x,
                self.weave_bias.y,
            ],
            j: self.resid,
            k: [self.cell, self.cell_anchor.x, self.cell_anchor.y, 0.0],
        }
    }
}

/// What a plan is built *against*, as opposed to the segments it is built *from*:
/// where the piece's region sits, how large its snapshot scratch is, and the stroke
/// constants every slot is filled from.
///
/// Bundled because these five travel together through the plan and its rect
/// arithmetic, and because a slot's geometry is only meaningful relative to them.
pub(super) struct PlanCtx<'a> {
    pub(super) rec: &'a StrokeRecord,
    /// The budget `rec` was flattened at, handed down from [`dynamics_setup`] rather
    /// than recomputed — one place answers what a stroke's segments are. Only the
    /// pen-up frame reads it ([`settle_tangent`]), which re-flattens a footprint's
    /// worth of the record and must cut it exactly as the segments in hand were cut.
    pub(super) tol: crate::path::FlattenTolerance,
    /// The region rectangle's top-left in canvas px — what every slot's coordinates
    /// are measured from, since the shader never learns where the piece sits.
    pub(super) region_origin: Vec2,
    /// The snapshot scratch's square ([`Snapshot::size`]), which every dispatch rect
    /// is clamped to.
    pub(super) dsize: u32,
    /// Everything both render paths read off the record and the scene
    /// ([`StrokeConstants`](super::super::StrokeConstants)) — the colour a slot's `c` is, the
    /// weave map its `i` carries, and the colour-dynamics lookup for `f`–`h`.
    pub(super) consts: &'a super::super::StrokeConstants,
    pub(super) surface: &'a crate::gpu::surface::Surface,
}

/// The margin, in canvas px, a dispatch rect is grown by each side so a fragment
/// sampling just outside its own texel still lands inside the rect.
const RECT_MARGIN: f32 = 1.5;

/// The texels a coverage box `span` px across can occupy once [`dispatch_rect`] has
/// added its margin and snapped the result outward — **wherever it sits**.
///
/// This is the one number the scratch and the dispatch are related by, and it is a
/// function of the span alone precisely so that they can share it. `2·RECT_MARGIN` for
/// the margin either side; `+2` because the rect then floors its origin and rounds its
/// far edge outward, which can each cost a texel however the box falls across the
/// grid.
///
/// Monotone in `span`, which is what makes [`snapshot_size`]'s maximum a bound on
/// every rect rather than on the one that happened to be widest.
fn rect_extent(span: f32) -> u32 {
    (span + 2.0 * RECT_MARGIN).ceil() as u32 + 2
}

/// The snapshot scratch's square for a piece: large enough for any one slot's
/// [`dispatch_rect`], measured from the coverage boxes rather than bounded
/// analytically, since `coverage_bounds` is already the exact box and a curved sweep
/// has no closed-form "worst rotation" to fall back on.
///
/// The fit is now structural rather than an argument spread over four numbers: both
/// sides go through [`rect_extent`], and this takes its maximum, so no rect a piece
/// can build overruns the scratch that piece sized. It used to be `+3` and `+2` here
/// against a margin and a rounding there — an equality nothing enforced, backed by a
/// `debug_assert` that in release let a too-large rect clip into a silently truncated
/// footprint.
///
/// Split out from [`DynamicsRun::snapshot_scratch`] so that fit can be exercised
/// without a GPU (`tests`).
pub(super) fn snapshot_size(segments: &[Segment], fires: &[(usize, Segment)]) -> u32 {
    segments
        .iter()
        .chain(fires.iter().map(|(_, f)| f))
        .fold(rect_extent(1.0), |m, s| {
            let (lo, hi) = coverage_bounds(s);
            m.max(rect_extent(hi.x - lo.x))
                .max(rect_extent(hi.y - lo.y))
        })
}

/// One slot's dispatch rectangle over a canvas-space coverage box: its integral origin
/// in region texels, and the workgroup counts covering it.
///
/// The slot's own box rather than the piece-wide worst case, so an axis-aligned sweep
/// dispatches ~4·r² threads where a square would spend ~10·r². Texels the rounding adds
/// beyond the box read zero exposure and fall out of `deposit` untouched.
///
/// Every rect fits `dsize` **by construction**: its extent is at most
/// [`rect_extent`] of the box's own span, and [`snapshot_size`] took the maximum of
/// exactly that over exactly these boxes. The assertion states the relation rather
/// than defending against it — and it is a real one, because the alternative it
/// replaced (a `debug_assert` and a `min`) let a release build clip a too-large rect
/// into a silently truncated footprint, which is wrong pixels with no signal at all.
///
/// The extent is measured exactly here rather than taken from `rect_extent`, which
/// only bounds it: the bound is position-independent by design and so is a texel loose
/// wherever the box happens to land on the grid, and there is no reason to dispatch
/// that texel.
fn dispatch_rect(lo: Vec2, hi: Vec2, region_origin: Vec2, dsize: u32) -> Rect {
    let span = hi - lo;
    let lo = lo - region_origin - Vec2::splat(RECT_MARGIN);
    let hi = hi - region_origin + Vec2::splat(RECT_MARGIN);
    let origin = Vec2::new(lo.x.floor(), lo.y.floor());
    let (w, h) = (
        ((hi.x - origin.x).ceil() as u32) + 1,
        ((hi.y - origin.y).ceil() as u32) + 1,
    );
    assert!(
        w <= rect_extent(span.x) && h <= rect_extent(span.y) && w <= dsize && h <= dsize,
        "a {w}x{h} dispatch rect overruns the {dsize} snapshot scratch",
    );
    Rect {
        origin,
        groups: (w.div_ceil(8), h.div_ceil(8)),
    }
}

/// One slot's dispatch rectangle.
struct Rect {
    /// The rect's top-left in region texels, integral — the `d.xy` a slot carries.
    origin: Vec2,
    /// Workgroup counts covering it, at the shaders' 8×8.
    groups: (u32, u32),
}

/// The cell scratch's square for a piece, in cells: enough for any rect the piece's
/// snapshot scratch admits at the finest cell the coarse path runs (2), plus the
/// [`CELL_BORDER`] ring on each side. The same structural-fit move as
/// [`snapshot_size`]/[`rect_extent`]: both sides of the relation go through this
/// function — [`cell_geometry`] asserts against it, `DynamicsRun::cell_scratch`
/// allocates from it — so no slot a plan can build reads cells the scratch does not
/// hold.
pub(super) fn cell_scratch_size(dsize: u32) -> u32 {
    dsize.div_ceil(2) + 2 + 2 * CELL_BORDER
}

/// Cells of hoist beyond the rect's own, on each side, that `deposit_coarse`'s
/// bilinear read needs (§6.2): a texel in the leading half of the first cell
/// interpolates against the cell before it, and one in the trailing half of the last
/// against the cell after. The shader spends it at the low end — `cell_base` starts
/// the scratch one cell below the rect's first — so the count here covers both.
const CELL_BORDER: u32 = 1;

/// Cells the hoist must write to cover `span` texels from region texel `base`
/// (already offset by the anchor), border included — the window
/// `deposit_coarse`'s taps are asserted to sit inside.
fn cell_span(base: i32, span: i32, c: i32) -> u32 {
    ((base + span - 1).div_euclid(c) - base.div_euclid(c) + 1) as u32 + 2 * CELL_BORDER
}

/// The coarse deposit's per-slot geometry (§6.2): the cell grid's canvas anchor, and
/// the `cell_hoist` workgroup counts covering every cell the slot's texel grid can
/// touch — or `None` when the cell is 1 and the slot keeps the exact kernel.
///
/// The cells are counted over the texels the deposit will actually scan — the
/// dispatch rect *as rounded up to whole workgroups*, clamped to the snapshot
/// scratch — because a rounding texel past the rect still passes the shader's bounds
/// checks and looks its cell up; a cell it can name must be one the hoist wrote, or
/// the deposit reads whatever the previous segment left in the scratch. Each texel
/// names four, so the count carries a [`CELL_BORDER`] ring on each side.
fn cell_geometry(
    cell: u32,
    region_origin: Vec2,
    rect: &Rect,
    dsize: u32,
) -> (Vec2, Option<(u32, u32)>) {
    if cell <= 1 {
        return (Vec2::ZERO, None);
    }
    let c = cell as i32;
    // The anchor is congruence arithmetic, so it needs the origin to *be* the
    // integer it is (a tile origin less the apron): a fractional origin would put
    // the grid off the canvas texels everything above says it is on.
    debug_assert!(
        region_origin.x.fract() == 0.0 && region_origin.y.fract() == 0.0,
        "a region origin must sit on a canvas texel for the cell grid to anchor to it",
    );
    let anchor = Vec2::new(
        (region_origin.x as i32).rem_euclid(c) as f32,
        (region_origin.y as i32).rem_euclid(c) as f32,
    );
    let cells = |origin: f32, a: f32, groups: u32| -> u32 {
        cell_span(origin as i32 + a as i32, (groups * 8).min(dsize) as i32, c)
    };
    let cx = cells(rect.origin.x, anchor.x, rect.groups.0);
    let cy = cells(rect.origin.y, anchor.y, rect.groups.1);
    let fit = cell_scratch_size(dsize);
    assert!(
        cx <= fit && cy <= fit,
        "a {cx}x{cy}-cell hoist overruns the {fit}-cell scratch",
    );
    (anchor, Some((cx.div_ceil(8), cy.div_ceil(8))))
}

impl PlanCtx<'_> {
    /// [`dispatch_rect`] against this piece's origin and snapshot square.
    fn rect(&self, lo: Vec2, hi: Vec2) -> Rect {
        dispatch_rect(lo, hi, self.region_origin, self.dsize)
    }
}

/// Build the swept-exchange dispatch plan (§6.2): one `snapshot` +
/// `deposit` pair per flattened segment (the canvas-side exchange, swept through
/// the prefix-τ integral), each followed by the tool's own `exchange`.
/// λ = ln(1 − axis) makes every rate exponential in
/// exposure, so the exchange composes exactly across overlapping segment quads —
/// the continuous path integral, independent of any spacing. Pure CPU float math
/// → replay-deterministic.
///
/// Every painting dispatch is a segment: the tool exchanges once per segment rather
/// than on a cadence of its own, so there is no interval state to carry between
/// ranges. The bleed cadence and the pen-up ride as their own [`SlotKind`]s.
pub(super) fn dynamics_plan(
    ctx: &PlanCtx<'_>,
    segments: &[Segment],
    fires: &[(usize, Segment)],
    settle: bool,
) -> Vec<LoopDispatch> {
    let &PlanCtx {
        rec,
        region_origin,
        consts,
        surface,
        ..
    } = ctx;
    let b = &rec.brush;
    // The canvas → weave map, folded so the shader can go straight from its *region*
    // texel to the ground under it: `uv = rt · grain_uv + grain_bias` (§6.4). Only the
    // bias belongs to the piece — the shader never learns where the piece sits, only
    // where the weave does; the scale is a stroke constant and comes off `consts`,
    // which is what keeps it the same number the swept path writes.
    let grain_bias = region_origin * consts.grain_uv;
    // What share of the ground a tip with this tooth, going this way, stands on — per
    // segment because the tooth is modulated per segment (§6.2) and because the
    // direction is the segment's own. The canvas side of the exchange asks the ground
    // ahead of each texel; the tool has none of its own and books against this mean,
    // which is what makes a toothed smear conserve (`Surface::bearing`).
    //
    // At the segment's **midpoint** tangent, the same second-order choice `mid` is
    // sampled at below: a curved segment's canvas side reads a tangent that turns
    // across the sweep, and the midpoint is the representative of that whose error is
    // second order where either endpoint's would be first.
    let bearing = |tooth: f32, dir: Vec2| surface.bearing(tooth, dir.to_array());
    // λ = ln(1 − axis), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped clean),
    // per [`TAU_PER_PASS`] — so an axis reads as a fraction *per pass of the tip*,
    // which is what a 0..1 knob should mean, rather than per unit optical depth.
    //
    // Taken **per segment**, off the rates the segment generator resolved from the
    // pen (§6.2), rather than once for the stroke. Nothing else about the loop
    // changes: every dispatch already carried its own λs in its slot, because a
    // segment is where the exchange happens.
    let lambda = |axis: f32| (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0) / TAU_PER_PASS;
    let common = SlotCommon {
        k: consts,
        weave: [consts.grain_uv, grain_bias.x, grain_bias.y],
    };

    let mut plan = Vec::new();
    // Drained in step with the walk below, which is only correct because `bleed_fires`
    // emits them in segment order. Cheap to state, and the alternative — a firing
    // silently landing in the wrong piece of the plan — is not something a pixel would
    // show.
    debug_assert!(
        fires.is_sorted_by_key(|(after, _)| *after),
        "bleed firings must arrive in segment order",
    );
    let mut pending = fires.iter().peekable();
    for (si, s) in segments.iter().enumerate() {
        // The segment's swept exchange: the frame is (start, travel tangent at the
        // start, curvature), over the segment's own coverage box.
        let p = s.start - region_origin;
        let (clo, chi) = coverage_bounds(s);
        let rect = ctx.rect(clo, chi);
        // The tangent at the segment's **midpoint**, along the arc rather than the
        // chord: what the bearing below is read along, since a curved segment's canvas
        // side sees a heading that turns across the sweep and the midpoint is the
        // representative of that whose error is second order where either endpoint's
        // would be first.
        let (_, mid_dir) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length * 0.5);
        // The footprint cell this segment's deposit may evaluate the exchange at
        // (§6.2): a pure function of the brush shape and the segment's own radius
        // ([`footprint_cell`]), so a live tail and its commit pick the same cell —
        // and 1, the exact kernel, for every tip whose shoulder proves nothing.
        let cell = footprint_cell(&b.shape, s.radius);
        let (cell_anchor, cell_groups) = cell_geometry(cell, region_origin, &rect, ctx.dsize);
        plan.push(LoopDispatch {
            groups: rect.groups,
            cell_groups,
            kind: SlotKind::Segment,
            slot: Slot {
                start: p,
                dir: s.dir,
                radius: s.radius,
                travel_radii: s.length / s.radius,
                lambda_lift: lambda(s.lift),
                lambda_deposit: lambda(s.deposit),
                rect_origin: rect.origin,
                orient: s.orient,
                drain: b.drain,
                // The `add` source rate is passed through **unscaled**, exactly as
                // `stamp.wesl` takes it. It used to carry a gain of 2 ("tuned so
                // `add = 1` lays roughly a full-thickness deposit per pass"), which made
                // the same slider mean two different amounts of paint depending on
                // whether some *other* axis happened to be non-zero — nudging `deposit`
                // off zero doubled the flow. The tuning it claimed is already met without
                // it: a pass of the tip is `TAU_PER_PASS ≈ 6.9` of exposure, so
                // `add = 1` lays 6.9 of height, which the slab law reads as 0.999
                // coverage.
                //
                // Off the segment, since the pen can drive it (§6.2) — the same number
                // the swept path now reads off its instance.
                add: s.add,
                curvature: s.curvature,
                tooth: s.tooth,
                cell: cell as f32,
                cell_anchor,
                // No `bleed_reach` and no `lambda_bleed`: the lateral flux runs only on
                // the dedicated firings, so a painting segment takes the no-bleed path
                // bit-for-bit (§6.2). Both are `Slot::default`'s zero.
                ..common.painting(s.dist, bearing(s.tooth, mid_dir))
            }
            .pack(),
        });

        // The bleed slots that fire at this segment's end (§6.2, `bleed_fires`):
        // a quad whose sweep is the firing's travel window, with every vertical rate
        // and the source zeroed — the dispatch is the identity everywhere except the
        // lateral flux. The noise lanes are zeroed too, so the deposit skips its
        // colour-jitter taps.
        while let Some((_, fire)) = pending.next_if(|(after, _)| *after == si) {
            let p = fire.start - region_origin;
            let (clo, chi) = coverage_bounds(fire);
            let rect = ctx.rect(clo, chi);
            // The stencil this firing diffuses with: how far it reaches, and how hard
            // it relaxes to get there. Both come out of the diffusivity the axis asks
            // for — see [`bleed_stencil`], which is where the axis's whole meaning is.
            let (reach, lambda_bleed) = bleed_stencil(fire.bleed, fire.radius, fire.length);
            plan.push(LoopDispatch {
                groups: rect.groups,
                // Bleed slots keep the exact deposit whatever the tip: the ladder's
                // flux pairs need both threads of a pair to read per-texel exposures,
                // and the firings are rare and small next to the painting they cut.
                cell_groups: None,
                kind: SlotKind::Bleed,
                // Everything a painting segment carries and this does not is
                // `Slot::default`'s zero, which is what the slot *means*: λ_lift = 0 so
                // the canvas keeps everything, λ_deposit = 0 so the (uninvolved) tool
                // lays nothing, no drain because nothing is laid, no `add` because this
                // is not a stretch of painting, no tooth because there is no `add` for
                // the ground to gate, and no colour jitter — which is zeroed rather
                // than shared, so the deposit skips its noise taps entirely.
                slot: Slot {
                    start: p,
                    dir: fire.dir,
                    radius: fire.radius,
                    travel_radii: fire.length / fire.radius,
                    rect_origin: rect.origin,
                    orient: fire.orient,
                    // The window's own curvature, so the relaxed band follows the paint
                    // rather than cutting the corner off it (`bleed_fires`).
                    curvature: fire.curvature,
                    // The stencil's longest tap — the only slot that carries one.
                    bleed_reach: reach,
                    dist: fire.dist,
                    // The rate that lands this window's exposure on the blend its reach
                    // needs — not `lambda(axis)`, which is the vertical rates' mapping
                    // and would make the axis a rate rather than a diffusivity. A firing
                    // whose modulated axis has fallen to zero still dispatches: λ = 0
                    // makes it the identity, and keeping the plan a pure function of the
                    // segmentation is worth more than the dispatch it would save.
                    lambda_bleed,
                    ..common.slot()
                }
                .pack(),
            });
        }
    }

    // The pen-up (`dynamics.wesl::settle`), as one more slot on the same uniform: the
    // tip standing at the stroke's last point with **zero travel**, which is what makes
    // the shared `segment_frame`/`outside_sweep` reduce to the tip's own footprint and
    // `snapshot` copy exactly the texels the settle will write. Everything the settle
    // reads is already here — the frame, the radius, the two λs and the orientation —
    // so it costs a slot rather than a second uniform.
    if let Some(s) = settle.then(|| segments.last()).flatten() {
        let (end, _) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length);
        // The frame comes off the *record* — see [`settle_tangent`]. `segments` is one
        // piece of one range, and a lookback that walked it would stop wherever that
        // cut fell, so a live tail and its commit would settle at different angles.
        let tan = settle_tangent(rec, ctx.tol, segments);
        let p = end - region_origin;
        // The tip's own square rather than a swept box — a pen-up is a standing tip.
        // Its half-extent is the tip's `reach`, which is the radius only for a shape
        // that stays inside its own disc (`segments::tip_reach`); the settle writes the
        // same footprint the pass was laying, corners included. It cannot overrun the
        // snapshot scratch: that was sized from coverage boxes, and a segment's box is
        // this square grown by its travel.
        let rect = ctx.rect(end - Vec2::splat(s.reach), end + Vec2::splat(s.reach));
        plan.push(LoopDispatch {
            groups: rect.groups,
            // The settle is one dispatch at the end of a stroke — nothing to amortize
            // — and its p-norm handover is exactly the smooth structure a cell would
            // staircase, so it stays exact whatever the tip.
            cell_groups: None,
            kind: SlotKind::Settle,
            slot: Slot {
                start: p,
                dir: tan,
                // No travel: a pen-up is a break of contact, not a stretch of it. The
                // rates are the *last* segment's, which is where the pen was when it
                // left the page — the same segment this slot takes its radius and
                // orientation from. (`travel_radii` stays at its default 0.)
                radius: s.radius,
                lambda_lift: lambda(s.lift),
                lambda_deposit: lambda(s.deposit),
                rect_origin: rect.origin,
                orient: s.orient,
                drain: b.drain,
                // The last segment's tooth: the settle delivers what the pass still
                // owed, and it owes it through the same ground the pass was laying
                // through. What the valleys do not take stays on the tool, which is
                // discarded — a knife lifted off a canvas keeps what it did not
                // reach (§6.4).
                tooth: s.tooth,
                // No `add`: the source is a rate per unit of travel, and there is none.
                // No curvature, for the same reason — the frame is a standing tip. No
                // bleed reach: a settle is not a firing. And no λ_bleed either — that
                // axis carries no reservoir, every firing having applied its window as
                // the tip passed, so a break of contact strands nothing for a settle to
                // finish, unlike the vertical transfer whose in-flight half lives on the
                // tool. All four are `Slot::default`'s zero.
                //
                // The bearing is the neutral 1: the tool is not written back at pen-up,
                // so nothing reads it — the settle's own gate is per texel, from the
                // weave. The colour channels are filled consistently with a segment slot
                // rather than left as junk, though the settle lays the tool's *carried*
                // paint and so reads none of them.
                ..common.painting(s.dist + s.length, 1.0)
            }
            .pack(),
        });
    }
    plan
}

/// The bleed cadence (§6.2): one dedicated **bleed slot** per crossing of
/// [`BLEED_TRAVEL_QUANTUM`] of absolute arc, as `(after, window)` pairs — the index
/// of the piece segment the firing follows, and a synthetic segment whose sweep is
/// the firing's travel window: **exactly one quantum** of path, bending the way the
/// crossing segment bends. A segment that crosses the cadence twice fires twice, and
/// the two windows tile back from its end rather than merging into one.
///
/// **One quantum per firing is what makes the axis a diffusivity** rather than a
/// number that means less the faster the hand moves. A window asks the stencil for
/// `σ² ∝ its own travel`, and what one firing can carry is `2·Σ(share·d²)` — a
/// property of the stencil, flat in the travel. So a merged N-quantum window asks for
/// N times what a firing can give and is clamped back to roughly `1/N` of it
/// ([`bleed_stencil`]). That is not the exotic case: a segment at the travel cap
/// crosses a half-radius cadence twice, so an ordinary fast stroke was already
/// diffusing a tenth short before this fired per crossing. Variance adds linearly in
/// travel across firings, so N of them deliver N quanta's worth exactly — more steps,
/// not bigger ones, as in any explicit diffusion solver.
///
/// Counted off the **absolute** arc, so the firings, and the windows they sweep,
/// are a pure function of the record, independent of how the path was cut (§6.2,
/// live == committed). Why the lateral flux cannot simply ride the painting segments is a
/// numeric story told at the shader (`dynamics.wesl`, the bleed-slot note): on real
/// slow input the fitter emits sub-pixel segments, whose per-texel exposure is
/// prefix-cancellation noise and whose per-segment fluxes sit under the f16 ULP of
/// the heights they edit — measured as a 20-level directional ghost on a 177-knot
/// repro. A half-radius window has neither problem.
///
/// Each window is an **arc**, not the chord across one. At this cadence the two are
/// a fraction of a texel apart, so this is not a correction — it is that a window
/// *is* a stretch of the path, and a representation that says so cannot be wrong at
/// whatever cadence some later tuning picks. Its start is walked **back along the
/// crossing segment's own arc** rather than looked up among the segments in hand, so a
/// window is never truncated by where the range being drawn happens to begin — see the
/// note at the walk itself for what that truncation cost.
pub(super) fn bleed_fires(bleed: f32, segments: &[Segment]) -> Vec<(usize, Segment)> {
    let mut fires = Vec::new();
    // The brush's own axis, so *which* windows fire stays a function of the geometry
    // and the brush alone; how hard each one relaxes is the pen's business, and comes
    // off the crossing segment below. A brush at zero bleed can be modulated nowhere
    // above zero, so this early-out is exact (`document::Modulation`).
    if bleed <= 0.0 {
        return fires;
    }
    for (i, s) in segments.iter().enumerate() {
        let bq = BLEED_TRAVEL_QUANTUM * s.radius;
        // Before the division, not after it. A tip with no width sweeps nothing and has
        // nothing to relax, and asking how many quanta fit in it first made `crossings`
        // a NaN that only fell through by the grace of `NaN < 1.0` being false.
        // `generate_segments_in` floors the radius at 0.5, so no real segment reaches
        // here — which is the reason to state the guard plainly rather than lean on the
        // ordering of two comparisons.
        if bq <= 1e-3 {
            continue;
        }
        let crossings = ((s.dist + s.length) / bq).floor() - (s.dist / bq).floor();
        if crossings < 1.0 {
            continue;
        }
        // Capped so a plan stays bounded. `crossings` is the segment's travel over its
        // *own* radius' quantum, and those two are priced apart: the flattener buys
        // segment length off the brush's nominal radius while the cadence is the
        // modulated one, so a pen thinning the tip drives the count up without
        // shortening anything. Sixteen covers a tip down to a quarter of the brush;
        // under that the axis under-delivers, on a tip carrying almost no paint to
        // spread. Without a cap this is a memory blow-up on a degenerate stroke, which
        // is a worse failure than a gentle one.
        let crossings = (crossings as usize).min(MAX_BLEED_FIRES_PER_SEGMENT);
        let (end, end_dir) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length);
        // Walked **back along the crossing segment's own arc**, rather than looked up
        // in the segments this piece happens to hold. Reversing an arc is negating
        // both its direction and its curvature, so this is the same circle traced the
        // other way and is exact for any path the segment itself describes.
        //
        // It is history-free, and that is the point. Looking the position up meant
        // clamping to the first segment in hand, so a window reaching further back
        // than the range being drawn came out short — and a live tail always starts at
        // a span boundary while the commit renders the whole stroke from zero, so the
        // two relaxed different amounts of paint at exactly that seam. That is a
        // `preview == committed` break (§1.3), in the one place it cannot be
        // repainted, and it was visible: a bleeding stroke lightened when the pointer
        // came up.
        //
        // What it costs is extrapolating one segment's curvature over the window,
        // where the old form used the true path — the same bend for the whole span
        // rather than each segment's own. Bounded by
        // [`MAX_TIP_TURN`](super::budget::MAX_TIP_TURN), which caps how far the tip's
        // curvature may move at all, and the window is the arc that extrapolation
        // describes rather than a chord across it, so nothing else is given up on top.
        //
        // Emitted oldest first: the firings tile back from the segment's end, but they
        // edit the canvas in sequence and paint laid earlier should relax first.
        for n in (0..crossings).rev() {
            let back = (n + 1) as f32 * bq;
            let (start, back_dir) = crate::path::arc_at(end, end_dir * -1.0, -s.curvature, back);
            fires.push((
                i,
                Segment {
                    start,
                    // The reversed walk arrives pointing back the way it came, so the
                    // window's own heading is its negation — the tangent the path had
                    // at `start`, which is where the arc below is measured from.
                    dir: back_dir * -1.0,
                    // **The window bends with the path it stands for.** Its two
                    // endpoints were always on the arc; carrying the curvature is what
                    // puts the sweep between them there too. At this cadence a chord
                    // would sit `span²·κ/8` off the paint, which the tip covers many
                    // times over — so this is not a correction, it is that a window
                    // *is* a stretch of the path and nothing is gained by representing
                    // it as something else. Nothing downstream needs telling:
                    // `coverage_bounds` already grows a box by the sagitta, and
                    // `deposit` sweeps an arc for every painting segment by unrolling
                    // the annulus (`stamp_common::sweep_at`) — a bleed slot just takes
                    // the same path. The unroll's own error is `radius·|curvature|/2`,
                    // which the window inherits from the crossing segment and the
                    // flattener has capped
                    // ([`MAX_TIP_TURN`](super::budget::MAX_TIP_TURN)).
                    curvature: s.curvature,
                    radius: s.radius,
                    // The crossing segment's shape is the window's shape — a firing is
                    // that segment relaxing its own footprint, so it reaches exactly as
                    // far from the centreline.
                    reach: s.reach,
                    // One quantum of arc length, which is what `sweep_at` measures
                    // travel in — and what `bleed_stencil` is calibrated against.
                    length: bq,
                    orient: s.orient,
                    dist: s.dist + s.length - back,
                    // The window inherits the crossing segment's rates: it is that
                    // segment's own firing, and `bleed` is the only one it will use —
                    // every other axis is zeroed in the slot it becomes. Reading them
                    // from one point of the window is the cadence's usual
                    // approximation about the radius it fires at.
                    add: s.add,
                    lift: s.lift,
                    deposit: s.deposit,
                    bleed: s.bleed,
                    tooth: s.tooth,
                },
            ));
        }
    }
    fires
}

/// The travel direction the pen-up settle measures `owed` and `received` along: the
/// chord over the **last footprint's worth of path**, rather than the last segment's
/// own tangent.
///
/// The last segment's tangent cannot be trusted, and the reason is a property of real
/// input rather than a rare edge case. A hand pauses before it lifts, so a pen-up
/// arrives as a cluster of samples at almost one point; the fitter turns that into
/// spans of no length, and the flattener into edges whose chord is a rounding error
/// and whose direction is therefore arbitrary — measured on a straight drag down, the
/// final edges came out at 0°, −90°, 90° and 180° against a stroke running at 90°.
///
/// Nothing else in the loop notices: a segment of no length deposits nothing, so its
/// direction never reaches a pixel. The settle is the exception, because it takes a
/// whole tip's worth of exchange from that one frame — and its `min(owed, received)`
/// lens is elongated *along* it, so a wrong direction lands a tip-shaped disc across
/// the stroke instead of along it, at a different angle every time the hand pauses
/// differently. That is exactly what it looked like: a fade-out cap whose orientation
/// wandered from stroke to stroke, and worse the higher `lift` and `deposit` were.
///
/// One radius is the natural window because it is the extent of the thing being
/// settled — the tip's own footprint — so this is the direction the tip was travelling
/// over precisely the stretch of canvas the settle acts on, and no new constant.
///
/// **Measured on the record, not on the segments in hand**, and that is the whole of
/// why it takes a `rec`. The slice a plan is built from is one *piece* of one *range*
/// — `chunk_segments`'s cut of what `render_range` was asked for — so a lookback that
/// walked it stopped at whichever boundary came first. A live tail always starts at a
/// span boundary while the commit renders the whole stroke from zero, so a tail
/// carrying less than a radius of travel measured its frame over a shorter window than
/// the commit measured the same frame over, and on a curving stroke the two came out
/// pointing different ways: the fade-out cap turned as the pointer came up, which is a
/// `preview == committed` break (§1.3) in the one place it cannot be repainted.
///
/// This is the same defect [`bleed_fires`] was fixed for, and the cure is the same in
/// spirit — ask the record rather than the range — but not in mechanism. Walking back
/// along the last segment's own arc, as a firing's window does, is exactly what this
/// function exists to avoid: the last segments *are* the degenerate ones. So it walks
/// the curve's own polyline instead, flattening only the trailing spans a radius
/// reaches back over (`span_end` prices a span boundary without subdividing anything,
/// so finding them costs no polyline).
///
/// `segments` is still read, for the two things only it knows: the radius of the tip
/// being settled, and a click's frame — a lone control point is not a curve, and
/// `generate_segments_in` gives its dab a real direction where the path has none.
fn settle_tangent(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    segments: &[Segment],
) -> Vec2 {
    let radius = segments.last().map_or(1.0, |s| s.radius);
    // A click's frame: the dab's own direction, which is deliberate rather than
    // fitted. `generate_segments_in` sweeps it symmetrically about the point pressed,
    // so which direction it is cannot matter — but it has to be *a* direction.
    let fallback = || segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.dir);
    let last = crate::path::span_count(rec.path.len());
    if last == 0 {
        return fallback();
    }
    let tip = crate::path::span_end(&rec.path, last - 1);
    // The first span boundary a radius or more back from the tip, measured on chords
    // — which under-estimate arc length, so the span this admits genuinely holds a
    // footprint's worth of path behind it. Walking boundaries rather than the polyline
    // is what keeps the flatten below proportional to the *radius* instead of to the
    // length of the stroke.
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
    // Back one radius of **travel**, not of displacement — the window is a footprint's
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
    use super::*;
    use crate::geom::Vec2;
    use crate::gpu::stroke::StrokeSpans;
    use crate::gpu::stroke::budget::flatten_tolerance;
    use crate::gpu::stroke::segments::generate_segments_in;

    /// A straight segment of `length` from `start` along `dir`, at arc length `dist`.
    /// The plan builders read the frame, the radius and the arc clock; the paint rates
    /// are left at zero except where a test sets one, so a value that mattered would
    /// have to be given deliberately.
    fn seg(start: Vec2, dir: Vec2, length: f32, radius: f32, dist: f32) -> Segment {
        Segment {
            start,
            dir,
            curvature: 0.0,
            radius,
            // A round tip's reach: the plan builders are being measured here, not the
            // width of any one shape.
            reach: radius,
            length,
            orient: 0.0,
            dist,
            add: 0.0,
            lift: 0.0,
            deposit: 0.0,
            bleed: 0.0,
            tooth: 0.0,
        }
    }

    /// `n` straight segments of `len` each, running +x from the origin — a stroke cut
    /// the way the flattener would cut a steady drag.
    fn run(n: usize, len: f32, radius: f32) -> Vec<Segment> {
        (0..n)
            .map(|i| {
                let d = i as f32 * len;
                seg(Vec2::new(d, 0.0), Vec2::new(1.0, 0.0), len, radius, d)
            })
            .collect()
    }

    // --- the slot's lane packing -------------------------------------------

    /// [`Slot::pack`] is an **ABI**, not a convenience: the lanes it fills are read by
    /// `dynamics.wesl`'s accessors, which name them from the other side. So every field
    /// is pinned to the component it lands in, with a value that appears nowhere else —
    /// a swap of two same-typed neighbours (`lambda_lift` for `lambda_deposit`, `orient`
    /// for `drain`) is otherwise a silent wrong picture that no golden would attribute
    /// to the right cause.
    ///
    /// The generated `offset_of` assertions pin where a *lane* starts. This is what
    /// pins what lives inside one.
    #[test]
    fn every_slot_field_lands_in_the_lane_the_shader_reads_it_from() {
        let packed = Slot {
            start: Vec2::new(1.0, 2.0),
            dir: Vec2::new(3.0, 4.0),
            radius: 5.0,
            travel_radii: 6.0,
            lambda_lift: 7.0,
            lambda_deposit: 8.0,
            channels: [9.0, 10.0, 11.0, 12.0],
            rect_origin: Vec2::new(13.0, 14.0),
            orient: 15.0,
            drain: 16.0,
            add: 17.0,
            curvature: 18.0,
            bleed_reach: 19.0,
            noise_freq: [20.0, 21.0, 22.0, 23.0],
            noise_amp: [24.0, 25.0, 26.0],
            noise_off: [27.0, 28.0],
            dist: 29.0,
            bearing: 30.0,
            lambda_bleed: 31.0,
            tooth: 32.0,
            weave_scale: 33.0,
            weave_bias: Vec2::new(34.0, 35.0),
            resid: [36.0, 37.0, 38.0, 39.0],
            cell: 40.0,
            cell_anchor: Vec2::new(41.0, 42.0),
        }
        .pack();

        assert_eq!(packed.a, [1.0, 2.0, 3.0, 4.0], "a: start.xy, dir.zw");
        assert_eq!(
            packed.b,
            [5.0, 6.0, 7.0, 8.0],
            "b: radius, travel, λ_lift, λ_dep"
        );
        assert_eq!(packed.c, [9.0, 10.0, 11.0, 12.0], "c: colour + opacity");
        assert_eq!(
            packed.d,
            [13.0, 14.0, 15.0, 16.0],
            "d: rect.xy, orient, drain"
        );
        assert_eq!(
            packed.e,
            [17.0, 18.0, 19.0, 0.0],
            "e: add, curvature, reach, —"
        );
        assert_eq!(packed.f, [20.0, 21.0, 22.0, 23.0], "f: noise frequency");
        assert_eq!(
            packed.g,
            [24.0, 25.0, 26.0, 29.0],
            "g: noise amplitude, dist"
        );
        assert_eq!(
            packed.h,
            [27.0, 28.0, 30.0, 31.0],
            "h: noise off, bearing, λ_bleed"
        );
        assert_eq!(
            packed.i,
            [32.0, 33.0, 34.0, 35.0],
            "i: tooth, weave scale + bias"
        );
        assert_eq!(
            packed.j,
            [36.0, 37.0, 38.0, 39.0],
            "j: the colour's residual (§6.7)"
        );
        assert_eq!(
            packed.k,
            [40.0, 41.0, 42.0, 0.0],
            "k: the footprint cell + its canvas anchor (§6.2)"
        );
    }

    /// The neutral slot is neutral *in the shader's terms*, which for one field is not
    /// zero: a `bearing` of 0 books the tool's half of every transfer against no ground
    /// at all — infinite tooth, not absent tooth — so a derived `Default` would make
    /// the two slot kinds that leave it alone quietly wrong.
    #[test]
    fn the_default_slot_is_neutral_rather_than_zeroed() {
        let d = Slot::default().pack();
        assert_eq!(d.h[2], 1.0, "the default bearing must be 1, not 0");
        assert_eq!(d.k[0], 1.0, "the default cell must be 1 — exact — not 0");
        for (lane, name) in [
            (d.a, "a"),
            (d.b, "b"),
            (d.c, "c"),
            (d.d, "d"),
            (d.e, "e"),
            (d.f, "f"),
            (d.g, "g"),
            (d.i, "i"),
        ] {
            assert_eq!(lane, [0.0; 4], "lane {name} is not neutral by default");
        }
        assert_eq!(
            [d.h[0], d.h[1], d.h[3]],
            [0.0; 3],
            "lane h beyond the bearing"
        );
        assert_eq!([d.k[1], d.k[2], d.k[3]], [0.0; 3], "lane k beyond the cell");
    }

    /// **The cell grid is anchored to the canvas, not to the region** (§6.4). Where a
    /// cell boundary falls must be a property of the canvas position and the brush
    /// alone: region origins differ per piece and per live fold, and a grid that
    /// moved with them would break tile aprons against neighbour interiors and
    /// `preview == committed` in one stroke. This replays the shader's own index
    /// arithmetic (`div_floor(rt + anchor, c)`) against [`cell_geometry`]'s anchor
    /// for a spread of origins and asserts every boundary lands on the same canvas
    /// texel as it does with the origin at zero.
    #[test]
    fn cell_boundaries_are_canvas_anchored_whatever_the_region_origin() {
        let rect = Rect {
            origin: Vec2::ZERO,
            groups: (8, 8),
        };
        let boundaries = |origin: f32| -> Vec<i32> {
            let (anchor, groups) = cell_geometry(5, Vec2::new(origin, origin), &rect, 64);
            assert!(groups.is_some(), "a cell of 5 must take the coarse path");
            let (c, a) = (5i32, anchor.x as i32);
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
    /// or the deposit reads whatever the previous segment left in the scratch: not a
    /// wrong colour but a stale one, from another segment's tip, which is the kind of
    /// artifact a golden reports without naming.
    ///
    /// So this replays the shader's own tap arithmetic in f32, for every texel a
    /// dispatch can scan, and asserts the window sits inside [`cell_span`]'s count —
    /// the one the host dispatches and [`cell_scratch_size`] is asserted against.
    #[test]
    fn the_bilinear_read_never_taps_a_cell_the_hoist_skipped() {
        let dsize = 64;
        for cell in [2u32, 3, 5, 10, 16] {
            for region_origin in [-640.0f32, -5.0, 0.0, 3.0, 17.0, 999.0] {
                for rect_origin in [0.0f32, 1.0, 7.0, 33.0] {
                    let rect = Rect {
                        origin: Vec2::splat(rect_origin),
                        groups: (3, 5),
                    };
                    let (anchor, groups) =
                        cell_geometry(cell, Vec2::splat(region_origin), &rect, dsize);
                    assert!(groups.is_some(), "a cell above 1 must take the coarse path");
                    let (c, a) = (cell as i32, anchor.x as i32);
                    // The shader's `cell_base`: the rect's first cell, less the border.
                    let base = (rect_origin as i32 + a).div_euclid(c) - CELL_BORDER as i32;
                    // The texels the deposit scans: its rect as rounded to whole
                    // workgroups, clamped to the snapshot square — what `cells` counts.
                    let span = (rect.groups.0 * 8).min(dsize) as i32;
                    let count = cell_span(rect_origin as i32 + a, span, c) as i32;
                    for t in 0..span {
                        let rt = rect_origin as i32 + t;
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

    // --- the bleed cadence -------------------------------------------------

    /// **The claim `bleed_fires` is built on**: which windows fire, and what each one
    /// sweeps, is a pure function of the record — not of where the renderer happened to
    /// cut the stroke into pieces or ranges.
    ///
    /// This is a `preview == committed` property (§1.3) in the one place it cannot be
    /// repainted. A live tail always starts at a span boundary while the commit renders
    /// the whole stroke from zero, so if a window came out shorter for one than the
    /// other, a bleeding stroke would visibly lighten the moment the pointer came up —
    /// which it did, before the window learned to walk back along the crossing
    /// segment's own arc instead of looking its start up among the segments in hand.
    ///
    /// Checked at every cut point rather than one, since the interesting cuts are
    /// exactly the ones that land mid-window.
    #[test]
    fn bleed_firings_do_not_depend_on_where_the_stroke_was_cut() {
        // Segments well under the quantum (0.5 · radius = 5px), so windows routinely
        // reach back over several of them and a cut can land inside one.
        let all = run(40, 1.5, 10.0);
        let whole: Vec<_> = bleed_fires(0.4, &all)
            .into_iter()
            .map(|(i, f)| (i, f.start, f.length, f.dist))
            .collect();
        assert!(
            whole.len() > 3,
            "the case does not fire often enough to be interesting: {}",
            whole.len()
        );

        for cut in 1..all.len() {
            let mut split: Vec<_> = bleed_fires(0.4, &all[..cut])
                .into_iter()
                .map(|(i, f)| (i, f.start, f.length, f.dist))
                .collect();
            split.extend(
                bleed_fires(0.4, &all[cut..])
                    .into_iter()
                    .map(|(i, f)| (i + cut, f.start, f.length, f.dist)),
            );
            assert_eq!(
                split, whole,
                "cutting after segment {cut} changed the firings"
            );
        }
    }

    /// The firings of a curved segment **tile it**, a quantum each, and every one of
    /// them lies on the path rather than on a chord across it.
    ///
    /// Two properties that hold each other up. Tiling is what makes the axis a
    /// diffusivity: a firing carries a fixed variance, so N quanta of travel have to
    /// arrive as N firings or the axis is quietly scaled by `1/N` (`bleed_stencil`).
    /// And a window on the arc is what makes each of those tiles a stretch of the path
    /// rather than an approximation to one — at this cadence the bow a chord would sit
    /// off it is under a thousandth of a texel, so this is not a correction the picture
    /// needs today; it is that the representation cannot go wrong if the cadence is
    /// ever coarsened, which is the lever `BLEED_REACH_MAX` names as the way to
    /// diffuse further.
    #[test]
    fn a_segments_firings_tile_it_along_its_own_arc() {
        use crate::gpu::stroke::budget::MAX_TIP_TURN;
        // A 40 px brush at the tightest arc the flattener will sweep it along, its
        // size modulated down to a 3 px tip — and segments at the travel cap, which is
        // priced off the 40 rather than the 3, so each crosses the cadence many times.
        let (nominal, tip, len) = (40.0f32, 3.0f32, 40.0f32);
        let kappa = MAX_TIP_TURN / nominal;
        let r = 1.0 / kappa;
        let centre = Vec2::new(0.0, r);
        let quantum = BLEED_TRAVEL_QUANTUM * tip;

        let mut segs = Vec::new();
        let (mut p, mut d, mut dist) = (Vec2::ZERO, Vec2::new(1.0, 0.0), 0.0);
        for _ in 0..20 {
            let mut s = seg(p, d, len, tip, dist);
            s.curvature = kappa;
            segs.push(s);
            (p, d) = crate::path::arc_at(p, d, kappa, len);
            dist += len;
        }

        let fires = bleed_fires(0.4, &segs);
        // The cap is what stops this being `len / quantum` = 53 per segment.
        assert_eq!(fires.len(), 20 * MAX_BLEED_FIRES_PER_SEGMENT);
        for (i, f) in &fires {
            assert_eq!(
                f.length, quantum,
                "a firing swept more than its own quantum"
            );
            // Every point of the window sits on the circle the path traced — not just
            // its two ends, which the walk back along the crossing segment's own arc
            // already put there.
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let on = crate::path::arc_at(f.start, f.dir, f.curvature, f.length * t).0;
                assert!(
                    ((on - centre).length() - r).abs() < 1e-2,
                    "the window left the path {} px at t = {t}",
                    (on - centre).length() - r,
                );
            }
            // Butted end to end, back from the segment's end: no quantum of travel is
            // diffused twice, and none is skipped between the cap and that end.
            let s = &segs[*i];
            let from_end = s.dist + s.length - f.dist;
            let quanta = from_end / quantum;
            assert!(
                (quanta - quanta.round()).abs() < 1e-3
                    && quanta <= MAX_BLEED_FIRES_PER_SEGMENT as f32 + 1e-3,
                "a firing sits {quanta} quanta back from its segment's end",
            );
        }
    }

    /// A brush that does not bleed fires nothing at all — the early-out is exact, and
    /// is what lets every non-bleeding stroke keep the no-bleed path bit-for-bit.
    #[test]
    fn a_brush_that_does_not_bleed_fires_nothing() {
        assert!(bleed_fires(0.0, &run(40, 1.5, 10.0)).is_empty());
    }

    /// **Why the cadence exists at all**: a firing's window is a quarter-radius of travel
    /// however finely the path was cut, so its exposure is a well-conditioned prefix
    /// difference rather than the f16 noise a per-segment flux would be.
    ///
    /// A hand that draws slowly is fitted at a control point per pointer sample — the
    /// repro that prompted this carried 177 knots over 68 px — and at that cut a texel's
    /// per-segment flux lands under the f16 ULP of the height it is editing, so every
    /// store either snaps it away or ratchets a whole ULP. One firing moves what those
    /// micro-segments would each have tried to move, in a step far above the floor.
    #[test]
    fn a_firing_sweeps_its_own_quantum_however_finely_the_path_was_cut() {
        let radius = 20.0;
        let quantum = BLEED_TRAVEL_QUANTUM * radius;
        // 0.39 px a segment — the repro's mean span.
        let fine = run(400, 0.39, radius);
        let fires = bleed_fires(0.4, &fine);
        assert!(fires.len() > 5, "only {} firings", fires.len());
        for (_, f) in &fires {
            assert!(
                (f.length - quantum).abs() < 0.5,
                "a firing swept {} of the {quantum} its cadence carries",
                f.length,
            );
            assert!(
                f.length > 10.0 * 0.39,
                "the window is segment-sized, which is the regime the cadence exists \
                 to leave",
            );
        }
    }

    // --- the pen-up frame --------------------------------------------------

    /// A brush that manipulates paint, so a stroke of it settles at all.
    fn smearing(radius: f32) -> crate::document::BrushParams {
        crate::document::BrushParams {
            radius,
            dynamics: crate::document::BrushDynamics {
                lift: 0.8,
                deposit: 0.8,
                ..crate::document::BrushDynamics::default()
            },
            ..crate::document::BrushParams::default()
        }
    }

    /// A stroke through `pts` with `brush`, as plain full-pressure knots.
    fn record(brush: crate::document::BrushParams, pts: &[Vec2]) -> StrokeRecord {
        StrokeRecord {
            layer: crate::document::LayerId(0),
            brush,
            path: pts
                .iter()
                .map(|p| crate::path::ControlPoint::at(*p))
                .collect(),
            seed: 0,
        }
    }

    /// The segments of `range` of `rec`, at the budget the loop would flatten it with.
    fn segments_of(rec: &StrokeRecord, range: std::ops::Range<usize>) -> Vec<Segment> {
        generate_segments_in(
            rec,
            flatten_tolerance(&rec.brush),
            StrokeSpans { range, dist: 0.0 },
        )
        .0
    }

    /// What [`settle_tangent`] used to do: walk back a radius through **the segments in
    /// hand**. Kept here as the thing the test below measures against — it is the
    /// behaviour that made a stroke's fade-out cap turn as the pointer came up.
    fn piece_local_tangent(segments: &[Segment], end: Vec2) -> Vec2 {
        let radius = segments.last().map_or(1.0, |s| s.radius);
        let mut back = end;
        let mut acc = 0.0;
        for s in segments.iter().rev() {
            back = s.start;
            acc += s.length;
            if acc >= radius {
                break;
            }
        }
        let v = end - back;
        let len = v.length();
        if len > 1e-4 {
            v / len
        } else {
            segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.dir)
        }
    }

    /// **The claim [`settle_tangent`] is built on**: the pen-up frame is a function of
    /// the record, not of where the renderer happened to cut the stroke into ranges and
    /// pieces.
    ///
    /// A live tail starts at a span boundary while the commit renders the whole stroke
    /// from zero, and both run the settle — the tail's range reaches the stroke's end,
    /// which is exactly the condition that asks for one. So a frame measured over "the
    /// segments in hand" came out over a shorter window for the tail than for the
    /// commit, and on a curving stroke that is a different direction: the settle's
    /// `min(owed, received)` lens is elongated along it, so the fade-out cap visibly
    /// turned at pen-up. That is a `preview == committed` break (§1.3) in the one place
    /// it cannot be repainted — the same one `bleed_fires` was fixed for.
    ///
    /// Checked at every cut point, since the interesting ones are those that leave the
    /// tail shorter than the tip being settled.
    #[test]
    fn the_settle_frame_does_not_depend_on_where_the_stroke_was_cut() {
        // A circle of radius 200 under a 60 px tip: a footprint's worth of path turns
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
        let want = settle_tangent(&rec, flatten_tolerance(&rec.brush), &whole);

        let mut ever_differed = false;
        for cut in 1..all {
            let tail = segments_of(&rec, cut..all);
            if tail.is_empty() {
                continue;
            }
            let got = settle_tangent(&rec, flatten_tolerance(&rec.brush), &tail);
            assert!(
                (got - want).length() < 1e-4,
                "cutting at span {cut} moved the settle frame from {want:?} to {got:?}",
            );
            // And the old piece-local walk really does move on these cuts, so the
            // assertion above is not passing because the case is uninteresting.
            let last = whole.last().expect("segments");
            let end = crate::path::arc_at(last.start, last.dir, last.curvature, last.length).0;
            let stale = piece_local_tangent(&tail, end);
            ever_differed |= (stale - piece_local_tangent(&whole, end)).length() > 1e-2;
        }
        assert!(
            ever_differed,
            "no cut left a tail short enough to move the old frame — the test proves \
             nothing",
        );
    }

    /// [`settle_tangent`] must survive the way a real pen-up arrives: a hand pauses
    /// before it lifts, so the last samples cluster at one point and the flattener
    /// turns them into edges whose chord is a rounding error and whose direction is
    /// therefore arbitrary.
    ///
    /// Nothing else in the loop notices — a segment of no length deposits nothing, so
    /// its direction never reaches a pixel. The settle is the exception: it takes a
    /// whole tip's worth of exchange from that one frame, and its `min(owed, received)`
    /// lens is elongated *along* it, so a wrong direction lays a tip-shaped disc across
    /// the stroke instead of along it. That is what a wandering fade-out cap was.
    ///
    /// Reading the record's own polyline is what makes this structural rather than a
    /// rule to remember: knots piled on one spot contribute nothing to the chord over
    /// the last radius, so they cannot steer it whatever direction the fitter gave the
    /// edges between them.
    #[test]
    fn the_settle_frame_ignores_a_paused_hands_arbitrary_last_edges() {
        // A straight drag along +y, then the pause: four knots on the stopping point.
        let mut pts: Vec<Vec2> = (0..20).map(|i| Vec2::new(0.0, i as f32 * 2.0)).collect();
        let stop = Vec2::new(0.0, 38.0);
        pts.extend([stop; 4]);
        let rec = record(smearing(12.0), &pts);
        let all = crate::path::span_count(rec.path.len());
        let segs = segments_of(&rec, 0..all);

        let tan = settle_tangent(&rec, flatten_tolerance(&rec.brush), &segs);
        assert!(
            (tan - Vec2::new(0.0, 1.0)).length() < 1e-2,
            "the settle frame followed the paused hand: {tan:?}"
        );
    }

    /// A click has no travel to measure a direction over — a lone control point is not
    /// a curve, so there is no polyline to walk — and still gets a real frame: the
    /// dab's own, which `generate_segments_in` gave it deliberately.
    #[test]
    fn a_click_settles_along_its_dab() {
        let rec = record(smearing(10.0), &[Vec2::new(4.0, -7.0)]);
        let segs = segments_of(&rec, 0..crate::path::span_count(rec.path.len()));
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        assert_eq!(
            settle_tangent(&rec, flatten_tolerance(&rec.brush), &segs),
            segs[0].dir,
        );
    }

    // --- the dispatch rects fit the scratch --------------------------------

    /// Every dispatch rect a piece builds fits the snapshot scratch that piece sized.
    ///
    /// The two now go through one function — [`rect_extent`], of which `snapshot_size`
    /// takes the maximum — so the fit is structural rather than an equality between a
    /// margin here and two magic numbers there. `dispatch_rect` asserts it; this runs
    /// that assert over shapes chosen to stress it, on the CPU, with no adapter needed.
    ///
    /// The stress is in the fractional origins: a rect floors its origin and rounds its
    /// far edge outward, so the worst case is a box straddling texel boundaries at both
    /// ends — which is exactly the case `rect_extent`'s `+2` is for.
    #[test]
    fn every_dispatch_rect_fits_the_scratch_its_piece_sized() {
        // Sub-pixel origins are the point: the rect floors its origin and rounds its
        // far edge out, so the worst case is a box straddling texel boundaries at both
        // ends. Curvature is swept too, since a bent sweep bows a sagitta out of its box.
        for &radius in &[0.5f32, 1.0, 7.3, 40.0, 120.0] {
            for &length in &[0.0f32, 0.37, 4.0, 60.0] {
                for &kappa in &[0.0f32, 0.004, -0.02] {
                    for &frac in &[0.0f32, 0.499, 0.5, 0.999] {
                        let start = Vec2::new(frac, -frac);
                        let mut s = seg(start, Vec2::new(1.0, 0.0), length, radius, 0.0);
                        s.curvature = kappa;
                        let segments = [s];
                        let dsize = snapshot_size(&segments, &[]);

                        let (lo, hi) = coverage_bounds(&s);
                        // Region origins that put the box at both ends of the region.
                        for &origin in &[Vec2::ZERO, lo.floor(), Vec2::new(-13.7, 91.2)] {
                            dispatch_rect(lo, hi, origin, dsize);
                            // The pen-up's square, which is sized from the same scratch
                            // but built from the tip alone rather than the swept box.
                            let end = crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0;
                            dispatch_rect(
                                end - Vec2::splat(radius),
                                end + Vec2::splat(radius),
                                origin,
                                dsize,
                            );
                        }
                    }
                }
            }
        }
    }

    /// A bleed window can be the largest footprint in its piece — it sweeps up to a
    /// half-radius where the piece's own segments may be sub-pixel — so the scratch has
    /// to be sized with the firings in it, not just the segments.
    #[test]
    fn the_scratch_is_sized_with_the_bleed_windows_in_it() {
        // Long enough to cross the 0.5 · 30 = 15 px cadence, cut far finer than it.
        let segments = run(200, 0.2, 30.0);
        let fires = bleed_fires(0.5, &segments);
        assert!(!fires.is_empty(), "no firing to size against");
        let with = snapshot_size(&segments, &fires);
        let without = snapshot_size(&segments, &[]);
        assert!(
            with > without,
            "a firing's window did not widen the scratch ({without} -> {with})"
        );
        for (_, f) in &fires {
            let (lo, hi) = coverage_bounds(f);
            dispatch_rect(lo, hi, Vec2::ZERO, with);
        }
    }

    // `the_host_and_the_shader_agree_on_the_loops_constants` stood here, reading
    // `BAKE_RES` out of the linked shader. It is generated now, so there is one
    // declaration of it (§6.10).

    // `the_stamp_struct_has_the_same_nine_lanes_on_both_sides` stood here, counting
    // `vec4<f32>` in the shader source and comparing against [`SLOT`]. There is no
    // longer a second declaration for it to disagree with: `Stamp` is generated from
    // the WESL, and the generator emits `offset_of` assertions per lane, so a tenth
    // lane moves both sides at once and a mistake in the layout is a build failure.
}
