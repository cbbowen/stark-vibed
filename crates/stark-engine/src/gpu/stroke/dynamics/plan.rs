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
//! The one GPU *type* that reaches in is the canvas [`SubstrateMap`](crate::gpu::substrate),
//! and only for `bearing` — plain CPU arithmetic over the substrate's statistics, which
//! happens to live on the struct that also owns its texture.
//!
//! Every slot is a pure function of the record and the piece's own geometry, in plain
//! CPU float math, so replay is deterministic (§12.1).

use stark_model::document::StrokeRecord;
use stark_model::geom::Vec2;

use super::super::budget::{extent_cell, lambda};
use super::super::region::{coverage_bounds, segment_end};
use super::super::segments::{BleedFire, Segment, Stretch};
use super::bleed::bleed_stencil;
// The `Stamp` uniform, generated from `dynamics.wesl`'s own declaration at build
// time (`stark-shaders/build/mirror.rs`) — lanes, offsets, and the documentation of
// what each lane holds, all on the generated fields.
//
// **The shader decides how the lanes are read, so it is the only place they are
// written down** (§6.10). A hand-written twin here — nine `[f32; 4]` fields against
// the shader's nine `vec4`s, with its own copy of the lane map in the doc comments —
// is a second declaration nothing checks against the first.
//
// Every slot is a pure function of the `StrokeRecord` and the piece's own geometry,
// computed in plain CPU float math, so replay is deterministic (§12.1).
use stark_shaders::mirror::dynamics::{CELL_BORDER, Stamp, TILE_WG};

/// One slot's window into the stamp buffer, and the `min_binding_size` its layout
/// declares — both of which have to be `Stamp`'s own size, so they are taken from it
/// rather than written down.
pub(super) const SLOT: usize = std::mem::size_of::<Stamp>();

/// Stride between those windows: the padded size a dynamic offset must be a multiple
/// of, from the type rather than from a number written here.
///
/// [`SLOT`] is what a slot *holds* and this is what it *occupies*, and the difference
/// is the whole reason to derive it: as a bare `256` beside a `copy_from_slice` of
/// [`SLOT`] bytes, a `Stamp` that outgrew the quantum would have written over the next
/// slot rather than widening them — silently, and only for whichever brush reached
/// that dispatch. See `swept::XFORM_STRIDE`, which is the same law for the sweep's own
/// per-tile uniform.
pub(super) const STAMP_STRIDE: u64 = crate::gpu::uniforms::UniformSlots::<Stamp>::STRIDE;

/// One slot of the sequential swept-exchange loop (§6.2), and the dispatches it
/// stands for.
pub(super) struct LoopDispatch {
    pub(super) slot: Stamp,
    /// Workgroup counts for the slot's extent work — the `deposit`, and the
    /// `snapshot` that rides in `exchange`'s grid. The slot's own coverage box
    /// rather than the piece-wide worst-case square, so an axis-aligned sweep pays for
    /// the ~4·r² texels its extent can reach instead of the ~10·r² a diagonal one
    /// might have needed.
    pub(super) groups: (u32, u32),
    /// Workgroup counts for the `cell_hoist` grid when this slot takes the **coarse
    /// deposit** (§6.2) — `Some` exactly when [`extent_cell`] beat 1 for this
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
    /// One segment of a liquify stroke (§6.13): `snapshot` (over the whole
    /// square — the gather pulls from upstream of any one texel's sweep test) →
    /// `warp`. The tool plays no part, so there is nothing to bake or exchange,
    /// the reservoir ping-pong never advances, and a plan of these is the whole
    /// of what [`liquify_plan`] emits.
    Warp,
}

/// The [`Stamp`] lanes every slot in a plan fills the same way, resolved once — so
/// the three slot kinds below list only what actually differs between them, which
/// is the whole of what makes a bleed slot or a settle slot readable against a
/// painting segment.
struct SlotCommon<'a> {
    /// The stroke's own constants: `c` outright, and the color-dynamics lookup that
    /// fills `noise_freq` and `noise_amp`. Borrowed rather than copied out, so a slot
    /// and the swept path's `TileXform` are demonstrably reading one resolution of
    /// them.
    k: &'a super::super::StrokeConstants,
    /// `i.yzw`: the region texel → substrate map, with the piece's origin already
    /// folded into the bias. Only `i.x` — how deep this slot's tip bites — varies.
    ///
    /// The one lane here that is not a stroke constant: the bias is where the *piece*
    /// sits, which `k` cannot know.
    substrate: [f32; 3],
    /// The region's canvas origin in texels — the deposit jitter's map from a
    /// region texel to the absolute canvas one its gate is keyed on (§6.2).
    /// Integral by the cell grid's own argument ([`cell_geometry`]'s assert), and
    /// a plan fact like `substrate` above: the piece's placement, which `k`
    /// cannot know.
    origin: [i32; 2],
}

impl SlotCommon<'_> {
    /// The lanes every slot fills the same way — the stroke's color and the substrate
    /// map — over the neutral value of everything a slot kind may leave alone.
    ///
    /// A slot kind then names only what it actually differs by, which is the whole of
    /// what makes a bleed or settle slot readable against a painting segment.
    fn slot(&self) -> Slot {
        Slot {
            channels: self.k.channels,
            // The other half of the same color (§6.7) — filled wherever `channels` is,
            // and zero in a space whose three channels already say everything.
            resid: self.k.resid,
            // The paint effect's ceiling (§6.2), off `k` like the color: one
            // resolution for every slot kind and for the swept path's integrate.
            // Only a painting segment's mint reads it; a bleed or settle slot
            // carries it inertly, the jitter's argument one field down.
            opacity: self.k.opacity,
            // …and whether the pen drives it, which is likewise the stroke's. A
            // bleed or settle slot carries the flag inertly with the dial.
            ceiling_lane: self.k.ceiling_lane,
            substrate_uv_scale: self.substrate[0],
            substrate_uv_bias: Vec2::new(self.substrate[1], self.substrate[2]),
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
            canvas_origin: self.origin,
            ..Slot::default()
        }
    }

    /// [`Self::slot`] plus the color-dynamics jitter, for a slot that lays the
    /// brush's own `add` paint: the shared field, this slot's arc length, and the
    /// bearing fraction it books the tool's half of the transfer against.
    ///
    /// `lambda_bleed` stays 0, which is what every such slot wants: the lateral flux
    /// runs only on the dedicated bleed slots, so between firings the canvas takes the
    /// no-bleed path bit-for-bit (§6.2).
    fn painting(&self, dist: f32, bearing: f32) -> Slot {
        let namp = self.k.namp;
        Slot {
            noise_freq: self.k.nfreq,
            noise_amp: [namp[0], namp[1], namp[2]],
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
    /// The tip's radius in region px — and so the frame the sweep is unrolled in
    /// (§6.6) — with its travel as a multiple of that radius: 0 on a settle, which
    /// is a break of contact rather than a stretch of it.
    radius: f32,
    travel_radii: f32,
    /// `λ = ln(1 − axis) ≤ 0`, clamped away from −∞. Zero is "no transfer".
    lambda_lift: f32,
    lambda_deposit: f32,
    /// The brush's own color channels. **Undrained** — and no per-unit opacity
    /// beside them: minted paint is opaque per unit, faded only by the drain
    /// (§6.2).
    channels: [f32; 3],
    /// The paint effect's opacity — the ceiling the mint's prefix-difference law
    /// runs under in the shader (§6.2). 1 is the identity, exactly.
    opacity: f32,
    /// The ceiling's modulation factor across this segment (§6.2) — the mean of
    /// its two ends and the difference between them, read at the texel's own
    /// travel by the shader. Only where `ceiling_lane` is set; `(1, 0)` — a
    /// scale's neutral and no ramp — everywhere else.
    opacity_mod: f32,
    opacity_ramp: f32,
    /// How this segment's other **per-texel** rates change across it (§6.2) —
    /// `Paint`'s three ramps, carried through unchanged. The exchange's own lambdas
    /// have none: that solve is one problem per dispatch (see `dynamics.wesl`).
    add_ramp: f32,
    tooth_give_ramp: f32,
    drag_ramp: f32,
    /// Whether the stroke's ceiling is pen-driven, off `k` like the dial: routes
    /// the mint through the ceiling lane the region aux's `.w` carries.
    ceiling_lane: bool,
    /// The same color's **residual** (§6.7) in `.xyz`; `.w` unused. Undrained like
    /// `channels`, and zero in a space that has no residual to carry.
    resid: [f32; 4],
    /// The dispatch rect's top-left in region texels, integral.
    rect_origin: Vec2,
    /// Shape orientation in turns ∈ [0, 1) — picks the prefix-τ slice (§6.6).
    orient: f32,
    /// The `drain` falloff per canvas px — [`BrushParams::drain_px`](stark_model::document::BrushParams::drain_px), the brush's own
    /// per-*radius* knob already divided, so the shader's arc length (canvas px) and
    /// this rate are in the same units.
    drain: f32,
    /// The `add` source rate per unit exposure, **undrained** like the opacity.
    add: f32,
    /// Signed curvature of the sweep (1/region px); 0 is a straight one.
    curvature: f32,
    /// The liquify follow rate (§6.13): the segment's modulated strength over
    /// the tip's peak τ density — nonzero **only** on a warp slot, and the lane
    /// `snapshot` reads to know it must copy its whole square.
    drag: f32,
    /// The bleed stencil's longest tap in texels — nonzero **only** on a bleed slot.
    bleed_reach: f32,
    /// The color-dynamics lookup (§6.2): frequency per axis + 1/NOISE_TILE_PX and
    /// per-channel amplitude. All zero = no jitter.
    noise_freq: [f32; 4],
    noise_amp: [f32; 3],
    /// Arc length at the slot's start (canvas px) — the noise's third axis.
    dist: f32,
    /// The tooth's bearing fraction: the share of the substrate a tip with this tooth,
    /// going this way, stands on (§6.4). What the *tool* books its half of the
    /// transfer against, having no substrate of its own. 1 where there is nothing to bite.
    bearing: f32,
    /// The lateral canvas diffusion rate (≤ 0) — nonzero **only** on a bleed slot.
    lambda_bleed: f32,
    /// How little give this slot's tip has (0 = the substrate gates nothing), how wide
    /// the transition around that is, and the region texel → substrate map
    /// `uv = rt · substrate_uv_scale + substrate_uv_bias` (§6.4).
    tooth_give: f32,
    tooth_softness: f32,
    substrate_uv_scale: f32,
    substrate_uv_bias: Vec2,
    /// The extent cell's edge in texels (§6.2) — 1 is the exact per-texel deposit,
    /// which is also the neutral value: the exact kernels never read the lane.
    cell: f32,
    /// The cell grid's canvas anchor: the region's canvas origin modulo the cell,
    /// so cell boundaries are pinned to canvas texels whatever region surrounds them
    /// (§6.4). Zero whenever `cell` is 1.
    cell_anchor: Vec2,
    /// How much the tip grows across this slot's sweep, as a fraction of
    /// [`frame`](Self::radius) — [`Sweep::ramp`](super::super::segments::Sweep).
    ///
    /// Zero on a bleed window and on a settle, and both are meant: a firing is a
    /// stretch of *diffusion* at one tip (which is also the radius
    /// [`bleed_stencil`] solved its stencil against), and a settle is the tip
    /// standing still with no travel to ramp along. Zero is also
    /// [`Default`]'s value, so a slot kind that has never heard of ramps cannot
    /// accidentally acquire one.
    radius_ramp: f32,
    /// The tip drawn out along its facing axis (§6.6), solved into the map that
    /// carries a point of the reference travel frame into the frame the prefix-τ
    /// volume and the bake rows are indexed in
    /// ([`Stretch`]).
    ///
    /// The identity `(1, 0, 1)` for every brush that does not stretch — and, like
    /// the other neutral-at-1 lanes below, that is a triple of *scales* and not of
    /// zeroes: a zeroed lane is not "no stretch" but a tip of no width and infinite
    /// gain. [`Stretch::NONE`](super::super::segments::Stretch::NONE) states it, so
    /// neither this default nor the shader's neutral value is written twice.
    stretch: Stretch,
    /// The deposit jitter (§6.2): the gate's half-range (0 = off, the neutral
    /// value), the stroke's seed for it, and the region's canvas origin — the
    /// jitter's map from region texels to the absolute canvas grid it is keyed on.
    jitter_eps: f32,
    jitter_seed: u32,
    canvas_origin: [i32; 2],
}

impl Default for Slot {
    /// Zero everywhere except the three fields whose neutral value is **1**. Two are
    /// there for one reason: they are *scales*, and a zeroed scale does not mean "none
    /// of this" but "none of the thing it multiplies". `bearing` is the share of the
    /// substrate a tip stands on where there is nothing to bite — zeroed it would book the
    /// tool's half of every transfer against no substrate at all, which is not "no tooth"
    /// but "infinite tooth". `cell` at 1 is the exact per-texel deposit.
    ///
    /// The third, `tooth_give`, is there for a reason of its own: the knob runs the
    /// other way (`BrushParams::tooth_give`), so a zeroed lane is not "no tooth" but
    /// the driest tip there is. 1 is a tip that follows every fall, which is what a
    /// bleed slot needs — it is the only lane keeping the lateral flux clear of the
    /// substrate (`dynamics.wesl::substrate_tooth`) — and what a slot kind that never
    /// heard of the tooth ought to get.
    fn default() -> Self {
        Self {
            start: Vec2::ZERO,
            dir: Vec2::ZERO,
            radius: 0.0,
            travel_radii: 0.0,
            lambda_lift: 0.0,
            lambda_deposit: 0.0,
            channels: [0.0; 3],
            // A scale, so its neutral value is 1 — the identity the shader
            // branches to exactly — not 0, which would be the strongest ceiling
            // there is (`stretch`'s argument, one field over).
            opacity: 1.0,
            opacity_mod: 1.0,
            opacity_ramp: 0.0,
            add_ramp: 0.0,
            tooth_give_ramp: 0.0,
            drag_ramp: 0.0,
            ceiling_lane: false,
            resid: [0.0; 4],
            rect_origin: Vec2::ZERO,
            orient: 0.0,
            drain: 0.0,
            add: 0.0,
            curvature: 0.0,
            drag: 0.0,
            bleed_reach: 0.0,
            noise_freq: [0.0; 4],
            noise_amp: [0.0; 3],
            dist: 0.0,
            bearing: 1.0,
            lambda_bleed: 0.0,
            tooth_give: 1.0,
            // Never read where `tooth_give` is 1, so the value is free; zero rather
            // than the brush's default because a slot that reads it is a slot
            // `SlotCommon` filled, and this is what an unfilled one looks like.
            tooth_softness: 0.0,
            substrate_uv_scale: 0.0,
            substrate_uv_bias: Vec2::ZERO,
            cell: 1.0,
            cell_anchor: Vec2::ZERO,
            radius_ramp: 0.0,
            stretch: Stretch::NONE,
            jitter_eps: 0.0,
            jitter_seed: 0,
            canvas_origin: [0, 0],
        }
    }
}

impl Slot {
    /// This slot as the uniform `dynamics.wesl` reads.
    ///
    /// **A rename, not a packing.** `Stamp`'s members are named now, and the mirror
    /// generates them from the shader's own declaration (§6.10), so the field names on
    /// both sides of each line below are one name checked by the compiler. What stood
    /// here was a lane map — `e: [self.add, self.curvature, self.bleed_reach, …]` —
    /// whose correspondence to the shader's `st.e.z` nothing could see: the sizes
    /// matched whatever order the four were written in.
    ///
    /// The casts are the members the shader declares **integral** because they are:
    /// a rect origin, a cell edge, a stencil reach. They are `f32` in the plan because
    /// the rect arithmetic around them is, and every one is a whole number by
    /// construction — `rect.origin` is a texel corner, `cell` an edge in texels,
    /// `reach` a tap count.
    fn pack(&self) -> Stamp {
        Stamp {
            start: self.start.to_array(),
            dir: self.dir.to_array(),
            radius: self.radius,
            travel_radii: self.travel_radii,
            radius_ramp: self.radius_ramp,
            lambda_lift: self.lambda_lift,
            lambda_deposit: self.lambda_deposit,
            lambda_bleed: self.lambda_bleed,
            curvature: self.curvature,
            brush_lat: self.channels,
            opacity: self.opacity,
            brush_res: [self.resid[0], self.resid[1], self.resid[2]],
            add: self.add,
            drag: self.drag,
            noise_freq: [self.noise_freq[0], self.noise_freq[1], self.noise_freq[2]],
            arc_at_start: self.dist,
            noise_amp: self.noise_amp,
            drain: self.drain,
            stretch: [
                self.stretch.travel,
                self.stretch.shear,
                self.stretch.lateral,
            ],
            orientation: self.orient,
            substrate_uv_bias: self.substrate_uv_bias.to_array(),
            rect_origin: [self.rect_origin.x as i32, self.rect_origin.y as i32],
            cell_anchor: [self.cell_anchor.x as i32, self.cell_anchor.y as i32],
            tooth_give: self.tooth_give,
            tooth_softness: self.tooth_softness,
            tooth_bearing: self.bearing,
            substrate_uv_scale: self.substrate_uv_scale,
            cell_px: self.cell as i32,
            bleed_reach: self.bleed_reach as i32,
            jitter_eps: self.jitter_eps,
            jitter_seed: self.jitter_seed,
            canvas_origin: self.canvas_origin,
            opacity_mod: self.opacity_mod,
            opacity_ramp: self.opacity_ramp,
            add_ramp: self.add_ramp,
            tooth_give_ramp: self.tooth_give_ramp,
            drag_ramp: self.drag_ramp,
            ceiling_lane: u32::from(self.ceiling_lane),
            // The struct's own trailing padding, generated because the scalars
            // above end 8 bytes short of the uniform's 16-byte round (§6.10) —
            // `TileXform`'s own tail, one uniform over.
            _pad_38: [0; 8],
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
    /// The budget `rec` was flattened at, handed down from [`dynamics_setup`](super::dynamics_setup) rather
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
const RECT_MARGIN: f32 = 1.5;

/// The snapshot square's pool quantum: [`snapshot_square`] rounds the measured maximum
/// up to a multiple of this.
///
/// For the scratch pool's sake alone. The maximum drifts a few texels per fold as the
/// tail's geometry evolves, which would make nearly every checkout a miss
/// ([`ScratchPool`](crate::gpu::scratch)); rounded, the handful of sizes a stroke's
/// folds actually take recur. Nothing the shaders compute changes — stores and reads
/// are gated by the slot rects and the sweep test, and the `textureDimensions` bounds
/// only widen onto texels those gates reject — so the round-up moves no pixel.
const SNAPSHOT_QUANTUM: u32 = 64;

/// One slot's dispatch rectangle over a canvas-space coverage box: its integral origin
/// in region texels and its extent, from which the workgroup counts follow.
///
/// The slot's own box rather than the piece-wide worst case, so an axis-aligned sweep
/// dispatches ~4·r² threads where a square would spend ~10·r². Texels the rounding adds
/// beyond the box read zero exposure and fall out of `deposit` untouched.
fn dispatch_rect(lo: Vec2, hi: Vec2, region_origin: Vec2) -> Rect {
    let lo = lo - region_origin - Vec2::splat(RECT_MARGIN);
    let hi = hi - region_origin + Vec2::splat(RECT_MARGIN);
    let origin = Vec2::new(lo.x.floor(), lo.y.floor());
    Rect {
        origin,
        w: ((hi.x - origin.x).ceil() as u32) + 1,
        h: ((hi.y - origin.y).ceil() as u32) + 1,
    }
}

/// One slot's dispatch rectangle.
struct Rect {
    /// The rect's top-left in region texels, integral — the `d.xy` a slot carries.
    origin: Vec2,
    /// Its extent in texels.
    w: u32,
    h: u32,
}

impl Rect {
    /// Workgroup counts covering it, at the shaders' own tile size (§6.10).
    fn groups(&self) -> (u32, u32) {
        (groups_for(self.w), groups_for(self.h))
    }
}

/// Workgroups covering `texels` along one axis, at the tile kernels' declared side.
///
/// `TILE_WG` is generated from `dynamics.wesl`'s own `const` — the side that decides
/// it — rather than transcribed here (§6.10). Every host expression that turns texels
/// into groups, or groups back into texels, goes through this constant.
pub(super) fn groups_for(texels: u32) -> u32 {
    texels.div_ceil(TILE_WG)
}

/// The snapshot scratch's square for a piece: **the largest rect the piece will
/// actually dispatch**, rounded up to [`SNAPSHOT_QUANTUM`].
///
/// **A maximum, not a bound**, and the distinction is the point. A bound would be a
/// second derivation of this number — a position-independent extent folded over the
/// coverage boxes — related to the real rects by an argument ("monotone in span") and
/// defended by an assertion in the render path. The rects are computed once, so the
/// scratch is sized by taking their maximum, and there is nothing for an assertion to
/// state: a maximum is not a claim about the things it was taken over.
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
/// [`CELL_BORDER`] ring on each side. The same structural-fit move as
/// [`snapshot_square`]/[`dispatch_rect`]: both sides of the relation go through this
/// function — [`cell_geometry`] asserts against it, `DynamicsRun::cell_scratch`
/// allocates from it — so no slot a plan can build reads cells the scratch does not
/// hold.
pub(super) fn cell_scratch_size(dsize: u32) -> u32 {
    dsize.div_ceil(2) + 2 + (2 * CELL_BORDER) as u32
}

/// Cells the hoist must write to cover `span` texels from region texel `base`
/// (already offset by the anchor), border included — the window
/// `deposit_coarse`'s taps are asserted to sit inside.
fn cell_span(base: i32, span: i32, c: i32) -> u32 {
    ((base + span - 1).div_euclid(c) - base.div_euclid(c) + 1) as u32 + (2 * CELL_BORDER) as u32
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
    // Restating `RegionRect::origin`'s own guarantee rather than establishing it — see
    // there for why a fractional origin cannot arrive, and why it would be a seam if
    // one did.
    let anchor = Vec2::new(
        (region_origin.x as i32).rem_euclid(c) as f32,
        (region_origin.y as i32).rem_euclid(c) as f32,
    );
    let groups = rect.groups();
    let cells = |origin: f32, a: f32, groups: u32| -> u32 {
        cell_span(
            origin as i32 + a as i32,
            (groups * TILE_WG).min(dsize) as i32,
            c,
        )
    };
    let cx = cells(rect.origin.x, anchor.x, groups.0);
    let cy = cells(rect.origin.y, anchor.y, groups.1);
    // Derived rather than defended, now that `dsize` is the maximum of the very rects
    // this is asked about: a rect spans at most `dsize` texels, so at a cell of `c ≥ 2`
    // it names at most `ceil(dsize/c) + 1 + 2·CELL_BORDER ≤ dsize.div_ceil(2) + 3`
    // cells, and [`cell_scratch_size`] is that plus one. Debug-only for the reason the
    // dispatch rect's assertion is gone altogether — a panic mid-render is a worse
    // failure than the thing it guards, and this one is arithmetic.
    let fit = cell_scratch_size(dsize);
    debug_assert!(
        cx <= fit && cy <= fit,
        "a {cx}x{cy}-cell hoist overruns the {fit}-cell scratch",
    );
    (anchor, Some((groups_for(cx), groups_for(cy))))
}

/// What one slot of the plan is built from, and — because the walk that produces these
/// is the walk that defines the plan's order — the single statement of that order.
///
/// The rects are measured over this list and the slots are built by zipping the two, so
/// the two passes cannot drift into disagreeing about which rect belongs to which slot.
/// That is the price of computing a rect once instead of twice, and it is paid here
/// rather than by two `for` loops that happen to be written the same way.
enum SlotSource<'a> {
    Segment(&'a Segment),
    Bleed(&'a BleedFire),
    /// The pen-up, which is the last segment read a different way — a standing tip
    /// rather than a stretch of travel.
    Settle(&'a Segment),
}

impl SlotSource<'_> {
    /// The canvas box this slot's dispatch has to cover.
    fn bounds(&self) -> (Vec2, Vec2) {
        match self {
            SlotSource::Segment(s) => coverage_bounds(&s.sweep),
            SlotSource::Bleed(f) => coverage_bounds(&f.window),
            // The tip's own extent rather than a swept box — a pen-up is a standing
            // tip. Its half-extent is the tip's `reach` (`Sweep::reach`), so the
            // settle writes the same extent the pass was laying. It cannot be the
            // largest box in the piece: a segment's box is this square grown by its
            // travel, and this is the last segment's.
            SlotSource::Settle(s) => {
                let end = segment_end(&s.sweep);
                let reach = Vec2::splat(s.sweep.reach);
                (end - reach, end + reach)
            }
        }
    }
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

/// The plan's slots in dispatch order, and the snapshot square they fit.
///
/// The square rides with the slots because it is derived from them — see
/// [`snapshot_square`]. A caller that has one has the other, so there is no way to
/// allocate a scratch for a plan other than the one that measured it.
pub(super) struct DynamicsPlan {
    pub(super) slots: Vec<LoopDispatch>,
    pub(super) dsize: u32,
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
    fires: &[BleedFire],
    settle: bool,
) -> DynamicsPlan {
    let &PlanCtx {
        rec,
        region_origin,
        consts,
        substrate,
        ..
    } = ctx;
    let b = &rec.brush;
    // The canvas → substrate map, folded so the shader can go straight from its *region*
    // texel to the substrate under it: `uv = rt · substrate_uv_scale + substrate_uv_bias` (§6.4). Only the
    // bias belongs to the piece — the shader never learns where the piece sits, only
    // where the substrate does; the scale is a stroke constant and comes off `consts`,
    // which is what keeps it the same number the swept path writes.
    let substrate_uv_bias = region_origin * consts.substrate_uv_scale;
    // What share of the substrate a tip with this tooth, going this way, stands on — per
    // segment because the tooth's *give* is modulated per segment (§6.2) and because
    // the direction is the segment's own. The canvas side of the exchange asks the substrate
    // ahead of each texel; the tool has none of its own and books against this mean,
    // which is what makes a toothed smear conserve (`SubstrateMap::bearing`).
    //
    // The width of the transition is the brush's outright, so it closes over rather
    // than being passed in: it is the same number on every segment, and the mean has
    // to be taken through the very gate the canvas side evaluates or the two halves
    // of the transfer stop agreeing.
    //
    // At the segment's **midpoint** tangent, the same second-order choice `mid` is
    // sampled at below: a curved segment's canvas side reads a tangent that turns
    // across the sweep, and the midpoint is the representative of that whose error is
    // second order where either endpoint's would be first.
    let bearing = |give: f32, dir: Vec2| substrate.bearing(give, b.tooth.softness, dir.to_array());
    // λ per axis is [`lambda`](super::super::budget::lambda) — one definition, the
    // same clamp the flattening budget prices. Taken **per segment**, off the rates
    // the segment generator resolved from the pen (§6.2), rather than once for the
    // stroke: every dispatch carries its own λs in its slot, because a segment is
    // where the exchange happens.
    let common = SlotCommon {
        k: consts,
        substrate: [
            consts.substrate_uv_scale,
            substrate_uv_bias.x,
            substrate_uv_bias.y,
        ],
        origin: [region_origin.x as i32, region_origin.y as i32],
    };

    // Drained in step with the walk below, which is only correct because `bleed_fires`
    // emits them in segment order. Cheap to state, and the alternative — a firing
    // silently landing in the wrong piece of the plan — is not something a pixel would
    // show.
    debug_assert!(
        fires.is_sorted_by_key(|f| f.after),
        "bleed firings must arrive in segment order",
    );

    // ---- Pass one: what the plan dispatches, in order. This walk is the *only*
    // statement of that order; the rects and the slots below both hang off it.
    let mut sources: Vec<SlotSource<'_>> = Vec::new();
    let mut pending = fires.iter().peekable();
    for (si, s) in segments.iter().enumerate() {
        sources.push(SlotSource::Segment(s));
        // The bleed slots that fire at this segment's end (§6.2, `bleed_fires`).
        while let Some(fire) = pending.next_if(|f| f.after == si) {
            sources.push(SlotSource::Bleed(fire));
        }
    }
    // The pen-up (`dynamics.wesl::settle`), as one more slot on the same uniform: the
    // tip standing at the stroke's last point with **zero travel**, which is what makes
    // the shared `segment_frame`/`outside_sweep` reduce to the tip's own extent and
    // `snapshot` copy exactly the texels the settle will write. Everything the settle
    // reads is already here — the frame, the radius, the two λs and the orientation —
    // so it costs a slot rather than a second uniform.
    if let Some(s) = settle.then(|| segments.last()).flatten() {
        sources.push(SlotSource::Settle(s));
    }

    // ---- Pass two: the rect each of those dispatches over, and the scratch square
    // that is their maximum. Computed once and carried, so the rect that sizes the
    // scratch and the rect that is dispatched are the same number rather than two
    // measurements an assertion in the render path has to hold together.
    let rects = rects_for(&sources, region_origin);
    let dsize = snapshot_square(&rects);

    // ---- Pass three: the slots themselves, zipped to the rects measured for them.
    let mut plan = Vec::with_capacity(sources.len());
    for (src, rect) in sources.iter().zip(&rects) {
        let groups = rect.groups();
        let dispatch = match src {
            SlotSource::Segment(s) => {
                let (sw, paint) = (&s.sweep, &s.paint);
                // The segment's swept exchange: the frame is (start, travel tangent at
                // the start, curvature), over the segment's own coverage box.
                let p = sw.start - region_origin;
                // The tangent at the segment's **midpoint**, along the arc rather than
                // the chord: what the bearing below is read along, since a curved
                // segment's canvas side sees a heading that turns across the sweep and
                // the midpoint is the representative of that whose error is second
                // order where either endpoint's would be first.
                let (_, mid_dir) =
                    crate::path::arc_at(sw.start, sw.dir, sw.curvature, sw.length * 0.5);
                // The extent cell this segment's deposit may evaluate the exchange
                // at (§6.2): a pure function of the brush shape and the segment's own
                // radius ([`extent_cell`]), so a live tail and its commit pick the
                // same cell — and 1, the exact kernel, for every tip whose shoulder
                // proves nothing.
                let cell = extent_cell(&b.shape, sw.radius);
                let (cell_anchor, cell_groups) = cell_geometry(cell, region_origin, rect, dsize);
                LoopDispatch {
                    groups,
                    cell_groups,
                    kind: SlotKind::Segment,
                    slot: Slot {
                        start: p,
                        dir: sw.dir,
                        // The tip, and the travel in its units.
                        radius: sw.radius,
                        travel_radii: sw.length / sw.radius,
                        // The tip's growth across this segment ([`Sweep::ramp`]).
                        // Everything the host prices off `sw.radius` — the cell, the
                        // bleed cadence, the exchange step — stays on the reference
                        // tip, which is what that radius is.
                        radius_ramp: sw.radius_ramp,
                        // Scaled by the segment's flow (§6.2): λ rides the
                        // exponent, so this is exposure scaling — a pass at flow f
                        // trades exactly what f passes at flow 1 would, and the
                        // exchange still composes exactly across overlapping
                        // segments. `add` and `bleed`, linear in exposure, took
                        // the factor at the segment generator instead.
                        lambda_lift: paint.flow * lambda(paint.lift),
                        lambda_deposit: paint.flow * lambda(paint.deposit),
                        rect_origin: rect.origin,
                        orient: sw.orient,
                        stretch: sw.stretch,
                        drain: b.drain_px(),
                        // The `add` rate as the segment resolved it — mappings and
                        // the wet flow already folded (`generate_segments_in`) —
                        // with **no further gain here**, exactly as `stamp.wesl`
                        // takes it. A gain here would make the same slider mean two
                        // different amounts of paint depending on whether some
                        // *other* axis happened to be non-zero — nudging `deposit`
                        // off zero would change the flow. Nor is one needed to
                        // make an effective rate of 1 lay a full-thickness deposit
                        // per pass: a pass of the tip is `TAU_PER_PASS ≈ 6.9` of
                        // exposure, so a rate of 1 lays 6.9 of height, which the
                        // slab law reads as 0.999 coverage. The effect's opacity is
                        // *not* folded in either — the raw rate is exactly what the
                        // ceiling's prefix-difference law must accumulate (§6.2,
                        // `dynamics.wesl::lay_parcel`), and the ceiling rides the
                        // slot's own lane.
                        //
                        // Off the segment, since the pen can drive it (§6.2) — the same
                        // number the swept path now reads off its instance.
                        add: paint.add,
                        // The ceiling's factor as the segment resolved it, beside
                        // the rate it caps — the same pair the swept path reads
                        // off its instance, ramp included, so a texel's factor is
                        // the same number on either path.
                        opacity_mod: paint.opacity,
                        opacity_ramp: paint.opacity_ramp,
                        add_ramp: paint.add_ramp,
                        tooth_give_ramp: paint.tooth_give_ramp,
                        curvature: sw.curvature,
                        tooth_give: paint.tooth_give,
                        cell: cell as f32,
                        cell_anchor,
                        // No `bleed_reach` and no `lambda_bleed`: the lateral flux runs
                        // only on the dedicated firings, so a painting segment takes
                        // the no-bleed path bit-for-bit (§6.2). Both are
                        // `Slot::default`'s zero.
                        ..common.painting(sw.dist, bearing(paint.tooth_give, mid_dir))
                    }
                    .pack(),
                }
            }
            // A quad whose sweep is the firing's travel window, with every vertical
            // rate and the source zeroed — the dispatch is the identity everywhere
            // except the lateral flux. The noise lanes are zeroed too, so the deposit
            // skips its color-jitter taps.
            SlotSource::Bleed(fire) => {
                let w = &fire.window;
                let p = w.start - region_origin;
                // The stencil this firing diffuses with: how far it reaches, and how
                // hard it relaxes to get there. Both come out of the diffusivity the
                // axis asks for — see [`bleed_stencil`], which is where the axis's
                // whole meaning is.
                let (reach, lambda_bleed) = bleed_stencil(fire.bleed, w.radius, w.length);
                LoopDispatch {
                    groups,
                    // Bleed slots keep the exact deposit whatever the tip: the ladder's
                    // flux pairs need both threads of a pair to read per-texel
                    // exposures, and the firings are rare and small next to the
                    // painting they cut.
                    cell_groups: None,
                    kind: SlotKind::Bleed,
                    // Everything a painting segment carries and this does not is
                    // `Slot::default`'s zero, which is what the slot *means*: λ_lift = 0
                    // so the canvas keeps everything, λ_deposit = 0 so the (uninvolved)
                    // tool lays nothing, no drain because nothing is laid, no `add`
                    // because this is not a stretch of painting, no tooth because there
                    // is no `add` for the substrate to gate, and no color jitter — which is
                    // zeroed rather than shared, so the deposit skips its noise taps
                    // entirely.
                    //
                    // A [`BleedFire`] cannot carry those rates in the first place: it
                    // holds a [`Sweep`] and its one axis. Holding a whole `Segment`
                    // instead, its five rates would be copied in by `bleed_fires` for
                    // this arm to write straight back out.
                    slot: Slot {
                        start: p,
                        dir: w.dir,
                        radius: w.radius,
                        travel_radii: w.length / w.radius,
                        rect_origin: rect.origin,
                        orient: w.orient,
                        stretch: w.stretch,
                        // The window's own curvature, so the relaxed band follows the
                        // paint rather than cutting the corner off it (`bleed_fires`).
                        curvature: w.curvature,
                        // The stencil's longest tap — the only slot that carries one.
                        bleed_reach: reach,
                        dist: w.dist,
                        // The rate that lands this window's exposure on the blend its
                        // reach needs — not `lambda(axis)`, which is the vertical rates'
                        // mapping and would make the axis a rate rather than a
                        // diffusivity. A firing whose modulated axis has fallen to zero
                        // still dispatches: λ = 0 makes it the identity, and keeping the
                        // plan a pure function of the segmentation is worth more than
                        // the dispatch it would save.
                        lambda_bleed,
                        ..common.slot()
                    }
                    .pack(),
                }
            }
            SlotSource::Settle(s) => {
                let (sw, paint) = (&s.sweep, &s.paint);
                let p = segment_end(sw) - region_origin;
                // The frame comes off the *record* — see [`settle_tangent`]. `segments`
                // is one piece of one range, and a lookback that walked it would stop
                // wherever that cut fell, so a live tail and its commit would settle at
                // different angles.
                let tan = settle_tangent(rec, ctx.tol, segments);
                LoopDispatch {
                    groups,
                    // The settle is one dispatch at the end of a stroke — nothing to
                    // amortize — and its p-norm handover is exactly the smooth structure
                    // a cell would staircase, so it stays exact whatever the tip.
                    cell_groups: None,
                    kind: SlotKind::Settle,
                    slot: Slot {
                        start: p,
                        dir: tan,
                        // No travel: a pen-up is a break of contact, not a stretch of
                        // it. The rates are the *last* segment's, which is where the pen
                        // was when it left the page — the same segment this slot takes
                        // its radius and orientation from. (`travel_radii` stays at its
                        // default 0.) The λs carry that segment's flow with them,
                        // as the painting slots' do.
                        radius: sw.radius,
                        lambda_lift: paint.flow * lambda(paint.lift),
                        lambda_deposit: paint.flow * lambda(paint.deposit),
                        rect_origin: rect.origin,
                        orient: sw.orient,
                        stretch: sw.stretch,
                        drain: b.drain_px(),
                        // The last segment's tooth: the settle delivers what the pass
                        // still owed, and it owes it through the same substrate the pass
                        // was laying through. What the valleys do not take stays on the
                        // tool, which is discarded — a knife lifted off a canvas keeps
                        // what it did not reach (§6.4).
                        tooth_give: paint.tooth_give,
                        // No `add`: the source is a rate per unit of travel, and there
                        // is none. No curvature, for the same reason — the frame is a
                        // standing tip. No bleed reach: a settle is not a firing. And no
                        // λ_bleed either — that axis carries no reservoir, every firing
                        // having applied its window as the tip passed, so a break of
                        // contact strands nothing for a settle to finish, unlike the
                        // vertical transfer whose in-flight half lives on the tool. All
                        // four are `Slot::default`'s zero.
                        //
                        // The bearing is the neutral 1: the tool is not written back at
                        // pen-up, so nothing reads it — the settle's own gate is per
                        // texel, from the substrate. The color channels are filled
                        // consistently with a segment slot rather than left as junk,
                        // though the settle lays the tool's *carried* paint and so reads
                        // none of them.
                        ..common.painting(sw.dist + sw.length, 1.0)
                    }
                    .pack(),
                }
            }
        };
        plan.push(dispatch);
    }
    DynamicsPlan { slots: plan, dsize }
}

/// Build the liquify dispatch plan (§6.13): one warp slot per flattened segment,
/// in stroke order, and nothing else — no bleed cadence (the effect has no such
/// axis) and no pen-up (the warp strands nothing at a break of contact: each
/// segment's follow is applied whole as the tip passes, the bleed axis's own
/// argument).
///
/// The follow rate the slot carries is the segment's modulated strength over the
/// tip's **peak τ density** (`assets::peak_tau`, through `inv_peak_tau`), so the
/// shader's `drag · e · radius` is px of travel kept pace with — see
/// `dynamics.wesl::warp`. Pure CPU float math over the record and the piece's
/// geometry, like [`dynamics_plan`], and replay-deterministic for the same
/// reason (§12.1).
pub(super) fn liquify_plan(
    ctx: &PlanCtx<'_>,
    segments: &[Segment],
    inv_peak_tau: f32,
) -> DynamicsPlan {
    let &PlanCtx {
        rec,
        region_origin,
        consts,
        ..
    } = ctx;
    let b = &rec.brush;
    let substrate_uv_bias = region_origin * consts.substrate_uv_scale;
    let common = SlotCommon {
        k: consts,
        substrate: [
            consts.substrate_uv_scale,
            substrate_uv_bias.x,
            substrate_uv_bias.y,
        ],
        origin: [region_origin.x as i32, region_origin.y as i32],
    };
    // The same three-pass shape as [`dynamics_plan`], with one source kind: the
    // walk *is* the order, the rects are measured over it, and the slots zip to
    // them — so the rect that sizes the scratch is the rect that dispatches.
    let sources: Vec<SlotSource<'_>> = segments.iter().map(SlotSource::Segment).collect();
    let rects = rects_for(&sources, region_origin);
    let dsize = snapshot_square(&rects);
    let mut plan = Vec::with_capacity(sources.len());
    for (s, rect) in segments.iter().zip(&rects) {
        let (sw, paint) = (&s.sweep, &s.paint);
        let p = sw.start - region_origin;
        plan.push(LoopDispatch {
            groups: rect.groups(),
            // The warp gathers the field whole; a cell would staircase exactly
            // the structure the effect exists to carry.
            cell_groups: None,
            kind: SlotKind::Warp,
            slot: Slot {
                start: p,
                dir: sw.dir,
                radius: sw.radius,
                travel_radii: sw.length / sw.radius,
                radius_ramp: sw.radius_ramp,
                // The one lane only a warp slot carries — and the lane
                // `snapshot` reads to know its square must be copied whole.
                drag: paint.drag * inv_peak_tau,
                // Normalised by the same peak, so the two ends stay the pair
                // they were (§6.13).
                drag_ramp: paint.drag_ramp * inv_peak_tau,
                rect_origin: rect.origin,
                orient: sw.orient,
                stretch: sw.stretch,
                curvature: sw.curvature,
                // The drag runs dry as a deposit would (§6.2): the drain at the
                // texel's own arc, which needs the arc at the slot's start.
                drain: b.drain_px(),
                dist: sw.dist,
                // …and catches on the substrate as a deposit would (§6.4). No
                // `bearing`: there is no tool side to book the transfer against.
                tooth_give: paint.tooth_give,
                // Every rate a deposit would run on stays `Slot::default`'s
                // zero — a warp slot lays nothing, lifts nothing, bleeds
                // nothing — and the noise lanes stay zero with them: the effect
                // has no color for a jitter to wander (§6.13).
                ..common.slot()
            }
            .pack(),
        });
    }
    DynamicsPlan { slots: plan, dsize }
}

/// The travel direction the pen-up settle measures `owed` and `received` along: the
/// chord over the **last extent's worth of path**, rather than the last segment's
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
/// settled — the tip's own extent — so this is the direction the tip was travelling
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
/// This is the same defect [`bleed_fires`](super::bleed::bleed_fires) was fixed for, and the cure is the same in
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
    // — which under-estimate arc length, so the span this admits genuinely holds a
    // extent's worth of path behind it. Walking boundaries rather than the polyline
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
    use crate::gpu::stroke::budget::flatten_tolerance;
    use crate::gpu::stroke::segments::Stretch;
    use crate::gpu::stroke::segments::generate_segments_in;
    use crate::gpu::stroke::segments::testing::{record, run, seg, smearing};
    use stark_model::geom::Vec2;

    // --- the slot's lane packing -------------------------------------------

    /// Every field of the plan's slot reaches the member of `Stamp` the shader reads
    /// it from.
    ///
    /// **Much less is left to check than there was.** `Stamp`'s members are named now
    /// (§6.10), so `pack` reads `lambda_lift: self.lambda_lift` and a swap of two
    /// same-typed neighbours is a compile error rather than a wrong picture. What
    /// survives is the handful of lines where the two sides spell one quantity
    /// differently — `orient`/`orientation`, `dist`/`arc_at_start`,
    /// `ramp`/`radius_ramp`, `bearing`/`tooth_bearing` — and the three that split a
    /// host array (`channels`, `resid`, `noise_freq`). Those are the assertions
    /// below; the rest are the compiler's.
    ///
    /// (`pack` used to close with a `..Default::default()` for the generated
    /// padding's sake; `drag` filled the last hole, so the struct is written out
    /// whole — the one trailing pad the ceiling's pair reopened named as the
    /// zeroes it is — and a forgotten member is the compiler's to catch.)
    #[test]
    fn every_slot_field_lands_in_the_member_the_shader_reads_it_from() {
        let packed = Slot {
            start: Vec2::new(1.0, 2.0),
            dir: Vec2::new(3.0, 4.0),
            radius: 5.0,
            travel_radii: 6.0,
            lambda_lift: 7.0,
            lambda_deposit: 8.0,
            channels: [9.0, 10.0, 11.0],
            opacity: 53.0,
            opacity_mod: 55.0,
            opacity_ramp: 56.0,
            add_ramp: 57.0,
            tooth_give_ramp: 58.0,
            drag_ramp: 59.0,
            ceiling_lane: true,
            rect_origin: Vec2::new(13.0, 14.0),
            orient: 15.0,
            drain: 16.0,
            add: 17.0,
            curvature: 18.0,
            drag: 54.0,
            bleed_reach: 19.0,
            noise_freq: [20.0, 21.0, 22.0, 23.0],
            noise_amp: [24.0, 25.0, 26.0],
            dist: 29.0,
            bearing: 30.0,
            lambda_bleed: 31.0,
            tooth_give: 32.0,
            tooth_softness: 48.0,
            substrate_uv_scale: 33.0,
            substrate_uv_bias: Vec2::new(34.0, 35.0),
            resid: [36.0, 37.0, 38.0, 39.0],
            cell: 40.0,
            cell_anchor: Vec2::new(41.0, 42.0),
            radius_ramp: 44.0,
            stretch: Stretch {
                travel: 45.0,
                shear: 46.0,
                lateral: 47.0,
                turns: 0.0,
            },
            jitter_eps: 49.0,
            jitter_seed: 50,
            canvas_origin: [51, 52],
        }
        .pack();

        // The pairs the two sides spell differently.
        assert_eq!(packed.travel_radii, 6.0, "travel_radii");
        assert_eq!(packed.orientation, 15.0, "orient → orientation");
        assert_eq!(packed.arc_at_start, 29.0, "dist → arc_at_start");
        assert_eq!(packed.radius_ramp, 44.0, "ramp → radius_ramp (§6.2)");
        assert_eq!(packed.tooth_bearing, 30.0, "bearing → tooth_bearing (§6.4)");
        assert_eq!(packed.substrate_uv_scale, 33.0, "substrate_uv_scale");
        assert_eq!(packed.substrate_uv_bias, [34.0, 35.0], "substrate_uv_bias");
        assert_eq!(packed.cell_px, 40, "cell → cell_px, as an integer (§6.2)");
        assert_eq!(packed.cell_anchor, [41, 42], "cell_anchor, as integers");
        assert_eq!(packed.rect_origin, [13, 14], "rect_origin, as integers");
        assert_eq!(packed.bleed_reach, 19, "bleed_reach, as an integer");
        // The three the host keeps as one array and the shader splits.
        assert_eq!(packed.brush_lat, [9.0, 10.0, 11.0], "channels → brush_lat");
        assert_eq!(packed.brush_res, [36.0, 37.0, 38.0], "resid (§6.7)");
        assert_eq!(packed.noise_freq, [20.0, 21.0, 22.0], "noise_freq");
        // And the ones whose names already match, so that a member never assigned at
        // all — which `..Default::default()` would zero in silence — still fails here.
        assert_eq!(packed.start, [1.0, 2.0]);
        assert_eq!(packed.dir, [3.0, 4.0]);
        assert_eq!(packed.radius, 5.0);
        assert_eq!(packed.lambda_lift, 7.0);
        assert_eq!(packed.lambda_deposit, 8.0);
        assert_eq!(packed.lambda_bleed, 31.0);
        assert_eq!(packed.curvature, 18.0);
        assert_eq!(packed.add, 17.0);
        assert_eq!(packed.drag, 54.0, "the liquify follow rate (§6.13)");
        assert_eq!(packed.drain, 16.0);
        assert_eq!(packed.noise_amp, [24.0, 25.0, 26.0]);
        assert_eq!(packed.tooth_give, 32.0);
        assert_eq!(packed.tooth_softness, 48.0);
        assert_eq!(
            packed.stretch,
            [45.0, 46.0, 47.0],
            "the facing stretch (§6.6)"
        );
        assert_eq!(packed.jitter_eps, 49.0, "the deposit jitter (§6.2)");
        assert_eq!(packed.jitter_seed, 50);
        assert_eq!(packed.canvas_origin, [51, 52]);
        assert_eq!(packed.opacity, 53.0, "the effect's ceiling (§6.2)");
        assert_eq!(packed.opacity_mod, 55.0, "the ceiling's pen factor (§6.2)");
        assert_eq!(
            packed.opacity_ramp, 56.0,
            "…and its ramp across the segment"
        );
        assert_eq!(packed.add_ramp, 57.0, "the source rate's ramp (§6.2)");
        assert_eq!(packed.tooth_give_ramp, 58.0, "the tooth give's (§6.4)");
        assert_eq!(packed.drag_ramp, 59.0, "the liquify follow's (§6.13)");
        assert_eq!(packed.ceiling_lane, 1, "the ceiling lane's flag (§6.2)");
    }

    /// The neutral slot is neutral *in the shader's terms*, which for seven fields
    /// is not zero. Six are **scales**, and a zeroed scale does not say "none of
    /// this" but "none of the thing it multiplies": a `bearing` of 0 books the tool's
    /// half of every transfer against no substrate at all — infinite tooth, not absent
    /// tooth; a `cell` of 0 is no deposit grid rather than the exact one; a
    /// zeroed stretch is a tip of no width whose every prefix difference is divided by
    /// it (§6.6); a zeroed `opacity` is the strongest ceiling there is where 1 is
    /// the exact identity branch (§6.2), and a zeroed `opacity_mod` would take the
    /// whole of it away on the lane. The seventh, `tooth_give`, is not a scale but
    /// a knob that runs the other way (`BrushParams::tooth_give`): zeroed it is the
    /// driest tip there is, where 1 is the short-circuit a slot with no tooth wants.
    /// A derived `Default` would make every slot kind that leaves one alone quietly
    /// wrong, and this is the list that says which they are.
    #[test]
    fn the_default_slot_is_neutral_rather_than_zeroed() {
        let d = Slot::default().pack();
        assert_eq!(d.tooth_bearing, 1.0, "the default bearing must be 1, not 0");
        assert_eq!(d.cell_px, 1, "the default cell must be 1 — exact — not 0");
        assert_eq!(
            d.tooth_give, 1.0,
            "the default give must be 1 — full contact — not 0, which is the dry mark"
        );
        assert_eq!(
            d.stretch,
            [1.0, 0.0, 1.0],
            "the default stretch must be the identity map, not zeroes"
        );
        assert_eq!(
            d.opacity, 1.0,
            "the default opacity must be 1 — the identity ceiling — not 0"
        );
        assert_eq!(
            d.opacity_mod, 1.0,
            "the default ceiling factor must be 1 — the pen taking nothing — not 0"
        );
        // And everything else is zero, which for the rest of the slot *is* neutral —
        // stated as the complement of the five above so a new member has to be
        // classified rather than silently joining whichever list it was written near.
        let z = Stamp {
            tooth_bearing: 1.0,
            cell_px: 1,
            tooth_give: 1.0,
            stretch: [1.0, 0.0, 1.0],
            opacity: 1.0,
            opacity_mod: 1.0,
            ..Default::default()
        };
        assert_eq!(
            bytemuck::bytes_of(&d),
            bytemuck::bytes_of(&z),
            "a member of the default slot is neither zero nor one of the seven neutrals",
        );
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
            w: 64,
            h: 64,
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
    /// wrong color but a stale one, from another segment's tip, which is the kind of
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
                        w: 24,
                        h: 40,
                    };
                    let (anchor, groups) =
                        cell_geometry(cell, Vec2::splat(region_origin), &rect, dsize);
                    assert!(groups.is_some(), "a cell above 1 must take the coarse path");
                    let (c, a) = (cell as i32, anchor.x as i32);
                    // The shader's `cell_base`: the rect's first cell, less the border.
                    let base = (rect_origin as i32 + a).div_euclid(c) - CELL_BORDER;
                    // The texels the deposit scans: its rect as rounded to whole
                    // workgroups, clamped to the snapshot square — what `cells` counts.
                    let span = (rect.groups().0 * TILE_WG).min(dsize) as i32;
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

    // --- the pen-up frame --------------------------------------------------

    /// The segments of `range` of `rec`, at the budget the loop would flatten it with.
    fn segments_of(rec: &StrokeRecord, range: std::ops::Range<usize>) -> Vec<Segment> {
        generate_segments_in(
            rec,
            flatten_tolerance(&rec.brush),
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
    /// from zero, and both run the settle — the tail's range reaches the stroke's end,
    /// which is exactly the condition that asks for one. A frame measured over "the
    /// segments in hand" therefore spans a shorter window for the tail than for the
    /// commit, and on a curving stroke that is a different direction: the settle's
    /// `min(owed, received)` lens is elongated along it, so the fade-out cap visibly
    /// turns at pen-up. That is a `preview == committed` break (§1.3) in the one place
    /// it cannot be repainted — the same class `bleed_fires` has to answer.
    ///
    /// Checked at every cut point, since the interesting ones are those that leave the
    /// tail shorter than the tip being settled.
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

    /// The sources a piece's plan would walk, in the order [`dynamics_plan`] walks
    /// them — segments, each followed by its own firings, then the pen-up.
    fn sources_of<'a>(
        segments: &'a [Segment],
        fires: &'a [BleedFire],
        settle: bool,
    ) -> Vec<SlotSource<'a>> {
        let mut sources = Vec::new();
        let mut pending = fires.iter().peekable();
        for (si, s) in segments.iter().enumerate() {
            sources.push(SlotSource::Segment(s));
            while let Some(f) = pending.next_if(|f| f.after == si) {
                sources.push(SlotSource::Bleed(f));
            }
        }
        if let Some(s) = settle.then(|| segments.last()).flatten() {
            sources.push(SlotSource::Settle(s));
        }
        sources
    }

    /// **Every dispatch grid a piece builds fits the snapshot scratch that piece
    /// sized.**
    ///
    /// That a *rect* fits is no claim at all: [`snapshot_square`] takes the maximum of
    /// the very rects the plan dispatches. What is a claim, and what the shaders'
    /// bounds checks rest on, is one step further out — a dispatch is rounded **up to
    /// whole 8×8 workgroups**, so the texels a slot scans reach `groups·8`, past its
    /// own rect. That stays inside the scratch only because [`SNAPSHOT_QUANTUM`] is
    /// itself a multiple of 8, and this is what pins it.
    ///
    /// The stress is in the fractional origins: a rect floors its origin and rounds its
    /// far edge outward, so the worst case is a box straddling texel boundaries at both
    /// ends. Curvature is swept too, since a bent sweep bows a sagitta out of its box.
    #[test]
    fn every_dispatch_grid_fits_the_scratch_its_piece_sized() {
        for &radius in &[0.5f32, 1.0, 7.3, 40.0, 120.0] {
            for &length in &[0.0f32, 0.37, 4.0, 60.0] {
                for &kappa in &[0.0f32, 0.004, -0.02] {
                    for &frac in &[0.0f32, 0.499, 0.5, 0.999] {
                        let start = Vec2::new(frac, -frac);
                        let mut s = seg(start, Vec2::new(1.0, 0.0), length, radius, 0.0);
                        s.sweep.curvature = kappa;
                        let segments = [s];
                        // With the pen-up, whose square is built from the tip alone
                        // rather than from the swept box.
                        let sources = sources_of(&segments, &[], true);
                        // Region origins that put the box at both ends of the region.
                        let (lo, _) = coverage_bounds(&s.sweep);
                        for &origin in &[Vec2::ZERO, lo.floor(), Vec2::new(-13.7, 91.2)] {
                            let rects = rects_for(&sources, origin);
                            let dsize = snapshot_square(&rects);
                            for r in &rects {
                                let (gx, gy) = r.groups();
                                assert!(
                                    gx * 8 <= dsize && gy * 8 <= dsize,
                                    "a {}x{} rect scans {}x{} texels of a {dsize} \
                                     scratch (radius {radius}, length {length}, \
                                     curvature {kappa}, frac {frac})",
                                    r.w,
                                    r.h,
                                    gx * 8,
                                    gy * 8,
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
    /// has to be sized with the firings in it, not just the segments.
    ///
    /// Still worth stating with the rects computed once: the maximum is only over what
    /// the walk collected, so a walk that forgot the firings would size the scratch
    /// short and go on asserting nothing.
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
            rects_for(&sources_of(&segments, f, false), Vec2::ZERO)
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
                &sources_of(&segments, &fires, false),
                Vec2::ZERO
            )) >= with,
        );
    }
}
