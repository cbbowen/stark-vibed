//! **The stroke corpus** (§9): one entry per *kind of stroke*, so that every
//! invariant the engine claims about strokes can be asked of all of them at once.
//!
//! The suite grew the other way round — a stroke was written to catch a particular
//! bug, and it checked the one property that bug broke. That left holes nobody could
//! see, because a hole is exactly the combination no test happens to draw. Sweeping
//! the flattener's tangent bound over an 8× range left **every committed golden
//! bit-identical**: no golden put a stamp, or a toothed tip, on the sequential stamp
//! loop *around a curve*, so the suite would have green-lit any value. Sweeping the
//! attribute bound over ×0.125…×8 did the same, because nothing painted a real
//! pressure ramp on that path. Both times the picture only moved on a stroke built on
//! purpose afterwards.
//!
//! So the two axes are separated. **This file is the strokes**: what to draw, and what
//! each one is the only cover for. [`corpus.rs`](../corpus.rs) is the battery: what
//! must be true of a stroke, applied to all of them. A new brush axis is a `Case`
//! here and inherits every check; a new invariant is a check there and is immediately
//! asked of everything already drawn.
//!
//! A case earns its place by being the *only* cover for some axis — the list is meant
//! to be read as a claim about coverage, not as a pile of pictures.

use stark_engine::Engine;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_model::document::{BrushParams, BrushShape, OrientationSource, Tool};
use stark_model::geom::{Extent2, Vec2};

use super::{SIZE, brush, engine_or_skip_sized, stroke_with};

const RED: [f32; 4] = [0.86, 0.16, 0.12, 1.0];

/// How far a case is allowed to move under each check. Per case rather than global
/// because the answers differ by two orders of magnitude and *the differences are the
/// interesting part*: a straight swept line converges to the bit, while a wide smear
/// resamples through a reservoir coarser than the canvas and cannot. A single loose
/// global bound would hide the first, and a tight one would fail the second.
#[derive(Copy, Clone, Debug)]
pub struct Tol {
    /// Per-channel levels for the committed golden.
    pub golden: u8,
    /// Per-channel levels between a stroke drawn in pieces and the same stroke drawn
    /// in one pass — the frozen head against the commit, and the incremental preview
    /// against a fresh render. **No** texel may exceed this, however many are off by
    /// one; a cut that went wrong shows as a step, not as a haze.
    pub seam: u8,
    /// Share of the viewport (in **percent**) allowed to move visibly when the same
    /// path is refitted four times finer.
    ///
    /// Unlike its two neighbours this is an area, not a level: the level is fixed
    /// globally at the battery's `VISIBLE_LEVELS`, so the numbers below are directly
    /// comparable and read as one table — *how much of this stroke is still moving when
    /// you stop asking for it more finely*. See the battery's refinement check for why
    /// area is the right instrument here and the worst texel is not.
    pub refine: f64,
    /// Share of the viewport (in **percent**) allowed to move visibly when the pen is
    /// **lifted off the tablet** at the end of the same gesture (see [`lifted`]).
    ///
    /// The same instrument as `refine` and for the same reason — what a release does to
    /// a stroke is not a step at one texel, it is the last stretch of the mark quietly
    /// changing width — so the two are directly comparable and the corpus reads as one
    /// table of *how much of this stroke the hand coming off it moves*.
    ///
    /// **Thirteen of the fourteen are 0, exactly, and that is the claim.** A release
    /// carries no arc length, so by §6.2 it deposits nothing — the swept integral is a
    /// definite integral over travel, and `generate_segments_in` drops any edge shorter
    /// than `1e-5` before the question reaches a shader at all.
    ///
    /// The renderer honours that; the **fit** is where it is easy to lose, and this
    /// column is what holds it. Least squares summed over *reports* weighs a pen that
    /// reports on a clock rather than on a ruler, so the eight reports a release spends
    /// crossing half a pixel outvote the whole last span of real curve: the fitted
    /// pressure comes down over 88 px of a 563 px `LOOP_STROKE` and 134 px of an 838 px
    /// `FAST_STROKE`, reaching the tip at 0.80 and 0.52 instead of 1.0, and every case
    /// here moves — `line` by 22% of the viewport. `path::arc_weights` and the
    /// attribute end condition beside it are what keep the weighting honest.
    pub lift: f64,
}

/// One stroke worth drawing, and everything needed to draw it.
pub struct Case {
    /// Identifies the case in failures and names its golden (`corpus_<name>.png`).
    pub name: &'static str,
    /// **What this case is the only cover for.** If another case would catch
    /// everything this one does, one of the two should not exist.
    pub what: &'static str,
    /// Viewport. The default [`SIZE`] unless the mark is too big to fit it.
    pub view: Extent2,
    /// Everything the canvas needs before the stroke — ground, undercoat, surface,
    /// media params, imported assets — returning the brush to paint with.
    ///
    /// One closure rather than a pile of `Option` fields: what a case needs laid down
    /// first is usually inseparable from the brush that will work on it (a smear needs
    /// something to smear; a toothed tip needs a ground with a weave), and splitting
    /// them invites a case that asks for one without the other.
    ///
    /// The `Vec2` is where the case is being drawn — added to **every canvas
    /// coordinate this lays down**, so that the whole configuration can be moved and
    /// still be the same picture (the battery's translation check). A closure that
    /// lays nothing ignores it;
    /// one that lays an undercoat must not, or the translated case would
    /// smear a stroke over a field that stayed behind, which is a different picture
    /// and would say nothing about tile boundaries.
    pub prepare: fn(&mut Engine, Vec2) -> BrushParams,
    /// The pointer reports, as the app would deliver them.
    pub path: fn() -> Vec<InputSample>,
    pub tol: Tol,
}

impl Case {
    /// A prepared engine and the case's brush, or `None` if this machine has no
    /// adapter and skipping is permitted.
    pub fn open(&self) -> Option<(Engine, BrushParams)> {
        self.open_at(Vec2::ZERO)
    }

    /// [`open`](Self::open) with the whole case laid down `offset` further along, and
    /// the **view moved to match** so the same mark falls on the same screen pixels.
    ///
    /// Both halves together, because either alone is a different question: moving the
    /// content without the view asks what the mark looks like somewhere else, and
    /// moving the view without the content asks what the *viewport* does at a
    /// fractional tile. What this asks is whether the tile grid the mark happens to
    /// land on can be seen at all.
    pub fn open_at(&self, offset: Vec2) -> Option<(Engine, BrushParams)> {
        let mut engine = engine_or_skip_sized(self.view)?;
        let brush = (self.prepare)(&mut engine, offset);
        let center = engine.view().center + offset;
        engine.process(ViewCommand::CenterOn(center));
        Some((engine, brush))
    }

    pub fn samples(&self) -> Vec<InputSample> {
        (self.path)()
    }

    /// The case's pointer reports, every one of them `offset` further along — the
    /// stroke's half of what [`open_at`](Self::open_at) does for the canvas.
    pub fn samples_at(&self, offset: Vec2) -> Vec<InputSample> {
        self.samples()
            .into_iter()
            .map(|s| InputSample {
                pos: s.pos + offset,
                ..s
            })
            .collect()
    }

    /// Paint the whole stroke and commit it, at a chosen input tolerance.
    pub fn paint(&self, engine: &mut Engine, b: BrushParams, tolerance: f32) {
        self.paint_input(engine, b, tolerance, &self.samples());
    }

    /// [`paint`](Self::paint) with the pointer reports supplied rather than taken from
    /// the case — for the checks that ask what a *different delivery* of the same
    /// gesture draws (see [`lifted`]).
    pub fn paint_input(
        &self,
        engine: &mut Engine,
        b: BrushParams,
        tolerance: f32,
        samples: &[InputSample],
    ) {
        engine.process(ViewCommand::SetBrush(b));
        let (first, rest) = samples.split_first().expect("a case draws something");
        engine.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: *first,
            tolerance,
            rope: 0.0,
        });
        for s in rest {
            engine.process(GestureCommand::To { sample: *s });
        }
        engine.process(GestureCommand::End);
    }
}

// --- shared ingredients -------------------------------------------------------

/// A brush that runs the **sequential stamp loop** rather than the swept fast path: it
/// lifts paint off the canvas onto the tool and lays it back down, so what it draws at
/// any point depends on everywhere it has already been (§6.2). That carried state is
/// why the loop deserves its own half of the corpus — the swept deposit composes
/// freely, so cutting its path anywhere is trivially safe, while the loop is only
/// cuttable because `gpu::stroke::ToolState` remembers the reservoir and the pickup
/// cadence across the cut.
pub fn smear_brush(radius: f32) -> BrushParams {
    let mut b = brush(RED, radius);
    b.dynamics.lift = 0.6;
    b.dynamics.deposit = 0.5;
    b.dynamics.add = 0.5;
    b.drain = 0.0;
    b
}

/// Two crossing bands for a smear brush to pick up. A `lift` brush over bare canvas
/// carries nothing and would exercise none of the reservoir.
///
/// Laid at `at`, like everything else the case puts down, so that a translated case
/// smears the *same* field rather than a field that stayed where it was.
fn undercoat(engine: &mut Engine, at: Vec2) {
    for (color, from, to) in [
        ([0.1, 0.9, 0.2, 1.0], (-110.0, -40.0), (110.0, -10.0)),
        ([0.95, 0.85, 0.1, 1.0], (-90.0, 50.0), (100.0, 20.0)),
    ] {
        let mut b = brush(color, 26.0);
        b.drain = 0.0;
        stroke_with(
            engine,
            b,
            &[at + Vec2::new(from.0, from.1), at + Vec2::new(to.0, to.1)],
        );
    }
}

/// Put the linen weave on the canvas and light it, for the cases about tooth (§6.4).
/// `surface_strength` defaults to 0, which leaves the relief there for paint to sit in
/// but keeps the light from embossing it — so a case about the grain has to ask.
fn linen_ground(engine: &mut Engine) {
    let linen = engine
        .import_surface(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    engine.process(DocCommand::SetSurface(linen));
    engine.process(ViewCommand::SetMediaParams(stark_engine::MediaParams {
        surface_strength: 0.6,
        ..Default::default()
    }));
}

fn bristle_stamp(engine: &mut Engine, radius: f32) -> BrushParams {
    let id = engine
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import brush shape");
    let mut b = brush(RED, radius);
    b.shape = BrushShape::Stamp(id);
    b.drain = 0.0;
    b
}

fn at(pts: &[(f32, f32)]) -> Vec<InputSample> {
    pts.iter()
        .map(|&(x, y)| InputSample::at(Vec2::new(x, y)))
        .collect()
}

/// A recorded stroke, moved so its bounding box is centred on the canvas origin.
///
/// **Centred rather than used as captured, and this is not a nicety.** The recordings
/// in `stark_testdata` carry the canvas coordinates the pen was actually at — `LOOP`
/// sits around `y = 980`, `C` around `x = 1310` — while a test viewport shows the 256
/// px about the origin. Using one raw draws the whole stroke off-screen, and *every
/// check in the battery still passes*, because bare paper previews as bare paper,
/// round-trips exactly, and converges perfectly. Five tests in `stroke.rs` were
/// passing that way when this corpus was written: they had been comparing blank
/// renders for as long as they had existed.
///
/// So the placement is computed from the data instead of being a constant somebody has
/// to keep true, and `every_case_leaves_a_mark` in the battery is the backstop that
/// makes a silently-empty case a failure rather than a pass.
fn centred(pts: &[[f32; 2]]) -> Vec<InputSample> {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    pts.iter()
        .map(|&[x, y]| InputSample::at(Vec2::new(x - mid[0], y - mid[1])))
        .collect()
}

/// How many pointer reports a pen leaving the tablet is modelled as, and how far the
/// nib slides across the glass while it happens.
///
/// Both are what a tablet actually delivers rather than a worst case picked to fail:
/// a pen at 133 Hz spends 40-60 ms coming out of range, and the hand rolls the nib a
/// fraction of a pixel over that. **Under one canvas px of travel carrying the whole
/// pressure range** is the shape of the thing, and it is the shape rather than either
/// number that matters — a release is an attribute change with no path under it.
const LIFT_REPORTS: usize = 8;
const LIFT_DRIFT: f32 = 0.6;

/// The same gesture, with the **pen lifted off the tablet at the end of it**.
///
/// A tablet does not stop reporting when the hand starts to leave: it keeps sampling
/// through the release, and those last reports carry the pressure down to zero across
/// [`LIFT_DRIFT`] px of nib travel. So they are, in the engine's own terms, a run of
/// segments the swept integral must integrate over nothing — §6.2's deposit is a
/// definite integral over travel, and there is none here.
///
/// The tail continues in the direction the nib was last moving, because that is what a
/// hand coming off does; a click has no such direction and gets `+x`, which cannot
/// matter for the same reason `gpu::stroke::segments::sample_at` gives — there is
/// nothing to lead in.
pub fn lifted(samples: &[InputSample]) -> Vec<InputSample> {
    lift_tail(samples, true)
}

/// **The control for [`lifted`]**: the identical tail, with the nib still down.
///
/// Six-tenths of a pixel is not nothing, and the fit is entitled to notice it — the arc
/// length it divides by grows, the endpoint it pins moves, and on a path described by
/// three or four reports that re-fits the whole curve by a fraction of a pixel, which a
/// hard-edged stroke answers with more than the battery's twelve levels along its whole
/// length. Measured on `curve`, that alone is 10.3% of the viewport, and on `wide_smear`
/// 9.6%. It is also not the defect: what is being claimed is about a **release**, not
/// about travel, and the two are only separable by holding one of them still.
///
/// So the battery compares a release against this rather than against the untailed
/// stroke, and what survives the comparison is exactly what the pen letting go cost.
pub fn held_down(samples: &[InputSample]) -> Vec<InputSample> {
    lift_tail(samples, false)
}

fn lift_tail(samples: &[InputSample], release: bool) -> Vec<InputSample> {
    let mut out = samples.to_vec();
    let last = *out.last().expect("a case draws something");
    let dir = out
        .iter()
        .rev()
        .find_map(|s| {
            let v = last.pos - s.pos;
            let len = v.length();
            (len > 1e-3).then(|| v / len)
        })
        .unwrap_or(Vec2::new(1.0, 0.0));
    for k in 1..=LIFT_REPORTS {
        let f = k as f32 / LIFT_REPORTS as f32;
        out.push(InputSample {
            pos: last.pos + dir * (LIFT_DRIFT * f),
            pressure: if release {
                last.pressure * (1.0 - f)
            } else {
                last.pressure
            },
            // Held, not faded: a pen's tilt is where the hand is holding it and does
            // not go anywhere as the nib comes off. Only the pressure is leaving.
            tilt: last.tilt,
            time: last.time + f as f64 * 0.05,
        });
    }
    out
}

/// A half-turn arc: the tangent sweeps 180°, so how finely the path is cut *is* how
/// finely an oriented footprint rotates. Shared by the two cases that care which way
/// the tip is pointing.
fn half_turn_arc() -> Vec<InputSample> {
    let n = 48;
    (0..=n)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / n as f32);
            InputSample::at(Vec2::new(-70.0 * a.cos(), 70.0 * a.sin() - 20.0))
        })
        .collect()
}

/// A dense sine sweep — many samples over a smoothly turning path, which is what
/// makes the fitter freeze repeatedly rather than settling the whole stroke at once.
fn sine_sweep(n: usize, span: f32, amp: f32, turns: f32) -> Vec<InputSample> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            InputSample::at(Vec2::new(
                t * span - span * 0.5,
                (t * turns * std::f32::consts::TAU).sin() * amp,
            ))
        })
        .collect()
}

// --- the corpus ---------------------------------------------------------------

/// Every stroke the battery runs. Order is swept path first, then the stamp loop.
pub const CASES: &[Case] = &[
    Case {
        name: "line",
        what: "The floor. A soft round tip straight across, on the swept fast path with \
               nothing else switched on — so a failure here is the deposit itself rather \
               than any axis layered over it, and every other case can be read as this \
               plus one thing.",
        view: SIZE,
        prepare: |_, _| {
            let mut b = brush(RED, 40.0);
            b.drain = 0.0;
            b
        },
        path: || at(&[(-95.0, 0.0), (0.0, 0.0), (95.0, 0.0)]),
        // Zero, and it means it: a straight line is fitted exactly at any tolerance,
        // so refitting it four times finer leaves the curve where it was and moves
        // only where it is cut into segments. Nothing at all may move as a result —
        // which is §6.2's additivity in `τ` as directly as this suite can state it.
        tol: Tol {
            golden: 6,
            seam: 4,
            refine: 0.0,
            lift: 0.0,
        },
    },
    Case {
        name: "curve",
        what: "Curvature: a coarse zigzag the fitter must round into a smooth curve, \
               then flatten into swept arcs. The only cover for `max_arc_curvature` \
               and for the corner-preserving half of the angle bound.",
        view: SIZE,
        prepare: |_, _| {
            let mut b = brush(RED, 18.0);
            b.drain = 0.0;
            b
        },
        path: || {
            at(&[
                (-90.0, 40.0),
                (-45.0, -50.0),
                (0.0, 40.0),
                (45.0, -50.0),
                (90.0, 40.0),
            ])
        },
        tol: Tol {
            golden: 6,
            seam: 4,
            refine: 0.0,
            lift: 0.0,
        },
    },
    Case {
        name: "hairpin",
        what: "Real pen input rather than synthetic: 305 reports over 334 px, slow and \
               dense enough that the fit visibly struggles. Nothing generated from a \
               formula has this stroke's sample spacing, and the fitter's freeze cadence \
               answers to spacing.",
        view: SIZE,
        prepare: |_, _| {
            let mut b = brush(RED, 16.0);
            b.drain = 0.0;
            b
        },
        path: || centred(stark_testdata::HAIRPIN_STROKE),
        tol: Tol {
            golden: 6,
            seam: 8,
            // The loosest of the swept cases, and the reason is this case's whole
            // point: its spacing runs from 0.37 px to over 1 px, so it is the stroke
            // where `arc_weights` does the most work, and which samples fall in the
            // solve window — and therefore what the weights normalize against — moves
            // with the tolerance. 0.8 before the weights existed, 1.36 measured after.
            refine: 1.5,
            lift: 0.0,
        },
    },
    Case {
        name: "pressure_ramp",
        what: "A pressure ramp on a swept stroke — the radius sweeping 10% to full and \
               back through `Modulations::PRESSURE_SIZE`. The only cover for the \
               flattener's *attribute* bound, which is the one that keeps a ramp from \
               drawing as a staircase of radii; sweeping it over ×0.125…×8 once left \
               every committed golden bit-identical, because nothing pressed the pen.",
        view: SIZE,
        prepare: |_, _| {
            let mut b = brush(RED, 46.0);
            b.drain = 0.0;
            b
        },
        path: || {
            (0..=60)
                .map(|i| {
                    let t = i as f32 / 60.0;
                    InputSample {
                        pos: Vec2::new(t * 200.0 - 100.0, 0.0),
                        // Straight, and **linear** in pressure, so the fit is exact in
                        // both channels at any tolerance: a cubic B-spline reproduces
                        // a line, and the curvature penalty has nothing to pull on.
                        // A sine ramp was tried first and moves 5% of the viewport
                        // under a finer fit — all of it the fitted *pressure curve*
                        // shifting, none of it about the attribute budget, which is
                        // the thing this case is here for. What covers the budget is
                        // the golden: relax it and the ramp draws as a staircase of
                        // radii, and until this case existed no golden in the suite
                        // had a pressure ramp in it to notice.
                        pressure: 0.1 + 0.9 * t,
                        ..InputSample::default()
                    }
                })
                .collect()
        },
        tol: Tol {
            golden: 6,
            seam: 6,
            // Not zero, and it was — this is the one bound the attribute end condition
            // costs (`PathFitter::solve`). The curve's last control value is held at
            // its neighbour rather than pinned to the last report, so the final span
            // does not complete the ramp; where that neighbour sits depends on the knot
            // count, and the knot count moves with the tolerance. That is the whole
            // 0.067% of it. A linear ramp is still fitted exactly everywhere the end
            // condition does not reach, and giving up the last sliver of one is what
            // buys a stroke that survives the pen coming off it (`Tol::lift`).
            refine: 0.1,
            lift: 0.0,
        },
    },
    Case {
        name: "taper",
        what: "Both tapers, on recorded input long enough to freeze several times. A \
               taper is measured from the ends of the *whole* stroke, so while the \
               pointer is down the far end has not been drawn yet — this is the only \
               cover for the freeze hold-back (`gpu::stroke::safe_frozen`), and a taper \
               baked into a span too early leaves a pinch that is never redrawn.",
        view: SIZE,
        prepare: |_, _| {
            let mut b = brush(RED, 16.0);
            b.shape = BrushShape::Round { hardness: 0.9 };
            b.drain = 0.0;
            b.dynamics.add = 1.0;
            b.start_taper_length = 4.0;
            b.end_taper_length = 9.0;
            b
        },
        path: || centred(stark_testdata::LOOP_STROKE),
        tol: Tol {
            golden: 6,
            seam: 8,
            refine: 4.0,
            lift: 0.0,
        },
    },
    // The "dab" case — a click, once rescued by `segments::DAB_TRAVEL`'s
    // fabricated minimum travel — retired with the dwell itself: a swept deposit
    // is a definite integral over travel, and a press that has not moved now
    // honestly lays nothing and commits nothing (`tests/stroke.rs` holds the new
    // truth), while this corpus asserts marks.
    Case {
        name: "stamp_arc",
        what: "An anisotropic stamp around a half turn, orientation following the \
               stroke — so how finely the path is cut *is* how finely the footprint \
               rotates. With `tooth_arc`, the pair the suite was completely blind to: \
               nothing else puts an orientable footprint on the dynamics path around a \
               curve, and loosening the tangent bound to what `position` allows moves \
               this by a worst 19 levels with every committed golden bit-identical.",
        view: SIZE,
        prepare: |e, _| {
            let mut b = bristle_stamp(e, 45.0);
            b.dynamics.lift = 0.6;
            b.dynamics.deposit = 0.5;
            b.dynamics.add = 0.5;
            b
        },
        path: half_turn_arc,
        tol: Tol {
            golden: 6,
            seam: 12,
            // The loosest of the arcs, and it is the footprint that spends it: the mask
            // is rotated once per segment, so a finer cut *re-orients* it rather than
            // merely re-sampling it, and a hard-edged bristle answers along its whole
            // rim. `tooth_arc` converges twice as tightly on the same curve with a disc.
            refine: 0.25,
            lift: 0.0,
        },
    },
    Case {
        name: "pen_stamp",
        what: "The same anisotropic mask pinned to the **pen's** tilt azimuth instead of \
               to the tangent, while the stroke bends through 90°. Travel direction and \
               footprint orientation diverge, which is the whole point of \
               `OrientationSource` and is a different code path from following the \
               stroke, not a parameter of it.",
        view: SIZE,
        prepare: |e, _| {
            let mut b = bristle_stamp(e, 60.0);
            b.orientation = OrientationSource::Pen;
            b
        },
        path: || {
            // A fixed 45° tilt azimuth on every sample, while the path turns.
            [(-80.0, -50.0), (0.0, -50.0), (60.0, -10.0), (60.0, 70.0)]
                .iter()
                .map(|&(x, y)| InputSample {
                    pos: Vec2::new(x, y),
                    tilt: Vec2::new(1.0, 1.0),
                    ..InputSample::default()
                })
                .collect()
        },
        tol: Tol {
            golden: 6,
            seam: 6,
            refine: 0.0,
            lift: 0.0,
        },
    },
    Case {
        name: "tooth_arc",
        what: "A round tip on a woven ground around the same half turn — the \
               counter-intuitive case, and the one the suite had no cover for at all. A \
               disc has no orientation, so the intuition is that it cannot care which \
               way it travels; it can, because `rise_ahead` reads the weave's gradient \
               *along* the segment (§6.4), so the grain a stroke prints turns with the \
               cut. Loosening the tangent bound to what `position` allows moves this by \
               a worst 50 levels, which is what the coverage claim is worth.",
        view: SIZE,
        prepare: |e, _| {
            linen_ground(e);
            let mut b = smear_brush(45.0);
            b.shape = BrushShape::Round { hardness: 0.9 };
            b.tooth = 0.9;
            b
        },
        path: half_turn_arc,
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 0.10,
            lift: 0.0,
        },
    },
    Case {
        name: "smear",
        what: "The stamp loop at ordinary size: lift and deposit over an undercoat, on \
               a path that freezes many times. The base case for everything sequential — \
               each segment reads the canvas the previous one left and the tool the \
               previous one loaded, so a cut that dropped the reservoir shows here as a \
               step in color trailing back across the stroke.",
        view: SIZE,
        prepare: |e, at| {
            undercoat(e, at);
            smear_brush(15.0)
        },
        path: || sine_sweep(140, 124.0, 34.0, 1.3),
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 0.06,
            lift: 0.0,
        },
    },
    Case {
        name: "smear_taper",
        what: "A taper *on* the stamp loop, which reaches it twice over: the taper cuts \
               finer segments near the ends, and the reservoir reloads every \
               `RESERVOIR_CADENCE · radius` of travel — a radius the taper is shrinking. \
               So both how many segments the loop walks and how often it exchanges with \
               the canvas change inside a taper, and head and commit must still walk \
               exactly the same sequence.",
        view: SIZE,
        prepare: |e, at| {
            undercoat(e, at);
            let mut b = smear_brush(15.0);
            b.start_taper_length = 4.0;
            b.end_taper_length = 9.0;
            b
        },
        path: || centred(stark_testdata::C_STROKE),
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 2.0,
            lift: 0.0,
        },
    },
    Case {
        name: "bleed",
        what: "Lateral diffusion — the one flux internal to the canvas, where the tool \
               neither takes nor gives. It fires on its own cadence rather than per \
               segment (`budget::bleed_stencil`), so it is the axis a per-segment \
               argument about the stamp loop says nothing about.",
        view: SIZE,
        prepare: |e, at| {
            undercoat(e, at);
            let mut b = brush(RED, 30.0);
            b.drain = 0.0;
            b.dynamics.add = 0.0;
            b.dynamics.bleed = 0.8;
            b
        },
        path: || sine_sweep(80, 180.0, 30.0, 0.75),
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 0.0,
            lift: 0.0,
        },
    },
    Case {
        name: "wide_smear",
        what: "A smear at a tip far wider than anything else here draws — the regime \
               nothing else reaches. Everything the loop's cost and resolution are \
               quoted in scales with the radius: the exchange cadence, the reservoir's \
               own extent, and how many segments a turn is cut into. A curved smear \
               pins the result directly, since an arc's inner and outer edges are pure \
               along-travel structure. Radius 250.",
        view: Extent2 {
            width: 768,
            height: 512,
        },
        prepare: |e, at| {
            // A thick field to smear, laid flat so the smear's own structure is what
            // shows. Wider than the tip that will work it, so the mark stays inside it.
            let mut lay = brush([0.0, 0.0, 0.0, 1.0], 260.0);
            lay.dynamics.add = 1.5;
            lay.drain = 0.0;
            stroke_with(
                e,
                lay,
                &[at + Vec2::new(-260.0, 0.0), at + Vec2::new(260.0, 0.0)],
            );
            let mut b = smear_brush(250.0);
            b.shape = BrushShape::Round { hardness: 0.9 };
            b.dynamics.add = 0.0;
            b.dynamics.lift = 0.9;
            b.dynamics.deposit = 0.9;
            b
        },
        path: || at(&[(-300.0, -120.0), (0.0, 120.0), (300.0, -120.0)]),
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 0.0,
            // **The only case the release still moves**, and it is the arithmetic of a
            // three-report path rather than anything about tablets. Eight release
            // reports against three real ones leaves the solve one free row with the
            // release as the nearest data to it, so the weight that protects every
            // other case has nothing to outvote it *with*; and at a 250 px tip the
            // sliver of radius that buys shows up along the whole rim. Real pen input
            // does not arrive three reports at a time — every recorded stroke in
            // `stark_testdata` is cured to the bit — and a mouse, which does, reports
            // no pressure to release.
            lift: 0.7,
        },
    },
    Case {
        name: "oversized_smear",
        what: "A smear whose bounding box outgrows one region, so the commit runs it in \
               several pieces while a live range runs each in one. The pieces are cut \
               where the *region* fills up, which has nothing to do with where the \
               fitter freezes — so this is the only case where the two renders cut the \
               same stroke in genuinely different places.",
        view: Extent2 {
            width: 1280,
            height: 256,
        },
        prepare: |e, at| {
            let mut lay = brush([0.1, 0.9, 0.2, 1.0], 30.0);
            lay.drain = 0.0;
            let band: Vec<Vec2> = (0..220)
                .map(|i| {
                    let t = i as f32 / 219.0;
                    at + Vec2::new(t * 2560.0 - 2000.0, (t * 26.0).sin() * 40.0)
                })
                .collect();
            stroke_with(e, lay, &band);
            smear_brush(15.0)
        },
        path: || {
            (0..220)
                .map(|i| {
                    let t = i as f32 / 219.0;
                    InputSample::at(Vec2::new(t * 2560.0 - 2000.0, (t * 26.0).sin() * 40.0))
                })
                .collect()
        },
        tol: Tol {
            golden: 6,
            seam: 12,
            refine: 0.12,
            lift: 0.0,
        },
    },
];
