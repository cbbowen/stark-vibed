//! Gradients: a color ramp fitted from a line traced through the painting (§22).
//!
//! A gradient here is a small list of positioned color stops, interpolated in
//! **Oklab** so a ramp between two colors passes through the colors an artist
//! would mix on the way (§1.6). Stops store straight sRGB — the same convention
//! as every other color on a CPU boundary (`PaintEffect::color`, §6.5) — and
//! convert to Oklab only to interpolate, so a stop round-trips through the
//! picker and the library unchanged.
//!
//! The type exists to be *captured*, not authored point by point: the artist
//! traces a line through paint they have already mixed, the engine samples
//! colors along it (`stark-engine`'s `Engine::pick_gradient`,
//! §22.2), and [`fit`] reduces those samples to the fewest stops that still
//! reproduce the ramp within a perceptual tolerance. Placing and color-picking
//! control points by hand remains possible in principle — a `Gradient` is just
//! stops — but the trace is the front door.
//!
//! Nothing in here touches the document: a gradient library is something the
//! artist paints *with*, like brush presets, so it lives with the frontend
//! (§22.3). The type sits in core because the capture pipeline's fitting is
//! engine work, and because a position-varying `FillOp` will embed it when the
//! gradient fill lands at the seam §18.0.4 names.

use crate::Srgb;
use crate::color::{oklab_to_srgb, srgb_to_oklab};
use crate::geom::Vec2;

/// One color stop: a position along the ramp and the color there.
#[derive(
    Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, carbonite::Schema,
)]
pub struct GradientStop {
    /// Position in `[0,1]` along the ramp.
    pub t: f32,
    /// The color there. No alpha: a captured ramp is made of paint the canvas
    /// actually holds, and what a fill does with coverage is the fill's parameter,
    /// not the gradient's.
    pub color: Srgb,
}

/// A color ramp: at least two stops, positions ascending, endpoints at 0 and 1.
///
/// The invariants are held by construction on both doors, which answer a list that
/// names no ramp differently and deliberately:
///
/// - [`new`](Self::new) **refuses**. The authoring path: a caller who traced a line
///   needs to hear that the samples describe one (§22.2).
/// - `Deserialize`/`Schema` **repair**, through `From<Vec<GradientStop>>`. A refusal
///   here would take the whole document with it — a ramp sits inside a `Parcel`, a
///   `Filter::GradientMap` and a [`FillOp`](crate::document::FillOp)'s parcel, where it
///   would be refused *before* that op's own clamp could run — and §19 admits only two
///   refusals on the load path, neither of them a malformed ramp.
///
/// So a condition added here — monotonic lightness, say — can only arrive as a repair,
/// since refusing it would retroactively unload files that were valid when saved.
/// [`MAX_STOPS`] is the one added so far, which is why it thins rather than refuses,
/// on both doors alike.
///
/// The stop list is the wire shape in both directions: `carbonite(as)` states the
/// schema as `Vec<GradientStop>`, so nothing drives the conversion to find out what the
/// type looks like (§8). Being infallible, it also leaves `carbonite::compat`'s probe a
/// verdict to give about every type a ramp sits inside — the probe writes one-element
/// sequences, and a funnel that refused one returned no answer for `Action` either.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, carbonite::Schema)]
#[serde(from = "Vec<GradientStop>", into = "Vec<GradientStop>")]
#[carbonite(as = "Vec<GradientStop>")]
pub struct Gradient {
    stops: Vec<GradientStop>,
}

impl Gradient {
    /// Build a gradient from stops, normalizing to the invariants: sorted by
    /// `t`, at most [`MAX_STOPS`] of them, positions rescaled so the first sits
    /// at 0 and the last at 1. `None` if fewer than two stops arrive, if a
    /// position is non-finite, or if every stop sits at one position (there is
    /// no ramp to rescale).
    ///
    /// **Only the positions are checked**, because only the positions can be wrong:
    /// an [`Srgb`] is finite and bounded by construction.
    ///
    /// **The authoring door, and it refuses**: a caller that traced a line expecting a
    /// ramp needs to hear "these samples do not describe one" (§22.2). A file carrying
    /// the same list is repaired instead — see [`Gradient`].
    ///
    /// **Except for the count, which is thinned rather than refused.** Every other
    /// condition here is about a list that names no ramp at all; a long list names one
    /// perfectly well and simply names more of it than a uniform can hold, so refusing
    /// would unload a file over something this can answer.
    pub fn new(mut stops: Vec<GradientStop>) -> Option<Self> {
        if stops.len() < 2 {
            return None;
        }
        if !stops.iter().all(|s| s.t.is_finite()) {
            return None;
        }
        stops.sort_by(|a, b| a.t.total_cmp(&b.t));
        thin(&mut stops);
        let (lo, hi) = (stops[0].t, stops[stops.len() - 1].t);
        if hi <= lo {
            return None;
        }
        for s in &mut stops {
            s.t = (s.t - lo) / (hi - lo);
        }
        Some(Self { stops })
    }

    /// The stops, ascending, endpoints at 0 and 1.
    pub fn stops(&self) -> &[GradientStop] {
        &self.stops
    }

    /// The color at `t` (clamped to `[0,1]`), interpolated **in Oklab** between
    /// the surrounding stops — the same interpolation CSS's `in oklab` performs,
    /// so a frontend strip previews exactly what this returns. Straight sRGB out.
    pub fn sample(&self, t: f32) -> Srgb {
        let t = t.clamp(0.0, 1.0);
        let after = self.stops.partition_point(|s| s.t <= t);
        let (a, b) = match after {
            0 => return self.stops[0].color,
            n if n == self.stops.len() => return self.stops[n - 1].color,
            n => (&self.stops[n - 1], &self.stops[n]),
        };
        // Two stops sharing a position are a hard edge; standing exactly on it
        // reads the earlier side, which the `<=` in the partition already chose.
        let span = b.t - a.t;
        if span <= 0.0 {
            return a.color;
        }
        let f = (t - a.t) / span;
        // Standing on a stop returns the stop, bit for bit — a color must not
        // pick up conversion dust just by being read back off the ramp.
        if f <= 0.0 {
            return a.color;
        }
        if f >= 1.0 {
            return b.color;
        }
        let la = to_lab(a.color.get());
        let lb = to_lab(b.color.get());
        from_lab([
            la[0] + (lb[0] - la[0]) * f,
            la[1] + (lb[1] - la[1]) * f,
            la[2] + (lb[2] - la[2]) * f,
        ])
    }

    /// A two-stop ramp, `a` at 0 and `b` at 1. Normalized as written, so it needs no
    /// pass through [`new`](Self::new) — which could not refuse it anyway.
    fn ends(a: Srgb, b: Srgb) -> Self {
        Self {
            stops: vec![
                GradientStop { t: 0.0, color: a },
                GradientStop { t: 1.0, color: b },
            ],
        }
    }

    /// The same ramp run the other way: `reversed().sample(t) == sample(1 - t)`.
    ///
    /// A captured gradient runs in whatever direction the hand traced, which is
    /// no direction at all until a consumer gives `t` a meaning — and the
    /// gradient map (§21.11) gives it one, dark at 0. Reversing is the one-click
    /// answer to a trace made the other way; without it the fix is re-tracing
    /// the same line backwards.
    pub fn reversed(&self) -> Self {
        let stops = self
            .stops
            .iter()
            .rev()
            .map(|s| GradientStop {
                t: 1.0 - s.t,
                color: s.color,
            })
            .collect();
        // Direct construction rather than `new`: reversing preserves every
        // invariant (ascending order flips with the iteration, the endpoints
        // swap), and the funnel's rescale would be a no-op it could not refuse.
        Self { stops }
    }
}

impl From<Gradient> for Vec<GradientStop> {
    fn from(g: Gradient) -> Self {
        g.stops
    }
}

/// The load path's door, and it repairs (§19): whatever a file, a peer or a stored
/// library entry carries comes back as a ramp.
///
/// The normalization is [`Gradient::new`]'s — same sort, same thinning, same rescale —
/// so a ramp that arrives well-formed is the one `new` would have built, bit for bit.
/// Only the three shapes `new` calls no ramp at all are answered here rather than
/// refused.
impl From<Vec<GradientStop>> for Gradient {
    fn from(mut stops: Vec<GradientStop>) -> Self {
        // A non-finite position names no place on the ramp, and any place chosen for it
        // would be one the file did not write — so the stop goes, and what is left is
        // repaired by the rules below.
        stops.retain(|s| s.t.is_finite());
        // The color a pile of stops at one position samples as (`sample` reads the
        // earlier side of a hard edge), and the one thing a lone stop carries that
        // black→white would discard.
        let held = stops.first().map(|s| s.color);
        match (Gradient::new(stops), held) {
            (Some(g), _) => g,
            (None, Some(c)) => Self::ends(c, c),
            // Nothing arrived, so there is nothing to preserve: black→white is the ramp
            // that states no choice — the identity of the gradient map, dark at 0
            // (§22.5).
            (None, None) => Self::ends(Srgb::BLACK, Srgb::WHITE),
        }
    }
}

/// Reduce a sorted stop list to at most [`MAX_STOPS`], keeping both endpoints and
/// spreading the rest evenly across the ramp — [`Gradient::new`]'s one repair.
///
/// **Evenly across the list, not across `t`.** A capture puts stops where the ramp
/// *turns* ([`fit`]), so index spacing is already a rough measure of where the
/// structure is; thinning by position instead would spend the budget on whichever
/// stretch happened to be long. Keeping the ends is what makes this safe to run
/// before the rescale below it: `lo` and `hi` are the same stops either way, so a
/// thinned ramp still spans exactly the range the full one did.
///
/// Consecutive indices are strictly increasing whenever this runs at all
/// (`len > MAX_STOPS` makes the step `(len - 1)/(MAX_STOPS - 1) > 1`), so no stop is
/// picked twice and the result cannot collapse.
fn thin(stops: &mut Vec<GradientStop>) {
    let len = stops.len();
    if len <= MAX_STOPS {
        return;
    }
    // Through `pick_index` for the reason the lasso's decimation is: `usize` is 32
    // bits in the browser. Safe by size alone here — the product needs 286M stops to
    // overflow — but there is no reason for the crate's two decimations to be spelled
    // differently.
    *stops = (0..MAX_STOPS)
        .map(|i| stops[crate::pick_index(i, len - 1, MAX_STOPS - 1)])
        .collect();
}

fn to_lab(srgb: [f32; 3]) -> [f32; 3] {
    let l = srgb_to_oklab([srgb[0], srgb[1], srgb[2], 1.0]);
    [l[0], l[1], l[2]]
}

fn from_lab(lab: [f32; 3]) -> Srgb {
    let s = oklab_to_srgb([lab[0], lab[1], lab[2], 1.0]);
    // A lerp between two stops can leave the sRGB cube (Oklab is wider), and that
    // is **kept**: extended sRGB is what a color is (§6.5), so the excursion is a
    // wide color rather than an error. The constructor holds only the bound.
    Srgb::new([s[0], s[1], s[2]])
}

/// How far apart, in canvas px, the capture samples a traced path (§22.2).
/// Comparable to the 5×5 patch each sample averages, so consecutive patches
/// overlap and the ramp cannot skip a color narrower than a patch.
pub const SAMPLE_SPACING: f32 = 4.0;

/// Most samples one capture may take. A bound on what a trace may cost, the way
/// `MAX_PICK_RADIUS` bounds one sample: each sample is a rendered patch and a
/// readback slot, so an unbounded trace would be an unbounded render.
pub const MAX_SAMPLES: usize = 128;

/// Largest stop count a [`Gradient`] may hold. Sixteen is far more structure than
/// a hand places; past it [`fit`] lets the tolerance give rather than the list
/// grow, and [`Gradient::new`] thins what arrives some other way.
///
/// **An invariant of the type, not a budget of the fitter**, and the difference is
/// what a shader does with the number. Every consumer copies the stops into a fixed
/// array of exactly this length — `fill.wesl`'s `stop_c`, `matte.wesl`'s `Ramp`,
/// `filter_common.wesl`'s gradient map, each asserted equal to this constant
/// host-side (§6.10) — so the constant, the shader and *the data* have to agree. A
/// `const` assert can pin the first two but not the data, which is why the bound is
/// in the constructor.
pub const MAX_STOPS: usize = 16;

/// The fitting tolerance: the largest Oklab distance allowed between the
/// smoothed samples and the fitted ramp. Oklab's L spans `[0,1]`, where ~0.01
/// is around a just-noticeable difference — so the fitted gradient is the
/// traced one to the eye, with stops only where the ramp genuinely turns.
pub const FIT_TOLERANCE: f32 = 0.01;

/// Resample a traced polyline evenly by arc length: `(t, position)` pairs with
/// `t` the arc-length fraction in `[0,1]`. Returns an empty vec for a path with
/// no length. Spacing aims at [`SAMPLE_SPACING`] canvas px, clamped to
/// [`MAX_SAMPLES`]; both ends are always sampled.
pub fn resample(path: &[Vec2]) -> Vec<(f32, Vec2)> {
    let mut cum = Vec::with_capacity(path.len());
    let mut total = 0.0f32;
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            total += p.distance(path[i - 1]);
        }
        cum.push(total);
    }
    if !total.is_finite() || total <= 0.0 {
        return Vec::new();
    }
    let n = ((total / SAMPLE_SPACING).ceil() as usize + 1).clamp(2, MAX_SAMPLES);
    let mut out = Vec::with_capacity(n);
    let mut seg = 0usize;
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let want = t * total;
        while seg + 1 < cum.len() && cum[seg + 1] < want {
            seg += 1;
        }
        let span = cum[seg + 1] - cum[seg];
        let f = if span > 0.0 {
            (want - cum[seg]) / span
        } else {
            0.0
        };
        out.push((t, path[seg].lerp(path[seg + 1], f.clamp(0.0, 1.0))));
    }
    out
}

/// Fit a gradient to colors sampled along a trace: `(t, straight sRGB)` pairs,
/// `t` ascending. The mechanical half of the capture (§22.2) — the artist
/// supplies the line, this supplies the control points.
///
/// Three stages, each with one job:
///
/// 1. **Median-of-3** per Oklab channel — a single outlier sample (the trace
///    nicking a dark line or a stray bristle track) becomes a stop under any
///    least-error criterion, and no artist means one sample of it.
/// 2. **Box-3 smoothing** — the 5×5 patch average each sample already carries
///    handles texel noise; this handles the sample-to-sample grain of paint
///    itself, so the fitter chases the ramp and not the tooth.
/// 3. **Greedy stop insertion** in Oklab: start with the endpoints, repeatedly
///    add the sample farthest from the current piecewise-linear ramp, stop when
///    the worst error drops under [`FIT_TOLERANCE`] or [`MAX_STOPS`] is
///    reached. Farthest-point insertion rather than a corner detector because
///    the criterion *is* the promise: nowhere along the trace does the fitted
///    ramp drift a visible distance from the paint.
///
/// `None` when fewer than two samples arrive or the positions do not span —
/// the same refusals as [`Gradient::new`], reached a step earlier.
pub fn fit(samples: &[(f32, [f32; 3])]) -> Option<Gradient> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len();
    let lab: Vec<[f32; 3]> = samples.iter().map(|(_, c)| to_lab(*c)).collect();

    // Median-of-3, endpoints kept verbatim: the ends are the colors the artist
    // deliberately started and finished on, not noise to vote away.
    let med: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            if i == 0 || i == n - 1 {
                return lab[i];
            }
            let mut m = [0.0f32; 3];
            for (ch, v) in m.iter_mut().enumerate() {
                let (a, b, c) = (lab[i - 1][ch], lab[i][ch], lab[i + 1][ch]);
                *v = a.max(b.min(c)).min(b.max(c));
            }
            m
        })
        .collect();
    let smooth: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            if i == 0 || i == n - 1 {
                return med[i];
            }
            let mut s = [0.0f32; 3];
            for (ch, v) in s.iter_mut().enumerate() {
                *v = (med[i - 1][ch] + med[i][ch] + med[i + 1][ch]) / 3.0;
            }
            s
        })
        .collect();

    let t = |i: usize| samples[i].0;
    let mut picked = vec![0, n - 1];
    loop {
        // Worst sample against the current ramp, measured in Oklab distance.
        let mut worst = (0usize, 0.0f32);
        for w in picked.windows(2) {
            let (a, b) = (w[0], w[1]);
            let span = t(b) - t(a);
            for i in a + 1..b {
                let f = if span > 0.0 {
                    (t(i) - t(a)) / span
                } else {
                    0.0
                };
                let mut d2 = 0.0f32;
                for ((sa, sb), si) in smooth[a].iter().zip(&smooth[b]).zip(&smooth[i]) {
                    let on = sa + (sb - sa) * f;
                    d2 += (si - on) * (si - on);
                }
                if d2 > worst.1 {
                    worst = (i, d2);
                }
            }
        }
        if worst.1.sqrt() <= FIT_TOLERANCE || picked.len() >= MAX_STOPS {
            break;
        }
        let at = picked.partition_point(|&i| i < worst.0);
        picked.insert(at, worst.0);
    }

    Gradient::new(
        picked
            .iter()
            .map(|&i| GradientStop {
                t: t(i),
                color: from_lab(smooth[i]),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stop(t: f32, color: [f32; 3]) -> GradientStop {
        GradientStop {
            t,
            color: Srgb::new(color),
        }
    }

    #[test]
    fn new_normalizes_and_refuses() {
        assert!(Gradient::new(vec![]).is_none());
        assert!(Gradient::new(vec![stop(0.5, [1.0; 3])]).is_none());
        assert!(Gradient::new(vec![stop(0.5, [1.0; 3]), stop(0.5, [0.0; 3])]).is_none());
        assert!(Gradient::new(vec![stop(f32::NAN, [1.0; 3]), stop(1.0, [0.0; 3])]).is_none());

        let g = Gradient::new(vec![stop(0.9, [1.0; 3]), stop(0.1, [0.0; 3])]).unwrap();
        assert_eq!(g.stops()[0].t, 0.0);
        assert_eq!(g.stops()[0].color, Srgb::BLACK);
        assert_eq!(g.stops()[1].t, 1.0);
    }

    #[test]
    fn sample_hits_stops_and_interpolates_in_oklab() {
        let a = [0.8, 0.2, 0.1];
        let b = [0.1, 0.3, 0.9];
        let g = Gradient::new(vec![stop(0.0, a), stop(1.0, b)]).unwrap();
        assert_eq!(g.sample(-1.0), Srgb::new(a));
        assert_eq!(g.sample(2.0), Srgb::new(b));

        // Midpoint = the Oklab midpoint, not the sRGB one.
        let la = srgb_to_oklab([a[0], a[1], a[2], 1.0]);
        let lb = srgb_to_oklab([b[0], b[1], b[2], 1.0]);
        let mid = oklab_to_srgb([
            (la[0] + lb[0]) / 2.0,
            (la[1] + lb[1]) / 2.0,
            (la[2] + lb[2]) / 2.0,
            1.0,
        ]);
        let got = g.sample(0.5);
        for ch in 0..3 {
            assert!((got[ch] - mid[ch]).abs() < 1e-5, "{got:?} vs {mid:?}");
        }
    }

    #[test]
    fn reversed_runs_the_same_ramp_the_other_way() {
        let g = Gradient::new(vec![
            stop(0.0, [0.1, 0.2, 0.3]),
            stop(0.3, [0.9, 0.5, 0.1]),
            stop(1.0, [0.2, 0.8, 0.6]),
        ])
        .unwrap();
        let r = g.reversed();
        // The invariants survive without the funnel: ascending, endpoints exact.
        assert_eq!(r.stops()[0].t, 0.0);
        assert_eq!(r.stops()[2].t, 1.0);
        // Stops land bit-exact — a color must not pick up dust by being turned
        // around — and interior positions mirror.
        assert_eq!(r.stops()[0].color, Srgb::new([0.2, 0.8, 0.6]));
        assert_eq!(r.stops()[2].color, Srgb::new([0.1, 0.2, 0.3]));
        assert!((r.stops()[1].t - 0.7).abs() < 1e-6);
        // And the ramp between them is the same ramp: r(t) == g(1 - t).
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let (a, b) = (r.sample(t), g.sample(1.0 - t));
            for ch in 0..3 {
                assert!((a[ch] - b[ch]).abs() < 1e-5, "t={t}: {a:?} vs {b:?}");
            }
        }
    }

    /// What a stored ramp can smuggle past the funnel: nothing. What it can be refused
    /// for: nothing either (§19).
    ///
    /// Driven through the *encoding*, since a stop list is exactly what a file carries
    /// and that is the path the funnel exists for. The per-case repairs are asserted
    /// below; this says the two things true of every case.
    #[test]
    fn deserialization_funnels_through_the_same_gate() {
        for stops in [
            vec![],
            vec![stop(0.5, [1.0; 3])],
            vec![stop(0.5, [0.0; 3]), stop(0.5, [1.0; 3])],
            vec![stop(f32::NAN, [0.0; 3]), stop(1.0, [1.0; 3])],
            vec![stop(0.0, [0.0; 3]), stop(1.0, [1.0; 3])],
        ] {
            let g = loaded(stops);
            let n = g.stops().len();
            assert!((2..=MAX_STOPS).contains(&n), "{n} stops");
            assert_eq!((g.stops()[0].t, g.stops()[n - 1].t), (0.0, 1.0));
            assert!(g.stops().windows(2).all(|w| w[0].t <= w[1].t));
        }
    }

    /// Decode a stop list the way a file does: bytes in, `Gradient` out.
    fn loaded(stops: Vec<GradientStop>) -> Gradient {
        let bytes = carbonite::to_vec_static(&stops).expect("encodes");
        carbonite::from_slice_static::<Gradient>(&bytes).expect("a stop list always loads")
    }

    /// **The load path repairs; it does not refuse** (§19).
    ///
    /// Each case asserts the ramp that comes back, not merely that one does — a
    /// repair is only worth the refusal it replaces if it is the *stated* one.
    #[test]
    fn an_empty_stop_list_loads_as_black_to_white() {
        let g = loaded(vec![]);
        assert_eq!(g.stops().len(), 2);
        assert_eq!((g.stops()[0].t, g.stops()[1].t), (0.0, 1.0));
        assert_eq!(g.stops()[0].color, Srgb::BLACK);
        assert_eq!(g.stops()[1].color, Srgb::WHITE);
    }

    #[test]
    fn a_lone_stop_loads_as_a_flat_ramp_of_its_own_color() {
        let c = Srgb::new([0.8, 0.2, 0.1]);
        let g = loaded(vec![stop(0.5, c.get())]);
        assert_eq!(
            g.stops(),
            &[
                GradientStop { t: 0.0, color: c },
                GradientStop { t: 1.0, color: c },
            ],
        );
    }

    #[test]
    fn stops_at_one_position_load_as_a_flat_ramp() {
        let c = Srgb::new([0.1, 0.6, 0.3]);
        let g = loaded(vec![stop(0.25, c.get()), stop(0.25, [0.9, 0.9, 0.9])]);
        assert_eq!(
            g.stops(),
            &[
                GradientStop { t: 0.0, color: c },
                GradientStop { t: 1.0, color: c },
            ],
        );
    }

    #[test]
    fn a_non_finite_position_drops_its_stop_and_the_rest_is_repaired() {
        let a = Srgb::new([1.0, 0.0, 0.0]);
        let b = Srgb::new([0.0, 0.0, 1.0]);
        let g = loaded(vec![
            stop(0.0, a.get()),
            stop(f32::NAN, [0.0; 3]),
            stop(1.0, b.get()),
            stop(f32::INFINITY, [0.0; 3]),
        ]);
        assert_eq!(
            g.stops(),
            &[
                GradientStop { t: 0.0, color: a },
                GradientStop { t: 1.0, color: b },
            ],
        );

        // And when dropping leaves too little to be a ramp, the rules above apply
        // to what is left rather than to what arrived.
        let lone = loaded(vec![stop(f32::NAN, [0.0; 3]), stop(0.5, a.get())]);
        assert_eq!(
            lone.stops(),
            &[
                GradientStop { t: 0.0, color: a },
                GradientStop { t: 1.0, color: a },
            ],
        );
    }

    /// **A well-formed ramp comes back bit for bit.** The load-bearing one: a repair
    /// that perturbed valid data would be worse than the refusal it replaced.
    #[test]
    fn a_well_formed_ramp_survives_the_load_path_unchanged() {
        let stops = vec![
            stop(0.0, [0.13, 0.27, 0.41]),
            stop(0.317, [0.91, 0.52, 0.08]),
            stop(0.742, [0.04, 0.66, 0.55]),
            stop(1.0, [0.98, 0.99, 0.97]),
        ];
        let g = Gradient::new(stops.clone()).expect("a ramp");
        assert_eq!(g.stops(), stops.as_slice());
        assert_eq!(loaded(stops.clone()).stops(), stops.as_slice());

        // Round-tripping the `Gradient` itself, which is what a document does — and
        // its bytes *are* the stop list's, which is what `carbonite(as)` promises and
        // what keeps the repair off the encoding (§8).
        let bytes = carbonite::to_vec_static(&g).expect("encodes");
        assert_eq!(bytes, carbonite::to_vec_static(&stops).expect("encodes"));
        assert_eq!(
            carbonite::from_slice_static::<Gradient>(&bytes).expect("decodes"),
            g,
        );
    }

    /// The two paths stay distinguished: the load path repairs, the authoring one
    /// still tells a caller their trace named no ramp (§22.2).
    #[test]
    fn new_still_refuses_what_the_load_path_repairs() {
        for stops in [
            vec![],
            vec![stop(0.5, [1.0; 3])],
            vec![stop(0.5, [1.0; 3]), stop(0.5, [0.0; 3])],
            vec![stop(f32::NAN, [1.0; 3]), stop(1.0, [0.0; 3])],
        ] {
            assert!(Gradient::new(stops.clone()).is_none());
            assert_eq!(loaded(stops).stops().len(), 2);
        }
    }

    /// **A ramp can never be longer than the array every consumer reads it into.**
    ///
    /// `fill.wesl`, `matte.wesl` and `filter_common.wesl` each hold exactly
    /// [`MAX_STOPS`] lanes, and the hosts that fill them index by the stop's own
    /// position — so one stop past the bound is an index off the end of a uniform,
    /// which is a panic on two of those three paths.
    ///
    /// Thinned rather than refused, because a long list names a perfectly good ramp
    /// (see [`Gradient`]) — so what is asserted is that the ramp *survives*, keeps its
    /// ends, and stays a ramp.
    #[test]
    fn a_ramp_longer_than_the_uniform_is_thinned_rather_than_refused() {
        let long: Vec<GradientStop> = (0..=1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                stop(t, [t, 1.0 - t, 0.5])
            })
            .collect();
        let (first, last) = (long[0], long[long.len() - 1]);

        // Through the constructor *and* through the encoding, since a file carries
        // the stop list and that is the path the bound exists for.
        let bytes = carbonite::to_vec_static(&long).expect("encodes");
        for g in [
            Gradient::new(long).expect("a long ramp is still a ramp"),
            carbonite::from_slice_static::<Gradient>(&bytes).expect("and decodes as one"),
        ] {
            assert!(
                g.stops().len() <= MAX_STOPS,
                "{} stops would index past every consumer's array",
                g.stops().len(),
            );
            assert!(g.stops().len() >= 2, "thinning must not collapse the ramp");
            // The ends are kept, so the thinned ramp spans what the full one did —
            // which is also what makes thinning safe to run before the rescale.
            assert_eq!(g.stops()[0].color, first.color);
            assert_eq!(g.stops()[g.stops().len() - 1].color, last.color);
            assert_eq!(
                (g.stops()[0].t, g.stops()[g.stops().len() - 1].t),
                (0.0, 1.0)
            );
            // Still ascending, with no stop picked twice.
            assert!(
                g.stops().windows(2).all(|w| w[0].t < w[1].t),
                "thinning must leave the positions strictly ascending",
            );
        }

        // Exactly at the bound nothing moves, and one past it thins to the bound —
        // the two sides of the boundary, since an off-by-one here is the whole bug.
        let at = |n: usize| -> Vec<GradientStop> {
            (0..n)
                .map(|i| stop(i as f32 / (n - 1) as f32, [0.0; 3]))
                .collect()
        };
        let exact = Gradient::new(at(MAX_STOPS)).expect("the bound itself is fine");
        assert_eq!(exact.stops().len(), MAX_STOPS);
        assert_eq!(
            exact.stops(),
            at(MAX_STOPS),
            "a ramp at the bound is untouched"
        );
        assert_eq!(
            Gradient::new(at(MAX_STOPS + 1))
                .expect("one past it")
                .stops()
                .len(),
            MAX_STOPS,
        );
    }

    #[test]
    fn resample_is_even_in_arc_length() {
        // An L-shaped path: two 100 px legs. Samples must be evenly spaced along
        // the *path*, unaffected by where the vertices sit.
        let path = [
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
        ];
        let s = resample(&path);
        assert!(s.len() >= 2);
        assert_eq!(s[0].1, path[0]);
        assert_eq!(s[s.len() - 1].1, path[2]);
        let step = 200.0 / (s.len() - 1) as f32;
        for w in s.windows(2) {
            let d = w[0].1.distance(w[1].1);
            // Around the corner the chord is shorter than the arc; everywhere
            // else the spacing is exact.
            assert!(d <= step + 1e-3, "spacing {d} exceeds {step}");
        }
        assert!(resample(&[Vec2::ZERO, Vec2::ZERO]).is_empty());
        assert!(resample(&[]).is_empty());
    }

    #[test]
    fn fit_of_a_clean_ramp_is_two_stops() {
        let a = to_lab([0.9, 0.1, 0.1]);
        let b = to_lab([0.1, 0.2, 0.9]);
        let samples: Vec<(f32, [f32; 3])> = (0..64)
            .map(|i| {
                let t = i as f32 / 63.0;
                let lab = [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ];
                (t, from_lab(lab).get())
            })
            .collect();
        let g = fit(&samples).unwrap();
        assert_eq!(
            g.stops().len(),
            2,
            "a linear Oklab ramp needs no interior stops"
        );
    }

    #[test]
    fn fit_places_a_stop_at_a_turn() {
        // Dark red → bright yellow → white: the ramp turns hard at 0.5, and a
        // two-stop fit would sail straight past yellow.
        let red = [0.5, 0.05, 0.05];
        let yellow = [0.95, 0.85, 0.1];
        let white = [1.0, 1.0, 1.0];
        let ra = to_lab(red);
        let ya = to_lab(yellow);
        let wa = to_lab(white);
        let samples: Vec<(f32, [f32; 3])> = (0..65)
            .map(|i| {
                let t = i as f32 / 64.0;
                let lab = if t < 0.5 {
                    let f = t * 2.0;
                    [
                        ra[0] + (ya[0] - ra[0]) * f,
                        ra[1] + (ya[1] - ra[1]) * f,
                        ra[2] + (ya[2] - ra[2]) * f,
                    ]
                } else {
                    let f = (t - 0.5) * 2.0;
                    [
                        ya[0] + (wa[0] - ya[0]) * f,
                        ya[1] + (wa[1] - ya[1]) * f,
                        ya[2] + (wa[2] - ya[2]) * f,
                    ]
                };
                (t, from_lab(lab).get())
            })
            .collect();
        let g = fit(&samples).unwrap();
        assert!(g.stops().len() >= 3, "the turn at yellow needs a stop");
        let turn = g
            .stops()
            .iter()
            .min_by(|a, b| (a.t - 0.5).abs().total_cmp(&(b.t - 0.5).abs()))
            .unwrap();
        assert!((turn.t - 0.5).abs() < 0.06, "stop at {}", turn.t);
        for (got, want) in turn.color.iter().zip(&yellow) {
            assert!((got - want).abs() < 0.05);
        }
        // And the fitted ramp reproduces the trace everywhere, not just at stops.
        for (t, c) in &samples {
            let got = to_lab(g.sample(*t).get());
            let want = to_lab(*c);
            let d = ((got[0] - want[0]).powi(2)
                + (got[1] - want[1]).powi(2)
                + (got[2] - want[2]).powi(2))
            .sqrt();
            assert!(d <= FIT_TOLERANCE * 2.0, "drift {d} at t={t}");
        }
    }

    #[test]
    fn fit_ignores_a_single_outlier() {
        let samples: Vec<(f32, [f32; 3])> = (0..33)
            .map(|i| {
                let t = i as f32 / 32.0;
                let c = if i == 16 {
                    [0.0, 0.0, 0.0]
                } else {
                    [0.6, 0.6, 0.6]
                };
                (t, c)
            })
            .collect();
        let g = fit(&samples).unwrap();
        assert_eq!(
            g.stops().len(),
            2,
            "one nicked sample must not become a stop"
        );
    }
}
