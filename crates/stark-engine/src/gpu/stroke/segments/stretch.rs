//! The stretched tip (§6.6): the map that reads a tip drawn out along its facing
//! axis out of the very prefix-τ volume the unstretched tip binds — three floats and
//! no new texture.

/// **The stretch of the tip along its facing axis, as the renderer reads it** (§6.6):
/// the map carrying a point's place in the tip's reference travel frame into the frame
/// the prefix-τ volume is looked up in, once the extent is drawn out by
/// [`BrushParams::elongation`](stark_model::document::BrushParams::elongation).
///
/// The whole feature is here, and it costs three floats and no new texture.
/// Stretching the tip by `s` along a canvas axis `û` is the linear map
/// `A = R_û·diag(s, 1)·R_ûᵀ` on the extent, and the deposit is that extent's
/// integral as it is dragged along the travel `t̂`. Substituting `q = A⁻¹p` turns that
/// integral into one of the **unstretched** extent — dragged along
/// `v̂ = normalize(A⁻¹t̂)` instead of `t̂`, over a travel `m = |A⁻¹t̂|` times as long,
/// with `1/m` on the result. Every one of those is something the existing volume
/// already answers: it is indexed by the angle between the mask's native axis and the
/// direction of integration, so a different direction is a different *slice*, not a
/// different bake.
///
/// That holds because the axis is the brush's **facing** axis
/// ([`orientation_turns`](super::orientation_turns)'s), which is what makes the whole of it fit in the volume the
/// brush already binds:
///
/// - `FollowStroke` faces along the tangent, so `û = t̂` and therefore `v̂ = t̂` — the
///   relative angle stays 0 and the single identity layer still serves.
/// - A round tip is rotation-invariant: one slice answers every angle (§6.6).
/// - `Pen` on a stamp already reads the stack of every angle, so a shifted slice
///   index is free.
///
/// An axis free of the facing one would break all three at once — a follow-stroke
/// stamp would need the rotatable stack it never builds — which is why there is no
/// second direction to set.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::gpu::stroke) struct Stretch {
    /// `m` — the segment's travel measured in the stretched frame, as a multiple of
    /// its travel in the tip's own. Also the divisor a prefix difference read in that
    /// frame is worth in the tip's, which is why the shaders carry the one number and
    /// not the pair.
    pub(in crate::gpu::stroke) travel: f32,
    /// The lookup frame's travel coordinate picks up this much of the point's lateral
    /// offset — the leading edge of an obliquely-stretched tip is not square to its
    /// own travel. 0 wherever the facing axis is along or across the travel.
    pub(in crate::gpu::stroke) shear: f32,
    /// What the point's lateral offset is scaled by. **Below 1 for any real stretch**,
    /// which is the direction that matters: the mask's `|y| ≤ 1` then stands for a
    /// wider band of canvas, and that is the stroke getting fatter.
    pub(in crate::gpu::stroke) lateral: f32,
    /// How much further round the volume's orientation axis to read, in turns — the
    /// angle from `t̂` to `v̂`, which is the whole of what "another slice" means above.
    /// Added to [`Sweep::orient`](super::Sweep::orient) rather than carried, so the shaders see one angle.
    pub(in crate::gpu::stroke) turns: f32,
}

impl Stretch {
    /// The identity: the tip as its mask draws it. What every brush without a stretch
    /// gets, and **exactly** the neutral element of every expression that reads one —
    /// `travel = 1`, `shear = 0`, `lateral = 1` leave the shaders' arithmetic an
    /// identity in floats, not merely close to one.
    pub(in crate::gpu::stroke) const NONE: Self = Self {
        travel: 1.0,
        shear: 0.0,
        lateral: 1.0,
        turns: 0.0,
    };

    /// Solve the map for an elongation `s` along a facing axis sitting `orient` turns
    /// round from the travel direction — which is [`orientation_turns`](super::orientation_turns)'s own answer,
    /// for both sources: `FollowStroke` faces along the tangent and reports 0, and
    /// `Pen` reports the azimuth relative to the travel, which is the same angle.
    ///
    /// Short-circuited at `s = 1` rather than left to fall out of the general path.
    /// The trigonometry below *does* return the identity there — `A⁻¹` is `I`, `v` is
    /// `(1, 0)` — but a brush with no stretch should not be relying on `atan2(0, 1)`
    /// being exactly zero to render what it always rendered.
    pub(super) fn solve(elongation: f32, orient: f32) -> Self {
        // A non-finite elongation takes the same exit as an absent one, since the
        // values here arrive from files, presets and peers: a NaN would otherwise reach
        // a lane the shaders divide by.
        if !elongation.is_finite() || elongation <= 1.0 {
            return Self::NONE;
        }
        // `A⁻¹ = I + (k − 1)·ûûᵀ` in the travel frame, where `k = 1/s` and `û` is the
        // facing axis expressed there. Symmetric, so its first column is `(a00, a01)`
        // and that column is `v = A⁻¹t̂` — the direction the sweep is integrated along
        // once the stretch is taken out.
        let k = 1.0 / elongation;
        let (sn, cs) = (orient * std::f32::consts::TAU).sin_cos();
        let a00 = 1.0 + (k - 1.0) * cs * cs;
        let a01 = (k - 1.0) * cs * sn;
        let a11 = 1.0 + (k - 1.0) * sn * sn;
        let m = (a00 * a00 + a01 * a01).sqrt();
        // `M = R(v̂ → x̂)·A⁻¹`, whose first column is `(m, 0)` by construction — so the
        // map is upper triangular and three numbers state it. Its determinant is
        // `det A⁻¹ = k`, which is where `lateral` comes from without a second dot
        // product, and which says the same thing the shape does: a tip stretched `s`
        // along one axis covers `s` times the mask per unit of the frame.
        let (vx, vy) = (a00 / m, a01 / m);
        Self {
            travel: m,
            shear: vx * a01 + vy * a11,
            lateral: k / m,
            turns: -a01.atan2(a00) / std::f32::consts::TAU,
        }
    }

    /// The box in the tip's **reference travel frame** that holds everything the mask
    /// can put on the canvas, as a multiple of the frame radius: `(along, across)`.
    ///
    /// **The shaders' — `stamp_common::stretch_hull` is this function**, and only they
    /// need it: it is the sweep strip and the dynamics loop's rim test that are drawn
    /// in the reference travel frame, where the host's boxes are canvas-aligned and
    /// take [`Sweep::reach`](super::Sweep::reach) instead. So this side is `#[cfg(test)]`, existing to hold
    /// the shader to a formula rather than to be called in anger — the derivation is
    /// short enough to restate and wrong enough to matter, since under-reporting it is
    /// a stroke cut off along a straight line where its own geometry ran out.
    ///
    /// A point takes paint only where the map lands it inside the mask's `|x| ≤ 1,
    /// |y| ≤ 1`, so `|y| ≤ 1/lateral` and `|x| ≤ (1 + |shear|/lateral)/travel`.
    /// `(1, 1)` exactly for [`NONE`](Self::NONE).
    #[cfg(test)]
    fn hull(&self) -> (f32, f32) {
        let across = 1.0 / self.lateral;
        ((1.0 + self.shear.abs() * across) / self.travel, across)
    }
}

/// The stretched tip (§6.6).
///
/// One claim is being tested here, and everything else is a reading of it: **the swept
/// integral of an extent drawn out along an axis is the integral of the *undrawn*
/// extent, along another direction, over another travel, times a constant.** That is
/// what lets a stretch cost three floats and no new texture, and it is exactly the kind
/// of claim that is either exact or quietly wrong by a few percent everywhere — the mask
/// is still swept, the stroke still looks like a stroke, and the profile it draws is not
/// the one the brush names.
///
/// So it is checked against a **direct numerical sweep** of the stretched extent,
/// with a mask that is deliberately neither round nor symmetric: a rotation-invariant
/// tip would satisfy the identity at every angle for the wrong reason (any slice would
/// do), and a symmetric one would hide the shear.
#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::BrushParams;
    use std::f32::consts::TAU;

    /// Rotate `p` by `a` radians.
    fn rot(p: (f32, f32), a: f32) -> (f32, f32) {
        let (s, c) = a.sin_cos();
        (p.0 * c - p.1 * s, p.0 * s + p.1 * c)
    }

    /// A stand-in for a brush mask's optical-depth field `κ`, in mask coordinates.
    ///
    /// **Anisotropic and off-centre on purpose.** The identity under test moves the
    /// direction of integration, so a field that reads the same along every direction
    /// would pass it however the slice was chosen; and it puts a shear on the travel
    /// coordinate, which a field symmetric about `y = 0` would leave undetectable.
    /// Smooth and compactly supported so the quadrature below converges quickly.
    fn mask(q: (f32, f32)) -> f32 {
        let r2 = q.0 * q.0 + q.1 * q.1;
        if r2 >= 1.0 {
            return 0.0;
        }
        let lobe = (-6.0 * (q.0 - 0.2) * (q.0 - 0.2) - 2.0 * (q.1 + 0.1) * (q.1 + 0.1)).exp();
        (1.0 - r2) * lobe
    }

    /// The prefix-τ volume's own lookup, evaluated by quadrature rather than baked:
    /// slice `w` holds the mask turned by `+w` turns and integrated along `+x`, so a
    /// read at `(x, y)` is `∫ mask(R(−wτ)·(u, y)) du` up to `x`
    /// (`assets::rotate_layers`, `assets::build_prefix`).
    fn prefix(x: f32, y: f32, w: f32) -> f32 {
        const STEPS: usize = 4000;
        let (lo, hi) = (-1.5f32, x.min(1.5));
        if hi <= lo {
            return 0.0;
        }
        let du = (hi - lo) / STEPS as f32;
        (0..STEPS)
            .map(|i| {
                let u = lo + (i as f32 + 0.5) * du;
                mask(rot((u, y), -w * TAU)) * du
            })
            .sum()
    }

    /// The swept depth at `p`, computed the long way: the stretched extent dragged
    /// along `+x` for `len`, integrated step by step in the travel frame. The thing the
    /// renderer's three floats have to reproduce.
    fn swept_directly(p: (f32, f32), len: f32, elongation: f32, orient: f32) -> f32 {
        const STEPS: usize = 4000;
        let psi = orient * TAU;
        let k = 1.0 / elongation;
        let (sn, cs) = psi.sin_cos();
        // `A⁻¹ = I + (k − 1)·ûûᵀ`, the map the extent is read through.
        let inv = |q: (f32, f32)| {
            let d = (k - 1.0) * (q.0 * cs + q.1 * sn);
            (q.0 + d * cs, q.1 + d * sn)
        };
        let dt = len / STEPS as f32;
        (0..STEPS)
            .map(|i| {
                let t = (i as f32 + 0.5) * dt;
                // The tip's own extent at this instant: the mask turned to face the
                // shape's angle, then drawn out along it.
                mask(rot(inv((p.0 - t, p.1)), -psi)) * dt
            })
            .sum()
    }

    /// The same depth as the renderer takes it: one prefix difference on one slice,
    /// with the map and the gain [`Stretch::solve`] hands back.
    fn swept_through_the_volume(p: (f32, f32), len: f32, elongation: f32, orient: f32) -> f32 {
        let st = Stretch::solve(elongation, orient);
        let slice = (orient + st.turns).rem_euclid(1.0);
        // `stretch_look`, in Rust: the lookup frame's travel coordinate at the sweep's
        // end and at its start, and the lateral offset.
        let x1 = st.travel * p.0 + st.shear * p.1;
        let x0 = x1 - st.travel * len;
        let y = st.lateral * p.1;
        (prefix(x1, y, slice) - prefix(x0, y, slice)) / st.travel
    }

    /// **The whole feature, against a sweep that knows nothing about it.**
    ///
    /// Across elongations, facing angles, sample points and travels: the definite
    /// integral of the stretched extent equals the unstretched volume read at
    /// [`Stretch`]'s slice, over its travel, times its gain. If the derivation is wrong
    /// anywhere — the sign of the slice shift, the shear, the Jacobian — this is where
    /// it shows, because the left-hand side is the picture and the right-hand side is
    /// the renderer.
    #[test]
    fn a_stretched_sweep_is_the_unstretched_volume_read_at_another_slice() {
        let mut worst = 0.0f32;
        for &elongation in &[1.0, 1.5, 2.5, 4.0, 8.0] {
            for &orient in &[0.0, 0.05, 0.125, 0.25, 0.4, 0.5, 0.75, 0.9] {
                for &len in &[0.3, 1.0, 3.0] {
                    for &p in &[
                        (0.0, 0.0),
                        (0.4, 0.3),
                        (-0.6, -0.45),
                        (1.1, 0.2),
                        (0.2, -0.8),
                    ] {
                        let direct = swept_directly(p, len, elongation, orient);
                        let volume = swept_through_the_volume(p, len, elongation, orient);
                        let err = (direct - volume).abs();
                        worst = worst.max(err);
                        assert!(
                            err < 2e-3,
                            "s={elongation} orient={orient} len={len} at {p:?}: \
                             swept {direct}, volume {volume}",
                        );
                    }
                }
            }
        }
        // Both sides are 4000-step midpoint rules over a field with a `1 − r²` kink at
        // the rim, so the floor here is the quadrature's and not the identity's.
        assert!(worst < 2e-3, "worst disagreement {worst}");
    }

    /// The identity is the identity **in floats**, not merely to a tolerance — which is
    /// what lets every existing brush keep its pixels. Checked at the ways a brush says
    /// it does not stretch: the knob at rest, a knob the pen has modulated to nothing,
    /// and the two malformed values that arrive from files, presets and peers.
    #[test]
    fn a_brush_that_does_not_stretch_gets_the_exact_identity() {
        for &orient in &[0.0, 0.125, 0.25, 0.5, 0.9] {
            for &e in &[BrushParams::elongation(0.0), 1.0, 0.5, f32::NAN] {
                assert_eq!(
                    Stretch::solve(e, orient),
                    Stretch::NONE,
                    "elongation {e} at orient {orient} is not the identity",
                );
            }
        }
        // And the identity leaves the shaders' arithmetic exact — `stretch_look`,
        // `stretch_gain` and `stretch_hull` spelled here as they are spelled there.
        let st = Stretch::NONE;
        for &(x, y, lr) in &[(0.37f32, -0.62f32, 1.7f32), (-1.0, 0.0, 0.0)] {
            assert_eq!(st.travel * x + st.shear * y, x);
            assert_eq!(st.lateral * y, y);
            assert_eq!(st.travel * x + st.shear * y - st.travel * lr, x - lr);
        }
        assert_eq!(1.0 / st.travel, 1.0);
        assert_eq!(st.hull(), (1.0, 1.0));
    }

    /// `hull` has to hold **everything the mask can paint**, because what is drawn for
    /// the extent is drawn from it: the sweep strip in the shader, and the tile box
    /// on the host. Under-report it and the stroke is cut off along a straight line
    /// where its own geometry ran out — the failure every under-reported reach lands
    /// on ([`Sweep::reach`]), which a stretch reintroduces at a different scale.
    ///
    /// So: every point of the reference travel frame that the map lands *inside* the
    /// mask's square must be inside the hull.
    #[test]
    fn the_hull_holds_every_point_the_stretched_mask_can_paint() {
        for &elongation in &[1.0, 1.5, 2.5, 4.0, 8.0] {
            for &orient in &[0.0, 0.05, 0.125, 0.25, 0.4, 0.6, 0.875] {
                let st = Stretch::solve(elongation, orient);
                let (along, across) = st.hull();
                // Walk the mask's own square back through the map: its corners are the
                // extremes, but the edges are walked too so nothing rests on that.
                for i in 0..=64 {
                    let f = i as f32 / 32.0 - 1.0;
                    for (mx, my) in [(1.0, f), (-1.0, f), (f, 1.0), (f, -1.0)] {
                        // `stretch_unlook`: the map is upper triangular, so its inverse
                        // is three reciprocals.
                        let y = my / st.lateral;
                        let x = (mx - st.shear * y) / st.travel;
                        assert!(
                            x.abs() <= along + 1e-4 && y.abs() <= across + 1e-4,
                            "s={elongation} orient={orient}: mask ({mx}, {my}) rides at \
                             ({x}, {y}), outside the hull ({along}, {across})",
                        );
                    }
                }
            }
        }
    }

    /// [`Sweep::reach`] is the same promise in canvas px and in every direction at once
    /// — a *box* the segment is drawn into — so it is scaled by the elongation alone
    /// rather than by the hull's two axes. This is why that is sound: the map's operator
    /// norm is exactly the elongation, so no point of the mask lands further out than
    /// `elongation` times where the unstretched tip's did, whichever way it faces.
    #[test]
    fn the_reach_covers_every_texel_the_stretched_mask_can_paint() {
        for &elongation in &[1.0, 2.0, 5.0, 8.0] {
            for &orient in &[0.0, 0.1, 0.25, 0.33, 0.5] {
                let psi = orient * TAU;
                let (sn, cs) = psi.sin_cos();
                // `A = I + (s − 1)·ûûᵀ` — the forward map, mask to canvas.
                let fwd = |q: (f32, f32)| {
                    let d = (elongation - 1.0) * (q.0 * cs + q.1 * sn);
                    (q.0 + d * cs, q.1 + d * sn)
                };
                for i in 0..=64 {
                    let f = i as f32 / 32.0 - 1.0;
                    // The disc's rim is the frontier: nothing any canonical shape
                    // can paint lies outside its disc ([`Sweep::reach`]), so the rim
                    // covering under the map is the whole promise.
                    let q = (f.cos(), f.sin());
                    let p = fwd(q);
                    let d = (p.0 * p.0 + p.1 * p.1).sqrt();
                    assert!(
                        d <= elongation + 1e-4,
                        "s={elongation} orient={orient}: mask {q:?} lands {d} out, \
                         past a reach of {elongation}",
                    );
                }
            }
        }
    }

    /// What the axis is *for*, stated as the two readings a hand would recognise — and
    /// the reason a pencil could not be built out of the size mapping it used to use,
    /// which scales both of these together.
    ///
    /// Lean the pen **along** the stroke and the mark gets heavier without getting
    /// wider; lean it **across** and it gets wider without the centreline getting
    /// heavier per unit travel. Measured off the solved map rather than off a picture,
    /// which is what makes it a statement about the model.
    #[test]
    fn leaning_along_the_stroke_darkens_it_and_leaning_across_widens_it() {
        let s = 3.0;
        // Along the travel: the lookup's lateral axis is untouched, so the profile
        // across the stroke is the shape it was — and a full pass, whose prefix
        // saturates at the row total whatever the travel scaled to, lays `s` times as
        // much.
        let along = Stretch::solve(s, 0.0);
        assert!((along.lateral - 1.0).abs() < 1e-6, "{along:?}");
        assert!((1.0 / along.travel - s).abs() < 1e-5, "gain: {along:?}");
        assert_eq!(along.shear, 0.0, "an axis along the travel cannot shear it");
        assert!(along.turns.abs() < 1e-6, "nor turn the slice: {along:?}");

        // Across it: the mask's own `|y| ≤ 1` now stands for `s` radii of canvas, which
        // is the stroke `s` times wider, and the depth per unit travel is untouched.
        let across = Stretch::solve(s, 0.25);
        assert!((1.0 / across.lateral - s).abs() < 1e-5, "{across:?}");
        assert!((across.travel - 1.0).abs() < 1e-6, "gain: {across:?}");
        assert!(
            across.shear.abs() < 1e-6,
            "an axis across the travel cannot shear it"
        );
        assert!(across.turns.abs() < 1e-6, "nor turn the slice: {across:?}");

        // Obliquely: both, plus the shear that leans the leading edge — the term only
        // an oblique lean has, and the one a per-axis scale could not express.
        let oblique = Stretch::solve(s, 0.125);
        assert!(oblique.shear.abs() > 0.1, "{oblique:?}");
        assert!(oblique.turns.abs() > 0.01, "{oblique:?}");
        // The map's determinant is `1/s` at every angle: a tip drawn out `s` along one
        // axis covers `s` times the mask per unit of the frame, however it is turned.
        for st in [along, across, oblique] {
            assert!((st.travel * st.lateral - 1.0 / s).abs() < 1e-5, "{st:?}");
        }
    }

    /// The knob's own contract: exactly 1 at rest, monotone, and bounded — the last
    /// because the elongation prices the stroke, every tile the drawn-out tip reaches
    /// being one the loop rasterizes and dispatches over.
    #[test]
    fn the_elongation_knob_is_the_identity_at_rest_and_bounded_at_the_top() {
        assert_eq!(BrushParams::elongation(0.0), 1.0);
        assert_eq!(
            BrushParams::elongation(-1.0),
            1.0,
            "a negative knob is no stretch, not a squash"
        );
        assert_eq!(
            BrushParams::elongation(f32::NAN),
            1.0,
            "and neither is a NaN"
        );
        assert_eq!(BrushParams::elongation(1.0), BrushParams::MAX_ELONGATION);
        assert_eq!(BrushParams::elongation(9.0), BrushParams::MAX_ELONGATION);
        let mut prev = 0.0;
        for i in 0..=100 {
            let e = BrushParams::elongation(i as f32 / 100.0);
            assert!(e >= prev, "not monotone at {i}");
            assert!(e <= BrushParams::MAX_ELONGATION);
            prev = e;
        }
    }
}
