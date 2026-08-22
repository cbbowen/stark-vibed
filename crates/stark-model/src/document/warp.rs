//! The warp mesh (§16.9): a coarse control grid over a source rect, smoothly
//! interpolated, moving the selected paint under the resulting map.
//!
//! The wire form is the **control grid** — `cols × rows` dragged points over
//! `[min, max]` — and everything else is derived deterministically from it:
//! a Catmull-Rom (tensor Hermite) surface through the control points, sampled
//! onto a fine lattice of `SUBDIV × SUBDIV` bilinear sub-cells per control
//! cell. The GPU consumes only the fine lattice, whose cells are straight-edged
//! quads (a bilinear map sends axis-aligned lines to lines), so the parcel
//! machinery of §16.4/§16.5 carries over: watertight quads, at most one parcel
//! per destination texel, inverse mapping per fragment.
//!
//! **Identity is bit-exact by construction.** The substrate is evaluated in
//! *deviation form*: control points are split into `base + delta` against the
//! undeformed grid, only the deltas go through the Hermite arithmetic (in a
//! form whose every term is a multiple of a delta difference), and the base
//! positions are added back untouched. An untouched mesh has all-zero deltas,
//! so every lattice point lands exactly on its base, every sub-cell is exactly
//! its own source rect, and the GPU takes the exact inverse-affine path —
//! which is what lets §16.4's identity invariant extend to the warp.

use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Most control points per axis an action may carry. Small on purpose: the log
/// stores the grid, and the UI's 4×4 already spans "gentle bend" to "full
/// puppet" once the smooth interpolation is in play.
pub const MAX_WARP_GRID: u32 = 8;

/// Bilinear sub-cells per control cell per axis — the fine-lattice density the
/// smooth substrate is sampled at. Fixed by design (not by quality settings), so
/// every peer and every replay rasterizes the identical lattice.
pub(crate) const SUBDIV: usize = 8;

/// A sub-cell whose Jacobian falls below this (px²) folds or collapses; the
/// whole action is rejected rather than resampling through a crease.
const MIN_JACOBIAN: f32 = 1e-6;

/// A warp of the selected paint inside `[min, max]` (§16.9): the control grid's
/// points are the images of the rect's uniform `cols × rows` grid, and the paint
/// under the author's mask *within the rect* follows the interpolated substrate.
/// Paint and mask outside the rect stay put — the mask coverage gates the cut,
/// so a feathered selection's warp stays seamless at the rect edge.
///
/// Wire format note (§8): these field *names* are what a saved mesh is read back by,
/// so renaming one needs a `#[serde(alias)]`. Order is free.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct WarpMap {
    /// Source rect, canvas px.
    pub min: Vec2,
    pub max: Vec2,
    /// Control grid dimensions: points per axis, each in `2..=MAX_WARP_GRID`.
    pub cols: u32,
    pub rows: u32,
    /// Row-major control points, `cols * rows` of them: `points[j*cols + i]` is
    /// the image of the grid node at fraction `(i/(cols−1), j/(rows−1))`.
    pub points: Vec<Vec2>,
}

/// `a` toward `b` by `t`, with `t == 1` returning `b` **bitwise**. The branch is
/// load-bearing: two neighbouring cells evaluate their shared edge from the two
/// ends, and `a + (b−a)·1` is not `b` in f32 — the branch is what makes shared
/// lattice geometry bitwise equal from both sides, which is what makes the
/// rasterized quads watertight (§16.4).
pub(crate) fn lerp2(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    if t == 1.0 { b } else { a + (b - a) * t }
}

/// Scalar twin of [`lerp2`], for the base coordinate axes.
fn lerp1(a: f32, b: f32, t: f32) -> f32 {
    if t == 1.0 { b } else { a + (b - a) * t }
}

/// The bilinear point of a cell with corner images `g00, g10, g01, g11` at
/// parameter `(u, v)` — evaluated so that cell-edge points (`u` or `v` at 0/1)
/// are pure 1-D lerps of the shared corners, bitwise identical from whichever
/// neighbouring cell computes them.
pub fn cell_point(g: &[Vec2; 4], u: f32, v: f32) -> Vec2 {
    lerp2(lerp2(g[0], g[2], v), lerp2(g[1], g[3], v), u)
}

/// One cubic-Hermite span over `d[k]..d[k+1]` at fraction `f`, Catmull-Rom
/// tangents (one-sided at the ends), in **deviation form**: every term is a
/// multiple of a difference of deltas, so all-zero deltas give exactly zero and
/// `f == 0` gives exactly `d[k]` — no reconstitution arithmetic to drift.
fn hermite_axis(d: &[Vec2], k: usize, f: f32) -> Vec2 {
    let (v1, v2) = (d[k], d[k + 1]);
    let m1 = if k > 0 {
        (d[k + 1] - d[k - 1]) * 0.5
    } else {
        d[1] - d[0]
    };
    let m2 = if k + 2 < d.len() {
        (d[k + 2] - d[k]) * 0.5
    } else {
        d[k + 1] - d[k]
    };
    let f2 = f * f;
    let h01 = f2 * (3.0 - 2.0 * f);
    let h10 = f * (1.0 - f) * (1.0 - f);
    let h11 = f2 * (f - 1.0);
    v1 + (v2 - v1) * h01 + m1 * h10 + m2 * h11
}

/// Per-control-point weights of [`hermite_axis`] at `(k, f)`: the coefficient
/// each `d[i]` contributes to the evaluated point. What lets a frontend solve
/// "move the substrate point under the finger exactly" — the least-norm control
/// move is `Δpᵢ = Bᵢ·Δ / ΣB²` (§16.9).
fn axis_basis(len: usize, k: usize, f: f32) -> Vec<f32> {
    let mut c = vec![0.0f32; len];
    let f2 = f * f;
    let h01 = f2 * (3.0 - 2.0 * f);
    let h10 = f * (1.0 - f) * (1.0 - f);
    let h11 = f2 * (f - 1.0);
    c[k] += 1.0 - h01;
    c[k + 1] += h01;
    if k > 0 {
        c[k + 1] += 0.5 * h10;
        c[k - 1] -= 0.5 * h10;
    } else {
        c[1] += h10;
        c[0] -= h10;
    }
    if k + 2 < len {
        c[k + 2] += 0.5 * h11;
        c[k] -= 0.5 * h11;
    } else {
        c[k + 1] += h11;
        c[k] -= h11;
    }
    c
}

/// The fine lattice a [`WarpMap`] rasterizes through: base coordinates per axis
/// and the image of every lattice node, row-major (`ny` rows of `nx`).
///
/// **Fields are private and [`WarpMap::lattice`] is the only way to one**, which
/// is what makes the methods below total rather than merely lucky: `positive`
/// walks `0..ny - 1` and `aabb` reads `pts[0]`, so a hand-built empty lattice
/// panicked on both. The constructor guarantees at least `SUBDIV + 1` nodes an
/// axis (a map is refused below two control points), so there is no degenerate
/// lattice left to defend against — §1's preference for ruling out a class over
/// checking for its instances.
pub struct Lattice {
    nx: usize,
    ny: usize,
    /// Base (undeformed) lattice coordinates, strictly increasing, spanning
    /// `[min, max]` exactly at the ends.
    xs: Vec<f32>,
    ys: Vec<f32>,
    /// Image of node `(a, b)` at `pts[b * nx + a]`.
    pts: Vec<Vec2>,
}

impl Lattice {
    /// Nodes per row, and rows — both at least `SUBDIV + 1`.
    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Base (undeformed) coordinates per axis, strictly increasing.
    pub fn xs(&self) -> &[f32] {
        &self.xs
    }

    pub fn ys(&self) -> &[f32] {
        &self.ys
    }

    /// Image of every node, row-major (`ny` rows of `nx`).
    pub fn pts(&self) -> &[Vec2] {
        &self.pts
    }

    /// The corner images of sub-cell `(a, b)` in `cell_point` order
    /// (00, 10, 01, 11).
    pub fn cell(&self, a: usize, b: usize) -> [Vec2; 4] {
        [
            self.pts[b * self.nx + a],
            self.pts[b * self.nx + a + 1],
            self.pts[(b + 1) * self.nx + a],
            self.pts[(b + 1) * self.nx + a + 1],
        ]
    }

    /// Whether every sub-cell keeps a positive Jacobian — no folds, no
    /// collapses, orientation preserved. The Jacobian of a bilinear cell is
    /// bilinear in `(u, v)`, so its extrema are at the corners: four cross
    /// products per cell decide the whole cell.
    pub fn positive(&self) -> bool {
        for b in 0..self.ny - 1 {
            for a in 0..self.nx - 1 {
                let g = self.cell(a, b);
                let e = g[1] - g[0];
                let f = g[2] - g[0];
                let d = g[3] - g[1] - g[2] + g[0];
                let corners = [
                    e.perp_dot(f),
                    e.perp_dot(f + d),
                    (e + d).perp_dot(f),
                    (e + d).perp_dot(f + d),
                ];
                // A NaN Jacobian fails the positive test and folds the mesh,
                // which is why the test is spelled as "all strictly positive".
                if !corners.iter().all(|j| *j > MIN_JACOBIAN) {
                    return false;
                }
            }
        }
        true
    }

    /// The bounding box of every lattice image — a conservative bound on where
    /// the warp can carry paint (each bilinear cell stays inside its corners'
    /// hull).
    pub fn aabb(&self) -> (Vec2, Vec2) {
        let mut lo = self.pts[0];
        let mut hi = self.pts[0];
        for p in &self.pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }
}

impl WarpMap {
    /// The undeformed mesh over `[min, max]`: control points on the rect's
    /// uniform grid. This is the *reference* every identity comparison is made
    /// against — frontends must start from these exact points, not re-derive
    /// their own, for "untouched" to stay bit-exact.
    pub fn identity(min: Vec2, max: Vec2, cols: u32, rows: u32) -> Self {
        let mut points = Vec::with_capacity((cols * rows) as usize);
        for j in 0..rows {
            for i in 0..cols {
                points.push(grid_base(min, max, cols, rows, i, j));
            }
        }
        Self {
            min,
            max,
            cols,
            rows,
            points,
        }
    }

    /// The undeformed position of control node `(i, j)`.
    pub fn base(&self, i: u32, j: u32) -> Vec2 {
        grid_base(self.min, self.max, self.cols, self.rows, i, j)
    }

    /// Structural validity: finite, a proper rect, a grid within bounds, and
    /// the right number of points. Cheap; [`usable`](Self::usable) adds the
    /// geometric (fold) check.
    fn shape_ok(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.max.x > self.min.x
            && self.max.y > self.min.y
            && (2..=MAX_WARP_GRID).contains(&self.cols)
            && (2..=MAX_WARP_GRID).contains(&self.rows)
            && self.points.len() == (self.cols * self.rows) as usize
            && self.points.iter().all(|p| p.is_finite())
    }

    /// Whether this warp may be applied at all: well-shaped, a lattice that
    /// exists (the rect is wide enough that its fine columns don't collapse in
    /// f32), and no folded sub-cell. Deterministic — peers and replays agree
    /// about rejection (§16.1).
    pub fn usable(&self) -> bool {
        self.lattice().is_some_and(|lat| lat.positive())
    }

    /// Sample the smooth substrate onto the fine lattice. `None` when the map is
    /// malformed or the rect is degenerate enough that the base lattice loses
    /// strict monotonicity in f32.
    pub fn lattice(&self) -> Option<Lattice> {
        if !self.shape_ok() {
            return None;
        }
        let (cols, rows) = (self.cols as usize, self.rows as usize);
        let nx = (cols - 1) * SUBDIV + 1;
        let ny = (rows - 1) * SUBDIV + 1;

        // Base axes: each fine coordinate is a lerp *within its control cell*,
        // hitting the control base coordinates exactly at cell ends (lerp1's
        // t == 1 branch), so the lattice tiles the rect without gaps.
        let axis = |n_ctrl: usize, count: usize, at: &dyn Fn(u32) -> f32| -> Option<Vec<f32>> {
            let mut out = Vec::with_capacity(count);
            for a in 0..count {
                let k = (a / SUBDIV).min(n_ctrl - 2);
                let f = (a - k * SUBDIV) as f32 / SUBDIV as f32;
                out.push(lerp1(at(k as u32), at(k as u32 + 1), f));
            }
            // Strictly increasing, or the sub-cells degenerate.
            out.windows(2).all(|w| w[1] > w[0]).then_some(out)
        };
        let xs = axis(cols, nx, &|i| self.base(i, 0).x)?;
        let ys = axis(rows, ny, &|j| self.base(0, j).y)?;

        // Deltas against the undeformed grid — the only thing the Hermite
        // arithmetic ever sees.
        let deltas = self.deltas();

        let mut pts = Vec::with_capacity(nx * ny);
        let mut row_dev = vec![Vec2::ZERO; rows];
        for (b, y) in ys.iter().enumerate() {
            let ky = (b / SUBDIV).min(rows - 2);
            let fy = (b - ky * SUBDIV) as f32 / SUBDIV as f32;
            for (a, x) in xs.iter().enumerate() {
                let kx = (a / SUBDIV).min(cols - 2);
                let fx = (a - kx * SUBDIV) as f32 / SUBDIV as f32;
                for (j, dev) in row_dev.iter_mut().enumerate() {
                    *dev = hermite_axis(&deltas[j * cols..(j + 1) * cols], kx, fx);
                }
                let delta = hermite_axis(&row_dev, ky, fy);
                pts.push(Vec2::new(*x, *y) + delta);
            }
        }
        Some(Lattice {
            nx,
            ny,
            xs,
            ys,
            pts,
        })
    }

    /// Which control cell the grid fraction `t` falls in on one axis, and how far
    /// through it — the shared half of [`eval`](Self::eval) and
    /// [`basis`](Self::basis), which carried a verbatim copy each.
    fn locate(t: f32, n: usize) -> (usize, f32) {
        let u = (t * (n - 1) as f32).clamp(0.0, (n - 1) as f32);
        let k = (u.floor() as usize).min(n - 2);
        (k, u - k as f32)
    }

    /// The control points as **deviations** from the undeformed grid — the only
    /// thing the Hermite arithmetic ever sees (see the module header).
    ///
    /// Hoisted out of [`eval`](Self::eval) so a caller evaluating the substrate many
    /// times can compute it once: see [`prepared`](Self::prepared).
    fn deltas(&self) -> Vec<Vec2> {
        (0..self.rows)
            .flat_map(|j| (0..self.cols).map(move |i| (i, j)))
            .map(|(i, j)| self.points[(j * self.cols + i) as usize] - self.base(i, j))
            .collect()
    }

    /// The mesh with its delta grid computed once, for a caller that is about to
    /// evaluate the substrate many times (§16.9).
    ///
    /// **The frontend is that caller, in a loop.** Finding the grabbed point is a
    /// search over the substrate and drawing the mesh is one `eval` per point of
    /// every curve, so a drag was rebuilding the whole delta grid — a `Vec`
    /// allocation and `cols · rows` subtractions — per sample, tens to hundreds of
    /// times a frame. Nothing about the answer changes between them.
    pub fn prepared(&self) -> Prepared<'_> {
        Prepared {
            map: self,
            deltas: self.deltas(),
        }
    }

    /// The smooth substrate at grid fraction `t ∈ [0,1]²` — what a frontend draws
    /// its mesh curves with and inverts to find the grabbed point.
    ///
    /// Evaluating more than one point? [`prepared`](Self::prepared) hoists the
    /// delta grid this rebuilds on every call.
    pub fn eval(&self, t: Vec2) -> Vec2 {
        self.prepared().eval(t)
    }

    /// Per-control-point influence at grid fraction `t`: how far the substrate
    /// point moves per unit move of each control point (row-major, like
    /// `points`). Weights sum to 1. The exact-follow surface drag rests on
    /// this (§16.9).
    pub fn basis(&self, t: Vec2) -> Vec<f32> {
        let (cols, rows) = (self.cols as usize, self.rows as usize);
        let (kx, fx) = Self::locate(t.x, cols);
        let (ky, fy) = Self::locate(t.y, rows);
        let cx = axis_basis(cols, kx, fx);
        let cy = axis_basis(rows, ky, fy);
        let mut out = Vec::with_capacity(cols * rows);
        for wy in &cy {
            for wx in &cx {
                out.push(wx * wy);
            }
        }
        out
    }

    /// A conservative bound on where the warp can carry paint (the fine
    /// lattice's bounding box — every bilinear cell stays inside its corners'
    /// hull). `None` when the map is unusable.
    pub fn image_aabb(&self) -> Option<(Vec2, Vec2)> {
        self.lattice().map(|lat| lat.aabb())
    }
}

/// A [`WarpMap`] with its delta grid already computed — see
/// [`WarpMap::prepared`]. Borrows the map, so it cannot outlive an edit to it.
pub struct Prepared<'a> {
    map: &'a WarpMap,
    deltas: Vec<Vec2>,
}

impl Prepared<'_> {
    /// The smooth substrate at grid fraction `t ∈ [0,1]²`.
    ///
    /// Bit-for-bit what [`WarpMap::eval`] computes — it *is* what `eval` computes,
    /// which is what keeps the hoist from being a second implementation that could
    /// drift from the first. §16.4's identity invariant is stated bitwise, so a
    /// reordering here would be a real difference and not a rounding one.
    pub fn eval(&self, t: Vec2) -> Vec2 {
        let (cols, rows) = (self.map.cols as usize, self.map.rows as usize);
        let (kx, fx) = WarpMap::locate(t.x, cols);
        let (ky, fy) = WarpMap::locate(t.y, rows);
        let row_dev: Vec<Vec2> = (0..rows)
            .map(|j| hermite_axis(&self.deltas[j * cols..(j + 1) * cols], kx, fx))
            .collect();
        let delta = hermite_axis(&row_dev, ky, fy);
        let base = self.map.base(kx as u32, 0);
        let next_x = self.map.base(kx as u32 + 1, 0);
        let bx = lerp1(base.x, next_x.x, fx);
        let by = lerp1(
            self.map.base(0, ky as u32).y,
            self.map.base(0, ky as u32 + 1).y,
            fy,
        );
        Vec2::new(bx, by) + delta
    }
}

/// The undeformed grid node — one shared formula, because "identity" is defined
/// as "the points *are* these values", bitwise.
fn grid_base(min: Vec2, max: Vec2, cols: u32, rows: u32, i: u32, j: u32) -> Vec2 {
    Vec2::new(
        lerp1(min.x, max.x, i as f32 / (cols - 1) as f32),
        lerp1(min.y, max.y, j as f32 / (rows - 1) as f32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-40.0, 16.0), Vec2::new(216.0, 272.0))
    }

    #[test]
    fn identity_mesh_lattices_exactly_onto_its_base() {
        let (min, max) = rect();
        let map = WarpMap::identity(min, max, 4, 4);
        assert!(map.usable());
        let lat = map.lattice().unwrap();
        for b in 0..lat.ny {
            for a in 0..lat.nx {
                let p = lat.pts[b * lat.nx + a];
                // Bitwise, not approximately: §16.4's identity invariant rides
                // on every lattice point being its base.
                assert_eq!(p, Vec2::new(lat.xs[a], lat.ys[b]), "node ({a}, {b})");
            }
        }
        assert_eq!(lat.xs[0], min.x);
        assert_eq!(*lat.xs.last().unwrap(), max.x);
        assert_eq!(lat.ys[0], min.y);
        assert_eq!(*lat.ys.last().unwrap(), max.y);
    }

    #[test]
    fn the_surface_interpolates_its_control_points() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        map.points[5] += Vec2::new(20.0, -14.0); // an interior node
        let t = Vec2::new(1.0 / 3.0, 1.0 / 3.0);
        let at = map.eval(t);
        assert!(
            (at - map.points[5]).length() < 1e-3,
            "substrate at the node's fraction is {at:?}, node is {:?}",
            map.points[5]
        );
    }

    #[test]
    fn basis_weights_sum_to_one_and_predict_the_surface() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        map.points[6] += Vec2::new(-11.0, 7.0);
        map.points[9] += Vec2::new(4.0, 13.0);
        let t = Vec2::new(0.41, 0.77);
        let b = map.basis(t);
        let sum: f32 = b.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights sum to {sum}");
        // Moving every control point by its weighted share moves the substrate
        // point by exactly the sum of shares — linearity, which the
        // exact-follow drag depends on.
        let before = map.eval(t);
        let delta = Vec2::new(9.0, -5.0);
        let norm: f32 = b.iter().map(|w| w * w).sum();
        let mut moved = map.clone();
        for (p, w) in moved.points.iter_mut().zip(&b) {
            *p += delta * (*w / norm);
        }
        let after = moved.eval(t);
        assert!(
            (after - (before + delta)).length() < 1e-2,
            "substrate moved {:?}, wanted {delta:?}",
            after - before
        );
    }

    /// The hoist must be the **same arithmetic**, not merely a close one.
    ///
    /// §16.4's identity invariant is stated bitwise and the watertightness of the
    /// rasterized quads rests on shared points agreeing to the bit, so a `Prepared`
    /// that reassociated anything would be a second implementation quietly
    /// disagreeing with the first — the class of drift this codebase keeps ruling
    /// out with one shared formula (`grid_base`, `rect_corners`, `lerp2`).
    #[test]
    fn preparing_changes_nothing_about_the_answer() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        map.points[5] += Vec2::new(21.0, -13.0);
        map.points[9] += Vec2::new(-7.0, 18.0);
        let prepared = map.prepared();
        for j in 0..=16 {
            for i in 0..=16 {
                let t = Vec2::new(i as f32 / 16.0, j as f32 / 16.0);
                assert_eq!(prepared.eval(t), map.eval(t), "at {t:?}");
            }
        }
        // Including the untouched mesh, whose every point must still be its base.
        let flat = WarpMap::identity(min, max, 3, 5);
        let flat_prepared = flat.prepared();
        for i in 0..=8 {
            let t = Vec2::splat(i as f32 / 8.0);
            assert_eq!(flat_prepared.eval(t), flat.eval(t));
        }
    }

    #[test]
    fn a_folded_mesh_is_refused() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        assert!(map.usable());
        // Drag one interior node across its neighbour: some sub-cell must fold.
        map.points[5] = map.points[6] + Vec2::new(60.0, 0.0);
        assert!(!map.usable(), "a crossed mesh has to be rejected");
    }

    #[test]
    fn malformed_maps_are_refused() {
        let (min, max) = rect();
        let mut short = WarpMap::identity(min, max, 4, 4);
        short.points.pop();
        assert!(!short.usable());
        let inverted = WarpMap::identity(max, min, 4, 4);
        assert!(!inverted.usable());
        let mut nan = WarpMap::identity(min, max, 3, 3);
        nan.points[4] = Vec2::new(f32::NAN, 0.0);
        assert!(!nan.usable());
        assert!(!WarpMap::identity(min, max, 1, 4).usable());
        assert!(!WarpMap::identity(min, max, 4, MAX_WARP_GRID + 1).usable());
    }

    #[test]
    fn a_translated_mesh_stays_unfolded_and_lands_where_it_points() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        let shift = Vec2::new(120.0, -35.0);
        for p in &mut map.points {
            *p += shift;
        }
        assert!(map.usable());
        let lat = map.lattice().unwrap();
        for (i, p) in lat.pts.iter().enumerate() {
            let base = Vec2::new(lat.xs[i % lat.nx], lat.ys[i / lat.nx]);
            assert!(
                (*p - base - shift).length() < 1e-2,
                "lattice node {i} drifted: {:?}",
                *p - base
            );
        }
    }

    #[test]
    fn cell_edges_are_shared_bitwise() {
        let (min, max) = rect();
        let mut map = WarpMap::identity(min, max, 4, 4);
        map.points[5] += Vec2::new(25.0, 10.0);
        map.points[10] += Vec2::new(-8.0, 22.0);
        let lat = map.lattice().unwrap();
        // A point on the shared edge of two horizontally adjacent sub-cells,
        // computed from each side's corner set, must agree bitwise (§16.4's
        // watertightness, extended to the mesh).
        let (a, b) = (7, 5);
        let left = lat.cell(a, b);
        let right = lat.cell(a + 1, b);
        for v in [0.0, 0.25, 0.6, 1.0] {
            assert_eq!(cell_point(&left, 1.0, v), cell_point(&right, 0.0, v));
        }
    }
}
