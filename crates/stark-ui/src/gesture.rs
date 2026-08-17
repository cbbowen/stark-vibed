//! The transform mode's algebra (§16.6, §16.8, §16.9): the three families of map
//! the artist composes, as pure geometry.
//!
//! **Nothing here knows about signals, dioxus, or the browser.** Each family is a
//! value plus the functions that move it — `translated`, `turned_scaled`,
//! `stretched`, `corner_dragged`, `surface_dragged` — and every one of them takes
//! the gesture's *start* rather than its previous step, so a long drag is one
//! accumulated map instead of a chain of them and rounding cannot walk over the
//! length of it. The chrome that drives them is `panels::transform`; what holds
//! them between events is `state::AppState::transform`.
//!
//! That is why it is a file of its own. It lived in `crate::state`, which is about
//! the app's signals and the one door to the engine, and this is the part of that
//! file that could be tested — 18 of the crate's tests are here, and they are the
//! ones that can say a rim drag really does carry the grabbed point to the pointer
//! and that four mirrors really do cancel bit-exactly. Sitting inside the state
//! module, they were the tests hardest to find and the code most likely to be read
//! as UI plumbing.
//!
//! Three shapes, one rule: **the grabbed point follows the pointer exactly**,
//! within whatever family is composing. A drag the family cannot express — a
//! perspective quad turned concave, a warp mesh folded over itself — holds at the
//! last valid shape rather than tearing through it.

use stark_model::document::{LayerId, PerspectiveMap, TransformMap, WarpMap, rect_corners};
use stark_model::geom::{Affine2, Mat2, Vec2};

/// Where a pointer stands relative to the transform widget's ellipse — which
/// decides what a drag starting there does (§16.6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransformRegion {
    /// Strictly inside: dragging translates.
    Inside,
    /// On the rim: dragging turns and scales uniformly — tangential motion is
    /// pure rotation, radial motion pure scale, anything between blends the two.
    Rim,
    /// Outside: dragging stretches and shears along the grab direction, pinning
    /// the perpendicular diameter.
    Outside,
}

/// The transform gesture being composed (§16.6). The widget is
/// an **ellipse**: the image of a reference **circle** under the accumulated
/// linear map, so the widget's shape *is* the transform — it stays a circle
/// exactly as long as the transform is a similarity, and any distortion shows
/// as eccentricity. The affine the engine sees is derived on every change —
/// `x ↦ center + linear·(x − anchor)` — so a long drag is one accumulated
/// transform, never a chain of them, and the preview resamples the committed
/// tiles exactly once ("lossless" until "Done").
///
/// Every shaping gesture left-composes a world-space factor onto `linear`, each
/// solved so that **the grabbed point follows the pointer** within its family:
/// a similarity for the rim, a rank-1 stretch/shear for the outside. Gestures
/// that were never used leave their factors out entirely — a pure move keeps
/// `linear` bit-exactly the identity, which is what keeps it a pure translation
/// through the engine's exactness invariants (§16.4).
#[derive(Clone, Copy, PartialEq)]
pub struct TransformState {
    /// The layer whose selected paint is being transformed.
    pub layer: LayerId,
    /// The reference ellipse's centre — the hull's — in canvas px. Fixed for the
    /// mode's life; the affine pivots here.
    pub anchor: Vec2,
    /// The reference **circle**'s radius, canvas px. A circle, not the hull's
    /// own aspect: the widget's shape carries meaning — a circle says the
    /// accumulated transform is a similarity (rotation, uniform scale,
    /// translation), and any other shape says distortion has been applied.
    /// Encompasses the hull; floored so a hairline selection still mounts a
    /// grabbable widget.
    pub radius: f32,
    /// Where the gesture has carried the centre.
    pub center: Vec2,
    /// The accumulated linear map, applied about the centre.
    pub linear: Mat2,
}

impl TransformState {
    pub fn begin(layer: LayerId, hull: (Vec2, Vec2), min_radius: f32) -> Self {
        let anchor = (hull.0 + hull.1) * 0.5;
        let half = (hull.1 - hull.0) * 0.5;
        Self {
            layer,
            anchor,
            radius: half.max(Vec2::ZERO).length().max(min_radius),
            center: anchor,
            linear: Mat2::IDENTITY,
        }
    }

    /// Whether committing would change nothing — "Done" then skips the commit
    /// rather than spending an undo step on a no-op.
    pub fn is_identity(&self) -> bool {
        self.center == self.anchor && self.linear == Mat2::IDENTITY
    }

    /// The affine this gesture stands for — what the preview shows and "Done"
    /// commits.
    pub fn affine(&self) -> Affine2 {
        if self.linear == Mat2::IDENTITY {
            // The untouched-linear case stays a *pure* translation, not a
            // translation reconstituted through matrix arithmetic.
            return Affine2::from_translation(self.center - self.anchor);
        }
        Affine2::from_mat2_translation(self.linear, self.center - self.linear * self.anchor)
    }

    /// Classify a canvas-space pointer against the widget
    /// (§16.6): pull it back through the linear map into the reference circle's own
    /// space, where the test is a radius. `band` is the rim's grab half-width in
    /// canvas px, converted to circle units by the widget's local radius along
    /// the pointer's direction.
    pub fn region(&self, pointer: Vec2, band: f32) -> TransformRegion {
        let det = self.linear.determinant();
        if det.abs() < 1e-6 {
            // Collapsed to a sliver: everything reads as inside, so the widget
            // can still be moved (the shaping clamps keep this unreachable in
            // practice).
            return TransformRegion::Inside;
        }
        let u = (self.linear.inverse() * (pointer - self.center)) / self.radius;
        let rho = u.length();
        if rho < 1e-6 {
            return TransformRegion::Inside;
        }
        let local_radius = (self.linear * (self.radius * (u / rho))).length();
        let band = band / local_radius.max(1e-3);
        if rho < 1.0 - band {
            TransformRegion::Inside
        } else if rho <= 1.0 + band {
            TransformRegion::Rim
        } else {
            TransformRegion::Outside
        }
    }

    /// An inside drag: translate. `eps` (canvas px) snaps a jiggle back to the
    /// start, so touching the widget without meaning to never resamples.
    pub fn translated(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        Self {
            center: self.center + (to - from),
            ..self
        }
    }

    /// A rim drag: the similarity (rotation + uniform scale about the centre)
    /// that carries the grabbed point `from` exactly to the pointer `to` — the
    /// complex ratio `(to − c)/(from − c)`. Tangential motion is thereby pure
    /// rotation and radial motion pure scale, with no mode to pick.
    pub fn turned_scaled(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        let v0 = from - self.center;
        let v = to - self.center;
        let n = v0.length_squared();
        if n < 1e-6 {
            return self;
        }
        // Keep the widget grabbable: never scale below 5% in one gesture.
        let v = clamp_len(v, 0.05 * n.sqrt());
        let (a, b) = (v.dot(v0) / n, v0.perp_dot(v) / n);
        Self {
            linear: Mat2::from_cols(Vec2::new(a, b), Vec2::new(-b, a)) * self.linear,
            ..self
        }
    }

    /// An outside drag: the rank-1 update `I + (Δ ⊗ d̂)/λ` that carries the
    /// grabbed point exactly to the pointer while **pinning the diameter
    /// perpendicular to the grab** — radial pull scales along the grab
    /// direction, tangential drag shears, and everything on the pinned axis
    /// stays put, which is what makes the gesture predictable.
    pub fn stretched(self, from: Vec2, to: Vec2, eps: f32) -> Self {
        if to.distance(from) < eps {
            return self;
        }
        let v0 = from - self.center;
        let lambda = v0.length();
        if lambda < 1e-3 {
            return self;
        }
        let dir = v0 / lambda;
        let mut delta = to - from;
        // Pulling in past the pinned axis would run the determinant through
        // zero (the paint would vanish into a line, and the engine would refuse
        // the commit); floor the radial component at 90% pulled-in.
        let radial = delta.dot(dir) / lambda;
        if radial < -0.9 {
            delta += dir * ((-0.9 - radial) * lambda);
        }
        let g = Mat2::from_cols(
            Vec2::new(1.0 + delta.x * dir.x / lambda, delta.y * dir.x / lambda),
            Vec2::new(delta.x * dir.y / lambda, 1.0 + delta.y * dir.y / lambda),
        );
        Self {
            linear: g * self.linear,
            ..self
        }
    }

    /// Mirror left↔right, about the vertical axis through the centre.
    pub fn flipped_h(self) -> Self {
        Self {
            linear: Mat2::from_diagonal(Vec2::new(-1.0, 1.0)) * self.linear,
            ..self
        }
    }

    /// Mirror top↕bottom, about the horizontal axis through the centre.
    pub fn flipped_v(self) -> Self {
        Self {
            linear: Mat2::from_diagonal(Vec2::new(1.0, -1.0)) * self.linear,
            ..self
        }
    }
}

/// `v`, no shorter than `min` (direction kept; zero stays zero).
fn clamp_len(v: Vec2, min: f32) -> Vec2 {
    let len = v.length();
    if len < min && len > 1e-9 {
        v * (min / len)
    } else {
        v
    }
}

/// The transform mode's whole in-flight state (§16.6, §16.8, §16.9): which of
/// the three families the bar has selected, with that family's own gesture
/// state. One value in one signal, because the mode is *modal* — there is
/// always exactly one family composing, and switching families is an explicit
/// act on the bar (which carries the deformation along when the new family
/// contains the old one exactly, and commits it first when it cannot).
#[derive(Clone, Copy, PartialEq)]
pub enum TransformUi {
    /// The ellipse widget over the whole affine group — `rect` is the hull the
    /// mode was entered around, kept so a switch to a rect-scoped family knows
    /// its source rect.
    Affine {
        rect: (Vec2, Vec2),
        ts: TransformState,
    },
    Perspective(PerspectiveUi),
    Warp(WarpUi),
}

impl TransformUi {
    pub fn layer(&self) -> LayerId {
        match self {
            TransformUi::Affine { ts, .. } => ts.layer,
            TransformUi::Perspective(p) => p.layer,
            TransformUi::Warp(w) => w.layer,
        }
    }

    /// The map this gesture stands for — what the preview shows and "Done"
    /// commits.
    pub fn map(&self) -> TransformMap {
        match self {
            TransformUi::Affine { ts, .. } => TransformMap::Affine(ts.affine()),
            TransformUi::Perspective(p) => TransformMap::Perspective(p.map()),
            TransformUi::Warp(w) => TransformMap::Warp(w.map()),
        }
    }

    /// Whether committing would change nothing — "Done" then skips the commit
    /// rather than spending an undo step on a no-op.
    pub fn is_identity(&self) -> bool {
        match self {
            TransformUi::Affine { ts, .. } => ts.is_identity(),
            TransformUi::Perspective(p) => p.is_identity(),
            TransformUi::Warp(w) => w.is_identity(),
        }
    }
}

/// Where a pointer stands relative to the perspective quad (§16.8) — which
/// decides what a drag starting there does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuadRegion {
    /// On a corner handle: dragging carries that corner exactly, pinning the
    /// other three.
    Corner(usize),
    /// On an edge (named by its two corner indices): dragging shifts the whole
    /// edge — the foreshortening gesture.
    Edge(usize, usize),
    /// Inside the quad, or anywhere else: dragging translates all four
    /// corners together.
    Inside,
    Outside,
}

/// The perspective gesture being composed (§16.8): the image of the source
/// rect is a quad, its corners are the handles, and **the grabbed corner
/// follows the pointer exactly** — the map is defined as "the homography
/// putting the corners where the hand put them", so the widget cannot disagree
/// with the paint. A drag that would cross the quad (a concave or reflected
/// configuration — the map's horizon) holds at the last valid shape rather
/// than letting the paint fly through infinity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PerspectiveUi {
    pub layer: LayerId,
    /// The source rect the map acts on — the hull the mode was entered around.
    pub rect: (Vec2, Vec2),
    /// The corner images, in [`rect_corners`] order (00, 10, 01, 11).
    pub corners: [Vec2; 4],
}

impl PerspectiveUi {
    pub fn begin(layer: LayerId, rect: (Vec2, Vec2)) -> Self {
        Self {
            layer,
            rect,
            // The exact base values — identity is "the corners *are* the
            // rect's", bitwise (§16.4).
            corners: rect_corners(rect.0, rect.1),
        }
    }

    pub fn map(&self) -> PerspectiveMap {
        PerspectiveMap {
            min: self.rect.0,
            max: self.rect.1,
            corners: self.corners,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.corners == rect_corners(self.rect.0, self.rect.1)
    }

    /// Classify a canvas-space pointer: corner handles win, then edges, then
    /// the quad's inside. `grab` and `band` are canvas-px radii (screen px
    /// over the zoom).
    pub fn region(&self, p: Vec2, grab: f32, band: f32) -> QuadRegion {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.corners.iter().enumerate() {
            let d = p.distance(*c);
            if d <= grab && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        if let Some((i, _)) = best {
            return QuadRegion::Corner(i);
        }
        for (a, b) in EDGES {
            if segment_distance(p, self.corners[a], self.corners[b]) <= band {
                return QuadRegion::Edge(a, b);
            }
        }
        if point_in_quad(&self.corners, p) {
            QuadRegion::Inside
        } else {
            QuadRegion::Outside
        }
    }

    /// A corner drag, recomputed from the drag's start: the corner follows the
    /// pointer exactly while the shape stays convex; a pull past validity
    /// holds at `current` (the last valid shape) instead of tearing through
    /// the horizon. `eps` snaps a jiggle back to the start (§16.6).
    pub fn corner_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.corners[i] = start.corners[i] + delta;
        if next.map().usable() { next } else { current }
    }

    /// An edge drag: both of its corners follow together.
    pub fn edge_dragged(
        start: Self,
        current: Self,
        (a, b): (usize, usize),
        delta: Vec2,
        eps: f32,
    ) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.corners[a] = start.corners[a] + delta;
        next.corners[b] = start.corners[b] + delta;
        if next.map().usable() { next } else { current }
    }

    /// An inside (or outside) drag: the whole quad translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for c in &mut next.corners {
            *c += delta;
        }
        if next.map().usable() { next } else { current }
    }
}

/// The quad's edges as corner-index pairs, walking the boundary
/// (corner order is 00, 10, 01, 11, so the boundary is 0 → 1 → 3 → 2).
const EDGES: [(usize, usize); 4] = [(0, 1), (1, 3), (3, 2), (2, 0)];

fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Point-in-convex-quad, corners in (00, 10, 01, 11) order. Same-side test
/// against every boundary edge; orientation-agnostic so it also serves a
/// mid-drag shape the validity clamp has not yet vetoed.
fn point_in_quad(c: &[Vec2; 4], p: Vec2) -> bool {
    let b = [c[0], c[1], c[3], c[2]];
    let mut sign = 0.0f32;
    for i in 0..4 {
        let cross = (b[(i + 1) % 4] - b[i]).perp_dot(p - b[i]);
        if cross.abs() < 1e-9 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// Control points per axis of the warp gesture's mesh. 4×4 spans "gentle bend"
/// to "full puppet" through the smooth interpolation; the engine accepts up to
/// [`stark_model::document::MAX_WARP_GRID`] if a denser UI ever wants one.
pub const WARP_GRID: usize = 4;

/// Where a pointer stands relative to the warp mesh (§16.9).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshRegion {
    /// On a control point: dragging carries that point exactly.
    Point(usize),
    /// On the surface between points: dragging grabs the *paint* — the
    /// control points share the motion so the grabbed surface point follows
    /// the pointer exactly.
    Inside,
    /// Anywhere else: dragging translates the whole mesh.
    Outside,
}

/// The warp gesture being composed (§16.9): a 4×4 control grid over the source
/// rect, smoothly interpolated by the engine's own surface — the mesh the
/// overlay draws is sampled from the very lattice the paint resamples through,
/// so the curves *are* the deformation. Two ways to shape it, both
/// exact-follow: drag a control point, or grab the surface anywhere and the
/// least-norm control move puts that spot of paint under the pointer
/// (`Δpᵢ = Bᵢ·Δ / ΣB²`, with `B` the surface basis at the grab). A drag that
/// would fold the mesh holds at the last valid shape.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WarpUi {
    pub layer: LayerId,
    pub rect: (Vec2, Vec2),
    /// Row-major control points — the images of the rect's uniform grid.
    pub points: [Vec2; WARP_GRID * WARP_GRID],
}

impl WarpUi {
    pub fn begin(layer: LayerId, rect: (Vec2, Vec2)) -> Self {
        // The engine's own base points, not a re-derivation: identity is "the
        // points *are* these values", bitwise (§16.4).
        let base = WarpMap::identity(rect.0, rect.1, WARP_GRID as u32, WARP_GRID as u32);
        let mut points = [Vec2::ZERO; WARP_GRID * WARP_GRID];
        points.copy_from_slice(&base.points);
        Self {
            layer,
            rect,
            points,
        }
    }

    pub fn map(&self) -> WarpMap {
        WarpMap {
            min: self.rect.0,
            max: self.rect.1,
            cols: WARP_GRID as u32,
            rows: WARP_GRID as u32,
            points: self.points.to_vec(),
        }
    }

    pub fn is_identity(&self) -> bool {
        let base = WarpMap::identity(self.rect.0, self.rect.1, WARP_GRID as u32, WARP_GRID as u32);
        self.points[..] == base.points[..]
    }

    /// Classify a canvas-space pointer: the nearest control point within
    /// `grab` wins; otherwise inside the mesh's boundary polygon is a surface
    /// grab; otherwise a whole-mesh translate.
    pub fn region(&self, p: Vec2, grab: f32) -> MeshRegion {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.points.iter().enumerate() {
            let d = p.distance(*c);
            if d <= grab && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        if let Some((i, _)) = best {
            return MeshRegion::Point(i);
        }
        if point_in_polygon(&self.boundary(), p) {
            MeshRegion::Inside
        } else {
            MeshRegion::Outside
        }
    }

    /// The mesh's border control points, walking the boundary clockwise.
    fn boundary(&self) -> Vec<Vec2> {
        let n = WARP_GRID;
        let at = |i: usize, j: usize| self.points[j * n + i];
        let mut out = Vec::with_capacity(4 * (n - 1));
        for i in 0..n - 1 {
            out.push(at(i, 0));
        }
        for j in 0..n - 1 {
            out.push(at(n - 1, j));
        }
        for i in (1..n).rev() {
            out.push(at(i, n - 1));
        }
        for j in (1..n).rev() {
            out.push(at(0, j));
        }
        out
    }

    /// The grid fraction whose surface point is nearest `p`, with the surface
    /// basis there and its squared norm — everything a surface drag needs,
    /// computed once at the press. Coarse scan plus local refinement; the
    /// surface is smooth and unfolded, so nearest-on-a-grid converges fast.
    pub fn grab(&self, p: Vec2) -> (Vec2, [f32; WARP_GRID * WARP_GRID], f32) {
        let map = self.map();
        // The delta grid hoisted out of the search: this is 81 coarse probes plus
        // six refinement passes of 25, and `WarpMap::eval` rebuilds the grid on
        // every call. Same arithmetic to the bit (`Prepared::eval`), which §16.4's
        // identity invariant requires.
        let surface = map.prepared();
        let mut best = (Vec2::splat(0.5), f32::INFINITY);
        let scan = |from: Vec2, step: f32, best: &mut (Vec2, f32)| {
            for j in -2..=2i32 {
                for i in -2..=2i32 {
                    let t =
                        (from + Vec2::new(i as f32, j as f32) * step).clamp(Vec2::ZERO, Vec2::ONE);
                    let d = surface.eval(t).distance_squared(p);
                    if d < best.1 {
                        *best = (t, d);
                    }
                }
            }
        };
        for j in 0..=8 {
            for i in 0..=8 {
                let t = Vec2::new(i as f32 / 8.0, j as f32 / 8.0);
                let d = surface.eval(t).distance_squared(p);
                if d < best.1 {
                    best = (t, d);
                }
            }
        }
        let mut step = 1.0 / 16.0;
        for _ in 0..6 {
            let from = best.0;
            scan(from, step, &mut best);
            step *= 0.5;
        }
        let basis = map.basis(best.0);
        let mut b = [0.0f32; WARP_GRID * WARP_GRID];
        b.copy_from_slice(&basis);
        let norm: f32 = b.iter().map(|w| w * w).sum();
        (best.0, b, norm.max(1e-6))
    }

    /// A control-point drag, recomputed from the drag's start; a fold holds at
    /// `current`, the last valid shape.
    pub fn point_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        next.points[i] = start.points[i] + delta;
        if next.map().usable() { next } else { current }
    }

    /// A surface drag: the least-norm control move that carries the grabbed
    /// surface point exactly with the pointer — the hand holds the paint, not
    /// a handle (§16.9). `basis`/`norm` come from [`grab`](Self::grab) at the
    /// press.
    pub fn surface_dragged(
        start: Self,
        current: Self,
        basis: &[f32; WARP_GRID * WARP_GRID],
        norm: f32,
        delta: Vec2,
        eps: f32,
    ) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for (pt, w) in next.points.iter_mut().zip(basis) {
            *pt += delta * (*w / norm);
        }
        if next.map().usable() { next } else { current }
    }

    /// An outside drag: the whole mesh translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        if delta.length() < eps {
            return start;
        }
        let mut next = start;
        for pt in &mut next.points {
            *pt += delta;
        }
        if next.map().usable() { next } else { current }
    }
}

/// Even-odd point-in-polygon over an arbitrary boundary walk.
fn point_in_polygon(poly: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    fn state() -> TransformState {
        TransformState::begin(
            LayerId(0),
            (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0)),
            10.0,
        )
    }

    #[test]
    fn untouched_gesture_is_the_identity() {
        let ts = state();
        assert!(ts.is_identity());
        assert_eq!(ts.affine(), Affine2::IDENTITY);
    }

    #[test]
    fn translation_alone_keeps_the_linear_part_exact() {
        let ts = state().translated(Vec2::ZERO, Vec2::new(37.5, -12.0), 0.5);
        assert_eq!(ts.linear, Mat2::IDENTITY);
        let a = ts.affine();
        assert_eq!(a.matrix2, Mat2::IDENTITY);
        assert_eq!(a.translation, Vec2::new(37.5, -12.0));
    }

    #[test]
    fn a_sub_epsilon_jiggle_changes_nothing() {
        let ts = state();
        let from = Vec2::new(100.0, 0.0);
        assert!(ts.turned_scaled(from, from + Vec2::splat(0.1), 0.5) == ts);
        assert!(ts.stretched(from, from + Vec2::splat(0.1), 0.5) == ts);
        assert!(ts.translated(from, from + Vec2::splat(0.1), 0.5) == ts);
    }

    #[test]
    fn rim_drag_carries_the_grab_point_to_the_pointer() {
        // Grab east, drag to twice-north: a quarter turn plus a 2× scale.
        let ts = state();
        let from = ts.center + Vec2::new(100.0, 0.0);
        let to = ts.center + Vec2::new(0.0, 200.0);
        let turned = ts.turned_scaled(from, to, 0.5);
        let moved = turned.linear * (from - ts.center);
        assert!((moved - (to - ts.center)).length() < 1e-3, "got {moved:?}");
        assert!(turned.linear.determinant() > 0.0);
    }

    #[test]
    fn outside_drag_pins_the_perpendicular_diameter() {
        // Grab east of the widget and drag: the north–south diameter must not move.
        let ts = state();
        let from = ts.center + Vec2::new(300.0, 0.0);
        let to = from + Vec2::new(80.0, 55.0);
        let stretched = ts.stretched(from, to, 0.5);
        let moved = stretched.linear * (from - ts.center);
        assert!((moved - (to - ts.center)).length() < 1e-3, "got {moved:?}");
        let pinned = stretched.linear * Vec2::new(0.0, 1.0);
        assert!(
            (pinned - Vec2::new(0.0, 1.0)).length() < 1e-6,
            "got {pinned:?}"
        );
    }

    #[test]
    fn flips_are_involutions() {
        let ts = state().flipped_h().flipped_v();
        assert!(!ts.is_identity());
        let back = ts.flipped_v().flipped_h();
        assert!(back.is_identity(), "four mirrors must cancel bit-exactly");
    }

    #[test]
    fn the_reference_is_a_circle_matching_the_hull_ellipses_area() {
        // A 200×100 hull: the circle's area equals the inscribed ellipse's
        // (π·100·50), i.e. r = √(100·50) — not the ellipse itself, because a
        // circle is what says "no distortion yet" (§16.6).
        let r = state().radius;
        assert!((r - 100.0f32.hypot(50.0)).abs() < 1e-3, "got {r}");
    }

    #[test]
    fn regions_classify_by_the_deformed_circle() {
        let ts = state();
        let (c, r) = (ts.center, ts.radius);
        assert_eq!(ts.region(c, 4.0), TransformRegion::Inside);
        assert_eq!(
            ts.region(c + Vec2::new(0.6 * r, 0.0), 4.0),
            TransformRegion::Inside
        );
        assert_eq!(ts.region(c + Vec2::new(r, 0.0), 4.0), TransformRegion::Rim);
        assert_eq!(ts.region(c + Vec2::new(0.0, -r), 4.0), TransformRegion::Rim);
        assert_eq!(
            ts.region(c + Vec2::new(1.6 * r, 0.0), 4.0),
            TransformRegion::Outside
        );

        // Stretch the widget to 2× along x: the rim moves with it.
        let wide = ts.stretched(c + Vec2::new(r, 0.0), c + Vec2::new(2.0 * r, 0.0), 0.5);
        assert_eq!(
            wide.region(c + Vec2::new(2.0 * r, 0.0), 4.0),
            TransformRegion::Rim
        );
        assert_eq!(
            wide.region(c + Vec2::new(r, 0.0), 4.0),
            TransformRegion::Inside
        );
    }
}

#[cfg(test)]
mod gesture_tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0))
    }

    #[test]
    fn a_fresh_perspective_is_the_identity_and_usable() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert!(p.is_identity());
        assert!(p.map().usable());
        assert!(TransformUi::Perspective(p).is_identity());
    }

    #[test]
    fn a_dragged_corner_lands_under_the_pointer() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        let delta = Vec2::new(-30.0, 22.0);
        let dragged = PerspectiveUi::corner_dragged(p, p, 3, delta, 0.5);
        assert_eq!(dragged.corners[3], p.corners[3] + delta);
        assert!(!dragged.is_identity());
        assert!(dragged.map().usable());
    }

    #[test]
    fn a_corner_pulled_across_the_quad_holds_at_the_last_valid_shape() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        // Almost across: still convex, accepted.
        let near = PerspectiveUi::corner_dragged(p, p, 0, Vec2::new(150.0, 60.0), 0.5);
        assert!(near.map().usable());
        // All the way across the opposite corner: the candidate is concave, so
        // the drag holds at `current` rather than folding the map.
        let held = PerspectiveUi::corner_dragged(p, near, 0, Vec2::new(500.0, 300.0), 0.5);
        assert_eq!(held, near);
    }

    #[test]
    fn quad_regions_classify_corners_edges_and_inside() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert_eq!(p.region(p.corners[1], 8.0, 5.0), QuadRegion::Corner(1));
        // Mid-top edge.
        let mid = (p.corners[0] + p.corners[1]) * 0.5;
        assert_eq!(p.region(mid, 8.0, 5.0), QuadRegion::Edge(0, 1));
        assert_eq!(p.region(Vec2::ZERO, 8.0, 5.0), QuadRegion::Inside);
        assert_eq!(
            p.region(Vec2::new(400.0, 400.0), 8.0, 5.0),
            QuadRegion::Outside
        );
    }

    #[test]
    fn a_fresh_warp_is_the_identity_and_usable() {
        let w = WarpUi::begin(LayerId(0), rect());
        assert!(w.is_identity());
        assert!(w.map().usable());
    }

    #[test]
    fn a_dragged_control_point_lands_under_the_pointer() {
        let w = WarpUi::begin(LayerId(0), rect());
        let delta = Vec2::new(14.0, -9.0);
        let dragged = WarpUi::point_dragged(w, w, 5, delta, 0.5);
        assert_eq!(dragged.points[5], w.points[5] + delta);
        assert!(dragged.map().usable());
        assert!(!dragged.is_identity());
    }

    #[test]
    fn a_surface_drag_carries_the_grabbed_paint_exactly() {
        let w = WarpUi::begin(LayerId(0), rect());
        let grab_at = Vec2::new(20.0, -10.0);
        let (t, basis, norm) = w.grab(grab_at);
        let before = w.map().eval(t);
        assert!(before.distance(grab_at) < 1.0, "grab missed: {before:?}");
        let delta = Vec2::new(18.0, 12.0);
        let dragged = WarpUi::surface_dragged(w, w, &basis, norm, delta, 0.5);
        let after = dragged.map().eval(t);
        assert!(
            after.distance(before + delta) < 0.1,
            "the paint under the finger moved {:?}, the finger moved {delta:?}",
            after - before
        );
    }

    #[test]
    fn a_folding_drag_holds_at_the_last_valid_shape() {
        let w = WarpUi::begin(LayerId(0), rect());
        // Drag an interior point far past its neighbour: the mesh would fold.
        let held = WarpUi::point_dragged(w, w, 5, Vec2::new(250.0, 0.0), 0.5);
        assert_eq!(held, w, "a fold must hold, not tear");
    }

    #[test]
    fn mesh_regions_classify_points_surface_and_outside() {
        let w = WarpUi::begin(LayerId(0), rect());
        assert_eq!(w.region(w.points[0], 8.0), MeshRegion::Point(0));
        assert_eq!(w.region(Vec2::new(15.0, 5.0), 8.0), MeshRegion::Inside);
        assert_eq!(w.region(Vec2::new(500.0, 0.0), 8.0), MeshRegion::Outside);
    }

    #[test]
    fn sub_epsilon_jiggles_snap_back_to_the_start() {
        let p = PerspectiveUi::begin(LayerId(0), rect());
        assert_eq!(
            PerspectiveUi::corner_dragged(p, p, 2, Vec2::splat(0.1), 0.5),
            p
        );
        let w = WarpUi::begin(LayerId(0), rect());
        assert_eq!(WarpUi::point_dragged(w, w, 5, Vec2::splat(0.1), 0.5), w);
        assert_eq!(WarpUi::translated(w, w, Vec2::splat(0.1), 0.5), w);
    }
}
