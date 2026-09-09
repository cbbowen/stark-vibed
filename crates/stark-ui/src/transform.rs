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
//! That is why it is a file of its own. It lived in the web frontend's `state`, which is about
//! the app's signals and the one door to the engine, and this is the part of that
//! file that could be tested — most of the crate's transform tests are here, and they
//! are the ones that can say a rim drag really does carry the grabbed point to the
//! pointer and that four mirrors really do cancel bit-exactly. Sitting inside the state
//! module, they were the tests hardest to find and the code most likely to be read
//! as UI plumbing.
//!
//! Three shapes, one rule: **the grabbed point follows the pointer exactly**,
//! within whatever family is composing. A drag the family cannot express — a
//! perspective quad turned concave, a warp mesh folded over itself, an affine
//! collapsed onto a line — holds at the last valid shape rather than tearing through
//! it. That rule is written **once**, in `shaped`, and every gesture below is only
//! its own edit.
//!
//! Two things a frontend would otherwise have to derive for itself live here as well,
//! because deriving them twice is how the two drawings came to disagree:
//!
//! - [`Bands`] — the grab widths, in canvas px. Every one of them is a screen-px
//!   figure over the zoom, and the division is the part a call site can forget.
//! - [`outline`] / [`grid`] / [`handles`] — the widget's geometry, in canvas px. A
//!   frontend is left with the stroking, which is the only part its toolkit owns.

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
#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// Mount around `hull`, with `min_radius` the canvas-px floor on the circle.
    ///
    /// An inverted hull mounts the same circle as its normalized twin: the radius is a
    /// length, and `bounds::inflate` already answers "what is a backwards rectangle"
    /// by normalizing it rather than collapsing it. Two answers to that would be two
    /// widgets for one selection.
    pub fn begin(layer: LayerId, hull: (Vec2, Vec2), min_radius: f32) -> Self {
        let anchor = (hull.0 + hull.1) * 0.5;
        let half = (hull.1 - hull.0) * 0.5;
        Self {
            layer,
            anchor,
            radius: half.length().max(min_radius),
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
    /// canvas px (a [`Bands`]'s rim), converted to circle units by the widget's local
    /// radius along the pointer's direction.
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

    /// An inside drag: translate.
    pub fn translated(start: Self, current: Self, from: Vec2, to: Vec2, eps: f32) -> Self {
        shaped(start, current, to - from, eps, |next| {
            next.center = start.center + (to - from);
        })
    }

    /// A rim drag: the similarity (rotation + uniform scale about the centre)
    /// that carries the grabbed point `from` exactly to the pointer `to` — the
    /// complex ratio `(to − c)/(from − c)`. Tangential motion is thereby pure
    /// rotation and radial motion pure scale, with no mode to pick.
    pub fn turned_scaled(start: Self, current: Self, from: Vec2, to: Vec2, eps: f32) -> Self {
        shaped(start, current, to - from, eps, |next| {
            let v0 = from - start.center;
            let v = to - start.center;
            let n = v0.length_squared();
            if n < 1e-6 {
                return;
            }
            // Keep the widget grabbable: never scale below 5% in one gesture.
            let v = clamp_len(v, 0.05 * n.sqrt());
            let (a, b) = (v.dot(v0) / n, v0.perp_dot(v) / n);
            next.linear = Mat2::from_cols(Vec2::new(a, b), Vec2::new(-b, a)) * start.linear;
        })
    }

    /// An outside drag: the rank-1 update `I + (Δ ⊗ d̂)/λ` that carries the
    /// grabbed point exactly to the pointer while **pinning the diameter
    /// perpendicular to the grab** — radial pull scales along the grab
    /// direction, tangential drag shears, and everything on the pinned axis
    /// stays put, which is what makes the gesture predictable.
    pub fn stretched(start: Self, current: Self, from: Vec2, to: Vec2, eps: f32) -> Self {
        shaped(start, current, to - from, eps, |next| {
            let v0 = from - start.center;
            let lambda = v0.length();
            if lambda < 1e-3 {
                return;
            }
            let dir = v0 / lambda;
            let mut delta = to - from;
            // Pulling in past the pinned axis would run the determinant through
            // zero (the paint would vanish into a line, and the engine would refuse
            // the commit); floor the radial component at 90% pulled-in. That bounds
            // *this* gesture; what bounds a run of them is `shaped`'s own check,
            // since each press starts from what the last one accumulated.
            let radial = delta.dot(dir) / lambda;
            if radial < -0.9 {
                delta += dir * ((-0.9 - radial) * lambda);
            }
            let g = Mat2::from_cols(
                Vec2::new(1.0 + delta.x * dir.x / lambda, delta.y * dir.x / lambda),
                Vec2::new(delta.x * dir.y / lambda, 1.0 + delta.y * dir.y / lambda),
            );
            next.linear = g * start.linear;
        })
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

/// A gesture shape that can say whether the map it stands for may be applied.
///
/// Private, and one method, because its whole job is to let [`shaped`] state the
/// module's headline promise **once**. It was stated six times; a rule written six
/// times is a rule that can be left out a seventh, and the affine family is where it
/// had been.
trait Shapeable: Copy {
    fn usable(&self) -> bool;
}

impl Shapeable for TransformState {
    fn usable(&self) -> bool {
        stark_model::document::affine_usable(self.affine())
    }
}

impl Shapeable for PerspectiveUi {
    fn usable(&self) -> bool {
        self.map().usable()
    }
}

impl Shapeable for WarpUi {
    fn usable(&self) -> bool {
        self.map().usable()
    }
}

/// The shape `edit` makes of `start`, or the last shape that was allowed.
///
/// Three outcomes, and every gesture in the module has all three:
///
/// - a travel under `eps` — or a pointer that is not a number at all — is a jiggle,
///   and returns `start` untouched, so an accidental touch never resamples (§16.6).
///   Non-finite has to be named: `NaN < eps` is *false*, so the snap alone would wave
///   a NaN pointer through into the map, where `is_identity` reads false and "Done"
///   commits something the engine refuses;
/// - an edit the family can express is the answer;
/// - one it cannot **holds at `current`**, the last valid shape, rather than tearing
///   through the horizon or the fold (§16.8, §16.9).
///
/// `edit` may bail by returning without touching `next`, which is the degenerate-grab
/// case: an untouched `next` is `start`, which is where those want to end up anyway.
fn shaped<T: Shapeable>(
    start: T,
    current: T,
    delta: Vec2,
    eps: f32,
    edit: impl FnOnce(&mut T),
) -> T {
    if !delta.is_finite() || delta.length() < eps {
        return start;
    }
    let mut next = start;
    edit(&mut next);
    if next.usable() { next } else { current }
}

/// The transform mode's whole in-flight state (§16.6, §16.8, §16.9): which of
/// the three families the bar has selected, with that family's own gesture
/// state. One value in one signal, because the mode is *modal* — there is
/// always exactly one family composing, and switching families is an explicit
/// act on the bar (which carries the deformation along when the new family
/// contains the old one exactly, and commits it first when it cannot).
#[derive(Clone, Copy, PartialEq, Debug)]
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

    /// Which family is composing.
    pub fn family(&self) -> Family {
        match self {
            TransformUi::Affine { .. } => Family::Free,
            TransformUi::Perspective(_) => Family::Perspective,
            TransformUi::Warp(_) => Family::Warp,
        }
    }

    /// The source rect this gesture was mounted around — the paint it is holding,
    /// before the gesture moved it.
    pub fn rect(&self) -> (Vec2, Vec2) {
        match self {
            TransformUi::Affine { rect, .. } => *rect,
            TransformUi::Perspective(p) => p.rect,
            TransformUi::Warp(w) => w.rect,
        }
    }

    /// Where the paint has been carried to, as an axis-aligned bound — what a fresh
    /// gesture should be mounted around after this one is committed.
    pub fn image_rect(&self) -> (Vec2, Vec2) {
        match self.map() {
            TransformMap::Affine(a) => {
                let rect = self.rect();
                let corners = rect_corners(rect.0, rect.1).map(|c| a.transform_point2(c));
                crate::bounds::aabb(corners).unwrap_or(rect)
            }
            TransformMap::Perspective(p) => p.image_aabb().unwrap_or((p.min, p.max)),
            TransformMap::Warp(w) => w.image_aabb().unwrap_or((w.min, w.max)),
        }
    }
}

/// Half-width of the rim's / an edge's grab band, screen px.
const RIM_BAND_PX: f32 = 10.0;

/// Grab radius of a corner or control-point handle, screen px.
///
/// The one figure of the five that is public, and for one reader: the native
/// frontend *draws* a handle smaller than this and asserts the pair at compile time,
/// so a target stays easier to hit than it looks. A drawn size is the frontend's and
/// the grab radius is this module's, which is exactly why the relation between them
/// has to be stated somewhere both can see.
pub const HANDLE_PX: f32 = 14.0;

/// Screen-px floor for the widget's radius at entry, so a hairline selection still
/// mounts a circle with an inside to translate by.
const MIN_RADIUS_PX: f32 = 28.0;

/// Screen-px floor for a perspective/warp source rect's extent at entry — a hairline
/// hull still mounts a quad with corners apart enough to grab.
const MIN_RECT_PX: f32 = 56.0;

/// Pointer travel below which a gesture reads as a jiggle and snaps back to its start
/// (screen px): an accidental touch must never resample the paint.
const SNAP_PX: f32 = 2.0;

/// The widths a gesture is measured against, in **canvas** px (§16.6).
///
/// One value rather than loose arguments, because every one of them is a screen-px
/// constant over the zoom and the division is the part a call site can forget. It was
/// spelled at six call sites across two frontends — each of which grew a private
/// helper for it — while [`mount`] took the raw zoom and divided internally, so the
/// interface disagreed with itself in the one place a mix-up makes no noise: a band
/// left in screen px is a widget that is merely hard to grab at some magnifications.
///
/// Fields are private and there is one constructor, so screen px cannot arrive here
/// by accident.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bands {
    /// The rim's / an edge's grab half-width.
    rim: f32,
    /// A corner's or control point's grab radius.
    handle: f32,
    /// Travel below which a gesture is a jiggle.
    snap: f32,
    /// Floor on the affine widget's radius at entry.
    min_radius: f32,
    /// Floor on a rect-scoped family's source extent at entry.
    min_rect: f32,
}

impl Bands {
    /// The bands at `zoom`, so a handle is equally grabbable at any magnification.
    ///
    /// A zoom that is not a positive number divides into bands that are infinite,
    /// negative or NaN, and an infinite rim band reads the *whole plane* as the rim —
    /// a gesture nobody asked for, which is worse than one that misses.
    pub fn at(zoom: f32) -> Self {
        let zoom = if zoom.is_finite() && zoom > 0.0 {
            zoom
        } else {
            1.0
        };
        Self {
            rim: RIM_BAND_PX / zoom,
            handle: HANDLE_PX / zoom,
            snap: SNAP_PX / zoom,
            min_radius: MIN_RADIUS_PX / zoom,
            min_rect: MIN_RECT_PX / zoom,
        }
    }
}

/// The three families, as a selector.
///
/// An enum of its own rather than a `match` on [`TransformUi`], because a bar has to
/// name the family the gesture is *not* currently in — that is what its chips are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Free,
    Perspective,
    Warp,
}

/// The layer a transform would act on, and the rectangle to mount it around.
///
/// Both answers come off one read of `ObservableState`, so they cannot be taken from
/// two different moments — a hull from before a layer change and a layer from after
/// would mount the widget around paint it is not holding.
pub struct Entry {
    pub layer: LayerId,
    /// The selection's hull, or the paint's, or what is on screen — see [`entry`].
    pub hull: (Vec2, Vec2),
}

/// What entering transform mode should act on, or `None` when nothing can be.
///
/// The layer is the active one, or the topmost paintable layer when a matte is
/// selected — a matte refuses transforms the same way it refuses strokes (§15.2).
///
/// The rectangle is a ladder, because the widget must always exist: the selection's
/// analytic hull; failing that (select-all, an inversion — an *unbounded* selection,
/// which holds the whole layer) the painted content's bounds; failing that, on an
/// empty canvas, what is on screen.
pub fn entry(o: &stark_engine::ObservableState) -> Option<Entry> {
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id)?;
    let hull = o
        .selection_hull
        .or_else(|| crate::bounds::content(o))
        .unwrap_or_else(|| crate::bounds::view(o));
    Some(Entry { layer, hull })
}

/// Mount a fresh gesture of `family` around `rect`.
pub fn mount(layer: LayerId, family: Family, rect: (Vec2, Vec2), bands: Bands) -> TransformUi {
    match family {
        Family::Free => TransformUi::Affine {
            rect,
            ts: TransformState::begin(layer, rect, bands.min_radius),
        },
        Family::Perspective => TransformUi::Perspective(PerspectiveUi::begin(
            layer,
            crate::bounds::inflate(rect, bands.min_rect),
        )),
        Family::Warp => TransformUi::Warp(WarpUi::begin(
            layer,
            crate::bounds::inflate(rect, bands.min_rect),
        )),
    }
}

/// What a press on the warp mesh took hold of (§16.9), with the solve the one region
/// that needs it carries.
///
/// Not [`MeshRegion`], which is where the pointer *is*: this is what the press
/// **decided**, and the difference is the basis. Folding it into the surface arm is
/// what keeps a point drag and a whole-mesh translate from carrying a basis they must
/// never read — a fabricated `[0.0; 16]` was two regions' worth of state that could
/// only ever be wrong, and 64 bytes of it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MeshGrab {
    /// A control point, by index: it follows the pointer exactly.
    Point(usize),
    /// The surface itself, with the least-norm solve at the grabbed spot — see
    /// [`WarpUi::grab`]. Solved once at the press, because it is a property of *where
    /// the paint was grabbed*; re-solving per move would let the grabbed point slide
    /// out from under the pointer.
    Surface {
        basis: [f32; WARP_GRID * WARP_GRID],
        /// `Σ B²`, the divisor of the least-norm move.
        norm_sq: f32,
    },
    /// Outside the mesh: the whole thing translates.
    Translate,
}

/// What a press on the widget took hold of, and everything the drag needs from the
/// moment it landed.
///
/// **The whole of a transform drag is this type plus [`follow`](Self::follow).** Each
/// arm carries two shapes of its family. `start` is where the gesture began — every
/// shaping function takes the start rather than the previous step (see the module
/// note), so a drag is one accumulated map instead of a chain of them. `held` is the
/// last shape the family could express, which is a fact about *what the drag has
/// reached*: a pull past the horizon or into a fold stops there and stays there.
///
/// The grab owns `held` rather than being handed it, the way `nav::Mode` owns what
/// its press decided. It was an argument, and an argument of a type that could name
/// another family — a mismatch the callee quietly repaired, and which cost both
/// frontends a second read of live state on every pointer move.
#[derive(Clone, Copy, PartialEq, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "the arms are the three families, and a mesh's two shapes are sixteen \
              control points each. Boxing that arm would buy back 250 bytes at the \
              cost of an allocation per press and of `Copy`, which is what makes a \
              grab a value both frontends can hold in a signal or a field; and there \
              is exactly one of these alive at a time, boxed already where a frontend \
              cares (the native `Held::Transform`)"
)]
pub enum Grab {
    Affine {
        region: TransformRegion,
        from: Vec2,
        rect: (Vec2, Vec2),
        start: TransformState,
        held: TransformState,
    },
    Quad {
        region: QuadRegion,
        from: Vec2,
        start: PerspectiveUi,
        held: PerspectiveUi,
    },
    Mesh {
        region: MeshGrab,
        from: Vec2,
        start: WarpUi,
        held: WarpUi,
    },
}

/// What a press would do here — for the cursor a frontend shows.
///
/// Three answers rather than each frontend's own spelling of them: a CSS cursor
/// string and a `winit` cursor icon are different alphabets for one classification,
/// and the classification is the part that could disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hint {
    /// The whole thing moves.
    Move,
    /// Something is taken hold of and carried: a rim, a corner, a control point, the
    /// surface itself.
    Hold,
    /// The shape is worked from outside it — the affine's stretch and shear.
    Shape,
}

/// What a press at `at` would do, without deciding it.
///
/// **The resting pointer's answer**, and it is a separate function because taking a
/// grab is not free: a warp press solves the least-norm basis, which is 231 surface
/// evaluations, and a hovering mouse asked for one of those per move and then read
/// three bits of it. This calls only the region tests.
pub fn hint_at(ui: &TransformUi, at: Vec2, bands: Bands) -> Hint {
    match ui {
        TransformUi::Affine { ts, .. } => match ts.region(at, bands.rim) {
            TransformRegion::Inside => Hint::Move,
            TransformRegion::Rim => Hint::Hold,
            TransformRegion::Outside => Hint::Shape,
        },
        TransformUi::Perspective(p) => match p.region(at, bands.handle, bands.rim) {
            QuadRegion::Corner(_) | QuadRegion::Edge(..) => Hint::Hold,
            QuadRegion::Inside | QuadRegion::Outside => Hint::Move,
        },
        TransformUi::Warp(w) => match w.region(at, bands.handle) {
            MeshRegion::Point(_) | MeshRegion::Inside => Hint::Hold,
            MeshRegion::Outside => Hint::Move,
        },
    }
}

impl Grab {
    /// Classify a press at `at` (canvas px) and take hold.
    pub fn take(ui: TransformUi, at: Vec2, bands: Bands) -> Self {
        match ui {
            TransformUi::Affine { rect, ts } => Grab::Affine {
                region: ts.region(at, bands.rim),
                from: at,
                rect,
                start: ts,
                held: ts,
            },
            TransformUi::Perspective(p) => Grab::Quad {
                region: p.region(at, bands.handle, bands.rim),
                from: at,
                start: p,
                held: p,
            },
            TransformUi::Warp(w) => Grab::Mesh {
                region: match w.region(at, bands.handle) {
                    MeshRegion::Point(i) => MeshGrab::Point(i),
                    // The one region with a solve, and the only one that pays for it.
                    MeshRegion::Inside => {
                        let g = w.grab(at);
                        MeshGrab::Surface {
                            basis: g.basis,
                            norm_sq: g.norm_sq,
                        }
                    }
                    MeshRegion::Outside => MeshGrab::Translate,
                },
                from: at,
                start: w,
                held: w,
            },
        }
    }

    /// What the gesture stands for with the pointer at `at`, advancing the shape the
    /// validity clamps hold at.
    pub fn follow(&mut self, at: Vec2, bands: Bands) -> TransformUi {
        let snap = bands.snap;
        match self {
            Grab::Affine {
                region,
                from,
                rect,
                start,
                held,
            } => {
                let ts = match region {
                    TransformRegion::Inside => {
                        TransformState::translated(*start, *held, *from, at, snap)
                    }
                    TransformRegion::Rim => {
                        TransformState::turned_scaled(*start, *held, *from, at, snap)
                    }
                    TransformRegion::Outside => {
                        TransformState::stretched(*start, *held, *from, at, snap)
                    }
                };
                *held = ts;
                TransformUi::Affine { rect: *rect, ts }
            }
            Grab::Quad {
                region,
                from,
                start,
                held,
            } => {
                let delta = at - *from;
                let next = match *region {
                    QuadRegion::Corner(i) => {
                        PerspectiveUi::corner_dragged(*start, *held, i, delta, snap)
                    }
                    QuadRegion::Edge(a, b) => {
                        PerspectiveUi::edge_dragged(*start, *held, (a, b), delta, snap)
                    }
                    QuadRegion::Inside | QuadRegion::Outside => {
                        PerspectiveUi::translated(*start, *held, delta, snap)
                    }
                };
                *held = next;
                TransformUi::Perspective(next)
            }
            Grab::Mesh {
                region,
                from,
                start,
                held,
            } => {
                let delta = at - *from;
                let next = match region {
                    MeshGrab::Point(i) => WarpUi::point_dragged(*start, *held, *i, delta, snap),
                    MeshGrab::Surface { basis, norm_sq } => {
                        WarpUi::surface_dragged(*start, *held, basis, *norm_sq, delta, snap)
                    }
                    MeshGrab::Translate => WarpUi::translated(*start, *held, delta, snap),
                };
                *held = next;
                TransformUi::Warp(next)
            }
        }
    }
}

/// What switching to another family costs (§16.8, §16.9).
pub enum Switch {
    /// Already there. Nothing to do, and saying so is not the same as an identity
    /// carry — a bar that re-mounted on every press of the lit chip would throw away
    /// the gesture in hand.
    Nothing,
    /// The deformation rode across exactly. Replace the gesture and keep going —
    /// **and show it**: the map is a different one from the map on screen, exact to
    /// within a resample, so the preview owes the new family's own picture.
    Carried(TransformUi),
    /// Nothing was composed, so the switch is free: no undo step, and nothing to
    /// show that is not already shown. Separate from [`Carried`](Self::Carried)
    /// because a carry has a deformation to re-preview and this has none, and a
    /// preview of the identity is not free — it resamples the selected paint.
    Fresh(TransformUi),
    /// Commit what is composed — one honest undo step — then reopen around where the
    /// paint now is. What a lossy carry would have hidden.
    Commit {
        map: TransformMap,
        then: TransformUi,
    },
}

/// Decide what switching `ui` to `to` should do.
///
/// Three outcomes, and which one is a fact about the two families rather than a
/// preference:
///
/// - An **orientation-preserving affine is exactly** a parallelogram perspective, and
///   exactly a mesh whose smooth surface reproduces it — cubic interpolation
///   reproduces affine functions. So it carries, and the artist keeps composing.
/// - A **mirrored or degenerate** affine has no image in either: both of those maps
///   preserve orientation. So it commits first. So does anything leaving a
///   rect-scoped family with a deformation on it — a homography is not reproducible
///   by a cubic mesh, nor a mesh by a homography, nor either by an affine.
/// - Nothing composed yet carries trivially, whichever way it is going: there is no
///   deformation to lose, so the switch is free and spends no undo step.
///
/// The last case is why this cannot be "carry when you can, commit otherwise": a
/// perspective quad nobody has dragged has to reach the warp family without an undo
/// step appearing for it, and `is_identity` is what says so.
pub fn switch(ui: TransformUi, to: Family, bands: Bands) -> Switch {
    if ui.family() == to {
        return Switch::Nothing;
    }
    let layer = ui.layer();
    if let TransformUi::Affine { rect, ts } = ui
        && ts.affine().matrix2.determinant() > 0.0
    {
        let affine = ts.affine();
        let inflated = crate::bounds::inflate(rect, bands.min_rect);
        let carried = match to {
            Family::Free => unreachable!("the same family returned above"),
            Family::Perspective => {
                let mut p = PerspectiveUi::begin(layer, inflated);
                p.corners = p.corners.map(|c| affine.transform_point2(c));
                TransformUi::Perspective(p)
            }
            Family::Warp => {
                let mut w = WarpUi::begin(layer, inflated);
                for pt in &mut w.points {
                    *pt = affine.transform_point2(*pt);
                }
                TransformUi::Warp(w)
            }
        };
        if carried.map().usable() {
            return Switch::Carried(carried);
        }
    }
    if ui.is_identity() {
        return Switch::Fresh(mount(layer, to, ui.rect(), bands));
    }
    Switch::Commit {
        map: ui.map(),
        then: mount(layer, to, ui.image_rect(), bands),
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
    /// the quad's inside. `grab` and `band` are canvas-px radii ([`Bands`]).
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
    /// pointer exactly while the shape stays convex.
    pub fn corner_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        shaped(start, current, delta, eps, |next| {
            next.corners[i] = start.corners[i] + delta;
        })
    }

    /// An edge drag: both of its corners follow together.
    pub fn edge_dragged(
        start: Self,
        current: Self,
        (a, b): (usize, usize),
        delta: Vec2,
        eps: f32,
    ) -> Self {
        shaped(start, current, delta, eps, |next| {
            next.corners[a] = start.corners[a] + delta;
            next.corners[b] = start.corners[b] + delta;
        })
    }

    /// An inside (or outside) drag: the whole quad translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        shaped(start, current, delta, eps, |next| {
            for c in &mut next.corners {
                *c += delta;
            }
        })
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

/// Where the warp surface was grabbed, and the solve that carries it (§16.9).
///
/// A named value rather than a triple: every caller outside a test wants two of the
/// three, and which two was decided by position.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SurfaceGrab {
    /// The grid fraction whose surface point is nearest the press.
    pub at: Vec2,
    /// Per-control-point influence there — how far the surface moves per unit move
    /// of each control point.
    pub basis: [f32; WARP_GRID * WARP_GRID],
    /// `Σ B²`, the divisor of the least-norm move. Floored, so it can be divided by.
    pub norm_sq: f32,
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
        // Against the points `begin` laid, which are the model's own — the same
        // comparison, without a second mesh built to make it.
        self.points == Self::begin(self.layer, self.rect).points
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

    /// Everything a surface drag needs, computed once at the press: the grid fraction
    /// whose surface point is nearest `p`, the basis there and its squared norm.
    /// Coarse scan plus local refinement; the surface is smooth and unfolded, so
    /// nearest-on-a-grid converges fast.
    ///
    /// **Not on the hover path** — this is 81 coarse probes plus six refinement passes
    /// of 25, and a resting pointer wants [`hint_at`] instead.
    pub fn grab(&self, p: Vec2) -> SurfaceGrab {
        let map = self.map();
        // The delta grid hoisted out of the search: every probe would otherwise
        // rebuild it. `WARP_GRID` is 4, so the mesh is always well-shaped.
        let surface = map.prepared().expect("a WARP_GRID mesh is well-shaped");
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
        let mut basis = [0.0f32; WARP_GRID * WARP_GRID];
        basis.copy_from_slice(&surface.basis(best.0));
        let norm_sq: f32 = basis.iter().map(|w| w * w).sum();
        SurfaceGrab {
            at: best.0,
            basis,
            norm_sq: norm_sq.max(1e-6),
        }
    }

    /// A control-point drag, recomputed from the drag's start.
    pub fn point_dragged(start: Self, current: Self, i: usize, delta: Vec2, eps: f32) -> Self {
        shaped(start, current, delta, eps, |next| {
            next.points[i] = start.points[i] + delta;
        })
    }

    /// A surface drag: the least-norm control move that carries the grabbed
    /// surface point exactly with the pointer — the hand holds the paint, not
    /// a handle (§16.9). `basis`/`norm_sq` come from [`grab`](Self::grab) at the
    /// press.
    pub fn surface_dragged(
        start: Self,
        current: Self,
        basis: &[f32; WARP_GRID * WARP_GRID],
        norm_sq: f32,
        delta: Vec2,
        eps: f32,
    ) -> Self {
        shaped(start, current, delta, eps, |next| {
            for (pt, w) in next.points.iter_mut().zip(basis) {
                *pt += delta * (*w / norm_sq);
            }
        })
    }

    /// An outside drag: the whole mesh translates.
    pub fn translated(start: Self, current: Self, delta: Vec2, eps: f32) -> Self {
        shaped(start, current, delta, eps, |next| {
            for pt in &mut next.points {
                *pt += delta;
            }
        })
    }
}

/// Even-odd point-in-polygon over an arbitrary boundary walk.
fn point_in_polygon(poly: &[Vec2], p: Vec2) -> bool {
    // An empty walk encloses nothing — and the wrap-around index below underflows on
    // one, which is a panic rather than an answer.
    let Some(mut j) = poly.len().checked_sub(1) else {
        return false;
    };
    let mut inside = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// How many points the affine's ellipse is sampled at. Enough that the polygon reads
/// as a curve at any zoom the widget is usable at.
const ELLIPSE_STEPS: usize = 96;

/// How many parts the perspective grid divides the source rect into: its thirds, so
/// two interior lines per axis.
const GRID_DIVISIONS: usize = 3;

/// How finely one warp curve is sampled. Enough that the cubic reads as a curve at
/// any deformation, and this is `WARP_GRID * 2 * (MESH_SAMPLES + 1)` evaluations on
/// every frame of a drag, so not much more.
const MESH_SAMPLES: usize = 24;

/// The widget's own boundary, canvas px — one polyline per run, **closed by repeating
/// its first point**, so a frontend strokes every run the same way and no run carries
/// a flag saying which.
///
/// Here rather than in a frontend because it was in both, and the two had drifted:
/// the native app drew the warp mesh as straight lines through the control points and
/// the perspective grid at a different line count from the web's. Which is not a
/// style difference — §16.9 makes the drawing a claim about the paint, and a claim
/// derived twice is two claims.
pub fn outline(ui: &TransformUi) -> Vec<Vec<Vec2>> {
    match ui {
        TransformUi::Affine { ts, .. } => {
            // Sampled rather than fitted with arcs: under a shear it is an ellipse at
            // an angle, which no axis-aligned arc primitive can state.
            let mut ring: Vec<Vec2> = (0..ELLIPSE_STEPS)
                .map(|i| {
                    let t = i as f32 / ELLIPSE_STEPS as f32 * std::f32::consts::TAU;
                    ts.center + ts.linear * (ts.radius * Vec2::new(t.cos(), t.sin()))
                })
                .collect();
            ring.push(ring[0]);
            vec![ring]
        }
        TransformUi::Perspective(p) => {
            let c = p.corners;
            vec![vec![c[0], c[1], c[3], c[2], c[0]]]
        }
        // The mesh has no boundary of its own: every curve `grid` draws is draggable
        // paint, and a heavier border would say one of them is not.
        TransformUi::Warp(_) => Vec::new(),
    }
}

/// The lines drawn *through* the widget, canvas px — what says what the map does
/// between the handles.
///
/// **Sampled from the very maps the paint resamples through** (§16.8, §16.9), which
/// is the whole point of them: a straight mesh grid says "untouched", and every bend
/// is a bend the paint has taken. A perspective's lines need two points each and no
/// sampling at all — a line stays a line under a homography, so the run between the
/// images of its ends *is* the image of the line.
///
/// Empty when the map cannot be built: a concave quad has no homography to draw, and
/// the corners already show that.
pub fn grid(ui: &TransformUi) -> Vec<Vec<Vec2>> {
    match ui {
        // The affine's whole shape is its rim; there is nothing between handles it
        // does not already say.
        TransformUi::Affine { .. } => Vec::new(),
        TransformUi::Perspective(p) => {
            let Some(h) = p.map().forward() else {
                return Vec::new();
            };
            let (lo, hi) = p.rect;
            let mut runs = Vec::with_capacity(2 * (GRID_DIVISIONS - 1));
            for i in 1..GRID_DIVISIONS {
                let t = i as f32 / GRID_DIVISIONS as f32;
                let x = lo.x + (hi.x - lo.x) * t;
                let y = lo.y + (hi.y - lo.y) * t;
                runs.push(vec![
                    h.apply(Vec2::new(x, lo.y)),
                    h.apply(Vec2::new(x, hi.y)),
                ]);
                runs.push(vec![
                    h.apply(Vec2::new(lo.x, y)),
                    h.apply(Vec2::new(hi.x, y)),
                ]);
            }
            runs
        }
        TransformUi::Warp(w) => {
            let map = w.map();
            // Prepared once for the whole overlay rather than once per sample.
            let Some(surface) = map.prepared() else {
                return Vec::new();
            };
            let mut runs = Vec::with_capacity(2 * WARP_GRID);
            for k in 0..WARP_GRID {
                let t = k as f32 / (WARP_GRID - 1) as f32;
                runs.push(
                    (0..=MESH_SAMPLES)
                        .map(|s| surface.eval(Vec2::new(s as f32 / MESH_SAMPLES as f32, t)))
                        .collect(),
                );
                runs.push(
                    (0..=MESH_SAMPLES)
                        .map(|s| surface.eval(Vec2::new(t, s as f32 / MESH_SAMPLES as f32)))
                        .collect(),
                );
            }
            runs
        }
    }
}

/// Where the widget's handles sit, canvas px — the same points [`Grab::take`]
/// classifies against, so a mark can never be drawn where a press would miss it.
///
/// The affine's is its centre: the ellipse is grabbed anywhere along its rim, so the
/// only thing worth marking is what a translate aims at.
pub fn handles(ui: &TransformUi) -> Vec<Vec2> {
    match ui {
        TransformUi::Affine { ts, .. } => vec![ts.center],
        TransformUi::Perspective(p) => p.corners.to_vec(),
        TransformUi::Warp(w) => w.points.to_vec(),
    }
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    fn state() -> TransformState {
        TransformState::begin(
            LayerId::ROOT,
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
        let s = state();
        let ts = TransformState::translated(s, s, Vec2::ZERO, Vec2::new(37.5, -12.0), 0.5);
        assert_eq!(ts.linear, Mat2::IDENTITY);
        let a = ts.affine();
        assert_eq!(a.matrix2, Mat2::IDENTITY);
        assert_eq!(a.translation, Vec2::new(37.5, -12.0));
    }

    #[test]
    fn a_sub_epsilon_jiggle_changes_nothing() {
        let ts = state();
        let from = Vec2::new(100.0, 0.0);
        let to = from + Vec2::splat(0.1);
        assert!(TransformState::turned_scaled(ts, ts, from, to, 0.5) == ts);
        assert!(TransformState::stretched(ts, ts, from, to, 0.5) == ts);
        assert!(TransformState::translated(ts, ts, from, to, 0.5) == ts);
    }

    #[test]
    fn rim_drag_carries_the_grab_point_to_the_pointer() {
        // Grab east, drag to twice-north: a quarter turn plus a 2× scale.
        let ts = state();
        let from = ts.center + Vec2::new(100.0, 0.0);
        let to = ts.center + Vec2::new(0.0, 200.0);
        let turned = TransformState::turned_scaled(ts, ts, from, to, 0.5);
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
        let stretched = TransformState::stretched(ts, ts, from, to, 0.5);
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

    /// The reference circle **circumscribes** the hull — the hypotenuse of its
    /// half-extents, not the inscribed ellipse's radius — so the widget encloses
    /// every corner of what it is holding. A circle rather than the hull's own
    /// aspect because the shape carries meaning: a circle says "no distortion yet"
    /// (§16.6), and an ellipse-shaped reference would say "distorted" before the hand
    /// had done anything.
    #[test]
    fn the_reference_circle_circumscribes_the_hull() {
        let r = state().radius;
        assert!((r - 100.0f32.hypot(50.0)).abs() < 1e-3, "got {r}");
    }

    /// A hull given back to front is the same widget as the one given the right way
    /// round: the radius is a length, and `bounds::inflate` answers the same way.
    #[test]
    fn an_inverted_hull_mounts_the_same_circle() {
        let hull = (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0));
        let flipped = (hull.1, hull.0);
        assert_eq!(
            TransformState::begin(LayerId::ROOT, hull, 10.0).radius,
            TransformState::begin(LayerId::ROOT, flipped, 10.0).radius
        );
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
        let wide = TransformState::stretched(
            ts,
            ts,
            c + Vec2::new(r, 0.0),
            c + Vec2::new(2.0 * r, 0.0),
            0.5,
        );
        assert_eq!(
            wide.region(c + Vec2::new(2.0 * r, 0.0), 4.0),
            TransformRegion::Rim
        );
        assert_eq!(
            wide.region(c + Vec2::new(r, 0.0), 4.0),
            TransformRegion::Inside
        );
    }

    /// A NaN pointer coordinate is not a gesture. `NaN < eps` is false, so before
    /// [`shaped`] named it the snap waved one through into `linear`, `is_identity`
    /// read false, and "Done" committed a map the engine refuses — nothing happening,
    /// with nothing said about why.
    #[test]
    fn a_pointer_that_is_not_a_number_moves_nothing() {
        let ts = state();
        let nan = Vec2::new(f32::NAN, 0.0);
        for got in [
            TransformState::translated(ts, ts, Vec2::ZERO, nan, 0.5),
            TransformState::turned_scaled(ts, ts, Vec2::new(100.0, 0.0), nan, 0.5),
            TransformState::stretched(ts, ts, Vec2::new(300.0, 0.0), nan, 0.5),
        ] {
            assert_eq!(got, ts);
            assert!(got.usable());
        }
    }

    /// A run of pull-ins cannot collapse the paint onto a line. The radial floor
    /// bounds one *gesture* to a 10× shrink of the determinant, but every press
    /// starts from what the last one left, so ten of them would take it under
    /// `f32::EPSILON` — and the engine would then refuse a commit the widget had
    /// shown no sign of trouble with.
    #[test]
    fn a_chain_of_pull_ins_holds_at_the_last_usable_shape() {
        let mut ts = state();
        for _ in 0..10 {
            let from = ts.center + Vec2::new(300.0, 0.0);
            // Maximal: past the pinned axis, so the floor is what decides.
            let to = from - Vec2::new(600.0, 0.0);
            ts = TransformState::stretched(ts, ts, from, to, 0.5);
            assert!(ts.usable(), "det {}", ts.linear.determinant());
        }
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
        let p = PerspectiveUi::begin(LayerId::ROOT, rect());
        assert!(p.is_identity());
        assert!(p.map().usable());
        assert!(TransformUi::Perspective(p).is_identity());
    }

    #[test]
    fn a_dragged_corner_lands_under_the_pointer() {
        let p = PerspectiveUi::begin(LayerId::ROOT, rect());
        let delta = Vec2::new(-30.0, 22.0);
        let dragged = PerspectiveUi::corner_dragged(p, p, 3, delta, 0.5);
        assert_eq!(dragged.corners[3], p.corners[3] + delta);
        assert!(!dragged.is_identity());
        assert!(dragged.map().usable());
    }

    #[test]
    fn a_corner_pulled_across_the_quad_holds_at_the_last_valid_shape() {
        let p = PerspectiveUi::begin(LayerId::ROOT, rect());
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
        let p = PerspectiveUi::begin(LayerId::ROOT, rect());
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
        let w = WarpUi::begin(LayerId::ROOT, rect());
        assert!(w.is_identity());
        assert!(w.map().usable());
    }

    #[test]
    fn a_dragged_control_point_lands_under_the_pointer() {
        let w = WarpUi::begin(LayerId::ROOT, rect());
        let delta = Vec2::new(14.0, -9.0);
        let dragged = WarpUi::point_dragged(w, w, 5, delta, 0.5);
        assert_eq!(dragged.points[5], w.points[5] + delta);
        assert!(dragged.map().usable());
        assert!(!dragged.is_identity());
    }

    #[test]
    fn a_surface_drag_carries_the_grabbed_paint_exactly() {
        let w = WarpUi::begin(LayerId::ROOT, rect());
        let grab_at = Vec2::new(20.0, -10.0);
        let g = w.grab(grab_at);
        let before = w.map().prepared().expect("well-shaped").eval(g.at);
        assert!(before.distance(grab_at) < 1.0, "grab missed: {before:?}");
        let delta = Vec2::new(18.0, 12.0);
        let dragged = WarpUi::surface_dragged(w, w, &g.basis, g.norm_sq, delta, 0.5);
        let after = dragged.map().prepared().expect("well-shaped").eval(g.at);
        assert!(
            after.distance(before + delta) < 0.1,
            "the paint under the finger moved {:?}, the finger moved {delta:?}",
            after - before
        );
    }

    #[test]
    fn a_folding_drag_holds_at_the_last_valid_shape() {
        let w = WarpUi::begin(LayerId::ROOT, rect());
        // Drag an interior point far past its neighbour: the mesh would fold.
        let held = WarpUi::point_dragged(w, w, 5, Vec2::new(250.0, 0.0), 0.5);
        assert_eq!(held, w, "a fold must hold, not tear");
    }

    #[test]
    fn mesh_regions_classify_points_surface_and_outside() {
        let w = WarpUi::begin(LayerId::ROOT, rect());
        assert_eq!(w.region(w.points[0], 8.0), MeshRegion::Point(0));
        assert_eq!(w.region(Vec2::new(15.0, 5.0), 8.0), MeshRegion::Inside);
        assert_eq!(w.region(Vec2::new(500.0, 0.0), 8.0), MeshRegion::Outside);
    }

    #[test]
    fn sub_epsilon_jiggles_snap_back_to_the_start() {
        let p = PerspectiveUi::begin(LayerId::ROOT, rect());
        assert_eq!(
            PerspectiveUi::corner_dragged(p, p, 2, Vec2::splat(0.1), 0.5),
            p
        );
        let w = WarpUi::begin(LayerId::ROOT, rect());
        assert_eq!(WarpUi::point_dragged(w, w, 5, Vec2::splat(0.1), 0.5), w);
        assert_eq!(WarpUi::translated(w, w, Vec2::splat(0.1), 0.5), w);
    }

    /// An empty boundary walk encloses nothing. Unreachable through [`WarpUi`], whose
    /// mesh always has one — but this is a free function over a slice, and the
    /// wrap-around index underflows rather than answering.
    #[test]
    fn an_empty_polygon_encloses_nothing() {
        assert!(!point_in_polygon(&[], Vec2::ZERO));
    }
}

#[cfg(test)]
mod drag_tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0))
    }

    fn bands() -> Bands {
        Bands::at(1.0)
    }

    /// Where the map this gesture stands for carries `p`.
    fn carried(ui: &TransformUi, p: Vec2) -> Vec2 {
        match ui.map() {
            TransformMap::Affine(a) => a.transform_point2(p),
            TransformMap::Perspective(m) => m.forward().expect("a usable quad").apply(p),
            TransformMap::Warp(m) => {
                // The mesh's own surface, at the grid fraction `p` sits at in the
                // source rect — the mesh is the map, so this is what "carried" means.
                let (lo, hi) = (m.min, m.max);
                let t = (p - lo) / (hi - lo);
                let prepared = m.prepared().expect("a usable mesh");
                prepared.eval(t)
            }
        }
    }

    /// **The path both frontends actually use, end to end.** Every other test here
    /// calls a shaping function directly, so swapping the rim's gesture for the
    /// outside's passed the whole suite: take a grab where the widget offers each of
    /// its regions, follow it to a target, and the grabbed point has to arrive there.
    #[test]
    fn a_grab_carries_the_grabbed_point_to_the_pointer() {
        let ui = mount(LayerId::ROOT, Family::Free, rect(), bands());
        let TransformUi::Affine { ts, .. } = ui else {
            unreachable!()
        };
        let r = ts.radius;
        // One press per region of the affine widget, each with somewhere to be
        // dragged to that the family can express.
        let table = [
            (
                "inside",
                ts.center + Vec2::new(0.2 * r, 0.0),
                Vec2::new(40.0, -25.0),
            ),
            ("rim", ts.center + Vec2::new(r, 0.0), Vec2::new(-30.0, 60.0)),
            (
                "outside",
                ts.center + Vec2::new(1.8 * r, 0.0),
                Vec2::new(55.0, 35.0),
            ),
        ];
        for (what, from, delta) in table {
            let mut grab = Grab::take(ui, from, bands());
            let next = grab.follow(from + delta, bands());
            let landed = carried(&next, from);
            assert!(
                landed.distance(from + delta) < 0.05,
                "{what}: the grab landed at {landed:?}, the pointer at {:?}",
                from + delta
            );
        }
    }

    /// The same for the two rect-scoped families, over every region each offers.
    #[test]
    fn every_region_of_every_family_follows_the_pointer() {
        for family in [Family::Perspective, Family::Warp] {
            let ui = mount(LayerId::ROOT, family, rect(), bands());
            // A handle, the surface between handles, and outside the shape.
            let handle = handles(&ui)[0];
            let table = [
                ("handle", handle, Vec2::new(-18.0, -14.0)),
                ("surface", Vec2::new(10.0, 6.0), Vec2::new(-20.0, 12.0)),
            ];
            for (what, from, delta) in table {
                let mut grab = Grab::take(ui, from, bands());
                let next = grab.follow(from + delta, bands());
                let landed = carried(&next, from);
                assert!(
                    landed.distance(from + delta) < 0.2,
                    "{family:?}/{what}: landed {landed:?}, pointer {:?}",
                    from + delta
                );
            }
        }
    }

    /// A translate arm moves everything by exactly the pointer's travel — the whole
    /// shape, not the part under the hand.
    #[test]
    fn the_translate_arms_move_everything_by_the_delta() {
        let delta = Vec2::new(33.0, -21.0);
        for family in [Family::Free, Family::Perspective, Family::Warp] {
            let ui = mount(LayerId::ROOT, family, rect(), bands());
            // Well outside the widget for the two rect-scoped families (which
            // translate from outside) and inside it for the affine.
            let from = match family {
                Family::Free => Vec2::ZERO,
                _ => Vec2::new(900.0, 900.0),
            };
            let mut grab = Grab::take(ui, from, bands());
            let next = grab.follow(from + delta, bands());
            for (before, after) in handles(&ui).into_iter().zip(handles(&next)) {
                assert!(
                    (after - before - delta).length() < 1e-3,
                    "{family:?}: {before:?} moved to {after:?}, not by {delta:?}"
                );
            }
        }
    }

    /// A press outside the mesh carries no basis to read. The two regions that never
    /// use one used to be handed a fabricated `[0.0; 16]`, which is a value that can
    /// only be wrong if anything ever looked at it.
    #[test]
    fn only_a_surface_press_solves_a_basis() {
        let ui = mount(LayerId::ROOT, Family::Warp, rect(), bands());
        let TransformUi::Warp(w) = ui else {
            unreachable!()
        };
        let cases = [
            (w.points[0], MeshGrab::Point(0)),
            (Vec2::new(900.0, 900.0), MeshGrab::Translate),
        ];
        for (at, want) in cases {
            let Grab::Mesh { region, .. } = Grab::take(ui, at, bands()) else {
                unreachable!()
            };
            assert_eq!(region, want);
        }
        let Grab::Mesh { region, .. } = Grab::take(ui, Vec2::new(10.0, 6.0), bands()) else {
            unreachable!()
        };
        assert!(matches!(region, MeshGrab::Surface { .. }));
    }

    /// A hover asks what a press would do without taking one — which for the warp is
    /// the difference between three region tests and 231 surface evaluations. Three
    /// points each: on a handle, on the surface between handles, and well outside.
    #[test]
    fn the_hover_hint_names_what_a_press_would_do() {
        // The affine's outside is its *shaping* gesture; the two rect-scoped
        // families translate from outside, which is why the third column differs.
        let table = [
            (Family::Free, Hint::Move, Hint::Move, Hint::Shape),
            (Family::Perspective, Hint::Hold, Hint::Move, Hint::Move),
            (Family::Warp, Hint::Hold, Hint::Hold, Hint::Move),
        ];
        for (family, on_handle, between, outside) in table {
            let ui = mount(LayerId::ROOT, family, rect(), bands());
            let cases = [
                ("handle", handles(&ui)[0], on_handle),
                ("between", Vec2::new(10.0, 6.0), between),
                ("outside", Vec2::new(900.0, 900.0), outside),
            ];
            for (what, at, want) in cases {
                assert_eq!(hint_at(&ui, at, bands()), want, "{family:?} {what}");
            }
        }
    }

    /// The clamp is the grab's own now: a fold reached mid-drag stays where the last
    /// valid shape was, without the frontend handing the state back in.
    #[test]
    fn a_grab_holds_its_own_last_valid_shape() {
        let ui = mount(LayerId::ROOT, Family::Warp, rect(), bands());
        let TransformUi::Warp(w) = ui else {
            unreachable!()
        };
        let from = w.points[5];
        let mut grab = Grab::take(ui, from, bands());
        // Far enough to bend, not far enough to fold.
        let bent = grab.follow(from + Vec2::new(20.0, 0.0), bands());
        assert!(bent.map().usable());
        // Now past the fold: the answer is the shape before it, not the one it began
        // at and not the folded one.
        let held = grab.follow(from + Vec2::new(400.0, 0.0), bands());
        assert_eq!(held, bent);
    }
}

#[cfg(test)]
mod overlay_tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0))
    }

    /// **The rule the native app broke** (§16.9): the curves are sampled from the
    /// surface the paint resamples through, so a mesh bent between its control points
    /// draws bent. A polyline through the control points would draw this one straight.
    #[test]
    fn the_mesh_curves_come_off_the_surface_not_the_control_net() {
        let w = WarpUi::begin(LayerId::ROOT, rect());
        // Pull one interior point off the net. Its row's *midpoints* now bulge, even
        // where the control points on that row have not moved.
        let bent = WarpUi::point_dragged(w, w, 5, Vec2::new(0.0, 12.0), 0.5);
        let ui = TransformUi::Warp(bent);
        let runs = grid(&ui);
        assert_eq!(runs.len(), 2 * WARP_GRID, "one curve per row and column");
        let straight = runs.iter().all(|run| {
            let (a, b) = (run[0], run[run.len() - 1]);
            run.iter()
                .all(|p| (*p - a).perp_dot(b - a).abs() < 1e-2 * (b - a).length())
        });
        assert!(!straight, "a bent mesh drew as an untouched one");
    }

    /// Every run is closed by repeating its first point, so a frontend strokes them
    /// all the same way and no run needs a flag saying which.
    #[test]
    fn boundary_runs_close_by_repeating_their_first_point() {
        for family in [Family::Free, Family::Perspective] {
            let ui = mount(LayerId::ROOT, family, rect(), Bands::at(1.0));
            for run in outline(&ui) {
                assert!(run.len() > 2);
                assert_eq!(run[0], run[run.len() - 1], "{family:?} left its run open");
            }
        }
    }

    /// The perspective grid is straight two-point runs, because a line stays a line
    /// under a homography — the run between the images of its ends *is* the image of
    /// the line, and sampling it would be sampling a straight edge.
    #[test]
    fn the_perspective_grid_is_the_images_of_the_rects_thirds() {
        let mut p = PerspectiveUi::begin(LayerId::ROOT, rect());
        p.corners[1] += Vec2::new(-40.0, 25.0);
        let runs = grid(&TransformUi::Perspective(p));
        assert_eq!(runs.len(), 2 * (GRID_DIVISIONS - 1));
        assert!(runs.iter().all(|r| r.len() == 2));
    }

    /// Handles are drawn where a press is classified, which is the only thing
    /// stopping a mark from sitting where a grab would miss it.
    #[test]
    fn every_handle_is_where_a_press_takes_hold_of_it() {
        let bands = Bands::at(1.0);
        for family in [Family::Perspective, Family::Warp] {
            let ui = mount(LayerId::ROOT, family, rect(), bands);
            for h in handles(&ui) {
                assert_eq!(hint_at(&ui, h, bands), Hint::Hold, "{family:?} at {h:?}");
                // And the press agrees with the cursor. The hover path and the press
                // path classify separately now, so this is the seam between them.
                let took_it = match Grab::take(ui, h, bands) {
                    Grab::Quad { region, .. } => matches!(region, QuadRegion::Corner(_)),
                    Grab::Mesh { region, .. } => matches!(region, MeshGrab::Point(_)),
                    Grab::Affine { .. } => unreachable!("no affine in the table"),
                };
                assert!(took_it, "{family:?} drew a handle a press misses at {h:?}");
            }
        }
    }

    /// A quad with no homography draws its corners and nothing between them — a
    /// concave one has no map to sample, and the corners already say so.
    #[test]
    fn an_unusable_quad_draws_no_grid() {
        let mut p = PerspectiveUi::begin(LayerId::ROOT, rect());
        p.corners[0] += Vec2::new(500.0, 300.0);
        let ui = TransformUi::Perspective(p);
        assert!(!p.map().usable());
        assert!(grid(&ui).is_empty());
        assert_eq!(handles(&ui).len(), 4);
    }
}

#[cfg(test)]
mod switch_tests {
    use super::*;

    fn rect() -> (Vec2, Vec2) {
        (Vec2::new(-100.0, -50.0), Vec2::new(100.0, 50.0))
    }

    fn bands() -> Bands {
        Bands::at(1.0)
    }

    fn free() -> TransformUi {
        mount(LayerId::ROOT, Family::Free, rect(), bands())
    }

    /// Switching to the family already composing is not a re-mount: a bar that took
    /// the lit chip as an instruction would throw the gesture away on a stray press.
    #[test]
    fn the_lit_family_is_left_alone() {
        assert!(matches!(
            switch(free(), Family::Free, bands()),
            Switch::Nothing
        ));
    }

    /// An orientation-preserving affine *is* a parallelogram perspective, so it rides
    /// across and the artist keeps composing — no undo step, no reopen.
    #[test]
    fn an_ordinary_affine_carries_into_the_other_families() {
        let TransformUi::Affine { rect, ts } = free() else {
            unreachable!()
        };
        let turned = TransformState::turned_scaled(
            ts,
            ts,
            Vec2::new(100.0, 0.0),
            Vec2::new(0.0, 100.0),
            0.5,
        );
        let turned = TransformUi::Affine {
            rect,
            ts: TransformState::translated(turned, turned, Vec2::ZERO, Vec2::new(20.0, -5.0), 0.5),
        };
        for to in [Family::Perspective, Family::Warp] {
            let Switch::Carried(next) = switch(turned, to, bands()) else {
                panic!("an ordinary affine should carry into {to:?}");
            };
            assert_eq!(next.family(), to);
            // Carried *exactly*: the paint under the widget must not move on a
            // switch, or the family chips would be an edit.
            let before = turned.image_rect();
            let after = next.image_rect();
            for (a, b) in [(before.0, after.0), (before.1, after.1)] {
                assert!(
                    (a - b).length() < 0.01,
                    "the carry moved the paint: {a:?} vs {b:?}"
                );
            }
        }
    }

    /// A mirrored affine has no image in either — both of those maps preserve
    /// orientation — so it commits first, and the reopen is around where the paint
    /// now is rather than where it started.
    #[test]
    fn a_mirrored_affine_commits_before_it_leaves() {
        let TransformUi::Affine { rect, ts } = free() else {
            unreachable!()
        };
        let flipped = ts.flipped_h();
        let mirrored = TransformUi::Affine {
            rect,
            ts: TransformState::translated(
                flipped,
                flipped,
                Vec2::ZERO,
                Vec2::new(500.0, 0.0),
                0.5,
            ),
        };
        let Switch::Commit { map, then } = switch(mirrored, Family::Perspective, bands()) else {
            panic!("a mirrored affine cannot carry");
        };
        assert!(matches!(map, TransformMap::Affine(_)));
        assert_eq!(then.family(), Family::Perspective);
        // Reopened around the moved paint: the translation above put it 500 to the
        // right, so a quad still sitting over the origin would be the widget having
        // let go of the picture.
        assert!(then.rect().0.x > 300.0, "reopened at {:?}", then.rect());
    }

    /// Nothing composed yet costs nothing to switch, whichever way it goes — an undo
    /// step appearing for a family chip nobody dragged under would be a lie about
    /// what happened.
    #[test]
    fn an_untouched_gesture_switches_free() {
        let fresh = mount(LayerId::ROOT, Family::Perspective, rect(), bands());
        assert!(fresh.is_identity());
        let Switch::Fresh(next) = switch(fresh, Family::Warp, bands()) else {
            panic!("an identity should never spend an undo step");
        };
        assert_eq!(next.family(), Family::Warp);
        assert!(next.is_identity());
    }

    /// **The four transitions that rested on a fall-through.** A homography is not
    /// reproducible by a cubic mesh and a mesh is not reproducible by a homography,
    /// and neither is an affine, so every one of these commits — the honest undo step
    /// rather than a silent approximation of what "Done" would have produced.
    #[test]
    fn a_deformed_rect_family_commits_whichever_way_it_leaves() {
        let mut p = PerspectiveUi::begin(LayerId::ROOT, rect());
        p.corners[1] += Vec2::new(-40.0, 25.0);
        let perspective = TransformUi::Perspective(p);

        let w = WarpUi::begin(LayerId::ROOT, rect());
        let warp = TransformUi::Warp(WarpUi::point_dragged(w, w, 5, Vec2::new(0.0, 12.0), 0.5));

        for (ui, to) in [
            (perspective, Family::Warp),
            (perspective, Family::Free),
            (warp, Family::Perspective),
            (warp, Family::Free),
        ] {
            assert!(!ui.is_identity());
            let Switch::Commit { map, then } = switch(ui, to, bands()) else {
                panic!("{:?} → {to:?} must commit, not carry", ui.family());
            };
            assert!(map.usable());
            assert_eq!(then.family(), to);
            assert!(then.is_identity(), "the reopened gesture starts fresh");
        }
    }
}
