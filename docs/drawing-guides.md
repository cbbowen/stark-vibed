# Drawing guides

The perspective grid: one projective camera, three familiar cases — §20.
What a stroke aligns to is in §20.6 and §20.7.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 20 Drawing guides

Guides are scaffolding for the hand: geometry drawn over the canvas that a
stroke can be aligned to (§20.6) and that no pixel ever records. The first
guide is the classical perspective grid, and its design principle is the one
this chapter keeps returning to: **the guide is a camera, and everything the
artist sees is derived from it.**

## 20.1 One camera, three special cases

Art tools habitually ship "1-point", "2-point" and "3-point perspective" as
three modes with three data models — one draggable dot, two dots on a shared
horizon, three free dots — and each mode's invariants (horizontals stay level,
verticals stay parallel) are enforced by *code* in that mode. Stark refuses the
split. The guide's state is exactly a projective camera (`guides.rs`):

- a **center of view** `c` — the principal point, where the view axis meets the
  picture plane, in canvas px;
- a **focal length** `f` — the eye's distance from the picture plane, in canvas
  px;
- an **orientation** — one quaternion turning the world's orthogonal axis frame
  relative to the camera. A quaternion rather than Euler angles because the
  orientation is *composed*, not set: every canvas drag multiplies a small
  rotation onto it (§20.5), and there is no slider left that would want an
  angle read back out.

Everything else is projection. The vanishing point of world axis `a` (a unit
direction in camera space — x right, y down to match the canvas, z forward) is

```
V(a) = c + f · (a.x, a.y) / a.z
```

— a *projective* point, allowed to be at infinity when `a.z = 0`. The familiar
cases are then just counts of finite vanishing points: view straight down an
axis and two axes lie in the picture plane (1-point); level the view and only
the verticals do (2-point); tilt it and none do (3-point). There is no mode
switch anywhere in the code, and the panel's case chips are *derived* from the
count of finite VPs rather than stored — dragging yaw through zero is watching
2-point collapse into 1-point, which is the honest picture.

Because the state is a camera, the classical drawing-office constructions come
out as theorems instead of decorations, and the overlay shows them:

- the **center of view** itself, which for a 3-point triangle is provably the
  **orthocenter** of the three vanishing points (unit-tested);
- the **45° circle** about `c` with radius exactly `f` (`tan 45° = 1`): the
  cone at 45° off the view axis, the classical "keep the drawing inside this
  or it will look stretched" bound. It is the focal length made visible, and
  later the handle by which the lens is dragged;
- the three **station points** (§20.2).

Directions are treated unsigned throughout — an axis and its negation name the
same pencil of lines, the same vanishing point — so nothing downstream ever
branches on a sign. (The curvilinear lens is the one deliberate exception: it
sees the two signs at two places, which is §20.8's whole subject.)

## 20.2 Vanishing lines and station points

Each *pair* of axes spans a plane, and the plane's images carry two more
constructions.

**Vanishing line.** All planes parallel to span(`a_i`, `a_j`) share one image
line — for the ground pair, the horizon. With `m = a_i × a_j` (unit, because
the axes are orthonormal), the line is the trace of the parallel plane through
the eye:

```
m.x·x + m.y·y + (f·m.z − m·c) = 0
```

normalized so evaluating it is signed distance in canvas px. It passes through
`V(a_i)` and `V(a_j)` — including correctly through the one at infinity, which
is why it is computed from the plane normal rather than by joining two points
that may not exist. When `m.xy ≈ 0` the plane faces the camera square-on and
its line (and station point) is at infinity: drawn as nothing, not as a special
case.

**Station point.** Rotate the eye about a pair's vanishing line until it lands
in the picture plane: that is the classical station point, the viewer's
position folded into the drawing so distances can be measured on it. The eye
sits at height `f` over `c`; if the line lies at distance `a` from `c`, the
rotation preserves the eye's distance `√(a² + f²)` to the line and lands on the
ray from the foot of `c`'s perpendicular through `c`. In exact 2-point the view
axis lies *in* the ground plane (`a = 0`) and either side is the same rotation;
the canvas-down side is the drawing-board convention. The payoff, and the test:
a station point still sees its two vanishing points at the right angle they
subtend in the world, so it lies on the **Thales circle** over them — which is
where a future measuring tool will anchor.

## 20.3 The fans: equal steps of visual angle

What should "grid lines toward a vanishing point" mean? Any fan through the VP
is projectively *a* pencil; the question is how to space its members. Stark
parametrizes the pencil at its source: the guide lines of axis `a` are the
planes through the eye containing `a`, stepped by equal **dihedral angle** —
equal turns of the eye, `π / density` apart. That choice is uniform across all
three cases (nothing about it mentions whether the VP is finite), degrades
gracefully to evenly-stepped parallel lines as the VP goes to infinity, and
bunches near a pair's vanishing line exactly the way receding structure
foreshortens — the fan carries its own horizon-crowding.

A texel's pencil coordinate is computed from the eye's ray through it,
`r = ((p − c)/f, 1)`: the plane through the eye containing both `a` and `r`
has normal `n = a × r`, and resolving `n` against the *other two axes* gives
`θ = atan2(n·a_{i+1}, n·a_{i+2})`. Measuring against the other axes anchors the
fan's phase structurally: the plane spanned with either partner axis — whose
trace is a vanishing line — lands on a quarter turn of the pencil, so with an
even density **every vanishing line is a fan line** and the ground grid from
one VP passes exactly through the other. Scale never matters (`r` is used
unnormalized; every expression is homogeneous), and no VP is ever computed on
this path — the fans work entirely in direction space, which is what makes the
1/2/3-point uniformity real rather than asserted.

## 20.4 Pass D: the guide overlay

Rendering is one fullscreen triangle drawn after the selection outline —
pass D (`guides.wesl`, wired in `composite.rs`), over the lit image, because
the grid is chrome the whole canvas is read *through*. It is gated exactly as
pass C is: screen renders carry the derived `GuideScene`s in their
`CompositeScene`, exports and the navigator's miniature never do (§15.6), and
`render_to_image` sees them only if a test adds a guide — the default-empty
list leaves every golden untouched.

Every element is an analytic distance field, evaluated at the fragment's
*canvas* position (the uniform carries the screen→canvas map) and converted to
*screen* px for anti-aliasing — crisp at any zoom, rotation or mirror, no
geometry, no textures. Two details are load-bearing:

- **The fan's gradient is taken analytically.** θ has a branch cut, so
  `fwidth(θ)` lies along it; the true gradient of the `atan2` quotient,
  `|u∇v − v∇u| / (u² + v²)`, is continuous everywhere and gives both the AA
  width and the line spacing in screen px.
- **A moiré fade, not a clip.** Where fan lines pack tighter than a few px —
  approaching vanishing lines and VPs — coverage fades smoothly to nothing
  instead of shimmering. The fade is driven by the same measured spacing, so
  it tracks zoom for free.

The markers (CoV crosshair, VP discs in their axis's hue, station-point rings,
the dashed 45° circle) ride in the same pass as more distance fields, packed
as `position + valid` uniform slots so the shader branches on data, never on
pipeline variants. Axis hues follow the X/Y/Z semantics every 3D tool taught.

## 20.5 The list, the edit mode, and the drag

Guides are a **list** (`Session::guides`), and the whole list is view state:
per-client, unlogged, unsent — an aid for the hand holding the pen, like the
pan and the zoom. Every mutation — a slider, a drag sample, a row toggle —
travels as one `ViewCommand::SetGuides` carrying the whole list, the same
read-modify-commit shape `SetMediaParams` uses; the engine never needs one
command per control. If guides later become part of what a document carries (a
shared scaffold peers should see), that is a new `DocCommand` and an action;
`SetGuides` would remain as the in-flight preview half, the same bargain the
matte-rect drag strikes (§4). Rendering-side, each visible guide gets a
dynamic-offset uniform slot and its own fullscreen draw in pass D — slots
rather than one rewritten buffer for the reason `BLEND_SLOT` records.

The **Drawing Guides panel** is the roster, shaped like the Layers panel
because it answers the same question about a different stack: "Add
Perspective" (placed at the view center, so the grid lands where you look),
one row per guide, carrying what a layer row carries in the same order and
glyphs: the name, then duplicate, then remove, then the eye. Duplicating drops
the copy in the row below and picks it up, because duplicating a guide is asking
to shape a variant of it — the same argument that has "Add Perspective" open the
mode on what it made. The copy keeps the source's name for the reason the layer
duplicate does (§14.8): a name is the author's own word. The name renames on a
double-click, exactly as a layer's does — one draft per row, committed on Enter
or blur, abandoned on Escape, and a field left empty is *no name* rather than a
blank one, so the row goes back to describing its position ("Perspective 2").
That last rule is `normalize_name` in the engine, shared with layers and
applied to every guide the `SetGuides` arm accepts: the list arrives whole, so
there is no path to a guide's name that avoids it. Selecting a row — or
adding — enters the **edit mode**: a full-viewport catcher owns the pointer for the
mode's duration (the transform-mode bargain, §16.6; navigation still works),
and the **Perspective Guide bar** stands at the bottom with the per-axis
locks, per-axis visibility, the Fisheye lens toggle (§20.8), density, opacity,
and "Done". Which of 1/2/3-point you are in is never stored or displayed as a
mode: the canvas shows it, as the count of finite vanishing points.

The drag *is* the manipulation, classified by what the press lands on:

- **Anywhere: grab the world.** The world direction under the pointer follows
  it — `PerspectiveGuide::dragged` rotates the frame by the arc from the
  press's eye-ray to the current one, always recomputed from the drag's start
  so nothing drifts and nothing flickers. When the implied rotation axis lies
  within ~15° of a world axis, the drag **snaps** to a pure turn about it: a
  roughly-horizontal drag orbits the vertical axis and a 2-point setup stays
  exactly 2-point, without a mode.
- **Locks** are the same constraint made deliberate: rotations fixing an axis
  are exactly the turns about it, so one locked axis confines the drag to its
  orbit, and two pin the frame (the identity is the only rotation fixing two
  axes). Locks are gesture state, held by the mode and released with it.
- **The 45° circle: drag the lens.** The circle's radius *is* the focal
  length, so `f` becomes the distance from the center to the pointer and the
  ring follows the hand exactly — the §20.1 identity made into a handle.
- **The crosshair: move the construction.** The center of view follows the
  drag, grab-offset preserved.

Deferred, deliberately (§18 discipline — nothing inert ships):

- **Dragging vanishing points themselves** — a real design question (what does
  it hold fixed: the focal length? the other VPs?) that the orbit-with-locks
  vocabulary may make unnecessary.
- **Other guide kinds** (isometric, ellipse/vanishing-scale, symmetry): each a
  new derivation over the same pass-D machinery and the same list.

## 20.6 Strokes on the grid

The grid earns its keep when the hand can put a line on it. Drag out a rough
line and hold (§6.9): if it lands near an axis of a guide that is on the
screen, the stroke that snaps is aimed exactly along that axis, and the rest of
the drag runs the end out **along** it.

**What a stroke aligns to is a pencil, not a direction.** `AxisPencil` — one
axis of one guide — is the whole of a perspective guide that `assist.rs` sees,
and the one thing it can answer is a direction *at a point*:

```
through(p) = f·(a.x, a.y) + a.z·(c − p)
```

which is `V(a) − p` cleared of the denominator the vanishing point divides by.
It stays finite as `a.z → 0`, where it becomes the parallel direction of an
axis lying in the picture plane, so no vanishing point is computed on this path
and the 1/2/3-point uniformity is again structural rather than asserted —
exactly as for the fans (§20.3).

**The line is turned, not moved.** The pencil is evaluated at the stroke's
*start*, which §6.9 already treats as the deliberate end of a drag, and the
line keeps it. So a snapped stroke stays where the hand put it and only its
angle comes from the grid. That is also why it snaps to the axis rather than to
the nearest **fan line**, which is what §20.5 originally deferred: the fans are
a sampling of the pencil at whatever the density slider says, and there is no
reason a stroke's position should quantize to a display setting. Every line of
the pencil is equally a line of the grid.

**How near is near enough is an angle, spelled as a residual.** The trace is
re-scored against the pencil's line by exactly the measure that just accepted
it as a line — worst perpendicular distance, as a fraction of its own length —
against a bar (`GUIDE_RESIDUAL`, 0.1 ≈ 5.7°) wider than the recognizer's own.
The two bars are asking different questions and are deliberately not
interchangeable: `LINE_RESIDUAL` asks *whether the hand drew a line*, where a
false positive replaces a considered curve; this one asks *which line it
meant*, after the artist has dwelt to ask for an ideal one with a grid up to
answer with. So the guide question is put strictly **after** recognition has
accepted the stroke, never instead of it — a curve that happens to bow along a
fan line is still a curve — and among the axes that pass, the closest wins.

**Only what is shown may bend a stroke.** `PerspectiveGuide::pencils` is gated
on the guide's eye, the per-axis fans, and the overlay opacity, in one place:
a snap the artist cannot see coming reads as the tool bending a considered
line, so anything invisible offers nothing. A document with no guides up
gathers an empty list and the assist behaves precisely as it did before.

**The axis is held for the rest of the drag.** Steering resolves the pointer's
travel onto the line and drops the rest, so the far end runs out and back along
the grid line while the hand wanders. Adjustment preserving what recognition
established is the same bargain that keeps a drawn loop's eccentricity (§6.9),
and it is what makes the feature usable at all: an alignment a single sideways
nudge could break would not be one. To escape, lift and draw again.

The whole coupling is `AssistShape::Line`'s one extra bool. A guided line is
still a segment — the pencil's line through a point *is* a straight canvas line
— so nothing about realization, the path, the wire format or the goldens
changes, and the guides are read but never touched.

## 20.7 Circles on a plane

The other half, and the harder one: draw a rough loop where a circle on the
grid would be and hold, and what snaps is **the circle on that plane**, seen
from here. Not the ellipse the hand managed — the circle it was trying to draw.

This is the construction art teaching spends the most time on and gets the
least far with, because the two things that make a perspective circle right are
the two an eye cannot judge: how **open** it is and which way it **leans**, both
fixed entirely by where on the plane it sits. Its size and its position, which a
hand does get about right, are the free parameters. So the snap corrects exactly
what is hard and keeps exactly what was meant.

**A plane is a chart.** `AxisPlane` — one *pair* of axes — is the map between
the canvas and the plane's own flat, metric coordinates, both ways. A pair plane
needs no depth chosen for it, because scaling the depth scales every circle on
it by the same factor and leaves the images alone; taken at unit distance along
the plane's normal it is one 3×3:

```
canvas_from_plane = K · [ a_i | a_j | a_i × a_j ]
```

with `K` the lens. The three planes of a guide are that product with the axis
frame's columns *cyclically shifted* — one matrix read three ways — and it is
invertible for every pose, since a lens and a rotation both are. **There is no
degenerate plane to guard against**, which is the whole reason the chart is the
representation and not, say, a basis and a fallback.

**The question is asked in the plane, and answered on the canvas.** A trace is
pulled back through the chart, where "which circle is this" is a question with a
closed-form answer — and it is answered by the *same* `fit_ellipse` §6.9 already
has, because a circle is an ellipse whose radii agree, and the measure
corrections that fit earned (speed, overshoot, undershoot) are as necessary on a
pulled-back trace as on a drawn one. Its two radii are then collapsed to the one
of equal area.

But the **residual is measured on the canvas**, never in the plane. A plane's
own metric is stretched by the perspective, unboundedly toward its vanishing
line, so a residual measured there would mean something different at every depth
and the far half of a loop would count for orders of magnitude more than the
near half. What decides is the same worst-sample distance the free ellipse was
judged by, in the space the artist drew in.

**What comes back is an ellipse like any other.** The image of a conic under a
homography is a conic — `Hᵀ C H` on the matrix — so the circle is carried to the
canvas in closed form rather than fitted from sampled points, and the *same*
operation the other way recovers the circle from the ellipse. That exactness is
what lets the shape stay a plain `AssistShape::Ellipse`: realization, the path,
the wire format and the goldens are untouched, and the plane rides along as one
`Option` field that only the adjustment reads. It is `on_axis` again, with more
in it.

Two things fall out of the conic classification rather than being cases:

- a circle crossing its plane's vanishing line images to a **hyperbola**, which
  is not something a stroke can be — so `circle_seen` declines, and the
  positive-definiteness test *is* the check;
- a trace crossing that line has no preimage on one piece of the plane, so the
  chart declines it, because no circle in front of the eye is ever seen across
  the image of that plane's infinity.

**Steering sizes it, and only sizes it.** A circle has no orientation, so the
turn the free ellipse spends a degree of freedom on is not there to spend: the
drag's travel resolves onto the radius, in the plane, about the centre it has
*there*. It cannot be done by scaling the drawn ellipse, because the image of a
circle is **not centred on the image of its centre** — the classical fact, and a
unit test. Eccentricity and tilt then keep following the grid for the rest of the
drag, which is the whole reason the plane is carried at all.

**The bar** (`GUIDE_CIRCLE_RESIDUAL`, 0.18 of the drawn loop's mean radius) is
wider again than the free ellipse's, for §20.6's reason: it is not asking
whether a loop was drawn but which circle it meant. Measured, on ellipses a few
hundred px across, it admits a loop about a sixth too round or leaning 5° out of
true, and declines at around a fifth and 8°. Being an isotropic fraction of the
mean radius, it forgives eccentricity more readily than tilt on a strongly
foreshortened circle — which is the right way round, since how open a
near-edge-on ellipse should be is genuinely hard to see and which way it leans
is not.

Gating is §20.6's, with one addition: a plane needs **both** of its axes shown,
being the thing the two of them span.

## 20.8 The curvilinear lens

The Fisheye toggle on the perspective bar swaps the guide's **lens** — the one
map from a canvas point to the eye's ray through it — and nothing else. That is
the whole design: the camera, the fans, the orbit drag, the axis snap and the
locks are all stated in *direction space* (§20.3, §20.5), so a guide seen
through a different lens keeps every behavior and every theorem that lives on
the view sphere, while everything drawn on the canvas bends to the new
projection. `PerspectiveGuide::ray` and `::project` are the only functions that
branch on it.

**Why stereographic.** Of the classical fisheye mappings (equidistant `f·θ`,
orthographic `f·sin θ`, stereographic `2f·tan(θ/2)`), Stark draws the
stereographic one, for two reasons that matter to a *drawing* tool. It is
**conformal** — angles survive, so the grid's local structure reads the same
everywhere and a brush-size circle stays a circle — and it takes **circles to
circles**: the image of any world line (a great circle of directions) is an
exact canvas circle, closed-form, which is what keeps every element of pass D
an analytic distance field rather than a sampled curve. The scale `2f·tan(θ/2)`
agrees with the rectilinear `f·tan θ` to first order at the center of view, so
toggling the lens leaves the drawing's heart at the same size and bows its
edges.

The forward map is `c + 2f·(d.x, d.y)/(1 + d.z)`; the inverse, with
`u = (p − c)/2f` and `s = |u|²`, is `(2u, 1 − s)/(1 + s)` — no trigonometry in
either direction, and `s > 1` reaches past the 90° ring to directions *behind*
the camera: the fisheye's field of view is the whole sphere except the exact
backward pole, its projection point.

What the artist sees change:

- **Both poles.** Directions are no longer projective — an axis and its
  negation image at two different points — so each axis can show two vanishing
  points. A 1-point pose becomes the classical **5-point curvilinear grid**:
  the view axis at the center, the four transverse poles on the 90° ring at
  its compass points (a unit test states exactly this).
- **Traces bow.** A pair plane's vanishing line becomes the stereographic
  image of its great circle: center `c + 2f·m.xy/m.z`, radius `2f/|m.z|` —
  substituting the inverse map into `m·d = 0` and completing the square. A
  plane containing the view axis (`m.z ≈ 0`) stays a straight line through
  `c`, as any great circle through the projection axis must; near that pose
  the circle is drawn *as* its limiting line below `FISHEYE_LINE_EPS`, because
  a ten-million-pixel radius is where f32's distance-to-ring subtraction
  starts to wobble, and the swap is sub-pixel where it happens.
- **Two rings.** The 45° ring moves to its stereographic radius
  `2(√2 − 1)·f ≈ 0.83f`, and the **90° ring** appears at exactly `2f` — the
  rim of the forward hemisphere, the circle the classical 5-point grid is
  drawn inside. Both are the focal length's handles: dragging either ring in
  the edit mode sets `f` through that ring's own factor, and the drag holds
  the ring it grabbed rather than handing off to the nearer one.
- **Station points go.** Rotating the eye into the picture plane is a
  flat-plane measuring construction; under a curved lens the distances it
  transfers do not exist on the canvas, so drawing them would be decoration
  pretending to be geometry.

In the shader the lens is one uniform branch producing the ray — the fan
arithmetic after it is untouched, and the guide lines emerge as circles simply
because that is where the pencil's coordinate now falls. The pair traces
travel as tagged slots (line or circle, one distance test each), the vanishing
points as six slots instead of three.

**Deliberately withheld:** a fisheye guide puts up no snapping scaffold
(§20.6, §20.7). Its guide lines are arcs, and the assist's pencils and charts
describe straight lines and flat planes — offering them would snap a stroke to
geometry the guide does not draw. Snapping strokes to the fisheye's circles is
its own future piece of work, on the same `Scaffold` seam.
