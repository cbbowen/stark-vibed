# Drawing guides

The perspective grid: one projective camera, three familiar cases — §20.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 20 Drawing guides

Guides are chrome for the hand: geometry drawn over the canvas that no tool yet
snaps to and no pixel ever records. The first guide is the classical perspective
grid, and its design principle is the one this chapter keeps returning to:
**the guide is a camera, and everything the artist sees is derived from it.**

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
- an **orientation** — yaw, pitch and roll turning the world's orthogonal axis
  frame relative to the camera.

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
branches on a sign.

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
forshortens — the fan carries its own horizon-crowding.

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
pass C is: screen renders carry the derived `GuideScene` in their
`CompositeScene`, exports and the navigator's miniature never do (§15.6), and
`render_to_image` sees it only if a test turns the guide on — the default-off
guide leaves every golden untouched.

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

## 20.5 State, panel, and what is deliberately deferred

The guide is **view state** (`Session::guide`, `ViewCommand::SetGuide`,
projected through `ObservableState`): per-client, unlogged, unsent — an aid for
the hand holding the pen, like the pan and the zoom. If guides later become
part of what a document carries (a shared scaffold peers should see), that is a
new `DocCommand` and an action; `SetGuide` would remain as the in-flight
preview half, the same bargain the matte-rect drag strikes (§4).

The Drawing Guides panel edits the camera read-modify-commit, like the
lighting panel edits `MediaParams`. Its case chips are presets that *turn* the
camera — set yaw/pitch/roll, bring `c` to the view center, enable — and light
up from the derived finite-VP count, never from a stored mode.

Deferred, deliberately (§18 discipline — nothing inert ships):

- **Direct manipulation** — dragging VPs, the horizon, the CoV and the 45°
  circle on the canvas, which will subsume most of the panel's sliders. Its
  design is the next step and owns questions like "what does dragging a VP
  hold fixed?"
- **Snapping** — strokes constrained to the nearest fan line; the assist layer
  (§6.9) is where it will live.
- **Other guide kinds** (isometric, ellipse/vanishing-scale, symmetry): each a
  new derivation over the same pass-D machinery.
