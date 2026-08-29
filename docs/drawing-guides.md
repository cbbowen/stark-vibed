# Drawing guides

The perspective grid: one projective camera, three familiar cases — §20.
What a stroke aligns to is in §20.6 and §20.7.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 20 Drawing guides

Guides are scaffolding for the construction: geometry drawn over the canvas that
a stroke can be aligned to (§20.6) and that no pixel ever records. They are
**part of the document** all the same (§20.5) — saved, replicated, undoable —
because a perspective built over a drawing is part of how that drawing was made,
even though nothing it draws is paint. The first guide is the classical
perspective grid, and its design principle is the one this chapter keeps
returning to: **the guide is a camera, and everything the artist sees is derived
from it.**

## 20.1 One camera, three special cases

Art tools habitually ship "1-point", "2-point" and "3-point perspective" as
three modes with three data models — one draggable dot, two dots on a shared
horizon, three free dots — and each mode's invariants (horizontals stay level,
verticals stay parallel) are enforced by *code* in that mode. Stark refuses the
split. The guide's state is exactly a projective camera
(`stark-model`'s `document::guide`):

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

## 20.3 The fans: a lattice of squares

What should "grid lines toward a vanishing point" mean? Any fan through the VP
is projectively *a* pencil; the question is how to space its members, and the
answer decides whether the grid is something to measure on or something to look
at. **The cells two fans cut on the plane they span must be squares.** Not
rectangles that happen to be square somewhere, and not rectangles of some fixed
proportion: one lattice cell each, everywhere, so that counting cells across a
drawing is measuring it and an object placed on the grid has a size.

The obvious parametrization fails this, and it is worth saying how, because it
is the one nearly every tool ships and it looks plausible on screen. Step the
pencil by equal **dihedral angle** — equal turns of the eye — and the guide
lines land on the pair plane at `y = −cot θ`. A cotangent run is not an
arithmetic one: consecutive cells differ in size by a factor that itself
changes cell to cell, so no two rows of the grid are the same shape and nothing
on it measures anything. It crowds toward the vanishing line, which is why it
passes a glance, but not by the law foreshortening actually obeys.

**One cube, three planes, six families.** So the guide carries a lattice: a
cubic grid of cells, whose three coordinate planes through one **corner** each
carry a grid of squares. The whole of it is one vector — `lattice`, how far each
plane lies from the eye in *cells*, one component per world axis, which is the
same thing as the corner they meet at. It has to be a world quantity, because a
camera has no world scale of its own and the only visible fact about a cell is
how many of them lie between the eye and a plane. Scaling that vector is the
density control (twice as many cells to the same planes is a grid of half the
size); turning it is where the grid sits.

**The grid's phase is the eye, not the corner.** Every guide line lies a whole
number of cells from the *viewer*, so the foot of the eye's own perpendicular to
each plane is one of that plane's cell corners: look straight down any axis and
the crossing beneath the center of view is the grid's own — at every scale, and
for any lattice whatever. Written out, the family of lines parallel to `a_i` on
the plane at distance `p` is just `p·u = −m·v` for integer `m`; the corner never
enters it, and the two families of an axis differ only in which plane's distance
they carry.

Anchoring the phase at the corner instead — counting cells from *it* — makes
that true only when the corner's components happen to be whole, and a scale
control that halves is exactly what breaks that: `(-4, 3, 6)` halves twice to
`(-1, ¾, 1½)`, one axis on a line, one a quarter off and one a half. Three grids
laid out separately is what that reads as. With the phase on the viewer nothing
about the corner needs to be round, a corner placed anywhere keeps the grid hung
on the eye, and the density scale expands and contracts *about the viewer*
rather than pivoting on a line some cells away.

**The scale halves and doubles, and does nothing in between.** The control offers
only powers of two of the default, because
a grid meant to be *counted* on should refine rather than slide. At exactly
twice the cells, every line of the coarser grid is still a line of the finer one
and one new line appears between each pair — nothing already counted against
moves. At any other ratio every line slides along its pencil toward the corner's
own edge, and since a scaling has a fixed line, the visible part of the grid
mostly sits to one side of it and the whole thing reads as drifting sideways
rather than as a change of scale. That is a property of scaling a lattice, not a
defect in this one: the only way for a density control not to slide is for it
not to be continuous.

Six families and not three, and that is forced rather than chosen. Axis `a_i`
lies in two pair planes, and its guide lines on one of them are the images of
world lines parallel to `a_i` offset along `a_j`. Those same planes-through-the-
eye meet the *other* pair plane at offsets that are the **reciprocals** of the
first sequence — so a family evenly spaced on one of an axis's two planes is a
harmonic run on the other, and no family can square both. Three families can
square exactly one plane; squaring all three costs two per axis. What makes the
six of them one lattice rather than three grids drawn side by side is that they
are all counted from the same place: the zeroth line of every family is the one
directly beneath the eye on its own plane.

**A family is a linear level set, and that is the whole of the shader.** Let `r`
be the eye's ray through a texel (`((p − c)/f, 1)`, or the fisheye's inverse
map), let `P` be the lattice, and resolve the ray on the axes other than `a_i`:

```
u = r·a_{i+1}      v = −r·a_{i+2}
```

Then the family stepping along `a_{i+1}` — which lies in the plane at distance
`P_{i+2}` — is `−P_{i+2}·u = m·v` for integer `m`, and the one stepping along
`a_{i+2}` is `−P_{i+1}·v = m·u`. Each guide line is the zero set of
`w − i·den`, which is linear in the ray — so its distance field is exact, its
gradient is the same one that gives the family's spacing (`|den| / |∇|`, which
drives the moiré fade), and the level set is taken rather than the quotient
`|w/den − i|` so the arithmetic stays sound where the quotient runs away.

Everything the old parametrization was chosen for survives, and the reasons are
better ones. Scale never matters — `r` is used unnormalized and every expression
is homogeneous, so the fisheye's ray cancels. **No vanishing point is ever
computed on this path** and there is no case to branch on, which is what makes
the 1/2/3-point uniformity structural: an axis lying in the picture plane simply
has `u` or `v` constant and its family comes out as evenly stepped parallel
lines. The fan still crowds toward a pair's vanishing line, but now it crowds
*correctly*: `den → 0` there, the spacing closes up harmonically, and the fade
takes it to nothing before it can shimmer. And the vanishing line is no longer
one of the fan lines but their accumulation point, which is what it is.

**A family belongs to its plane, not to its axis.** Two rules follow, and both
are about keeping the overlay legible rather than about geometry:

One thing the cube buys that is worth stating on its own: **where two planes
meet, their grids meet**. Along the edge a pair shares, both put their cell
corners on the same points, one cell apart (unit-tested) — so a count carried
along an edge means the same on either side of it, and a box drawn in one plane
lands on the grid of the next. What the cube does *not* buy is agreement away
from the edges: two planes' lines toward a shared vanishing point are two
different families of world lines, and where the image happens to bring one of
each close together they read as a doubled line. That is not a phase error to be
corrected, it is what showing three planes at once costs; the levers against it
are switching a plane off and the obliquity fade below.

- **The plane is what is shown or hidden** (`PerspectiveGuide::pairs`, three
  flags: XY, YZ, ZX). It is the unit the drawing has: a guide line lies *in* a
  plane, so the two families ruling one go together, and everything else the
  plane carries — its vanishing trace, the horizon that turns about its normal
  (§20.5), its station point — goes with them. An axis is not a thing that can
  be hidden; what an axis has is lines, and every one of them is on one of its
  two planes.
- **An axis is *drawn* when either of its planes is** (`is_drawn`). That is the
  derived half, and what the markers belonging to an axis alone follow: its
  vanishing point, and the pencil §20.6 offers a stroke. Lose both of an axis's
  planes and it rules nothing anywhere, however well defined its direction still
  is — so nothing of it may bend a stroke either.

  Naming the plane rather than the axis is also what makes the control
  complete. Three axis toggles reached three of the eight states — none, one
  plane, or all three — because a plane needed *both* its axes, so the second
  axis always brought a second plane free with it. "The ground and the near
  wall, not the far one" is three of those eight rows and was not sayable at
  all, and it is the most common thing to want.
- **A plane recedes in the periphery, the sooner the more squarely it faces the
  eye.** A plane the view axis is normal to repeats across the whole canvas
  without ever foreshortening: out at the edges it is saying nothing the middle
  did not already say, and it says it over everything else. A plane the view
  axis *lies in* is doing the opposite out there — running away toward its own
  vanishing line, where every cell is a fresh statement — so it recedes at a
  fraction of the rate and only well past the useful cone. The rate is set by
  the plane normal's component along the view axis, which for the plane two axes
  span is just the third axis's own `z`, and the distance is measured in units
  of the **45° ring**, so the rule names the same visual angle at any focal
  length and under either lens. One pose reads the whole scale: look straight
  down an axis and the picture-plane wall thins to a patch around the center
  while the floor and side walls, whose lines all converge, stay.

  This is the overlay's second legibility rule on the fans and it deliberately
  does not duplicate the first (§20.4): the moiré fade answers *these lines are
  too close together to resolve*, and this one answers *these lines are too far
  out to mean anything*.

The one pose the lattice cannot answer for is a corner with a zero component:
that puts the eye *inside* the pair plane normal to that axis, which then images
as a line, and the two families lying in it have no grid left to show. The pass
declines it by the test that says what it actually is — a level set with no
**gradient** is not a curve, so there is no line here rather than a line
everywhere. A corner at the eye itself is the same fact for all six families at
once, and is `None` at the source.

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

- **A fan line is a level set, not a quotient.** A family's guide lines are
  `w − i·den = 0` (§20.3), linear in the eye's ray, so `|w − i·den| / |∇|` is
  the distance in screen px and `|den| / |∇|` is the spacing — one gradient,
  taken analytically, serving both. Measuring `|w/den − i|` instead would be
  the same number where it is well conditioned and nonsense where it is not,
  which is precisely near a vanishing line, where the lines are.
- **A moiré fade, not a clip.** Where fan lines pack tighter than a few px —
  approaching vanishing lines and VPs — coverage fades smoothly to nothing
  instead of shimmering. The fade is driven by the same measured spacing, so
  it tracks zoom for free. The obliquity fade (§20.3) rides in the same
  product and answers the opposite question: not *too close to resolve* but
  *too far out to mean anything*.

The markers (CoV crosshair, VP discs in their axis's hue, station-point rings,
the dashed 45° circle) ride in the same pass as more distance fields, packed
as `position + valid` uniform slots so the shader branches on data, never on
pipeline variants. Axis hues follow the X/Y/Z semantics every 3D tool taught.

## 20.5 The roster, the edit mode, and the drag

Guides are a **roster**, and the roster is **document state**: saved with the
file, replicated to peers, and reachable by undo, exactly like the layer stack.
Each guide has a `GuideId`, a camera (`PerspectiveGuide`), a name, and a place
in the arrangement, and every edit is a logged action — `AddGuide`,
`SetGuide`, `SetGuideName`, `MoveGuide`, `RemoveGuide`.

This was the other way round. A guide was view state — per-client, unlogged,
unsent, outside the undo history — on the argument that it is an aid for the
hand holding the pen, like the pan and the zoom. That argument was wrong about
what a guide *is*. A perspective built over a drawing is part of the drawing's
construction: it is what the artist reasons in, it takes as much care to set up
as a layer takes to paint, and losing it on reload — or leaving a collaborator
unable to see the scaffold the work is being built on — is losing work. It is
the argument §6.4 makes for the canvas substrate and §15.5 makes for the substrate
color, and the guides sat on the wrong side of it.

**The one exception is the eye.** Whether a guide is *drawn* is per-client
(`Session::visible_guides`, `ViewCommand::SetGuideVisible`): shutting a guide to see
the picture underneath must not reach across a shared session, must not be saved,
and must not cost an undo step — an artist who hides a guide and then presses
undo wants their last *edit* back. So a guide is two things kept in two places,
and the two are combined into one per-client reading of the roster (`GuideInfo`)
by `Engine::observe`.

**A guide is hidden until this client asks for it.** That is the second half of
the same decision, and it is the half that only became necessary once the first
one landed: a document now carries every perspective anyone ever built over it,
so opening a painting from last year — or joining somebody's session — would
otherwise lay all of them across the canvas at once. Scaffolding you did not ask
for is not scaffolding; it is something to clear away, and a tool that hands you
that on open has made the feature cost more than it gives. The construction is
kept because it is part of the drawing; looking at it stays a thing you do. It
also makes the guides free to accumulate — a document may carry a dozen, one per
part of the picture, without any of them being in the way of the others.

So the state is the set of guides this client has **opened**, and an absent entry
means not drawn. That is the cheaper end as well as the right default: a client
typically has one guide up out of however many the document holds, so the set
holds what is in use rather than what is being ignored. An id stays in it when
its guide is removed, deliberately — the removal may be undone, and a guide that
shut its own eye on the way back would be the tool changing what you are looking
at, which here would look like an undo that only half worked.

Nothing opens an eye but the artist, and the panel needs exactly one line for it:
**picking a guide up to shape it opens it** (`begin_guide_edit`). Adding and
duplicating come free, because both end by picking up what they made — so "the
guide I am working on is one I can see" is stated once rather than at each path
that can produce a guide, and there is no way to end up in the edit mode dragging
a construction that is not on the screen. The row's eye is the only other way in,
and it is the artist saying so directly.

### Identity, and why there is no counter

**A guide's id is the id of the action that added it.** An `ActionId` is already
the log's total-order key `(lamport, actor)`, so it is already globally unique:
two actions cannot share one, therefore two guides cannot. That makes "no two
guides carry the same id" a property of the representation rather than a rule a
call site could forget (§1), and it means there is no per-actor counter to
partition and nothing for `resync_counters` to resume past.

**Layers followed** (§17.9). What kept them from doing so at first was a
difference in the vocabularies rather than a preference: one `AddGuide` mints
exactly one guide, where `DuplicateLayer` mints one id per layer of a subtree, and
an action id can name one thing. A `LayerId` therefore carries a `k` beside the
action id — which of that action's layers this is — and a guide, minting one, has
no use for it.

The engine therefore mints nothing when a guide is added, and the frontend
learns the new id by reading the roster back off the projection — which is why
`AddGuide` carries the *name* too, rather than leaving a following
`SetGuideName` to supply it: duplicating a guide copies the artist's own word for
it, and two actions would put one gesture two undo steps deep with a nameless
guide in between. It is `PlaceImage`'s bargain (§23), for its reason.

`AddGuide` and `MoveGuide` anchor with `after: Option<GuideId>` — the guide the
new one lands directly after, or the head of the roster when `None`. A flat list
of `n` guides has `n + 1` places to land in and one optional id reaches every one
of them, which is why guides need no `Place` where a layer stack does (§14.8):
`Place` exists because a stack's two-state anchor could not say "under the bottom
layer", and `None` meaning the head is that third state spelled without a type.

### One resource, one preview

The whole roster is **one footprint resource**, `Resource::Guides` (§12.6). Two
concurrent guide edits therefore do not commute, and that is the trade taken
knowingly: spelling it finely — a resource per guide, plus one for the
arrangement — buys only the case of two artists shaping two *different* guides at
once, while a footprint may always claim too much. Nothing on the drawing path
contends with it. A stroke reads the guides to snap through them, but it reads
them before it is an action at all, so the action it commits names this nowhere.

Every control that is *dragged* previews and commits, the bargain
`crate::preview` states once for the whole app: `ViewCommand::PreviewGuide` shows
a pose per pointer sample without logging it, and one `DocCommand::SetGuide` is
laid down when the gesture settles (§15.7, §21.6). So shaping a perspective costs
one undo step per adjustment rather than one per sample of the hand — and
`SetGuide` carries the **whole camera** for the reason `SetFilter` carries the
whole filter: a guide that grows a knob then needs no new action and no
wire-format break. The controls that settle the moment they are used — a plane
chip, the lens toggle — commit directly. Rendering-side, each guide this client
draws gets a dynamic-offset uniform slot and its own fullscreen draw in pass D —
slots rather than one rewritten buffer for the reason `BLEND_SLOT` records.

Retiring the old whole-list `SetGuides` view command was free where retiring a
*logged* action would not have been (§8, §19): a `ViewCommand` is never
serialized, so it has no files to keep readable. The five actions that replaced
it are new variants in `ActionKind`, which is a change to the wire's shape — the
ALPN moves to `stark/collab/9`.

### The panel

The **Drawing Guides panel** is the roster, shaped like the Layers panel
because it answers the same question about a different stack: "Add
Perspective" (placed at the view center, so the grid lands where you look),
one row per guide, carrying what a layer row carries in the same order and
glyphs: the name, then duplicate, then remove, then the eye. Every one of those
is a logged edit except the last, which is the eye. Duplicating drops the copy in
the row below and picks it up, because duplicating a guide is asking to shape a
variant of it — the same argument that has "Add Perspective" open the mode on
what it made. The copy keeps the source's name for the reason the layer duplicate
does (§14.8): a name is the author's own word. The name renames on a
double-click, exactly as a layer's does — one draft per row, committed on Enter
or blur, abandoned on Escape, and a field left empty is *no name* rather than a
blank one, so the row goes back to describing its position ("Perspective 2").
That last rule is `normalize_name` in the engine, shared with layers and applied
to every guide name that reaches a `SetGuideName` or an `AddGuide`.

A row **moves by being dragged**, and it is the same gesture the layer tree is
arranged with, down to the code (§14.6, `panels::reorder`): the rows open the
slot the guide is going into and the guide floats over it, so the preview is the
answer rather than a symbol for one. What differs is only what a landing *means*
— this list is flat, so nothing sideways is asked of the hand where the layer
tree asks for a depth. The drag measures drawn rows against each other and so
speaks in positions; the action has to name a *guide*, or it would mean something
else on a peer whose roster has moved, and `anchor_at` is the one place the two
readings meet. Reordering deliberately does *not* enter the edit mode: arranging
the roster is tidying, and tidying must not take over the canvas.

What the ids removed is a whole class of bookkeeping. The mode holds a `GuideId`,
so a removal or a reorder elsewhere in the roster moves nothing it points at —
where an index had to be re-pointed by every path that could move the list, and
would silently address a *different* guide from any path that forgot.

Selecting a row — or adding — enters the **edit mode**, and opens that guide's
eye on the way in (a guide is hidden until asked for, and shaping one you cannot
see would be dragging an invisible construction): a full-viewport catcher owns
the pointer for the mode's duration (the transform-mode bargain, §16.6;
navigation still works), and the **Perspective Guide bar** stands at the bottom
with the per-axis locks, per-plane visibility (§20.3 — three chips, XY / YZ / ZX,
each lettered in its two axes' own hues), the Fisheye lens toggle (§20.8), the
cell count, opacity, and "Done". The locks stay per-*axis* and the visibility is
per-*plane*, which is not an inconsistency: a lock constrains a rotation, and a
rotation is about an axis; visibility governs what is drawn, and what is drawn is
a plane. That per-plane visibility is document state, unlike the guide's own eye,
and for the reason the split runs where it does — which planes a perspective
rules is a statement about the construction, where whether you are looking at it
is a statement about you. The cell count is the *length* of the lattice (§20.3) —
how many cells lie between the eye and its corner — so the slider scales the grid
and leaves the corner where it is, and it steps in **halvings** so that a change
of scale subdivides the grid instead of sliding it. Which of 1/2/3-point you are
in is never stored or displayed as a mode: the canvas shows it, as the count of
finite vanishing points.

There is no Cancel chip on the bar, alone among the mode bars, and it is not the
lie it would have been when a guide was live view state: every control lays its
own value down as it settles, so by the time you could reach a Cancel there is
nothing uncommitted for it to keep back. What a mistake costs is one undo step,
which is what it costs everywhere else in the app.

### The drag

The drag *is* the manipulation, classified by what the press lands on:

- **Anywhere: grab the world.** The world direction under the pointer follows
  it — `PerspectiveGuide::dragged` rotates the frame by the arc from the
  press's eye-ray to the current one, always recomputed from the drag's start
  so nothing drifts. The grabbed direction lands exactly under the pointer, at
  every pose and under either lens.
- **A horizon: turn about one axis.** A pair plane's vanishing line runs
  between the two vanishing points of the axes spanning it, which makes it the
  line belonging to the *third* — the plane it is normal to. So grabbing the
  line between the X and Z vanishing points orbits Y, and a 2-point setup
  dragged by its own horizon stays exactly 2-point. `PerspectiveGuide::horizons`
  is that list indexed the way the hand thinks of it, by the axis turned about,
  and it offers a horizon only where the guide draws one — a pair missing an
  axis, an overlay turned down to nothing, or a trace at infinity puts up no
  handle, the gating §20.6 states for the same reason. The guide's own eye is
  the one gate that is *not* asked there, because it is not a fact the camera
  carries: it is applied one level up, by the engine handing `Scaffold::of` only
  the guides this client draws.
- **Locks** are the same constraint made standing rather than momentary:
  rotations fixing an axis are exactly the turns about it, so one locked axis
  confines *every* drag to its orbit, and two pin the frame (the identity is
  the only rotation fixing two axes). A horizon grab is a lock on its axis held
  for the drag's duration, and it arrives at `dragged` as one — so there is one
  constrained-turn path rather than two, and a lit lock the horizon contradicts
  pins the frame by the standing two-lock rule instead of by a new one. Locks
  are gesture state, held by the mode and released with it: a constraint on the
  hand for one sitting, not a fact about the guide, which is why they are the
  one thing on that bar the document never hears about.
- **The 45° circle: drag the lens.** The circle's radius *is* the focal
  length, so `f` becomes the distance from the center to the pointer and the
  ring follows the hand exactly — the §20.1 identity made into a handle.
- **The crosshair: move the construction.** The center of view follows the
  drag, grab-offset preserved.

The crosshair is topmost, as it is drawn; below it the rings and the horizons
compete on distance in **screen** px, so a press between two curves takes the
one it is nearer and every handle stays equally grabbable at any magnification.
A dead tie goes to the ring, which matters under the fisheye, where a pair
trace can *be* one (§20.8).

**There is no snap.** A free drag that passes near an axis turn stays free.
This was the other way round — a rotation axis landing within ~15° of a world
axis was taken to mean a pure turn about it — and the cone had to be entered
*during* the drag, since the axis is only known once the hand has moved. So the
same gesture was free at the press and constrained a moment later, and a
rotation that changes what it is halfway through reads as the tool taking the
guide out of the hand. Making it a region of the canvas instead moves the whole
decision to the press, where the artist makes it: reach for the horizon and get
the constrained turn, reach anywhere else and get the free one. The constraint
is also now *visible before it applies* — the line you grab is the line drawn
between the two vanishing points that stay put — where the cone could only be
discovered by falling into it.

Deferred, deliberately (§18 discipline — nothing inert ships):

- **Dragging vanishing points themselves** — a real design question (what does
  it hold fixed: the focal length? the other VPs?) that the orbit-with-locks
  vocabulary may make unnecessary.
- **Dragging the lattice's corner**, which would place the grid in the world
  rather than only scale it (§20.3). The slider says how far away the corner is
  in cells; nothing yet says which way, beyond the default and whatever the
  orbit carries it to. It is one more region for the press to classify into,
  and it waits until the question of what a placed corner should hold fixed
  under a subsequent orbit has an answer worth shipping.
- **Other guide kinds** (isometric, ellipse/vanishing-scale, symmetry): each a
  new derivation over the same pass-D machinery and the same roster.

## 20.6 Strokes on the grid

The grid earns its keep when the hand can put a line on it. Drag out a rough
line and hold (§6.9): if it lands near an axis of a guide that is on the
screen, the stroke that snaps is aimed exactly along that axis, and the rest of
the drag runs the end out **along** it.

**What a stroke aligns to is a pencil, not a direction.** `AxisPencil` — one
axis of one guide — is the whole of a perspective guide that `assist/recognize.rs` sees,
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
a sampling of the pencil at whatever cell size the slider says, and there is no
reason a stroke's position should quantize to a display setting. Every line of
the pencil is equally a line of the grid.

**How near is near enough is an angle, spelled as a residual.** The trace is
re-scored against the pencil's line by exactly the measure that just accepted
it as a line — worst perpendicular distance, as a fraction of its own length —
against a bar (`GUIDE_LINE_RESIDUAL`, 0.15 ≈ 8.6°) wider than the recognizer's
own — wide enough to hold the *aim* rather than the wrist, since where a stroke
lands is decided at the moment the hand commits to a direction, and a vanishing
point off the page is not somewhere the eye can look while drawing toward it.
The two bars are asking different questions and are deliberately not
interchangeable: `LINE_RESIDUAL` asks *whether the hand drew a line*, where a
false positive replaces a considered curve; this one asks *which line it
meant*, after the artist has dwelt to ask for an ideal one with a grid up to
answer with. So the guide question is put strictly **after** recognition has
accepted the stroke, never instead of it — a curve that happens to bow along a
fan line is still a curve — and among the axes that pass, the closest wins.

**Only what is shown may bend a stroke.** `PerspectiveGuide::pencils` is gated
on the guide's eye, the planes it draws, and the overlay opacity, in one place:
a snap the artist cannot see coming reads as the tool bending a considered
line, so anything invisible offers nothing. An axis whose two planes are both
switched off is covered by the same rule rather than by an exception — its lines
live on those planes (§20.3), so it draws nothing, and an axis that draws
nothing offers nothing. A document with no guides up gathers an empty list and
the assist behaves precisely as it did before.

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

**The bar** (`GUIDE_CIRCLE_RESIDUAL`, 0.26 of the drawn loop's mean radius) is
wider again than the free ellipse's, for §20.6's reason: it is not asking
whether a loop was drawn but which circle it meant. Measured, on ellipses a few
hundred px across, it admits a loop about a quarter too round or leaning 9° out
of true, and declines at around a third and 11° — figures for the plane that
shows the tilt most, since a plane seen closer to face-on forgives 15° or 40°,
and correctly, a rounder image having proportionally less tilt to be wrong
about. Being an isotropic fraction of the
mean radius, it forgives eccentricity more readily than tilt on a strongly
foreshortened circle — which is the right way round, since how open a
near-edge-on ellipse should be is genuinely hard to see and which way it leans
is not.

Gating is §20.6's, asked of the plane's own flag rather than of an axis: a
circle is drawn *on a plane*, so the plane switched off is exactly the plane no
loop may be read as a circle on.

## 20.8 The curvilinear lens

The Fisheye toggle on the perspective bar swaps the guide's **lens** — the one
map from a canvas point to the eye's ray through it — and nothing else. That is
the whole design: the camera, the fans, the orbit drag, the horizon grab and
the locks are all stated in *direction space* (§20.3, §20.5), so a guide seen
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
because a family's condition is linear in the ray (§20.3) and the inverse
stereographic map takes a line in the ray to a circle on the canvas. The
lattice bows with them, so the cells stay one cell each along every arc. The pair traces
travel as tagged slots (line or circle, one distance test each), the vanishing
points as six slots instead of three.

**Deliberately withheld:** a fisheye guide puts up no snapping scaffold
(§20.6, §20.7). Its guide lines are arcs, and the assist's pencils and charts
describe straight lines and flat planes — offering them would snap a stroke to
geometry the guide does not draw. Snapping strokes to the fisheye's circles is
its own future piece of work, on the same `Scaffold` seam.
