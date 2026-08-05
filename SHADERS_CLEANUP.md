# stark-shaders cleanup

A review of `crates/stark-shaders/src` — the WESL tree and the build step that
links it. Twelve items, ordered by what unblocks what. Retire this file when the
last one lands, the way `CORE_CLEANUP.md` and `STROKE_CLEANUP.md` were retired.

The organising complaint: **modules mix bindings with pure helpers, so a helper
cannot be shared without dragging a bind group along.** Everything in §1–§4 is
downstream of that. `paint_common.wesl` is the one module that got it right —
binding-free, so five pipelines import from it without friction — and the fix is
to make that the rule rather than the exception.

---

## 1. The binding-free `lib/` split

**Status: done.**

The smoking gun was in `blend_common.wesl`, on its own copy of the fullscreen
vertex:

> *A local copy rather than `media_common`'s: importing that module would drag
> its whole `@group(0)` in, and this pass binds something else entirely.*

That is an admission that the module graph is wrong, and the cost was visible:
the fullscreen triangle was written out **ten times** (blend_common, fill,
guides, integrate, matte, media_common, resolve, selection, slice, transform).

`src/shaders/lib/` now holds the binding-free leaves — `color`, `noise`,
`paint_common`, and the new `quad` (the fullscreen vertex) and `sample` (the
clamped 1×1-or-real-tile load, and the shared bilinear index arithmetic).
Anything that owns a `@group` stays at the top level. The rule is mechanical:
**a module under `lib/` may not declare a binding.**

Note for whoever revisits the blend_common comment: with wesl 0.4's defaults
(`strip: true`, `lazy: true`) unused bindings in an imported module are stripped
anyway, so the stated reason may already have been stale. The binding-free
module makes the question moot either way, which is the point.

## 2. Constants that will drift

**Status: done.**

- `OPACITY_K = 1.0` was declared **four times** — `paint_common`, `composite`,
  `matte`, `media_common` — three of them carrying a "must match" comment, while
  `paint_common` already exported it and `stamp_oklab` already imported it.
  Now one definition in `lib/paint_common.wesl`.
- `EPS` was worse: same name, **different values**. 1e-5 in
  paint_common/media_common/composite/resolve, 1e-4 in blend_common, and a bare
  `1e-4` literal in both media shaders. `blend_oklab` imported `EPS` from
  `blend_common` while `fill` imported `EPS` from `paint_common` — same
  identifier, different number, nothing at either call site to say so.

  The split is now by **which representation is being guarded**, because that is
  what the two values were tracking all along. A *tile* carries per-unit opacity
  in alpha; a *composited target* carries the stack's coverage. Un-premultiplying
  the two divides by different quantities on different scales:
  `PAINT_EPS = 1e-5` and `COVERAGE_EPS = 1e-4`, both in `lib/paint_common.wesl`.
  `resolve` keeps a local `WEIGHT_EPS` — its alpha is the box filter's own
  accumulated weight, which is neither. No value changed.
- **The tooth trio was the highest-risk instance.** `gpu/surface/tooth.rs`
  mirrored `TOOTH_SOFTNESS` and `TOOTH_RISE` by comment only, and the failure is
  silent by construction: the two halves of a toothed transfer stop balancing and
  nothing crashes. `RISE_LIMIT` was worse still — `paint_common::rise_ahead`
  folded it into the literals `255.0/512.0` and `0.25`, so even a parser could not
  check it. It is now named, and the decode reads `255.0·L/128.0 − L`, which is
  the same number to the bit (both operands exact, and the divisor a power of
  two). All three are asserted — see §3.

## 3. Assert the constants instead of asking for them in a comment

**Status: done.**

`wesl_const` already read a scalar `const` out of linked WESL and asserted the
host agreed — but it was a private helper in `dynamics.rs`'s test module, applied
to two constants in one shader. It now lives in `gpu/wesl.rs` next to
`mirrors_wesl!`, which is the module whose whole stated job this is: *"whatever
can be checked here should be, because the failure is quiet."* `mirrors_wesl!`
pins a struct's size; `wesl_const` pins a scalar both sides compute with.

Asserted now: the tooth trio (`TOOTH_SOFTNESS`, `TOOTH_RISE`, `RISE_LIMIT`) in
`gpu/surface/tooth.rs`, and `SWEEP_VERTS` in `gpu/stroke/swept.rs`, alongside the
two that already were.

Three things learned in the doing, all now in `wesl_const`'s own docs:

- **`OPACITY_K` and `MATTE_THICKNESS` have no CPU counterpart at all** — they are
  shader-only, so there was never a pair to assert. This list over-reached.
- **Imported constants arrive mangled.** `TOOTH_RISE` links as
  `package_lib__1paint_common__1TOOTH_RISE`, while a root-module constant keeps
  its name. The helper had to learn to match the suffix.
- **Compare as `f32`, not `f64`.** Both sides hold `f32`; widening the host's
  `0.06f32` gives 0.059999998…, which is not the source's `0.06`. The first run
  of the tooth assertion failed on exactly this, which at least proved it was
  reading the shader.

`SWEEP_VERTS` is checked through `SWEEP_SLICES`, because the shader states
`SWEEP_VERTS` for the host's benefit and never computes with it — so the linker
strips it, the same trap `WICK_RATE` falls into.

## 4. The duplicated helpers the split enables

- **The clamped 1×1-or-real-tile load** was `load1` in `fill` and `integrate`,
  and open-coded five times in `transform`, twice in `dynamics`, once in
  `selection`, once in `overlay`'s neighbourhood. Four separate comments called
  it "§6.8's pattern". Now `lib/sample.wesl::load1`.
- **The hand-rolled asymmetric bilinear** is written twice:
  `stamp_common::prefix_slice` and `dynamics::bake_at`, the latter's comment
  saying "for the same reasons". The texture types differ
  (`texture_2d_array` vs `texture_2d`) so they cannot fully merge, but the index
  and weight arithmetic — the part that is easy to get subtly wrong — is now one
  shared `lib/sample.wesl::bilerp_at`.

## 5. Dead code in `noise.wesl`

**Status: done.**

85 of its 111 lines were unreachable: `_pcg2d`, `_pcg3d`, `_pcg4d`, `_unit` and
all seven `noise_N_to_M` wrappers have zero callers across the whole tree. The
only live function is `color_jitter`, which samples a CPU-baked texture and
touches none of the hashes. `media_common` imports `noise_2_to_4` and never uses
it — the sole reference to the entire family.

Per the project's own rule, this was scaffolding that cannot change a pixel, and
it went the way of `tooth`, `drag` and `wetness`.

One thing it turned up: `noise.rs`'s live CPU `pcg4d` was documented as mirroring
`noise.wesl` "so CPU and GPU noise stay in the same family". There is no GPU side
to that family — the fields are CPU-baked *precisely* so they are bit-identical
across adapters, and the shader only ever samples the texture. That comment
described a contract neither side exercised, and has been corrected.

## 6. Prose describing a pass that no longer exists

**Status: done.** Six live references rewritten; the one surviving mention is
explicitly flagged as history, so a reader grepping the name learns immediately
that there is nothing to find.

`pickup` was retired — `dynamics.wesl`'s header says so at length — but it is
still named in ten places as if live: the `Stamp.a` and `Stamp.b` field docs
("for a pickup slot…", "0 for pickup"), the `region_color` binding comment
("Read by snapshot and pickup"), the reservoir ping-pong comment ("the host
swaps the bind group at each pickup"), `deposit`'s selection lerp ("the fraction
`pickup` scaled its own side of the transfer by"), and `composite.wesl`'s
`fs_raw` header. A reader cannot tell which references are historical narrative
and which are current wiring. `exchange` is the name in all six live cases.

## 7. `dynamics.wesl`: the packed uniform

**Status: outstanding.**

1624 lines is five passes plus their shared kernel, but the length is not the
tax — the **nine-`vec4` `Stamp` uniform with a prose field map** is. The body
reads `st.b.y * st.b.x`, `st.d.z`, `st.h.z`, `st.i.y`, `st.e.y`. WGSL's uniform
layout rules permit named scalar and `vec2` members, so the packing buys nothing
a properly ordered `#[repr(C)]`-matched struct would not. Failing that, a dozen
one-line accessors (`fn lift_rate() -> f32 { return -st.b.z; }`) recover most of
it at no ABI risk.

The same pattern runs through `Media` (`surf_a.z`, `surf_b.y`), `View`
(`misc.x/y/z/w`), `selection::Params` (`c.x` = kind, `c.y` = mode),
`Resolve.n.x` and `Guide`.

## 8. `transform.wesl`: three pipelines, one file, aliased layouts

**Status: outstanding.**

`Quad` at `@group(0) @binding(0)` and `Gated` at `@group(0) @binding(0)` are
different structs at the same slot; the combine pass then claims bindings 2–7 of
the same group. It works only because no entry point references two of them and
DCE prunes the rest. Nothing in the file says this out loud, and an entry point
that touched both would be silently invalid. Split into three modules over a
shared `transform_common`, or state the aliasing loudly at the top.

Related: `mixbox_lut.wesl` hardcodes `@group(0) @binding(5)` and `(6)`, correct
only because `blend_common` happens to stop at 4. Neither file mentions the
other's binding budget.

## 9. The generated `mixbox_poly.wesl` is written into the source tree

**Status: outstanding.**

Which is what forces the whole freshness apparatus in `build.rs`: a hand-rolled
directory walk, a write-only-on-change guard, a `.gitignore` entry, and a
nine-line comment about the tile-shaped-artifact failure mode that a stale
artifact produces.

wesl 0.4 has `Router::mount_resolver` + `VirtualResolver::add_module`: serve
`package::mixbox_poly` from memory and the file, the gitignore entry, the mtime
guard and the entire class of staleness bug disappear together. There is also
`wesl::emit_rerun_if_changed(&modules, &resolver)`, which emits the dependency
set from the actual import graph rather than a directory glob.

## 10. The entry-point list is maintained twice

**Status: outstanding.**

`build.rs` and `lib.rs` each carry all seventeen entry points, with no check that
they agree; adding a shader and forgetting one side fails at the wrong layer.
One `const ENTRY_POINTS: &[&str]` in a file `include!`d by both removes it.
`lib.rs`'s seventeen `pub fn x() -> &'static str { include_wesl!("x") }` can also
just be `pub const X: &str = include_wesl!("x")`.

## 11. `stamp_oklab` is a misnomer

**Status: outstanding.**

Its own header says the Mixbox space reuses it verbatim, and `colorspace.rs`
confirms both spaces call it. `stamp` is the honest name, and would stop it
reading as the surviving half of a `stamp_mixbox` pair that never existed.

## 12. Smaller items

**Status: three done, one outstanding.**

- ~~`selection.wesl` uses `f32` enum tags compared with `==`.~~ Now `u32` codes
  read out of the float lane, matching `blend_common`'s `BlendMode`. The equality
  was in fact safe — every code is a small integer, exact in f32 — but a tag is an
  enumerand, not a measurement, and the next code to be added might not be.
- ~~`noise.wesl` and `media_common` use `vec2f`/`vec2u`/`vec3f`.~~ All long-form
  now.
- ~~`noise.wesl::_unit` carries a commented-out alternative implementation.~~ Gone
  with §5.
- `blend_common::merge` takes `cb`/`cs` as parameters but reads `back_aux` /
  `src_aux` from globals — hence its otherwise-unused `p` argument. Asymmetric
  enough to slow a reader down. **Outstanding**, and a genuine design call rather
  than a tidy-up: `merge` either takes everything or reads everything.
