# `stark-shaders` cleanup

A review of `crates/stark-shaders` — `build.rs`, the mirror generator, all 37 WESL
modules, and the `stark-core` code that consumes them (`gpu/desc.rs`,
`gpu/stroke/dynamics/slots.rs`, `gpu/composite/blend.rs`, `gpu/selection.rs`).

**Nothing here is a live bug.** The two closest are §1 (a trap that springs the moment
someone extends `BINDINGS`, which `build.rs`'s own comment invites) and §5 (a
divergence that would be real but invisible until a particular brush radius).

Almost every finding is one shape: **a good mechanism applied to a list of instances
instead of to a class.** `CLAUDE.md` states the rule — *rule out a class rather than
enumerate its instances* — and §6.10 is currently the largest violation of it in the
repo.

Since there are no stability guarantees yet, none of this needs to be bit-identical;
where a fix moves a uniform's layout, the generator's `offset_of` assertions turn a
mistake into a build failure at the struct it got wrong.

## Status

| # | Finding | Status |
|---|---|---|
| 1 | [`Binding` carries no `@group`](#1-binding-carries-no-group) | **done** |
| 2 | [The generator is enumerated where it should be exhaustive](#2-the-generator-is-enumerated-where-it-should-be-exhaustive) | **generator done**; ~10 host layouts still hand-written |
| 3 | [`BindKind` pays for a `wgpu` dependency it already has](#3-bindkind-pays-for-a-wgpu-dependency-it-already-has) | **done** |
| 4 | [`Stamp`'s twelve anonymous lanes defeat the mirror](#4-stamps-twelve-anonymous-lanes-defeat-the-mirror) | **done** |
| 5 | [`deposit` and `deposit_coarse` are held together by a comment](#5-deposit-and-deposit_coarse-are-held-together-by-a-comment) | **done** |
| 6 | [The group-0 index space is partitioned by comment](#6-the-group-0-index-space-is-partitioned-by-comment) | **done** |
| 7 | [The bleed ladder recomputes its neighbours ~36× over](#7-performance-the-bleed-ladder-recomputes-its-neighbours-36-over) | **the per-texel wins done**; the scratch pass open |
| 8 | [The `@if(resid)` tax is 128 sites](#8-the-ifresid-tax-is-128-sites) | open |
| 9 | [Smaller items](#9-smaller-items) | **three done**, two open |

**One thing the work found that the review did not.** wesl 0.4's `validate: true`
resolves names, counts call arguments and rejects cycles — it does **not** typecheck.
Handing a `Sweep` to a parameter declared `vec3<f32>` linked cleanly, deposited an
artifact, and failed at `create_shader_module`: at run time, on a GPU, in the half of
the suite CI runs without comparing pixels. It cost eleven red tests and a bisect to
find something the compiler knew. `build.rs` now runs `naga` — the same front end
`wgpu` uses — over every linked artifact, so that class fails at `cargo build` with the
line and the two types named. Finding 9 listed this as a hypothetical; it is not.

---

## 1. `Binding` carries no `@group`

> **Done.** `Binding` carries `group`, and a host names a slot by taking the whole
> declaration (`decl::REGION_COLOR`) rather than resolving an index against a table.
> `Binding::lookup` is gone with the question it answered, and `desc::layout_for` can
> now assert that a slot list is one group's worth.

`build/mirror.rs::emit_bindings` walks every `@binding` declaration in a module and
emits `Binding { index, name, kind, resid }`. It never reads `@group`.
`stark_shaders::Binding::lookup` and the generated `binding_of` key on `index` alone,
and `gpu::desc::layout_for` takes one table per layout.

This is correct today only by accident. `BINDINGS` in `build.rs` contains exactly
`["dynamics"]`, and `dynamics.wesl`'s own file happens to declare a single group — the
group-1 `prefix_tex` lives in the imported `stamp_common`, which `emit_bindings` never
reads, because it reads the *unlinked* source of the named module alone.

Add `stamp_common` to the list and it breaks quietly: `xf@g0b0`, `prefix_tex@g1b0` and
`noise_tex@g2b0` all have index 0, so `lookup(table, 0)` returns `xf` and the other two
become unreachable. The name-collision assertion does not fire, because the *names*
differ. `composite`, `transform`, `matte`, `overlay` and `mask_region` are all
multi-group too.

**Fix.** Carry `group: u32` on `Binding`, key `lookup` on `(group, index)`, and emit
either a per-group table or one table the host filters. The `binding::<NAME>` constants
are unaffected — they are already unambiguous, since the name is the WESL variable's.

## 2. The generator is enumerated where it should be exhaustive

> **Generator done; the host layouts are the remainder.** `MIRRORS`, `CONSTS` and
> `BINDINGS` are deleted: `build/mirror.rs` reads the whole tree and emits every
> `@binding`, every `const` with a Rust spelling, and every struct a `var<uniform>`
> names, across all 24 modules. What is still named is what the shader does not say —
> `SHARED` (two modules, one host type) and `VERTEX` (the *name* of a record), and the
> latter is now a build failure to omit rather than a list to remember. Anything
> discovery cannot spell is skipped with a note in the generated file's header; five
> `vec3` consts in `guides.wesl` are the whole of that.
>
> The ABI codes are done: `blend_code` and selection's kind/mode codes are the shaders'
> own numbers now.
>
> **What is left** is the other side — the ~10 host layouts in `composite/{blend,
> filter,media,overlay,resolve,tiles}.rs`, `fill.rs`, `merge.rs`, `selection.rs`,
> `transform.rs` and `stroke/{swept,dynamics/kit}.rs` that still write
> `desc::load_tex(3, frag)` and close with `if resid { entries.push(..) }`. Each needs
> a slot list in `slots.rs`'s shape. Three of them (`blend`, `filter`, `media`) share
> one layout across both color spaces while `blend_mixbox` declares its residual
> bindings *unconditionally* — reached only by the space that has one — so those need a
> `Slot::when_resid` for a gate the shader genuinely does not state.

Four hand-maintained lists in `build.rs` (`MIRRORS`, `CONSTS`, `VERTEX`, `BINDINGS`)
name things the shader already declares. `MIRRORS` and `VERTEX` are currently
complete — all 18 uniform structs and all 5 vertex entry points with `@location`
parameters are covered. `CONSTS` and `BINDINGS` are not, and the gaps are exactly the
drift class §6.10 exists to close.

**`BINDINGS` covers 1 of ~21 entry-point modules.** Everywhere else the host writes the
index *and* the binding type *and* the residual gate by hand:

- `gpu/composite/blend.rs`, entries 1–8
- `gpu/composite/filter.rs`, entries 1–7
- `gpu/composite/media.rs`, entries 0–7
- `gpu/composite/{overlay,resolve,tiles}.rs`
- `gpu/fill.rs` 0–5, `gpu/merge.rs` 0–6 (twice), `gpu/selection.rs`,
  `gpu/transform.rs`, `gpu/stroke/{swept,dynamics/kit}.rs`

Several of these still close with `if resid { entries.push(desc::load_tex(N, frag)) }` —
the same hand-counted residual tail that the `Binding::resid` flag was introduced to
retire for `dynamics`.

**Shader ABI codes are half generated.** `FILTER_COLOR` / `FILTER_CHROMATIC` /
`FILTER_GRADIENT_MAP` are in `CONSTS` and reach the host through the mirror.
`MODE_NORMAL` / `MODE_REINHARD` / `MODE_DRAGO` / `MODE_MULTIPLY` are transcribed by hand
in `gpu/composite/blend.rs::blend_code`; selection's `MODE_INVERT` is a bare
`const MODE_INVERT: f32 = 4.0` in `gpu/selection.rs`; and selection's
`KIND_ALL` / `KIND_RECT` / `KIND_ELLIPSE` / `KIND_LASSO` are mirrored nowhere at all.
`blend_common.wesl`'s comment says they are "mirrored by `blend_code` in
`gpu/composite.rs`" — mirrored *by hand* is the thing §6.10 was written against.

**Fix.** Invert the default. Auto-discover every `@binding`, every struct reachable from
a `var<uniform>`, every `@vertex` `@location` record, and every `const` with an explicit
scalar type. **Skip** rather than panic on anything with no Rust spelling — a nested
struct, a non-scalar const — since discovery must not fail on a declaration nothing
asked for. Keep the explicit lists as a *required*-set assertion, so an item the host
depends on that gets silently skipped is still a build failure at the use site rather
than a missing symbol.

The four lists then stop being something to remember to update when a shader grows a
binding, which is what ruling out the class means here.

## 3. `BindKind` pays for a `wgpu` dependency it already has

> **Done.** `BindKind` carries `wgpu::TextureFormat` and `wgpu::TextureViewDimension`;
> `desc.rs`'s `storage_format` and `view_dimension` are deleted with their runtime
> panics, and an unmapped format now stops the build that generated it. The stale
> `lib.rs` claim is rewritten.

`src/lib.rs` states, on `BindKind`:

> `wgpu` is deliberately not a dependency of this crate, so the kind is spelled
> structurally here and turned into a `BindingType` by the one consumer that has
> `wgpu` in scope (`stark_core::gpu::desc`).

`Cargo.toml` has `wgpu = { workspace = true }` in `[dependencies]`, for the generated
vertex attribute arrays. The doc is stale.

The stale doc is minor; the code it justifies is not. Because
`BindKind::Storage { format: &'static str }` carries a WGSL *name*, `gpu/desc.rs`
carries `storage_format` and `view_dimension` — two string matches with runtime panics,
for a fact the generator knew at build time.

**Fix.** Have `mirror.rs` emit `wgpu::TextureFormat::Rgba16Float` and
`wgpu::TextureViewDimension::D2Array` directly. Roughly 40 lines of host code and two
panic paths delete, and an unmapped format becomes a build error rather than a startup
one. Fix the `BindKind` doc either way.

## 4. `Stamp`'s twelve anonymous lanes defeat the mirror

> **Done.** `Stamp` has 29 named members and no interior padding — the same 192 bytes.
> Four that are integral by construction (`rect_origin`, `cell_anchor`, `cell_px`,
> `bleed_reach`) say so in the type, which removes an `i32(...)` truncation from every
> read. Twenty-six of the thirty accessors were pure renames and are gone; the four
> that remain are arithmetic. `TileXform`'s milder version of the same thing is
> untouched.

`dynamics.wesl` declares `struct Stamp` as twelve bare `vec4<f32>` named `a`–`l`. The
generator faithfully produces `pub a: [f32; 4]`, so the Rust side carries no semantics,
and the lane map exists **twice, by hand**:

- the ~30 accessor functions under *Reading the slot* in `dynamics.wesl`
- `Slot::pack` in `gpu/stroke/dynamics/plan.rs`

The compiler checks the *size*; nothing checks the *meaning*. Swapping `e.y` and `e.z`
between the two sides compiles clean, and the generated `offset_of` assertions cannot
see it.

The stated reason — "packed `vec4`s because that is what a uniform wants" — is not a
WGSL rule, and the crate already disproves it. `filter_common.wesl`'s `Filter` declares
`kind: u32, strength: f32, clip: u32, disp: vec2<f32>, params: vec4<f32>, …` — mixed
scalars, mirrored correctly with generated padding and per-field assertions. The uniform
layout constraint that forces 16-byte rounding applies to array element strides and to
the struct's own alignment, not to scalar members.

**Fix.** Declare `Stamp` with named members and let the generator lay them out. The
shader's accessors collapse to field reads; `pack()` becomes a struct literal whose
field names *are* the contract, checked by the compiler on both sides. `TileXform` in
`stamp_common.wesl` has the same problem in milder form (`paint.x` is the drain,
`paint.y` the weave uv scale), as do the `extra: vec4` and `geom: vec3` vertex lanes in
`stamp.wesl`, which the generator mirrors into `SegmentInstance`.

192 bytes written once per slot: the packing buys nothing that the naming costs.

## 5. `deposit` and `deposit_coarse` are held together by a comment

> **Done.** The shared tail is `lay_parcel`, over three bundles — `Under` (the three
> snapshot texels), `ToolMean` (the front half's whole output) and `Canvas` (the state
> the transfer acts on, plus the bleed's verdict). The two kernels are now what the
> comment claimed: the same tail reached by two front halves, one differencing the bake
> and one blending four cell centres. `dynamics.wesl` lost ~90 lines.

`dynamics.wesl` says it outright, above `deposit_coarse`:

> Everything else — the frame, the tooth, the selection, the arc, the jitter, the parcel
> algebra, the guard, the stores — is line for line the same, and must stay so: the two
> kernels are one model at two resolutions, and a divergence between them is a
> `preview == committed` bug waiting for the radius that flips the host's choice.

They genuinely are in sync today; the `inv_e` / `inv_pre` substitution cancels correctly
(`cell_hoist` stores `wg · d_load`, so `tool.y / (dpre · wg)` is `deposit`'s
`d_load.x / dpre`). But ~100 lines of identical tail are enforced by prose, on the one
kernel where a divergence stays invisible until a specific brush radius flips the host's
choice of path.

**Fix.** Extract the shared tail — tooth, selection, `exchange_at`, snapshot loads, the
lift, arc/drain/jitter, the parcel algebra, the rewrite guard and the stores — into one
function taking a small `struct ToolMean { h, m, lat, @if(resid) res }`. The two kernels
then differ only in their front half: the exact one differences the bake and may bleed,
the coarse one blends its cells. That is precisely what the comment claims the model is,
said structurally.

## 6. The group-0 index space is partitioned by comment

> **Done.** `build.rs::bindings_do_not_collide` checks the **linked** artifact, where
> the answer is, and reports the two *files*. Only cross-module collisions are a fault:
> `transform.wesl` deliberately puts `Quad` and `Gated` at one slot because no entry
> point reaches both, and one file can state that about itself where two splitting a
> group cannot. (The check found that case on its first run, which is how the
> distinction got drawn.)

`mixbox_lut.wesl` states the problem better than a reviewer can:

> **Bindings 5 and 6 of group 0, which is a claim on someone else's group.** … The three
> files have to agree on where each stops and the next starts, and nothing but this
> comment says so. If `blend_common` ever grows a sixth binding, it collides here, and
> the error will name a mangled identifier rather than any of the files.

The same arrangement holds for `media_common` (0–6) / `media_mixbox` (7) and
`filter_common` (0–3) / `mixbox_lut` (5–6) / `filter_mixbox` (7).

**Fix.** Once §6.10 is exhaustive (finding 2), have the generator assert no
`(group, index)` collision across a root's transitive import set. That turns the failure
into a build error naming both files, instead of a naga validation error naming
`package__1mixbox_lut_pigment_lut`. A bind group per shared module is the stronger fix
but costs a group slot, and probably is not worth it.

## 7. Performance: the bleed ladder recomputes its neighbours ~36× over

> **The per-texel wins are done; the scratch pass is open.** `segment_look` is built
> once per texel instead of three times, and once per bleed neighbour instead of twice
> — the ladder evaluates it 36 times a texel, so that alone halves its frame work.
> `prefix_span` replaces the two `prefix_at` calls every swept-depth reading used to
> make, so the lateral guard, `textureNumLayers` and the whole orientation-slice
> selection happen once for a difference instead of twice; it also differences *inside*
> the slice mix, which is the better-conditioned order on the one lookup this engine
> keeps in R32Float precisely because that subtraction is where the digits go.
> `f16_nearest4` is branch-free across all four channels.
>
> **What is left** is the `w_n` scratch pass — one thread per region texel writing the
> neighbour blend the ladder reads 36 times. Still the right shape, and it would make
> the pair antisymmetry structural rather than an argument about two threads deriving
> identical floats.

`dynamics.wesl`'s bleed block runs 9 rungs × 4 offsets, and each tap calls
`segment_frame(rtn)`, `outside_sweep(swn)` and `swept_pre(swn)`. That is two `prefix_at`
per tap at 4 `textureLoad` each — **~290 prefix taps per bleed texel** — and every `w_n`
is recomputed independently by up to 36 neighbouring threads.

The fix pattern is already in this file: `cell_hoist`. One thread per region texel
writing `w = 1 − exp(−k_bleed · dpre · warp · gain)` into an R32Float scratch turns the
ladder into 36 plain loads. It also makes the pair antisymmetry **structural** — both
threads of a pair read the same texel — rather than an argument about two threads
deriving bit-identical floats from one uniform.

Smaller ones in the same kernel:

- **`segment_look` runs three times per `deposit` texel** — inside `outside_sweep`,
  inside `swept_pre`, and again for `lk`. Compute once, thread it through. Twice per tap
  in the bleed loop, on top of the above.
- **`swept_pre` calls `prefix_at` twice with the same `yl` and `w`.** A fused
  `prefix_span(x1, x0, yl, w)` computes the lateral taps and the layer selection once.
  This is the file's own "hottest lookup the dynamics loop has".
- **`f16_nearest4` is four scalar calls with two branches each**, on every store of the
  hottest kernel. It vectorizes cleanly — `bitcast<vec4<u32>>`, vector shifts, `select`
  on `vec4<bool>` — branch-free and roughly a quarter of the instructions.

Bracket any of these against `cargo bench -p stark-core --bench stroke`.

## 8. The `@if(resid)` tax is 128 sites

128 gated lines across 10 modules, 66 of them in `dynamics.wesl`. Every color statement
is written twice.

The *variant* design is right — Oklab genuinely should not pay eight bytes per texel of
zeroes on the default space's hot path — but the source duplication is avoidable. A
`struct Latent { lat: vec3<f32>, @if(resid) res: vec3<f32> }` with `latent_blend`,
`latent_store` and `latent_from_premult` helpers moves the gate from the call site into
the operation. `@if` on a struct member already works: `composite.wesl`'s `FsOut` uses it
for `@location(2)`.

That should take `dynamics.wesl` from 66 gated lines to roughly 15, and it serves
finding 5 directly — the extracted tail then needs the gate in one place rather than
five.

Prototype on `integrate.wesl` (10 sites) before touching `dynamics`.

## 9. Smaller items

> **Three done.** `binding_of` is deleted with `Binding::lookup`. The nine copied
> accessor paragraphs went with the `resid`-taking accessors' rewrite. And build-time
> validation is now a real typecheck — see the note at the top, which is the one item
> here that turned out not to be hypothetical.
>
> **Two open.** `dynamics.wesl` is down to ~2130 lines but still holds seven entry
> points; the `*_common` split is unstarted. `EXCHANGE_STEPS` is still a loop that
> cannot iterate twice.

- **`dynamics.wesl` is 2228 lines with 7 entry points.** The `*_common` pattern already
  solves this: the `Stamp` struct, the bindings, the accessors and the kernels move to
  `dynamics_common.wesl` and the entry points stay in `dynamics.wesl`. Same artifact,
  same `ShaderModule`, a root that reads in one screen.
- **`EXCHANGE_STEPS: u32 = 1u`** is a loop that cannot iterate twice and a divide by 1.
  The comment argues the case well, but by *do not add inert scaffolding* it is a knob
  that cannot change a pixel. Either delete the loop and keep the note as prose, or keep
  it as a knowing exception.
- **`binding_of` is generated and called from nowhere.** `Binding::lookup` is what every
  caller uses. Two spellings of one question; drop the generated one.
- **Nine accessors in `src/lib.rs` carry a byte-identical three-line paragraph**
  ("Takes `resid` because this pass carries a tile's color…"). Once `RESID_ENTRY_POINTS`
  is the source of the accessors, that sentence can be generated rather than copied nine
  times.
- **Build-time validation is name-resolution only.** `wesl::Wesl::new` sets
  `validate: true`, but wesl 0.4's validator checks undefined symbols, call arity and
  cycles — it does **not** typecheck. The first real typecheck is naga at pipeline
  creation, i.e. at test time, and CI is `workflow_dispatch`-only. If that ever bites,
  `wesl`'s `to_naga` behind a build-dependency feature moves it to `cargo build`.
- **The Mixbox GLSL transpile is textual** (`replace("float ", "let ")`,
  `strip_unary_plus`) over vendored source that a submodule bump can reformat. It is
  self-guarding in practice — a botched transform yields WESL that does not parse — and
  `tests/blend.rs::black_is_the_identity_through_the_round_trip` pins the polynomial and
  its inverse LUT end to end through a real render. Noted only so the coupling is on the
  record.

## What is working, and should not be disturbed

- **The `lib/` rule** — a module there may not declare a binding — is what let nine
  copies of `fullscreen_clip` and two hand-rolled bilinear taps collapse. It is the
  single best structural decision in the crate.
- **The generated `offset_of` / `size_of` / `align_of` assertions.** They are why a
  layout change can be made at all without a golden re-bless: a mistake stops the build
  at the struct it got wrong.
- **Reading from the *unlinked* source.** Mangling, per-artifact duplication and dead-code
  stripping all make the linked WGSL the wrong input, and the comments the generator
  carries over exist only before linking.
- **`OUT_DIR` as the only write target**, so no input's fingerprint is entangled with the
  script's own output. The stale-half failure that guards against is exactly the kind
  that discredits whatever you happened to be changing at the time.
- **The two-form rule** — additive in τ, or `1 − exp(−k·τ)` — and the complementary-share
  `Exchange`. Both are stated once and evaluated on both sides of every transfer, which
  is why conservation is a property rather than a test.
