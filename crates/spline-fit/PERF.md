# Performance Notes

> **Removed 2026-07-25, when this crate was vendored into Stark.** Stark refits a small
> *live* window of points per update (tens, not thousands — see `IncrementalFit`), so the
> work aimed at large `n` was cost without benefit, and one of the three traded away
> exactness. Gone: **item 10** (`parallel`/rayon), **items 15–16** (`simd.rs` and its
> `f32` root-finder), and **items 12/17's approximate half** — the FastDTW-style skeleton
> E-step (`best_ordered_assignment_approx` and the `*_approx`/`*_exact` fit variants),
> which now leaves `fit_monotonic` exact. The notes below are kept as the record of what
> was measured and what was tried; read them as history, not as a description of the code.
> What survives and is still load-bearing: item 11 (Descartes root isolation), item 13/18
> (geometric range pruning — the spline keeps growing while the live window does not, so
> this is what stops each update from root-finding over the whole stroke), item 17's
> banded DP, and item 9 (SQUAREM), none of which cost accuracy.

Findings from profiling and benchmarking (criterion, `benches/fit.rs`), recorded 2026-07-03.
Reference timings at that date, after the `Poly` → `SVector` change (`033684a`):
`evaluate` 76.6 ns/call, `locally_closest_points` 3.3 µs (21 spans),
`best_ordered_assignment/200` 2.12 ms, `fit_monotonic/200` 193.6 ms,
`fit_monotonic_adaptive/100` 688.6 ms.

## Done

0. **`Poly` as `Vec` → `nalgebra::SVector`** (`033684a`). Coefficient counts are
   compile-time constants (cubic spans: 4; distance-derivative quintic: 6), so all
   polynomial machinery — including `to_bernstein`/`split_half` inside the root-finder
   recursion, which allocated two `Vec`s per subdivision node — is stack-only.
   Measured: `locally_closest_points` −90% (33 µs → 3.3 µs),
   `best_ordered_assignment/50` −88%.

1. **EM loop always hit its 100-iteration cap.** `fit_monotonic` cost ≈ 91–113× one
   E-step at every benchmark size; instrumentation confirmed `iters=100` always. The
   error trace shows ~90% of the reduction in the first ~20 iterations, then a slow
   tail (~0.2–0.3%/iteration at iteration 100) that never approaches the old
   `sqrt(eps)` relative threshold. Fix: `rel_tol` parameter on the fitting functions —
   stop when an iteration improves the error by ≤ `rel_tol * err` (plus a `sqrt(eps)`
   absolute floor preserving exact-fit behavior). Speed/quality knob: 0 reproduces the
   old max-quality behavior; ~1e-3 is a practical default.
   Measured (with items 4–5, benches at `rel_tol = 1e-3`): `fit_monotonic`
   −18…−45%, `fit_monotonic_adaptive` −35…−41% (e.g. adaptive/100 688.6 → 416.7 ms).

3. **`evaluate` recomputed the Cox–de Boor basis matrix per call** (most of the
   76 ns; the matrix depends only on the degree, not the knots). Fix: `evaluate_batch`,
   which computes the basis once for a whole batch of parameters; internal hot paths
   instead use the span-polynomial cache of item 4. Single-shot `evaluate` keeps the
   recompute — caching in the struct wasn't worth the type-level plumbing.
   Measured: 19.7 ns/eval batched vs 77 ns single (~3.9×).

4. **`locally_closest_points` rebuilt per-span polynomials for every query point.**
   The span polynomials depend on the point only through the constant term:
   f = C′·C − Σ_d p_d·C′_d. Fix: precompute per span (C_d, C′_d, C′·C) once per
   spline state (`span_polys`), share across all n points of an E-step, and derive
   f, distances, and sign samples from the cache. Also removed the per-span `Vec`
   from `roots_in_unit_interval` (append-into-buffer API).

5. **DP allocation churn in `best_ordered_assignment`.** Fresh `next`/`parent`/
   `prefix_best` vectors per point; fixed by reusing buffers across the DP sweep and
   storing parents in one flat matrix. (Grid near-duplicate merging turned out to be a
   non-issue: within-point duplicates are already merged in `locally_closest_points`,
   and cross-point candidates are genuinely distinct.)
   Measured (items 4+5 together): `best_ordered_assignment` −13…−26%
   (e.g. /200 2.12 → 1.70 ms).

8. **Post-pass re-ran the full closest-point search for every singleton block.**
   Profiling `fit_monotonic/200` (recorded 2026-07-03) split the run as: E-step
   ≈99.5% of the fit (M-step just 0.4ms over 76 EM iterations), and within one E-step
   the three sub-phases were comparable — candidate-gen 34ms, DP 26ms, tied-run
   post-pass 35ms. The post-pass was suspiciously as expensive as candidate-gen even
   on near-monotonic data: for each *singleton* block it called
   `locally_closest_in(&centroid)`, but a singleton's centroid *is* its point, so this
   recomputed exactly the `candidates[i]` list already produced in the candidate phase.
   Fix: reuse `candidates[i]` for singleton blocks; only genuine multi-point (tied) runs
   re-search. Bit-identical output (errors unchanged to all digits). Post-pass 35 → 1.4ms.
   Measured (criterion, vs prior baseline): `best_ordered_assignment` −35…−59%,
   `fit_monotonic` −36…−46%, `fit_monotonic_adaptive` −39…−42%
   (e.g. `fit_monotonic/200` 88 → 56ms).

9. **SQUAREM acceleration of the EM loop.** After item 8 the fit cost is
   `EM iterations × E-step`, and the dominant lever was the **iteration count**:
   `fit_monotonic/200` ran 76 EM iterations at `rel_tol = 1e-3`. The error trace drops
   5.0 → 0.26 in the first ~10 iterations, then crawls 0.26 → 0.13 over the next ~66 at
   ~1%/iteration — a genuine slow *linear* tail (hard-assignment coordinate descent /
   ICP), not dithering. Fix: SQUAREM (Varadhan & Roland 2008) on the control-point
   matrix `K` (the smoothly evolving parameter; the assignment `ts` is discrete/tied and
   a poor extrapolation target). Each cycle takes two ordinary EM steps `K₀→K₁→K₂`, then
   jumps to `Kₐ = K₀ − 2α r + α² v` (`r = K₁−K₀`, `v = K₂−2K₁+K₀`, S3 steplength
   `α = −‖r‖/‖v‖` clamped to `[−1e6, −1]`; `α = −1` reproduces `K₂`). The objective is
   the E-step error `L(K)`, kept monotone by the M-step's proximal ridge
   (`L(K₂) ≤ L(K₁) ≤ L(K₀)`); `Kₐ` is accepted only when `L(Kₐ) ≤ L(K₂)`, so acceleration
   never loses to plain EM. Costs 3 E-steps/cycle but leaps far along the tail.
   Implemented by extracting `m_step` (M-step made independent of `self.control_points`,
   taking an explicit proximal `prior`) so the E-step objective can be evaluated at any
   candidate `K`. Measured (criterion, vs the item-8 baseline): `fit_monotonic` −29…−42%,
   `fit_monotonic_adaptive` −31…−35% (adaptive/20 unchanged — few EM iterations there, so
   the 3-evals/cycle overhead breaks even). Combined with item 8: `fit_monotonic` 2.3–3.4×
   overall (e.g. /200 88 → 39ms, /100 25.6 → 9.6ms). Quality *improves* slightly at fixed
   `rel_tol` (the big jumps land deeper before the relative-improvement stop trips, e.g.
   /200 err 1.676 → 1.628); exact recovery still reaches ~0.

10. **Parallel candidate-gen (`parallel` feature, rayon).** The E-step's per-point
   closest-point searches are independent (they share only the immutable `span_polys`
   cache), so they farm out to a rayon `par_iter` behind an off-by-default `parallel`
   feature; output is bit-identical (order-preserving collect, 39 tests pass under both
   configs). Two lessons from tuning:
   - **Curve evaluation must stay serial.** Parallelizing the grid→`curve` map was
     net-*negative* even at large grids (e.g. `best_ordered_assignment/50` +200%): each
     grid point is only a few poly evals, far too fine-grained for a thread hand-off.
   - **Coarsen the tasks.** With rayon's default one-item-at-a-time splitting the
     candidate-gen *regressed* mid sizes (`fit_monotonic/100` +19%) and barely helped at
     n=200 (+8%) — the per-point search is only ~2µs and the region runs ~10×/fit, so
     scheduling overhead dominated. `.with_min_len(32)` (≥32 points/task) flipped this:
     no regression anywhere, and `fit_monotonic/100` −20%, `/200` −24%.
   The win **peaks at n≈100–200 and fades at large n** (`/500` −1%, `/1000` −5%):
   candidate-gen is O(n·spans) (linear at fixed m) but the *sequential* DP is O(n·grid)
   ≈ O(n²), so as n grows the un-parallelizable DP dominates the E-step. Confirms the DP
   is the real ceiling — the motivation for the windowed-DP / shrunk-grid work (items in
   "Remaining hotspots"). Gates: 32 cores; `PARALLEL_THRESHOLD = 64` (serial below),
   `PARALLEL_MIN_LEN = 32`.

11. **Recursive-Descartes root isolation in the root-finder** (`poly.rs`). Re-profiling
   (2026-07-04, below) overturned the old candidate-gen/DP split: the E-step is
   dominated by the per-span **root-finder**, and the old `subdivide` (recursive de
   Casteljau, bisecting to `tol`) was the bulk of it. Instrumentation on
   `fit_monotonic/200`: 53% of (span, point) pairs have a Bernstein control polygon that
   straddles zero and so recurse, averaging ~21 subdivision nodes each — but **67.7% of
   those have exactly one Bernstein sign change.** By Descartes' rule of signs in the
   Bernstein basis, one sign change (endpoints strictly nonzero) proves exactly one
   simple root, bracketed by the endpoints.
   Replaced `subdivide`+polish with an `isolate` recursion driven by the sign-change
   count: all-one-sign ⇒ no root (discard); one sign change with straddling endpoints ⇒
   solve directly with a safeguarded (bisection-guarded) Newton, *no* subdivision; else
   split and recurse, isolating each root into its own subinterval before Newton finishes
   it. Only genuinely clustered / even-multiplicity roots that never isolate fall to the
   `tol`-width guard (a polished midpoint). This is **exact** — bit-accurate to tolerance,
   all 40 tests pass, no quality change (unlike the deferred blunt depth-cap, item 7,
   which trades multiple-root accuracy) — and it unifies three code paths into one.
   Landed in two steps: (11a) the single-root fast path with subdivision fallback, then
   (11b) generalizing the fast path recursively so the ≥2-sign-change spans isolate their
   roots too instead of bisecting to `tol`. Measured (vs the item-10 baseline):
   `locally_closest_points` −26%, `best_ordered_assignment/200` −31%, `fit_monotonic`
   −28…−37% (e.g. /100 9.9 → 6.6ms, /200 39.6 → 28.6ms). 11b added ~3–6% over 11a alone
   (more at larger n, where multi-root spans accumulate). Slightly smaller at n=200
   overall because the DP (which doesn't shrink) is a larger share there.

13. **Geometric range pruning of candidate generation.** Since candidate-gen (the
   per-span root-finder) is ~93% of the E-step and root-finds *every* span for *every*
   point, the win is to root-find fewer spans. Instrumentation (`fit_monotonic`, n=200,
   11 spans) showed the assignment only *lands* each point ~1 span past its predecessor
   (~10% of the eager span-scans), but exactness forbids a naive forward scan: 108/200
   points have their *best* minimum in a later lobe than their first, so the search must
   look past the first minimum. The safe lever is a per-span **bounding-box distance
   lower bound** `lb(k, p) = dist(p, bbox of span k's control points)²` (valid by the
   convex-hull property). For point `i` in span `k`, `Σ_{j<i} minlb_j(0) + lb(k,p_i) +
   Σ_{j>i} minlb_j(k)` (with `minlb_j(k)=min_{k'≥k} lb`, respecting monotonicity) lower-
   bounds the total error of any assignment placing `i` there; if it exceeds a feasible
   incumbent (a cheap greedy assignment built from `next_locally_closest_point` — the
   lazy primitive of item 12's aftermath), span `k` can host no *optimal* placement of
   `i` and is skipped. Surviving spans form a contiguous range searched by a new
   `locally_closest_in_range`; minima inside it are classified identically to the full
   search (sign is constant between consecutive critical points; the range ends bracket
   the same intervals), and a beneficial minimum can never sit on an interior cut (that
   would make the adjacent pruned span beneficial too). **Exact** — a 400-case random
   differential test (`bounded_candidate_pruning_matches_full_search`) confirms bit-
   identical assignment and error vs the unpruned search; the pruning provably drops only
   candidates no optimum uses. Enabled by `first_root_after`/`next_locally_closest_point`
   (the "next root is cheaper than all roots" primitives). Measured (vs the pre-bound
   baseline, serial): `best_ordered_assignment/100` −25%, `/200` −34%; `fit_monotonic/50`
   −16%, `/100` −23%, `/200` −30% (29.2 → 20.4ms). The win grows with n (more spans to
   prune per point); `/20` is ~−3% (few EM iterations amortize the O(n·spans) precompute).
   Note this refines candidate-gen, not the DP — consistent with the profile correction
   below that the DP is only ~7%.

14. **Clamped-endpoint root-finder thrash** (`poly.rs`). Uncovered while profiling the
   SIMD prototype (item 15): one span dominated candidate generation. The clamped first
   span's triple-coincident knot makes the curve's tangent vanish *exactly* at the domain
   start (`|C'(0)| = 0`), so `f = C'·(C − p)` is identically zero at `u = 0` for **every**
   query point. That exact `bern[0] = 0` fails `isolate`'s `ends_bracket` test (which
   requires `bern[0] ≠ 0`), so instead of solving the endpoint root the recursion
   subdivided down the interval's left spine to the depth cap — ~50 fruitless splits per
   point, making span 0 ~10× every other span and ~47% of an *unpruned* candidate search.
   (The last span's tangent also vanishes but its value lands at ~1e-15, not exactly 0, so
   it squeaks through the fast path and does not thrash.) Fix: Descartes' rule already says
   `sign_changes == 0 ⇒ no root strictly inside`, so when that holds, record any *exactly*
   zero endpoint as a root and return, instead of subdividing toward it. Exact (an exact
   endpoint zero is a genuine root; the de-Casteljau cluster boundaries that must *not*
   trigger this are only ever tiny-but-nonzero, so the `== 0` test discriminates cleanly),
   and it also drops the spurious `polish`-midpoint roots the old path emitted for
   interior touch-zero polygons. All 40 root-finder tests pass unchanged. Measured
   (criterion, `-C target-cpu=native`): unpruned candidate-gen (`all_critical_points`)
   −31…−33%; **real fit `fit_monotonic` −9…−13.5%** (e.g. /200 20.55 → 18.78ms, /20 496 →
   429µs) — pure scalar, no SIMD. Smaller than the candidate-gen delta because the E-step
   already geometrically prunes (item 13) most span-0 searches away.

15. **SIMD candidate generation** (`simd.rs`, `simd` feature, nightly `portable_simd`;
   off by default). Concrete-`f64` kernel that vectorizes the per-span root-finder across
   *points* (SoA, one point per lane, AVX2 `f64` width 4). Per span it batches: the
   Bernstein discard test (~40% of pairs, no root), a lockstep safeguarded Newton for the
   single-sign-change lanes (the common single-root case), and a per-lane scalar fallback
   for the multi-root minority. Reproduces the scalar `all_critical_points` to root-finding
   tolerance (a conservative near-zero guard routes fp-fragile classifications to the
   authoritative scalar path). Findings: the pure fast path runs ~3.8× (near the 4× lane
   width); `L = 8` bought nothing (no AVX-512 on the test box — two AVX2 ops); after item
   14 removed the span-0 fallback, the kernel is **1.5×** on unpruned candidate-gen
   (`all_critical_points/200` 116µs scalar → 77µs SIMD).

16. **Wiring the SIMD kernel into the pruned E-step.** `best_ordered_assignment`'s
   candidate generation was refactored to a single seam (`candidates_in_ranges`), which
   dispatches to a batched f64 generator (`candidates_in_ranges_simd`) when the `simd`
   feature is on and `T = f64` (via `TypeId` + a checked reinterpret; scalar path for every
   other `T`). The generator is span-major: for each span it root-finds — batched across
   lanes — exactly the points whose pruned range (item 13) covers it (contiguous, so they
   batch coherently), accumulates each point's all-span `f_scale`, then runs the shared
   scalar minima classification (`minima_from_roots`, split out of `locally_closest_in_range`
   so both paths reuse it). A differential test (`simd_candidates_match_scalar_in_ranges`)
   confirms the SIMD candidates match the scalar ones on full and windowed ranges.
   Measured (criterion, `-C target-cpu=native`, SIMD vs scalar both *with* item 14):
   `fit_monotonic` **−5…−6%** (/200 20.21 → 19.03ms, /100 −6.2%). Much smaller than the
   kernel's 1.5× because pruning + item 14 already shrank candidate root-finding to a
   modest slice of the fit; the remainder (the scalar minima classification, the DP, and
   the M-step/SQUAREM) is untouched — vectorizing `f_scale` too moved nothing. So SIMD adds
   an exact ~5% on top of item 14's ~10%; the exact scalar fix remains the larger win.

17. **Exact banded DP** (`assignment_from_candidates`). After the 2026-07-05 profile
   correction below, the DP core is the biggest E-step phase (33% at n=200) and the
   worst-scaling — O(n·grid) ≈ O(n²), as SIMD + pruning shrank the root-finder out of the
   top spot. The DP's per-point pin/prefix sweeps ran over the *whole* grid, but
   `search_ranges` already proves point `i`'s optimal parameter lies in its pruned span
   window `ranges[i] = [lo_i, hi_i]` (no optimal assignment leaves it — the same guarantee
   candidate-gen relies on). So each point's DP states can be confined to the grid indices
   in that window (its "band"): the pin loop, the prefix-best scan, and the parent row all
   shrink from `grid.len()` to the band width, turning the sweep into ≈ O(n·band). Bands
   are narrow (the pruned ranges span a few knots), and the parent matrix went from
   `(n−1)·grid` to band-sized rows (time *and* space). **Exact** — every optimal path stays
   within the bands, so only unreachable/dominated out-of-window states are skipped; the
   400-case differential test now runs the banded (pruned) DP against the plain full-grid
   DP (`ranges = (0, num_spans)` disables banding) and confirms bit-identical assignment and
   error. Threaded `ranges` through `assignment_from_candidates`. Measured (criterion,
   `-C target-cpu=native`, SIMD+parallel, vs the item-16 baseline): the win **grows with n**
   exactly as the DP's O(n²) share predicts — `best_ordered_assignment` −3.0% (n=20) →
   −8.4% (50) → −11.9% (100) → **−14.2% (200)**; `fit_monotonic` −2.1% → −4.6% → −5.5% →
   **−11.5%** (/200 7.44 → 6.59ms). This is the follow-through the rejected item 12's banded
   DP never got to deliver on its own (item 12 bundled it with an *inexact* windowed E-step
   and was written off when the DP was thought to be ~7%).

18. **Cheaper pruning precompute** (`search_ranges`). The 2026-07-05 profile put the
   pruning precompute at ~24% of the E-step, dominated by the incumbent greedy (17%): a
   full *serial, scalar* closest-point pass whose per-point `next_locally_closest_in`
   allocated a fresh `Vec<Poly<T,6>>` over **all** spans and computed an exact all-span
   `f_scale` on every call (O(n·spans) builds + n heap allocations). Two exact changes:
   (a) `next_locally_closest_in` now builds each span's `f` on demand — the lazy
   critical-point scan only forms `f` for the spans it reaches, and `f_scale` takes its
   exact max over spans transiently, so **no per-call all-span buffer is allocated**;
   (b) `search_ranges` computes each point's suffix-min bounding-box row (`min_lb_row`)
   **once** and shares it between the two passes instead of recomputing it in each. Both
   preserve the exact ranges (the 400-case differential test is unchanged). Measured
   (criterion, `-C target-cpu=native`, SIMD+parallel, controlled A/B vs the item-17
   baseline): `best_ordered_assignment` −2…−6%, `fit_monotonic` −4.0% (n=50) → −5.6%
   (100) → **−7.5% (200)**.
   *Rejected sub-approach (pitfall):* also skipping the all-span `f_scale` scan via a cheap
   per-point upper bound `gmax[c] + Σ_d |p_d|·dcmax[d][c]` (per-span coefficient maxima
   precomputed once). Sound in exact arithmetic — an over-estimate only widens the tiny
   `sign_tol`, and the incumbent is a valid upper bound however its minima classify — but it
   **regressed the real fit** because the benchmark spline is `f32` (eps ≈ 1.2e-7, not
   f64's 2.2e-16): when the curve is far from the points (early EM iterations) the bound
   overestimates the true `f_scale` 10–100× through cancellation in `g − Σ p·dc`, inflating
   `sign_tol` to ~1e-3, which loosened the incumbent, widened the pruned ranges, and did
   more root-finding + wider DP bands every E-step. The standalone
   `best_ordered_assignment` bench (a well-fit spline, points near the curve) *improved*,
   masking it — only the full-fit trajectory exposed the regression. Keep the exact
   `f_scale`; the win is the removed allocation, not a cheaper scale.

## Profile correction (2026-07-04): the root-finder dominates, not the DP

Earlier notes (items 8–10) assumed candidate-gen and the DP were comparable (~34ms vs
~29ms) and pointed at the DP as the ceiling. Direct measurement refuted this. Short-
circuiting all root-finding drops `best_ordered_assignment/200` by **92.5%** and
`locally_closest_points` by 62%; within a candidate search, `subdivide` alone is ~73%.
So the E-step is **~93% candidate-gen / ~7% DP**, and candidate-gen is mostly the
root-finder's recursive subdivision. Consequences:
- The DP is *not* worth optimizing (≤7% ceiling) — see the rejected windowed-DP work.
- The remaining lever is still the root-finder, now largely spent: item 11 handles
  single-root spans (the majority) directly and isolates the multi-root minority via
  recursive Descartes. The only exact refinement left is faster *isolation* of the
  ≥2-sign-change spans (Bézier clipping's convex-hull clip converges quadratically vs the
  midpoint split), but at degree 5 the hull-construction overhead roughly cancels the
  fewer iterations — not worth the complexity.

## Profile correction (2026-07-05): the DP now dominates, not the root-finder

The 2026-07-04 correction above ("~93% candidate-gen / ~7% DP") was measured on the
**scalar, pre-pruning, pre-SIMD** path. Items 13 (range pruning) and 16 (SIMD root-finder)
have since inverted it. Re-profiling `fit_monotonic` **with `--features parallel,simd`,
`-C target-cpu=native`** (sub-phase timers inside `best_ordered_assignment`), per E-step
call:

| phase                                            | n=50   | n=100  | n=200  | share@200 |
|--------------------------------------------------|--------|--------|--------|-----------|
| DP core (`assignment_from_candidates`)           | 12.8µs | 38.8µs | 143µs  | **33%**   |
| candidate root-find + `f_scale` + classify       | ~24µs  | ~49µs  | 96µs   | 22%       |
| incumbent greedy (pruning ceiling)               | 19.7µs | 36.2µs | 73µs   | 17%       |
| pruning passes (`search_ranges`)                 | 8.4µs  | 15.8µs | 32µs   | 7%        |
| tied-run post-pass                               | 4.5µs  | 9.2µs  | 22µs   | 5%        |

Absolute (criterion, same flags): `fit_monotonic/200` ≈ 7.4ms, `/100` ≈ 3.4ms;
`best_ordered_assignment/200` ≈ 337µs; `all_critical_points/simd/200` = 66µs (unpruned
root-finding over *all* spans). `fit_monotonic` ≈ 22 E-steps × E-step; M-step still <0.5%.

The takeaways completely reverse the previous note:
- **The DP core is now the single biggest phase (33% at n=200), not 7%** — and the
  worst-scaling: 12.8 → 38.8 → 143µs is 3.0× then 3.7× per doubling of n, i.e.
  O(n·grid) ≈ **O(n²)**. It dominates increasingly as n grows.
- **The root-finder is now only ~22%**, already SIMD-vectorized and pruned; items 11–16
  optimized it out of the top spot. Further root-finder work has a low ceiling.
- **The geometric pruning precompute (incumbent 17% + passes 7% ≈ 24%) is now a quarter
  of the E-step** — the incumbent greedy runs a full *serial, scalar* closest-point pass
  just to set the pruning ceiling. Pruning is still net-positive (disabling it makes
  `fit_monotonic/200` slower, 7.3 → 10.1ms), but its payoff is now **shrinking the DP
  grid**, not saving root-finding (unpruned SIMD root-finding is only 66µs). It was tuned
  for the scalar path where root-finding dominated.

### Remaining opportunities (ranked)

1. **Exact banded DP** *(implemented — item 17 below)*. Confine each point's DP sweep to
   the grid indices in its already-proven pruned parameter window `ranges[i]` instead of
   the whole grid. Exact (the pruning guarantee already forbids the optimum from leaving
   the window); turns the O(n·grid) sweep into ≈ O(n·band). Biggest lever, grows with n.
2. **Cheapen the pruning precompute (~24%).** *(partly done — item 18.)* The incumbent
   greedy is a full serial scalar pass while the candidate-gen it gates is parallel+SIMD.
   Item 18 removed its per-call all-span `Vec` allocation (lazy `f`) and the duplicated
   `min_lb_row` for −4…−7.5%. Residual: the incumbent still scans all spans to form the
   *exact* `f_scale` (a cheap bound is unsafe at f32 — see item 18), and the greedy is
   still serial. Remaining options: parallelize it, or find a tighter-but-cheap `f_scale`
   that holds at f32.
3. **The `f_scale` loop in `candidates_in_ranges_simd`** is an unconditional
   O(n·spans·D) scalar loop (part of the 22% candidate phase) run for every point at every
   span regardless of pruning — fold into the SIMD kernel or restrict to pruned spans.

## Tried and rejected

12. **Windowed E-step + banded DP** (approaches "2 & 3"; the intended follow-on to the
   parallel candidate-gen). Idea: since the spline moves little between EM iterations,
   let each point search only a window of spans around its previous parameter (#2) and
   confine the DP to a per-point grid band (#3), with an exact-EM "polish" at the end to
   restore correctness. Built it fully (shared `e_step(warm: Option<&[T]>)`, windowed
   candidate search, banded DP, cold→windowed→polish `em_fit`); it was correct (exact
   `warm=None` path bit-identical, all tests pass) but **~60–88% slower** at every size.
   Two fatal problems: (a) the windowed E-step's assignment error was **1.4×–29×** the
   exact one — the "spline barely moves" premise is false (early EM iterations and
   SQUAREM's extrapolation jumps move the control points a lot, and a wiggly curve puts a
   point's true closest in a far lobe a local window can't see); (b) those bad
   correspondences corrupt the M-step, and EM's path-dependence means the exact polish
   converges to a *worse* local optimum (`/200` err 3.32 vs 1.63). Reverted. The later
   profile correction showed the DP was only ~7% of the E-step anyway, so #3 could never
   have paid off regardless. Lesson: the E-step cannot be *approximated* mid-trajectory;
   only exact speedups (item 11) are safe here.

2. **Warm-starting `fit_monotonic_adaptive`** (each fit seeded with knots sampled from
  the previous best curve and the previous assignment rescaled). Measured *slower*
  (+11–13% at n=20/50) *and worse*: cold polyline initialization threads the knots
  through the data and reaches near-zero error in 2–3 EM iterations when m
  approaches n (e.g. n=20, m=16: cold err 0.0018 vs warm 0.050 after 100 capped
  iterations), while the warm init starts in the coarser fit's oversmoothed basin
  and crawls. Adaptive fits stay cold-started.

## Deferred — use a library rather than reinvent

6. **Banded Cholesky in the M-step.** `btb` has bandwidth 3 (7 diagonals) from the
   cubic basis's local support, but is stored dense and factored with dense O(m³)
   Cholesky, plus a clone per ridge attempt. Irrelevant at m≈10; matters for adaptive
   fits where m grows exponentially. Adopt a banded/sparse solver crate.
   (2026-07-03 profiling: the whole M-step — assembly + ridge Cholesky — is 0.4ms over
   76 iterations of `fit_monotonic/200`, i.e. <0.5% of the fit at these sizes; only pursue
   this for large-m adaptive fits.)

7. **Root-finder subdivision depth.** *Done, exactly, via item 11.* The original idea —
   cap Bernstein subdivision depth and finish with Newton — was inexact (Newton is only
   linearly convergent at multiple roots; the `triple_root_reported_once` 1e-5 tolerance
   is the canary). Item 11 achieves the same speedup *without* the accuracy tradeoff by
   isolating on the Bernstein sign-change count instead of a blunt depth cap: single-root
   spans skip subdivision entirely, multi-root spans subdivide only until each root is
   alone, then Newton. Note the earlier "root-finding is ~15–20% of the E-step" estimate
   was wrong — it is the large majority (see the profile correction above).
