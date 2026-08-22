// Writing f16 state back without walking it (§6.2).
//
// Under `lib/`, so nothing here declares a binding: the caller passes the value.
//
// **A `textureStore` to an `rgba16float` target is not guaranteed to round.** The
// f32 → binary16 conversion a storage write performs is the backend's, and D3D12's
// (HLSL `f32tof16`) **truncates toward zero**. Render-target writes go through the
// output merger, which is round-to-nearest and safe; storage writes are not, and the
// whole dynamics loop is storage writes.
//
// That is harmless for a value computed once. It is not harmless for a
// **read-modify-write iterated over a stroke**, which is what every pass in
// `dynamics.wesl` is: a texel under a 64 px tip is re-stored by every segment whose
// sweep reaches it — a hundred and more of them on a path with any wiggle in it — and
// each store that lands a hair below the stored value drops a whole f16 ULP, `2⁻¹¹`
// relative. The recompute lands low about half the time by chance (`premult =
// (premult/op)·op`) and *always* when the pass is nearly the identity (`h·keep` with
// `keep` a whisker under 1 is below `h` by far less than an ULP, so truncation takes
// the full ULP every time). Either way the drift is one-directional, so it
// accumulates instead of cancelling: measured at −3.7% of a field's total height over
// one wiggly smear whose rates were `1e−4` — a stroke that physically moves nothing.
// Height goes down and the premultiplied latent goes down with it, which reads as
// paint that is both thinner and darker than the pass that "did nothing" found it.
//
// The fix is to hand the store a value that is **already on the f16 lattice**, so
// whichever way the backend converts, it converts exactly. Then an identity pass is
// an identity store on every adapter — a structural property rather than a guard each
// store site could forget — and a real change rounds to nearest instead of down.
//
// The rewrite guards in `deposit` and `settle` are still worth their branch: they skip
// the *traffic*, not just the rounding. This is what covers the stores that do happen.

// The nearest binary16-representable value to each channel of `v`, ties to even — as
// f32s, since that is what a `textureStore` takes.
//
// The rounding is done on the f32 bit pattern rather than by round-tripping through
// `pack2x16float`, because that builtin's conversion is the same backend one this
// exists to work around.
//
// **Written across the four channels at once, and branch-free.** This runs on every
// store of the loop's hottest kernels — the deposit writes three targets per extent
// texel — and as four scalar calls it was four copies of two branches, on a machine
// whose lanes have to agree about which way each one goes. Every step below is the
// scalar law componentwise: same bit pattern, same tie rule, same treatment of what has
// no lattice point to snap to.
fn f16_nearest4(v: vec4<f32>) -> vec4<f32> {
    let a = abs(v);
    let u = bitcast<vec4<u32>>(v);
    let e = vec4<i32>((u >> vec4<u32>(23u)) & vec4<u32>(0xFFu)) - vec4<i32>(127);
    // How many bits of the f32 significand binary16 cannot keep: 13 throughout its
    // normal range, and one more for each binade below `2⁻¹⁴`, where the subnormals
    // give up a bit at a time. `−1 − e` meets 13 exactly at the normal/subnormal
    // boundary, so the `max` is the whole rule.
    let shift = max(vec4<i32>(13), vec4<i32>(-1) - e);
    // Held at 23 for the arithmetic below. A shift wider than the significand is
    // undefined in WGSL, and the lanes that would want one are exactly the lanes
    // `tiny` answers for — so this keeps the shared arithmetic well-defined rather
    // than deciding anything.
    let sh = vec4<u32>(min(shift, vec4<i32>(23)));
    // Round to nearest, ties to even, by biasing the discarded bits and masking. A
    // significand that carries out lands in the exponent field, which is exactly the
    // increment that case wants.
    let m = vec4<u32>(1u) << sh;
    let lsb = (u >> sh) & vec4<u32>(1u);
    let rounded = bitcast<vec4<f32>>(
        (u + (m >> vec4<u32>(1u)) - vec4<u32>(1u) + lsb) & ~(m - vec4<u32>(1u)),
    );
    // Under half the smallest subnormal step there is nothing left to keep: the
    // neighbours are zero and `2⁻²⁴`, and an exact tie goes to the even one (zero).
    // `v * 0.0` rather than a literal, to carry the sign of a negative latent.
    let tiny = select(sign(v) * 5.9604645e-8, v * 0.0, a <= vec4<f32>(2.9802322e-8));
    let snapped = select(rounded, tiny, shift > vec4<i32>(23));
    // NaN, an infinity, or past binary16's largest finite value: there is no lattice
    // point to snap to, and what the backend does with it is its own business. The
    // NaN case is why the test is a negated `<` — here, a `select` that keeps `v`
    // whenever `a < 65504` is *not* true.
    return select(v, snapped, a < vec4<f32>(65504.0));
}
