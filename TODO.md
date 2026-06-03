* Stroke smoothing/interpolation.
* Oklab color picker.
* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Lighting panel.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* Reorderable and hidable panels (show from menu).

---

Option 1: Simpler but doesn't support digital-style painting as well.

I've identified the problem, and it's far-reaching but imminently solvable. Our current paint representation is denormalized. We have both coverage (alpha) and paint height. But these are not separate things, they're two representations of the same thing. We're hackily reconcile this by scaling opacity by functions of both of them in @media.wgls, but it's an imperfect solution. We're hitting one of those imperfections now with the knife dynamics because we store premultiplied colors.

Here's how I want to fix it:
1. Only store paint height. We'll compute coverage from it in the media shader with the current opacity calculation.
2. Do not premultiply color. This will mean we need to write strokes to temporary tiles and then integrate them into the layer with a compute shader. We would've needed this eventually anyway to support blend modes.

Overall, I think this approach will make it dramatically easier to implement brush dynamics because we don't have to keep coverage and height in sync.

---

Option 2: More complex but more accurate and supports digital-style better.

I've identified the problem, and it's far-reaching but imminently solvable. Our current paint representation is denormalized. We have both coverage (stored in the alpha channel) and paint height. But we currently treat them as two representations of the same thing that we hackily reconcile in @media.wgls. We're hitting one of the limitations of this with the knife dynamics because we store premultiplied colors.

Here's how I want to fix it:
1. Treat the alpha channel not as coverage but as the per-unit-height opacity of the paint at that location. In the media shader, we'll compute the final alpha by combining this opacity with the paint thickness, treating it as a translucent material.
2. Only premultiply color by opacity, not by height. This enables correct blending.
3. To avoid double-applying the coverage of the brush shape, we should scale thickness and opacity by its square root (instead of it's value).
4. This approach may mean we need to write strokes to temporary tiles and then integrate them into the layer with a compute shader. We would've needed this eventually anyway to support blend modes.

Overall, I think this approach will make it dramatically easier to implement brush dynamics because we don't have to keep opacity and height in sync.

---