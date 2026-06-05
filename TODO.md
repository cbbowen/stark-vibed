* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* Reorderable and hidable panels (show from menu).
* `Engine::apply_ctx` does a _lot_ of cloning.
* BUG: Smear loses paint because it hits the `RESERVOIR_CAPACITY` cap. But the math in the shader currently breaks down if it's higher than 1.0.

* Wet dynamics runs out of memory in Firefox and the instance crashes in Chromium, which I suspect is the same issue. Look into whether resources are not being cleaned up correctly. It may be useful to log the high water mark for allocated tiles (per format). If necessary, we could even track the source of each allocation by introducing an enum.
