* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* Reorderable and hidable panels (show from menu).
* `Engine::apply_ctx` does a _lot_ of cloning.
* BUG: Smear loses paint because it hits the `RESERVOIR_CAPACITY` cap. But the math in the shader currently breaks down if it's higher than 1.0.
