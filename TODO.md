* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* `Engine::apply_ctx` does a _lot_ of cloning.
* De-duplicate brushes in save file (flyweight pattern?).

Please update your memory and @DESIGN.md to reflect how the reservoir is actually parameterized (what the x and y coordinates mean).
