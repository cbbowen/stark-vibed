* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* Reorderable and hidable panels (show from menu).
* `Engine::apply_ctx` does a _lot_ of cloning.

With that bugfix out of the way, let's return to the UI improvements. I want to make Panels a unified, first-class UI element. This includes all of the existing panels: Color, Brush, Lighting, and Layers. I want Panels to have a few key capabilities:
1. They can be reordered by dragging the "title bar". This should be accompanied by a pleasant animation where panels shift to their new positions.
2. They can be closed with a button in the top-right of the panel.
3. When closed, they can be shown again via a dedicated menu.
