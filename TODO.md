* Oklab color picker.
* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* Support changing surface without resetting the document.
* Reorderable and hidable panels (show from menu).

---

Next up is another unification. I want to unify the dry, knife, and mixer dynamics, parameterizing it by:
* How much paint is moved. This is roughly equivalent to the current "pickup" in the mixer dynamics and "carry" in the knife dynamics.
* How much paint is removed. I think currently this is currently called "bite".
* How much paint is added. I think this is roughly equivalent to the current "mix" in the mixer dynamics and "load" in the knife dynamics.

This unification will let us vary between erasing, smearing, painting, and everything in between. The current dry brush is simply no paint moved or removed, only added.