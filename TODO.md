* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Avoid calling `build_gpu` to change environment.
* `Engine::apply_ctx` does a _lot_ of cloning.
* De-duplicate brushes in save file (flyweight pattern?).
* Make the paper color selectable.
* Networked multi-user editing.
* Knob to add less or more paint with wet (allow pure drag or pure bleed).

Time for some more UI work. With dry brush dynamics, I want to change how we present the "Add", "Lift", and "Deposit" knobs. They way they work, instead of scaling all of them up or down, you'd just change "Rate", so it's overactuated. I'd like fix this by presenting it as a triangle with pure "Add", pure "Lift" and pure "Deposit" at each vertex. The user can drag a point around to get different combinations. This effectively normalizes them to sum to one.

Let's also make this a reusable component because the next step will be to add a knob to control how much paint is added with the wet brush, and it will get the same treatment.