# Blend Modes

Every mode is ordinary addition conjugated by a curve, `f(a,b) = T(T⁻¹(a) + T⁻¹(b))`,
evaluated in CIE XYZ normalized to the display white.

## Tonemapping-derived screen

In XYZ color space:
* Reinhard: f(a,b) = (a + b - 2ab) / (1 - ab)
* Drago: f(a,b) = k log(e^(a/k) + e^(b/k) - 1)

Identity: black. `T` is the tonemap; what adds is light.

## Subtractive

* Multiply: f(a,b) = ab

The same construction with `T(x) = e^(-x)`. What adds is optical density, so this is
Beer-Lambert — stacked glazes — and the identity is white, with black an annihilator.
Screen is this conjugated again by `x -> 1-x`, and that second step is the one with no
physical referent; hence multiply and not Screen.
