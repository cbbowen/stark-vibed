* The color reservoir used for brush dynamics describes how the brush color changes across the stroke. However, at a given point in time, it is a single color. It does not capture that one part of the brush might pick up a different color from another part. However, that would be easy to fix by capturing multiple samples at different lateral offsets along the path. Furthermore, we get this almost for free. The mixer compute shader can trivially load these extra samples (and could even do each lateral offset in parallel). We might also need to make the region captured for sampling from slightly larger.
* Stroke smoothing/interpolation.
* Canvas texture image.
* Seams between tiles.
* SVG noise textures for panel backgrounds.
* Oklab color picker.
* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Lighting panel.

One issue I'm seeing that makes the paint strokes looks less natural is that with a uniform application of paint, the bumps of the surface are almost perfectly reproduced on the surface of the paint instead of only where the paint is thin. One way to fix this would be with diffusion, but I have an idea for how to fix it that I think will work and also help support other features in the future.

Change the interpretation of paint height to be the actual height, not the thickness of the paint on top of the surface. Then in the media shader, compute the paint thickness as `paint_height - surface_height` and use that to modulate opacity.

Why I think that will make future changes easier:
* It let's us change the surface texture of a document without altering the fully painted surface.
* It simplifies fluid advection and diffusion calculations and subtractive strokes.

---

I made some updates to @crates/stark-shaders/src/shaders/media_common.wesl , @crates/stark-shaders/src/shaders/media_mixbox.wesl , and @crates/stark-shaders/src/shaders/media_oklab.wesl  to ensure that the media shaders do blending in the correct color space (oklab or mixbox) so we get strokes that fade out beautifully as the paint thins. I updated the test goldens, and they look good, but please check my work. Especially check that we're using `surface_strength` consistently.