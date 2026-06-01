* Stroke smoothing/interpolation.
* Canvas texture image.
* Seams between tiles.
* SVG noise textures for panel backgrounds.
* Oklab color picker.
* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing?
* Brush editor similar to Procreate.
* Lighting panel.

One issue I'm seeing that makes the paint strokes looks less natural is that with a uniform application of paint, the bumps of the surface are almost perfectly reproduced on the surface of the paint instead of only where the paint is thin. One way to fix this would be with diffusion, but I have an idea for how to fix it that I think will work and also help support other features in the future.

Change the interpretation of height paint height to be the actual height, not the thickness of the paint on top of the surface. Then in the media shader, compute the paint thickness as `paint_height - surface_height` and use that to modulate opacity.

Why I think that will make future changes easier:
* It let's us change the surface texture of a document without altering the fully painting surface.
* It simplifies fluid advection and diffusion calculations and subtractive strokes.