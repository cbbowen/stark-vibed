* Shape image.
* Stroke smoothing/interpolation.
* Canvas texture image.
* Seams between tiles.
* SVG noise textures for panel backgrounds.
* Oklab color picker.
* Use WGSL derivatives (`dpdx` and `dpdy`) in @media.wesl instead of manual finite differencing.
* Pigment color space with Kubelka–Munk blending.
  * `trait ColorSpace` converts `Vec4` raw texture values to a from sRGB and specifies a media shader.
* Brush editor similar to Procreate.

Before we dive into `PigmentColorSpace`, there's another task we should look at because it may affect the stamp "interface". There are three problems that I believe we can solve with one change. Specifically, cubic stroke interpolation. The problems it will solve:
* Trying to paint a diagonal stroke currently exhibits stair-step aliasing as the cursor often moves only one pixel left then one pixel up, making the angle entirely horizontal or vertical.
* The discrete stamps are very visible with hard-edges brush shapes. Stroke interpolation would enable us to better approximation continuous stamping (i.e. what we would get if we computed a path integral over the stroke).
* It would will reduce save file size by providing a more compact representation of the stroke.