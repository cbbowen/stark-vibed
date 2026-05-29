* Shape image.
* Canvas texture image.

* Currently, the next item in the suggested build order is implement an LOD system. However, the motivation for that is zooming and panning, which we haven't implemented yet. So let's revise the plan to implement zoom and pan first then follow up with an LOD system only if it proves necessary. Pan should use middle mouse button drag and zoom should use the scroll wheel, with the zoom intuitively centered such that the cursor continues to point to the same location on the canvas after the zoom.