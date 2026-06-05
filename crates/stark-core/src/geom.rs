//! Canvas geometry: tile coordinates and the pan/zoom view transform.
//!
//! Canvas space is in pixels with x to the right and y downward. Tile `(i, j)`
//! covers the square `[i*TILE_SIZE, (i+1)*TILE_SIZE) × [j*TILE_SIZE, ...)`.
//! The infinite canvas (DESIGN.md §6) is realized by tiles being sparse and
//! addressed by signed integer coordinates.

pub use glam::Vec2;

/// Apron (halo) width in pixels carried around each tile's interior, replicated
/// from the neighboring canvas content (DESIGN.md §6.4). The compositor samples a
/// tile's interior with bilinear filtering; without an apron the filter clamps at
/// the tile edge instead of reaching into the neighbor, leaving a visible seam at
/// every boundary under sub-pixel pan or non-1:1 zoom (the seam is then amplified
/// by the media pass's height→normal gradient). One pixel is all bilinear needs;
/// widen this if a future media effect needs more neighbor context.
pub const TILE_APRON: u32 = 1;

/// Physical edge length of a tile's channel textures: interior plus an apron on
/// every side. Tiles are stored at this size; only the interior is presented.
pub const TILE_TEX: u32 = 256;

/// Edge length of a square tile's *interior*, in canvas pixels (DESIGN.md §6.1).
/// This is the addressing stride: tile `(i, j)` owns canvas
/// `[i*TILE_SIZE, (i+1)*TILE_SIZE)` — aprons (below) overlap neighbors and are
/// not owned.
pub const TILE_SIZE: u32 = TILE_TEX - 2 * TILE_APRON;

/// Maps a tile's interior quad corner (`∈ [0, 1]`) to a UV coordinate in the
/// apron'd texture: `uv = corner * INTERIOR_UV_SCALE + INTERIOR_UV_BIAS`. The
/// compositor and presenter sample only the interior sub-rect; bilinear taps at
/// the interior edge then fall into the apron (neighbor content), not a clamp.
pub const INTERIOR_UV_SCALE: f32 = TILE_SIZE as f32 / TILE_TEX as f32;
pub const INTERIOR_UV_BIAS: f32 = TILE_APRON as f32 / TILE_TEX as f32;

/// Integer address of a tile on the infinite canvas.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Canvas-space position of this tile's top-left corner, in pixels.
    pub fn origin(self) -> Vec2 {
        Vec2::new(
            self.x as f32 * TILE_SIZE as f32,
            self.y as f32 * TILE_SIZE as f32,
        )
    }
}

/// A pixel size (e.g. a render target's dimensions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extent2 {
    pub width: u32,
    pub height: u32,
}

impl Extent2 {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// The pan/zoom transform applied when presenting the canvas to a surface
/// (DESIGN.md §6.4). This is session state and is never historized.
#[derive(Copy, Clone, Debug)]
pub struct ViewTransform {
    /// Canvas-space point shown at the center of the viewport.
    pub center: Vec2,
    /// Zoom factor: screen pixels per canvas pixel (1.0 = 1:1).
    pub zoom: f32,
    /// Size of the target surface, in pixels.
    pub viewport: Extent2,
}

impl ViewTransform {
    /// A 1:1 view centered on the canvas origin for a viewport of `size`.
    pub fn identity(size: Extent2) -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
            viewport: size,
        }
    }

    /// Linear map from canvas pixels to normalized device coordinates:
    /// `ndc = canvas_pos * scale + translate`. Derivation in DESIGN.md §6.4.
    ///
    /// Returns `(scale, translate)`. The y component of `scale` is negative
    /// because canvas y is downward while NDC y is upward.
    pub fn canvas_to_ndc(self) -> (Vec2, Vec2) {
        let vw = self.viewport.width.max(1) as f32;
        let vh = self.viewport.height.max(1) as f32;
        let scale = Vec2::new(2.0 * self.zoom / vw, -2.0 * self.zoom / vh);
        let translate = -self.center * scale;
        (scale, translate)
    }

    /// Inverse of the view: map a screen-pixel position (origin top-left) to a
    /// canvas-space point. Used to turn pointer input into stroke samples.
    pub fn screen_to_canvas(self, screen: Vec2) -> Vec2 {
        let half = Vec2::new(self.viewport.width as f32, self.viewport.height as f32) * 0.5;
        (screen - half) / self.zoom + self.center
    }

    /// Scale the zoom by `factor` while keeping the canvas point under `anchor`
    /// (a screen-pixel position) pinned in place — cursor-anchored zoom.
    pub fn zoom_about(&mut self, anchor: Vec2, factor: f32) {
        let before = self.screen_to_canvas(anchor);
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.screen_to_canvas(anchor);
        self.center += before - after;
    }

    const MIN_ZOOM: f32 = 0.05;
    const MAX_ZOOM: f32 = 64.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_center_maps_to_view_center() {
        let view = ViewTransform {
            center: Vec2::new(123.0, -45.0),
            zoom: 2.0,
            viewport: Extent2::new(800, 600),
        };
        let center_px = Vec2::new(400.0, 300.0);
        let mapped = view.screen_to_canvas(center_px);
        assert!((mapped - view.center).length() < 1e-3, "got {mapped:?}");
    }

    #[test]
    fn zoom_scales_screen_to_canvas_distance() {
        let view = ViewTransform {
            center: Vec2::ZERO,
            zoom: 2.0,
            viewport: Extent2::new(800, 600),
        };
        // 100 screen px right of center is 50 canvas px at 2x zoom.
        let p = view.screen_to_canvas(Vec2::new(500.0, 300.0));
        assert!((p - Vec2::new(50.0, 0.0)).length() < 1e-3, "got {p:?}");
    }

    #[test]
    fn zoom_about_keeps_cursor_point_fixed() {
        let mut view = ViewTransform {
            center: Vec2::new(10.0, 20.0),
            zoom: 1.0,
            viewport: Extent2::new(800, 600),
        };
        let anchor = Vec2::new(620.0, 130.0); // arbitrary off-center cursor
        let canvas_under = view.screen_to_canvas(anchor);
        view.zoom_about(anchor, 2.5);
        let after = view.screen_to_canvas(anchor);
        assert!(
            (after - canvas_under).length() < 1e-3,
            "anchor drifted: {canvas_under:?} -> {after:?}"
        );
        assert!((view.zoom - 2.5).abs() < 1e-4);
    }
}
