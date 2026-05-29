//! Step-1 skeleton tests (DESIGN.md §12 build order, step 1): a recycling tile
//! pool, and tiles presented to an offscreen target under a pan/zoom transform.
//!
//! These need a GPU adapter; where none is available (e.g. a headless CI box
//! without a software fallback) the render test skips rather than fails.

use stark_core::geom::{Extent2, TileCoord, Vec2, ViewTransform};
use stark_core::gpu::{GpuContext, Presenter, TilePool};

const RED: wgpu::Color = wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
const GREEN: wgpu::Color = wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
const BLUE: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };

/// Acquire a context or skip the test if the machine has no usable adapter.
fn context_or_skip() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::headless()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

#[test]
fn pool_recycles_dropped_tiles() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx);

    assert_eq!(pool.free_count(), 0, "fresh pool has no recycled tiles");

    let a = pool.acquire();
    let b = pool.acquire();
    assert_eq!(pool.free_count(), 0, "live tiles are not in the free list");

    drop(a);
    assert_eq!(pool.free_count(), 1, "dropping the last handle recycles the tile");
    drop(b);
    assert_eq!(pool.free_count(), 2);

    // A subsequent acquire reuses a recycled texture rather than allocating.
    let _c = pool.acquire();
    assert_eq!(pool.free_count(), 1, "acquire reuses a recycled tile");
}

#[test]
fn presents_tiles_under_view_transform() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx.clone());

    // 512x512 target. With zoom 1 and center (256,256), canvas (0,0) maps to the
    // top-left screen pixel, so tile (0,0) occupies the top-left 256x256 quadrant
    // and tile (1,0) the top-right; the bottom half stays background.
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let size = Extent2::new(512, 512);
    let view = ViewTransform {
        center: Vec2::new(256.0, 256.0),
        zoom: 1.0,
        viewport: size,
    };

    let tiles = vec![
        (TileCoord::new(0, 0), pool.acquire_filled(RED)),
        (TileCoord::new(1, 0), pool.acquire_filled(GREEN)),
    ];

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test target"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut presenter = Presenter::new(&ctx, target_format);
    presenter.render(&target_view, view, BLUE, &tiles);

    let pixels = read_back(&ctx, &target, size);
    let at = |x: u32, y: u32| -> [u8; 4] {
        let row = (size.width * 4) as usize;
        let i = y as usize * row + x as usize * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };

    // Top-left quadrant: tile (0,0) = red.
    assert!(is_near(at(128, 128), [255, 0, 0, 255]), "top-left should be red, got {:?}", at(128, 128));
    // Top-right quadrant: tile (1,0) = green.
    assert!(is_near(at(384, 128), [0, 255, 0, 255]), "top-right should be green, got {:?}", at(384, 128));
    // Bottom half: no tiles, so background blue.
    assert!(is_near(at(256, 384), [0, 0, 255, 255]), "bottom should be background blue, got {:?}", at(256, 384));
}

fn is_near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (*x as i32 - y as i32).abs() <= 2)
}

/// Copy an `Rgba8Unorm` texture back to CPU as tightly-packed RGBA bytes.
fn read_back(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<u8> {
    let unpadded = size.width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * size.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.height),
            },
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * size.height) as usize);
    for row in 0..size.height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}
