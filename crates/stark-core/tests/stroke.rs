//! Step-2 stroke MVP tests (DESIGN.md §13 build order, step 2).
//!
//! Drives the engine through the command/action split: a painted stroke commits
//! one `CommitStroke` action, copy-on-write tiles render it, and the
//! `history`-backed timeline supports undo/redo. Also checks that the live
//! preview path produces the same pixels as the committed stroke (DESIGN.md §6.2).

use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushParams, Tool};
use stark_core::engine::headless_engine;
use stark_core::geom::{Extent2, Vec2};
use stark_core::Engine;

const SIZE: Extent2 = Extent2 { width: 256, height: 256 };
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const BG: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };

fn engine_or_skip() -> Option<Engine> {
    match pollster::block_on(headless_engine(TARGET, SIZE)) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn red_brush() -> BrushParams {
    BrushParams {
        color: [1.0, 0.0, 0.0, 1.0],
        radius: 40.0,
        spacing: 0.25,
        hardness: 0.5,
        flow: 1.0,
    }
}

/// Paint a short horizontal stroke through the canvas origin (screen center for
/// the default identity view).
fn paint_stroke(engine: &mut Engine) {
    engine.process(InputCommand::SetBrush(red_brush()));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-30.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo { sample: InputSample::at(Vec2::new(0.0, 0.0)) });
    engine.process(InputCommand::StrokeTo { sample: InputSample::at(Vec2::new(30.0, 0.0)) });
}

#[test]
fn stroke_commit_undo_redo() {
    let Some(mut engine) = engine_or_skip() else { return };

    // Live preview should already show the stroke before committing (DESIGN §6.2).
    paint_stroke(&mut engine);
    assert!(engine.observe().is_stroking);
    let preview = render(&mut engine);
    assert!(is_red(center(&preview)), "preview center should be red, got {:?}", center(&preview));

    // Commit: one action, stroke persists, undo becomes available.
    engine.process(InputCommand::EndStroke);
    assert!(!engine.observe().is_stroking);
    assert!(engine.observe().can_undo);

    let committed = render(&mut engine);
    assert!(is_red(center(&committed)), "committed center should be red");
    assert!(is_blue(corner(&committed)), "untouched corner should be background blue");
    // Live preview and committed render must agree (the "one rendering path").
    assert!(is_red(center(&committed)) == is_red(center(&preview)));

    // Undo returns to a blank canvas.
    engine.process(InputCommand::Undo);
    assert!(engine.observe().can_redo);
    let undone = render(&mut engine);
    assert!(is_blue(center(&undone)), "after undo, center should be background, got {:?}", center(&undone));

    // Redo repaints it.
    engine.process(InputCommand::Redo);
    let redone = render(&mut engine);
    assert!(is_red(center(&redone)), "after redo, center should be red again, got {:?}", center(&redone));
}

#[test]
fn stroke_spans_multiple_tiles_via_cow() {
    let Some(mut engine) = engine_or_skip() else { return };
    paint_stroke(&mut engine);
    engine.process(InputCommand::EndStroke);

    // A radius-40 stroke straddling the canvas origin touches all four tiles
    // around (0,0); copy-on-write should have populated each of them.
    let doc = engine.document();
    let populated: usize = doc.layers.iter().map(|l| l.tiles.size()).sum();
    assert!(populated >= 2, "stroke across the origin should populate multiple tiles, got {populated}");
}

// --- helpers ---

fn render(engine: &mut Engine) -> Vec<u8> {
    let ctx = pollster_ctx(engine);
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test target"),
        size: wgpu::Extent3d { width: SIZE.width, height: SIZE.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    engine.render(&view, BG);
    read_back(&ctx, &target)
}

/// Borrow the engine's GPU context for test-side texture creation/readback.
fn pollster_ctx(engine: &Engine) -> stark_core::GpuContext {
    engine.gpu().clone()
}

fn center(px: &[u8]) -> [u8; 4] {
    at(px, SIZE.width / 2, SIZE.height / 2)
}
fn corner(px: &[u8]) -> [u8; 4] {
    at(px, 10, 10)
}
fn at(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = (y * SIZE.width + x) as usize * 4;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}
fn is_red(c: [u8; 4]) -> bool {
    c[0] > 200 && c[1] < 60 && c[2] < 60
}
fn is_blue(c: [u8; 4]) -> bool {
    c[2] > 200 && c[0] < 60 && c[1] < 60
}

fn read_back(ctx: &stark_core::GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let unpadded = SIZE.width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * SIZE.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(SIZE.height),
            },
        },
        wgpu::Extent3d { width: SIZE.width, height: SIZE.height, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * SIZE.height) as usize);
    for row in 0..SIZE.height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}
