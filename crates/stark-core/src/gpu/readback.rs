//! GPU → CPU texture readback (DESIGN.md §9). Used for export and golden tests.
//!
//! Readback is **inherently asynchronous** — it is the one GPU operation that is
//! (DESIGN.md §7), and the one place where native and web genuinely differ:
//!
//! - Natively, `Device::poll(Wait)` blocks until the queue drains, so the map
//!   callback has already fired by the time it returns.
//! - On WebGPU there is no blocking poll. `mapAsync` returns a JS promise that
//!   only settles when the browser's event loop runs, so `poll` is a no-op and
//!   `getMappedRange` on the next line fails with `OperationError` — the buffer
//!   is simply not mapped yet.
//!
//! So the real entry point is [`read_rgba8`], which is `async` and correct on
//! both. The blocking [`read_rgba8_blocking`] is kept for the golden tests, which
//! are native by construction, and is **compiled out on wasm** so that the
//! failure above cannot be reintroduced by calling it from the frontend.

use crate::geom::Extent2;
use crate::gpu::context::GpuContext;

/// Copy a texture into a mappable buffer, and return it with the row padding the
/// copy required. Shared by the async and blocking paths, which differ only in
/// how they wait.
fn begin_read(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    size: Extent2,
    bytes_per_texel: u32,
) -> (wgpu::Buffer, u32, u32) {
    let unpadded = size.width * bytes_per_texel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark readback"),
        size: (padded * size.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark readback encoder"),
        });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
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
    (buffer, unpadded, padded)
}

/// Strip the row padding a texture→buffer copy required, leaving tightly-packed
/// bytes. Consumes the mapping and unmaps the buffer.
fn take_rows(buffer: &wgpu::Buffer, size: Extent2, unpadded: u32, padded: u32) -> Vec<u8> {
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer is mapped");
    let mut out = Vec::with_capacity((unpadded * size.height) as usize);
    for row in 0..size.height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}

/// Read any texture back to tightly-packed bytes, awaiting the map.
/// `bytes_per_texel` must match the texture format.
async fn read_texture_bytes(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    size: Extent2,
    bytes_per_texel: u32,
) -> Vec<u8> {
    let (buffer, unpadded, padded) = begin_read(ctx, texture, size, bytes_per_texel);

    let (tx, rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    // How the map callback actually gets driven is the one thing that genuinely
    // differs between the two targets, and getting it wrong deadlocks rather than
    // failing loudly:
    //
    //  · Native — nothing polls the device on its own, and the executor awaiting
    //    this future is very likely blocking the only thread (`pollster`). So
    //    block *here*, before awaiting: `Wait` drains the queue and fires the
    //    callback, and the await below then resolves immediately. A non-blocking
    //    `Poll` hangs forever — the thread parks and no one ever polls again.
    //  · Web — there is no blocking poll; `mapAsync` is a promise that the
    //    browser's event loop settles while this future is suspended. Calling
    //    poll would do nothing, so awaiting *is* the wait.
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");

    rx.await
        .expect("readback callback dropped")
        .expect("map readback buffer");

    take_rows(&buffer, size, unpadded, padded)
}

/// Read an 8-bit, 4-channel (e.g. `Rgba8Unorm`) texture back to tightly-packed
/// RGBA bytes.
pub async fn read_rgba8(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<u8> {
    read_texture_bytes(ctx, texture, size, 4).await
}

/// Blocking readback, for native callers only — the golden tests, which are
/// native by construction (`STARK_ALLOW_NO_GPU`, adapter-specific goldens).
///
/// Compiled out on wasm on purpose: WebGPU has no blocking poll, so this shape
/// cannot work there, and a `cfg` is the only guard that makes calling it from
/// the frontend a compile error rather than a runtime `OperationError`.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_rgba8_blocking(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<u8> {
    let (buffer, unpadded, padded) = begin_read(ctx, texture, size, 4);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    take_rows(&buffer, size, unpadded, padded)
}

/// Read an `Rgba16Float` texture back as `f32` RGBA (4 per texel). The texture must carry
/// `COPY_SRC`. Used by reservoir-visualization debugging (DESIGN.md §6.2).
pub async fn read_rgba16f(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<f32> {
    let bytes = read_texture_bytes(ctx, texture, size, 8).await; // 4 × f16
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
        .collect()
}

/// Decode an IEEE-754 half-precision float to `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = match exp {
        0 => (mant as f32) * 2f32.powi(-24), // subnormal (and zero)
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -val } else { val }
}
