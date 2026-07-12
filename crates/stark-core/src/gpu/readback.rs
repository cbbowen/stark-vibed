//! GPU → CPU texture readback (DESIGN.md §9). Used for export and golden tests.

use crate::geom::Extent2;
use crate::gpu::context::GpuContext;

/// Read any texture back to tightly-packed bytes (row padding removed), blocking
/// until the copy completes. `bytes_per_texel` must match the texture format.
fn read_texture_bytes(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    size: Extent2,
    bytes_per_texel: u32,
) -> Vec<u8> {
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

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");

    let data = slice.get_mapped_range().expect("slice is mapped");
    let mut out = Vec::with_capacity((unpadded * size.height) as usize);
    for row in 0..size.height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}

/// Read an 8-bit, 4-channel (e.g. `Rgba8Unorm`) texture back to tightly-packed
/// RGBA bytes. Blocks until the copy completes.
pub fn read_rgba8(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<u8> {
    read_texture_bytes(ctx, texture, size, 4)
}

/// Read an `Rgba16Float` texture back as `f32` RGBA (4 per texel). The texture must carry
/// `COPY_SRC`. Used by reservoir-visualization debugging (DESIGN.md §6.2).
pub fn read_rgba16f(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<f32> {
    let bytes = read_texture_bytes(ctx, texture, size, 8); // 4 × f16
    bytes
        .chunks_exact(2)
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
    if sign == 1 {
        -val
    } else {
        val
    }
}
