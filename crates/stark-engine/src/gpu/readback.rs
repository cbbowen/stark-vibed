//! GPU → CPU texture readback (§9). Used for export and golden tests.
//!
//! Readback is **inherently asynchronous** — it is the one GPU operation that is
//! (§7), and the one place where native and web genuinely differ:
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

use crate::error::{EngineError, Result};
use crate::gpu::context::GpuContext;
use crate::gpu::half::f16_to_f32;
use stark_model::geom::Extent2;

/// A texel of `texture`, in bytes.
///
/// Asked of the texture rather than of the caller. As a parameter it would make every
/// reader restate a fact the texture already carries — and a mismatch would not fail,
/// it would hand back rows shifted by the difference, which reads as a picture skewing
/// progressively sideways rather than as an error.
fn bytes_per_texel(texture: &wgpu::Texture) -> u32 {
    texture
        .format()
        .block_copy_size(None)
        .expect("readback of an uncompressed texture")
}

/// One texture→buffer copy's geometry: how wide a row really is, how wide the copy
/// has to make it, and how far apart the textures sit in the staging buffer.
///
/// Three numbers rather than three arguments threaded separately, because a mismatch
/// between any two of them does not fail — it hands back rows shifted by the
/// difference, which reads as a picture skewing progressively sideways.
struct Rows {
    /// A row's real bytes.
    unpadded: u32,
    /// A row's bytes in the staging buffer, rounded up to `COPY_BYTES_PER_ROW_ALIGNMENT`.
    padded: u32,
    /// One texture's whole span in the staging buffer.
    slot: u64,
}

impl Rows {
    fn of(texture: &wgpu::Texture, size: Extent2) -> Self {
        let unpadded = size.width * bytes_per_texel(texture);
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        Self {
            unpadded,
            padded,
            slot: u64::from(padded) * u64::from(size.height),
        }
    }
}

/// Copy `textures` into **one** mappable buffer, one slot each, and submit. Returns
/// the buffer and the geometry [`take_rows`] undoes.
///
/// One buffer however many textures, which is what a batched read *is*: a map is a
/// round trip to the queue and a gradient capture reads up to
/// [`MAX_SAMPLES`](stark_model::gradient::MAX_SAMPLES) patches (§22.2), so a map
/// apiece would be a map per texel of latency. Every texture must carry `COPY_SRC`,
/// share `size`, and share a format — the slot stride is taken from the first, and a
/// second texture of a different texel size would be copied at the wrong pitch.
fn begin_read(
    ctx: &GpuContext,
    textures: &[&wgpu::Texture],
    size: Extent2,
) -> (wgpu::Buffer, Rows) {
    let first = textures.first().expect("a read of no textures");
    let rows = Rows::of(first, size);

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark readback"),
        size: rows.slot * textures.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark readback encoder"),
        });
    for (i, texture) in textures.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: rows.slot * i as u64,
                    bytes_per_row: Some(rows.padded),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
    }
    ctx.queue.submit([encoder.finish()]);
    (buffer, rows)
}

/// Strip the row padding a texture→buffer copy required, leaving one tightly-packed
/// byte string per texture. Consumes the buffer: unmaps it, then **destroys** it.
///
/// Destroyed rather than dropped, for `ScopedResources`' reason (§6.2): on the web a
/// dropped buffer only releases its JS handle and waits for GC. A readback buffer is
/// the largest single allocation in the subsystem — an 8192² export is 268 MB — and
/// leaving that to a collector is how the tab OOMs. Safe here for the same reason it
/// is there: the copy that filled it has already been submitted *and* waited on, so
/// nothing is in flight against it.
fn take_rows(buffer: wgpu::Buffer, count: usize, size: Extent2, rows: &Rows) -> Vec<Vec<u8>> {
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer is mapped");
    let out = (0..count)
        .map(|i| {
            let base = (rows.slot * i as u64) as usize;
            let mut bytes = Vec::with_capacity((rows.unpadded * size.height) as usize);
            for row in 0..size.height {
                let start = base + (row * rows.padded) as usize;
                bytes.extend_from_slice(&data[start..start + rows.unpadded as usize]);
            }
            bytes
        })
        .collect();
    drop(data);
    buffer.unmap();
    buffer.destroy();
    out
}

/// Await the map of `buffer`, driving it the way this target needs.
///
/// How the map callback actually gets driven is the one thing that genuinely differs
/// between the two targets, and getting it wrong deadlocks rather than failing loudly:
///
///  · Native — nothing polls the device on its own, and the executor awaiting this
///    future is very likely blocking the only thread (`pollster`). So block *here*,
///    before awaiting: `Wait` drains the queue and fires the callback, and the await
///    below then resolves immediately. A non-blocking `Poll` hangs forever — the
///    thread parks and no one ever polls again.
///  · Web — there is no blocking poll; `mapAsync` is a promise that the browser's
///    event loop settles while this future is suspended. Calling poll would do
///    nothing, so awaiting *is* the wait.
///
/// **Reports rather than panics**, unlike the blocking sibling below, because this is
/// the path a *shipping* build takes: a failed map here is a lost device or an
/// exhausted one, and turning that into an `expect` was an abort — on the web, with
/// the painting unsaved (§5). The action log is untouched by any of it, so a caller
/// told the readback failed can still save the document; that is the whole point of
/// there being an error to tell it with.
async fn wait_mapped(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Result<()> {
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    #[cfg(not(target_arch = "wasm32"))]
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| readback_failed(ctx, format!("poll: {e}")))?;

    rx.await
        .map_err(|_| readback_failed(ctx, "the map callback was dropped".into()))?
        .map_err(|e| readback_failed(ctx, format!("map: {e}")))?;
    Ok(())
}

/// The error a failed readback reports, preferring what the **device** said over
/// what the map said.
///
/// A lost device fails the map too, so the map's own message is a symptom and the
/// device-lost callback holds the cause (`GpuHealth`). Reporting the cause where
/// there is one is what makes the message name a driver reset rather than a buffer.
fn readback_failed(ctx: &GpuContext, detail: String) -> EngineError {
    match ctx.health().failure() {
        Some(failure) => EngineError::Gpu(failure.to_string()),
        None => EngineError::Gpu(format!("readback failed ({detail})")),
    }
}

/// Read an 8-bit, 4-channel (e.g. `Rgba8Unorm`) texture back to tightly-packed
/// RGBA bytes.
pub async fn read_rgba8(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    size: Extent2,
) -> Result<Vec<u8>> {
    let (buffer, rows) = begin_read(ctx, &[texture], size);
    wait_mapped(ctx, &buffer).await?;
    Ok(take_rows(buffer, 1, size, &rows)
        .pop()
        .expect("one texture in, one string out"))
}

/// Blocking readback, for native callers only — the golden tests, which are
/// native by construction (`STARK_ALLOW_NO_GPU`, adapter-specific goldens).
///
/// Compiled out on wasm on purpose: WebGPU has no blocking poll, so this shape
/// cannot work there, and a `cfg` is the only guard that makes calling it from
/// the frontend a compile error rather than a runtime `OperationError`.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_rgba8_blocking(ctx: &GpuContext, texture: &wgpu::Texture, size: Extent2) -> Vec<u8> {
    let (buffer, rows) = begin_read(ctx, &[texture], size);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    take_rows(buffer, 1, size, &rows)
        .pop()
        .expect("one texture in, one string out")
}

fn decode_rgba16f(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
        .collect()
}

/// Read many same-sized `Rgba16Float` textures back in **one** buffer map — the
/// gradient capture's readback (§22.2), where a trace is up to
/// [`MAX_SAMPLES`](stark_model::gradient::MAX_SAMPLES) patches. Every texture must
/// carry `COPY_SRC` and share `size`; results come back in argument order, 4 `f32`
/// per texel.
///
/// **The format is checked here rather than by the caller**, and unconditionally.
/// The decode is four halves a texel, which is a claim about the texture and not
/// about the reader — so the reader is where it belongs, and the two callers that
/// each carried their own `debug_assert` were two copies of one fact that a release
/// build did not hold at all. A color space whose `color_format` is something else
/// (§6.7 leaves that open) would otherwise hand back a picture decoded at the wrong
/// stride: not an error, a wrong colour. The formats come from this build's own
/// pipeline, never from a file or a peer, so an assert is the right shape — there is
/// no outside input to refuse (§5).
pub async fn read_many_rgba16f(
    ctx: &GpuContext,
    textures: &[&wgpu::Texture],
    size: Extent2,
) -> Result<Vec<Vec<f32>>> {
    if textures.is_empty() {
        return Ok(Vec::new());
    }
    for texture in textures {
        assert_eq!(
            texture.format(),
            wgpu::TextureFormat::Rgba16Float,
            "this readback decodes four halves per texel"
        );
    }
    let (buffer, rows) = begin_read(ctx, textures, size);
    wait_mapped(ctx, &buffer).await?;
    Ok(take_rows(buffer, textures.len(), size, &rows)
        .iter()
        .map(|bytes| decode_rgba16f(bytes))
        .collect())
}
