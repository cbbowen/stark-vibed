//! Mixbox's pigment LUT on the GPU — the inverse of the mixing polynomial
//! (§18.0.4, §6.7).
//!
//! One owner, the blend pass — which combines *light* and then has to say which
//! mixture of pigments would have produced it — and the filter pass borrows it for the
//! same question (§21). Every other GPU path runs Mixbox forwards
//! (`lib/mixbox.wesl`) and the CPU handles the rare reverse; a per-texel inverse is
//! what forces the table onto the GPU.
//!
//! The bytes are the vendored submodule's own `mixbox_lut.png` — the 64³ cube unrolled
//! into an 8×8 grid of 64×64 slices, 512×512 RGBA8, the layout Mixbox's shaders expect.
//! Embedded rather than fetched: this is part of what the color space *is*, not content
//! a document supplies. Only a Mixbox document decodes it.
//!
//! Mixbox 2.0 (c) 2022 Secret Weapons, authors Sarka Sochorova and Ondrej Jamriska.
//! Licensed CC BY-NC 4.0; see `vendor/mixbox/LICENSE`.

use stark_shaders::{Binding, EntryPoint};

use crate::gpu::context::GpuContext;

/// The vendored LUT image (git submodule; CC BY-NC 4.0).
///
/// Behind the `mixbox` feature because `include_bytes!` is resolved at *compile* time:
/// an unconditional one would keep the submodule a hard build requirement even in a
/// build that can never decode it.
#[cfg(feature = "mixbox")]
const LUT_PNG: &[u8] = include_bytes!("../../../../vendor/mixbox/shaders/mixbox_lut.png");

/// Edge of the unrolled LUT image, in texels. Read only by [`PigmentLut::load`], so
/// it goes with it.
#[cfg(feature = "mixbox")]
const LUT_DIM: u32 = 512;

/// The pigment LUT bound to the blend pass: a texture plus the sampler that reads
/// it.
pub struct PigmentLut {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

/// Whether `fs` reads `lut`, the LUT slot its own module declares — which is what puts
/// it in that pass's layout.
///
/// **The shader's answer, not the space's.** Only `blend_mixbox.wesl` and
/// `filter_mixbox.wesl` declare a LUT, so a colorimetric space's layout has no slot for
/// it — there is nothing to stand in for, and no `needs_pigment_lut` for a space to
/// answer twice. Each pass asks it of its own shader, and passes its own declaration
/// because the two are different slots of different modules (§6.10).
pub fn read_by(fs: EntryPoint, lut: Binding) -> bool {
    fs.uses.iter().any(|u| u.decl == lut)
}

impl PigmentLut {
    /// The LUT `fs` reads, or `None` where it declares none ([`read_by`]).
    pub fn of(ctx: &GpuContext, fs: EntryPoint, lut: Binding) -> Option<Self> {
        read_by(fs, lut).then(|| Self::load(ctx))
    }

    /// Decode and upload the vendored LUT.
    ///
    /// The texture is **`Rgba8Unorm`, not sRGB**: these texels are polynomial
    /// coefficients that happen to be stored in an image, and letting the hardware
    /// apply a transfer curve to them would silently corrupt every mixture.
    #[cfg(feature = "mixbox")]
    fn load(ctx: &GpuContext) -> Self {
        let decoder = png::Decoder::new(std::io::Cursor::new(LUT_PNG));
        let mut reader = decoder.read_info().expect("pigment lut: read png info");
        let size = reader
            .output_buffer_size()
            .expect("pigment lut: png output size");
        let mut buf = vec![0u8; size];
        let info = reader
            .next_frame(&mut buf)
            .expect("pigment lut: decode png frame");
        assert_eq!(
            (info.width, info.height),
            (LUT_DIM, LUT_DIM),
            "pigment lut: unexpected LUT dimensions"
        );

        // Widen to RGBA if the vendored image is ever re-encoded without alpha; only
        // `.rgb` is ever read, so the alpha value is immaterial.
        let rgba: Vec<u8> = match info.color_type {
            png::ColorType::Rgba => buf,
            png::ColorType::Rgb => buf
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            other => panic!("pigment lut: unsupported PNG color type {other:?}"),
        };

        let extent = wgpu::Extent3d {
            width: LUT_DIM,
            height: LUT_DIM,
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark pigment lut"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            texture.as_image_copy(),
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * LUT_DIM),
                rows_per_image: Some(LUT_DIM),
            },
            extent,
        );

        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            // Linear in x/y: Mixbox interpolates within a slice in hardware and only
            // blends the two z-slices in the shader. Clamped, so the `iz == 63` fetch
            // that runs off the end of the grid (at weight 0) stays in bounds.
            sampler: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("stark pigment lut sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// Unreachable without the feature, and structurally so: the LUT is declared only
    /// by the two Mixbox shaders, which a build without it does not link — so
    /// [`read_by`] is false for every entry point there is.
    #[cfg(not(feature = "mixbox"))]
    fn load(_ctx: &GpuContext) -> Self {
        unreachable!("no shader in a build without Mixbox declares the pigment LUT")
    }
}
