//! Mixbox's pigment LUT on the GPU — the inverse of the mixing polynomial
//! (§18.0.4, §6.7).
//!
//! One caller: the blend pass, which combines *light* and then has to say which
//! mixture of pigments would have produced it. Every other GPU path in the engine
//! runs Mixbox one way — `mixbox_poly.wesl` evaluates concentrations → colour — and
//! the CPU handles the rare reverse with the vendored `mixbox` crate. A per-texel
//! inverse is what forces the table onto the GPU.
//!
//! The bytes come from the vendored submodule's own `mixbox_lut.png`, which is the
//! layout Mixbox's shaders expect: the 64³ cube unrolled into an 8×8 grid of 64×64
//! slices, 512×512 RGBA8. It is embedded rather than fetched by the frontend, on the
//! same footing as the polynomial `build.rs` transpiles out of the same submodule:
//! this is part of what the colour space *is*, not content a document supplies. It
//! is also loaded only in a Mixbox document — an Oklab one never decodes it.
//!
//! Mixbox 2.0 (c) 2022 Secret Weapons, authors Sarka Sochorova and Ondrej Jamriska.
//! Licensed CC BY-NC 4.0; see `vendor/mixbox/LICENSE`.

use crate::gpu::context::GpuContext;

/// The vendored LUT image (git submodule; CC BY-NC 4.0).
const LUT_PNG: &[u8] = include_bytes!("../../../../vendor/mixbox/shaders/mixbox_lut.png");

/// Edge of the unrolled LUT image, in texels.
const LUT_DIM: u32 = 512;

/// The pigment LUT bound to the blend pass: a texture plus the sampler that reads
/// it.
pub struct PigmentLut {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl PigmentLut {
    /// Decode and upload the vendored LUT.
    ///
    /// The texture is **`Rgba8Unorm`, not sRGB**: these texels are polynomial
    /// coefficients that happen to be stored in an image, and letting the hardware
    /// apply a transfer curve to them would silently corrupt every mixture.
    pub fn load(ctx: &GpuContext) -> Self {
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

    /// A 1×1 stand-in for colour spaces whose blend pass never samples the LUT.
    ///
    /// The blend pass has one bind group layout across every space, so the binding
    /// exists either way; this is what fills it without decoding 512² of table a
    /// document will not use.
    pub fn placeholder(ctx: &GpuContext) -> Self {
        let extent = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark pigment lut placeholder"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            sampler: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("stark pigment lut placeholder sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }
}
