//! Mixbox's pigment LUT on the GPU — the inverse of the mixing polynomial
//! (§18.0.4, §6.7).
//!
//! Decoded once per space, by the blend pass — which combines *light* and then has to
//! say which mixture of pigments would have produced it — and bound by the filter pass
//! too, for the same question (§21). Every other GPU path runs Mixbox forwards
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

use crate::device::context::GpuContext;

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

/// The pigment LUT the blend and filter passes bind: a texture plus the sampler that
/// reads it. Handles only, so a clone binds the same table rather than decoding
/// another.
#[derive(Clone)]
pub(crate) struct PigmentLut {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl PigmentLut {
    /// The LUT `fs` reads, or `None` where it reaches none.
    ///
    /// **The shader's answer, not the space's.** Only the pigment builds of
    /// `blend.wesl` and `filter.wesl` declare a LUT, so a colorimetric space's layout
    /// has no slot
    /// for it — there is nothing to stand in for, and no `needs_pigment_lut` for a space
    /// to answer twice. `lut` is the caller's own module's declaration, the two being
    /// different slots of different modules (§6.10).
    pub(crate) fn of(ctx: &GpuContext, fs: EntryPoint, lut: Binding) -> Option<Self> {
        fs.reads(lut).then(|| Self::load(ctx))
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

        let size = wgpu::Extent3d {
            width: LUT_DIM,
            height: LUT_DIM,
            depth_or_array_layers: 1,
        };
        let texture = crate::device::desc::texture_with_data(
            ctx,
            &crate::device::desc::sampled_texture(
                "stark pigment lut",
                wgpu::TextureFormat::Rgba8Unorm,
                size,
            ),
            &rgba,
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
    /// by the two Mixbox shaders, which a build without it does not link — so no entry
    /// point there reaches one.
    #[cfg(not(feature = "mixbox"))]
    fn load(_ctx: &GpuContext) -> Self {
        unreachable!("no shader in a build without Mixbox declares the pigment LUT")
    }
}
