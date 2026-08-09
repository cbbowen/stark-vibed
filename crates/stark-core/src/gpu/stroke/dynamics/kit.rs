//! The stamp loop's GPU objects, built once (§6.2).
//!
//! Pipelines, bind group layouts and samplers — nothing that varies with a stroke, a
//! piece or a segment, all of which is [`plan`](super::plan)'s business or
//! [`run`](super::run)'s. Immutable throughout, which is what lets the kit be cloned
//! with its renderer and live in an `Action::Context` (§5).

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::tile::SCRATCH_AUX_FORMAT;

use super::BAKE_FORMAT;
use super::plan::SLOT;
use super::run::SLICE_SLOT;
/// GPU objects for the brush-dynamics stamp loop (§6.2), built once.
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
///
/// Immutable throughout, which the type now says rather than merely intends: it
/// used to carry the round tip's coverage cache, the one mutable thing in a struct
/// documented as built-once. That lives with its sibling on the renderer
/// ([`StrokeRenderer::round_tip`](super::StrokeRenderer)), where the rest of the
/// lazily-baked brush textures already were.
#[derive(Clone)]
pub(in crate::gpu::stroke) struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (colour + wide aux).
    pub(in crate::gpu::stroke) composite_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) composite_view_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) composite_tile_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) composite_sampler: wgpu::Sampler,
    // The stamp-loop dispatches (one compute shader, several entry points).
    /// The footprint copy that gives the `deposit`/`settle` something to read while
    /// they storage-write the region.
    ///
    /// A painting segment does not dispatch it: its snapshot rides in the tail of its
    /// own `exchange` grid, since it depends on nothing that pass writes
    /// (`dynamics.wesl::exchange`). The two slot kinds with no exchange to ride in —
    /// [`SlotKind::Bleed`] and [`SlotKind::Settle`] — dispatch it standalone. (The
    /// settle could not have shared a grid in any case: it *reads* the snapshot,
    /// rather than merely sharing a consumer with it.)
    pub(in crate::gpu::stroke) snapshot_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) snapshot_bgl: wgpu::BindGroupLayout,
    /// The tool's own side of one segment's transfer — the complement of every share
    /// the `deposit` after it hands the canvas (`dynamics.wesl::exchange`).
    pub(in crate::gpu::stroke) exchange_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) exchange_bgl: wgpu::BindGroupLayout,
    /// Paint migrating *within* the tip, ahead of both halves of the transfer
    /// (`dynamics.wesl::wick_axis`). Shares `exchange`'s bind group: it reads and writes
    /// the same reservoir ping-pong and needs a strict subset of the same bindings.
    ///
    /// Two pipelines because the stencil is **separable** and runs as one 1-D pass per
    /// axis: `.0` along the tip's travel, `.1` across it. Splitting them is what lets
    /// the kernel be four times as wide for the same work per unit travel — and it is
    /// the shape a stroke-space march needs, where the along-travel pass lives inside a
    /// lateral row's own workgroup and only the across-row pass is a barrier.
    pub(in crate::gpu::stroke) wick_pipelines: (wgpu::ComputePipeline, wgpu::ComputePipeline),
    /// Integrates the reservoir along the segment's travel axis so the deposit can
    /// read the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(in crate::gpu::stroke) bake_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) bake_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) deposit_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) deposit_bgl: wgpu::BindGroupLayout,
    /// The pen-up: settles the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`). Reads the reservoir through its own
    /// `bake` dispatch — the zero-travel slot bakes the *remaining pass's* delivery
    /// integral, not a per-segment window — never the cell that sits overhead.
    pub(in crate::gpu::stroke) settle_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) settle_bgl: wgpu::BindGroupLayout,
    /// The deposit's prefix-τ volume binding (group 1) — the same texture the
    /// swept fast path samples, so the exchange footprint *is* the definite
    /// integral of the brush along the travel (compute-visible variant).
    pub(in crate::gpu::stroke) prefix_bgl: wgpu::BindGroupLayout,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    pub(in crate::gpu::stroke) exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back.
    pub(in crate::gpu::stroke) slice_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) slice_bgl: wgpu::BindGroupLayout,
}

/// Build the brush-dynamics stamp-loop kit (§6.2): the region
/// composite, the loop's seven compute pipelines, and the region→tile slice.
pub(in crate::gpu::stroke) fn build_dynamics_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> DynamicsKit {
    // The loop's storage-texture declarations are `rgba16float`; both color
    // spaces use that tile colour format (§6.7), so the region can hold either.
    debug_assert_eq!(color_space.color_format(), wgpu::TextureFormat::Rgba16Float);
    let frag = wgpu::ShaderStages::FRAGMENT;
    // Whether this space carries a **residual** (§6.7). It selects the `_resid` build
    // of every shader here that touches a tile's colour, and adds the bindings and
    // targets that build declares. Oklab leaves every one of them off, so its layouts
    // are shorter rather than bound to stand-ins.
    let resid = color_space.has_resid();

    // ---- Region composite: the `composite` shader over region-sized targets
    // (colour + the wide aux, so nothing is narrowed until the write-back).
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics composite"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::composite(resid).into()),
    });
    let composite_view_bgl = desc::bind_group_layout(
        device,
        "stark dynamics composite view bgl",
        &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    );
    let composite_tile_bgl = desc::bind_group_layout(
        device,
        "stark dynamics composite tile bgl",
        // Binding 2 is the tile's residual (§6.7) — the source tiles are composited
        // into the region whole, so the channel rides in with the colour it belongs to.
        &[
            desc::sample_tex(0, frag),
            desc::sample_tex(1, frag),
            desc::sample_tex(2, frag),
        ][..2 + usize::from(resid)],
    );
    let composite_layout = desc::pipeline_layout(
        device,
        "stark dynamics composite layout",
        &[Some(&composite_view_bgl), Some(&composite_tile_bgl)],
    );
    let composite_pipeline = desc::render_pipeline(
        device,
        desc::RenderPipe {
            label: "stark dynamics composite pipeline",
            layout: &composite_layout,
            module: &composite_shader,
            vs: "vs_main",
            // `fs_raw`, NOT the screen path's `fs_main`: the loop's region must hold
            // the tile representation itself (opacity in alpha), not the
            // coverage-weighted channels pass A shows — the exchange reads this
            // region and the slice writes it back to persistent tiles.
            fs: "fs_raw",
            primitive: desc::QUAD_STRIP,
            buffers: &[Some(stark_shaders::mirror::composite::instance_layout(
                wgpu::VertexStepMode::Instance,
            ))],
            targets: &[
                desc::blended_target(color_space.color_format(), Some(color_space.color_blend())),
                desc::blended_target(SCRATCH_AUX_FORMAT, Some(color_space.aux_blend())),
                // The region's residual, over-blended by the colour's own rule
                // because it is the rest of the same colour (§6.7).
                color_space
                    .resid_format()
                    .and_then(|f| desc::blended_target(f, Some(color_space.color_blend()))),
            ][..2 + usize::from(resid)],
        },
    );
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- The stamp loop: one module, seven entry points — `snapshot`, `exchange`,
    // `wick_x`, `wick_y`, `bake`, `deposit`, `settle` — over five bind group layouts,
    // since the two wick axes share `exchange`'s (they need a strict subset of it).
    // Every layout includes the dynamic-offset stamp uniform at binding 0; the binding
    // numbers partition the module's group(0) — see dynamics.wesl.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics loop"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::dynamics(resid).into()),
    });
    // Every layout below is compute-visible and opens with the dynamic-offset stamp
    // slot; the binding numbers partition the module's group(0), so a layout lists
    // only the bindings its own entry point reads.
    let comp = wgpu::ShaderStages::COMPUTE;
    let params_entry = desc::uniform_slot(0, comp, SLOT as u64);
    let ctex = |binding: u32, filterable: bool| {
        if filterable {
            desc::sample_tex(binding, comp)
        } else {
            desc::load_tex(binding, comp)
        }
    };
    let stor = |binding: u32| desc::storage_tex(binding, comp, wgpu::TextureFormat::Rgba16Float);
    // The baked swept prefix is fp32 — it is differenced per fragment, like the
    // prefix-τ volume, so f16 would band exactly where the difference is smallest.
    let stor32 = |binding: u32| desc::storage_tex(binding, comp, BAKE_FORMAT);
    let csamp = desc::sampler(5, comp);
    // The residual bindings (23–30) each sit beside the colour binding they ride with;
    // a layout that lists one lists the other. See the block at the head of
    // dynamics.wesl for what each carries.
    let snapshot_bgl = desc::bind_group_layout(
        device,
        "stark dynamics snapshot bgl",
        &[
            params_entry,
            ctex(1, false),
            ctex(2, false),
            stor(3),
            stor(4),
            ctex(23, false), // region_resid
            stor(26),        // under_resid_w
        ][..5 + 2 * usize::from(resid)],
    );
    let exchange_bgl = desc::bind_group_layout(
        device,
        "stark dynamics exchange bgl",
        &[
            params_entry,
            ctex(1, true),
            ctex(2, true),
            // The footprint snapshot's targets: the segment's `snapshot` runs from the
            // tail of the `exchange` grid rather than from a dispatch of its own
            // (`dynamics.wesl::exchange`), so its writes belong to this layout. The
            // `wick`, which shares this layout, leaves them alone.
            stor(3),
            stor(4),
            csamp,
            ctex(6, true),
            ctex(7, false),
            ctex(8, false),
            stor(9),
            stor(10),
            // The selection mask over the region (§6.8) — sampled bilinearly here,
            // since a reservoir texel sits over an arbitrary sub-pixel spot.
            ctex(21, true),
            ctex(23, true), // region_resid — the same bilinear tap as 1/2
            stor(26),       // under_resid_w, for the snapshot riding in this grid
            ctex(27, true), // brush_src_resid
            stor(28),       // brush_dst_resid_w
        ][..12 + 4 * usize::from(resid)],
    );
    // `bake` integrates the reservoir along the travel axis for one segment; the
    // deposit then reads the result instead of point-sampling the reservoir.
    let bake_bgl = desc::bind_group_layout(
        device,
        "stark dynamics bake bgl",
        &[
            params_entry,
            csamp,
            ctex(7, true),
            ctex(8, true),
            stor32(17),
            stor32(18),
            ctex(27, true), // brush_src_resid
            stor32(29),     // bake_rlm_w — fp32, differenced like the two beside it
        ][..6 + 2 * usize::from(resid)],
    );
    // The pen-up settle: the deposit's targets and snapshot, and the deposit's *baked*
    // reservoir reads too — its parcel is the delivery integral of the remaining pass,
    // which the settle slot's own `bake` dispatch stores (`dynamics.wesl::settle`),
    // not the cell that happens to sit overhead.
    let settle_bgl = desc::bind_group_layout(
        device,
        "stark dynamics settle bgl",
        &[
            params_entry,
            ctex(19, false),
            ctex(20, false),
            ctex(11, false),
            ctex(12, false),
            stor(13),
            stor(14),
            ctex(21, false),
            // The ground (§6.4): the settle lays paint, so it reads the tooth too.
            ctex(22, false),
            ctex(30, false), // bake_rlm
            ctex(25, false), // under_resid
            stor(24),        // region_resid_w
        ][..9 + 3 * usize::from(resid)],
    );
    let deposit_bgl = desc::bind_group_layout(
        device,
        "stark dynamics deposit bgl",
        &[
            params_entry,
            csamp,
            ctex(19, false),
            ctex(20, false),
            ctex(11, false),
            ctex(12, false),
            stor(13),
            stor(14),
            // The colour-dynamics noise tile + its repeat sampler (§6.2).
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 16,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // The selection mask over the region (§6.8) — read 1:1 with the region
            // here, so `textureLoad` suffices.
            ctex(21, false),
            // The canvas surface's height map — the deposition tooth (§6.4). Read
            // nearest, so it needs no sampler and is not filterable.
            ctex(22, false),
            ctex(30, false), // bake_rlm
            ctex(25, false), // under_resid
            stor(24),        // region_resid_w
        ][..12 + 3 * usize::from(resid)],
    );
    // The deposit's prefix-τ volume (group 1) — same shape as the fast path's
    // prefix binding, but compute-visible.
    let prefix_bgl = desc::bind_group_layout(
        device,
        "stark dynamics prefix bgl",
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }],
    );
    let cpipe = |label: &str, entry: &str, bgls: &[Option<&wgpu::BindGroupLayout>]| {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: bgls,
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let snapshot_pipeline = cpipe(
        "stark dynamics snapshot",
        "snapshot",
        &[Some(&snapshot_bgl)],
    );
    let exchange_pipeline = cpipe(
        "stark dynamics exchange",
        "exchange",
        &[Some(&exchange_bgl)],
    );
    // The wick reads the reservoir and writes the other half of the ping-pong, which is
    // exactly what `exchange_bgl` already describes — the region and selection bindings
    // it also carries simply go unused, which a pipeline layout is free to do.
    //
    // One pipeline per axis: the stencil is separable, so a firing is two 1-D passes
    // (`dynamics.wesl::wick_axis`). The axis rides in the entry point rather than the
    // uniform because both passes of a firing share the segment's slot.
    let wick_pipelines = (
        cpipe("stark dynamics wick x", "wick_x", &[Some(&exchange_bgl)]),
        cpipe("stark dynamics wick y", "wick_y", &[Some(&exchange_bgl)]),
    );
    // The bake reads the prefix-τ volume too (group 1) — the exposure weights in
    // its integral are that volume's own differences.
    let bake_pipeline = cpipe(
        "stark dynamics bake",
        "bake",
        &[Some(&bake_bgl), Some(&prefix_bgl)],
    );
    let deposit_pipeline = cpipe(
        "stark dynamics deposit",
        "deposit",
        &[Some(&deposit_bgl), Some(&prefix_bgl)],
    );
    // The settle reads the prefix-τ volume too (group 1): its exposure is a pair of
    // readings of it, which is what makes the pen-up fade over the whole tip rather
    // than over the few pixels of its coverage knee.
    let settle_pipeline = cpipe(
        "stark dynamics settle",
        "settle",
        &[Some(&settle_bgl), Some(&prefix_bgl)],
    );
    let exchange_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics exchange sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- Region → tile slice (write-back).
    let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics slice"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::slice(resid).into()),
    });
    // One slot per tile the piece writes back, selected by a dynamic offset: the
    // region bindings beside it are the same for every tile, so the whole group is
    // built once per piece ([`UNIFORM_STRIDE`]).
    let slice_bgl = desc::bind_group_layout(
        device,
        "stark dynamics slice bgl",
        &[
            desc::uniform_slot(0, frag, SLICE_SLOT),
            desc::load_tex(1, frag),
            desc::load_tex(2, frag),
            desc::load_tex(3, frag), // the region's residual (§6.7)
        ][..3 + usize::from(resid)],
    );
    let slice_layout =
        desc::pipeline_layout(device, "stark dynamics slice layout", &[Some(&slice_bgl)]);
    let slice_pipeline = desc::fullscreen_pipeline(
        device,
        "stark dynamics slice pipeline",
        &slice_layout,
        &slice_shader,
        ("vs_main", "fs_main"),
        &[
            desc::target(color_space.color_format()),
            desc::target(color_space.aux_format()),
            color_space.resid_format().and_then(desc::target),
        ][..2 + usize::from(resid)],
    );

    DynamicsKit {
        composite_pipeline,
        composite_view_bgl,
        composite_tile_bgl,
        composite_sampler,
        snapshot_pipeline,
        snapshot_bgl,
        exchange_pipeline,
        exchange_bgl,
        wick_pipelines,
        bake_pipeline,
        bake_bgl,
        deposit_pipeline,
        deposit_bgl,
        settle_pipeline,
        settle_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        prefix_bgl,
    }
}
