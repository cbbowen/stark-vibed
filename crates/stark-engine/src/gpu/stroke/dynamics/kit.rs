//! The stamp loop's GPU objects, built once (§6.2).
//!
//! Pipelines, bind group layouts and samplers — nothing that varies with a stroke, a
//! piece or a segment, all of which is [`plan`](super::plan)'s business or
//! [`run`](super::run)'s. Immutable throughout, which is what lets the kit be cloned
//! with its renderer and live in an `Action::Context` (§5).

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use stark_shaders::mirror::slice::decl as sld;
use stark_shaders::mirror::stamp_common::decl as scd;

/// The prefix-τ volume at group 1, compute-visible — the same slot the fast path binds
/// at fragment visibility (`stroke::swept`), from the same declaration (§6.6).
pub(super) const PREFIX_SLOTS: &[Slot] = &[Slot::at(scd::PREFIX_TEX)];

/// The write-back's aux narrowing (`slice.wesl`, §6.2/§6.4): the wide region aux in,
/// the tile's one channel out.
pub(super) const SLICE_SLOTS: &[Slot] = &[Slot::at(sld::REGION_AUX)];
use crate::gpu::tile::SCRATCH_AUX_FORMAT;

use super::slots;
/// GPU objects for the brush-dynamics stamp loop (§6.2), built once.
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
///
/// **Immutable throughout**, and the type says so rather than merely intending it: no
/// cache lives here. The round tip's coverage cache and the rest of the lazily-baked
/// brush textures sit together on the renderer
/// ([`TipCache::round_tip`](super::super::tips::TipCache)).
#[derive(Clone)]
pub(in crate::gpu::stroke) struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (color + wide aux).
    pub(in crate::gpu::stroke) composite_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) composite_view_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) composite_tile_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) composite_sampler: wgpu::Sampler,
    // The stamp-loop dispatches (one compute shader, several entry points).
    /// The extent copy that gives the `deposit`/`settle` something to read while
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
    /// The bleed pair's mobility pass (§6.2) and its layout.
    pub(in crate::gpu::stroke) bleed_weight_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) bleed_weight_bgl: wgpu::BindGroupLayout,
    /// What a **painting** segment's deposit binds where a firing binds the scratch: a
    /// 1×1 zero. Such a slot carries `lambda_bleed = 0` and never reads it, so this is
    /// the §6.8 stand-in pattern rather than a case the shader has to branch on.
    pub(in crate::gpu::stroke) bleed_placeholder: wgpu::TextureView,
    /// The tool's own side of one segment's transfer — the complement of every share
    /// the `deposit` after it hands the canvas (`dynamics.wesl::exchange`).
    pub(in crate::gpu::stroke) exchange_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) exchange_bgl: wgpu::BindGroupLayout,
    /// Integrates the reservoir along the segment's travel axis so the deposit can
    /// read the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(in crate::gpu::stroke) bake_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) bake_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) deposit_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) deposit_bgl: wgpu::BindGroupLayout,
    /// The **coarse deposit** pair (§6.2), for the slots whose tip's shoulder lets
    /// the exchange be evaluated per cell instead of per texel
    /// (`budget::extent_cell`): `cell_hoist` distils the prefix and the bake into
    /// per-cell means, `deposit_coarse` reads them back over the exact kernel's own
    /// texel grid. Slots with a cell of 1 — every hard or small tip, every bleed and
    /// settle slot — never touch either and keep `deposit_pipeline` bit-for-bit.
    pub(in crate::gpu::stroke) hoist_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) hoist_bgl: wgpu::BindGroupLayout,
    pub(in crate::gpu::stroke) deposit_coarse_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) deposit_coarse_bgl: wgpu::BindGroupLayout,
    /// The pen-up: settles the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`). Reads the reservoir through its own
    /// `bake` dispatch — the zero-travel slot bakes the *remaining pass's* delivery
    /// integral, not a per-segment window — never the cell that sits overhead.
    pub(in crate::gpu::stroke) settle_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) settle_bgl: wgpu::BindGroupLayout,
    /// The deposit's prefix-τ volume binding (group 1) — the same texture the
    /// swept fast path samples, so the exchange extent *is* the definite
    /// integral of the brush along the travel (compute-visible variant).
    pub(in crate::gpu::stroke) prefix_bgl: wgpu::BindGroupLayout,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    pub(in crate::gpu::stroke) exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back: the aux narrow pass. Color and residual leave
    // the region as plain texture copies (`DynamicsRun::write_back`), so the one
    // pipeline the write-back keeps is the narrowing of the wide region aux to the
    // persistent height channel — once over the whole region, not once per tile.
    pub(in crate::gpu::stroke) slice_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) slice_bgl: wgpu::BindGroupLayout,
}

/// Build the brush-dynamics stamp-loop kit (§6.2): the region
/// composite, the loop's nine compute pipelines, and the region→tile slice.
pub(in crate::gpu::stroke) fn build_dynamics_kit(
    ctx: &crate::gpu::context::GpuContext,
    color_space: &dyn ColorSpace,
    composite_tile_bgl: wgpu::BindGroupLayout,
) -> DynamicsKit {
    let device = &ctx.device;
    // The loop's storage-texture declarations are `rgba16float`; both color
    // spaces use that tile color format (§6.7), so the region can hold either.
    debug_assert_eq!(
        color_space.color_format(),
        wgpu::TextureFormat::Rgba16Float,
        "the loop declares rgba16float storage; this space's tiles are not that"
    );
    let frag = wgpu::ShaderStages::FRAGMENT;
    // Whether this space carries a **residual** (§6.7). It selects the `_resid` build
    // of every shader here that touches a tile's color, and adds the bindings and
    // targets that build declares. Oklab leaves every one of them off, so its layouts
    // are shorter rather than bound to stand-ins.
    let resid = color_space.has_resid();

    // ---- Region composite: the `composite` shader over region-sized targets
    // (color + the wide aux, so nothing is narrowed until the write-back).
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics composite"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::composite(resid).into()),
    });
    // The very layout pass A builds, because it *is* pass A's: the group this loop
    // binds per tile is the one the tile itself caches, which answers to one layout
    // (`composite::tile_bind_group_layout`). The view group has no such cache and so
    // is built here, from the same declarations pass A reads — this loop composites
    // its working region through `composite.wesl` itself (§6.3).
    let composite_view_bgl = desc::layout_for(
        device,
        "stark dynamics composite view bgl",
        crate::gpu::composite::COMPOSITE_VIEW_SLOTS,
        frag,
        resid,
    );
    let composite_layout = desc::pipeline_layout(
        device,
        "stark dynamics composite layout",
        &[Some(&composite_view_bgl), Some(&composite_tile_bgl)],
    );
    // Not `ChannelFormats::blended`: the region's aux is the *wide* scratch format and
    // takes the aux blend where the two color targets take the color's. Built rather
    // than sliced to a hand-counted length (§6.7).
    let mut composite_targets = vec![
        desc::blended_target(color_space.color_format(), Some(color_space.color_blend())),
        desc::blended_target(SCRATCH_AUX_FORMAT, Some(color_space.aux_blend())),
    ];
    if let Some(f) = color_space.resid_format() {
        // The region's residual, over-blended by the color's own rule because it is the
        // rest of the same color (§6.7).
        composite_targets.push(desc::blended_target(f, Some(color_space.color_blend())));
    }
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
            targets: &composite_targets,
        },
    );
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- The stamp loop: one module, seven entry points — `snapshot`, `exchange`,
    // `bake`, `deposit`, `cell_hoist`, `deposit_coarse`, `settle` — over seven bind
    // group layouts, each built from the slot list in [`slots`](super::slots).
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics loop"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::dynamics(resid).into()),
    });
    // Every layout below is compute-visible and opens with the dynamic-offset stamp
    // slot; the binding numbers partition the module's group(0), so a layout lists only
    // the bindings its own entry point reads.
    //
    // **The list is all the host says.** What kind of thing each slot holds — a uniform
    // and how wide, a sampler, a texture, a storage texture of a particular format —
    // and whether it exists at all without the residual, come from the generated
    // `BINDINGS` table (§6.10). No array here closes with a `resid` count
    // (`[..12 + 4 * usize::from(resid)]`, recounted by hand on every edit) — that gate
    // is the `@if(resid)` on the declaration itself.
    let bgl = |label: &str, list: &[desc::Slot]| {
        desc::layout_for(device, label, list, wgpu::ShaderStages::COMPUTE, resid)
    };
    let snapshot_bgl = bgl("stark dynamics snapshot bgl", slots::SNAPSHOT);
    let bleed_weight_bgl = bgl("stark dynamics bleed weight bgl", slots::BLEED_WEIGHT);
    let exchange_bgl = bgl("stark dynamics exchange bgl", slots::EXCHANGE);
    let bake_bgl = bgl("stark dynamics bake bgl", slots::BAKE);
    let settle_bgl = bgl("stark dynamics settle bgl", slots::SETTLE);
    let deposit_bgl = bgl("stark dynamics deposit bgl", slots::DEPOSIT);
    let hoist_bgl = bgl("stark dynamics cell hoist bgl", slots::HOIST);
    let deposit_coarse_bgl = bgl("stark dynamics deposit coarse bgl", slots::DEPOSIT_COARSE);
    // The deposit's prefix-τ volume (group 1) — same shape as the fast path's
    // prefix binding, but compute-visible.
    let prefix_bgl = desc::layout_for(
        device,
        "stark dynamics prefix bgl",
        PREFIX_SLOTS,
        wgpu::ShaderStages::COMPUTE,
        false,
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
    // The bleed ladder's mobility, hoisted (§6.2). It reads the prefix-τ volume, so it
    // takes group 1 like every other pass that does — one `swept_pre` per texel is the
    // whole of it.
    let bleed_weight_pipeline = cpipe(
        "stark dynamics bleed weight",
        "bleed_weight",
        &[Some(&bleed_weight_bgl), Some(&prefix_bgl)],
    );
    let exchange_pipeline = cpipe(
        "stark dynamics exchange",
        "exchange",
        &[Some(&exchange_bgl)],
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
    // The hoist takes the same prefix-τ taps the deposit's front half did; the coarse
    // deposit takes none, so its layout stops at group 0.
    let hoist_pipeline = cpipe(
        "stark dynamics cell hoist",
        "cell_hoist",
        &[Some(&hoist_bgl), Some(&prefix_bgl)],
    );
    let deposit_coarse_pipeline = cpipe(
        "stark dynamics deposit coarse",
        "deposit_coarse",
        &[Some(&deposit_coarse_bgl)],
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

    // ---- Region → tile write-back: the aux narrow pass. The color and residual
    // channels are copied out of the region bit-exactly (`DynamicsRun::write_back`),
    // so this draws once over the whole region rather than once per tile, and needs
    // neither a per-tile uniform nor a residual variant.
    let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics slice"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::slice().into()),
    });
    let slice_bgl = desc::layout_for(device, "stark dynamics slice bgl", SLICE_SLOTS, frag, false);
    let slice_layout =
        desc::pipeline_layout(device, "stark dynamics slice layout", &[Some(&slice_bgl)]);
    let slice_pipeline = desc::fullscreen_pipeline(
        device,
        "stark dynamics slice pipeline",
        &slice_layout,
        &slice_shader,
        ("vs_main", "fs_main"),
        &[desc::target(color_space.aux_format())],
    );

    // The 1×1 a painting segment binds where a firing binds the real scratch (§6.8's
    // stand-in pattern): such a slot carries `lambda_bleed = 0` and never reads it.
    let bleed_placeholder = desc::zero_texture(
        ctx,
        wgpu::TextureFormat::R32Float,
        "stark dynamics bleed w 1x1",
    );

    DynamicsKit {
        composite_pipeline,
        composite_view_bgl,
        composite_tile_bgl,
        composite_sampler,
        snapshot_pipeline,
        snapshot_bgl,
        bleed_weight_pipeline,
        bleed_weight_bgl,
        bleed_placeholder,
        exchange_pipeline,
        exchange_bgl,
        bake_pipeline,
        bake_bgl,
        deposit_pipeline,
        deposit_bgl,
        hoist_pipeline,
        hoist_bgl,
        deposit_coarse_pipeline,
        deposit_coarse_bgl,
        settle_pipeline,
        settle_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        prefix_bgl,
    }
}
