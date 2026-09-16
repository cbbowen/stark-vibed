//! The stamp loop's GPU objects, built once (§6.2).
//!
//! Pipelines, bind group layouts and samplers — nothing that varies with a stroke, a
//! piece or a segment, all of which is [`plan`](super::plan)'s business or
//! [`run`](super::run)'s. Immutable throughout, which is what lets the kit be cloned
//! with its renderer and live in an `Action::Context` (§5).

use crate::colorspace::ColorSpace;
use crate::gpu::desc;
use crate::gpu::tile::SCRATCH_AUX_FORMAT;
use stark_shaders::Stages;
use stark_shaders::mirror::dynamics::decl as d;
use stark_shaders::mirror::dynamics_common::decl as sd;
use stark_shaders::mirror::slice::decl as sld;
use stark_shaders::mirror::stamp_common::decl as scd;
/// GPU objects for the brush-dynamics stamp loop (§6.2), built once. All handles are
/// `Arc`-backed, so the kit is cheap to clone with its renderer.
///
/// **Immutable throughout**: no cache lives here. The lazily-baked brush textures sit
/// on the renderer instead ([`TipCache::round_tip`](super::super::tips::TipCache)).
#[derive(Clone)]
pub(in crate::gpu::stroke) struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (color + wide aux).
    pub(in crate::gpu::stroke) composite_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) composite_view_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) composite_tile_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) composite_sampler: wgpu::Sampler,
    // The stamp-loop dispatches (one compute shader, several entry points).
    /// The extent copy that gives the `deposit`/`settle` something to read while
    /// they storage-write the region.
    ///
    /// A painting segment does not dispatch it — its snapshot rides in the tail of
    /// its own `exchange` grid, depending on nothing that pass writes
    /// (`dynamics.wesl::exchange`). Only the two slot kinds with no exchange to ride
    /// in, [`SlotKind::Bleed`](super::plan::SlotKind) and
    /// [`SlotKind::Settle`](super::plan::SlotKind), dispatch it standalone.
    pub(in crate::gpu::stroke) snapshot_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) snapshot_bgl: desc::Bindings,
    /// The bleed pair's mobility pass (§6.2) and its layout.
    pub(in crate::gpu::stroke) bleed_weight_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) bleed_weight_bgl: desc::Bindings,
    /// What a **painting** segment's deposit binds where a firing binds the scratch: a
    /// 1×1 zero. Such a slot carries `lambda_bleed = 0` and never reads it, so this is
    /// the §6.8 stand-in pattern rather than a case the shader has to branch on.
    pub(in crate::gpu::stroke) bleed_placeholder: wgpu::TextureView,
    /// The ceiling lane's 1×1 zero (§6.2), bound where a stroke has no lane.
    pub(in crate::gpu::stroke) levels_placeholder: wgpu::TextureView,
    /// The tool's own side of one segment's transfer — the complement of every share
    /// the `deposit` after it hands the canvas (`dynamics.wesl::exchange`).
    pub(in crate::gpu::stroke) exchange_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) exchange_bgl: desc::Bindings,
    /// Integrates the reservoir along the segment's travel axis so the deposit can
    /// read the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(in crate::gpu::stroke) bake_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) bake_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) deposit_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) deposit_bgl: desc::Bindings,
    /// The **coarse deposit** pair (§6.2), for the slots whose tip's shoulder lets
    /// the exchange be evaluated per cell instead of per texel
    /// (`budget::extent_cell`): `cell_hoist` distils the prefix and the bake into
    /// per-cell means, `deposit_coarse` reads them back over the exact kernel's own
    /// texel grid. A slot with a cell of 1 touches neither and keeps
    /// `deposit_pipeline` bit-for-bit.
    pub(in crate::gpu::stroke) hoist_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) hoist_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) deposit_coarse_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) deposit_coarse_bgl: desc::Bindings,
    /// The pen-up: settles the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`). Reads the reservoir through its own
    /// `bake` dispatch — the zero-travel slot bakes the *remaining pass's* delivery
    /// integral, not a per-segment window — never the cell that sits overhead.
    pub(in crate::gpu::stroke) settle_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) settle_bgl: desc::Bindings,
    /// The liquify field's three kernels (§6.13, `liquify.wesl`): the field's
    /// snapshot under a segment's square, the composition of one segment's step into
    /// it (`warp`), and the one resample of a piece through it (`warp_apply`). Each
    /// over its own layout; only the composition takes group 1, bound to the tip's
    /// coverage prefix.
    pub(in crate::gpu::stroke) snapshot_field_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) snapshot_field_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) warp_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) warp_bgl: desc::Bindings,
    pub(in crate::gpu::stroke) warp_apply_pipeline: wgpu::ComputePipeline,
    pub(in crate::gpu::stroke) warp_apply_bgl: desc::Bindings,
    /// The deposit's prefix-τ volume binding (group 1) — the same texture the
    /// swept fast path samples, so the exchange extent *is* the definite
    /// integral of the brush along the travel (compute-visible variant).
    pub(in crate::gpu::stroke) prefix_bgl: desc::Bindings,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    pub(in crate::gpu::stroke) exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back: the aux narrow pass. Color and residual leave
    // the region as plain texture copies (`DynamicsRun::write_back`), so the one
    // pipeline the write-back keeps is the narrowing of the wide region aux to the
    // persistent height channel — once over the whole region, not once per tile.
    pub(in crate::gpu::stroke) slice_pipeline: wgpu::RenderPipeline,
    pub(in crate::gpu::stroke) slice_bgl: desc::Bindings,
}

/// Build the brush-dynamics stamp-loop kit (§6.2, §6.13): the region
/// composite, the loop's compute pipelines — the warp among them — and the
/// region→tile slice.
pub(in crate::gpu::stroke) fn build_dynamics_kit(
    ctx: &crate::gpu::context::GpuContext,
    color_space: &dyn ColorSpace,
    composite_tile_bgl: desc::Bindings,
) -> DynamicsKit {
    let device = &ctx.device;
    // The loop stores a tile's color through `region_color_w` and copies the region
    // back into tiles texel-for-texel, which is legal only where the space's tile
    // format *is* the format that slot declares. Both spaces use `rgba16float` (§6.7),
    // so either can hold the region. Compared against the shader's own declaration
    // rather than a literal (§6.10), so either side drifting is caught.
    debug_assert_eq!(
        color_space.color_format(),
        sd::REGION_COLOR_W.storage_format(),
        "the loop stores tile color through `region_color_w`; this space's tiles are not that format",
    );
    // ---- Region composite: the `composite` shader over region-sized targets
    // (color + the wide aux, so nothing is narrowed until the write-back).
    let composite = stark_shaders::composite(color_space.resid());
    let composite_shader = desc::Module::new(device, "stark dynamics composite", composite);
    // Pass A's own tile layout, because the group this loop binds per tile is the one
    // the tile itself caches (`composite::tile_bind_group_layout`). The view group has
    // no such cache, so it is built here off the two stages this loop runs — which
    // composite their working region through `composite.wesl` itself (§6.3).
    let composite_view_bgl = desc::Bindings::of(
        device,
        "stark dynamics composite view bgl",
        stark_shaders::Stages::Render(composite.vs_main, composite.fs_raw),
        stark_shaders::mirror::view::decl::VIEW,
        &[stark_shaders::mirror::view::decl::VIEW],
    );
    let composite_layout = desc::pipeline_layout_of(
        device,
        "stark dynamics composite layout",
        &[&composite_view_bgl, &composite_tile_bgl],
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
            vs: composite.vs_main,
            // `fs_raw`, NOT the screen path's `fs_main`: the loop's region must hold
            // the tile representation itself (opacity in alpha), not the
            // coverage-weighted channels pass A shows — the exchange reads this
            // region and the slice writes it back to persistent tiles.
            fs: composite.fs_raw,
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

    // ---- The stamp loop: one module with eight entry points, and the liquify
    // field's module beside it with three more (§6.13), over as many bind group
    // layouts.
    let dynamics = stark_shaders::dynamics(color_space.resid());
    let module = desc::Module::new(device, "stark dynamics loop", dynamics);
    let liquify = stark_shaders::liquify(color_space.resid());
    let liquify_module = desc::Module::new(device, "stark liquify field", liquify);
    // Every layout below is one kernel's group 0, **exact** rather than shared: these
    // dispatches disagree about the region, which one samples and the next
    // storage-writes (`layout_shared_by`, §6.10). `ST`, the dynamic-offset stamp slot,
    // anchors the group for all of them — both modules take that declaration from
    // `dynamics_common.wesl`.
    let bgl = |label: &str, kernel| {
        desc::Bindings::of(device, label, Stages::Compute(kernel), sd::ST, &[sd::ST])
    };
    let snapshot_bgl = bgl("stark dynamics snapshot bgl", dynamics.snapshot);
    let bleed_weight_bgl = bgl("stark dynamics bleed weight bgl", dynamics.bleed_weight);
    let exchange_bgl = bgl("stark dynamics exchange bgl", dynamics.exchange);
    let bake_bgl = bgl("stark dynamics bake bgl", dynamics.bake);
    let settle_bgl = bgl("stark dynamics settle bgl", dynamics.settle);
    let deposit_bgl = bgl("stark dynamics deposit bgl", dynamics.deposit);
    let snapshot_field_bgl = bgl("stark dynamics snapshot field bgl", liquify.snapshot_field);
    let warp_bgl = bgl("stark dynamics warp bgl", liquify.warp);
    let warp_apply_bgl = bgl("stark dynamics warp apply bgl", liquify.warp_apply);
    let hoist_bgl = bgl("stark dynamics cell hoist bgl", dynamics.cell_hoist);
    let deposit_coarse_bgl = bgl("stark dynamics deposit coarse bgl", dynamics.deposit_coarse);
    // The prefix-τ volume at group 1 — the very texture the fast path samples, so the
    // exchange extent *is* the definite integral of the brush along the travel (§6.6).
    // Shared safely: one read-only texture, which no sharer can disagree about.
    let prefix_bgl = desc::Bindings::shared_by(
        device,
        "stark dynamics prefix bgl",
        &[
            Stages::Compute(dynamics.bleed_weight),
            Stages::Compute(dynamics.bake),
            Stages::Compute(dynamics.deposit),
            Stages::Compute(dynamics.cell_hoist),
            Stages::Compute(dynamics.settle),
            Stages::Compute(liquify.warp),
        ],
        scd::PREFIX_TEX,
        &[],
    );
    let pipe_in = |module: &desc::Module, label: &str, entry, bindings: &[&desc::Bindings]| {
        let layout = desc::pipeline_layout_of(device, label, bindings);
        desc::compute_pipeline(device, label, &layout, module, entry)
    };
    let cpipe =
        |label: &str, entry, bindings: &[&desc::Bindings]| pipe_in(&module, label, entry, bindings);
    let lpipe = |label: &str, entry, bindings: &[&desc::Bindings]| {
        pipe_in(&liquify_module, label, entry, bindings)
    };
    let snapshot_pipeline = cpipe(
        "stark dynamics snapshot",
        dynamics.snapshot,
        &[&snapshot_bgl],
    );
    // The bleed ladder's mobility, hoisted (§6.2). It reads the prefix-τ volume, so it
    // takes group 1 like every other pass that does — one `swept_pre` per texel is the
    // whole of it.
    let bleed_weight_pipeline = cpipe(
        "stark dynamics bleed weight",
        dynamics.bleed_weight,
        &[&bleed_weight_bgl, &prefix_bgl],
    );
    let exchange_pipeline = cpipe(
        "stark dynamics exchange",
        dynamics.exchange,
        &[&exchange_bgl],
    );
    // The bake reads the prefix-τ volume too (group 1) — the exposure weights in
    // its integral are that volume's own differences.
    let bake_pipeline = cpipe(
        "stark dynamics bake",
        dynamics.bake,
        &[&bake_bgl, &prefix_bgl],
    );
    let deposit_pipeline = cpipe(
        "stark dynamics deposit",
        dynamics.deposit,
        &[&deposit_bgl, &prefix_bgl],
    );
    // The hoist takes the same prefix-τ taps the deposit's front half did; the coarse
    // deposit takes none, so its layout stops at group 0.
    let hoist_pipeline = cpipe(
        "stark dynamics cell hoist",
        dynamics.cell_hoist,
        &[&hoist_bgl, &prefix_bgl],
    );
    let deposit_coarse_pipeline = cpipe(
        "stark dynamics deposit coarse",
        dynamics.deposit_coarse,
        &[&deposit_coarse_bgl],
    );
    // The settle reads the prefix-τ volume too (group 1): its exposure is a pair of
    // readings of it, which is what makes the pen-up fade over the whole tip rather
    // than over the few pixels of its coverage knee.
    let settle_pipeline = cpipe(
        "stark dynamics settle",
        dynamics.settle,
        &[&settle_bgl, &prefix_bgl],
    );
    // The liquify field's kernels (§6.13). The composition reads its exposure
    // from a prefix volume at group 1 like every deposit — the **coverage**
    // prefix, which the liquify path binds there in the prefix-τ's place; the
    // snapshot and the resample read no tip at all.
    let snapshot_field_pipeline = lpipe(
        "stark liquify snapshot field",
        liquify.snapshot_field,
        &[&snapshot_field_bgl],
    );
    let warp_pipeline = lpipe(
        "stark liquify warp",
        liquify.warp,
        &[&warp_bgl, &prefix_bgl],
    );
    let warp_apply_pipeline = lpipe(
        "stark liquify warp apply",
        liquify.warp_apply,
        &[&warp_apply_bgl],
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
    let slice = stark_shaders::slice();
    let slice_shader = desc::Module::new(device, "stark dynamics slice", slice);
    let slice_bgl = desc::Bindings::of(
        device,
        "stark dynamics slice bgl",
        Stages::Render(slice.vs_main, slice.fs_main),
        sld::REGION_AUX,
        &[],
    );
    let slice_layout =
        desc::pipeline_layout_of(device, "stark dynamics slice layout", &[&slice_bgl]);
    let slice_pipeline = desc::fullscreen_pipeline(
        device,
        "stark dynamics slice pipeline",
        &slice_layout,
        &slice_shader,
        (slice.vs_main, slice.fs_main),
        &[desc::target(color_space.aux_format())],
    );

    // The 1×1 a painting segment binds where a firing binds the real scratch (§6.8's
    // stand-in pattern): such a slot carries `lambda_bleed = 0` and never reads it.
    let bleed_placeholder = desc::zero_texture(
        ctx,
        d::BLEED_W_W.storage_format(),
        "stark dynamics bleed w 1x1",
    );
    // The same stand-in for the ceiling lane (§6.2): a stroke whose ceiling the
    // pen does not drive carries `ceiling_lane = 0` and never reads it.
    let levels_placeholder = desc::zero_texture(
        ctx,
        crate::gpu::stroke::swept::CEILING_FORMAT,
        "stark dynamics levels 1x1",
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
        levels_placeholder,
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
        snapshot_field_pipeline,
        snapshot_field_bgl,
        warp_pipeline,
        warp_bgl,
        warp_apply_pipeline,
        warp_apply_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        prefix_bgl,
    }
}
