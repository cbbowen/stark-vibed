//! The stamp loop's GPU objects, built once (§6.2).
//!
//! Pipelines, bind group layouts and samplers — nothing that varies with a stroke, a
//! piece or a segment, all of which is [`plan`](super::plan)'s business or
//! [`run`](super::run)'s. Immutable throughout, which is what lets the kit be cloned
//! with its renderer and live in an `Action::Context` (§5).

use crate::colorspace::ColorSpace;
use crate::device::channels::{ChannelFormats, TILE_AUX_FORMAT, TILE_COLOR_FORMAT};
use crate::device::desc;
use crate::device::tile::SCRATCH_AUX_FORMAT;
use stark_shaders::mirror::dynamics::decl as d;
use stark_shaders::mirror::dynamics_common::decl as sd;
use stark_shaders::mirror::slice::decl as sld;
use stark_shaders::mirror::stamp_common::decl as scd;
use stark_shaders::{EntryPoint, Stages};

/// **One texture, three declarations.** A region's channels are filled by the composite
/// below — whose targets are [`ChannelFormats`] over [`SCRATCH_AUX_FORMAT`] — then
/// storage-written by the loop through `region_*_w`, then copied texel-for-texel into
/// tiles. A texture has one format, so all of that has to name one, and nothing at run
/// time would say otherwise: `create_texture` takes whichever the host passed.
///
/// `wgpu::TextureFormat` has no `const` equality — `PartialEq` is not a `const` trait,
/// and the enum carries payloads, so it cannot be cast to compare — so the declarations
/// are pinned against one pattern rather than against each other. Either side moving
/// fails this.
const _: () = {
    const fn region_format(f: wgpu::TextureFormat) -> bool {
        matches!(f, wgpu::TextureFormat::Rgba16Float)
    }
    assert!(
        region_format(sd::REGION_COLOR_W.storage_format())
            && region_format(sd::REGION_AUX_W.storage_format())
            && region_format(sd::REGION_RESID_W.storage_format())
            && region_format(TILE_COLOR_FORMAT)
            && region_format(SCRATCH_AUX_FORMAT),
        "the region composite writes the trio at `TILE_COLOR_FORMAT`/`SCRATCH_AUX_FORMAT` \
         and the loop storage-writes the same textures through `region_*_w`",
    );
};

/// Whether `entry` reads the prefix volume, and so binds it at group 1.
///
/// **The one statement of it** (§6.10). The shader says which kernels take a tip, and
/// this used to be written down three more times besides: in the `shared_by` list the
/// prefix layout was built over, in each pipeline's own layout slice, and at every
/// `set_bind_group(1, …)` in the two record loops.
fn reads_prefix(entry: EntryPoint) -> bool {
    entry.reads(scd::PREFIX_TEX)
}

/// One compute kernel of the loop: the pipeline, the group-0 layout every group over
/// it is built from, and the entry point both came from.
#[derive(Clone)]
pub(super) struct Kernel {
    pipeline: wgpu::ComputePipeline,
    /// The kernel's group 0, **exact** rather than shared: these dispatches disagree
    /// about the region, which one samples and the next storage-writes
    /// (`layout_shared_by`, §6.10). `ST`, the dynamic-offset stamp slot, anchors it —
    /// both modules take that declaration from `dynamics_common.wesl`.
    group0: desc::Bindings,
    entry: EntryPoint,
}

impl Kernel {
    /// Build `entry` over `module`: its own group 0, and `prefix` at group 1 exactly
    /// where [`reads_prefix`] says so.
    fn build(
        device: &wgpu::Device,
        module: &desc::Module,
        label: &str,
        entry: EntryPoint,
        prefix: &desc::Bindings,
    ) -> Self {
        let group0 = desc::Bindings::of(
            device,
            &format!("{label} bgl"),
            Stages::Compute(entry),
            sd::ST,
            &[sd::ST],
        );
        let groups: &[&desc::Bindings] = if reads_prefix(entry) {
            &[&group0, prefix]
        } else {
            &[&group0]
        };
        let layout = desc::pipeline_layout_of(device, &format!("{label} layout"), groups);
        Self {
            pipeline: desc::compute_pipeline(device, label, &layout, module, entry),
            group0,
            entry,
        }
    }

    /// A bind group over this kernel's group 0, `resource` asked for exactly the
    /// entries the layout declares ([`desc::Bindings::group`]).
    pub(super) fn group<'a>(
        &self,
        device: &wgpu::Device,
        label: &str,
        resource: impl FnMut(u32) -> wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        self.group0.group(device, label, resource)
    }

    /// The workgroup counts covering `extent` at this kernel's declared grid (§6.10) —
    /// for the one dispatch that lays two grids side by side.
    pub(super) fn groups(&self, extent: (u32, u32)) -> (u32, u32) {
        self.entry.groups(extent)
    }

    /// Dispatch this kernel over `extent`, at the grid it declares (§6.10), with
    /// `group0` at its own group and `slot`'s dynamic offset.
    ///
    /// `extent` is whatever this kernel's `@workgroup_size` is counted over: texels for
    /// every kernel but `cell_hoist`, whose grid is cells.
    pub(super) fn dispatch(
        &self,
        at: &mut LoopPass<'_>,
        group0: &wgpu::BindGroup,
        slot: u32,
        extent: (u32, u32),
    ) {
        self.dispatch_groups(at, group0, slot, self.groups(extent));
    }

    /// [`dispatch`](Self::dispatch) over a grid the caller composed — the `exchange`'s,
    /// which lays the segment's extent beside the reservoir's on x
    /// (`dynamics.wesl::exchange`).
    pub(super) fn dispatch_groups(
        &self,
        at: &mut LoopPass<'_>,
        group0: &wgpu::BindGroup,
        slot: u32,
        groups: (u32, u32),
    ) {
        at.pass.set_pipeline(&self.pipeline);
        self.group0.set(&mut at.pass, group0, &[slot]);
        // Re-bound per dispatch because WebGPU requires it, not out of habit: setting a
        // pipeline whose layout is not compatible with the last one's at group 0 leaves
        // every group above it unset, and each of these kernels is reached only across
        // such a switch.
        if reads_prefix(self.entry) {
            at.prefix.bgl.set(&mut at.pass, at.prefix.group, &[]);
        }
        at.pass.dispatch_workgroups(groups.0, groups.1, 1);
    }
}

/// The prefix volume as a dispatch binds it: the shared layout, which says which
/// `@group` it stands at, and the run's group over it.
#[derive(Clone, Copy)]
pub(super) struct Prefix<'a> {
    pub(super) bgl: &'a desc::Bindings,
    pub(super) group: &'a wgpu::BindGroup,
}

/// A compute pass the loop's kernels are recorded into, carrying the [`Prefix`] the
/// ones that read a tip take at group 1 — so a dispatch is one call and binding the
/// prefix is not something a call site can forget.
pub(super) struct LoopPass<'a> {
    pass: wgpu::ComputePass<'a>,
    prefix: Prefix<'a>,
}

impl<'a> LoopPass<'a> {
    pub(super) fn open(
        encoder: &'a mut wgpu::CommandEncoder,
        label: &'static str,
        prefix: Prefix<'a>,
    ) -> Self {
        Self {
            pass: encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            }),
            prefix,
        }
    }
}

/// The region machinery's own objects (`region.rs`): the composite that fills a region
/// from base tiles, and the narrow pass that slices its aux back out.
#[derive(Clone)]
pub(super) struct RegionKit {
    pub(super) composite_pipeline: wgpu::RenderPipeline,
    pub(super) composite_view_bgl: desc::Bindings,
    pub(super) composite_tile_bgl: desc::Bindings,
    pub(super) composite_sampler: wgpu::Sampler,
    /// Region → CoW tile write-back: the aux narrow pass. Color and residual leave the
    /// region as plain texture copies (`slice_region`), so the one pipeline the
    /// write-back keeps is the narrowing of the wide region aux to the persistent
    /// height channel — once over the whole region, not once per tile.
    pub(super) slice_pipeline: wgpu::RenderPipeline,
    pub(super) slice_bgl: desc::Bindings,
}

/// The wet loop's kernels (§6.2) and the two stand-ins its groups bind where a stroke
/// has no scratch for them.
#[derive(Clone)]
pub(super) struct WetKernels {
    /// The extent copy that gives the `deposit`/`settle` something to read while they
    /// storage-write the region.
    ///
    /// A painting segment does not dispatch it — its snapshot rides in the tail of its
    /// own `exchange` grid, depending on nothing that pass writes
    /// (`dynamics.wesl::exchange`). Only the two steps with no exchange to ride in,
    /// [`WetStep::Bleed`](super::plan::WetStep) and [`WetStep::Settle`](super::plan::WetStep),
    /// dispatch it standalone.
    pub(super) snapshot: Kernel,
    /// The bleed pair's mobility pass (§6.2).
    pub(super) bleed_weight: Kernel,
    /// The tool's own side of one segment's transfer — the complement of every share
    /// the `deposit` after it hands the canvas (`dynamics.wesl::exchange`).
    pub(super) exchange: Kernel,
    /// Integrates the reservoir along the segment's travel axis so the deposit can read
    /// the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(super) bake: Kernel,
    pub(super) deposit: Kernel,
    /// The **coarse deposit** pair (§6.2), for the slots whose tip's shoulder lets the
    /// exchange be evaluated per cell instead of per texel (`budget::extent_cell`):
    /// `cell_hoist` distils the prefix and the bake into per-cell means,
    /// `deposit_coarse` reads them back over the exact kernel's own texel grid. A slot
    /// with a cell of 1 touches neither and keeps [`deposit`](Self::deposit)
    /// bit-for-bit.
    pub(super) hoist: Kernel,
    pub(super) deposit_coarse: Kernel,
    /// The pen-up: settles the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`). Reads the reservoir through its own
    /// `bake` dispatch — the zero-travel slot bakes the *remaining pass's* delivery
    /// integral, not a per-segment window — never the cell that sits overhead.
    pub(super) settle: Kernel,
    /// What a **painting** segment's deposit binds where a firing binds the scratch: a
    /// 1×1 zero. Such a slot carries `lambda_bleed = 0` and never reads it, so this is
    /// the §6.8 stand-in pattern rather than a case the shader has to branch on.
    pub(super) bleed_placeholder: wgpu::TextureView,
    /// The ceiling lane's 1×1 zero (§6.2), bound where a stroke has no lane.
    pub(super) levels_placeholder: wgpu::TextureView,
    /// Bilinear clamp sampler for the region and reservoir lookups.
    pub(super) sampler: wgpu::Sampler,
}

/// The liquify field's three kernels (§6.13, `liquify.wesl`): the field's snapshot
/// under a segment's square, the composition of one segment's step into it, and the one
/// resample of a piece through it. Only the composition takes group 1, bound to the
/// tip's coverage prefix.
#[derive(Clone)]
pub(super) struct WarpKernels {
    pub(super) snapshot_field: Kernel,
    pub(super) warp: Kernel,
    pub(super) apply: Kernel,
}

/// GPU objects for the brush-dynamics stamp loop (§6.2), built once. All handles are
/// `Arc`-backed, so the kit is cheap to clone with its renderer.
///
/// **Immutable throughout**: no cache lives here. The lazily-baked brush textures sit
/// on the renderer instead ([`TipCache::round_tip`](super::super::tips::TipCache)).
#[derive(Clone)]
pub(in crate::gpu::stroke) struct DynamicsKit {
    pub(super) region: RegionKit,
    pub(super) wet: WetKernels,
    pub(super) warp: WarpKernels,
    /// The prefix volume's layout (group 1) — the deposit's prefix-τ, the same texture
    /// the swept fast path samples, so the exchange extent *is* the definite integral
    /// of the brush along the travel; and, on the liquify path, the coverage prefix
    /// bound in its place (§6.13).
    ///
    /// Shared safely by every kernel that reads it: one read-only texture, which no
    /// sharer can disagree about.
    pub(super) prefix_bgl: desc::Bindings,
}

/// Build the brush-dynamics stamp-loop kit (§6.2, §6.13): the region composite, the
/// loop's compute pipelines — the warp among them — and the region→tile slice.
pub(in crate::gpu::stroke) fn build_dynamics_kit(
    ctx: &crate::device::context::GpuContext,
    color_space: &dyn ColorSpace,
    composite_tile_bgl: desc::Bindings,
) -> DynamicsKit {
    let device = &ctx.device;
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
        Stages::Render(composite.vs_main, composite.fs_raw),
        stark_shaders::mirror::view::decl::VIEW,
        &[stark_shaders::mirror::view::decl::VIEW],
    );
    let composite_layout = desc::pipeline_layout_of(
        device,
        "stark dynamics composite layout",
        &[&composite_view_bgl, &composite_tile_bgl],
    );
    // The deposit's blends, over the *wide* scratch aux: nothing is narrowed until the
    // write-back.
    let composite_targets = ChannelFormats::of(color_space).deposit_targets(SCRATCH_AUX_FORMAT);
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
            targets: &desc::targets_for(composite.fs_raw, &composite_targets),
        },
    );
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- The stamp loop: one module with eight entry points, and the liquify field's
    // module beside it with three more (§6.13).
    let dynamics = stark_shaders::dynamics(color_space.resid());
    let module = desc::Module::new(device, "stark dynamics loop", dynamics);
    let liquify = stark_shaders::liquify(color_space.resid());
    let liquify_module = desc::Module::new(device, "stark liquify field", liquify);
    // The prefix volume's layout, over the kernels that read it — **filtered from the
    // whole list rather than listed** (§6.10), which is the same question
    // `Kernel::build` asks of each one.
    let every: [EntryPoint; 11] = [
        dynamics.snapshot,
        dynamics.bleed_weight,
        dynamics.exchange,
        dynamics.bake,
        dynamics.deposit,
        dynamics.cell_hoist,
        dynamics.deposit_coarse,
        dynamics.settle,
        liquify.snapshot_field,
        liquify.warp,
        liquify.warp_apply,
    ];
    let readers: Vec<Stages> = every
        .into_iter()
        .filter(|e| reads_prefix(*e))
        .map(Stages::Compute)
        .collect();
    let prefix_bgl = desc::Bindings::shared_by(
        device,
        "stark dynamics prefix bgl",
        &readers,
        scd::PREFIX_TEX,
        &[],
    );
    let wet_kernel = |label: &str, entry| Kernel::build(device, &module, label, entry, &prefix_bgl);
    let warp_kernel =
        |label: &str, entry| Kernel::build(device, &liquify_module, label, entry, &prefix_bgl);
    let wet = WetKernels {
        snapshot: wet_kernel("stark dynamics snapshot", dynamics.snapshot),
        bleed_weight: wet_kernel("stark dynamics bleed weight", dynamics.bleed_weight),
        exchange: wet_kernel("stark dynamics exchange", dynamics.exchange),
        bake: wet_kernel("stark dynamics bake", dynamics.bake),
        deposit: wet_kernel("stark dynamics deposit", dynamics.deposit),
        hoist: wet_kernel("stark dynamics cell hoist", dynamics.cell_hoist),
        deposit_coarse: wet_kernel("stark dynamics deposit coarse", dynamics.deposit_coarse),
        settle: wet_kernel("stark dynamics settle", dynamics.settle),
        // The 1×1 a painting segment binds where a firing binds the real scratch (§6.8's
        // stand-in pattern): such a slot carries `lambda_bleed = 0` and never reads it.
        bleed_placeholder: desc::zero_texture(
            ctx,
            d::BLEED_W_W.storage_format(),
            "stark dynamics bleed w 1x1",
        ),
        // The same stand-in for the ceiling lane (§6.2): a stroke whose ceiling the pen
        // does not drive carries `ceiling_lane = 0` and never reads it.
        levels_placeholder: desc::zero_texture(
            ctx,
            crate::gpu::stroke::swept::CEILING_FORMAT,
            "stark dynamics levels 1x1",
        ),
        sampler: device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark dynamics exchange sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
    };
    let warp = WarpKernels {
        snapshot_field: warp_kernel("stark liquify snapshot field", liquify.snapshot_field),
        warp: warp_kernel("stark liquify warp", liquify.warp),
        apply: warp_kernel("stark liquify warp apply", liquify.warp_apply),
    };

    // ---- Region → tile write-back: the aux narrow pass. The color and residual
    // channels are copied out of the region bit-exactly (`slice_region`), so this draws
    // once over the whole region rather than once per tile, and needs neither a per-tile
    // uniform nor a residual variant.
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
        &[desc::target(TILE_AUX_FORMAT)],
    );

    DynamicsKit {
        region: RegionKit {
            composite_pipeline,
            composite_view_bgl,
            composite_tile_bgl,
            composite_sampler,
            slice_pipeline,
            slice_bgl,
        },
        wet,
        warp,
        prefix_bgl,
    }
}
