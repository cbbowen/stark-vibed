//! GPU context: the wgpu handles the engine renders with (§7).
//!
//! Per CLAUDE.md the backend is *given* its wgpu resources by the frontend.
//! [`GpuContext::from_parts`] is that path; [`GpuContext::headless`] is a
//! convenience for tests and tools that need an offscreen device (§9).

use std::sync::{Arc, Mutex};

use crate::error::Result;

/// Why the GPU stopped being usable.
///
/// Kept as a kind plus the driver's own words rather than as a `wgpu::Error`, because
/// this outlives the callback that produced it and crosses into
/// [`ObservableState`](crate::ObservableState) — where it has to be `Clone`,
/// comparable, and free of any borrow of the device that just died.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// The device was lost: a driver reset, a GPU hot-unplug, a browser tab whose
    /// context went away, or `destroy()`. **Nothing will work again on this device**
    /// — recovery means building a new one.
    Lost,
    /// The device ran out of memory. Not necessarily terminal, but the operation
    /// that hit it produced nothing.
    OutOfMemory,
    /// A validation or internal error that no error scope caught. A bug in the
    /// engine rather than a fact about the machine, which is why it is reported
    /// rather than recovered from.
    Invalid,
}

/// A GPU failure the engine was told about out-of-band, with what the driver said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFailure {
    pub kind: FailureKind,
    /// The driver's message. For a human and for a bug report; nothing branches on it.
    pub detail: String,
}

impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            FailureKind::Lost => "the GPU device was lost",
            FailureKind::OutOfMemory => "the GPU ran out of memory",
            FailureKind::Invalid => "the GPU reported an error",
        };
        write!(f, "{what}: {}", self.detail)
    }
}

/// Whether the device is still usable, shared by every clone of a [`GpuContext`].
///
/// **The first failure is the one kept.** A lost device produces a cascade — every
/// subsequent operation fails too — and the first report is the one that says what
/// actually happened; the rest are its echoes.
#[derive(Clone, Default)]
pub struct GpuHealth(Arc<Mutex<Option<DeviceFailure>>>);

impl GpuHealth {
    /// The failure this device has suffered, if any.
    pub fn failure(&self) -> Option<DeviceFailure> {
        self.lock().clone()
    }

    /// Whether the device is still usable.
    pub fn is_ok(&self) -> bool {
        self.lock().is_none()
    }

    /// Record a failure, unless one is already recorded.
    fn report(&self, failure: DeviceFailure) {
        let mut slot = self.lock();
        if slot.is_none() {
            tracing::error!(failure = %failure, "GPU device failure");
            *slot = Some(failure);
        }
    }

    /// Poisoning cannot matter here: what is guarded is one `Option` that is only
    /// ever moved in whole, and the alternative — propagating another thread's panic
    /// as ours — would turn a reportable failure into an unreportable one.
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<DeviceFailure>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// **The largest 2D texture the engine will create, and therefore exactly what it
/// asks a device for** ([`GpuContext::minimum_required_limits`]).
///
/// It bounds the canvas substrate — a larger source is box-downsampled by an integer
/// factor, which preserves tileability — and the stamp loop's region, which is what
/// puts a ceiling on a brush's reach (`gpu::stroke::budget::max_tip_reach`, §6.2).
///
/// **It is the requirement rather than a reading of one.** The limit is *set* from
/// this constant rather than raised to meet it, so the number here and the number
/// the device was asked for cannot drift apart, and a build that raises it asks for
/// the bigger device instead of quietly creating a texture nobody promised.
///
/// What it costs is device breadth, and 8192 is where WebGPU itself sits: it is
/// `wgpu::Limits::default()`'s cap and the floor the WebGPU spec requires of a
/// conformant implementation, so every device Stark can run on already meets it.
/// The value was 2048 — the `downlevel_defaults()`/WebGL2 floor — which bought
/// portability to a backend this workspace does not build: no crate enables wgpu's
/// `webgl` feature, so a WebGL2 device was never going to be handed to the engine
/// anyway. What that 2048 was actually costing was the reach of a brush, since the
/// stamp loop's region is a texture (`gpu::stroke::budget::MAX_REGION_DIM`).
///
/// **A fixed constant, never a device query**, which is load-bearing beyond texture
/// allocation: the substrate downsample is part of the canonical form an asset id is
/// taken over (§19), so following the adapter's real limit would canonicalize the
/// same PNG differently on two machines and the id would stop naming one thing.
pub(crate) const MAX_TEXTURE_DIM_2D: u32 = 8192;

// The canonical-form caps that become **textures** have to fit inside what the device
// was asked for, and both are frozen one-way ratchets (§19) that a future raise could
// walk into this ceiling.
//
// `MAX_PICTURE_DIM` is deliberately absent rather than overlooked: a placed picture is
// built into tiles on the CPU and never bound as a texture at all, which is why it is
// allowed to be larger (§23).
const _: () = assert!(
    stark_assetid::MAX_SUBSTRATE_DIM <= MAX_TEXTURE_DIM_2D,
    "a substrate would not fit the texture limit the device was asked for",
);
const _: () = assert!(
    stark_assetid::MAX_SHAPE_DIM <= MAX_TEXTURE_DIM_2D,
    "a brush shape would not fit the texture limit the device was asked for",
);

/// The wgpu device, queue, and adapter the engine draws with.
///
/// `wgpu::Device` and `wgpu::Queue` are cheaply clonable (reference-counted),
/// so this struct is too.
#[derive(Clone)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Whether this device is still usable (§5).
    ///
    /// Shared by every clone, which is what makes one cell enough: the context is
    /// cloned into the tile pool, every renderer, the `ApplyCtx` and each preview
    /// engine, and all of them are talking to the same device.
    health: GpuHealth,
}

impl GpuContext {
    /// Wrap wgpu handles supplied by the frontend (CLAUDE.md).
    ///
    /// **Installs this crate's device callbacks**, which is not a courtesy — it is
    /// the only way the engine can find out that its device has died.
    /// `Action::Error` is `Infallible` on the stated substrates that "GPU work reports
    /// failure via wgpu's device error callbacks"; for a long time nothing installed
    /// one, so the sentence described a mechanism that did not exist and the first
    /// anyone knew of a lost device was an `expect` in the readback path — an abort,
    /// on the web, with the painting unsaved.
    ///
    /// A frontend that had installed its own handler will find it replaced. That is
    /// the right way round: the engine is what has to stop issuing work, and it
    /// publishes what it learns through
    /// [`ObservableState::gpu_failure`](crate::ObservableState::gpu_failure) so the
    /// frontend loses nothing by not owning the callback.
    pub fn from_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let health = GpuHealth::default();
        install_callbacks(&device, &health);
        Self {
            instance,
            adapter,
            device,
            queue,
            health,
        }
    }

    /// Whether this device is still usable, and what went wrong if not.
    ///
    /// **The action log survives what the device does not**, which is the whole
    /// reason this is worth reporting rather than panicking on: the document is a
    /// list of actions held in ordinary memory (§1), so a frontend told the device
    /// has gone can still save the file. Every path that would otherwise discover
    /// this by dying — the readback, the next submit — leaves the log untouched.
    pub fn health(&self) -> &GpuHealth {
        &self.health
    }

    pub fn minimum_required_limits() -> wgpu::Limits {
        // A conservative floor for everything the engine does not have an opinion
        // about; the two it does are set from that opinion below.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        // **Assigned, not raised to meet.** [`MAX_TEXTURE_DIM_2D`] is the largest
        // texture the engine will create, so it is also precisely what the device is
        // asked for — one number, and no way for the size we allocate and the size we
        // required to disagree. Written as `.max(…)` against a preset, the preset was
        // silently the real limit whenever it was the larger, and the constant was
        // documented as matching a `wgpu` default it had no way to keep matching.
        required_limits.max_texture_dimension_2d = MAX_TEXTURE_DIM_2D;
        // **The stamp loop's `exchange` writes six storage textures where WebGPU
        // guarantees four.**
        //
        // The four it always wrote — the extent snapshot's color and aux, and the
        // reservoir's color and aux, since the segment's `snapshot` rides in the tail
        // of that same dispatch (§6.2) — sit exactly on the downlevel limit, so the
        // residual channel's two (§6.7) put it over. This is the one limit Stark asks
        // for above the guaranteed floor for a *feature* rather than for canvas size,
        // and it is worth saying what that buys and what it costs.
        //
        // It is asked of every device, including one that will only ever open Oklab
        // documents, because limits are settled when the device is created and the
        // color space is a property of a document opened long after. Every adapter
        // Stark targets — D3D12, Vulkan, Metal, and WebGPU in Chrome — reports at
        // least eight; a conformant device reporting exactly four would fail to start
        // rather than fail to open a Mixbox file, which is the honest failure but not
        // a graceful one.
        //
        // The way back to four, if such a device ever turns up, is packing rather than
        // a second code path, and both halves of it are free: `brush_dst_aux_w` and
        // `under_aux_w` each carry height in `.x` and nothing in `.yzw`, so each one's
        // residual fits beside the height it belongs to. That is the whole excess —
        // no other entry point in the module declares more than three.
        //
        // Asked for only with the `mixbox` feature, which is the other way back under
        // four and the one that already exists: the residual belongs to a pigment
        // space, so a build without one declares no residual textures anywhere and
        // runs on WebGPU's guaranteed floor.
        #[cfg(feature = "mixbox")]
        {
            required_limits.max_storage_textures_per_shader_stage =
                required_limits.max_storage_textures_per_shader_stage.max(6);
        }
        required_limits
    }

    /// Create an offscreen context with no substrate, for headless rendering.
    pub async fn headless() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stark headless device"),
                required_features: wgpu::Features::empty(),
                required_limits: Self::minimum_required_limits(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        Ok(Self::from_parts(instance, adapter, device, queue))
    }
}

/// Point the device's two out-of-band failure reports at `health`.
///
/// Both, because they answer different questions and neither implies the other: an
/// uncaptured error is one operation that was refused (a validation bug, an
/// allocation that did not fit), while device-lost is the device itself going away
/// and every operation after it failing too.
///
/// This is the whole of what the engine can do about a GPU failure synchronously —
/// wgpu reports errors asynchronously and by callback, so there is no return value
/// anywhere in the submit path to check. What it buys is that the failure is *known*
/// rather than discovered by a later `expect`.
fn install_callbacks(device: &wgpu::Device, health: &GpuHealth) {
    let lost = health.clone();
    device.set_device_lost_callback(move |reason, detail| {
        lost.report(DeviceFailure {
            kind: FailureKind::Lost,
            detail: format!("{reason:?}: {detail}"),
        });
    });
    let uncaptured = health.clone();
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        let (kind, detail) = match &error {
            wgpu::Error::OutOfMemory { .. } => (FailureKind::OutOfMemory, error.to_string()),
            // Validation and internal errors are both bugs rather than facts about
            // the machine, and both mean the operation produced nothing. They are one
            // kind here because nothing downstream would act on the difference.
            wgpu::Error::Validation { .. } | wgpu::Error::Internal { .. } => {
                (FailureKind::Invalid, error.to_string())
            }
        };
        uncaptured.report(DeviceFailure { kind, detail });
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context, or `None` where the machine has no adapter *and*
    /// `STARK_ALLOW_NO_GPU=1` permits the skip — `gpu::submit`'s guard, for the same
    /// reason: a skipped GPU test still reports `ok`.
    fn context_or_skip() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::headless()) {
            Ok(ctx) => Some(ctx),
            Err(e) if std::env::var("STARK_ALLOW_NO_GPU").is_ok_and(|v| v == "1") => {
                eprintln!("skipping GPU test (STARK_ALLOW_NO_GPU=1): {e}");
                None
            }
            Err(e) => {
                panic!("no usable GPU adapter: {e}\nset STARK_ALLOW_NO_GPU=1 to skip GPU tests")
            }
        }
    }

    /// **A GPU error reaches the health cell instead of the default panic.**
    ///
    /// This is the test that could not be written at all before: `Action::Error` is
    /// `Infallible` on the grounds that "GPU work reports failure via wgpu's device
    /// error callbacks", and nothing installed one — so the first anyone knew of a
    /// failure was an `expect` in the readback path, which on the web is an abort that
    /// takes the unsaved painting with it (§5).
    ///
    /// Provoked with a **validation** error — a texture past the device's own limit —
    /// rather than by losing the device, and the difference is not squeamishness.
    /// `Device::destroy()` on a real adapter is a driver-level device removal, and on
    /// Windows the display compositor shares that adapter: an earlier version of this
    /// test took `dwm.exe` down with it on the machine that ran it. A test suite may
    /// not reach outside its own process, and nothing about the wiring under test
    /// needs it to — `on_uncaptured_error` and `set_device_lost_callback` are
    /// installed together by [`install_callbacks`], so proving one is delivered proves
    /// the handler is attached. What kind of failure arrives is wgpu's business.
    ///
    /// The device stays perfectly usable afterwards, which is the other half of
    /// choosing this error: a validation failure is a bug in the request, not a fact
    /// about the machine.
    #[test]
    fn a_gpu_error_reaches_the_health_cell() {
        let Some(ctx) = context_or_skip() else { return };
        assert!(ctx.health().is_ok(), "a fresh device is healthy");

        // One texel past what this device will allow. Refused by validation, and with
        // no error scope pushed it goes to the uncaptured handler — which, without
        // one installed, is wgpu's own panic.
        let over = ctx.device.limits().max_texture_dimension_2d + 1;
        let _refused = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark health test: deliberately oversized"),
            size: wgpu::Extent3d {
                width: over,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let failure = ctx.health().failure().expect(
            "an oversized texture is a validation error, and the uncaptured handler \
             should have recorded it",
        );
        assert_eq!(failure.kind, FailureKind::Invalid);

        // Reported once and kept: a failing device fails everything after it, and the
        // first report is the cause where the rest are its echoes.
        let _also_refused = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark health test: refused again"),
            size: wgpu::Extent3d {
                width: over,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        assert_eq!(
            ctx.health().failure(),
            Some(failure),
            "a later failure overwrote the first one",
        );
    }

    /// The cell's own rule, with no GPU in it: the **first** failure is the one kept.
    ///
    /// Separate from the test above because it is the part that has to hold on every
    /// machine, including one with no adapter at all — and because "which report
    /// survives" is a decision this module made, not something wgpu tells us.
    #[test]
    fn the_first_failure_is_the_one_kept() {
        let health = GpuHealth::default();
        assert!(health.is_ok());
        assert_eq!(health.failure(), None);

        let cause = DeviceFailure {
            kind: FailureKind::Lost,
            detail: "the cause".into(),
        };
        health.report(cause.clone());
        // Every operation after a lost device fails too, so the reports that follow
        // are symptoms. Keeping the first is what makes the message name the driver
        // reset rather than the buffer map that noticed it.
        health.report(DeviceFailure {
            kind: FailureKind::Invalid,
            detail: "an echo".into(),
        });
        assert!(!health.is_ok());
        assert_eq!(health.failure(), Some(cause));
    }
}
