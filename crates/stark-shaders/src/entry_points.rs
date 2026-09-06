// The WESL modules that become standalone WGSL artifacts — the single list.
//
// `include!`d by **both** `build.rs` (which compiles each one) and `lib.rs` (which
// embeds each one), so there is no second list for either to disagree with. Adding a
// shader to one and not the other fails at the wrong layer: a missing `include_wesl!`
// artifact is a build-script-ordering error several frames from the shader you just
// wrote.
//
// Everything *not* in this list is a module reached only by import — the binding-free
// leaves under `shaders/lib/`, and the binding-owning shared modules
// (`blend_common`, `dynamics_common`, `media_common`, `stamp_common`, `mixbox_lut`).
// Those have no entry point of their own and would fail to link as a root.

/// Every WESL module compiled to its own WGSL artifact, by module name.
///
/// Kept sorted, which is not cosmetic: the pipeline this list drives is a `for` loop,
/// so the order decides the order build errors substrate in, and an alphabetical list is
/// the one a reader can check for a missing entry at a glance.
pub const ENTRY_POINTS: &[&str] = &[
    "blend_mixbox",
    "blend_oklab",
    "blur",
    "composite",
    "dynamics",
    "erase",
    "fill",
    "filter_mixbox",
    "filter_oklab",
    "guides",
    "integrate",
    "liquify",
    "mask_region",
    "matte",
    "media_mixbox",
    "media_oklab",
    "merge",
    "overlay",
    "resolve",
    "selection",
    "slab",
    "slice",
    "stamp",
    "transform",
];

/// The entries of [`ENTRY_POINTS`] that exist only in a build carrying the `mixbox`
/// cargo feature — the shaders that evaluate the pigment polynomial (§6.7).
///
/// They are the only importers of `package::gen::mixbox_poly` and
/// `package::mixbox_lut`, so leaving them unbuilt is what unlinks the vendored
/// CC BY-NC 4.0 code entirely rather than merely leaving it unreachable.
pub const MIXBOX_ENTRY_POINTS: &[&str] = &["blend_mixbox", "filter_mixbox", "media_mixbox"];

/// Whether `name` is built in a configuration whose `mixbox` feature is `mixbox`.
///
/// One predicate with two callers, because the two sides of this crate learn about
/// the feature differently and **must not** be allowed to disagree: `lib.rs` reads
/// `cfg(feature = "mixbox")`, while `build.rs` — a build script is compiled without
/// the crate's own features, so it has no such cfg — reads the `CARGO_FEATURE_MIXBOX`
/// environment variable cargo sets when it runs. A second list on either side would
/// be a way for an accessor to exist with no artifact behind it, which is exactly the
/// link-time failure [`ENTRY_POINTS`] is one list to prevent.
pub fn entry_point_enabled(name: &str, mixbox: bool) -> bool {
    mixbox || !MIXBOX_ENTRY_POINTS.contains(&name)
}

/// The WESL conditional-compilation feature that turns on a tile's **residual**
/// channel — the third color texture a pigment space needs (§6.7).
pub const RESID_FEATURE: &str = "resid";

/// The WESL feature that turns on the sweep's **ceiling lane** (§6.2): the
/// target a stroke whose opacity the pen drives accumulates its claimed coverage
/// into, beside the parcel.
pub const CEILING_FEATURE: &str = "ceiling";

/// The subset of [`ENTRY_POINTS`] built again with [`CEILING_FEATURE`] on,
/// deposited as `<module>_ceiling` — and, in a residual build, once more as
/// `<module>_resid_ceiling`.
///
/// Only the sweep. The lane is a render *target*, and a pipeline's target list is
/// fixed when the pipeline is, so the sweep that writes it is a second pipeline
/// over a second artifact. The passes that *read* the lane (`integrate`, `erase`,
/// `dynamics`) declare its slot unconditionally and bind a 1×1 zero behind a
/// uniform flag when there is none — a flag costs those passes nothing, where a
/// fourth attachment on every unmodulated stroke would cost the sweep two bytes a
/// fragment.
pub const CEILING_ENTRY_POINTS: &[&str] = &["stamp"];

/// The subset of [`ENTRY_POINTS`] that also gets built a *second* time with
/// [`RESID_FEATURE`] enabled, deposited as `<module>_resid`.
///
/// These are the passes that carry a tile's color, and a residual goes wherever a
/// latent goes — it is the same premultiplied "over" on the same coverage, so every
/// one of them does to `resid` exactly what it already does to `color` (§6.7).
///
/// A **variant** rather than one shader that always carries the channel, because
/// Oklab has no residual to carry: its three channels reproduce every sRGB color
/// exactly, so a third target there would be eight bytes per texel of zeroes written
/// on the default space's hot path. `@if(resid)` is what keeps the two laws in one
/// file instead of a `*_resid.wesl` beside each of these.
///
/// `blend_mixbox` and `media_mixbox` are **not** here and need no WESL feature: they
/// are reached only by the space that has a residual, so they declare the extra
/// binding unconditionally — the trick `mixbox_lut.wesl` already plays on the blend
/// group. They are gated by the *cargo* feature instead
/// ([`MIXBOX_ENTRY_POINTS`]), which is a different question: whether the shader is
/// built at all, rather than which of its two variants this is.
///
/// **This whole pass is skipped without the `mixbox` cargo feature.** A residual is
/// something a *pigment* space has, Mixbox is the only one, and `resid_format()` is
/// then `None` for every space in the build — so the variants would be seven artifacts
/// nothing could ever select. `build.rs` skips the pass and `lib.rs` compiles the
/// `resid`-taking accessors down to their plain arm.
///
/// `slice` is **not** here: since the write-back's color and residual became plain
/// texture copies (§6.2), the one pass left under that name narrows the wide region
/// aux, which every color space has and neither varies.
///
/// Kept sorted, like [`ENTRY_POINTS`] and for the same reason.
pub const RESID_ENTRY_POINTS: &[&str] = &[
    "composite",
    "dynamics",
    "erase",
    "fill",
    "integrate",
    "liquify",
    "matte",
    "merge",
    "slab",
    "stamp",
    "transform",
];
