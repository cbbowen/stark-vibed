//! Compiles WESL shader modules to WGSL at build time (§2).
//!
//! The work is `stark-shaders-build`'s. What is left here is what belongs to *this*
//! crate: the axes a shader is linked along twice, the one thing the shaders cannot
//! say about themselves, and the paths.

use std::path::{Path, PathBuf};

use stark_shaders_build::{Axis, RESID_FEATURE};

/// The axes a shader is linked along a **second** time, with a WESL feature on (§6.10).
///
/// This is the whole of what is declared; everything else about which artifacts exist
/// is read off the tree. What no shader can state is the host's side of an `@if`: the
/// feature has a name, but the type a caller picks a build with does not exist until it
/// is written here.
const AXES: &[Axis<'_>] = &[
    Axis {
        feature: "pigment",
        what: "the **working space** a color is stored in (§6.7) — three perceptual \
               channels, or a mixture of pigments plus what the polynomial cannot say \
               about it",
        ty: "Pigment",
        off: "Oklab",
        on: "Mixbox",
        pigment_only: true,
        // The artifacts keep the names the two spaces were separate files under, which
        // is what let three pairs of near-identical shaders become three modules
        // without renaming anything downstream.
        artifacts: Some(("oklab", "mixbox")),
        // The three passes bracketed by a color space's own conversion (§6.7): out to
        // light and back for the blend, out to Oklab and back for the filter, out to
        // display for the media pass. Nothing else in the tree converts — a tile's
        // channels ride every other pass unread.
        modules: &["blend", "filter", "media"],
    },
    Axis {
        // Named by the generator, which reads `@if(resid)` off a binding to fill
        // `Binding::resid` — one spelling, not two that happen to match.
        feature: RESID_FEATURE,
        what: "the **residual** channel a pigment space needs (§6.7) — the third color \
               texture, the half of a color its three channels cannot express",
        ty: "Resid",
        off: "Without",
        on: "With",
        pigment_only: true,
        artifacts: None,
        // Every pass that carries a tile's color: a residual goes wherever a latent
        // goes, being the same premultiplied "over" on the same coverage (§6.7).
        //
        // `blend`, `filter` and `media` are **not** here: they carry a residual on the
        // `pigment` axis above, which is the same texture answering a different
        // question — what a color *is* in this space, rather than whether the pass
        // that carries one needs a third target. `slice` is not here either — the one
        // pass left under that name narrows the region aux, which every space has and
        // neither varies.
        modules: &[
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
        ],
    },
    Axis {
        feature: "ceiling",
        what: "the sweep's **ceiling lane** (§6.2), the target a stroke whose opacity \
               the pen drives accumulates its claimed coverage into",
        ty: "Lane",
        off: "Plain",
        on: "Ceiling",
        pigment_only: false,
        artifacts: None,
        // Only the sweep. The lane is a render *target* and a pipeline's target list is
        // fixed when the pipeline is, so the sweep that writes it is a second pipeline
        // over a second artifact. The passes that *read* it (`integrate`, `erase`,
        // `dynamics`) declare its slot unconditionally and bind a 1×1 zero behind a
        // uniform flag when there is none — a flag costs them nothing, where a fourth
        // attachment on every unmodulated stroke would cost the sweep two bytes a
        // fragment.
        modules: &["stamp"],
    },
];

/// The vendored Mixbox shader (git submodule), source of the pigment-mixing
/// polynomial. Licensed CC BY-NC 4.0 — see `vendor/mixbox/LICENSE`.
const MIXBOX_GLSL: &str = "../../vendor/mixbox/shaders/mixbox.glsl";

/// Where the shader tree lives, relative to this crate.
const SHADER_DIR: &str = "src/shaders";

fn main() {
    // A build script is compiled *without* the crate's features, so `cfg(feature =
    // ..)` is not available here — cargo passes the answer in the environment instead.
    // It is the only thing this side says about the feature: which shaders and which
    // axes it takes away follows from the tree.
    let mixbox = std::env::var_os("CARGO_FEATURE_MIXBOX").is_some();
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    stark_shaders_build::run(&stark_shaders_build::Config {
        shader_dir: Path::new(SHADER_DIR),
        out_dir: &out_dir,
        mixbox_glsl: mixbox.then(|| Path::new(MIXBOX_GLSL)),
        axes: AXES,
    });

    // The tree, by directory: cargo scans a named directory recursively, so this
    // covers `lib/` and anything nested under it.
    println!("cargo::rerun-if-changed={SHADER_DIR}");
    // Only when it is actually read: naming a path that need not exist would make
    // cargo re-run this script on every build in a configuration without the
    // submodule checked out.
    if mixbox {
        println!("cargo::rerun-if-changed={MIXBOX_GLSL}");
    }
}
