//! The body of `stark-shaders`'s build script (§6.10): one read of the WESL tree,
//! two products.
//!
//! * The **artifacts** — each entry point linked with its imports, checked, and
//!   deposited as WGSL in `OUT_DIR` (`link`, `collide`, `check`).
//! * The **mirrors** — the Rust structs, constants, binding tables and vertex layouts
//!   the tree's own declarations describe (`emit`).
//!
//! It is a library because a build script's `#[cfg(test)]` is never compiled, so none
//! of this could be tested where it used to live. Failures are still reported by
//! panicking: the caller is a build script, and a panic there is its diagnostic.

mod check;
mod collide;
mod docs;
mod emit;
mod eval;
mod layout;
mod link;
mod mixbox_poly;
mod tree;

use std::path::Path;

/// The module prefix generated WESL is mounted under. Importers say
/// `package::gen::mixbox_poly`, so the import site itself says "build-time generated".
const GEN_PREFIX: &str = "package::gen";

/// What the build script tells the generator about its own crate.
///
/// Everything here is either a path cargo decides or a fact the shader tree cannot
/// state about itself; nothing that a `.wesl` file already says (§6.10).
#[derive(Debug, Clone, Copy)]
pub struct Config<'a> {
    /// The shader tree's root, relative to the crate being built.
    pub shader_dir: &'a Path,
    /// Cargo's `OUT_DIR`. Everything generated lands here and nowhere else — an input
    /// directory this script also wrote into would entangle the two fingerprints, and
    /// cargo could then call the artifacts fresh after a shader edit.
    pub out_dir: &'a Path,
    /// The vendored Mixbox GLSL, when this build carries the pigment space (§6.7).
    ///
    /// `None` skips the transpile, leaves `package::gen` unmounted, and skips the whole
    /// residual pass — a residual is something a *pigment* space has, Mixbox is the
    /// only one, and the variants would be artifacts nothing could select.
    ///
    /// This spelling of the path is quoted in the generated WESL's provenance header,
    /// so pass the relative one: an absolute path would make the artifact differ
    /// between machines.
    pub mixbox_glsl: Option<&'a Path>,
    /// The modules to link, each deposited under its own name. Already filtered to
    /// what this feature set builds.
    pub entry_points: &'a [&'a str],
    /// Linked a second time with [`Config::resid_feature`] on, as `<module>_resid`.
    pub resid_entry_points: &'a [&'a str],
    /// Linked again with [`Config::ceiling_feature`] on, as `<module>_ceiling` — and,
    /// in a residual build, once more as `<module>_resid_ceiling`.
    pub ceiling_entry_points: &'a [&'a str],
    /// The WESL conditional-compilation feature gating a tile's residual channel.
    pub resid_feature: &'a str,
    /// The WESL feature gating the sweep's ceiling lane.
    pub ceiling_feature: &'a str,
    /// The structs **two or more modules declare identically** against one host type,
    /// as `(modules, struct)`.
    ///
    /// The first module is generated from and names the Rust module; the rest are
    /// checked to agree member for member and offset for offset, then skipped so
    /// discovery does not emit a second copy. Every *other* uniform struct is
    /// discovered — this is only about which declarations claim to be one declaration.
    pub shared: &'a [(&'a [&'a str], &'a str)],
}

/// Generate the mirrors and deposit every artifact.
///
/// # Panics
/// On anything the shader tree gets wrong — a module that will not parse, link,
/// typecheck or lay out, two modules claiming one binding slot, a vertex entry point
/// whose record has no name. The message names the declaration.
pub fn run(cfg: &Config<'_>) {
    let gen_dir = cfg
        .mixbox_glsl
        .map(|glsl| mixbox_poly::generate(glsl, cfg.out_dir));

    // Read from the *unlinked* sources: the linker mangles `Stamp` to
    // `package__1dynamics_common_Stamp`, emits it once per artifact that reaches it,
    // and has already stripped whatever no entry point uses.
    emit::generate(cfg.shader_dir, &cfg.out_dir.join("mirror.rs"), cfg.shared);

    link::compile_all(cfg, gen_dir.as_deref());
}
