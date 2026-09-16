//! The body of `stark-shaders`'s build script (§6.10): one read of the WESL tree,
//! three products.
//!
//! * The **artifacts** — each entry point linked with its imports, checked, and
//!   deposited as WGSL in `OUT_DIR` (`link`, `collide`, `check`).
//! * The **mirrors** — the Rust structs, constants, binding tables and vertex layouts
//!   the tree's own declarations describe (`emit`).
//! * The **accessors** — one Rust function per artifact, taking a typed parameter per
//!   axis the shader is linked along (`entries`, `accessors`).
//!
//! It is a library because a build script's `#[cfg(test)]` is never compiled, so none
//! of this could be tested where it used to live. Failures are still reported by
//! panicking: the caller is a build script, and a panic there is its diagnostic.

mod accessors;
mod check;
mod collide;
mod docs;
mod emit;
mod entries;
mod eval;
mod layout;
mod link;
mod mixbox_poly;
mod tree;

use std::path::Path;

pub use entries::{Axis, RESID_FEATURE};

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
    /// `None` leaves `package::gen` unmounted, which is what takes the shaders that
    /// import it out of the build — and with them every [`Axis::pigment_only`] axis.
    ///
    /// This spelling of the path is quoted in the generated WESL's provenance header,
    /// so pass the relative one: an absolute path would make the artifact differ
    /// between machines.
    pub mixbox_glsl: Option<&'a Path>,
    /// The axes a shader can be linked along a second time (§6.7, §6.2) — the one
    /// thing here the tree cannot say about itself. Which modules link at all, and
    /// which of them the pigment space owns, are both discovered (`entries`).
    pub axes: &'a [Axis<'a>],
}

/// Generate the mirrors and the accessors, and deposit every artifact.
///
/// # Panics
/// On anything the shader tree gets wrong — a module that will not parse, link,
/// typecheck or lay out, two modules claiming one binding slot, a vertex entry point
/// whose record has no name, a declared axis that changes nothing. The message names
/// the declaration.
pub fn run(cfg: &Config<'_>) {
    let gen_dir = cfg
        .mixbox_glsl
        .map(|glsl| mixbox_poly::generate(glsl, cfg.out_dir));

    // One read, and the *unlinked* sources: the linker mangles `Stamp` to
    // `package__1dynamics_common_Stamp`, emits it once per artifact that reaches it,
    // and has already stripped whatever no entry point uses — so both generators read
    // the tree as it is written, and the linker is the only thing that sees it linked.
    let pigment = cfg.mixbox_glsl.is_some();
    let modules = tree::read_tree(cfg.shader_dir);
    let entries = entries::discover(&modules, cfg.axes, pigment);

    emit::generate(&modules, &cfg.out_dir.join("mirror.rs"));
    accessors::generate(
        &entries,
        cfg.axes,
        pigment,
        &cfg.out_dir.join("accessors.rs"),
    );
    link::compile_all(cfg, &entries, gen_dir.as_deref());
}
