//! The type check `wgpu` would otherwise run on a GPU — and the answer it gives.

/// Parse and validate the linked WGSL the way `wgpu` will, keeping what naga worked
/// out.
///
/// **`wesl`'s own validation is not a type check.** It resolves names, counts call
/// arguments and rejects cycles — real checks, and the ones that catch a typo — but it
/// has no opinion about types. A `Sweep` passed to a parameter declared `vec3<f32>`
/// linked without complaint, deposited an artifact, and failed at
/// `create_shader_module`: at run time, on a GPU, in the half of the suite CI runs
/// without comparing pixels. It cost eleven red tests and a bisect to find something
/// the compiler knew.
///
/// So the artifact is parsed and validated here, by `naga` — which is what `wgpu`
/// itself uses, so this is the *same* answer, moved from the run to the build and
/// attributed to the shader that earned it.
///
/// **And the answer is kept.** Validation is the pass that works out which globals
/// each entry point reaches through its callees and which of them it samples; throwing
/// that away left the host to restate it by hand, per pipeline. [`crate::reflect`]
/// turns the pair into the generated record.
///
/// Validated with no capabilities beyond the default set, which is the honest bound: a
/// shader this rejects is one some target would reject too.
pub(crate) fn typechecks(wgsl: &str, artifact: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| {
        panic!(
            "`{artifact}` is not valid WGSL:\n{}",
            e.emit_to_string(wgsl)
        );
    });
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!(
            "`{artifact}` does not validate:\n{}",
            e.emit_to_string(wgsl)
        );
    });
    (module, info)
}
