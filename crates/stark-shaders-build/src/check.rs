//! The type check `wgpu` would otherwise run on a GPU.

/// Fail unless the linked WGSL passes the same front end `wgpu` will run on it.
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
/// Validated with no capabilities beyond the default set, which is the honest bound: a
/// shader this rejects is one some target would reject too.
pub(crate) fn typechecks(wgsl: &str, artifact: &str) {
    let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| {
        panic!(
            "`{artifact}` is not valid WGSL:\n{}",
            e.emit_to_string(wgsl)
        );
    });
    naga::valid::Validator::new(
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
}
