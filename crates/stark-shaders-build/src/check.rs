//! The type check `wgpu` would otherwise run on a GPU — and the answer it gives.

/// Parse and validate the linked WGSL the way `wgpu` will, keeping what naga worked
/// out.
///
/// `wesl`'s own validation resolves names and rejects cycles but has no opinion about
/// types, so a type error reaches `create_shader_module` — at run time, on a GPU. This
/// is `naga`, the front end `wgpu` uses, so it is the same answer moved to the build.
///
/// The answer is **kept**: validation is the pass that works out which globals an entry
/// point reaches through its callees and which of them it samples ([`crate::reflect`]).
/// Default capabilities only, which is the honest bound.
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
