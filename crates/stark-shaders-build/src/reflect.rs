//! What an entry point *is*, read off the validated artifact (§6.10).
//!
//! Three facts the host had been restating by hand, each of which naga works out while
//! type-checking and used to discard: which stage an entry point is and how wide its
//! workgroup, which `@location`s a fragment entry point writes, and which of the
//! artifact's globals it actually reaches — through its callees, and whether it reads
//! them through a sampler.
//!
//! **Read on the linked side, reported on the unlinked one.** Only the linked artifact
//! knows what a pipeline's entry point reaches, since half of it arrives by import; but
//! the names there are mangled, so every global is resolved back through the compiler's
//! own sourcemap to the `.wesl` that declares it and named by that module's generated
//! `decl::` constant.

use std::collections::BTreeSet;

use wesl::SourceMap;

use crate::tree::Module;

/// One global an entry point reaches, named by the WESL declaration that states it.
pub(crate) struct Used {
    /// The Rust mirror module the declaration is emitted under — `m.rust`.
    pub(crate) module: String,
    /// The `decl::` constant's name: the WESL variable's, uppercased.
    pub(crate) decl: String,
    /// Whether this entry point reads it **through a sampler**, which is what makes a
    /// texture's layout entry filterable.
    ///
    /// The image side of the pair only. A sampler's own layout entry is a filtering
    /// sampler either way, so the flag would say nothing about it.
    pub(crate) sampled: bool,
    /// `@group`/`@binding`, for the sort that keeps the generated record stable — the
    /// linker's declaration order is not.
    slot: (u32, u32),
}

/// One entry point of a linked artifact.
pub(crate) struct Reflected {
    pub(crate) name: String,
    pub(crate) stage: naga::ShaderStage,
    /// `[0, 0, 0]` for anything but a compute entry point — naga's own answer, left as
    /// it is rather than given a meaning it does not have.
    pub(crate) workgroup_size: [u32; 3],
    /// The `@location`s a fragment entry point writes, ascending. Empty for every
    /// other stage: a vertex output is the rasterizer's, not a color target.
    pub(crate) targets: Vec<u32>,
    pub(crate) uses: Vec<Used>,
}

/// Every entry point of one linked artifact, sorted by name.
///
/// Sorted because `wesl` does not promise a declaration order between runs, and this
/// becomes the generated struct's field order: unsorted, a rebuild that changed nothing
/// would rewrite the file.
///
/// `root` is the module that was linked — the one whose own declarations the linker
/// leaves unmangled, and so the one the sourcemap has nothing to say about.
///
/// # Panics
/// On an entry point whose workgroup size is an override expression, a global the
/// sourcemap attributes to a module outside the tree, or a fragment result this cannot
/// read a location off.
pub(crate) fn entry_points(
    naga: &naga::Module,
    info: &naga::valid::ModuleInfo,
    sourcemap: &impl SourceMap,
    root: &str,
    tree: &[Module],
    artifact: &str,
) -> Vec<Reflected> {
    let mut out: Vec<Reflected> = naga
        .entry_points
        .iter()
        .enumerate()
        .map(|(i, ep)| {
            assert!(
                ep.workgroup_size_overrides.is_none(),
                "`{artifact}`'s `{}` sizes its workgroup with an override expression, \
                 which this mirror reports as three numbers — and a host dispatching \
                 against the wrong ones is a kernel that reads outside its grid.",
                ep.name,
            );
            Reflected {
                name: ep.name.clone(),
                stage: ep.stage,
                workgroup_size: ep.workgroup_size,
                targets: targets(naga, ep, artifact),
                uses: uses(
                    naga,
                    info.get_entry_point(i),
                    sourcemap,
                    root,
                    tree,
                    artifact,
                ),
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The globals `fi`'s entry point reaches, ascending by slot.
fn uses(
    naga: &naga::Module,
    fi: &naga::valid::FunctionInfo,
    sourcemap: &impl SourceMap,
    root: &str,
    tree: &[Module],
    artifact: &str,
) -> Vec<Used> {
    // `sampling_set` is exhaustive for an entry point — naga says so: an entry point
    // takes no texture or sampler as an argument, so every pair it can sample with is
    // in terms of globals.
    let sampled: BTreeSet<_> = fi.sampling_set.iter().map(|k| k.image).collect();
    let mut out: Vec<Used> = naga
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            // A global with no `@binding` occupies no slot — `var<private>` and the
            // workgroup scratch the wet loop hoists into.
            let slot = global.binding.as_ref()?;
            // `global_uses` merges what the callees use, which is the whole reason to
            // ask naga rather than to read the entry point's own body: `deposit`
            // touches most of its slots through `lib/` helpers.
            if fi[handle].is_empty() {
                return None;
            }
            let name = global
                .name
                .as_deref()
                .unwrap_or_else(|| panic!("`{artifact}` has an unnamed `@binding` global"));
            let (module, decl) = declared_at(sourcemap, name, root, tree, artifact);
            Some(Used {
                module,
                decl: decl.to_uppercase(),
                sampled: sampled.contains(&handle),
                slot: (slot.group, slot.binding),
            })
        })
        .collect();
    out.sort_by(|a, b| (a.slot, &a.decl).cmp(&(b.slot, &b.decl)));
    out
}

/// The Rust mirror module and WESL item name a linked global came from.
///
/// Through the compiler's own sourcemap rather than by unpicking the mangled spelling:
/// the mangler is a setting, and a name this build's mangler did not produce would be
/// read as a different declaration rather than refused. A name the sourcemap does not
/// hold is the root module's, which is not mangled — [`crate::collide`] leans on that
/// same fact, though it names the root by the artifact it is linking rather than by the
/// module path, its message being about which *files* claimed one slot.
fn declared_at(
    sourcemap: &impl SourceMap,
    name: &str,
    root: &str,
    tree: &[Module],
    artifact: &str,
) -> (String, String) {
    let (path, item) = sourcemap.get_decl(name).map_or_else(
        || (root.to_string(), name),
        |(path, item)| (path.components.join("/"), item),
    );
    let module = tree.iter().find(|m| m.path == path).unwrap_or_else(|| {
        panic!(
            "`{artifact}` binds `{item}`, which the sourcemap attributes to \
                 `{path}` — a module outside the shader tree, so it has no generated \
                 `decl::` constant to name. A binding belongs to the module that owns \
                 the pipeline (§2)."
        )
    });
    (module.rust.clone(), item.to_string())
}

/// The `@location`s a fragment entry point writes.
///
/// Both spellings: a bare `-> @location(0) vec4<f32>`, and the struct half the tree
/// returns because it writes two or three attachments at once.
fn targets(naga: &naga::Module, ep: &naga::EntryPoint, artifact: &str) -> Vec<u32> {
    if ep.stage != naga::ShaderStage::Fragment {
        return Vec::new();
    }
    let Some(result) = &ep.function.result else {
        return Vec::new();
    };
    let at = |binding| location(binding, &ep.name, artifact);
    let mut out: Vec<u32> = match &result.binding {
        Some(binding) => at(binding).into_iter().collect(),
        None => match &naga.types[result.ty].inner {
            naga::TypeInner::Struct { members, .. } => members
                .iter()
                .filter_map(|m| m.binding.as_ref().and_then(at))
                .collect(),
            _ => panic!(
                "`{artifact}`'s `{}` returns an unbound non-struct, which names no \
                 color target",
                ep.name,
            ),
        },
    };
    out.sort_unstable();
    out
}

/// The `@location` one output carries, or `None` for a builtin.
///
/// **`@blend_src` is refused rather than reported.** Dual-source blending gives one
/// target two outputs at one `@location`, so the list this builds would hold that
/// location twice — and the host's target array is checked against it one entry per
/// location (`desc::render_pipeline`). A shader that wanted it would want that check
/// thought about rather than silently doubled.
fn location(binding: &naga::Binding, ep: &str, artifact: &str) -> Option<u32> {
    match binding {
        naga::Binding::Location {
            location,
            blend_src: Some(src),
            ..
        } => panic!(
            "`{artifact}`'s `{ep}` writes `@location({location}) @blend_src({src})`. \
             Dual-source blending is two outputs at one color target, which this \
             mirror reports as two targets — and the host builds its target array \
             against that count."
        ),
        naga::Binding::Location { location, .. } => Some(*location),
        naga::Binding::BuiltIn(_) => None,
    }
}

/// Written against hand-linked WGSL and a hand-built sourcemap — the shape the linker
/// actually leaves, without linking anything.
#[cfg(test)]
mod tests {
    use super::*;
    use wesl::{BasicSourceMap, NoSourceMap};

    fn sourcemap(decls: &[(&str, &str, &str)]) -> BasicSourceMap {
        let mut map = BasicSourceMap::new();
        for (mangled, module, item) in decls {
            map.add_decl(
                (*mangled).to_string(),
                module.parse().expect("a module path"),
                (*item).to_string(),
            );
        }
        map
    }

    /// Reflect `wgsl` as though it were `root`'s artifact, over a tree of `paths`.
    fn reflect(wgsl: &str, root: &str, paths: &[&str], map: &impl SourceMap) -> Vec<Reflected> {
        let tree: Vec<Module> = paths.iter().map(|p| Module::parse(p, "")).collect();
        let (naga, info) = crate::check::typechecks(wgsl, "probe");
        entry_points(&naga, &info, map, root, &tree, "probe")
    }

    fn named<'a>(eps: &'a [Reflected], name: &str) -> &'a Reflected {
        eps.iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("no `{name}`"))
    }

    fn uses_of(ep: &Reflected) -> Vec<(&str, &str, bool)> {
        ep.uses
            .iter()
            .map(|u| (u.module.as_str(), u.decl.as_str(), u.sampled))
            .collect()
    }

    const TWO_ENTRY_POINTS: &str = "\
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var other: texture_2d<f32>;

fn tap(uv: vec2<f32>) -> vec4<f32> { return textureSample(src, samp, uv); }

@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }

@fragment fn fs_main() -> @location(0) vec4<f32> { return tap(vec2<f32>(0.0)); }

@compute @workgroup_size(8, 4, 2) fn kernel() { _ = textureLoad(other, vec2<i32>(0), 0); }
";

    /// The whole record for the shapes the tree has: a vertex entry point that touches
    /// nothing, a fragment one that samples through a callee, and a compute one with a
    /// workgroup size of its own.
    #[test]
    fn an_entry_point_carries_its_stage_its_targets_and_what_it_reaches() {
        let eps = reflect(TWO_ENTRY_POINTS, "probe", &["probe"], &NoSourceMap);
        assert_eq!(
            eps.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            ["fs_main", "kernel", "vs_main"],
            "sorted by name, so the generated field order does not follow the linker",
        );

        let fs = named(&eps, "fs_main");
        assert_eq!(fs.stage, naga::ShaderStage::Fragment);
        assert_eq!(fs.targets, [0]);
        // `src` is sampled, `samp` is the sampler it is sampled with, and `other`
        // belongs to the kernel alone.
        assert_eq!(
            uses_of(fs),
            [("probe", "SRC", true), ("probe", "SAMP", false)],
        );

        let vs = named(&eps, "vs_main");
        assert_eq!(
            vs.targets,
            Vec::<u32>::new(),
            "a vertex output is no target"
        );
        assert!(vs.uses.is_empty());

        let kernel = named(&eps, "kernel");
        assert_eq!(kernel.workgroup_size, [8, 4, 2]);
        assert_eq!(uses_of(kernel), [("probe", "OTHER", false)]);
    }

    /// A struct result is the half of the tree that writes two attachments at once.
    /// The builtin beside the locations is not one of them.
    #[test]
    fn a_struct_result_names_every_location_it_writes() {
        let eps = reflect(
            "struct Out {\n\
             \x20 @location(2) aux: vec4<f32>,\n\
             \x20 @builtin(frag_depth) depth: f32,\n\
             \x20 @location(0) color: vec4<f32>,\n\
             }\n\
             @fragment fn fs_main() -> Out {\n\
             \x20 return Out(vec4<f32>(0.0), 0.0, vec4<f32>(0.0));\n\
             }\n",
            "probe",
            &["probe"],
            &NoSourceMap,
        );
        assert_eq!(named(&eps, "fs_main").targets, [0, 2]);
    }

    /// The linked shape: an imported binding arrives mangled and is resolved back to
    /// the module that declares it, where the root's own is not mangled at all.
    #[test]
    fn a_mangled_global_is_named_by_the_module_the_sourcemap_gives_it_to() {
        let map = sourcemap(&[(
            "package__1dynamics_common_st",
            "package::dynamics_common",
            "st",
        )]);
        let eps = reflect(
            "struct St { n: u32 }\n\
             @group(0) @binding(0) var<uniform> package__1dynamics_common_st: St;\n\
             @group(0) @binding(1) var region: texture_2d<f32>;\n\
             @compute @workgroup_size(1) fn deposit() {\n\
             \x20 _ = package__1dynamics_common_st.n;\n\
             \x20 _ = textureLoad(region, vec2<i32>(0), 0);\n\
             }\n",
            "dynamics",
            &["dynamics", "dynamics_common"],
            &map,
        );
        assert_eq!(
            uses_of(named(&eps, "deposit")),
            [
                ("dynamics_common", "ST", false),
                ("dynamics", "REGION", false),
            ],
        );
    }

    /// A module under `lib/` mirrors as its leaf name, which is the path the generated
    /// `decl::` constants are under.
    #[test]
    fn a_nested_module_is_named_by_its_rust_leaf() {
        let map = sourcemap(&[("package__1lib_view_v", "package::lib::view", "v")]);
        let eps = reflect(
            "struct V { n: u32 }\n\
             @group(0) @binding(0) var<uniform> package__1lib_view_v: V;\n\
             @compute @workgroup_size(1) fn k() { _ = package__1lib_view_v.n; }\n",
            "probe",
            &["probe", "lib/view"],
            &map,
        );
        assert_eq!(uses_of(named(&eps, "k")), [("view", "V", false)]);
    }

    /// An unused binding is not a use. wgpu would accept the extra layout entry, so
    /// nothing else would ever say the host had listed one the entry point never reads.
    #[test]
    fn a_binding_the_entry_point_does_not_reach_is_not_a_use() {
        let eps = reflect(
            "@group(0) @binding(0) var used: texture_2d<f32>;\n\
             @group(0) @binding(1) var unused: texture_2d<f32>;\n\
             @compute @workgroup_size(1) fn k() { _ = textureLoad(used, vec2<i32>(0), 0); }\n",
            "probe",
            &["probe"],
            &NoSourceMap,
        );
        assert_eq!(uses_of(named(&eps, "k")), [("probe", "USED", false)]);
    }

    /// Two outputs at one color target: the list would hold `[0, 0]`, and the host
    /// builds its target array one entry per location.
    #[test]
    #[should_panic(expected = "@location(0) @blend_src(1)")]
    fn dual_source_blending_is_refused() {
        reflect(
            "enable dual_source_blending;\n\
             struct Out {\n\
             \x20 @location(0) @blend_src(0) a: vec4<f32>,\n\
             \x20 @location(0) @blend_src(1) b: vec4<f32>,\n\
             }\n\
             @fragment fn fs_main() -> Out {\n\
             \x20 return Out(vec4<f32>(0.0), vec4<f32>(0.0));\n\
             }\n",
            "probe",
            &["probe"],
            &NoSourceMap,
        );
    }

    /// The sourcemap naming a module the tree does not hold — the transpiled
    /// polynomial's shape, were it ever to declare a binding.
    #[test]
    #[should_panic(expected = "a module outside the shader tree")]
    fn a_global_from_outside_the_tree_is_refused() {
        let map = sourcemap(&[("package__1gen_poly_lut", "package::gen::poly", "lut")]);
        reflect(
            "@group(0) @binding(0) var package__1gen_poly_lut: texture_2d<f32>;\n\
             @compute @workgroup_size(1) fn k() {\n\
             \x20 _ = textureLoad(package__1gen_poly_lut, vec2<i32>(0), 0);\n\
             }\n",
            "probe",
            &["probe"],
            &map,
        );
    }
}
