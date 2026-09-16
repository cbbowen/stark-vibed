//! The WESL tree, read and parsed once.

use std::path::Path;

use wesl::syntax::{GlobalDeclaration, Struct, TranslationUnit};

/// One module of the shader tree.
///
/// The *unlinked* source, always. The linker mangles `Stamp` to
/// `package__1dynamics_common_Stamp`, emits it once per artifact that reaches it, and strips
/// whatever no entry point uses — the reason the check this replaces could not see a
/// constant that survived only in prose (the retired wick's `WICK_RATE` was the case
/// that proved it). And it drops the comments that are half of what is generated here.
pub(crate) struct Module {
    /// The WESL path — `dynamics`, or `lib/paint_common`. How a diagnostic names it.
    pub(crate) path: String,
    /// The Rust module the mirrors land in: the file's own name, without the
    /// directory. `lib` holds the binding-free leaves and is a placement rule rather
    /// than a namespace, so it does not reach the generated paths.
    pub(crate) rust: String,
    pub(crate) src: String,
    pub(crate) tu: TranslationUnit,
}

impl Module {
    /// Parse one module's source, under the WESL path it was found at.
    ///
    /// # Panics
    /// If `src` is not a WESL translation unit.
    pub(crate) fn parse(path: &str, src: &str) -> Self {
        let tu: TranslationUnit = src
            .parse()
            .unwrap_or_else(|e| panic!("cannot parse {path}.wesl: {e}"));
        Self {
            path: path.to_string(),
            rust: path.rsplit('/').next().unwrap_or(path).to_string(),
            src: src.to_string(),
            tu,
        }
    }

    /// The `struct name` this module declares, if it declares one.
    pub(crate) fn struct_named(&self, name: &str) -> Option<&Struct> {
        self.tu.global_declarations.iter().find_map(|d| match &**d {
            GlobalDeclaration::Struct(s) if s.ident.name().as_str() == name => Some(s),
            _ => None,
        })
    }
}

/// Read every `.wesl` in the tree, sorted by path so the generated file is
/// deterministic (`read_dir` order is not).
pub(crate) fn read_tree(shader_dir: &Path) -> Vec<Module> {
    let mut out: Vec<Module> = Vec::new();
    for dir in ["", "lib"] {
        let at = if dir.is_empty() {
            shader_dir.to_path_buf()
        } else {
            shader_dir.join(dir)
        };
        let mut paths: Vec<_> = std::fs::read_dir(&at)
            .unwrap_or_else(|e| panic!("read {}: {e}", at.display()))
            .map(|e| e.expect("shader dir entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "wesl"))
            .collect();
        paths.sort();
        for p in paths {
            let stem = p
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a shader file has a name");
            let path = if dir.is_empty() {
                stem.to_string()
            } else {
                format!("{dir}/{stem}")
            };
            let src = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
            let module = Module::parse(&path, &src);
            // Two shaders with the same file name in different directories would land
            // in one Rust module and silently merge their items. Refused rather than
            // merged — the mirror's whole job is that one declaration answers for one
            // thing.
            if let Some(other) = out.iter().find(|m| m.rust == module.rust) {
                panic!(
                    "`{}.wesl` and `{}.wesl` would both mirror as `{}`",
                    other.path, module.path, module.rust
                );
            }
            out.push(module);
        }
    }
    out
}
