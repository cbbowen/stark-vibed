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

    /// Whether this is one of the binding-free leaves (§2).
    ///
    /// The rule is the *directory*, at any depth, which is why it is read off the path
    /// here rather than from a list anywhere.
    pub(crate) fn under_lib(&self) -> bool {
        self.path.starts_with("lib/")
    }
}

/// Read every `.wesl` in the tree, at any depth.
///
/// Recursive — `lib/` is a placement rule at any depth, not a fixed list, and a walk of
/// `["", "lib"]` gave anything below it no mirror at all without saying so.
///
/// Deterministic, because `read_dir` order is not and the generated file's module order
/// is this one: a directory's own files sorted, then its subdirectories sorted,
/// depth-first.
pub(crate) fn read_tree(shader_dir: &Path) -> Vec<Module> {
    let mut out: Vec<Module> = Vec::new();
    walk(shader_dir, "", &mut out);
    out
}

fn walk(at: &Path, prefix: &str, out: &mut Vec<Module>) {
    let (mut files, mut dirs) = (Vec::new(), Vec::new());
    for entry in std::fs::read_dir(at).unwrap_or_else(|e| panic!("read {}: {e}", at.display())) {
        let path = entry
            .unwrap_or_else(|e| panic!("read an entry of {}: {e}", at.display()))
            .path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.extension().is_some_and(|e| e == "wesl") {
            files.push(path);
        }
    }
    files.sort();
    dirs.sort();

    for p in files {
        let src = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        let module = Module::parse(&format!("{prefix}{}", utf8(p.file_stem(), &p)), &src);
        // Two shaders with the same file name in different directories would land in
        // one Rust module and silently merge their items. Refused rather than merged —
        // the mirror's whole job is that one declaration answers for one thing.
        if let Some(other) = out.iter().find(|m| m.rust == module.rust) {
            panic!(
                "`{}.wesl` and `{}.wesl` would both mirror as `{}`",
                other.path, module.path, module.rust
            );
        }
        out.push(module);
    }
    for d in dirs {
        // The whole name, where a file contributes its stem: `ramp.v2/` and `ramp/` are
        // two directories, and trimming at the dot would silently make them one.
        walk(&d, &format!("{prefix}{}/", utf8(d.file_name(), &d)), out);
    }
}

/// A path component as a `str`, refused rather than lossily converted.
fn utf8<'a>(name: Option<&'a std::ffi::OsStr>, p: &Path) -> &'a str {
    name.and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("{} has no usable name", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shader tree on disk, removed when the test ends.
    struct Tree(std::path::PathBuf);

    impl Tree {
        /// `files` are `(path under the root, source)`, directories created as needed.
        ///
        /// The process id is in the name because this project routinely runs the suite
        /// from several worktrees at once, and a fixed name would have two runs deleting
        /// each other's tree.
        fn new(tag: &str, files: &[(&str, &str)]) -> Self {
            let root = std::env::temp_dir()
                .join(format!("stark-shaders-build-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for (path, src) in files {
                let at = root.join(path);
                std::fs::create_dir_all(at.parent().expect("a file has a parent"))
                    .expect("create the probe tree");
                std::fs::write(&at, src).expect("write a probe module");
            }
            Self(root)
        }

        fn read(&self) -> Vec<(String, String)> {
            read_tree(&self.0)
                .into_iter()
                .map(|m| (m.path, m.rust))
                .collect()
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The order the generated file's modules come out in: a directory's files, then its
    /// subdirectories, each sorted.
    #[test]
    fn the_tree_is_read_depth_first_files_before_subdirectories() {
        let tree = Tree::new(
            "order",
            &[
                ("stamp.wesl", ""),
                ("blur.wesl", ""),
                ("lib/sample.wesl", ""),
                ("lib/color.wesl", ""),
                ("notes.md", "not a shader"),
            ],
        );
        assert_eq!(
            tree.read(),
            [
                ("blur".to_string(), "blur".to_string()),
                ("stamp".to_string(), "stamp".to_string()),
                ("lib/color".to_string(), "color".to_string()),
                ("lib/sample".to_string(), "sample".to_string()),
            ]
        );
    }

    /// The walk that this replaces visited exactly `["", "lib"]`, so this module linked
    /// and got no mirror at all — the one failure a generated file cannot show, because
    /// what is missing is the file's absence.
    #[test]
    fn a_module_below_lib_is_read() {
        let tree = Tree::new("nested", &[("lib/ramp/stops.wesl", "const N: u32 = 3u;\n")]);
        assert_eq!(
            tree.read(),
            [("lib/ramp/stops".to_string(), "stops".to_string())]
        );
    }

    /// A directory keeps its whole name. Trimming at the dot the way a file's stem is
    /// trimmed would make `ramp.v2/` and `ramp/` one WESL path.
    #[test]
    fn a_directory_with_a_dot_in_its_name_keeps_it() {
        let tree = Tree::new("dotted", &[("lib/ramp.v2/stops.wesl", "")]);
        assert_eq!(
            tree.read(),
            [("lib/ramp.v2/stops".to_string(), "stops".to_string())]
        );
    }

    /// Two directories deep is still one Rust module name.
    #[test]
    #[should_panic(expected = "`lib/color.wesl` and `lib/ramp/color.wesl` would both mirror as")]
    fn one_rust_name_from_two_directories_is_refused() {
        Tree::new(
            "collide",
            &[("lib/color.wesl", ""), ("lib/ramp/color.wesl", "")],
        )
        .read();
    }
}
