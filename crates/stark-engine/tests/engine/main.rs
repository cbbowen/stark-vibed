//! The engine's integration tests, as one binary: a test crate per file would each
//! recompile `common` and link wgpu again. Cargo discovers only this file, so a module
//! missing from the list below is never compiled.

mod common;

mod assets;
mod assist;
mod blend;
mod collab;
mod color_dynamics;
mod commute;
mod commute_pairs;
mod composite;
mod corpus;
mod dynamics;
mod erase;
mod export;
mod fill;
mod filter;
mod footprint;
mod golden;
mod gradient;
mod groups;
mod guides;
mod hdr;
mod hover;
mod layers;
mod liquify;
mod matte;
mod merge;
mod minify;
mod modulation;
mod opacity;
mod peer_state;
mod pick;
mod place;
mod reference;
mod replay;
mod reset;
mod retention;
mod save_load;
mod seam;
mod selection;
mod sharing;
mod stroke;
mod tile_pool;
mod tooth;
mod transform;
mod translate;
mod view;
mod wide;

/// Every module file beside this one has a `mod` line above: a file added without one
/// compiles nothing and reports nothing, which no other test can notice.
#[test]
fn every_file_here_is_a_module_of_this_binary() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/engine");
    let main = std::fs::read_to_string(dir.join("main.rs")).expect("main.rs is readable");
    let declared: std::collections::HashSet<&str> = main
        .lines()
        .filter_map(|l| l.strip_prefix("mod ")?.strip_suffix(';'))
        .collect();
    let mut missing: Vec<String> = std::fs::read_dir(&dir)
        .expect("tests/engine is listable")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs") || p.join("mod.rs").is_file())
        .filter_map(|p| p.file_stem()?.to_str().map(str::to_owned))
        .filter(|name| name != "main" && !declared.contains(name.as_str()))
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "no `mod` line in tests/engine/main.rs, so nothing in them runs: {missing:?}"
    );
}
