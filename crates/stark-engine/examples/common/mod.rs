//! Shared by the dev repro harnesses: paying a captured document's content bill.
//!
//! Not a Cargo example itself — `examples/*/mod.rs` matches neither of the patterns
//! Cargo auto-discovers, so this is included with `mod common;` exactly the way
//! `tests/common` is.

/// Resolve everything `file` names out of the frontend's bundled assets, so a
/// captured document replays as the app replays it (§8).
///
/// **A capture is a lean file and this is not optional.** The app strips any asset it
/// ships from the bundle and keeps only the content id, on the promise that whoever
/// opens it can produce those bytes — a promise only `stark-dioxus-frontend` could
/// keep, because only its build script hashes the shipped PNGs. So a harness that
/// skipped this got a substrate that would not resolve, and a substrate that does not
/// resolve is `Flat`, whose relief is 0: the deposition tooth is inert, and a toothed
/// capture replays perfectly smooth with nothing to say it did (§6.4). A real bug
/// report was unreproducible here for that reason alone.
///
/// It is a bill and not a fetch: the engine refuses to replay a document whose content
/// is still outstanding (`stark_model::DocError::MissingContent`), so anything left unpaid here
/// substrates at the load rather than in the pixels.
pub fn settle(engine: &mut stark_engine::Engine, file: &stark_model::DocumentFile) {
    for need in engine.unresolved_content(file) {
        // The id says which store the bytes belong in, so the lookup needs no kind —
        // a shape is hashed from its coverage and a substrate from its height, so one
        // image filed as both would earn two ids.
        let Some(bytes) = stark_testdata::assets::bundled(need.content()) else {
            eprintln!("this build ships nothing for {need:?}; the load will refuse it");
            continue;
        };
        let installed = match need.substrate() {
            Some(id) => engine.accept_substrate(id, &bytes).map(|_| ()),
            None => engine.import_brush(&bytes).map(|_| ()),
        };
        match installed {
            Ok(()) => eprintln!("resolved {need:?} from the app's assets"),
            Err(e) => eprintln!("could not install {need:?}: {e}"),
        }
    }
}
