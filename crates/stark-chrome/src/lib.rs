//! The **frontend's model** — what a chrome is written in, below any toolkit
//! (§11.2).
//!
//! `frontend → chrome → engine → model`. Two frontends sit above this crate
//! (`stark-dioxus-frontend`, `stark-wgpui-frontend`); without it they grow two
//! copies of every rule, and the copies drift. That is not a prediction: the
//! native frontend was one commit old and already carrying its own
//! `ROPE_MAX_SCREEN_PX` beside the web one's.
//!
//! # What belongs here
//!
//! The rule is §2's, one level up:
//!
//! > If it names a toolkit type or holds a `Signal`, it is **chrome** and stays in
//! > its frontend. If it is arithmetic over
//! > [`ObservableState`](stark_engine::ObservableState),
//! > [`BrushParams`](stark_model::document::BrushParams),
//! > [`ViewTransform`](stark_engine::ViewTransform) or a pointer report, it is the
//! > frontend's **model** and belongs here.
//!
//! **This crate names no toolkit type at all** — no `dioxus`, no `wgpui`, no
//! `web-sys`, no `winit` — which is `stark-net`'s bargain applied one level up, and
//! which `tests::no_toolkit_types` holds by reading the source rather than by
//! trusting the manifest: a type can arrive through a re-export the dependency list
//! does not show.
//!
//! It compiles to wasm, because the web app is one of its two consumers.
//!
//! # What is here
//!
//! Every module was already this crate before this crate existed — each was split
//! out of the web frontend *because it was the part that could be tested*, which is
//! the same line drawn one file early. The 45 tests came with them.
//!
//! - [`brush_config`] — the brush as a frontend carries it: the durable half (what
//!   the tool *is*) beside the transient one (the size, flow and colour in hand),
//!   and `params()`, the one projection down to the engine's `BrushParams` (§6.2).
//! - [`transform`] — the transform mode's algebra (§16.6, §16.8, §16.9). Named for
//!   what it computes; it was `gesture` next to five *input* gestures that are not
//!   this, and the name would have been read as those here.
//! - [`layer_tree`] — what the Layers panel draws, and what a drop into it means
//!   (§14.6, §14.8).
//! - [`reorder`] — moving a row of a list by dragging it, with no opinion about
//!   what the list is. Two panels are rosters; this is the gesture they share.
//! - [`library`] — the gallery thumbnails a browser-held asset library shows
//!   (§6.4, §6.6).
//! - [`storage`] — the ten records a client keeps between visits, the one JSON format
//!   they are kept in, and the [`Backend`](storage::Backend) a frontend installs to
//!   say where they actually go (§25.6).
//! - [`identity`] — the key this client's `ActorId` derives from, and the run counter
//!   beside it (§17).
//! - [`prefs`] — the standing preferences a settings dialog sets.
//! - [`input`] — the two screen-denominated lengths a gesture declares, and the map
//!   from a knob to each (§6.2, §6.11). The module this crate was built to prevent a
//!   second copy of.

pub mod brush_config;
pub mod identity;
pub mod input;
pub mod layer_tree;
pub mod library;
pub mod prefs;
pub mod presets;
pub mod reorder;
pub mod slots;
pub mod storage;
pub mod transform;

#[cfg(test)]
mod tests {
    /// **No toolkit type reaches this crate.**
    ///
    /// Read off the source rather than the manifest, because the manifest is the
    /// weaker claim: a `dioxus::` path can arrive through a dependency's re-export
    /// without ever appearing in `[dependencies]`, and what would go wrong is not a
    /// build failure but a module that quietly stops being movable.
    ///
    /// The check a reviewer would do, made a thing that runs.
    #[test]
    fn no_toolkit_types() {
        // Path-shaped, so that *naming the frontend crate* in prose stays legal while
        // naming one of its types does not: `stark-dioxus-frontend` is a neighbour worth
        // pointing at, `dioxus::Event` is the thing this crate must not know.
        const BANNED: &[&str] = &["dioxus::", "wgpui::", "web_sys::", "winit::", "gpui::"];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&src).expect("the crate has a src directory") {
            let path = entry.expect("a readable directory entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            // This file names all five, which is the one legitimate mention.
            if path.file_name().is_some_and(|n| n == "lib.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a readable source file");
            for (n, line) in text.lines().enumerate() {
                if let Some(name) = BANNED.iter().find(|b| line.contains(**b)) {
                    found.push(format!("{}:{}: {name}", path.display(), n + 1));
                }
            }
        }
        assert!(
            found.is_empty(),
            "a toolkit type reached stark-chrome, which is what it exists not to name (§11.2):\n{}",
            found.join("\n")
        );
    }
}
