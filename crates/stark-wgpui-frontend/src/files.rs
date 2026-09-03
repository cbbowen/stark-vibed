//! Getting work in and out as files: save, open and export (§8, §15.6).
//!
//! Two different things share this module because they are the same gesture from the
//! user's side — a file leaves or arrives — but they are not the same object:
//!
//! - **Save** writes the *document*: the action log, replayable, still editable, with
//!   its whole undo history intact when reopened. This is the native format and the
//!   thing that must never be lossy.
//! - **Export** writes a *picture*: one frame, flattened and lit. It is an output,
//!   not a document, and nothing can be recovered from it.
//!
//! # What is different here from the web
//!
//! Both frontends reach the same three engine calls; what differs is the *door*. The
//! browser has no filesystem, so it downloads a blob and picks files through a hidden
//! `<input>`; here there are real paths, which buys two things a download cannot:
//! **Save over the file you opened**, and a window title that says which file that is.
//!
//! The paths come from wgpui's own dialogs, which answer on a channel rather than a
//! callback — so each act is a task that waits, and the result comes back to the view
//! through a message rather than by writing a signal from inside the picker.
//!
//! # Lean files, both ways
//!
//! A `.stark` file may *name* content rather than carry it (§8's version 6), leaving
//! out substrates and shapes it knows the opener ships. Doing that needs a catalog on
//! both sides, and until N7 this frontend had none — so it saved everything, and
//! refused any file naming a shipped substrate.
//!
//! It carries the images now (`crate::assets`), so both halves are the catalog's:
//! [`save_bytes`] leaves out what any build can resolve, and `Canvas::load` settles an
//! owed asset out of the binary before replaying. A file naming content **no** build
//! ships is still refused, with the painting on screen untouched — a file has no peer
//! to ask, and a substrate that silently failed to arrive would bake a smooth deposit
//! into tiles that no later arrival un-bakes (§6.4).

use std::path::{Path, PathBuf};

use stark_chrome::files::DOC_EXT;
use stark_engine::{Background, ExportScale, Rendered};

use crate::render::Renderer;

/// What a file act produced, handed back to the view.
///
/// A message rather than a signal write, because the dialogs answer asynchronously
/// and the view owns the engine: an act that reached in from a task would be a second
/// door onto the state `send` is the only door onto (§4).
pub enum Done {
    /// A document was written at this revision — the client is clean up to it.
    Saved { path: PathBuf, revision: u64 },
    /// A document was read; the bytes are the caller's to load.
    Opened { path: PathBuf, bytes: Vec<u8> },
    /// A picture was written. Not a document, but a copy of the work, so it settles
    /// the same question: whether closing this loses any.
    Exported { revision: u64 },
    /// The user called the dialog off. Not a failure and not reported as one.
    Cancelled,
    /// Something went wrong, in words a person could act on.
    Failed(String),
}

/// The document as bytes, **lean**: what any build can resolve out of its own
/// catalog is named rather than carried (§8's version 6).
///
/// The list is `stark_chrome::assets::resolvable`, which is the same list the web
/// frontend passes — so the two write files of the same shape, and a document on the
/// linen substrate is a few kilobytes on both rather than fourteen megabytes on one.
pub fn save_bytes(renderer: &Renderer) -> Result<Vec<u8>, String> {
    renderer
        .save_bytes_resolvable(&stark_chrome::assets::resolvable())
        .map_err(|e| format!("could not serialize the document: {e}"))
}

/// Write `bytes` to `path`, adding the extension if the dialog did not.
///
/// A name the user typed without an extension is meant as a `.stark` — a dialog that
/// took it literally would leave a file the app itself would not offer to open.
pub fn write(path: &Path, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = with_extension(path, DOC_EXT);
    std::fs::write(&path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

/// `path` with `ext`, unless it already ends in it (caselessly — a `.STARK` a file
/// manager produced is the same file).
pub fn with_extension(path: &Path, ext: &str) -> PathBuf {
    match path.extension() {
        Some(had) if had.eq_ignore_ascii_case(ext) => path.to_path_buf(),
        _ => path.with_extension(ext),
    }
}

/// Read a document off disk.
pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))
}

/// Encode a rendered picture for `path`'s extension, defaulting to PNG.
///
/// The two the engine can write (§15.6), chosen by the name the user typed — which is
/// the convention every other app on the machine follows, and cheaper than a format
/// dropdown the dialog would have to grow.
pub fn encode(image: &stark_engine::RgbaImage, path: &Path) -> Result<Vec<u8>, String> {
    let jpeg = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));
    let encoded = if jpeg {
        image.to_jpeg(EXPORT_JPEG_QUALITY)
    } else {
        image.to_png()
    };
    encoded.map_err(|e| format!("could not encode the picture: {e}"))
}

/// The quality a JPEG export is written at.
///
/// High enough that the artefacts are not what a viewer notices first, and short of
/// the top where the file grows faster than the picture improves. The same figure the
/// web frontend offers as its default.
const EXPORT_JPEG_QUALITY: u8 = 90;

/// The title bar's text: the file this window holds, and whether it has unsaved work.
///
/// A window title is where a desktop app says which document it is, and the marker is
/// the convention for "not written yet" on every platform this runs on.
pub fn window_title(path: Option<&Path>, unsaved: bool) -> String {
    let name = path
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string());
    let mark = if unsaved { "• " } else { "" };
    format!("{mark}{name} — Stark")
}

/// What the engine should render for an export, and at what scale.
///
/// Fixed for now: the whole canvas, over its substrate, at 1×. The web app's export
/// dialog offers a frame, a scale and a transparent background (§15.6), and each of
/// those is a control this frontend has nowhere to put yet — so the defaults are
/// stated here rather than spread across the call.
pub const EXPORT: (
    Option<stark_model::document::LayerId>,
    ExportScale,
    Background,
    Rendered,
) = (
    None,
    ExportScale::Factor(1.0),
    Background::Substrate,
    Rendered::Committed,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A name typed without an extension is meant as a document — the app would not
    /// otherwise offer to open what it had just written.
    #[test]
    fn a_document_gets_its_extension() {
        assert_eq!(
            with_extension(Path::new("/x/painting"), DOC_EXT),
            PathBuf::from("/x/painting.stark")
        );
    }

    /// One it already has is left alone, whatever case a file manager produced it in.
    #[test]
    fn an_extension_is_not_doubled() {
        for name in ["/x/p.stark", "/x/p.STARK"] {
            let out = with_extension(Path::new(name), DOC_EXT);
            assert_eq!(out, PathBuf::from(name), "{name} already names the format");
        }
    }

    /// A name carrying some *other* extension gains the document's rather than
    /// keeping one that would mislead a file manager about what it holds.
    #[test]
    fn another_extension_gives_way() {
        assert_eq!(
            with_extension(Path::new("/x/p.png"), DOC_EXT),
            PathBuf::from("/x/p.stark")
        );
    }

    /// The title says which file, and whether it holds work the disk does not.
    #[test]
    fn the_title_names_the_file_and_its_state() {
        assert_eq!(window_title(None, false), "Untitled — Stark");
        assert_eq!(window_title(None, true), "• Untitled — Stark");
        assert_eq!(
            window_title(Some(Path::new("/x/sketch.stark")), true),
            "• sketch.stark — Stark"
        );
    }
}
