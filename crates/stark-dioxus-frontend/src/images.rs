//! Getting a picture in: an image file, or the clipboard (§23).
//!
//! Three gestures that feel nothing alike — a menu and a file picker, Ctrl+V, a file
//! dragged onto the window — and are the same thing from here down. All three arrive as
//! encoded bytes, go through the browser's decoder, and commit one
//! [`DocCommand::PlaceImage`]. So the
//! shared half is [`place_bytes`] and the three entry points are a few lines each,
//! which is the right proportion: what differs between them is only where the bytes
//! came from, whether there is a filename to name the layer after, and whether the
//! gesture **pointed** anywhere ([`At`]).
//!
//! Sibling to [`crate::files`] rather than part of it, and the distinction is the one
//! that file's header draws so carefully. Save and Open move the *document*; this moves
//! a **picture into** one. An image placed here is paint from the moment it lands —
//! stroke over it, glaze it, lift it, transform it — not an attachment the document
//! keeps a reference to.

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::HasFileData;
use dioxus::prelude::*;

use crate::platform::{decode_image, pick_file};
use crate::state::{AppState, dispatch};
use stark_assetid::Picture;
use stark_engine::command::DocCommand;
use stark_model::geom::{IVec2, Vec2};

/// What the file picker offers. `image/*` rather than a list of extensions, because
/// the decoder is the browser's (§23) — enumerating formats here would be this app
/// claiming to know which ones it supports, and getting it wrong in whichever direction
/// the browser moved.
const ACCEPT: &str = "image/*";

/// Where a placement lands.
///
/// The one thing that genuinely differs between the three ways in, so it is a type
/// rather than an `Option<Vec2>` threaded through: a menu import and a paste do not
/// point anywhere and a drop does, and naming that difference is what keeps the
/// arithmetic for each in one place.
#[derive(Copy, Clone)]
enum At {
    /// The middle of what is being looked at — the placement that needs no
    /// explanation, since an image arrives where the eye already is.
    Viewport,
    /// Under the pointer, in page pixels: a drop *is* a positional gesture, and
    /// centring it anywhere else would be ignoring the one thing the hand said.
    Pointer(Vec2),
}

/// Ask for an image file and place it — the menu entry.
pub fn import_image(state: AppState) {
    // Called straight from the menu handler: a file picker may only be opened inside
    // the user gesture that asked for it (`platform::pick_file`).
    pick_file(ACCEPT, move |name, bytes| {
        place_bytes(state, Some(name), bytes, At::Viewport)
    });
}

/// Take whatever was dropped onto the window (§23.4).
///
/// **Every drop, not only an image**, because the alternative is not "nothing happens":
/// an unclaimed drop is handled by the browser, and what the browser does with a
/// dropped file is navigate to it, discarding an unsaved painting. The root therefore
/// claims all of them (`main`), which makes deciding what each one *is* this function's
/// job.
///
/// A `.stark` opens as a document — the other thing this app can be handed, and the
/// gesture the manifest's `file_handlers` already answers for a double-click (§11,
/// [`crate::files::bind_file_launch`]). Anything else is offered to the image decoder,
/// which is the browser's, so what it accepts is the platform's question rather than
/// this build's (§23.4); a file it cannot read logs and places nothing.
///
/// The first file only, like the launch queue and for its reason: this is one gesture
/// with one drop point, and a second image placed at the same spot would land under the
/// first where nobody would see it.
pub fn drop_files(state: AppState, e: &Event<DragData>) {
    let Some(file) = e.files().into_iter().next() else {
        // A drag that carried text, a link, or nothing this app can read. Claimed and
        // discarded rather than handed back to the browser — see above.
        return;
    };
    // Read here rather than in the task: the event is borrowed, and where the hand let
    // go is the whole of what a drop says that the other two ways in do not.
    let p = e.page_coordinates();
    let at = At::Pointer(Vec2::new(p.x as f32, p.y as f32));
    spawn_forever(async move {
        let name = file.name();
        let bytes = match file.read_bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => return tracing::error!(name, "could not read the dropped file: {e}"),
        };
        if is_document(&name) {
            return crate::files::open_bytes(state, bytes);
        }
        place_bytes(state, Some(name), bytes, at);
    });
}

/// Whether a dropped file is a Stark document rather than a picture.
fn is_document(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case(stark_ui::files::DOC_EXT))
}

/// Place whatever image is pasted into the page — bound once, for the life of the page.
///
/// Ctrl+V is not among [`crate::commands`]' chord rows and deliberately is not: a paste is
/// delivered as its own event carrying the clipboard with it, and a keystroke handler
/// would have to go and *ask* for the clipboard afterwards, through an API that prompts
/// for a permission and does not carry images in every browser
/// ([`platform::on_window_paste`](crate::platform::on_window_paste)).
pub fn bind_paste(state: AppState) {
    crate::platform::on_window_paste(move |bytes| {
        // No name: the clipboard has no filename to offer, so the layer is described by
        // its place in the stack, which is what an unnamed layer is for. And no point:
        // Ctrl+V is a keystroke, so the pointer is wherever it was left rather than
        // anywhere the gesture chose.
        place_bytes(state, None, bytes, At::Viewport);
    });
}

/// Decode `bytes` and commit the placement — the half both entry points share.
///
/// Detached (`spawn_forever`) because no caller has a scope to tie the work to: a menu
/// item unmounts the moment it is clicked, and a paste and a drop have no component
/// behind them at all. The task writes nothing but the document, through
/// [`dispatch`], so there is no signal of a dead scope to write
/// through — the hazard `files::ExportModal` avoids by doing the opposite.
fn place_bytes(state: AppState, name: Option<String>, bytes: Vec<u8>, at: At) {
    spawn_forever(async move {
        let (width, height, pixels) = match decode_image(bytes).await {
            Ok(decoded) => decoded,
            Err(e) => return tracing::error!("could not read that image: {e}"),
        };
        // Into the engine's picture store, which answers with the **content id** the
        // action references — the same two-step a custom brush shape takes (§6.6,
        // §23). Re-encoded as a canonical PNG on the way in, so what is stored, saved
        // and sent is the form the id actually names.
        let png = match (Picture {
            width,
            height,
            pixels,
        })
        .encode()
        {
            Ok(png) => png,
            Err(e) => return tracing::error!("could not encode that image: {e}"),
        };
        let Some(Ok(id)) = crate::state::with_engine_quiet(state, |r| r.import_picture_id(&png))
        else {
            return tracing::error!(width, height, "the canvas could not take that image");
        };
        // **Before the commit that names it**, which is the ordering `add_content`
        // documents: an action broadcast without its content registered leaves every
        // receiver unable to fetch what it needs (§12.4). A no-op when solo.
        seed_session(state, id, png);

        // Centred on whatever the gesture meant ([`At`]). The view is the frontend's
        // (§18.1.2), so this is the frontend's arithmetic — the engine is told a
        // position, not a policy.
        //
        // Rounded to whole canvas pixels, because that is what the action carries and
        // what makes the placement resample nothing (§23). The engine could not round
        // it for us without the rounding becoming invisible at the call site.
        let Some(at) = crate::state::with_engine_quiet(state, |r| {
            let view = r.view();
            let middle = match at {
                At::Viewport => view.center,
                At::Pointer(p) => view.screen_to_canvas(p),
            };
            IVec2::new(
                (middle.x - width as f32 * 0.5).round() as i32,
                (middle.y - height as f32 * 0.5).round() as i32,
            )
        }) else {
            return tracing::error!("the canvas is not ready yet");
        };

        // Above the layer being worked on, so a reference photograph lands over the
        // drawing rather than under it — and in that layer's own stack, so placing one
        // while inside a group keeps it in the group (§14.8).
        //
        // Read once into a scope that ends before the dispatch below. A projection
        // guard held across the command would be borrowed while `dispatch` publishes
        // the new one, which is the borrow panic `crate::state` warns about.
        let (above, carrier) = {
            let obs = state.obs.peek();
            let Some(o) = obs.as_ref() else {
                return tracing::error!("the canvas is not ready yet");
            };
            let above = o.active_layer;
            let carrier = o
                .layers
                .iter()
                .find(|l| l.id == above)
                .and_then(|l| l.carrier);
            (Some(above), carrier)
        };

        tracing::info!(width, height, ?name, "placing an image");
        dispatch(
            state,
            DocCommand::PlaceImage {
                carrier,
                above,
                at,
                name,
                image: id,
            },
        );
    });
}

/// Register the picture with a live session's mirror so peers can fetch it by hash.
/// A no-op when solo; idempotent when repeated, because it is content-addressed.
///
/// `shapes::seed_session`'s twin, and the reason a placed image needs no chunking of
/// its own: the log carries 32 bytes and the pixels travel over the blob ALPN, so a
/// photograph never has to fit in a gossip message (§12.4).
fn seed_session(state: AppState, id: stark_model::AssetId, bytes: Vec<u8>) {
    if let Some(broadcaster) = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    {
        broadcaster.add_content(stark_net::AssetNeed::Picture(id), bytes);
    }
}
