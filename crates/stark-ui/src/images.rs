//! Getting a picture in: an image file, or the clipboard (§23).
//!
//! Two gestures that feel nothing alike — a menu, a file picker, a chosen file; or
//! Ctrl+V — and are the same thing from here down. Both arrive as encoded bytes, both
//! go through the browser's decoder, and both commit one
//! [`DocCommand::PlaceImage`](stark_engine::command::DocCommand::PlaceImage). So the
//! shared half is [`place_bytes`] and the two entry points are three lines each, which
//! is the right proportion: what differs between them is only where the bytes came
//! from and whether there is a filename to name the layer after.
//!
//! Sibling to [`crate::files`] rather than part of it, and the distinction is the one
//! that file's header draws so carefully. Save and Open move the *document*; this moves
//! a **picture into** one. An image placed here is paint from the moment it lands —
//! stroke over it, glaze it, lift it, transform it — not an attachment the document
//! keeps a reference to.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use crate::platform::{decode_image, pick_file};
use crate::state::{AppState, dispatch};
use stark_assetid::Picture;
use stark_engine::command::DocCommand;
use stark_model::geom::IVec2;

/// What the file picker offers. `image/*` rather than a list of extensions, because
/// the decoder is the browser's (§23) — enumerating formats here would be this app
/// claiming to know which ones it supports, and getting it wrong in whichever direction
/// the browser moved.
const ACCEPT: &str = "image/*";

/// Ask for an image file and place it — the menu entry.
pub fn import_image(state: AppState) {
    // Called straight from the menu handler: a file picker may only be opened inside
    // the user gesture that asked for it (`platform::pick_file`).
    pick_file(ACCEPT, move |name, bytes| {
        place_bytes(state, Some(name), bytes)
    });
}

/// Place whatever image is pasted into the page — bound once, for the life of the page.
///
/// Ctrl+V is not among [`crate::input`]'s shortcuts and deliberately is not: a paste is
/// delivered as its own event carrying the clipboard with it, and a keystroke handler
/// would have to go and *ask* for the clipboard afterwards, through an API that prompts
/// for a permission and does not carry images in every browser
/// ([`platform::on_window_paste`](crate::platform::on_window_paste)).
pub fn bind_paste(state: AppState) {
    crate::platform::on_window_paste(move |bytes| {
        // No name: the clipboard has no filename to offer, so the layer is described by
        // its place in the stack, which is what an unnamed layer is for.
        place_bytes(state, None, bytes);
    });
}

/// Decode `bytes` and commit the placement — the half both entry points share.
///
/// Detached (`spawn_forever`) because neither caller has a scope to tie the work to: a
/// menu item unmounts the moment it is clicked, and a paste has no component behind it
/// at all. The task writes nothing but the document, through
/// [`dispatch`](crate::state::dispatch), so there is no signal of a dead scope to write
/// through — the hazard `files::ExportModal` avoids by doing the opposite.
fn place_bytes(state: AppState, name: Option<String>, bytes: Vec<u8>) {
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

        // **Centred on what is being looked at**, which is the only placement that
        // needs no explanation: an image arrives where the eye already is. The view is
        // the frontend's (§18.1.2), so this is the frontend's arithmetic — the engine
        // is told a position, not a policy.
        //
        // Rounded to whole canvas pixels, because that is what the action carries and
        // what makes the placement resample nothing (§23). The engine could not round
        // it for us without the rounding becoming invisible at the call site.
        let Some(at) = crate::state::with_engine_quiet(state, |r| {
            let center = r.view().center;
            IVec2::new(
                (center.x - width as f32 * 0.5).round() as i32,
                (center.y - height as f32 * 0.5).round() as i32,
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
            let obs = state.obs.read();
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
