//! Stark's Dioxus **web** frontend (§11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. The canvas fills the window;
//! unobtrusive floating panels (color, brush, layers) sit on top.
//!
//! Run with `dx serve --web -p stark-dioxus-frontend` in a WebGPU-capable browser.
//!
//! # What is in this file
//!
//! The root, and only the root: [`run`] — all the binary's `main` does — and `app`,
//! which is the list of what is on screen and the order it is stacked in. What the app
//! does once at start is `startup`'s.

// `rsx!` lowers every interpolated attribute and text node — `id: "{CANVAS_ID}"`,
// `"{title}"` — through `format!`, so clippy sees a `format!` with nothing to
// format and suggests `.to_string()`. The suggestion applies to the expansion,
// not to anything writable in the source: plain interpolation *is* the idiom
// here. Suppressed crate-wide because it fires wherever rsx! does.
#![expect(
    clippy::useless_format,
    reason = "rsx! expands plain interpolation through format!; the suggestion applies to the expansion"
)]

mod anchor;
mod base64;
mod brush_editor;
mod builtin_ids;
mod builtins;
mod canvas;
mod cards;
mod collab;
mod commands;
mod credits;
mod dialogs;
mod drags;
mod failure;
mod files;
mod gradients;
mod icons;
mod identity;
mod images;
mod input;
mod layer_thumbs;
mod layout;
mod library;
mod modes;
mod navigator;
mod overlays;
mod panels;
mod platform;
mod prefs;
mod presets;
mod preview;
mod rail;
mod records;
mod render;
mod settings;
mod shapes;
mod shipped;
mod slots;
mod startup;
mod state;
mod substrates;
mod thumb_cache;
mod thumbs;
mod timings;
mod tutor;
mod visibility;
mod widgets;

use dioxus::prelude::*;

use canvas::Canvas;
use dialogs::DialogStack;
use input::{bind_context_menu, bind_pen, bind_shortcuts};
use layout::PanelStack;
use layout::{resize_end, resize_move};
use navigator::NavigatorOverlay;
use overlays::{BrushCursor, PeerCursors, PickLoupe, TowStringOverlay, TuneReadoutOverlay};
use panels::{
    FilterBar, FrameBar, ModeBars, ModeCatcher, PickBar, SelectionBar, StackPopouts, TimelineBar,
};
use rail::CommandRail;
use slots::SlotOverlay;
use state::{AppState, use_pref};

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by `app` so the rsx!
/// blocks carry class names, not inline styles.
static STARK_CSS: Asset = asset!("/assets/stark.css");

/// Launch the app: the whole of the binary's `main`.
pub fn run() {
    platform::install_panic_hook();
    platform::install_tracing();
    // Before anything reads the browser's store, say what the store *is*: the format
    // is `stark_ui`'s and shared with the native frontend, and where the bytes go
    // is this frontend's alone (§11.2). Nothing here fails without it — every read
    // answers "nothing stored" — which is why it is an install rather than a
    // constructor everything else threads, and why it is **here** rather than in the
    // root's body: `AppState::new` reads four records while it is building its
    // signals, so a hook in the body is already too late for them, whatever it sits
    // above. The native frontend installs in its own `main` for the same reason.
    stark_ui::storage::install(platform::LocalStore);
    // Then drop the keys the old formats were kept under, so their bytes are not
    // still spending this origin's quota (`storage::drop_retired`, which says when to
    // delete this line).
    stark_ui::storage::drop_retired();
    dioxus::launch(app);
}

fn app() -> Element {
    // Root-owned, because the collaboration pumps and the renderer's async init are
    // detached tasks living in `ScopeId::ROOT` — see `state::root_signal`.
    let state = AppState::new();
    use_context_provider(|| state);

    // The keyboard shortcuts live on the window, not on the root element below, so
    // they answer whatever has focus — including `document.body`, where the browser
    // leaves it after a clicked button unmounts itself (see `platform::on_window_key`).
    use_hook(|| bind_shortcuts(state));
    // And the pen's other end, on the window for the same reason: it is a hold
    // like a number key's, so it belongs to no one surface (§18.1.8).
    use_hook(|| bind_pen(state));
    // And the browser's context menu, refused on the same grounds: a pen held
    // still is this app's own gesture (§6.9), and Windows reads the same hold as
    // a right-click.
    use_hook(bind_context_menu);
    // And the way out: the browser asks before the tab goes, if there is committed
    // work here that neither Save nor Export has taken out (`files::guard_unload`).
    // In the root's body rather than at the end of the startup task, unlike the
    // launch queue and the paste hook — this one binds a predicate over the
    // signals, so it wants no engine and cannot be left unbound by a start that
    // fails before there is one.
    use_hook(|| files::guard_unload(state));

    use_hook(|| startup::load_records(state));

    // Every brush with a picture to show wants a rendered stroke (`crate::thumbs`,
    // §11): the preset library's rows and the quick-brush rack's overlay
    // (§18.1.8). In the **root** rather than in either viewer, because neither is
    // always mounted — the Brush panel closes, and the rack's overlay exists only
    // while a key is held, which is far too late to start rendering the thing it
    // is there to show. Generation needs the main renderer, so this watches the
    // renderer alongside the two libraries: whichever lands last kicks it off, and a
    // slot tuned under a hold re-runs it on the release that stores it.
    //
    // Subscribing reads, because `thumbs::refresh` peeks. The renderer through
    // `renderer_ready`, which moves once, rather than the renderer signal, which every
    // command writes (U2).
    use_effect(move || {
        let _ = state.presets.read();
        let _ = state.slots.brushes.read();
        let _ = (state.renderer_ready)();
        thumbs::refresh(state);
    });

    use_hook(|| spawn(startup::run(state)));
    // Readers that re-render at cost read a slice: the whole tree hangs off this one.
    let minimal = use_pref(state, |p| p.minimal);

    rsx! {
        document::Stylesheet { href: STARK_CSS }

        div {
            class: "app-root",
            // Two modes that the whole tree is inside of, so both ride the root element
            // and the stylesheet does the rest — one rule each, rather than every
            // control learning about a state it does not own.
            //
            // Timeline mode reserves the foot of the window for its bar, and the
            // stylesheet lifts the bottom-bars column clear of it. Minimal mode hides
            // the `.label` spans the chrome marks its words with (`icons::label`); it
            // is a class here rather than a prop for the same reason — a boolean
            // threaded through the tree would have to reach every control, and the one
            // that failed to pass it on would be the one that kept its word.
            class: if (state.timeline.open)() { "timeline-mode" },
            class: if minimal() { "minimal" },
            // A panel resize by the bottom-edge grip is driven here — events bubble up even
            // over the canvas, so it keeps tracking wherever the pointer goes, and leaving
            // the window ends it so it cannot get stuck. A no-op unless armed.
            //
            // The title-bar *reorder* drag is not here: it captures the pointer and handles
            // its own move and release (`layout::Panel`), which is the only way to be sure
            // of getting the release.
            onpointermove: move |e| resize_move(state.panels, &e),
            onpointerup: move |_| resize_end(state.panels),
            onpointerleave: move |_| resize_end(state.panels),

            // Dropping a file anywhere on the app (§23.4). At the **root** rather
            // than on the canvas, and that is not for convenience: an unclaimed
            // drop is one the browser handles, and what the browser does with a
            // dropped file is navigate to it — discarding an unsaved painting. So
            // every drop over the window has to be claimed, including the ones that
            // land on a panel, and `preventDefault` on `dragover` is what makes the
            // window a drop target at all.
            //
            // A descendant that wants a drop for itself takes it by stopping
            // propagation — both asset libraries do (`widgets::AssetGallery`), because
            // dropping a stamp or a weave into a library and placing a photograph on
            // the canvas are different acts.
            ondragover: move |e| e.prevent_default(),
            ondrop: move |e| {
                e.prevent_default();
                images::drop_files(state, &e);
            },

            Canvas {}

            // Over the canvas but *under* all the floating chrome: the live mode's
            // catcher — the transform's box (§16.6), the guide edit (§20.5), the
            // gradient trace (§22.2) or the gradient fill's axis (§22.4) — or, with no
            // mode composing, the selected frame's edges and handles (§15.7), whose
            // interior passes pointer events through to the painting.
            ModeCatcher {}

            // Collaborators' pointers, over the canvas and under the chrome
            // (§17.4). Empty and free when solo.
            PeerCursors {}

            // The live brush under the resting pointer (§18.1.10), same layer:
            // the size the next stroke would land at, riding the hover. Empty
            // and free while the pointer is off the canvas.
            BrushCursor {}

            // The brush-tuning drag's readout — the size ring or the flow bar — in
            // the same layer and mounted for the same reason (§18.1.9). Empty and
            // free unless a tuning drag is in flight.
            TuneReadoutOverlay {}

            // The tow string while a smoothing brush draws (§6.11), same layer
            // again. Empty and free unless a rope is in flight.
            TowStringOverlay {}

            // The held touch pick's swatch (§18.1.11), same layer once more.
            // Empty and free unless a finger is holding one.
            PickLoupe {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // The chrome down the left, under the rail: the quick-brush rack while
            // a number key is held (§18.1.8), and the navigator's miniature in the
            // corner (§11). Neither is a panel — one is summoned by a finger already
            // on the keyboard, the other is a picture read at a glance — and both
            // stand on this edge, which is the whole reason they share a box.
            //
            // A column rather than two independently placed overlays, because the
            // two are sized by things that have nothing to do with each other: the
            // rack by how many digits are filled, the miniature by the artwork's
            // aspect. Any pair of fixed offsets that kept them apart on this window
            // would put one over the other on a shorter one. Here the navigator
            // takes what it needs from the bottom and the rack centres its rows in
            // whatever is left, so "they do not overlap" is a fact about the layout
            // rather than an arithmetic nobody re-checks.
            //
            // The box itself takes no pointer events; its children decide for
            // themselves (the rack only while it is pinned, the miniature always).
            div { class: "left-chrome",
                SlotOverlay {}
                NavigatorOverlay {}
            }

            // Floating tool panels, stacked top-right — order + visibility are data-driven.
            PanelStack {}

            // Whatever a panel has flown out beside itself: the canvas colour's
            // picker, the surface gallery (§25.7). Here rather than in the panel
            // because the column clips, so a pop-out has to be placed against its
            // row's measured box instead of drawn in it (`panels::popout`). After
            // the stack in the DOM for the tour card's reason — it stands *over*
            // the panel it flew out of — and empty whenever none is open.
            StackPopouts {}

            // The guided tour's card, beside whichever panel or bar the lesson on
            // screen is about (§24). After the stack in the DOM, which is what puts
            // it over the panel it points at rather than under it — it is positioned
            // against that panel's own measured box, so an overlap of a pixel or two
            // where the shadows meet is expected and must land the right way round.
            // Empty and free whenever no lesson is showing.
            tutor::TutorCard {}

            // Bottom-centre: the bars that are mounted only while the thing they act
            // on exists. Stacked in one column so a frame and a selection in force at
            // the same time sit above one another instead of on top of each other.
            //
            // Two registers share the column (MODAL_DESIGN.md): standing-state
            // bars describe a fact and dissolve with it, while a composing
            // mode's bar (`mode-bar`) fronts a catcher that has taken the
            // pointer, wears the armed accent, and stands the others down —
            // *recessed* rather than unmounted, so the place its Done and Esc
            // return to stays on screen.
            // The column is anchored at its foot, so an earlier child stands
            // *higher* — and the order here is the stack's: deepest first, so
            // each bar lands **above** the one it covers, the way a card lands
            // on a pile. A trace stands over the gradient bar it parked, a
            // mode over the standing bars it recessed.
            //
            // The covered bars leave the flow entirely (`.recessed` is
            // absolute), which is what holds the live bar still: it is the
            // column's one in-flow child, at the baseline whatever the stack
            // holds, and the bars it covers peek out beneath it — lower and
            // fainter with depth, priced by a sibling chain that this order
            // is load-bearing for.
            div { class: "bottom-bars",
                // The eyedropper's options, present only while Alt arms it
                // (§18.0.2). First, deliberately: in a foot-anchored column a
                // child mounting *above* the others moves none of them, so the
                // rack coming and going with the Alt key never bumps the bar
                // somebody is working in. It is no part of the stack below —
                // it coexists with painting rather than covering anything — so
                // it floats above the pile.
                PickBar {}
                // The composing mode's bar, deepest first: the trace's Cancel
                // (§22.2), the transform's flips (§16.6), the gradient fill's axis
                // kinds (§22.4) — recessed under the trace's bar while one has it
                // parked — and the guide edit's locks and dials (§20.5). Each stands
                // over the standing bars below it.
                ModeBars {}
                // The whole selection's opacity and commands, present while
                // there is a selection or a shape tool is armed to make one — so
                // it doubles as the "canvas is masked" indicator.
                SelectionBar {}
                // The frame's composition controls, present only while a frame is
                // selected for composing (§15.7).
                FrameBar {}
                // The selected filter layer's own numbers, present only while one is
                // selected (§21.6) — the same bargain as the frame bar, and mutually
                // exclusive with it by construction, since both key off the single
                // selected layer being of their kind.
                FilterBar {}
            }

            // Timeline mode's own bar, below the bottom-bars column and spanning
            // the window rather than hugging its contents: the others are commands
            // that happen to apply right now, while this is a *mode* the whole
            // canvas is in, and a scrubber wants every pixel of width it can get
            // (§18.2.4). Mounted from here rather than gated inside
            // itself, because it owns hooks — a component may not gain or lose
            // those between renders.
            if (state.timeline.open)() {
                TimelineBar {}
            }

            // Every open dialog, in the order they were opened, so the last one opened
            // is drawn over the rest (`crate::dialogs`). Each is mounted only while
            // open, so it opens fresh.
            DialogStack {}

            // Last, and over everything: the GPU has died and the canvas is showing
            // its final frame (§5, `crate::failure`). It gates on its own read of
            // the projection rather than on a condition here, because the same read
            // is what stops the engine doors — one field, so the app cannot be
            // stopped without saying so or say so while still running. Empty and
            // free on a healthy device, which is every ordinary session.
            failure::GpuFailureModal {}
        }
    }
}
