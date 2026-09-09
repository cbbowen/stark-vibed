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
//! which `tests/no_toolkit_types.rs` holds by reading the source rather than by
//! trusting the manifest: a type can arrive through a re-export the dependency list
//! does not show. It lives outside `src` so that the strings it bans are not in the
//! tree it walks — this file used to be exempted from its own check for naming all
//! five of them, and an exemption is a hole shaped like whatever grows into it.
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
//! - [`brush_editor`] — what the full brush editor shows (§6.2): every parameter it
//!   offers, the range each is offered over, which group it belongs to, and the test
//!   stroke the preview lays. The rows are here and the markup is each frontend's.
//! - [`transform`] — the transform mode's algebra (§16.6, §16.8, §16.9). Named for
//!   what it computes; it was `gesture` next to five *input* gestures that are not
//!   this, and the name would have been read as those here.
//! - [`layer_tree`] — what the Layers panel draws, and what a drop into it means
//!   (§14.6, §14.8).
//! - [`reorder`] — moving a row of a list by dragging it, with no opinion about
//!   what the list is. Two panels are rosters; this is the gesture they share.
//! - [`library`] — the gallery thumbnails a browser-held asset library shows
//!   (§6.4, §6.6).
//! - [`storage`] — the records a client keeps between visits, the one JSON format
//!   they are kept in, and the [`Backend`](storage::Backend) a frontend installs to
//!   say where they actually go (§25.6).
//! - [`identity`] — the key this client's `ActorId` derives from, and the run counter
//!   beside it (§17).
//! - [`prefs`] — the standing preferences a settings dialog sets.
//! - [`input`] — the two screen-denominated lengths a gesture declares, and the map
//!   from a knob to each (§6.2, §6.11). The module this crate was built to prevent a
//!   second copy of.
//! - [`collab`] — a shared session as a *link* (§12.4): the address a peer opens,
//!   and the ticket read back out of one. The whole of what both frontends say
//!   about sharing that is not the network.
//! - [`pick`] — the eyedropper's options and its arming (§18.0.2): what a sample is
//!   taken with, how a reach resolves against the layer selected *now*, and whether
//!   a press would sample rather than paint.
//! - [`commands`] — the command registry (§11, §25): every simple act the chrome can
//!   ask for, with its name, mark, hint and greying on the variant, and the chord
//!   table a rebinding writes over.
//! - [`drags`] — that registry for the pointer (§25): which chord and button opens
//!   which canvas drag, and the preset tables a hand arrives from another app with.
//! - [`keys`] — one keystroke as both binding tables read it, and the three modifiers
//!   they start from (§25).
//! - [`panels`] — the register vocabulary a chrome is arranged in (§11): which
//!   floating tool panels there are, what each is called and what mark it wears.
//! - [`visibility`] — what a client last had on screen, as one record over the
//!   visibility menu's own list (§11, §25.6).
//! - [`nav`] — what a press, a drag and a wheel notch do to the **view**, at the
//!   rates both apps travel at (§18.1.7).
//! - [`tune`] — and what one does to the **brush** (§18.1.9): the size sideways, the
//!   flow up and down, and the axis lock that keeps a gesture to one knob.
//! - [`selection`] — what a shape gesture is about to do (§6.8, §18.0.4): which tool
//!   draws the region, and what the region it encloses lands on.
//! - [`slots`] — the ten brushes under the hand (§18.1.8): what a digit holds, what a
//!   hold's release keeps and hands back, and when two presses are a pick.
//! - [`presets`] — the brush preset library (§6.2, §18.1.8): named `BrushConfig`
//!   snapshots, and the ones the app ships.
//! - [`assets`] — what an imported image *becomes*, and what a client's asset library
//!   is made of (§6.4, §6.6, §19, §25.6).
//! - [`color`] — the Oklab picker's geometry (§6.7, §11.2): the display gamut's rim,
//!   the wheel fitted to it, and the pictures of both.
//! - [`icons`] — which glyph each control wears, and why (§11, §25).
//! - [`bounds`] — the canvas-space rectangles a frontend asks the document for, and
//!   the one way it grows them.
//! - [`files`] — what a document file is called, and whether closing it would lose
//!   work (§8, §15.6).

pub mod assets;
pub mod bounds;
pub mod brush_config;
pub mod brush_editor;
pub mod collab;
pub mod color;
pub mod commands;
pub mod drags;
pub mod files;
pub mod icons;
pub mod identity;
pub mod input;
pub mod keys;
pub mod layer_tree;
pub mod library;
pub mod nav;
pub mod panels;
pub mod pick;
pub mod prefs;
pub mod presets;
pub mod reorder;
pub mod selection;
pub mod slots;
pub mod storage;
pub mod transform;
pub mod tune;
pub mod visibility;
