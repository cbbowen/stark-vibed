//! The icons the controls wear (`assets/icons`).
//!
//! Embedded with `include_str!` and dropped into the DOM **inline**, rather than
//! fetched as an `asset!` and hung in an `<img>`. Every icon in that directory
//! paints with `fill="currentColor"`, and inline is the only place that resolves
//! the way we want: the glyph inherits the colour of the control around it, so one
//! file covers a resting chip, a lit chip's white-on-blue, and a disabled chip's
//! fade — with nothing to keep in sync. Inside an `<img>`, `currentColor` resolves
//! against the *image's* own root instead, which on our dark chrome means a black
//! glyph on a near-black chip.
//!
//! Inlining also spends no fetch, which is worth more here than elsewhere: this is
//! the wasm build, and these are ~200-byte files.

use dioxus::prelude::*;

/// `NAME => "file"` for each icon, embedded from `assets/icons/<file>.svg`.
///
/// The names are what the *control* means rather than what the glyph draws
/// (`SELECTION_NEW`, not `RECTANGLE_DASHED`), because the call sites are controls —
/// except the three shape tools, where the glyph *is* the meaning.
macro_rules! icons {
    ($($name:ident => $file:literal),* $(,)?) => {
        $(pub const $name: &str = include_str!(concat!("../assets/icons/", $file, ".svg"));)*
    };
}

icons! {
    RECTANGLE => "rectangle-bold",
    CIRCLE => "circle-bold",
    LASSO => "lasso-bold",
    SELECTION_NEW => "rectangle-dashed-bold",
    // The three combining modes, drawn as the set operation itself: two overlapping
    // squares with the kept part filled. They are one family where the marks they
    // replaced were three unrelated pictures of a marquee — and the family matters
    // more than the picture here, because the row's whole claim is that Add, Sub and
    // Isect are the same question answered three ways (see `panels::select`). Squares
    // rather than the bare `unite`/`subtract` circles so they sit at the weight of the
    // rectangle beside them.
    SELECTION_ADD => "unite-square-bold",
    SELECTION_SUB => "subtract-square-bold",
    SELECTION_ISECT => "intersect-square-bold",
    SELECTION_NONE => "selection-slash-bold",
    SELECTION_INVERT => "selection-inverse-bold",
    // The one solid glyph in the set, and the only one that ever wears a colour of its
    // own ([`icon_tinted`]): an outline weight has barely any interior to tint, so the
    // bucket is `fill` where its neighbours are `bold`. That break is not an
    // inconsistency — it is the *reason* this chip looks different from the four beside
    // it, which is the same job the Fill chip's swatch used to do.
    PAINT_BUCKET => "paint-bucket-fill",
    // The Layers panel's header. A frame is a layer, so the two "add" buttons sit side
    // by side and the glyphs have to carry the difference the words used to: a stack
    // gains a member, versus a single bordered region coming into being — which is what
    // a frame is (§15.7), and why it is a square rather than a stack.
    ADD_LAYER => "stack-plus-bold",
    REMOVE_LAYER => "stack-minus-bold",
    ADD_FRAME => "plus-square-bold",
    // The frame bar's own mark. Crop marks rather than a frame outline, because the bar
    // is not *about* the rectangle — it is about deciding what the piece is, which is
    // the one job a frame does and the reason it clips nothing.
    FRAME => "crop-bold",
    // Grouping, as a folder gaining or losing a member. The pair has to read as one
    // gesture and its undo, which is why it is one glyph mirrored rather than two
    // pictures — the same argument the stack pair above is built on. A folder rather
    // than a stack because these two commands are about *membership* (§14.2), which is
    // the fact the panel draws as an indent; the stack pair is about existence.
    CARRY => "folder-simple-plus-bold",
    RELEASE => "folder-simple-minus-bold",
    // The clip toggle. What a clipped layer is bounded by is the paint under it, so the
    // glyph is a picture in a frame — an image with a silhouette, which is the thing
    // doing the bounding. It sits beside the blend picker because both answer *how does
    // this layer meet what is below it*, and the two together are one row of that
    // question rather than a drop-down and a sentence with a tick-box.
    CLIP => "image-square-bold",
    // The two flips on the transform bar. Here the glyph *is* the meaning — each one
    // draws its own axis of mirroring, which is the whole difference between the two
    // buttons, and is what the `\u{2194}` / `\u{2195}` arrows in their labels used to
    // carry.
    FLIP_H => "flip-horizontal-bold",
    FLIP_V => "flip-vertical-bold",
    // Per-row visibility. Unlike every other icon here this one is a *state* rather
    // than an act: the row shows the eye it currently is, not the one clicking would
    // give you. That is the way the tick-box it replaces read, and the way the same
    // control reads in every other painting application — a row of eyes is scannable
    // in a way a row of "would-hide" glyphs is not.
    VISIBLE => "eye-bold",
    HIDDEN => "eye-slash-bold",
}

/// One icon, sized and coloured by whatever it sits in (`.icon` in `stark.css`).
///
/// `dangerous_inner_html` is what the module doc is about: the markup is ours,
/// compiled in from a file in this repo, so no untrusted string comes near it.
pub fn icon(svg: &'static str) -> Element {
    rsx! { span { class: "icon", dangerous_inner_html: svg } }
}

/// The same icon holding a paint colour rather than the colour of its control — for
/// the one glyph that has to say *which* paint the act would lay, not only which act.
///
/// The colour arrives as the brush's RGBA, and its alpha is the paint's opacity
/// (per-unit, as everywhere in Stark), so a thin wash has to *look* thin. That is why
/// the glyph is drawn twice, the paint over an untinted copy of itself: the copy
/// underneath is the white base the old swatch had, and it is what a 15% wash tints.
/// Laid straight onto the chip's dark ground the same wash would come out a dim grey
/// bucket — the paint would read as *dark* rather than as *thin*, which is the one
/// thing this glyph exists to get right.
///
/// The base takes the control's own colour rather than a hard white, so at zero
/// opacity the bucket is exactly its four neighbours, and it fades with a disabled
/// chip like they do. One path, twice: the layers cannot drift apart.
pub fn icon_tinted(svg: &'static str, color: [f32; 4]) -> Element {
    let c = |i: usize| (color[i] * 255.0).round().clamp(0.0, 255.0) as u8;
    let paint = format!("color: rgba({}, {}, {}, {})", c(0), c(1), c(2), color[3]);
    rsx! {
        span { class: "icon tinted",
            span { dangerous_inner_html: svg }
            span { class: "icon-paint", style: "{paint}", dangerous_inner_html: svg }
        }
    }
}
