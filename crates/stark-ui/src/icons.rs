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
//!
//! "Every icon in that directory" is a claim about files nobody reviews — a Phosphor
//! download arrives with `fill="#000000"` baked in, and an icon that keeps it looks
//! *right* in a file browser and paints a black glyph on near-black chrome. Nothing
//! about the call site would say which happened, so [`tests::every_icon_inherits_its_colour`]
//! checks the directory rather than the table: an icon added and not yet wired up is
//! caught before it is ever drawn, which is when a wrong fill is cheapest to fix.

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
    ADD_FRAME => "plus-square-bold",
    // Take this one away, drawn in the row it would take: a layer in the Layers panel,
    // a perspective in the Guides panel. One glyph for both, because it is one control
    // — the two rosters differ in what they list, not in what removing means.
    //
    // A trash rather than the stack-minus that mirrored "+ Layer". That mirror was an
    // argument about *header* buttons — one stack gaining a member, the other losing
    // one — and it stopped being true when Remove moved onto the rows, where there is
    // no "+" beside it to mirror and no stack in a guide to lose from. What the glyph
    // has to say now is that this is the destructive control, which is the one thing a
    // trash says everywhere.
    REMOVE => "trash-bold",
    // The frame bar's own mark. Crop marks rather than a frame outline, because the bar
    // is not *about* the rectangle — it is about deciding what the piece is, which is
    // the one job a frame does and the reason it clips nothing.
    FRAME => "crop-bold",
    // Grouping, drawn as the move itself rather than as a picture of a group. The pair
    // has to read as one gesture and its undo, which is why it is one glyph mirrored
    // rather than two pictures — the same argument the stack pair above is built on.
    // An elbow rather than a folder because what these commands change is *membership*
    // (§14.2), and membership is a thing the panel already draws: the indent. So the
    // arrow turns the way the row's own indent is about to turn — in to the right, out
    // to the left — and the glyph is a preview of the panel after the click.
    CARRY => "arrow-elbow-down-right-bold",
    RELEASE => "arrow-elbow-up-left-bold",
    // The disclosure mark on a group, centred on the top edge of its row. Both states
    // point *up*, because in this panel what a layer carries is drawn above it
    // (§14.6) — so the caret is aimed at the members either way, and what changes is
    // whether it can get to them. Open, it is a bare caret pointing at rows that are
    // there; shut, it runs into a lid, and what is behind the lid is what has been
    // folded away. One picture with one thing added, which is the same argument the
    // Carry/Release pair above is built on — a rotation would have said "shut" by
    // pointing away from the very rows it is about.
    FOLD_OPEN => "caret-up-bold",
    FOLD_SHUT => "caret-line-up-bold",
    // Opacity sliders.
    OPACITY => "ghost-bold",
    // The blend mode drop-down.
    BLEND => "waves-bold",
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
    // Dismissal, everywhere: a panel's header, Timeline mode's bar. One constant
    // rather than a glyph per site, because closing is one control wherever it is
    // drawn — the same argument [`REMOVE`] is one mark across two rosters.
    CLOSE => "x-bold",
    // The command rail's two triggers and its one button. A rail is read as a column
    // of destinations rather than of verbs, so these say *where* rather than what:
    // a list of commands, a grid of panels, the preferences.
    MENU => "list-bold",
    PANELS => "dots-nine-bold",
    SETTINGS => "gear-bold",
    // The history, in both directions. Deliberately *not* the elbow arrows above —
    // those already mean membership in the Layers panel, and a glyph that means two
    // things means neither. A turning arrow is what undo is drawn as everywhere, and
    // the pair differs only in which way it turns, which is the whole difference
    // between the two commands.
    UNDO => "arrow-counter-clockwise-bold",
    REDO => "arrow-clockwise-bold",
    // The brush editor's "restore the default test stroke". The same file as [`UNDO`],
    // and that is the point rather than an oversight: putting the preview back is an
    // undo narrowed to the one thing the dialog owns. Named for the control, as
    // everything here is, so the call site says what it does and not where the glyph
    // came from.
    RESET => "arrow-counter-clockwise-bold",
    // The file commands. Nothing in the drawing looks like these, so they are the
    // conventional pictures — the ones a hand arrives already knowing — rather than
    // anything derived from what Stark does with them.
    OPEN_DOC => "folder-open-bold",
    SAVE => "floppy-disk-bold",
    EXPORT => "export-bold",
    SHARE => "share-network-bold",
    // The invite link's Copy. A clipboard is what the act *is* here, which is worth
    // saying because the button's label changes to "Copied" and back: the glyph is
    // the half that holds still. Named for the destination rather than for the verb,
    // because the verb is shared: [`DUPLICATE`] below is also a copy, and the two are
    // different acts — one puts a string somewhere else, the other puts a second
    // layer in the document.
    COPY_TO_CLIPBOARD => "clipboard-bold",
    // Duplicate, in the row it would duplicate — a layer in the Layers panel, a
    // perspective in the Guides panel, one glyph for both on the argument [`REMOVE`]
    // is one mark across the two rosters. Two sheets, one behind the other: the
    // picture is the panel *after* the click, which is the same thing the Carry and
    // Release elbows are drawn to say.
    DUPLICATE => "copy-bold",
    // Timeline mode and its transport (§18.2.4). A slideshow for the mode, because
    // what the mode offers is the log played back rather than a clock or a strip of
    // film; play and pause for the transport, which have no second reading.
    TIMELINE => "slideshow-bold",
    PLAY => "play-bold",
    PAUSE => "pause-bold",
    // A menu entry that is a *mode* carries a tick (§18.2.4). Like [`VISIBLE`] and
    // unlike everything else here, it answers "am I in it?" rather than naming an act
    // — which is why it is a state where its neighbours in the menu are verbs.
    CHECK => "check-bold",
    // The same file, and a second name because the *control* is a different one: every
    // "Done" in the application — the transform, guide and frame bars, and the dialogs
    // that have nothing to cancel. A tick is what it means to be finished, so this is
    // the one act in the set drawn as its own outcome.
    //
    // Not `CHECK` at the call sites, even though the bytes are identical: a Done button
    // is a verb and a menu tick is a state, and reading `icons::CHECK` on a button
    // would invite someone to "unify" the two the next time either wants to change.
    // The same argument [`RESET`] and [`UNDO`] are two names for one arrow.
    DONE => "check-bold",
    // Last in the commands menu, and the only entry there that is not a thing to do
    // to the drawing — so it gets the one glyph in the set that is not about one.
    CREDITS => "heart-bold",
    // Making one, wherever there is no stack to add to and no region to bound: a new
    // document, a brush shape imported into the gallery. A bare plus rather than one
    // of the two loaded "add" marks above, because neither a document nor a stamp is
    // a layer or a frame.
    ADD => "plus-bold",
    // The transform gesture (§16.6). Arrows out of a centre rather than a box of
    // handles, deliberately: the widget is an ellipse and the module's whole claim is
    // that there are no handles — a glyph of eight grips would promise a control that
    // is not there. What the four arrows say instead is that the paint is about to be
    // moved and reshaped, which is true of all three of the mode's gestures.
    TRANSFORM => "arrows-out-cardinal-bold",
    PERSPECTIVE => "perspective-bold",
    // The transform bar's Warp family (§16.9). A curve with its control
    // points, which is what the gesture *is*: a smooth surface bent by the
    // handles through it. Perspective takes [`PERSPECTIVE`] — the same subject
    // the guides panel wears, and the sharing is the claim: both are one
    // projective space looked at two ways.
    WARP => "bezier-curve-bold",
    // Alt over the brush (§18.0.2). The bar comes up on the modifier and names itself
    // with the tool it arms — which is the whole discoverability argument that bar is
    // built on, made in a picture as well as a word.
    EYEDROPPER => "eyedropper-bold",
    // The brush editor, off the Brush panel. A wrench rather than a brush: the panel
    // that hosts the button already wears the brush ([`BRUSH`]), and what this opens
    // is the place where the brush is *adjusted*, not another brush.
    EDIT_BRUSH => "wrench-bold",
    // The brush editor's parameter groups. Each says what the group is *about* — the
    // footprint the stroke sweeps, the paint the brush carries down, and the canvas
    // paint it lifts and moves — which is the same job the group's sentence does, in
    // the space beside the chevron. Its fourth group, colour, takes [`COLOR`] below.
    //
    // `PAINT` is the *outline* twin of [`PAINT_BUCKET`], and that is the reason both
    // files are here: the solid one is the Fill action, the one glyph in the set that
    // ever carries a colour, and letting a section header wear it would spend that
    // distinction on a heading.
    TIP => "shapes-bold",
    PAINT => "paint-bucket-bold",
    PICKUP => "drop-simple-bold",
    // Sliders. Each says what the number *does* rather than naming the parameter: a
    // rule for how wide, moving air for how much the brush lets go, a feather for an
    // edge that stops being an edge.
    //
    // An earlier note here said a slider's word and track already identify it and so
    // only the two or three reached for mid-stroke needed marking. That was true of a
    // chrome whose words are always on screen, and it stops being true the moment the
    // words can be turned off: a control's mark is what survives its label, so a bare
    // label is a control that would disappear. Marks lead now, and the word is the
    // half that is optional.
    SIZE => "ruler-bold",
    FLOW => "wind-bold",
    FEATHER => "feather-bold",
    // Two of the perspective bar's four group heads (§20.5). `LOCK` and "Show" label
    // three identical X/Y/Z chips each, so the word was carrying the whole difference
    // between two triplets that look the same; `DENSITY` draws a fan of lines from a
    // point, which is exactly what its number counts — the fan *is* the guide's
    // parametrization. The fourth, Opacity, takes [`OPACITY`].
    //
    // "Show" has no constant of its own: it takes the eye the layer and guide rows
    // wear ([`VISIBLE`]), because it is the same question asked of a fan of guide
    // lines instead of of a layer.
    LOCK => "lock-bold",
    DENSITY => "vector-three-bold",
    // The perspective bar's lens toggle (§20.8): a gridded globe — straight
    // lines bowed onto a sphere, which is what the stereographic fisheye does
    // to the guide. Hand-drawn in the Phosphor bold grammar; not from the set.
    FISHEYE => "fisheye-bold",
    // The nouns the floating panels are about (§11), worn by the title bar and by the
    // entry that opens that panel in the Panels menu — so a panel and the way back to
    // it are one picture rather than two names that happen to match.
    //
    // Three of them are shared rather than duplicated, and the sharing is the claim:
    // `SELECTION` and `PERSPECTIVE_GRID` also head the bars that serve those panels, and
    // `COLOR` also heads the brush editor's colour-dynamics group. A panel, its bar
    // and the dialog that expands it are three views of one subject, so they are one
    // mark — the same argument the frame bar's crop marks are the frame's.
    NAVIGATOR => "magnifying-glass-bold",
    COLOR => "palette-bold",
    BRUSH => "paint-brush-bold",
    SELECTION => "selection-bold",
    LAYERS => "stack-bold",
    PERSPECTIVE_GRID => "vector-three-bold",
    LIGHTING => "sphere-bold",
}

/// One icon, sized and coloured by whatever it sits in (`.icon` in `stark.css`).
///
/// `dangerous_inner_html` is what the module doc is about: the markup is ours,
/// compiled in from a file in this repo, so no untrusted string comes near it.
pub fn icon(svg: &'static str) -> Element {
    rsx! { span { class: "icon", dangerous_inner_html: svg } }
}

/// A control's word, marked as the half that minimal mode may take away (§11).
///
/// The rule this encodes is about *where* a control is, not what it does. Chrome over
/// the canvas — the panels and the bottom bars — is what minimal mode quiets, because
/// there the mark and the slot already say which control this is and the words are
/// what stands between the artist and the painting. Four kinds of text are left alone:
///
/// - **dialogs**, which obscure the canvas anyway, so there is nothing to win;
/// - **anything transient** — an open menu, an expanded drop-down — which is on screen
///   only because it was asked for, and is read rather than recognised;
/// - **panel titles**, whose space is the drag handle and would be spent either way;
/// - **names the user chose** — layers, guides, presets — which no mark can stand in
///   for, since the whole point of them is that they are not predictable.
///
/// So the failure mode is chosen deliberately: a word that is *not* wrapped keeps
/// itself. Forgetting the wrapper leaves a label on screen in minimal mode, which is
/// merely untidy; the opposite default would make a forgotten exception erase a
/// control's only description, and nothing on screen would say which had happened.
/// That is also why this is a wrapper rather than a `font-size: 0` on the container:
/// hiding text by inheritance would put every piece of kept text one missed override
/// away from vanishing.
///
/// What this wrapper claims is only that the text *is a control's name* — not that it
/// will be hidden. Where it is rendered decides that, because a shared component cannot
/// know: [`crate::widgets::Slider`] is used in the panels and in the brush editor
/// alike, and the same `.label` has to go in the first and stay in the second. The
/// stylesheet holds that half, scoped to the dialog backdrop rather than to any
/// particular control.
pub fn label(text: &str) -> Element {
    rsx! { span { class: "label", "{text}" } }
}

/// The same icon at the command rail's weight (`.icon.icon-lg`).
///
/// The rail is 40px of button carrying one mark and no word, where every other icon
/// in the application sits beside a label at a shared 14px. A glyph sized for that
/// company reads as lost in a button nearly three times its height — which is what
/// the 18px characters the rail wore were already compensating for. One extra class
/// rather than a second `icons!` table: it is the *setting*, not the glyph, that is
/// different here.
pub fn icon_large(svg: &'static str) -> Element {
    rsx! { span { class: "icon icon-lg", dangerous_inner_html: svg } }
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

#[cfg(test)]
mod tests {
    /// Every file in `assets/icons` paints with `currentColor` — the one property the
    /// module is built on (see the module docs).
    ///
    /// The *directory*, not the table above, and that is the whole value: the table
    /// only holds icons someone has already wired up, and an icon is at its most
    /// fixable in the commit that adds the file. Checking the table would move the
    /// failure to whenever the glyph first got a call site, which may be a different
    /// change by a different hand.
    ///
    /// It reads bytes rather than parsing SVG on purpose. A hard fill is the one way
    /// these files go wrong — Phosphor exports `fill="#000000"` — and the check that
    /// catches it is "no colour literal anywhere in the file", which is stricter than
    /// "the root fill is right" and cannot be fooled by a colour further down a path.
    #[test]
    fn every_icon_inherits_its_colour() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icons");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("assets/icons is readable") {
            let path = entry.expect("directory entry").path();
            if path.extension().is_none_or(|e| e != "svg") {
                continue;
            }
            let svg = std::fs::read_to_string(&path).expect("icon is UTF-8");
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            assert!(
                svg.contains(r#"fill="currentColor""#),
                "{name} does not paint with currentColor, so it will ignore the colour \
                 of the control it sits in"
            );
            // A `#` in one of these files is a hex colour and nothing else — the paths
            // are numbers and letters, and there is no `url(#…)` or gradient in the set.
            // Whatever it painted would be worn *instead* of the control's colour, which
            // on Stark's dark chrome is usually a black mark on a near-black chip.
            assert!(
                !svg.contains('#'),
                "{name} carries a hard-coded colour; icons take their colour from the \
                 control around them"
            );
            checked += 1;
        }
        // A rename or a moved directory must fail loudly rather than pass by finding
        // nothing to check.
        assert!(checked > 0, "no icons found in {}", dir.display());
    }
}
