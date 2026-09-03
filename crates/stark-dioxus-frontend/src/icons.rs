//! The icons the controls wear, as this frontend draws them (§11, §25).
//!
//! **Which glyph a control wears is `stark_chrome::icons`'**, prose and all — it is a
//! statement about what the control means, and two apps disagreeing about that would
//! be two apps. What is here is the carrier.
//!
//! And the carrier is the reason the catalog had to be a *name* rather than a blob:
//! an icon is dropped into the DOM **inline** here, rather than fetched as an
//! `asset!` and hung in an `<img>`. Every icon paints with `fill="currentColor"`, and
//! inline is the only place that resolves the way we want — the glyph inherits the
//! color of the control around it, so one file covers a resting chip, a lit chip's
//! white-on-blue, and a disabled chip's fade, with nothing to keep in sync. Inside an
//! `<img>`, `currentColor` resolves against the *image's* own root instead, which on
//! our dark chrome means a black glyph on a near-black chip.
//!
//! Inlining also spends no fetch, which is worth more here than elsewhere: this is
//! the wasm build, and these are ~200-byte files.
//!
//! The native frontend reaches the same behaviour by a different route — resvg to an
//! alpha mask, tinted per draw — which is why the catalog is shared and this is not.

use dioxus::prelude::*;
use stark_chrome::icons::Icon;

/// The SVG source for a mark. Empty for a name this build ships no file for, which
/// `stark_chrome::icons::tests::every_icon_has_its_file` rules out — so what a missing
/// one costs is a blank span rather than a panic in a render.
fn svg(mark: Icon) -> &'static str {
    mark.svg().unwrap_or_default()
}

/// One icon, sized and colored by whatever it sits in (`.icon` in `stark.css`).
///
/// `dangerous_inner_html` is what the module doc is about: the markup is ours,
/// compiled in from a file in this repo, so no untrusted string comes near it.
pub fn icon(mark: Icon) -> Element {
    rsx! { span { class: "icon", dangerous_inner_html: svg(mark) } }
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
pub fn icon_large(mark: Icon) -> Element {
    rsx! { span { class: "icon icon-lg", dangerous_inner_html: svg(mark) } }
}

/// The same icon holding a paint color rather than the color of its control — for
/// the one glyph that has to say *which* paint the act would lay, not only which act.
///
/// The color arrives as the brush's RGB with the strength the act would lay it
/// at in the fourth lane — the marquee fill's own opacity, or 1 for the
/// mask-bounded fill — so a thin wash has to *look* thin. That is why
/// the glyph is drawn twice, the paint over an untinted copy of itself: the copy
/// underneath is the light base a 15% wash tints, the same job a swatch's white
/// backing does. Laid straight onto the chip's dark ground the same wash comes out a dim grey
/// bucket — the paint would read as *dark* rather than as *thin*, which is the one
/// thing this glyph exists to get right.
///
/// The base takes the control's own color rather than a hard white, so at zero
/// opacity the bucket is exactly its four neighbours, and it fades with a disabled
/// chip like they do. One path, twice: the layers cannot drift apart.
pub fn icon_tinted(mark: Icon, color: [f32; 4]) -> Element {
    let c = |i: usize| (color[i] * 255.0).round().clamp(0.0, 255.0) as u8;
    let paint = format!("color: rgba({}, {}, {}, {})", c(0), c(1), c(2), color[3]);
    let svg = svg(mark);
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
    /// catches it is "no color literal anywhere in the file", which is stricter than
    /// "the root fill is right" and cannot be fooled by a color further down a path.
    #[test]
    fn every_icon_inherits_its_color() {
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
                "{name} does not paint with currentColor, so it will ignore the color \
                 of the control it sits in"
            );
            // A `#` in one of these files is a hex color and nothing else — the paths
            // are numbers and letters, and there is no `url(#…)` or gradient in the set.
            // Whatever it painted would be worn *instead* of the control's color, which
            // on Stark's dark chrome is usually a black mark on a near-black chip.
            assert!(
                !svg.contains('#'),
                "{name} carries a hard-coded color; icons take their color from the \
                 control around them"
            );
            checked += 1;
        }
        // A rename or a moved directory must fail loudly rather than pass by finding
        // nothing to check.
        assert!(checked > 0, "no icons found in {}", dir.display());
    }
}
