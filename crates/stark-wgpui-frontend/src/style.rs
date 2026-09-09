//! The chrome's palette and its style classes (§11.2).
//!
//! wgpui ships no stylesheet, so every control dressed itself — sixty-odd hex
//! literals across seven modules, and the same six-call chain wherever a chip could
//! be lit. Rust's answer to a stylesheet is an extension trait: a blanket impl over
//! [`wgpui::Styled`] whose default methods are the classes. A class can take an
//! argument, which a CSS class cannot, so [`StyleExt::lit`] is the whole
//! selected/resting fork rather than two rules and a predicate at every call site.
//!
//! **The colours are `u32` rather than `Rgba`** because they have two consumers: the
//! `Styled` methods, which want `rgb(_)`, and [`crate::icons::icon`], which tints an
//! alpha mask and takes the packed value.
//!
//! Deliberately not the web app's palette — parity is of acts, not appearance (§11.2)
//! — but its vocabulary, so `LIT` here and `--lit-bg` there name the same idea.
//!
//! Two pairs sit closer together than a restyle would want: `INK_ROW` beside
//! `INK_LIT`, `INK_CHORD` beside `INK_LABEL`, each one call site in the menu. Naming
//! them rather than collapsing them is what makes the choice visible to whoever
//! restyles this.
//!
//! [`tip`] is the other half of what the columns are built on. Once a control wears a
//! mark and no word (`crate::panel`), the hover is where its name went — so every
//! chip, track and row in the chrome hangs one, and a control that hangs none is a
//! square nothing on screen explains.

use wgpui::{SharedString, StatefulInteractiveElement, Styled, px, rgb};
use wgpui_component::tooltip::Tooltip;

// --- grounds, darkest first -----------------------------------------------

/// The lightness track and a gallery card: a field something is *shown in*, sunk
/// below the panel so what sits on it reads as ink rather than as chrome.
pub const WELL: u32 = 0x14161a;
/// The menu bar.
pub const BAR: u32 = 0x1a1c1f;
/// A panel column.
pub const PANEL: u32 = 0x1e2124;
/// A drop-down, which floats over the bar and so must be lighter than it.
pub const MENU: u32 = 0x24272b;
/// A resting control: a chip nobody has chosen.
pub const CONTROL: u32 = 0x2a2d31;
/// A row under the pointer.
pub const HOVER: u32 = 0x2f3337;
/// The chosen one — an armed tool, an open menu, the active layer.
pub const LIT: u32 = 0x35496b;
/// How far a dial has been turned — the widget layer's slider bar (`crate::theme`).
pub const FILL: u32 = 0x40474e;
/// The same, on the dial a drag is holding. The one saturated colour in the chrome,
/// spent on the single thing the hand is doing.
pub const ACCENT: u32 = 0x5b9dd9;

/// What a modal dims the window with (`crate::brush_editor`) — a near-black at
/// three-quarters, so the painting still reads through it. `rgba`, like the ground
/// below: the low byte is the alpha.
///
/// Dimmed rather than covered because of what the dialog is *for*. A brush editor is
/// about the next stroke, and the canvas it will land on staying visible is the point
/// — the same argument the transform bar makes one control down.
pub const SCRIM: u32 = 0x0b0d_10c0;

/// [`PANEL`] laid *over* the canvas rather than beside it, for the transform bar:
/// the same ground with the painting showing through, so the bar costs no painting
/// room it was not already costing. `rgba` reads the alpha out of the low byte.
pub const PANEL_OVER_CANVAS: u32 = (PANEL << 8) | 0xe0;

/// A panel's border against its neighbour.
pub const EDGE: u32 = 0x35393d;
/// A drop-down's border, and the rule between two runs of its rows.
pub const RULE: u32 = 0x3d4247;

// --- inks, brightest first ------------------------------------------------

/// On a lit ground, and the default the panels set for anything that names none.
pub const INK_LIT: u32 = 0xe8eaed;
/// A menu row that can be pressed.
pub const INK_ROW: u32 = 0xdfe3e6;
/// Ordinary text on a resting control.
pub const INK: u32 = 0xb0b4b8;
/// A label naming what is beside it, rather than saying anything itself.
pub const INK_LABEL: u32 = 0x9aa0a6;
/// The chord printed beside a live menu row.
pub const INK_CHORD: u32 = 0x8b9196;
/// An unlit chip's glyph. Below [`INK_LABEL`] because a mask at 14px reads heavier
/// than text does at the same value.
pub const INK_MARK: u32 = 0x767b80;
/// A fold triangle: present, but not competing with the title it sits opposite.
pub const INK_FOLD: u32 = 0x6c7378;
/// A command with nothing to act on. Dimmed rather than absent, so the row keeps its
/// shape and a person can see what a selection would buy them.
pub const INK_DEAD: u32 = 0x5a5f64;
/// The chord beside one of those.
pub const INK_CHORD_DEAD: u32 = 0x4c5155;

/// The classes, as default methods over anything wgpui can style.
///
/// Blanket-implemented, so this reaches `Div`, `Svg` and — because `hover` hands one
/// out — the `StyleRefinement` inside a hover closure.
pub trait StyleExt: Styled {
    /// A panel column: its metrics, its ground and the ink anything in it inherits.
    /// The border is the caller's, since which side it is on is which column this is.
    fn panel_column(self, width: f32) -> Self {
        self.flex()
            .flex_col()
            .w(px(width))
            .h_full()
            .p_3()
            .gap_2()
            .bg(rgb(PANEL))
            .text_color(rgb(INK_LIT))
    }

    /// A section's title.
    fn heading(self) -> Self {
        self.text_sm().text_color(rgb(INK_LABEL))
    }

    /// Small text that names something rather than saying it.
    fn caption(self) -> Self {
        self.text_xs().text_color(rgb(INK_LABEL))
    }

    /// A small pressable thing. Padding is the caller's — these run from a full-width
    /// menu row to a quarter-width segment, and only the shape is shared.
    fn chip(self) -> Self {
        self.relative().rounded_sm().text_xs().cursor_pointer()
    }

    /// A control nobody has chosen.
    fn resting(self) -> Self {
        self.bg(rgb(CONTROL)).text_color(rgb(INK))
    }

    /// Chosen, or [`resting`](Self::resting). The fork rather than the two ends of
    /// it: a chip that is lit on one axis and resting on another is the bug this
    /// stops.
    fn lit(self, on: bool) -> Self {
        if on {
            self.bg(rgb(LIT)).text_color(rgb(INK_LIT))
        } else {
            self.resting()
        }
    }

    /// The same for a row in a list, which has no ground of its own until it is
    /// chosen — a column of grounds would be a column of chips.
    fn lit_row(self, on: bool) -> Self {
        if on {
            self.bg(rgb(LIT)).text_color(rgb(INK_LIT))
        } else {
            self.text_color(rgb(INK))
        }
    }
}

impl<T: Styled> StyleExt for T {}

/// Hang a tooltip on a control: the word for a mark, the chord for a word.
///
/// The widget layer's `Tooltip` (§11.1) on wgpui's own hover, so the chips this
/// reaches carry an id where they did not before — a hover has to belong to
/// something. This is where a chord went once the labels stopped carrying one:
/// the same information in less space, which is what §11.2 said native gets.
pub fn tip<E: StatefulInteractiveElement>(el: E, text: impl Into<SharedString>) -> E {
    let text: SharedString = text.into();
    el.tooltip(move |window, cx| Tooltip::new(text.clone()).build(window, cx))
}
