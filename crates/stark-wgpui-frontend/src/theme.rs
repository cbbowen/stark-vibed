//! The widget layer's theme, set from the chrome's palette (§11.1).
//!
//! `wgpui-component` dresses its controls from one global [`Theme`], and ships a
//! light and a dark one of its own. Neither is this chrome: [`crate::style`] is —
//! a dark palette named by role, which every hand-built control here already
//! wears. So the library's dark theme is the starting point, and its semantic
//! slots are then written from the same constants, role for role: a popover is
//! [`style::MENU`] because a drop-down is, a slider's fill is [`style::FILL`]
//! because the troughs are. One palette, two consumers, and no second copy of a
//! colour to drift.
//!
//! What is *not* mapped keeps the library's dark default — the status colours
//! (danger, warning, success, info), the chart series, the code-editor slots —
//! because the chrome has no word for them yet, and inventing one here would be a
//! second palette by another route.

use wgpui::{App, Hsla, Window, px, rgb, rgba};
use wgpui_component::{Theme, ThemeMode, ThemeTokens};

use crate::style;

/// A packed `0xRRGGBB` from the palette, as the theme stores it.
fn ink(packed: u32) -> Hsla {
    rgb(packed).into()
}

/// The same with an alpha in the low byte, for the few translucent slots.
fn wash(packed: u32, alpha: u8) -> Hsla {
    rgba((packed << 8) | u32::from(alpha)).into()
}

/// Register the widget layer and set its theme. Once per app, before the first
/// window's root view is built — the components read the global as they render,
/// and a `Root` built first would take the library's light default for a frame.
pub fn install(window: &mut Window, cx: &mut App) {
    wgpui_component::init(cx);
    Theme::change(ThemeMode::Dark, Some(window), cx);

    let theme = Theme::global_mut(cx);
    theme.radius = px(4.);
    theme.radius_lg = px(6.);
    theme.shadow = true;

    let c = &mut theme.colors;
    // grounds
    c.background = ink(style::PANEL);
    c.foreground = ink(style::INK_LIT);
    c.border = ink(style::EDGE);
    c.popover = ink(style::MENU);
    c.popover_foreground = ink(style::INK_ROW);
    c.muted = ink(style::CONTROL);
    c.muted_foreground = ink(style::INK_LABEL);
    c.title_bar = ink(style::BAR);
    c.title_bar_border = ink(style::EDGE);
    c.status_bar = ink(style::BAR);
    c.status_bar_border = ink(style::EDGE);
    c.sidebar = ink(style::PANEL);
    c.sidebar_border = ink(style::EDGE);
    c.sidebar_foreground = ink(style::INK);
    c.sidebar_accent = ink(style::HOVER);
    c.sidebar_accent_foreground = ink(style::INK_LIT);
    c.sidebar_primary = ink(style::LIT);
    c.sidebar_primary_foreground = ink(style::INK_LIT);
    c.overlay = wash(style::WELL, 0xa0);

    // buttons: resting, hovered, pressed — the library reads these slots for a
    // button and the generic ones below for everything else, and its dark default
    // puts a *white* primary button on the dark ground, which is nothing here.
    c.button = ink(style::CONTROL);
    c.button_foreground = ink(style::INK);
    c.button_hover = ink(style::HOVER);
    c.button_active = ink(style::LIT);
    c.button_primary = ink(style::LIT);
    c.button_primary_foreground = ink(style::INK_LIT);
    c.button_primary_hover = ink(style::ACCENT);
    c.button_primary_active = ink(style::LIT);
    c.button_secondary = ink(style::CONTROL);
    c.button_secondary_foreground = ink(style::INK);
    c.button_secondary_hover = ink(style::HOVER);
    c.button_secondary_active = ink(style::LIT);

    // controls: resting, hovered, chosen
    c.secondary = ink(style::CONTROL);
    c.secondary_foreground = ink(style::INK);
    c.secondary_hover = ink(style::HOVER);
    c.secondary_active = ink(style::LIT);
    c.primary = ink(style::LIT);
    c.primary_foreground = ink(style::INK_LIT);
    c.primary_hover = ink(style::ACCENT);
    c.primary_active = ink(style::LIT);
    c.accent = ink(style::HOVER);
    c.accent_foreground = ink(style::INK_LIT);
    c.ring = ink(style::ACCENT);
    c.input = ink(style::RULE);
    c.caret = ink(style::INK_LIT);
    c.selection = wash(style::ACCENT, 0x60);
    c.link = ink(style::ACCENT);
    c.link_hover = ink(style::INK_LIT);
    c.link_active = ink(style::ACCENT);

    // lists and tabs
    c.list = ink(style::PANEL);
    c.list_even = ink(style::PANEL);
    c.list_head = ink(style::BAR);
    c.list_hover = ink(style::HOVER);
    c.list_active = ink(style::LIT);
    c.list_active_border = ink(style::LIT);
    c.tab_bar = ink(style::BAR);
    c.tab_bar_segmented = ink(style::CONTROL);
    c.tab = ink(style::BAR);
    c.tab_foreground = ink(style::INK);
    c.tab_active = ink(style::LIT);
    c.tab_active_foreground = ink(style::INK_LIT);
    c.table = ink(style::PANEL);
    c.table_even = ink(style::PANEL);
    c.table_head = ink(style::BAR);
    c.table_head_foreground = ink(style::INK_LABEL);
    c.table_hover = ink(style::HOVER);
    c.table_active = ink(style::LIT);
    c.table_active_border = ink(style::LIT);
    c.table_row_border = ink(style::RULE);

    // dials and switches
    c.slider_bar = ink(style::FILL);
    c.slider_thumb = ink(style::ACCENT);
    c.progress_bar = ink(style::FILL);
    c.switch = ink(style::CONTROL);
    c.switch_thumb = ink(style::INK_LIT);
    c.scrollbar = ink(style::PANEL);
    c.scrollbar_thumb = ink(style::CONTROL);
    c.scrollbar_thumb_hover = ink(style::HOVER);
    c.skeleton = ink(style::CONTROL);
    c.drag_border = ink(style::ACCENT);
    c.drop_target = wash(style::ACCENT, 0x30);
    c.accordion = ink(style::PANEL);
    c.group_box = ink(style::PANEL);
    c.group_box_foreground = ink(style::INK_LIT);

    // The controls read the *tokens*, which are derived from the slots above when
    // a theme is applied — so they are derived again here, and the base layer's
    // copy of the theme rebuilt from the result.
    theme.tokens = ThemeTokens::from(&theme.colors);
    Theme::sync_base(cx);
}
