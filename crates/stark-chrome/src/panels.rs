//! The register vocabulary a chrome is arranged in (§11).
//!
//! One enum so far. It is here because two *records* name a panel — which the client
//! left open, and a rebinding of `Command::TogglePanel` — and a record cannot outrun
//! its vocabulary: that is the rule N1 found when `visibility` could not move, and
//! this is the half of it being paid.
//!
//! The frames the panels float in, the order they stack in and the gestures that
//! rearrange them are a frontend's, and stay there.

/// Identity of a floating tool panel. The set is fixed; a frontend's own layout
/// tracks their order and which are open (§11).
///
/// Serde, because a panel is named in two stored records — which panels this browser
/// left open, and a rebinding of `Command::TogglePanel` — and the derive spells a
/// variant exactly as `Debug` does. So the stored name, the `data-panel` attribute and
/// the drag key (a frontend's `panel_key`) are one word by construction, and a variant renamed
/// costs the stored row rather than mis-matching it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum PanelId {
    Color,
    Brush,
    Select,
    Layers,
    Guides,
    Lighting,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order. Color leads: it is what the
    /// next stroke is made of, and the one panel an artist reaches for between
    /// nearly every pair of them.
    pub const ALL: [PanelId; 6] = [
        PanelId::Color,
        PanelId::Brush,
        PanelId::Select,
        PanelId::Layers,
        PanelId::Guides,
        PanelId::Lighting,
    ];

    /// The panel's title-bar label.
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Color => "Color",
            PanelId::Brush => "Brush",
            PanelId::Select => "Select",
            PanelId::Layers => "Layers",
            PanelId::Guides => "Drawing Guides",
            PanelId::Lighting => "Lighting",
        }
    }

    ///
    /// The height a panel opens at, in px — and, by being `Some`, that it is
    /// **vertically resizable**.
    ///
    /// One method rather than a `resizable()` flag beside a `default_height()`,
    /// because a panel that can be resized is exactly a panel whose height the layout
    /// owns; two sources for that would be one to get out of step.
    ///
    /// Everything else hugs its controls, which is the right answer for a fixed set of
    /// knobs. Only a panel holding a list the user grows (Brush, via its presets) has
    /// an appetite for height.
    pub fn default_height(self) -> Option<f32> {
        match self {
            // Tall enough for the quick controls plus four or five presets — a library
            // worth scrolling rather than a slot — and no taller, because the panel
            // stack is a column and every pixel here is one the panels under it lose.
            PanelId::Brush => Some(340.0),
            _ => None,
        }
    }
}
