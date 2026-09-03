//! The brush in hand: which tool, at what size, flow and colour (§6.2, §18.1.8).
//!
//! Both halves, as `stark-chrome` splits them — the **durable**
//! [`BrushConfig`] (what the tool *is*) beside the **transient** [`Transient`] (the
//! size, flow and colour the hand is working it at). The split is not this
//! frontend's invention and not the web one's either: it is what a preset stores and
//! what a quick slot substitutes, so both frontends carry the pair and neither
//! carries a third shape.
//!
//! What is here that is not shared is the *library this frontend can show*. The
//! shipped table names two bundled stamps by content id (`presets::shipped`), and an
//! id is the hash of bytes nothing has fetched yet — this frontend has no asset
//! import (that is N7) — so the two stamp presets stand on the round tip until then.
//! They are still in the list, because dropping them would mean a different set of
//! tools on the two frontends for a reason that is temporary.

use stark_chrome::brush_config::{BrushConfig, Transient};
use stark_chrome::presets::{BuiltinShapes, PresetEntry, shipped};
use stark_engine::command::ViewCommand;

/// The brush the app opens on and the library it can be swapped for.
pub struct Brush {
    /// What the tool is.
    pub config: BrushConfig,
    /// The size, flow and colour it is being worked at.
    pub tune: Transient,
    /// The shipped presets, in the order the table declares. Not a user library:
    /// saving one is a record and a dialog, and neither exists here yet.
    pub library: Vec<PresetEntry>,
    /// Which preset the tool came from, so a row can show as the one in hand.
    /// `None` once a knob has been moved off it.
    pub from: Option<String>,
}

impl Default for Brush {
    /// Opens on the library's first entry, which is the everyday brush — the same
    /// rule the web frontend's `apply_first` follows, and the reason the table leads
    /// with Hard Round.
    fn default() -> Self {
        // `BuiltinShapes::default()` is the round tip twice over: see the module note.
        let library = shipped(BuiltinShapes::default());
        let first = library.first().cloned();
        Self {
            config: first.as_ref().map(|e| e.brush).unwrap_or_default(),
            tune: first.as_ref().map(|e| e.transient).unwrap_or_default(),
            from: first.map(|e| e.name),
            library,
        }
    }
}

impl Brush {
    /// The command that puts this brush in the engine's hand.
    ///
    /// One command rather than two, because the engine takes the pair together
    /// (`ViewCommand::SetBrush`): the hand's colour is not always the brush's — an
    /// erasing brush carries no pigment — so sending them apart would let the two
    /// arrive out of step.
    pub fn set(&self) -> ViewCommand {
        ViewCommand::SetBrush {
            brush: self.config.params(self.tune),
            color: self.tune.color,
        }
    }

    /// Wear the preset called `name`, keeping the colour in hand.
    ///
    /// The colour stays because it is the Colour panel's rather than the tool's
    /// (§18.1.8) — picking up a different brush does not repaint your palette. Every
    /// other field, the effect's own opacity included, comes from the preset.
    pub fn wear(&mut self, name: &str) {
        let Some(entry) = self.library.iter().find(|e| e.name == name) else {
            return;
        };
        let color = self.tune.color;
        self.config = entry.brush;
        self.tune = Transient {
            color,
            ..entry.transient
        };
        self.from = Some(entry.name.clone());
    }

    /// Note that a knob has been moved: the tool is no longer *the* preset, though
    /// it still descends from it.
    ///
    /// The size and the flow are exempt, which is the whole of the durable/transient
    /// split showing through: working a brush at another size is the same tool, and
    /// the rack is built on that (§18.1.8). Only an edit to the durable half takes
    /// the name off.
    pub fn tuned_off_preset(&mut self) {
        self.from = None;
    }
}
