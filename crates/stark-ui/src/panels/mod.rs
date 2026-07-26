//! The floating tool panels (DESIGN.md §11). One module per panel; the chrome that
//! frames them and the order they stack in belong to [`crate::layout`].

pub mod brush;
pub mod color;
pub mod layer;
pub mod lighting;
pub mod select;

pub use brush::BrushPanel;
pub use color::ColorPanel;
pub use layer::LayerPanel;
pub use lighting::LightingPanel;
pub use select::SelectPanel;
