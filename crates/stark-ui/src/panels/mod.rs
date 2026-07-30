//! The floating tool panels (DESIGN.md §11). One module per panel; the chrome that
//! frames them and the order they stack in belong to [`crate::layout`].

pub mod brush;
pub mod color;
pub mod frame;
pub mod layer;
pub mod lighting;
pub mod navigator;
pub mod pick;
pub mod select;
pub mod timeline;
pub mod transform;

pub use brush::BrushPanel;
pub use color::ColorPanel;
pub use frame::{FrameBar, FrameOverlay};
pub use layer::LayerPanel;
pub use lighting::LightingPanel;
pub use navigator::NavigatorPanel;
pub use pick::PickBar;
pub use select::{SelectPanel, SelectionBar};
pub use timeline::TimelineBar;
pub use transform::{TransformBar, TransformOverlay};
