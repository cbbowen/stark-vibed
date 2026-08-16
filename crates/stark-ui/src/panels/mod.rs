//! The floating tool panels (§11). One module per panel; the chrome that
//! frames them and the order they stack in belong to [`crate::layout`].

pub mod brush;
pub mod color;
pub mod filter;
pub mod frame;
pub mod gradient_bar;
pub mod gradients;
pub mod guides;
pub mod layer;
/// The Layers panel's arithmetic — the rows, and what a drop into them means.
/// Split from the panel it serves because it is the half that can be tested.
pub mod layer_tree;
pub mod lighting;
pub mod navigator;
pub mod pick;
/// The drag that moves a row of a list — shared by the two panels that are
/// rosters of a stack the artist arranges (the layer tree and the guide list).
pub mod reorder;
pub mod select;
pub mod timeline;
pub mod transform;

pub use brush::BrushPanel;
pub use color::ColorPanel;
pub use filter::FilterBar;
pub use frame::{FrameBar, FrameOverlay};
pub use gradient_bar::{GradientBar, GradientBarOverlay};
pub use gradients::{GradientTraceOverlay, GradientsPanel};
pub use guides::{GuideEditOverlay, GuidesPanel, PerspectiveGuideBar};
pub use layer::LayerPanel;
pub use lighting::LightingPanel;
pub use navigator::NavigatorPanel;
pub use pick::PickBar;
pub use select::{SelectPanel, SelectionBar};
pub use timeline::TimelineBar;
pub use transform::{TransformBar, TransformOverlay};
