//! The chrome a feature puts on screen (§11). The frame a panel floats in and
//! the order the stack keeps belong to [`crate::layout`].
//!
//! # Three registers, one directory
//!
//! The name says *panels* and the directory holds more than panels, which is
//! worth stating rather than leaving to be worked out from the exports. §11
//! treats these as three registers with different rules, and a module here may
//! own one, two or all three of them for the same feature:
//!
//! - a **panel** stacks in the right-hand column, wears a title bar, is dragged
//!   and folded and closed, and is remembered between visits ([`BrushPanel`],
//!   [`ColorPanel`], [`SelectPanel`], [`LayerPanel`], [`GuidesPanel`],
//!   [`LightingPanel`] — the six of [`PanelId`](crate::layout::PanelId), and
//!   the only six there will be without an edit to that enum);
//! - a **bar** mounts at the bottom with the thing it acts on and dissolves with
//!   it, so it doubles as the indicator that the thing exists ([`SelectionBar`],
//!   [`FrameBar`], [`FilterBar`], [`TimelineBar`], [`PickBar`]) — or wears the
//!   composing register (`mode-bar`) and stands the others down
//!   ([`TransformBar`], [`GradientBar`], [`TraceBar`], [`PerspectiveGuideBar`]);
//! - an **overlay** is a full-viewport catcher that takes the pointer away from
//!   painting for a mode's duration ([`TransformOverlay`],
//!   [`GuideEditOverlay`], [`GradientBarOverlay`], [`GradientTraceOverlay`]) —
//!   or, like [`FrameOverlay`], sits over the canvas and passes presses through
//!   it. `crate::modes` is what keeps at most one catcher live;
//! - a **pop-out** is a surface flown open beside the well that opened it, for a
//!   choice made by looking rather than by reading — a colour, a ramp, a canvas
//!   surface. `widgets::PopoutId` names every one and keeps at most one open, and
//!   where each is *drawn* turns on a single fact: a bar draws its own in place,
//!   while a panel's is clipped by the column it lives in and so is mounted at the
//!   app root and placed ([`StackPopouts`]).
//!
//! **The feature is the module and the register is the item**, which is why a
//! module is not renamed for whichever of the three it happens to hold most of:
//! `gradient_bar` owns a bar and the catcher it fronts because they are one
//! composition, and splitting them by register would put the two halves of one
//! gesture in two files.
//!
//! Two modules here are neither, and say so at their own declaration: they are
//! the arithmetic a panel hides, split out so that it can be tested.

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
pub mod lighting;
pub mod pick;
/// Where a panel's pop-out is drawn — the one register in this directory that
/// cannot be drawn where it belongs, and so is mounted at the app root and placed.
pub mod popout;
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
pub use gradients::{GradientTraceOverlay, TraceBar};
pub use guides::{GuideEditOverlay, GuidesPanel, PerspectiveGuideBar};
pub use layer::LayerPanel;
pub use lighting::LightingPanel;
pub use pick::PickBar;
pub use popout::StackPopouts;
pub use select::{SelectPanel, SelectionBar};
pub use timeline::TimelineBar;
pub use transform::{TransformBar, TransformOverlay};
