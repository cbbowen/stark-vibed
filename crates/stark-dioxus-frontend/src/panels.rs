//! The chrome a feature puts on screen (§11). The frame a panel floats in and
//! the order the stack keeps belong to [`crate::layout`].
//!
//! # Several registers, one directory
//!
//! The name says *panels* and the directory holds more than panels, which is
//! worth stating rather than leaving to be worked out from the exports. §11
//! treats these as registers with different rules, and a module here may own
//! several of them for the same feature:
//!
//! - a **panel** stacks in the right-hand column, wears a title bar, is dragged
//!   and folded and closed, and is remembered between visits ([`BrushPanel`],
//!   [`ColorPanel`], [`SelectPanel`], [`LayerPanel`], [`GuidesPanel`],
//!   [`LightingPanel`] — the six of [`PanelId`](stark_ui::panels::PanelId), and
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
//!   it. [`ModeCatcher`] and [`ModeBars`] mount whichever of those a mode owns;
//! - a **pop-out** is a surface flown open beside the well that opened it, for a
//!   choice made by looking rather than by reading — a colour, a ramp, a canvas
//!   surface. `widgets::PopoutId` names every one and keeps at most one open, and
//!   where each is *drawn* turns on a single fact: a bar draws its own in place,
//!   while a panel's is clipped by the column it lives in and so is mounted at the
//!   app root and placed ([`StackPopouts`]);
//! - a **dialog** is on the root's stack (`crate::dialogs`) — [`new_document`]'s.
//!
//! **The feature is the module and the register is the item**, which is why a
//! module is not renamed for whichever register it happens to hold most of:
//! `gradient_bar` owns a bar and the catcher it fronts because they are one
//! composition, and splitting them by register would put the two halves of one
//! gesture in two files.
//!
//! Two modules here are neither, and say so at their own declaration: they are
//! the arithmetic a panel hides, split out so that it can be tested.

use dioxus::prelude::*;

use crate::state::AppState;
use stark_ui::modes::{Composing, GradientUi};

pub mod brush;
pub mod color;
pub mod filter;
pub mod frame;
pub mod gradient_bar;
pub mod gradients;
pub mod guides;
pub mod layer;
pub mod lighting;
pub mod new_document;
pub mod pick;
/// Where a panel's pop-out is drawn — the one register in this directory that
/// cannot be drawn where it belongs, and so is mounted at the app root and placed.
pub mod popout;
/// The drag that moves a row of a list — shared by the two panels that are
/// rosters of a stack the artist arranges (the layer tree and the guide list) and
/// by the panel stack's own title bars.
pub mod reorder;
pub mod select;
pub mod substrates;
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

/// The live composing mode's catcher over the canvas (`crate::modes`), or the frame's
/// handles while no mode is live.
///
/// At most one catcher is live, since entering a mode leaves the last. The frame's handles
/// stand down under a mode because they sit above the catchers' rung (`.frame-overlay` at
/// 10, catchers at 9), where a grip over a transform box or an axis drag would take presses
/// meant for it.
///
/// The `match` sits inside `rsx!` because a key is honoured only among siblings: at a
/// component's root it is ignored, and a guide picked up in place of another would keep
/// the last one's drag and hover.
#[component]
pub fn ModeCatcher() -> Element {
    let state = use_context::<AppState>();
    rsx! {
        {
            match crate::modes::composing(state) {
                None => rsx! { FrameOverlay {} },
                Some(Composing::Transform(ui)) => rsx! { TransformOverlay { ui } },
                Some(Composing::GuideEdit(edit)) => rsx! {
                    GuideEditOverlay { key: "{edit.id:?}", edit }
                },
                Some(Composing::GradientTrace) => rsx! { GradientTraceOverlay {} },
                Some(Composing::GradientFill(ui)) => rsx! { GradientBarOverlay { ui } },
            }
        }
    }
}

/// The live composing mode's bar, and the gradient bar a trace parked.
///
/// Deepest first — the trace, the transform, the gradient, the guide — so a trace's bar
/// lands above the gradient bar it parked. Each bar holds its own slot, so the gradient bar
/// keeps its element (and its recess transition) when a trace parks it and when the trace
/// hands it back.
#[component]
pub fn ModeBars() -> Element {
    let state = use_context::<AppState>();
    let mode = crate::modes::composing(state);
    let gradient = gradient_slot(mode.as_ref(), state.gradient_resume.read().clone());
    let trace = matches!(mode, Some(Composing::GradientTrace));
    let transform = mode.clone().and_then(Composing::transform);
    let guide = mode.and_then(Composing::guide_edit);
    rsx! {
        if trace {
            TraceBar {}
        }
        if let Some(ui) = transform {
            TransformBar { ui }
        }
        if let Some((ui, parked)) = gradient {
            GradientBar { ui, parked }
        }
        if let Some(edit) = guide {
            PerspectiveGuideBar { edit }
        }
    }
}

/// What the gradient bar composes, and whether it is `parked`: the live fill's own, or
/// else the gesture a trace set aside (`gradient_bar::suspend`), drawn recessed under
/// whatever mode is live.
fn gradient_slot(
    mode: Option<&Composing>,
    parked: Option<GradientUi>,
) -> Option<(GradientUi, bool)> {
    match mode {
        Some(Composing::GradientFill(ui)) => Some((ui.clone(), false)),
        _ => parked.map(|ui| (ui, true)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId, GuideId, LayerId};
    use stark_ui::modes::{GradientAxisKind, GradientTarget, GuideEdit};

    fn fill(kind: GradientAxisKind) -> GradientUi {
        GradientUi {
            target: GradientTarget::Fill {
                layer: LayerId::ROOT,
            },
            kind,
            drag: None,
        }
    }

    #[test]
    fn a_parked_gradient_bar_stands_recessed_under_any_mode_but_a_live_fill() {
        let parked = fill(GradientAxisKind::Linear);
        for mode in [None, Some(Composing::GradientTrace)] {
            assert_eq!(
                gradient_slot(mode.as_ref(), Some(parked.clone())),
                Some((parked.clone(), true)),
                "under {mode:?}"
            );
        }
        let live = fill(GradientAxisKind::Radial);
        assert_eq!(
            gradient_slot(Some(&Composing::GradientFill(live.clone())), Some(parked)),
            Some((live, false))
        );
        assert_eq!(gradient_slot(Some(&Composing::GradientTrace), None), None);
    }

    /// Picking up another guide remounts the catcher, so its drag and hover start clean; the
    /// same guide re-rendered keeps it. With nothing to draw the overlay renders a
    /// placeholder, so a remount is the only thing that writes to the DOM.
    #[test]
    fn a_guide_picked_up_in_place_of_another_gets_a_fresh_catcher() {
        use dioxus::dioxus_core::{ScopeId, VirtualDom};

        fn app() -> Element {
            let state = AppState::new();
            use_context_provider(|| state);
            rsx! { ModeCatcher {} }
        }
        let guide = |lamport| GuideEdit {
            id: GuideId(ActionId {
                lamport,
                actor: ActorId(1),
            }),
            locked: [false; 3],
        };
        let mut dom = VirtualDom::new(app);
        dom.rebuild_in_place();
        let mut enter = |edit: GuideEdit| {
            dom.in_scope(ScopeId::APP, || {
                let mut mode = consume_context::<AppState>().mode;
                mode.set(Some(Composing::GuideEdit(edit)));
            });
            dom.process_events();
            dom.render_immediate_to_vec().edits.len()
        };

        assert_ne!(
            enter(guide(1)),
            0,
            "the catcher mounts over the frame's handles"
        );
        let relocked = GuideEdit {
            locked: [true, false, false],
            ..guide(1)
        };
        assert_eq!(enter(relocked), 0, "the same guide keeps its catcher");
        assert_ne!(enter(guide(2)), 0, "another guide gets a fresh one");
    }
}
