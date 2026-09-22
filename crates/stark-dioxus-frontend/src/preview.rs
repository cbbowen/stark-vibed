//! Showing an edit on the canvas before laying it down — the bargain every
//! continuous control in the app makes, stated once.
//!
//! A slider, a color picker, a frame handle and a transform widget all want the
//! same thing: **the value is judged by looking**, so every value the pointer
//! crosses has to reach the canvas — and only the answer belongs in the log. So
//! each of them previews per sample through [`ViewCommand::Preview`], which logs
//! nothing, and commits once through [`DocCommand::Set`] when the gesture settles.
//! One undo step per adjustment, and one replicated action, rather than one per
//! pointer move (§6.4, §14.6, §15.7, §16.6, §21.6, §22.4).
//!
//! # The pair is one value
//!
//! Both halves carry the same [`Edit`], so a control that previews one thing and
//! commits another cannot be written — the failure is silent where it happens,
//! since both halves work on their own. What a row here adds is the control's own
//! shape: which `Edit` the value it holds becomes, named once in the table below.
//!
//! What the type does not rule out is *forgetting to settle* a canvas drag, which
//! has to wire [`Preview::settle`] to every event that can end it — hence the method
//! is idempotent. A slider cannot forget: [`PreviewSlider`](crate::widgets::PreviewSlider)
//! owns its three.

use dioxus::prelude::*;
use stark_model::{Srgb, SubstrateScale};

use crate::state::{AppState, dispatch};
use stark_engine::command::{DocCommand, Edit, ViewCommand};
use stark_model::document::LayerId;
use stark_model::document::{
    BlendMode, FillOp, Filter, GuideId, Parcel, PerspectiveGuide, TransformMap,
};
use stark_model::geom::Vec2;

/// One control's edit: what the value it holds becomes, shown on the canvas without
/// logging it or laid down.
///
/// `Copy`, and a function pointer wide, so the constants below are `const` and
/// handing one to an event closure costs nothing — for every `T`, which is why the
/// impls are written out: a derive would ask `T` to be `Copy` too.
pub struct Preview<T: 'static> {
    edit: fn(T) -> Edit,
}

impl<T: 'static> Clone for Preview<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: 'static> Copy for Preview<T> {}

/// The same row, by the address of its edit — what a
/// [`PreviewSlider`](crate::widgets::PreviewSlider) compares its props by.
impl<T: 'static> PartialEq for Preview<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.edit, other.edit)
    }
}

impl<T: 'static> Preview<T> {
    const fn new(edit: fn(T) -> Edit) -> Self {
        Self { edit }
    }

    /// Show `value` on the canvas, logging nothing.
    pub fn show(self, state: AppState, value: T) {
        dispatch(state, ViewCommand::Preview(Some((self.edit)(value))));
    }

    /// Drop whatever is being shown, laying nothing.
    ///
    /// What an abandoned composition needs (`crate::modes`), and what a control
    /// whose gesture ended with nothing to lay needs. Deliberately not `commit`
    /// with a neutral value: dropping the preview restores the document, while
    /// committing a neutral would be an edit that says the artist chose it.
    pub fn clear(self, state: AppState) {
        dispatch(state, ViewCommand::Preview(None));
    }

    /// Lay `value` down — one logged action, one undo step.
    ///
    /// The commit supersedes the preview engine-side, so nothing has to drop it
    /// first and there is no intermediate frame showing the document without it.
    pub fn commit(self, state: AppState, value: T) {
        dispatch(state, DocCommand::Set((self.edit)(value)));
    }
}

impl<T: Clone + 'static> Preview<T> {
    /// One sample of a drag: show `value`, and remember it as what a
    /// [`settle`](Self::settle) would lay.
    ///
    /// The two halves are one call because they are one act, and splitting them
    /// is how a control ends up showing one value and committing another — the
    /// pending value **is** what the canvas is showing, by construction.
    ///
    /// The value is held rather than read back off the projection at commit time,
    /// which matters and is easy to get wrong: the projection is reporting the
    /// *preview* mid-drag (§14.6), so a commit that re-read it would be reading
    /// its own echo.
    pub fn during(self, state: AppState, mut pending: Signal<Option<T>>, value: T) {
        pending.set(Some(value.clone()));
        self.show(state, value);
    }

    /// End a drag: lay down what the previews have been showing, once, and
    /// disarm.
    ///
    /// **A no-op when nothing is pending**, which is what lets a control wire this
    /// to all three of the events that can end a drag — `change`, `pointerup`,
    /// `pointercancel` — without spending an undo step per event. `change` alone
    /// is not enough, because it is not sent when a drag ends on the value it
    /// started on, which would strand a preview with no commit to supersede it.
    /// A commit to the value the document already holds is refused engine-side, so
    /// a drag that travelled out and came back logs nothing.
    pub fn settle(self, state: AppState, mut pending: Signal<Option<T>>) {
        let settled = pending.write().take();
        if let Some(value) = settled {
            self.commit(state, value);
        }
    }
}

// --- the table -------------------------------------------------------------
//
// Every previewing control in the app, as the edit its value becomes. Adding one is
// a row here and nothing else.

/// The selection's overall strength — the selection bar's Opacity slider (§6.8).
///
/// The one row here whose preview shows *nothing on the canvas*, and it is still a
/// row rather than a bare commit: a mask's strength changes no pixel until something
/// paints through it, but it is document state all the same, so a drag that logged
/// every value it crossed would spend an undo step per pointer move on an adjustment
/// the hand made once. What the preview buys is the bar — the engine reports the
/// previewed number back through `observe`, so the track follows the pointer instead
/// of snapping back to the committed value under it.
pub const SELECTION_OPACITY: Preview<f32> = Preview::new(Edit::SelectionOpacity);

/// A layer's opacity — the Layers panel's slider (§14.6).
pub const LAYER_OPACITY: Preview<(LayerId, f32)> =
    Preview::new(|(layer, opacity)| Edit::LayerOpacity(layer, opacity));

/// A layer's blend mode — the Bend slider, which drags a parameter *of* the mode
/// and so carries the whole mode (§18.0.4).
pub const LAYER_BLEND: Preview<(LayerId, BlendMode)> =
    Preview::new(|(layer, mode)| Edit::LayerBlend(layer, mode));

/// A filter layer's settings — every knob on the filter bar (§21.6).
pub const FILTER: Preview<(LayerId, Filter)> =
    Preview::new(|(layer, filter)| Edit::Filter(layer, filter));

/// A frame's rect — the on-canvas handle drag (§15.7).
pub const MATTE_RECT: Preview<(LayerId, Vec2, Vec2)> =
    Preview::new(|(layer, min, max)| Edit::MatteRect(layer, min, max));

/// A matte's paint — the frame bar's color pop-out, and the gradient bar's axis
/// when it is aimed at a matte (§15.4, §22.4).
pub const MATTE_PAINT: Preview<(LayerId, Parcel)> =
    Preview::new(|(layer, paint)| Edit::MattePaint(layer, paint));

/// The canvas substrate color — the Lighting panel's pop-out (§15.5).
pub const BACKGROUND: Preview<Srgb> = Preview::new(Edit::SubstrateColor);

/// How large the canvas substrate is laid — the Lighting panel's scale slider (§6.4).
///
/// The row where the bargain buys the most. Every other row here spends an undo step
/// per pointer sample if it is got wrong; this one would also **bake a substrate** per
/// sample, since a `Surface` is built from the substrate and its scale together
/// (`stark-engine`'s `gpu::substrate::Substrate`). The preview costs one number in the
/// media pass's uniform, because the height field a light reads is the same field
/// however large it is laid — only the deposit's rise channels move with the scale,
/// and nothing deposits while a slider is under the hand.
pub const SUBSTRATE_SCALE: Preview<SubstrateScale> = Preview::new(Edit::SubstrateScale);

/// A gradient fill of the selection — the gradient bar's composing axis (§22.4).
pub const FILL: Preview<(LayerId, FillOp)> = Preview::new(|(layer, op)| Edit::Fill { layer, op });

/// A drawing guide's camera — every control on the Perspective Guide bar, and
/// every drag on the canvas while one is being shaped (§20.5).
///
/// The whole camera travels, for the reason [`FILTER`]'s whole filter does: an
/// orbit writes the rotation, the ring drag writes the focal length and the cell
/// slider writes the lattice, and one row covers all of them without an edit
/// per knob.
///
/// The newest row, and the one the table was waiting for: guides became document
/// state after this module was written, which turned every control on that bar
/// from a live view-state write into exactly the bargain described above.
pub const GUIDE: Preview<(GuideId, PerspectiveGuide)> =
    Preview::new(|(id, guide)| Edit::Guide(id, guide));

/// The transform gesture — the widget's whole composition, committed on "Done"
/// (§16.6, §16.8, §16.9).
pub const TRANSFORM: Preview<(LayerId, TransformMap)> =
    Preview::new(|(layer, map)| Edit::Transform { layer, map });

/// A layer's place on the canvas — the pick-and-translate drag (§16.11, §14.12).
///
/// The row that replaced the carry's use of [`TRANSFORM`]: a translation is a
/// property write, so this preview moves no tile where that one resampled the
/// layer per changed sample — the whole of what the frame field buys.
pub const TRANSLATE: Preview<(LayerId, stark_model::geom::IVec2)> =
    Preview::new(|(layer, to)| Edit::Translate { layer, to });
