//! What a shape gesture is about to do (§6.8, §18.0.4): which tool draws it, and what
//! the region it encloses lands on.
//!
//! Two rules, and both are the frontend's rather than the engine's — the session is
//! sent [`SetTool`](stark_engine::command::ViewCommand::SetTool) and
//! [`SetShapeAction`](stark_engine::command::ViewCommand::SetShapeAction) and nothing
//! else, so what a *press* on an already-lit chip means, and what a held modifier
//! means, are decided up here. Decided **once**: a chip, a chord and a palette row
//! reach the same act, and two frontends reach the same rule.

use stark_engine::command::Tool;
use stark_model::document::{SelectionMode, ShapeAction};

use crate::keys::Mods;

/// The three shape tools, in the order a row of them is drawn.
///
/// The brush is not among them and is not an omission: arming is momentary — a
/// selecting gesture hands the canvas back when it ends (§6.8) — so *no* tool lit is
/// the brush, and a fourth chip for it would be a chip that can never be off.
pub const SHAPE_TOOLS: [Tool; 3] = [Tool::SelectRect, Tool::SelectEllipse, Tool::SelectLasso];

/// The five answers to "what does this shape do?", in row order.
///
/// Five answers to one question rather than four combine modes plus an odd one out.
/// The shape tools never produced selections; they produce **coverage**, and the four
/// modes are the four ways coverage can land on the mask. [`ShapeAction::Fill`] lands
/// it on the paint instead, with the same shapes, the same rasterizer and the same
/// feather (§18.0.4).
pub const SHAPE_ACTIONS: [ShapeAction; 5] = [
    ShapeAction::Select(SelectionMode::Replace),
    ShapeAction::Select(SelectionMode::Union),
    ShapeAction::Select(SelectionMode::Subtract),
    ShapeAction::Select(SelectionMode::Intersect),
    ShapeAction::Fill,
];

/// What each of [`SHAPE_ACTIONS`] is called — terse, because the row is five chips
/// wide in the narrowest column either frontend has.
///
/// Here rather than in `commands` because these are not commands: an action is a
/// *setting* the next gesture reads, and the registry names acts. Both frontends draw
/// the row, so the words are shared for the reason every other pair of words in this
/// crate is — and the mark beside each stays each frontend's, since one of them is
/// inline SVG and the other is not.
pub fn action_word(action: ShapeAction) -> &'static str {
    match action {
        ShapeAction::Select(SelectionMode::Replace) => "New",
        ShapeAction::Select(SelectionMode::Union) => "Add",
        ShapeAction::Select(SelectionMode::Subtract) => "Sub",
        ShapeAction::Select(SelectionMode::Intersect) => "Isect",
        ShapeAction::Fill => "Fill",
    }
}

/// The tool arming `asked` should leave in hand, given what is in hand now.
///
/// **Pressing the lit chip disarms it**, so the way out of an armed tool is the same
/// control that armed it — which matters because arming is otherwise only undone by
/// making the gesture, and a person who armed one by accident should not have to draw
/// something to get the brush back.
pub fn arm(current: Tool, asked: Tool) -> Tool {
    if current == asked { Tool::Brush } else { asked }
}

/// The selection mode a gesture's held modifiers ask for, or `None` to keep whatever
/// the action row is set to. The conventional marquee modifiers.
///
/// Consulted only when the row's action is a *selecting* one
/// ([`ShapeAction::is_select`]): under Fill there is nothing to combine, so shift and
/// alt mean nothing rather than quietly turning a fill back into a selection — which
/// would be the worst kind of surprise, since the paint would not land and the mask
/// would move instead.
pub fn modifier_mode(m: Mods) -> Option<SelectionMode> {
    match (m.shift, m.alt) {
        (true, true) => Some(SelectionMode::Intersect),
        (true, false) => Some(SelectionMode::Union),
        (false, true) => Some(SelectionMode::Subtract),
        (false, false) => None,
    }
}

/// What a shape gesture should be opened with, given the row's action and the
/// modifiers actually held.
///
/// `Some(action)` is an **override** the frontend must put back when the gesture ends
/// — the engine holds one shape action and a modifier borrows it for one drag. `None`
/// means the row's setting already says what to do and nothing has to be restored,
/// which is the common case and the one worth being cheap.
pub fn override_for(action: ShapeAction, mods: Mods) -> Option<ShapeAction> {
    if !action.is_select() {
        return None;
    }
    modifier_mode(mods)
        .map(ShapeAction::Select)
        .filter(|next| *next != action)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arming the tool already in hand puts it down. The row's escape hatch.
    #[test]
    fn pressing_the_lit_tool_hands_the_canvas_back() {
        assert_eq!(arm(Tool::SelectRect, Tool::SelectRect), Tool::Brush);
        assert_eq!(arm(Tool::SelectRect, Tool::SelectLasso), Tool::SelectLasso);
        assert_eq!(arm(Tool::Brush, Tool::SelectEllipse), Tool::SelectEllipse);
    }

    /// The four conventional marquee modifiers, and the bare press that means
    /// "whatever the row says".
    #[test]
    fn the_modifiers_are_the_conventional_four() {
        let m = |shift, alt| Mods {
            ctrl: false,
            shift,
            alt,
        };
        assert_eq!(modifier_mode(m(false, false)), None);
        assert_eq!(modifier_mode(m(true, false)), Some(SelectionMode::Union));
        assert_eq!(modifier_mode(m(false, true)), Some(SelectionMode::Subtract));
        assert_eq!(modifier_mode(m(true, true)), Some(SelectionMode::Intersect));
    }

    /// Under Fill the modifiers say nothing: there is no combining to do, and letting
    /// shift turn a fill into a union-select would move the mask where paint was
    /// asked for.
    #[test]
    fn a_fill_ignores_the_marquee_modifiers() {
        let held = Mods {
            ctrl: false,
            shift: true,
            alt: false,
        };
        assert_eq!(override_for(ShapeAction::Fill, held), None);
    }

    /// A modifier naming the mode the row is already on is not an override — there
    /// would be nothing to put back, and sending the setter twice per gesture for a
    /// change of nothing is work the common case should not do.
    #[test]
    fn a_modifier_agreeing_with_the_row_overrides_nothing() {
        let held = Mods {
            ctrl: false,
            shift: true,
            alt: false,
        };
        let union = ShapeAction::Select(SelectionMode::Union);
        assert_eq!(override_for(union, held), None);
        assert_eq!(
            override_for(ShapeAction::Select(SelectionMode::Replace), held),
            Some(union)
        );
    }
}
