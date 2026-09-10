//! What a shape gesture is about to do (§6.8, §18.0.4): which tool draws it, and what
//! the region it encloses lands on.
//!
//! Two rules, and both are the frontend's rather than the engine's — the session is
//! sent [`SetTool`](stark_engine::command::ViewCommand::SetTool) and
//! [`SetShapeAction`](stark_engine::command::ViewCommand::SetShapeAction) and nothing
//! else, so what a *press* on an already-lit chip means, and what a held modifier
//! means, are decided up here. Decided **once**: a chip, a chord and a palette row
//! reach the same act, and two frontends reach the same rule.

use stark_engine::ObservableState;
use stark_engine::command::Tool;
use stark_model::document::{SelectionMode, ShapeAction};

use crate::icons::Icon;
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

/// The widest edge the feather dial offers, canvas px.
///
/// A ceiling this control owns rather than one the model states — the rasterizer will
/// strike a softer edge than any panel offers. It was a bare `max: 64.0` in a web
/// `Slider` and a named constant natively whose doc called it "the same ceiling the web
/// panel's slider carries" (§11.2).
pub const MAX_FEATHER: f32 = 64.0;

/// The three dials a selection panel can show, each mounted only while it means
/// something.
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray, strum::EnumCount)]
pub enum Dial {
    /// The edge the next shape gesture's rasterizer strikes, canvas px. Chosen
    /// *before* the gesture, so it is shown for exactly as long as one is pending.
    Feather,
    /// How strongly a **fill** gesture's paint lands — mounted under the Fill action,
    /// because that is the only action it is about.
    FillOpacity,
    /// How strongly the whole mask gates. Not a gesture's setting but the selection's,
    /// set after the fact, so it appears with the selection rather than with the tool.
    MaskOpacity,
}

impl Dial {
    /// Which seat this dial has, so a view holding one state per dial can index rather
    /// than search. Exhaustive, so a fourth does not compile until it says where it sits.
    pub fn index(self) -> usize {
        match self {
            Dial::Feather => 0,
            Dial::FillOpacity => 1,
            Dial::MaskOpacity => 2,
        }
    }

    /// The mark the track wears.
    ///
    /// The fill's dial takes the bucket its own action chip wears, and the adjacency is
    /// the point: it is mounted only while Fill is the armed action, so it appears
    /// directly under the lit chip whose strength it sets.
    pub fn glyph(self) -> Icon {
        match self {
            Dial::Feather => crate::icons::FEATHER,
            Dial::FillOpacity => crate::icons::PAINT_BUCKET,
            Dial::MaskOpacity => crate::icons::OPACITY,
        }
    }

    /// The one word a caption gives it.
    pub fn label(self) -> &'static str {
        match self {
            Dial::Feather => "Feather",
            Dial::FillOpacity => "Fill opacity",
            Dial::MaskOpacity => "Selection strength",
        }
    }

    /// What the hover says the mark means.
    pub fn tip(self) -> &'static str {
        match self {
            Dial::Feather => {
                "Feather \u{2014} how far the next shape's edge is softened, in canvas px"
            }
            Dial::FillOpacity => "Fill opacity \u{2014} how strongly a fill gesture's paint lands",
            Dial::MaskOpacity => {
                "Selection strength \u{2014} how hard the mask gates what every tool does"
            }
        }
    }

    /// The ends of its track. Feather is a canvas-px length; the other two are
    /// strengths.
    pub const fn range(self) -> (f32, f32) {
        match self {
            Dial::Feather => (0.0, MAX_FEATHER),
            Dial::FillOpacity | Dial::MaskOpacity => (0.0, 1.0),
        }
    }

    /// The step a track moves in: whole px for a length, a hundredth for a strength.
    pub const fn step(self) -> f32 {
        match self {
            Dial::Feather => 1.0,
            Dial::FillOpacity | Dial::MaskOpacity => 0.01,
        }
    }

    /// Where the dial stands, off the engine's projection.
    pub fn read(self, o: &ObservableState) -> f32 {
        match self {
            Dial::Feather => o.selection_feather,
            Dial::FillOpacity => o.shape_opacity,
            Dial::MaskOpacity => o.selection_opacity,
        }
    }

    /// The value a fraction along this dial's track means.
    pub fn value_at(self, fraction: f32) -> f32 {
        let (lo, hi) = self.range();
        lo + fraction.clamp(0.0, 1.0) * (hi - lo)
    }
}

/// Which dials this frame shows, and in which order.
///
/// Each is mounted on what it is *about*: the feather belongs to a pending shape
/// gesture, the fill's strength to the Fill action alone, and the mask's to there being
/// a mask at all. A dial mounted where it means nothing is a control that answers a
/// question nobody asked.
pub fn dials(o: Option<&ObservableState>) -> Vec<Dial> {
    let Some(o) = o else { return Vec::new() };
    let mut out = Vec::new();
    if o.tool.is_selection() {
        out.push(Dial::Feather);
        if o.shape_action == ShapeAction::Fill {
            out.push(Dial::FillOpacity);
        }
    }
    if o.has_selection {
        out.push(Dial::MaskOpacity);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every selecting tool has a chip.** Both rows are hand-written arrays, and
    /// nothing but this stops a fourth tool or a fifth mode from being reachable by
    /// no control at all in either frontend.
    ///
    /// The device, here and below, is a `match` that maps each variant to its place
    /// in a local roster: a new variant fails to compile, the arm added to answer
    /// that has to name an index, an index the roster does not have panics, and a
    /// roster that grew forces the row to grow with it. Each of those three is one of
    /// the ways the loop could otherwise be satisfied without the row being right.
    #[test]
    fn every_selecting_tool_has_a_chip() {
        const ALL: [Tool; 4] = [
            Tool::Brush,
            Tool::SelectRect,
            Tool::SelectEllipse,
            Tool::SelectLasso,
        ];
        for tool in ALL {
            let k = match tool {
                Tool::Brush => 0,
                Tool::SelectRect => 1,
                Tool::SelectEllipse => 2,
                Tool::SelectLasso => 3,
            };
            assert_eq!(ALL[k], tool);
            assert_eq!(
                SHAPE_TOOLS.contains(&tool),
                tool.is_selection(),
                "{tool:?} is in the row but does not select, or the other way round"
            );
        }
        assert_eq!(
            SHAPE_TOOLS.len(),
            ALL.iter().filter(|t| t.is_selection()).count()
        );
    }

    /// And every combine mode has one, plus the fill that is the row's fifth answer.
    #[test]
    fn every_combine_mode_has_a_chip() {
        const ALL: [SelectionMode; 4] = [
            SelectionMode::Replace,
            SelectionMode::Union,
            SelectionMode::Subtract,
            SelectionMode::Intersect,
        ];
        for mode in ALL {
            let k = match mode {
                SelectionMode::Replace => 0,
                SelectionMode::Union => 1,
                SelectionMode::Subtract => 2,
                SelectionMode::Intersect => 3,
            };
            assert_eq!(ALL[k], mode);
            assert!(
                SHAPE_ACTIONS.contains(&ShapeAction::Select(mode)),
                "{mode:?} has no chip"
            );
        }
        // Fill is the one action that is not a mode, and it is exhaustive here for
        // the same reason: a third `ShapeAction` is a build error.
        for action in SHAPE_ACTIONS {
            match action {
                ShapeAction::Select(_) | ShapeAction::Fill => {}
            }
        }
        assert_eq!(
            SHAPE_ACTIONS.len(),
            ALL.len() + 1,
            "a mode left off the row"
        );
    }

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

    /// The seat and the roster are one order — a view indexes by the first into a list
    /// built from the second.
    #[test]
    fn every_dial_sits_in_the_seat_its_index_names() {
        for (i, dial) in <Dial as strum::VariantArray>::VARIANTS.iter().enumerate() {
            assert_eq!(dial.index(), i, "{dial:?} names a seat it does not sit in");
        }
    }

    /// Every dial says what it does, leading with its own caption so the two cannot
    /// come to call one knob two things.
    #[test]
    fn every_dial_says_what_it_does() {
        for dial in <Dial as strum::VariantArray>::VARIANTS {
            assert!(
                dial.tip().starts_with(dial.label()),
                "{dial:?}: {:?}",
                dial.tip()
            );
            let (lo, hi) = dial.range();
            assert!(
                lo < hi && dial.step() > 0.0 && dial.step() <= hi - lo,
                "{dial:?}"
            );
        }
    }

    /// A fraction along a track is the value it points at, and past either end is that
    /// end — a drag off the trough cannot ask for a feather the rasterizer refuses.
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "a fraction of an exactly-representable track lands on the value itself, so the assertion is identity rather than proximity"
    )]
    fn a_fraction_along_a_track_is_the_value_under_it() {
        assert_eq!(Dial::Feather.value_at(0.5), MAX_FEATHER / 2.0);
        assert_eq!(Dial::Feather.value_at(-1.0), 0.0);
        assert_eq!(Dial::MaskOpacity.value_at(2.0), 1.0);
    }

    /// With nothing in hand there is nothing to set. The bar draws no track at all
    /// rather than a row of dim ones.
    #[test]
    fn nothing_in_hand_mounts_no_dial() {
        assert!(dials(None).is_empty());
    }
}
