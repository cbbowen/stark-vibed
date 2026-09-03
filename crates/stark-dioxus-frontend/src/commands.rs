//! What a command *does*, and what it says about itself that only this frontend
//! knows (§25).
//!
//! The registry itself is `stark_chrome::commands` — the variants, the chords, the
//! words, the tables. What is here is the half that could not travel:
//!
//! - **`run`** dispatches, opens dialogs, writes signals and asks gates that are the
//!   app's. A free function rather than a method, because `Command` belongs to
//!   another crate now and the orphan rule says so (CLAUDE.md) — which is the
//!   boundary reporting itself: every arm reaches `AppState`.
//! - **`active`** reads this frontend's own state rather than the engine's
//!   projection, so it cannot be a rule over `ObservableState` the way `enabled` is.
//! - **`icon`** is inline SVG (`crate::icons`), which is a DOM idiom.
//! - **`find`** and the four store doors, which hold signals.

use dioxus::html::{Key, Modifiers};
use dioxus::prelude::{ReadableExt, Signal, WritableExt};
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::SelectionOp;

use crate::icons;
use crate::input::accel;
use crate::platform;
use crate::state::{AppState, dispatch, update_brush};
use stark_chrome::brush_config::{MAX_RADIUS, MIN_RADIUS};
use stark_chrome::commands::{Bindings, Chord, Command, PickScope, StoredBinding};
use stark_chrome::keys::{Keystroke, Mods, Role};

pub fn load(state: AppState) {
    let Some(stored) = stark_chrome::storage::load_list::<StoredBinding>() else {
        return;
    };
    let overrides = stored
        .into_iter()
        .map(|row| (row.command, row.chord))
        .collect();
    let mut bindings = state.bindings;
    bindings.set(Bindings { overrides });
}

/// Give `command` the captured chord, and persist the table — the palette's
/// commit (`rail::CommandSearch`), written through [`Bindings::rebind`] so a
/// stolen chord and its victim's row change in the same write.
pub fn rebind(state: AppState, command: Command, chord: Chord) {
    edit(state, |b| b.rebind(command, chord));
}

/// Take `command`'s binding away and persist the table — the palette's other
/// commit, Backspace where a chord would be.
pub fn unbind(state: AppState, command: Command) {
    edit(state, |b| b.unbind(command));
}

/// One change to the table, written to the signal and to storage as one act —
/// so what the rows show, what the keyboard answers, and what the next visit
/// loads cannot be three states.
fn edit(state: AppState, change: impl FnOnce(&mut Bindings)) {
    let mut bindings = state.bindings;
    let mut next = bindings.peek().clone();
    change(&mut next);
    bindings.set(next);
    let stored: Vec<StoredBinding> = bindings
        .peek()
        .overrides
        .iter()
        .map(|(command, chord)| StoredBinding {
            command: *command,
            chord: chord.clone(),
        })
        .collect();
    stark_chrome::storage::save_list(&stored);
}

/// This browser's keydown, as the shared tables read it (`stark_chrome::keys`).
///
/// **The one translation this frontend owes the registry**, and the reason the
/// registry could travel at all: `accel` is Ctrl here and Command on a Mac, `typed`
/// is whatever the layout produces, and the three keys a capture spends on itself
/// have DOM names. None of that is derivable a crate down.
///
/// `Key::Character` is taken only when it is exactly one `char`: a dead key or an IME
/// composition reports a longer string, and neither is a chord.
///
/// `code` is a parameter because the event hands it back as an owned `String` and a
/// keystroke borrows it — the caller keeps it alive for the length of the lookup.
pub fn stroke<'a>(e: &platform::KeyEvent, code: &'a str) -> Keystroke<'a> {
    stroke_of(e.modifiers(), &e.key(), code)
}

/// [`stroke`] from the parts, for the surface whose event is Dioxus's own rather
/// than the window's ([`crate::rail`]'s capture field).
pub fn stroke_of<'a>(m: Modifiers, key: &Key, code: &'a str) -> Keystroke<'a> {
    let role = match key {
        Key::Escape => Role::Escape,
        Key::Backspace => Role::Backspace,
        Key::Control | Key::Shift | Key::Alt | Key::AltGraph | Key::Meta => Role::Modifier,
        _ => Role::Ordinary,
    };
    let typed = match key {
        Key::Character(c) => {
            let mut chars = c.chars();
            match (chars.next(), chars.next()) {
                (Some(k), None) => Some(k),
                _ => None,
            }
        }
        _ => None,
    };
    Keystroke {
        mods: Mods {
            ctrl: accel(m),
            shift: m.contains(Modifiers::SHIFT),
            alt: m.contains(Modifiers::ALT),
        },
        typed,
        code,
        role,
    }
}

/// The command `e` asks for, if any — the one reader on the dispatch path, asking
/// this browser's own table. `peek`: a keydown is no reason for anything to
/// re-render.
pub fn find(state: AppState, e: &platform::KeyEvent) -> Option<Command> {
    let code = e.code();
    state
        .bindings
        .peek()
        .lookup(&stroke(e, &code))
        // A row that matches may still decline the *keystroke* — today only
        // FinishMode's bare Enter ([`claims`]). Filtered here rather
        // than in `run`, because the caller `prevent_default`s whatever this
        // answers, and a claim is the one thing a declined Enter must not make.
        .filter(|c| claims(*c, state))
}

/// The command's mark (`crate::icons`). Total, because a control rendering
/// a command has nothing else to wear; the three keyboard-only commands
/// wear the mark of the knob they step or the subject they turn, on the
/// sharing argument `icons` already makes — the bracket keys are the Size
/// slider's own knob (§18.1.9), so they wear its ruler.
pub fn icon(command: Command) -> &'static str {
    match command {
        Command::Undo => icons::UNDO,
        Command::Redo => icons::REDO,
        Command::Deselect => icons::SELECTION_NONE,
        Command::InvertSelection => icons::SELECTION_INVERT,
        // The one family where the glyph *is* the meaning rather than the
        // control's (`icons`): a tool that draws a rectangle is marked
        // with a rectangle.
        Command::SelectRect => icons::RECTANGLE,
        Command::SelectEllipse => icons::CIRCLE,
        Command::SelectLasso => icons::LASSO,
        Command::MirrorView => icons::MIRROR_VIEW,
        Command::BrushSmaller | Command::BrushLarger => icons::SIZE,
        Command::NewDocument => icons::NEW_DOCUMENT,
        Command::OpenDocument => icons::OPEN_DOC,
        Command::SaveDocument => icons::SAVE,
        Command::ImportImage => icons::IMPORT_IMAGE,
        Command::ExportImage => icons::EXPORT,
        Command::Share => icons::SHARE,
        Command::ToggleTimeline => icons::TIMELINE,
        Command::TimingStats => icons::TIMING,
        Command::Credits => icons::CREDITS,
        Command::ToggleNavigator => icons::NAVIGATOR,
        Command::ToggleQuickBrushes => icons::QUICK_BRUSHES,
        Command::Settings => icons::SETTINGS,
        Command::EditBrush => icons::EDIT_BRUSH,
        Command::SavePreset => icons::SAVE,
        Command::Transform => icons::TRANSFORM,
        Command::FloatSelection => icons::FLOAT,
        Command::FillSelection => icons::PAINT_BUCKET,
        Command::GradientFill => icons::GRADIENT,
        Command::AddLayer => icons::ADD_LAYER,
        Command::AddFrame => icons::ADD_FRAME,
        Command::AddPerspective => icons::ADD_LAYER,
        // The dismissal mark every panel header wears, and the tick every
        // Done chip does — the two acts these commands are the names of.
        Command::CancelMode => icons::CLOSE,
        Command::FinishMode => icons::DONE,
        // The bar's own three marks, which are a picture of the question:
        // one sheet, a sheet over what is under it, a stack.
        Command::SetPickScope(scope) => match scope {
            PickScope::ThisLayer => icons::ONE_LAYER,
            PickScope::AndBelow => icons::AND_BELOW,
            PickScope::AllLayers => icons::ALL_LAYERS,
        },
        // The mark its own title bar wears, so the menu and the palette
        // both stay a picture of the stack.
        Command::TogglePanel(id) => crate::layout::panel_glyph(id),
    }
}
pub fn active(command: Command, state: AppState) -> Option<bool> {
    match command {
        Command::SelectRect => Some(armed(state, Tool::SelectRect)),
        Command::SelectEllipse => Some(armed(state, Tool::SelectEllipse)),
        Command::SelectLasso => Some(armed(state, Tool::SelectLasso)),
        Command::ToggleTimeline => Some(*state.timeline.open.read()),
        Command::ToggleNavigator => Some(*state.navigator.read()),
        Command::ToggleQuickBrushes => Some(*state.slots.pinned.read()),
        // Exactly one of the three is lit, always, which is the claim that
        // the row is one question rather than three switches — and it is
        // read here so a chord pressed under the bar moves the light the
        // chip would have moved.
        Command::SetPickScope(scope) => Some(*state.pick.scope.read() == scope),
        Command::TogglePanel(id) => Some(!state.panels.hidden.read().contains(&id)),
        Command::Share => Some(*state.collab.phase.read() == crate::collab::CollabPhase::Shared),
        _ => None,
    }
}

/// Whether the chrome should offer this command right now — the menu's
/// greyed rows and a bar's greyed chips (`widgets::CommandButton`), read
/// off the projection so a disabled entry is a fact about the document
/// ("nothing to undo", "nothing selected") rather than a mood.
///
/// **Presentation only.** The act's own gate lives on [`run`]
/// and asks different questions, deliberately: undo during playback is
/// *enabled* — nothing on screen says otherwise — and resolves what is in
/// flight rather than refusing (see [`edit_history`]). A caller must not
/// skip `run`'s gate because this said yes.
///
/// `None` is startup — no engine yet, so no document: the commands that ask
/// the projection answer no, and everything else (a dialog, a file pick)
/// needs nothing from it.
pub fn run(command: Command, state: AppState) {
    match command {
        Command::Undo => edit_history(state, DocCommand::Undo),
        Command::Redo => edit_history(state, DocCommand::Redo),
        Command::Deselect => {
            if may_edit(state) {
                dispatch(state, DocCommand::Select(SelectionOp::select_all()));
            }
        }
        Command::InvertSelection => {
            if may_edit(state) {
                dispatch(state, DocCommand::InvertSelection);
            }
        }
        Command::SelectRect => arm_tool(state, Tool::SelectRect),
        Command::SelectEllipse => arm_tool(state, Tool::SelectEllipse),
        Command::SelectLasso => arm_tool(state, Tool::SelectLasso),
        Command::MirrorView => dispatch(state, ViewCommand::MirrorH),
        Command::BrushSmaller => step_radius(state, 1.0 / SIZE_STEP),
        Command::BrushLarger => step_radius(state, SIZE_STEP),
        Command::NewDocument => open_dialog(state.dialogs.new_document),
        Command::OpenDocument => crate::files::open_document(state),
        Command::SaveDocument => crate::files::save_document(state),
        Command::ImportImage => crate::images::import_image(state),
        Command::ExportImage => open_dialog(state.dialogs.export),
        Command::Share => {
            crate::collab::share(state);
            open_dialog(state.dialogs.session);
        }
        Command::ToggleTimeline => {
            let open = *state.timeline.open.peek();
            crate::panels::timeline::set_open(state, !open);
        }
        Command::TimingStats => open_dialog(state.dialogs.timing),
        Command::Credits => open_dialog(state.dialogs.credits),
        Command::ToggleNavigator => {
            let open = *state.navigator.peek();
            crate::navigator::set_open(state, !open);
        }
        // Through `slots::set_pinned` rather than writing the signal, for the
        // reason the two above go through theirs: the rack's visibility is
        // remembered, and the one writer is what makes that structural
        // (`crate::visibility`).
        Command::ToggleQuickBrushes => {
            let pinned = *state.slots.pinned.peek();
            crate::slots::set_pinned(state, !pinned);
        }
        // Ungated like the other toggles: which panels are up is chrome,
        // not document. The two halves an entry must not forget — waking a
        // sleeping stack on open, telling the tour on close — live in
        // `layout`'s own functions, which is why this goes through
        // `toggle_panel` rather than writing `hidden`.
        Command::TogglePanel(id) => {
            crate::layout::toggle_panel(state, state.panels, id);
        }
        Command::Settings => open_dialog(state.dialogs.settings),
        Command::EditBrush => {
            open_dialog(state.brush_editor_open);
            // The dialog is frontend state and reaches no engine, so there
            // is no command for the tour to read (§24.2). Its series of
            // cards is the one thing this click owes anybody.
            crate::tutor::did(state, crate::tutor::Deed::OpenedBrushEditor);
        }
        Command::SavePreset => open_dialog(state.preset_save_open),
        // Ungated, with the view and brush acts: how far a sample reaches is
        // an argument to a *request* (`Engine::pick_color`), read at the
        // moment of the sample and committing nothing — and the bar's own
        // chips have never been refused mid-playback either. The gate that
        // matters is the sample's, and it is the drag table's
        // (`DragAction::claims`).
        Command::SetPickScope(scope) => {
            let mut want = state.pick.scope;
            want.set(scope);
        }
        Command::Transform => {
            if may_edit(state) {
                crate::panels::transform::begin_transform(state);
            }
        }
        Command::FloatSelection => {
            if may_edit(state) {
                crate::panels::select::float_selection(state);
            }
        }
        Command::FillSelection => {
            if may_edit(state) {
                crate::panels::select::fill_selection(state);
            }
        }
        Command::GradientFill => {
            if may_edit(state) {
                crate::panels::gradient_bar::begin_fill(state);
            }
        }
        Command::AddLayer => {
            if may_edit(state) {
                crate::panels::layer::add_layer(state);
            }
        }
        Command::AddFrame => {
            if may_edit(state) {
                crate::panels::frame::add_frame(state);
            }
        }
        // Half of [`may_edit`], and the halves are asked separately on purpose.
        // Adding a guide *is* a document edit now (§20.5), so it is refused
        // while the timeline is playing back, like every other one: what is on
        // screen then is a historical state, and editing it would be editing
        // the wrong document. The composing half is deliberately not asked —
        // this command puts down whatever was composing itself
        // (`modes::leave`), so it replaces a mode rather than being refused by
        // one, which is the behaviour it has always had.
        Command::AddPerspective => {
            if !crate::panels::timeline::is_playing(state) {
                crate::panels::guides::add_perspective(state);
            }
        }
        Command::CancelMode => escape(state),
        // Gated on the dialogs where CancelMode ladders through them:
        // Enter under a dialog belongs to the dialog's form, and a commit
        // it could not see landing beneath it would be the worse surprise.
        Command::FinishMode => {
            if !dialog_open(state) {
                crate::modes::finish(state);
            }
        }
    }
}

/// Whether this command claims its keystroke right now — asked by [`find`]
/// **before** the caller's `prevent_default`, where [`run`]'s
/// own gates decide only what happens after the claim.
///
/// `true` for almost everything: a declined act still claims its chord,
/// because the browser's default would answer it with something worse (see
/// `input`'s keydown handler). The exception is bare **Enter**, which is
/// the keyboard's activation of whatever control has focus — a Done that
/// claimed it unconditionally would eat every focused button and dialog
/// form in the app, so FinishMode claims it only while there is a mode for
/// it to finish and no dialog over that mode. Esc has no such double life:
/// outside text entry (already carved out before the table is consulted),
/// the browser does nothing with it worth keeping.
fn claims(command: Command, state: AppState) -> bool {
    match command {
        Command::FinishMode => crate::modes::is_composing(state) && !dialog_open(state),
        _ => true,
    }
}

/// Raise a root-mounted dialog's flag; the dialog's own `on_close` lowers it.
fn open_dialog(mut flag: Signal<bool>) {
    flag.set(true);
}

/// One tap of `[` or `]`, as a ratio.
///
/// Equal *ratios* rather than equal pixels, because the hand feels radius
/// proportionally: the +1px that is a visible jump on a 5px liner is nothing on
/// a 300px wash. A tenth is about the smallest change a mark reliably shows,
/// and it compounds across the whole range quickly under the key's own
/// auto-repeat — 1 → 500 is ~65 repeats, a couple of seconds of holding `]`.
/// Up and down are exact inverses (multiply by it, divide by it), so a tap too
/// far is a tap back rather than a slowly drifting number.
const SIZE_STEP: f32 = 1.1;

/// Step the live brush's radius by `factor` — the keyboard sibling of the Size
/// slider and the accelerator drag (§18.1.9), writing through the same
/// [`update_brush`] and clamped to the same bounds, so a tap cannot put the
/// brush anywhere the panel could not show or take back.
///
/// Ungated by [`may_edit`] on purpose: tuning the brush edits no document, and
/// the slider this shadows is not refused mid-playback either — the keyboard
/// says what the panel says.
fn step_radius(state: AppState, factor: f32) {
    update_brush(state, move |_, t| {
        t.size = (t.size * factor).clamp(MIN_RADIUS, MAX_RADIUS);
    });
}

/// Arm `tool` for the next canvas gesture — or hand the brush back, if it is
/// the tool already in hand (§6.8). One act rather than two, because the chip
/// and the chord must mean the same thing on a second press: the control that
/// armed a tool is the one that takes it back, and R pressed twice is the
/// keyboard reaching that same control twice.
///
/// **Ungated**, with the view and brush acts: arming commits nothing to the
/// document — `SetTool` is a `ViewCommand` — and the panel's chips have never
/// been refused mid-playback or under a composing mode either. The *gesture*
/// an armed tool then makes is a different act with a gate of its own, which
/// the canvas has always asked (`crate::input`).
fn arm_tool(state: AppState, tool: Tool) {
    let already = crate::panels::select::current_tool(state) == tool;
    let next = if already { Tool::Brush } else { tool };
    // Which of the three was last in hand, kept where the session cannot keep it:
    // a selecting gesture disarms to `Tool::Brush` (§6.8), so the engine's own
    // `tool` has forgotten which marquee drew by the time anything asks. Recorded
    // here because this is the one door into arming — the chip, the chord and the
    // palette are all this call — and read back by [`arm_shape_tool`].
    if next.is_selection() {
        let mut last = state.shape_tool;
        last.set(next);
    }
    dispatch(state, ViewCommand::SetTool(next));
}

/// Hand back a shape tool without naming one — the last one armed
/// ([`Signals::shape_tool`](crate::state::Signals::shape_tool)), and nothing at all if one is already in hand.
///
/// The Select panel's action row is what asks (`crate::panels::select`): picking
/// what a shape *does* is a statement about a gesture that has not been made yet,
/// and with the brush in hand there is nothing for it to be a statement about.
/// Which of the three would draw it is a question the row does not answer, so it
/// takes the answer the chips above left behind.
///
/// Leaving an armed tool alone is the same rule read the other way: the row says
/// nothing about which of the three, so a lasso stays a lasso.
pub fn arm_shape_tool(state: AppState) {
    if !crate::panels::select::current_tool(state).is_selection() {
        // Read out first, never `*state.shape_tool.peek()` in the argument: a
        // signal's read guard lives to the end of the *statement*, and the call it
        // would be an argument to writes that same signal — which is a panic, taken
        // in a handler that has already dispatched half of what it came to do.
        let last = *state.shape_tool.peek();
        arm_tool(state, last);
    }
}

/// Whether `tool` is the one the next gesture would use — reactively (`read`,
/// not `peek`), because this is the answer a lit chip is mounted on.
fn armed(state: AppState, tool: Tool) -> bool {
    state.obs.read().as_ref().is_some_and(|o| o.tool == tool)
}

/// Whether a **document edit** may be accepted right now.
///
/// The two questions the canvas already asks of a press, asked of every other
/// door into the document — the keyboard shortcuts and the chrome's own rows,
/// which between them were the doors that asked neither:
///
/// - **The playhead is moving.** A commit clears the withheld half of the
///   timeline, so an edit laid under a running playback deletes the rest of the
///   piece (`crate::panels::timeline`). The canvas refuses a press for this;
///   Ctrl+A went through and truncated the history from the keyboard — and the
///   menu's Deselect kept doing it after the keyboard was fixed, which is what
///   putting the gate on the act rather than the call site is for.
/// - **A mode is composing.** Its preview is computed against the committed
///   document (`crate::modes`), and the bar that carries these very commands
///   stands recessed and inert behind the mode's own — deselecting
///   mid-transform would move the wrong region on "Done"
///   (`crate::panels::select::SelectionBar`). The chrome says what the screen
///   says — and this gate is also what lets a recessed bar keep its chips
///   mounted: a click that somehow reached one would be refused here.
fn may_edit(state: AppState) -> bool {
    !crate::panels::timeline::is_playing(state) && !crate::modes::is_composing(state)
}

/// Esc's ladder (MODAL_DESIGN.md), one rung per press: the open dialog, else
/// the composing mode, else the frame or filter layer selected for composing,
/// else Timeline. Ordered outermost-in — a dialog stands over a mode, a mode
/// over the bar that raised it, the bars over the timeline's — so each press
/// peels the layer the eye reads as topmost, and never two at once: Esc from
/// a gradient matte drops the axis first and leaves the frame second.
///
/// The dialogs are closed *here*, not declined in deference to their own Esc
/// handlers, because outside a text field they have none: every element-level
/// Escape in the app lives on an input (the palette's field, the rename and
/// name drafts), where the window's keydown binding is already withheld
/// (`platform::KeyEvent::on_text_entry`). One actor per keystroke, so the
/// dioxus-vs-window handler ordering that a second actor would hang on never
/// gets asked.
fn escape(state: AppState) {
    // A pop-out flown out of a bar stands over everything below, so it is the
    // first thing Escape puts down (`widgets::PopoutId`, §25.7). Above the
    // dialogs because it is the innermost surface, and kept off the dialog list
    // for a reason of its own: that list is what stands `FinishMode` down, and a
    // library opened from a composing bar must not take Enter's "Done" away.
    if crate::widgets::close_popout(state) {
        return;
    }
    if close_dialogs(state) {
        return;
    }
    if crate::modes::is_composing(state) {
        crate::modes::cancel(state);
        return;
    }
    // The two layer kinds that are composed rather than painted (§15.7,
    // §21.6): Esc is their bars' own Done — the topmost paint layer selected
    // instead, the only way a frame or filter is ever "deselected". The guide
    // bar's bargain: nothing is uncommitted, so leaving is the whole act.
    // Enter deliberately does *not* reach here — these are standing states,
    // and an Enter claimed through one would eat every focused button's
    // Enter for as long as the layer stayed selected ([`claims`]).
    if composing_layer_selected(state) {
        crate::panels::frame::done_composing(state);
        return;
    }
    if *state.timeline.open.peek() {
        crate::panels::timeline::set_open(state, false);
    }
}

/// Whether the selected layer is a frame or a filter — the kinds whose bar is
/// up for as long as they are selected, asked handler-time (`peek`).
fn composing_layer_selected(state: AppState) -> bool {
    state
        .obs
        .peek()
        .as_ref()
        .is_some_and(|o| crate::panels::frame::selected_frame_of(o).is_some())
        || crate::panels::filter::selected_filter(state).is_some()
}

/// Whether any root-mounted dialog is up — Esc's first rung, and the fact that
/// stands FinishMode down ([`claims`]).
fn dialog_open(state: AppState) -> bool {
    state.root_dialogs().iter().any(|flag| *flag.peek())
}

/// Lower the root dialog on top, if one is up; `true` if one was. Lowering the
/// flag *is* the dialog's own close — every `on_close` in `main` does nothing
/// else (`AppState::root_dialogs`).
///
/// The topmost only, one per press: the list is in stacking order, and the
/// preset-name dialog stands over the brush editor that raised it — an Esc
/// meant for the name must not take the editor down with it.
fn close_dialogs(state: AppState) -> bool {
    let top = state
        .root_dialogs()
        .into_iter()
        .rev()
        .find(|flag| *flag.peek());
    match top {
        Some(mut flag) => {
            flag.set(false);
            true
        }
        None => false,
    }
}

/// Undo or redo, having first put down whatever was in hand.
///
/// Not [`may_edit`]'s flat refusal, because these two are not refusable in the
/// same sense. Nothing on screen says undo is unavailable — no bar stood down to
/// carry the message — so a shortcut that silently did nothing would read as a
/// broken keyboard rather than as a rule. Editing the history is instead an
/// unambiguous statement that the composition in flight is over, so it ends the
/// way scrubbing ends one: the preview dropped, nothing committed. Playback
/// stops for the same reason it stops when the transport is touched — the hand
/// has taken the playhead back off the loop that was moving it.
fn edit_history(state: AppState, command: DocCommand) {
    crate::panels::timeline::stop(state);
    crate::modes::leave(state);
    dispatch(state, command);
}
