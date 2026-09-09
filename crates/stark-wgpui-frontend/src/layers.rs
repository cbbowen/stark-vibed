//! The layers panel: the stack, and the acts that rearrange it (§14, §11.2 N4).
//!
//! **Almost nothing here is a decision.** What the rows *are* — which are folded
//! away, which can be removed, what Carry and Release would each mean, how deep a
//! drop lands — is `stark_ui::layer_tree`, and has been since N0: it was already
//! split out of the web panel because it was the part that could be tested. This
//! module is the markup over it, plus one thing the tree cannot answer: where each
//! row was laid out, so a press can find it.
//!
//! It measures rather than predicts, for the reason [`crate::panel`] gives and the
//! bug that taught it.
//!
//! # The two acts a *tree* has
//!
//! A flat roster is dragged and that is all. A tree has two more, and they are the
//! same mechanism read from either side (§14.2): **Carry** puts a layer into the
//! group below it, and **Release** takes it out of the one it is in. `Row` answers
//! both — [`Row::carry`] and [`Row::release`], each the whole `MoveLayer` — so a
//! row's two buttons are a `Some` each rather than a rule written here.
//!
//! Nor is *which controls are live* a decision here. Blend and clip go inert with
//! nothing under the layer and part company on a filter, and both answers are the
//! row's ([`Row::blend_inert`]). This panel used to ask neither, so its clip chip and
//! its blend picker were live on the bottom layer of the document, where every mode
//! is the identity and a clip would leave nothing to show.

use stark_engine::ObservableState;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::LayerId;
use stark_ui::commands::{Bindings, Command};
use stark_ui::icons::Icon;
use stark_ui::layer_tree::{self, Row};
use wgpui::{
    App, Bounds, IntoElement, Pixels, Point, RenderOnce, SharedString, Window, canvas, div,
    prelude::*, px, rgb,
};

use wgpui_component::select::Select;

use crate::controls::Controls;
use crate::style::{self, StyleExt};

/// What a press on the layers panel landed on.
///
/// The row-bound acts name their **layer**, not a position: the panel draws the rows
/// in display order (`layer_tree::display`) and [`act`] reads them in the engine's,
/// and an index would have meant one of the two orders without saying which. It also
/// keeps Remove working on a layer folded away under a shut group, which is not in
/// the displayed list at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The row's body: select this layer to paint on.
    Row(LayerId),
    /// Its eye.
    Visible(LayerId),
    /// Its fold triangle — only a group has one.
    Fold(LayerId),
    /// Put this layer into the group below it (§14.2).
    Carry(LayerId),
    /// Take it out of the group it is in.
    Release(LayerId),
    /// Clip it to what it sits on.
    Clip(LayerId),
    /// One of the acts on the whole stack.
    Add,
    Duplicate,
    Remove,
}

/// Where each control was laid out, as of the last painted frame.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .size_full()
}

/// Which control a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// A small square control — an eye, a carry, a clip mark.
///
/// The mark is `stark_ui::icons`' rather than a character: which glyph a control
/// wears says what the control *means*, and the two frontends agreeing about that is
/// the whole reason the catalog is shared (§11.2 N8).
#[derive(IntoElement)]
struct Chip {
    glyph: Icon,
    on: bool,
    /// Drawn, but with nothing to say about this row. Shown rather than hidden: the
    /// control belongs to the layer wherever it sits, and a row that loses a control
    /// when it is dragged to the bottom of the document reads as a bug.
    inert: bool,
    region: Region,
    regions: Regions,
    /// What the hover says a mark means — the word the web app prints beside it.
    tip: SharedString,
}

impl Chip {
    /// A chip that is simply a chip: lit by nothing, refused by nothing.
    fn plain(glyph: Icon, region: Region, regions: &Regions, tip: impl Into<SharedString>) -> Self {
        Self {
            glyph,
            on: false,
            inert: false,
            region,
            regions: regions.clone(),
            tip: tip.into(),
        }
    }
}

impl RenderOnce for Chip {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let chip = div()
            // The region is the chip's identity already; a hover needs it as an id.
            .id(SharedString::from(format!("{:?}", self.region)))
            .relative()
            .w(px(20.))
            .h(px(18.))
            .flex()
            .items_center()
            .justify_center()
            .rounded_sm()
            .text_xs()
            .child(probe(&self.regions, self.region))
            // The colour is passed rather than inherited: a rasterized glyph is
            // tinted by its *own* element, not by the row around it
            // (`crate::icons`).
            .when(self.on && !self.inert, |el| el.bg(rgb(style::LIT)))
            .child(crate::icons::icon(
                self.glyph,
                match (self.inert, self.on) {
                    (true, _) => style::INK_DEAD,
                    (false, true) => style::INK_LIT,
                    (false, false) => style::INK_MARK,
                },
            ));
        style::tip(chip, self.tip)
    }
}

/// Build the panel's element tree.
///
/// `rows` is `layer_tree::rows`' answer in the engine's own order, unmodified: which
/// rows exist, which are folded away and which order to draw them in
/// (`layer_tree::display`) are all the tree's, and drawing them is this module's.
pub fn layers_body(
    obs: Option<&ObservableState>,
    rows: &[Row],
    bindings: &Bindings,
    controls: &Controls,
    regions: &Regions,
) -> impl IntoElement {
    let active = obs.map(|o| o.active_layer);
    let selected = active.and_then(|id| rows.iter().find(|r| r.info.id == id));
    let opacity = selected.map_or(1.0, |r| r.info.opacity);
    // Whether the blend picker has anything to say about the selected layer. The row
    // answers it (`layer_tree::Row::blend_inert`); this panel used not to ask, and
    // offered every mode on the bottom layer of the document, where they are all the
    // identity.
    let blend_inert = selected.is_none_or(Row::blend_inert);
    // Top of the document first, folded rows gone — the tree's own turn of the list
    // (`layer_tree::display`) rather than one written here. It is the same list a drag
    // is resolved against, which is why the rule is a function and not a line of
    // iterator: handed the engine's order instead, `landing` answers the mirror image
    // of the right drop and says nothing about it.
    let shown = layer_tree::display(rows);

    div()
        .flex()
        .flex_col()
        .gap_1()
        // The selected layer's two continuous knobs, on the widget layer's controls
        // (`crate::controls`): the blend mode is a drop-down, which §25.9 asks for
        // once the answers stop fitting on one line, and the opacity a track. Both
        // wear their mark rather than their word (`crate::panel`), so each sits on one
        // line where the labelled pair took two.
        .child(style::tip(
            div()
                .id("layer-blend")
                .flex()
                .items_center()
                .gap_2()
                .child(crate::icons::icon(
                    stark_ui::icons::BLEND,
                    if blend_inert {
                        style::INK_DEAD
                    } else {
                        style::INK_MARK
                    },
                ))
                .child(
                    div()
                        .flex_1()
                        .child(Select::new(&controls.blend).w_full().disabled(blend_inert)),
                ),
            if blend_inert {
                "Nothing composites under this layer, so every mode looks the same here"
            } else {
                "Blend \u{2014} how this layer meets what is under it"
            },
        ))
        .child(crate::panel::Slider::new(
            stark_ui::icons::OPACITY,
            "Opacity \u{2014} how much of this layer shows",
            format!("{opacity:.2}"),
            &controls.opacity,
        ))
        // The acts on the whole stack, above the roster they act on.
        .child(
            div().flex().gap_1().children(
                // The catalog's own marks (`stark_ui::icons`), so the three
                // acts here and the three in the web app's header are one control
                // apiece rather than two that resemble each other. A stack gaining a
                // member, a copy of one, and the destructive one — which is what a
                // trash says everywhere.
                [
                    (
                        Region::Add,
                        stark_ui::icons::ADD_LAYER,
                        Command::AddLayer.tooltip(bindings),
                    ),
                    (
                        Region::Duplicate,
                        stark_ui::icons::DUPLICATE,
                        "Duplicate the layer".to_string(),
                    ),
                    (
                        Region::Remove,
                        stark_ui::icons::REMOVE,
                        "Remove the layer".to_string(),
                    ),
                ]
                .map(|(region, glyph, tip)| Chip::plain(glyph, region, regions, tip)),
            ),
        )
        // The roster, as the panel shows it — `layer_tree::display`, not this
        // module's own turn of the engine's list.
        .child(
            div()
                .flex()
                .flex_col()
                .gap_0p5()
                .py_1()
                .children(shown.iter().copied().map(|row| {
                    let id = row.info.id;
                    let worn = active == Some(id);
                    let clip_inert = row.clip_inert();
                    div()
                        .relative()
                        .flex()
                        .items_center()
                        .gap_1()
                        .pl(px(4.0 + row.info.depth as f32 * layer_tree::INDENT as f32))
                        .pr_1()
                        .py_0p5()
                        .rounded_sm()
                        .lit_row(worn)
                        .child(Chip {
                            glyph: if row.info.visible {
                                stark_ui::icons::VISIBLE
                            } else {
                                stark_ui::icons::HIDDEN
                            },
                            on: row.info.visible,
                            inert: false,
                            region: Region::Visible(id),
                            regions: regions.clone(),
                            tip: if row.info.visible { "Hide" } else { "Show" }.into(),
                        })
                        // The fold slot is drawn whatever the row is, empty for
                        // a layer that carries nothing: a triangle only some rows
                        // have would push their *names* right, and a column of
                        // names that do not line up is what makes a tree
                        // unreadable — the indent would stop meaning depth.
                        .child(if row.info.is_group {
                            Chip::plain(
                                if row.collapsed {
                                    stark_ui::icons::FOLD_SHUT
                                } else {
                                    stark_ui::icons::FOLD_OPEN
                                },
                                Region::Fold(id),
                                regions,
                                if row.collapsed { "Unfold" } else { "Fold" },
                            )
                            .into_any_element()
                        } else {
                            div().w(px(20.)).into_any_element()
                        })
                        .child(
                            // The name takes the slack, so the controls stay put
                            // down the column however long a layer is called.
                            div()
                                .relative()
                                .flex_1()
                                .text_sm()
                                .truncate()
                                .child(probe(regions, Region::Row(id)))
                                .child(layer_tree::layer_label(&row.info).into_owned()),
                        )
                        .when(row.info.clip, |el| {
                            el.child(crate::icons::icon(stark_ui::icons::CLIP, style::INK_LABEL))
                        })
                        // Carry and Release are a `Some` each rather than a rule
                        // written here — see the module note.
                        .when(row.carry().is_some(), |el| {
                            el.child(Chip::plain(
                                stark_ui::icons::CARRY,
                                Region::Carry(id),
                                regions,
                                "Carry on the layer below",
                            ))
                        })
                        .when(row.release().is_some(), |el| {
                            el.child(Chip::plain(
                                stark_ui::icons::RELEASE,
                                Region::Release(id),
                                regions,
                                "Release from its group",
                            ))
                        })
                        .child(Chip {
                            glyph: stark_ui::icons::CLIP,
                            on: row.info.clip,
                            // A mode over nothing is harmlessly the identity; a clip
                            // over nothing would erase the layer, which is why this
                            // one has to be stopped rather than left to do nothing
                            // (§14.4.3).
                            inert: clip_inert,
                            region: Region::Clip(id),
                            regions: regions.clone(),
                            tip: if clip_inert {
                                "Nothing composites under this layer, so clipping it \
                                 would leave nothing to show"
                            } else if row.info.clip {
                                "Unclip"
                            } else {
                                "Clip to the layer below"
                            }
                            .into(),
                        })
                })),
        )
}

/// What a press on `region` means as a command, given the rows it was drawn over.
///
/// A function rather than a `match` in the view, so the mapping is testable — which
/// matters more here than it looks: every arm is a claim about §14's vocabulary, and
/// three of them (Carry, Release, Clip) are the tree's answer rather than this
/// module's.
pub fn act(region: Region, rows: &[Row], active: Option<LayerId>) -> Option<Act> {
    let row = |id: LayerId| rows.iter().find(|r| r.info.id == id);
    Some(match region {
        Region::Row(id) => Act::Peer(PeerCommand::SetActiveLayer(row(id)?.info.id)),
        Region::Visible(id) => {
            let info = &row(id)?.info;
            Act::Doc(DocCommand::SetLayerVisible(info.id, !info.visible))
        }
        Region::Fold(id) => Act::Fold(row(id)?.info.id),
        // Refused where a clip would leave nothing to show, which is the same answer
        // the chip is drawn dim from (§14.4.3) — one question, asked of the row.
        Region::Clip(id) => {
            let r = row(id)?;
            if r.clip_inert() {
                return None;
            }
            Act::Doc(DocCommand::SetLayerClip(r.info.id, !r.info.clip))
        }
        // Into the group below and out of the group it is in: the two moves are the
        // tree's whole, not a rule reassembled here (§14.2), and a row with nowhere
        // to go offers no button at all.
        Region::Carry(id) => Act::Doc(row(id)?.carry()?),
        Region::Release(id) => Act::Doc(row(id)?.release()?),
        Region::Add => Act::Doc(DocCommand::AddLayer {
            carrier: None,
            above: active,
        }),
        Region::Duplicate => Act::Doc(DocCommand::DuplicateLayer(active?)),
        // The tree says whether a removal would leave a document behind (§14.2), so
        // the refusal is a property of the row rather than a count kept here.
        Region::Remove => {
            let id = active?;
            row(id)?
                .removable
                .then_some(Act::Doc(DocCommand::RemoveLayer(id)))?
        }
    })
}

/// What a press turns into. Two kinds, because they are two kinds of state (§4).
pub enum Act {
    /// A document edit: logged, undoable, replicated.
    Doc(DocCommand),
    /// Which layer this client paints on — presence, not the document (§17.4).
    Peer(PeerCommand),
    /// Folding a group away is the panel's own state: nothing about the document
    /// changes, and a collaborator's panel is theirs to fold.
    Fold(LayerId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_engine::LayerInfo;
    use stark_model::document::{ActionId, ActorId, BlendMode, Place};
    use std::collections::HashSet;

    /// A stand-in layer, spelled out because `LayerInfo` is the engine's projection
    /// and has no `Default` — a roster is something the engine *answers*, not
    /// something a caller builds, and this is the one place that has to.
    fn info(id: u64, depth: usize, is_group: bool) -> LayerInfo {
        LayerInfo {
            id: LayerId {
                action: ActionId {
                    lamport: id,
                    actor: ActorId::SOLO,
                },
                k: 0,
            },
            // Derived from the depth, which is all these fixtures ever nest: a row
            // one level down is carried by the row before it.
            blend: BlendMode::Normal,
            clip: false,
            opacity: 1.0,
            visible: true,
            carrier: (depth > 0).then(|| LayerId {
                action: ActionId {
                    lamport: id - 1,
                    actor: ActorId::SOLO,
                },
                k: 0,
            }),
            depth,
            is_group,
            has_backdrop: false,
            name: None,
            number: Some(1),
            matte: None,
            filter: None,
            has_underlay: false,
            merge_down: None,
            content_revision: None,
            translation: Default::default(),
        }
    }

    fn id_of(n: u64) -> LayerId {
        info(n, 0, false).id
    }

    /// A stack of two: a group at the foot, a layer above it. The upper one has
    /// something under it, which is what makes its blend and its clip mean anything
    /// (§14.4.3).
    fn stack() -> Vec<Row> {
        let mut top = info(2, 0, false);
        top.has_backdrop = true;
        top.has_underlay = true;
        let layers = vec![info(1, 0, true), top];
        layer_tree::rows(&layers, &HashSet::new())
    }

    /// The eye and the clip mark toggle what they show, rather than setting a fixed
    /// value — a control that always sent `true` would be dead the second time.
    #[test]
    fn the_row_toggles_read_the_row() {
        let rows = stack();
        let visible = rows[1].info.visible;
        match act(Region::Visible(id_of(2)), &rows, None) {
            Some(Act::Doc(DocCommand::SetLayerVisible(_, to))) => assert_eq!(to, !visible),
            _ => panic!("the eye sets visibility"),
        }
        match act(Region::Clip(id_of(2)), &rows, None) {
            Some(Act::Doc(DocCommand::SetLayerClip(_, to))) => assert!(to),
            _ => panic!("the clip mark sets clipping"),
        }
    }

    /// A clip over nothing would erase the layer, so the chip that is drawn dim is
    /// refused as well as dimmed (§14.4.3). This panel used to send the command.
    #[test]
    fn the_clip_chip_is_refused_where_it_is_dim() {
        let rows = stack();
        assert!(rows[0].clip_inert(), "the foot of the document");
        assert!(act(Region::Clip(id_of(1)), &rows, None).is_none());
    }

    /// Selecting a row is **presence**, not a document edit: two collaborators paint
    /// on different layers of one document (§17.4).
    #[test]
    fn choosing_a_row_is_presence_rather_than_an_edit() {
        let rows = stack();
        assert!(matches!(
            act(Region::Row(id_of(1)), &rows, None),
            Some(Act::Peer(PeerCommand::SetActiveLayer(_)))
        ));
    }

    /// Carry is the move the *tree* spelled, rather than one reassembled here out of
    /// the pieces it handed over.
    #[test]
    fn carry_is_the_move_the_tree_spelled() {
        let rows = stack();
        let Some(spelled) = rows[1].carry() else {
            panic!("a layer over a group can be carried onto it");
        };
        match (act(Region::Carry(id_of(2)), &rows, None), spelled) {
            (
                Some(Act::Doc(DocCommand::MoveLayer { carrier, at, .. })),
                DocCommand::MoveLayer {
                    carrier: want,
                    at: place,
                    ..
                },
            ) => {
                assert_eq!(carrier, want);
                assert_eq!((at, place), (Place::Top, Place::Top));
            }
            _ => panic!("carry moves the layer"),
        }
    }

    /// An act with nothing selected asks for nothing, rather than reaching for a
    /// layer that is not there.
    #[test]
    fn the_stack_acts_need_a_selection() {
        let rows = stack();
        assert!(act(Region::Duplicate, &rows, None).is_none());
        assert!(act(Region::Remove, &rows, None).is_none());
        // Add is the exception: with nothing selected it goes on top.
        assert!(matches!(
            act(Region::Add, &rows, None),
            Some(Act::Doc(DocCommand::AddLayer { above: None, .. }))
        ));
    }

    /// Removing the last row would leave no document, and the tree says so — the
    /// panel does not keep a count of its own (§14.2).
    #[test]
    fn the_only_stack_refuses_to_be_removed() {
        let layers = vec![info(1, 0, false)];
        let rows = layer_tree::rows(&layers, &HashSet::new());
        assert!(!rows[0].removable, "the sole stack is what a document is");
        assert!(act(Region::Remove, &rows, Some(id_of(1))).is_none());
    }

    /// A layer folded away under a shut group is still a layer, and Remove still
    /// answers for it — which is what naming a **layer** rather than a display index
    /// buys, since a hidden row is not in the list the panel draws.
    #[test]
    fn a_folded_away_layer_can_still_be_removed() {
        let layers = vec![info(1, 0, true), info(2, 1, false)];
        let rows = layer_tree::rows(&layers, &HashSet::from([id_of(1)]));
        assert!(rows[1].hidden, "the group is shut over it");
        assert!(matches!(
            act(Region::Remove, &rows, Some(id_of(2))),
            Some(Act::Doc(DocCommand::RemoveLayer(_)))
        ));
    }
}
