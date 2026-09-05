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
//! both — `carry_onto` and `release_to` — so a row's two buttons are a `Some` each
//! rather than a rule written here.

use stark_engine::ObservableState;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::{BlendMode, LayerId, Place};
use stark_ui::icons::Icon;
use stark_ui::layer_tree::{self, Row};
use wgpui::{
    App, Bounds, IntoElement, Pixels, Point, RenderOnce, Window, canvas, div, prelude::*, px, rgb,
};

/// The panel's width in logical px — wider than the brush's, because a row carries a
/// name, a depth indent and four controls.
pub const WIDTH: f32 = 268.0;

/// What a press on the layers panel landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The row's body: select this layer to paint on.
    Row(usize),
    /// Its eye.
    Visible(usize),
    /// Its fold triangle — only a group has one.
    Fold(usize),
    /// Put this layer into the group below it (§14.2).
    Carry(usize),
    /// Take it out of the group it is in.
    Release(usize),
    /// Clip it to what it sits on.
    Clip(usize),
    /// One of the acts on the whole stack.
    Add,
    Duplicate,
    Remove,
    /// The blend-mode cycle on the selected layer.
    Blend,
    /// The opacity track.
    Opacity,
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

/// How far along the opacity track a position is, `0..=1`.
pub fn opacity_at(regions: &Regions, at: Point<Pixels>) -> Option<f32> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == Region::Opacity)
        .map(|(_, b)| *b)?;
    let left = f32::from(bounds.origin.x);
    let width = f32::from(bounds.size.width);
    (width > 0.0).then(|| ((f32::from(at.x) - left) / width).clamp(0.0, 1.0))
}

/// The blend mode after `mode` in [`BlendMode::ALL`], wrapping.
///
/// A cycle rather than a picker, because a pop-out is its own design (§25.7) and the
/// list is short enough to walk. `same_mode`, not `==`, so a layer already on
/// Radiance at a `k` of its own is not skipped; `ALL` starts Radiance where the model
/// does ([`DRAGO_K`](stark_model::document::DRAGO_K)) and leaves dialling it to a
/// surface with a slider, which this panel is not yet.
pub fn next_blend(mode: BlendMode) -> BlendMode {
    let i = BlendMode::ALL
        .iter()
        .position(|m| m.same_mode(mode))
        .unwrap_or(0);
    BlendMode::ALL[(i + 1) % BlendMode::ALL.len()]
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
    region: Region,
    regions: Regions,
}

impl RenderOnce for Chip {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        div()
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
            .when(self.on, |el| el.bg(rgb(0x35496b)))
            .child(crate::icons::icon(
                self.glyph,
                if self.on { 0xe8eaed } else { 0x767b80 },
            ))
    }
}

/// Build the panel's element tree.
///
/// `rows` is `layer_tree::rows`' answer, unmodified: which rows exist and which are
/// folded away is the tree's, and drawing them is this module's.
pub fn layers_panel(
    obs: Option<&ObservableState>,
    rows: &[Row],
    regions: &Regions,
) -> impl IntoElement {
    regions.borrow_mut().clear();
    let active = obs.map(|o| o.active_layer);
    let selected = active.and_then(|id| rows.iter().find(|r| r.info.id == id));
    let opacity = selected.map_or(1.0, |r| r.info.opacity);
    let blend = selected.map_or(BlendMode::Normal, |r| r.info.blend);

    div()
        .flex()
        .flex_col()
        .w(px(WIDTH))
        .h_full()
        .p_3()
        .gap_2()
        .bg(rgb(0x1e2124))
        .border_l_1()
        .border_color(rgb(0x35393d))
        .text_color(rgb(0xe8eaed))
        .child(div().text_sm().text_color(rgb(0x9aa0a6)).child("Layers"))
        // The selected layer's two continuous knobs.
        .child(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .pt_2()
                .child(
                    div()
                        .relative()
                        .mt_1()
                        .px_2()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(0x2a2d31))
                        .text_xs()
                        .text_color(rgb(0xb0b4b8))
                        .child(probe(regions, Region::Blend))
                        .child(format!("Blend: {}", blend.label())),
                )
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .text_xs()
                        .text_color(rgb(0x9aa0a6))
                        .child("Opacity")
                        .child(format!("{opacity:.2}")),
                )
                .child(
                    div()
                        .relative()
                        .h(px(18.))
                        .w_full()
                        .rounded_sm()
                        .bg(rgb(0x2a2d31))
                        .child(probe(regions, Region::Opacity))
                        .child(
                            div()
                                .h_full()
                                .w(wgpui::relative(opacity.clamp(0.0, 1.0)))
                                .rounded_sm()
                                .bg(rgb(0x40474e)),
                        ),
                ),
        )
        // The acts on the whole stack, above the roster they act on.
        .child(
            div().flex().gap_1().children(
                // The catalog's own marks (`stark_ui::icons`), so the three
                // acts here and the three in the web app's header are one control
                // apiece rather than two that resemble each other. A stack gaining a
                // member, a copy of one, and the destructive one — which is what a
                // trash says everywhere.
                [
                    (Region::Add, stark_ui::icons::ADD_LAYER),
                    (Region::Duplicate, stark_ui::icons::DUPLICATE),
                    (Region::Remove, stark_ui::icons::REMOVE),
                ]
                .map(|(region, glyph)| Chip {
                    glyph,
                    on: false,
                    region,
                    regions: regions.clone(),
                }),
            ),
        )
        // The roster, top of the stack first: a layer list is read the way the
        // picture is, and the engine's roster is bottom-to-top.
        .child(
            div().flex().flex_col().gap_0p5().py_1().children(
                rows.iter()
                    .enumerate()
                    .rev()
                    .filter(|(_, r)| !r.hidden)
                    .map(|(i, row)| {
                        let worn = active == Some(row.info.id);
                        div()
                            .relative()
                            .flex()
                            .items_center()
                            .gap_1()
                            .pl(px(4.0 + row.info.depth as f32 * layer_tree::INDENT as f32))
                            .pr_1()
                            .py_0p5()
                            .rounded_sm()
                            .when_else(
                                worn,
                                |el| el.bg(rgb(0x35496b)),
                                |el| el.text_color(rgb(0xb0b4b8)),
                            )
                            .child(Chip {
                                glyph: if row.info.visible {
                                    stark_ui::icons::VISIBLE
                                } else {
                                    stark_ui::icons::HIDDEN
                                },
                                on: row.info.visible,
                                region: Region::Visible(i),
                                regions: regions.clone(),
                            })
                            // The fold slot is drawn whatever the row is, empty for
                            // a layer that carries nothing: a triangle only some rows
                            // have would push their *names* right, and a column of
                            // names that do not line up is what makes a tree
                            // unreadable — the indent would stop meaning depth.
                            .child(if row.info.is_group {
                                Chip {
                                    glyph: if row.collapsed {
                                        stark_ui::icons::FOLD_SHUT
                                    } else {
                                        stark_ui::icons::FOLD_OPEN
                                    },
                                    on: false,
                                    region: Region::Fold(i),
                                    regions: regions.clone(),
                                }
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
                                    .child(probe(regions, Region::Row(i)))
                                    .child(layer_tree::layer_label(&row.info)),
                            )
                            .when(row.info.clip, |el| {
                                el.child(crate::icons::icon(stark_ui::icons::CLIP, 0x9aa0a6))
                            })
                            // Carry and Release are a `Some` each rather than a rule
                            // written here — see the module note.
                            .when(row.carry_onto.is_some(), |el| {
                                el.child(Chip {
                                    glyph: stark_ui::icons::CARRY,
                                    on: false,
                                    region: Region::Carry(i),
                                    regions: regions.clone(),
                                })
                            })
                            .when(row.release_to.is_some(), |el| {
                                el.child(Chip {
                                    glyph: stark_ui::icons::RELEASE,
                                    on: false,
                                    region: Region::Release(i),
                                    regions: regions.clone(),
                                })
                            })
                            .child(Chip {
                                glyph: stark_ui::icons::CLIP,
                                on: row.info.clip,
                                region: Region::Clip(i),
                                regions: regions.clone(),
                            })
                    }),
            ),
        )
}

/// What a press on `region` means as a command, given the rows it was drawn over.
///
/// A function rather than a `match` in the view, so the mapping is testable — which
/// matters more here than it looks: every arm is a claim about §14's vocabulary, and
/// three of them (Carry, Release, Clip) are the tree's answer rather than this
/// module's.
pub fn act(region: Region, rows: &[Row], active: Option<LayerId>) -> Option<Act> {
    let row = |i: usize| rows.get(i);
    Some(match region {
        Region::Row(i) => Act::Peer(PeerCommand::SetActiveLayer(row(i)?.info.id)),
        Region::Visible(i) => {
            let info = &row(i)?.info;
            Act::Doc(DocCommand::SetLayerVisible(info.id, !info.visible))
        }
        Region::Fold(i) => Act::Fold(row(i)?.info.id),
        Region::Clip(i) => {
            let info = &row(i)?.info;
            Act::Doc(DocCommand::SetLayerClip(info.id, !info.clip))
        }
        // Into the group below: the tree already worked out which that is, and a row
        // with nothing under it in its own stack offers no button at all.
        Region::Carry(i) => {
            let r = row(i)?;
            Act::Doc(DocCommand::MoveLayer {
                id: r.info.id,
                carrier: Some(r.carry_onto?),
                at: Place::Top,
            })
        }
        // Out of the group it is in, and directly above it.
        Region::Release(i) => {
            let r = row(i)?;
            let (group, carrier) = r.release_to?;
            Act::Doc(DocCommand::MoveLayer {
                id: r.info.id,
                carrier,
                at: Place::Above(group),
            })
        }
        Region::Add => Act::Doc(DocCommand::AddLayer {
            carrier: None,
            above: active,
        }),
        Region::Duplicate => Act::Doc(DocCommand::DuplicateLayer(active?)),
        // The tree says whether a removal would leave a document behind (§14.2), so
        // the refusal is a property of the row rather than a count kept here.
        Region::Remove => {
            let id = active?;
            let removable = rows.iter().any(|r| r.info.id == id && r.removable);
            removable.then_some(Act::Doc(DocCommand::RemoveLayer(id)))?
        }
        Region::Blend => {
            let id = active?;
            let info = &rows.iter().find(|r| r.info.id == id)?.info;
            Act::Doc(DocCommand::SetLayerBlend(id, next_blend(info.blend)))
        }
        // A drag, not a click: the caller reads the fraction and previews.
        Region::Opacity => return None,
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
    use stark_model::document::{ActionId, ActorId, DRAGO_K};
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
            blend: BlendMode::Normal,
            clip: false,
            opacity: 1.0,
            visible: true,
            carrier: None,
            depth,
            is_group,
            has_backdrop: false,
            name: None,
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

    /// A stack of two: a group at the foot, a layer above it.
    fn stack() -> Vec<Row> {
        let layers = vec![info(1, 0, true), info(2, 0, false)];
        layer_tree::rows(&layers, &HashSet::new())
    }

    /// The blend cycle visits every mode and comes back — a cycle with a hole in it
    /// would leave one mode unreachable from the panel.
    #[test]
    fn the_blend_cycle_closes() {
        let mut m = BlendMode::Normal;
        let mut seen = vec![m.label()];
        for _ in 1..BlendMode::ALL.len() {
            m = next_blend(m);
            seen.push(m.label());
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            BlendMode::ALL.len(),
            "every mode is on the cycle"
        );
        assert!(
            next_blend(m).same_mode(BlendMode::Normal),
            "and the cycle returns"
        );

        // A layer already on Radiance at its own `k` is still on Radiance — the
        // cycle must not skip a mode, nor land elsewhere, because `k` was dialled.
        let dialled = BlendMode::Drago { k: 3.0 };
        assert_eq!(dialled.label(), "Radiance");
        assert!(next_blend(dialled).same_mode(next_blend(BlendMode::Drago { k: DRAGO_K })));
    }

    /// The eye and the clip mark toggle what they show, rather than setting a fixed
    /// value — a control that always sent `true` would be dead the second time.
    #[test]
    fn the_row_toggles_read_the_row() {
        let rows = stack();
        let visible = rows[1].info.visible;
        match act(Region::Visible(1), &rows, None) {
            Some(Act::Doc(DocCommand::SetLayerVisible(_, to))) => assert_eq!(to, !visible),
            _ => panic!("the eye sets visibility"),
        }
        match act(Region::Clip(1), &rows, None) {
            Some(Act::Doc(DocCommand::SetLayerClip(_, to))) => assert!(to),
            _ => panic!("the clip mark sets clipping"),
        }
    }

    /// Selecting a row is **presence**, not a document edit: two collaborators paint
    /// on different layers of one document (§17.4).
    #[test]
    fn choosing_a_row_is_presence_rather_than_an_edit() {
        let rows = stack();
        assert!(matches!(
            act(Region::Row(0), &rows, None),
            Some(Act::Peer(PeerCommand::SetActiveLayer(_)))
        ));
    }

    /// Carry puts a layer into the group the *tree* named, rather than into whatever
    /// this module would have guessed at.
    #[test]
    fn carry_uses_the_row_the_tree_named() {
        let rows = stack();
        let Some(onto) = rows[1].carry_onto else {
            panic!("a layer over a group can be carried onto it");
        };
        match act(Region::Carry(1), &rows, None) {
            Some(Act::Doc(DocCommand::MoveLayer { carrier, at, .. })) => {
                assert_eq!(carrier, Some(onto));
                assert_eq!(at, Place::Top);
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
        assert!(act(Region::Blend, &rows, None).is_none());
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
}
