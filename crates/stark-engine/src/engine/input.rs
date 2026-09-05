//! Commands into state (§4): the four arms of [`Engine::process`], and what an arm
//! reads off the committed document before it commits — the frame a canvas-space
//! payload is converted into, the subtree a translate expands to, the name funnel.
//!
//! An arm commits through `commit`'s doors or previews through `live`'s, and decides
//! nothing about either.

use super::Engine;
use crate::command::{DocCommand, GestureCommand, PeerCommand, Tool, ViewCommand};
use crate::session::ShapeResult;
use stark_model::document::{
    ActionId, ActionKind, BlendMode, Filter, GuideId, LayerId, Parcel, PerspectiveGuide,
};
use stark_model::geom::{IVec2, Vec2};
use stark_model::{Srgb, SubstrateScale};

/// Longest name that will be recorded, in `char`s — the wire's bound, reached for
/// rather than restated.
///
/// The argument is one argument: a name travels, so it is bounded, and nothing about
/// a text field stops a paste from being a megabyte. It is stated where the *wire*
/// can also reach it ([`stark_model::MAX_NAME`]), because a presence frame's
/// name is capped by the same number and the model cannot depend on this crate (§2).
/// Two constants agreeing at 64 would be two things to keep level.
use stark_model::MAX_NAME;

/// The name to record, given what a frontend collected: surrounding whitespace
/// trimmed, length capped, and anything that comes out empty treated as *no name*
/// rather than as a name that happens to be blank.
///
/// One funnel for every source — the panel's field, a script, a peer's command —
/// so "a name is either absent or something you can read" is a property of the
/// model rather than a habit of the UI. The logged action carries the result, so
/// replay reproduces it without re-running these rules.
///
/// Shared by layers and drawing guides: the two are named through different commands
/// — one logged, one view state — and the rule for what a name *is* should not be a
/// property of which command carried it.
///
/// Generic at both ends for that reason too, and only for that reason: the two
/// callers hold their names differently — a logged action carries a `String`,
/// because that is what goes on the wire, while a guide holds an `Arc<str>`, because
/// its list is re-projected at pointer rate — and neither difference is about what a
/// name is. `String: From<String>` is the identity, so the logged path still moves
/// its bytes rather than copying them.
fn normalize_name<T: From<String>>(name: Option<impl AsRef<str>>) -> Option<T> {
    let trimmed = name?;
    let capped: String = trimmed.as_ref().trim().chars().take(MAX_NAME).collect();
    (!capped.is_empty()).then(|| T::from(capped))
}

/// The payload of a setter: a document command whose drag previews by folding the
/// very action its release commits (§21.6). One variant per `DocCommand` /
/// `ViewCommand::Preview*` pair, and both mint their kind through
/// [`Engine::setter_kind`].
enum Setter {
    LayerBlend(LayerId, BlendMode),
    LayerOpacity(LayerId, f32),
    SelectionOpacity(f32),
    Filter(LayerId, Filter),
    MatteRect(LayerId, Vec2, Vec2),
    MattePaint(LayerId, Parcel),
    SubstrateColor(Srgb),
    SubstrateScale(SubstrateScale),
    Translate(LayerId, IVec2),
    Guide(GuideId, PerspectiveGuide),
}

/// Whether a captured pointer report opens a stroke or continues one — see
/// [`Engine::note_debug_sample`]. A named pair rather than a `bool`, because a
/// bare `true` at the call site says nothing about which way round it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Capture {
    /// The press: forget the last stroke's samples first.
    Restart,
    /// A move within the stroke in hand.
    Continue,
}

impl Engine {
    /// The canvas offset of a layer's frame (§14.12) — zero for a layer that is
    /// not there, which every action aimed at one is refused as anyway. Off the
    /// **committed** document, which is what a mint reads and what every preview
    /// entry point reads too, so the frame a gesture is converted into is the
    /// frame its commit will carry.
    pub(crate) fn frame_of(&self, layer: LayerId) -> stark_model::geom::IVec2 {
        self.timeline
            .current()
            .layer(layer)
            .map_or(stark_model::geom::IVec2::ZERO, |l| l.translation)
    }

    /// The whole move a translate gesture means (§14.12): `layer`'s subtree —
    /// a group moves as one, and translation does not inherit — with each
    /// member that has a frame to move
    /// ([`Layer::is_translatable`](crate::document::Layer::is_translatable): paint and
    /// mattes) displaced by the same delta. Filters are left out rather than
    /// named-and-refused: a move naming one would answer "not a no-op" forever,
    /// and a drag out and back would log a step that does nothing. Read off the
    /// committed document, like every mint. An absent layer expands to no moves,
    /// which `is_noop_on` then declines to log.
    fn translate_moves(
        &self,
        layer: LayerId,
        to: stark_model::geom::IVec2,
    ) -> Vec<(LayerId, stark_model::geom::IVec2)> {
        use stark_model::document::FRAME_LIMIT;
        // The command's `to` has not been through the funnel yet, and this
        // arithmetic runs before the commit that would clamp it — so hold it
        // here, where the subtraction below would otherwise be the first thing
        // an unbounded value reaches.
        let to = to.clamp(
            stark_model::geom::IVec2::splat(-FRAME_LIMIT),
            stark_model::geom::IVec2::splat(FRAME_LIMIT),
        );
        let doc = self.timeline.current();
        let delta = to - self.frame_of(layer);
        doc.subtree_ids(layer)
            .unwrap_or_default()
            .into_iter()
            .filter(|id| doc.layer(*id).is_some_and(|l| l.is_translatable()))
            .map(|id| (id, self.frame_of(id) + delta))
            .collect()
    }

    /// The action a setter commits — and the very one its preview folds (§21.6).
    ///
    /// **One function, so `preview == committed` (§1.3) is a property of the code
    /// rather than of a setter's two arms agreeing.** Whatever a kind needs beyond its payload
    /// is read here, once, off the committed document: the canvas-to-frame conversion
    /// a matte's rect and paint carry (§14.12), and the subtree a translate expands to.
    fn setter_kind(&self, setter: Setter) -> ActionKind {
        match setter {
            Setter::LayerBlend(id, blend) => ActionKind::SetLayerBlend(id, blend),
            Setter::LayerOpacity(id, opacity) => ActionKind::SetLayerOpacity(id, opacity),
            Setter::SelectionOpacity(opacity) => ActionKind::SetSelectionOpacity(opacity),
            Setter::Filter(id, filter) => ActionKind::SetFilter(id, filter),
            Setter::MatteRect(id, min, max) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMatteRect(id, min - d, max - d)
            }
            Setter::MattePaint(id, paint) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMattePaint(id, paint.translated(-d))
            }
            Setter::SubstrateColor(rgb) => ActionKind::SetSubstrateColor(rgb),
            Setter::SubstrateScale(scale) => ActionKind::SetSubstrateScale(scale),
            Setter::Translate(layer, to) => ActionKind::TranslateLayers {
                moves: self.translate_moves(layer, to),
            },
            Setter::Guide(id, guide) => ActionKind::SetGuide(id, guide),
        }
    }

    /// The press-drag-release lifecycle. One path for both kinds of tool
    /// (§6.8): the selection tools build an op where the brush builds a
    /// stroke, and both preview through the same `preview` DocState.
    pub(super) fn process_gesture(&mut self, command: GestureCommand) {
        match command {
            GestureCommand::Start {
                tool,
                sample,
                tolerance,
                rope,
            } => {
                if tool.is_selection() {
                    // A marquee or lasso fits no curve, so it has no use for the
                    // tolerance (or the rope); its own decimation is a mask-cost
                    // knob (§6.8).
                    //
                    // What it does need is whether there is a mask to combine with,
                    // which only this side holds: an Add drawn over nothing is a New
                    // (`session::against_selection`). Off the committed document and
                    // read at the press, so the gesture's meaning is fixed before it
                    // has drawn anything.
                    let has_selection = self.document().has_selection(self.actor());
                    let frame = self.frame_of(self.session.active_layer());
                    self.session
                        .start_selection(tool, sample.pos, has_selection, frame);
                } else {
                    let seed = self.authoring.clock;
                    let frame = self.frame_of(self.session.active_layer());
                    self.session
                        .start_stroke(tool, sample, seed, tolerance, rope, frame);
                    self.note_debug_sample(Capture::Restart, sample);
                }
                self.mark_live_stale();
            }
            GestureCommand::To { sample } => {
                // The CPU half of a pointer sample, and the *whole* of what arrives at
                // input rate — the fold and the render it marks stale are paid once a
                // frame instead (`mark_live_stale`). Worth its own row because it is
                // the one phase that grows with stroke length rather than with the
                // tail: the fitter re-solves its unfrozen prefix on every push, and
                // that has measured ~350× the flattening beside it.
                crate::timing::span!("input.fit");
                if self.session.is_selecting() {
                    self.session.selection_to(sample.pos);
                } else {
                    self.session.stroke_to(sample);
                    self.note_debug_sample(Capture::Continue, sample);
                }
                self.mark_live_stale();
            }
            // A held pointer: snap the stroke to the shape it resembles (§6.9). Nothing
            // is committed and nothing is decided about the gesture's end — a snap
            // changes what the *same* drag builds, and the release still commits one
            // stroke either way.
            GestureCommand::Hold => {
                // Built here rather than inside the session, because the two halves
                // of a guide live in two places now (§20.5) and this is the only
                // side holding both. Off the **committed** document, which is what
                // a stroke is drawn over — a guide being dragged elsewhere in the
                // same instant previews without moving what a snap aims at, and a
                // snap that changed underfoot mid-gesture is the surprise §20.5
                // rules out.
                let scaffold = self.scaffold(self.timeline.current());
                if self.session.assist_stroke(&scaffold) {
                    self.mark_live_stale();
                }
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                if self.session.is_selecting() {
                    // One gesture, two things it can commit — which one was decided
                    // when the drag started (§18.0.4).
                    match self.session.end_shape() {
                        Some(ShapeResult::Select(op)) => self.commit(ActionKind::Select(op)),
                        // The layer the drag pinned at the press, not the active
                        // layer now: the op was converted into *that* layer's
                        // frame, and the two must not part (`ShapeResult::Fill`).
                        Some(ShapeResult::Fill {
                            layer,
                            op,
                            translation: frame,
                        }) => self.commit(ActionKind::Fill {
                            layer,
                            op,
                            translation: frame,
                        }),
                        None => {}
                    }
                } else {
                    // Fold first, so what is offered to the commit is the stroke as
                    // it stands at the release — the frame that would have shown the
                    // last few samples, drawn now instead of never. A fold costs the
                    // live tail; the render it saves the commit costs the stroke.
                    self.flush_live();
                    if let Some(rec) = self.session.end_stroke() {
                        self.log_debug_samples();
                        self.commit_stroke(rec);
                    }
                }
                self.mark_live_stale();
            }
            GestureCommand::Cancel => {
                self.session.cancel_stroke();
                self.mark_live_stale();
            }
        }
    }

    /// Per-client state that is published rather than logged (§17.7).
    /// Nothing here enters the history or the save file; it rides the presence
    /// channel so collaborators can see where this client is working.
    pub(super) fn process_peer(&mut self, command: PeerCommand) {
        match command {
            // Any existing layer, including a matte. `active_layer` is *the
            // selected layer*, not "a paint target" — a frame is selected the same
            // way a paint layer is, which is what lets the frontend have one
            // selection concept instead of two (§15.7). A stroke aimed
            // at a matte then simply draws nothing, refused identically by `apply`
            // and by the preview path.
            PeerCommand::SetActiveLayer(id) => {
                if self.session.set_active_layer(id, self.timeline.current()) {
                    // The hover mark follows the brush's target (§18.1.10): it
                    // is built against the active layer at fold time, so moving
                    // the selection has to re-lay it there. Free when nothing is
                    // in flight — a clean fold's rebuild is an early return.
                    self.mark_live_stale();
                }
            }
            PeerCommand::SetCursor(pos) => self.session.set_cursor(pos),
            PeerCommand::SetName(name) => self.session.set_name(name),
        }
    }

    /// Document-state mutations: every arm here either commits an action or
    /// navigates the history that holds them.
    pub(super) fn process_doc(&mut self, command: DocCommand) {
        self.process_doc_inner(command);
        // Every arm changes the document the in-flight previews are drawn over, so
        // the fold is rebuilt once, here, rather than at each of a dozen call sites.
        // Cheap when nothing is in flight (there is nothing to fold) and correct when
        // a peer is mid-stroke while this client edits.
        self.mark_live_stale();
    }

    fn process_doc_inner(&mut self, command: DocCommand) {
        match command {
            // Shared sessions log undo as an action peers can order (§5.4, §12.3);
            // solo falls back to navigation. Redo is an `Undo` of an `Undo`, which
            // is why the two differ only in which pair of timeline methods they
            // name — see [`Self::navigate`].
            DocCommand::Undo => {
                self.navigate(|t| t.undo_as_action(), |t, ctx| t.undo(ctx));
            }
            DocCommand::Redo => {
                self.navigate(|t| t.redo_as_action(), |t, ctx| t.redo(ctx));
            }
            DocCommand::Seek(to) => {
                self.preview.set_doc(None);
                if self.timeline.seek(to, &mut self.shared.apply) {
                    // A scrub crosses layer additions wholesale — dragging to the
                    // start of the log withdraws every one of them — so the selected
                    // layer routinely stops existing here. `committed_changed`
                    // repoints the brush for every such cause at once (§17.9).
                    self.committed_changed();
                    self.apply_document_substrate();
                }
            }
            DocCommand::Select(op) => self.commit(ActionKind::Select(op)),
            DocCommand::InvertSelection => self.commit(ActionKind::InvertSelection),
            DocCommand::SetSelectionOpacity(opacity) => {
                self.commit(self.setter_kind(Setter::SelectionOpacity(opacity)))
            }
            DocCommand::Fill { layer, op } => {
                // The command's op is on the canvas, where every gesture is; the
                // action's is in the layer's frame — the same pair of reads
                // `preview_fill` makes, so preview == committed (§14.12).
                let frame = self.frame_of(layer);
                self.commit(ActionKind::Fill {
                    layer,
                    op: op.translated(-frame.as_vec2()),
                    translation: frame,
                });
            }
            DocCommand::Transform { layer, map } => {
                // A degenerate or non-finite map would be rejected by `apply`
                // anyway (deterministically — §16.1); refusing it
                // here as well keeps a knowably-dead action out of the log.
                // Each family goes to its own action kind — the wire format
                // never carries the routing enum, only the map it named.
                if map.usable() {
                    use stark_model::document::TransformMap;
                    // The map stays stated on the canvas; the frame rides beside
                    // it and `apply` conjugates (§14.12) — the same value
                    // `preview_transform` reads.
                    let frame = self.frame_of(layer);
                    self.commit(match map {
                        TransformMap::Affine(affine) => ActionKind::Transform {
                            layer,
                            affine,
                            translation: frame,
                        },
                        TransformMap::Perspective(map) => ActionKind::TransformPerspective {
                            layer,
                            map,
                            translation: frame,
                        },
                        TransformMap::Warp(map) => ActionKind::TransformWarp {
                            layer,
                            map,
                            translation: frame,
                        },
                    });
                } else {
                    // Nothing is logged, but the gesture's preview still has to be
                    // superseded — `commit`'s bargain, made by hand because the
                    // refusal is about the map rather than about the document.
                    self.preview.set_doc(None);
                }
            }
            DocCommand::TranslateLayer { layer, to } => {
                self.commit(self.setter_kind(Setter::Translate(layer, to)))
            }
            DocCommand::FloatSelection { layer } => {
                // Asked before an action is spent, exactly as `MergeLayerDown`
                // asks its plan (§16.12): the same refusals `apply` makes, off the
                // same committed state, so a command that would no-op logs nothing.
                let frame = self.frame_of(layer);
                let doc = self.timeline.current();
                let offered = doc.layer(layer).and_then(|l| l.tiles()).is_some_and(|t| {
                    let selection = doc.selection_of(self.actor());
                    !selection.is_universal()
                        && crate::document::transform::plan_float(t, &selection, frame).is_some()
                });
                if offered {
                    let action = self.commit_minting(|a| ActionKind::FloatSelection {
                        layer,
                        child: LayerId::new(a, 0),
                        translation: frame,
                    });
                    // The float is what the hand is about to move — and it is
                    // paint, so the next stroke has somewhere to go (`AddLayer`'s
                    // reason).
                    self.arm_active(LayerId::new(action, 0));
                }
            }
            DocCommand::SetSubstrate(id) => {
                self.commit(ActionKind::SetSubstrate(id));
                // Unconditional, and a no-op when the substrate did not move: the
                // registry is brought level with the document rather than with what
                // this command asked for.
                self.apply_document_substrate();
            }
            DocCommand::SetSubstrateScale(scale) => {
                self.commit(self.setter_kind(Setter::SubstrateScale(scale)));
                // The same call for the same reason, and it is the same *state*: a
                // `SubstrateMap` is built from the substrate and its scale together, so laying
                // the substrate larger invalidates the bound substrate exactly as switching
                // it does (`gpu::substrate::Substrate`).
                self.apply_document_substrate();
            }
            DocCommand::AddLayer { carrier, above } => {
                // A freshly added layer becomes the active painting target — but only
                // if it landed and can take a stroke, which is `arm_active`'s whole
                // question.
                let action = self.commit_minting(|a| ActionKind::AddLayer {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                });
                self.arm_active(LayerId::new(action, 0));
            }
            DocCommand::PlaceImage {
                carrier,
                above,
                at,
                name,
                image,
            } => {
                // The active layer, exactly as an `AddLayer` is and for its reason:
                // it is paint, so the next stroke has somewhere to go.
                let action = self.commit_minting(|a| ActionKind::PlaceImage {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    at,
                    name,
                    image,
                });
                self.arm_active(LayerId::new(action, 0));
            }
            DocCommand::AddMatte {
                carrier,
                at,
                region,
                paint,
            } => {
                self.commit_minting(|a| ActionKind::AddMatte {
                    id: LayerId::new(a, 0),
                    carrier,
                    at,
                    region,
                    paint,
                });
                // Deliberately *not* made the active layer, unlike `AddLayer`: a
                // matte has no tile map, so painting on it is refused
                // (§15.7) and arming it as the target would just
                // swallow the user's next stroke.
            }
            DocCommand::AddFilter {
                carrier,
                above,
                filter,
            } => {
                self.commit_minting(|a| ActionKind::AddFilter {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    filter,
                });
                // Deliberately *not* made the active layer, for the reason
                // `AddMatte` is not: a filter has no tile map, so arming it as the
                // paint target would swallow the next stroke (§21.4). The frontend
                // selects it, which is what raises its bar.
            }
            DocCommand::SetFilter(id, filter) => {
                self.commit(self.setter_kind(Setter::Filter(id, filter)))
            }
            DocCommand::SetMatteRect(id, min, max) => {
                self.commit(self.setter_kind(Setter::MatteRect(id, min, max)))
            }
            DocCommand::SetMattePaint(id, paint) => {
                self.commit(self.setter_kind(Setter::MattePaint(id, paint)))
            }
            DocCommand::SetSubstrateColor(rgb) => {
                self.commit(self.setter_kind(Setter::SubstrateColor(rgb)))
            }
            DocCommand::DuplicateLayer(source) => {
                // One minted id per layer of the subtree, paired with the layer it
                // copies, in composite order — the map the action carries (§14.8).
                // The copies are this action's own ids at `k = 0..n`, so the map is
                // only a list of *sources* wearing its positions; it is still written
                // as pairs because that is the shape `apply` reads and the shape the
                // footprint claims a `Layer(src)` from.
                //
                // Through the document's own walk, not a second one here: `apply`
                // declines the action unless `ids` names exactly the subtree
                // `duplicate_layer` walks, so a copy of the traversal in the engine is
                // two walks that must agree — on this client and on every peer.
                if let Some(sources) = self.document().subtree_ids(source) {
                    let action = self.commit_minting(|a| ActionKind::DuplicateLayer {
                        ids: sources
                            .iter()
                            .enumerate()
                            .map(|(k, &src)| (src, LayerId::new(a, k as u32)))
                            .collect(),
                    });
                    // The copy is what you go on to work on.
                    self.arm_active(LayerId::new(action, 0));
                }
            }
            // The subtree travels in the action, read off the document the command
            // was aimed at (§12.6) — see `ActionKind::RemoveLayer`. A layer that is
            // not there mints an empty list and the fold declines it, which is what
            // every other action naming an absent layer does.
            DocCommand::RemoveLayer(id) => {
                let carried = self.document().carried_ids(id).unwrap_or_default();
                self.commit(ActionKind::RemoveLayer { id, carried })
            }
            DocCommand::MergeLayerDown(id) => {
                // Asked here rather than only inside `apply`, so a merge that cannot
                // preserve the document's appearance never reaches the log at all —
                // the same argument `Transform` makes about a degenerate map. `apply`
                // asks again anyway, because a peer's action arrives without passing
                // through here (§14.11).
                if let Some(plan) = crate::document::merge::plan(self.document(), id) {
                    // The frame bake's own refusal, asked here for the reason the
                    // plan is (§14.12.3): a source too large to restate in the
                    // destination's frame is declined by `apply`, and an offer
                    // that outran that would log a dead action and still repoint
                    // the brush below as if it had worked.
                    let shift = self.frame_of(plan.source) - self.frame_of(plan.dest);
                    if shift != stark_model::geom::IVec2::ZERO {
                        let bakeable = self
                            .document()
                            .layer(plan.source)
                            .and_then(|l| l.tiles())
                            .is_none_or(|tiles| {
                                crate::document::transform::plan_paint(
                                    tiles,
                                    &crate::document::selection::Selection::everything(),
                                    stark_model::geom::Affine2::from_translation(shift.as_vec2()),
                                )
                                .is_some()
                            });
                        if !bakeable {
                            return;
                        }
                    }
                    // Read **before** the commit, which is what makes it answerable:
                    // the commit repoints the brush off the layer it is about to fold
                    // away (§17.9), so afterwards there is nothing left to compare.
                    let follow = self.session.active_layer() == id;
                    self.commit(ActionKind::MergeLayerDown {
                        source: plan.source,
                        dest: plan.dest,
                    });
                    // The merged layer is where the work now is, so the brush follows
                    // it. The repoint has already put it somewhere that exists; this
                    // says *which* somewhere, because picking the nearest paintable
                    // layer is not the same as picking the paint that just absorbed
                    // what you were working on.
                    if follow {
                        self.arm_active(plan.dest);
                    }
                }
            }
            DocCommand::SetLayerBlend(id, blend) => {
                self.commit(self.setter_kind(Setter::LayerBlend(id, blend)))
            }
            DocCommand::SetLayerClip(id, clip) => self.commit(ActionKind::SetLayerClip(id, clip)),
            DocCommand::SetLayerOpacity(id, opacity) => {
                self.commit(self.setter_kind(Setter::LayerOpacity(id, opacity)))
            }
            DocCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            DocCommand::SetLayerName(id, name) => {
                self.commit(ActionKind::SetLayerName(id, normalize_name(name)))
            }
            DocCommand::MoveLayer { id, carrier, at } => {
                self.commit(ActionKind::MoveLayer { id, carrier, at })
            }

            // The drawing guides (§20.5). A guide's identity is the id of the action
            // that adds it, minted through the same door a layer's is — so there is
            // no counter here and nothing for `resync_counters` to resume past
            // (`GuideId`, §17.9).
            DocCommand::AddGuide { guide, after, name } => {
                self.commit_minting(|a| ActionKind::AddGuide {
                    id: GuideId(a),
                    guide,
                    after,
                    name: normalize_name(name),
                });
            }
            DocCommand::RemoveGuide(id) => self.commit(ActionKind::RemoveGuide(id)),
            DocCommand::SetGuide(id, guide) => {
                self.commit(self.setter_kind(Setter::Guide(id, guide)))
            }
            DocCommand::SetGuideName(id, name) => {
                self.commit(ActionKind::SetGuideName(id, normalize_name(name)))
            }
            DocCommand::MoveGuide { id, after } => self.commit(ActionKind::MoveGuide { id, after }),
        }
    }

    /// Show the document committing `setter` would leave behind, without logging it —
    /// the body every `Preview*` setter arm below shares (§21.6).
    ///
    /// `None` clears the preview, which is what the release of a drag that changed
    /// nothing sends. The sanitize and the fold both happen inside
    /// [`crate::document::apply::preview_of`], which is the point: an arm cannot forget a
    /// step it does not perform. See that function for the two arms that had.
    fn preview_setter(&mut self, setter: Option<Setter>) {
        let kind = setter.map(|s| self.setter_kind(s));
        let actor = self.actor();
        let preview = kind.map(|kind| {
            crate::document::apply::preview_of(
                kind,
                self.timeline.current(),
                actor,
                &mut self.shared.apply,
            )
        });
        self.set_doc_preview(preview);
    }

    /// View-state mutations: nothing here is logged, replicated, or reachable by
    /// undo.
    pub(super) fn process_view(&mut self, command: ViewCommand) {
        match command {
            ViewCommand::SetTool(tool) => {
                // Switching away mid-gesture abandons it rather than committing a
                // half-dragged marquee.
                self.session.set_tool(tool);
                self.mark_live_stale();
            }
            ViewCommand::SetBrush { brush, color } => {
                // Held here for the reason `PeerFrame::sanitized` holds a peer's:
                // a committed stroke's brush is held by `ActionKind::sanitized`,
                // and a live one is drawn by the same renderer without ever
                // becoming an action, so nothing else would. `preview ==
                // committed` needs both doors (§6.2).
                self.session.set_brush(brush);
                self.session.set_color(color);
                self.mark_live_stale();
            }
            // Grab-and-drag: content follows the cursor, so the view center moves
            // opposite by the drag delta, carried into canvas units — through the
            // whole map, since a turned or mirrored canvas sends a screen-space drag
            // somewhere else entirely. Every arm here names a mutator rather than
            // writing a view field, so a command carrying a non-finite number is
            // refused by the view rather than stored (see [`ViewTransform`]).
            ViewCommand::Pan { delta } => self.session.view.pan_by(delta),
            ViewCommand::SetRotation(radians) => self.session.view.set_rotation(radians),
            ViewCommand::MirrorH => self.session.view.mirror_screen_h(),
            ViewCommand::CenterOn(point) => self.session.view.center_on(point),
            ViewCommand::ShowPiece(frame) => self.show_piece(frame),
            ViewCommand::Zoom { anchor, factor } => {
                self.session.view.zoom_about(anchor, factor);
            }
            ViewCommand::Pinch {
                anchor,
                to,
                scale,
                turn,
            } => self.session.view.pinch(anchor, to, scale, turn),
            ViewCommand::Resize(viewport) => self.session.view.resize(viewport),
            ViewCommand::SetShapeAction(action) => self.session.shape_action = action,
            ViewCommand::SetSelectionFeather(feather) => {
                self.session.set_selection_feather(feather);
            }
            ViewCommand::SetShapeOpacity(opacity) => self.session.set_shape_opacity(opacity),
            ViewCommand::SetShowPeerSelections(show) => self.session.show_peer_selections = show,
            ViewCommand::SetGuideVisible(id, visible) => {
                // The eye is the one per-client thing about a guide (§20.5), so it
                // moves the session and never the document. The bump is what a
                // frontend's memo on the roster watches: nothing in `doc_revision`
                // moves when an eye does, and without saying so the panel would
                // keep showing the eye it drew last time.
                if self.session.set_guide_visible(id, visible) {
                    self.guide_epoch.bump();
                    self.mark_live_stale();
                }
            }
            ViewCommand::PreviewGuide(drag) => {
                self.preview_setter(drag.map(|(id, guide)| Setter::Guide(id, guide)));
            }
            ViewCommand::PreviewMatteRect(drag) => {
                self.preview_setter(drag.map(|(id, min, max)| Setter::MatteRect(id, min, max)));
            }
            ViewCommand::PreviewSubstrateColor(rgb) => {
                self.preview_setter(rgb.map(Setter::SubstrateColor));
            }
            // The preview moves the *document* the compositor reads, and stops there:
            // no `apply_document_substrate`, so nothing is baked while the hand is on
            // the slider. What that costs: a preview shows the scale in the
            // **light**, since the media pass re-reads the substrate every frame off
            // one uniform, and not in the **tooth**, whose substrate is a stored bake.
            // Paint already down looks right immediately;
            // what the next stroke will bite is right from the commit.
            ViewCommand::PreviewSubstrateScale(scale) => {
                self.preview_setter(scale.map(Setter::SubstrateScale));
            }
            ViewCommand::PreviewParcel(pick) => {
                self.preview_setter(pick.map(|(id, paint)| Setter::MattePaint(id, paint)));
            }
            ViewCommand::PreviewSelectionOpacity(opacity) => {
                self.preview_setter(opacity.map(Setter::SelectionOpacity));
            }
            ViewCommand::PreviewLayerOpacity(set) => {
                self.preview_setter(set.map(|(id, opacity)| Setter::LayerOpacity(id, opacity)));
            }
            ViewCommand::PreviewFilter(set) => {
                self.preview_setter(set.map(|(id, filter)| Setter::Filter(id, filter)));
            }
            ViewCommand::PreviewLayerBlend(set) => {
                self.preview_setter(set.map(|(id, blend)| Setter::LayerBlend(id, blend)));
            }
            ViewCommand::PreviewTransform(t) => {
                let preview = t.and_then(|(layer, map)| self.preview_transform(layer, &map));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewTranslate(set) => {
                self.preview_setter(set.map(|(layer, to)| Setter::Translate(layer, to)));
            }
            ViewCommand::PreviewFill(f) => {
                let preview = f.and_then(|(layer, op)| self.preview_fill(layer, &op));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewHover(report) => match report {
                Some(r) => {
                    // The CPU half of a hover report — the window refit — on its
                    // own row, so the cost of following a resting pointer is
                    // never folded into what painting costs (`input.fit`, §7.1).
                    crate::timing::span!("input.hover");
                    // A report the window declined — sub-tolerance drift under a
                    // resting pen — refolds nothing.
                    if self.session.hover_to(r.sample, r.tolerance, r.reach) {
                        self.mark_live_stale();
                    }
                }
                None => {
                    if self.session.clear_hover() {
                        self.mark_live_stale();
                    }
                }
            },
            ViewCommand::SetMediaParams(params) => self.compositor_pipeline.set_media(params),
            ViewCommand::SetOutput(output) => self.compositor_pipeline.set_output(output),
            ViewCommand::SetEnvironment(id) => self.set_environment(id),
            ViewCommand::SetHistoryBudget(bytes) => self.history_budget = bytes,
            ViewCommand::SetFastCommit(on) => self.fast_commit = on,
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, without the per-sample staleness marks. Interactive samples go
    /// through `GestureCommand::To`, whose marks a frame's `flush_live` services
    /// by rendering the in-flight tail — right for drawing (the user must see
    /// each frame's moves), pointless across a replay where nothing is presented
    /// in between. This renders the stroke exactly once, at commit. Used by the
    /// brush editor's test-stroke replay.
    /// Answers the id of the action it committed, or `None` where the samples held no
    /// stroke — empty, or a hand that never left the first point.
    pub fn replay_stroke(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
    ) -> Option<ActionId> {
        self.replay_stroke_seeded(tool, samples, self.authoring.clock, 0.0)
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the
    /// Lamport clock. Replaying the same samples repeatedly advances the clock
    /// (each replay commits), so the seed — and with it the color dynamics and
    /// dither — changes on every replay. A caller re-rendering *one* stroke to
    /// show the effect of a brush change (the brush editor's preview) wants the
    /// jitter held fixed, so only the edited parameter moves.
    /// `rope` is the §6.11 smoothing string, and it is a parameter here — where
    /// [`Engine::replay_stroke`] pins it to zero — because the brush editor's
    /// preview replays a *recorded hand* (the user's own test stroke) and has to
    /// show what the smoothing slider beside it would do to that hand.
    /// **Answers what it committed**, which is §4's requirement of anything that
    /// mutates and is not a command: this is a batch of inputs ending in a logged,
    /// replicated action, so a caller has to be able to tell a committed stroke from a
    /// refused one. `None` is "these samples held no stroke": none at all, or a hand
    /// that never left its first point.
    ///
    /// Not routed through `GestureCommand` instead, deliberately. The command tier's
    /// payloads are values a frontend builds per event; this takes a borrowed slice a
    /// bench replays in a loop, and making it a command would mean an `Arc<[_]>` per
    /// call to say the same thing. Answering is what §4 actually asks for.
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
        seed: u64,
        rope: f32,
    ) -> Option<ActionId> {
        let mut it = samples.iter();
        let first = it.next()?;
        // Replayed samples are already in canvas space and came from a fit or from a
        // generator, not from a device, so there is no device tolerance to declare.
        let frame = self.frame_of(self.session.active_layer());
        self.session.start_stroke(
            tool,
            *first,
            seed,
            crate::path::DEFAULT_TOLERANCE,
            rope,
            frame,
        );
        for s in it {
            self.session.stroke_to(*s);
        }
        let committed = self.session.end_stroke().map(|rec| {
            let id = self.next_action_id();
            self.commit_with_id(id, ActionKind::CommitStroke(rec));
            id
        });
        self.mark_live_stale();
        committed
    }

    /// Keep a raw pointer report of the stroke in hand, so a misfit seen in the app
    /// can be dumped on release and replayed as a test
    /// ([`log_debug_samples`](Self::log_debug_samples)).
    ///
    /// A diagnostic, so a shipping build carries neither the samples nor the field
    /// that would hold them: this is `#[cfg]`, not a runtime `cfg!` around a `Vec`
    /// that exists either way. Keeping the capture behind a *call* rather than an
    /// `#[cfg]` block at each site keeps the gesture arms readable, and stops the two
    /// of them disagreeing about the gate — one gated and one not means a shipping
    /// build accumulates the first sample of every stroke and drops the rest.
    #[cfg(feature = "debug-unfrozen")]
    fn note_debug_sample(&mut self, capture: Capture, sample: crate::command::InputSample) {
        if capture == Capture::Restart {
            self.debug_samples.clear();
        }
        self.debug_samples.push(sample);
    }

    #[cfg(not(feature = "debug-unfrozen"))]
    fn note_debug_sample(&mut self, _capture: Capture, _sample: crate::command::InputSample) {}
}
