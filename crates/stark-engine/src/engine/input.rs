//! Commands into state (§4): the four arms of [`Engine::process`], and what an arm
//! reads off the committed document before it commits — the frame a canvas-space
//! payload is converted into, the subtree a translate expands to, the name funnel.
//!
//! An arm commits through `commit`'s doors or previews through `live`'s, and decides
//! nothing about either.

use super::Engine;
use super::live::Held;
use crate::command::{DocCommand, Edit, GestureCommand, PeerCommand, ViewCommand};
use crate::document::Step;
use crate::gesture::LiveGesture;
use stark_model::document::{ActionId, ActionKind, GuideId, LayerId, TransformMap};

/// Longest name that will be recorded, in `char`s — the wire's bound, reached for
/// rather than restated.
///
/// A name travels, so it is bounded, and nothing about a text field stops a paste
/// from being a megabyte. Stated where the *wire* can also reach it
/// ([`stark_model::MAX_NAME`]), since a presence frame's name is capped by the same
/// number and the model cannot depend on this crate (§2).
use stark_model::MAX_NAME;

/// The name to record, given what a frontend collected: surrounding whitespace
/// trimmed, length capped at `MAX_NAME`, and anything that comes out empty treated as
/// *no name* rather than as a name that happens to be blank.
///
/// One funnel for every source — the panel's field, a script, a peer's command — and
/// for both layers and drawing guides, so what a name *is* is not a property of which
/// command carried it. The logged action carries the result, so replay reproduces it
/// without re-running these rules.
///
/// Generic at both ends because the callers hold their names differently: a logged
/// action a `String`, a guide an `Arc<str>`. `String: From<String>` is the identity,
/// so the logged path still moves its bytes rather than copying them.
fn normalize_name<T: From<String>>(name: Option<impl AsRef<str>>) -> Option<T> {
    let trimmed = name?;
    let capped: String = trimmed.as_ref().trim().chars().take(MAX_NAME).collect();
    (!capped.is_empty()).then(|| T::from(capped))
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
    /// The canvas offset of a layer's frame (§14.12) — zero for a layer that is not
    /// there, which every action aimed at one is refused as anyway. Off the
    /// **committed** document, which every mint and every preview entry point reads,
    /// so the frame a gesture is converted into is the frame its commit will carry.
    pub(super) fn frame_of(&self, layer: LayerId) -> stark_model::geom::IVec2 {
        self.timeline
            .current()
            .layer(layer)
            .map_or(stark_model::geom::IVec2::ZERO, |l| l.translation)
    }

    /// The whole move a translate gesture means (§14.12): `layer`'s subtree — a group
    /// moves as one, and translation does not inherit — with each member that has a
    /// frame to move
    /// ([`Layer::is_translatable`](crate::document::Layer::is_translatable): paint and
    /// mattes) displaced by the same delta. Filters are left out rather than
    /// named-and-refused, so a drag out and back does not log a step that does
    /// nothing. Read off the committed document, like every mint; an absent layer
    /// expands to no moves, which `is_noop_on` then declines to log.
    fn translate_moves(
        &self,
        layer: LayerId,
        to: stark_model::geom::IVec2,
    ) -> Vec<(LayerId, stark_model::geom::IVec2)> {
        use stark_model::document::FRAME_LIMIT;
        // Clamped before the subtraction below, which is otherwise the first thing an
        // unbounded `to` reaches: the commit's own clamp comes too late for it.
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

    /// The action an [`Edit`] commits — and the very one its preview folds (§21.6) — or
    /// `None` for a transform whose map is unusable, a knowably dead action kept out
    /// of the log (§16.1).
    ///
    /// **One function, so `preview == committed` (§1.3) is a property of the code
    /// rather than of two arms agreeing.** Whatever a kind needs beyond its payload is
    /// read here, once, off the committed document: the canvas-to-frame conversion a
    /// matte's rect and paint and a fill's op carry, the frame a transform's map is
    /// conjugated into (§14.12), and the subtree a translate expands to.
    fn edit_kind(&self, edit: Edit) -> Option<ActionKind> {
        Some(match edit {
            Edit::LayerBlend(id, blend) => ActionKind::SetLayerBlend(id, blend),
            Edit::LayerOpacity(id, opacity) => ActionKind::SetLayerOpacity(id, opacity),
            Edit::SelectionOpacity(opacity) => ActionKind::SetSelectionOpacity(opacity),
            Edit::Filter(id, filter) => ActionKind::SetFilter(id, filter),
            Edit::MatteRect(id, min, max) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMatteRect(id, min - d, max - d)
            }
            Edit::MattePaint(id, paint) => {
                let d = self.frame_of(id).as_vec2();
                ActionKind::SetMattePaint(id, paint.translated(-d))
            }
            Edit::SubstrateColor(rgb) => ActionKind::SetSubstrateColor(rgb),
            Edit::SubstrateScale(scale) => ActionKind::SetSubstrateScale(scale),
            Edit::Translate { layer, to } => ActionKind::TranslateLayers {
                moves: self.translate_moves(layer, to),
            },
            Edit::Guide(id, guide) => ActionKind::SetGuide(id, guide),
            Edit::Fill { layer, op } => {
                let frame = self.frame_of(layer);
                ActionKind::Fill {
                    layer,
                    op: op.translated(-frame.as_vec2()),
                    translation: frame,
                }
            }
            Edit::Transform { map, .. } if !map.usable() => return None,
            // The map stays stated on the canvas, with the frame beside it; the fold
            // conjugates. Each family is its own kind, since the wire never carries
            // the routing enum.
            Edit::Transform { layer, map } => {
                let translation = self.frame_of(layer);
                match map {
                    TransformMap::Affine(affine) => ActionKind::Transform {
                        layer,
                        affine,
                        translation,
                    },
                    TransformMap::Perspective(map) => ActionKind::TransformPerspective {
                        layer,
                        map,
                        translation,
                    },
                    TransformMap::Warp(map) => ActionKind::TransformWarp {
                        layer,
                        map,
                        translation,
                    },
                }
            }
        })
    }

    /// Commit what `edit` mints — or, where it mints nothing, drop the drag preview
    /// as a declined commit would ([`commit`](Self::commit)'s bargain).
    fn commit_edit(&mut self, edit: Edit) {
        // A committed scale is a substrate the registry bakes, and a `SubstrateMap` is
        // built from the substrate and its scale together — so laying it at another
        // size invalidates the bound one exactly as `SetSubstrate` does.
        let rebakes = matches!(edit, Edit::SubstrateScale(_));
        match self.edit_kind(edit) {
            Some(kind) => self.commit(kind),
            None => self.preview.hold(None),
        }
        if rebakes {
            self.apply_document_substrate();
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
                let frame = self.frame_of(self.session.active_layer());
                // **The press's dispatch is the tool's own answer** (§6.8): a tool
                // either drags a region out or lays paint, and answering with the
                // region leaves the drag no way to hold a tool that drags none.
                match tool.outline() {
                    // A marquee or lasso fits no curve, so it has no use for the
                    // tolerance (or the rope); its own decimation is a mask-cost knob
                    // (§6.8). What it does need is whether there is a mask to combine
                    // with — an Add drawn over nothing is a New
                    // (`session::against_selection`) — read at the press off the
                    // committed document, so the gesture's meaning is fixed before it
                    // has drawn anything.
                    Some(outline) => {
                        let has_selection = self.document().has_selection(self.actor());
                        self.session.start_selection(
                            tool,
                            outline,
                            sample.pos,
                            has_selection,
                            frame,
                        );
                    }
                    None => {
                        let seed = self.authoring.clock;
                        self.session
                            .start_stroke(sample, seed, tolerance, rope, frame);
                        self.note_debug_sample(Capture::Restart, sample);
                    }
                }
                self.mark_live_stale();
            }
            GestureCommand::To { sample } => {
                // The CPU half of a pointer sample, and the *whole* of what arrives at
                // input rate — the fold and the render it marks stale are paid once a
                // frame instead. Its own row because it is the phase that grows with
                // stroke length rather than with the tail: the fitter re-solves its
                // unfrozen prefix on every push, ~350× the flattening beside it.
                crate::timing::span!("input.fit");
                // One door for every kind of gesture: which one is live is the
                // session's to know (`Session::gesture_to`), not this arm's to ask.
                self.session.gesture_to(sample);
                if self.session.is_stroking() {
                    self.note_debug_sample(Capture::Continue, sample);
                }
                self.mark_live_stale();
            }
            // A held pointer: snap the stroke to the shape it resembles (§6.9).
            // Nothing is committed — a snap changes what the *same* drag builds, and
            // the release still commits one stroke either way.
            GestureCommand::Hold => {
                // Built here rather than inside the session, because the two halves of
                // a guide live in two places (§20.5) and this is the only side holding
                // both. Off the **committed** document, so a guide being dragged
                // elsewhere in the same instant cannot move what a snap aims at —
                // the surprise §20.5 rules out.
                let scaffold = self.scaffold(self.timeline.current());
                if self.session.assist_stroke(&scaffold) {
                    self.mark_live_stale();
                }
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                // Fold first, so the commit is offered the stroke as it stands at the
                // release — the frame that would have shown the last few samples. A
                // fold costs the live tail; the render it saves the commit costs the
                // stroke. Only a stroke has such a render to save.
                if self.session.is_stroking() {
                    self.flush_live();
                }
                // One release, three things it can commit — which one was decided when
                // the gesture started (§18.0.4).
                match self.session.end_gesture() {
                    Some(LiveGesture::Stroke(rec)) => {
                        self.log_debug_samples();
                        self.commit_stroke(rec);
                    }
                    Some(LiveGesture::Selection(op)) => self.commit(ActionKind::Select(op)),
                    // The layer the drag pinned at the press, not the active layer
                    // now: the op was converted into *that* layer's frame, and the
                    // two must not part (`ShapeDrag::layer`).
                    Some(LiveGesture::Fill {
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
            // Any existing layer, including a matte. `active_layer` is *the selected
            // layer*, not "a paint target", which is what lets the frontend have one
            // selection concept instead of two (§15.7). A stroke aimed at a matte
            // draws nothing, refused identically by `apply` and by the preview path.
            PeerCommand::SetActiveLayer(id) => {
                if self.session.set_active_layer(id, self.timeline.current()) {
                    // The hover mark follows the brush's target (§18.1.10): it is
                    // built against the active layer at fold time, so moving the
                    // selection has to re-lay it there.
                    self.mark_live_stale();
                }
            }
            PeerCommand::SetCursor(pos) => self.session.set_cursor(pos),
            PeerCommand::SetName(name) => self.session.set_name(name),
        }
    }

    /// Document-state mutations: every arm here either commits an action or
    /// navigates the history that holds them. Neither marks the fold stale by hand:
    /// the doors that change the document under it do (`change_committed`), and an arm
    /// that changes nothing leaves the fold and its cached heads alone.
    pub(super) fn process_doc(&mut self, command: DocCommand) {
        match command {
            DocCommand::Undo => self.navigate(Step::Undo),
            DocCommand::Redo => self.navigate(Step::Redo),
            DocCommand::Seek(to) => self.navigate(Step::Seek(to)),
            DocCommand::Select(op) => self.commit(ActionKind::Select(op)),
            DocCommand::InvertSelection => self.commit(ActionKind::InvertSelection),
            DocCommand::Set(edit) => self.commit_edit(edit),
            DocCommand::FloatSelection { layer } => {
                // Asked before an action is spent, as `MergeLayerDown` asks its offer:
                // the fold's own refusals, off the same committed state, so a float
                // that would cut nothing logs nothing (§16.12).
                let frame = self.frame_of(layer);
                let (doc, actor) = (self.document(), self.actor());
                if crate::document::apply::float_cut_offered(doc, actor, layer, frame) {
                    // The float is also what the hand is about to move.
                    self.mint_layer(|a, number| ActionKind::FloatSelection {
                        layer,
                        child: LayerId::new(a, 0),
                        translation: frame,
                        number,
                    });
                }
            }
            DocCommand::SetSubstrate(id) => {
                self.commit(ActionKind::SetSubstrate(id));
                // Unconditional, and a no-op when the substrate did not move: the
                // registry is brought level with the document rather than with what
                // this command asked for.
                self.apply_document_substrate();
            }
            DocCommand::AddLayer { carrier, above } => {
                self.mint_layer(|a, number| ActionKind::AddLayer {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    number,
                });
            }
            DocCommand::PlaceImage {
                carrier,
                above,
                at,
                name,
                image,
            } => {
                self.mint_layer(|a, number| ActionKind::PlaceImage {
                    id: LayerId::new(a, 0),
                    carrier,
                    above,
                    at,
                    name,
                    image,
                    number,
                });
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
                // Deliberately *not* made the active layer, unlike `AddLayer`: a matte
                // has no tile map, so painting on it is refused (§15.7) and arming it
                // as the target would swallow the user's next stroke.
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
                // Deliberately *not* made the active layer, for `AddMatte`'s reason: a
                // filter has no tile map, so arming it as the paint target would
                // swallow the next stroke (§21.4).
            }
            DocCommand::DuplicateLayer(source) => {
                // One minted id per layer of the subtree, paired with the layer it
                // copies, in composite order — the map the action carries (§14.8).
                // Written as pairs because that is the shape `apply` reads and the
                // shape the footprint claims a `Layer(src)` from.
                //
                // Through the document's own walk, not a second one here: `apply`
                // declines the action unless `ids` names exactly the subtree
                // `duplicate_layer` walks, so a copy of the traversal here would be two
                // walks that must agree — on this client and on every peer.
                if let Some(sources) = self.document().subtree_ids(source) {
                    // `number` is the first of the run the copies wear, `ids`' own
                    // order deciding the rest — see `ActionKind::DuplicateLayer`.
                    self.mint_layer(|a, number| ActionKind::DuplicateLayer {
                        ids: sources
                            .iter()
                            .enumerate()
                            .map(|(k, &src)| (src, LayerId::new(a, k as u32)))
                            .collect(),
                        number,
                    });
                }
            }
            // The subtree travels in the action, read off the document the command was
            // aimed at (§12.6) — see `ActionKind::RemoveLayer`. An absent layer mints
            // an empty list and the fold declines it, as every other action naming one
            // does.
            DocCommand::RemoveLayer(id) => {
                let carried = self.document().carried_ids(id).unwrap_or_default();
                self.commit(ActionKind::RemoveLayer { id, carried })
            }
            DocCommand::MergeLayerDown(id) => {
                // Asked here rather than only inside `apply`, so a merge the fold would
                // decline never reaches the log or repoints the brush — the same
                // argument `Transform` makes about a degenerate map. `apply` asks again,
                // because a peer's action does not pass through here (§14.11).
                if let Some(plan) = crate::document::merge::offer(self.document(), id) {
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
                    // says *which* somewhere, since the nearest paintable layer is not
                    // the paint that just absorbed what you were working on.
                    if follow {
                        self.arm_active(plan.dest);
                    }
                }
            }
            DocCommand::SetLayerClip(id, clip) => self.commit(ActionKind::SetLayerClip(id, clip)),
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
            // that adds it, minted through the same door a layer's is — so there is no
            // counter here and nothing for `resync_counters` to resume past (§17.9).
            DocCommand::AddGuide { guide, after, name } => {
                self.commit_minting(|a| ActionKind::AddGuide {
                    id: GuideId(a),
                    guide,
                    after,
                    name: normalize_name(name),
                });
            }
            DocCommand::RemoveGuide(id) => self.commit(ActionKind::RemoveGuide(id)),
            DocCommand::SetGuideName(id, name) => {
                self.commit(ActionKind::SetGuideName(id, normalize_name(name)))
            }
            DocCommand::MoveGuide { id, after } => self.commit(ActionKind::MoveGuide { id, after }),
        }
    }

    /// Show the document committing `edit` would leave behind, without logging it
    /// (§21.6).
    ///
    /// `None` clears the preview, which is what the release of a drag that changed
    /// nothing sends, and so does an edit whose commit would be declined. The sanitize
    /// and the fold both happen inside [`crate::document::apply::preview_of`], so this
    /// cannot forget a step it does not perform.
    ///
    /// It moves the document the compositor reads and stops there: no
    /// `apply_document_substrate`, so nothing bakes while a hand is on a slider
    /// ([`Edit::SubstrateScale`]).
    fn preview_edit(&mut self, edit: Option<Edit>) {
        let held = edit.map(|e| self.held(e));
        self.preview.hold(held);
    }

    /// `edit` as the hand holds it: folded over the committed document through the
    /// action its release would commit.
    fn held(&self, edit: Edit) -> Held {
        let shown = self.edit_kind(edit.clone()).and_then(|kind| {
            crate::document::apply::preview_of(
                kind,
                self.timeline.current(),
                self.actor(),
                &self.shared.apply,
            )
        });
        Held::new(edit, shown)
    }

    /// Fold the held edit again over a committed document that a change the hand did
    /// not make has just replaced — a peer's action, a replay (`Drag::Keep`) — and
    /// invalidate what was drawn over the old one.
    ///
    /// Here rather than at the next pointer sample, because a slider held still and an
    /// idle transform mode send none: the drag would stay built over the old document,
    /// hiding everything that landed since.
    pub(super) fn refold_held(&mut self) {
        let held = self.preview.held_edit().cloned().map(|e| self.held(e));
        self.preview.rebase(held);
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
                // Sanitized behind the door, not here — see `Session::set_brush`.
                self.session.set_brush(brush);
                self.session.set_color(color);
                self.mark_live_stale();
            }
            // Grab-and-drag: content follows the cursor, so the view center moves
            // opposite by the drag delta, carried into canvas units through the whole
            // map — a turned or mirrored canvas sends a screen-space drag somewhere
            // else entirely. Every arm here names a mutator rather than writing a view
            // field, so a command carrying a non-finite number is refused by
            // `ViewTransform` rather than stored.
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
                // moves when an eye does.
                if self.session.set_guide_visible(id, visible) {
                    self.guide_epoch.bump();
                    self.mark_live_stale();
                }
            }
            ViewCommand::Preview(edit) => self.preview_edit(edit),
            ViewCommand::PreviewHover(report) => match report {
                Some(r) => {
                    // The CPU half of a hover report — the window refit — on its own
                    // row, so following a resting pointer is never folded into what
                    // painting costs (`input.fit`, §7.1).
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

    /// Replay a whole recorded stroke as a single commit: start → samples → end,
    /// rendering the stroke exactly once, at commit — where interactive samples go
    /// through `GestureCommand::To` and are folded once a frame. Used by the brush
    /// editor's test-stroke replay.
    ///
    /// Answers the id of the action it committed, or `None` where the samples held no
    /// stroke — empty, a hand that never left the first point, or a reach too large
    /// for any render to draw.
    pub fn replay_stroke(&mut self, samples: &[crate::command::InputSample]) -> Option<ActionId> {
        self.replay_stroke_seeded(samples, self.authoring.clock, 0.0)
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the Lamport
    /// clock, so a caller re-rendering *one* stroke to show the effect of a brush
    /// change (the brush editor's preview) can hold the color dynamics and dither fixed
    /// while only the edited parameter moves. `rope` is the §6.11 smoothing string, a
    /// parameter here — where [`Engine::replay_stroke`] pins it to zero — because that
    /// preview replays a recorded hand and has to show what the smoothing slider beside
    /// it would do to it.
    ///
    /// **Answers what it committed**, which is §4's requirement of anything that
    /// mutates and is not a command: a caller has to be able to tell a committed stroke
    /// from a refused one. `None` is "these samples held no stroke": none at all, a
    /// hand that never left its first point, or a record refused as too large, which
    /// every render refuses whole and so is never logged.
    ///
    /// Not routed through `GestureCommand`, deliberately: the command tier's payloads
    /// are values a frontend builds per event, and this takes a borrowed slice a bench
    /// replays in a loop.
    pub fn replay_stroke_seeded(
        &mut self,
        samples: &[crate::command::InputSample],
        seed: u64,
        rope: f32,
    ) -> Option<ActionId> {
        let mut it = samples.iter();
        let first = it.next()?;
        // Replayed samples are already in canvas space and came from a fit or from a
        // generator, not from a device, so there is no device tolerance to declare.
        let frame = self.frame_of(self.session.active_layer());
        self.session
            .start_stroke(*first, seed, crate::path::DEFAULT_TOLERANCE, rope, frame);
        for s in it {
            self.session.gesture_to(*s);
        }
        let committed = match self.session.end_gesture() {
            Some(LiveGesture::Stroke(rec)) => {
                let id = self.next_action_id();
                self.commit_with_id(id, ActionKind::CommitStroke(rec));
                Some(id)
            }
            // A replay opens a stroke, so nothing else can be what it ends.
            Some(LiveGesture::Selection(_) | LiveGesture::Fill { .. }) | None => None,
        };
        self.mark_live_stale();
        committed
    }

    /// Keep a raw pointer report of the stroke in hand, so a misfit seen in the app
    /// can be dumped on release and replayed as a test
    /// ([`log_debug_samples`](Self::log_debug_samples)).
    ///
    /// A diagnostic, so a shipping build carries neither the samples nor the field
    /// that would hold them: `#[cfg]`, not a runtime `cfg!` around a `Vec` that exists
    /// either way. Behind a *call* rather than an `#[cfg]` block at each site, so the
    /// two sites cannot come to disagree about the gate.
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
