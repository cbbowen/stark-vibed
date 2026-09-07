//! The projection a frontend reads (§7): [`ObservableState`], the rows it carries,
//! the keys its two rosters are memoized on, and the walk that builds them.
//!
//! The counterpart to `command`: what goes in there comes back out here, and nothing
//! here can be written to.

use std::sync::Arc;

use super::Engine;
use crate::command::Tool;
use crate::document::{CanvasBounds, Layer, LayerContent};
use crate::gpu::EnvironmentId;
use crate::projection::{Projected, Revision};
use crate::view::ViewTransform;
use stark_model::document::{GuideId, LayerId, PerspectiveGuide, ShapeAction};
use stark_model::{ColorSpaceId, Srgb, SubstrateId, SubstrateScale};

/// The layer list [`ObservableState`] carries: the whole tree flattened in composite
/// order, shared rather than copied — see [`Projected`] for what that buys and why
/// it is a type.
pub type Layers = Projected<LayerInfo>;

/// The drawing-guide roster, shared on exactly [`Layers`]' argument (§20.5).
pub type Guides = Projected<GuideInfo>;

/// What a projection off the **shown** document is a function of: the committed state,
/// and the unlogged edit standing in for it (§17.6).
///
/// - `doc_revision` advances whenever the committed document does.
/// - `preview` is `Preview::epoch`, which advances whenever the stand-in document is
///   installed, replaced or dropped.
///
/// The same two terms [`render::DrawKey`](super::render::DrawKey) keys the
/// compositor's draw list on.
///
/// **The live fold is absent because `shown` is not the fold.** A stroke in flight
/// bumps `Preview::fold` and neither term here, so its samples reproject nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ShownKey {
    doc_revision: u64,
    preview: u64,
}

/// [`ShownKey`] and the one term a guide roster has that no document does: whether
/// **this client** draws each guide (§20.5). Shutting an eye changes what the roster
/// answers while changing nothing about the document, so a key built from the
/// document alone would hand back a stale roster.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct GuideKey {
    shown: ShownKey,
    guide_epoch: Revision,
}

/// One row of the drawing-guide roster, **as this client sees it** (§20.5): the
/// document's half of a guide — camera, name, place — put together with this client's
/// **eye** on it, which is view state and so none of logged, saved or undoable.
///
/// Cheap to clone: the camera is `Copy` and the name an `Arc<str>` bump.
#[derive(Clone, Debug, PartialEq)]
pub struct GuideInfo {
    pub id: GuideId,
    /// What the artist calls it, or `None` for one never named — the panel then
    /// describes it by its place in the roster ("Perspective 2").
    pub name: Option<Arc<str>>,
    /// Whether **this client** draws it — `false` until somebody here asks for it
    /// (§20.5). Never saved, never sent, never undoable; see
    /// [`ViewCommand::SetGuideVisible`](crate::command::ViewCommand).
    pub visible: bool,
    /// The camera and its dressing: everything §20 derives from.
    pub guide: PerspectiveGuide,
}

/// A layer's presentation properties, for the UI's layer panel (§11).
///
/// `Clone` but not `Copy` since it carries the name — an `Arc<str>` bump, so
/// cloning one is still a handful of instructions and `observe()` stays cheap.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: LayerId,
    pub blend: stark_model::document::BlendMode,
    /// Whether this layer clips to the paint beneath it (§14.4).
    pub clip: bool,
    pub opacity: f32,
    pub visible: bool,
    /// The layer carrying this one — i.e. the group it is in — or `None` for one
    /// in the document's own stack (§14.2).
    pub carrier: Option<LayerId>,
    /// How deeply nested it is: 0 in the document's own stack, one more per level
    /// of carrying. What a panel indents by.
    pub depth: usize,
    /// Whether this layer carries others, i.e. whether it is a **group**. A panel
    /// gives one of these a disclosure triangle; nothing else distinguishes it.
    pub is_group: bool,
    /// Whether anything composites beneath it, so its blend mode and its clip do
    /// anything at all (§14.4.3). False on exactly one row — the bottom of the
    /// document — where a panel shows both controls inert.
    pub has_backdrop: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named — in which case it is for the frontend to describe it, since only the
    /// frontend knows how it presents a stack (see [`Layer::name`]).
    ///
    /// [`Layer::name`]: crate::document::Layer::name
    pub name: Option<std::sync::Arc<str>>,
    /// The **n in "Layer n"** for a row nobody has named — [`Layer::number`],
    /// projected. `None` for a matte or a filter, which are labelled by what they
    /// are (§14.6).
    ///
    /// Minted into the action that created the layer and never recomputed, so nothing
    /// an artist does to another row moves it.
    ///
    /// [`Layer::number`]: crate::document::Layer::number
    pub number: Option<u32>,
    /// Set when this layer is a **matte** (§15.2) — a frame rather than paint;
    /// `None` for an ordinary paint layer. Enough for the frontend to label it, draw
    /// its handles, and show that the brush has nowhere to go while it is selected.
    pub matte: Option<MatteInfo>,
    /// Set when this layer is a **filter** (§21) — a function of what is composited
    /// beneath it rather than content of its own. `None` for anything else.
    ///
    /// The whole filter, not a name for it: the filter bar's sliders read their
    /// current values off this and send the adjusted filter straight back, so a
    /// filter that grows a knob costs the projection nothing (§21.6).
    pub filter: Option<stark_model::document::Filter>,
    /// Whether the compositor would draw anything beneath this layer **within its
    /// own stack** — which is exactly what a filter layer rewrites (§21.2).
    ///
    /// Not [`has_backdrop`](Self::has_backdrop): it counts the **carrier's own
    /// content**, a group's base compositing at the bottom of the group, and it counts
    /// only what would actually **draw** — a hidden, transparent or never-painted
    /// layer is culled from the draw list (`render.rs`). `has_backdrop` stays
    /// positional because blend and clip are defined against position (§14.4.3); this
    /// follows the renderer because a filter's reach *is* the renderer's accumulator.
    pub has_underlay: bool,
    /// The layer this one would **merge down** onto, or `None` when there is no merge
    /// here that leaves the document looking the same (§14.11).
    ///
    /// The destination rather than a `bool`, so the panel's label and its click come
    /// from one answer — and the one control the panel *hides* rather than greys out
    /// when the answer is no: a merge that would change the picture is a different
    /// edit, not a weaker one.
    pub merge_down: Option<LayerId>,
    /// A number that changes exactly when this layer's own tiles do, or `None` for a
    /// layer that holds none — [`Layer::content_revision`], projected (§14.6).
    ///
    /// The layer panel keys its thumbnails on it. **The one field here that moves on
    /// an ordinary stroke** — every other describes the tree, which a stroke leaves
    /// alone, so without this the roster would compare equal across a commit that
    /// only painted.
    ///
    /// Read off `shown`, so an unlogged drag — which installs a fresh `DocState` per
    /// pointer sample — moves this row per sample where the tree beside it does not.
    /// That is what makes a thumbnail track the drag rather than lag it.
    ///
    /// [`Layer::content_revision`]: crate::document::Layer::content_revision
    pub content_revision: Option<u64>,
    /// Where the layer's frame sits on the canvas (§14.12) — what the
    /// pick-and-translate drag adds its delta to
    /// ([`DocCommand::TranslateLayer`](crate::command::DocCommand::TranslateLayer)).
    /// A matte stands in its frame like paint (§15.2); only a filter has no
    /// frame to move, and its row here is always zero.
    pub translation: stark_model::geom::IVec2,
}

impl LayerInfo {
    /// Whether a stroke aimed at this layer would draw anything. Neither a matte nor
    /// a filter has a tile map, so selecting one is legal but painting on it does
    /// nothing (§15.7, §21.4).
    pub fn is_paintable(&self) -> bool {
        self.matte.is_none() && self.filter.is_none()
    }
}

/// A matte layer's geometry and fill, for the frame chrome (§15.7).
#[derive(Clone, Debug, PartialEq)]
pub struct MatteInfo {
    /// The rect the region is defined against, in canvas px — the stored rect placed
    /// by the layer's translation (§14.12), so the chrome reads it where it shows.
    /// For a frame this is the *hole* — the piece — which the handles resize and
    /// export frames against (§15.6). `None` for a region defined against no rect
    /// ([`MatteRegion::Everything`](stark_model::document::MatteRegion::Everything)),
    /// where the handle box, the aspect readout and the export frame all stand down.
    pub rect: Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)>,
    /// The paint the region wears — flat, or a ramp (§15.4, §22.4).
    pub paint: stark_model::document::Parcel,
}

impl MatteInfo {
    /// Width × height of the rect, where there is one.
    pub fn dims(&self) -> Option<(f32, f32)> {
        let (min, max) = self.rect?;
        Some((max.x - min.x, max.y - min.y))
    }
}

/// A cheap, UI-facing projection of engine state (§7). Published to
/// the frontend so it can render chrome reactively without touching pixels.
///
/// `PartialEq` because "reactively" is the whole point: a frontend holding this in a
/// signal marks every subscriber dirty when it publishes, and needs a way to notice
/// that the answer did not move.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservableState {
    pub can_undo: bool,
    pub can_redo: bool,
    pub is_stroking: bool,
    pub tool: Tool,
    pub view: ViewTransform,
    pub bounds: CanvasBounds,
    /// A counter that changes whenever the **committed** document does — a commit,
    /// an undo, a merged remote action, a load.
    ///
    /// For a frontend keeping a *rendered* stand-in for the document, such as the
    /// navigator's miniature: nothing else here answers the question, since a second
    /// stroke over the same tiles leaves the bounds, the layer list and the undo flags
    /// untouched. Not bumped by an in-flight gesture or an unlogged drag preview —
    /// `is_stroking` is the in-flight question.
    pub doc_revision: u64,
    /// Whether the committed document has moved since it arrived — since the reset
    /// that made a new one, or since the last action of a load. False for a document
    /// that still stands exactly as the file it came out of.
    ///
    /// **Not "unsaved".** A frontend that asks before throwing the page away wants
    /// this *and* whether the revision on screen is one it has since written to a
    /// file, which only it knows (`stark-dioxus-frontend`'s `files::unsaved`).
    pub edited: bool,
    pub active_layer: LayerId,
    /// Layers bottom-to-top. Shared rather than copied — see [`Layers`].
    pub layers: Layers,
    /// Whether a selection is masking the canvas (§6.8) — drives the
    /// "Deselect"/"Invert" affordances and the selection indicator.
    pub has_selection: bool,
    /// A conservative canvas-space bounding box of this client's selected
    /// coverage, or `None` when the selection is unbounded or unknown
    /// ([`Selection::hull`](crate::document::Selection::hull)). What the
    /// transform chrome hangs its handles on; committed-only, like
    /// `has_selection`.
    pub selection_hull: Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)>,
    /// What the next shape gesture will do with the region it encloses — combine
    /// it into the selection one of four ways, or fill it (§18.0.4).
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) the next shape gesture will apply.
    pub selection_feather: f32,
    /// How strongly a **fill** gesture's parcel will land, `0..=1` (§18.0.4).
    pub shape_opacity: f32,
    /// How strongly this client's whole selection mask gates, `0..=1` (§6.8) — the
    /// selection bar's Opacity slider. Live with nothing selected too: there it is
    /// the strength the next region will take, and the whole canvas's until one is
    /// drawn ([`Selection::opacity`](crate::document::Selection::opacity)); a
    /// deselect puts it back to 1.
    pub selection_opacity: f32,
    /// Whether collaborators' selection outlines are drawn (§17.3).
    pub show_peer_selections: bool,
    /// How much resident tile memory history retention may hold before undo depth
    /// is given up, in bytes (§5) — what
    /// [`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand) sets.
    ///
    /// Projected for the reason `tool` and `brush` are: a frontend's own copy goes
    /// stale the moment anything else moves it (§4). The settings dialog reads its
    /// slider off this.
    pub history_budget: u64,
    /// Whether a stroke's commit takes the tiles its preview already drew (§6.2) —
    /// what [`ViewCommand::SetFastCommit`](crate::command::ViewCommand) sets.
    ///
    /// Projected for `history_budget`'s reason; the settings dialog reads its switch
    /// off this.
    pub fast_commit: bool,
    /// The drawing guides (§20.5), **as this client sees them**: the document's
    /// roster with each row carrying whether this client's eye on it is open, so the
    /// panel, the edit bar and the hit test read one roster rather than each putting
    /// the two halves of a guide together ([`GuideInfo`]).
    pub guides: Guides,

    // --- view settings (per-client, never historized) ---------------------
    //
    // Projected for `history_budget`'s reason (§4).
    /// Media/lighting parameters of the painterly pass (§6.3).
    pub media: crate::gpu::MediaParams,
    /// The display the screen is presented on (§6.5) — what
    /// [`ViewCommand::SetOutput`](crate::command::ViewCommand) set.
    pub output: crate::gpu::Output,
    /// The HDR lighting environment in use (§6.3).
    pub environment: EnvironmentId,

    // --- document properties fixed at creation ----------------------------
    /// The document's color space. Immutable for the document's life — changing
    /// it means starting a new document ([`Engine::new_with_color_space`]).
    pub color_space: ColorSpaceId,
    /// The physical canvas substrate (§6.4).
    pub substrate: SubstrateId,
    /// How large that substrate is laid (§6.4) — what the Lighting panel's scale slider
    /// reads and what it commits against.
    pub substrate_scale: SubstrateScale,
    /// The canvas substrate color, straight sRGB (§15.5). Document
    /// state, not a view setting — projected here so the frontend shows what the
    /// document says rather than a copy of its own that goes stale.
    pub substrate_color: Srgb,

    /// Set once the GPU has failed — a lost device, an out-of-memory, an error no
    /// scope caught (§5); `None` for a healthy device.
    ///
    /// **Projected because the document outlives the device.** The log is in ordinary
    /// memory, so a frontend told this can still write the file: it should stop
    /// dispatching and offer to save, and must not keep painting, since nothing past
    /// this point reaches a pixel.
    pub gpu_failure: Option<Arc<crate::gpu::DeviceFailure>>,
}

impl Engine {
    /// What every projection off `shown` is keyed on, read in one place so the two
    /// below cannot come to disagree about which counters say `shown` moved — see
    /// [`ShownKey`] for what the two terms are.
    fn shown_key(&self) -> ShownKey {
        ShownKey {
            doc_revision: self.doc_revision,
            preview: self.preview.epoch(),
        }
    }

    /// The layer roster for `shown`, **rebuilt only when the document it describes
    /// has moved** — the walk in [`observe`](Self::observe) is a pure function of
    /// `shown`, and [`ShownKey`] names exactly what `shown` is a function of.
    fn projected_layers(&self, build: impl FnOnce() -> Vec<LayerInfo>) -> Layers {
        self.layer_cache
            .get_or_build(self.shown_key(), || build().into())
    }

    /// The drawing-guide roster this client sees, keyed on [`GuideKey`] — memoized
    /// for the property rather than the cost (§20.5): an *unchanged* roster hands
    /// back the same `Arc`, so a frontend's "did this move?" stays the pointer
    /// comparison [`Projected`] exists for.
    fn projected_guides(&self, build: impl FnOnce() -> Vec<GuideInfo>) -> Guides {
        let key = GuideKey {
            shown: self.shown_key(),
            guide_epoch: self.guide_epoch,
        };
        self.guide_cache.get_or_build(key, || build().into())
    }

    /// A snapshot of UI-facing state (§7).
    pub fn observe(&self) -> ObservableState {
        /// Whether the compositor would draw anything at all for `l` — the same
        /// culls `render.rs` applies, asked of the document rather than of a
        /// viewport, so the answer does not change when the artist scrolls.
        fn contributes(l: &Layer) -> bool {
            l.is_shown() && (l.draws_content() || l.carries.iter().any(contributes))
        }
        let doc = self.timeline.current();
        // Layers and the substrate color come off the *previewed* document when one is
        // in flight (§15.7, §15.5), so the frame's handles track a drag and the color
        // swatch tracks the picker rather than lagging the canvas. Only those two:
        // `has_selection` must stay committed-only, or a marquee drag flashes the
        // selection bar in and out before anything is selected.
        let shown = self.preview.doc().unwrap_or(doc);
        // Flattened in **composite order** — each stack bottom-to-top, a group's base
        // before what it carries — with the tree carried alongside as `depth` and
        // `carrier` (§14.6). That is the order a panel draws in and the order the
        // compositor draws in, so one list means both.
        /// What the walk knows about the stack it is currently in, one per depth.
        /// Truncating on the way back up is what makes these per-*stack*: re-entering
        /// depth `d` from deeper keeps them, descending to a new `d` starts a fresh one.
        struct Cursor<'a> {
            /// Whether this stack has anything drawable beneath the row being visited
            /// — "would a filter here reach something" (§21.2), answered by the draw
            /// list's culls (`render.rs`) rather than by row order: a hidden,
            /// transparent or empty sibling fills nothing.
            filled: bool,
            /// The row visited before this one **in this stack** — a layer's lower
            /// sibling, which is what it would merge down onto (§14.11).
            below: Option<&'a Layer>,
            /// How far up this stack the row being visited sits, counting from the
            /// bottom — `LayerSite::index` without the search for it.
            index: usize,
        }
        // Rebuilt only when the document it describes has moved; every other
        // observation hands out the last one (see [`Engine::projected_layers`]).
        let layers = self.projected_layers(|| {
            let mut layers: Vec<LayerInfo> = Vec::new();
            // The carrier chain down to the current row: the layer (whose id
            // `LayerInfo::carrier` reports) and whether its own content draws anything —
            // the seed for the stack its carries open, since a base composites at the
            // bottom of its group (§14.1).
            let mut carriers: Vec<(&Layer, bool)> = Vec::new();
            let mut stack: Vec<Cursor<'_>> = Vec::new();
            shown.visit(&mut |l, depth| {
                carriers.truncate(depth);
                if stack.len() > depth {
                    stack.truncate(depth + 1);
                } else {
                    let filled = carriers.last().is_some_and(|&(_, draws)| draws);
                    stack.push(Cursor {
                        filled,
                        below: None,
                        index: 0,
                    });
                }
                let has_underlay = stack[depth].filled;
                stack[depth].filled = stack[depth].filled || contributes(l);
                layers.push(LayerInfo {
                    id: l.id,
                    blend: l.composite.blend,
                    clip: l.composite.clip,
                    opacity: l.composite.opacity,
                    visible: l.visible,
                    carrier: carriers.last().map(|&(c, _)| c.id),
                    depth,
                    is_group: l.is_group(),
                    // Composite order visits the bottom of the root stack first, and
                    // that is the *only* layer with nothing beneath it (§14.4.3) —
                    // every other one has a lower sibling or the content of the layer
                    // carrying it. So "has a backdrop" is "is not the first row".
                    has_backdrop: !layers.is_empty(),
                    name: l.name.clone(),
                    number: l.number,
                    matte: match &l.content {
                        // Placed onto the canvas, where the chrome lives: region and
                        // paint are stated in the layer's frame (§15.2), and the mint
                        // takes the same offset back out (`DocCommand::SetMatteRect`),
                        // so the handles never do frame arithmetic.
                        LayerContent::Matte { region, paint } => {
                            let d = l.translation.as_vec2();
                            Some(MatteInfo {
                                rect: region.translated(d).rect(),
                                paint: paint.translated(d),
                            })
                        }
                        LayerContent::Paint(_) | LayerContent::Filter(_) => None,
                    },
                    filter: l.filter(),
                    has_underlay,
                    // Off the walk, which already holds what the question needs: the
                    // lower sibling and the carrier. Asking `merge::plan` per row
                    // costs a `site_of` — a walk of the whole tree — making this
                    // projection quadratic in the layer count (79 µs at 60 layers
                    // against 1.3 µs at 4).
                    merge_down: stack[depth]
                        .below
                        .map(|d| (d, false))
                        .or_else(|| carriers.last().map(|&(c, _)| (c, true)))
                        .and_then(|(dest, dest_is_carrier)| {
                            crate::document::merge::plan_at(&crate::document::merge::MergeSite {
                                source: l,
                                dest,
                                // The destination is the whole backdrop where it carries
                                // the source, and — in the root stack — where the source
                                // sits second from the foot over an unclipped layer, whose
                                // accumulator starts cleared (§14.11.2).
                                backdrop_is_dest: dest_is_carrier
                                    || (depth == 0
                                        && stack[depth].index == 1
                                        && !dest.composite.clip),
                                dest_is_carrier,
                            })
                            .map(|p| p.dest)
                        }),
                    // Off the *shown* document like the rest of the row, so a thumbnail
                    // tracks a drag preview's tiles rather than the committed ones
                    // behind them.
                    content_revision: l.content_revision(),
                    translation: l.translation,
                });
                stack[depth].below = Some(l);
                stack[depth].index += 1;
                carriers.push((l, l.draws_content()));
            });
            layers
        });
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool(),
            view: self.session.view,
            bounds: doc.bounds(),
            doc_revision: self.doc_revision,
            edited: self.doc_revision != self.doc_origin,
            active_layer: self.session.active_layer(),
            layers,
            has_selection: doc.has_selection(self.actor()),
            selection_hull: doc.selection_of(self.actor()).hull(),
            // Off the *shown* document, unlike the two above: this one previews
            // (`PreviewSelectionOpacity`), and a bar reading the committed number
            // would fight its own slider mid-drag (§6.8).
            selection_opacity: shown.selection_of(self.actor()).opacity(),
            shape_action: self.session.shape_action,
            selection_feather: self.session.selection_feather(),
            shape_opacity: self.session.shape_opacity(),
            show_peer_selections: self.session.show_peer_selections,
            history_budget: self.history_budget,
            fast_commit: self.fast_commit,
            guides: self.projected_guides(|| {
                // Off `shown`, like the layer list and the substrate color: a
                // guide's drag previews through `PreviewGuide`, and a panel
                // reading the committed roster would show the pose the hand left
                // behind (§20.5).
                shown
                    .guides()
                    .iter()
                    .map(|g| GuideInfo {
                        id: g.id,
                        name: g.name.clone(),
                        visible: self.session.guide_visible(g.id),
                        guide: g.camera,
                    })
                    .collect()
            }),
            media: self.compositor_pipeline.media(),
            output: self.compositor_pipeline.output(),
            environment: self.shared.environment.id(),
            color_space: self.shared.color_space.id(),
            substrate: doc.substrate,
            substrate_scale: doc.substrate_scale,
            substrate_color: shown.substrate_color,
            gpu_failure: self.shared.gpu.health().failure().map(Arc::new),
        }
    }
}
