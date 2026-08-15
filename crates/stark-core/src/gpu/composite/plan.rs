//! What one frame's pass A does, decided in **one** walk of the group tree (§14.7,
//! §18.0.4, §21.3).
//!
//! Pure description — no GPU anywhere in this file, and no `wgpu` type in [`Plan`].
//! It is `group.rs`'s counterpart on the other side of the boundary: that file says
//! what the document *is*, this one says what drawing it *does*.
//!
//! # One walk, not four
//!
//! The alternative is a walk per product — the flat instance order, the blend and
//! filter uniform slots, how many levels isolate — with the encoder consuming all
//! three **positionally**, by cursors. Nothing then ties them together but a sentence
//! claiming they recurse alike. A slot walk that failed to recurse into a `Stack`
//! would render every group through its *sibling's* blend mode: silently, and
//! identically to the correct result whenever the two modes happened to agree. A
//! depth walk that disagreed would panic mid-encode, at a message naming whichever
//! feature happened to ask.
//!
//! So each [`Step`] carries the slot index it draws from and the targets it reads and
//! writes. The orders cannot drift because there is only one of them, and what the
//! encoder needs is a fact recorded when the decision was made rather than a count
//! reconstructed while executing it. The parity that lands the final accumulator in
//! the caller's own targets, the level a group isolates into, and which uniform slot a
//! merge binds are all byproducts of the same pass — and, being plain data, all
//! testable without an adapter.

use std::ops::Range;

use crate::geom::ViewTransform;
use crate::gpu::tile::TilePairHandle;

use super::blend::{self, BlendUniform};
use super::filter::FilterUniform;
use super::group::{CompositeGroup, CompositeItem, FilterDraw, GroupContent};
use super::tiles::{Instance, MatteInstance, Ramp};

/// Which set of channel targets a [`Step`] reads or writes.
///
/// Resolved against the caller's own targets and the frame's scratch at encode time
/// — see `Compositor::encode_plan`. A stack one level down composites into its
/// parent's `Iso`, which is why a nested group costs a level rather than a set of its
/// own (§14.7).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum Slot {
    /// The targets the render was asked to fill: the compositor's accumulator, or
    /// the eyedropper's own few hundred texels (§18.0.2).
    Target,
    /// The other half of level `l`'s ping-pong. A merge and a filter both read the
    /// accumulator and write the result, and a texture cannot be both.
    Swap(usize),
    /// Where a group at level `l` composites **alone**, which is what its mode and
    /// its clip are defined against.
    Iso(usize),
}

/// One pass-A draw: which instance stream, and which record in it.
#[derive(Copy, Clone, Debug)]
pub(super) enum Draw {
    Tile(u32),
    Matte(u32),
}

/// One thing the encoder does, in the order it does it.
#[derive(Clone, Debug)]
pub(super) enum Step {
    /// A run of items into `into`, `clear`ing as it goes when nothing has written
    /// there yet — the fast path, and the whole of an ordinary document (§14.7).
    Draw {
        into: Slot,
        draws: Range<usize>,
        clear: bool,
    },
    /// A render pass that only clears, for a stack whose first member *reads* the
    /// accumulator and so cannot fold the clear into its own load op.
    Clear { into: Slot },
    /// Merge the isolated `src` into the accumulator `back`, writing `out` (§18.0.4).
    Blend {
        back: Slot,
        src: Slot,
        out: Slot,
        slot: u32,
        phase: Phase,
    },
    /// Rewrite the accumulator `back` into `out` through a filter layer (§21.3).
    Filter {
        back: Slot,
        out: Slot,
        slot: u32,
        phase: Phase,
    },
}

/// Which way round a bouncing pass found the ping-pong, and at which level.
///
/// Recorded because it is exactly the identity of the bind group the pass reads
/// through: a merge at level `l` binds either that level's `swap` or the stack's own
/// target as its backdrop, and always that level's `iso` as its source. Two phases per
/// level, so a document with fifty merges still needs two bind groups per level rather
/// than one per merge per frame (`ScratchLevel::blend_bg`).
///
/// Derived here rather than recovered in the encoder because the plan is where the
/// ping-pong is decided — recovering it from `back` alone would mean asking which of
/// two slots a name refers to, which is the class of question this module exists to
/// stop being asked.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct Phase {
    pub(super) level: usize,
    /// Whether the backdrop is this level's `swap`, as against the stack's target.
    pub(super) back_is_swap: bool,
}

impl Step {
    /// What this step writes — every step writes exactly one set of targets, which is
    /// what makes "the accumulator ends up where the caller asked for it" a property
    /// of the last step alone.
    pub(super) fn out(&self) -> Slot {
        match self {
            Step::Draw { into, .. } | Step::Clear { into } => *into,
            Step::Blend { out, .. } | Step::Filter { out, .. } => *out,
        }
    }
}

/// One frame's compositing, as data.
///
/// Borrows the draw list it was built from: the tiles ride as handles rather than as
/// copies, and the filter layers as descriptions (see [`Self::filters`]).
pub(super) struct Plan<'a> {
    pub(super) steps: Vec<Step>,
    /// The flat draw order every [`Step::Draw`] indexes a run of.
    pub(super) draws: Vec<Draw>,

    /// Pass A's per-tile instances, flat across the whole tree, and the tile each one
    /// draws — the same index into both, which is what lets the bind groups be
    /// gathered by a loop rather than by a second walk.
    pub(super) instances: Vec<Instance>,
    pub(super) tiles: Vec<&'a TilePairHandle>,
    pub(super) mattes: Vec<MatteInstance>,
    /// One ramp slot per matte, zeroed for a solid one (§22.4). Same index as
    /// `mattes`, so a matte's instance index is also its ramp's.
    pub(super) ramps: Vec<Ramp>,

    pub(super) blends: Vec<BlendUniform>,
    /// The filter layers in slot order, kept as **descriptions** where the blends are
    /// kept as uniforms.
    ///
    /// The asymmetry is one lane's: the chromatic filter's dispersion is stated by
    /// the document in canvas terms and read by the pass in accumulator texels
    /// (§21.10), so the uniform cannot be built until the view is known — and the
    /// view is not known until the *sample count* is, which is chosen from this
    /// plan's own [`Self::scratch`] (§6.4). A blend uniform has no such lane.
    pub(super) filters: Vec<&'a FilterDraw>,

    /// One entry per level of nesting this frame reaches; `true` where the level
    /// isolates something and needs an `Iso` trio as well as a `Swap` pair. Empty
    /// when every group draws straight into the accumulator, which is the common
    /// document and allocates nothing.
    pub(super) scratch: Vec<bool>,
}

impl<'a> Plan<'a> {
    /// Walk `groups` as one stack into the caller's targets, deciding everything.
    pub(super) fn build(groups: &'a [CompositeGroup]) -> Self {
        let mut plan = Self {
            steps: Vec::new(),
            draws: Vec::new(),
            instances: Vec::new(),
            tiles: Vec::new(),
            mattes: Vec::new(),
            ramps: Vec::new(),
            blends: Vec::new(),
            filters: Vec::new(),
            scratch: Vec::new(),
        };
        plan.stack(groups, Slot::Target, 0);
        plan
    }

    /// Composite one stack's members into `target`, bottom-to-top (§14.7).
    ///
    /// Called on the document's root stack, and again on each group's members one
    /// level deeper. `level` selects this stack's ping-pong pair and the `iso` its
    /// members composite alone into; a member that is itself a group recurses into
    /// that `iso` at `level + 1`, which is why nesting costs a pair-set per level
    /// rather than per group.
    ///
    /// **The ping-pong, and why the caller's targets always win.** A blend pass reads
    /// the accumulator and writes the merge, so it needs somewhere else to write; the
    /// accumulator therefore alternates between `target` and this level's `swap`.
    /// Rather than copy at the end, the *start* is chosen by parity: with an odd
    /// number of bounces the stack begins in `swap`, and every flip lands the final
    /// result exactly where the caller asked for it. That is what lets the media pass
    /// keep one bind group and the eyedropper keep its own targets.
    fn stack(&mut self, members: &'a [CompositeGroup], target: Slot, level: usize) {
        // A merge and a filter both bounce, and it is the count of *bounces* that the
        // parity is about — not of blend modes.
        let bounces = members
            .iter()
            .filter(|m| m.as_direct_run().is_none())
            .count();
        // A level is consumed exactly when its stack has something to bounce, which
        // is also exactly when `Slot::Swap(level)` is named below — so the entry and
        // the name that needs it are one decision. A nested stack whose members all
        // draw directly consumes none, which is what makes organizing free (§14.7).
        if bounces > 0 && self.scratch.len() <= level {
            self.scratch.resize(level + 1, false);
        }
        // With nothing to bounce there is no ping-pong at all: `cur` is the caller's
        // target throughout and `alt` is never reached.
        let swap = if bounces > 0 {
            Slot::Swap(level)
        } else {
            target
        };
        let (mut cur, mut alt) = if bounces % 2 == 1 {
            (swap, target)
        } else {
            (target, swap)
        };

        // Whether `cur` holds a real accumulator yet. A direct member's draw clears
        // as it goes; a bounce cannot, because the pass *reads* what is under it, so
        // a stack that opens with one needs the clear as a step of its own.
        let mut written = false;
        for member in members {
            // The fast path, and its items in one step: `as_direct_run` is the test
            // and the extraction together, so there is no second match to disagree
            // with the first about what "direct" implies (§14.7).
            if let Some(items) = member.as_direct_run() {
                self.draw(items, cur, !written);
                written = true;
                continue;
            }
            if !written {
                self.steps.push(Step::Clear { into: cur });
                written = true;
            }
            // A **filter layer** takes the same ping-pong with nothing isolated into
            // it: there is no source, so it reads `cur` and writes the adjusted
            // result to `alt` directly (§21.3). That is the whole of what it shares
            // with a merge, and it is why this level may end up with a `Swap` pair
            // and no `Iso` trio at all.
            if let GroupContent::Filter(f) = &member.content {
                let slot = self.filters.len() as u32;
                self.filters.push(f);
                self.steps.push(Step::Filter {
                    back: cur,
                    out: alt,
                    slot,
                    phase: Phase {
                        level,
                        back_is_swap: cur == swap,
                    },
                });
                std::mem::swap(&mut cur, &mut alt);
                continue;
            }
            // The group, alone on nothing — the isolation its mode and its clip are
            // both defined against. Recorded here rather than by a separate walk, so
            // "this level isolates" and "this step reads an `Iso`" are one decision.
            self.scratch[level] = true;
            let iso = Slot::Iso(level);
            match &member.content {
                GroupContent::Run(items) => self.draw(items, iso, true),
                GroupContent::Stack(inner) => self.stack(inner, iso, level + 1),
                GroupContent::Filter(_) => unreachable!("handled above"),
            }
            let slot = self.blends.len() as u32;
            self.blends.push(BlendUniform {
                mode: blend::blend_code(member.params.blend),
                k: member.params.blend.drago_k(),
                clip: u32::from(member.params.clip),
                opacity: member.params.opacity,
            });
            self.steps.push(Step::Blend {
                back: cur,
                src: iso,
                out: alt,
                slot,
                phase: Phase {
                    level,
                    back_is_swap: cur == swap,
                },
            });
            // `alt` now holds the merged stack and becomes the accumulator; what was
            // `cur` is stale, and the next bounce overwrites all of it.
            std::mem::swap(&mut cur, &mut alt);
        }
        if !written {
            self.steps.push(Step::Clear { into: cur });
        }
    }

    /// Append one run's items to the streams, and the step that draws them.
    fn draw(&mut self, items: &'a [CompositeItem], into: Slot, clear: bool) {
        let start = self.draws.len();
        for item in items {
            match item {
                CompositeItem::Tile {
                    coord,
                    handle,
                    opacity,
                } => {
                    self.draws.push(Draw::Tile(self.instances.len() as u32));
                    self.instances.push(Instance {
                        origin: coord.origin().to_array(),
                        opacity: *opacity,
                    });
                    self.tiles.push(handle);
                }
                CompositeItem::Matte(m) => {
                    self.draws.push(Draw::Matte(self.mattes.len() as u32));
                    self.mattes.push(MatteInstance {
                        rect: m.rect,
                        channels: m.channels,
                        opacity: m.opacity,
                        resid: m.resid,
                        flags: m.flags,
                    });
                    // A gradient matte brings its ramp as a per-matte uniform
                    // (§22.4); a solid one takes a zeroed slot, whose stop count says
                    // "use the instance's own channels". One slot either way, so the
                    // matte's instance index is also its ramp's.
                    self.ramps
                        .push(m.ramp.as_deref().copied().unwrap_or_default());
                }
            }
        }
        self.steps.push(Step::Draw {
            into,
            draws: start..self.draws.len(),
            clear,
        });
    }
}

/// The filter pass's uniform for `f`, under this frame's `view` (§21).
///
/// Here rather than on [`FilterDraw`] for the reason [`Plan::filters`] gives: one of
/// its lanes is a fact about the view, which the draw deliberately has none of.
pub(super) fn filter_uniform(f: &FilterDraw, view: ViewTransform) -> FilterUniform {
    FilterUniform {
        kind: f.kind,
        strength: f.strength,
        clip: u32::from(f.clip),
        disp: chromatic_disp(f, view),
        params: f.params,
        params2: f.params2,
        // The gradient map's ramp, zeroed for every other kind — `disp`'s convention:
        // the true value, since no other kind has stops (§21.11).
        stops: f.stops.as_deref().copied().unwrap_or([[0.0; 4]; 16]),
        // The padding WGSL's alignment leaves around `clip`, which the generator
        // names and nothing reads (§6.10).
        ..Default::default()
    }
}

/// The chromatic filter's dispersion vector for this frame: the red-end → blue-end
/// displacement, carried from the canvas terms the document states (`params` =
/// spread in canvas px, angle in canvas radians) into the **accumulator texels**
/// the pass samples in, through the view's full canvas→screen linear map — zoom,
/// rotation and mirror alike, so the fringes stay attached to the artwork exactly
/// as the canvas weave does (§21.10, §6.4). Zero for every other filter kind,
/// which is the true value rather than a stand-in: no other kind disperses.
fn chromatic_disp(f: &FilterDraw, view: ViewTransform) -> [f32; 2] {
    if f.kind != stark_shaders::mirror::filter_common::FILTER_CHROMATIC {
        return [0.0; 2];
    }
    let (spread, angle) = (f.params[0], f.params[1]);
    let d = view.linear() * crate::geom::Vec2::new(angle.cos(), angle.sin()) * spread;
    [d.x, d.y]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{BlendMode, CompositeParams};
    use crate::gpu::composite::group::MatteDraw;

    /// A drawable with nothing in it — a matte, because it is the one
    /// [`CompositeItem`] that is plain data. A tile would need a GPU to make one, and
    /// nothing under test here has anything to do with paint.
    fn item() -> CompositeItem {
        CompositeItem::Matte(MatteDraw {
            rect: [0.0; 4],
            flags: 0.0,
            channels: [0.0; 4],
            resid: [0.0; 4],
            opacity: 1.0,
            ramp: None,
        })
    }

    fn leaf(params: CompositeParams) -> CompositeGroup {
        CompositeGroup::leaf(params, vec![item()])
    }

    fn plain() -> CompositeGroup {
        leaf(CompositeParams::IDENTITY)
    }

    /// A group that has to be isolated: a mode of its own.
    fn blended() -> CompositeGroup {
        leaf(CompositeParams {
            blend: BlendMode::Multiply,
            ..CompositeParams::IDENTITY
        })
    }

    fn filter() -> CompositeGroup {
        CompositeGroup::filter(FilterDraw {
            kind: stark_shaders::mirror::filter_common::FILTER_COLOR,
            strength: 1.0,
            clip: false,
            params: [0.0; 4],
            params2: [0.0; 4],
            stops: None,
        })
    }

    /// Every slot a step names must be one the scratch was told to allocate, and an
    /// `Iso` must be one the level was told to *isolate*.
    ///
    /// Checkable here because the plan is plain data. Split across an encoder
    /// recursion and a separate scratch-sizing walk, the invariant is held by nothing
    /// but a matched pair of `if`s, and its failure is an
    /// `expect("a merge without scratch targets")` mid-encode — a message that names
    /// one of the features depending on it.
    fn every_slot_is_allocated(plan: &Plan<'_>) {
        let check = |slot: Slot| match slot {
            Slot::Target => {}
            Slot::Swap(l) => assert!(
                l < plan.scratch.len(),
                "Swap({l}) with {} level(s) allocated",
                plan.scratch.len(),
            ),
            Slot::Iso(l) => assert!(
                plan.scratch.get(l).copied().unwrap_or(false),
                "Iso({l}) at a level not allocated with an iso trio",
            ),
        };
        for step in &plan.steps {
            match step {
                Step::Draw { into, .. } | Step::Clear { into } => check(*into),
                Step::Blend { back, src, out, .. } => {
                    check(*back);
                    check(*src);
                    check(*out);
                    assert_ne!(back, out, "a blend cannot read and write one texture");
                    assert_ne!(src, out, "a blend cannot read and write one texture");
                }
                Step::Filter { back, out, .. } => {
                    check(*back);
                    check(*out);
                    assert_ne!(back, out, "a filter cannot read and write one texture");
                }
            }
        }
    }

    /// The parity claim, stated once here instead of implied by arithmetic in a
    /// comment: however many times the accumulator bounced, it ends where the caller
    /// asked for it. This is what lets the media pass keep one bind group across
    /// every document, and the eyedropper read back its own targets.
    fn lands_in_the_callers_targets(plan: &Plan<'_>) {
        assert_eq!(
            plan.steps.last().map(Step::out),
            Some(Slot::Target),
            "the last step must write the caller's own targets",
        );
    }

    fn check(plan: &Plan<'_>) {
        every_slot_is_allocated(plan);
        lands_in_the_callers_targets(plan);
    }

    /// The common document: one run, straight into the caller's targets, clearing as
    /// it goes. No scratch, no bounce, no second render pass.
    #[test]
    fn a_plain_stack_is_one_cleared_draw_and_no_scratch() {
        let groups = vec![CompositeGroup::run(
            CompositeParams::IDENTITY,
            vec![item(), item()],
        )];
        let plan = Plan::build(&groups);
        assert!(plan.scratch.is_empty(), "a flat document allocates nothing");
        assert_eq!(plan.steps.len(), 1);
        assert!(matches!(
            plan.steps[0],
            Step::Draw {
                into: Slot::Target,
                clear: true,
                ..
            }
        ));
        assert_eq!(plan.draws.len(), 2);
        check(&plan);
    }

    /// An empty document still owes the caller a cleared accumulator — the media
    /// pass reads it either way.
    #[test]
    fn an_empty_stack_still_clears() {
        let plan = Plan::build(&[]);
        assert!(matches!(
            plan.steps.as_slice(),
            [Step::Clear { into: Slot::Target }]
        ));
        check(&plan);
    }

    /// One bounce is odd, so the stack starts in `swap` and the single flip lands it
    /// in the caller's targets. Getting this backwards renders the whole frame into
    /// scratch and presents whatever was in the accumulator last.
    #[test]
    fn an_odd_number_of_bounces_starts_in_the_scratch() {
        let groups = vec![plain(), blended()];
        let plan = Plan::build(&groups);
        assert_eq!(plan.scratch, vec![true]);
        assert!(
            matches!(
                plan.steps[0],
                Step::Draw {
                    into: Slot::Swap(0),
                    ..
                }
            ),
            "an odd bounce count must open in the swap: {:?}",
            plan.steps[0],
        );
        check(&plan);
    }

    /// And an even one starts in the caller's, for the same reason.
    #[test]
    fn an_even_number_of_bounces_starts_in_the_target() {
        let groups = vec![plain(), blended(), blended()];
        let plan = Plan::build(&groups);
        assert!(matches!(
            plan.steps[0],
            Step::Draw {
                into: Slot::Target,
                ..
            }
        ));
        check(&plan);
    }

    /// A stack opening with a merge cannot fold its clear into a draw, because the
    /// merge *reads* what is under it.
    #[test]
    fn a_stack_that_opens_with_a_merge_clears_on_its_own() {
        let groups = vec![blended()];
        let plan = Plan::build(&groups);
        assert!(matches!(plan.steps[0], Step::Clear { .. }));
        check(&plan);
    }

    /// A filter bounces but isolates nothing, so its level gets a `Swap` pair and no
    /// `Iso` trio — which is a third of the memory, and the case a bare "does this
    /// level need scratch" bool could not express (§21.3).
    #[test]
    fn a_filter_only_level_never_allocates_an_iso() {
        let groups = vec![plain(), filter()];
        let plan = Plan::build(&groups);
        assert_eq!(plan.scratch, vec![false], "a filter isolates nothing");
        assert!(plan.steps.iter().all(|s| !matches!(
            s,
            Step::Draw {
                into: Slot::Iso(_),
                ..
            }
        )));
        check(&plan);
    }

    /// Nesting costs one level per level, not one per group: two blended groups side
    /// by side share level 0's trio, and a group inside a group reaches level 1.
    #[test]
    fn nesting_costs_a_level_per_level() {
        let side_by_side = vec![blended(), blended()];
        assert_eq!(Plan::build(&side_by_side).scratch, vec![true]);

        let nested = vec![CompositeGroup::stack(
            CompositeParams {
                blend: BlendMode::Multiply,
                ..CompositeParams::IDENTITY
            },
            vec![plain(), blended()],
        )];
        let plan = Plan::build(&nested);
        assert_eq!(plan.scratch, vec![true, true]);
        check(&plan);
    }

    /// A group whose members all draw directly consumes no level of its own, even
    /// though the group itself is isolated at its parent's.
    #[test]
    fn a_group_of_plain_layers_consumes_no_level_of_its_own() {
        let groups = vec![CompositeGroup::stack(
            CompositeParams {
                blend: BlendMode::Multiply,
                ..CompositeParams::IDENTITY
            },
            vec![plain(), plain()],
        )];
        let plan = Plan::build(&groups);
        assert_eq!(plan.scratch, vec![true], "the parent's level, and no more");
        check(&plan);
    }

    /// The slot indices are dense and in step order, for both passes independently —
    /// a filter and a blend group side by side never count each other's slots.
    ///
    /// Recorded when the decision is made, rather than reconstructed by cursors while
    /// encoding from a list some other function built.
    #[test]
    fn uniform_slots_are_dense_and_independent() {
        let groups = vec![plain(), blended(), filter(), blended(), filter()];
        let plan = Plan::build(&groups);
        let blends: Vec<u32> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Blend { slot, .. } => Some(*slot),
                _ => None,
            })
            .collect();
        let filters: Vec<u32> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Filter { slot, .. } => Some(*slot),
                _ => None,
            })
            .collect();
        assert_eq!(blends, vec![0, 1]);
        assert_eq!(filters, vec![0, 1]);
        assert_eq!(plan.blends.len(), 2);
        assert_eq!(plan.filters.len(), 2);
        check(&plan);
    }

    /// A group's members merge before the group itself does, because the group cannot
    /// be merged until it has been composited — so the inner blend takes the lower
    /// slot. Post-order, and now by construction rather than by two functions
    /// agreeing to recurse the same way.
    #[test]
    fn an_inner_merge_takes_the_lower_slot() {
        let groups = vec![CompositeGroup::stack(
            CompositeParams {
                blend: BlendMode::Multiply,
                ..CompositeParams::IDENTITY
            },
            vec![plain(), blended()],
        )];
        let plan = Plan::build(&groups);
        let order: Vec<(u32, Slot)> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Blend { slot, out, .. } => Some((*slot, *out)),
                _ => None,
            })
            .collect();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].0, 0, "the inner merge is encoded first");
        assert_eq!(order[1].0, 1);
        assert_eq!(order[1].1, Slot::Target, "the outer merge lands the frame");
        check(&plan);
    }

    /// Every arrangement the encoder can meet, checked against both invariants at
    /// once. Cheap to extend, and the reason to have written the plan as data.
    #[test]
    fn every_shape_lands_where_it_should() {
        let group = |members| {
            CompositeGroup::stack(
                CompositeParams {
                    blend: BlendMode::Multiply,
                    ..CompositeParams::IDENTITY
                },
                members,
            )
        };
        let shapes: Vec<Vec<CompositeGroup>> = vec![
            vec![],
            vec![plain()],
            vec![blended()],
            vec![filter()],
            vec![plain(), filter()],
            vec![filter(), filter()],
            vec![blended(), filter()],
            vec![filter(), blended()],
            vec![plain(), blended(), plain()],
            vec![group(vec![plain()])],
            vec![group(vec![plain(), blended()])],
            vec![group(vec![group(vec![plain(), blended()]), blended()])],
            vec![plain(), group(vec![filter(), blended()]), filter()],
        ];
        for (i, shape) in shapes.iter().enumerate() {
            let plan = Plan::build(shape);
            every_slot_is_allocated(&plan);
            lands_in_the_callers_targets(&plan);
            assert!(!plan.steps.is_empty(), "shape {i} encoded nothing");
        }
    }
}
