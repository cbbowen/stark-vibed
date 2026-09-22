//! The **drawing-guide roster** (§20.5): the guides the artist has set up, in the
//! order they arranged them.
//!
//! The whole of what the five guide actions do to a document, and the one place the
//! roster's invariant — no two entries share an id — is kept.

use std::sync::Arc;

use rpds::Vector;

use stark_model::document::{GuideId, PerspectiveGuide};

use super::state::{insert_at, remove_at};

/// One guide in the roster: the camera, and what the artist calls it (§20.5).
///
/// The engine's half of the `GuideId`/`PerspectiveGuide` pair; all it adds is the
/// name, held as an `Arc<str>` for the reason
/// [`Layer::name`](super::layer::Layer) is one, where the action carrying it holds
/// the `String` that goes on the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct Guide {
    pub id: GuideId,
    /// What the artist calls this guide, or `None` for one never named — the panel
    /// then shows its *place* in the roster ("Perspective 2"), which is a description
    /// rather than a name (the same distinction
    /// [`Layer::name`](super::layer::Layer) draws).
    ///
    /// Normalized on the way in by the engine's command handler (`normalize_name`), so
    /// "either absent or something you can read" is a property of the document rather
    /// than a habit of whichever frontend collected it.
    pub name: Option<Arc<str>>,
    /// The camera, and how densely to dress it — everything §20 derives from.
    pub camera: PerspectiveGuide,
}

/// The roster: **a list with an invariant** — no two entries share an id — which is
/// kept by the vector being private, so [`insert`](Self::insert) is the only way a
/// guide gets in and the edits below can only rewrite or move one that is there.
///
/// Ordered rather than keyed: the order is the artist's arrangement and changes no
/// pixel, so lookup is a scan, deliberately — a roster is a handful of entries, and a
/// map would cost the order.
///
/// **Total, like every mutator on a [`DocState`](super::state::DocState).** An edit
/// naming a guide that is not there hands the roster back unchanged, so every peer
/// and every replay declines it the same way (§12.1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GuideRoster(Vector<Guide>);

impl GuideRoster {
    /// The guides, in roster order.
    pub fn iter(&self) -> impl Iterator<Item = &Guide> {
        self.0.iter()
    }

    /// The guide with this id, or `None` if it is not there — which is what a
    /// concurrent removal looks like from every action that names one.
    pub fn get(&self, id: GuideId) -> Option<&Guide> {
        self.0.iter().find(|g| g.id == id)
    }

    /// The roster with `guide` directly after `after`, or at its head when nothing is
    /// named (§20.5).
    ///
    /// An anchor that is not there lands the guide at the head — the shape
    /// [`Place::Above`](stark_model::document::Place::Above) takes for a sibling that
    /// has gone.
    ///
    /// An id already in the roster is refused rather than doubled. It cannot arise — a
    /// `GuideId` *is* an action's id and an action is applied once — but the roster is
    /// a list rather than a map, and a lookup answering the first of two would be the
    /// quiet kind of divergence.
    pub(crate) fn insert(&self, guide: Guide, after: Option<GuideId>) -> Self {
        if self.get(guide.id).is_some() {
            return self.clone();
        }
        Self(insert_at(&self.0, self.slot_after(after), &guide))
    }

    /// The roster without `id`. A no-op when it is already gone.
    pub(crate) fn remove(&self, id: GuideId) -> Self {
        match self.position(id) {
            Some(i) => Self(remove_at(&self.0, i)),
            None => self.clone(),
        }
    }

    /// Reshape a guide — the whole camera at once (§20.5).
    pub(crate) fn set_camera(&self, id: GuideId, camera: PerspectiveGuide) -> Self {
        self.map(id, |g| Guide {
            camera,
            ..g.clone()
        })
    }

    /// Name a guide, or with `None` take its name away so it goes back to being
    /// described by its place in the roster.
    pub(crate) fn set_name(&self, id: GuideId, name: Option<Arc<str>>) -> Self {
        self.map(id, |g| Guide {
            name: name.clone(),
            ..g.clone()
        })
    }

    /// Move a guide so it sits directly after `after`, or at the roster's head.
    ///
    /// Taken out and put back, **in that order**: the slot is re-derived against the
    /// shortened list, so an anchor above the row that left keeps the place it names
    /// rather than one past it. A guide anchored to itself therefore lands at the
    /// head, the anchor being gone by the time it is looked for.
    pub(crate) fn move_after(&self, id: GuideId, after: Option<GuideId>) -> Self {
        let Some(from) = self.position(id) else {
            return self.clone();
        };
        let moved = self.0.get(from).expect(IN_RANGE).clone();
        let rest = Self(remove_at(&self.0, from));
        Self(insert_at(&rest.0, rest.slot_after(after), &moved))
    }

    /// Where `id` sits, or `None` when it is not in the roster.
    fn position(&self, id: GuideId) -> Option<usize> {
        self.0.iter().position(|g| g.id == id)
    }

    /// Rewrite one guide in place, leaving the roster's order alone. A guide that is
    /// not there is left alone too, which is how every action naming one declines.
    fn map(&self, id: GuideId, f: impl FnOnce(&Guide) -> Guide) -> Self {
        let Some(i) = self.position(id) else {
            return self.clone();
        };
        let guide = f(self.0.get(i).expect(IN_RANGE));
        Self(self.0.set(i, guide).expect(IN_RANGE))
    }

    /// The index the anchor `after` names: one past the guide it names, or the head
    /// when it names nothing — or names a guide that is no longer there, which every
    /// peer reads the same way (§20.5).
    fn slot_after(&self, after: Option<GuideId>) -> usize {
        after.and_then(|id| self.position(id)).map_or(0, |i| i + 1)
    }
}

/// What `Vector::set` failing would mean here: an index that came from this roster.
const IN_RANGE: &str = "index came from this roster";

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId};

    /// A guide's identity is its adding action's own id (§20.5).
    const fn id(lamport: u64) -> GuideId {
        GuideId(ActionId {
            lamport,
            actor: ActorId(1),
        })
    }
    const A: GuideId = id(1);
    const B: GuideId = id(2);
    const C: GuideId = id(3);
    /// An id no roster below holds.
    const GONE: GuideId = id(9);

    /// A guide whose camera is distinguishable, so a rewrite that landed on the wrong
    /// row would show.
    fn guide(id: GuideId, focal: f32) -> Guide {
        Guide {
            id,
            name: None,
            camera: PerspectiveGuide {
                focal,
                ..PerspectiveGuide::default()
            },
        }
    }

    /// The roster's order, as the lamports that name its guides.
    fn order(roster: &GuideRoster) -> Vec<u64> {
        roster.iter().map(|g| g.id.0.lamport).collect()
    }

    /// A, B, C in roster order, each with its own focal length.
    fn three() -> GuideRoster {
        GuideRoster::default()
            .insert(guide(A, 100.0), None)
            .insert(guide(B, 200.0), Some(A))
            .insert(guide(C, 300.0), Some(B))
    }

    /// The anchor names the row a guide lands **after**; naming nothing, or naming a
    /// guide that is not there, lands it at the head.
    #[test]
    fn an_anchor_names_the_row_a_guide_lands_after() {
        assert_eq!(order(&three()), [1, 2, 3]);

        let head = three().insert(guide(id(4), 400.0), None);
        assert_eq!(order(&head), [4, 1, 2, 3], "no anchor is the head");

        let middle = three().insert(guide(id(4), 400.0), Some(A));
        assert_eq!(order(&middle), [1, 4, 2, 3]);

        let orphan = three().insert(guide(id(4), 400.0), Some(GONE));
        assert_eq!(
            order(&orphan),
            [4, 1, 2, 3],
            "an anchor that is not there reads as none",
        );
    }

    /// **No two entries share an id.** A second insert under a held id is refused
    /// whole — it does not double the row and it does not overwrite the one there.
    #[test]
    fn an_id_the_roster_already_holds_is_refused() {
        let doubled = three().insert(guide(B, 999.0), Some(C));
        assert_eq!(order(&doubled), [1, 2, 3], "the insert added a row");
        assert_eq!(
            doubled.get(B).map(|g| g.camera.focal),
            Some(200.0),
            "the refused insert overwrote the guide it collided with",
        );
    }

    /// …and nothing else can double one either: every edit below either rewrites a
    /// row that is there or hands the roster back.
    #[test]
    fn no_edit_can_double_an_id() {
        let mut roster = three();
        let edits: [fn(&GuideRoster, GuideId) -> GuideRoster; 5] = [
            |r, id| r.remove(id),
            |r, id| r.set_camera(id, PerspectiveGuide::default()),
            |r, id| r.set_name(id, Some("named".into())),
            |r, id| r.move_after(id, Some(A)),
            |r, id| r.insert(guide(id, 500.0), Some(C)),
        ];
        for edit in edits {
            for named in [A, B, C, GONE] {
                roster = edit(&roster, named);
                let mut seen: Vec<u64> = order(&roster);
                let held = seen.len();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), held, "an id is in the roster twice");
            }
        }
    }

    /// A rewrite lands on the row it names and leaves the order alone; a rewrite
    /// naming a guide that is not there changes nothing at all.
    #[test]
    fn a_rewrite_keeps_the_order_and_declines_an_absent_guide() {
        let renamed = three()
            .set_camera(
                B,
                PerspectiveGuide {
                    focal: 999.0,
                    ..PerspectiveGuide::default()
                },
            )
            .set_name(B, Some("horizon".into()));
        assert_eq!(order(&renamed), [1, 2, 3]);
        let b = renamed.get(B).expect("B is still there");
        assert_eq!(b.camera.focal, 999.0);
        assert_eq!(b.name.as_deref(), Some("horizon"));
        assert_eq!(
            renamed.get(A).map(|g| g.camera.focal),
            Some(100.0),
            "the rewrite reached a guide it did not name",
        );
        // …and the name comes off again, which is how a guide goes back to being
        // described by its place.
        assert_eq!(
            renamed
                .set_name(B, None)
                .get(B)
                .and_then(|g| g.name.clone()),
            None
        );

        for absent in [
            three().set_camera(GONE, PerspectiveGuide::default()),
            three().set_name(GONE, Some("nobody".into())),
            three().remove(GONE),
            three().move_after(GONE, Some(A)),
        ] {
            assert_eq!(absent, three(), "an absent guide moved the roster");
        }
    }

    /// A move re-derives its slot against the **shortened** roster, so an anchor above
    /// the row that left keeps the place it names rather than one past it.
    #[test]
    fn a_move_resolves_its_anchor_after_the_guide_has_left() {
        assert_eq!(order(&three().move_after(A, Some(C))), [2, 3, 1]);
        assert_eq!(
            order(&three().move_after(C, Some(A))),
            [1, 3, 2],
            "an anchor below the row that left keeps its place",
        );
        assert_eq!(order(&three().move_after(C, None)), [3, 1, 2]);
        assert_eq!(
            order(&three().move_after(B, Some(B))),
            [2, 1, 3],
            "a guide anchored to itself lands at the head, its anchor having left",
        );
        assert_eq!(
            order(&three().move_after(B, Some(GONE))),
            [2, 1, 3],
            "…as does one anchored to a guide that is not there",
        );
    }

    /// Removing takes the row out and leaves the rest in order.
    #[test]
    fn removing_takes_one_row_out() {
        assert_eq!(order(&three().remove(B)), [1, 3]);
        assert_eq!(order(&three().remove(B).remove(B)), [1, 3]);
        assert!(
            three()
                .remove(A)
                .remove(B)
                .remove(C)
                .iter()
                .next()
                .is_none()
        );
    }
}
