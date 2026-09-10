//! Which row an add just made (§4, §20.5).
//!
//! An add answers nothing — a command returns no value (§4) — and the engine mints the
//! new id itself: a layer's out of the action that created it, a guide's as that
//! action's own id (§20.5). So a frontend that selects what it added has to *find* it,
//! and the one way that does not depend on where the row landed is to compare the roster
//! before the dispatch with the roster after. "The topmost matte", "the last guide" or
//! "one past its source" each name somebody else's row the moment the insertion point
//! moves, or a peer's row arrives in the same place.

/// The one id in `after` that is not in `before`.
///
/// `None` when nothing is new — the add was refused — and `None` when more than one is,
/// since then the roster moved for some other reason too and no row is the add's alone.
/// Ids that left do not count: they are not the add's either way.
pub fn minted<Id: PartialEq>(before: &[Id], after: impl IntoIterator<Item = Id>) -> Option<Id> {
    let mut new = after.into_iter().filter(|id| !before.contains(id));
    match (new.next(), new.next()) {
        (Some(id), None) => Some(id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId, GuideId, LayerId};

    /// A layer roster as the engine projects it — bottom of the document first, a
    /// group's base *before* the rows it carries — which is the order `layer_tree`'s
    /// fixtures write and the order a frontend reads the ids off in.
    fn roster(ids: &[u64]) -> Vec<LayerId> {
        ids.iter().copied().map(LayerId::solo).collect()
    }

    /// The new layer is found wherever the add put it: at the top of the document, at
    /// the bottom (where a backing is born, §15.5), and inside a group — between the base
    /// and the rows it carries, where "+ Filter" lands while the artist works in one.
    #[test]
    fn an_add_is_found_wherever_it_lands() {
        // 1 and 4 in the root stack; 2 and 3 carried on 1.
        let before = roster(&[1, 2, 3, 4]);
        for (place, after) in [
            ("the top", roster(&[1, 2, 3, 4, 9])),
            ("the bottom", roster(&[9, 1, 2, 3, 4])),
            ("inside the group", roster(&[1, 2, 9, 3, 4])),
        ] {
            assert_eq!(
                minted(&before, after),
                Some(LayerId::solo(9)),
                "the layer added at {place} was not found"
            );
        }

        // The rule has no opinion about what a row is: a duplicated guide lands one past
        // its source, mid-roster, and is found the same way.
        let guide = |lamport| {
            GuideId(ActionId {
                lamport,
                actor: ActorId(1),
            })
        };
        let before = [guide(1), guide(2), guide(3)];
        assert_eq!(
            minted(&before, [guide(1), guide(2), guide(7), guide(3)]),
            Some(guide(7)),
            "the duplicate was not found"
        );
    }

    /// A refused add — the engine declined it, or there was no engine to ask — leaves
    /// the roster as it was, and there is nothing to select.
    #[test]
    fn a_refused_add_mints_nothing() {
        let before = roster(&[1, 2, 3]);
        assert_eq!(minted(&before, before.clone()), None);
        assert_eq!(minted(&[], roster(&[])), None);
    }

    /// Two new rows are not one add: selecting either would be a guess.
    #[test]
    fn two_new_rows_are_no_one_add() {
        let before = roster(&[1, 2]);
        assert_eq!(minted(&before, roster(&[1, 2, 8, 9])), None);
        assert_eq!(minted(&[], roster(&[8, 9])), None);
    }
}
