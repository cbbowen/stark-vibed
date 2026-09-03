//! Which rows of the record registry **this** frontend keeps (§25.6).
//!
//! A module with nothing in it but a test, because the claim spans every record type
//! in the crate and there is no one of them it belongs to. It was `storage.rs`'s until
//! the format moved down to `stark_chrome` (§11.2, N1) — and the split is the point:
//! the format, the keys and the failure policy are shared, while *which* records a
//! given frontend keeps is not.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use stark_chrome::storage::{Blob, Entry, Record, Store};

    /// Rows of the registry that are some **other** frontend's, and so have no type
    /// here to claim them.
    ///
    /// One line each, listed rather than inferred, so that a row added and forgotten
    /// still fails the check below instead of being waved through as "probably
    /// somebody else's".
    const ELSEWHERE: [Store; 1] = [
        // The native frontend's window bounds. A page is where the user put the tab.
        Store::Window,
    ];

    /// Every [`Store`] is claimed by exactly one type, and every type claims one.
    ///
    /// The pairing is compile-time in one direction — a type names its record, so it
    /// cannot be read out of the wrong key — and this is the other direction, which
    /// nothing else checks: **two types naming the same variant** would overwrite each
    /// other's record, and a variant no type claims is a row of the registry that does
    /// nothing. The old `Store` argument made the first mistake unwritable only by
    /// convention and the second invisible entirely.
    ///
    /// A record added without a line here fails on the count, not on a reviewer
    /// remembering: `ALL` grows and the claims do not.
    #[test]
    fn every_record_claims_one_store() {
        let claimed = [
            <stark_chrome::identity::Stored as Record>::STORE,
            <stark_chrome::prefs::Prefs as Record>::STORE,
            <crate::commands::StoredBinding as Entry>::STORE,
            <crate::drags::DragRow as Entry>::STORE,
            <crate::visibility::StoredVisible as Entry>::STORE,
            <crate::tutor::Row as Entry>::STORE,
            <crate::shapes::StoredShape as Entry>::STORE,
            <stark_chrome::presets::StoredPreset as Entry>::STORE,
            <crate::slots::StoredSlot as Entry>::STORE,
            <crate::gradients::GradientEntry as Entry>::STORE,
            <crate::substrates::StoredSubstrate as Entry>::STORE,
        ];
        let distinct: HashSet<Store> = claimed.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            claimed.len(),
            "two types naming one record overwrite each other"
        );
        let elsewhere: HashSet<Store> = ELSEWHERE.iter().copied().collect();
        assert!(
            distinct.is_disjoint(&elsewhere),
            "a row this frontend keeps is also listed as another's"
        );
        assert_eq!(
            distinct.union(&elsewhere).copied().collect::<HashSet<_>>(),
            Store::ALL.iter().copied().collect::<HashSet<_>>(),
            "every row of the registry is some type's, and every type has a row"
        );

        // [`Blob`] is deliberately *not* one of the claims above: a record's bytes are
        // the other half of a record that already has a row, never a record of their
        // own. What is checked instead is that they cannot invent one.
        let blobs = [<crate::shapes::ShapeEntry as Blob>::STORE];
        assert!(
            blobs.iter().all(|s| distinct.contains(s)),
            "bytes belong to a record some type already claims"
        );
    }
}
