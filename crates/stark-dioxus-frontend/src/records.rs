//! Which rows of the record registry **this** frontend keeps (§25.6).
//!
//! A module with nothing in it but a test, because the claim spans every record type
//! in the crate and there is no one of them it belongs to. It was `storage.rs`'s until
//! the format moved down to `stark_ui` (§11.2, N1) — and the split is the point:
//! the format, the keys and the failure policy are shared, while *which* records a
//! given frontend keeps is not.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use serde_json::json;
    use stark_ui::storage::{self, Blob, Entry, Record, Store};
    use strum::VariantArray;

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
    /// remembering: `VARIANTS` grows and the claims do not.
    #[test]
    fn every_record_claims_one_store() {
        let claimed = [
            <stark_ui::identity::Stored as Record>::STORE,
            <stark_ui::prefs::Prefs as Record>::STORE,
            <stark_ui::commands::StoredBinding as Entry>::STORE,
            <stark_ui::drags::DragRow as Entry>::STORE,
            <stark_ui::visibility::StoredVisible as Entry>::STORE,
            <crate::tutor::Row as Entry>::STORE,
            <stark_ui::assets::Row<stark_ui::assets::Shapes> as Entry>::STORE,
            <stark_ui::presets::StoredPreset as Entry>::STORE,
            <stark_ui::slots::StoredSlot as Entry>::STORE,
            <crate::gradients::GradientEntry as Entry>::STORE,
            <stark_ui::assets::Row<stark_ui::assets::Substrates> as Entry>::STORE,
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
            Store::VARIANTS.iter().copied().collect::<HashSet<_>>(),
            "every row of the registry is some type's, and every type has a row"
        );

        // [`Blob`] is deliberately *not* one of the claims above: a record's bytes are
        // the other half of a record that already has a row, never a record of their
        // own. That used to be checked here. It is now **structural**: one type
        // carries both impls for an asset library, so its bytes cannot name a store
        // its rows do not (`stark_ui::assets::Row`). What is left to check is
        // that the two libraries are still the only blob-bearing records, since a
        // third would be a claim this list does not make.
        let blobs = [
            <stark_ui::assets::Row<stark_ui::assets::Shapes> as Blob>::STORE,
            <stark_ui::assets::Row<stark_ui::assets::Substrates> as Blob>::STORE,
        ];
        assert!(
            blobs.iter().all(|s| distinct.contains(s)),
            "bytes belong to a record some type already claims"
        );
    }

    /// One record's sample, made the way the store makes one.
    ///
    /// Through JSON rather than a struct literal because several of these types keep
    /// their fields private to the module that owns them — which is right, and which
    /// leaves a sibling crate the store's own door as the only way in. What is written
    /// out here is then literally a row as it sits in `localStorage`.
    fn sample<T: serde::de::DeserializeOwned>(row: serde_json::Value) -> T {
        serde_json::from_value(row).unwrap_or_else(|e| panic!("the sample is not a row: {e}"))
    }

    /// [`every_field_may_be_absent`](storage::every_field_may_be_absent) for a whole
    /// record, **counting which one it was**, so the coverage below is accountable to
    /// the registry rather than to whoever remembered to add a line.
    fn whole<T: Record + serde::Serialize + serde::de::DeserializeOwned>(
        sample: &T,
        exempt: &[&str],
        covered: &mut HashSet<Store>,
    ) {
        storage::every_field_may_be_absent(sample, exempt);
        covered.insert(T::STORE);
    }

    /// The same for a record that is a list.
    fn each<T: Entry + serde::Serialize + serde::de::DeserializeOwned>(
        sample: &T,
        exempt: &[&str],
        covered: &mut HashSet<Store>,
    ) {
        storage::every_field_may_be_absent(sample, exempt);
        covered.insert(T::STORE);
    }

    /// **Every field of every stored record may be absent** — the property
    /// `#[serde(default)]` buys, stated once for every row of the registry.
    /// [`every_field_may_be_absent`](storage::every_field_may_be_absent) carries the
    /// argument; what is here is the coverage and the exceptions.
    ///
    /// The **exemptions are the interesting half** — each is a field whose absence is
    /// meant to be fatal, and this is the one place such a claim is written down. Four
    /// shapes of them recur: a row's own subject, which is how it is *reached* (an
    /// asset row's `id`, a preset's `name`, a slot's `digit`); the field an untagged
    /// enum is told apart by; a key whose default would be a value no client may hold
    /// (`identity`'s `secret`); and a binding's chord, where a defaulted field would
    /// not leave a damaged shortcut but a working one somewhere else.
    ///
    /// A record's **embedded parts are walked with it**, by path — so a field added to
    /// `Modulation`, three types down inside a stored preset and in another crate,
    /// fails here rather than emptying the preset library and the quick-brush rack on
    /// the next launch.
    #[test]
    fn every_stored_field_may_be_absent() {
        use stark_ui::assets::{self, Shapes, Substrates};
        use stark_ui::brush_config::{BrushConfig, Transient};
        use stark_ui::commands::{Chord, ChordKey, Command, StoredBinding, VisibilityToggle};
        use stark_ui::drags::{DragAction, DragButton, DragChord, DragRow};
        use stark_ui::keys::Mods;
        use stark_ui::panels::PanelId;
        use stark_ui::visibility::StoredVisible;

        let id = "ab".repeat(32);
        let covered = &mut HashSet::new();

        // This client's identity. `secret` alone: `[u8; 32]::default()` is a key every
        // client shares, where a *fresh* one is safe precisely because nothing can be
        // stale against it (`identity::Stored`).
        whole(
            &sample::<stark_ui::identity::Stored>(json!({"secret": id, "boot": 3})),
            &["secret"],
            covered,
        );

        // The settings — the one all-or-nothing record here, so a field without a
        // default costs all eight rather than one.
        whole(&stark_ui::prefs::Prefs::default(), &[], covered);

        // A chord override — **with a chord in it**, since what an `Option` is empty
        // of is what the walk cannot see. `command` is the row's whole subject.
        //
        // The chord's own three are the exception to the rule this test is about, and
        // the reason is that a binding is not a value but a *place*: a row that read
        // back missing its `ctrl` would not be a damaged shortcut, it would be a
        // working one somewhere else — Undo on plain Z, firing under the hand. Losing
        // the override is the smaller harm, so its absence stays fatal. `alt` is
        // defaulted and rightly so, and does not contradict this: that column was
        // *added*, so a row without it really did mean "no Alt" (`Chord`). A fourth
        // modifier added tomorrow must default for the same reason and will fail here
        // until it does.
        each(
            &StoredBinding {
                command: Command::Undo,
                chord: Some(Chord {
                    ctrl: true,
                    shift: false,
                    alt: false,
                    key: ChordKey::Char('z'),
                }),
            },
            &["command", "chord.ctrl", "chord.shift", "chord.key"],
            covered,
        );

        // The drag table, whose two row shapes are **untagged**: which fields are
        // present is what tells them apart, so the field that discriminates is
        // load-bearing in a way a defaulted one could not be. Its press is exempt for
        // the chord's reason above, and harder: a drag binding that lost its modifiers
        // would read as a bare press on the primary button, which is not a chord this
        // table can name at all — that press *is* painting (`drags::Mods::none`).
        each(
            &DragRow::Bound {
                action: DragAction::TuneBrush,
                chord: Some(DragChord {
                    mods: Mods {
                        ctrl: true,
                        shift: false,
                        alt: false,
                    },
                    button: DragButton::Left,
                }),
            },
            &["action", "chord.mods", "chord.button"],
            covered,
        );
        each(&DragRow::Offered { offered: true }, &["offered"], covered);

        // What was on screen. `what` is the row — a row that names nothing is not a
        // thing left open, and its variant tag is not a field it could default.
        each(
            &StoredVisible {
                what: VisibilityToggle::Panel(PanelId::Layers),
                collapsed: false,
            },
            &["what"],
            covered,
        );

        // The tour's ledger, untagged like the drag table. `count` is exempt for a
        // different reason from `deed`: a tally row with no tally says nothing, so
        // dropping it and reading it as zero are the same outcome and the row may as
        // well cost itself.
        each(
            &crate::tutor::Row::Deed {
                deed: crate::tutor::Deed::Stroke,
                count: 7,
            },
            &["deed", "count"],
            covered,
        );
        each(
            &crate::tutor::Row::Given {
                given: "brush-panel".to_string(),
            },
            &["given"],
            covered,
        );

        // Both asset libraries — one type, two records, so a field without a default
        // here drops the shapes and the substrates together and strands every blob
        // they named. `id` is the whole of the reference (§19) and a default would be
        // an id no bytes are under, where `name` is a label the gallery can draw
        // empty. That is the line: a row is *reached* by its id.
        let row = json!({"name": "Bristles", "id": id});
        each(
            &sample::<assets::Row<Shapes>>(row.clone()),
            &["id"],
            covered,
        );
        each(&sample::<assets::Row<Substrates>>(row), &["id"], covered);

        // A saved preset. `name` alone — a preset is *reached* by it, which is what a
        // quick slot binds to. The brush and the tune are walked *into*, which is
        // where a field added to `Modulation` or `ColorDynamics`, three types below a
        // preset and in another crate, is caught. `shape.Round` is the one structural
        // exception: an externally tagged variant's tag is the discriminant, not a
        // field, so there is nothing there to default.
        let tune = serde_json::to_value(Transient::default()).expect("a tune encodes");
        each(
            &sample::<stark_ui::presets::StoredPreset>(
                json!({"name": "Wet Oil", "brush": BrushConfig::default(), "transient": tune}),
            ),
            &["name", "brush.shape.Round"],
            covered,
        );

        // A quick-brush slot: its digit, and — flattened, so they are this record's
        // own keys — the preset it is bound to and the tune the number keeps. The
        // first two are what the slot *is*; a rack that loses its rows reads as
        // `Some(empty)`, which never re-seeds (§18.1.8).
        each(
            &sample::<stark_ui::slots::StoredSlot>(
                json!({"digit": 0, "preset": "Wet Oil", "transient": tune}),
            ),
            &["digit", "preset"],
            covered,
        );

        // The gradient library. Both fields are the entry: a gradient is reached by
        // name — the picker keeps its selection as one — and a ramp is what the row is
        // *for*. Exempting `gradient` leaves its stops unwalked, which is the one
        // subtree of the registry this check does not reach: `Gradient` has no
        // `Default` to give the field, because a ramp of fewer than two stops is not
        // one (§22.2), and inventing a two-stop one is the model's call.
        each(&gradient_entry(), &["name", "gradient"], covered);

        let elsewhere: HashSet<Store> = ELSEWHERE.iter().copied().collect();
        assert_eq!(
            covered.union(&elsewhere).copied().collect::<HashSet<_>>(),
            Store::VARIANTS.iter().copied().collect::<HashSet<_>>(),
            "a record of the registry has no sample here, so nothing says its fields \
             may be absent",
        );
    }

    /// A two-stop ramp under a name — the gradient library's row, which has no
    /// `Default` because a ramp needs at least two stops to be one (§22.2).
    fn gradient_entry() -> crate::gradients::GradientEntry {
        use stark_model::Srgb;
        use stark_model::gradient::{Gradient, GradientStop};
        crate::gradients::GradientEntry {
            name: "Gradient 1".to_string(),
            gradient: Gradient::new(vec![
                GradientStop {
                    t: 0.0,
                    color: Srgb::new([1.0, 0.0, 0.0]),
                },
                GradientStop {
                    t: 1.0,
                    color: Srgb::new([0.0, 0.0, 1.0]),
                },
            ])
            .expect("two stops are a ramp"),
        }
    }
}
