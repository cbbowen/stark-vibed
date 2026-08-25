//! The document's vocabulary as **data**: every [`ActionKind`] there is, in one
//! order, behind the one exhaustive match that stops compiling when a variant
//! appears.
//!
//! `ActionKind::label` is already exhaustive, and a new variant already has to be
//! given a caption there. But that match is in the model's *source*, so it forces
//! nothing about the model's *tests*: the suites need the roster the other way
//! round — a list they can iterate, and a way to be stopped when it is short.
//! [`slot`] is the stopper and [`LABELS`] is the list.
//!
//! # Why here
//!
//! Two suites in two crates need it, and neither crate can see the other's tests.
//! `stark-model`'s `action_kinds.rs` asks whether the funnel reaches every payload
//! that carries a number (§21.5, §8); `stark-engine`'s `footprint.rs` asks whether
//! a *run* reached every kind at all (§12.6), and needs a GPU to ask it, which is
//! why the first cannot simply live beside the second.
//!
//! So it was written twice, and the copies drifted the first time a variant landed
//! after them. `SetSelectionOpacity` (§6.8) got the arm the compiler demanded in
//! each `slot` — and was left out of every list those arms index, in both crates at
//! once. Both suites went on passing, having never once driven the new kind. This
//! is the same containment [`assets`](crate::assets) is: the coupling exists once,
//! in the crate whose whole job is fixtures, instead of once per reader.
//!
//! # The chain
//!
//! What makes it whole rather than merely shared is that no link can be left
//! half-done. A new variant forces an arm in [`slot`]; the arm has to name a
//! caption; [`at`] is a `const fn`, so a caption [`LABELS`] does not hold is a
//! **compile error** at that arm; extending `LABELS` moves [`KINDS`]; and `KINDS`
//! is the declared length of the one-of-each array `action_kinds.rs` drives. The
//! numbers are gone from all of it — the arms name captions, and the roster is the
//! only place an order is written down.

use stark_model::document::ActionKind;

/// How many kinds there are: [`LABELS`]'s length, and so the length of the
/// one-of-each list a suite drives from it.
pub const KINDS: usize = 33;

/// Every kind there is, by `ActionKind::label`'s own caption — which is what a
/// reader of a failure will go looking for, and which
/// `the_list_holds_one_of_every_kind` pins to the model's own captions rather than
/// leaving as a second set of names to keep in step.
///
/// In `ActionKind`'s declared order. Nothing rests on that — a caption is the key
/// here and the indices are private to the suites — but a roster wants *an* order,
/// and the enum's is the one already written down.
pub const LABELS: [&str; KINDS] = [
    "Stroke",
    "Add layer",
    "Remove layer",
    "Blend mode",
    "Layer opacity",
    "Layer visibility",
    "Reorder layer",
    "Undo",
    "Canvas substrate",
    "Substrate scale",
    "Select",
    "Invert selection",
    "Selection opacity",
    "Add matte",
    "Move frame",
    "Matte paint",
    "Canvas color",
    "Transform",
    "Rename layer",
    "Fill",
    "Clip layer",
    "Perspective",
    "Warp",
    "Duplicate layer",
    "Add filter",
    "Filter",
    "Merge down",
    "Place image",
    "Add guide",
    "Remove guide",
    "Perspective guide",
    "Rename guide",
    "Reorder guide",
];

/// The slot of every action kind — its place in [`LABELS`].
///
/// **Exhaustive, with no `_` arm, and that is the whole point of it.** A wildcard
/// would hand every future variant a slot it never asked for, and both suites would
/// pass having never driven it: that is how `MergeLayerDown`, `AddFilter` and
/// `SetFilter` came to be unchecked in the engine's run, and how the model's funnel
/// test came to be a hand-written 24 of 31. It is the device `Modulations::all`'s
/// `..`-free destructure already uses for a struct's fields.
pub fn slot(kind: &ActionKind) -> usize {
    match kind {
        ActionKind::CommitStroke(_) => const { at("Stroke") },
        ActionKind::AddLayer { .. } => const { at("Add layer") },
        ActionKind::RemoveLayer(_) => const { at("Remove layer") },
        ActionKind::SetLayerBlend(..) => const { at("Blend mode") },
        ActionKind::SetLayerOpacity(..) => const { at("Layer opacity") },
        ActionKind::SetLayerVisible(..) => const { at("Layer visibility") },
        ActionKind::MoveLayer { .. } => const { at("Reorder layer") },
        ActionKind::Undo(_) => const { at("Undo") },
        ActionKind::SetSubstrate(_) => const { at("Canvas substrate") },
        ActionKind::SetSubstrateScale(_) => const { at("Substrate scale") },
        ActionKind::Select(_) => const { at("Select") },
        ActionKind::InvertSelection => const { at("Invert selection") },
        ActionKind::SetSelectionOpacity(_) => const { at("Selection opacity") },
        ActionKind::AddMatte { .. } => const { at("Add matte") },
        ActionKind::SetMatteRect(..) => const { at("Move frame") },
        ActionKind::SetMattePaint(..) => const { at("Matte paint") },
        ActionKind::SetSubstrateColor(_) => const { at("Canvas color") },
        ActionKind::Transform { .. } => const { at("Transform") },
        ActionKind::SetLayerName(..) => const { at("Rename layer") },
        ActionKind::Fill { .. } => const { at("Fill") },
        ActionKind::SetLayerClip(..) => const { at("Clip layer") },
        ActionKind::TransformPerspective { .. } => const { at("Perspective") },
        ActionKind::TransformWarp { .. } => const { at("Warp") },
        ActionKind::DuplicateLayer { .. } => const { at("Duplicate layer") },
        ActionKind::AddFilter { .. } => const { at("Add filter") },
        ActionKind::SetFilter(..) => const { at("Filter") },
        ActionKind::MergeLayerDown { .. } => const { at("Merge down") },
        ActionKind::PlaceImage { .. } => const { at("Place image") },
        ActionKind::AddGuide { .. } => const { at("Add guide") },
        ActionKind::RemoveGuide(_) => const { at("Remove guide") },
        ActionKind::SetGuide(..) => const { at("Perspective guide") },
        ActionKind::SetGuideName(..) => const { at("Rename guide") },
        ActionKind::MoveGuide { .. } => const { at("Reorder guide") },
    }
}

/// Where `name` sits in [`LABELS`].
///
/// A `const fn`, and called from [`slot`]'s arms inside `const { .. }` so the
/// compiler is the one that runs it: a caption the roster does not hold fails the
/// **build**, at the arm that named it. That is the link the whole chain hangs
/// from — a slot resolved at run time would let a new kind sit unlisted for as long
/// as no test happened to reach it, which is exactly the state this file was
/// written to end.
const fn at(name: &str) -> usize {
    let mut i = 0;
    while i < KINDS {
        if same(LABELS[i], name) {
            return i;
        }
        i += 1;
    }
    panic!("no such caption in LABELS — a kind was given an arm but no place in the roster")
}

/// `str`'s own `==` is not a `const fn`, and this is only ever run by the compiler.
const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
