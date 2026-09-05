//! The wire: one message vocabulary for every channel.
//!
//! Three channels, one vocabulary (§12.4):
//!
//! - **Gossip** carries [`Stamped`] messages — one committed action or
//!   presence frame each, encoded against a schema both ends hold rather than one
//!   travelling with the message ([`codec`](crate::codec) says why). Actions are
//!   small (fitted control points, ids, params); pixels and image bytes never ride
//!   gossip, whatever kind of content names them.
//! - **The session [`ALPN`]** answers [`Request`]s over one bi-stream per
//!   request: the full session [`Snapshot`](Request::Snapshot) (the save-format
//!   container, assets bundled) for joins. Every response opens with a one-byte
//!   [tag](Tag) saying whether what follows is an answer at all.
//! - **The `iroh-blobs` ALPN** serves individual pieces of content to peers that
//!   see an action referencing one they don't hold — a brush image, a canvas
//!   substrate, a placed picture (§23) — hash-verified, addressed by the blob hash a
//!   [`Stamped`] message carries alongside such an action.
//!
//! The endpoint that *answers* these requests — the accept loop and the
//! client-side request call — is [`proto`](crate::proto).

use serde::{Deserialize, Serialize};
use stark_model::AssetId;
use stark_model::document::{Action, ActionId};
use stark_model::peer::PeerFrame;

use iroh::EndpointId;

/// The catch-up (snapshot) protocol. The trailing number moves with the wire:
/// gossip payloads carry no schema of their own, so two builds whose action shape
/// differs must fail to *meet* rather than decode each other's messages wrong.
///
/// **This is the one version number left, and the save format no longer has a sibling
/// for it** (§8). A file carries its schema and is reconciled by name, so it needs no
/// number; a message is encoded against the schema the far end is assumed to hold, so
/// it does — the trade [`codec`](crate::codec) explains. Bump it whenever the *shape*
/// of anything gossip carries changes: an action's fields or variants, a presence
/// frame's, or the envelope's.
///
/// **And whenever an action's *meaning* changes, which the shape rule does not
/// cover.** Retiring an action is the case that made this worth saying: §19 lets a
/// file keep opening while producing something different, so an action is retired by
/// keeping its variant and taking away what it does (`ActionKind`'s tombstone rule).
/// The wire shape is then untouched — and two peers, one still applying it and one
/// ignoring it, would fold the same log into two different documents, with pixels
/// unable to show which path ran (§12.6). A file may be read by a build that disagrees
/// with it; a *session* may not, which is what this number is for. Past bumps (1: `FillOp`'s parcel; 2: the matte's paint
/// and anchor, §22.4, §15.4; 3: a fill's strength became one field and a coverage,
/// §6.8; 4: `SelectionOp` gained its opacity, §6.8; 5: `BlendMode::Drago` gained its
/// bend, §6.3; 6: a response opens with a [`Tag`], so a member with nothing to serve
/// can say so instead of answering; 7: carbonite replaced postcard, so every payload
/// is columnar; 8: `BrushParams::drain` became a rate per *radius* rather than per
/// canvas px, §6.2 — the field's shape untouched and every stroke already in a log
/// rendering differently, which is the meaning rule above and not the shape one;
/// 9: the drawing guides became document state, so `ActionKind` gained five
/// variants, §20.5 — the plainest kind of shape change, and one an older peer
/// would decode as some other action entirely;
/// 10: how large the canvas substrate is laid became document state, so `ActionKind`
/// gained `SetSubstrateScale` — and *inserted*, beside the `SetSubstrate` it is the other
/// half of, which shifts the index of every variant after it, §6.4;
/// 11: `BrushParams` gained `jitter`, §6.2 — a plain shape change to a
/// struct every stroke gossips;
/// 12: what a brush *does* became `BrushEffect` — `Paint` or the new `Erase`
/// (§6.12) — with the dynamics, the color dynamics and the modulations
/// regrouped around it, the tooth's pair folded into `ToothParams`, both
/// effects carrying an `opacity` ceiling (§6.2), the brush color losing
/// its per-unit alpha to it and then moving inside `PaintEffect` — an erasing
/// brush carries no pigment at all: one reshaping of the struct every
/// stroke gossips);
/// 13: the selection mask gained an opacity of its own, set after the region
/// is drawn, so `ActionKind` gained `SetSelectionOpacity` (§6.8) — and the
/// mask stopped being a lerp of a stroke's result and became the other
/// factor of its opacity ceiling (§6.2), which is the meaning rule: the same
/// log under a feathered selection renders differently;
/// 14: a universal mask keeps its opacity (§6.8) — `SetSelectionOpacity` with
/// nothing selected stores where it used to discard, every later stroke and
/// the next region read at it, and a deselect is what lands on full strength:
/// the meaning rule again, with no shape touched;
/// 15: `RemoveLayer` carries the subtree it takes, so it can declare the layers
/// it writes (§12.6) — a reshaping *and* a meaning change, since a removal
/// whose named subtree is not what the document holds is now declined. A build
/// that sent the old shape would have its group removals read as a bare id;
/// 16: a `LayerId` is the **action that minted it**, not a counter partitioned by a
/// 32-bit fold of the actor (§17.9) — a reshaping of the one type every action that
/// names a layer carries, and a meaning change besides: two peers whose folds
/// coincided used to mint colliding ids, which is what the shape now rules out. A
/// build that sent the old shape would have its layer ids read as an action id and a
/// `k` cut out of the middle of one;
/// 17: `AddGuide` carries its `GuideId` where the fold used to derive it from
/// `action.id` (§20.5, §17.9) — a field added to a variant every guide gossips, and
/// the fix for a defect the derivation had: `start_collaboration` rewrites
/// solo-authored `ActionId`s, which moved a derived guide id out from under every
/// later action naming it;
/// 18: the wet-mixing loop became its own effect — `BrushEffect` gained `Wet`,
/// `PaintEffect` lost its `dynamics` to it and kept a bare `flow` (§6.2): a
/// reshaping of the struct every stroke gossips, with no migration — the alpha
/// rule (§19);
/// 19: `BrushEffect` gained `Liquify` (§6.13) — a variant added to the enum
/// every stroke gossips, which reshapes it on the wire even though every
/// existing stroke encodes as it did: a peer without the variant cannot decode
/// a stroke that carries it, and there is no schema on the wire to say so;
/// 20: the wet flow/add split (§6.2) — `WetEffect` gained its own `flow` (the
/// overall rate), `BrushDynamics::flow` became `add` (the source share, [0, 1]),
/// and `WetModulations` gained an `add` target: a reshaping of the struct every
/// wet stroke gossips, and a meaning change besides — the same bytes at the old
/// shape would read a source rate as a λ multiplier. No migration — the alpha
/// rule (§19);
/// 21: layers gained a frame (§14.12) — `ActionKind` gained `TranslateLayers`
/// and `FloatSelection`, every paint action and the presence stroke head and
/// fill frame gained a `frame` offset, and paint geometry is stated in the
/// layer's frame rather than on the canvas: a meaning change with the shape
/// nearly untouched, since a build without the field would fold a framed
/// stroke's path at the wrong place and gate it through an unshifted mask;
/// 22: mattes joined the frame (§15.2) — `TranslateLayers` moves a matte, and a
/// matte's rect and gradient axis are stated in the layer's frame rather than
/// on the canvas: a meaning change with no shape change at all, since a build
/// without it folds the move as a no-op and reads a translated matte's
/// `SetMatteRect` at the wrong place;
/// 23: sweeps gained a digest pre-check (§12.5) — `Request` gained `Digest`,
/// answered with a [`LogDigest`] (count + XOR of per-id BLAKE3), so a sweep
/// between identical logs costs tens of bytes instead of the id list — and
/// the client's read ceiling became per-request
/// ([`Request::response_ceiling`]). Inserted beside the [`Ids`](Request::Ids)
/// it pre-checks, which shifts the index of every variant after it — an older
/// peer would decode `Digest` as its `Ids` and answer with the megabyte id
/// list the digest exists to avoid;
/// 24: the opacity ceiling became a pen target (§6.2) — `PaintModulations`,
/// `WetModulations` and `EraseModulations` each gained `opacity`: a field added
/// to a struct every stroke gossips, which a file fills from its default and
/// the wire cannot;
/// 25: `Srgb` widened past the cube (§6.5). The same bytes in the same fields,
/// which a peer on 24 clamps — the same log, a narrower picture of it, which is
/// the disagreement the ALPN exists to keep from meeting.
pub(crate) const ALPN: &[u8] = b"stark/collab/25";

/// The number [`ALPN`] ends with, as a number, for a ticket to carry — see
/// `ticket`'s `TicketBody::proto` for why a link names it. Kept in step with
/// [`ALPN`] by a test rather than by building the byte-string from it: two
/// tokens side by side are not worth the compile-time ceremony.
pub(crate) const PROTO: u32 = 25;

/// Upper bound on an encoded request, over any transport.
///
/// A request carries the joiner's list of resolvable content ids, 32 bytes each,
/// so the ceiling has to clear a catalog that grows rather than the variant tag
/// alone: 64 KiB is two thousand of them.
pub(crate) const MAX_REQUEST: usize = 64 * 1024;

/// How many missing actions a reconciling member names per [`Request::Actions`]
/// — sized against [`MAX_REQUEST`]: an `ActionId` is two u64s, encoded
/// fixed-width, so 2048 × 16 B = 32 KiB < 64 KiB.
pub(crate) const RECOVER_BATCH: usize = 2048;

/// Upper bound on a snapshot response: a whole session (log + content
/// payloads). A session that outgrows it stops accepting new members, so
/// crossing most of the way there is worth saying out loud while joining still
/// works (`proto`'s warn).
pub(crate) const MAX_SNAPSHOT_RESPONSE: usize = 64 * 1024 * 1024;

/// The first byte of every response.
///
/// A refusal has to be distinguishable from an answer *in the payload*, because
/// the one thing it must never be confused with is the empty session — which is
/// what a peer still fetching its own snapshot would otherwise hand over, encoded
/// perfectly, indistinguishable from a document with nothing in it yet.
///
/// A byte rather than a `Response` enum wrapping the payload: a snapshot is
/// megabytes, and nesting it inside another encoded value would copy all of it to
/// say one thing about it.
pub(crate) struct Tag;

impl Tag {
    /// What follows is the answer to the request.
    pub const OK: u8 = 0;
    /// The responder is still joining and has no session to serve
    /// ([`NetError::NotReady`](crate::NetError::NotReady)). Nothing follows.
    pub const NOT_READY: u8 = 1;
}

/// One gossip broadcast: the payload plus who authored it.
///
/// Gossip forwards messages through intermediate peers and reports only the
/// *delivering* neighbor, so the author travels in the payload. It is
/// self-declared — the same trust already placed in the payload itself, since
/// anyone holding the ticket can write anything (§12.5 defers
/// authentication).
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
pub(crate) struct Stamped {
    /// Who produced the message — the authoritative source for anything it
    /// references (a presence frame's author, a stroke's brush asset).
    ///
    /// `carbonite(serde)` on this field and the next: an `EndpointId` and a blob hash
    /// are iroh's types, so the orphan rule puts a compile-time schema for them out of
    /// reach here and each is described by a memoized trace of its own `Deserialize`
    /// instead. Both are `[u8; 32]` underneath, so nothing about the shape changes.
    #[carbonite(serde)]
    pub origin: EndpointId,
    /// The blob hash of the brush image the payload references, if any. An
    /// [`AssetId`] names the *decoded coverage* (encoding-independent), so it
    /// is not itself fetchable over blobs — the author, who holds the bytes,
    /// supplies the transfer hash here. Trusted like the rest of the payload;
    /// the engine re-derives the real `AssetId` from the fetched bytes.
    #[carbonite(serde)]
    pub asset: Option<iroh_blobs::Hash>,
    pub wire: Wire,
}

/// [`Stamped`], borrowing what it sends.
///
/// It exists so that publishing an action does not duplicate it. The mirror keeps
/// the action (a joiner needs it whether or not the send succeeds) and the wire
/// encodes it, and without this those are two copies of a stroke's control points.
///
/// **It presents itself as a `Stamped`**, name for name, and that is what makes a
/// second spelling safe. One schema serves both
/// ([`encode_stamped_ref`](crate::codec::encode_stamped_ref)), and encoding checks the
/// struct name and every field and variant name as it goes — so a field reordered,
/// renamed or added in one and not the other fails to encode here rather than shipping
/// bytes the far end reads as something else. Under the positional format that came
/// before, the two were kept in step by hand, and the first symptom of a slip was a
/// decode failure on somebody else's machine.
///
/// It carries no schema of its own — it borrows, and a compile-time schema is only
/// defined for owned types. None is needed: it is written *against* `Stamped`'s
/// ([`encode_stamped_ref`](crate::codec::encode_stamped_ref)), which is the same
/// statement the `serde(rename)` makes.
#[derive(Debug, Serialize)]
#[serde(rename = "Stamped")]
pub(crate) struct StampedRef<'a> {
    pub origin: EndpointId,
    pub asset: Option<iroh_blobs::Hash>,
    pub wire: WireRef<'a>,
}

/// [`Wire`], borrowing its payload. See [`StampedRef`].
#[derive(Debug, Serialize)]
#[serde(rename = "Wire")]
pub(crate) enum WireRef<'a> {
    Action(&'a Action),
    Presence(&'a PeerFrame),
}

/// A live-wire message, inside [`Stamped`].
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
pub(crate) enum Wire {
    /// A freshly committed action for the shared log.
    Action(Action),
    /// One client's presence: cursor, selected layer, the gesture it is drawing
    /// (§17.4). **Never mirrored and never snapshotted** — it is not
    /// part of the document, and nothing in the log refers to it, which is the whole
    /// reason it may be dropped, coalesced or delayed without affecting convergence.
    ///
    /// The author is not in the frame: the receiver takes it from the
    /// [`Stamped`] envelope, whose `origin` names exactly one author
    /// (§17.7).
    Presence(PeerFrame),
}

/// A request over the collab ALPN (one per bi-stream; the response is the
/// stream's full contents).
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
pub(crate) enum Request {
    /// The whole session: a [`DocumentFile`](stark_model::DocumentFile) container.
    Snapshot,
    /// The session, minus the content the joiner says it can resolve without help
    /// — the ids of the assets that ship with its build (§12.4).
    ///
    /// A separate variant rather than a field on [`Request::Snapshot`], which is now
    /// a statement about *peers* rather than about bytes: a variant is matched by name
    /// (§8), so an older peer that only knows `Snapshot` keeps working, and a newer
    /// one asking an older host gets a decode error on a request rather than a
    /// silently full bundle. The [`ALPN`] is what should have caught it first.
    ///
    /// The list is a **promise**, not an inventory — "I can get these", not "I
    /// have these loaded". The joiner has to make it good before replaying, and
    /// the blob fetch is what catches it if it cannot.
    SnapshotWithout(Vec<AssetId>),
    /// This member's whole log, summarized for equality — the pre-check that lets
    /// a sweep between identical logs close for tens of bytes instead of
    /// [`Ids`](Request::Ids)' megabyte-class answer (see
    /// [`reconcile`](crate::reconcile)). Answered with an encoded [`LogDigest`].
    Digest,
    /// Every action id this member holds, in total order — the full half of
    /// reconciliation (see [`reconcile`](crate::reconcile)). Answered with an encoded
    /// `Vec<ActionId>`.
    ///
    /// The whole list, rather than a summary. A per-actor high-water mark would be
    /// a sixteenth the size and would not work: `ActionId` is `(lamport, actor)`
    /// and a lamport clock jumps when its owner observes someone else, so an
    /// actor's ids are sparse and a mark cannot see a hole in the middle of them.
    /// At 16 bytes each a hundred thousand actions is 1.6 MB, sent rarely.
    Ids,
    /// The named actions, for the ids a reconciling member found it was missing.
    /// Answered with an encoded `Vec<`[`Recovered`]`>`.
    Actions(Vec<ActionId>),
}

impl Request {
    /// What the client-side read accepts as this request's answer (`proto`'s
    /// `request`). Per request, because the honest answers differ by six orders
    /// of magnitude, and one ceiling sized for the snapshot would let a digest
    /// answer be 64 MiB of something else.
    pub(crate) fn response_ceiling(&self) -> usize {
        match self {
            Request::Snapshot | Request::SnapshotWithout(_) => MAX_SNAPSHOT_RESPONSE,
            // Recovery batches; the server truncates its answers well under
            // this (`proto`'s `MAX_RECOVER_RESPONSE` is defined as a fraction
            // of it, so the two cannot invert), so the headroom is free.
            Request::Actions(_) => MAX_ACTIONS_RESPONSE,
            // Half a million ids at 16 B each — five times the design target.
            Request::Ids => 8 * 1024 * 1024,
            // 40 bytes of digest plus framing.
            Request::Digest => 1024,
        }
    }
}

/// The client-side read ceiling on a [`Request::Actions`] answer. Named so the
/// server's truncation bound can be defined against it — a truncation past the
/// read ceiling would turn every large recovery into a client-side error,
/// rebuilt identically each sweep.
pub(crate) const MAX_ACTIONS_RESPONSE: usize = 32 * 1024 * 1024;

/// A member's action log, summarized for equality: how many ids it holds and
/// the XOR of the BLAKE3 of each. Ids are inserted once and never removed, so
/// XOR is a sound set digest — order-independent, with nothing to un-fold
/// ([`Mirror`](crate::mirror::Mirror) maintains it per insertion). `count`
/// guards the honest XOR collision; against *chosen* ids the digest is only as
/// trustworthy as the self-declared ids themselves (§12.5 defers
/// authentication), so it is an optimization, never an integrity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub(crate) struct LogDigest {
    pub count: u64,
    pub xor: [u8; 32],
}

/// An action as it travels during reconciliation: the action, and the transfer
/// hash for whatever content it names.
///
/// The hash has to come with it. A recovered action goes through the same door as
/// one off the flood, and that door needs to know what to fetch — without it a
/// recovered `SetSubstrate` would be applied against the flat stand-in, which is the
/// divergence reconciliation exists to undo (§6.4).
///
/// A named pair rather than the tuple it was, because a tuple has no *field* for the
/// hash's `carbonite(serde)` to sit on (§8) — and because "the action, and the hash for
/// what it names" is worth saying in the type rather than at every destructuring.
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
pub(crate) struct Recovered {
    pub action: Action,
    /// The transfer hash for the content the action names, if it names any.
    #[carbonite(serde)]
    pub hash: Option<iroh_blobs::Hash>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::action;

    /// [`StampedRef`] is a second spelling of [`Stamped`], and what makes a second
    /// spelling safe is that the two encode identically — asserted here, and checked
    /// again by the encoder itself, which matches the schema's names against the
    /// value's as it writes (see [`StampedRef`]).
    #[test]
    fn the_borrowed_encoding_is_the_owned_one() {
        let origin = iroh::SecretKey::from_bytes(&[5u8; 32]).public();
        let asset = Some(iroh_blobs::Hash::new(b"a brush"));
        let action = action(7);
        let frame = PeerFrame {
            boot: 3,
            seq: 9,
            name: Some("someone".into()),
            active_layer: stark_model::document::LayerId::ROOT,
            cursor: Some(stark_model::geom::Vec2::new(1.0, 2.0)),
            gesture: None,
            leaving: false,
        };

        let owned = |wire| {
            crate::codec::encode(&Stamped {
                origin,
                asset,
                wire,
            })
            .expect("encode")
        };
        let borrowed = |wire| {
            crate::codec::encode_stamped_ref(&StampedRef {
                origin,
                asset,
                wire,
            })
            .expect("encode")
        };

        // Both variants, so each one's payload is pinned and not just the envelope.
        let bytes = borrowed(WireRef::Action(&action));
        assert_eq!(bytes, owned(Wire::Action(action.clone())));
        assert_eq!(
            borrowed(WireRef::Presence(&frame)),
            owned(Wire::Presence(frame))
        );

        let back: Stamped = crate::codec::decode(&bytes).expect("decode the borrowed form");
        assert_eq!(back.origin, origin);
        assert!(matches!(back.wire, Wire::Action(a) if a.id == action.id));
    }

    /// [`PROTO`] is the number [`ALPN`] ends with — with its separator, so a
    /// one-digit `PROTO` cannot pass by matching the tail of a longer number.
    /// The test is what keeps the two from drifting (see [`PROTO`]).
    #[test]
    fn the_alpn_ends_with_proto() {
        let tail = format!("/{PROTO}");
        assert!(
            ALPN.ends_with(tail.as_bytes()),
            "ALPN {:?} does not end with {tail:?}",
            std::str::from_utf8(ALPN),
        );
    }
}
