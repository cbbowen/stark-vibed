//! The wire: message formats and the catch-up/asset protocol.
//!
//! Three channels, one vocabulary (§12.4):
//!
//! - **Gossip** carries [`Stamped`] messages — one committed action or
//!   presence frame each, postcard-encoded. Actions are small (fitted control
//!   points, ids, params); pixels and image bytes never ride gossip.
//! - **The `stark/collab/6` ALPN** answers [`Request`]s over one bi-stream per
//!   request: the full session [`Snapshot`](Request::Snapshot) (the save-format
//!   container, assets bundled) for joins. Every response opens with a one-byte
//!   [tag](Tag) saying whether what follows is an answer at all.
//! - **The `iroh-blobs` ALPN** serves individual brush images to peers that
//!   see a stroke referencing one they don't hold — hash-verified, addressed
//!   by the blob hash a [`Stamped`] message carries alongside such a stroke.

use std::sync::Mutex;

use bytes::Bytes;
use iroh::EndpointId;
use serde::{Deserialize, Serialize};
use stark_engine::peer::PeerFrame;
use stark_model::AssetId;
use stark_model::document::{Action, ActionId};

use crate::mirror::{Mirror, Served};

/// The catch-up (snapshot) protocol. The trailing number moves with the wire:
/// gossip payloads carry no version of their own, so two builds whose action
/// encoding differs must fail to *meet* rather than decode each other's
/// messages wrong — bumped with `WIRE_VERSION` whenever an action reshapes
/// (1: `FillOp`'s parcel; 2: the matte's paint and anchor, §22.4, §15.4;
/// 3: a fill's strength became one field and a coverage, §6.8; 4: `SelectionOp`
/// gained its opacity, §6.8; 5: `BlendMode::Drago` gained its bend, §6.3;
/// 6: a response opens with a [`Tag`], so a member with nothing to serve can say
/// so instead of answering).
pub(crate) const ALPN: &[u8] = b"stark/collab/6";

/// The first byte of every response.
///
/// A refusal has to be distinguishable from an answer *in the payload*, because
/// the one thing it must never be confused with is the empty session — which is
/// what a peer still fetching its own snapshot would otherwise hand over, encoded
/// perfectly, indistinguishable from a document with nothing in it yet.
///
/// A byte rather than a `Response` enum wrapping the payload: a snapshot is
/// megabytes, and nesting it inside another postcard value would copy all of it to
/// say one thing about it.
pub(crate) struct Tag;

impl Tag {
    /// What follows is the answer to the request.
    pub const OK: u8 = 0;
    /// The responder is still joining and has no session to serve
    /// ([`NetError::NotReady`](crate::NetError::NotReady)). Nothing follows.
    pub const NOT_READY: u8 = 1;
}

/// One gossip broadcast: the payload plus who authored it. Postcard-encoded.
///
/// Gossip forwards messages through intermediate peers and reports only the
/// *delivering* neighbor, so the author travels in the payload. It is
/// self-declared — the same trust already placed in the payload itself, since
/// anyone holding the ticket can write anything (§12.5 defers
/// authentication).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Stamped {
    /// Who produced the message — the authoritative source for anything it
    /// references (a presence frame's author, a stroke's brush asset).
    pub origin: EndpointId,
    /// The blob hash of the brush image the payload references, if any. An
    /// [`AssetId`] names the *decoded coverage* (encoding-independent), so it
    /// is not itself fetchable over blobs — the author, who holds the bytes,
    /// supplies the transfer hash here. Trusted like the rest of the payload;
    /// the engine re-derives the real `AssetId` from the fetched bytes.
    pub asset: Option<iroh_blobs::Hash>,
    pub wire: Wire,
}

/// [`Stamped`], borrowing what it sends.
///
/// Postcard writes fields in declaration order and an enum by variant index, and a
/// reference writes what it points at — so this produces byte-identical output to
/// [`Stamped`] and is decoded by it. **The field and variant order of the two pairs
/// must stay in step**; `the_borrowed_encoding_is_the_owned_one` is what says so.
///
/// It exists so that publishing an action does not duplicate it. The mirror keeps
/// the action (a joiner needs it whether or not the send succeeds) and the wire
/// encodes it, and without this those are two copies of a stroke's control points.
#[derive(Debug, Serialize)]
pub(crate) struct StampedRef<'a> {
    pub origin: EndpointId,
    pub asset: Option<iroh_blobs::Hash>,
    pub wire: WireRef<'a>,
}

/// [`Wire`], borrowing its payload. See [`StampedRef`].
#[derive(Debug, Serialize)]
pub(crate) enum WireRef<'a> {
    Action(&'a Action),
    Presence(&'a PeerFrame),
}

/// A live-wire message. Postcard-encoded, inside [`Stamped`].
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Request {
    /// The whole session: a [`DocumentFile`](stark_engine::DocumentFile) container.
    Snapshot,
    /// The session, minus the content the joiner says it can resolve without help
    /// — the ids of the assets that ship with its build (§12.4).
    ///
    /// A separate variant rather than a field on [`Request::Snapshot`] because
    /// postcard encodes enums by index and appending a variant is the safe
    /// change (§8): an older peer that only knows `Snapshot` keeps working, and a
    /// newer one asking an older host gets a decode error on a request rather
    /// than a silently full bundle.
    ///
    /// The list is a **promise**, not an inventory — "I can get these", not "I
    /// have these loaded". The joiner has to make it good before replaying, and
    /// the blob fetch is what catches it if it cannot.
    SnapshotWithout(Vec<AssetId>),
    /// Every action id this member holds, in total order — the digest half of
    /// reconciliation (see [`reconcile`](crate::reconcile)). Answered with a
    /// postcard `Vec<ActionId>`.
    ///
    /// The whole list, rather than a summary. A per-actor high-water mark would be
    /// a sixteenth the size and would not work: `ActionId` is `(lamport, actor)`
    /// and a lamport clock jumps when its owner observes someone else, so an
    /// actor's ids are sparse and a mark cannot see a hole in the middle of them.
    /// At 16 bytes each a hundred thousand actions is 1.6 MB, sent rarely.
    Ids,
    /// The named actions, for the ids a reconciling member found it was missing.
    /// Answered with a postcard `Vec<`[`Recovered`]`>`.
    Actions(Vec<ActionId>),
}

/// An action as it travels during reconciliation: the action, and the transfer
/// hash for whatever content it names.
///
/// The hash has to come with it. A recovered action goes through the same door as
/// one off the flood, and that door needs to know what to fetch — without it a
/// recovered `SetSurface` would be applied against the flat stand-in, which is the
/// divergence reconciliation exists to undo (§6.4).
pub(crate) type Recovered = (Action, Option<iroh_blobs::Hash>);

/// Answer one request from the shared [`Mirror`] — every peer is a provider, so
/// the session survives the original sharer leaving. `None` while this peer is
/// still joining and has no session of its own to serve (see [`Served`]).
///
/// This is the whole protocol; the transports below only move the bytes.
pub(crate) fn answer(served: &Served, req: Request) -> crate::Result<Option<Bytes>> {
    let Some(mirror) = served.get() else {
        return Ok(None);
    };
    Ok(Some(match req {
        Request::Snapshot => snapshot_bytes(mirror, &[])?,
        Request::SnapshotWithout(have) => snapshot_bytes(mirror, &have)?,
        // Both cheap enough to answer under the lock: one walks the log's keys,
        // the other looks up as many actions as the asker found it was missing.
        Request::Ids => {
            let ids = mirror.lock().expect("mirror poisoned").action_ids();
            postcard::to_allocvec(&ids)?.into()
        }
        Request::Actions(ids) => {
            let actions = mirror.lock().expect("mirror poisoned").recover(&ids);
            postcard::to_allocvec(&actions)?.into()
        }
    }))
}

/// Encode the session snapshot, leaving out any content in `have`.
///
/// Three phases, and the lock covers only the first and last. Taking the snapshot
/// is a handful of refcount bumps — the log is persistent — while turning it into
/// a [`DocumentFile`] copies every asset payload (the container owns its bytes) and
/// postcards the lot. A joiner arriving mid-session must not stall this peer's
/// receive loop for the length of that, and the next joiner should not repeat it.
fn snapshot_bytes(mirror: &Mutex<Mirror>, have: &[AssetId]) -> crate::Result<Bytes> {
    let snapshot = {
        let mirror = mirror.lock().expect("mirror poisoned");
        match mirror.encoded_for(have) {
            Some(bytes) => return Ok(bytes),
            None => mirror.snapshot(),
        }
    };
    let revision = snapshot.revision;
    let bytes = Bytes::from(snapshot.without(have).into_file().to_bytes()?);
    mirror
        .lock()
        .expect("mirror poisoned")
        .remember(revision, have, bytes.clone());
    Ok(bytes)
}

/// Decode a request received over any transport.
pub(crate) fn decode_request(bytes: &[u8]) -> crate::Result<Request> {
    Ok(postcard::from_bytes(bytes)?)
}

/// The iroh plumbing: the protocol handler and the client-side request call.
pub(crate) use iroh_wire::{CollabProto, request};

/// The one property the catch-up wire has to have that an empty document cannot
/// express: "I have nothing *yet*" is not "I have nothing *in* me".
///
/// Over real endpoints, because that is where the two are told apart — the
/// distinction is made by the response's first byte, and a test of `answer` alone
/// would not go near it. No GPU: this is about the wire.
#[cfg(test)]
mod tests {
    use stark_engine::DocumentFile;
    use stark_model::document::{Action, ActionId, ActionKind, ActorId};

    use super::*;
    use crate::backend::{self, Bound};
    use crate::session::NetOptions;

    async fn bound(served: Served) -> Bound {
        backend::bind(served, &NetOptions::local())
            .await
            .expect("bind a local endpoint")
    }

    /// Ask `host` for a snapshot, the way a joiner does.
    async fn ask(asker: &Bound, host: &Bound) -> crate::Result<Vec<u8>> {
        let addr = host
            .dialer
            .ticket_addr(&NetOptions::local())
            .await
            .expect("ticket address");
        let catchup = asker.dialer.open(addr).await.expect("dial the collab ALPN");
        let answer = catchup.request(Request::Snapshot).await;
        catchup.close().await;
        answer
    }

    fn action(lamport: u64) -> Action {
        Action {
            id: ActionId {
                lamport,
                actor: ActorId(1),
            },
            kind: ActionKind::SetBackground([0.0; 3]),
        }
    }

    /// [`StampedRef`] is a second spelling of [`Stamped`], and the only thing that
    /// makes a second spelling safe is that postcard cannot tell them apart. A
    /// field or variant reordered in one and not the other is a wire break whose
    /// first symptom is a decode failure on somebody else's machine.
    #[test]
    fn the_borrowed_encoding_is_the_owned_one() {
        let origin = iroh::SecretKey::from_bytes(&[5u8; 32]).public();
        let asset = Some(iroh_blobs::Hash::new(b"a brush"));
        let action = action(7);
        let frame = PeerFrame {
            boot: 3,
            seq: 9,
            name: Some("someone".into()),
            active_layer: stark_model::document::LayerId(0),
            cursor: Some(stark_model::Vec2::new(1.0, 2.0)),
            gesture: None,
            leaving: false,
        };

        let owned = |wire| {
            postcard::to_allocvec(&Stamped {
                origin,
                asset,
                wire,
            })
            .expect("encode")
        };
        let borrowed = |wire| {
            postcard::to_allocvec(&StampedRef {
                origin,
                asset,
                wire,
            })
            .expect("encode")
        };

        // Both variants, so the *indices* are pinned and not just the field order.
        let bytes = borrowed(WireRef::Action(&action));
        assert_eq!(bytes, owned(Wire::Action(action.clone())));
        assert_eq!(
            borrowed(WireRef::Presence(&frame)),
            owned(Wire::Presence(frame))
        );

        let back: Stamped = postcard::from_bytes(&bytes).expect("decode the borrowed form");
        assert_eq!(back.origin, origin);
        assert!(matches!(back.wire, Wire::Action(a) if a.id == action.id));
    }

    /// The encode cache answers the question it was asked, or none at all. Serving
    /// a snapshot from a stale key would be the same defect as serving an empty
    /// one: a document that is wrong and says nothing about it.
    #[test]
    fn a_remembered_snapshot_is_dropped_the_moment_the_log_moves() {
        let mut mirror = Mirror::from_file(&DocumentFile::new(vec![action(1)]));
        let have = [AssetId([3; 32])];

        let snapshot = mirror.snapshot();
        let revision = snapshot.revision;
        let bytes = Bytes::from(
            snapshot
                .without(&have)
                .into_file()
                .to_bytes()
                .expect("encode"),
        );
        mirror.remember(revision, &have, bytes.clone());

        assert_eq!(mirror.encoded_for(&have), Some(bytes));
        assert_eq!(
            mirror.encoded_for(&[]),
            None,
            "a different promise is a different snapshot"
        );

        mirror.insert(action(2));
        assert_eq!(
            mirror.encoded_for(&have),
            None,
            "the log moved; the remembered bytes are not what this session is"
        );

        // And an encode that finished after the log moved does not install itself.
        mirror.remember(revision, &have, Bytes::from_static(b"stale"));
        assert_eq!(mirror.encoded_for(&have), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_member_still_joining_refuses_rather_than_serving_an_empty_session() {
        let asker = bound(Served::default()).await;

        // Bound, listening, and with nothing behind it — a joiner between its
        // endpoint coming up and its own snapshot arriving.
        let joining = bound(Served::default()).await;
        let err = ask(&asker, &joining)
            .await
            .expect_err("a member with no session must not answer with one");
        assert!(matches!(err, crate::NetError::NotReady), "{err}");

        // The contrast is the point: the refusal above has to be the *state*
        // talking, not a server that never worked. Same call, same empty log,
        // published this time.
        let served = Served::default();
        let member = bound(served.clone()).await;
        let file = DocumentFile::new(vec![action(1)]);
        served.publish(std::sync::Arc::new(Mutex::new(Mirror::from_file(&file))));
        let bytes = ask(&asker, &member)
            .await
            .expect("a published member serves");
        let served_back = DocumentFile::from_bytes(&bytes).expect("decode the snapshot");
        assert_eq!(served_back.actions.len(), 1);

        for stack in [&asker, &joining, &member] {
            stack.shutdown.run().await;
        }
    }
}

mod iroh_wire {
    use iroh::endpoint::Connection;
    use iroh::protocol::{AcceptError, ProtocolHandler};

    use super::{Request, Tag, answer, decode_request};
    use crate::mirror::Served;

    /// Upper bound on an encoded request.
    ///
    /// A request carries the joiner's list of resolvable content ids, 32 bytes each,
    /// so the ceiling has to clear a catalog that grows rather than the variant tag
    /// alone: 64 KiB is two thousand of them.
    const MAX_REQUEST: usize = 64 * 1024;
    /// Upper bound on a response: a whole session snapshot (log + brush PNGs).
    /// A session that outgrows it stops accepting new members, so crossing most
    /// of the way there is worth saying out loud while joining still works.
    const MAX_RESPONSE: usize = 64 * 1024 * 1024;
    /// Fraction of [`MAX_RESPONSE`] a snapshot may reach before it is reported.
    const RESPONSE_WARN_AT: usize = MAX_RESPONSE / 2;

    /// Serves [`Request`]s over iroh connections.
    #[derive(Debug, Clone)]
    pub(crate) struct CollabProto {
        pub served: Served,
    }

    impl ProtocolHandler for CollabProto {
        async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
            // Serve requests until the peer closes the connection.
            loop {
                let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                    return Ok(());
                };
                let req = recv
                    .read_to_end(MAX_REQUEST)
                    .await
                    .map_err(AcceptError::from_err)?;
                let req = decode_request(&req).map_err(AcceptError::from_err)?;
                // A refusal is a complete, well-formed response — the requester
                // has to be able to tell "nothing yet" from "nothing in it".
                let Some(response) = answer(&self.served, req).map_err(AcceptError::from_err)?
                else {
                    send.write_all(&[Tag::NOT_READY])
                        .await
                        .map_err(AcceptError::from_err)?;
                    send.finish().map_err(AcceptError::from_err)?;
                    continue;
                };
                if response.len() > RESPONSE_WARN_AT {
                    tracing::warn!(
                        bytes = response.len(),
                        limit = MAX_RESPONSE,
                        "session snapshot is approaching the response ceiling; past \
                         it no new member can join"
                    );
                }
                send.write_all(&[Tag::OK])
                    .await
                    .map_err(AcceptError::from_err)?;
                send.write_all(&response)
                    .await
                    .map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
            }
        }
    }

    /// Issue one request over an open connection and return the raw response body.
    ///
    /// The tag is read on its own rather than split off a buffer: the body is a
    /// whole session, and shifting megabytes by one byte to look at the first of
    /// them would be the most expensive thing this function does.
    pub(crate) async fn request(conn: &Connection, req: Request) -> crate::Result<Vec<u8>> {
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&postcard::to_allocvec(&req)?).await?;
        send.finish()?;
        let mut tag = [0u8; 1];
        recv.read_exact(&mut tag).await?;
        match tag[0] {
            Tag::OK => Ok(recv.read_to_end(MAX_RESPONSE).await?),
            Tag::NOT_READY => Err(crate::NetError::NotReady),
            other => Err(crate::NetError::Other(format!(
                "response tagged {other}, which this build does not know"
            ))),
        }
    }
}
