//! The catch-up/asset endpoint: answering the wire's [`Request`]s from the
//! shared [`Mirror`](crate::mirror::Mirror) (§12.4).
//!
//! The vocabulary — the message formats, the [`ALPN`](crate::wire::ALPN) and its
//! version ledger — is [`wire`](crate::wire)'s; this module is the *serving* of
//! it. [`answer`] maps one request onto the mirror, and the iroh plumbing at the
//! bottom moves the bytes.

use bytes::Bytes;
use stark_model::AssetId;

use crate::mirror::{Served, SharedMirror};
use crate::wire::{Recovered, Request, Tag};

/// Upper bound on an encoded [`Request::Actions`] answer.
///
/// [`RECOVER_BATCH`](crate::wire::RECOVER_BATCH) caps how many actions are
/// *asked for*, but their encoded
/// size is unbounded — a stroke-heavy backlog could blow the read ceiling and
/// fail identically every sweep. Truncated rather than refused: the asker
/// treats ids not answered as still missing and re-asks next sweep, so a short
/// answer self-heals where a refusal repeats. Sized so even the largest single
/// action (bounded by the gossip message ceiling) always fits.
const MAX_RECOVER_RESPONSE: usize = 8 * 1024 * 1024;

/// Answer one request from the shared [`Mirror`](crate::mirror::Mirror) — every peer is a provider, so
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
        Request::SnapshotWithout(mut have) => {
            // No length guard: [`MAX_REQUEST`](crate::wire::MAX_REQUEST) already
            // bounds the list — an
            // `AssetId` is 32 bytes, so a list long enough to matter is refused
            // at the read, two layers before it could reach here.
            // Canonical order before anything reads the list: the encode cache
            // compares it verbatim, and two joiners enumerating the same catalog
            // in different orders are asking the same question.
            have.sort_unstable();
            have.dedup();
            snapshot_bytes(mirror, &have)?
        }
        // Neither is answered under the lock: the receive loop takes it per
        // arriving action, and a full-log id walk or per-action clones under it
        // stall this peer's painting. The lock covers a [`LogView`] — a
        // refcount bump — and the walk happens off it.
        Request::Ids => {
            let view = mirror.lock().log_view();
            crate::codec::encode(&view.action_ids())?.into()
        }
        Request::Actions(ids) => {
            let view = mirror.lock().log_view();
            // The view answers in plain pairs; the wire's shape is spelled here.
            let mut actions: Vec<Recovered> = view
                .recover(&ids)
                .into_iter()
                .map(|(action, hash)| Recovered { action, hash })
                .collect();
            let mut bytes = crate::codec::encode(&actions)?;
            while bytes.len() > MAX_RECOVER_RESPONSE && actions.len() > 1 {
                actions.truncate(actions.len() / 2);
                bytes = crate::codec::encode(&actions)?;
            }
            bytes.into()
        }
    }))
}

/// Encode the session snapshot, leaving out any content in `have`.
///
/// Three phases, and the lock covers only the first and last. Taking the snapshot
/// is a handful of refcount bumps — the log is persistent — while turning it into
/// a [`DocumentFile`](stark_model::DocumentFile) copies every asset payload (the
/// container owns its bytes) and
/// encodes the lot. A joiner arriving mid-session must not stall this peer's
/// receive loop for the length of that, and the next joiner should not repeat it.
fn snapshot_bytes(mirror: &SharedMirror, have: &[AssetId]) -> crate::Result<Bytes> {
    let snapshot = {
        let mirror = mirror.lock();
        match mirror.encoded_for(have) {
            Some(bytes) => return Ok(bytes),
            None => mirror.snapshot(),
        }
    };
    let revision = snapshot.revision;
    let bytes = Bytes::from(snapshot.without(have).into_file().to_bytes()?);
    mirror.lock().remember(revision, have, bytes.clone());
    Ok(bytes)
}

/// Decode a request received over any transport.
pub(crate) fn decode_request(bytes: &[u8]) -> crate::Result<Request> {
    Ok(crate::codec::decode(bytes)?)
}

/// What a response's first byte says: `Ok` when the answer follows, or the
/// typed refusal it spells. The one reading of the tag byte, so an unknown
/// value has one answer — [`NetError::UnknownTag`](crate::NetError::UnknownTag)
/// — whichever transport read it.
fn interpret_tag(tag: u8) -> crate::Result<()> {
    match tag {
        Tag::OK => Ok(()),
        Tag::NOT_READY => Err(crate::NetError::NotReady),
        tag => Err(crate::NetError::UnknownTag { tag }),
    }
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
    use stark_model::DocumentFile;
    use stark_model::Srgb;
    use stark_model::document::{Action, ActionId, ActionKind, ActorId};

    use super::*;
    use crate::backend::{self, Bound};
    use crate::events::NetOptions;
    use crate::mirror::Mirror;

    /// The tag protocol byte by byte: the two known values, and a typed refusal
    /// of everything else so the far end's message can name the byte.
    #[test]
    fn interpret_tag_knows_exactly_two_bytes() {
        assert!(interpret_tag(Tag::OK).is_ok());
        assert!(matches!(
            interpret_tag(Tag::NOT_READY),
            Err(crate::NetError::NotReady)
        ));
        assert!(matches!(
            interpret_tag(7),
            Err(crate::NetError::UnknownTag { tag: 7 })
        ));
    }

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
            kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
        }
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

    /// Two joiners enumerating the same catalog in different orders are asking
    /// the same question, and the second must not pay for a second encode —
    /// [`answer`] canonicalizes the list before the cache compares it.
    #[test]
    fn the_encode_cache_hits_across_permuted_have_lists() {
        let served = Served::default();
        served.publish(SharedMirror::new(Mirror::from_file(&DocumentFile::new(
            vec![action(1)],
        ))));
        let (a, b, c) = (AssetId([1; 32]), AssetId([2; 32]), AssetId([3; 32]));

        let first = answer(&served, Request::SnapshotWithout(vec![c, a, b]))
            .expect("encode")
            .expect("published");
        // Permuted, and with a duplicate — the other way one catalog spells
        // two questions.
        let second = answer(&served, Request::SnapshotWithout(vec![b, c, a, c]))
            .expect("encode")
            .expect("published");
        assert_eq!(first, second);
        assert_eq!(
            first.as_ptr(),
            second.as_ptr(),
            "the second answer is the remembered allocation, not a re-encode"
        );
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
        served.publish(SharedMirror::new(Mirror::from_file(&file)));
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

    use super::{answer, decode_request};
    use crate::mirror::Served;
    use crate::wire::{MAX_REQUEST, Request, Tag};

    /// Upper bound on a response: a whole session snapshot (log + brush PNGs).
    /// A session that outgrows it stops accepting new members, so crossing most
    /// of the way there is worth saying out loud while joining still works.
    const MAX_RESPONSE: usize = 64 * 1024 * 1024;
    /// Fraction of [`MAX_RESPONSE`] a snapshot may reach before it is reported.
    const RESPONSE_WARN_AT: usize = MAX_RESPONSE / 2;
    /// Snapshot requests served per connection before it is closed. Each answer
    /// is up to [`MAX_RESPONSE`] on a peer that is also painting, and a
    /// legitimate re-join opens a fresh connection — so the bound only cuts off
    /// a peer re-asking on one.
    const SNAPSHOTS_PER_CONN: u32 = 8;

    /// Serves [`Request`]s over iroh connections.
    #[derive(Debug, Clone)]
    pub(crate) struct CollabProto {
        pub served: Served,
    }

    impl ProtocolHandler for CollabProto {
        async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
            // Serve requests until the peer closes the connection.
            let mut snapshots: u32 = 0;
            loop {
                let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                    return Ok(());
                };
                let req = recv
                    .read_to_end(MAX_REQUEST)
                    .await
                    .map_err(AcceptError::from_err)?;
                let req = decode_request(&req).map_err(AcceptError::from_err)?;
                if matches!(req, Request::Snapshot | Request::SnapshotWithout(_)) {
                    snapshots += 1;
                    if snapshots > SNAPSHOTS_PER_CONN {
                        tracing::warn!(
                            limit = SNAPSHOTS_PER_CONN,
                            "closing a connection that keeps asking for snapshots"
                        );
                        connection.close(0u32.into(), b"snapshot budget spent");
                        return Ok(());
                    }
                }
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
        send.write_all(&crate::codec::encode(&req)?).await?;
        send.finish()?;
        let mut tag = [0u8; 1];
        recv.read_exact(&mut tag).await?;
        super::interpret_tag(tag[0])?;
        Ok(recv.read_to_end(MAX_RESPONSE).await?)
    }
}
