//! Joining's fallible walk (§12.4): reaching the members a link names, fetching
//! the snapshot through the first that serves, and closing the stack down when
//! setup does not finish.

use std::time::Duration;

use iroh::EndpointAddr;

use crate::backend::{Catchup, Dialer, Shutdown};
use crate::cancel::Cancel;
use crate::wire::Request;
use crate::{NetError, Result, TicketError};

/// How long a joiner waits to meet the swarm before fetching the snapshot.
pub(super) const JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// The bound on each attempt to reach one member a link names.
///
/// A member that has left gets no answer at all rather than a refusal (UDP
/// refuses nothing), so without a bound the members after it — the link's
/// insurance against exactly this — would never be tried.
pub(super) const DIAL_TIMEOUT: Duration = Duration::from_secs(5);

/// Run the fallible tail of session setup, stopping and closing the stack if it
/// does not finish.
///
/// Every step that can fail — dialling the ticket's members, subscribing, fetching
/// and decoding the snapshot, minting the ticket address — happens *after* the
/// endpoint exists, and dropping [`Bound`](crate::backend::Bound) closes none of
/// it: the endpoint keeps
/// its relay connection and the gossip, blob and router actors keep running. The
/// expected failures are the ones that repeat (a link from another build, a host
/// that has gone), so without this a user retrying accumulates a stack per
/// attempt — in a browser tab with a hard ceiling.
///
/// The stop signal goes first, as in
/// [`CollabSession::shutdown`](crate::CollabSession::shutdown): an
/// answering-side WebRTC bootstrap accepted during setup already holds the
/// session's [`Cancel`], and its retries must not outlive the stack they dial on.
pub(super) async fn closing_on_error<T>(
    cancel: Cancel,
    shutdown: Shutdown,
    setup: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match setup.await {
        Ok(ready) => Ok(ready),
        Err(e) => {
            cancel.stop();
            shutdown.run().await;
            Err(e)
        }
    }
}

/// One bounded attempt to reach one member a link names — see [`DIAL_TIMEOUT`]
/// for why unbounded would mean the members after this one are never tried.
async fn try_open(dialer: &Dialer, member: &EndpointAddr) -> Result<Catchup> {
    match n0_future::time::timeout(DIAL_TIMEOUT, dialer.open(member.clone())).await {
        Ok(result) => result,
        Err(_) => Err(NetError::NoAnswer { member: member.id }),
    }
}

/// One snapshot, from whoever will serve it: the link's members opened and
/// asked in order, first answer wins. The minter is first and usually still
/// there; every member after it is the link's insurance against exactly that
/// peer having left.
///
/// Two failures are walked past. A member that does not answer the dial is
/// skipped ([`try_open`]'s bound is what makes the ones after it reachable at
/// all), and a member that answers but cannot serve —
/// [`NetError::NotReady`], a member still fetching its *own* snapshot — is
/// walked past too: every member is an entry point (§12.4) and the answer to
/// one that cannot serve yet is to ask another, which is only possible when
/// the link names another.
///
/// Each miss is logged as it happens, because a join that eventually succeeds
/// through the third name should still say what it walked past; the error
/// reported is the *last* member's, by which point it describes a link none of
/// whose members would serve.
pub(super) async fn fetch_snapshot(
    dialer: &Dialer,
    members: &[EndpointAddr],
    request: Request,
) -> Result<Vec<u8>> {
    // What an empty link reports — unmintable, and refused by the parse, but a
    // `SessionTicket` is a plain struct anyone can assemble.
    let mut failure = NetError::Ticket(TicketError::Empty);
    for member in members {
        let catchup = match try_open(dialer, member).await {
            Ok(catchup) => catchup,
            Err(e) => {
                tracing::warn!(member = %member.id.fmt_short(), "could not reach session member: {e}");
                failure = e;
                continue;
            }
        };
        match catchup.request(request.clone()).await {
            Ok(snapshot) => {
                catchup.close().await;
                return Ok(snapshot);
            }
            Err(e) => {
                tracing::warn!("a session member answered but could not serve the snapshot: {e}");
                catchup.close().await;
                failure = e;
            }
        }
    }
    Err(failure)
}

/// The walk, over real endpoints: what a joiner does with a link whose first
/// name answers and still cannot serve.
#[cfg(test)]
mod tests {
    use stark_model::DocumentFile;

    use crate::events::NetOptions;
    use crate::mirror::Served;
    use crate::session::CollabSession;
    use crate::testutil::{action, bound};
    use crate::ticket::SessionTicket;

    /// Every member is an entry point (§12.4), including when the first one a
    /// link names is itself still joining: it answers the dial and refuses with
    /// [`NotReady`](crate::NetError::NotReady) rather than serving an empty
    /// session, and the walk treats that as "ask the next" — so the join still
    /// lands on the member with a session to serve.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_join_walks_past_a_member_that_is_still_joining() {
        // The member with a session: a real host, so the walk's success is the
        // whole join path and not just the fetch.
        let marker = action(1);
        let (host, _host_events) =
            CollabSession::host(DocumentFile::new(vec![marker.clone()]), NetOptions::local())
                .await
                .expect("host session");
        let minted = host.broadcaster().ticket().await;

        // A member that is bound — the ALPN answers — but never published:
        // a joiner between its endpoint coming up and its own snapshot
        // arriving (the same state proto's NotReady test pins).
        let joining = bound(Served::default()).await;
        let joining_addr = joining
            .dialer
            .ticket_addr(&NetOptions::local())
            .await
            .expect("the not-ready member's address");

        // A link naming the not-ready member first: the walk must get past it.
        let ticket = SessionTicket {
            members: vec![joining_addr, minted.members[0].clone()],
            topic: minted.topic,
        };
        let joined = CollabSession::join(&ticket, NetOptions::local())
            .await
            .expect("join through the second member the link names");
        assert!(
            joined.document.actions.iter().any(|a| a.id == marker.id),
            "the snapshot is the serving member's, not an empty stand-in"
        );

        joined.session.shutdown().await;
        host.shutdown().await;
        joining.shutdown.run().await;
    }
}
