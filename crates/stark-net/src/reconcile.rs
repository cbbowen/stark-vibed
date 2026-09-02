//! Anti-entropy: putting back what the flood dropped (§12.5).
//!
//! `iroh-gossip` is epidemic broadcast, not reliable delivery. A message can be
//! lost outright, a receiver can fall far enough behind that the swarm reports
//! [`Lagged`](iroh_gossip::api::Event::Lagged) and skips the rest, and a send can
//! fail at the sender — most plausibly a long stroke crossing the gossip message
//! ceiling. §12.6's convergence argument covers *ordering* — `merge_remote` is
//! idempotent by id, and an action landing behind newer ones makes the timeline
//! resync — but it says nothing about loss. A dropped `CommitStroke` is a stroke
//! that exists on some canvases and not others, forever, with both sides believing
//! they are in sync.
//!
//! So members compare logs. Every peer mirrors the whole total-ordered log and
//! every peer serves the catch-up ALPN, which is the entire mechanism: ask a
//! neighbour for its action ids, work out which are missing here, ask for those.
//!
//! **It is a pull, and that is enough in both directions.** A member whose own
//! broadcast failed cannot push it — but it mirrored the action before sending
//! (`Broadcaster::broadcast`), so it is holding it, and the next sweep by anyone
//! else finds it. Every member sweeps, so every gap closes from whichever side
//! noticed.
//!
//! Recovery is deliberately not a shortcut past anything. A recovered action goes
//! through the same door as one off the flood — parked if it names content this
//! peer lacks, released when that content lands — because a `SetSubstrate` recovered
//! and applied against the flat stand-in would be a divergence created by the
//! machinery meant to remove one (§6.4). The one gossip-door check with no analogue
//! here is dropping an action whose author does not match its sender — a sweep
//! partner legitimately forwards other authors' actions — so a future
//! authentication pass has two doors to cover, not one.

use std::sync::Arc;
use std::time::Duration;

use iroh::{EndpointAddr, EndpointId};
use n0_future::task;
use stark_model::document::ActionId;
use tokio::sync::Notify;

use crate::Result;
use crate::backend::Dialer;
use crate::cancel::Cancel;
use crate::content::Resolver;
use crate::neighbors::Neighbors;
use crate::waitlist::{MustFetch, Waitlist};
use crate::wire::{LogDigest, RECOVER_BATCH, Recovered, Request};

/// How long a quiet session waits before comparing logs with a neighbour.
///
/// This is the ceiling on how long a silently-dropped action stays dropped, so it
/// wants to be short; it is also a full id-list exchange per member per period, so
/// it wants to be long. Two minutes is far below the span over which anyone would
/// notice a missing stroke, and far above the rate at which the exchange costs
/// anything.
const SWEEP: Duration = Duration::from_secs(120);

/// How long after a *known* loss to wait before repairing it.
///
/// Not zero: `Lagged` means the swarm outran this peer, and the messages still in
/// flight behind it will arrive on their own. Reconciling immediately would ask
/// for actions that are seconds away, and would do it while the peer is already
/// behind.
const AFTER_LOSS: Duration = Duration::from_secs(5);

/// Consecutive failed sweeps before the failure log escalates to a warning.
///
/// One failure is routine — the partner may have left between the neighbour set
/// and the dial — but a run of them means repair is not happening at all.
const ESCALATE_AFTER: u32 = 5;

/// Raised when something is known to have been lost, so the next sweep does not
/// wait out [`SWEEP`].
#[derive(Debug, Clone, Default)]
pub(crate) struct Prompt(Arc<Notify>);

impl Prompt {
    /// Something was dropped — reconcile soon rather than on the slow cadence.
    pub fn raise(&self) {
        self.0.notify_one();
    }
}

/// The handles a session's background work runs on — what the receive loop and the
/// reconciler both need, assembled once by `finish` instead of threaded to each of
/// them a field at a time.
#[derive(Clone)]
pub(crate) struct Wiring {
    pub dialer: Dialer,
    pub neighbors: Neighbors,
    pub waitlist: Arc<Waitlist>,
    pub cancel: Cancel,
    pub prompt: Prompt,
}

impl Wiring {
    /// A resolver over the session's handles. Built per fetch rather than
    /// stored: it is cloned into every spawned fetch anyway, and it holds
    /// nothing the wiring does not.
    fn resolver(&self) -> Resolver<Dialer> {
        Resolver::new(
            self.dialer.clone(),
            self.neighbors.clone(),
            self.waitlist.clone(),
            self.cancel.clone(),
        )
    }

    /// The one door a resolver goes out through — detached, ended by the
    /// session's [`Cancel`]. A method so the call sites cannot drift.
    pub fn spawn_resolver(&self, fetch: MustFetch, origin: EndpointId, from: EndpointId) {
        task::spawn(
            self.resolver()
                .resolve(fetch.need, fetch.hash, origin, from),
        );
    }
}

pub(crate) struct Reconciler {
    wiring: Wiring,
    /// Which neighbour to ask next. Rotating rather than random: over a few sweeps
    /// it covers the swarm, which random selection only does in expectation, and it
    /// needs no source of randomness on a target that has to be careful about them.
    next: usize,
    /// Consecutive failed sweeps — see [`ESCALATE_AFTER`]. Reset by any success.
    failures: u32,
}

impl Reconciler {
    pub fn new(wiring: Wiring) -> Self {
        Self {
            wiring,
            next: 0,
            failures: 0,
        }
    }

    /// Sweep until the session ends.
    pub async fn run(mut self) {
        loop {
            let prompted = self.wait().await;
            if !self.alive() {
                return;
            }
            if prompted && !self.wiring.cancel.sleep(AFTER_LOSS).await {
                return;
            }
            match self.sweep().await {
                Ok(()) => self.failures = 0,
                Err(e) => {
                    self.failures += 1;
                    if self.failures >= ESCALATE_AFTER {
                        tracing::warn!(
                            consecutive = self.failures,
                            "reconciliation keeps failing; losses are going unrepaired: {e}"
                        );
                    } else {
                        // Expected often enough not to be a warning: the partner
                        // may have left between the neighbour set and the dial, or
                        // may itself still be joining. The next sweep picks
                        // someone else.
                        tracing::debug!("reconciliation with a neighbour failed: {e}");
                    }
                }
            }
        }
    }

    /// Wait for the next sweep. `true` if something was known to be lost, rather
    /// than the slow cadence coming round.
    async fn wait(&self) -> bool {
        let prompted = self.wiring.prompt.0.notified();
        n0_future::future::race(
            async {
                self.wiring.cancel.sleep(SWEEP).await;
                false
            },
            async {
                prompted.await;
                true
            },
        )
        .await
    }

    /// One exchange: ask a neighbour what it has, take what this peer does not.
    async fn sweep(&mut self) -> Result<()> {
        let Some(partner) = self.partner() else {
            // Alone in the swarm — nothing to compare against, and nothing lost
            // that anyone else could return.
            return Ok(());
        };
        let catchup = self.wiring.dialer.open(EndpointAddr::new(partner)).await?;
        // Equality first: most sweeps find nothing missing, and the id list is
        // ~1.6 MB at the design target where the digest is tens of bytes —
        // which is also what makes a `Lagged`-prompted burst of sweeps cheap.
        let digest: LogDigest = crate::codec::decode(&catchup.request(Request::Digest).await?)?;
        if digest == self.wiring.waitlist.log_digest() {
            catchup.close().await;
            return Ok(());
        }
        let theirs: Vec<ActionId> = crate::codec::decode(&catchup.request(Request::Ids).await?)?;
        let missing = self.wiring.waitlist.missing_from(&theirs);
        if missing.is_empty() {
            catchup.close().await;
            return Ok(());
        }

        // Loud on purpose. Reaching here means the flood lost something, which is
        // the failure §12.5 defers authentication in favour of taking seriously.
        tracing::info!(
            count = missing.len(),
            partner = %partner.fmt_short(),
            "recovering actions the flood dropped"
        );
        // Batched under the server's request ceiling: one ask naming every id a
        // far-behind peer lacks blows [`MAX_REQUEST`](crate::wire::MAX_REQUEST),
        // and the next sweep would rebuild it identically — repair would never
        // complete. Each batch is admitted as it lands, so a failure partway
        // keeps what already arrived.
        for chunk in missing.chunks(RECOVER_BATCH) {
            let recovered: Vec<Recovered> =
                crate::codec::decode(&catchup.request(Request::Actions(chunk.to_vec())).await?)?;
            for Recovered { action, hash } in recovered {
                // The same door the flood's actions go through, for the same reasons.
                // `partner` stands in for both the author and the deliverer: it is who
                // this peer can actually reach, and the resolver widens to the rest of
                // the swarm from its second round anyway.
                if let Some(fetch) = self.wiring.waitlist.admit(action, hash) {
                    self.wiring.spawn_resolver(fetch, partner, partner);
                }
            }
        }
        catchup.close().await;
        Ok(())
    }

    /// The next neighbour to compare against, or `None` when there are none.
    /// Sorted ([`Neighbors::snapshot_sorted`]) so the rotation covers the swarm.
    fn partner(&mut self) -> Option<EndpointId> {
        let ids = self.wiring.neighbors.snapshot_sorted();
        if ids.is_empty() {
            return None;
        }
        let index = self.next % ids.len();
        self.next = self.next.wrapping_add(1);
        Some(ids[index])
    }

    fn alive(&self) -> bool {
        // Only the session's end stops repair: the mirror this loop keeps
        // complete serves joiners and sweeping peers, so it outlives the local
        // engine's listener — a UI that stopped pumping must not quietly
        // degrade the mirror to flood-only completeness.
        !self.wiring.cancel.stopped()
    }
}

/// The property the whole file exists for: an action that never reached this peer
/// arrives anyway.
///
/// Over real endpoints — the exchange is two requests on the catch-up ALPN, and a
/// test of the diff alone would not go near either. No GPU: this is about the wire
/// and the log. Gossip is not involved at all, which is the point; a *dropped*
/// message is precisely one gossip never delivers.
#[cfg(test)]
mod tests {
    use stark_model::Srgb;
    use std::time::Duration;

    use stark_model::DocumentFile;
    use stark_model::document::{Action, ActionKind, ActorId};
    use tokio::sync::mpsc;

    use super::*;
    use crate::backend::{self, Bound};
    use crate::events::{NetOptions, RemoteEvent};
    use crate::mirror::{Mirror, Served, SharedMirror};

    fn action(lamport: u64) -> Action {
        Action {
            id: ActionId {
                lamport,
                actor: ActorId(1),
            },
            kind: ActionKind::SetSubstrateColor(Srgb::new([lamport as f32 / 8.0; 3])),
        }
    }

    /// A bound member serving `log`. The [`Served`] handle comes back too, for
    /// its request count.
    async fn member(log: Vec<Action>) -> (Bound, SharedMirror, Served) {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &NetOptions::local())
            .await
            .expect("bind");
        let mirror = SharedMirror::new(Mirror::from_file(&DocumentFile::new(log)));
        served.publish(mirror.clone());
        (bound, mirror, served)
    }

    /// A reconciler for `ours` wired to sweep against `them`, and the event
    /// stream its recoveries surface on. Also teaches the endpoint how to reach
    /// them, the way gossip's membership exchange does in a live session — the
    /// sweep dials by bare id.
    async fn reconciler_between(
        us: &Bound,
        ours: &SharedMirror,
        them: &Bound,
    ) -> (Reconciler, mpsc::UnboundedReceiver<RemoteEvent>) {
        let (tx, events) = mpsc::unbounded_channel();
        let waitlist = Arc::new(Waitlist::new(ours.clone(), tx, &[]));
        let reconciler = Reconciler::new(Wiring {
            dialer: us.dialer.clone(),
            neighbors: Neighbors::from_iter([them.dialer.local_id()]),
            waitlist,
            cancel: Cancel::default(),
            prompt: Prompt::default(),
        });
        let addr = them
            .dialer
            .ticket_addr(&NetOptions::local())
            .await
            .expect("their address");
        us.dialer.open(addr).await.expect("dial").close().await;
        (reconciler, events)
    }

    /// Every recovered action's lamport, in the order it surfaced.
    fn recovered_lamports(events: &mut mpsc::UnboundedReceiver<RemoteEvent>) -> Vec<u64> {
        std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|e| match e {
                RemoteEvent::Action(a) => Some(a.id.lamport),
                _ => None,
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_sweep_recovers_what_the_flood_dropped() {
        // One member painted four strokes; the other saw the first and the third.
        // Exactly the state a lost gossip message leaves behind, and one nothing
        // on either canvas reveals.
        let (them, _theirs, _) = member((1..=4).map(action).collect()).await;
        let (us, ours, _) = member(vec![action(1), action(3)]).await;
        let (mut reconciler, mut events) = reconciler_between(&us, &ours, &them).await;

        tokio::time::timeout(Duration::from_secs(20), reconciler.sweep())
            .await
            .expect("the sweep finishes")
            .expect("and succeeds");

        assert_eq!(
            recovered_lamports(&mut events),
            vec![2, 4],
            "the two that were dropped, and only those"
        );

        // A second sweep finds nothing: recovery is not a source of duplicates.
        reconciler.sweep().await.expect("a second sweep");
        assert!(events.try_recv().is_err());

        for stack in [&us, &them] {
            stack.shutdown.run().await;
        }
    }

    /// The steady state — two identical logs — costs the digest exchange and
    /// nothing further: no id list, nothing recovered, one request fielded.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_sweep_between_identical_logs_exchanges_only_the_digest() {
        let log: Vec<Action> = (1..=4).map(action).collect();
        let (them, _theirs, served) = member(log.clone()).await;
        let (us, ours, _) = member(log).await;
        let (mut reconciler, mut events) = reconciler_between(&us, &ours, &them).await;

        tokio::time::timeout(Duration::from_secs(20), reconciler.sweep())
            .await
            .expect("the sweep finishes")
            .expect("and succeeds");

        assert!(events.try_recv().is_err(), "nothing surfaced to the engine");
        assert_eq!(
            served.requests_fielded(),
            1,
            "the digest request, and nothing after it"
        );

        for stack in [&us, &them] {
            stack.shutdown.run().await;
        }
    }

    /// A peer that fell *far* behind still repairs completely. 5,000 missing ids
    /// encode past the server's request ceiling ([`crate::wire::MAX_REQUEST`]),
    /// so a single `Request::Actions` naming them all was refused — and rebuilt
    /// identically every sweep, so repair never finished.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_sweep_recovers_more_actions_than_one_request_can_name() {
        let (them, _theirs, _) = member((1..=5_000).map(action).collect()).await;
        let (us, ours, _) = member(Vec::new()).await;
        let (mut reconciler, mut events) = reconciler_between(&us, &ours, &them).await;

        tokio::time::timeout(Duration::from_secs(30), reconciler.sweep())
            .await
            .expect("the sweep finishes")
            .expect("and succeeds");

        assert_eq!(
            recovered_lamports(&mut events),
            (1..=5_000).collect::<Vec<u64>>(),
            "all five thousand, in total order"
        );

        for stack in [&us, &them] {
            stack.shutdown.run().await;
        }
    }
}
