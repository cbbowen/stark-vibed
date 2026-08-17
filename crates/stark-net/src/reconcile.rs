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
//! peer lacks, released when that content lands — because a `SetSurface` recovered
//! and applied against the flat stand-in would be a divergence created by the
//! machinery meant to remove one (§6.4).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use iroh::{EndpointAddr, EndpointId};
use n0_future::task;
use stark_engine::document::ActionId;
use tokio::sync::Notify;

use crate::Result;
use crate::proto::{Recovered, Request};
use crate::session::{Admission, Wiring};

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

pub(crate) struct Reconciler {
    wiring: Wiring,
    /// Which neighbour to ask next. Rotating rather than random: over a few sweeps
    /// it covers the swarm, which random selection only does in expectation, and it
    /// needs no source of randomness on a target that has to be careful about them.
    next: AtomicUsize,
}

impl Reconciler {
    pub fn new(wiring: Wiring) -> Self {
        Self {
            wiring,
            next: AtomicUsize::new(0),
        }
    }

    /// Sweep until the session ends.
    pub async fn run(self) {
        loop {
            let prompted = self.wait().await;
            if !self.alive() {
                return;
            }
            if prompted && !self.wiring.cancel.sleep(AFTER_LOSS).await {
                return;
            }
            if let Err(e) = self.sweep().await {
                // Expected often enough not to be a warning: the partner may have
                // left between the neighbour set and the dial, or may itself still
                // be joining. The next sweep picks someone else.
                tracing::debug!("reconciliation with a neighbour failed: {e}");
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
    async fn sweep(&self) -> Result<()> {
        let Some(partner) = self.partner() else {
            // Alone in the swarm — nothing to compare against, and nothing lost
            // that anyone else could return.
            return Ok(());
        };
        let catchup = self.wiring.dialer.open(EndpointAddr::new(partner)).await?;
        let theirs: Vec<ActionId> = postcard::from_bytes(&catchup.request(Request::Ids).await?)?;
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
        let recovered: Vec<Recovered> =
            postcard::from_bytes(&catchup.request(Request::Actions(missing)).await?)?;
        catchup.close().await;

        for (action, hash) in recovered {
            // The same door the flood's actions go through, for the same reasons.
            // `partner` stands in for both the author and the deliverer: it is who
            // this peer can actually reach, and the resolver widens to the rest of
            // the swarm from its second round anyway.
            match Admission::of(&action, hash, &self.wiring.waitlist) {
                Admission::Ready => self.wiring.waitlist.accept(action),
                Admission::Waiting => {}
                Admission::Fetching { need, hash } => {
                    task::spawn(
                        self.wiring
                            .resolver
                            .clone()
                            .resolve(need, hash, partner, partner),
                    );
                }
            }
        }
        Ok(())
    }

    /// The next neighbour to compare against, or `None` when there are none.
    fn partner(&self) -> Option<EndpointId> {
        let neighbors = self.wiring.neighbors.lock().expect("neighbors poisoned");
        if neighbors.is_empty() {
            return None;
        }
        // Sorted, so the rotation visits every member rather than following the
        // hash set's iteration order, which moves as the set changes.
        let mut ids: Vec<EndpointId> = neighbors.iter().copied().collect();
        drop(neighbors);
        ids.sort_by_key(|id| *id.as_bytes());
        let index = self.next.fetch_add(1, Ordering::Relaxed) % ids.len();
        Some(ids[index])
    }

    fn alive(&self) -> bool {
        !self.wiring.cancel.stopped() && self.wiring.waitlist.is_live()
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
    use std::time::Duration;

    use stark_engine::DocumentFile;
    use stark_engine::document::{Action, ActionKind, ActorId};
    use tokio::sync::mpsc;

    use std::collections::HashSet;
    use std::sync::Mutex;

    use super::*;
    use crate::backend::{self, Bound, Cancel};
    use crate::content::Resolver;
    use crate::mirror::{Mirror, Served};
    use crate::session::{NetOptions, RemoteEvent};
    use crate::waitlist::Waitlist;

    fn action(lamport: u64) -> Action {
        Action {
            id: ActionId {
                lamport,
                actor: ActorId(1),
            },
            kind: ActionKind::SetBackground([lamport as f32 / 8.0; 3]),
        }
    }

    /// A bound member serving `log`.
    async fn member(log: Vec<Action>) -> (Bound, Arc<Mutex<Mirror>>) {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &NetOptions::local())
            .await
            .expect("bind");
        let mirror = Arc::new(Mutex::new(Mirror::from_file(&DocumentFile::new(log))));
        served.publish(mirror.clone());
        (bound, mirror)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_sweep_recovers_what_the_flood_dropped() {
        // One member painted four strokes; the other saw the first and the third.
        // Exactly the state a lost gossip message leaves behind, and one nothing
        // on either canvas reveals.
        let (them, _theirs) = member((1..=4).map(action).collect()).await;
        let (us, ours) = member(vec![action(1), action(3)]).await;

        let (tx, mut events) = mpsc::unbounded_channel();
        let waitlist = Arc::new(Waitlist::new(ours.clone(), tx, &[]));
        let cancel = Cancel::default();
        let neighbors = Arc::new(Mutex::new(HashSet::from([them.dialer.local_id()])));
        let reconciler = Reconciler::new(Wiring {
            resolver: Resolver::new(
                us.dialer.clone(),
                neighbors.clone(),
                waitlist.clone(),
                cancel.clone(),
            ),
            dialer: us.dialer.clone(),
            neighbors,
            waitlist,
            cancel,
            prompt: Prompt::default(),
        });

        // Teach the endpoint how to reach them, the way gossip's membership
        // exchange does in a live session — the sweep dials by bare id.
        let addr = them
            .dialer
            .ticket_addr(&NetOptions::local())
            .await
            .expect("their address");
        us.dialer.open(addr).await.expect("dial").close().await;

        tokio::time::timeout(Duration::from_secs(20), reconciler.sweep())
            .await
            .expect("the sweep finishes")
            .expect("and succeeds");

        let recovered: Vec<u64> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|e| match e {
                RemoteEvent::Action(a) => Some(a.id.lamport),
                _ => None,
            })
            .collect();
        assert_eq!(
            recovered,
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
}
