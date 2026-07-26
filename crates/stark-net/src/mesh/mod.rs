//! A small peer-to-peer broadcast mesh — the live-edit wire, in place of
//! `iroh-gossip`.
//!
//! # Why this exists
//!
//! On the web iroh cannot hole-punch, so every byte would otherwise cross a
//! relay. WebRTC fixes that, but the only way to get WebRTC under iroh is a
//! transport whose connections are established explicitly (dial/accept) — which
//! `iroh-gossip`, a self-dialing swarm protocol, cannot use. This module
//! provides the same guarantees over explicit connections instead.
//!
//! # What it guarantees (the properties gossip gave us)
//!
//! - **Everyone receives everything.** A payload is flooded to every neighbour
//!   and forwarded onward, so it reaches peers that are not directly connected
//!   to its origin — the epidemic property, over whatever partial connectivity
//!   exists.
//! - **Exactly once.** Every payload is stamped `(origin, seq)`; duplicates
//!   arriving over redundant paths are dropped and never re-forwarded, so
//!   flooding terminates.
//! - **Peers find each other.** Connecting exchanges peer lists, and newly
//!   learned peers are propagated, so a joiner that knows *one* member converges
//!   on the whole swarm.
//! - **It heals.** Dropped connections are redialed with exponential backoff; a
//!   peer is never forgotten, so partitions recover on their own.
//! - **Loss is visible.** A gap that never fills raises [`MeshEvent::Lagged`],
//!   the counterpart of gossip's `Lagged`, so the session can resync.
//!
//! # Shape
//!
//! One task owns all state and receives everything — commands, inbound
//! connections, decoded frames, timer ticks — through a single queue. Network
//! work (dialing, reading, writing) happens in satellite tasks that report back
//! through that same queue, so the core never blocks and needs no locks and no
//! `select!` (which keeps it identical on wasm).
//!
//! The mesh knows nothing of iroh or WebRTC: see [`MeshTransport`].

mod proto;
mod state;
mod transport;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use n0_future::task;
use tokio::sync::{mpsc, oneshot};

use proto::Frame;
use state::DeliveryTracker;

pub use proto::TopicId;
pub use transport::{
    MaybeSend, MaybeSync, MeshConn, MeshRecv, MeshSender, MeshTransport, MeshTransportError,
    PeerId, TransportResult,
};

/// Tuning. Defaults suit a drawing session: a handful of peers, small payloads.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Which swarm this is. Connections announcing a different topic are refused.
    pub topic: TopicId,
    /// Frames larger than this are refused rather than allocated.
    pub max_frame: usize,
    /// How often to retry disconnected peers and re-examine the swarm.
    pub maintenance_interval: Duration,
    /// Cap on the exponential redial backoff, in maintenance intervals.
    pub max_backoff_ticks: u32,
    /// How far past a hole the sequence may run before the missing payloads are
    /// written off and [`MeshEvent::Lagged`] is raised.
    pub lag_threshold: u64,
}

impl MeshConfig {
    pub fn new(topic: TopicId) -> Self {
        Self {
            topic,
            // Comfortably past any single action; pixels never ride the mesh.
            max_frame: 512 * 1024,
            maintenance_interval: Duration::from_secs(2),
            max_backoff_ticks: 15,
            lag_threshold: 128,
        }
    }
}

/// Something the mesh observed. Delivered in arrival order.
#[derive(Debug, Clone)]
pub enum MeshEvent {
    /// A payload from the swarm, not seen before.
    Received {
        /// Who produced it — the authoritative source for anything it references.
        origin: PeerId,
        /// The neighbour that handed it to us (may be a forwarder, not `origin`).
        from: PeerId,
        payload: Vec<u8>,
    },
    /// Payloads from `origin` were lost. The session should resync.
    Lagged { origin: PeerId },
    /// A direct connection came up.
    NeighborUp(PeerId),
    /// A direct connection went away. The mesh will keep trying to restore it.
    NeighborDown(PeerId),
}

/// The mesh is no longer running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshClosed;

impl std::fmt::Display for MeshClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mesh is closed")
    }
}

impl std::error::Error for MeshClosed {}

enum Command {
    Broadcast {
        payload: Vec<u8>,
        /// Resolves once the payload has been queued to every current neighbour.
        ack: oneshot::Sender<()>,
    },
    AddPeer(PeerId),
    Neighbors(oneshot::Sender<Vec<PeerId>>),
    Shutdown,
}

/// A handle to a running mesh. Cheap to clone; all clones drive the same swarm.
#[derive(Clone)]
pub struct Mesh {
    local: PeerId,
    commands: mpsc::UnboundedSender<Command>,
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("local", &self.local)
            .finish_non_exhaustive()
    }
}

impl Mesh {
    /// Start a mesh over `transport`, seeded with `bootstrap` peers (the members
    /// we already know how to reach — often just the one from a ticket).
    ///
    /// Returns the handle and the stream of [`MeshEvent`]s.
    pub fn spawn<T: MeshTransport>(
        transport: T,
        config: MeshConfig,
        bootstrap: impl IntoIterator<Item = PeerId>,
    ) -> (Self, mpsc::UnboundedReceiver<MeshEvent>) {
        let local = transport.local_id();
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let (inbox_tx, inbox_rx) = mpsc::unbounded_channel::<Ev<T::Conn>>();
        let transport = Arc::new(transport);

        let mut known: HashMap<PeerId, PeerState> = HashMap::new();
        for peer in bootstrap {
            if peer != local {
                known.insert(peer, PeerState::default());
            }
        }

        // Commands are funnelled into the one inbox the driver reads.
        {
            let inbox_tx = inbox_tx.clone();
            let mut commands_rx = commands_rx;
            task::spawn(async move {
                while let Some(cmd) = commands_rx.recv().await {
                    if inbox_tx.send(Ev::Cmd(cmd)).is_err() {
                        break;
                    }
                }
            });
        }

        // Inbound connections.
        {
            let transport = transport.clone();
            let inbox_tx = inbox_tx.clone();
            task::spawn(async move {
                while let Some(conn) = transport.accept().await {
                    if inbox_tx.send(Ev::Inbound(conn)).is_err() {
                        break;
                    }
                }
            });
        }

        // Maintenance clock: redials and swarm upkeep.
        {
            let inbox_tx = inbox_tx.clone();
            let interval = config.maintenance_interval;
            task::spawn(async move {
                loop {
                    n0_future::time::sleep(interval).await;
                    if inbox_tx.send(Ev::Tick).is_err() {
                        break;
                    }
                }
            });
        }

        let driver = Driver {
            transport,
            local,
            delivery: DeliveryTracker::new(config.lag_threshold),
            config,
            known,
            conns: HashMap::new(),
            dialing: HashSet::new(),
            next_seq: 0,
            next_conn_id: 0,
            events: events_tx,
            inbox: inbox_tx,
        };
        task::spawn(driver.run(inbox_rx));

        (
            Self {
                local,
                commands: commands_tx,
            },
            events_rx,
        )
    }

    /// This node's identity.
    pub fn local_id(&self) -> PeerId {
        self.local
    }

    /// Flood `payload` to the swarm. Resolves once it has been queued to every
    /// current neighbour (delivery itself is asynchronous, as with gossip).
    pub async fn broadcast(&self, payload: Vec<u8>) -> Result<(), MeshClosed> {
        let (ack, acked) = oneshot::channel();
        self.commands
            .send(Command::Broadcast { payload, ack })
            .map_err(|_| MeshClosed)?;
        acked.await.map_err(|_| MeshClosed)
    }

    /// Introduce a peer. The mesh connects to it and folds it into the swarm.
    pub fn add_peer(&self, peer: PeerId) -> Result<(), MeshClosed> {
        self.commands
            .send(Command::AddPeer(peer))
            .map_err(|_| MeshClosed)
    }

    /// Currently connected neighbours (diagnostics and tests).
    pub async fn neighbors(&self) -> Result<Vec<PeerId>, MeshClosed> {
        let (tx, rx) = oneshot::channel();
        self.commands
            .send(Command::Neighbors(tx))
            .map_err(|_| MeshClosed)?;
        rx.await.map_err(|_| MeshClosed)
    }

    /// Stop the mesh and drop every connection.
    pub fn shutdown(&self) {
        let _ = self.commands.send(Command::Shutdown);
    }
}

/// Everything the driver reacts to, from every source.
enum Ev<C: MeshConn> {
    Cmd(Command),
    Inbound(C),
    Dialed {
        peer: PeerId,
        conn: Option<C>,
    },
    Frame {
        from: PeerId,
        frame: Frame,
        raw: Vec<u8>,
    },
    ConnLost {
        peer: PeerId,
        conn_id: u64,
    },
    Tick,
}

/// A peer we know of, connected or not.
#[derive(Debug, Default)]
struct PeerState {
    failures: u32,
    /// Maintenance ticks to wait before the next dial attempt.
    wait_ticks: u32,
}

/// A live connection, reduced to what the driver needs: an ordered outgoing
/// queue. Dropping this closes the connection (the writer task exits).
struct Neighbor {
    conn_id: u64,
    out: mpsc::UnboundedSender<Vec<u8>>,
    /// Who opened this connection — the tie-break for simultaneous dials, and
    /// identical on both sides.
    dialer: PeerId,
}

struct Driver<T: MeshTransport> {
    transport: Arc<T>,
    config: MeshConfig,
    local: PeerId,
    /// Every peer we have ever heard of. Never pruned: a peer that is gone today
    /// may be back tomorrow, and the swarm is small.
    known: HashMap<PeerId, PeerState>,
    conns: HashMap<PeerId, Neighbor>,
    dialing: HashSet<PeerId>,
    delivery: DeliveryTracker,
    next_seq: u64,
    next_conn_id: u64,
    events: mpsc::UnboundedSender<MeshEvent>,
    inbox: mpsc::UnboundedSender<Ev<T::Conn>>,
}

impl<T: MeshTransport> Driver<T> {
    async fn run(mut self, mut inbox: mpsc::UnboundedReceiver<Ev<T::Conn>>) {
        // Reach out to the bootstrap set immediately rather than waiting a tick.
        self.dial_due_peers(true);

        while let Some(ev) = inbox.recv().await {
            match ev {
                Ev::Cmd(Command::Broadcast { payload, ack }) => {
                    self.broadcast(payload);
                    let _ = ack.send(());
                }
                Ev::Cmd(Command::AddPeer(peer)) => {
                    if self.learn_peers(std::iter::once(peer)) {
                        self.dial_due_peers(true);
                    }
                }
                Ev::Cmd(Command::Neighbors(reply)) => {
                    let _ = reply.send(self.conns.keys().copied().collect());
                }
                Ev::Cmd(Command::Shutdown) => break,
                Ev::Inbound(conn) => self.register(conn, false),
                Ev::Dialed { peer, conn } => {
                    self.dialing.remove(&peer);
                    match conn {
                        Some(conn) => self.register(conn, true),
                        None => self.back_off(peer),
                    }
                }
                Ev::Frame { from, frame, raw } => self.handle_frame(from, frame, raw),
                Ev::ConnLost { peer, conn_id } => self.drop_conn(peer, conn_id),
                Ev::Tick => self.dial_due_peers(false),
            }
        }

        // Dropping the neighbours closes their writer tasks and connections.
        self.conns.clear();
    }

    // --- connections ---

    fn register(&mut self, conn: T::Conn, initiated_by_us: bool) {
        let peer = conn.peer();
        if peer == self.local {
            return;
        }
        let dialer = if initiated_by_us { self.local } else { peer };

        // Both peers may dial at once. Both sides resolve it the same way —
        // keep the connection opened by the lower id — so exactly one survives.
        if let Some(existing) = self.conns.get(&peer) {
            if existing.dialer <= dialer {
                tracing::debug!(%peer, "dropping redundant mesh connection");
                return;
            }
            tracing::debug!(%peer, "replacing mesh connection after simultaneous dial");
        }

        let conn_id = self.next_conn_id;
        self.next_conn_id += 1;
        let (sender, recv) = conn.split();

        // One writer task per connection keeps frame order without locking.
        let (out, mut queue) = mpsc::unbounded_channel::<Vec<u8>>();
        task::spawn(async move {
            while let Some(frame) = queue.recv().await {
                if let Err(e) = sender.send(frame).await {
                    tracing::debug!("mesh send failed: {e}");
                    break;
                }
            }
            sender.close();
        });

        // Reader task: decode frames and hand them to the driver.
        {
            let inbox = self.inbox.clone();
            let max_frame = self.config.max_frame;
            task::spawn(read_loop::<T::Conn>(recv, peer, conn_id, inbox, max_frame));
        }

        self.conns.insert(
            peer,
            Neighbor {
                conn_id,
                out,
                dialer,
            },
        );
        self.known.entry(peer).or_default().reset();

        // Greet: prove which swarm we are in and share who we know.
        let hello = Frame::Hello {
            topic: self.config.topic,
            from: self.local,
            peers: self.known.keys().copied().collect(),
        };
        self.send_to(peer, &hello);

        tracing::debug!(%peer, "mesh neighbour up");
        let _ = self.events.send(MeshEvent::NeighborUp(peer));
    }

    fn drop_conn(&mut self, peer: PeerId, conn_id: u64) {
        // A late notice from a replaced connection must not evict the live one.
        if self.conns.get(&peer).map(|n| n.conn_id) != Some(conn_id) {
            return;
        }
        self.conns.remove(&peer);
        // Retry promptly: this was a working peer a moment ago.
        if let Some(state) = self.known.get_mut(&peer) {
            state.wait_ticks = 0;
        }
        tracing::debug!(%peer, "mesh neighbour down");
        let _ = self.events.send(MeshEvent::NeighborDown(peer));
    }

    fn back_off(&mut self, peer: PeerId) {
        let max = self.config.max_backoff_ticks;
        let state = self.known.entry(peer).or_default();
        state.failures = state.failures.saturating_add(1);
        state.wait_ticks = 1u32
            .checked_shl(state.failures.saturating_sub(1))
            .unwrap_or(max)
            .min(max);
    }

    /// Dial every known peer that is due. `immediate` ignores the backoff clock
    /// (used when a peer is first introduced).
    fn dial_due_peers(&mut self, immediate: bool) {
        let due: Vec<PeerId> = self
            .known
            .iter_mut()
            .filter(|(peer, _)| !self.conns.contains_key(*peer) && !self.dialing.contains(*peer))
            .filter_map(|(peer, state)| {
                if immediate || state.wait_ticks == 0 {
                    Some(*peer)
                } else {
                    state.wait_ticks -= 1;
                    None
                }
            })
            .collect();

        for peer in due {
            self.dialing.insert(peer);
            let transport = self.transport.clone();
            let inbox = self.inbox.clone();
            task::spawn(async move {
                let conn = match transport.dial(peer).await {
                    Ok(conn) => Some(conn),
                    Err(e) => {
                        tracing::debug!(%peer, "mesh dial failed: {e}");
                        None
                    }
                };
                let _ = inbox.send(Ev::Dialed { peer, conn });
            });
        }
    }

    // --- frames ---

    fn handle_frame(&mut self, from: PeerId, frame: Frame, raw: Vec<u8>) {
        match frame {
            Frame::Hello {
                topic,
                from: claimed,
                peers,
            } => {
                if topic != self.config.topic {
                    tracing::warn!(%from, "refusing mesh peer from another session");
                    self.conns.remove(&from);
                    self.known.remove(&from);
                    return;
                }
                if claimed != from {
                    tracing::warn!(%from, "mesh peer announced a mismatched identity");
                    self.conns.remove(&from);
                    return;
                }
                if self.learn_peers(peers) {
                    self.dial_due_peers(true);
                }
            }
            Frame::Peers { peers } => {
                if self.learn_peers(peers) {
                    self.dial_due_peers(true);
                }
            }
            Frame::Data {
                origin,
                seq,
                payload,
            } => {
                // Our own payload came back around; dedup already covers this,
                // but skip the bookkeeping entirely.
                if origin == self.local {
                    return;
                }
                let delivery = self.delivery.observe(origin, seq);
                if delivery.lagged {
                    tracing::warn!(%origin, "mesh lost payloads from a peer");
                    let _ = self.events.send(MeshEvent::Lagged { origin });
                }
                if !delivery.fresh {
                    return;
                }
                // Keep it moving before delivering locally: peers behind us in
                // the flood should not wait on our consumer.
                self.forward(&raw, from);
                let _ = self.events.send(MeshEvent::Received {
                    origin,
                    from,
                    payload,
                });
            }
        }
    }

    /// Fold `peers` into the known set; returns whether anything was new.
    /// New peers are announced onward, so the swarm converges without every
    /// member having to meet every other directly.
    fn learn_peers(&mut self, peers: impl IntoIterator<Item = PeerId>) -> bool {
        let added: Vec<PeerId> = peers
            .into_iter()
            .filter(|p| *p != self.local && !self.known.contains_key(p))
            .collect();
        if added.is_empty() {
            return false;
        }
        for peer in &added {
            self.known.insert(*peer, PeerState::default());
        }
        // Monotonic: a peer is only ever announced by someone the first time it
        // learns of it, so this terminates.
        let announce = Frame::Peers { peers: added };
        for peer in self.conns.keys().copied().collect::<Vec<_>>() {
            self.send_to(peer, &announce);
        }
        true
    }

    // --- sending ---

    fn broadcast(&mut self, payload: Vec<u8>) {
        self.next_seq += 1;
        let frame = Frame::Data {
            origin: self.local,
            seq: self.next_seq,
            payload,
        };
        let Some(raw) = encode(&frame) else { return };
        for neighbor in self.conns.values() {
            let _ = neighbor.out.send(raw.clone());
        }
    }

    /// Pass a payload on to everyone except whoever just gave it to us.
    fn forward(&self, raw: &[u8], from: PeerId) {
        for (peer, neighbor) in &self.conns {
            if *peer != from {
                let _ = neighbor.out.send(raw.to_vec());
            }
        }
    }

    fn send_to(&self, peer: PeerId, frame: &Frame) {
        let Some(neighbor) = self.conns.get(&peer) else {
            return;
        };
        let Some(raw) = encode(frame) else { return };
        let _ = neighbor.out.send(raw);
    }
}

impl PeerState {
    fn reset(&mut self) {
        self.failures = 0;
        self.wait_ticks = 0;
    }
}

fn encode(frame: &Frame) -> Option<Vec<u8>> {
    match frame.encode() {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::error!("failed to encode mesh frame: {e}");
            None
        }
    }
}

/// Decode frames off one connection until it ends, then report the loss.
async fn read_loop<C: MeshConn>(
    mut recv: C::Recv,
    peer: PeerId,
    conn_id: u64,
    inbox: mpsc::UnboundedSender<Ev<C>>,
    max_frame: usize,
) {
    loop {
        let raw = match recv.recv().await {
            Ok(Some(raw)) => raw,
            Ok(None) => break,
            Err(e) => {
                tracing::debug!(%peer, "mesh connection read failed: {e}");
                break;
            }
        };
        if raw.len() > max_frame {
            tracing::warn!(%peer, len = raw.len(), "oversized mesh frame; dropping peer");
            break;
        }
        match Frame::decode(&raw) {
            // One malformed frame is not worth dropping a peer over.
            Err(e) => tracing::warn!(%peer, "undecodable mesh frame: {e}"),
            Ok(frame) => {
                if inbox
                    .send(Ev::Frame {
                        from: peer,
                        frame,
                        raw,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }
    }
    let _ = inbox.send(Ev::ConnLost { peer, conn_id });
}
