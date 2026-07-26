//! Mesh behaviour, proven against an in-memory transport.
//!
//! These cover the properties `iroh-gossip` used to provide, which are the
//! reason the mesh is more than "send to everyone you're connected to":
//! multi-hop delivery over partial connectivity, termination of the flood,
//! swarm discovery from a single contact, and healing after failure.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Notify, mpsc};

use super::*;

// --- an in-memory transport with a controllable topology ---

/// A simulated network. Links can be cut and healed to test partitions.
#[derive(Clone, Default)]
struct Network {
    inner: Arc<Mutex<NetworkInner>>,
}

#[derive(Default)]
struct NetworkInner {
    inbound: HashMap<PeerId, mpsc::UnboundedSender<MemConn>>,
    /// Pairs that may not connect, normalized so order does not matter.
    blocked: HashSet<(PeerId, PeerId)>,
    /// Live links, so an existing connection can be severed mid-flight.
    links: Vec<(PeerId, PeerId, Arc<Notify>)>,
}

fn pair(a: PeerId, b: PeerId) -> (PeerId, PeerId) {
    if a <= b { (a, b) } else { (b, a) }
}

impl Network {
    /// Attach a node and hand back its transport.
    fn node(&self, local: PeerId) -> MemTransport {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.lock().unwrap().inbound.insert(local, tx);
        MemTransport {
            local,
            net: self.clone(),
            inbound: tokio::sync::Mutex::new(rx),
        }
    }

    /// Sever the link between two peers and refuse further dials.
    fn cut(&self, a: PeerId, b: PeerId) {
        let mut inner = self.inner.lock().unwrap();
        inner.blocked.insert(pair(a, b));
        let key = pair(a, b);
        inner.links.retain(|(x, y, notify)| {
            if pair(*x, *y) == key {
                notify.notify_waiters();
                false
            } else {
                true
            }
        });
    }

    /// Allow the two peers to connect again.
    fn heal(&self, a: PeerId, b: PeerId) {
        self.inner.lock().unwrap().blocked.remove(&pair(a, b));
    }

    fn connect(&self, from: PeerId, to: PeerId) -> TransportResult<MemConn> {
        let mut inner = self.inner.lock().unwrap();
        if inner.blocked.contains(&pair(from, to)) {
            return Err(MeshTransportError::new("link is cut"));
        }
        let peer_inbound = inner
            .inbound
            .get(&to)
            .cloned()
            .ok_or_else(|| MeshTransportError::new("no such peer"))?;

        let (dialer_tx, acceptor_rx) = mpsc::unbounded_channel();
        let (acceptor_tx, dialer_rx) = mpsc::unbounded_channel();
        let severed = Arc::new(Notify::new());
        inner.links.push((from, to, severed.clone()));

        peer_inbound
            .send(MemConn {
                peer: from,
                tx: acceptor_tx,
                rx: acceptor_rx,
                severed: severed.clone(),
            })
            .map_err(|_| MeshTransportError::new("peer is gone"))?;

        Ok(MemConn {
            peer: to,
            tx: dialer_tx,
            rx: dialer_rx,
            severed,
        })
    }
}

struct MemTransport {
    local: PeerId,
    net: Network,
    inbound: tokio::sync::Mutex<mpsc::UnboundedReceiver<MemConn>>,
}

impl MeshTransport for MemTransport {
    type Conn = MemConn;

    fn local_id(&self) -> PeerId {
        self.local
    }

    async fn dial(&self, peer: PeerId) -> TransportResult<MemConn> {
        self.net.connect(self.local, peer)
    }

    async fn accept(&self) -> Option<MemConn> {
        self.inbound.lock().await.recv().await
    }
}

struct MemConn {
    peer: PeerId,
    tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    severed: Arc<Notify>,
}

impl MeshConn for MemConn {
    type Sender = MemSender;
    type Recv = MemRecv;

    fn peer(&self) -> PeerId {
        self.peer
    }

    fn split(self) -> (MemSender, MemRecv) {
        (
            MemSender { tx: self.tx },
            MemRecv {
                rx: self.rx,
                severed: self.severed,
            },
        )
    }
}

#[derive(Clone)]
struct MemSender {
    tx: mpsc::UnboundedSender<Vec<u8>>,
}

impl MeshSender for MemSender {
    async fn send(&self, frame: Vec<u8>) -> TransportResult<()> {
        self.tx
            .send(frame)
            .map_err(|_| MeshTransportError::new("connection closed"))
    }

    fn close(&self) {}
}

struct MemRecv {
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    severed: Arc<Notify>,
}

impl MeshRecv for MemRecv {
    async fn recv(&mut self) -> TransportResult<Option<Vec<u8>>> {
        tokio::select! {
            frame = self.rx.recv() => Ok(frame),
            _ = self.severed.notified() => Ok(None),
        }
    }
}

// --- harness ---

fn peer_id(n: u8) -> PeerId {
    PeerId([n; 32])
}

const TOPIC: TopicId = TopicId::from_bytes([7u8; 32]);

fn config() -> MeshConfig {
    MeshConfig {
        // Fast clock so healing is observable within a test.
        maintenance_interval: Duration::from_millis(25),
        ..MeshConfig::new(TOPIC)
    }
}

/// Bring up one mesh node on `net`.
fn spawn_node(
    net: &Network,
    id: u8,
    bootstrap: &[u8],
) -> (Mesh, mpsc::UnboundedReceiver<MeshEvent>) {
    spawn_node_with(net, id, bootstrap, config())
}

fn spawn_node_with(
    net: &Network,
    id: u8,
    bootstrap: &[u8],
    config: MeshConfig,
) -> (Mesh, mpsc::UnboundedReceiver<MeshEvent>) {
    let transport = net.node(peer_id(id));
    let bootstrap: Vec<PeerId> = bootstrap.iter().copied().map(peer_id).collect();
    Mesh::spawn(transport, config, bootstrap)
}

/// Block until `mesh` has at least `want` neighbours.
async fn wait_for_neighbors(mesh: &Mesh, want: usize) -> Vec<PeerId> {
    for _ in 0..400 {
        let neighbors = mesh.neighbors().await.expect("mesh running");
        if neighbors.len() >= want {
            return neighbors;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for {want} neighbour(s)");
}

/// The next delivered payload, ignoring membership churn.
async fn next_payload(events: &mut mpsc::UnboundedReceiver<MeshEvent>) -> (PeerId, Vec<u8>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(MeshEvent::Received {
                origin, payload, ..
            })) => return (origin, payload),
            Ok(Some(_)) => continue,
            Ok(None) => panic!("mesh event stream ended"),
            Err(_) => panic!("timed out waiting for a payload"),
        }
    }
}

/// Drain payloads that have already arrived, without waiting for more.
fn drained_payloads(events: &mut mpsc::UnboundedReceiver<MeshEvent>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let MeshEvent::Received { payload, .. } = event {
            out.push(payload);
        }
    }
    out
}

// --- tests ---

#[tokio::test]
async fn two_peers_exchange_payloads_in_both_directions() {
    let net = Network::default();
    let (a, mut a_events) = spawn_node(&net, 1, &[]);
    let (b, mut b_events) = spawn_node(&net, 2, &[1]);

    wait_for_neighbors(&a, 1).await;
    wait_for_neighbors(&b, 1).await;

    a.broadcast(b"from a".to_vec()).await.unwrap();
    let (origin, payload) = next_payload(&mut b_events).await;
    assert_eq!(origin, peer_id(1));
    assert_eq!(payload, b"from a");

    b.broadcast(b"from b".to_vec()).await.unwrap();
    let (origin, payload) = next_payload(&mut a_events).await;
    assert_eq!(origin, peer_id(2));
    assert_eq!(payload, b"from b");
}

/// The property that makes this a mesh and not a broadcast list: A and C cannot
/// talk directly, so C only hears A if B forwards. This is what `iroh-gossip`
/// gave us for free.
#[tokio::test]
async fn payload_reaches_a_peer_it_cannot_connect_to_directly() {
    let net = Network::default();
    net.cut(peer_id(1), peer_id(3));

    let (a, _a_events) = spawn_node(&net, 1, &[]);
    let (b, _b_events) = spawn_node(&net, 2, &[1]);
    let (c, mut c_events) = spawn_node(&net, 3, &[2]);

    // A—B and B—C, but never A—C.
    wait_for_neighbors(&b, 2).await;
    wait_for_neighbors(&a, 1).await;
    wait_for_neighbors(&c, 1).await;

    a.broadcast(b"relayed by b".to_vec()).await.unwrap();

    let (origin, payload) = next_payload(&mut c_events).await;
    assert_eq!(origin, peer_id(1), "origin survives the hop");
    assert_eq!(payload, b"relayed by b");

    assert!(
        !c.neighbors().await.unwrap().contains(&peer_id(1)),
        "C must not have reached A directly"
    );
}

/// Flooding a cycle must terminate: every peer delivers each payload once.
#[tokio::test]
async fn duplicates_are_suppressed_around_a_cycle() {
    let net = Network::default();
    let (a, _a_events) = spawn_node(&net, 1, &[]);
    let (b, mut b_events) = spawn_node(&net, 2, &[1]);
    let (c, mut c_events) = spawn_node(&net, 3, &[1, 2]);

    // Fully connected triangle: two paths from A to each of B and C.
    wait_for_neighbors(&a, 2).await;
    wait_for_neighbors(&b, 2).await;
    wait_for_neighbors(&c, 2).await;

    a.broadcast(b"once".to_vec()).await.unwrap();

    assert_eq!(next_payload(&mut b_events).await.1, b"once");
    assert_eq!(next_payload(&mut c_events).await.1, b"once");

    // Give any echo time to circulate before asserting it did not.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        drained_payloads(&mut b_events).is_empty(),
        "B saw a duplicate"
    );
    assert!(
        drained_payloads(&mut c_events).is_empty(),
        "C saw a duplicate"
    );
}

/// A joiner is told about one member (the ticket) and must find the rest.
#[tokio::test]
async fn a_joiner_discovers_the_whole_swarm_from_one_contact() {
    let net = Network::default();
    let (a, _a) = spawn_node(&net, 1, &[]);
    let (b, _b) = spawn_node(&net, 2, &[1]);
    let (c, _c) = spawn_node(&net, 3, &[1, 2]);
    wait_for_neighbors(&a, 2).await;

    // D knows only A.
    let (d, _d) = spawn_node(&net, 4, &[1]);

    let neighbors = wait_for_neighbors(&d, 3).await;
    let found: HashSet<PeerId> = neighbors.into_iter().collect();
    assert_eq!(
        found,
        HashSet::from([peer_id(1), peer_id(2), peer_id(3)]),
        "joiner should have been introduced to the whole swarm"
    );

    // And the existing members learned about the newcomer.
    for member in [&a, &b, &c] {
        let neighbors = wait_for_neighbors(member, 3).await;
        assert!(neighbors.contains(&peer_id(4)));
    }
}

/// Connections fail and come back; the mesh must recover without help.
#[tokio::test]
async fn a_severed_link_is_redialed_and_traffic_resumes() {
    let net = Network::default();
    let (a, _a_events) = spawn_node(&net, 1, &[]);
    let (b, mut b_events) = spawn_node(&net, 2, &[1]);
    wait_for_neighbors(&a, 1).await;
    wait_for_neighbors(&b, 1).await;

    net.cut(peer_id(1), peer_id(2));
    for _ in 0..400 {
        if a.neighbors().await.unwrap().is_empty() && b.neighbors().await.unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(
        a.neighbors().await.unwrap().is_empty(),
        "cut should disconnect"
    );

    net.heal(peer_id(1), peer_id(2));

    // No nudge from the application: the maintenance loop must do this itself.
    wait_for_neighbors(&a, 1).await;
    wait_for_neighbors(&b, 1).await;

    a.broadcast(b"after healing".to_vec()).await.unwrap();
    assert_eq!(next_payload(&mut b_events).await.1, b"after healing");
}

/// Backoff must not become "give up": a peer that is down for a long time is
/// still picked up once it returns.
#[tokio::test]
async fn a_peer_that_is_unreachable_for_a_while_is_still_recovered() {
    let net = Network::default();
    net.cut(peer_id(1), peer_id(2));

    let (a, _a_events) = spawn_node(&net, 1, &[]);
    let (_b, mut b_events) = spawn_node(&net, 2, &[1]);

    // Long enough for the exponential backoff to stretch out.
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(a.neighbors().await.unwrap().is_empty());

    net.heal(peer_id(1), peer_id(2));

    wait_for_neighbors(&a, 1).await;
    a.broadcast(b"finally".to_vec()).await.unwrap();
    assert_eq!(next_payload(&mut b_events).await.1, b"finally");
}

/// Sessions must not bleed into each other.
#[tokio::test]
async fn a_peer_from_another_topic_is_refused() {
    let net = Network::default();
    let (a, _a_events) = spawn_node(&net, 1, &[]);

    let other_topic = MeshConfig {
        maintenance_interval: Duration::from_millis(25),
        ..MeshConfig::new(TopicId::from_bytes([9u8; 32]))
    };
    let (b, _b_events) = spawn_node_with(&net, 2, &[1], other_topic);

    // Both sides must reject: whichever direction the connection came from.
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(
        a.neighbors().await.unwrap().is_empty(),
        "a joined a peer from a different session"
    );
    assert!(
        b.neighbors().await.unwrap().is_empty(),
        "b joined a peer from a different session"
    );
}

/// Both peers dialing at once must converge on one connection, not oscillate.
#[tokio::test]
async fn simultaneous_dials_settle_on_a_single_connection() {
    let net = Network::default();
    // Each bootstraps from the other, so both dial immediately.
    let (a, mut a_events) = spawn_node(&net, 1, &[2]);
    let (b, mut b_events) = spawn_node(&net, 2, &[1]);

    wait_for_neighbors(&a, 1).await;
    wait_for_neighbors(&b, 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(a.neighbors().await.unwrap(), vec![peer_id(2)]);
    assert_eq!(b.neighbors().await.unwrap(), vec![peer_id(1)]);

    // The surviving connection must actually work in both directions.
    a.broadcast(b"ping".to_vec()).await.unwrap();
    assert_eq!(next_payload(&mut b_events).await.1, b"ping");
    b.broadcast(b"pong".to_vec()).await.unwrap();
    assert_eq!(next_payload(&mut a_events).await.1, b"pong");
}

/// Ordering and completeness under a burst, across a forwarding hop.
#[tokio::test]
async fn a_burst_arrives_complete_and_in_order_across_a_hop() {
    let net = Network::default();
    net.cut(peer_id(1), peer_id(3));
    let (a, _a) = spawn_node(&net, 1, &[]);
    let (b, _b) = spawn_node(&net, 2, &[1]);
    let (c, mut c_events) = spawn_node(&net, 3, &[2]);
    wait_for_neighbors(&b, 2).await;
    wait_for_neighbors(&c, 1).await;
    let _ = (a.local_id(), c.local_id());

    for i in 0..50u32 {
        a.broadcast(i.to_le_bytes().to_vec()).await.unwrap();
    }

    for expected in 0..50u32 {
        let (_, payload) = next_payload(&mut c_events).await;
        assert_eq!(payload, expected.to_le_bytes(), "payloads must not reorder");
    }
}

/// A peer dropping out must not cost the node its ability to take on new ones.
/// The browser transport originally treated the error raised by a closing peer
/// as "this node is finished accepting", which left the survivors talking to
/// each other but unjoinable — the session quietly died with its founder.
#[tokio::test]
async fn a_peer_departing_does_not_stop_the_node_accepting_newcomers() {
    let net = Network::default();
    let (founder, _founder_events) = spawn_node(&net, 1, &[]);
    let (survivor, mut survivor_events) = spawn_node(&net, 2, &[1]);
    wait_for_neighbors(&survivor, 1).await;

    // The founder goes away abruptly, as a closed browser tab would.
    founder.shutdown();
    net.cut(peer_id(1), peer_id(2));
    for _ in 0..400 {
        if survivor.neighbors().await.unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // A newcomer arrives knowing only the survivor.
    let (newcomer, mut newcomer_events) = spawn_node(&net, 3, &[2]);
    wait_for_neighbors(&survivor, 1).await;
    wait_for_neighbors(&newcomer, 1).await;

    newcomer
        .broadcast(b"hello from the newcomer".to_vec())
        .await
        .unwrap();
    assert_eq!(
        next_payload(&mut survivor_events).await.1,
        b"hello from the newcomer"
    );
    survivor.broadcast(b"welcome".to_vec()).await.unwrap();
    assert_eq!(next_payload(&mut newcomer_events).await.1, b"welcome");
}

#[tokio::test]
async fn shutdown_stops_the_mesh() {
    let net = Network::default();
    let (a, _a_events) = spawn_node(&net, 1, &[]);
    // Bound to a name so the peer stays attached to the network.
    let (_b, _b_events) = spawn_node(&net, 2, &[1]);
    wait_for_neighbors(&a, 1).await;

    a.shutdown();
    for _ in 0..200 {
        if a.neighbors().await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(a.neighbors().await.is_err(), "handle should report closed");
    assert!(a.broadcast(b"nope".to_vec()).await.is_err());
}
