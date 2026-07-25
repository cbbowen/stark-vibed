//! The mesh over ordinary iroh connections.
//!
//! One QUIC bi-stream per peer carries length-prefixed mesh frames. Outbound
//! connections are dialed by [`EndpointId`]; inbound ones arrive through a
//! [`ProtocolHandler`] on the session's router and are queued for
//! [`MeshTransport::accept`].
//!
//! This is the transport for native peers and for the web today (where iroh
//! rides n0's relays). The WebRTC transport replaces only *how* the bytes move;
//! the mesh above it is identical.

use std::sync::Arc;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::protocol::{AcceptError, ProtocolHandler};
use iroh::{Endpoint, EndpointAddr, EndpointId};
use tokio::sync::{mpsc, Mutex};

use crate::mesh::{
    MeshConn, MeshRecv, MeshSender, MeshTransport, MeshTransportError, PeerId, TransportResult,
};

/// The mesh's own ALPN, distinct from the catch-up/asset protocol.
pub(crate) const ALPN: &[u8] = b"stark/mesh/0";

/// Hard ceiling on a single frame, so a bad length prefix cannot make us
/// allocate wildly. Comfortably above the mesh's own `max_frame`.
const MAX_FRAME: usize = 1024 * 1024;

pub(crate) fn to_peer_id(id: EndpointId) -> PeerId {
    PeerId(*id.as_bytes())
}

pub(crate) fn to_endpoint_id(peer: PeerId) -> Option<EndpointId> {
    EndpointId::from_bytes(peer.as_bytes()).ok()
}

/// Dials peers over an [`Endpoint`] and receives connections from [`MeshProto`].
#[derive(Debug)]
pub(crate) struct IrohMeshTransport {
    endpoint: Endpoint,
    inbound: Mutex<mpsc::UnboundedReceiver<IrohConn>>,
}

impl IrohMeshTransport {
    /// Build the transport and the protocol handler that feeds it. Register the
    /// handler on the session router under [`ALPN`].
    pub fn new(endpoint: Endpoint) -> (Self, MeshProto) {
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        (
            Self {
                endpoint,
                inbound: Mutex::new(inbound_rx),
            },
            MeshProto { inbound: inbound_tx },
        )
    }
}

impl MeshTransport for IrohMeshTransport {
    type Conn = IrohConn;

    fn local_id(&self) -> PeerId {
        to_peer_id(self.endpoint.id())
    }

    async fn dial(&self, peer: PeerId) -> TransportResult<IrohConn> {
        let id = to_endpoint_id(peer).ok_or_else(|| MeshTransportError::new("bad peer id"))?;
        // A bare id resolves through address lookup, or through addresses the
        // endpoint already learned from an earlier connection. Peers we only
        // heard about second-hand (no relay, no lookup — the local-only case)
        // may not be dialable; the mesh copes by routing around them.
        let conn = self
            .endpoint
            .connect(EndpointAddr::new(id), ALPN)
            .await
            .map_err(MeshTransportError::new)?;
        let (send, recv) = conn.open_bi().await.map_err(MeshTransportError::new)?;
        Ok(IrohConn {
            peer,
            conn: Arc::new(conn),
            send,
            recv,
        })
    }

    async fn accept(&self) -> Option<IrohConn> {
        self.inbound.lock().await.recv().await
    }
}

/// Accepts inbound mesh connections and hands them to the transport.
#[derive(Debug, Clone)]
pub(crate) struct MeshProto {
    inbound: mpsc::UnboundedSender<IrohConn>,
}

impl ProtocolHandler for MeshProto {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let peer = to_peer_id(conn.remote_id());
        // The dialer writes its greeting immediately, so this resolves promptly.
        let (send, recv) = conn.accept_bi().await?;
        let conn = Arc::new(conn);
        if self
            .inbound
            .send(IrohConn {
                peer,
                conn: conn.clone(),
                send,
                recv,
            })
            .is_err()
        {
            // The session is gone; let the connection close.
            return Ok(());
        }
        // Hold the connection open for as long as the mesh is using it —
        // returning here would tear the streams down.
        conn.closed().await;
        Ok(())
    }
}

/// One peer-to-peer mesh connection.
#[derive(Debug)]
pub(crate) struct IrohConn {
    peer: PeerId,
    /// Kept alive by both halves: dropping the last reference closes the QUIC
    /// connection out from under the streams.
    conn: Arc<Connection>,
    send: SendStream,
    recv: RecvStream,
}

impl MeshConn for IrohConn {
    type Sender = IrohSender;
    type Recv = IrohRecv;

    fn peer(&self) -> PeerId {
        self.peer
    }

    fn split(self) -> (IrohSender, IrohRecv) {
        (
            IrohSender {
                conn: self.conn.clone(),
                send: Arc::new(Mutex::new(self.send)),
            },
            IrohRecv {
                _conn: self.conn,
                recv: self.recv,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IrohSender {
    conn: Arc<Connection>,
    send: Arc<Mutex<SendStream>>,
}

impl MeshSender for IrohSender {
    async fn send(&self, frame: Vec<u8>) -> TransportResult<()> {
        let len = u32::try_from(frame.len())
            .map_err(|_| MeshTransportError::new("frame too large"))?;
        let mut send = self.send.lock().await;
        send.write_all(&len.to_le_bytes())
            .await
            .map_err(MeshTransportError::new)?;
        send.write_all(&frame).await.map_err(MeshTransportError::new)
    }

    fn close(&self) {
        self.conn.close(0u32.into(), b"mesh closed");
    }
}

#[derive(Debug)]
pub(crate) struct IrohRecv {
    _conn: Arc<Connection>,
    recv: RecvStream,
}

impl MeshRecv for IrohRecv {
    async fn recv(&mut self) -> TransportResult<Option<Vec<u8>>> {
        let mut header = [0u8; 4];
        // A failure here is the ordinary end of the stream, not an error worth
        // shouting about: the peer closed, or the connection went away.
        if self.recv.read_exact(&mut header).await.is_err() {
            return Ok(None);
        }
        let len = u32::from_le_bytes(header) as usize;
        if len > MAX_FRAME {
            return Err(MeshTransportError::new(format!(
                "peer announced a {len}-byte frame"
            )));
        }
        let mut frame = vec![0u8; len];
        self.recv
            .read_exact(&mut frame)
            .await
            .map_err(MeshTransportError::new)?;
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::time::Duration;

    use iroh::address_lookup::MemoryLookup;
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::SecretKey;
    use tokio::sync::mpsc;

    use super::*;
    use crate::mesh::{Mesh, MeshConfig, MeshEvent, TopicId};

    const TOPIC: TopicId = TopicId::from_bytes([3u8; 32]);

    struct TestPeer {
        addr: EndpointAddr,
        lookup: MemoryLookup,
        endpoint: Endpoint,
        router: Router,
        transport: Option<IrohMeshTransport>,
    }

    /// Bind a real endpoint on loopback with no relays and no external lookup,
    /// so the test is hermetic. Peers can only reach what we put in `lookup`.
    async fn bind_peer() -> TestPeer {
        let lookup = MemoryLookup::new();
        let endpoint = Endpoint::builder(presets::Minimal)
            .secret_key(SecretKey::generate())
            .address_lookup(lookup.clone())
            .bind()
            .await
            .expect("bind endpoint");
        let (transport, mesh_proto) = IrohMeshTransport::new(endpoint.clone());
        let router = Router::builder(endpoint.clone()).accept(ALPN, mesh_proto).spawn();
        let addr = loopback_addr(&endpoint);
        TestPeer {
            addr,
            lookup,
            endpoint,
            router,
            transport: Some(transport),
        }
    }

    fn loopback_addr(endpoint: &Endpoint) -> EndpointAddr {
        let mut addr = EndpointAddr::new(endpoint.id());
        for sock in endpoint.bound_sockets() {
            let sock = if sock.ip().is_unspecified() {
                let loopback: IpAddr = if sock.is_ipv4() {
                    Ipv4Addr::LOCALHOST.into()
                } else {
                    Ipv6Addr::LOCALHOST.into()
                };
                SocketAddr::new(loopback, sock.port())
            } else {
                sock
            };
            addr = addr.with_ip_addr(sock);
        }
        addr
    }

    fn start_mesh(
        peer: &mut TestPeer,
        bootstrap: &[&TestPeer],
    ) -> (Mesh, mpsc::UnboundedReceiver<MeshEvent>) {
        let config = MeshConfig {
            maintenance_interval: Duration::from_millis(100),
            ..MeshConfig::new(TOPIC)
        };
        let bootstrap: Vec<PeerId> =
            bootstrap.iter().map(|p| to_peer_id(p.endpoint.id())).collect();
        Mesh::spawn(peer.transport.take().expect("one mesh per peer"), config, bootstrap)
    }

    async fn wait_for_neighbors(mesh: &Mesh, want: usize) {
        for _ in 0..400 {
            if mesh.neighbors().await.expect("mesh running").len() >= want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for {want} neighbour(s) over iroh");
    }

    async fn next_payload(events: &mut mpsc::UnboundedReceiver<MeshEvent>) -> (PeerId, Vec<u8>) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, events.recv()).await {
                Ok(Some(MeshEvent::Received { origin, payload, .. })) => return (origin, payload),
                Ok(Some(_)) => continue,
                Ok(None) => panic!("mesh event stream ended"),
                Err(_) => panic!("timed out waiting for a payload over iroh"),
            }
        }
    }

    /// The transport binding itself: real QUIC streams, real framing, real
    /// `ProtocolHandler` accept path.
    #[tokio::test(flavor = "multi_thread")]
    async fn two_peers_exchange_payloads_over_real_endpoints() {
        let mut a = bind_peer().await;
        let mut b = bind_peer().await;
        a.lookup.add_endpoint_info(b.addr.clone());
        b.lookup.add_endpoint_info(a.addr.clone());

        let (a_mesh, mut a_events) = start_mesh(&mut a, &[]);
        let (b_mesh, mut b_events) = start_mesh(&mut b, &[&a]);

        wait_for_neighbors(&a_mesh, 1).await;
        wait_for_neighbors(&b_mesh, 1).await;

        a_mesh.broadcast(b"over quic".to_vec()).await.unwrap();
        let (origin, payload) = next_payload(&mut b_events).await;
        assert_eq!(origin, to_peer_id(a.endpoint.id()));
        assert_eq!(payload, b"over quic");

        b_mesh.broadcast(b"and back".to_vec()).await.unwrap();
        assert_eq!(next_payload(&mut a_events).await.1, b"and back");

        shutdown(a, a_mesh).await;
        shutdown(b, b_mesh).await;
    }

    /// The gossip-equivalence property on the real transport: A can dial B and
    /// B can dial C, but A and C have no way to reach each other — A's payload
    /// must still arrive at C, forwarded by B.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_payload_crosses_a_peer_that_cannot_be_reached_directly() {
        let mut a = bind_peer().await;
        let mut b = bind_peer().await;
        let mut c = bind_peer().await;

        // Only B knows how to reach both ends; A and C never learn each other's
        // address, so their dials cannot succeed.
        a.lookup.add_endpoint_info(b.addr.clone());
        c.lookup.add_endpoint_info(b.addr.clone());
        b.lookup.add_endpoint_info(a.addr.clone());
        b.lookup.add_endpoint_info(c.addr.clone());

        let (a_mesh, _a_events) = start_mesh(&mut a, &[]);
        let (b_mesh, _b_events) = start_mesh(&mut b, &[&a]);
        let (c_mesh, mut c_events) = start_mesh(&mut c, &[&b]);

        wait_for_neighbors(&b_mesh, 2).await;
        wait_for_neighbors(&a_mesh, 1).await;
        wait_for_neighbors(&c_mesh, 1).await;

        a_mesh.broadcast(b"forwarded by b".to_vec()).await.unwrap();

        let (origin, payload) = next_payload(&mut c_events).await;
        assert_eq!(origin, to_peer_id(a.endpoint.id()), "origin survives the hop");
        assert_eq!(payload, b"forwarded by b");
        assert!(
            !c_mesh
                .neighbors()
                .await
                .unwrap()
                .contains(&to_peer_id(a.endpoint.id())),
            "C must not have reached A directly"
        );

        shutdown(a, a_mesh).await;
        shutdown(b, b_mesh).await;
        shutdown(c, c_mesh).await;
    }

    async fn shutdown(peer: TestPeer, mesh: Mesh) {
        mesh.shutdown();
        let _ = peer.router.shutdown().await;
        peer.endpoint.close().await;
    }
}
