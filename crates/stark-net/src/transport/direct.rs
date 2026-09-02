//! WebRTC bootstrap for the single-endpoint backend (`webrtc` feature).
//!
//! The session's ordinary iroh endpoint carries a WebRTC
//! [`CustomTransport`](::iroh::endpoint::transports::CustomTransport)
//! (vendor/iroh-webrtc-transport). [`Direct`] is the `webrtc` half of the
//! facade in [`transport`](crate::transport) — the backend holds one either
//! way and never cfg's — and this module owns everything around it:
//!
//! - **Addressing.** A peer's WebRTC [`CustomAddr`] is derived from its
//!   [`EndpointId`] ([`custom_addr_for`]), so no ticket or wire format carries
//!   WebRTC addresses — knowing a peer's id is knowing its custom addr. It
//!   also makes the addr peer-level rather than channel-level: if crossed
//!   simultaneous bootstraps ever leave the two sides sending on different
//!   channels, both still resolve to the same addr and iroh's path validation
//!   is unaffected.
//! - **Signaling.** JSEP (SDP offer/answer, ICE inside the SDP) over one iroh
//!   bi-stream on [`SIGNALING_ALPN`] — which rides the relay on the web, and
//!   whatever exists natively. [`JsepProto`] is the answering side,
//!   registered on the session router by [`Direct::register`].
//! - **Bootstrap.** [`Direct::ensure_direct`] is called for every gossip
//!   neighbor (both ends). The side with the smaller endpoint id offers; the
//!   other side answers via [`JsepProto`]. Once the channel attaches, *both*
//!   sides [`announce`] the peer's custom addr with `Endpoint::add_addr` (a
//!   patched-iroh API), which opens the custom path on every live connection
//!   to that peer — gossip and catch-up conns *migrate* onto WebRTC, keeping
//!   their relay paths as fallback. Crucially the addr is introduced only
//!   *after* the channel attaches, and never at dial time; see
//!   [`Direct::bootstrap`].
//! - **Repair.** [`Direct::maintain`] re-runs `ensure_direct` over the live
//!   neighbor set on a slow cadence, because a channel can die with no gossip
//!   event to say so.
//!
//! Nothing here is in the dial or data path: if bootstrap fails (no WebRTC,
//! peer without the feature, handshake failure), connections simply stay on
//! the paths they already have.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ::iroh::endpoint::{Builder, Connection};
use ::iroh::protocol::{AcceptError, ProtocolHandler, RouterBuilder};
use ::iroh::{Endpoint, EndpointAddr, EndpointId, TransportAddr};
use iroh_base::CustomAddr;
use iroh_webrtc_transport::{
    AttachOptions, QuicSignaling, WebRtcTransport, custom_addr_from_opaque_data,
};

use crate::cancel::Cancel;
use crate::neighbors::Neighbors;

/// JSEP signaling for the session's WebRTC channels, distinct from the gossip
/// and catch-up ALPNs.
const SIGNALING_ALPN: &[u8] = b"stark/webrtc-sig/0";

const DC_LABEL: &str = "stark/webrtc";

/// Give up on one bootstrap attempt after this long (covers the QUIC dial,
/// SDP exchange, ICE, and the channel opening).
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Delays before each bootstrap attempt.
const ATTEMPT_DELAYS: [Duration; 3] = [
    Duration::ZERO,
    Duration::from_secs(1),
    Duration::from_secs(4),
];

/// Cadence of the [`Direct::maintain`] sweep over the live neighbor set.
const REBOOTSTRAP_SWEEP: Duration = Duration::from_secs(60);

/// A peer's WebRTC custom addr, derived from its endpoint id.
fn custom_addr_for(id: EndpointId) -> CustomAddr {
    custom_addr_from_opaque_data(id.as_bytes())
}

/// `addr` plus the WebRTC custom addr derived from its id.
fn with_custom_addr(mut addr: EndpointAddr) -> EndpointAddr {
    let custom = TransportAddr::Custom(custom_addr_for(addr.id));
    addr.addrs.insert(custom);
    addr
}

/// Everything WebRTC the backend touches, behind one handle: the transport on
/// the session's endpoint, the bootstraps, the dial-time addr upgrade.
#[derive(Debug, Clone)]
pub(crate) struct Direct {
    transport: Arc<WebRtcTransport>,
    /// The session's stop signal, for the bootstraps
    /// [`ensure_direct`](Direct::ensure_direct) spawns.
    cancel: Cancel,
    /// Peers with a bootstrap in flight. Two quick NeighborUp events for
    /// one peer must not race two tasks — the same structural rule-out the
    /// waitlist uses for fetches.
    bootstrapping: Arc<Mutex<HashSet<EndpointId>>>,
}

impl Direct {
    /// The transport's advertised custom addr is derived from `local`, our own
    /// endpoint id.
    pub fn new(local: EndpointId, cancel: Cancel) -> Self {
        Self {
            transport: Arc::new(WebRtcTransport::new(local.as_bytes().to_vec())),
            cancel,
            bootstrapping: Arc::default(),
        }
    }

    /// Add the WebRTC custom transport to the endpoint being built.
    pub fn install(&self, builder: Builder) -> Builder {
        builder.add_custom_transport(self.transport.clone())
    }

    /// Accept JSEP signaling on the session router — the answering side of
    /// [`ensure_direct`](Direct::ensure_direct).
    pub fn register(&self, router: RouterBuilder, endpoint: &Endpoint) -> RouterBuilder {
        router.accept(
            SIGNALING_ALPN,
            JsepProto::new(
                endpoint.clone(),
                self.transport.clone(),
                self.cancel.clone(),
            ),
        )
    }

    /// Fire-and-forget: make sure a WebRTC channel to `remote` exists or is
    /// being bootstrapped. Called on every gossip neighbor, both ends, as
    /// often as neighbors come and go — cheap when the channel already exists.
    pub fn ensure_direct(&self, endpoint: &Endpoint, remote: EndpointId) {
        // Exactly one side offers — the smaller endpoint id — the other answers
        // via [`JsepProto`]. Gossip neighbor links are mutual, so the offerer
        // always learns of the peer.
        if endpoint.id().as_bytes() >= remote.as_bytes() {
            return;
        }
        if self.transport.is_attached(&custom_addr_for(remote)) {
            return;
        }
        if self.cancel.stopped() {
            return;
        }
        // Two quick NeighborUp events for one peer must not race two bootstraps:
        // the insert is the claim, and the guard's drop the release — held by the
        // task so even a panic inside it frees the slot, rather than leaving the
        // peer marked in-flight and never bootstrapped again.
        let Some(claim) = Claim::take(&self.bootstrapping, remote) else {
            return;
        };
        let this = self.clone();
        let endpoint = endpoint.clone();
        n0_future::task::spawn(async move {
            let _claim = claim;
            this.offer(&endpoint, remote).await;
        });
    }

    /// If a WebRTC channel to this peer is already attached, add its
    /// (derived) custom addr so the connection starts direct. Never before
    /// it attaches: a custom path opened against an unattached channel
    /// fails validation and blocks the later real open (see
    /// [`Direct::bootstrap`]).
    pub fn dial_addr(&self, addr: EndpointAddr) -> EndpointAddr {
        if self.transport.is_attached(&custom_addr_for(addr.id)) {
            with_custom_addr(addr)
        } else {
            addr
        }
    }

    /// The re-bootstrap sweep, for the session's life: walk the live neighbors
    /// and re-run [`ensure_direct`](Direct::ensure_direct) — nearly free, since
    /// it returns early when attached, answering-side, in flight, or stopped.
    ///
    /// Without it, migration is one-shot: `ensure_direct` fires on NeighborUp,
    /// but a data channel that dies later (the vendored pump removes its route)
    /// re-fires nothing — gossip keeps flowing on the relay, so no NeighborUp
    /// comes — and the same holds when every bootstrap attempt fell in one
    /// transient outage. This sweep is what makes "relay kept as fallback"
    /// true over time rather than at bootstrap only.
    pub async fn maintain(&self, endpoint: &Endpoint, neighbors: Neighbors) {
        while self.cancel.sleep(REBOOTSTRAP_SWEEP).await {
            for peer in neighbors.snapshot() {
                self.ensure_direct(endpoint, peer);
            }
        }
    }

    /// The offerer's bounded attempts, ending with the channel attached, the
    /// retries spent, or the session over.
    async fn offer(&self, endpoint: &Endpoint, remote: EndpointId) {
        for delay in ATTEMPT_DELAYS {
            if !delay.is_zero() && !self.cancel.sleep(delay).await {
                return;
            }
            if self.cancel.stopped() {
                return;
            }
            if self.transport.is_attached(&custom_addr_for(remote)) {
                return;
            }
            let attempt = self.bootstrap(endpoint, remote);
            match n0_future::time::timeout(ATTEMPT_TIMEOUT, attempt).await {
                Ok(Ok(())) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc channel attached (offerer)");
                    return;
                }
                Ok(Err(e)) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc bootstrap attempt failed: {e:#}");
                }
                Err(_) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc bootstrap attempt timed out");
                }
            }
        }
        tracing::debug!(
            remote = %remote.fmt_short(),
            "webrtc bootstrap gave up; traffic stays on existing paths"
        );
    }

    /// One bootstrap attempt: dial JSEP, negotiate as offerer, attach, announce.
    async fn bootstrap(&self, endpoint: &Endpoint, remote: EndpointId) -> anyhow::Result<()> {
        // Deliberately dialed WITHOUT the custom addr: introducing it before the
        // channel is attached makes iroh open the path immediately, whose
        // validation then fails against the unattached transport — and a path
        // mid-validation blocks the post-attach open (it is "already open"),
        // leaving the connection stuck on the relay. The dial rides whatever
        // paths already work (relay on the web).
        let conn = endpoint
            .connect(EndpointAddr::new(remote), SIGNALING_ALPN)
            .await?;
        let (send, recv) = conn.open_bi().await?;
        let mut sig = QuicSignaling::new(send, recv);
        let peer = peer::offer(&mut sig).await?;
        self.transport.attach_data_channel(
            peer,
            custom_addr_for(remote),
            AttachOptions::default(),
        )?;
        conn.close(0u32.into(), b"signaling done");
        announce(endpoint.clone(), remote, self.cancel.clone());
        Ok(())
    }
}

/// The negotiated peer type and the negotiate calls differ per target (str0m
/// on native, the browser's RTCPeerConnection on wasm); the protocol is
/// identical. The vendored crate's `anyhow` stays contained at this edge.
mod peer {
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    mod imp {
        use iroh_webrtc_transport::{QuicSignaling, Str0mPeer};

        pub async fn offer(sig: &mut QuicSignaling) -> anyhow::Result<Str0mPeer> {
            iroh_webrtc_transport::negotiate_dc_as_offerer(sig, super::super::DC_LABEL).await
        }

        pub async fn answer(sig: &mut QuicSignaling) -> anyhow::Result<Str0mPeer> {
            iroh_webrtc_transport::negotiate_dc_as_answerer(sig).await
        }
    }

    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    mod imp {
        use iroh_webrtc_transport::{QuicSignaling, WebPeerConfig, WebRtcPeer};

        pub async fn offer(sig: &mut QuicSignaling) -> anyhow::Result<WebRtcPeer> {
            iroh_webrtc_transport::negotiate_dc_as_offerer(
                sig,
                super::super::DC_LABEL,
                &WebPeerConfig::default(),
            )
            .await
        }

        pub async fn answer(sig: &mut QuicSignaling) -> anyhow::Result<WebRtcPeer> {
            iroh_webrtc_transport::negotiate_dc_as_answerer(sig, &WebPeerConfig::default()).await
        }
    }

    pub(super) use imp::{answer, offer};
}

/// Answer one JSEP offer arriving on `conn` and attach the channel.
///
/// The answerer announces too (not just the offerer): path opening is only
/// initiated by the QUIC *client* of each connection, so whichever end dialed
/// a given connection must know the peer's custom addr for that connection to
/// migrate.
async fn answer_one(
    endpoint: &Endpoint,
    transport: &Arc<WebRtcTransport>,
    cancel: &Cancel,
    conn: &Connection,
) -> anyhow::Result<()> {
    let remote = conn.remote_id();
    let (send, recv) = conn.accept_bi().await?;
    let mut sig = QuicSignaling::new(send, recv);
    let peer = peer::answer(&mut sig).await?;
    transport.attach_data_channel(peer, custom_addr_for(remote), AttachOptions::default())?;
    tracing::debug!(remote = %remote.fmt_short(), "webrtc channel attached (answerer)");
    announce(endpoint.clone(), remote, cancel.clone());
    Ok(())
}

/// Answers JSEP offers and attaches the negotiated channel (native: inline in
/// the handler, whose future is `Send`).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
#[derive(Debug, Clone)]
struct JsepProto {
    endpoint: Endpoint,
    transport: Arc<WebRtcTransport>,
    cancel: Cancel,
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
impl JsepProto {
    fn new(endpoint: Endpoint, transport: Arc<WebRtcTransport>, cancel: Cancel) -> Self {
        Self {
            endpoint,
            transport,
            cancel,
        }
    }
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
impl ProtocolHandler for JsepProto {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Bounded: a stuck offerer must not hold an accept task open for the
        // session's life. Quiet on timeout — bootstrap is not the data path —
        // and returning drops `conn`, which closes it.
        let answered = n0_future::time::timeout(
            ATTEMPT_TIMEOUT,
            answer_one(&self.endpoint, &self.transport, &self.cancel, &conn),
        )
        .await;
        match answered {
            Ok(result) => result.map_err(|e| AcceptError::from_boxed(e.into()))?,
            Err(_) => {
                tracing::debug!(remote = %conn.remote_id().fmt_short(), "webrtc answer timed out");
                return Ok(());
            }
        }
        // Hold the signaling connection open until the offerer closes it; it
        // is their signal that bootstrap (attach + addr announcement) is done.
        conn.closed().await;
        Ok(())
    }
}

/// Answers JSEP offers and attaches the negotiated channel. On wasm the
/// browser negotiation is `!Send` (`JsFuture`, `Rc`) but iroh requires `Send`
/// handler futures, so the handler only queues the connection; a local worker
/// task does the negotiating.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
#[derive(Debug, Clone)]
struct JsepProto {
    worker: tokio::sync::mpsc::UnboundedSender<Arc<Connection>>,
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
impl JsepProto {
    fn new(endpoint: Endpoint, transport: Arc<WebRtcTransport>, cancel: Cancel) -> Self {
        let (worker, mut inbox) = tokio::sync::mpsc::unbounded_channel::<Arc<Connection>>();
        // Exits when the router (and with it every JsepProto clone) is
        // dropped. Sequential is fine: one channel per peer, rarely — but only
        // because each answer is bounded, or one stuck offerer would starve
        // every later bootstrap for the session's life.
        n0_future::task::spawn(async move {
            while let Some(conn) = inbox.recv().await {
                let answered = n0_future::time::timeout(
                    ATTEMPT_TIMEOUT,
                    answer_one(&endpoint, &transport, &cancel, &conn),
                )
                .await;
                match answered {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::debug!("webrtc answer failed: {e:#}"),
                    Err(_) => tracing::debug!("webrtc answer timed out"),
                }
            }
        });
        Self { worker }
    }
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
impl ProtocolHandler for JsepProto {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let conn = Arc::new(conn);
        if self.worker.send(conn.clone()).is_err() {
            return Ok(());
        }
        // Hold the connection open for the worker (returning would tear it
        // down); the offerer closes it when bootstrap is done.
        conn.closed().await;
        Ok(())
    }
}

/// A peer's slot in the in-flight bootstrap set, released on drop.
struct Claim {
    set: Arc<Mutex<HashSet<EndpointId>>>,
    peer: EndpointId,
}

impl Claim {
    /// `None` when a bootstrap to `peer` is already in flight.
    fn take(set: &Arc<Mutex<HashSet<EndpointId>>>, peer: EndpointId) -> Option<Self> {
        set.lock()
            .expect("bootstraps poisoned")
            .insert(peer)
            .then(|| Self {
                set: set.clone(),
                peer,
            })
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.set
            .lock()
            .expect("bootstraps poisoned")
            .remove(&self.peer);
    }
}

/// How long to keep re-announcing before concluding migration won't happen.
const ANNOUNCE_RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(3),
    Duration::from_secs(9),
];

/// Announce `remote`'s (derived) custom addr now that its channel is
/// attached: the patched iroh opens it as a path on every live connection to
/// the peer, and the path selector migrates traffic onto it once it
/// validates. Path opening is only initiated by each connection's QUIC
/// client, so *both* ends announce after attaching.
///
/// Re-announces (bounded) until the selected path is the custom one: a
/// connection that had opened the path *before* the channel attached (its
/// addr can be known from an earlier session) ignores the first announcement
/// while that stale path is still failing validation, and accepts a re-open
/// only after it is abandoned.
fn announce(endpoint: Endpoint, remote: EndpointId, cancel: Cancel) {
    n0_future::task::spawn(async move {
        if cancel.stopped() {
            return;
        }
        let mixed = with_custom_addr(EndpointAddr::new(remote));
        endpoint.add_addr(mixed.clone()).await;
        for delay in ANNOUNCE_RETRY_DELAYS {
            if !cancel.sleep(delay).await {
                return;
            }
            let selected = endpoint
                .remote_info(remote)
                .await
                .and_then(|info| info.selected_addr().cloned());
            if matches!(selected, Some(::iroh::TransportAddr::Custom(_))) {
                return;
            }
            if cancel.stopped() {
                return;
            }
            endpoint.add_addr(mixed.clone()).await;
        }
    });
}

/// The full production flow, natively: two peers with relay + WebRTC custom
/// transport only (IP cleared, like a browser), gossip between them over the
/// relay, JSEP-over-QUIC on each new neighbor, and the patched iroh migrates
/// the live gossip connection onto the WebRTC path — exactly what a browser
/// session does, minus the browser (str0m stands in for RTCPeerConnection;
/// the protocol is the same).
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ::iroh::address_lookup::MemoryLookup;
    use ::iroh::endpoint::presets;
    use ::iroh::protocol::Router;
    use ::iroh::test_utils::run_relay_server;
    use ::iroh::tls::CaTlsConfig;
    use ::iroh::{RelayMap, RelayMode, SecretKey, Watcher};
    use iroh_gossip::api::{Event as GossipEvent, GossipSender};
    use iroh_gossip::net::Gossip;
    use iroh_gossip::proto::TopicId;
    use n0_future::StreamExt;
    use tokio::sync::mpsc;

    use super::*;

    const TOPIC: TopicId = TopicId::from_bytes([7u8; 32]);

    struct TestPeer {
        endpoint: Endpoint,
        direct: Direct,
        lookup: MemoryLookup,
        router: Router,
        gossip: Gossip,
    }

    /// Relay + WebRTC only (IP transports cleared): on loopback a direct UDP
    /// path would always win selection and the migration would be invisible.
    /// This mirrors the browser, which has no IP transport at all.
    async fn bind_peer(relay: RelayMap) -> TestPeer {
        let secret = SecretKey::generate();
        let direct = Direct::new(secret.public(), Cancel::default());
        let lookup = MemoryLookup::new();
        let builder = Endpoint::builder(presets::N0)
            .secret_key(secret)
            .relay_mode(RelayMode::Custom(relay))
            .ca_tls_config(CaTlsConfig::insecure_skip_verify())
            .clear_address_lookup()
            .address_lookup(lookup.clone())
            .clear_ip_transports();
        let endpoint = direct.install(builder).bind().await.expect("bind endpoint");
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let router = Router::builder(endpoint.clone()).accept(iroh_gossip::ALPN, gossip.clone());
        let router = direct.register(router, &endpoint).spawn();
        TestPeer {
            endpoint,
            direct,
            lookup,
            router,
            gossip,
        }
    }

    /// Wait until the endpoint has published a relay address, and return it.
    async fn relay_addr(peer: &TestPeer) -> EndpointAddr {
        let mut w = peer.endpoint.watch_addr();
        for _ in 0..300 {
            let a = w.get();
            if a.addrs.iter().any(|x| x.is_relay()) {
                return a;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for a relay addr");
    }

    /// Subscribe and run the session's receive-loop shape: bootstrap WebRTC
    /// on every new neighbor, forward received payloads. The returned oneshot
    /// fires on the first neighbor — the "joined" moment.
    async fn join(
        peer: &TestPeer,
        bootstrap: &[&TestPeer],
    ) -> (
        GossipSender,
        mpsc::UnboundedReceiver<Vec<u8>>,
        tokio::sync::oneshot::Receiver<()>,
    ) {
        let bootstrap = bootstrap.iter().map(|p| p.endpoint.id()).collect();
        let sub = peer
            .gossip
            .subscribe(TOPIC, bootstrap)
            .await
            .expect("subscribe");
        let (sender, mut receiver) = sub.split();
        let (tx, rx) = mpsc::unbounded_channel();
        let (joined_tx, joined_rx) = tokio::sync::oneshot::channel();
        let mut joined_tx = Some(joined_tx);
        let endpoint = peer.endpoint.clone();
        let direct = peer.direct.clone();
        tokio::spawn(async move {
            while let Some(Ok(event)) = receiver.next().await {
                match event {
                    GossipEvent::NeighborUp(id) => {
                        direct.ensure_direct(&endpoint, id);
                        if let Some(joined) = joined_tx.take() {
                            let _ = joined.send(());
                        }
                    }
                    GossipEvent::Received(message) => {
                        // Best effort: the test may have finished with the rx.
                        let _ = tx.send(message.content.to_vec());
                    }
                    _ => {}
                }
            }
        });
        (sender, rx, joined_rx)
    }

    /// Wait until traffic to every live remote travels over the WebRTC path
    /// (the remote map's *selected* path — relay stays open as backup).
    async fn wait_for_webrtc_selected(peer: &TestPeer, remote: EndpointId) {
        for _ in 0..600 {
            let selected = peer
                .endpoint
                .remote_info(remote)
                .await
                .and_then(|info| info.selected_addr().cloned());
            if matches!(selected, Some(TransportAddr::Custom(_))) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for the WebRTC path to be selected");
    }

    async fn next_payload(rx: &mut mpsc::UnboundedReceiver<Vec<u8>>) -> Vec<u8> {
        tokio::time::timeout(Duration::from_secs(20), rx.recv())
            .await
            .expect("timed out waiting for a payload")
            .expect("gossip event stream ended")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn gossip_links_migrate_from_relay_to_webrtc() {
        let (relay_map, _url, _relay_guard) = run_relay_server().await.expect("local relay server");

        let a = bind_peer(relay_map.clone()).await;
        let b = bind_peer(relay_map.clone()).await;
        let a_addr = relay_addr(&a).await;
        let b_addr = relay_addr(&b).await;
        a.lookup.add_endpoint_info(b_addr);
        b.lookup.add_endpoint_info(a_addr);

        let (a_send, mut a_recv, a_joined) = join(&a, &[]).await;
        let (b_send, mut b_recv, b_joined) = join(&b, &[&a]).await;
        tokio::time::timeout(Duration::from_secs(20), async {
            a_joined.await.expect("a receive loop died");
            b_joined.await.expect("b receive loop died");
        })
        .await
        .expect("timed out waiting for the swarm to form");

        // The topic works from the start (over the relay)...
        a_send
            .broadcast(bytes::Bytes::from_static(b"over the relay, probably"))
            .await
            .expect("broadcast");
        assert_eq!(next_payload(&mut b_recv).await, b"over the relay, probably");

        // ...and migrates onto WebRTC without any reconnect, on BOTH ends.
        wait_for_webrtc_selected(&a, b.endpoint.id()).await;
        wait_for_webrtc_selected(&b, a.endpoint.id()).await;

        // Still the same connection, now direct: traffic keeps flowing.
        b_send
            .broadcast(bytes::Bytes::from_static(b"over webrtc"))
            .await
            .expect("broadcast");
        assert_eq!(next_payload(&mut a_recv).await, b"over webrtc");

        let _ = a.router.shutdown().await;
        let _ = b.router.shutdown().await;
        a.endpoint.close().await;
        b.endpoint.close().await;
    }
}
