//! Bridges iroh [`CustomSender::poll_send`] / [`CustomEndpoint::poll_recv`] to WebRTC data channels.
//!
//! One [`WebRtcTunnel`] is shared by [`crate::WebRtcTransport`], its [`crate::endpoint::WebRtcEndpoint`], and
//! [`crate::sender::WebRtcSender`]. After JSEP establishes a channel, `WebRtcTransport::attach_data_channel`
//! registers a **route** for the peer's [`CustomAddr`].
//!
//! ## Multi-peer routing (local change; upstream was one channel per transport)
//!
//! The tunnel holds a `CustomAddr -> outbound queue` routing table. Each attached data channel
//! owns its queue's receiving end in its pump task; `poll_send` looks up the destination and
//! `is_valid_send_addr` answers from the table, so one transport (= one iroh endpoint) can hold
//! direct channels to many peers at once. Inbound needs no table: every channel feeds the shared
//! inbound queue and each packet carries its `source_custom`.
//!
//! Attaching an addr that is already routed **replaces** the route: the old queue's senders are
//! dropped, the old pump drains and exits, and its exit-time cleanup no-ops because cleanup is
//! guarded by a per-route generation id. This makes re-negotiation after a dead channel safe even
//! when the old pump has not yet noticed the death.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use iroh_base::CustomAddr;
use tokio::sync::mpsc;

/// Custom transport id for [`CustomAddr`] parts (see iroh `TRANSPORTS.md` for registration).
pub const WEBRTC_TRANSPORT_ID: u64 = u64::from_le_bytes(*b"irohwebr");

/// One inbound datagram worth of bytes from a data channel, tagged with the peer's [`CustomAddr`].
#[derive(Debug)]
pub(crate) struct InboundPacket {
    pub(crate) source_custom: CustomAddr,
    pub(crate) payload: Vec<u8>,
}

const IN_QUEUE: usize = 1024;

/// Optional behavior when attaching a data channel to a [`crate::WebRtcTransport`].
#[derive(Debug, Default, Clone)]
pub struct AttachOptions {
    /// If true, every inbound payload is also sent back on the same data channel (demo echo).
    pub mirror_sctp_echo: bool,
    /// If set, a copy of each inbound payload is forwarded here (e.g. for example logging).
    pub tap_inbound_to: Option<mpsc::UnboundedSender<Vec<u8>>>,
}

/// The outbound side of one attached data channel.
#[derive(Debug)]
struct Route {
    /// Generation id: [`WebRtcTunnel::remove_route`] only removes when it matches,
    /// so a replaced route's late cleanup cannot tear down its successor.
    id: u64,
    out_tx: mpsc::UnboundedSender<Vec<u8>>,
}

/// Shared bridge between iroh custom transport I/O and the attached data channels.
#[derive(Debug)]
pub(crate) struct WebRtcTunnel {
    /// Opaque local address bytes (same as [`crate::WebRtcTransport::local_addr`] data).
    #[allow(dead_code)]
    local_addr_bytes: Vec<u8>,
    bound: AtomicBool,
    in_tx: mpsc::Sender<InboundPacket>,
    in_rx: Mutex<Option<mpsc::Receiver<InboundPacket>>>,
    routes: RwLock<HashMap<CustomAddr, Route>>,
    next_route_id: AtomicU64,
    recv_waker: Mutex<Option<std::task::Waker>>,
}

impl WebRtcTunnel {
    pub(crate) fn new(local_addr_bytes: Vec<u8>) -> Arc<Self> {
        let (in_tx, in_rx) = mpsc::channel(IN_QUEUE);
        Arc::new(Self {
            local_addr_bytes,
            bound: AtomicBool::new(false),
            in_tx,
            in_rx: Mutex::new(Some(in_rx)),
            routes: RwLock::new(HashMap::new()),
            next_route_id: AtomicU64::new(0),
            recv_waker: Mutex::new(None),
        })
    }

    pub(crate) fn mark_bound(&self) -> io::Result<()> {
        if self
            .bound
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(io::Error::other(
                "WebRtcTransport::bind: only one bind() is supported per WebRtcTransport instance",
            ));
        }
        Ok(())
    }

    pub(crate) fn take_inbound_receiver(&self) -> io::Result<mpsc::Receiver<InboundPacket>> {
        self.in_rx
            .lock()
            .map_err(|_| io::Error::other("poisoned tunnel lock"))?
            .take()
            .ok_or_else(|| io::Error::other("inbound receiver already taken"))
    }

    /// Registers (or replaces) the outbound route for `remote`, returning the
    /// route's generation id and the receiver its pump must drain.
    pub(crate) fn add_route(
        &self,
        remote: CustomAddr,
    ) -> io::Result<(u64, mpsc::UnboundedReceiver<Vec<u8>>)> {
        let (out_tx, out_rx) = mpsc::unbounded_channel();
        let id = self.next_route_id.fetch_add(1, Ordering::Relaxed);
        let mut routes = self
            .routes
            .write()
            .map_err(|_| io::Error::other("poisoned tunnel lock"))?;
        // Replace: dropping the old Route drops its sender; the old pump sees
        // its queue close and exits.
        routes.insert(remote, Route { id, out_tx });
        Ok((id, out_rx))
    }

    /// Removes `remote`'s route if it is still generation `id` (pump exit-time
    /// cleanup; a replaced route's late cleanup no-ops).
    pub(crate) fn remove_route(&self, remote: &CustomAddr, id: u64) {
        if let Ok(mut routes) = self.routes.write()
            && routes.get(remote).is_some_and(|route| route.id == id)
        {
            routes.remove(remote);
        }
    }

    /// True if a data channel is attached for `addr`.
    pub(crate) fn has_route(&self, addr: &CustomAddr) -> bool {
        self.routes
            .read()
            .is_ok_and(|routes| routes.contains_key(addr))
    }

    /// The outbound queue for `addr`, if a data channel is attached.
    pub(crate) fn route_sender(&self, addr: &CustomAddr) -> Option<mpsc::UnboundedSender<Vec<u8>>> {
        self.routes
            .read()
            .ok()
            .and_then(|routes| routes.get(addr).map(|route| route.out_tx.clone()))
    }

    pub(crate) fn wake_recv_pollers(&self) {
        if let Ok(mut g) = self.recv_waker.lock() {
            if let Some(w) = g.take() {
                w.wake();
            }
        }
    }

    pub(crate) fn register_recv_waker(&self, waker: &std::task::Waker) {
        if let Ok(mut g) = self.recv_waker.lock() {
            *g = Some(waker.clone());
        }
    }

    pub(crate) fn inbound_sender(&self) -> mpsc::Sender<InboundPacket> {
        self.in_tx.clone()
    }
}

/// Build the [`CustomAddr`] for a peer that advertises the given opaque address bytes on this transport id.
pub fn custom_addr_from_opaque_data(addr_data: &[u8]) -> CustomAddr {
    CustomAddr::from_parts(WEBRTC_TRANSPORT_ID, addr_data)
}
