//! How a session binds to the network and reaches its peers.
//!
//! Binding the stack — the endpoint, the router, gossip — plus opening
//! catch-up connections, minting the ticket address, and shutting down.
//! [`session`](crate::session) drives it and owns the protocol above it.
//!
//! The backend is an ordinary iroh [`Endpoint`](iroh::Endpoint) and
//! [`Router`](iroh::protocol::Router) on every target; in the browser iroh
//! rides its relay (WebSocket) transport. With the **`webrtc`** feature the
//! endpoint additionally carries a WebRTC custom transport: connections
//! establish over whatever works and migrate onto a WebRTC path once a data
//! channel is bootstrapped (see [`transport::direct`](crate::transport::direct))
//! — which is what gives the *web* direct connections.

mod imp {
    //! Plain iroh: an endpoint, a router, and gossip.

    #[cfg(not(target_arch = "wasm32"))]
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use iroh::endpoint::{Connection, presets};
    use iroh::protocol::Router;
    use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, TransportAddr};
    use iroh_gossip::net::Gossip;

    use crate::Result;
    use crate::mirror::Mirror;
    use crate::proto::{self, CollabProto, Request};
    use crate::session::NetOptions;

    /// How long to wait for relay/publish readiness before minting a ticket.
    const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

    /// A long stroke's fitted control points can be sizeable, so the gossip
    /// message ceiling sits well past any plausible single action (paths are
    /// RDP-simplified; pixels never ride gossip).
    const MAX_MESSAGE_SIZE: usize = 256 * 1024;

    pub(crate) struct Bound {
        pub dialer: Dialer,
        pub gossip: Gossip,
        pub shutdown: Shutdown,
    }

    pub(crate) async fn bind(mirror: Arc<Mutex<Mirror>>, opts: &NetOptions) -> Result<Bound> {
        let secret = opts.secret.clone().unwrap_or_else(SecretKey::generate);
        // The WebRTC custom transport rides the same endpoint; peers derive
        // its addr from our endpoint id (see transport::direct).
        #[cfg(feature = "webrtc")]
        let webrtc = crate::transport::direct::make_transport(secret.public());

        let builder = if opts.local_only {
            Endpoint::builder(presets::Minimal).secret_key(secret)
        } else {
            Endpoint::builder(presets::N0).secret_key(secret)
        };
        #[cfg(feature = "webrtc")]
        let builder = builder.add_custom_transport(webrtc.clone());
        let endpoint = builder.bind().await?;

        let gossip = Gossip::builder()
            .max_message_size(MAX_MESSAGE_SIZE)
            .spawn(endpoint.clone());

        let router = Router::builder(endpoint.clone())
            .accept(iroh_gossip::ALPN, gossip.clone())
            .accept(proto::ALPN, CollabProto { mirror });
        #[cfg(feature = "webrtc")]
        let router = router.accept(
            crate::transport::direct::SIGNALING_ALPN,
            crate::transport::direct::JsepProto::new(endpoint.clone(), webrtc.clone()),
        );
        let router = router.spawn();

        Ok(Bound {
            dialer: Dialer {
                endpoint: endpoint.clone(),
                #[cfg(feature = "webrtc")]
                webrtc,
            },
            gossip,
            shutdown: Shutdown { endpoint, router },
        })
    }

    #[derive(Clone)]
    pub(crate) struct Dialer {
        endpoint: Endpoint,
        #[cfg(feature = "webrtc")]
        webrtc: Arc<iroh_webrtc_transport::WebRtcTransport>,
    }

    impl Dialer {
        pub fn local_id(&self) -> Result<EndpointId> {
            Ok(self.endpoint.id())
        }

        /// Fire-and-forget: bootstrap a WebRTC channel to `remote` unless one
        /// exists (or we are the answering side). Called on every gossip
        /// neighbor; connections migrate onto the channel once it attaches.
        pub fn ensure_direct(&self, remote: EndpointId) {
            #[cfg(feature = "webrtc")]
            crate::transport::direct::ensure_direct(&self.endpoint, &self.webrtc, remote);
            #[cfg(not(feature = "webrtc"))]
            let _ = remote;
        }

        /// The address traffic to `remote` currently travels over — the remote
        /// map's selected path. `None` when no path selection exists (no live
        /// connection). Sampled per call; paths migrate.
        pub async fn selected_addr(&self, remote: EndpointId) -> Option<TransportAddr> {
            self.endpoint
                .remote_info(remote)
                .await?
                .selected_addr()
                .cloned()
        }

        /// Connecting also teaches the endpoint how to reach `addr`, which is
        /// what later lets gossip dial the same peer by bare id.
        pub async fn open(&self, addr: EndpointAddr) -> Result<Catchup> {
            // If a WebRTC channel to this peer is already attached, dial with
            // its (derived) custom addr so the connection starts direct. Never
            // before it attaches: a custom path opened against an unattached
            // channel fails validation and blocks the later real open (see
            // transport::direct::bootstrap).
            #[cfg(feature = "webrtc")]
            let addr = if self
                .webrtc
                .is_attached(&crate::transport::direct::custom_addr_for(addr.id))
            {
                crate::transport::direct::with_custom_addr(addr)
            } else {
                addr
            };
            let conn = self.endpoint.connect(addr, proto::ALPN).await?;
            Ok(Catchup { conn })
        }

        /// With public infrastructure, wait (bounded) for the relay handshake so
        /// the ticket carries a relay URL; local-only tickets carry the bound
        /// sockets, loopback-normalized.
        pub async fn ticket_addr(&self, opts: &NetOptions) -> Result<EndpointAddr> {
            if !opts.local_only {
                // `online()` pends forever with no WAN; bound wait, then best effort.
                let _ = n0_future::time::timeout(ONLINE_TIMEOUT, self.endpoint.online()).await;
                return Ok(self.endpoint.addr());
            }
            // Local-only is native-only: a browser has no UDP sockets to
            // advertise, so there it yields a bare-id ticket that only
            // same-machine tests could ever have used anyway.
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut addr = EndpointAddr::new(self.endpoint.id());
                for sock in self.endpoint.bound_sockets() {
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
                Ok(addr)
            }
            #[cfg(target_arch = "wasm32")]
            Ok(EndpointAddr::new(self.endpoint.id()))
        }
    }

    pub(crate) struct Catchup {
        conn: Connection,
    }

    impl Catchup {
        pub async fn request(&self, req: Request) -> Result<Vec<u8>> {
            proto::request(&self.conn, req).await
        }

        pub async fn close(self) {
            self.conn.close(0u8.into(), b"done");
        }
    }

    pub(crate) struct Shutdown {
        endpoint: Endpoint,
        router: Router,
    }

    impl Shutdown {
        pub async fn run(self) {
            if let Err(e) = self.router.shutdown().await {
                tracing::warn!("router shutdown: {e}");
            }
            self.endpoint.close().await;
        }
    }
}

// `Bound`, `Catchup` and the transport type are reached through inference from
// `bind`, so only the names the session spells out are re-exported.
pub(crate) use imp::{Dialer, Shutdown, bind};
