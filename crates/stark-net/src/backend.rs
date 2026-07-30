//! How a session binds to the network and reaches its peers.
//!
//! Binding the stack, opening a catch-up connection, minting the ticket
//! address, and shutting down — one seam so [`session`](crate::session) never
//! touches iroh types directly, and the live wire above it — the
//! [`mesh`](crate::mesh) — stays network-agnostic.
//!
//! The backend is an ordinary iroh [`Endpoint`](iroh::Endpoint) and
//! [`Router`](iroh::protocol::Router) on every target; in the browser iroh
//! rides its relay (WebSocket) transport. With the **`webrtc`** feature the
//! endpoint additionally carries a WebRTC custom transport: connections
//! establish over whatever works and migrate onto a WebRTC path once a data
//! channel is bootstrapped (see [`transport::direct`](crate::transport::direct))
//! — which is what gives the *web* direct connections.

mod imp {
    //! Plain iroh: an endpoint and a router.

    #[cfg(not(target_arch = "wasm32"))]
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use iroh::endpoint::{Connection, presets};
    use iroh::protocol::Router;
    use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};

    use crate::Result;
    use crate::mirror::Mirror;
    use crate::proto::{self, CollabProto, Request};
    use crate::session::NetOptions;
    use crate::transport::iroh::IrohMeshTransport;

    pub(crate) type SessionTransport = IrohMeshTransport;

    /// How long to wait for relay/publish readiness before minting a ticket.
    const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

    pub(crate) struct Bound {
        pub dialer: Dialer,
        pub transport: SessionTransport,
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

        let (transport, mesh_proto) = IrohMeshTransport::new(endpoint.clone());
        #[cfg(feature = "webrtc")]
        let transport = transport.with_direct(webrtc.clone());

        let router = Router::builder(endpoint.clone())
            .accept(crate::transport::MESH_ALPN, mesh_proto)
            .accept(proto::ALPN, CollabProto { mirror });
        #[cfg(feature = "webrtc")]
        let router = router.accept(
            crate::transport::direct::SIGNALING_ALPN,
            crate::transport::direct::JsepProto::new(webrtc),
        );
        let router = router.spawn();

        Ok(Bound {
            dialer: Dialer {
                endpoint: endpoint.clone(),
            },
            transport,
            shutdown: Shutdown { endpoint, router },
        })
    }

    #[derive(Clone)]
    pub(crate) struct Dialer {
        endpoint: Endpoint,
    }

    impl Dialer {
        pub fn local_id(&self) -> Result<EndpointId> {
            Ok(self.endpoint.id())
        }

        /// Connecting also teaches the endpoint how to reach `addr`, which is
        /// what later lets the mesh dial the same peer by bare id.
        pub async fn open(&self, addr: EndpointAddr) -> Result<Catchup> {
            // Teach iroh the peer's (derived) WebRTC addr from first contact,
            // so this connection migrates too once a channel attaches.
            #[cfg(feature = "webrtc")]
            let addr = crate::transport::direct::with_custom_addr(addr);
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
