//! How a session binds to the network and reaches its peers.
//!
//! Everything that differs between "plain iroh" and "browser WebRTC" is behind
//! this one seam: binding the stack, opening a catch-up connection, minting the
//! ticket address, and shutting down. [`session`](crate::session) uses it
//! without a single `cfg`, and the live wire above it — the
//! [`mesh`](crate::mesh) — is identical either way.
//!
//! The active implementation is chosen at compile time:
//!
//! - **`webrtc` feature on wasm** → the vendored facade. Peer connections are
//!   direct WebRTC where the handshake succeeds and iroh relay where it does
//!   not, decided per connection.
//! - **everything else** → an ordinary iroh [`Endpoint`](iroh::Endpoint) and
//!   [`Router`](iroh::protocol::Router). Native peers hole-punch already, so
//!   WebRTC buys them nothing.

#[cfg(all(feature = "webrtc", target_family = "wasm", target_os = "unknown"))]
mod imp {
    //! Browser WebRTC, via the vendored facade.

    use std::sync::{Arc, Mutex};

    use iroh::{EndpointAddr, EndpointId, SecretKey};
    use iroh_webrtc_transport::browser::{
        BrowserDialOptions, BrowserWebRtcConnection, BrowserWebRtcNode, BrowserWebRtcNodeConfig,
    };

    use crate::mirror::Mirror;
    use crate::proto::{self, Request};
    use crate::session::NetOptions;
    use crate::transport::webrtc::{self, WebRtcMeshTransport};
    use crate::{NetError, Result};

    pub(crate) type SessionTransport = WebRtcMeshTransport;

    fn js_err(e: impl std::fmt::Debug) -> NetError {
        NetError::Other(format!("{e:?}"))
    }

    pub(crate) struct Bound {
        pub dialer: Dialer,
        pub transport: SessionTransport,
        pub shutdown: Shutdown,
    }

    /// Bind the facade node, start the mesh transport, and serve catch-up.
    ///
    /// `local_only` has no meaning here — a browser has no sockets to advertise
    /// and the facade always uses n0's relays for signalling — so it is ignored.
    pub(crate) async fn bind(mirror: Arc<Mutex<Mirror>>, opts: &NetOptions) -> Result<Bound> {
        let secret = opts.secret.clone().unwrap_or_else(SecretKey::generate);
        let node = BrowserWebRtcNode::builder(BrowserWebRtcNodeConfig::default(), secret)
            // Both wires ride the facade: live actions and catch-up/assets.
            .accept_facade(crate::transport::MESH_ALPN)
            .accept_facade(proto::ALPN)
            .spawn()
            .await
            .map_err(js_err)?;

        let transport = WebRtcMeshTransport::new(node.clone())
            .await
            .map_err(|e| NetError::Other(e.to_string()))?;
        webrtc::serve_catchup(node.clone(), mirror);

        Ok(Bound {
            dialer: Dialer { node: node.clone() },
            transport,
            shutdown: Shutdown { node },
        })
    }

    /// Opens catch-up connections to peers.
    #[derive(Clone)]
    pub(crate) struct Dialer {
        node: BrowserWebRtcNode,
    }

    impl Dialer {
        pub fn local_id(&self) -> Result<EndpointId> {
            self.node
                .endpoint_id()
                .parse()
                .map_err(|e| NetError::Other(format!("facade returned an unusable id: {e}")))
        }

        /// The facade resolves peers by bare id through n0's address lookup, so
        /// only `addr.id` is used.
        pub async fn open(&self, addr: EndpointAddr) -> Result<Catchup> {
            let remote = addr.id.to_string();
            let conn = self
                .node
                .dial(&remote, proto::ALPN, BrowserDialOptions::webrtc_preferred())
                .await
                .map_err(js_err)?;
            Ok(Catchup { conn })
        }

        /// A browser advertises no addresses of its own; peers reach it by id.
        pub async fn ticket_addr(&self, _opts: &NetOptions) -> Result<EndpointAddr> {
            Ok(EndpointAddr::new(self.local_id()?))
        }
    }

    pub(crate) struct Catchup {
        conn: BrowserWebRtcConnection,
    }

    impl Catchup {
        pub async fn request(&self, req: Request) -> Result<Vec<u8>> {
            let stream = self.conn.open_bi().await.map_err(js_err)?;
            stream
                .send_all(&postcard::to_allocvec(&req)?)
                .await
                .map_err(js_err)?;
            // Half-close so the peer knows the request is complete.
            stream.close_send().await.map_err(js_err)?;
            stream.read_to_end().await.map_err(js_err)
        }

        pub async fn close(self) {
            let _ = self.conn.close("done").await;
        }
    }

    pub(crate) struct Shutdown {
        node: BrowserWebRtcNode,
    }

    impl Shutdown {
        pub async fn run(self) {
            if let Err(e) = self.node.close("session ended").await {
                tracing::warn!("facade shutdown: {e:?}");
            }
        }
    }
}

#[cfg(not(all(feature = "webrtc", target_family = "wasm", target_os = "unknown")))]
mod imp {
    //! Plain iroh: an endpoint and a router.

    #[cfg(not(target_arch = "wasm32"))]
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use iroh::endpoint::{presets, Connection};
    use iroh::protocol::Router;
    use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};

    use crate::mirror::Mirror;
    use crate::proto::{self, CollabProto, Request};
    use crate::session::NetOptions;
    use crate::transport::iroh::IrohMeshTransport;
    use crate::Result;

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
        let endpoint = if opts.local_only {
            Endpoint::builder(presets::Minimal).secret_key(secret).bind().await?
        } else {
            Endpoint::builder(presets::N0).secret_key(secret).bind().await?
        };
        let (transport, mesh_proto) = IrohMeshTransport::new(endpoint.clone());
        let router = Router::builder(endpoint.clone())
            .accept(crate::transport::MESH_ALPN, mesh_proto)
            .accept(proto::ALPN, CollabProto { mirror })
            .spawn();

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
            let conn = self.endpoint.connect(addr, proto::ALPN).await?;
            Ok(Catchup { conn })
        }

        /// With public infrastructure, wait (bounded) for the relay handshake so
        /// the ticket carries a relay URL; local-only tickets carry the bound
        /// sockets, loopback-normalized.
        pub async fn ticket_addr(&self, opts: &NetOptions) -> Result<EndpointAddr> {
            if !opts.local_only {
                // `online()` pends forever with no WAN; bound wait, then best effort.
                let _ =
                    n0_future::time::timeout(ONLINE_TIMEOUT, self.endpoint.online()).await;
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
pub(crate) use imp::{bind, Dialer, Shutdown};
