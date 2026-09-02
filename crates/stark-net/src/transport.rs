//! Extra transports under the session's endpoint.
//!
//! The live wire itself is `iroh-gossip` on the plain endpoint; this module
//! holds what augments it. WebRTC on the single endpoint (`webrtc` feature):
//! not a wire of its own — just the bootstrap that gives the endpoint direct
//! WebRTC *paths*, which every protocol on it (gossip, catch-up) then rides.
//!
//! [`Direct`] is the whole surface the backend sees: one type, two cfg'd
//! definitions. With the feature it is [`direct`]'s handle around the custom
//! transport; without, the stand-in below, whose every method is the identity
//! or nothing — so the backend carries no feature awareness at all.

#[cfg(feature = "webrtc")]
mod direct;

#[cfg(feature = "webrtc")]
pub(crate) use direct::Direct;

#[cfg(not(feature = "webrtc"))]
pub(crate) use noop::Direct;

/// A module rather than cfg'd items so the imports live and die with the one
/// cfg — a `#[cfg]` cannot drift off a `use` it never guarded.
#[cfg(not(feature = "webrtc"))]
mod noop {
    use iroh::endpoint::Builder;
    use iroh::protocol::RouterBuilder;
    use iroh::{Endpoint, EndpointAddr, EndpointId};

    use crate::cancel::Cancel;
    use crate::neighbors::Neighbors;

    /// The `webrtc`-off stand-in: same surface, no transport.
    #[derive(Debug, Clone)]
    pub(crate) struct Direct;

    impl Direct {
        pub fn new(_local: EndpointId, _cancel: Cancel) -> Self {
            Self
        }

        pub fn install(&self, builder: Builder) -> Builder {
            builder
        }

        pub fn register(&self, router: RouterBuilder, _endpoint: &Endpoint) -> RouterBuilder {
            router
        }

        pub fn ensure_direct(&self, _endpoint: &Endpoint, _remote: EndpointId) {}

        pub fn dial_addr(&self, addr: EndpointAddr) -> EndpointAddr {
            addr
        }

        /// Returns immediately rather than sleeping out a sweep whose only
        /// step, `ensure_direct`, does nothing here.
        pub async fn maintain(&self, _endpoint: &Endpoint, _neighbors: Neighbors) {}
    }
}
