//! This client's identity as the *network* wants it: the bytes
//! [`stark_ui::identity`] keeps, turned into the `SecretKey` an
//! [`ActorId`](stark_model::document::ActorId) derives from.
//!
//! The same two lines the web frontend has, and the duplication is the point rather
//! than an oversight — what is shared is the record, the policy and the reason both
//! exist, and what is left here is that a secret key is `stark-net`'s type and the
//! crate under both frontends does not name it (§17).
//!
//! What differs is where the bytes are kept: the store backend is `crate::store`'s
//! (a file beside the app's other records) where the web app's is `localStorage`.
//! Nothing here says so, which is the seam working.

use stark_net::SecretKey;

/// This client's identity for the life of the process.
#[derive(Clone)]
pub struct ClientIdentity {
    pub secret: SecretKey,
    /// Which run of this identity the process is; see [`stark_engine::Identity`].
    pub boot: u64,
}

/// This client's identity, minting and storing one on first run.
pub fn get() -> ClientIdentity {
    let stored = stark_ui::identity::get(|| SecretKey::generate().to_bytes());
    ClientIdentity {
        secret: SecretKey::from_bytes(&stored.secret),
        boot: stored.boot,
    }
}
