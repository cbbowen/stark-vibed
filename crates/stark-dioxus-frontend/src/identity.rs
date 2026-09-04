//! This client's identity as the *network* wants it: the bytes
//! [`stark_ui::identity`] keeps, turned into the `SecretKey` an
//! [`ActorId`](stark_model::document::ActorId) derives from.
//!
//! Two lines and a type, because that is the whole of what does not travel. Which
//! bytes to keep, how to count runs of them, and what an absent record means are the
//! shared crate's — it is the record and the policy, and it says why both exist. What
//! is left here is that a secret key is `stark-net`'s type, and a crate two frontends
//! share should not name it: the native one has no collaboration yet, and putting
//! iroh under it for a 32-byte array would be the tail wagging the dog.
//!
//! The minting is here for the same reason. `SecretKey::generate` is where the CSPRNG
//! choice belongs, so the shared crate asks for a closure rather than choosing one
//! (`stark_ui::identity::get`).

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
