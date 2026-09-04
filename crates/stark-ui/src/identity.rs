//! This client's durable identity: the key its
//! [`ActorId`](stark_model::document::ActorId) derives from, and a
//! counter distinguishing runs of it.
//!
//! # Why it is persisted
//!
//! A fresh key per session made every reload a different author, and the `ActorId`
//! is not just a presence handle:
//!
//! - **undo targets your own actions**, so a reload orphaned your history on a
//!   document you were still looking at;
//! - **`DocState` keeps one selection per actor forever**, because replay needs it —
//!   so every session anyone ever opened left a dead entry in the log and the save
//!   file, growing without bound over a document's life.
//!
//! # Why that needs a run counter
//!
//! `PeerFrame::seq` restarts at zero with the process. With a fresh key that was
//! harmless — a new `ActorId` has no history to be stale against — but a durable one
//! reloading inside `PEER_TIMEOUT` would have every frame rejected as an overtaken
//! duplicate until it out-numbered its previous run. So each run also bumps a stored
//! counter, and peers order frames on `(boot, seq)`.
//!
//! # Privacy
//!
//! A persisted key is a stable pseudonymous identity that follows this client
//! across every document it opens — which is the point, and is also a thing someone
//! may not want. It is per-origin (`localStorage`), never leaves the machine except
//! as the public half every peer already sees, and clearing the browser's site data
//! discards it for a fresh one. Where storage is unavailable — private windows, storage disabled — this
//! degrades to the previous behaviour of a new identity per run, which costs the two
//! properties above and breaks nothing.

use crate::storage::{self, Record, Store};

/// What is kept between visits: the key itself, and how many runs have used it.
///
/// One record rather than the two keys this used to be, because they are one fact —
/// a boot counter without the key it counts runs of has nothing to say — and because
/// a half-written pair is a state nothing here could have made sense of.
///
/// No `#[serde(default)]` on either field: an absent or damaged record means a fresh
/// identity, and a fresh identity is the safe answer precisely because it is new —
/// nothing can be stale against it (see [`resolve`]). A defaulted secret would be a
/// key everyone shares.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Stored {
    #[serde(with = "storage::hex")]
    secret: [u8; 32],
    boot: u64,
}

impl Record for Stored {
    const STORE: Store = Store::Identity;
}

/// This client's identity for the life of the process, as **bytes**.
///
/// Not a `stark_net::SecretKey`, which is what every caller actually wants: that type
/// is the network crate's, and naming it here would put iroh under a crate whose other
/// consumer has no collaboration at all yet. The 32 bytes are the whole of what is
/// *stored*, and turning them into a key is one line at the frontend that already
/// depends on the crate that owns it.
#[derive(Clone, Copy)]
pub struct ClientIdentity {
    pub secret: [u8; 32],
    /// Which run of this identity the process is; see [`stark_engine::Identity`].
    pub boot: u64,
}

thread_local! {
    /// Resolved once per process: the run counter must count *runs*, not calls, so
    /// sharing and then joining must not look like two clients.
    static RESOLVED: std::cell::RefCell<Option<ClientIdentity>> =
        const { std::cell::RefCell::new(None) };
}

/// This client's identity, minting and storing one on first run.
///
/// `mint` is asked only when there is nothing stored, and it is the caller's because
/// a secret key's *randomness* belongs with the crate that defines the key — this one
/// would otherwise be choosing a CSPRNG on the network layer's behalf. It is called at
/// most once per process whatever happens, so a caller may reach for the expensive
/// generator without guarding it.
pub fn get(mint: impl FnOnce() -> [u8; 32]) -> ClientIdentity {
    RESOLVED.with(|slot| *slot.borrow_mut().get_or_insert_with(|| resolve(mint)))
}

fn resolve(mint: impl FnOnce() -> [u8; 32]) -> ClientIdentity {
    // A client with no storage reads as one that has stored nothing, so this mints a
    // fresh key per run — safe precisely because the id is new: nothing can be stale
    // against it. The write below then warns and carries on, and the whole arrangement
    // degrades to what it was before any of this was persisted ([`crate::storage`]).
    let stored = storage::load::<Stored>();
    let secret = stored.as_ref().map_or_else(mint, |s| s.secret);
    // Count this run. Wrapping is unreachable in practice and harmless if reached:
    // a peer that saw the old value drops us for `PEER_TIMEOUT` and re-adds us.
    let boot = stored.map_or(0, |s| s.boot).wrapping_add(1);
    storage::save(&Stored { secret, boot });
    ClientIdentity { secret, boot }
}
