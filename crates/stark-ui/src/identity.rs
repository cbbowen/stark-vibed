//! This client's durable identity: the key its [`ActorId`] derives from, and a
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
//! A persisted key is a stable pseudonymous identity that follows this browser
//! across every document it opens — which is the point, and is also a thing someone
//! may not want. It is per-origin (`localStorage`), never leaves the machine except
//! as the public half every peer already sees, and [`reset`] discards it for a fresh
//! one. Where storage is unavailable — private windows, storage disabled — this
//! degrades to the previous behaviour of a new identity per run, which costs the two
//! properties above and breaks nothing.

use crate::storage;
use stark_net::SecretKey;

/// Storage keys. Namespaced because `localStorage` is shared per origin.
const KEY_SECRET: &str = "stark.identity.secret";
const KEY_BOOT: &str = "stark.identity.boot";

/// This client's identity for the life of the process.
#[derive(Clone)]
pub struct ClientIdentity {
    pub secret: SecretKey,
    /// Which run of this identity the process is; see [`stark_core::peer::Identity`].
    pub boot: u64,
}

thread_local! {
    /// Resolved once per process: the run counter must count *runs*, not calls, so
    /// sharing and then joining must not look like two clients.
    static RESOLVED: std::cell::RefCell<Option<ClientIdentity>> =
        const { std::cell::RefCell::new(None) };
}

/// This client's identity, minting and storing one on first run.
pub fn get() -> ClientIdentity {
    RESOLVED.with(|slot| slot.borrow_mut().get_or_insert_with(resolve).clone())
}

fn resolve() -> ClientIdentity {
    // A browser with no storage reads as one that has stored nothing, so this
    // mints a fresh key per run — safe precisely because the id is new: nothing
    // can be stale against it. The writes below then warn and carry on, and the
    // whole arrangement degrades to what it was before any of this was persisted
    // (`crate::storage`).
    let secret = storage::get(KEY_SECRET)
        .as_deref()
        .and_then(decode_secret)
        .unwrap_or_else(|| {
            let fresh = SecretKey::generate();
            storage::set(
                KEY_SECRET,
                "this browser's identity",
                &encode_secret(&fresh),
            );
            fresh
        });
    // Count this run. Wrapping is unreachable in practice and harmless if reached:
    // a peer that saw the old value drops us for `PEER_TIMEOUT` and re-adds us.
    let boot = storage::get(KEY_BOOT)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
        .wrapping_add(1);
    storage::set(KEY_BOOT, "this browser's identity", &boot.to_string());
    ClientIdentity { secret, boot }
}

fn encode_secret(secret: &SecretKey) -> String {
    secret
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn decode_secret(hex: &str) -> Option<SecretKey> {
    if hex.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(SecretKey::from_bytes(&bytes))
}
