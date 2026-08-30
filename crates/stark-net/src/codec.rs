//! The wire's encoding (§8, §12.4).
//!
//! The save format and the wire speak the same encoding, which is the whole reason this
//! module is a handful of functions and not a format. What differs is **where the schema
//! lives**, and that difference is the interesting part:
//!
//! - A **file** carries its own schema, so a document written today opens in a build
//!   that has never heard of it (`stark_model::DocumentFile`, §8). It has to: a file is
//!   a message to the future, and the future cannot be asked to agree.
//! - A **message** does not. Both ends encode against their own copy, and the
//!   [ALPN](crate::proto) is what makes a disagreement fail to *meet* rather than decode
//!   wrong. A live session is a meeting of builds, and a meeting may require agreement —
//!   which buys back the schema, and that is not a small thing here: a blob's header
//!   spends a varint per column, an `Action`'s schema is some four hundred columns wide,
//!   and carrying it would take a presence frame from a couple of hundred bytes to four
//!   kilobytes. Gossip floods, and a flood pays for its headers once per hop.
//!
//! Every schema here is built at **compile time**, from `#[derive(carbonite::Schema)]`
//! on the types themselves. Nothing is discovered at runtime, so there is nothing to
//! cache and nothing to fail on the first message of a session — and that is not an
//! optimization but a requirement: a schema can also be found by *tracing* a type's
//! `Deserialize` impl, and an `Action` cannot be traced, because the invariant funnels
//! inside it (`FillOp`, `SelectionOp`, `Gradient`) refuse the synthetic values tracing
//! feeds them. Every payload on this wire holds an `Action`.

use std::sync::LazyLock;

use carbonite::{Schema, Serializer, StaticSchema};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::proto::{Stamped, StampedRef};

/// Encode one message against its own compile-time schema.
///
/// Carbonite's error rather than this crate's: both [`NetError`](crate::NetError) and
/// [`TicketError`](crate::TicketError) fold it in, and which of the two a caller wants
/// is the caller's question.
pub(crate) fn encode<T>(value: &T) -> carbonite::Result<Vec<u8>>
where
    T: Serialize + carbonite::SerializeColumns,
{
    carbonite::Serializer::new(T::schema()).to_vec_columns(value)
}

/// Decode one message, reconciling against the sender's schema if it differs.
///
/// It should not differ — the ALPN is what keeps two builds that disagree from meeting —
/// so this is the belt to that braces: a field the far end does not send yet arrives
/// from its `#[serde(default)]` rather than as a dropped session.
pub(crate) fn decode<T>(bytes: &[u8]) -> carbonite::Result<T>
where
    T: DeserializeOwned + StaticSchema,
{
    carbonite::Deserializer::new_static(T::schema()).from_slice(bytes)
}

/// [`Stamped`]'s schema, retyped once for the spelling that borrows what it sends.
static STAMPED_REF: LazyLock<Schema<StampedRef<'static>>> =
    LazyLock::new(|| Stamped::schema().cast());

/// Encode a broadcast **without copying what it references** (see [`StampedRef`]).
///
/// One schema serves both spellings, and that it may is *checked* rather than hoped for:
/// carbonite serializes against a schema by name, and `StampedRef` presents itself as a
/// `Stamped` (`#[serde(rename)]`), so a field or variant that moved in one and not the
/// other fails to encode here instead of shipping bytes the far end reads as something
/// else.
///
/// The `'static` schema serves a shorter-lived value because [`Schema`](struct@Schema) is covariant in
/// its parameter — it holds the schema tree and a `PhantomData<fn() -> T>` — so the
/// reference retypes without touching the tree. The `LazyLock` is what keeps the retype
/// from rebuilding the tree per broadcast; the tree itself is a `const`-shaped thing the
/// derive assembles, not a discovery.
pub(crate) fn encode_stamped_ref(stamped: &StampedRef<'_>) -> carbonite::Result<Vec<u8>> {
    let schema: &Schema<StampedRef<'_>> = &STAMPED_REF;
    Serializer::new(schema).to_vec(stamped)
}

#[cfg(test)]
mod tests {
    use stark_model::document::ActionId;

    use super::*;
    use crate::proto::{Recovered, Request};

    /// The borrowed spelling of a broadcast describes the same shape as the owned one.
    ///
    /// Both come from the same tree — one is a retype of the other — so this is really
    /// a check that the retype is of the thing it claims. It is cheap, and the failure
    /// it guards against is bytes that only the sender can read.
    #[test]
    fn the_borrowed_spelling_shares_the_owned_schema() {
        assert_eq!(STAMPED_REF.to_bytes(), Stamped::schema().to_bytes());
    }

    /// Every message type on this wire has a schema, and it is built rather than
    /// discovered — so an `Action` whose funnels refuse a trace is no obstacle (§8).
    ///
    /// Forcing each one is the assertion: a type that could not describe itself would
    /// not have compiled, which is the whole difference from where this started.
    #[test]
    fn every_wire_type_describes_itself() {
        for (what, bytes) in [
            ("a gossip payload", Stamped::schema().to_bytes()),
            ("a catch-up request", Request::schema().to_bytes()),
            ("an action-id digest", Vec::<ActionId>::schema().to_bytes()),
            (
                "a recovered action batch",
                Vec::<Recovered>::schema().to_bytes(),
            ),
            (
                "a session ticket",
                crate::ticket::TicketBody::schema().to_bytes(),
            ),
        ] {
            assert!(!bytes.is_empty(), "{what} has no schema");
        }
    }
}
