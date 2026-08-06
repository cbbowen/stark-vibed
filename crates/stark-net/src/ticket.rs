//! The shareable session ticket: everything a peer needs to join — how to
//! reach one member (an [`EndpointAddr`]) and the session's topic.
//!
//! Displayed as `stark…` + base32 of the postcard encoding, so it survives
//! chat clients and clipboards.
//!
//! One member is enough: the joiner bootstraps gossip from it and the swarm's
//! membership exchange introduces everyone else, so any member can hand out a
//! ticket.

use std::fmt;
use std::str::FromStr;

use iroh::EndpointAddr;
use iroh_gossip::proto::TopicId;
use serde::{Deserialize, Serialize};

/// Human-pasteable prefix so tickets are recognizable in the wild.
const PREFIX: &str = "stark";

/// The encoding this build mints, and the only one it reads.
///
/// The wire protocols carry their version in the ALPN (`stark/collab/0`); the
/// ticket had nowhere to carry one, so the first change to [`EndpointAddr`] or
/// [`TopicId`]'s postcard shape would have surfaced as an unexplained decode
/// failure on a link someone pasted (§19). One byte buys the message instead.
const VERSION: u8 = 0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTicket {
    /// A reachable member of the session (initially the sharer).
    pub addr: EndpointAddr,
    /// The swarm all live actions ride on.
    pub topic: TopicId,
}

impl fmt::Display for SessionTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = postcard::to_allocvec(&(VERSION, self)).map_err(|_| fmt::Error)?;
        let mut encoded = data_encoding::BASE32_NOPAD.encode(&bytes);
        encoded.make_ascii_lowercase();
        write!(f, "{PREFIX}{encoded}")
    }
}

impl FromStr for SessionTicket {
    type Err = crate::NetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = |m: &str| crate::NetError::Ticket(m.to_string());
        let s = s.trim();
        let encoded = s
            .strip_prefix(PREFIX)
            .ok_or_else(|| bad("missing 'stark' prefix"))?;
        let bytes = data_encoding::BASE32_NOPAD
            .decode(encoded.to_ascii_uppercase().as_bytes())
            .map_err(|e| bad(&e.to_string()))?;
        let (version, ticket): (u8, Self) =
            postcard::from_bytes(&bytes).map_err(|e| bad(&e.to_string()))?;
        if version != VERSION {
            // Named rather than guessed at: past the version byte the fields are
            // a different shape, so anything else this could say about them would
            // be about the wrong shape.
            return Err(bad(&format!(
                "ticket is version {version}; this build speaks {VERSION} — \
                 both ends need the same version of Stark"
            )));
        }
        Ok(ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::{EndpointAddr, SecretKey};

    #[test]
    fn ticket_roundtrips_through_display() {
        let key = SecretKey::from_bytes(&[7u8; 32]);
        let ticket = SessionTicket {
            addr: EndpointAddr::new(key.public()).with_ip_addr("127.0.0.1:4433".parse().unwrap()),
            topic: TopicId::from_bytes([9u8; 32]),
        };
        let s = ticket.to_string();
        assert!(s.starts_with("stark"));
        let back: SessionTicket = s.parse().expect("parse ticket");
        assert_eq!(back.addr.id, ticket.addr.id);
        assert_eq!(back.topic, ticket.topic);
        assert_eq!(
            back.addr.ip_addrs().collect::<Vec<_>>(),
            ticket.addr.ip_addrs().collect::<Vec<_>>()
        );
    }

    /// The version byte earns its place only if a mismatch is *named*. Decoding
    /// a future ticket as a current one otherwise yields a postcard error about
    /// whatever field happened to move, which tells a user nothing.
    #[test]
    fn a_ticket_from_another_version_says_so() {
        let bytes = postcard::to_allocvec(&(
            VERSION + 1,
            SessionTicket {
                addr: EndpointAddr::new(SecretKey::from_bytes(&[3u8; 32]).public()),
                topic: TopicId::from_bytes([1u8; 32]),
            },
        ))
        .expect("encode");
        let mut encoded = data_encoding::BASE32_NOPAD.encode(&bytes);
        encoded.make_ascii_lowercase();

        let err = format!("{PREFIX}{encoded}")
            .parse::<SessionTicket>()
            .expect_err("a version this build does not speak");
        assert!(err.to_string().contains("version 1"), "{err}");
    }

    #[test]
    fn a_ticket_without_the_prefix_is_rejected() {
        let err = "not-a-ticket"
            .parse::<SessionTicket>()
            .expect_err("no prefix");
        assert!(err.to_string().contains("prefix"), "{err}");
    }
}
