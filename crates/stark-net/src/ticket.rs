//! The shareable session ticket: everything a peer needs to join — how to
//! reach one member (an [`EndpointAddr`]) and the session's gossip topic.
//!
//! Displayed as `stark…` + base32 of the postcard encoding, so it survives
//! chat clients and clipboards.

use std::fmt;
use std::str::FromStr;

use iroh::EndpointAddr;
use iroh_gossip::TopicId;
use serde::{Deserialize, Serialize};

/// Human-pasteable prefix so tickets are recognizable in the wild.
const PREFIX: &str = "stark";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTicket {
    /// A reachable member of the session (initially the sharer).
    pub addr: EndpointAddr,
    /// The gossip topic all live actions ride on.
    pub topic: TopicId,
}

impl fmt::Display for SessionTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = postcard::to_allocvec(self).map_err(|_| fmt::Error)?;
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
        postcard::from_bytes(&bytes).map_err(|e| bad(&e.to_string()))
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
}
