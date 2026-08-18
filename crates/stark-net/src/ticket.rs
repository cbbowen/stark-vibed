//! The shareable session ticket: everything a peer needs to join — how to
//! reach a few *members* (each an [`EndpointAddr`]) and the session's topic.
//!
//! Displayed as `stark…` + base32 of the encoded ticket, so it survives chat
//! clients and clipboards.
//!
//! One live member is enough: the joiner bootstraps gossip from it and the
//! swarm's membership exchange introduces everyone else, so any member can hand
//! out a ticket. Every member *past* the first is insurance — the joiner tries
//! them in order, so a link keeps working while anyone it names is still in the
//! session, including after the member that minted it has left.

use std::fmt;
use std::net::SocketAddr;
use std::str::FromStr;

use iroh::{EndpointAddr, EndpointId};
use iroh_base::{CustomAddr, TransportAddr};
use iroh_gossip::proto::TopicId;
use serde::{Deserialize, Serialize};

use crate::TicketError;

/// Human-pasteable prefix so tickets are recognizable in the wild.
const PREFIX: &str = "stark";

/// The encoding this build mints, and the only one it reads.
///
/// A ticket is the one thing here that keeps a version byte of its own, and the reason
/// is what a ticket *is*: a string someone pasted into a chat window, decoded with no
/// schema alongside it (§8 — the save format carries one, so it needs no number). One
/// byte buys a message naming the mismatch instead of a decode error about whichever
/// field happened to move (§19).
///
/// It rides *ahead* of the body — the first byte after the prefix — rather than inside
/// it, because inside it the byte cannot do its one job: a build decodes the body
/// against its own compile-time schema, so the moment the body's shape changes, a
/// version field inside that shape is unreadable by exactly the build that needed it.
/// Version 0 kept it as the body's first field and paid for that here, when the shape
/// first changed (one member became several) and the byte moved out.
const VERSION: u8 = 1;

/// What a link actually encodes — **Stark's own shape, in primitives**.
///
/// A mirror rather than the [`SessionTicket`] itself, and the reason is the same one
/// [`VERSION`] exists for. A link outlives builds with nothing beside it to reconcile
/// against, and the types a ticket is *made* of belong to iroh: an [`EndpointAddr`]
/// holds a set of [`TransportAddr`]s, one of which wraps a parsed URL. Pinning a
/// pasted link's bytes to another crate's internals is what the version byte was
/// apologizing for; spelled in primitives, the link format is ours to keep stable.
///
/// It is also the shape that *can* be described. An `EndpointAddr` is iroh's, so only a
/// runtime trace of its own `Deserialize` could describe it (§8) — and that trace fails:
/// it holds a `RelayUrl`, which parses its string and refuses the empty one. So this is
/// the one place in the crate where a mirror is not a preference. A `String` is what a
/// relay URL *is* on a link anyway.
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
pub(crate) struct TicketBody {
    /// The members a joiner may enter through, in the order to try them —
    /// whoever minted the link first.
    members: Vec<Member>,
    /// The gossip topic all live actions ride on.
    topic: [u8; 32],
}

/// One member a link names: an [`EndpointAddr`], in primitives.
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
struct Member {
    /// The member's endpoint id (its public key).
    endpoint: [u8; 32],
    /// Every way the link offers to reach this member, in iroh's order. May be
    /// empty: a bare id still resolves through address lookup on a WAN session.
    hops: Vec<Hop>,
}

/// One way to reach a member a link names: [`TransportAddr`], in primitives.
#[derive(Debug, Clone, Serialize, Deserialize, carbonite::Schema)]
enum Hop {
    /// A relay server, by URL. A string here, parsed on the way back in.
    Relay(String),
    /// A direct socket, for a same-machine or same-network session.
    ///
    /// `carbonite(serde)` because a `SocketAddr` is std's and carries no compile-time
    /// schema; its own `Deserialize` describes it (an enum of v4 and v6 over byte
    /// arrays, in a binary format), which is the shape a link already carried.
    Ip(#[carbonite(serde)] SocketAddr),
    /// A custom transport's address: its transport id and opaque data (the vendored
    /// WebRTC transport's, in a browser — see `transport`).
    Custom { transport: u64, data: Vec<u8> },
}

impl From<&SessionTicket> for TicketBody {
    fn from(ticket: &SessionTicket) -> Self {
        let members = ticket
            .members
            .iter()
            .map(|addr| Member {
                endpoint: *addr.id.as_bytes(),
                hops: addr
                    .addrs
                    .iter()
                    .filter_map(|addr| match addr {
                        TransportAddr::Relay(url) => Some(Hop::Relay(url.as_str().to_owned())),
                        TransportAddr::Ip(sock) => Some(Hop::Ip(*sock)),
                        TransportAddr::Custom(custom) => Some(Hop::Custom {
                            transport: custom.id(),
                            data: custom.data().to_vec(),
                        }),
                        // `TransportAddr` is `#[non_exhaustive]`: a kind this build cannot
                        // spell is dropped rather than refused, because a link with one fewer
                        // way to reach a peer still reaches it by the others.
                        _ => None,
                    })
                    .collect(),
            })
            .collect();
        TicketBody {
            members,
            topic: *ticket.topic.as_bytes(),
        }
    }
}

impl TryFrom<TicketBody> for SessionTicket {
    type Error = TicketError;

    fn try_from(body: TicketBody) -> Result<Self, Self::Error> {
        if body.members.is_empty() {
            return Err(TicketError::Empty);
        }
        let members = body
            .members
            .into_iter()
            .map(|member| {
                let id = EndpointId::from_bytes(&member.endpoint)
                    .map_err(|_| TicketError::NotAnEndpoint)?;
                let addrs = member.hops.into_iter().filter_map(|hop| match hop {
                    // A URL that will not parse is dropped, not refused, for the reason an
                    // unknown transport kind is: the other hops may still reach.
                    Hop::Relay(url) => match url.parse() {
                        Ok(url) => Some(TransportAddr::Relay(url)),
                        Err(_) => {
                            tracing::warn!(%url, "a link named a relay this build cannot parse");
                            None
                        }
                    },
                    Hop::Ip(sock) => Some(TransportAddr::Ip(sock)),
                    Hop::Custom { transport, data } => Some(TransportAddr::Custom(
                        CustomAddr::from_parts(transport, &data),
                    )),
                });
                Ok(EndpointAddr::from_parts(id, addrs))
            })
            .collect::<Result<Vec<_>, TicketError>>()?;
        Ok(SessionTicket {
            members,
            topic: TopicId::from_bytes(body.topic),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SessionTicket {
    /// Reachable members of the session, in the order a joiner should try them —
    /// whoever minted the link first, then members it could vouch were alive when
    /// it did.
    ///
    /// One live member is enough to join through, so every further name is
    /// insurance: the link keeps working while *any* member it names is still in
    /// the session, including after its minter has left.
    pub members: Vec<EndpointAddr>,
    /// The swarm all live actions ride on.
    pub topic: TopicId,
}

impl fmt::Display for SessionTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body = crate::codec::encode(&TicketBody::from(self)).map_err(|_| fmt::Error)?;
        let mut bytes = Vec::with_capacity(1 + body.len());
        bytes.push(VERSION);
        bytes.extend_from_slice(&body);
        let mut encoded = data_encoding::BASE32_NOPAD.encode(&bytes);
        encoded.make_ascii_lowercase();
        write!(f, "{PREFIX}{encoded}")
    }
}

impl FromStr for SessionTicket {
    type Err = TicketError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let encoded = s.trim().strip_prefix(PREFIX).ok_or(TicketError::NoPrefix)?;
        let bytes = data_encoding::BASE32_NOPAD.decode(encoded.to_ascii_uppercase().as_bytes())?;
        // The version byte is checked *before* the body is decoded — the body's
        // shape is exactly what another version disagrees about, so any question
        // asked of the bytes past this one would be asked in the wrong shape.
        let Some((&version, body)) = bytes.split_first() else {
            return Err(TicketError::Empty);
        };
        if version != VERSION {
            return Err(TicketError::Version {
                found: version,
                expected: VERSION,
            });
        }
        crate::codec::decode::<TicketBody>(body)?.try_into()
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
            members: vec![
                EndpointAddr::new(key.public()).with_ip_addr("127.0.0.1:4433".parse().unwrap()),
            ],
            topic: TopicId::from_bytes([9u8; 32]),
        };
        let s = ticket.to_string();
        assert!(s.starts_with("stark"));
        let back: SessionTicket = s.parse().expect("parse ticket");
        assert_eq!(back.members[0].id, ticket.members[0].id);
        assert_eq!(back.topic, ticket.topic);
        assert_eq!(
            back.members[0].ip_addrs().collect::<Vec<_>>(),
            ticket.members[0].ip_addrs().collect::<Vec<_>>()
        );
    }

    /// A link carries **every way to reach a peer**, and the shape it carries them in
    /// is Stark's rather than iroh's (see [`TicketBody`]).
    ///
    /// A mirror buys the link a stable format and costs it a hand-written conversion,
    /// which is a place a hop kind can be quietly dropped — and a dropped hop is not a
    /// broken link, it is one that connects over the relay when it should have gone
    /// direct, or fails to connect only for the peers that needed the kind that went
    /// missing. So all three are asserted, `Custom` included: it is what a browser
    /// session's WebRTC path advertises (`transport::direct`), and the least likely to
    /// be exercised by hand.
    #[test]
    fn a_link_carries_every_kind_of_hop() {
        let custom = CustomAddr::from_parts(7, b"a data channel");
        let ticket = SessionTicket {
            members: vec![EndpointAddr::from_parts(
                SecretKey::from_bytes(&[11u8; 32]).public(),
                [
                    TransportAddr::Relay("https://relay.example/".parse().expect("a relay url")),
                    TransportAddr::Ip("192.0.2.7:4433".parse().expect("a socket")),
                    TransportAddr::Custom(custom.clone()),
                ],
            )],
            topic: TopicId::from_bytes([5u8; 32]),
        };

        let back: SessionTicket = ticket.to_string().parse().expect("parse the link back");
        assert_eq!(back.members[0].id, ticket.members[0].id);
        assert_eq!(
            back.members[0].addrs, ticket.members[0].addrs,
            "every hop must survive the round trip, whatever kind it is",
        );
        // Spelled out for the one that travels as two loose primitives.
        let Some(TransportAddr::Custom(landed)) = back.members[0]
            .addrs
            .iter()
            .find(|a| matches!(a, TransportAddr::Custom(_)))
        else {
            panic!("the custom transport's address did not survive");
        };
        assert_eq!((landed.id(), landed.data()), (7, &b"a data channel"[..]));
    }

    /// A link names members in the order a joiner should try them, and that order
    /// is part of what it says: the minter is first because it is the member most
    /// recently known alive, and the members after it are the insurance a joiner
    /// falls back on. A round trip that reordered them would silently change which
    /// peer every joiner dials first.
    #[test]
    fn a_link_carries_several_members_in_order() {
        let member = |seed: u8, hops: Vec<TransportAddr>| {
            EndpointAddr::from_parts(SecretKey::from_bytes(&[seed; 32]).public(), hops)
        };
        let ticket = SessionTicket {
            members: vec![
                member(
                    1,
                    vec![TransportAddr::Relay(
                        "https://relay.example/".parse().expect("a relay url"),
                    )],
                ),
                member(
                    2,
                    vec![TransportAddr::Ip(
                        "192.0.2.7:4433".parse().expect("a socket"),
                    )],
                ),
                // A bare id — the WAN case, where address lookup resolves it.
                member(3, Vec::new()),
            ],
            topic: TopicId::from_bytes([5u8; 32]),
        };

        let back: SessionTicket = ticket.to_string().parse().expect("parse the link back");
        assert_eq!(back.members.len(), 3);
        for (landed, minted) in back.members.iter().zip(&ticket.members) {
            assert_eq!(landed.id, minted.id, "members must keep their order");
            assert_eq!(landed.addrs, minted.addrs);
        }
    }

    /// The version byte earns its place only if a mismatch is *named*. Decoding a
    /// future ticket as a current one otherwise yields a decode error about whatever
    /// field happened to move, which tells a user nothing — and a link is the one
    /// thing here that travels without a schema beside it to reconcile against.
    #[test]
    fn a_ticket_from_another_version_says_so() {
        let ticket = SessionTicket {
            members: vec![EndpointAddr::new(
                SecretKey::from_bytes(&[3u8; 32]).public(),
            )],
            topic: TopicId::from_bytes([1u8; 32]),
        };
        let body = crate::codec::encode(&TicketBody::from(&ticket)).expect("encode");
        // A future build's link: a version this one does not speak, ahead of a
        // body whose shape this build has no idea of — here, today's body, since
        // the check must not depend on reading past the byte.
        let mut bytes = vec![VERSION + 1];
        bytes.extend_from_slice(&body);
        let mut encoded = data_encoding::BASE32_NOPAD.encode(&bytes);
        encoded.make_ascii_lowercase();

        let err = format!("{PREFIX}{encoded}")
            .parse::<SessionTicket>()
            .expect_err("a version this build does not speak");
        assert!(
            matches!(err, TicketError::Version { found: 2, .. }),
            "{err}"
        );
        // And it says which, because the person reading it pasted a link.
        assert!(err.to_string().contains("version 2"), "{err}");
    }

    /// A link that names nobody is refused whole rather than joined nowhere:
    /// every use of a ticket begins with "dial a member", so the failure belongs
    /// to the parse, where the person who pasted it is still looking.
    #[test]
    fn a_link_naming_nobody_is_refused() {
        let body = crate::codec::encode(&TicketBody {
            members: Vec::new(),
            topic: [1u8; 32],
        })
        .expect("encode");
        let mut bytes = vec![VERSION];
        bytes.extend_from_slice(&body);
        let mut encoded = data_encoding::BASE32_NOPAD.encode(&bytes);
        encoded.make_ascii_lowercase();

        let err = format!("{PREFIX}{encoded}")
            .parse::<SessionTicket>()
            .expect_err("a link naming no member");
        assert!(matches!(err, TicketError::Empty), "{err}");
        // The degenerate spelling of the same nothing: a prefix with no bytes
        // after it at all, which must not reach the version check.
        let err = PREFIX.parse::<SessionTicket>().expect_err("an empty link");
        assert!(matches!(err, TicketError::Empty), "{err}");
    }

    #[test]
    fn a_ticket_without_the_prefix_is_rejected() {
        let err = "not-a-ticket"
            .parse::<SessionTicket>()
            .expect_err("no prefix");
        assert!(matches!(err, TicketError::NoPrefix), "{err}");
    }
}
