//! The shareable session ticket: everything a peer needs to join — how to
//! reach a few *members* (each an [`EndpointAddr`]) and the session's topic.
//!
//! Displayed as `stark…` + base64url of the deflated ticket, so it survives chat
//! clients, clipboards and URL bars — and so it stays as short as a thing a person
//! pastes ought to be (see [`wrap`], which holds both halves of that argument).
//!
//! One live member is enough: the joiner bootstraps gossip from it and the
//! swarm's membership exchange introduces everyone else, so any member can hand
//! out a ticket. Every member *past* the first is insurance — the joiner tries
//! them in order, so a link keeps working while anyone it names is still in the
//! session, including after the member that minted it has left.

use std::fmt;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::str::FromStr;

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
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
/// first changed (one member became several) and the byte moved out. Version 2 is the
/// body gaining the wire protocol it is *for* ([`TicketBody::proto`]) — a shape change
/// like any other, even though what the new field carries is a guard of its own.
const VERSION: u8 = 2;

/// The ceiling on what a pasted link may inflate to.
///
/// Deflate's ratio runs to about a thousand to one, so a string short enough to paste
/// can name as many megabytes as it likes. A file gets an unbounded door beside the
/// bounded one, because a painting off the artist's own disk is as large as they made it
/// (§8); a link gets only the bounded one, because there is no such thing as a ticket its
/// reader wrote themselves — every link arrived from somewhere else. Generous rather than
/// tight: the honest article is a few hundred bytes, so this is a bound on the absurd and
/// not a budget anyone can spend.
const MAX_BODY: u64 = 64 * 1024;

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
    /// The wire protocol ([`crate::wire::PROTO`]) the minting build speaks — what
    /// the link is *for*, where [`VERSION`] guards only what the link *is*. A
    /// mismatch here would otherwise surface as an ALPN transport error at
    /// `connect`, nowhere near the person who pasted the link; checked at the
    /// parse instead, where they are still looking.
    proto: u32,
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
            proto: crate::wire::PROTO,
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

/// Wrap an encoded body as the string a person pastes:
/// `PREFIX | base64url(version | deflate(body))`.
///
/// Both layers here are about **length**, which for a ticket is not a detail: a link is
/// pasted into a chat window and carried in a URL bar, and one that wraps across four
/// lines is one that arrives cut in half.
///
/// **Deflated because carbonite is columnar** (§8). The property that makes an action
/// log small does the same for a ticket, and for the same reason — like bytes end up
/// beside each other, so the several members a link names put their key material, their
/// ports and their relay URLs each in one run, and the columnar framing's own
/// bookkeeping (a length per column, most of them tiny and identical) is among the first
/// things to go. A representative link loses a third of its bytes.
///
/// **`base64url` because a link *is* a URL half the time** — the page fragment a shared
/// session rides in (`stark-ui`'s `collab`). Base32 spends 8 characters per 5 bytes
/// where base64 spends 4 per 3, so the alphabet alone is a fifth off the length; the
/// `url` in the name is what makes that free, `-` and `_` for the two extra digits and
/// no padding, so a link needs no percent-encoding anywhere it is put. What it costs is
/// case-sensitivity, which is the one thing base32 was buying: a link is now something
/// to copy rather than to retype.
///
/// **`best` rather than `default`**, the opposite of the save container's choice (§8),
/// because it is not the same trade at all — a body of a few hundred bytes makes the
/// extra hunting microseconds nobody waits through, and what it buys is characters a
/// person has to carry.
///
/// Nothing here branches on whether the compression *won*, and it need not: the shortest
/// link there is — one member, no hops, so two thirds of it is key material and topic —
/// still comes back at 69 bytes from 163, because what compresses is the framing rather
/// than the payload. A body that beat deflate outright would cost the handful of bytes
/// of a stored block, which is not worth a second thing the version byte would have to
/// say.
fn wrap(version: u8, body: &[u8]) -> std::io::Result<String> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(body)?;
    let deflated = encoder.finish()?;

    let mut bytes = Vec::with_capacity(1 + deflated.len());
    bytes.push(version);
    bytes.extend_from_slice(&deflated);
    Ok(format!(
        "{PREFIX}{}",
        data_encoding::BASE64URL_NOPAD.encode(&bytes)
    ))
}

/// The body [`wrap`] deflated, back as it went in.
///
/// Bounded *before* it expands rather than after (§8): the compressed length says
/// nothing about the decompressed one, so a reader that finds out by inflating has
/// already spent the memory. `take` one byte past the ceiling, so "filled the buffer"
/// and "reached the ceiling" stay distinguishable.
fn inflate(deflated: &[u8]) -> Result<Vec<u8>, TicketError> {
    let mut body = Vec::new();
    DeflateDecoder::new(deflated)
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)?;
    if body.len() as u64 > MAX_BODY {
        return Err(TicketError::TooLarge { limit: MAX_BODY });
    }
    Ok(body)
}

impl fmt::Display for SessionTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body = crate::codec::encode(&TicketBody::from(self)).map_err(|_| fmt::Error)?;
        f.write_str(&wrap(VERSION, &body).map_err(|_| fmt::Error)?)
    }
}

impl FromStr for SessionTicket {
    type Err = TicketError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let encoded = s.trim().strip_prefix(PREFIX).ok_or(TicketError::NoPrefix)?;
        let bytes = data_encoding::BASE64URL_NOPAD.decode(encoded.as_bytes())?;
        // The version byte is checked *before* the body is inflated, let alone decoded —
        // how the body is compressed and what shape it is in are exactly what another
        // version disagrees about, so any question asked of the bytes past this one
        // would be asked of the wrong thing.
        let Some((&version, deflated)) = bytes.split_first() else {
            return Err(TicketError::Empty);
        };
        if version != VERSION {
            return Err(TicketError::Version {
                found: version,
                expected: VERSION,
            });
        }
        let body = crate::codec::decode::<TicketBody>(&inflate(deflated)?)?;
        // After the decode — the shape is this build's, only the number inside
        // disagrees — and before the conversion, so the mismatch is named
        // rather than surfacing later as a transport error at `connect`.
        if body.proto != crate::wire::PROTO {
            return Err(TicketError::Protocol {
                found: body.proto,
                expected: crate::wire::PROTO,
            });
        }
        body.try_into()
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
                    TransportAddr::Custom(custom),
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
        // A future build's link: a version this one does not speak, ahead of bytes it
        // has no way to read at all — not a body of some other shape but one that is not
        // a deflate stream either, since the check must not depend on inflating what
        // follows it any more than on decoding it.
        let mut bytes = vec![VERSION + 1];
        bytes.extend_from_slice(b"a compression this build has never heard of");
        let encoded = data_encoding::BASE64URL_NOPAD.encode(&bytes);

        let err = format!("{PREFIX}{encoded}")
            .parse::<SessionTicket>()
            .expect_err("a version this build does not speak");
        assert!(
            matches!(err, TicketError::Version { found, .. } if found == VERSION + 1),
            "{err}"
        );
        // And it says which, because the person reading it pasted a link.
        assert!(
            err.to_string()
                .contains(&format!("version {}", VERSION + 1)),
            "{err}"
        );
    }

    /// The by-far-most-common mismatch is not the ticket's own shape but the
    /// wire behind it: the version byte holds still while the ALPN moves, so a
    /// foreign build's link parses perfectly and the join dies at `connect` as
    /// a transport error. The protocol number in the body is what turns that
    /// into an answer at the parse, where the person who pasted is looking.
    #[test]
    fn a_ticket_for_another_protocol_says_so() {
        let mut body = TicketBody::from(&a_realistic_ticket());
        body.proto += 1;
        let foreign = body.proto;
        let link = wrap(
            VERSION,
            &crate::codec::encode(&body).expect("encode the body"),
        )
        .expect("wrap");

        let err = link
            .parse::<SessionTicket>()
            .expect_err("a protocol this build does not speak");
        assert!(
            matches!(err, TicketError::Protocol { found, expected }
                if found == foreign && expected == crate::wire::PROTO),
            "{err}"
        );
        // And it says which, in the same voice as a version mismatch.
        assert!(
            err.to_string().contains(&format!("protocol {foreign}")),
            "{err}"
        );
        assert!(
            err.to_string().contains("the same version of Stark"),
            "{err}"
        );
    }

    /// A link that names nobody is refused whole rather than joined nowhere:
    /// every use of a ticket begins with "dial a member", so the failure belongs
    /// to the parse, where the person who pasted it is still looking.
    #[test]
    fn a_link_naming_nobody_is_refused() {
        let body = crate::codec::encode(&TicketBody {
            proto: crate::wire::PROTO,
            members: Vec::new(),
            topic: [1u8; 32],
        })
        .expect("encode");

        let err = wrap(VERSION, &body)
            .expect("wrap")
            .parse::<SessionTicket>()
            .expect_err("a link naming no member");
        assert!(matches!(err, TicketError::Empty), "{err}");
        // The degenerate spelling of the same nothing: a prefix with no bytes
        // after it at all, which must not reach the version check.
        let err = PREFIX.parse::<SessionTicket>().expect_err("an empty link");
        assert!(matches!(err, TicketError::Empty), "{err}");
    }

    /// A link shaped like the ones a session actually mints (`Broadcaster::ticket`):
    /// the minter with everything known about how to reach it, then the handful of
    /// neighbors it vouches for, each by the one path its traffic rides right now.
    fn a_realistic_ticket() -> SessionTicket {
        let relay = || {
            TransportAddr::Relay(
                "https://usw1-1.relay.n0.iroh.link./"
                    .parse()
                    .expect("a relay url"),
            )
        };
        let ip = |i: u8| TransportAddr::Ip(format!("192.0.2.{i}:4433").parse().expect("a socket"));
        let key = |i: u8| SecretKey::from_bytes(&[i; 32]).public();
        SessionTicket {
            members: vec![
                EndpointAddr::from_parts(key(1), [relay(), ip(1)]),
                EndpointAddr::from_parts(key(2), [relay()]),
                EndpointAddr::from_parts(key(3), [ip(3)]),
                EndpointAddr::from_parts(key(4), [relay()]),
            ],
            topic: TopicId::from_bytes([0x5a; 32]),
        }
    }

    /// **A link's length is part of whether it works**, and both halves of how one is
    /// spelled are there to hold it down (see [`wrap`]): the body is deflated, and the
    /// bytes are base64url rather than base32.
    ///
    /// Measured on the ticket above — four members, relay and direct hops, the shape a
    /// real session hands out — the plain spelling costs 629 characters and this one
    /// costs 352. The threshold is a three fifths that **neither half clears alone**:
    /// the wider alphabet by itself lands at 84% of plain and the deflate by itself at
    /// 67%. So a link that quietly stops being compressed, or quietly goes back to
    /// base32, fails here rather than merely getting long out in the world, where
    /// nothing is asserting anything about it.
    #[test]
    fn a_link_costs_half_of_the_plain_spelling() {
        let ticket = a_realistic_ticket();
        let link = ticket.to_string();
        let body = crate::codec::encode(&TicketBody::from(&ticket)).expect("encode");

        // The plain spelling of the same ticket: the body uncompressed, in base32.
        let mut plain_bytes = vec![VERSION];
        plain_bytes.extend_from_slice(&body);
        let plain = PREFIX.len() + data_encoding::BASE32_NOPAD.encode(&plain_bytes).len();
        assert!(
            link.len() * 5 < plain * 3,
            "a link costs {} characters where the plain spelling costs {plain}",
            link.len(),
        );
        // And it is still a link: nothing above is worth a character if the round trip
        // does not survive it.
        let back: SessionTicket = link.parse().expect("parse the link back");
        assert_eq!(back.members.len(), ticket.members.len());
        for (landed, minted) in back.members.iter().zip(&ticket.members) {
            assert_eq!((landed.id, &landed.addrs), (minted.id, &minted.addrs));
        }
    }

    /// A pasted link is a stranger's, and deflate's ratio means a short string can name
    /// as much memory as it likes. Refused by what it *would* expand to, before it does.
    ///
    /// The bomb is minted by the same [`wrap`] a real link is, which is the cheapest way
    /// to be sure this is the door a real link comes through rather than a hand-built
    /// near-miss beside it.
    #[test]
    fn a_link_that_expands_without_bound_is_refused() {
        let link = wrap(VERSION, &vec![0u8; 8 * 1024 * 1024]).expect("deflate");
        assert!(
            link.len() < 16 * 1024,
            "the bomb has to be pasteable to be a bomb: {} characters",
            link.len(),
        );

        let err = link
            .parse::<SessionTicket>()
            .expect_err("a link naming megabytes");
        assert!(matches!(err, TicketError::TooLarge { .. }), "{err}");
    }

    /// The characters can all be legal and still spell nothing — base64url has no
    /// checksum, and a link truncated on a chat client's line wrap decodes fine.
    /// Deflate is what notices, and it must say so as a damaged link rather than
    /// escaping as an I/O error from a parse that touches no I/O.
    #[test]
    fn a_link_that_is_not_a_deflate_stream_says_it_is_damaged() {
        let mut bytes = vec![VERSION];
        bytes.extend_from_slice(b"legal characters, no stream");
        let err = format!("{PREFIX}{}", data_encoding::BASE64URL_NOPAD.encode(&bytes))
            .parse::<SessionTicket>()
            .expect_err("not a deflate stream");
        assert!(matches!(err, TicketError::Compressed(_)), "{err}");
        assert!(err.to_string().contains("damaged"), "{err}");
    }

    #[test]
    fn a_ticket_without_the_prefix_is_rejected() {
        let err = "not-a-ticket"
            .parse::<SessionTicket>()
            .expect_err("no prefix");
        assert!(matches!(err, TicketError::NoPrefix), "{err}");
    }
}
