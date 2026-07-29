//! Spike: does the anchalshivank-style integration — ONE ordinary iroh endpoint
//! with a WebRTC `CustomTransport` — give us direct connections with per-peer
//! relay fallback?
//!
//! This re-litigates the 2026-07-24 spike in
//! `vendor/iroh-webrtc-transport/tests/single_endpoint_webrtc.rs` (NEGATIVE:
//! iroh never attempted the custom path once relay was known) against iroh
//! 1.0.3 and this crate's much simpler bridge.
//!
//! ## Findings (2026-07-29, iroh 1.0.3)
//!
//! 1. **The bridge works.** Full iroh QUIC connections — handshake and app
//!    data — ride the str0m WebRTC data channel ([`webrtc_carries_the_connection_when_the_relay_is_down`]).
//!    The old spike's "iroh never calls `is_valid_send_addr`" dead end is gone.
//! 2. **Relay fallback works per connection** ([`unattached_webrtc_falls_back_to_relay`]).
//! 3. **But path choice is a dial-time race, not a preference.** A new
//!    connection's Initial is sent to every known addr **only when no
//!    `selected_path` is cached for the peer** (so the previous connection to
//!    the peer must be closed first — see `handle_msg_send_datagram` in iroh's
//!    `remote_state.rs`), and whichever transport completes the handshake
//!    becomes PathId 0 — **forever**. iroh 1.0.3 never opens a custom path
//!    after the handshake: holepunching opens IP paths only, the post-hoc
//!    re-add logic exists only for relay paths, and `PathSelector` (public,
//!    unstable) only picks among already-open paths, so it cannot pull traffic
//!    onto WebRTC ([`with_relay_up_the_dial_race_decides_the_path`] — on a
//!    loopback relay, relay always wins the race).
//!
//! Consequence: "prefer WebRTC when it works, relay otherwise" needs a small
//! iroh patch (feed known-but-unopened custom addrs into the existing
//! `pending_open_paths` / `open_path_on_all_conns` machinery, which does
//! exactly this for CID-exhaustion retries; once a custom path opens, even the
//! DEFAULT selector prefers it — custom is Primary, relay is Backup), or
//! upstream support for the same.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId, RelayMap, RelayMode, TransportAddr, Watcher,
    address_lookup::MemoryLookup,
    endpoint::{
        Connection, presets,
        transports::{Addr, PathSelection, PathSelectionContext, PathSelector},
    },
    protocol::{AcceptError, ProtocolHandler, Router},
    test_utils::run_relay_server,
    tls::CaTlsConfig,
};
use iroh_webrtc_transport::{
    AttachOptions, QuicSignaling, WEBRTC_TRANSPORT_ID, WebRtcTransport,
    custom_addr_from_opaque_data, negotiate_dc_as_answerer, negotiate_dc_as_offerer,
};
use tokio::time::timeout;

const SIGNAL_ALPN: &[u8] = b"stark-spike/jsep/0";
const ECHO_ALPN: &[u8] = b"stark-spike/echo/0";
const STEP: Duration = Duration::from_secs(15);

const SERVER_OPAQUE: [u8; 16] = [1u8; 16];
const CLIENT_OPAQUE: [u8; 16] = [2u8; 16];

/// Prefer a WebRTC custom path once it has stats (i.e. it is open and measured);
/// otherwise lowest RTT — iroh's own `examples/custom-transport.rs` pattern,
/// made fallback-safe. NOTE (finding 3): this can only choose among paths that
/// are already open on the connection; it cannot cause a WebRTC path to open.
#[derive(Debug)]
struct PreferWebRtc;

impl PathSelector for PreferWebRtc {
    fn select(&self, ctx: &PathSelectionContext<'_>) -> PathSelection {
        let mut selection = PathSelection::none();
        if let Some(p) = ctx.paths().find(|p| {
            matches!(p.network_path().remote(), Addr::Custom(c) if c.id() == WEBRTC_TRANSPORT_ID)
                && p.stats().is_some()
        }) {
            selection.set(&p);
            return selection;
        }
        if let Some(p) = ctx
            .paths()
            .filter_map(|p| p.stats().map(|s| (p, s.rtt)))
            .min_by_key(|(_, rtt)| *rtt)
            .map(|(p, _)| p)
        {
            selection.set(&p);
        }
        selection
    }
}

#[derive(Debug, Clone)]
struct Echo;

impl ProtocolHandler for Echo {
    async fn accept(&self, conn: Connection) -> std::result::Result<(), AcceptError> {
        loop {
            let Ok((mut send, mut recv)) = conn.accept_bi().await else {
                return Ok(());
            };
            let msg = recv.read_to_end(1024).await.map_err(AcceptError::from_err)?;
            send.write_all(&msg).await.map_err(AcceptError::from_err)?;
            send.finish().map_err(AcceptError::from_err)?;
        }
    }
}

/// Answerer half of the JSEP bootstrap: accept the signaling stream, negotiate
/// the str0m data channel, attach it to this peer's transport.
#[derive(Clone)]
struct JsepAnswerer {
    transport: Arc<WebRtcTransport>,
    remote_opaque: [u8; 16],
}

impl std::fmt::Debug for JsepAnswerer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsepAnswerer").finish_non_exhaustive()
    }
}

impl ProtocolHandler for JsepAnswerer {
    async fn accept(&self, conn: Connection) -> std::result::Result<(), AcceptError> {
        let (send, recv) = conn.accept_bi().await?;
        let mut sig = QuicSignaling::new(send, recv);
        let peer = negotiate_dc_as_answerer(&mut sig)
            .await
            .map_err(|e| AcceptError::from_boxed(e.into()))?;
        self.transport
            .attach_data_channel(
                peer,
                custom_addr_from_opaque_data(&self.remote_opaque),
                AttachOptions::default(),
            )
            .map_err(|e| AcceptError::from_boxed(e.into()))?;
        conn.closed().await;
        Ok(())
    }
}

/// One peer = one ordinary endpoint: relay (bootstrap + fallback) + the WebRTC
/// custom transport, IP cleared (a loopback IP path would otherwise always win
/// and the relay-vs-WebRTC question would be meaningless — this also mirrors
/// the browser, which has no IP transport), `PreferWebRtc` selecting paths.
async fn bind_peer(
    relay: RelayMap,
    opaque: [u8; 16],
) -> Result<(Endpoint, Arc<WebRtcTransport>, MemoryLookup)> {
    let transport = Arc::new(WebRtcTransport::new(opaque.to_vec()));
    let lookup = MemoryLookup::new();
    let ep = Endpoint::builder(presets::N0)
        .relay_mode(RelayMode::Custom(relay))
        .ca_tls_config(CaTlsConfig::insecure_skip_verify())
        .clear_address_lookup()
        .address_lookup(lookup.clone())
        .clear_ip_transports()
        .add_custom_transport(transport.clone())
        .path_selector(Arc::new(PreferWebRtc))
        .bind()
        .await?;
    Ok((ep, transport, lookup))
}

/// Wait until the endpoint has published a relay address.
async fn relay_addr(ep: &Endpoint) -> Result<EndpointAddr> {
    timeout(STEP, async {
        let mut w = ep.watch_addr();
        loop {
            let a = w.get();
            if a.addrs.iter().any(|x| x.is_relay()) {
                return Ok::<_, anyhow::Error>(a);
            }
            w.updated().await?;
        }
    })
    .await
    .map_err(|_| anyhow!("timed out waiting for relay addr"))?
}

/// Relay + custom addr for a peer (what a stark ticket/lookup would carry).
fn mixed_addr(id: EndpointId, relay: &EndpointAddr, opaque: &[u8; 16]) -> EndpointAddr {
    EndpointAddr::from_parts(
        id,
        relay
            .addrs
            .iter()
            .cloned()
            .chain([TransportAddr::Custom(custom_addr_from_opaque_data(opaque))]),
    )
}

fn selected_kind(conn: &Connection) -> Option<&'static str> {
    conn.paths()
        .iter()
        .find(|p| p.is_selected())
        .map(|p| match p.remote_addr() {
            TransportAddr::Ip(_) => "ip",
            TransportAddr::Relay(_) => "relay",
            TransportAddr::Custom(_) => "custom",
            _ => "other",
        })
}

fn describe_paths(conn: &Connection) -> String {
    conn.paths()
        .iter()
        .map(|p| format!("{:?}(selected={})", p.remote_addr(), p.is_selected()))
        .collect::<Vec<_>>()
        .join(", ")
}

async fn run_echo(conn: &Connection, msg: &[u8]) -> Result<()> {
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(msg).await?;
    send.finish()?;
    let got = recv.read_to_end(1024).await?;
    if got != msg {
        bail!("echo mismatch: {got:?} != {msg:?}");
    }
    Ok(())
}

/// Poll until the connection's selected path is `want` (selection settles
/// asynchronously after paths validate).
async fn wait_selected(conn: &Connection, want: &str) -> Result<()> {
    timeout(STEP, async {
        loop {
            if selected_kind(conn) == Some(want) {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await
    .map_err(|_| {
        anyhow!(
            "timed out waiting for a {want} path to be selected; paths = [{}]",
            describe_paths(conn)
        )
    })?
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

struct Peers {
    server: Endpoint,
    server_id: EndpointId,
    server_relay: EndpointAddr,
    client: Endpoint,
    _router: Router,
}

/// Bind both peers, JSEP over a relay connection, attach the channel on both
/// sides, publish mixed addrs, and close the signaling connection.
///
/// Closing the signaling connection before the app dial is load-bearing, not
/// hygiene: iroh keeps a sticky `selected_path` per peer while any connection
/// is open, and a NEW connection's Initial goes ONLY to the selected path — so
/// dialing while the (relay-selected) signaling connection is open can never
/// touch the WebRTC path. Closing the last connection clears `selected_path`
/// and the next dial sends to ALL known addrs.
async fn bootstrap_webrtc(relay_map: RelayMap) -> Result<Peers> {
    let (server, server_transport, server_lookup) = bind_peer(relay_map.clone(), SERVER_OPAQUE).await?;
    let server_id = server.id();
    let router = Router::builder(server.clone())
        .accept(
            SIGNAL_ALPN,
            JsepAnswerer {
                transport: server_transport.clone(),
                remote_opaque: CLIENT_OPAQUE,
            },
        )
        .accept(ECHO_ALPN, Echo)
        .spawn();

    let (client, client_transport, client_lookup) = bind_peer(relay_map, CLIENT_OPAQUE).await?;

    let server_relay = relay_addr(&server).await?;
    let client_relay = relay_addr(&client).await?;
    client_lookup.add_endpoint_info(server_relay.clone());
    server_lookup.add_endpoint_info(client_relay.clone());

    let sig_conn = timeout(STEP, client.connect(server_relay.clone(), SIGNAL_ALPN))
        .await
        .map_err(|_| anyhow!("signaling connect timed out"))??;
    let (send, recv) = sig_conn.open_bi().await?;
    let mut sig = QuicSignaling::new(send, recv);
    let peer = timeout(STEP, negotiate_dc_as_offerer(&mut sig, "stark-spike"))
        .await
        .map_err(|_| anyhow!("JSEP negotiation timed out"))??;
    client_transport.attach_data_channel(
        peer,
        custom_addr_from_opaque_data(&SERVER_OPAQUE),
        AttachOptions::default(),
    )?;

    client_lookup.add_endpoint_info(mixed_addr(server_id, &server_relay, &SERVER_OPAQUE));
    server_lookup.add_endpoint_info(mixed_addr(client.id(), &client_relay, &CLIENT_OPAQUE));

    sig_conn.close(0u32.into(), b"signaling done");
    tokio::time::sleep(Duration::from_millis(750)).await;

    Ok(Peers {
        server,
        server_id,
        server_relay,
        client,
        _router: router,
    })
}

/// Finding 1 (POSITIVE): the WebRTC bridge carries a complete iroh connection.
/// With the relay shut down after signaling, the app dial's only viable addr is
/// the custom one — handshake and echo both ride the str0m data channel.
#[tokio::test]
async fn webrtc_carries_the_connection_when_the_relay_is_down() -> Result<()> {
    init_logging();
    let (relay_map, _url, relay_guard) = run_relay_server().await.context("local relay")?;
    let peers = bootstrap_webrtc(relay_map).await?;

    // Kill the relay: from here on, only the WebRTC channel can deliver.
    // (Dropping the Server is not enough — established relay client
    // connections keep being served; `shutdown()` tears them down too.)
    relay_guard
        .shutdown()
        .await
        .map_err(|e| anyhow!("relay shutdown: {e}"))?;
    tokio::time::sleep(Duration::from_millis(250)).await;

    let conn = timeout(
        STEP,
        peers.client.connect(
            mixed_addr(peers.server_id, &peers.server_relay, &SERVER_OPAQUE),
            ECHO_ALPN,
        ),
    )
    .await
    .map_err(|_| anyhow!("connect over WebRTC timed out"))??;

    run_echo(&conn, b"echo with no relay").await?;
    wait_selected(&conn, "custom").await?;
    run_echo(&conn, b"echo over webrtc, confirmed selected").await?;
    eprintln!("paths at end: [{}]", describe_paths(&conn));

    conn.close(0u32.into(), b"done");
    peers.client.close().await;
    peers.server.close().await;
    Ok(())
}

/// Finding 2 (POSITIVE): a peer addr advertising a custom addr with no attached
/// data channel (as when the WebRTC handshake failed) still connects — over the
/// relay. Per-connection fallback works.
#[tokio::test]
async fn unattached_webrtc_falls_back_to_relay() -> Result<()> {
    init_logging();
    let (relay_map, _url, _guard) = run_relay_server().await.context("local relay")?;

    let (server, _server_transport, server_lookup) = bind_peer(relay_map.clone(), SERVER_OPAQUE).await?;
    let server_id = server.id();
    let _router = Router::builder(server.clone()).accept(ECHO_ALPN, Echo).spawn();

    let (client, _client_transport, client_lookup) = bind_peer(relay_map.clone(), CLIENT_OPAQUE).await?;

    let server_relay = relay_addr(&server).await?;
    let client_relay = relay_addr(&client).await?;
    client_lookup.add_endpoint_info(mixed_addr(server_id, &server_relay, &SERVER_OPAQUE));
    server_lookup.add_endpoint_info(mixed_addr(client.id(), &client_relay, &CLIENT_OPAQUE));

    // Dial with relay + (dead) custom addr. No channel attached anywhere.
    let conn = timeout(
        STEP,
        client.connect(mixed_addr(server_id, &server_relay, &SERVER_OPAQUE), ECHO_ALPN),
    )
    .await
    .map_err(|_| anyhow!("fallback connect timed out"))??;

    run_echo(&conn, b"echo over relay").await?;
    wait_selected(&conn, "relay").await?;
    run_echo(&conn, b"still on relay").await?;
    eprintln!("paths at end: [{}]", describe_paths(&conn));

    conn.close(0u32.into(), b"done");
    client.close().await;
    server.close().await;
    Ok(())
}

/// Finding 3 (documented limitation): with both the relay and the WebRTC
/// channel live, the app dial RACES its Initial over both, and whichever
/// completes the handshake first is the path — permanently. On a loopback
/// relay that is always the relay (observed: the Initial demonstrably arrives
/// over the custom transport too, ~7ms later, and is discarded as a handshake
/// duplicate). In production the race would be genuinely nondeterministic.
/// Nothing in iroh 1.0.3 migrates an established connection onto a custom
/// path, so `PreferWebRtc` never gets a WebRTC candidate to prefer.
///
/// Asserted loosely (the connection works; the winner is logged) because the
/// winner is an artifact of relative latency, not a contract.
#[tokio::test]
async fn with_relay_up_the_dial_race_decides_the_path() -> Result<()> {
    init_logging();
    let (relay_map, _url, _guard) = run_relay_server().await.context("local relay")?;
    let peers = bootstrap_webrtc(relay_map).await?;

    let conn = timeout(
        STEP,
        peers.client.connect(
            mixed_addr(peers.server_id, &peers.server_relay, &SERVER_OPAQUE),
            ECHO_ALPN,
        ),
    )
    .await
    .map_err(|_| anyhow!("echo connect timed out"))??;

    run_echo(&conn, b"echo over whichever path won").await?;
    // Give selection time to settle before reporting.
    tokio::time::sleep(Duration::from_secs(2)).await;
    eprintln!(
        "race winner: {:?}; paths = [{}]",
        selected_kind(&conn),
        describe_paths(&conn)
    );
    run_echo(&conn, b"second echo").await?;

    conn.close(0u32.into(), b"done");
    peers.client.close().await;
    peers.server.close().await;
    Ok(())
}
