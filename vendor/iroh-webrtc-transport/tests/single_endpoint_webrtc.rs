//! Spike (NEGATIVE RESULT): can `stark-net` keep one ordinary `presets::N0`
//! endpoint — so iroh-gossip, ALPNs and relays work unchanged — and merely *add*
//! a WebRTC custom transport that traffic migrates onto once a data channel is
//! bootstrapped over the relay? **No, not with this crate + iroh 1.0.**
//!
//! Without using the facade, we bind one endpoint per peer via
//! `configure_endpoint(builder, transport)` (relay + custom, IP cleared), run a
//! hand-rolled SDP/ICE exchange over an ordinary iroh ALPN driving
//! `NativeWebRtcSession`, and then dial an echo ALPN. The WebRTC data channel
//! genuinely establishes (INFO logs show both peers `connected` / channel
//! `open`), but iroh keeps the app connection on relay and never even attempts
//! the custom path — see the test's doc comment for the full analysis.
//!
//! For contrast, iroh's own `test_custom_transport_wins_over_relay` (in-memory
//! transport) *does* select custom over relay, and the crate's `native_facade.rs`
//! WebRTC test passes — but the facade uses a *second, relay-cleared* endpoint
//! for app traffic. That two-endpoint split is what makes WebRTC actually win.

#![cfg(all(
    feature = "native",
    not(all(target_family = "wasm", target_os = "unknown"))
))]

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId, RelayMap, RelayMode, TransportAddr, Watcher,
    address_lookup::MemoryLookup,
    endpoint::{Connection, RecvStream, SendStream, presets},
    protocol::{AcceptError, ProtocolHandler, Router},
    test_utils::run_relay_server,
    tls::CaTlsConfig,
};
use iroh_webrtc_transport::{
    config::WebRtcSessionConfig,
    native::{
        LocalIceEvent, NativeWebRtcSession, WEBRTC_DATA_CHANNEL_LABEL, WebRtcAddr, WebRtcIceCandidate,
    },
    transport::{WebRtcTransport, WebRtcTransportConfig, configure_endpoint},
};
use n0_future::{task, time::timeout};
use serde::{Deserialize, Serialize};

const SIGNAL_ALPN: &[u8] = b"stark-spike/webrtc-signal/0";
const ECHO_ALPN: &[u8] = b"stark-spike/echo/0";
const SESSION_ID: [u8; 16] = [0x5au8; 16];
const STEP: Duration = Duration::from_secs(15);

/// One hand-rolled signaling message over the bootstrap stream.
#[derive(Debug, Serialize, Deserialize)]
enum Sig {
    Offer { session_id: [u8; 16], sdp: String },
    Answer { sdp: String },
    Ice { candidate: String, sdp_mid: Option<String>, sdp_mline_index: Option<u16> },
    IceEnd,
}

// ---- length-prefixed framing over an ordered iroh bi-stream ----

async fn write_msg(send: &mut SendStream, msg: &Sig) -> Result<()> {
    let bytes = serde_json::to_vec(msg)?;
    let len = u32::try_from(bytes.len())?.to_le_bytes();
    send.write_all(&len).await?;
    send.write_all(&bytes).await?;
    Ok(())
}

async fn read_msg(recv: &mut RecvStream) -> Result<Option<Sig>> {
    let mut len = [0u8; 4];
    if recv.read_exact(&mut len).await.is_err() {
        return Ok(None); // clean EOF / peer finished
    }
    let n = u32::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| anyhow!("short signaling frame: {e}"))?;
    Ok(Some(serde_json::from_slice(&buf)?))
}

/// Forward this session's locally-gathered ICE onto the signaling stream until
/// end-of-candidates, then finish the send side. Runs as its own task.
fn spawn_ice_forwarder(session: NativeWebRtcSession, mut send: SendStream) -> task::JoinHandle<()> {
    task::spawn(async move {
        loop {
            match session.next_local_ice().await {
                Ok(LocalIceEvent::Candidate(c)) => {
                    let msg = Sig::Ice {
                        candidate: c.candidate,
                        sdp_mid: c.sdp_mid,
                        sdp_mline_index: c.sdp_mline_index,
                    };
                    if write_msg(&mut send, &msg).await.is_err() {
                        return;
                    }
                }
                Ok(LocalIceEvent::EndOfCandidates) => {
                    let _ = write_msg(&mut send, &Sig::IceEnd).await;
                    let _ = send.finish();
                    return;
                }
                Err(_) => return,
            }
        }
    })
}

/// Apply remote signaling (answer/offer already handled by caller) until the
/// remote sends end-of-candidates.
async fn drain_remote_ice(session: &NativeWebRtcSession, recv: &mut RecvStream) -> Result<()> {
    while let Some(sig) = read_msg(recv).await? {
        match sig {
            Sig::Ice { candidate, sdp_mid, sdp_mline_index } => {
                session
                    .add_ice_candidate(WebRtcIceCandidate { candidate, sdp_mid, sdp_mline_index })
                    .await?;
            }
            Sig::IceEnd => {
                session.add_end_of_candidates().await?;
                return Ok(());
            }
            other => bail!("unexpected signaling message during ICE: {other:?}"),
        }
    }
    Ok(())
}

/// Client/offerer half of the bootstrap. Returns the live session (keep it
/// alive for the data channel to persist).
async fn run_offerer(
    transport: &WebRtcTransport,
    local_id: EndpointId,
    remote_id: EndpointId,
    mut send: SendStream,
    mut recv: RecvStream,
) -> Result<NativeWebRtcSession> {
    let hub = transport.session_hub();
    let remote_addr = WebRtcAddr::session(remote_id, SESSION_ID).to_custom_addr();
    transport.advertise_local_addr(WebRtcAddr::session(local_id, SESSION_ID).to_custom_addr());

    let session = NativeWebRtcSession::new_offerer_with_config(
        hub,
        remote_addr,
        SESSION_ID,
        WEBRTC_DATA_CHANNEL_LABEL,
        WebRtcSessionConfig::no_ice_servers(),
    )
    .await?;

    let offer = session.create_offer().await?;
    write_msg(&mut send, &Sig::Offer { session_id: SESSION_ID, sdp: offer }).await?;
    let ice = spawn_ice_forwarder(session.clone(), send);

    // First remote message must be the answer, then remote ICE (stream ordered).
    match read_msg(&mut recv).await?.context("offerer got no answer")? {
        Sig::Answer { sdp } => session.apply_answer(sdp).await?,
        other => bail!("expected answer, got {other:?}"),
    }
    drain_remote_ice(&session, &mut recv).await?;
    let _ = ice.await;
    Ok(session)
}

/// A protocol handler that runs the answerer bootstrap and stashes the live
/// session so it outlives this signaling connection.
#[derive(Clone)]
struct SignalHandler {
    transport: WebRtcTransport,
    local_id: EndpointId,
    sessions: std::sync::Arc<std::sync::Mutex<Vec<NativeWebRtcSession>>>,
}

impl std::fmt::Debug for SignalHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalHandler").finish_non_exhaustive()
    }
}

impl ProtocolHandler for SignalHandler {
    async fn accept(&self, conn: Connection) -> std::result::Result<(), AcceptError> {
        let remote_id = conn.remote_id();
        let (send, mut recv) = conn.accept_bi().await?;

        let session = answer(&self.transport, self.local_id, remote_id, send, &mut recv)
            .await
            .map_err(|e| AcceptError::from_boxed(e.into()))?;
        // Keep the session (and its data channel) alive for the whole test by
        // stashing it and holding the connection open until the peer closes.
        self.sessions.lock().unwrap().push(session);
        conn.closed().await;
        Ok(())
    }
}

/// The answerer flow: read the offer, answer it, exchange ICE.
async fn answer(
    transport: &WebRtcTransport,
    local_id: EndpointId,
    remote_id: EndpointId,
    mut send: SendStream,
    recv: &mut RecvStream,
) -> Result<NativeWebRtcSession> {
    let sdp = match read_msg(recv).await?.context("answerer got no offer")? {
        Sig::Offer { session_id, sdp } if session_id == SESSION_ID => sdp,
        other => bail!("expected offer, got {other:?}"),
    };

    let hub = transport.session_hub();
    let remote_addr = WebRtcAddr::session(remote_id, SESSION_ID).to_custom_addr();
    transport.advertise_local_addr(WebRtcAddr::session(local_id, SESSION_ID).to_custom_addr());

    let session = NativeWebRtcSession::new_answerer_with_config(
        hub,
        remote_addr,
        SESSION_ID,
        WebRtcSessionConfig::no_ice_servers(),
    )
    .await?;
    session.apply_offer(sdp).await?;
    let ans = session.create_answer().await?;
    write_msg(&mut send, &Sig::Answer { sdp: ans }).await?;
    let ice = spawn_ice_forwarder(session.clone(), send);
    drain_remote_ice(&session, recv).await?;
    let _ = ice.await;
    Ok(session)
}

#[derive(Debug, Clone)]
struct Echo;

impl ProtocolHandler for Echo {
    async fn accept(&self, conn: Connection) -> std::result::Result<(), AcceptError> {
        let (mut send, mut recv) = conn.accept_bi().await?;
        let msg = recv.read_to_end(1024).await.map_err(AcceptError::from_err)?;
        send.write_all(&msg).await.map_err(AcceptError::from_err)?;
        send.finish().map_err(AcceptError::from_err)?;
        conn.closed().await;
        Ok(())
    }
}

fn webrtc_transport() -> WebRtcTransport {
    WebRtcTransport::new(WebRtcTransportConfig::default())
}

/// Bind one peer as a single, otherwise-normal endpoint: relay (for bootstrap +
/// fallback) + a WebRTC custom transport, with IP transports cleared so the only
/// competing path is the relay — mirroring the browser, and making "did iroh
/// pick WebRTC over relay?" a meaningful question (a fast loopback IP path would
/// otherwise always win). This is the shape we'd use in `stark-net`, minus IP.
async fn bind_peer(relay: RelayMap) -> Result<(Endpoint, WebRtcTransport, MemoryLookup)> {
    let transport = webrtc_transport();
    // A `MemoryLookup` is how we teach iroh a peer's *custom* address. Passing a
    // custom addr inline to `connect()` is NOT enough — iroh only probes custom
    // paths that arrive via address lookup / the remote map (this is exactly how
    // the crate's own facade injects them, via `protocol_transport_lookup`).
    let lookup = MemoryLookup::new();
    let ep = configure_endpoint(
        Endpoint::builder(presets::N0)
            .relay_mode(RelayMode::Custom(relay))
            .ca_tls_config(CaTlsConfig::insecure_skip_verify())
            .clear_address_lookup()
            .address_lookup(lookup.clone())
            .clear_ip_transports(),
        transport.clone(),
    )
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

fn selected_kind(conn: &Connection) -> Option<String> {
    conn.paths().iter().find(|p| p.is_selected()).map(|p| {
        match p.remote_addr() {
            TransportAddr::Ip(_) => "ip",
            TransportAddr::Relay(_) => "relay",
            TransportAddr::Custom(_) => "custom",
            _ => "other",
        }
        .to_string()
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

/// Spike result (NEGATIVE): a single iroh endpoint carrying **both** a relay
/// path and a real WebRTC custom transport does **not** migrate an app
/// connection onto WebRTC. We bootstrap a genuine WebRTC data channel over the
/// relay (it reaches `connected` / `open` — see the INFO logs), advertise the
/// peer's custom addr every way iroh offers (inline in `connect`, via a
/// `MemoryLookup`, and as a custom-only `EndpointAddr`), yet iroh never once
/// calls `CustomSender::is_valid_send_addr` — it never even *attempts* the
/// custom path once a relay path is already known for that peer. The echo rides
/// relay and stays there.
///
/// Why: WebRTC signaling must ride an existing connection (the relay), so the
/// remote map always learns the relay path first; iroh then sticks to it on that
/// endpoint. This is exactly why the crate's facade uses a *second*,
/// relay-cleared endpoint for app traffic (see `native_facade.rs`, which passes)
/// — the two-endpoint split is fundamental, not incidental.
///
/// Kept (as `#[ignore]`) as executable documentation of the finding and a
/// re-usable single-endpoint bootstrap harness. Run with `--ignored`.
#[tokio::test]
#[ignore = "spike: documents that a single relay+custom endpoint does NOT migrate to WebRTC"]
async fn single_endpoint_does_not_migrate_to_webrtc() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();

    let (relay_map, _url, _relay_guard) = run_relay_server()
        .await
        .context("failed to start local relay server")?;

    let sessions = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let (server, server_transport, server_lookup) = bind_peer(relay_map.clone()).await?;
    let server_id = server.id();
    let _router = Router::builder(server.clone())
        .accept(
            SIGNAL_ALPN,
            SignalHandler {
                transport: server_transport.clone(),
                local_id: server_id,
                sessions: sessions.clone(),
            },
        )
        .accept(ECHO_ALPN, Echo)
        .spawn();

    let (client, client_transport, client_lookup) = bind_peer(relay_map.clone()).await?;
    let client_id = client.id();

    // Make sure both peers are reachable over the relay before bootstrapping.
    let server_addr = relay_addr(&server).await?;
    let client_addr = relay_addr(&client).await?;
    // Teach each peer how to reach the other over the relay (offline: no DNS).
    client_lookup.add_endpoint_info(server_addr.clone());
    server_lookup.add_endpoint_info(client_addr.clone());

    // 1) Bootstrap the WebRTC session over the relay connection.
    let signal_conn = timeout(STEP, client.connect(server_addr.clone(), SIGNAL_ALPN))
        .await
        .map_err(|_| anyhow!("signal connect timed out"))??;
    let (send, recv) = signal_conn.open_bi().await?;
    let _client_session = timeout(
        STEP,
        run_offerer(&client_transport, client_id, server_id, send, recv),
    )
    .await
    .map_err(|_| anyhow!("offerer bootstrap timed out"))??;

    // Now that the WebRTC data channel is up, publish each peer's *custom* addr
    // (session-scoped) into the other's lookup, alongside its relay addr. This
    // is the piece the earlier attempt missed: iroh only probes custom paths
    // learned via address lookup, not custom addrs passed inline to connect().
    let server_custom = TransportAddr::Custom(WebRtcAddr::session(server_id, SESSION_ID).to_custom_addr());
    let client_custom = TransportAddr::Custom(WebRtcAddr::session(client_id, SESSION_ID).to_custom_addr());
    client_lookup.add_endpoint_info(EndpointAddr::from_parts(
        server_id,
        server_addr.addrs.iter().cloned().chain(std::iter::once(server_custom)),
    ));
    server_lookup.add_endpoint_info(EndpointAddr::from_parts(
        client_id,
        client_addr.addrs.iter().cloned().chain(std::iter::once(client_custom)),
    ));

    // Give the data channel a moment to be fully open + its hub pump running.
    tokio::time::sleep(Duration::from_secs(1)).await;

    // 2) Dial the echo ALPN to the CUSTOM addr explicitly (the facade's pattern),
    //    to see whether iroh will use the WebRTC path when told to directly.
    let custom_only = EndpointAddr::from_parts(
        server_id,
        [TransportAddr::Custom(
            WebRtcAddr::session(server_id, SESSION_ID).to_custom_addr(),
        )],
    );
    let echo = timeout(STEP, client.connect(custom_only, ECHO_ALPN))
        .await
        .map_err(|_| anyhow!("echo connect timed out"))??;

    // The echo works — but over relay. Document the observed (negative) result:
    // the real WebRTC channel is up, yet iroh never selected the custom path.
    run_echo(&echo, b"hello over webrtc").await?;
    let kind = selected_kind(&echo);
    eprintln!("echo selected path kind: {kind:?}; all paths = [{}]", describe_paths(&echo));
    assert_eq!(
        kind.as_deref(),
        Some("relay"),
        "unexpected: single endpoint DID migrate to WebRTC — revisit the integration plan"
    );

    echo.close(0u32.into(), b"done");
    signal_conn.close(0u32.into(), b"done");
    Ok(())
}
