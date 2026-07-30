# Vendoring notes: iroh-webrtc-transport (anchalshivank variant)

Vendored from <https://github.com/anchalshivank/iroh-webrtc-transport.git>,
commit `a982968c07536b6033be0ca5e69e5c151b1410ac` (2026), repo root (this
upstream is a single crate, unlike SuddenlyHazel's workspace).

This is a **second** crate named `iroh-webrtc-transport` in `vendor/`. It is a
different project by a different author than `vendor/iroh-webrtc-transport`
(SuddenlyHazel's), which stark-net currently depends on. The two share a name
and a goal but no code or architecture:

| | SuddenlyHazel (current) | anchalshivank (this one) |
|---|---|---|
| Integration | Facade owning TWO endpoints (relay + relay-cleared webrtc) + its own dial/accept API | ONE ordinary `Endpoint` with `add_custom_transport` + `path_selector` |
| WebRTC stack (native) | webrtc-rs | str0m (sans-IO) + local UDP socket |
| Browser story | Full: wasm custom transport over `RtcPeerConnection` | None yet — upstream's browser demo only runs JSEP over iroh; the data channel stays in JS |
| Signaling | Internal (facade-managed) | Pluggable `Signaling` trait; JSEP offer/answer; caller attaches the channel |
| Peers per transport | Many (session hub) | ONE data channel per `WebRtcTransport` instance (single `bind`, single attach) |

## Why this exists

The facade model forced stark-net to abandon iroh-gossip and to route every
protocol through the facade's own connection API. A custom transport on the
*single* endpoint would let `iroh-gossip`/`iroh-blobs`/`iroh-docs` run
unchanged. Our 2026-07-24 spike (see
`vendor/iroh-webrtc-transport/tests/single_endpoint_webrtc.rs`) concluded
single-endpoint migration was a dead end — but that spike predates using
iroh 1.0.2's public `path_selector` API, which this upstream's examples are
built around (via the 0.97-era `transport_bias`, since removed). The spike in
`tests/` here re-litigates that question.

## What was vendored

`src/` minus `src/bin/` (demo servers/chats) and minus `src/jsep_ws.rs`
(WebSocket signaling; would drag in tokio-tungstenite/futures-util which stark
does not need — stark signals over iroh QUIC streams). `browser-iroh/`,
`static/`, `examples/`, and `scripts/` were not vendored. Upstream's
binary-only deps (nym-sdk, nym-noise-keys, axum, tower-http, tracing-subscriber)
were dropped with them.

## Local additions

**`src/web_peer.rs` (2026-07-29, NOT upstream): the wasm implementation of the
custom transport.** Upstream's browser story stops at signaling — its demo
keeps the data channel in JavaScript and never bridges it into iroh. This
module is the browser twin of `str0m_peer.rs`: the browser's own
`RTCPeerConnection` + data channel (web-sys), negotiated with the same
two-message `SignalEnvelope` offer/answer over any `Signaling` impl (ICE rides
inside the SDP: the browser waits for gathering to complete — "vanilla ICE" —
which interoperates with str0m's candidate-in-SDP on native), then attached to
the same `WebRtcTunnel` via a wasm `WebRtcTransport::attach_data_channel`
(inbound: `onmessage` → bounded queue → `poll_recv`, drops when full — QUIC
handles loss; outbound: `spawn_local` pump with `bufferedamountlow`
backpressure at 1 MiB/256 KiB water marks; the pump owns the
`RtcPeerConnection` so it is not GC'd). Event-handling patterns follow the
browser-verified code in `vendor/iroh-webrtc-transport`. `Signaling` and its
impls are `#[async_trait(?Send)]` on wasm (single-threaded; iroh's wasm stream
types are not `Send`). Compile-checked for `wasm32-unknown-unknown`; NOT yet
run in a browser.

## Local changes (delta from upstream)

1. **iroh 0.97 → 1.0** (same delta as the sibling crate's port, recorded in its
   VENDORING.md):
   - `CustomEndpoint::poll_recv`: `source_addrs: &mut [Addr]` →
     `recv_infos: &mut [RecvInfo]`; write `RecvInfo::new(remote_custom, None)`.
   - `CustomSender::poll_send` gained `src: Option<&CustomAddr>` (ignored).
   - deps: iroh/iroh-base `0.97` → `1`, noq-udp `0.9` → `1`, n0-watcher
     `0.6` → `1`.
   - Upstream examples' `Endpoint::builder(...).transport_bias(AddrKind::Custom(id), TransportBias...)`
     does not exist in iroh 1.0 (`TransportBias` went `pub(crate)`); the 1.0
     replacement is `.path_selector(Arc<dyn PathSelector>)` — see `tests/`.
2. **str0m crypto backend**: `str0m-aws-lc-rs` → `str0m-rust-crypto`
   (`build_rtc` in `src/str0m_peer.rs`). aws-lc-sys needs NASM on Windows
   MSVC; the pure-Rust backend also keeps a wasm door open.
3. **`src/lib.rs`**: removed `mod jsep_ws;` / `pub use jsep_ws::TcpWebSocket;`.

## ⚠ License

Upstream has **no LICENSE file and no `license` field** at the vendored commit.
Fine for a private experiment; must be resolved with the author (or the code
rewritten) before any release that includes it.

## Spike results (2026-07-29, iroh 1.0.3 — `tests/single_endpoint_direct_and_fallback.rs`)

The 2026-07-24 "single-endpoint migration is a dead end" conclusion is
**partially overturned**:

1. **POSITIVE — the bridge carries real iroh connections.** With the relay shut
   down after JSEP, an app dial completes its QUIC handshake and echoes data
   entirely over the str0m WebRTC data channel, and the custom path is
   selected. The old spike's "iroh never calls `is_valid_send_addr`" behavior
   is gone in iroh 1.0.3 — initial dial datagrams go to every known addr,
   including custom ones.
2. **POSITIVE — per-connection relay fallback works.** A dial whose custom
   addr has no attached channel connects over the relay and stays healthy.
3. **LIMITATION — path choice is a dial-time race, then sticky forever.**
   Preconditions for the custom path to be tried at all: the peer's previous
   connections must be CLOSED (iroh keeps a sticky per-peer `selected_path`
   while any connection is open, and new dials send ONLY to it), and the
   custom addr must be in the dial addr / lookup. Then whichever transport
   wins the handshake race is PathId 0 — permanently. iroh 1.0.3 never opens
   a custom path post-handshake (holepunching is IP-only; post-hoc re-add
   exists only for relay; the public `PathSelector` only chooses among
   already-open paths). On the loopback test relay, relay always wins the
   race; in production it would be nondeterministic.

**RESOLVED same day by patching iroh** — `vendor/iroh` (1.0.3 + an
`open_custom_paths` hook, see its VENDORING.md), substituted via
`[patch.crates-io]` in this crate's manifest and in the root workspace. With
the patch, `established_connection_migrates_to_webrtc` shows the target
behavior deterministically: the app connection completes its handshake over
the relay (worst case — the signaling connection is deliberately left open,
so the sticky `selected_path` forces relay), then migrates onto the WebRTC
path once it opens and validates, keeping the relay path as live backup —
and the still-open signaling connection is pulled onto WebRTC too. Relay
fallback (dead custom addr) is unaffected.

## Known limitations (upstream design, unchanged)

- ~~One `WebRtcTransport` = one WebRTC peer.~~ RESOLVED (2026-07-29, local
  rework; upstream remains single-channel): the tunnel now holds a
  `CustomAddr -> outbound queue` routing table. `attach_data_channel` may be
  called once per remote peer; `poll_send` routes by `dst`, inbound demuxes by
  each packet's `source_custom`, `is_valid_send_addr` answers from the table.
  Re-attaching an addr REPLACES its route (old pump drains out and exits; its
  cleanup is generation-guarded so it cannot tear down the successor) — safe
  re-negotiation after a dead channel. A dead channel's pump removes its route
  on exit, flipping `is_valid_send_addr` back to false. Proven by the
  `one_endpoint_reaches_two_webrtc_peers` test (one endpoint, two live
  channels, relay down, interleaved traffic). `bind()` is still once-only
  (iroh binds a transport once per endpoint). The demo-oriented
  `webrtc_out_sender()` accessor was removed with the single queue.
- Native negotiation gathers **host ICE candidates only** (no STUN/TURN), so
  cross-NAT native↔native won't punch. Irrelevant for stark: native iroh
  already hole-punches; WebRTC is for browsers, where the browser does ICE.
- ~~No wasm/browser implementation of the custom transport yet.~~ RESOLVED by
  `src/web_peer.rs` (see Local additions) — though real-browser verification
  is still pending. That iroh's wasm build accepts custom transports is
  already proven in production by the sibling crate's facade.
