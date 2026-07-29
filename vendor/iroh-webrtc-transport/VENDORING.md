# Vendoring notes

This crate is vendored from
[SuddenlyHazel/iroh-webrtc-transport](https://github.com/SuddenlyHazel/iroh-webrtc-transport.git),
subdirectory `crates/iroh-webrtc-transport`, at commit
`fdb511da1650600dba5b97e04880b1b9aa2738a6`.

It gives `stark-net` direct (WebRTC) connections on the web target, where iroh
cannot hole-punch and would otherwise relay all traffic through n0's relays. It
is pulled in only behind `stark-net`'s `webrtc` feature.

## Local changes vs. upstream

- **iroh 0.98 → 1.0.** Upstream targets `iroh`/`iroh-base` 0.98; we run iroh 1.0.
  The custom-transport API (`iroh::endpoint::transports`,
  `iroh_base::CustomAddr`) is unchanged across the bump, so the change is
  confined to `Cargo.toml`: `iroh` and `iroh-base` to `1`, and the shared
  transport types `noq-udp` (0.10 → 1) and `n0-watcher` (0.6 → 1) to match
  iroh's own versions.
- **wasm-bindgen pin relaxed** from `=0.2.118` to `0.2` so it unifies with the
  version `stark-ui` (Dioxus) resolves.
- `publish = false`.
- **WebRTC application dials no longer restart the WebRTC endpoint.**
  Upstream's `dial_webrtc_application_connection` called
  `restart_webrtc_endpoint()` on every dial, closing the node's single WebRTC
  endpoint — and with it **every established WebRTC connection the node had**
  (dialed and accepted, all peers, all ALPNs). Invisible when a node makes one
  connection ever; fatal for stark-net, where a join dials twice (catch-up +
  mesh) and the mesh redials on every drop: each dial killed the connections
  before it, each kill triggered a redial, and the swarm flapped forever.
  Dials now go through `ensure_webrtc_endpoint()`, which reuses the live
  endpoint and only binds when none exists.
- **Connections share a peer's live channel (browser runtime).** Reusing the
  endpoint surfaced why upstream restarted it: iroh's remote map prefers an
  already-validated path, so every later connection to a peer selects the
  *first* session's channel — and upstream's per-session guards
  (`require_webrtc_selected_path` exact-match, one-shot acceptor sessions)
  rejected them ("selected WebRTC custom path does not match session").
  A channel is just a wire, so the model is now: a connection belongs to
  whichever **live** session's address its selected path names.
  `accept_webrtc_connection` looks sessions up by that address (dropping the
  role/ALPN/unresolved conditions), the dial path resolves the carrying
  session the same way and admits the connection under it, and
  `connection_close` only closes a session once **no other open connection
  rides it** (previously: first close killed the shared channel).
  Redundant cost that remains: each dial still negotiates a fresh
  RTCPeerConnection that goes unused when an existing channel is selected;
  skipping bootstrap when a live channel exists is a possible follow-up.
- **`BrowserResolvedTransport` made public and exposed on connections.** The
  runtime already records whether each facade connection resolved to direct
  WebRTC or the iroh relay fallback (`BrowserConnectionInfo.transport`), but the
  facade dropped it. The enum is now `pub` (re-exported from `browser`), and
  `BrowserWebRtcConnection` carries it, readable via a new `transport()` method
  — `stark-net` uses it to tell the UI how each peer is reached.

Any further deltas from upstream should be recorded here.
