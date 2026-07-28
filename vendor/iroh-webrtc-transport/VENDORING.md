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
- **`BrowserResolvedTransport` made public and exposed on connections.** The
  runtime already records whether each facade connection resolved to direct
  WebRTC or the iroh relay fallback (`BrowserConnectionInfo.transport`), but the
  facade dropped it. The enum is now `pub` (re-exported from `browser`), and
  `BrowserWebRtcConnection` carries it, readable via a new `transport()` method
  — `stark-net` uses it to tell the UI how each peer is reached.

Any further deltas from upstream should be recorded here.
