# Vendoring notes: iroh (patched)

iroh `1.0.3`, copied verbatim from the crates.io source
(`registry/src/.../iroh-1.0.3`, upstream commit in `.cargo_vcs_info.json`),
plus one local patch. Substituted for the crates.io crate via
`[patch.crates-io]` in the root workspace manifest AND in
`vendor/iroh-webrtc-transport2/Cargo.toml` (that crate is workspace-excluded,
so the root patch does not reach it). License: BSD-3 (`LICENSE-BSD3`, kept).

## The patch (all sites marked `STARK PATCH`)

`src/socket/remote_map/remote_state.rs`: new `RemoteStateActor::open_custom_paths`
— opens a path for every known custom-transport addr on every live connection
(`open_path_on_conn` already no-ops for addrs a connection has open). Called
from three places:

1. `handle_msg_add_connection` — a new connection opens paths for custom addrs
   already known for the remote (e.g. from the dial `EndpointAddr`).
2. the `ResolveRemote` message arm — a dial that introduces custom addrs opens
   them on existing connections.
3. the address-lookup result arm — same, for addrs arriving via lookup.

## Why

Upstream iroh 1.0.3 only ever establishes a custom path when the custom
transport wins the initial dial handshake race: holepunching opens IP paths
only, only relay paths are re-added post-handshake, and `PathSelector`
(public, unstable) picks solely among already-open paths. So a connection that
lands on the relay — which is guaranteed when any earlier connection to the
peer holds the sticky `selected_path`, and otherwise decided by raw latency —
can never migrate onto WebRTC. With the patch, custom paths open post-handshake,
validate over the custom transport, and the selector (even the DEFAULT one:
custom = Primary, relay = Backup) migrates traffic onto them. Proven by
`vendor/iroh-webrtc-transport2/tests/single_endpoint_direct_and_fallback.rs`.

Dead custom addrs (advertised but no channel attached) are harmless: their
paths never validate, gain no stats, and are never selected; relay fallback is
unaffected.

This looks upstreamable as a feature request: "open known custom-transport
addrs as paths on live connections, like relay re-add / IP holepunching".

## Updating

Re-copy a newer crates.io source over this directory and re-apply the
`STARK PATCH` sites (grep for the marker).
