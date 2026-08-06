# `stark-net` cleanup

A review of `crates/stark-net/src` (2026-08-05), ordered by what to change
first. The crate's layering is sound — `session` (protocol) over `backend`
(plumbing) over `proto` (wire) beside `mirror` (served state), with the WebRTC
bootstrap isolated behind a feature. Everything below is about the paths where a
network hiccup turns into a frozen session or silently divergent pixels, and
then about the accumulated shape.

Work happens on the `net-cleanup` branch, in the order listed. Each item is
checked off as it lands.

## Architectural

### A1. The gossip receive loop blocks on asset fetches — for everything

`recv_loop` awaits `resolve_asset` inline (`session.rs`, the action path). That
loop is the single consumer of the gossip stream, so one unresolvable asset
stalls **all** traffic: other peers' actions, presence frames, and
`NeighborUp`/`NeighborDown` bookkeeping. Worst case is `ASSET_RETRIES` (5)
rounds × 2 sources × (dial timeout + `ASSET_RETRY_DELAY`) — tens of seconds in
which the session looks dead.

The `await` is justified by ordering, and the ordering constraint is real for
grounds (§6.4). But it is a *per-need* constraint, not a global one:
`Engine::merge_remote` is idempotent by id and order-insensitive — that is what
the `ReplicatedTimeline` is for (§12.1).

**Fix.** Park rather than block. A map of `AssetNeed -> Vec<Action>` holds
actions waiting on an in-flight fetch, drained onto the event channel when the
resolver completes. Ordering between an asset and its dependents is preserved;
independent traffic keeps flowing. It also de-duplicates concurrent fetches of
the same need, which today can start twice (once off a presence head, once off
the commit).

### A2. Exhausting the retries silently produces the divergence the ground arm exists to prevent

When `fetch_asset` returns `None`, `resolve_asset` warns and the loop merges the
action anyway. For a brush that is the documented graceful degradation — the
stroke draws with the round tip. For a ground it is exactly the failure the code
comment calls "the whole fix": a `SetSurface` applied against `Flat` leaves every
later stroke depositing as though the canvas were smooth, and those pixels are
stored (§6.4). No later arrival un-bakes them. Five retries over ~1.5 s is a
short window for a peer that is itself still fetching.

**Fix.** Once actions are parked (A1) there is no deadline pressure, because
parking no longer blocks anything. A `Ground` need parks indefinitely and retries
on a backoff; a `Brush` keeps a bounded retry and its round-tip fallback.

### A3. Asset bytes are held three times, and the snapshot clones under the lock

Every asset lives in `Mirror.assets`/`Mirror.surfaces`, in the blobs `MemStore`,
and again in the engine. `AssetHashes` is a fourth structure holding the
`AssetId -> Hash` translation the mirror could derive.

`Mirror::document_file` deep-clones every action *and* every asset while holding
the `Mutex`, called synchronously from the protocol handler. A joiner arriving
into a large session stalls the local receive loop and every in-flight
`broadcast` for the duration.

**Fix**, two independent halves:

- Carry asset payloads as `Bytes` (`insert_content`, `RemoteEvent::Asset`,
  `add_blob` all take `Vec<u8>` and clone today).
- Have `Mirror` hold `AssetId -> Hash` and read bytes back from the blob store
  when serving a snapshot, collapsing the mirror's two byte maps *and*
  `AssetHashes` into one map.

### A4. `add_content`-before-commit is an unenforced ordering rule

`add_content` documents that callers must register content *before* committing an
action that references it, and `publish_wire` silently sends `asset: None` when
they do not. The violation is detected at the *far end*, as a warning, by which
point the peer has degraded content. Per the "rule out a class rather than
enumerate its instances" convention, this should not be a rule a call site can
forget.

**Fix.** `publish_wire` reports when `referenced_asset` yields a need whose hash
is unregistered, so the fault surfaces where it was committed rather than on
someone else's canvas.

### A5. Unbounded event channel with no backpressure

`take_events` hands out an `mpsc::UnboundedReceiver`, and the UI drains it while
taking `renderer.write()` per event. Presence arrives at ~30 Hz *per stroking
peer*; if the render lock is contended the queue grows without bound, and stale
presence frames are delivered in order rather than coalesced — so the UI works
through a backlog of cursor positions nobody wants.

`RemoteEvent::Presence`'s own doc says it "may be dropped freely" (§17.4).

**Fix.** A bounded channel where presence is dropped when full and actions and
assets still block, so the doc comment describes the code.

### A6. `take_events` returning `Option`

`host` returns `Self`; `join` returns `(Self, DocumentFile)`; the event stream is
then taken out of an `Option` afterwards with the caller asserting freshness
(`.expect("fresh session events")`). Returning the receiver from both
constructors removes the field, the `Option` and the `expect`.

## Cleanups

- **`backend::imp` is a vestigial wrapper** — the whole file is one `mod imp`
  re-exported at the bottom, the shape left over from having two backends.
- **`AssetNeed::Ground(SurfaceId)` should be `Ground(AssetId)`.** `SurfaceId` is
  `Flat | Image(AssetId)` and `Flat` is never a valid need, so the type can
  express nonsense and four sites pay for it: `content() -> Option`, the two
  `let Some(id) = … else { return }` guards, `ground_content_id`, and
  `referenced_asset`'s `.content().map(|_| …)`. Carrying the `AssetId` is
  lossless — the receiver reconstructs `SurfaceId::Image`.
- **`CollabSession` should hold a `Broadcaster`**, not rebuild one per call; its
  fields *are* `Broadcaster`'s plus four, and `broadcast`/`add_content`/`links`
  each clone a `GossipSender`, an `Endpoint`, a blob `Store` and three `Arc`s.
- **`finish` takes six positional arguments**, four straight off `Bound`.
- **`NetError::Other(String)` erases three distinct sources** — gossip
  subscribe/broadcast and blob fetch all collapse to a string, losing the cause
  chain `thiserror` exists to keep.
- **`.lock().expect("… poisoned")` appears ~15 times.**
- **`Dialer::local_id` returns `Result` but cannot fail.**
- **Visibility is inconsistent in `proto`**: `Stamped` is `pub(crate)` while
  `Wire` and `ALPN` in the same private module are `pub`.
- **`require_hash` doesn't require anything** — it warns and returns its argument.
- **The ticket has no version tag** though the ALPN does (`stark/collab/0`). A
  postcard `EndpointAddr + TopicId` will fail opaquely the first time either
  changes shape (§19).
- **`MAX_RESPONSE = 64 MiB` is a hard join ceiling with no diagnostic** — a long
  session simply stops accepting members.
- **Timing and limit policy is spread across four files.** One place makes the
  session's whole budget legible, which matters because A1 and A2 turn on those
  numbers.

## Test gaps

The three integration tests cover the happy paths well (convergence + undo,
mid-session shape and ground replication, founder handoff). Not covered, each
mapping to a finding above:

- An asset that never resolves — that no other traffic stalls behind it (A1),
  and what the peer does when retries run out (A2).
- Three peers where `origin != from`, exercising the `asset_sources` fallback the
  two-peer tests never reach.
- `GossipEvent::Lagged`, currently a `warn!` and `continue` with no recovery
  (§12.5).
- A `broadcast` of an action whose content was never registered (A4).
