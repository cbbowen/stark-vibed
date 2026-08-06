# `stark-net` cleanup

A review of `crates/stark-net/src` (2026-08-05), ordered by what to change
first. The crate's layering is sound — `session` (protocol) over `backend`
(plumbing) over `proto` (wire) beside `mirror` (served state), with the WebRTC
bootstrap isolated behind a feature. Everything below is about the paths where a
network hiccup turns into a frozen session or silently divergent pixels, and
then about the accumulated shape.

Work happened on the `net-cleanup` branch, in the order listed. **All of it has
landed**; the status line under each item records what shipped, and the two
places where the implementation departed from the plan say why.

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

**Landed** as `waitlist.rs`. It holds the mirror handle, because it is the only
place that locks both mutexes and the lock order is better as a fact about one
file than a convention every call site could break; and it owns the release, so
"content reaches the engine before the actions that named it" is what delivering
*is* rather than something a caller does afterwards.

One thing the plan missed: a **local** `add_content` satisfies a parked remote
action exactly as a fetch does, and going straight to the mirror would have left
that action waiting for bytes already in hand. `Broadcaster::add_content` goes
through the waitlist.

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

**Landed.** The backoff widens to a 30 s ceiling and stops when the engine goes
away, so an unbounded retry cannot outlive its session. Verified against
`ReplicatedTimeline::insert` first: an out-of-order insert resyncs rather than
appending, so the strokes that merged ahead of a late `SetSurface` really are
replayed against the real ground.

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

**Landed, with the second half cut down.** Payloads are `Bytes`, `AssetHashes`
is gone (the mirror holds the transfer hash beside the content it describes), and
`Mirror::snapshot` clones handles under the lock while `Snapshot::into_file`
materializes off it — so the lock now covers the log and nothing else.

The mirror still keeps its own `Bytes` rather than reading back from the blob
store. Once a payload is refcounted the "third copy" is a pointer, so what was
left to win was small, and the cost was not: `from_file` and the snapshot path
would both become async and store-dependent, dragging the blob store into the
catch-up protocol handler. Not worth it for a pointer.

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

**Landed** as a `tracing::error!` naming the content. The payload is still sent:
the action is committed locally already, so withholding it would guarantee the
divergence rather than risk it.

### A5. Unbounded event channel with no backpressure

`take_events` hands out an `mpsc::UnboundedReceiver`, and the UI drains it while
taking `renderer.write()` per event. Presence arrives at ~30 Hz *per stroking
peer*; if the render lock is contended the queue grows without bound, and stale
presence frames are delivered in order rather than coalesced — so the UI works
through a backlog of cursor positions nobody wants.

`RemoteEvent::Presence`'s own doc says it "may be dropped freely" (§17.4).

**Fix.** A bounded channel where presence is dropped when full and actions and
assets still block, so the doc comment describes the code.

**Landed as a quota rather than a bounded channel.** A bounded channel makes
every send async, which would have pushed `async` onto `Broadcaster::add_content`
— a synchronous call from Dioxus event handlers — for the sake of a queue that
actions can never grow anyway. A counter capping only what can be queued reaches
the same place: `PresenceQuota` takes a slot when a frame is queued and frees it
when one is handed over.

### A6. `take_events` returning `Option`

`host` returns `Self`; `join` returns `(Self, DocumentFile)`; the event stream is
then taken out of an `Option` afterwards with the caller asserting freshness
(`.expect("fresh session events")`). Returning the receiver from both
constructors removes the field, the `Option` and the `expect`.

**Landed** as `Events`, which is also where A5's quota is accounted — the two are
one change, because both ends of a quota have to agree on a number and that is
what turns the receiver into a type. It carries `recv` and `try_recv`, the latter
for a pump driven by a frame clock rather than by the stream.

## Cleanups

All landed except the last two, which are noted below with the reason.

- **`backend::imp` is a vestigial wrapper** — the whole file is one `mod imp`
  re-exported at the bottom, the shape left over from having two backends.
  *Flattened.*
- **`AssetNeed::Ground(SurfaceId)` should be `Ground(AssetId)`.** `SurfaceId` is
  `Flat | Image(AssetId)` and `Flat` is never a valid need, so the type can
  express nonsense and four sites pay for it: `content() -> Option`, the two
  `let Some(id) = … else { return }` guards, `ground_content_id`, and
  `referenced_asset`'s `.content().map(|_| …)`. Carrying the `AssetId` is
  lossless — the receiver reconstructs `SurfaceId::Image`. *Done;
  `AssetNeed::ground` answers the `Flat` question once, at the one place it
  arises, and `content()` is total.*
- **`CollabSession` should hold a `Broadcaster`**, not rebuild one per call; its
  fields *are* `Broadcaster`'s plus four, and `broadcast`/`add_content`/`links`
  each clone a `GossipSender`, an `Endpoint`, a blob `Store` and three `Arc`s.
  *Done.*
- **`finish` takes six positional arguments**, four straight off `Bound`. *Takes
  `Bound`.*
- **`NetError::Other(String)` erases three distinct sources** — gossip
  subscribe/broadcast and blob fetch all collapse to a string, losing the cause
  chain `thiserror` exists to keep. *Typed `Gossip`, `BlobFetch` and `BlobRead`
  variants with `#[from]`; one `Other` remains, for a `finish()` on a stream.*
- **`Dialer::local_id` returns `Result` but cannot fail.** *Done.*
- **Visibility is inconsistent in `proto`**: `Stamped` is `pub(crate)` while
  `Wire` and `ALPN` in the same private module are `pub`. *All `pub(crate)`.*
- **`require_hash` doesn't require anything** — it warns and returns its
  argument. *`hash_or_warn`.*
- **The ticket has no version tag** though the ALPN does (`stark/collab/0`). A
  postcard `EndpointAddr + TopicId` will fail opaquely the first time either
  changes shape (§19). *A leading version byte, with a test that a mismatch is
  named rather than guessed at.*
- **`MAX_RESPONSE = 64 MiB` is a hard join ceiling with no diagnostic** — a long
  session simply stops accepting members. *Crossing half of it warns, while
  joining still works.*

Not done:

- **`.lock().expect("… poisoned")` appears ~15 times.** A1 and A3 took most of
  them out on their own: the waitlist absorbed the mirror locking and the
  `AssetHashes` map went away entirely, leaving few enough that a wrapper would
  add a layer to save less than it costs.
- **Timing and limit policy is spread across four files.** Left where it is. The
  premise was that A1 and A2 turn on those numbers — and they do, but the numbers
  they turn on now sit together in `session.rs` next to the code that reads them.
  The rest are each one constant in the file that owns it, carrying the rationale
  that only reads well beside its use; gathering them would trade that for a list.

## Test gaps

The three integration tests covered the happy paths well (convergence + undo,
mid-session shape and ground replication, founder handoff). What was missing,
each mapping to a finding above:

- **An asset that never resolves** — that no other traffic stalls behind it (A1),
  and what the peer does when retries run out (A2). *Covered as `Waitlist` unit
  tests: parking, release ordering, resolver de-duplication, abandonment, and
  that one parked need does not hold up another. The invariants live in that type
  and reaching them through two endpoints would test the swarm instead, more
  slowly and less exactly. Both mutations tried (releasing before the asset event;
  `imported` recording without releasing) fail exactly one test each.*
- **A `broadcast` of an action whose content was never registered** (A4).
  *Covered — and the interesting half turned out to be the receiver's: a need with
  no transfer hash has nothing to fetch, so parking it would park it forever.*
- **Three peers.** *Covered as content reaching a peer that joined through an
  intermediary — the first test where the deliverer is not the author.*

Still uncovered:

- **The `asset_sources` fallback** from author to deliverer. Forcing it needs the
  author unreachable at the moment the action is delivered, which is a race to
  arrange rather than a state to set up.
- **The receive loop not blocking**, end to end. The logical property is pinned
  in the waitlist tests; pinning it over a real swarm needs a forged `Stamped`
  carrying a hash nobody serves, and the type is crate-private — reachable only
  by adding a test-only publish hook, which is the inert scaffolding the
  conventions rule out.
- **`GossipEvent::Lagged`**, still a `warn!` and `continue` with no recovery
  (§12.5). Unchanged by this branch, and not reachable without driving gossip
  into overflow.
