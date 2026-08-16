# `stark-net` cleanup

A review of [crates/stark-net/](crates/stark-net/) — correctness, performance,
and structure — with what to do about each finding and how you would know it
worked.

> **The identifiers here are `N1`–`N11`, deliberately not `§n.m`.** Design-doc
> section numbers are stable and cited from ~1000 places in the source
> (CLAUDE.md); a work list is neither stable nor citable, and giving it §numbers
> would put entries into a namespace that must keep resolving. Where a finding
> contradicts or extends a design section, it says so and cites it.

3,640 lines across `src/` and `tests/`. The crate does the hard part well:
`waitlist.rs` is the best-designed piece of concurrency in the workspace — the
lock order lives in one file instead of in a convention every call site could
break, the claim/deliver race is closed by construction rather than by a check,
and it is the one module here with unit tests, because it is the one module that
can be tested without a swarm. The mirror/blob-store split is right, the ticket
carries a version byte for the reason §19 says it should, and `transport/direct.rs`
reasons carefully about the single ordering that matters (never announce the
custom addr before the channel attaches). What follows is the edges.

Three of these — [N1](#n1-there-is-no-anti-entropy),
[N2](#n2-a-joining-peer-serves-an-empty-document-as-if-it-were-complete) and
[N4](#n4-a-ground-can-wait-forever-on-two-peers-who-have-left) — can produce a
wrong picture with no error anywhere. They are the ones worth doing first.

## Ranked

| | Finding | Kind | Size | Status |
|---|---|---|---|---|
| [N1](#n1-there-is-no-anti-entropy) | There is no anti-entropy | correctness | large | open |
| [N2](#n2-a-joining-peer-serves-an-empty-document-as-if-it-were-complete) | A joining peer serves an empty document as if it were complete | correctness | small | open |
| [N3](#n3-nothing-checks-the-stamped-origin-against-the-actions-author) | Nothing checks the stamped origin against the action's author | correctness | small | open |
| [N4](#n4-a-ground-can-wait-forever-on-two-peers-who-have-left) | A ground can wait forever on two peers who have left | correctness | small | open |
| [N5](#n5-a-failed-join-leaks-its-endpoint) | A failed join leaks its endpoint | correctness | small | open |
| [N6](#n6-the-snapshot-clones-the-whole-log-under-the-receive-loops-lock) | The snapshot clones the whole log under the receive loop's lock | performance | medium | open |
| [N7](#n7-every-join-copies-every-asset-twice-and-re-encodes-the-document) | Every join copies every asset twice and re-encodes the document | performance | small | open |
| [N8](#n8-one-send-task-per-dispatch-makes-wire-order-accidental) | One send task per dispatch makes wire order accidental | structure | small | open |
| [N9](#n9-dialer-is-the-missing-seam) | `Dialer` is the missing seam | test health | medium | open |
| [N10](#n10-the-session-spawns-tasks-it-cannot-stop) | The session spawns tasks it cannot stop | structure | medium | open |
| [N11](#n11-smaller-things) | Smaller things | mixed | small | open |

Suggested order: **N2**, **N3**, **N5** first — each is a contained change and
two of them close a silent-divergence path. Then **N9**, because the fake it
introduces is what makes **N1** and **N4** testable at all. Then **N1**, which is
the large one and wants the reconciliation request that **N4** also uses. **N6**
and **N7** are independent and can go any time.

---

## N1. There is no anti-entropy

Three paths drop an action permanently for every peer that is already in the
session, and none of them has a recovery mechanism.

**Gossip lagged.** `session.rs`'s receive loop:

```rust
Ok(GossipEvent::Lagged) => {
    tracing::warn!("gossip lagged; some remote actions may be missing");
    continue;
}
```

with the comment "peers converge again on the next snapshot fetch (§12.5)".
**Nothing ever fetches a snapshot after join.** `Catchup` is opened once, in
`CollabSession::join`, and closed immediately after the one request. That
sentence describes a mechanism that does not exist — the same shape of claim
[C1](CORE_CLEANUP.md#c1-there-is-no-gpu-failure-path) was.

**A failed broadcast.** `Broadcaster::publish_wire` returns the gossip error and
the only caller logs it:

```rust
if let Err(e) = tx.broadcast(action).await {
    tracing::warn!("broadcast failed: {e}");
}
```

(`stark-ui/src/collab.rs`, `flush_outbox`). The action is already in the local
mirror, so a *future* joiner gets it — existing members never do. The most likely
trigger is real: `MAX_MESSAGE_SIZE` is 256 KiB and a `CommitStroke` carries its
fitted control points, so a long stroke at a fine tolerance is the one payload
that can cross it, and it fails on exactly the peers who were watching it being
drawn.

**Gossip itself.** `iroh-gossip` is epidemic broadcast, not reliable delivery.
Nothing in the crate treats it otherwise.

### Why this one is worse here than it sounds

§12.6's convergence argument rests on every peer eventually seeing every action —
`merge_remote` is idempotent by id and order-insensitive, which makes *ordering*
free but says nothing about *loss*. A missing `CommitStroke` is a stroke that
exists on some canvases and not others, forever, with both sides believing they
are in sync. It is the divergence content-addressing was built to prevent,
arrived at from the other end.

### The fix

The repair is one request away, because the pieces are already here: every peer
mirrors the total-ordered log (`Mirror::actions`, a `BTreeMap<ActionId, Action>`),
and every peer already serves the catch-up ALPN. Add to `proto::Request` —
appending variants, which is the safe postcard change (§8):

- `Request::Ids` → the sorted `Vec<ActionId>`. 16 bytes each; 100k actions is
  1.6 MB, which is far cheaper than the snapshot and is only sent when
  reconciliation is triggered.
- `Request::Actions(Vec<ActionId>)` → just those actions.

Then reconcile with a gossip neighbour: on `Lagged`, on a broadcast failure, and
on a slow timer (minutes). Feed whatever comes back through
`Waitlist::accept` — the existing dedupe by `Mirror::insert` means over-sending
costs bandwidth and nothing else.

A per-actor high-water mark is **not** a sufficient digest, and it is worth
writing that down so the cheap version is not attempted: `ActionId` is
`(lamport, actor)` and lamport clocks jump when a peer observes others, so an
actor's ids are sparse and a high-water mark cannot see a hole in the middle. The
id list, or a rolling hash of it with the list as the fallback, is what actually
detects a gap.

Also: bump `MAX_MESSAGE_SIZE`, or split an oversized `CommitStroke` across the
blob path the way assets already travel. A ceiling that a legal document action
can cross is a ceiling that will be crossed.

**How you would know.** A test that drops a broadcast on the floor — easiest via
the `Waitlist`/`Mirror` pair directly, or by broadcasting an action on a topic one
peer has not yet subscribed to — then triggers reconciliation and asserts the two
mirrors hold the same id set. A second test that broadcasts a stroke over
`MAX_MESSAGE_SIZE` and asserts it still arrives.

---

## N2. A joining peer serves an empty document as if it were complete

`CollabSession::join` builds the mirror empty and binds the router — `CollabProto`
included — before it has anything to serve:

```rust
let mirror = Arc::new(Mutex::new(Mirror::from_file(&DocumentFile::new(Vec::new()))));
let bound = backend::bind(mirror.clone(), &opts).await?;   // collab ALPN is live here
let catchup = bound.dialer.open(ticket.addr.clone()).await?;
let mut sub = bound.gossip.subscribe(ticket.topic, vec![ticket.addr.id]).await?;
// … up to JOIN_TIMEOUT (10 s) waiting to meet the swarm …
let snapshot = catchup.request(request).await?;
// … only now:
*mirror.lock().expect("mirror poisoned") = Mirror::from_file(&file);
```

For that whole window, anyone dialling `stark/collab/5` on this peer gets a
well-formed, empty, silently-wrong session snapshot. Not an error — a document.

**Reachability is low today and that is not the argument.** A peer's ticket is
only published after `join` returns, so in practice nobody has its address yet.
But `tests/handoff.rs` (`a_newcomer_can_join_through_any_member_after_the_founder_leaves`)
exists precisely because joining through a non-founder is a supported path, and
the ticket is a `SessionTicket` anyone can mint from any member. The failure mode
is a joiner who loads an empty canvas and then rides the gossip tail, believing
it has the document.

**The fix**, in the shape CLAUDE.md asks for ("rule out a class rather than
enumerate its instances" / "a representation that cannot express the wrong
thing"): the type should not be able to say "empty" and "complete" with the same
value.

```rust
pub(crate) enum MirrorState {
    Joining,
    Ready(Mirror),
}
```

`proto::answer` returns an error for `Joining`, the requester surfaces it as
"that member is still joining", and picks another. The host path constructs
`Ready` immediately, so nothing about hosting changes. `Waitlist` already funnels
every mirror access through itself, so the `Option`-handling lands in one file.

**How you would know.** A test that binds a joiner against a host whose snapshot
response is delayed, dials the joiner's collab ALPN mid-join, and asserts an error
rather than an empty `DocumentFile`. Today that test would pass while describing
a bug.

---

## N3. Nothing checks the stamped origin against the action's author

`Stamped::origin` is self-declared — it has to be, since `iroh-gossip`'s `Message`
exposes only `delivered_from` ("This is not the same as the original author") and
does not sign payloads. The doc comment says as much and defers authentication to
§12.5. Fine.

What is not fine is that `origin` and `action.id.actor` are never compared,
despite both being identity:

- `origin` picks the asset source (`asset_sources`) and, for presence, *is* the
  attribution: `actor_from_endpoint_id(origin)`.
- `action.id.actor` decides undo scope (§12.3) and half the total order key.

**This is not authentication, it is consistency.** The check costs one comparison
and rules out two things at once. The adversarial one is a peer claiming another's
undo scope. The one that will actually happen is a client whose collaboration
identity has drifted from its endpoint key — and the frontend derives those in two
separate places today:

```rust
let actor = actor_from_endpoint_id(id.secret.public());   // → start_collaboration
…
let opts = NetOptions { secret: Some(id.secret), … };     // → the endpoint
```

(`stark-ui/src/collab.rs`, `share`). They agree because the same `id` feeds both.
Nothing enforces that they keep agreeing, and if they stop, actions are attributed
to an actor nobody can undo — with no error at either end.

**The fix.** In `recv_loop`, before `waitlist.accept`:

```rust
if actor_from_endpoint_id(origin) != action.id.actor {
    tracing::warn!(?origin, actor = ?action.id.actor, "action author does not match its sender");
    continue;
}
```

Dropping rather than accepting is right: an action whose author is wrong is one
whose undo scope is wrong, and applying it makes the document unfixable rather
than merely incomplete.

Worth considering alongside it: have `CollabSession` *derive* the identity the
engine should use, rather than the frontend computing it separately — the session
already knows the endpoint key, and `actor_id()` already exists. Making
`start_collaboration` take it from there closes the drift structurally instead of
checking for it.

**How you would know.** A unit test on the check itself, plus an integration test
that broadcasts a hand-built `Stamped` whose `origin` and `id.actor` disagree and
asserts the receiving mirror does not grow.

---

## N4. A ground can wait forever on two peers who have left

`asset_sources` is two peers wide:

```rust
fn asset_sources(origin: EndpointId, from: EndpointId) -> Vec<EndpointId> {
    let mut ids = vec![origin];
    if from != origin { ids.push(from); }
    ids
}
```

and `resolve_asset` gives a ground `attempts: None` — retry until the content
arrives or the session ends, which is the right call (§6.4: applying `SetSurface`
against `Flat` bakes a smooth deposit no later arrival un-bakes). The two
decisions combine badly. If the author and the forwarding neighbour both leave,
the `SetSurface` parks for the life of the session while `fetch_asset` dials two
dead endpoints on a widening backoff, forever.

**And the bytes are right there.** Every member seeded that ground into its own
blob store at join — `Mirror::seed_blobs` puts "every piece of content this peer
already holds" in, brushes and grounds alike, precisely so any peer can serve any
other. The resolver is the only thing that does not know it.

**The fix.** `Broadcaster` already holds `neighbors: Arc<Mutex<HashSet<EndpointId>>>`,
maintained by the receive loop. Give the resolver a handle to it and widen after
the first round fails: try `[origin, from]`, then the current neighbour set on
each subsequent round. Re-read it per round rather than snapshotting it, so a peer
that joins later becomes a source.

The same widening is what makes [N1](#n1-there-is-no-anti-entropy)'s
reconciliation pick a live partner, which is why the two are worth doing near each
other.

**How you would know.** Three peers; the third holds a ground the first two do
not. Broadcast a `SetSurface` from a peer that then shuts down, and assert the
remaining peer resolves the ground from the third rather than parking. This is
also the test that is very hard to write without [N9](#n9-dialer-is-the-missing-seam).

---

## N5. A failed join leaks its endpoint

`CollabSession::join` has five `?` after `backend::bind`, and neither `Bound` nor
`Shutdown` implements `Drop`:

```rust
let bound = backend::bind(mirror.clone(), &opts).await?;
let catchup = bound.dialer.open(ticket.addr.clone()).await?;   // ← wrong version, host gone
let mut sub = bound.gossip.subscribe(…).await?;
…
let snapshot = catchup.request(request).await?;                // ← snapshot over MAX_RESPONSE
let file = DocumentFile::from_bytes(&snapshot)?;               // ← decode failure
let ticket_addr = bound.dialer.ticket_addr(&opts).await?;
```

Each of those drops `Bound` without ever calling `Shutdown::run`, leaving a live
iroh endpoint holding a relay connection, a spawned gossip actor, a blob store
actor and a router — none of them closed.

**Where it bites.** The failures listed above are the *expected* ones: a version
mismatch is the designed behaviour when two builds meet (the ALPN carries the
version so they fail to meet), and a user whose join fails reloads the page and
tries again. On the web that is a leak per attempt, in a tab with a hard ceiling.
`CollabSession::host` has the same shape, with two `?` after `bind`.

**The fix.** `impl Drop for Bound` that spawns `shutdown.run()`, or — cleaner given
that `run` is async and `Drop` is not — move the fallible tail of `host`/`join`
into an inner function and have the outer one close the endpoint on `Err`:

```rust
pub async fn join(ticket: &SessionTicket, opts: NetOptions) -> Result<Joined> {
    let bound = backend::bind(…).await?;
    match join_inner(&bound, ticket, &opts).await {
        Ok(joined) => Ok(joined),
        Err(e) => { bound.shutdown.run().await; Err(e) }
    }
}
```

**How you would know.** Join against a ticket pointing at a closed endpoint, twice,
and assert the process holds no more bound sockets after than before
(`Endpoint::bound_sockets` on a native test, or simply that a subsequent bind on a
fixed port succeeds).

---

## N6. The snapshot clones the whole log under the receive loop's lock

`Mirror::snapshot` says the lock is cheap:

> Every clone here is a refcount bump or a `BTreeMap` walk, so the caller's lock
> covers the log and nothing else

and `proto::snapshot_bytes` repeats it:

> the only real work the lock covers is the log — and a joiner arriving
> mid-session does not stall this peer's receive loop for the size of its own
> brush library.

Both are true about the *assets*, which are `Bytes` and cost a refcount. Neither
is true about the log, and the log is the part that grows without bound:

```rust
actions: self.actions.values().cloned().collect(),
```

That is a deep clone of every `Action` — each `CommitStroke` dragging its fitted
control-point vector along — into a fresh `Vec`, under the mutex that the receive
loop takes for *every arriving action* (`Waitlist::release` → `Mirror::insert`) and
that every broadcast takes for its transfer-hash lookup. A joiner arriving into a
long session stalls the host's live painting for the duration.

**This is the one place in the workspace that did not get the `DocState`
treatment.** CLAUDE.md names it as a founding consequence:

> **`DocState` is cheap to clone** — persistent (`rpds`) maps of `Arc<GpuTile>`
> handles, never pixels.

`stark-core` already depends on `rpds`; `stark-net` does not. Holding
`Mirror::actions` in an `rpds::RedBlackTreeMap<ActionId, Action>` (ordered, which
is what `BTreeMap` is here for) makes `snapshot()` an O(1) structural clone and
the lock genuinely as cheap as the comment claims. `insert` stays O(log n) with
path copying, which is nothing against the per-action GPU work already happening.

Two smaller allocations on the same path, worth taking while in there:

- `Waitlist::release` clones each action to insert it and then sends the original.
  Have `Mirror::insert` hand the action back on a duplicate instead.
- `Broadcaster::broadcast` clones the whole action to mirror it before encoding.
  A borrowed `StampedRef<'a>` / `WireRef<'a>` encodes byte-identically under
  postcard, so the encode can take `&Action` and the mirror can take the original.

**How you would know.** A criterion bench over `Mirror::snapshot` at 1k / 10k /
100k actions, before and after. Correctness is already pinned by
`tests/handoff.rs` and `tests/sync.rs`, which replay a real snapshot into a real
engine — so a change that loses or reorders an action fails rather than getting
re-blessed.

---

## N7. Every join copies every asset twice and re-encodes the document

`Snapshot::into_file` materializes owned payloads because `DocumentFile` owns
`Vec<u8>`:

```rust
file.assets = self.assets.into_iter().map(|(id, b)| (id, b.to_vec())).collect();
file.surfaces = self.surfaces.into_iter().map(|(id, b)| (SurfaceId::Image(id), b.to_vec())).collect();
```

and `snapshot_bytes` then postcard-encodes the result into a third buffer. With
the bundled grounds — 2.0 and 2.8 MB of canonicalized weave, per
`stark-ui/src/collab.rs` — that is ~5 MB copied twice and encoded once, **per
joiner**, on a peer that is also painting.

The `resolvable` mechanism already removes the common case of this, which is why
it is small rather than medium: a joiner running the same build asks for
`SnapshotWithout` and the grounds never leave. It bites for imported brushes and
imported grounds, which are exactly the content a real session accumulates.

**The fix.** Cache the encoded bytes on the mirror, keyed by (log revision,
`have` set). Most joins in a session share an answer — every peer on the same
build sends the same `resolvable` list — so the second joiner and after pay
nothing. A revision counter bumped by `insert`/`insert_content` is the whole
invalidation.

Making `DocumentFile` hold `Bytes` would remove the copies outright, but that is a
`stark-core` change touching the save format's in-memory shape, and the cache gets
most of the win without it.

**How you would know.** Bench `snapshot_bytes` with ~5 MB of assets, once cold and
twice warm. `tests/sync.rs`'s
`a_promised_ground_is_left_out_of_the_snapshot_and_still_replays` already pins the
`have`-set behaviour the key depends on.

---

## N8. One send task per dispatch makes wire order accidental

`flush_outbox` spawns a fresh task per dispatched command:

```rust
spawn_forever(async move {
    for action in actions {
        if let Err(e) = tx.broadcast(action).await { … }
    }
});
```

Two dispatches in the same frame produce two tasks racing on the same
`GossipSender`, so actions can reach the wire out of order. The CRDT absorbs it —
that is what §12.6's order-insensitivity is for — but every inversion buys a
timeline resync on *every* receiver, which replays the actions that landed after
it. Paying for that on the interactive path, to save a queue, is the wrong trade.

**The fix.** Give `Broadcaster` an `mpsc` and one drain task. `broadcast` becomes
a non-async `send` that cannot fail for ordering reasons, the per-dispatch
`spawn_forever` disappears from the frontend, and the drain task is the natural
home for [N1](#n1-there-is-no-anti-entropy)'s retry — a failed send has somewhere
to be retried *from*, which it currently does not.

It also removes a subtler coupling: `flush_outbox` reads the session out of a
signal on every dispatch to get a `Broadcaster`, purely because there is no
standing consumer to hand the actions to.

**How you would know.** Broadcast a burst of actions from two concurrent tasks and
assert the receiving mirror observed them in ascending `ActionId` order. Today
that test fails intermittently.

---

## N9. `Dialer` is the missing seam

Every test in this crate except `waitlist.rs`'s needs two real iroh endpoints.
That is why `waitlist.rs` is the only module with unit tests, and its own doc
comment gives the reason:

> These are here rather than in an integration test because the properties worth
> pinning are about *this* type … and reaching them through two iroh endpoints
> would test the swarm instead, more slowly and less exactly.

The same argument applies to the content-acquisition policy, and nothing there
can currently be reached. `Dialer` is a concrete struct wrapping `Endpoint` +
`iroh_blobs::api::Store`, used by `resolve_asset` and `fetch_asset` through four
methods: `local_id`, `add_blob`, `fetch_blob`, `ensure_direct`. A trait over those
four, with a fake that fails N times, disappears, or resolves late, makes these
testable for the first time:

- the brush/ground asymmetry — `BRUSH_ATTEMPTS` gives up and releases to the round
  tip, a ground never does;
- `LOCAL_GRACE` and `ResolveLocally` — the frontend answers, the frontend answers
  *late*, the frontend never answers;
- the widening backoff and its `ASSET_RETRY_MAX_DELAY` cap;
- `is_live` cutting an unbounded retry short;
- [N4](#n4-a-ground-can-wait-forever-on-two-peers-who-have-left)'s source widening,
  which is otherwise a three-endpoint test with a shutdown in the middle.

None of those is reachable deterministically through a real swarm, which is why
none of them is tested today. `tests/sync.rs` covers the happy paths well — seven
integration tests, including the two `resolvable` paths — and should stay exactly
as it is; this is about the branches underneath them.

**The fix.** A `pub(crate) trait ContentSource` with those four methods, `Dialer`
as its production impl, and the resolver generic over it (or over `Arc<dyn …>`,
given the resolver is already spawned per fetch and one vtable dispatch per blob
is free).

**How you would know.** The list above, as tests. The measure of this finding is
that they exist.

---

## N10. The session spawns tasks it cannot stop

`CollabSession::shutdown` shuts the router and closes the endpoint. It cancels
nothing it spawned, and the crate spawns a lot: `recv_loop`, a `resolve_asset` per
missing asset, `ensure_direct` per gossip neighbour, `announce` per attached
channel, an `add_blob` per import, and on wasm the JSEP worker.

What stands in for cancellation is liveness inferred from a channel:

```rust
/// Whether the engine is still listening. A resolver that may retry
/// indefinitely polls this so it cannot outlive the session.
pub fn is_live(&self) -> bool {
    !self.events.is_closed()
}
```

That is "does the UI still hold the `Events` receiver", which is a frontend
convention, not a session fact. It works today because `collab::leave` cancels the
pump task that owns `Events` — but the coupling is invisible from either side, and
a frontend that keeps `Events` alive while dropping the session leaves ground
resolvers dialling forever, since a ground's retry is uncapped by design.

**The fix.** A cancellation token (`tokio_util::sync::CancellationToken`, or an
`Arc<AtomicBool>` plus a `Notify` to avoid the dependency) owned by `CollabSession`,
cloned into every spawned task, cancelled by `shutdown`. `fetch_asset`'s sleep
becomes a `select!` on it, so shutdown is immediate rather than up to
`ASSET_RETRY_MAX_DELAY` late. `Waitlist::is_live` then means what its doc comment
says.

Worth pairing with a related tidy: **`session.rs` is three modules wearing one
hat** at 846 lines — the public API (`CollabSession`, `Broadcaster`, `NetOptions`,
`Joined`, `Events`, `PresenceQuota`, `LinkKind`), the gossip receive loop, and the
content-acquisition policy (`resolve_asset`, `fetch_asset`, `asset_sources`,
`hash_or_warn`, `BRUSH_ATTEMPTS`, `LOCAL_GRACE`, `ASSET_RETRY_*`). That third
piece is the other half of `waitlist.rs`: the waitlist decides *whether* to fetch,
the resolver decides *how hard*, and the brush-vs-ground asymmetry is currently
explained at length in both files. Moving the resolver next to the waitlist puts
the whole "what does an action wait for, and for how long" policy behind the doc
comment that already explains it — and is the natural moment to introduce
[N9](#n9-dialer-is-the-missing-seam)'s trait.

**How you would know.** A test that shuts a session down while a ground fetch is
mid-backoff and asserts the task is gone — observable as the blob store's client
handle count, or simply as `shutdown()` completing and no further dial attempts
being logged.

---

## N11. Smaller things

### `fetch_blob` opens a fresh connection per blob

```rust
let conn = self.endpoint.connect(self.dial_addr(EndpointAddr::new(provider)), iroh_blobs::ALPN).await?;
self.blobs.remote().fetch(conn, hash).await?;
```

The connection is dropped at the end of the call, so `fetch_asset`'s retry loop
re-dials on every round against every source — up to `BRUSH_ATTEMPTS × sources`
QUIC handshakes for one 20 KB PNG, and unbounded for a ground. Hold one connection
per provider for the life of a resolver; the retry is *for* the case where the
provider is slow, which is exactly when re-handshaking hurts most.

### `MAX_MESSAGE_SIZE` is a ceiling a legal action can cross

Covered under [N1](#n1-there-is-no-anti-entropy) because the consequence is
permanent loss, but the ceiling itself deserves a number chosen against the
fitter's actual worst case rather than against "any plausible single action".
`benches/path.rs`'s longest recorded stroke is the place to get it.

### `NetError` is stringly typed at the edges

`Other(String)` and `Ticket(String)` lose the source, and `proto::request` funnels
a perfectly good `ClosedStream` into `Other`:

```rust
send.finish().map_err(|e| crate::NetError::Other(e.to_string()))?;
```

`Ticket` in particular wants to be an enum — missing prefix, bad base32, version
mismatch, decode failure are four distinct things the UI shows verbatim to a user
who pasted a link.

### `Snapshot::without` is quadratic

`have.contains(id)` inside a `retain`, over two collections. Tens of ids today, so
it costs nothing; it is a `HashSet` away from never mattering.

### Nothing exercises the configuration that ships

Every test uses `NetOptions::local()` — `presets::Minimal`, no relays, loopback
addresses. The product runs `presets::N0` with relay-based addressing, and the one
test that touches a relay (`transport/direct.rs`'s migration test) builds its own
endpoints rather than going through `backend::bind`. `ONLINE_TIMEOUT`, the
`endpoint.online()` wait, and the relay-URL ticket path have no coverage at all.
`iroh`'s `test-utils` relay server is already a dev-dependency, so the fixture
exists — this is a matter of pointing `bind` at it.

### The comments are the reason this review was cheap

Same note as [C8](CORE_CLEANUP.md#the-documentation), and it applies here more
strongly: `waitlist.rs`'s module doc, the `LOCAL_GRACE` rationale, the
`resolve_asset` brush-vs-ground argument and `transport/direct.rs`'s "never
announce before attach" are all better than any design doc that could have been
written for them. **Do not cut them.**

The drift risk is live in three places, all of them findings above:
`Mirror::snapshot`'s "the caller's lock covers the log and nothing else"
([N6](#n6-the-snapshot-clones-the-whole-log-under-the-receive-loops-lock)), the
`Lagged` arm's "peers converge again on the next snapshot fetch"
([N1](#n1-there-is-no-anti-entropy)), and `is_live`'s "cannot outlive the session"
([N10](#n10-the-session-spawns-tasks-it-cannot-stop)). Each asserts a mechanism
that does not exist. The antidote is the one the workspace already uses — turn the
claim into an assertion — and for the first of those, a bench is the assertion.
