# Collaboration and per-client state

The CRDT over the action log, the iroh transport, owned selections, and the presence roster — §12, §17.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 12. Collaboration (peer-to-peer)

Multi-user editing over `iroh` — **implemented**, exactly as the additive layer
this section always planned: `ReplicatedTimeline` in `stark-engine` (merge
semantics), `stark-net` (the wire), a share/join dialog in `stark-ui`. Engine and
GPU code were untouched. Three properties already in place made it tractable:
the document is a **log of id-tagged deterministic actions** (§4); replay is
**bit-for-bit deterministic** (§6.5, §9); the timeline is behind a **trait** (§5).

### 12.1 Convergence model — a CRDT over the action log

The document is a grow-only set of actions with a **total order** given by
`ActionId = (lamport, actor)`. The canonical `DocState` is the deterministic
replay of all actions in that order. Two peers that have seen the same set
compute identical pixels — **strong eventual consistency** — because ordering is
total and replay deterministic. This is the well-trodden op-based CRDT /
replicated log pattern, and it fits almost for free since replay is already how
every pixel is derived.

- **Lamport clocks** give causal-consistent ordering; ties break on `actor`.
  Every merge advances the local clock past the remote action, so an action
  always orders after everything its author had seen — which also guarantees an
  `Undo` orders after its target.
- **Commutativity isn't required**, only a deterministic order — paint is not
  commutative, and a fixed order captures exactly the "whoever's stroke is
  ordered later wins the overlap" intuition.
- **`Undo` is resolved at the timeline layer, not in `apply`** (§5.4): one
  descending pass over the total order computes which actions are undone, and the
  **effective sequence** is what the `history` cache materializes. Duplicates
  (gossip redelivery) are rejected by id — merging is idempotent.

### 12.2 Inserting a late action (the one real cost)

When a remote action arrives with an id *earlier* than actions already applied
locally (or an `Undo` changes effectiveness mid-log), `ReplicatedTimeline` diffs
the new effective sequence against the materialized one, pops `history` back to
the first divergence, and replays forward. The untouched prefix keeps its
snapshots (and their tiles' `Arc`s); dense snapshot retention keeps the pops
shallow. For an undo the rewind rarely happens at all — see §12.6.

### 12.3 Undo under collaboration

This is why `ActionKind::Undo(target)` exists: in a shared log, undo must be *my*
action others can observe and order, and "undo my last stroke" must skip peers'
intervening strokes. The engine asks the timeline first
(`undo_as_action`/`redo_as_action`) and falls back to navigation undo only when
they return `None` (solo).

- **Undo targets** my most recent *effective* ordinary (non-`Undo`) action.
- **Redo** emits an `Undo` of my most recent effective `Undo` whose target is an
  ordinary action still undone — but only if that `Undo` is newer than my newest
  effective ordinary action, so a fresh edit clears the redo stack, matching solo
  expectations. Chains (Z Z Y Y) walk correctly because each redo suppresses
  exactly one undo.
- **Redo-at-top:** a revived action re-materializes at the *reviving redo's* slot
  rather than its original position (`revival_keys` in `timeline.rs`: the
  effective sequence orders by id, except a revived action takes its latest
  effective revival's id as its key). Deliberately "good, not perfect": a redone
  stroke lands *over* work that happened while it was undone rather than back
  underneath it — and in exchange redo is a plain append for every caught-up
  client instead of a mid-log insert replaying everything after the original
  slot. Peers converge because the key is a pure function of the shared log; solo
  sessions cannot tell the difference.
- A file saved mid-session carries the **full log**; a solo load replays the
  effective sequence (undone work flattens away), while a joining peer gets the
  full log so later redos still resolve. **Leaving performs that same flattening
  in place, and without a replay** (`Timeline::unshare`): the effective sequence
  is what the timeline has been materializing all along, so the linear history it
  hands back on the way out is the one already on screen. The scrubber returns
  with it (§18.2.4), and so does undo by navigation — with no session left to own
  it, a peer's stroke is this document's like any other.

### 12.4 Transport — `stark-net` over iroh

Core stays **network-agnostic**; `stark-net` adapts iroh 1.0 to the engine's
hooks (`start_collaboration` / `join_collaboration` / `merge_remote` /
`take_outbox` / `take_presence` / `merge_presence`):

- **Identity:** an iroh `EndpointId` maps to the `ActorId` — its first 8 bytes
  (`actor_from_endpoint_id`). No central server. At share time the host's
  `ActorId::SOLO` actions are rewritten to its real actor — before any peer has
  seen them — so pre-share strokes stay undoable.
- **Live edits:** `iroh-gossip` broadcasts each newly committed `Action` (small — a
  fitted path, not pixels) on the session's random `TopicId`. The message ceiling is
  raised (256 KiB) so long strokes fit. A broadcast carries **no schema**, unlike a
  saved file (§8): both ends encode against their own, and the ALPN below is what makes
  a mismatch fail to meet rather than decode wrong — see `stark-net`'s `codec` for why a
  flooded channel cannot afford to carry one.
- **Join / catch-up:** a joining peer connects over the `stark/collab/1` ALPN and
  requests a **snapshot** — the save-format container (§8), assets and substrates
  bundled — then rides the gossip tail. It joins the topic *before* fetching, so
  the snapshot/gossip overlap covers the seam (dedup by id). Every member serves
  snapshots from a session **mirror** (log + assets + substrates, CPU-side), so
  sessions survive the original sharer leaving, and any member can mint a **ticket**
  (`stark…`: a version byte, then a few members' `EndpointAddr`s + the topic,
  deflated and spelled in base64url — a link is pasted by a person, and carbonite's
  columns compress, so the two together halve what one costs to carry). A ticket
  names its minter first and then up to a handful of members the minter was
  connected to when it minted — the joiner tries them in order for the
  snapshot and hands *all* of them to gossip as bootstrap candidates, so a link
  also survives its **minter** leaving between the minting and the pasting. One
  live name in it is enough.
- **What the joiner already has** is left out of that bundle. Content-addressing
  makes provenance irrelevant to correctness, which is what it is for — but it
  also means the app cannot tell that a substrate it is being sent is the one it
  ships with. So the joiner *says*: it sends the ids of its bundled assets and the
  host omits them. The app's own substrates canonicalize to 2.0 and 2.8 MB — against
  a log that is a handful of fitted paths — and they were moving into installs that
  already had them.

  The list is a **promise**, not an inventory — "I can get these", not "I have
  these loaded" — and it is knowable ahead of the bytes only because the frontend
  hashes its bundle at build time, which is what `stark-assetid` exists to make
  possible without linking a renderer. The promise is called in twice: `owed`
  comes back from the join and must be installed **before the log is replayed**
  (a `SetSubstrate` whose height map is not registered when its strokes replay
  deposits them through the flat stand-in, and those pixels are stored, §6.4);
  and for the rest of the session a need whose id was promised raises a
  local-resolution request instead of a dial.

  Breaking the promise is safe by construction. The log still names the content,
  so the ordinary blob fetch pulls it off a peer exactly as it would have — a
  frontend that cannot deliver, or does not implement any of this, loses a grace
  period and nothing else. Being wrong costs a transfer, not a picture.
- **Assets:** an action referencing content the receiver lacks fetches those
  bytes over the blobs ALPN from its author, falling back to the peer that
  delivered it, and the action is **parked** on a waitlist until they arrive — so
  the content reaches the engine before the action that needs it. Parked, not
  awaited: what an action needs ordering against is the content it names and
  nothing else, since `merge_remote` is idempotent by id and order-insensitive,
  and an action landing behind newer ones makes the timeline resync (§12.6).
  Waiting inside the receive loop instead stalls every *other* peer's actions,
  every presence frame and the neighbor bookkeeping behind one unreachable blob.
  The waitlist is also what de-duplicates the fetch, since a live gesture's head
  and the commit it becomes name the same content.

  Two kinds ride this path, and the transport says which — the referencing action
  is the only thing that knows, and the two decode differently at the far end.
  They differ in exactly one thing beyond that, and it is how long the fetch
  tries:
  - a **brush shape** an unknown `AssetId` names. A miss degrades to the round
    tip and the stroke is still visibly a stroke, so after a few rounds the fetch
    gives up and lets the action through.
  - a **canvas substrate** a `SetSubstrate` names (§6.4). A miss is not cosmetic: the
    deposition tooth reads the substrate, an absent one falls back to the flat
    stand-in, and the resulting deposit is *stored* — so a peer that applied the
    switch before the height map landed diverged permanently, with nothing on
    either screen to say so. That was the bug that made substrates content-addressed
    in the first place; they were previously labels, and a label cannot be
    fetched. So a substrate is never given up on: the action waits indefinitely, and
    the strokes that merged ahead of it are replayed against the real substrate when
    it lands. Parking is what makes that affordable — an unbounded wait costs
    nothing when nothing waits behind it.

  Content the receiver said it could produce itself is asked of the frontend
  before any of this: one request, a grace period, and only then a dial. What
  comes back arrives through the same `add_content` a local import uses, so a
  locally-resolved substrate is not a different kind of content — only a different
  way of getting hold of it.

  A mid-session import seeds the mirror at import time — before the action goes
  out, since the broadcast attaches a transfer hash looked up from it — and
  releases any remote action parked on that same content, which a local import
  satisfies as well as a fetch would. A *presence* stroke head referencing an
  unknown shape starts the fetch without parking anything, so a peer's live
  preview upgrades from the round-tip fallback without waiting for the commit.
- **Browser:** iroh runs in wasm over its relay (WebSocket) transport, plus a
  vendored **WebRTC custom transport** on the same endpoint, so the Dioxus UI
  uses the same code path the native loopback tests exercise. The UI glue is two
  pumps: `dispatch` drains the engine outbox into `CollabSession::broadcast`, and
  a spawned task feeds `RemoteEvent`s into
  `merge_remote`/`import_brush`/`accept_substrate` and repaints. **The page URL is the invitation:** a live session's ticket rides the
  URL fragment (`…#stark…`, via `replaceState`; cleared on leave), and opening a
  link with one auto-joins on load — the fragment never leaves the browser, so no
  server sees the ticket. The fragment is re-minted on the link-poll cadence
  (rewritten only when its text changes), so the members a copied invitation
  names are the ones reachable *now*, not the ones of an hour ago.
- **Presence** (cursors, selected layer, names, live strokes) — see §17.
  Broadcast as `Wire::Presence`, **never historized, never mirrored and never
  snapshotted**.

### 12.5 What is deliberately deferred

Authentication/permissions (anyone with a ticket can write), large-session
scaling (gossip fan-out, log compaction/GC of fully-superseded tiles), recovery
from gossip loss (a lagged receiver warns; a re-join resnapshots), and
offline-merge UX. None perturb the convergence model; they layer on top of it.
See also §17.13.

### 12.6 Commutation fast paths — undo and late merges without replay

§12.2's rewind is the honest fallback, but most concurrent edits do not
interleave: your undo usually sits under a pile of *other people's* strokes on
other layers or other parts of the canvas. When the changed action **commutes**
with everything materialized after it, the reordered replay would recompute
exactly the pixels already on screen — so the timeline does not run it.

- **Footprints** (`document/footprint.rs`). Every `ActionKind` declares the
  resources its `apply` reads and writes: a layer's paint within a tile rect (a
  stroke's padded control-point bbox — the B-spline stays in its hull; a
  transform claims the whole layer), layer existence, one per-layer property
  (`Prop::Blend`, `Prop::Clip`, … — separate variants, because actions written by
  different commands must commute with each other), the stack order (one coarse
  resource — concurrent reorders genuinely do not commute, and nesting rides on
  it unchanged), the author's selection, the substrate, the substrate color. Two actions
  commute when no write overlaps the other's read or write set. This encodes the
  intuitive cases structurally: strokes on different layers commute; same-layer
  strokes commute when they share no tile (tile granularity is honest, not lazy:
  removal swaps whole tile handles, so strokes sharing a tile genuinely conflict
  even if their texels do not touch); a rename commutes with everything but its
  own layer's rename/removal; a selection edit blocks only its *own author's*
  later strokes. `Footprint` is the action's **`Centralizer`**. False conflicts
  only cost the fast path; a missed one would silently diverge peers.
- **`Action::inverse`** (`document/patch.rs`). Removes an action's effect from a
  state by restoring what its footprint wrote from the state it was originally
  applied to — the replaced tile *handles* (`Arc` bumps; tiles are CoW, so
  identity is change detection), the prior prop value, the prior selection.
  Nothing is stored ahead of time and nothing re-renders. The restore is bounded
  to the footprint, and must be — the two states are not adjacent, and the
  suffix's own work sits outside the footprint on the same layer.
  `PatchOp::Structure(Vec<(LayerId, Option<LayerId>)>)` records the flattened
  order *plus* each layer's carrier, and `PatchOp::Present { .. }` records a
  `LayerSite` — a carrier id and a position in that stack — rather than a flat
  index or an index path, because ids are stable under everything below them
  moving. Restoring only the *shape*, and taking each layer's current record from
  the state being rebuilt, is what lets a commuting action that painted in the
  gap keep its work.
- **`History::try_remove_action_with`** (upstream `history`). Servicing an undo,
  it shifts the target past the run of later actions it commutes with — O(log n)
  cached-state fixes via `inverse`, no re-render at all when the whole suffix
  commutes — and replays only what sits past the first conflict. Degradation is
  graceful.

Inserts stay simple on purpose: a fresh commit, a caught-up remote arrival and a
redo are all plain appends; the rare concurrent arrival landing mid-sequence
takes §12.2's rewind — shallow by construction, because a concurrent action's
Lamport slot is near the top of the stack.

Convergence is untouched: disjoint footprints mean the shifted materialization
computes bit-identical pixels to the canonical replay — **provided every `apply`
touches only what its footprint declares, which is an invariant `action.rs`
changes must maintain.** `TimelineStats` (`Engine::timeline_stats`) counts fast
removes vs. rebuilds, because pixels *cannot* show which path ran;
`tests/commute.rs` asserts both the stats and exact pixel equality against a
fresh peer's canonical materialization of the same log.


## 17. Per-client state — owned document state and presence

The third class of engine state: state **owned by one client, read by all**. It
covers the selected layer, the selection region, and live in-progress gestures.

### 17.1 What was actually broken

Three separate things, only one a missing feature:

1. **The selection was shared when it must not be.** `DocState::selection` was a
   single mask, edited by logged actions and read by `CommitStroke::apply` from
   the state it folds over. In a shared session that is a genuine defect: if I
   lasso a region, *your* next stroke is clipped to my lasso — on your screen
   too, because the mask is in the replayed state.
2. **Live strokes were invisible until commit.** A peer's stroke arrived as one
   `CommitStroke` on release, so a thirty-second stroke appeared as a single
   event thirty seconds late.
3. **The selected layer was fine, and is not the problem it looks like.**
   `Session::active_layer` was per-client already, and `StrokeRecord::layer`
   closes each stroke over its target, so two clients painting different layers
   already converged. What was missing was only *visibility* — nobody could see
   who was working where — plus one latent id-minting bug (§17.9).

### 17.2 The classification

§4 states one rule and derives everything from it. Adding an owner axis gives a
2×2, and every piece of per-client state falls into exactly one cell:

|  | **private** — this client only | **published** — every client reads |
|---|---|---|
| **not logged** | **view state** — pan/zoom, tool, brush, media params, environment, viewport | **presence** — cursor, selected layer, live gesture, display name |
| **logged** | *(empty — the log is shared by construction)* | **document state** — either **shared** (layers, paint, substrate, substrate color) or **owned** (the selection) |

The discriminator is the one this codebase already uses: **does replay need it to
reproduce pixels?**

- The **selection** does. A stroke's pixels depend on the mask it was drawn
  through, so a peer replaying my stroke must reconstruct *my* mask at *that
  point* in the log. It is document state; it just has an owner.
- The **selected layer** does not. It is already closed over by
  `StrokeRecord::layer`. Logging it would turn every click in the layers panel
  into an undo step, for no reproducible consequence.
- A **live gesture** does not. It is by definition the thing that has not
  committed; when it does, the `Action` is authoritative and the live copy is
  discarded.

So the "third class" is really two mechanisms, and the split is not arbitrary —
it is forced by the rule already governing §4.

**On "a map from clients inside the shared state":** exactly right for the
selection, exactly wrong for the rest. For the selection it is the *only*
formulation that keeps §12's convergence argument intact. For the cursor, the
selected layer and the in-flight path, a map inside `DocState` would be a
category error with three concrete costs: every hover would enter the undo
history and the save file; the timeline would have to re-materialize on
pointer-rate traffic (`resync` pops and replays from the first divergence); and a
lost packet would become a convergence failure instead of a dropped frame.
Presence must be allowed to be lossy, and nothing in `DocState` is allowed to be.
Hence: **one map inside the shared state, one roster outside it.**

### 17.3 Owned document state — the selection, keyed by actor

`DocState.selections: HashTrieMap<ActorId, Selection>` (§5.1). Absent = everything
selected, which is the free representation `Selection::everything()` already is —
so an actor who never selects costs nothing, and a solo document has exactly one
entry. `HashTrieMap` for the same reason as everything else in `DocState`:
cloning a version stays a handful of `Arc` bumps, and masks are tile maps that
structurally share across versions exactly like paint.

**The owner is derived, never declared.** `Action::apply` already receives
`&self`, so it has `self.id.actor`:

```rust
ActionKind::Select(op)      => state.select(self.id.actor, ctx, op),
ActionKind::InvertSelection => state.invert_selection(self.id.actor, ctx),
ActionKind::CommitStroke(rec) => /* ... */ selection: state.selection_of(self.id.actor),
```

This is the load-bearing detail. Ownership is read off the action *id*, which is
already the total-order key and already authenticated exactly as much as the rest
of the log. There is no `owner:` field in any payload, so "only its owner may
change it" is not a permission check that could be forgotten at a call site — it
is structurally impossible to express the other thing. A peer can no more write
my selection than it can author an action under my id, and those are the same
statement.

Consequences that fall out with no further work:

- **Undo is already scoped.** `undo_target` targets *my* most recent effective
  action (§12.3), so undoing a selection is mine to do and reaches only mine.
- **Ordering is already right.** A `Select` and a `CommitStroke` by the same
  actor are ordered by their Lamport ids, and the stroke reads the state folded
  up to its own position — so a late-arriving remote `Select` re-orders through
  the existing `resync` path and everyone still converges.
- **The wire and file formats did not change.** `ActionKind::Select(op)` is
  byte-identical; only its interpretation moved from "the selection" to "the
  author's selection". Every existing document has all its `Select` actions under
  `ActorId::SOLO` and therefore loads and replays to the same pixels.
- **`start_collaboration`'s SOLO→actor rewrite already covers it** — it rewrites
  ids, the map is keyed off ids, and the rebuild is a `resync`.

**Rendering the outline.** The selection overlay pass (§6.8) takes a short list
rather than one mask: the local actor's selection in the marching-ants style,
and — **only if this client asks for it** — the selection of every actor
*currently present*, drawn as a flat line in that peer's color so the two never
read as the same thing. `ViewCommand::SetShowPeerSelections` gates the second
group and it is **off by default**: knowing which region someone else is working
inside is occasionally useful, but a second contour over the artwork is a cost
paid on every frame you look at it, and most of the time the answer to "what is
that line?" should be "the one I drew". It is view state, so each client decides
for itself and nothing about the choice is logged or sent.

That is also where the control lives: not a painting control and not part of the
document, so it sits in the **Settings dialog** (§11) rather than the Select
panel — a standing per-client preference, set once and left alone. It is listed
there whether or not a session is live: a tool panel earns the opposite rule
(§6.8), but a settings dialog is read as the map of what is configurable, so the
row stays put and says in its own text that it takes effect while sharing.

Note the asymmetry underneath, which is deliberate: `DocState` holds a selection
for every actor that ever selected, because replay needs them; the roster says
which of those are here; the setting says whether to draw them. **The log decides
what exists, presence decides what could be shown, and the client decides whether
it is.**

### 17.4 Presence — the ephemeral roster

Everything else per-client lives outside the timeline, as a sibling of `Session`,
because that is precisely what it is: other people's sessions.

```rust
/// What one participant is doing right now. Never logged, never saved, never
/// referenced by anything in the action log.
pub struct Peer {
    pub actor: ActorId,
    pub name: String,
    /// Derived from `actor` by hash, so every client agrees with no negotiation
    /// and no allocation protocol.
    pub color: [f32; 3],
    pub active_layer: LayerId,
    pub cursor: Option<Vec2>,       // canvas space
    pub gesture: Option<LiveGesture>,
    seq: u64,        // monotonic per actor; a frame that does not advance it is stale
    last_seen: f64,
}

pub enum LiveGesture {
    /// The in-flight stroke *as the record it will commit as* — the same type
    /// `Session::preview_record()` produces and `CommitStroke` carries.
    Stroke(StrokeRecord),
    /// The marquee or lasso being dragged (§6.8).
    Selection(SelectionOp),
}
```

`LiveGesture::Stroke(StrokeRecord)` is the economy that makes the whole feature
small: **a live gesture is a preview of the action it will become**, in the same
type, rendered through the same entry point (`StrokeRenderer::render_range`) the
commit will use. There is no second stroke representation to keep in step, and
therefore no second way for live and committed pixels to disagree.

**The local client is just another peer.** `Session` is the local peer's record
plus the private view state, and exposes one projection — symmetric with
`Engine::observe()`, which is the UI-facing one. The engine holds one roster
containing everybody (`peers: BTreeMap<ActorId, Peer>`, including the local
actor), so the local in-flight stroke and remote in-flight strokes are the *same*
mechanism, folded in the same order, on every client. That uniformity is what
makes every client see the same canvas mid-stroke, and it deletes the special
case rather than adding one.

**Lifecycle.** A peer publishes at least every 2 s even when idle; a peer unheard
from for 6 s is dropped. An explicit `Leave` frame on a graceful exit makes the
common case instant. **A gesture is cleared by its own commit**: any `Action`
merged from actor *A* clears *A*'s `LiveGesture` — a gesture is a thing that
becomes an action, so the action's arrival is the end-of-gesture signal, with no
id to correlate and no window in which both are drawn. Cancels send an explicit
`GestureEnd`, and a gesture with no update for 2 s is dropped anyway, so a peer
that crashes mid-stroke does not leave a smear.

**Loss is a design property, not a failure mode.** Nothing in the action log ever
references presence. That invariant buys everything else: presence may be
dropped, coalesced, reordered or arbitrarily delayed without any effect on
convergence. The worst outcome of total presence loss is that a session looks
like it did before the feature — strokes appear on commit. So the transport is
free to shed presence first under congestion, and the receiver free to drop
frames it cannot use.

### 17.5 Live gestures on the wire

Sending the whole fitted path on every pointer move is O(n²) bytes over a stroke.
The fitter already solves this: `PathFitter` **freezes** a prefix of control
points that is final and never revised (§6.2). The wire form is the same
partition:

```rust
pub struct PeerFrame {
    pub seq: u64,
    pub name: Option<String>,          // sent on change and on resync
    pub active_layer: LayerId,
    pub cursor: Option<Vec2>,
    pub gesture: Option<GestureFrame>,
}

pub struct GestureFrame {
    pub id: GestureId,                 // (actor, ordinal) — a restart is unambiguous
    /// The invariant part: tool, brush, layer, seed. Present on the gesture's
    /// first frame and re-sent on every resync frame.
    pub head: Option<GestureHead>,
    /// Index of the first control point in `points`; 0 on a resync frame.
    pub from: u32,
    /// Everything frozen since the last frame, plus the provisional knot under
    /// the cursor.
    pub points: Vec<ControlPoint>,
    /// Where on the assembled curve the stroke begins (`StrokeRecord::start`,
    /// §6.2). Per frame, not in the head: a curve *parameter* names a
    /// different place as the path grows, and it refines until the entry
    /// spans freeze — after which it is final before any cached head bakes it.
    pub start: f32,
}
```

- **The receiver** does `path.truncate(from); path.extend(points)` — valid
  because frozen points never change, which is a property of the fitter, not an
  assumption about the network.
- **A gap** (`from > path.len()`, or a `MeshEvent::Lagged` for that origin) drops
  that peer's live gesture and waits. Nothing is requested and nothing is
  retransmitted on demand: the next **resync frame** repairs it.
- **Resync frames** carry `head` and the full path (`from = 0`) at ~1 Hz. A
  stroke rarely outlives a few seconds, so this bounds worst-case repair latency
  at about a second while costing roughly 1 KB/s — and it is exactly what a
  client joining mid-stroke needs, with no join-time presence exchange to design.

**Coalescing: the outbox is a latch, not a queue.** This is the structural
difference from the action path. `take_outbox()` returns a **log** — every
action, in order, none droppable. `take_presence()` returns a **snapshot** — the
current value, or nothing if unchanged since the last drain. Pen input arrives at
240 Hz+; the wire runs at one frame per publish tick (~30 Hz). Coalescing is safe
because the latch stores the *current full gesture state* and the delta is
computed **at drain time** against `last_sent_from`. Eight pointer moves between
drains produce one frame carrying all eight points.

**Waking is not working.** A latch has to be drained on a cadence, so the pump
wakes at a fixed rate for as long as a session is live — but a tick on which
nothing moved must cost nothing. Two `&self` tests, `Engine::presence_due(now)`
and `Engine::peers_revision()`, decide that before anything is borrowed mutably:

- `presence_due` is deliberately **conservative** — it may say yes where the
  drain then finds nothing, but never the reverse, since a pump trusting a false
  negative would drop a frame on the floor. `presence_due_never_hides_a_frame`
  pins exactly that implication across cursor moves, name changes, a whole
  gesture, the frame that *clears* a gesture, and the heartbeat.
- `peers_revision` lets the frontend notice the roster moved without rebuilding
  and comparing a projection of it — an allocation per tick, otherwise, per peer.

The third cost is the least visible from the engine side and the largest in
practice: taking a mutable borrow means writing to the signal the engine lives
in, and `Signal::write` marks its subscribers dirty whether or not the value
changed. An unconditional drain therefore re-rendered every component that reads
the renderer, thirty times a second, for the whole life of a session in which
nobody was doing anything. **Components driven by presence read the renderer with
`peek`, not `read`**, for the same reason: they already re-render when `peers`
changes, and subscribing to the engine as well buys nothing but churn.

### 17.6 Rendering — the preview fold

One preview per in-flight gesture:

```
presented = for each actor in ascending ActorId order:
                overlay that actor's live tiles onto the running state
            starting from timeline.current()
```

Two decisions in that sentence, both deliberate:

- **Ascending `ActorId`, with the local actor taking its place like any other.** A
  fixed order every client can compute means every client composites the same
  picture, and it removes "the local one is special" from the render path.
- **A live gesture renders over the *committed* document, not over the previous
  peer's preview — unless the two reach the same tiles.** Chaining unconditionally
  would cost far more: peer *k*'s cached head would be invalidated by every move
  of peers < *k*, so with two painters each move invalidates the other's cache and
  the incremental repaint collapses. Rooting every head at the committed state
  keeps per-move cost O(1) in the number of peers. The overlay is per-tile in
  actor order, using the dirty set the renderer already computes
  (`affected_tiles`, surfaced in `StrokeCarry`).

**But the overlay copies whole tiles, so a shared tile cannot be split between two
strokes.** Each copy carries the committed pixels *plus* one stroke's paint, so
the second copy puts back exactly what the first had drawn there and one of the
two strokes disappears from that tile until it commits. Rooting at the committed
state is therefore conditional on the tiles being uncontested, and the condition
is checked with the very footprint the commit will use (`stroke_rect`,
`fill_rect`, §12.6 — so the fold cannot call two strokes independent where the log
would call them conflicting):

> A stroke whose reach meets that of a gesture already in the fold, on the same
> layer, renders over **the fold** instead — and gives up its cached head to do
> it, because the fold is rebuilt on every move and a head rooted there is stale
> as soon as it is stored.

The price is paid only while two people are painting the *same tiles*, rather than
merely at the same time, which is what keeps the cache argument above intact for
the case it was made about. A fill needs no such rule: it already reads the fold
and replaces the layer's whole tile map rather than copying tiles across.

What stays provisional is which of two concurrent strokes ends up on top, and that
deserves a plain statement rather than a hedge: **a preview of concurrent strokes
is provisional and the commit is authoritative.** It has to be — the true result
depends on the total order, which is not known until both strokes commit. When
they commit, replay produces the ordered, correct pixels everywhere.

Two related consequences:

- **The `preview == committed` invariant (§1.3) is restated, not weakened:** it
  holds in the absence of concurrent remote edits, which is what it has meant
  since `merge_remote` began rebasing the preview.
- **A remote live stroke is masked by that peer's selection**, read from
  `state.selections[peer]` — durable state the receiver already has. This is
  where §17.3 and §17.5 meet: the reason a peer's live stroke can be reproduced
  faithfully at all is that the mask it is being drawn through is replicated
  durably, in the log, where replay can find it.

**Head invalidation is an epoch, not a blanket drop.** The epoch is bumped by
everything that replaces the base a preview is drawn over (a commit, an undo, a
remote merge, a load, a frame drag), and a `FrozenHead` stamped with an older
epoch is discarded. That keeps §6.2's "rule out the whole class" guarantee
without the previous code's side effect of dropping the cache on *every*
non-gesture command — which with peers painting would have thrown away their
heads whenever this client so much as panned.

The epoch, the fold, the head cache and the unlogged drag preview live together
in one `Preview` type, and that is what makes the rule hold rather than merely
state it. As four fields of `Engine` any method could move one without the
others, and the epoch's bumps were written out at the call sites: `Seek` cleared
the drag-preview slot up front and bumped only inside `if timeline.seek(..)`, so
a seek that declined dropped the base a head was stamped against while leaving
the stamp valid. Moving the slot now goes through one method that bumps as it
goes, so the whole class is unrepresentable instead of fixed one arm at a time.
It is also why the epoch is *observable* (`Engine::preview_epoch`) despite being
a cache detail: a drag preview that changes no tiles leaves a stale head drawing
exactly the right paint, so nothing on screen can tell you the rule has been
broken until a later preview moves tiles and the cause is long gone.

**A head's lifetime is its gesture's.** The epoch above says when a head is
*wrong*; this says when it stops being anyone's. The fold rebuilds the cache into
a fresh map each time and carries a head across only for an actor still drawing,
so a gesture that ends releases its head by construction rather than by a call
somebody has to remember to make. That the two rules are separate matters,
because their failures look nothing alike: a stale head draws the wrong picture
and is caught by a golden, while a head that outlives its gesture draws the
*right* picture and quietly holds a `DocState`'s worth of tile handles the pool
cannot reclaim. Only a second painter can reach it — with nobody drawing the fold
clears the cache wholesale — so the leak was one peer lifting while another
painted on, which is exactly when there is least GPU memory to spare.

**The local stroke's commit takes the fold's tiles** (§6.2). A fold that draws
this client's own stroke over the committed document — and over nothing else:
not a drag's stand-in, not a contested tile already carrying another peer's
paint — keeps that render in a slot the next `CommitStroke` is offered, and the
slot follows the epoch rule above exactly as a head does: whatever replaces the
base drops it. A peer's stroke is never offered this way; its commit arrives as
an action and renders at the fold, like every replay.

### 17.7 Commands and transport

§4's own principle — *the class is in the type, not in a comment* — is why
`PeerCommand` exists as its own arm (§4). `SetCursor` at pointer rate is fine: it
writes a field and marks the latch dirty, and §17.5 does the rest. Its class is a
statement about what it *is* — a fact about the hand, published, in no file and
reached by no undo — and not about who reads it, which is why nothing moved when
a second reader appeared on this side of the glass: a guide draws its rays
through the same cursor (§20.9), the way this client's own next stroke goes to
`SetActiveLayer`'s layer.
`DocCommand::Select` does **not** move — it is logged, so it stays where it is;
only its effect is now owner-scoped. `GestureCommand` is unchanged: it already
built in per-client state and committed document state, which is now simply
visible to others while it builds.

`merge_presence` takes the actor **from the transport's authenticated origin**,
not from the frame body — the same discipline §17.3 gets for free from
`ActionId`, made explicit here because a presence frame has no id to derive it
from.

`stark-net`'s `Wire` enum already reserved the room
(`Wire::{Action, Presence}`), with three rules that keep the existing model
untouched:

- **Presence never enters the `Mirror`, a snapshot, or a file.** One rule, and
  the save format and catch-up protocol need no changes at all.
- **Presence is dropped, never resynced.** `MeshEvent::Lagged { origin }` already
  reports loss per origin; for actions it means "resync", for presence it means
  "drop this peer's gesture and wait for their next resync frame".
- **Presence is shed first under congestion**, and is rate-capped per origin,
  since it is the only traffic a peer can generate without limit.

The UI pump gains one symmetric line each way:

```
engine.take_outbox()   → Wire::Action      RemoteEvent::Action   → engine.merge_remote
engine.take_presence() → Wire::Presence    RemoteEvent::Presence → engine.merge_presence
```

Two things the implementation settled that the design left open:
`Session::presence()` became **`Session::publish(now)`** — the latch and its
bookkeeping (sequence, resync clock, sent-path watermark) belong with the state
they latch, and the delta has to be computed where the fitter is; change
detection is by *comparison* against what was last published rather than by a
dirty flag, because a comparison cannot be forgotten at a call site.

### 17.8 Save format and compatibility

The container is unchanged and no `format_version` bump was needed. **Presence** is
never serialized; it is not in `DocState`, so it *cannot* be
(`presence_never_enters_the_snapshot` pins the transport half,
`presence_never_reaches_the_save_file` the engine half). **`ActionKind::Select`**
keeps its exact encoding; only the key it applies under changes.

**`LayerId` did not.** It was a `u64` when this section was written, and the
per-client identity §17.9 describes changed only the *values* it took. Closing that
finding properly changed the type: a layer's id is now the action that minted it,
`{ action: ActionId, k: u32 }`, and a struct where a scalar was is the one reshaping
carbonite's name-based reconciliation cannot absorb (§8). **Files written before it
do not open**, and they report as a decode failure rather than as a named refusal.
§19's beta rung is unclaimed, which is the permission this was taken under and the
only thing that makes it a decision rather than a mistake — but it is a decision, and
this is where a reader asking "does this break my files" is told so. `AddGuide` moved
its id into the payload in the same pass (§20.5's `GuideId` was *derived* from the
action id, which `start_collaboration`'s rewrite then moved out from under every
later `SetGuide` and `RemoveGuide` naming it), which is a shape change on the same
terms.

### 17.9 Two latent defects this fixed on the way past

**Concurrent layer ids collided.** `Engine::process_doc` minted
`LayerId(self.next_layer)` from a local counter resynced from the log. Two peers
adding a layer concurrently both minted the same `LayerId`, the log then
contained two layers with one id, and `layer_index` found whichever came first —
a real convergence failure, and exactly the class of bug "per-client identity" is
supposed to prevent.

**A layer's id is the id of the action that minted it**, and which of that
action's layers it is: `LayerId { action: ActionId, k: u32 }`. An `ActionId` is
already the log's total-order key `(lamport, actor)` and so already globally
unique — two actions cannot share one, therefore two layers cannot share an id.
That is the guarantee made structural rather than kept: there is no counter, so
there is nothing to resync when a log is picked back up, no partition to reason
about, and no rule about re-sharing to remember. `GuideId` took this answer from
the start (§20.5); `k` is what let layers follow, since one `AddGuide` mints one
guide where a `DuplicateLayer` mints one per layer of a subtree.

The mint is a door rather than a convention: `Engine::commit_minting` draws the
action id and hands it to the closure that builds the kind, so a kind naming a
layer it mints cannot be built without the id it will be committed under.
`LayerId::ROOT` is the one id no action produces — a reserved `k` of `u32::MAX`,
not a reserved action, because the Lamport clock starts at zero and
`(0, SOLO)` is a perfectly ordinary first action of a solo document.

**The two doors this replaced.** For as long as the id was a counter partitioned
by a mixed 32-bit fold of the `ActorId`, the guarantee was statistical — two
actors whose folds coincided minted colliding ids and nothing said so — and the
counter had to be recovered from the log at *both* places a document arrives by.
A load did it (`resync_counters`); **sharing** did not, restarting outright on the
reading that a new session is a new actor with an empty half of the id space. An
identity is a browser's persisted key rather than a session's, so re-sharing a
painting — or sharing one this client shared before and has reopened — restarted
inside a half it had already minted in. Both doors are gone with the counter.

One consequence worth stating: `start_collaboration` rewrites the log's
`ActorId::SOLO` actions to the sharer, and the layer ids *inside* those actions
keep saying `SOLO`. They must — rewriting them would mean rewriting every later
reference to those layers, which is a rewrite of the whole log to change nothing
observable — and they stay unique anyway, since `SOLO` authors no action in a
shared session.

**A remote `RemoveLayer` could strand the active layer.** The engine repointed
`session.active_layer` after a *local* `RemoveLayer`, but `merge_remote` had no
equivalent, so a peer deleting the layer I am painting on left me pointed at a
layer that no longer existed — after which my strokes were silently refused by
`apply` with nothing on screen to explain it. Repointing belongs in one place
both paths reach, and the same check applies to every peer's `active_layer` in
the roster before it is drawn.

### 17.10 Testing

The valuable tests are headless and need no network, because §17.3 put the
semantics in `stark-engine`. `tests/peer_state.rs` covers:

- **The masking defect.** `one_peers_selection_does_not_clip_anothers_stroke` — A
  selects the left half, B paints across the boundary, and both halves must land
  on both screens. This failed before §17.3, which is the point of writing it
  first.
- **The other half of the same rule.**
  `a_peers_stroke_is_reproduced_through_the_authors_own_mask` — the author's own
  mask *does* gate the stroke, on every peer. That is what makes replicating the
  mask necessary rather than merely tidy.
- **Independent masks converge.** Two peers holding different selections, plus a
  late joiner rebuilding both from the log. One thing this made explicit:
  convergence is about the *artwork*, not the chrome — the marching ants are
  drawn for whoever's selection is in force on this client, so peers with
  different masks legitimately show different outlines, and the test deselects
  before comparing pixels.
- **Undo scoping**, which falls out of keying by the action's own author.
- **A solo document is unaffected**, the check that the re-keying is invisible
  where there is nothing to key by.
- **Layer ids** (§17.9): concurrent adds mint distinct ids; a solo document's ids
  are exactly the `SOLO` actions that minted them; a remote removal repoints the
  active layer and painting still works.
- **Presence end to end:** a peer's live stroke previews before it commits and
  the commit lands the same pixels; a silent peer loses its gesture, then its
  place; presence never becomes an action.

`peer.rs`'s own unit tests cover the roster as pure CPU logic — stale `seq`
dropped, delta reassembly, gap → drop → repair on the next resync frame, a new
ordinal starting over, expiry, `leaving`. `stark-net/tests/presence.rs` covers
the wire: a frame reaches peers attributed to its **sender** (from the
transport's origin, not the payload), and reaches neither the mirror nor a
joiner's snapshot.

### 17.11 Alternatives considered and rejected

- **Put everything in `DocState`, keyed by actor.** Rejected for cursors and
  gestures only, on the three costs in §17.2. Adopted without reservation for the
  selection.
- **Keep everything out of the log and close each stroke over its own mask** —
  `StrokeRecord` carrying a content-addressed `MaskId` resolved through a store,
  in the manner of brush assets. Genuinely attractive: strokes become hermetic,
  and the ordering of `Select` against remote strokes stops mattering. Rejected
  because it buys immunity to an ordering problem the total order already solves,
  at the price of a new content-addressed store, a save format change, and either
  losing undoable selections or building a second mechanism to restore them.
- **A separate presence CRDT** (LWW-register map with vector clocks). Rejected:
  presence has a single writer per key and no merge semantics worth the name —
  "latest from that actor wins" is the entire specification, and a per-actor `seq`
  plus an expiry implements it in a few lines.
- **Chaining live previews peer-over-peer *unconditionally*.** Rejected on the
  cache argument in §17.6. Chaining only where the footprints meet is not the same
  proposal and is not optional: without it the tile-wise overlay drops one of the
  two strokes outright, which is a missing stroke rather than a provisional
  ordering.
- **Making presence an `Action` with a TTL.** Rejected — it puts pointer-rate
  traffic in a grow-only log, which no amount of compaction makes right.

### 17.12 Deliberately deferred

Authority and trust are unchanged from §12.5: anyone with a ticket can write, and
a peer that forges action ids can forge selections along with everything else, so
§17.3's ownership guarantee is exactly as strong as the log's and no stronger.
Also deferred: audio/text chat, follow-the-peer view sync (trivial once the
roster exists — it is a peer's `ViewTransform`, private today by choice, not by
necessity), presence over the catch-up ALPN for peers with no mesh path, and
per-peer permissioning of layers.

---


