# Stark — Per-client state

A design for the third class of engine state: state **owned by one client, read by
all**. It covers the selected layer, the selection region, and live in-progress
gestures, and it is written to slot into the existing architecture
([DESIGN.md](DESIGN.md) §4 commands/actions, §6.8 selections, §12 collaboration)
rather than beside it.

## 1. What is actually broken

Three separate things, only one of which is a missing feature:

1. **The selection is shared when it must not be.** `DocState::selection` is a
   single mask (§6.8), edited by logged `Select`/`InvertSelection` actions and read
   by `CommitStroke::apply` from the state it folds over. In a shared session that
   is a genuine defect, not a cosmetic one: if I lasso a region, *your* next stroke
   is clipped to my lasso — on your screen too, because the mask is in the replayed
   state. Two people cannot mask independently, and neither can paint freely while
   the other has a selection in force.
2. **Live strokes are invisible until commit.** A peer's stroke arrives as one
   `CommitStroke` on release, so a thirty-second stroke appears as a single event
   thirty seconds late. §12.4 already names this as future work.
3. **The selected layer is fine, and is not the problem it looks like.**
   `Session::active_layer` is per-client already, and `StrokeRecord::layer` closes
   each stroke over its target, so two clients painting different layers already
   converge. What is missing is only *visibility* — nobody can see who is working
   where — plus one latent id-minting bug (§9).

Those three are not the same kind of thing, and the design's main job is to stop
treating them as one.

## 2. The classification

DESIGN §4 states one rule and derives everything from it: *which class of state a
command touches decides whether it is logged, whether peers see it, whether undo
reaches it.* Adding an owner axis to that rule gives a 2×2, and every piece of
per-client state falls into exactly one cell:

|  | **private** — this client only | **published** — every client reads |
|---|---|---|
| **not logged** | **view state** — pan/zoom, tool, brush, media params, environment, viewport | **presence** — cursor, selected layer, live gesture, display name |
| **logged** | *(empty — the log is shared by construction)* | **document state** — either **shared** (layers, paint, surface, background) or **owned** (the selection) |

The discriminator between the two new cells is the one this codebase already
uses, applied without exception:

> **Does replay need it to reproduce pixels?**

- The **selection** does. A stroke's pixels depend on the mask it was drawn
  through (§6.8), so a peer replaying my stroke must be able to reconstruct *my*
  mask at *that point* in the log. It is document state; it just has an owner.
- The **selected layer** does not. It is already closed over by
  `StrokeRecord::layer`. Logging it would turn every click in the layers panel
  into an undo step, for no reproducible consequence.
- A **live gesture** does not. It is by definition the thing that has not
  committed yet; when it does, the `Action` is authoritative and the live copy is
  discarded.

So the "third class" the use-cases point at is really two mechanisms, and the split
is not arbitrary — it is forced by the rule already governing §4.

### 2.1 On "a map from clients inside the shared state"

That is exactly right for the selection, and exactly wrong for the rest.

For the selection it is not merely an implementation option, it is the *only*
formulation that keeps §12's convergence argument intact: the mask must be a
deterministic function of the ordered log, because a stroke's pixels are. Putting
per-actor selections in `DocState` keeps them derived by replay, undoable, and
ordered against the strokes that read them — all properties they already have,
now correctly scoped.

For the cursor, the selected layer and the in-flight path, a map inside `DocState`
would be a category error with three concrete costs: every hover would enter the
undo history and the save file; the timeline would have to re-materialize on
pointer-rate traffic (`ReplicatedTimeline::resync` pops and replays from the first
divergence — pointer noise would drive that loop); and a lost packet would become
a convergence failure instead of a dropped frame. Presence must be allowed to be
lossy, and nothing in `DocState` is allowed to be.

Hence: **one map inside the shared state, one roster outside it.** The rule above
says which goes where, so the boundary is derivable rather than remembered.

## 3. Owned document state — the selection, keyed by actor

```rust
pub struct DocState {
    pub layers: Vector<Layer>,
    pub bounds: CanvasBounds,
    /// One selection per actor (§6.8). Absent = everything selected, which is the
    /// free representation `Selection::everything()` already is — so an actor who
    /// never selects costs nothing, and a solo document has exactly one entry.
    pub selections: HashTrieMap<ActorId, Selection>,
    pub surface: SurfaceId,
    pub background: [f32; 3],
}
```

`HashTrieMap` for the same reason as everything else in `DocState`: cloning a
version must stay a handful of `Arc` bumps (§5.1), and the masks are tile maps
that structurally share across versions exactly like paint does.

**The owner is derived, never declared.** `Action::apply` already receives `&self`,
so it has `self.id.actor`:

```rust
ActionKind::Select(op)      => state.select(self.id.actor, ctx, op),
ActionKind::InvertSelection => state.invert_selection(self.id.actor, ctx),
ActionKind::CommitStroke(rec) => /* ... */ selection: state.selection_of(self.id.actor),
```

This is the load-bearing detail. Ownership is read off the action *id*, which is
already the total-order key and already authenticated exactly as much as the rest
of the log. There is no `owner:` field in any payload, so "only its owner may
change it" is not a permission check that could be forgotten at a call site — it is
structurally impossible to express the other thing. A peer can no more write my
selection than it can author an action under my id, and those are the same
statement.

Consequences that fall out with no further work:

- **Undo is already scoped.** `ReplicatedTimeline::undo_target` targets *my* most
  recent effective action (§12.3), so undoing a selection is mine to do and
  reaches only mine.
- **Ordering is already right.** A `Select` and a `CommitStroke` by the same actor
  are ordered by their Lamport ids, and the stroke reads the state folded up to its
  own position — so a late-arriving remote `Select` re-orders through the existing
  `resync` path (§12.2) and everyone still converges.
- **The wire and file formats do not change.** `ActionKind::Select(op)` is
  byte-identical; only its interpretation moves from "the selection" to "the
  author's selection". Every existing document has all of its `Select` actions
  under `ActorId::SOLO` and therefore loads and replays to the same pixels. No
  migration, no `format_version` bump.
- **`start_collaboration`'s SOLO→actor rewrite already covers it** — it rewrites
  ids, the map is keyed off ids, and the rebuild is a `resync`.

**Rendering the outline.** The selection overlay pass (§6.8) draws one instanced
quad per mask tile. It takes a short list rather than one mask: the local actor's
selection in the marching-ants style, and — **only if this client asks for it** — the
selection of every actor *currently present*, drawn as a flat line in that peer's
colour so the two never read as the same thing.

`ViewCommand::SetShowPeerSelections` gates the second group, and it is **off by
default**. Knowing which region someone else is working inside is occasionally
useful; a second contour over the artwork is a cost paid on every frame you look at
it, and most of the time the answer to "what is that line?" should be "the one I
drew". It is view state, so each client decides for itself and nothing about the
choice is logged or sent — it changes what you look at, not what the drawing is. The
control is mounted in the Select panel only while a session is live, on the same
argument §6.8 makes for the selection bar: a control that is present or absent says
whether the thing it governs exists, which a permanently-visible one greyed out does
not.

Note the asymmetry underneath, which is deliberate and worth stating: `DocState`
holds a selection for every actor that ever selected, because replay needs them; the
roster says which of those are here; the setting says whether to draw them. **The log
decides what exists, presence decides what could be shown, and the client decides
whether it is.**

## 4. Presence — the ephemeral roster

Everything else per-client lives outside the timeline, as a sibling of `Session`,
because that is precisely what it is: other people's sessions.

```rust
/// What one participant is doing right now. Never logged, never saved, never
/// referenced by anything in the action log.
pub struct Peer {
    pub actor: ActorId,
    pub name: String,
    /// Derived from `actor` by hash, so every client agrees on it with no
    /// negotiation and no allocation protocol.
    pub color: [f32; 3],
    pub active_layer: LayerId,
    pub cursor: Option<Vec2>,       // canvas space
    pub gesture: Option<LiveGesture>,
    /// Monotonic per actor; a frame that does not advance it is stale and dropped.
    seq: u64,
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
type, rendered through the same entry point (`StrokeRenderer::render_range`) that
the commit will use. There is no second stroke representation to keep in step, and
therefore no second way for live and committed pixels to disagree.

### 4.1 The local client is just another peer

`Session` becomes the local peer's record plus the private view state, and exposes
one projection:

```rust
impl Session {
    /// The publishable half of this session — what other clients get to see.
    /// The private half (view transform, brush, media params) is everything this
    /// does not return, which is where the private/published line is *recorded*
    /// rather than described.
    pub fn presence(&self) -> PeerFrame;
}
```

symmetric with `Engine::observe()`, which is the UI-facing projection. The engine
then holds one roster containing everybody:

```rust
peers: BTreeMap<ActorId, Peer>,   // includes the local actor
```

so the local in-flight stroke and remote in-flight strokes are the *same*
mechanism, folded in the same order, on every client. That uniformity is what makes
every client see the same canvas mid-stroke, and it deletes the special case rather
than adding one.

### 4.2 Lifecycle

- **Heartbeat.** A peer publishes at least every 2 s even when idle; a peer unheard
  from for 6 s is dropped from the roster. An explicit `Leave` frame on a graceful
  exit makes the common case instant.
- **A gesture is cleared by its own commit.** Any `Action` merged from actor *A*
  clears *A*'s `LiveGesture` — a gesture is a thing that becomes an action, so the
  action's arrival is the end-of-gesture signal, with no id to correlate and no
  window in which both are drawn.
- **Cancels** send an explicit `GestureEnd`, and a gesture with no update for 2 s is
  dropped anyway, so a peer that crashes mid-stroke does not leave a smear.

### 4.3 Loss is a design property, not a failure mode

Nothing in the action log ever references presence. That is the invariant that
buys everything else: presence may be dropped, coalesced, reordered or arbitrarily
delayed without any effect on convergence. The worst outcome of total presence loss
is that a session looks like today's — strokes appear on commit. So the transport
is free to shed presence first under congestion, and the receiver is free to drop
frames it cannot use.

## 5. Live gestures on the wire

Sending the whole fitted path on every pointer move is O(n²) bytes over a stroke.
The fitter already solves this: `PathFitter` **freezes** a prefix of control points
that is final and never revised (§6.2), which is the same property that lets the
renderer cache a `FrozenHead`. The wire form is the same partition:

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
    /// The control points from `from` on: everything frozen since the last frame,
    /// plus the provisional knot under the cursor.
    pub points: Vec<ControlPoint>,
}
```

- **The receiver** does `path.truncate(from); path.extend(points)` — valid because
  frozen points never change, which is a property of the fitter, not an assumption
  about the network.
- **A gap** (`from > path.len()`, or a `MeshEvent::Lagged` for that origin) drops
  that peer's live gesture and waits. Nothing is requested and nothing is
  retransmitted on demand: the next **resync frame** repairs it.
- **Resync frames** carry `head` and the full path (`from = 0`) at ~1 Hz. A stroke
  rarely outlives a few seconds, so this bounds worst-case repair latency at about
  a second while costing roughly 1 KB/s — and it is also exactly what a client that
  joins mid-stroke needs, with no join-time presence exchange to design.

### 5.1 Coalescing: the outbox is a latch, not a queue

This is the structural difference from the action path and is worth naming.
`take_outbox()` returns a **log** — every action, in order, none droppable.
`take_presence()` returns a **snapshot** — the current value, or nothing if
unchanged since the last drain.

Pen input arrives at 240 Hz+; the wire runs at one frame per publish tick (~30 Hz).
Coalescing must therefore be safe, and it is, because the latch stores the *current
full gesture state* and the delta is computed **at drain time** against
`last_sent_from`. Eight pointer moves between drains produce one frame carrying all
eight points. Nothing accumulates and nothing is lost by dropping intermediate
states — which is the definition of the state being a snapshot, and why it can be
allowed to be lossy while the action log cannot.

**Waking is not working.** A latch has to be drained on a cadence, so the pump wakes
at a fixed rate for as long as a session is live — but a tick on which nothing moved
must cost nothing. Two `&self` tests, `Engine::presence_due(now)` and
`Engine::peers_revision()`, decide that before anything is borrowed mutably:

- `presence_due` is deliberately **conservative** — it may say yes where the drain
  then finds nothing, but never the reverse, since a pump that trusted a false
  negative would drop a frame on the floor. `presence_due_never_hides_a_frame` pins
  exactly that implication across cursor moves, name changes, a whole gesture, the
  frame that *clears* a gesture, and the heartbeat.
- `peers_revision` lets the frontend notice the roster moved without rebuilding and
  comparing a projection of it — an allocation per tick, otherwise, per peer.

The third cost is the one least visible from the engine side and the largest in
practice: taking a mutable borrow means writing to the signal the engine lives in,
and `Signal::write` marks its subscribers dirty whether or not the value changed. An
unconditional drain therefore re-rendered every component that reads the renderer,
thirty times a second, for the whole life of a session in which nobody was doing
anything. Components that are driven by presence read the renderer with `peek`, not
`read`, for the same reason: they already re-render when `peers` changes, and
subscribing to the engine as well buys nothing but churn.

## 6. Rendering — the preview fold

Today `Engine::preview: Option<DocState>` holds a single CoW document that replaces
the committed one at composite time, and `FrozenHead` caches the settled prefix so
per-move cost follows the tail rather than the stroke (§6.2).

Generalize to one preview per in-flight gesture:

```
presented = for each actor in ascending ActorId order:
                overlay that actor's live tiles onto the running state
            starting from timeline.current()
```

Two decisions in that sentence, both deliberate:

- **Ascending `ActorId`, with the local actor taking its place in that order like
  any other.** A fixed order every client can compute means every client composites
  the same picture, and it removes "the local one is special" from the render path.
- **Every live gesture renders over the *committed* document, not over the previous
  peer's preview.** Chaining would be marginally more faithful for two strokes
  overlapping in the same instant, and would cost far more: peer *k*'s cached head
  would be invalidated by every move of peers < *k*, so with two painters each move
  invalidates the other's cache and the incremental repaint collapses. Rooting every
  head at the committed state keeps per-move cost O(1) in the number of peers,
  which is the property that has to hold.

  The overlay is per-tile in actor order, using the dirty set the renderer already
  computes (`gpu::stroke::segments::affected_tiles`) — surfacing it in `StrokeCarry`
  is the one small addition this needs. Where two peers' live strokes touch the same
  tile in the same instant, the higher `ActorId` wins that tile.

That last case deserves a plain statement rather than a hedge: **a preview of
concurrent strokes is provisional and the commit is authoritative.** It has to be —
the true result depends on the total order, which is not known until both strokes
commit, so *any* preview of two simultaneous overlapping strokes is a guess. When
they commit, replay produces the ordered, correct pixels on every client.

Two related consequences:

- **The `preview == committed` invariant (§1.3) is restated, not weakened:** it
  holds in the absence of concurrent remote edits, which is exactly what it has
  meant since `merge_remote` began rebasing the preview. Every remote merge drops
  the frozen heads and rebuilds the live tails against the new committed state,
  which is what today's code already does for one stroke.
- **A remote live stroke is masked by that peer's selection**, read from
  `state.selections[peer]` — durable state the receiver already has. This is where
  §3 and §5 meet: the reason a peer's live stroke can be reproduced faithfully at
  all is that the mask it is being drawn through is replicated durably, in the log,
  where replay can find it.

## 7. Commands

DESIGN §4's own principle — *the class is in the type, not in a comment* — says the
published/private split inside `ViewCommand` should be a type. So:

```rust
pub enum InputCommand {
    Gesture(GestureCommand),
    Doc(DocCommand),
    View(ViewCommand),      // private: never logged, never sent
    Peer(PeerCommand),      // published: never logged, but broadcast
}

pub enum PeerCommand {
    /// Moves out of `ViewCommand` — the selected layer is now something others
    /// can see, which is the whole difference between the two classes.
    SetActiveLayer(LayerId),
    /// Hover position, in canvas space; `None` when the pointer leaves the canvas.
    SetCursor(Option<Vec2>),
    SetName(String),
}
```

`SetCursor` at pointer rate is fine: it writes a field and marks the latch dirty,
and §5.1 does the rest. `DocCommand::Select` does **not** move — it is logged, so
it stays where it is; only its effect is now owner-scoped (§3). `GestureCommand` is
unchanged: it already builds in per-client state and commits document state, which
is now simply visible to others while it builds.

The engine grows two hooks alongside the two it already has for actions, which is
the whole of the new frontend-facing API:

```rust
fn take_presence(&mut self) -> Option<PeerFrame>;   // latch drain (§5.1)
fn merge_presence(&mut self, actor: ActorId, frame: PeerFrame) -> bool;
fn peers(&self) -> impl Iterator<Item = &Peer>;     // or projected into ObservableState
```

`merge_presence` takes the actor **from the transport's authenticated origin**, not
from the frame body — the same discipline §3 gets for free from `ActionId`, made
explicit here because a presence frame has no id to derive it from.

## 8. Transport

`stark-net`'s `Wire` enum already reserves the room:

```rust
pub enum Wire {
    Action(Action),
    Presence(PeerFrame),
}
```

with three rules that keep the existing model untouched:

- **Presence never enters the `Mirror`, a snapshot, or a file.** One rule, and the
  save format and catch-up protocol need no changes at all.
- **Presence is dropped, never resynced.** `MeshEvent::Lagged { origin }` already
  reports loss per origin; for actions it means "resync", for presence it means
  "drop this peer's gesture and wait for their next resync frame" (§5).
- **Presence is shed first under congestion**, and is rate-capped per origin, since
  it is the only traffic on the wire that a peer can generate without limit.

The UI pump gains one symmetric line each way:

```
engine.take_outbox()   → Wire::Action      RemoteEvent::Action   → engine.merge_remote
engine.take_presence() → Wire::Presence    RemoteEvent::Presence → engine.merge_presence
```

## 9. Two latent defects this fixes on the way past

**Concurrent layer ids collide.** `Engine::process_doc` minted
`LayerId(self.next_layer)` from a local counter resynced from the log. Two peers
adding a layer concurrently both mint the same `LayerId`, the log then contains two
layers with one id, and `layer_index` finds whichever comes first — a real
convergence failure, and exactly the class of bug "per-client identity" is supposed
to prevent.

Ids are now minted from the author (`LayerId::mint`): a mixed 32-bit fold of the
`ActorId` in the high half, the per-actor counter in the low. `ActorId::SOLO` maps to
high half 0, so a document that was never shared keeps the small, readable ids it
always had — including the root layer's `LayerId(0)`, which every peer must agree on
because it predates any actor. `Engine::resync_counters` resumes only *this* actor's
counter, since resuming past someone else's would skip ids for no reason and hide the
fact that they cannot collide.

**A remote `RemoveLayer` can strand the active layer.** The engine repoints
`session.active_layer` after a *local* `RemoveLayer`, but `merge_remote` has no
equivalent, so a peer deleting the layer I am painting on leaves me pointed at a
layer that no longer exists — after which my strokes are silently refused by
`apply` (the absent-layer arm) with nothing on screen to explain it. Repointing
belongs in one place both paths reach, and the same check applies to every peer's
`active_layer` in the roster before it is drawn in the layers panel.

## 10. Save format and compatibility

Unchanged — nothing about the on-disk container moved, and no `format_version` bump
was needed:

- **Presence** is never serialized. It is not in `DocState`, so it *cannot* be; the
  `stark-net` test `presence_never_enters_the_snapshot` pins the transport half of
  that, and `presence_never_reaches_the_save_file` the engine half.
- **`ActionKind::Select`** keeps its exact encoding; only the key it applies under
  changes. Existing files carry `ActorId::SOLO` throughout and replay to identical
  pixels — every golden is unchanged, which is the check on whether §3 is the right
  shape: a correct scoping of something that was never scoped should be invisible on
  documents that only ever had one actor, and it is.
- **`LayerId`** keeps its type and its encoding (a `u64`); only the *values* newly
  minted ids take changed, and only for actors other than `SOLO`. An old file's
  `LayerId(1)` still decodes and still means what it meant.

## 11. Testing

The valuable tests are headless and need no network, because §3 put the semantics
in `stark-core`. `crates/stark-core/tests/peer_state.rs` covers:

- **The masking defect.** `one_peers_selection_does_not_clip_anothers_stroke` — A
  selects the left half, B paints across the boundary, and both halves must land on
  both screens. This failed before §3, which is the point of writing it first.
- **The other half of the same rule.**
  `a_peers_stroke_is_reproduced_through_the_authors_own_mask` — the author's own mask
  *does* gate the stroke, on every peer. That is what makes replicating the mask
  necessary rather than merely tidy.
- **Independent masks converge.** Two peers holding different selections, plus a late
  joiner rebuilding both from the log. One thing this made explicit: convergence is
  about the *artwork*, not the chrome — the marching ants are drawn for whoever's
  selection is in force on this client, so peers with different masks legitimately
  show different outlines, and the test deselects before comparing pixels.
- **Undo scoping**, which falls out of keying by the action's own author rather than
  needing anything to enforce it.
- **A solo document is unaffected**, the check that the re-keying is invisible where
  there is nothing to key by.
- **Layer ids** (§9): concurrent adds mint distinct ids; a solo document keeps small
  ones; a remote removal repoints the active layer and painting still works.
- **Presence end to end:** a peer's live stroke previews before it commits and the
  commit lands the same pixels; a silent peer loses its gesture, then its place;
  presence never becomes an action.

`peer.rs`'s own unit tests cover the roster as pure CPU logic — stale `seq` dropped,
delta reassembly, gap → drop → repair on the next resync frame, a new ordinal
starting over, expiry, `leaving`. `crates/stark-net/tests/presence.rs` covers the
wire: a frame reaches peers attributed to its **sender** (from the transport's
origin, not the payload), and reaches neither the mirror nor a joiner's snapshot.

## 12. Alternatives considered and rejected

- **Put everything in `DocState`, keyed by actor** (the literal reading of "a map
  from clients inside the shared state"). Rejected for cursors and gestures only:
  it makes hover undoable and saved, drives `ReplicatedTimeline::resync` at pointer
  rate, and converts a dropped packet into a convergence failure. Adopted, without
  reservation, for the selection.
- **Keep everything out of the log and close each stroke over its own mask** —
  `StrokeRecord` carrying a content-addressed `MaskId` resolved through a store,
  in the manner of brush assets (§6.6). This is genuinely attractive: strokes
  become hermetic, and the ordering of `Select` against remote strokes stops
  mattering at all. Rejected because it buys immunity to an ordering problem the
  total order already solves, at the price of a new content-addressed store, a save
  format change, and either losing undoable selections or building a second
  mechanism to restore them. §3 is a one-field change with none of that.
- **A separate presence CRDT** (LWW-register map with vector clocks). Rejected:
  presence has a single writer per key and no merge semantics worth the name —
  "latest from that actor wins" is the entire specification, and a per-actor `seq`
  plus an expiry implements it in a few lines.
- **Chaining live previews peer-over-peer** rather than rooting each at the
  committed state. Rejected on the cache argument in §6; the fidelity it buys
  applies only to strokes overlapping in the same instant, which is provisional
  either way.
- **Making presence an `Action` with a `TTL`.** Rejected — it puts pointer-rate
  traffic in a grow-only log, which no amount of compaction makes right.

## 13. Deliberately deferred

Authority and trust are unchanged from §12.5: anyone with a ticket can write, and
a peer that forges action ids can forge selections along with everything else, so
§3's ownership guarantee is exactly as strong as the log's and no stronger.
Also deferred: audio/text chat, follow-the-peer view sync (trivial once the roster
exists — it is a peer's `ViewTransform`, which is private today by choice, not by
necessity), presence over the catch-up ALPN for peers with no mesh path, and
per-peer permissioning of layers.

## 14. Build order

Each step is independently useful and independently testable, and the first two
carry all of the correctness. Status lives here, in the same spirit as DESIGN §13.

| # | Step | Status |
|---|---|---|
| 1 | `DocState.selections` keyed by `ActorId`; `apply` reads `self.id.actor` (§3) | done |
| 2 | Layer id minting from the actor; repoint `active_layer` on remote removal (§9) | done |
| 3 | The `Peer` roster, `PeerCommand`, `Session::publish()` (§4, §7) | done |
| 4 | Roster in the UI: peer chips on layer rows, remote cursors, other actors' selection outlines (§3, §4) | done |
| 5 | `Wire::Presence`, the latch drain, the pump (§5.1, §8) | done |
| 6 | Live gesture frames: delta + resync, receiver reassembly (§5) | done |
| 7 | The preview fold and per-peer frozen heads (§6) | done |

Two things the implementation settled that the design left open:

- **`Session::presence()` became `Session::publish(now)`** — the latch and its
  bookkeeping (sequence, resync clock, sent-path watermark) belong with the state
  they latch, and the delta has to be computed where the fitter is. Change detection
  is by *comparison* against what was last published rather than by a dirty flag: a
  comparison cannot be forgotten at a call site.
- **Head invalidation is an epoch, not a blanket drop.** `Engine::doc_epoch` is
  bumped by everything that replaces the base a preview is drawn over (a commit, an
  undo, a remote merge, a load, a frame drag), and a `FrozenHead` stamped with an
  older epoch is discarded. That keeps DESIGN §6.2's "rule out the whole class rather
  than enumerate the ways it arises" guarantee, without the previous code's side
  effect of dropping the cache on *every* non-gesture command — which with peers
  painting would have thrown away their heads whenever this client so much as
  panned.
