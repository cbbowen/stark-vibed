//! A layer's **liquify run** (§6.13): the state a sequence of liquify strokes
//! composes through, so that working a spot over and over resamples the picture
//! once rather than once per stroke.
//!
//! A liquify stroke is a homeomorphism of the canvas. Applied to pixels it would
//! be a resample, and a resample per stroke — or, as the first design had it, per
//! *segment* — blurs an edge a little each time until nothing sharp is left. So
//! the stroke is kept as what it is: a **displacement field** `d`, composed from
//! every step of every stroke in the run by gathering the field itself (smooth, so
//! the gather is nearly lossless), and the picture the layer shows is the run's
//! **pristine base** resampled through the composed field, `out(x) = base(x + d(x))`
//! — one resample, whatever the run has been through.
//!
//! Three maps and a bound, all persistent and all handles (§5.1), so a `DocState`
//! holding one is as cheap to keep as one holding tiles:
//!
//! - **`base`** — the layer's tile at every coordinate the run has *read*, as it stood
//!   when the run first reached it. What the resample reads. `None` is a real entry:
//!   the run read bare canvas there, and a tile arriving since is a change.
//! - **`field`** — the displacement at every coordinate the run has *written*, one
//!   `Rg32Float` tile texture each, aprons kept by the same rule as paint (§6.4). An
//!   absent tile is the identity.
//! - **`produced`** — the picture tile the run last wrote at each written coordinate.
//!   With `base`, this is what the run *expects* the layer to hold: a tile that is not
//!   the one expected was painted by something else since, and the run is stale there.
//! - **`reach`** — per written tile, a bound on `|d|` over it, canvas px. What sizes
//!   the base a stroke has to composite, and what the engine caps: past the cap a run
//!   re-bases, so a stroke's reads stay inside the reach its footprint declares
//!   (`LiquifyEffect::REACH_PX`).
//!
//! **Only a liquify stroke ever writes one**, and that is the §12.6 argument. A paint
//! stroke leaves the run alone; the next liquify stroke reads the tiles inside its
//! declared reach and compares them to what the run expects, by identity — the same
//! change detection the undo patch uses (§5.2) — and starts afresh from the picture
//! if anything differs. Every decision is then a function of tile identities and
//! the run, both of which the commuting splice restores exactly, so a liquify stroke
//! and a paint stroke beyond its reach land the same picture in either order.

use std::rc::Rc;

use rpds::HashTrieMap;

use crate::gpu::tile::{TexHandle, TileMap, TilePairHandle};
use stark_model::geom::{TileCoord, TileRect};

/// The state a sequence of liquify strokes composes through — see the module.
#[derive(Clone, Default)]
pub struct LiquifyRun {
    base: HashTrieMap<TileCoord, Option<TilePairHandle>>,
    field: HashTrieMap<TileCoord, TexHandle>,
    produced: HashTrieMap<TileCoord, TilePairHandle>,
    reach: HashTrieMap<TileCoord, f32>,
}

impl LiquifyRun {
    /// A run that has read and written nothing: the identity over the picture as it
    /// stands, which is what a stroke starts from when the layer has no run or its
    /// run has gone stale under it.
    pub(crate) fn fresh() -> Self {
        Self::default()
    }

    /// Whether the layer's tiles are what this run expects everywhere it has read or
    /// written **inside `within`** — the stroke's declared read rect, and the only
    /// tiles it may consult (§12.6).
    ///
    /// Identity, never content: a committed tile is never rewritten in place (§5.2),
    /// so a shared handle *is* an unchanged tile, and a handle the undo splice put
    /// back reads as unchanged exactly as the canonical replay would have it.
    pub(crate) fn is_fresh(&self, tiles: &TileMap, within: TileRect) -> bool {
        let same = |c: &TileCoord, want: Option<&TilePairHandle>| match (tiles.get(c), want) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b),
            _ => false,
        };
        self.produced
            .iter()
            .filter(|(c, _)| within.contains(**c))
            .all(|(c, h)| same(c, Some(h)))
            && self
                .base
                .iter()
                .filter(|(c, _)| within.contains(**c) && !self.produced.contains_key(c))
                .all(|(c, h)| same(c, h.as_ref()))
    }

    /// What the run reads at `c`: `Some(entry)` once the run has recorded the tile
    /// there (`None` inside meaning bare canvas), or `None` for a coordinate it has
    /// never reached — which a stroke then records from the layer's own tile.
    pub(crate) fn base_at(&self, c: TileCoord) -> Option<Option<&TilePairHandle>> {
        self.base.get(&c).map(Option::as_ref)
    }

    /// The displacement tile at `c`, or `None` for the identity.
    pub(crate) fn field_at(&self, c: TileCoord) -> Option<&TexHandle> {
        self.field.get(&c)
    }

    /// The bound on the displacement over tile `c`, canvas px — 0 where the run has
    /// written nothing.
    pub(crate) fn reach_at(&self, c: TileCoord) -> f32 {
        self.reach.get(&c).copied().unwrap_or(0.0)
    }

    /// Whether the run has written anything at all — a run that has not is the
    /// identity, and costs nothing to keep.
    pub fn is_identity(&self) -> bool {
        self.field.is_empty()
    }

    /// How many tiles the run has written.
    pub fn written(&self) -> usize {
        self.field.size()
    }

    /// Record what the run reads at `c` — the layer's tile there as the run first
    /// found it. A no-op once recorded: the base is pristine by never being replaced.
    pub(crate) fn record_base(&mut self, c: TileCoord, tile: Option<TilePairHandle>) {
        if !self.base.contains_key(&c) {
            self.base.insert_mut(c, tile);
        }
    }

    /// Record what the run wrote at `c`: the picture tile the layer now holds there,
    /// the displacement tile under it, and the bound on that displacement.
    pub(crate) fn record_written(
        &mut self,
        c: TileCoord,
        picture: TilePairHandle,
        field: TexHandle,
        reach: f32,
    ) {
        self.produced.insert_mut(c, picture);
        self.field.insert_mut(c, field);
        self.reach.insert_mut(c, reach);
    }

    /// Whether two runs are the same run — the change test the undo patch and the
    /// fold audit use (§12.6), by the same argument tile identity is one: a run is
    /// replaced, never edited, once a stroke has committed it.
    pub(crate) fn same(a: Option<&Rc<Self>>, b: Option<&Rc<Self>>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}
