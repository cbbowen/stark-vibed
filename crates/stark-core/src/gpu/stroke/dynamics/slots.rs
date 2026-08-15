//! Which bindings each of the stamp loop's entry points reads, in layout order
//! (§6.2, §6.10).
//!
//! **One list per entry point, read by both sides.** [`kit`](super::kit) builds the
//! bind group *layout* from it and [`run`](super::run) builds the bind *group* from it,
//! so the two cannot disagree about which slots are present, in what order, or of what
//! type. What that replaced was two hand-kept arrays per entry point — seven pairs of
//! them — joined by nothing but the order they happened to be written in and a magic
//! element count per layout (`[..12 + 4 * usize::from(resid)]`).
//!
//! **Everything a list does not say comes from the shader.** The generated `BINDINGS`
//! table (`stark_shaders::mirror::dynamics`) carries each slot's kind, its storage
//! format, its uniform's `min_binding_size`, and whether it is `@if(resid)`-gated — all
//! read off the WESL declaration, which is the side that decides them. A list here is
//! therefore *only* a membership statement, plus the one thing a declaration cannot
//! decide: whether **this** entry point reads a texture through a sampler
//! ([`Slot::sampled`]) or with `textureLoad` ([`Slot::at`]). That really is per pair —
//! `region_color` is loaded by `snapshot` and sampled by `exchange`.
//!
//! The residual entries are listed inline, beside the color binding each rides with,
//! rather than heaped at the end of the array: the gate is on the declaration now, so
//! there is no reason to keep them in a countable tail. See the block at the head of
//! `dynamics.wesl` for what each carries (§6.7).

use crate::gpu::desc::Slot;
use stark_shaders::mirror::dynamics::binding as b;

/// The footprint copy that gives `deposit`/`settle` something to read while they
/// storage-write the region.
pub(super) const SNAPSHOT: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::REGION_COLOR),
    Slot::at(b::REGION_AUX),
    Slot::at(b::UNDER_COLOR_W),
    Slot::at(b::UNDER_AUX_W),
    Slot::at(b::REGION_RESID),
    Slot::at(b::UNDER_RESID_W),
];

/// The tool's own side of one segment's transfer.
///
/// The footprint `snapshot`'s targets are here too: a painting segment's snapshot runs
/// from the tail of the `exchange` grid rather than from a dispatch of its own
/// (`dynamics.wesl::exchange`), so its writes belong to this layout.
pub(super) const EXCHANGE: &[Slot] = &[
    Slot::at(b::ST),
    // Bilinear, unlike `snapshot`'s load of the same two slots — the reservoir texel
    // asking sits over an arbitrary sub-pixel spot on the region.
    Slot::sampled(b::REGION_COLOR),
    Slot::sampled(b::REGION_AUX),
    Slot::sampled(b::REGION_RESID),
    Slot::at(b::UNDER_COLOR_W),
    Slot::at(b::UNDER_AUX_W),
    Slot::at(b::UNDER_RESID_W),
    Slot::at(b::SAMP),
    Slot::sampled(b::COV_TEX),
    Slot::at(b::BRUSH_SRC_COLOR),
    Slot::at(b::BRUSH_SRC_AUX),
    Slot::sampled(b::BRUSH_SRC_RESID),
    Slot::at(b::BRUSH_DST_COLOR_W),
    Slot::at(b::BRUSH_DST_AUX_W),
    Slot::at(b::BRUSH_DST_RESID_W),
    // The selection mask over the region (§6.8) — sampled bilinearly here, since a
    // reservoir texel sits over an arbitrary sub-pixel spot.
    Slot::sampled(b::SEL_MASK),
];

/// Integrates the reservoir along the segment's travel axis so the deposit can read
/// the whole pass instead of one mid-pass sample.
pub(super) const BAKE: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::SAMP),
    Slot::sampled(b::BRUSH_SRC_COLOR),
    Slot::sampled(b::BRUSH_SRC_AUX),
    Slot::sampled(b::BRUSH_SRC_RESID),
    Slot::at(b::BAKE_LOAD_W),
    Slot::at(b::BAKE_LATM_W),
    Slot::at(b::BAKE_RLM_W),
];

/// The canvas's half of the transfer, exact per texel.
pub(super) const DEPOSIT: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::SAMP),
    Slot::at(b::BAKE_LOAD),
    Slot::at(b::BAKE_LATM),
    Slot::at(b::BAKE_RLM),
    Slot::at(b::UNDER_COLOR),
    Slot::at(b::UNDER_AUX),
    Slot::at(b::UNDER_RESID),
    Slot::at(b::REGION_COLOR_W),
    Slot::at(b::REGION_AUX_W),
    Slot::at(b::REGION_RESID_W),
    // The color-dynamics noise tile and its repeat sampler (§6.2).
    Slot::sampled(b::DYN_NOISE_TEX),
    Slot::at(b::DYN_NOISE_SAMP),
    // The selection mask over the region (§6.8) — read 1:1 with the region here, so
    // `textureLoad` suffices.
    Slot::at(b::SEL_MASK),
    // The canvas surface's height map — the deposition tooth (§6.4). Read nearest, so
    // it needs no sampler and is not filterable.
    Slot::at(b::SURFACE_TEX),
];

/// `cell_hoist`: the exact deposit's front half — the baked prefixes in, the per-cell
/// means out — plus the prefix-τ volume at group 1 (§6.2).
pub(super) const HOIST: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::BAKE_LOAD),
    Slot::at(b::BAKE_LATM),
    Slot::at(b::BAKE_RLM),
    Slot::at(b::CELL_TOOL_W),
    Slot::at(b::CELL_LAT_W),
    Slot::at(b::CELL_RES_W),
];

/// `deposit_coarse`: the deposit list with the baked prefixes swapped for the cell
/// means. It takes no prefix-τ tap and no bake tap of its own, which is the whole
/// point, so neither appears here (nor does group 1).
pub(super) const DEPOSIT_COARSE: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::UNDER_COLOR),
    Slot::at(b::UNDER_AUX),
    Slot::at(b::UNDER_RESID),
    Slot::at(b::REGION_COLOR_W),
    Slot::at(b::REGION_AUX_W),
    Slot::at(b::REGION_RESID_W),
    Slot::sampled(b::DYN_NOISE_TEX),
    Slot::at(b::DYN_NOISE_SAMP),
    Slot::at(b::SEL_MASK),
    Slot::at(b::SURFACE_TEX),
    Slot::at(b::CELL_TOOL),
    Slot::at(b::CELL_LAT),
    Slot::at(b::CELL_RES),
];

/// The pen-up: the deposit's targets and snapshot, and its *baked* reservoir reads too
/// — the settle's parcel is the delivery integral of the remaining pass, which the
/// settle slot's own `bake` dispatch stores (`dynamics.wesl::settle`).
pub(super) const SETTLE: &[Slot] = &[
    Slot::at(b::ST),
    Slot::at(b::BAKE_LOAD),
    Slot::at(b::BAKE_LATM),
    Slot::at(b::BAKE_RLM),
    Slot::at(b::UNDER_COLOR),
    Slot::at(b::UNDER_AUX),
    Slot::at(b::UNDER_RESID),
    Slot::at(b::REGION_COLOR_W),
    Slot::at(b::REGION_AUX_W),
    Slot::at(b::REGION_RESID_W),
    Slot::at(b::SEL_MASK),
    // The ground (§6.4): the settle lays paint, so it reads the tooth too.
    Slot::at(b::SURFACE_TEX),
];

#[cfg(test)]
mod tests {
    use super::*;
    use stark_shaders::mirror::dynamics::BINDINGS;

    /// Every list names slots the shader actually declares, and names none of them
    /// twice.
    ///
    /// The lists are the one thing on this boundary still written by hand, so this is
    /// what stands behind them. A duplicate is a wgpu validation failure at bind-group
    /// creation and an unknown index is a panic in `Binding::lookup` — both loud, but
    /// both on a GPU, which is the half of the suite CI does not run against pixels.
    /// Here they are arithmetic.
    #[test]
    fn every_slot_list_names_real_bindings_once() {
        let lists: [(&str, &[Slot]); 7] = [
            ("snapshot", SNAPSHOT),
            ("exchange", EXCHANGE),
            ("bake", BAKE),
            ("deposit", DEPOSIT),
            ("hoist", HOIST),
            ("deposit_coarse", DEPOSIT_COARSE),
            ("settle", SETTLE),
        ];
        for (what, list) in lists {
            let mut seen: Vec<u32> = Vec::new();
            for slot in list {
                let index = slot.binding();
                // Panics if the shader declares no such binding.
                let decl = stark_shaders::Binding::lookup(BINDINGS, index);
                assert!(
                    !seen.contains(&index),
                    "{what} lists `{}` (binding {index}) twice",
                    decl.name,
                );
                seen.push(index);
            }
            assert!(!list.is_empty(), "{what} lists no bindings at all");
        }
    }

    /// A residual binding is listed **beside** the color binding it rides with, and a
    /// list that takes one takes all of them — the all-or-nothing the shader's
    /// `@if(resid)` block expresses on the other side (§6.7).
    ///
    /// Stated as a count rather than as an ordering, because the ordering is now free:
    /// the gate is on the declaration, so a residual entry can sit wherever it reads
    /// best. What would still be a bug is a list that takes *some* of a pair.
    #[test]
    fn the_residual_pairs_are_taken_whole() {
        // Each residual slot and the plain slot it rides with, by name.
        let pairs = [
            (b::REGION_RESID, b::REGION_COLOR),
            (b::UNDER_RESID, b::UNDER_COLOR),
            (b::UNDER_RESID_W, b::UNDER_COLOR_W),
            (b::REGION_RESID_W, b::REGION_COLOR_W),
            (b::BRUSH_SRC_RESID, b::BRUSH_SRC_COLOR),
            (b::BRUSH_DST_RESID_W, b::BRUSH_DST_COLOR_W),
            (b::BAKE_RLM, b::BAKE_LOAD),
            (b::BAKE_RLM_W, b::BAKE_LOAD_W),
            (b::CELL_RES, b::CELL_TOOL),
            (b::CELL_RES_W, b::CELL_TOOL_W),
        ];
        let lists: [(&str, &[Slot]); 7] = [
            ("snapshot", SNAPSHOT),
            ("exchange", EXCHANGE),
            ("bake", BAKE),
            ("deposit", DEPOSIT),
            ("hoist", HOIST),
            ("deposit_coarse", DEPOSIT_COARSE),
            ("settle", SETTLE),
        ];
        for (what, list) in lists {
            let has = |i: u32| list.iter().any(|s| s.binding() == i);
            for (resid, plain) in pairs {
                assert_eq!(
                    has(resid),
                    has(plain),
                    "{what} takes only one of binding {resid} and its partner {plain}",
                );
            }
        }
        // And every `@if(resid)` slot the shader declares is covered by the pairs
        // above, so the check cannot go stale by the shader gaining one.
        for decl in BINDINGS.iter().filter(|b| b.resid) {
            assert!(
                pairs.iter().any(|(r, _)| *r == decl.index),
                "`{}` is `@if(resid)` but is not paired here",
                decl.name,
            );
        }
    }
}
