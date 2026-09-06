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
//! **Everything a list does not say comes from the shader**, and it comes *in* the
//! list rather than being looked up beside it: `d::REGION_COLOR` is the generated
//! declaration itself (`stark_shaders::mirror::dynamics::decl`), carrying the slot's
//! group, its index, its kind, its storage format, its uniform's `min_binding_size`
//! and whether it is `@if(resid)`-gated. A list here is therefore *only* a membership
//! statement, plus the two things a declaration cannot decide: whether **this** entry
//! point reads a texture through a sampler ([`Slot::sampled`]) or with `textureLoad`
//! ([`Slot::at`]), and whether a uniform is bound whole or as one dynamic-offset slot
//! ([`Slot::dynamic`]). The first really is per pair — `region_color` is loaded by
//! `snapshot` and sampled by `exchange`.
//!
//! Naming the declaration rather than the index is also what makes the group
//! unambiguous. `@binding(0)` means a different slot in each of a module's groups, and
//! the `Binding::lookup(table, index)` this replaces answered with whichever came
//! first — correct for `dynamics`, which declares one group, and silently wrong for
//! any of the dozen modules that declare two or three.
//!
//! The residual entries are listed inline, beside the color binding each rides with,
//! rather than heaped at the end of the array: the gate is on the declaration now, so
//! there is no reason to keep them in a countable tail. See the block at the head of
//! `dynamics.wesl` for what each carries (§6.7).

use crate::gpu::desc::Slot;
use stark_shaders::mirror::dynamics::decl as d;

/// The extent copy that gives `deposit`/`settle` something to read while they
/// storage-write the region.
pub(super) const SNAPSHOT: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::REGION_COLOR),
    Slot::at(d::REGION_AUX),
    Slot::at(d::UNDER_COLOR_W),
    Slot::at(d::UNDER_AUX_W),
    Slot::at(d::REGION_RESID),
    Slot::at(d::UNDER_RESID_W),
];

/// The tool's own side of one segment's transfer.
///
/// The extent `snapshot`'s targets are here too: a painting segment's snapshot runs
/// from the tail of the `exchange` grid rather than from a dispatch of its own
/// (`dynamics.wesl::exchange`), so its writes belong to this layout.
pub(super) const EXCHANGE: &[Slot] = &[
    Slot::dynamic(d::ST),
    // Bilinear, unlike `snapshot`'s load of the same two slots — the reservoir texel
    // asking sits over an arbitrary sub-pixel spot on the region.
    Slot::sampled(d::REGION_COLOR),
    Slot::sampled(d::REGION_AUX),
    Slot::sampled(d::REGION_RESID),
    Slot::at(d::UNDER_COLOR_W),
    Slot::at(d::UNDER_AUX_W),
    Slot::at(d::UNDER_RESID_W),
    Slot::at(d::SAMP),
    Slot::sampled(d::COV_TEX),
    Slot::at(d::BRUSH_SRC_COLOR),
    Slot::at(d::BRUSH_SRC_AUX),
    Slot::sampled(d::BRUSH_SRC_RESID),
    Slot::at(d::BRUSH_DST_COLOR_W),
    Slot::at(d::BRUSH_DST_AUX_W),
    Slot::at(d::BRUSH_DST_RESID_W),
    // The selection mask over the region (§6.8) — sampled bilinearly here, since a
    // reservoir texel sits over an arbitrary sub-pixel spot.
    Slot::sampled(d::SEL_MASK),
];

/// Integrates the reservoir along the segment's travel axis so the deposit can read
/// the whole pass instead of one mid-pass sample.
pub(super) const BAKE: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::SAMP),
    Slot::sampled(d::BRUSH_SRC_COLOR),
    Slot::sampled(d::BRUSH_SRC_AUX),
    Slot::sampled(d::BRUSH_SRC_RESID),
    Slot::at(d::BAKE_LOAD_W),
    Slot::at(d::BAKE_LATM_W),
    Slot::at(d::BAKE_RLM_W),
];

/// The canvas's half of the transfer, exact per texel.
pub(super) const DEPOSIT: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::SAMP),
    Slot::at(d::BAKE_LOAD),
    Slot::at(d::BAKE_LATM),
    Slot::at(d::BAKE_RLM),
    Slot::at(d::UNDER_COLOR),
    Slot::at(d::UNDER_AUX),
    Slot::at(d::UNDER_RESID),
    Slot::at(d::REGION_COLOR_W),
    Slot::at(d::REGION_AUX_W),
    Slot::at(d::REGION_RESID_W),
    // The color-dynamics noise tile and its repeat sampler (§6.2).
    Slot::sampled(d::DYN_NOISE_TEX),
    Slot::at(d::DYN_NOISE_SAMP),
    // The selection mask over the region (§6.8) — read 1:1 with the region here, so
    // `textureLoad` suffices.
    Slot::at(d::SEL_MASK),
    // The canvas substrate's height map — the deposition tooth (§6.4). Read nearest, so
    // it needs no sampler and is not filterable.
    Slot::at(d::SUBSTRATE_TEX),
    // The bleed pair's hoisted mobility (§6.2). Bound on every deposit, not only a
    // firing's: a painting segment carries `lambda_bleed = 0` and never reads it, and a
    // 1×1 stand-in is what it binds there — the same "one shader, one layout" the 1×1
    // masks buy everywhere else (§6.8).
    Slot::at(d::BLEED_W),
    // The ceiling lane over the region (§6.2), drawn by the host between
    // deposits; the kit's 1×1 zero on a stroke whose ceiling the pen does not
    // drive, by the mobility's argument above.
    Slot::at(d::REGION_LEVELS),
];

/// `bleed_weight`: one thread per snapshot texel, writing the mobility the ladder reads
/// back thirty-six times (§6.2). Its own uniform and its own target, and nothing else.
pub(super) const BLEED_WEIGHT: &[Slot] = &[Slot::dynamic(d::ST), Slot::at(d::BLEED_W_W)];

/// `cell_hoist`: the exact deposit's front half — the baked prefixes in, the per-cell
/// means out — plus the prefix-τ volume at group 1 (§6.2).
pub(super) const HOIST: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::BAKE_LOAD),
    Slot::at(d::BAKE_LATM),
    Slot::at(d::BAKE_RLM),
    Slot::at(d::CELL_TOOL_W),
    Slot::at(d::CELL_LAT_W),
    Slot::at(d::CELL_RES_W),
];

/// `deposit_coarse`: the deposit list with the baked prefixes swapped for the cell
/// means. It takes no prefix-τ tap and no bake tap of its own, which is the whole
/// point, so neither appears here (nor does group 1).
pub(super) const DEPOSIT_COARSE: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::UNDER_COLOR),
    Slot::at(d::UNDER_AUX),
    Slot::at(d::UNDER_RESID),
    Slot::at(d::REGION_COLOR_W),
    Slot::at(d::REGION_AUX_W),
    Slot::at(d::REGION_RESID_W),
    Slot::sampled(d::DYN_NOISE_TEX),
    Slot::at(d::DYN_NOISE_SAMP),
    Slot::at(d::SEL_MASK),
    Slot::at(d::SUBSTRATE_TEX),
    Slot::at(d::CELL_TOOL),
    Slot::at(d::CELL_LAT),
    Slot::at(d::CELL_RES),
    Slot::at(d::REGION_LEVELS),
];

/// `snapshot_field`: the liquify field under a warp slot's square, copied so the
/// composition can gather it while storage-writing it (§6.13).
pub(super) const SNAPSHOT_FIELD: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::FIELD),
    Slot::at(d::UNDER_FIELD_W),
];

/// `warp`: one segment's step composed into the liquify field (§6.13) — the
/// snapshot in, the field out, the selection scaling the follow. Nothing of the
/// picture: the field is all this kernel evolves.
pub(super) const WARP: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::UNDER_FIELD),
    Slot::at(d::FIELD_W),
    Slot::at(d::SEL_MASK),
];

/// `warp_apply`: the one resample of a liquify piece (§6.13) — the field and the
/// base composite in, the region's channels out, which the write-back then slices
/// exactly as it slices the wet loop's.
pub(super) const WARP_APPLY: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::FIELD),
    Slot::at(d::BASE_COLOR),
    Slot::at(d::BASE_AUX),
    Slot::at(d::BASE_RESID),
    Slot::at(d::REGION_COLOR_W),
    Slot::at(d::REGION_AUX_W),
    Slot::at(d::REGION_RESID_W),
];

/// The pen-up: the deposit's targets and snapshot, and its *baked* reservoir reads too
/// — the settle's parcel is the delivery integral of the remaining pass, which the
/// settle slot's own `bake` dispatch stores (`dynamics.wesl::settle`).
pub(super) const SETTLE: &[Slot] = &[
    Slot::dynamic(d::ST),
    Slot::at(d::BAKE_LOAD),
    Slot::at(d::BAKE_LATM),
    Slot::at(d::BAKE_RLM),
    Slot::at(d::UNDER_COLOR),
    Slot::at(d::UNDER_AUX),
    Slot::at(d::UNDER_RESID),
    Slot::at(d::REGION_COLOR_W),
    Slot::at(d::REGION_AUX_W),
    Slot::at(d::REGION_RESID_W),
    Slot::at(d::SEL_MASK),
    // The substrate (§6.4): the settle lays paint, so it reads the tooth too.
    Slot::at(d::SUBSTRATE_TEX),
    // …and the ceiling lane, since it lays through `lay_parcel` (§6.2).
    Slot::at(d::REGION_LEVELS),
];

#[cfg(test)]
mod tests {
    use super::*;
    // The indices, for the pairs below — a residual slot and its partner are
    // stated as the two numbers a list is checked against, not as two declarations.
    use stark_shaders::mirror::dynamics::{BINDINGS, binding as b};

    /// Every list names slots the shader actually declares, and names none of them
    /// twice.
    ///
    /// The lists are the one thing on this boundary still written by hand, so this is
    /// what stands behind them. A duplicate is a wgpu validation failure at bind-group
    /// creation — loud, but on a GPU, which is the half of the suite CI does not run
    /// against pixels. Here it is arithmetic.
    ///
    /// "Names a *real* binding" needs no assertion any more: a slot carries the
    /// declaration itself (`decl::REGION_COLOR`), so an index the shader does not
    /// declare cannot be written down. That is the class the `Binding::lookup` this
    /// replaces could only check one instance of at a time.
    #[test]
    fn every_slot_list_names_real_bindings_once() {
        for (what, list) in LISTS {
            let mut seen: Vec<u32> = Vec::new();
            for slot in list {
                let index = slot.binding();
                assert!(
                    !seen.contains(&index),
                    "{what} lists `{}` (binding {index}) twice",
                    slot.decl().name,
                );
                seen.push(index);
            }
            assert!(!list.is_empty(), "{what} lists no bindings at all");
        }
    }

    /// Every list is one `@group`'s worth, which is what a bind group layout is.
    ///
    /// `desc::layout_for` asserts the same thing, but only for the layouts a GPU run
    /// actually builds; this states it for all seven without one.
    #[test]
    fn every_slot_list_is_one_group() {
        for (what, list) in LISTS {
            let g = list[0].decl().group;
            for slot in list {
                assert_eq!(
                    slot.decl().group,
                    g,
                    "{what} lists `{}` from @group({}) beside @group({g})",
                    slot.decl().name,
                    slot.decl().group,
                );
            }
        }
    }

    /// The entry points' lists, named — every test here is about all of them.
    const LISTS: [(&str, &[Slot]); 11] = [
        ("bleed_weight", BLEED_WEIGHT),
        ("snapshot", SNAPSHOT),
        ("exchange", EXCHANGE),
        ("bake", BAKE),
        ("deposit", DEPOSIT),
        ("hoist", HOIST),
        ("deposit_coarse", DEPOSIT_COARSE),
        ("settle", SETTLE),
        ("snapshot_field", SNAPSHOT_FIELD),
        ("warp", WARP),
        ("warp_apply", WARP_APPLY),
    ];

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
            (b::BASE_RESID, b::BASE_COLOR),
        ];
        for (what, list) in LISTS {
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
