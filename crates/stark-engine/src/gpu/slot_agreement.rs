//! Every hand-written slot list, against what the entry points bound through it
//! actually read (§6.10).
//!
//! A `&[Slot]` list is a **membership claim**: these are the bindings, and these are the
//! ones read through a sampler. Both halves are things the linked shader knows — naga
//! works them out while type-checking, callees included, and the generated
//! [`EntryPoint::uses`] carries the answer — and until now nothing compared the two. A
//! list naming one slot too many is a layout entry and a bind-group entry no shader
//! reads, which `wgpu` accepts in silence; one too few is a pipeline that will not
//! create, on a GPU, in whichever colour space links that variant.
//!
//! **The unit is the list, not the pipeline.** Several pipelines share one layout — the
//! blur's four kernels, the filter's three fragment entry points, the sweep's five —
//! and a shared layout is necessarily the *union* of what its pipelines read, so a
//! per-pipeline comparison would report every sharer's unused entries as a fault. The
//! table below is still the pipelines, because that is what a reader can check against
//! `desc`'s call sites; the check folds them by list.
//!
//! **It is not a `#[derive]` for the lists.** What a list still says, and the shader
//! cannot, is which `@group` a pipeline binds it as, in what order, and at what
//! visibility. The list stays written by hand; this says whether it is true.
//!
//! [`KNOWN`] holds the differences that stand today, each with why. Nothing here
//! changes the engine.

use std::collections::BTreeMap;

use stark_shaders::{EntryPoint, Lane, Resid};

use crate::gpu::composite::{blend, blur, filter, guides, media, overlay, resolve, tiles};
use crate::gpu::desc::Slot;
use crate::gpu::stroke::dynamics::{kit, slots as dyn_slots};
use crate::gpu::stroke::{erase, swept};
use crate::gpu::{fill, merge, selection, transform};

/// One bind group layout a pipeline binds: the list it is built from, named as the
/// host writes it, and **the residual the host built it under** — which is not always
/// the colour space's. The sweep, the prefix tap and the overlay pass `false`
/// unconditionally, having no `@if(resid)` declaration between them.
#[derive(Clone, Copy)]
struct Group {
    name: &'static str,
    slots: &'static [Slot],
    resid: bool,
}

const fn group(name: &'static str, slots: &'static [Slot], resid: bool) -> Group {
    Group { name, slots, resid }
}

/// One pipeline the engine creates: the entry points it is built from, and the groups
/// its pipeline layout lists.
struct Case {
    /// The label `desc` gives `wgpu`, shortened.
    what: &'static str,
    /// Every stage's entry point. A render pipeline's layout has to satisfy the vertex
    /// and fragment stages together, so the two are read as one set.
    entries: Vec<EntryPoint>,
    groups: Vec<Group>,
}

/// A slot as both sides can name it: the module that declares it, and the declaration.
///
/// Neither alone is unique — three modules partition group 0 between them in the blend
/// and filter layouts, and `stamp_common`'s `xf` is not `transform`'s.
type Key = (&'static str, &'static str);

/// Which way a list and its shader disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Diff {
    /// The list names a slot no entry point bound through it reads.
    Unread,
    /// The list and the shader disagree about whether it is read through a sampler.
    Sampling,
}

/// The differences that stand today: `(list, module, declaration, which, why)`.
///
/// Every one is checked to still be a difference ([`every_known_difference_still_stands`]),
/// so a waiver cannot outlive what it excuses. None of them is fixed here: the lists
/// are the next commit's business, and a check that quietly edits what it measures is
/// not a check.
const KNOWN: &[(&str, &str, &str, Diff, &str)] = &[
    // One layout serves both colour spaces, because whether the pigment LUT is real is
    // `ColorSpace::needs_pigment_lut`'s answer rather than the layout's: the
    // colorimetric shaders do not import `mixbox_lut` at all, and the host binds a 1×1
    // stand-in — the same "one shader, one layout" the zero masks buy elsewhere (§6.8).
    // So in an Oklab document these two entries describe a placeholder.
    (
        "BLEND_SLOTS",
        "mixbox_lut",
        "PIGMENT_LUT",
        Diff::Unread,
        "a placeholder in the colorimetric space, which declares no LUT",
    ),
    (
        "BLEND_SLOTS",
        "mixbox_lut",
        "PIGMENT_SAMP",
        Diff::Unread,
        "the placeholder LUT's sampler",
    ),
    (
        "FILTER_SLOTS",
        "mixbox_lut",
        "PIGMENT_LUT",
        Diff::Unread,
        "a placeholder in the colorimetric space, which declares no LUT",
    ),
    (
        "FILTER_SLOTS",
        "mixbox_lut",
        "PIGMENT_SAMP",
        Diff::Unread,
        "the placeholder LUT's sampler",
    ),
    // The three below are the lists being wrong, not the layout being shared. Left as
    // they are and recorded here, since fixing a list changes a layout and a bind
    // group, which is a pixel-affecting change this commit does not make.
    (
        "dynamics::DEPOSIT",
        "dynamics",
        "SAMP",
        Diff::Unread,
        "the bilinear sampler is `exchange`'s and `bake`'s; the deposit reads its noise \
         through `dyn_noise_samp` and everything else with `textureLoad`",
    ),
    (
        "dynamics::SETTLE",
        "dynamics",
        "REGION_LEVELS",
        Diff::Unread,
        "the list says the settle lays through `lay_parcel`, which reads the lane — it \
         does not: the pen-up builds its parcel in `settle` itself and stores through \
         `stack_and_store`",
    ),
    (
        "dynamics::EXCHANGE",
        "dynamics",
        "BRUSH_SRC_RESID",
        Diff::Sampling,
        "listed `sampled` beside its `at` partners `BRUSH_SRC_COLOR`/`_AUX`, which \
         `exchange` loads exactly as it loads the residual — the filterable flag is \
         `bake`'s, where all three really are sampled",
    ),
];

/// What the shader says: every binding these entry points reach **in `@group(group)`**,
/// and whether any of them samples it.
///
/// Restricted to the one group, because a bind group layout describes exactly one — so
/// what a list is answerable for is its group's share of what the entry point reads,
/// and nothing else the pipeline binds beside it.
///
/// `|`, not the last one's answer: a layout entry is one entry for every stage and
/// every sharer, so a texture one of them samples has to be declared filterable even
/// where the rest load it.
fn shader_uses(entries: &[EntryPoint], group: u32) -> BTreeMap<Key, bool> {
    let mut out = BTreeMap::new();
    for ep in entries {
        for used in ep.uses.iter().filter(|u| u.decl.group == group) {
            *out.entry((used.decl.module, used.decl.name))
                .or_insert(false) |= used.sampled;
        }
    }
    out
}

/// What one list claims, with the residual gate it is built under applied.
fn host_list(g: Group) -> BTreeMap<Key, bool> {
    let mut out = BTreeMap::new();
    for slot in g.slots.iter().filter(|s| s.present(g.resid)) {
        *out.entry((slot.decl().module, slot.decl().name))
            .or_insert(false) |= slot.is_sampled();
    }
    out
}

/// The `@group` a list describes. `desc::layout_for` refuses a list spanning two, and
/// `slots.rs` states it for the wet loop's eleven without a device.
fn list_group(g: Group) -> u32 {
    g.slots
        .first()
        .expect("a slot list names at least one binding")
        .decl()
        .group
}

/// The colour-space-specific passes: a document runs the pigment media, blend and
/// filter shaders or the colorimetric ones, never both.
///
/// Shaped like [`ColorSpace::resid`](crate::colorspace::ColorSpace::resid) for the same
/// reason — without the `mixbox` feature the pigment half does not exist to be named.
fn space_cases(resid: bool) -> Vec<Case> {
    #[cfg(feature = "mixbox")]
    if resid {
        let (m, b, f) = (
            stark_shaders::media_mixbox(),
            stark_shaders::blend_mixbox(),
            stark_shaders::filter_mixbox(),
        );
        return space_shaped(
            resid,
            [m.vs_main, m.fs_main],
            [b.vs_main, b.fs_main],
            [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
        );
    }
    let (m, b, f) = (
        stark_shaders::media_oklab(),
        stark_shaders::blend_oklab(),
        stark_shaders::filter_oklab(),
    );
    space_shaped(
        resid,
        [m.vs_main, m.fs_main],
        [b.vs_main, b.fs_main],
        [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
    )
}

/// The five pipelines the two spaces have the same shape of — one media pass, one
/// blend, and the filter's three fragment entry points over one layout.
fn space_shaped(
    resid: bool,
    media_eps: [EntryPoint; 2],
    blend_eps: [EntryPoint; 2],
    filter_eps: [EntryPoint; 4],
) -> Vec<Case> {
    let [vs, fs_main, fs_tile, fs_blur_decode] = filter_eps;
    let filter = |what, fs| Case {
        what,
        entries: vec![vs, fs],
        groups: vec![group("FILTER_SLOTS", filter::FILTER_SLOTS, resid)],
    };
    vec![
        Case {
            what: "media",
            entries: media_eps.to_vec(),
            groups: vec![group("MEDIA_SLOTS", media::MEDIA_SLOTS, resid)],
        },
        Case {
            what: "blend",
            entries: blend_eps.to_vec(),
            groups: vec![group("BLEND_SLOTS", blend::BLEND_SLOTS, resid)],
        },
        filter("filter", fs_main),
        filter("filter tile", fs_tile),
        filter("filter blur decode", fs_blur_decode),
    ]
}

/// Every pipeline the engine creates, for a colour space with (`r`/`resid`) or without
/// a residual.
#[expect(
    clippy::too_many_lines,
    reason = "one entry per pipeline, which is the point"
)]
fn cases(r: Resid, resid: bool) -> Vec<Case> {
    let composite = stark_shaders::composite(r);
    let matte = stark_shaders::matte(r);
    let ov = stark_shaders::overlay();
    let gu = stark_shaders::guides();
    let re = stark_shaders::resolve();
    let bl = stark_shaders::blur();
    let fi = stark_shaders::fill(r);
    let me = stark_shaders::merge(r);
    let sl = stark_shaders::slab(r);
    let tr = stark_shaders::transform(r);
    let se = stark_shaders::selection();
    let mr = stark_shaders::mask_region();
    let plain = stark_shaders::stamp(r, Lane::Plain);
    let ceiling = stark_shaders::stamp(r, Lane::Ceiling);
    let ig = stark_shaders::integrate(r);
    let er = stark_shaders::erase(r);
    let dy = stark_shaders::dynamics(r);
    let li = stark_shaders::liquify(r);
    let sc = stark_shaders::slice();

    // The composite's two groups, which the wet loop's own preview pass rebuilds from
    // the same two lists.
    let view = || group("composite::VIEW_SLOTS", tiles::VIEW_SLOTS, resid);
    let tile = || group("composite::TILE_SLOTS", tiles::TILE_SLOTS, resid);
    // The sweep's three, shared by five pipelines and built `false` throughout: nothing
    // `stamp_common` declares is `@if(resid)`.
    let sweep = || {
        vec![
            group("XFORM_SLOTS", swept::XFORM_SLOTS, false),
            group("PREFIX_SLOTS", swept::PREFIX_SLOTS, false),
            group("NOISE_SLOTS", swept::NOISE_SLOTS, false),
        ]
    };
    // The prefix tap the wet loop's kernels take at group 1, likewise — a second
    // layout object over the one list, compute-visible where the sweep's is the
    // fragment stage's.
    let prefix = || group("PREFIX_SLOTS", swept::PREFIX_SLOTS, false);
    let dyn_group = |name, slots| group(name, slots, resid);

    let mut all = vec![
        Case {
            what: "composite",
            entries: vec![composite.vs_main, composite.fs_main],
            groups: vec![view(), tile()],
        },
        Case {
            what: "matte",
            entries: vec![matte.vs_main, matte.fs_main],
            groups: vec![view(), group("RAMP_SLOTS", tiles::RAMP_SLOTS, resid)],
        },
        Case {
            what: "overlay",
            entries: vec![ov.vs_main, ov.fs_main],
            groups: vec![
                group("overlay::VIEW_SLOTS", overlay::VIEW_SLOTS, false),
                group("MASK_SLOTS", overlay::MASK_SLOTS, false),
            ],
        },
        Case {
            what: "guides",
            entries: vec![gu.vs_main, gu.fs_main],
            groups: vec![group("GUIDE_SLOTS", guides::GUIDE_SLOTS, false)],
        },
        Case {
            what: "resolve",
            entries: vec![re.vs_main, re.fs_main],
            groups: vec![group("RESOLVE_SLOTS", resolve::RESOLVE_SLOTS, false)],
        },
        Case {
            what: "blur fft",
            entries: vec![bl.fft_both],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur kernel fft",
            entries: vec![bl.fft_one],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur make kernel",
            entries: vec![bl.make_kernel],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur apply kernel",
            entries: vec![bl.apply_kernel],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "fill",
            entries: vec![fi.vs_main, fi.fs_main],
            groups: vec![group("FILL_SLOTS", fill::FILL_SLOTS, resid)],
        },
        Case {
            what: "merge",
            entries: vec![me.vs_main, me.fs_main],
            groups: vec![group("MERGE_SLOTS", merge::MERGE_SLOTS, resid)],
        },
        Case {
            what: "slab expand",
            entries: vec![sl.vs_main, sl.fs_expand],
            groups: vec![group("SLAB_SLOTS", merge::SLAB_SLOTS, resid)],
        },
        Case {
            what: "slab store",
            entries: vec![sl.vs_main, sl.fs_store],
            groups: vec![group("SLAB_SLOTS", merge::SLAB_SLOTS, resid)],
        },
        Case {
            what: "transform parcel",
            entries: vec![tr.vs_quad, tr.fs_parcel],
            groups: vec![
                group("QUAD_SLOTS", transform::QUAD_SLOTS, resid),
                group("SRC_SLOTS", transform::SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform mask",
            entries: vec![tr.vs_quad, tr.fs_mask],
            groups: vec![
                group("QUAD_SLOTS", transform::QUAD_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform parcel gated",
            entries: vec![tr.vs_gated, tr.fs_parcel_gated],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("SRC_SLOTS", transform::SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform mask gated",
            entries: vec![tr.vs_gated, tr.fs_mask_gated],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform combine",
            entries: vec![tr.vs_fill, tr.fs_combine],
            groups: vec![group("COMBINE_SLOTS", transform::COMBINE_SLOTS, resid)],
        },
        Case {
            what: "transform mask base",
            entries: vec![tr.vs_fill, tr.fs_mask_base],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "selection",
            entries: vec![se.vs_main, se.fs_main],
            groups: vec![group("RASTERIZE_SLOTS", selection::RASTERIZE_SLOTS, false)],
        },
        Case {
            what: "selection region",
            entries: vec![mr.vs_main, mr.fs_main],
            groups: vec![
                group("REGION_VIEW_SLOTS", selection::REGION_VIEW_SLOTS, false),
                group("REGION_TILE_SLOTS", selection::REGION_TILE_SLOTS, false),
            ],
        },
        Case {
            what: "sweep",
            entries: vec![plain.vs_main, plain.fs_main],
            groups: sweep(),
        },
        Case {
            what: "sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_main],
            groups: sweep(),
        },
        Case {
            what: "sweep levels",
            entries: vec![plain.vs_main, plain.fs_levels],
            groups: sweep(),
        },
        Case {
            what: "erase sweep",
            entries: vec![plain.vs_main, plain.fs_erase],
            groups: sweep(),
        },
        Case {
            what: "erase sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_erase],
            groups: sweep(),
        },
        Case {
            what: "integrate",
            entries: vec![ig.vs_main, ig.fs_main],
            groups: vec![group("INTEGRATE_SLOTS", swept::INTEGRATE_SLOTS, resid)],
        },
        Case {
            what: "erase",
            entries: vec![er.vs_main, er.fs_main],
            groups: vec![group("ERASE_SLOTS", erase::ERASE_SLOTS, resid)],
        },
        Case {
            what: "dynamics composite",
            entries: vec![composite.vs_main, composite.fs_raw],
            groups: vec![view(), tile()],
        },
        Case {
            what: "dynamics snapshot",
            entries: vec![dy.snapshot],
            groups: vec![dyn_group("dynamics::SNAPSHOT", dyn_slots::SNAPSHOT)],
        },
        Case {
            what: "dynamics bleed weight",
            entries: vec![dy.bleed_weight],
            groups: vec![
                dyn_group("dynamics::BLEED_WEIGHT", dyn_slots::BLEED_WEIGHT),
                prefix(),
            ],
        },
        Case {
            what: "dynamics exchange",
            entries: vec![dy.exchange],
            groups: vec![dyn_group("dynamics::EXCHANGE", dyn_slots::EXCHANGE)],
        },
        Case {
            what: "dynamics bake",
            entries: vec![dy.bake],
            groups: vec![dyn_group("dynamics::BAKE", dyn_slots::BAKE), prefix()],
        },
        Case {
            what: "dynamics deposit",
            entries: vec![dy.deposit],
            groups: vec![dyn_group("dynamics::DEPOSIT", dyn_slots::DEPOSIT), prefix()],
        },
        Case {
            what: "dynamics cell hoist",
            entries: vec![dy.cell_hoist],
            groups: vec![dyn_group("dynamics::HOIST", dyn_slots::HOIST), prefix()],
        },
        Case {
            what: "dynamics deposit coarse",
            entries: vec![dy.deposit_coarse],
            groups: vec![dyn_group(
                "dynamics::DEPOSIT_COARSE",
                dyn_slots::DEPOSIT_COARSE,
            )],
        },
        Case {
            what: "dynamics settle",
            entries: vec![dy.settle],
            groups: vec![dyn_group("dynamics::SETTLE", dyn_slots::SETTLE), prefix()],
        },
        Case {
            what: "liquify snapshot field",
            entries: vec![li.snapshot_field],
            groups: vec![dyn_group(
                "liquify::SNAPSHOT_FIELD",
                dyn_slots::SNAPSHOT_FIELD,
            )],
        },
        Case {
            what: "liquify warp",
            entries: vec![li.warp],
            groups: vec![dyn_group("liquify::WARP", dyn_slots::WARP), prefix()],
        },
        Case {
            what: "liquify warp apply",
            entries: vec![li.warp_apply],
            groups: vec![dyn_group("liquify::WARP_APPLY", dyn_slots::WARP_APPLY)],
        },
        Case {
            what: "dynamics slice",
            entries: vec![sc.vs_main, sc.fs_main],
            groups: vec![group("SLICE_SLOTS", kit::SLICE_SLOTS, false)],
        },
    ];
    all.extend(space_cases(resid));
    all
}

/// The table folded by list: each one, and every entry point bound through it.
///
/// # Panics
/// If one name stands for two lists, or for one list at two residuals — either would
/// make the union below a comparison against something no layout is.
fn by_list(cases: &[Case]) -> BTreeMap<&'static str, (Group, Vec<EntryPoint>)> {
    let mut out: BTreeMap<&'static str, (Group, Vec<EntryPoint>)> = BTreeMap::new();
    for case in cases {
        for g in &case.groups {
            let (seen, entries) = out.entry(g.name).or_insert_with(|| (*g, Vec::new()));
            // By content, not by address: a `const` reference is re-evaluated at every
            // use, so two mentions of one list need not be one allocation.
            let same = seen.resid == g.resid
                && seen.slots.len() == g.slots.len()
                && seen
                    .slots
                    .iter()
                    .zip(g.slots)
                    .all(|(a, b)| a.decl() == b.decl() && a.is_sampled() == b.is_sampled());
            assert!(
                same,
                "`{}` names two different lists, or one list at two residuals — the \
                 second is `{}`'s",
                g.name, case.what,
            );
            entries.extend(case.entries.iter().copied());
        }
    }
    out
}

/// Every difference between one list and what is bound through it, as lines.
fn differences(name: &str, g: Group, entries: &[EntryPoint]) -> Vec<String> {
    let used = shader_uses(entries, list_group(g));
    let listed = host_list(g);
    let waived = |key: Key, which: Diff| {
        KNOWN
            .iter()
            .any(|(l, m, n, w, _)| *l == name && (*m, *n) == key && *w == which)
    };
    let mut out = Vec::new();
    for (key, sampled) in &listed {
        match used.get(key) {
            None if !waived(*key, Diff::Unread) => out.push(format!(
                "`{name}` lists `{}.wesl`'s `{}`, which nothing bound through it reads",
                key.0, key.1,
            )),
            Some(is) if is != sampled && !waived(*key, Diff::Sampling) => out.push(format!(
                "`{name}` lists `{}.wesl`'s `{}` as {}, where the shader {} it",
                key.0,
                key.1,
                if *sampled { "sampled" } else { "loaded" },
                if *is { "samples" } else { "loads" },
            )),
            _ => {}
        }
    }
    for key in used.keys() {
        if !listed.contains_key(key) {
            out.push(format!(
                "`{name}` omits `{}.wesl`'s `{}`, which is bound through it — the \
                 pipeline cannot be created",
                key.0, key.1,
            ));
        }
    }
    out
}

/// The colour spaces this build has, which every check here runs over.
fn spaces() -> Vec<(Resid, bool)> {
    #[cfg(feature = "mixbox")]
    let spaces = vec![(Resid::Without, false), (Resid::With, true)];
    #[cfg(not(feature = "mixbox"))]
    let spaces = vec![(Resid::Without, false)];
    spaces
}

/// Every list, against what is bound through it, in every colour space this build has.
///
/// Needs no device: the declarations are `const`s and the uses are generated.
#[test]
fn every_slot_list_is_what_its_entry_points_bind() {
    let mut bad: Vec<String> = Vec::new();
    for (r, resid) in spaces() {
        for (name, (g, entries)) in by_list(&cases(r, resid)) {
            bad.extend(
                differences(name, g, &entries)
                    .into_iter()
                    .map(|line| format!("resid={resid}: {line}")),
            );
        }
    }
    assert!(
        bad.is_empty(),
        "the hand-written slot lists and the shaders disagree:\n{}",
        bad.join("\n"),
    );
}

/// Every [`KNOWN`] waiver still names a difference, wherever its slot is reached.
///
/// A waiver that has stopped applying is how an allow-list turns into a place the next
/// mistake hides: the day `blend_oklab` declares a LUT of its own, those two should
/// fail rather than quietly cover something else.
///
/// "Wherever it is reached", because a `@if(resid)` slot is in no list at all in a
/// build without the pigment space — unexercised rather than stale. The list *name* is
/// checked outright, so a waiver naming nothing cannot hide behind that.
#[test]
fn every_known_difference_still_stands() {
    let mut seen = vec![false; KNOWN.len()];
    let mut live = vec![false; KNOWN.len()];
    let mut lists: Vec<&'static str> = Vec::new();
    for (r, resid) in spaces() {
        for (name, (g, entries)) in by_list(&cases(r, resid)) {
            lists.push(name);
            let used = shader_uses(&entries, list_group(g));
            let listed = host_list(g);
            for (i, (list, module, decl, which, _)) in KNOWN.iter().enumerate() {
                if *list != name {
                    continue;
                }
                let key = (*module, *decl);
                let Some(sampled) = listed.get(&key) else {
                    continue;
                };
                seen[i] = true;
                live[i] |= match which {
                    Diff::Unread => !used.contains_key(&key),
                    Diff::Sampling => used.get(&key).is_some_and(|is| is != sampled),
                };
            }
        }
    }
    let named: Vec<&str> = KNOWN
        .iter()
        .filter(|(list, ..)| !lists.contains(list))
        .map(|(list, ..)| *list)
        .collect();
    assert!(named.is_empty(), "these waivers name no list: {named:?}");

    let stale: Vec<String> = KNOWN
        .iter()
        .zip(seen.iter().zip(&live))
        .filter(|(_, (seen, live))| **seen && !**live)
        .map(|((list, module, decl, ..), _)| format!("{list}'s `{module}.wesl`'s `{decl}`"))
        .collect();
    assert!(
        stale.is_empty(),
        "these waivers name no difference where their slot is listed: {stale:?}",
    );
}

/// The table covers every pipeline the engine creates — the one number a reader can
/// check against `desc`'s call sites.
#[test]
fn the_table_names_every_pipeline() {
    assert_eq!(
        cases(Resid::Without, false).len(),
        46,
        "a pipeline was added or removed without this table following it",
    );
}
